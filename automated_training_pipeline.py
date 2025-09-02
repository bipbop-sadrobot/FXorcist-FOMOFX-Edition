#!/usr/bin/env python3
"""
Automated Forex AI Training Pipeline
Downloads data, processes it, and trains models automatically.
"""

import pandas as pd
import numpy as np
import os
import sys
import subprocess
import logging
import psutil
import concurrent.futures
import torch
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import asyncio
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from memory_profiler import profile
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from dashboard.utils.enhanced_data_loader import EnhancedDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/automated_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutomatedTrainingPipeline:
    """Automated pipeline for forex data download, processing, and model training with advanced synthesis."""

    def __init__(self, synthesis_config: Optional[Dict[str, float]] = None):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models" / "trained"
        self.logs_dir = self.project_root / "logs"
        
        # Initialize enhanced data loader
        self.data_loader = EnhancedDataLoader(num_workers=os.cpu_count())
        if synthesis_config:
            self.data_loader.synthesis_config.update(synthesis_config)
        
        # Initialize GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = os.cpu_count()
        
        # Enhanced performance monitoring
        self.metrics = {
            'data_processing_time': 0,
            'training_time': 0,
            'peak_memory_usage': 0,
            'gpu_memory_usage': 0 if torch.cuda.is_available() else None,
            'synthetic_data_ratio': 0,
            'edge_cases_generated': 0,
            'cache_hit_rate': 0,
            'training_throughput': 0,
            'model_convergence_rate': {},
            'feature_importance_history': []
        }

        # Create directories
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Forex symbols to download
        self.symbols = [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
            "USDCHF", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY"
        ]

        # Years to download (2020-2024 for comprehensive training)
        self.years = list(range(2020, 2025))

        # Months (1-12)
        self.months = list(range(1, 13))

    async def download_data_batch(self, symbols: List[str], year: int) -> bool:
        """Download data for multiple symbols for a specific year."""
        logger.info(f"Downloading data for {symbols} in {year}")

        success_count = 0
        for symbol in symbols:
            try:
                # Run the fetch script for each symbol and month
                for month in self.months:
                    cmd = [
                        "bash", "scripts/fetch_data.sh",
                        "--source", "histdata",
                        "--symbols", symbol,
                        "--year", str(year),
                        "--month", str(month)
                    ]

                    result = subprocess.run(
                        cmd,
                        cwd=self.project_root,
                        capture_output=True,
                        text=True,
                        timeout=60
                    )

                    if result.returncode == 0:
                        logger.info(f"✅ Downloaded {symbol} {year}-{month:02d}")
                        success_count += 1
                    else:
                        logger.warning(f"❌ Failed {symbol} {year}-{month:02d}: {result.stderr}")

                await asyncio.sleep(1)  # Rate limiting

            except subprocess.TimeoutExpired:
                logger.error(f"Timeout downloading {symbol} {year}")
            except Exception as e:
                logger.error(f"Error downloading {symbol} {year}: {str(e)}")

        return success_count > 0

    async def download_all_data(self) -> bool:
        """Download comprehensive forex data."""
        logger.info("Starting comprehensive data download")

        total_downloads = 0
        successful_downloads = 0

        # Download in batches to avoid overwhelming the server
        batch_size = 3
        for i in range(0, len(self.symbols), batch_size):
            symbol_batch = self.symbols[i:i + batch_size]

            for year in self.years:
                success = await self.download_data_batch(symbol_batch, year)
                if success:
                    successful_downloads += len(symbol_batch)
                total_downloads += len(symbol_batch)

                # Progress logging
                progress = (successful_downloads / total_downloads) * 100
                logger.info(".1f")

        logger.info(f"Data download complete: {successful_downloads}/{total_downloads} successful")
        return successful_downloads > 0

    def _process_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Process data for a single symbol using parallel processing."""
        symbol_data = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for year in self.years:
                for month in self.months:
                    csv_path = self.data_dir / "raw" / "histdata" / symbol / str(year) / f"{month:02d}.csv"
                    if csv_path.exists():
                        futures.append(executor.submit(pd.read_csv, csv_path))
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    df = future.result()
                    df['symbol'] = symbol
                    symbol_data.append(df)
                except Exception as e:
                    logger.warning(f"Error loading data for {symbol}: {str(e)}")
        
        if symbol_data:
            return pd.concat(symbol_data, ignore_index=True)
        return None

    def process_data(self, augment_data: bool = True) -> Optional[pd.DataFrame]:
        """Process and combine all downloaded forex data with synthetic data generation."""
        logger.info("Starting enhanced data processing with synthesis")
        start_time = datetime.now()
        
        try:
            # Load and process data using enhanced data loader
            base_data = self.data_loader.load_forex_data(
                timeframe="1H",
                augment_data=augment_data
            )[0]
            
            if augment_data:
                # Generate synthetic data including edge cases
                synthetic_data = self.data_loader.generate_synthetic_data(
                    base_data,
                    num_samples=int(len(base_data) * 0.3),  # 30% additional synthetic data
                    include_edge_cases=True
                )
                
                # Combine real and synthetic data
                combined_data = pd.concat([base_data, synthetic_data])
                combined_data = combined_data.sort_index()
                
                # Update metrics
                self.metrics['synthetic_data_ratio'] = len(synthetic_data) / len(base_data)
                self.metrics['edge_cases_generated'] = int(len(synthetic_data) * 
                                                         self.data_loader.synthesis_config['edge_case_ratio'])
                
                logger.info(f"Generated {len(synthetic_data)} synthetic samples "
                          f"({self.metrics['edge_cases_generated']} edge cases)")
            else:
                combined_data = base_data

            # Enhanced preprocessing with parallel processing
            processed_data = self.preprocess_data(combined_data)

            # Save processed data
            output_path = self.data_dir / "processed" / "comprehensive_forex_data.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            processed_data.to_parquet(output_path)
            
            # Update processing time metric
            self.metrics['data_processing_time'] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Saved processed data to {output_path}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
            return None

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the combined forex data."""
        logger.info("Preprocessing data")

        # Convert timestamp if needed
        if 'timestamp' not in df.columns:
            # Assume first column is timestamp
            timestamp_col = df.columns[0]
            df = df.rename(columns={timestamp_col: 'timestamp'})

        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Add technical indicators
        df = self.add_technical_indicators(df)

        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp', 'symbol'])

        logger.info(f"Preprocessed data: {len(df)} rows, {len(df.columns)} columns")
        return df

    def _calculate_rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI using vectorized operations."""
        # Calculate price changes
        delta = np.diff(close, prepend=close[0])
        
        # Separate gains and losses
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        # Calculate average gains and losses
        avg_gains = pd.Series(gains).rolling(window=period).mean().values
        avg_losses = pd.Series(losses).rolling(window=period).mean().values
        
        # Calculate RS and RSI
        rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses != 0)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def _calculate_macd(self, close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MACD using vectorized operations."""
        # Calculate EMAs
        ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
        ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
        
        return macd_line, signal_line

    def _calculate_indicators_batch(self, group_data: Tuple[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate technical indicators for a batch of data."""
        symbol, group = group_data
        df = group.copy()
        
        # Optimize calculations using vectorized operations
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # Calculate indicators using NumPy for better performance
        df['returns'] = np.diff(close, prepend=close[0])
        df['log_returns'] = np.log(close[1:] / close[:-1])
        
        # Moving averages using convolution
        window_sizes = [5, 10, 20, 50, 100]
        for size in window_sizes:
            df[f'sma_{size}'] = pd.Series(close).rolling(size).mean().values
            df[f'ema_{size}'] = pd.Series(close).ewm(span=size).mean().values
        
        # Volatility indicators
        df['atr'] = pd.Series(high - low).rolling(14).mean().values
        df['volatility'] = pd.Series(df['returns']).rolling(20).std().values
        
        # Momentum indicators
        df['rsi'] = self._calculate_rsi(close)
        df['macd'], df['macd_signal'] = self._calculate_macd(close)
        
        # Additional indicators
        df['bollinger_upper'] = df['sma_20'] + 2 * df['volatility']
        df['bollinger_lower'] = df['sma_20'] - 2 * df['volatility']
        df['momentum'] = close / np.roll(close, 10) - 1
        df['rate_of_change'] = (close - np.roll(close, 10)) / np.roll(close, 10)
        
        # Volume-based indicators (if volume data available)
        if 'volume' in df.columns:
            df['volume_sma'] = pd.Series(df['volume']).rolling(20).mean().values
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using parallel processing."""
        logger.info("Adding technical indicators in parallel")
        
        # Split data by symbol for parallel processing
        grouped_data = [(symbol, group) for symbol, group in df.groupby('symbol')]
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self._calculate_indicators_batch, grouped_data))
        
        return pd.concat(results, ignore_index=True)

        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')

        return df

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """Select features based on importance and correlation analysis."""
        logger.info("Performing feature selection")
        
        # Remove non-feature columns
        exclude_cols = ['timestamp', 'symbol', 'target', 'close']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Calculate correlation with target
        df['target'] = df['close'].shift(-1)
        correlations = df[feature_cols + ['target']].corr()['target'].abs()
        
        # Select features with significant correlation
        significant_features = correlations[correlations > 0.1].index.tolist()
        
        # Remove highly correlated features
        correlation_matrix = df[significant_features].corr().abs()
        upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        final_features = [f for f in significant_features if f not in to_drop]
        
        logger.info(f"Selected {len(final_features)} features from {len(feature_cols)} total")
        return final_features

    def train_models(
        self,
        df: pd.DataFrame,
        use_gpu: bool = True,
        distributed: bool = True,
        optimization_config: Optional[Dict] = None
    ) -> Dict[str, Dict]:
        """Train multiple models with distributed processing and advanced optimization.
        
        Args:
            df: Training data
            use_gpu: Whether to use GPU acceleration
            distributed: Whether to use distributed training
            optimization_config: Configuration for advanced optimization
        """
        logger.info("Starting distributed training pipeline with advanced optimization")
        start_time = datetime.now()
        
        # Initialize monitoring
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Default optimization config
        default_config = {
            'early_stopping_rounds': 50,
            'learning_rate_schedule': 'cosine',
            'warmup_epochs': 5,
            'max_epochs': 1000,
            'batch_size': 1024,
            'validation_frequency': 10
        }
        optimization_config = {**default_config, **(optimization_config or {})}
        
        results = {}
        convergence_history = {}
        
        # Initialize distributed training if enabled
        if distributed and torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs for distributed training")
            torch.distributed.init_process_group(backend='nccl')

        try:
            # Enhanced feature selection with parallel processing
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                feature_cols = self._select_features(df)
                df['target'] = df['close'].shift(-1)
                df = df.dropna()
            
            # Advanced scaling with robustness to outliers
            scaler = RobustScaler()
            scaled_features = scaler.fit_transform(df[feature_cols])
            
            # Time series cross-validation with expanding window
            tscv = TimeSeriesSplit(n_splits=5, test_size=len(df) // 10)
            
            # Prepare data for distributed training
            dataset = self._prepare_training_dataset(
                scaled_features,
                df['target'].values,
                optimization_config['batch_size']
            )
            
            # Initialize models for parallel training
            models = {
                'catboost': self._init_catboost_model(optimization_config),
                'lightgbm': self._init_lightgbm_model(optimization_config),
                'xgboost': self._init_xgboost_model(optimization_config)
            }

            # Train models in parallel with cross-validation
            with ProcessPoolExecutor(max_workers=len(models)) as executor:
                future_to_model = {}
                
                for fold, (train_idx, test_idx) in enumerate(tscv.split(dataset)):
                    train_data = self._get_fold_data(dataset, train_idx)
                    test_data = self._get_fold_data(dataset, test_idx)
                    
                    for model_name, model in models.items():
                        future = executor.submit(
                            self._train_model_with_optimization,
                            model=model,
                            model_name=model_name,
                            train_data=train_data,
                            test_data=test_data,
                            feature_names=feature_cols,
                            fold=fold,
                            config=optimization_config
                        )
                        future_to_model[future] = (model_name, fold)
                
                # Collect results
                for future in concurrent.futures.as_completed(future_to_model):
                    model_name, fold = future_to_model[future]
                    try:
                        fold_results = future.result()
                        
                        # Update convergence history
                        if model_name not in convergence_history:
                            convergence_history[model_name] = []
                        convergence_history[model_name].append(fold_results['metrics'])
                        
                        # Track feature importance
                        self.metrics['feature_importance_history'].append({
                            'model': model_name,
                            'fold': fold,
                            'importance': fold_results['feature_importance'],
                            'features': feature_cols
                        })
                        
                        results[f'{model_name}_fold_{fold}'] = fold_results
                    except Exception as e:
                        logger.error(f"Error in {model_name} fold {fold}: {str(e)}")
                
            # Create ensemble model from best performers
            ensemble_results = self._create_ensemble_model(results, dataset)
            results['ensemble'] = ensemble_results

            # Update metrics
            training_time = (datetime.now() - start_time).total_seconds()
            self.metrics.update({
                'training_time': training_time,
                'peak_memory_usage': max(
                    process.memory_info().rss / 1024 / 1024 - initial_memory,
                    0
                ),
                'gpu_memory_usage': (
                    torch.cuda.max_memory_allocated() / 1024 / 1024
                    if use_gpu and torch.cuda.is_available() else None
                ),
                'training_throughput': len(df) / training_time,
                'model_convergence_rate': {
                    model: np.mean([
                        metrics['rmse_history'][-1] / metrics['rmse_history'][0]
                        for metrics in model_history
                    ])
                    for model, model_history in convergence_history.items()
                }
            })

            # Save training summary with enhanced metrics
            self.save_training_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in distributed training: {str(e)}")
            return {}
        finally:
            if distributed and torch.cuda.device_count() > 1:
                torch.distributed.destroy_process_group()

    def _prepare_training_dataset(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        batch_size: int
    ) -> torch.utils.data.DataLoader:
        """Prepare dataset for distributed training."""
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32)
        )
        
        sampler = torch.utils.data.distributed.DistributedSampler(dataset) \
            if torch.distributed.is_initialized() else None
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def _train_model_with_optimization(
        self,
        model: Any,
        model_name: str,
        train_data: torch.utils.data.DataLoader,
        test_data: torch.utils.data.DataLoader,
        feature_names: List[str],
        fold: int,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train a model with advanced optimization strategies."""
        logger.info(f"Training {model_name} (fold {fold}) with optimization")
        
        # Initialize optimizer and scheduler
        optimizer = self._get_optimizer(model, config)
        scheduler = self._get_scheduler(optimizer, config)
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience = config['early_stopping_rounds']
        patience_counter = 0
        loss_history = []
        
        for epoch in range(config['max_epochs']):
            # Training step
            train_loss = self._train_epoch(model, train_data, optimizer)
            
            # Validation step
            if epoch % config['validation_frequency'] == 0:
                val_loss = self._validate_epoch(model, test_data)
                loss_history.append(val_loss)
                
                # Early stopping check
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    self._save_model_checkpoint(model, model_name, fold)
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered for {model_name} (fold {fold})")
                    break
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
        
        # Load best model
        model = self._load_model_checkpoint(model_name, fold)
        
        # Generate predictions and metrics
        predictions = model.predict(test_data)
        metrics = self._calculate_metrics(predictions, test_data)
        
        return {
            'model': model,
            'metrics': metrics,
            'loss_history': loss_history,
            'feature_importance': self._get_feature_importance(model, feature_names)
        }

    def _create_ensemble_model(
        self,
        results: Dict[str, Dict],
        dataset: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """Create an ensemble model from the best performing models."""
        # Select best models based on validation performance
        best_models = self._select_best_models(results)
        
        # Create ensemble
        ensemble = self._create_weighted_ensemble(best_models)
        
        # Evaluate ensemble
        ensemble_metrics = self._evaluate_ensemble(ensemble, dataset)
        
        return {
            'model': ensemble,
            'metrics': ensemble_metrics,
            'component_models': [model.name for model in best_models]
        }

    def train_catboost_model(self, X_train, X_test, y_train, y_test):
        """Train CatBoost model."""
        logger.info("Training CatBoost model")

        from catboost import CatBoostRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function='RMSE',
            random_seed=42,
            verbose=False
        )

        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.models_dir / f"catboost_comprehensive_{timestamp}.cbm"
        model.save_model(str(model_path))

        result = {
            'model_path': str(model_path),
            'training_time': training_time,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            },
            'feature_importance': model.feature_importances_.tolist(),
            'feature_names': X_train.columns.tolist()
        }

        logger.info(".2f")
        logger.info(".6f")
        logger.info(".6f")
        logger.info(".6f")
        logger.info(".6f")

        return result

    def train_xgboost_model(self, X_train, X_test, y_train, y_test):
        """Train XGBoost model."""
        logger.info("Training XGBoost model")

        from xgboost import XGBRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbosity=0
        )

        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.models_dir / f"xgboost_comprehensive_{timestamp}.json"
        model.save_model(str(model_path))

        result = {
            'model_path': str(model_path),
            'training_time': training_time,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            },
            'feature_importance': model.feature_importances_.tolist(),
            'feature_names': X_train.columns.tolist()
        }

        logger.info(".2f")
        logger.info(".6f")
        logger.info(".6f")
        logger.info(".6f")
        logger.info(".6f")

        return result

    def save_training_summary(self, results: Dict[str, Dict]):
        """Save training summary."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'models_trained': list(results.keys()),
            'results': results
        }

        summary_path = self.logs_dir / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Training summary saved to {summary_path}")

    async def run_automated_pipeline(self):
        """Run the complete automated pipeline."""
        logger.info("🚀 Starting Automated Forex AI Training Pipeline")

        try:
            # Step 1: Download data
            logger.info("📥 Step 1: Downloading comprehensive forex data")
            download_success = await self.download_all_data()

            if not download_success:
                logger.warning("Data download had issues, proceeding with existing data")

            # Step 2: Process data
            logger.info("🔄 Step 2: Processing and combining data")
            processed_data = self.process_data()

            if processed_data is None or len(processed_data) == 0:
                logger.error("No data available for training")
                return False

            # Step 3: Train models
            logger.info("🤖 Step 3: Training ML models")
            training_results = self.train_models(processed_data)

            # Step 4: Summary
            logger.info("✅ Pipeline completed successfully!")
            logger.info(f"📊 Data processed: {len(processed_data)} rows")
            logger.info(f"🎯 Models trained: {len(training_results)}")

            for model_name, result in training_results.items():
                logger.info(f"  • {model_name}: R² = {result['metrics']['r2']:.6f}")

            return True

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return False

def main():
    """Main function to run the automated pipeline."""
    pipeline = AutomatedTrainingPipeline()

    # Run the async pipeline
    success = asyncio.run(pipeline.run_automated_pipeline())

    if success:
        print("\n" + "="*60)
        print("🎉 AUTOMATED TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("📁 Check the following directories for results:")
        print("  • Models: models/trained/")
        print("  • Processed data: data/processed/")
        print("  • Logs: logs/")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ AUTOMATED TRAINING PIPELINE FAILED")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()