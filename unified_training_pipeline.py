#!/usr/bin/env python3
"""
Unified Forex AI Training Pipeline
Combines features from automated, comprehensive, and focused pipelines:
- Distributed training and resource management
- Advanced ML features and interpretability
- Optimized forex-specific parameters
- Efficient data handling
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
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import asyncio
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from memory_profiler import profile

# Import our enhanced modules
from forex_ai_dashboard.pipeline.enhanced_feature_engineering import EnhancedFeatureEngineer
from forex_ai_dashboard.pipeline.hyperparameter_optimization import HyperparameterOptimizer
from forex_ai_dashboard.pipeline.model_comparison import ModelComparator
from forex_ai_dashboard.pipeline.model_interpretability import ModelInterpreter
from forex_ai_dashboard.models.catboost_model import CatBoostModel
from forex_ai_dashboard.pipeline.unified_feature_engineering import UnifiedFeatureEngineer
from dashboard.utils.enhanced_data_loader import EnhancedDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/unified_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UnifiedTrainingPipeline:
    """Unified pipeline combining features from all versions."""

    def __init__(self, synthesis_config: Optional[Dict[str, float]] = None):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models" / "trained"
        self.logs_dir = self.project_root / "logs"
        
        # Initialize components
        self.feature_engineer = EnhancedFeatureEngineer()
        self.unified_engineer = UnifiedFeatureEngineer()
        self.model_comparator = ModelComparator()
        self.interpreter = ModelInterpreter()
        self.data_loader = EnhancedDataLoader(num_workers=os.cpu_count())
        
        if synthesis_config:
            self.data_loader.synthesis_config.update(synthesis_config)
        
        # Initialize GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = os.cpu_count()
        
        # Create directories
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.symbols = ["EURUSD", "GBPUSD"]  # Focus on major pairs
        self.years = [2023, 2024]  # Recent data by default
        self.months = list(range(1, 13))
        
        # Comprehensive metrics tracking
        self.metrics = {
            'data_processing_time': 0,
            'training_time': 0,
            'data_synthesis_time': 0,
            'validation_time': 0,
            'peak_memory_usage': 0,
            'gpu_memory_usage': 0 if torch.cuda.is_available() else None,
            'cpu_utilization': [],
            'synthetic_data_ratio': 0,
            'edge_cases_generated': 0,
            'data_quality_score': 0,
            'pattern_diversity': 0,
            'training_throughput': 0,
            'model_convergence_rate': {},
            'feature_importance_history': []
        }

    async def download_data(self, comprehensive: bool = False) -> bool:
        """Download forex data with flexible scope."""
        symbols = self.symbols
        years = self.years
        
        if comprehensive:
            symbols = [
                "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
                "USDCHF", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY"
            ]
            years = list(range(2020, 2025))
        
        logger.info(f"Downloading data for {len(symbols)} symbols, years {years}")
        
        success_count = 0
        total_attempts = 0
        
        for symbol in symbols:
            for year in years:
                for month in self.months:
                    total_attempts += 1
                    
                    # Skip if file exists
                    csv_path = self.data_dir / "raw" / "histdata" / symbol / str(year) / f"{month:02d}.csv"
                    if csv_path.exists():
                        success_count += 1
                        continue
                    
                    try:
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
                            timeout=30
                        )
                        
                        if result.returncode == 0:
                            success_count += 1
                            
                    except Exception as e:
                        logger.debug(f"Error downloading {symbol} {year}-{month}: {str(e)}")
                    
                    await asyncio.sleep(0.5)  # Rate limiting
        
        return success_count > 0

    def process_data(self, augment_data: bool = True) -> Optional[pd.DataFrame]:
        """Process data with enhanced features and synthetic data generation."""
        logger.info("Starting enhanced data processing")
        process_start_time = datetime.now()
        
        try:
            # Load and process data
            data_load_start = datetime.now()
            base_data = self.data_loader.load_forex_data(
                timeframe="1H",
                augment_data=augment_data
            )[0]
            
            self.metrics['data_load_time'] = (datetime.now() - data_load_start).total_seconds()
            
            if augment_data:
                # Generate synthetic data
                synthesis_start = datetime.now()
                synthetic_data = self.data_loader.generate_synthetic_data(
                    base_data,
                    num_samples=int(len(base_data) * 0.3),
                    include_edge_cases=True
                )
                
                self.metrics['data_synthesis_time'] = (datetime.now() - synthesis_start).total_seconds()
                self.metrics['synthetic_data_ratio'] = len(synthetic_data) / len(base_data)
                
                # Combine real and synthetic data
                combined_data = pd.concat([base_data, synthetic_data])
                combined_data = combined_data.sort_index()
            else:
                combined_data = base_data
            
            # Enhanced feature engineering
            feature_groups = [
                'basic', 'momentum', 'volatility', 'trend', 'time',
                'microstructure', 'advanced_momentum', 'advanced_trend',
                'volume_advanced', 'statistical'
            ]
            
            processed_data = self.unified_engineer.process_data(
                combined_data,
                feature_groups=feature_groups
            )
            
            # Save processed data
            output_path = self.data_dir / "processed" / "unified_forex_data.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            processed_data.to_parquet(output_path)
            
            self.metrics['data_processing_time'] = (datetime.now() - process_start_time).total_seconds()
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
            return None

    def train_models(
        self,
        df: pd.DataFrame,
        optimize_hyperparams: bool = True,
        use_ensemble: bool = True,
        distributed: bool = True
    ) -> Dict[str, Dict]:
        """Train multiple models with distributed processing and optimization."""
        logger.info("Starting distributed training pipeline")
        start_time = datetime.now()
        
        # Initialize monitoring
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        results = {}
        
        try:
            # Prepare features and target
            feature_cols = [col for col in df.columns if col not in [
                'timestamp', 'symbol', 'target'
            ] and not col.startswith(('year', 'month'))]
            
            df['target'] = df['close'].shift(-1)
            df = df.dropna()
            
            # Split data
            split_idx = int(len(df) * 0.8)
            train_df = df[:split_idx]
            test_df = df[split_idx:]
            
            X_train = train_df[feature_cols]
            X_test = test_df[feature_cols]
            y_train = train_df['target']
            y_test = test_df['target']
            
            # Initialize distributed training
            if distributed and torch.cuda.device_count() > 1:
                logger.info(f"Using {torch.cuda.device_count()} GPUs")
                torch.distributed.init_process_group(backend='nccl')
            
            # Train models in parallel
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # CatBoost with forex-optimized parameters
                if optimize_hyperparams:
                    optimizer = HyperparameterOptimizer()
                    catboost_params = optimizer.optimize_catboost(X_train, X_test, y_train, y_test)
                else:
                    catboost_params = {
                        'iterations': 1000,
                        'learning_rate': 0.03,
                        'depth': 8,
                        'l2_leaf_reg': 1.0,
                        'grow_policy': 'SymmetricTree',
                        'min_data_in_leaf': 100,
                        'max_leaves': 64
                    }
                
                futures = []
                
                # CatBoost
                futures.append(
                    executor.submit(
                        self._train_catboost,
                        X_train, X_test, y_train, y_test,
                        catboost_params
                    )
                )
                
                # Additional models for ensemble
                if use_ensemble:
                    futures.extend([
                        executor.submit(self._train_lightgbm, X_train, X_test, y_train, y_test),
                        executor.submit(self._train_xgboost, X_train, X_test, y_train, y_test)
                    ])
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    model_result = future.result()
                    if model_result:
                        results[model_result['model_name']] = model_result
            
            # Create ensemble if multiple models
            if len(results) > 1:
                ensemble_result = self._create_ensemble(results, X_test, y_test)
                results['ensemble'] = ensemble_result
            
            # Update metrics
            self.metrics.update({
                'training_time': (datetime.now() - start_time).total_seconds(),
                'peak_memory_usage': max(
                    process.memory_info().rss / 1024 / 1024 - initial_memory,
                    0
                ),
                'gpu_memory_usage': (
                    torch.cuda.max_memory_allocated() / 1024 / 1024
                    if torch.cuda.is_available() else None
                )
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            return {}
        finally:
            if distributed and torch.cuda.device_count() > 1:
                torch.distributed.destroy_process_group()

    def _train_catboost(self, X_train, X_test, y_train, y_test, params: Dict) -> Dict:
        """Train CatBoost model with optimized parameters."""
        from catboost import CatBoostRegressor
        
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_test, y_test))
        
        return self._evaluate_model(model, X_test, y_test, 'catboost')

    def _train_lightgbm(self, X_train, X_test, y_train, y_test) -> Dict:
        """Train LightGBM model."""
        import lightgbm as lgb
        
        model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        return self._evaluate_model(model, X_test, y_test, 'lightgbm')

    def _train_xgboost(self, X_train, X_test, y_train, y_test) -> Dict:
        """Train XGBoost model."""
        from xgboost import XGBRegressor
        
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        return self._evaluate_model(model, X_test, y_test, 'xgboost')

    def _evaluate_model(self, model: Any, X_test, y_test, model_name: str) -> Dict:
        """Evaluate model performance."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        y_pred = model.predict(X_test)
        
        return {
            'model_name': model_name,
            'model': model,
            'metrics': {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
        }

    def _create_ensemble(self, models: Dict, X_test, y_test) -> Dict:
        """Create ensemble from trained models."""
        predictions = []
        weights = []
        
        for model_name, model_dict in models.items():
            if model_name != 'ensemble':
                pred = model_dict['model'].predict(X_test)
                predictions.append(pred)
                weights.append(model_dict['metrics']['r2'])
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average prediction
        ensemble_pred = np.average(predictions, weights=weights, axis=0)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        return {
            'model_name': 'ensemble',
            'metrics': {
                'mse': mean_squared_error(y_test, ensemble_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
                'mae': mean_absolute_error(y_test, ensemble_pred),
                'r2': r2_score(y_test, ensemble_pred)
            },
            'weights': dict(zip(models.keys(), weights))
        }

    async def run_unified_pipeline(
        self,
        comprehensive_data: bool = False,
        optimize_hyperparams: bool = True,
        use_ensemble: bool = True,
        enable_interpretability: bool = True
    ) -> bool:
        """Run the complete unified pipeline."""
        logger.info("🚀 Starting Unified Forex AI Training Pipeline")
        
        try:
            # Step 1: Download data
            logger.info("📥 Step 1: Downloading forex data")
            download_success = await self.download_data(comprehensive=comprehensive_data)
            
            if not download_success:
                logger.warning("Data download had issues, proceeding with existing data")
            
            # Step 2: Process data
            logger.info("🔄 Step 2: Processing and engineering features")
            processed_data = self.process_data(augment_data=True)
            
            if processed_data is None or len(processed_data) == 0:
                logger.error("No data available for training")
                return False
            
            # Step 3: Train models
            logger.info("🤖 Step 3: Training ML models")
            training_results = self.train_models(
                processed_data,
                optimize_hyperparams=optimize_hyperparams,
                use_ensemble=use_ensemble
            )
            
            # Step 4: Model interpretability
            if enable_interpretability and training_results:
                logger.info("🔍 Step 4: Generating model interpretability")
                interpretability_results = self.interpreter.explain_models(
                    training_results,
                    processed_data
                )
                training_results['interpretability'] = interpretability_results
            
            # Step 5: Save results
            logger.info("💾 Step 5: Saving results")
            self.save_training_summary(training_results)
            
            logger.info("✅ Unified pipeline completed successfully!")
            logger.info(f"📊 Data processed: {len(processed_data)} rows")
            logger.info(f"🎯 Models trained: {len(training_results)}")
            
            for model_name, result in training_results.items():
                if model_name != 'interpretability':
                    metrics = result.get('metrics', {})
                    logger.info(f"  • {model_name}: R² = {metrics.get('r2', 0):.6f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return False

    def save_training_summary(self, results: Dict[str, Dict]):
        """Save comprehensive training summary."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics,
            'models': results
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.logs_dir / f"unified_training_summary_{timestamp}.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Training summary saved to {summary_path}")

def main():
    """Main function to run the unified pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Forex AI Training Pipeline')
    parser.add_argument('--comprehensive', action='store_true', help='Use comprehensive data')
    parser.add_argument('--optimize', action='store_true', help='Enable hyperparameter optimization')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble methods')
    parser.add_argument('--interpretability', action='store_true', help='Enable model interpretability')
    
    args = parser.parse_args()
    
    pipeline = UnifiedTrainingPipeline()
    
    success = asyncio.run(pipeline.run_unified_pipeline(
        comprehensive_data=args.comprehensive,
        optimize_hyperparams=args.optimize,
        use_ensemble=args.ensemble,
        enable_interpretability=args.interpretability
    ))
    
    if success:
        print("\n" + "="*60)
        print("🎉 UNIFIED TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("📁 Check the following directories for results:")
        print("  • Models: models/trained/")
        print("  • Processed data: data/processed/")
        print("  • Logs: logs/")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ UNIFIED TRAINING PIPELINE FAILED")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()