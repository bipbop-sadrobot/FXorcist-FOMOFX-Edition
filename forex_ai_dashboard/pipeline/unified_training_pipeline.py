#!/usr/bin/env python3
"""
Unified Training Pipeline for FXorcist-FOMOFX-Edition
Consolidated training system with multiple algorithms, optimization, and monitoring.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from forex_ai_dashboard.pipeline.optimized_data_integration import OptimizedDataIntegrator
from forex_ai_dashboard.pipeline.data_format_detector import ForexDataFormatDetector, DataQualityValidator
from memory_system.core import MemoryManager

# Import ML libraries
try:
    from catboost import CatBoostRegressor, Pool
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler, RobustScaler
    import optuna
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost not available, using basic models")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/unified_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UnifiedTrainingPipeline:
    """Unified training pipeline with multiple algorithms and optimization."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._load_default_config()
        self.project_root = Path(__file__).parent.parent.parent

        # Initialize components
        self.data_integrator = OptimizedDataIntegrator()
        self.format_detector = ForexDataFormatDetector()
        self.quality_validator = DataQualityValidator()
        self.memory_manager = MemoryManager()

        # Training state
        self.models = {}
        self.feature_importance = {}
        self.training_history = []
        self.best_model = None
        self.best_score = float('-inf')

        # Setup directories
        self.models_dir = self.project_root / self.config.get('models_dir', 'models')
        self.models_dir.mkdir(exist_ok=True)

        logger.info("Unified Training Pipeline initialized")

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default training configuration."""
        return {
            'models_dir': 'models',
            'cross_validation_folds': 5,
            'test_size': 0.2,
            'random_state': 42,
            'early_stopping_rounds': 50,
            'hyperparameter_optimization': True,
            'n_trials': 50,
            'ensemble_methods': True,
            'feature_selection': True,
            'memory_integration': True,
            'algorithms': ['catboost', 'xgboost', 'lightgbm', 'ensemble']
        }

    def run_complete_training_pipeline(self, data_source: Optional[str] = None) -> Dict[str, Any]:
        """Run the complete training pipeline from data to model deployment."""
        start_time = time.time()

        logger.info("🚀 Starting Unified Training Pipeline")

        try:
            # Step 1: Data Integration
            logger.info("📊 Step 1: Data Integration")
            data_results = self._run_data_integration(data_source)
            if not data_results['success']:
                raise Exception("Data integration failed")

            # Step 2: Data Validation
            logger.info("🔍 Step 2: Data Validation")
            validation_results = self._validate_training_data()
            if not validation_results['quality_score'] >= 0.7:
                logger.warning(f"Low data quality: {validation_results['quality_score']}")

            # Step 3: Feature Engineering
            logger.info("⚙️  Step 3: Feature Engineering")
            feature_data = self._engineer_features()

            # Step 4: Model Training
            logger.info("🤖 Step 4: Model Training")
            training_results = self._train_models(feature_data)

            # Step 5: Model Evaluation
            logger.info("📊 Step 5: Model Evaluation")
            evaluation_results = self._evaluate_models()

            # Step 6: Model Selection & Deployment
            logger.info("🏆 Step 6: Model Selection & Deployment")
            deployment_results = self._deploy_best_model()

            # Step 7: Generate Report
            logger.info("📋 Step 7: Generate Training Report")
            report = self._generate_training_report(
                data_results, validation_results, training_results,
                evaluation_results, deployment_results, time.time() - start_time
            )

            logger.info("✅ Unified Training Pipeline completed successfully")
            return report

        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'stage': 'unknown',
                'duration': time.time() - start_time
            }

    def _run_data_integration(self, data_source: Optional[str] = None) -> Dict[str, Any]:
        """Run optimized data integration."""
        try:
            if data_source:
                # Process specific data source
                logger.info(f"Processing data source: {data_source}")
                # Implementation for specific data source
            else:
                # Run general data integration
                results = self.data_integrator.process_optimized_data()

            return {
                'success': True,
                'processed_files': results.get('processed', 0),
                'skipped_files': results.get('skipped', 0),
                'total_files': results.get('total', 0)
            }

        except Exception as e:
            logger.error(f"Data integration failed: {e}")
            return {'success': False, 'error': str(e)}

    def _validate_training_data(self) -> Dict[str, Any]:
        """Validate training data quality."""
        try:
            # Get processed data files
            processed_dir = self.project_root / "data" / "processed"
            parquet_files = list(processed_dir.glob("*.parquet"))

            if not parquet_files:
                return {'quality_score': 0.0, 'issues': ['No processed data files found']}

            # Validate each file
            total_quality = 0.0
            all_issues = []

            for file_path in parquet_files[:5]:  # Sample first 5 files
                try:
                    df = pd.read_parquet(file_path)
                    symbol = file_path.stem.split('_')[0].upper()
                    quality = self.quality_validator.validate_dataset(df, symbol)

                    total_quality += quality['overall_quality']
                    all_issues.extend(quality.get('issues', []))

                except Exception as e:
                    logger.warning(f"Failed to validate {file_path}: {e}")

            avg_quality = total_quality / max(len(parquet_files[:5]), 1)

            return {
                'quality_score': avg_quality,
                'issues': list(set(all_issues)),  # Remove duplicates
                'files_validated': len(parquet_files[:5])
            }

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return {'quality_score': 0.0, 'issues': [str(e)]}

    def _engineer_features(self) -> pd.DataFrame:
        """Engineer features for training."""
        try:
            # Load processed data
            processed_dir = self.project_root / "data" / "processed"
            all_data = []

            for parquet_file in processed_dir.glob("*.parquet"):
                try:
                    df = pd.read_parquet(parquet_file)
                    if len(df) > 100:  # Minimum data requirement
                        all_data.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {parquet_file}: {e}")

            if not all_data:
                raise Exception("No valid training data found")

            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)

            # Basic feature engineering
            combined_df = self._add_technical_features(combined_df)
            combined_df = self._add_temporal_features(combined_df)
            combined_df = self._add_memory_features(combined_df)

            # Prepare for training
            feature_cols = [col for col in combined_df.columns
                          if col not in ['timestamp', 'symbol', 'target']]

            # Create target (next period return)
            if 'close' in combined_df.columns:
                combined_df['target'] = combined_df['close'].shift(-1) / combined_df['close'] - 1
                combined_df = combined_df.dropna()

            logger.info(f"Feature engineering completed: {len(feature_cols)} features, {len(combined_df)} samples")

            return combined_df

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis features."""
        if 'close' not in df.columns:
            return df

        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ma_{period}_slope'] = df[f'ma_{period}'].diff()

        # RSI
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        df['rsi_14'] = calculate_rsi(df['close'])

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Volatility
        df['returns'] = df['close'].pct_change()
        df['volatility_20'] = df['returns'].rolling(window=20).std()

        return df.fillna(method='bfill').fillna(method='ffill')

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features."""
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['quarter'] = df['timestamp'].dt.quarter

            # Trading session features
            df['is_london_session'] = df['hour'].between(8, 16)
            df['is_new_york_session'] = df['hour'].between(14, 21)
            df['is_tokyo_session'] = df['hour'].between(0, 9) | (df['hour'] >= 23)

        return df

    def _add_memory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add memory-based features."""
        if not self.config.get('memory_integration', True):
            return df

        try:
            # Get memory insights
            insights = self.memory_manager.generate_insights_report()

            # Add memory-based features (simplified)
            df['memory_confidence'] = insights.get('average_confidence', 0.5)
            df['pattern_strength'] = insights.get('pattern_strength', 0.5)

        except Exception as e:
            logger.warning(f"Memory feature integration failed: {e}")

        return df

    def _train_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train multiple models with optimization."""
        results = {}

        # Prepare data
        feature_cols = [col for col in data.columns
                       if col not in ['timestamp', 'symbol', 'target'] and
                       data[col].dtype in ['float64', 'int64']]

        X = data[feature_cols].fillna(0)
        y = data['target'].fillna(0)

        # Split data (time series split)
        train_size = int(len(X) * (1 - self.config.get('test_size', 0.2)))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train models in parallel
        algorithms = self.config.get('algorithms', ['catboost'])

        with ThreadPoolExecutor(max_workers=min(len(algorithms), 4)) as executor:
            futures = {}

            for algorithm in algorithms:
                if algorithm == 'catboost' and CATBOOST_AVAILABLE:
                    future = executor.submit(self._train_catboost, X_train_scaled, y_train, X_test_scaled, y_test)
                    futures[future] = 'catboost'
                elif algorithm == 'xgboost' and XGBOOST_AVAILABLE:
                    future = executor.submit(self._train_xgboost, X_train_scaled, y_train, X_test_scaled, y_test)
                    futures[future] = 'xgboost'
                elif algorithm == 'lightgbm' and LIGHTGBM_AVAILABLE:
                    future = executor.submit(self._train_lightgbm, X_train_scaled, y_train, X_test_scaled, y_test)
                    futures[future] = 'lightgbm'

            # Collect results
            for future in as_completed(futures):
                algorithm = futures[future]
                try:
                    model_result = future.result()
                    results[algorithm] = model_result

                    # Track best model
                    if model_result['test_score'] > self.best_score:
                        self.best_score = model_result['test_score']
                        self.best_model = model_result['model']
                        self.models[algorithm] = model_result['model']

                except Exception as e:
                    logger.error(f"Training failed for {algorithm}: {e}")

        return results

    def _train_catboost(self, X_train, y_train, X_test, y_test):
        """Train CatBoost model with optimization."""
        if not CATBOOST_AVAILABLE:
            raise Exception("CatBoost not available")

        # Hyperparameter optimization
        if self.config.get('hyperparameter_optimization', True):
            best_params = self._optimize_catboost_params(X_train, y_train)
        else:
            best_params = {
                'iterations': 1000,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3,
                'verbose': False
            }

        # Train model
        model = CatBoostRegressor(**best_params)
        model.fit(X_train, y_train, verbose=False)

        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_score = r2_score(y_train, train_pred)
        test_score = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

        # Feature importance
        feature_importance = dict(zip(range(len(X_train[0])), model.feature_importances_))

        return {
            'model': model,
            'train_score': train_score,
            'test_score': test_score,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'feature_importance': feature_importance,
            'best_params': best_params
        }

    def _optimize_catboost_params(self, X_train, y_train):
        """Optimize CatBoost hyperparameters."""
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'verbose': False
            }

            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train, verbose=False)

            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
            return cv_scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.get('n_trials', 30))

        return study.best_params

    def _train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model."""
        if not XGBOOST_AVAILABLE:
            raise Exception("XGBoost not available")

        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        return {
            'model': model,
            'train_score': r2_score(y_train, train_pred),
            'test_score': r2_score(y_test, test_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'feature_importance': dict(enumerate(model.feature_importances_))
        }

    def _train_lightgbm(self, X_train, y_train, X_test, y_test):
        """Train LightGBM model."""
        if not LIGHTGBM_AVAILABLE:
            raise Exception("LightGBM not available")

        model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        return {
            'model': model,
            'train_score': r2_score(y_train, train_pred),
            'test_score': r2_score(y_test, test_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'feature_importance': dict(enumerate(model.feature_importances_))
        }

    def _evaluate_models(self) -> Dict[str, Any]:
        """Evaluate trained models."""
        evaluation_results = {}

        for model_name, model_info in self.models.items():
            model = model_info['model']

            # Additional evaluation metrics
            evaluation_results[model_name] = {
                'r2_score': model_info.get('test_score', 0),
                'mae': model_info.get('test_mae', 0),
                'rmse': model_info.get('test_rmse', 0),
                'feature_importance': model_info.get('feature_importance', {}),
                'best_params': model_info.get('best_params', {})
            }

        return evaluation_results

    def _deploy_best_model(self) -> Dict[str, Any]:
        """Deploy the best performing model."""
        if not self.best_model:
            return {'success': False, 'error': 'No models trained'}

        try:
            # Save best model
            model_path = self.models_dir / f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

            import joblib
            joblib.dump(self.best_model, model_path)

            # Save model metadata
            metadata = {
                'model_type': type(self.best_model).__name__,
                'best_score': self.best_score,
                'training_date': datetime.now().isoformat(),
                'feature_importance': self.feature_importance
            }

            metadata_path = model_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            logger.info(f"Best model deployed: {model_path}")

            return {
                'success': True,
                'model_path': str(model_path),
                'metadata_path': str(metadata_path),
                'best_score': self.best_score
            }

        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            return {'success': False, 'error': str(e)}

    def _generate_training_report(self, data_results, validation_results,
                                training_results, evaluation_results,
                                deployment_results, duration) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        report = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'data_integration': data_results,
            'data_validation': validation_results,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'deployment_results': deployment_results,
            'best_model_score': self.best_score,
            'models_trained': list(self.models.keys()),
            'config': self.config
        }

        # Save report
        report_path = self.project_root / "logs" / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Training report saved: {report_path}")

        return report

def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description="Unified Training Pipeline")
    parser.add_argument('--data-source', type=str, help='Specific data source to process')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--quick', action='store_true', help='Quick training mode')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Load configuration
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)

    if args.quick:
        config.update({
            'hyperparameter_optimization': False,
            'cross_validation_folds': 3,
            'algorithms': ['catboost']
        })

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run training pipeline
    pipeline = UnifiedTrainingPipeline(config)
    results = pipeline.run_complete_training_pipeline(args.data_source)

    if results['success']:
        print("✅ Training completed successfully!")
        print(f"Best model score: {results.get('best_model_score', 'N/A')}")
        print(f"Models trained: {results.get('models_trained', [])}")
        print(".2f")
    else:
        print(f"❌ Training failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()