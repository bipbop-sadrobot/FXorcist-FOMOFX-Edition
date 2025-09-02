#!/usr/bin/env python3
"""
Advanced Forex AI Training Pipeline
Enhanced with hyperparameter optimization, ensemble methods, and comprehensive evaluation.
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import asyncio
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('.')

from forex_ai_dashboard.pipeline.unified_feature_engineering import UnifiedFeatureEngineer
from forex_ai_dashboard.pipeline.hyperparameter_optimization import HyperparameterOptimizer
from forex_ai_dashboard.models.catboost_model import CatBoostModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/advanced_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedTrainingPipeline:
    """Advanced training pipeline with optimization and ensemble methods."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models" / "trained"
        self.logs_dir = self.project_root / "logs"

        # Create directories
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Feature engineer
        self.feature_engineer = UnifiedFeatureEngineer()

        # Symbols to focus on
        self.symbols = ["EURUSD", "GBPUSD"]
        self.years = [2023, 2024]

    def load_and_prepare_data(self) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """Load and prepare data for training."""
        logger.info("Loading and preparing data for advanced training")

        # Try to load processed data first
        processed_file = self.data_dir / "processed" / "focused_forex_data.parquet"
        if processed_file.exists():
            logger.info(f"Loading existing processed data from {processed_file}")
            df = pd.read_parquet(processed_file)
        else:
            logger.info("No processed data found, processing raw data")
            df = self._process_raw_data()

        if df is None or len(df) == 0:
            logger.error("No data available for training")
            return None

        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in [
            'timestamp', 'symbol', 'target', 'returns'
        ] and not col.startswith(('year', 'month'))]

        df['target'] = df['close'].shift(-1)  # 1-step ahead prediction
        df = df.dropna(subset=['target'])

        # Remove features with too many NaN values
        nan_threshold = 0.1
        valid_features = []
        for col in feature_cols:
            nan_ratio = df[col].isnull().sum() / len(df)
            if nan_ratio < nan_threshold:
                valid_features.append(col)
            else:
                logger.warning(f"Dropping feature {col} with {nan_ratio:.1%} NaN values")

        # Fill remaining NaN values
        df[valid_features] = df[valid_features].fillna(method='bfill').fillna(method='ffill')

        X = df[valid_features]
        y = df['target']

        logger.info(f"Prepared {len(X)} samples with {len(valid_features)} features")

        # Split data (80/20 train/test, time-based)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test

    def _process_raw_data(self) -> Optional[pd.DataFrame]:
        """Process raw forex data."""
        logger.info("Processing raw forex data")

        all_data = []

        for symbol in self.symbols:
            symbol_data = []

            for year in self.years:
                for month in range(1, 13):
                    csv_path = self.data_dir / "raw" / "histdata" / symbol / str(year) / "02d"

                    if csv_path.exists():
                        try:
                            df = pd.read_csv(csv_path)
                            df['symbol'] = symbol
                            df['year'] = year
                            df['month'] = month
                            symbol_data.append(df)
                        except Exception as e:
                            logger.warning(f"Error loading {csv_path}: {str(e)}")

            if symbol_data:
                symbol_df = pd.concat(symbol_data, ignore_index=True)
                all_data.append(symbol_df)
                logger.info(f"Combined {symbol}: {len(symbol_df)} total rows")

        if not all_data:
            return None

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Enhanced preprocessing
        combined_df = self.feature_engineer.process_data(
            combined_df,
            feature_groups=['basic', 'momentum', 'volatility', 'trend', 'time']
        )

        # Save processed data
        output_path = self.data_dir / "processed" / "focused_forex_data.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_parquet(output_path)

        return combined_df

    def train_optimized_catboost(self, X_train, X_test, y_train, y_test) -> Dict:
        """Train optimized CatBoost model."""
        logger.info("Training optimized CatBoost model")

        # Hyperparameter optimization
        optimizer = HyperparameterOptimizer(
            study_name="catboost_optimization",
            n_trials=30,  # Reduced for faster training
            timeout=600   # 10 minutes
        )

        # Run optimization
        opt_results = optimizer.optimize_catboost(X_train, X_test, y_train, y_test)

        # Train final model with best parameters
        best_params = opt_results['best_params']
        best_params.update({
            'loss_function': 'RMSE',
            'random_seed': 42,
            'verbose': False,
            'early_stopping_rounds': 100
        })

        model = CatBoostModel(**best_params)
        model.train(X_train, y_train, X_test, y_test)

        # Evaluate
        y_pred = model.predict(X_test)
        metrics = model.evaluate(y_test, y_pred)

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.models_dir / f"catboost_optimized_{timestamp}.cbm"
        model.save(str(model_path))

        result = {
            'model_type': 'catboost_optimized',
            'model_path': str(model_path),
            'metrics': metrics,
            'best_params': best_params,
            'optimization_results': opt_results,
            'feature_importance': dict(zip(X_train.columns, model.model.feature_importances_))
        }

        logger.info(".4f")
        return result

    def train_lightgbm_model(self, X_train, X_test, y_train, y_test) -> Optional[Dict]:
        """Train LightGBM model with optimization."""
        logger.info("Training LightGBM model")

        try:
            import lightgbm as lgb
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            # Hyperparameter optimization
            optimizer = HyperparameterOptimizer(
                study_name="lightgbm_optimization",
                n_trials=20,
                timeout=300
            )

            opt_results = optimizer.optimize_lightgbm(X_train, X_test, y_train, y_test)

            # Train final model
            best_params = opt_results['best_params']
            best_params.update({
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1
            })

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

            model = lgb.train(
                best_params,
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )

            # Evaluate
            y_pred = model.predict(X_test)
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }

            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.models_dir / f"lightgbm_optimized_{timestamp}.txt"
            model.save_model(str(model_path))

            result = {
                'model_type': 'lightgbm_optimized',
                'model_path': str(model_path),
                'metrics': metrics,
                'best_params': best_params,
                'optimization_results': opt_results,
                'feature_importance': dict(zip(X_train.columns, model.feature_importance()))
            }

            logger.info(".4f")
            return result

        except ImportError:
            logger.warning("LightGBM not available, skipping")
            return None

    def train_ensemble_model(self, X_train, X_test, y_train, y_test) -> Dict:
        """Train ensemble model."""
        logger.info("Training ensemble model")

        from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # Hyperparameter optimization for ensemble
        optimizer = HyperparameterOptimizer(
            study_name="ensemble_optimization",
            n_trials=15,
            timeout=300
        )

        opt_results = optimizer.optimize_ensemble(X_train, X_test, y_train, y_test)

        # Train final ensemble
        best_params = opt_results['best_params']

        rf_params = {k.replace('rf_', ''): v for k, v in best_params.items() if k.startswith('rf_')}
        et_params = {k.replace('et_', ''): v for k, v in best_params.items() if k.startswith('et_')}

        rf_model = RandomForestRegressor(**rf_params, random_state=42)
        et_model = ExtraTreesRegressor(**et_params, random_state=42)

        rf_model.fit(X_train, y_train)
        et_model.fit(X_train, y_train)

        # Ensemble predictions
        rf_pred = rf_model.predict(X_test)
        et_pred = et_model.predict(X_test)
        rf_weight = best_params['rf_weight']
        et_weight = best_params['et_weight']

        y_pred = rf_weight * rf_pred + et_weight * et_pred

        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

        # Save models
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rf_path = self.models_dir / f"ensemble_rf_{timestamp}.pkl"
        et_path = self.models_dir / f"ensemble_et_{timestamp}.pkl"

        import joblib
        joblib.dump(rf_model, rf_path)
        joblib.dump(et_model, et_path)

        result = {
            'model_type': 'ensemble',
            'model_paths': {'rf': str(rf_path), 'et': str(et_path)},
            'weights': {'rf': rf_weight, 'et': et_weight},
            'metrics': metrics,
            'best_params': best_params,
            'optimization_results': opt_results,
            'feature_importance': {
                'rf': dict(zip(X_train.columns, rf_model.feature_importances_)),
                'et': dict(zip(X_train.columns, et_model.feature_importances_))
            }
        }

        logger.info(".4f")
        return result

    def run_advanced_training(self) -> Dict:
        """Run the complete advanced training pipeline."""
        logger.info("🚀 Starting Advanced Forex AI Training Pipeline")

        try:
            # Load and prepare data
            data = self.load_and_prepare_data()
            if data is None:
                return {'error': 'No data available for training'}

            X_train, X_test, y_train, y_test = data

            # Train multiple models
            results = {}

            # CatBoost (optimized)
            results['catboost'] = self.train_optimized_catboost(X_train, X_test, y_train, y_test)

            # LightGBM (if available)
            lgb_result = self.train_lightgbm_model(X_train, X_test, y_train, y_test)
            if lgb_result:
                results['lightgbm'] = lgb_result

            # Ensemble
            results['ensemble'] = self.train_ensemble_model(X_train, X_test, y_train, y_test)

            # Compare and select best model
            best_model = self._select_best_model(results)

            # Save comprehensive results
            self._save_training_results(results, best_model)

            logger.info("✅ Advanced training pipeline completed successfully!")
            logger.info(f"🎯 Best model: {best_model}")
            logger.info(".4f")

            return {
                'success': True,
                'results': results,
                'best_model': best_model,
                'data_shape': (len(X_train), len(X_test))
            }

        except Exception as e:
            logger.error(f"Advanced training failed: {str(e)}", exc_info=True)
            return {'error': str(e)}

    def _select_best_model(self, results: Dict) -> str:
        """Select the best performing model."""
        best_model = None
        best_score = -float('inf')

        for model_name, result in results.items():
            if 'metrics' in result and 'r2' in result['metrics']:
                r2_score = result['metrics']['r2']
                if r2_score > best_score:
                    best_score = r2_score
                    best_model = model_name

        return best_model

    def _save_training_results(self, results: Dict, best_model: str):
        """Save comprehensive training results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        summary = {
            'timestamp': datetime.now().isoformat(),
            'best_model': best_model,
            'models_trained': list(results.keys()),
            'results': results,
            'data_info': {
                'symbols': self.symbols,
                'years': self.years
            }
        }

        # Save summary
        summary_path = self.logs_dir / f"advanced_training_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Save detailed results for each model
        for model_name, result in results.items():
            model_path = self.logs_dir / f"{model_name}_results_{timestamp}.json"
            with open(model_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)

        logger.info(f"Training results saved to {summary_path}")

def main():
    """Main function to run advanced training pipeline."""
    pipeline = AdvancedTrainingPipeline()
    results = pipeline.run_advanced_training()

    if results.get('success'):
        print("\n" + "="*60)
        print("🎉 ADVANCED FOREX AI TRAINING COMPLETED!")
        print("="*60)
        print(f"📊 Best Model: {results['best_model']}")
        print(f"📁 Models saved in: models/trained/")
        print(f"📋 Results saved in: logs/")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ ADVANCED TRAINING FAILED")
        print("="*60)
        print(f"Error: {results.get('error', 'Unknown error')}")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()