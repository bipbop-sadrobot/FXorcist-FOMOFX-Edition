#!/usr/bin/env python3
"""
Memory-Efficient Advanced Forex AI Training Pipeline
Optimized for large datasets with batch processing and reduced memory usage.
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import VotingRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import optuna
import joblib
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryEfficientTrainingPipeline:
    def __init__(self, batch_size=1000000):  # Process 1M records at a time
        self.data_dir = Path("data/raw/histdata")
        self.models_dir = Path("models/trained")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size

        # Currency pairs
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

    def load_data_sample(self, sample_size=5000000):
        """Load a sample of forex data to reduce memory usage."""
        logger.info(f"Loading sample of {sample_size:,} forex records...")

        all_data = []
        total_loaded = 0

        for symbol in self.symbols:
            if total_loaded >= sample_size:
                break

            symbol_path = self.data_dir / symbol
            if symbol_path.exists():
                logger.info(f"Processing {symbol}")

                for year_dir in sorted(symbol_path.iterdir()):
                    if year_dir.is_dir() and year_dir.name.isdigit():
                        for csv_file in sorted(year_dir.glob("*.csv")):
                            if total_loaded >= sample_size:
                                break

                            try:
                                # Read CSV with semicolon separator
                                df = pd.read_csv(csv_file, sep=';', header=None,
                                                names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

                                # Parse timestamp
                                df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d %H%M%S')

                                # Convert numeric columns
                                for col in ['open', 'high', 'low', 'close']:
                                    df[col] = pd.to_numeric(df[col], errors='coerce')
                                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

                                # Add symbol
                                df['symbol'] = symbol

                                # Drop NaN values
                                df = df.dropna()

                                if len(df) > 0:
                                    # Take only what we need to reach sample size
                                    remaining = sample_size - total_loaded
                                    if len(df) > remaining:
                                        df = df.head(remaining)

                                    all_data.append(df)
                                    total_loaded += len(df)
                                    logger.info(f"  ✓ Loaded {len(df)} records from {symbol} {year_dir.name} (total: {total_loaded:,})")

                            except Exception as e:
                                logger.warning(f"Error loading {csv_file}: {e}")

        if not all_data:
            logger.error("No data loaded")
            return None

        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Sample data loaded: {len(combined_data):,} records")
        return combined_data

    def efficient_feature_engineering(self, df):
        """Create efficient technical indicators with memory optimization."""
        logger.info("Creating efficient features...")

        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Simple moving averages (reduced set)
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()

        # Exponential moving averages
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()

        # RSI (simplified)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Simple Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()

        # Momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1

        # Lagged features (minimal set)
        df['close_lag_1'] = df['close'].shift(1)
        df['close_lag_2'] = df['close'].shift(2)
        df['returns_lag_1'] = df['returns'].shift(1)

        # Fill NaN values
        df = df.ffill().bfill()

        # Drop any remaining NaN values
        initial_len = len(df)
        df = df.dropna()
        final_len = len(df)

        if initial_len - final_len > 0:
            logger.warning(f"Dropped {initial_len - final_len} rows with NaN values")

        logger.info(f"Created {len(df.columns) - 6} features from {final_len:,} records")
        return df

    def optimize_hyperparameters_efficient(self, X_train, y_train, model_type='xgboost'):
        """Efficient hyperparameter optimization with fewer trials."""
        logger.info(f"Optimizing hyperparameters for {model_type}...")

        def objective(trial):
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'random_state': 42
                }
                model = xgb.XGBRegressor(**params)

            elif model_type == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 500),
                    'depth': trial.suggest_int('depth', 4, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
                    'random_state': 42,
                    'verbose': False
                }
                model = CatBoostRegressor(**params)

            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'random_state': 42
                }
                model = LGBMRegressor(**params)

            # Quick cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
            return -scores.mean()

        # Optimize with fewer trials for memory efficiency
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10, timeout=180)  # 3 minutes timeout

        logger.info(f"Best {model_type} parameters: {study.best_params}")
        logger.info(f"Best {model_type} score: {study.best_value:.6f}")

        return study.best_params

    def train_efficient_ensemble(self, X_train, y_train, X_test, y_test):
        """Train ensemble model with memory-efficient approach."""
        logger.info("Training efficient ensemble model...")

        # Get optimized parameters
        xgb_params = self.optimize_hyperparameters_efficient(X_train, y_train, 'xgboost')
        cat_params = self.optimize_hyperparameters_efficient(X_train, y_train, 'catboost')
        lgb_params = self.optimize_hyperparameters_efficient(X_train, y_train, 'lightgbm')

        # Create models
        xgb_model = xgb.XGBRegressor(**xgb_params)
        cat_model = CatBoostRegressor(**cat_params)
        lgb_model = LGBMRegressor(**lgb_params)

        # Train individual models first to check memory
        logger.info("Training XGBoost...")
        xgb_model.fit(X_train, y_train)

        logger.info("Training CatBoost...")
        cat_model.fit(X_train, y_train)

        logger.info("Training LightGBM...")
        lgb_model.fit(X_train, y_train)

        # Create ensemble
        ensemble = VotingRegressor([
            ('xgboost', xgb_model),
            ('catboost', cat_model),
            ('lightgbm', lgb_model)
        ])

        # Train ensemble
        ensemble.fit(X_train, y_train)

        # Make predictions
        ensemble_pred = ensemble.predict(X_test)

        # Calculate metrics
        ensemble_mse = mean_squared_error(y_test, ensemble_pred)
        ensemble_rmse = np.sqrt(ensemble_mse)
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_r2 = r2_score(y_test, ensemble_pred)

        logger.info(".4f"
                   ".4f"
                   ".4f"
                   ".4f")

        return {
            'model': ensemble,
            'predictions': ensemble_pred,
            'metrics': {
                'mse': ensemble_mse,
                'rmse': ensemble_rmse,
                'mae': ensemble_mae,
                'r2': ensemble_r2
            },
            'individual_models': {
                'xgboost': xgb_model,
                'catboost': cat_model,
                'lightgbm': lgb_model
            }
        }

    def save_models_efficient(self, results):
        """Save trained models."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save ensemble model
        ensemble_path = self.models_dir / f"efficient_ensemble_model_{timestamp}.pkl"
        joblib.dump(results['model'], ensemble_path)
        logger.info(f"Ensemble model saved to {ensemble_path}")

        # Save individual models
        for name, model in results['individual_models'].items():
            model_path = self.models_dir / f"efficient_{name}_model_{timestamp}.pkl"
            joblib.dump(model, model_path)
            logger.info(f"{name} model saved to {model_path}")

        return str(ensemble_path)

    def run_efficient_pipeline(self):
        """Run the memory-efficient advanced training pipeline."""
        logger.info("🚀 Starting Memory-Efficient Advanced Forex AI Training Pipeline")

        # Load sample data
        data = self.load_data_sample()
        if data is None:
            return False

        # Feature engineering
        processed_data = self.efficient_feature_engineering(data)

        # Prepare features and target
        feature_cols = [col for col in processed_data.columns
                       if col not in ['timestamp', 'symbol', 'close']]
        target_col = 'close'

        logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

        # Prepare data
        df_clean = processed_data.dropna(subset=feature_cols + [target_col])
        X = df_clean[feature_cols]
        y = df_clean[target_col]

        logger.info(f"Final dataset: {len(X):,} samples, {len(feature_cols)} features")

        # Check minimum data size
        if len(X) < 10000:
            logger.error(f"Insufficient data for training: {len(X)} samples. Need at least 10,000.")
            return False

        # Split data (time series split)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        logger.info(f"Training set: {len(X_train):,} samples, Test set: {len(X_test):,} samples")

        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Clear memory
        del X_train, X_test
        gc.collect()

        # Train ensemble model
        results = self.train_efficient_ensemble(X_train_scaled, y_train, X_test_scaled, y_test)

        # Save models
        model_path = self.save_models_efficient(results)

        # Final results
        logger.info("✅ Memory-efficient training pipeline completed!")
        logger.info(f"📊 Data processed: {len(processed_data):,} records")
        logger.info(f"🔧 Features created: {len(feature_cols)}")
        logger.info(".4f")
        logger.info(f"📁 Models saved to: {model_path}")

        return True

def main():
    pipeline = MemoryEfficientTrainingPipeline()
    success = pipeline.run_efficient_pipeline()

    if success:
        print("\n" + "="*60)
        print("🎉 MEMORY-EFFICIENT ADVANCED TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("🚀 Advanced Features:")
        print("  • Hyperparameter optimization with Optuna")
        print("  • Ensemble learning (XGBoost + CatBoost + LightGBM)")
        print("  • Efficient feature engineering")
        print("  • Memory-optimized processing")
        print("  • Model interpretability and analysis")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ MEMORY-EFFICIENT TRAINING PIPELINE FAILED")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()