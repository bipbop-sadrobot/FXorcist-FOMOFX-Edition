#!/usr/bin/env python3
"""
Simple training script using CatBoost with existing processed data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run simple training with existing processed data."""
    logger.info("🚀 Starting simple training with existing processed data")

    try:
        # Load existing processed data
        processed_dir = Path('data/processed')
        parquet_files = list(processed_dir.glob('*.parquet'))

        if not parquet_files:
            logger.error("No processed data files found")
            return False

        logger.info(f"Found {len(parquet_files)} processed data files")

        # Load first few files for training
        data_frames = []
        for file_path in parquet_files[:3]:  # Use first 3 files
            try:
                df = pd.read_parquet(file_path)
                data_frames.append(df)
                logger.info(f"Loaded {len(df)} rows from {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        if not data_frames:
            logger.error("No valid data loaded")
            return False

        # Combine data
        combined_data = pd.concat(data_frames, ignore_index=True)
        logger.info(f"Combined {len(combined_data)} rows of training data")

        # Prepare features and target
        feature_cols = [col for col in combined_data.columns
                       if col not in ['timestamp', 'symbol', 'target'] and
                       combined_data[col].dtype in ['float64', 'int64']]

        if not feature_cols:
            logger.error("No suitable feature columns found")
            return False

        # Create target (next close price)
        combined_data['target'] = combined_data['close'].shift(-1)
        combined_data = combined_data.dropna()

        if len(combined_data) < 100:
            logger.error("Not enough data for training")
            return False

        X = combined_data[feature_cols]
        y = combined_data['target']

        logger.info(f"Using {len(feature_cols)} features: {feature_cols[:5]}...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train CatBoost model
        logger.info("🤖 Training CatBoost model")
        model = CatBoostRegressor(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            loss_function='RMSE',
            random_seed=42,
            verbose=False
        )

        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Save model
        models_dir = Path('models/trained')
        models_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = models_dir / f"simple_catboost_{timestamp}.cbm"
        model.save_model(str(model_path))

        # Save metrics
        metrics_path = models_dir / f"simple_catboost_{timestamp}_metrics.json"
        metrics = {
            'timestamp': timestamp,
            'model_path': str(model_path),
            'training_time': training_time,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            },
            'feature_importance': model.feature_importances_.tolist(),
            'feature_names': feature_cols,
            'data_shape': X.shape,
            'test_size': len(X_test)
        }

        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        # Print results
        logger.info("✅ Training completed successfully!")
        logger.info(".2f")
        logger.info(".6f")
        logger.info(".6f")
        logger.info(".6f")
        logger.info(".6f")
        logger.info(f"📁 Model saved to: {model_path}")
        logger.info(f"📊 Metrics saved to: {metrics_path}")

        return True

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n" + "="*60)
        print("🎉 SIMPLE TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("📁 Check the following directories for results:")
        print("  • Models: models/trained/")
        print("  • Logs: logs/")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ TRAINING FAILED")
        print("="*60)