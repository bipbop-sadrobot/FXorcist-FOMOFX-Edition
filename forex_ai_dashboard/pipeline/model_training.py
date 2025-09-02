#!/usr/bin/env python3
"""
Forex AI Model Training Script
Trains ML models for forex price prediction using processed data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from datetime import datetime
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from forex_ai_dashboard.models.catboost_model import CatBoostModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_processed_data(data_path: str = "data/processed/eurusd_features_2024.parquet"):
    """Load and prepare processed forex data for training."""
    logger.info(f"Loading data from {data_path}")

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    # Basic data validation
    if len(df) == 0:
        raise ValueError("No data found in the processed file")

    # Ensure we have OHLC data
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df

def prepare_features_and_target(df: pd.DataFrame, target_horizon: int = 1):
    """Prepare features and target for training."""
    logger.info(f"Preparing features and target with {target_horizon}-step ahead prediction")

    # Feature columns (exclude target-related columns)
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'returns']]

    # Create target: future price movement (1-step ahead)
    df = df.copy()
    df['target'] = df['close'].shift(-target_horizon)

    # Remove rows with NaN target
    df = df.dropna(subset=['target'])

    # Features and target
    X = df[feature_cols]
    y = df['target']

    logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
    logger.info(f"Feature columns: {feature_cols[:10]}...")  # Show first 10

    return X, y, feature_cols

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_params: dict = None):
    """Train and evaluate the model."""
    logger.info("Initializing CatBoost model")

    # Default parameters
    default_params = {
        'iterations': 500,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3.0,
        'loss_function': 'RMSE',
        'random_seed': 42,
        'early_stopping_rounds': 50,
        'verbose': False
    }

    # Update with custom parameters if provided
    if model_params:
        default_params.update(model_params)

    # Initialize model
    model = CatBoostModel(**default_params)

    logger.info("Starting model training")
    start_time = datetime.now()

    # Train model
    model.train(X_train, y_train, X_test, y_test)

    training_time = (datetime.now() - start_time).total_seconds()
    logger.info(".2f")

    # Evaluate on test set
    logger.info("Evaluating model on test set")
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

    logger.info("Model evaluation metrics:")
    logger.info(".6f")
    logger.info(".6f")
    logger.info(".6f")
    logger.info(".6f")

    return model, metrics

def save_model_and_results(model, metrics, model_path: str = "models/trained/catboost_model"):
    """Save trained model and evaluation results."""
    logger.info(f"Saving model to {model_path}")

    # Create directory if it doesn't exist
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    # Save model
    model.save(f"{model_path}.cbm")

    # Save metrics
    metrics_path = f"{model_path}_metrics.json"
    import json
    with open(metrics_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'model_params': model.params
        }, f, indent=2)

    logger.info(f"Model and metrics saved successfully")

def main():
    """Main training function."""
    try:
        logger.info("Starting Forex AI Model Training")

        # Load processed data
        data_path = "data/processed/eurusd_features_2024.parquet"
        df = load_processed_data(data_path)

        # Prepare features and target
        X, y, feature_cols = prepare_features_and_target(df, target_horizon=1)

        # Split data
        test_size = 0.2
        val_size = 0.1
        train_size = 1 - test_size - val_size

        logger.info(".1f")

        # First split: train + val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(train_size + val_size), shuffle=False
        )

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        # Train model
        model_params = {
            'iterations': 300,  # Reduced for faster training
            'learning_rate': 0.1,
            'depth': 4
        }

        model, metrics = train_and_evaluate_model(
            X_train, X_test, y_train, y_test, model_params
        )

        # Save model and results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/trained/catboost_forex_{timestamp}"
        save_model_and_results(model, metrics, model_path)

        logger.info("Model training completed successfully!")
        logger.info(f"Model saved to: {model_path}.cbm")
        logger.info(f"Metrics saved to: {model_path}_metrics.json")

        # Print final summary
        print("\n" + "="*50)
        print("FOREX AI MODEL TRAINING SUMMARY")
        print("="*50)
        print(f"Data source: {data_path}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features used: {len(feature_cols)}")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")
        print(f"Model saved as: {model_path}.cbm")
        print("="*50)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        print(f"\n❌ Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()