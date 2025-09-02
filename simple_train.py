#!/usr/bin/env python3
"""
Simple Forex AI Model Training Script
Direct training using CatBoost without wrapper classes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from datetime import datetime
from catboost import CatBoostRegressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/simple_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Simple training function."""
    try:
        logger.info("Starting Simple Forex AI Model Training")

        # Load processed data
        data_path = "data/processed/eurusd_features_2024.parquet"
        logger.info(f"Loading data from {data_path}")

        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'returns']]
        df['target'] = df['close'].shift(-1)  # 1-step ahead prediction
        df = df.dropna(subset=['target'])

        X = df[feature_cols]
        y = df['target']

        logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
        logger.info(f"Feature columns: {feature_cols[:5]}...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        # Initialize and train CatBoost model
        logger.info("Initializing CatBoost model")

        model = CatBoostRegressor(
            iterations=200,
            learning_rate=0.1,
            depth=4,
            loss_function='RMSE',
            random_seed=42,
            verbose=False
        )

        logger.info("Starting model training")
        start_time = datetime.now()

        model.fit(X_train, y_train)

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

        logger.info("Model evaluation metrics:")
        logger.info(".6f")
        logger.info(".6f")
        logger.info(".6f")
        logger.info(".6f")

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/trained/simple_catboost_{timestamp}.cbm"
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save_model(model_path)

        logger.info(f"Model saved to: {model_path}")

        # Print final summary
        print("\n" + "="*50)
        print("SIMPLE FOREX AI MODEL TRAINING SUMMARY")
        print("="*50)
        print(f"Data source: {data_path}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features used: {len(feature_cols)}")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")
        print(f"Model saved as: {model_path}")
        print("="*50)

        logger.info("Simple training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        print(f"\n❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()