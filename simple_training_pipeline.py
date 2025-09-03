#!/usr/bin/env python3
"""
Simple Forex AI Training Pipeline
Loads existing data and trains models.
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
import zipfile
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from catboost import CatBoostRegressor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleTrainingPipeline:
    def __init__(self):
        self.data_dir = Path("data/data/data/raw/temp_fx1min/output")
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

    def load_data(self):
        """Load forex data from ZIP files."""
        logger.info("Loading forex data from ZIP files...")

        all_data = []

        for symbol in self.symbols:
            symbol_path = self.data_dir / symbol.lower()
            if symbol_path.exists():
                logger.info(f"Processing {symbol}")

                for zip_file in symbol_path.glob("*.zip"):
                    try:
                        filename = zip_file.name
                        if "_M1_" in filename:
                            year_part = filename.split("_M1_")[1].split(".")[0]
                            if len(year_part) == 4 and year_part.isdigit():
                                year = year_part

                                # Extract the ZIP file temporarily
                                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                                    # Get the CSV filename inside the ZIP
                                    csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                                    if csv_files:
                                        # Extract to memory
                                        with zip_ref.open(csv_files[0]) as csv_file:
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
                                                all_data.append(df)
                                                logger.info(f"  Loaded {len(df)} records from {symbol} {year}")

                    except Exception as e:
                        logger.warning(f"Error processing {zip_file}: {e}")

        if not all_data:
            logger.error("No data loaded")
            return None

        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Total data loaded: {len(combined_data)} records")
        return combined_data

    def preprocess_data(self, df):
        """Add basic technical indicators."""
        logger.info("Adding technical indicators...")

        # Simple moving averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Returns
        df['returns'] = df['close'].pct_change()

        # Drop NaN values
        df = df.dropna()

        logger.info(f"Data after preprocessing: {len(df)} records")
        return df

    def train_models(self, df):
        """Train ML models."""
        logger.info("Training models...")

        # Prepare features
        feature_cols = ['open', 'high', 'low', 'volume', 'sma_5', 'sma_10', 'sma_20', 'rsi', 'returns']
        target_col = 'close'

        # Remove rows with NaN in features
        df_clean = df.dropna(subset=feature_cols + [target_col])

        X = df_clean[feature_cols]
        y = df_clean[target_col]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        results = {}

        # Train XGBoost
        logger.info("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb_model.fit(X_train_scaled, y_train)

        xgb_pred = xgb_model.predict(X_test_scaled)
        xgb_mse = mean_squared_error(y_test, xgb_pred)
        xgb_r2 = r2_score(y_test, xgb_pred)

        results['xgboost'] = {
            'mse': xgb_mse,
            'r2': xgb_r2,
            'predictions': xgb_pred[:10].tolist()  # First 10 predictions
        }

        # Train CatBoost
        logger.info("Training CatBoost...")
        cat_model = CatBoostRegressor(iterations=100, learning_rate=0.1, random_state=42, verbose=False)
        cat_model.fit(X_train_scaled, y_train)

        cat_pred = cat_model.predict(X_test_scaled)
        cat_mse = mean_squared_error(y_test, cat_pred)
        cat_r2 = r2_score(y_test, cat_pred)

        results['catboost'] = {
            'mse': cat_mse,
            'r2': cat_r2,
            'predictions': cat_pred[:10].tolist()  # First 10 predictions
        }

        return results

    def run_pipeline(self):
        """Run the complete pipeline."""
        logger.info("🚀 Starting Simple Forex AI Training Pipeline")

        # Load data
        data = self.load_data()
        if data is None:
            return False

        # Preprocess data
        processed_data = self.preprocess_data(data)

        # Train models
        results = self.train_models(processed_data)

        # Print results
        logger.info("✅ Pipeline completed!")
        logger.info(f"📊 Data processed: {len(processed_data)} records")

        for model_name, result in results.items():
            logger.info(".4f"
                       ".4f")

        return True

def main():
    pipeline = SimpleTrainingPipeline()
    success = pipeline.run_pipeline()

    if success:
        print("\n" + "="*50)
        print("🎉 SIMPLE TRAINING PIPELINE COMPLETED!")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("❌ PIPELINE FAILED")
        print("="*50)
        sys.exit(1)

if __name__ == "__main__":
    main()