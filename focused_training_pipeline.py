#!/usr/bin/env python3
"""
Focused Forex AI Training Pipeline
Optimized for available data and realistic downloads.
"""

import pandas as pd
import numpy as np
import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import asyncio

from forex_ai_dashboard.pipeline.unified_feature_engineering import UnifiedFeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/focused_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FocusedTrainingPipeline:
    """Focused pipeline for forex data processing and model training."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models" / "trained"
        self.logs_dir = self.project_root / "logs"

        # Create directories
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Focus on major pairs and recent data
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
        self.years = [2023, 2024]  # Focus on recent, available data
        self.months = list(range(1, 13))

    async def download_recent_data(self) -> bool:
        """Download recent forex data that's more likely to be available."""
        logger.info("📥 Downloading recent forex data (2023-2024)")

        success_count = 0
        total_attempts = 0

        # Download EURUSD and GBPUSD for 2023-2024 (we know these exist)
        for symbol in ["EURUSD", "GBPUSD"]:
            for year in [2023, 2024]:
                for month in self.months:
                    total_attempts += 1

                    # Skip if file already exists
                    csv_path = self.data_dir / "raw" / "histdata" / symbol / str(year) / f"{month:02d}.csv"
                    if csv_path.exists():
                        logger.info(f"✅ {symbol} {year}-{month:02d} already exists")
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
                            logger.info(f"✅ Downloaded {symbol} {year}-{month:02d}")
                            success_count += 1
                        else:
                            logger.debug(f"❌ Failed {symbol} {year}-{month:02d}")

                    except subprocess.TimeoutExpired:
                        logger.debug(f"Timeout {symbol} {year}-{month:02d}")
                    except Exception as e:
                        logger.debug(f"Error {symbol} {year}-{month:02d}: {str(e)}")

                    # Rate limiting
                    await asyncio.sleep(0.5)

        success_rate = success_count / total_attempts * 100
        logger.info(".1f")

        return success_count > 0

    def process_available_data(self) -> Optional[pd.DataFrame]:
        """Process all available forex data."""
        logger.info("🔄 Processing available forex data")

        all_data = []

        # Process EURUSD and GBPUSD data
        for symbol in ["EURUSD", "GBPUSD"]:
            symbol_data = []

            for year in [2023, 2024]:
                for month in self.months:
                    csv_path = self.data_dir / "raw" / "histdata" / symbol / str(year) / f"{month:02d}.csv"

                    if csv_path.exists():
                        try:
                            df = pd.read_csv(csv_path)
                            df['symbol'] = symbol
                            df['year'] = year
                            df['month'] = month
                            symbol_data.append(df)
                            logger.info(f"Loaded {symbol} {year}-{month:02d}: {len(df)} rows")
                        except Exception as e:
                            logger.warning(f"Error loading {csv_path}: {str(e)}")

            if symbol_data:
                symbol_df = pd.concat(symbol_data, ignore_index=True)
                all_data.append(symbol_df)
                logger.info(f"Combined {symbol}: {len(symbol_df)} total rows")

        if not all_data:
            logger.error("No data available for processing")
            return None

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total combined data: {len(combined_df)} rows")

        # Enhanced preprocessing
        combined_df = self.enhanced_preprocessing(combined_df)

        # Save processed data
        output_path = self.data_dir / "processed" / "focused_forex_data.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_parquet(output_path)

        logger.info(f"Saved processed data to {output_path}")
        return combined_df

    def enhanced_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced preprocessing with more features."""
        logger.info("🔧 Enhanced preprocessing")

        # Basic timestamp handling
        if 'timestamp' not in df.columns:
            timestamp_col = df.columns[0]
            df = df.rename(columns={timestamp_col: 'timestamp'})

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Add comprehensive technical indicators using unified feature engineering
        feature_engineer = UnifiedFeatureEngineer()
        df = feature_engineer.process_data(df, feature_groups=['basic', 'momentum', 'volatility', 'trend', 'volume', 'microstructure', 'time'])

        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp', 'symbol'])

        # Add time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month

        logger.info(f"Enhanced preprocessing complete: {len(df)} rows, {len(df.columns)} columns")
        return df


    def train_advanced_models(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Train advanced models with comprehensive evaluation."""
        logger.info("🚀 Training advanced ML models")

        results = {}

        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in [
            'timestamp', 'symbol', 'target', 'returns'
        ] and not col.startswith(('year', 'month'))]

        df['target'] = df['close'].shift(-1)  # 1-step ahead prediction
        df = df.dropna()

        X = df[feature_cols]
        y = df['target']

        logger.info(f"Training data: {len(X)} samples, {len(feature_cols)} features")
        logger.info(f"Features: {feature_cols[:10]}...")

        # Split data (80/20 train/test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        # Train CatBoost with optimized parameters
        results['catboost'] = self.train_optimized_catboost(X_train, X_test, y_train, y_test)

        # Save training summary
        self.save_training_summary(results, feature_cols)

        return results

    def train_optimized_catboost(self, X_train, X_test, y_train, y_test):
        """Train optimized CatBoost model."""
        logger.info("Training optimized CatBoost model")

        from catboost import CatBoostRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # Optimized parameters for forex data
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=1.0,
            loss_function='RMSE',
            random_seed=42,
            early_stopping_rounds=100,
            verbose=False,
            # Forex-specific parameters
            grow_policy='SymmetricTree',
            min_data_in_leaf=100,
            max_leaves=64
        )

        start_time = datetime.now()
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=100,
            verbose=False
        )
        training_time = (datetime.now() - start_time).total_seconds()

        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Feature importance
        feature_importance = model.feature_importances_

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.models_dir / f"catboost_optimized_{timestamp}.cbm"
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
            'feature_importance': feature_importance.tolist(),
            'feature_names': X_train.columns.tolist(),
            'model_params': {
                'iterations': 1000,
                'learning_rate': 0.03,
                'depth': 8,
                'l2_leaf_reg': 1.0
            }
        }

        logger.info(".2f")
        logger.info(".8f")
        logger.info(".8f")
        logger.info(".8f")
        logger.info(".8f")

        # Log top features
        top_features = sorted(zip(X_train.columns, feature_importance),
                            key=lambda x: x[1], reverse=True)[:10]
        logger.info("Top 10 features by importance:")
        for feature, importance in top_features:
            logger.info(".4f")

        return result

    def save_training_summary(self, results: Dict[str, Dict], feature_cols: List[str]):
        """Save comprehensive training summary."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_info': {
                'total_samples': len(feature_cols),
                'features_used': len(feature_cols),
                'feature_list': feature_cols
            },
            'models_trained': list(results.keys()),
            'results': results
        }

        summary_path = self.logs_dir / f"focused_training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            import json
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Training summary saved to {summary_path}")

    async def run_focused_pipeline(self):
        """Run the complete focused pipeline."""
        logger.info("🎯 Starting Focused Forex AI Training Pipeline")

        try:
            # Step 1: Download recent data
            logger.info("📥 Step 1: Downloading recent forex data")
            download_success = await self.download_recent_data()

            # Step 2: Process available data
            logger.info("🔄 Step 2: Processing available data")
            processed_data = self.process_available_data()

            if processed_data is None or len(processed_data) == 0:
                logger.error("No data available for training")
                return False

            # Step 3: Train advanced models
            logger.info("🤖 Step 3: Training advanced ML models")
            training_results = self.train_advanced_models(processed_data)

            # Step 4: Summary
            logger.info("✅ Focused pipeline completed successfully!")
            logger.info(f"📊 Data processed: {len(processed_data)} rows")
            logger.info(f"🎯 Models trained: {len(training_results)}")

            for model_name, result in training_results.items():
                metrics = result.get('metrics', {})
                logger.info(f"  • {model_name}: R² = {metrics.get('r2', 0):.6f}")

            return True

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return False

def main():
    """Main function to run the focused pipeline."""
    pipeline = FocusedTrainingPipeline()

    # Run the async pipeline
    success = asyncio.run(pipeline.run_focused_pipeline())

    if success:
        print("\n" + "="*60)
        print("🎉 FOCUSED FOREX AI TRAINING PIPELINE COMPLETED!")
        print("="*60)
        print("📁 Check the following directories for results:")
        print("  • Models: models/trained/")
        print("  • Processed data: data/processed/")
        print("  • Logs: logs/")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ FOCUSED TRAINING PIPELINE FAILED")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()