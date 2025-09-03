#!/usr/bin/env python3
"""
Simple training script that uses existing processed data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dashboard.utils.enhanced_data_loader import EnhancedDataLoader
from automated_training_pipeline import AutomatedTrainingPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run training with existing processed data."""
    logger.info("🚀 Starting training with existing processed data")

    try:
        # Initialize data loader
        data_loader = EnhancedDataLoader()

        # Load existing processed data
        processed_dir = Path('data/processed')
        parquet_files = list(processed_dir.glob('*.parquet'))

        if not parquet_files:
            logger.error("No processed data files found")
            return False

        logger.info(f"Found {len(parquet_files)} processed data files")

        # Load first few files for training
        data_frames = []
        for file_path in parquet_files[:5]:  # Use first 5 files
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

        # Initialize training pipeline
        pipeline = AutomatedTrainingPipeline()

        # Skip data processing and go directly to training
        logger.info("🤖 Starting model training")
        training_results = pipeline.train_models(combined_data)

        # Save results
        pipeline.save_training_summary(training_results)

        # Print results
        logger.info("✅ Training completed successfully!")
        logger.info(f"📊 Data processed: {len(combined_data)} rows")
        logger.info(f"🎯 Models trained: {len(training_results)}")

        for model_name, result in training_results.items():
            if 'metrics' in result and 'r2' in result['metrics']:
                logger.info(f"  • {model_name}: R² = {result['metrics']['r2']:.6f}")

        return True

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("📁 Check the following directories for results:")
        print("  • Models: models/trained/")
        print("  • Logs: logs/")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ TRAINING FAILED")
        print("="*60)