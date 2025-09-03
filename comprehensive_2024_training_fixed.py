#!/usr/bin/env python3
"""
Comprehensive 2024 Forex Training Pipeline
Processes all available 2024 data from multiple currency pairs for extensive training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/comprehensive_2024_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_2024_data():
    """Load all available 2024 data from different currency pairs"""
    logger.info("🔍 Scanning for 2024 data sources...")

    histdata_dir = Path('data/raw/histdata')
    data_sources = []

    # Scan all currency directories for 2024 data
    for currency_dir in histdata_dir.iterdir():
        if currency_dir.is_dir() and not currency_dir.name.startswith('.'):
            currency_2024_dir = currency_dir / '2024'
            if currency_2024_dir.exists():
                for file_path in currency_2024_dir.glob('*'):
                    if file_path.suffix in ['.csv', '.txt']:
                        data_sources.append({
                            'path': file_path,
                            'currency': currency_dir.name,
                            'filename': file_path.name
                        })

    logger.info(f"📊 Found {len(data_sources)} data sources for 2024")
    return data_sources

def process_2024_file(file_info):
    """Process a single 2024 data file"""
    try:
        file_path = file_info['path']
        currency = file_info['currency']

        logger.info(f"📖 Processing {currency}: {file_path.name}")

        # Read the file
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:  # .txt file
            df = pd.read_csv(file_path, delimiter=',')

        # Standardize column names
        df.columns = [col.lower().strip() for col in df.columns]

        # Handle different column formats
        if 'timestamp' not in df.columns:
            # Try to find timestamp column
            time_cols = [col for col in df.columns if any(term in col for term in ['time', 'date'])]
            if time_cols:
                df = df.rename(columns={time_cols[0]: 'timestamp'})

        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            df = df.set_index('timestamp')

        # Ensure we have OHLC columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns in {file_path.name}: {missing_cols}")
            return None

        # Add volume if missing
        if 'volume' not in df.columns:
            df['volume'] = np.random.randint(50, 200, size=len(df))

        # Add symbol column
        df['symbol'] = currency

        # Filter for 2024 data only
        df = df[df.index.year == 2024]

        if len(df) == 0:
            logger.warning(f"No 2024 data found in {file_path.name}")
            return None

        logger.info(f"✅ Processed {currency}: {len(df)} rows")
        return df

    except Exception as e:
        logger.error(f"❌ Error processing {file_info['filename']}: {str(e)}")
        return None

def create_features(df):
    """Create technical indicators and features for training"""
    df = df.copy()

    # Basic price features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Moving averages
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

    # Volatility features
    df['volatility_10'] = df['returns'].rolling(window=10).std()
    df['volatility_20'] = df['returns'].rolling(window=20).std()

    # Volume features
    df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_10']

    # Price momentum
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1

    # High-Low range
    df['range'] = (df['high'] - df['low']) / df['close']
    df['range_sma_10'] = df['range'].rolling(window=10).mean()

    # Lag features
    for lag in [1, 2, 3, 5]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)

    # Target variable: next period return
    df['target'] = df['returns'].shift(-1)

    # Drop NaN values
    df = df.dropna()

    return df

def train_comprehensive_model(X_train, y_train, X_test, y_test):
    """Train CatBoost model with comprehensive 2024 data"""
    logger.info("🤖 Training comprehensive CatBoost model...")

    # Model parameters optimized for forex data
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3,
        border_count=254,
        random_strength=1,
        bagging_temperature=1,
        od_type='Iter',
        od_wait=50,
        verbose=100,
        random_seed=42,
        task_type='CPU'
    )

    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=100,
        use_best_model=True
    )

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    logger.info("📊 Model Performance:")
    logger.info(".6f")
    logger.info(".6f")
    logger.info(".6f")

    return model, {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'feature_importance': model.get_feature_importance()
    }

def main():
    """Main training pipeline"""
    logger.info("🚀 Starting Comprehensive 2024 Forex Training")
    start_time = datetime.now()

    try:
        # Load all 2024 data sources
        data_sources = load_2024_data()

        if not data_sources:
            logger.error("❌ No 2024 data sources found!")
            return

        # Process all data files
        processed_data = []
        total_rows = 0

        for file_info in data_sources:
            df = process_2024_file(file_info)
            if df is not None:
                processed_data.append(df)
                total_rows += len(df)

        if not processed_data:
            logger.error("❌ No valid data processed!")
            return

        logger.info(f"📈 Total processed data: {total_rows:,} rows from {len(processed_data)} sources")

        # Combine all data
        combined_df = pd.concat(processed_data, axis=0)
        combined_df = combined_df.sort_index()

        # Remove duplicates
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

        logger.info(f"🔄 Combined data: {len(combined_df):,} rows")

        # Create features
        logger.info("🔧 Creating technical features...")
        feature_df = create_features(combined_df)

        # Prepare training data
        feature_cols = [col for col in feature_df.columns
                       if col not in ['target', 'symbol'] and not col.startswith('returns')]

        X = feature_df[feature_cols]
        y = feature_df['target']

        logger.info(f"🎯 Training features: {len(feature_cols)}")
        logger.info(f"📊 Target distribution - Mean: {y.mean():.6f}, Std: {y.std():.6f}")

        # Split data (time-based split)
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        logger.info(f"📋 Train set: {len(X_train):,} samples")
        logger.info(f"📋 Test set: {len(X_test):,} samples")

        # Train model
        model, metrics = train_comprehensive_model(X_train, y_train, X_test, y_test)

        # Save model and metrics
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"models/trained/comprehensive_2024_model_{timestamp}.cbm"
        metrics_path = f"models/trained/comprehensive_2024_model_{timestamp}_metrics.json"

        # Save model
        model.save_model(model_path)

        # Save metrics
        import json
        metrics_data = {
            'timestamp': timestamp,
            'data_sources': len(data_sources),
            'total_rows': total_rows,
            'training_rows': len(X_train),
            'test_rows': len(X_test),
            'features': len(feature_cols),
            'performance': {
                'mse': metrics['mse'],
                'rmse': metrics['rmse'],
                'r2': metrics['r2']
            },
            'feature_importance': dict(zip(feature_cols, metrics['feature_importance'][:10]))
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)

        # Log completion
        duration = (datetime.now() - start_time).total_seconds()
        logger.info("🎉 Comprehensive 2024 Training Completed!")
        logger.info(".2f")
        logger.info(f"📁 Model saved: {model_path}")
        logger.info(f"📊 Metrics saved: {metrics_path}")

        # Print top features
        logger.info("🔝 Top 10 Features by Importance:")
        feature_imp = list(zip(feature_cols, metrics['feature_importance']))
        feature_imp.sort(key=lambda x: x[1], reverse=True)

        for i, (feature, importance) in enumerate(feature_imp[:10]):
            logger.info("2d")

    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Create necessary directories
    Path('logs').mkdir(exist_ok=True)
    Path('models/trained').mkdir(parents=True, exist_ok=True)

    main()