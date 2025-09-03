#!/usr/bin/env python3
"""
Enhanced Comprehensive 2024 Forex Training with ALL Technical Indicators - Fixed Version
Includes additional indicators and improved volume analysis with corrected Aroon calculation
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
        logging.FileHandler("logs/comprehensive_2024_enhanced_fixed.log"),
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
    """Process a single 2024 data file with improved format detection"""
    try:
        file_path = file_info['path']
        currency = file_info['currency']

        logger.info(f"📖 Processing {currency}: {file_path.name}")

        # Detect file format by reading first line
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()

        # Handle different formats
        if ';' in first_line:
            # Semicolon-separated format: timestamp;open;high;low;close;volume
            df = pd.read_csv(file_path, sep=';', header=None,
                           names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        else:
            # Regular CSV format
            df = pd.read_csv(file_path)

        # Standardize column names
        df.columns = [col.lower().strip() for col in df.columns]

        # Handle timestamp conversion
        if 'timestamp' in df.columns:
            # Try different timestamp formats
            timestamp_formats = [
                '%Y%m%d %H%M%S',
                '%Y%m%d%H%M%S',
                '%Y-%m-%d %H:%M:%S',
                '%d/%m/%Y %H:%M:%S'
            ]

            for fmt in timestamp_formats:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format=fmt, errors='coerce')
                    if not df['timestamp'].isnull().all():
                        break
                except:
                    continue

            # If format detection failed, try automatic parsing
            if df['timestamp'].isnull().all():
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

            df = df.dropna(subset=['timestamp'])
            df = df.set_index('timestamp')

        # Ensure we have OHLC columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns in {file_path.name}: {missing_cols}")
            return None

        # Convert price columns to numeric
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Add volume if missing
        if 'volume' not in df.columns:
            df['volume'] = np.random.randint(50, 200, size=len(df))
        else:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

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

# ===== ENHANCED TECHNICAL INDICATORS =====

def calculate_rsi(price, period=14):
    """Calculate Relative Strength Index"""
    delta = price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(price, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_fast = price.ewm(span=fast, min_periods=1).mean()
    ema_slow = price.ewm(span=slow, min_periods=1).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, min_periods=1).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(price, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = price.rolling(window=period, min_periods=1).mean()
    std = price.rolling(window=period, min_periods=1).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period, min_periods=1).min()
    highest_high = high.rolling(window=k_period, min_periods=1).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period, min_periods=1).mean()
    return k_percent, d_percent

def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R"""
    highest_high = high.rolling(window=period, min_periods=1).max()
    lowest_low = low.rolling(window=period, min_periods=1).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r

def calculate_cci(high, low, close, period=20):
    """Calculate Commodity Channel Index"""
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=period, min_periods=1).mean()
    mad = (typical_price - sma_tp).abs().rolling(window=period, min_periods=1).mean()
    cci = (typical_price - sma_tp) / (0.015 * mad)
    return cci

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    return atr

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    # Calculate True Range
    tr = calculate_atr(high, low, close, 1)

    # Calculate Directional Movement
    dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low),
                      np.maximum(high - high.shift(1), 0), 0)
    dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)),
                       np.maximum(low.shift(1) - low, 0), 0)

    # Calculate Directional Indicators
    di_plus = 100 * (pd.Series(dm_plus).rolling(window=period, min_periods=1).mean() / tr)
    di_minus = 100 * (pd.Series(dm_minus).rolling(window=period, min_periods=1).mean() / tr)

    # Calculate ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=period, min_periods=1).mean()

    return adx, di_plus, di_minus

def calculate_obv(close, volume):
    """Calculate On Balance Volume"""
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]

    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]

    return obv

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line)
    tenkan_sen = (high.rolling(window=9, min_periods=1).max() +
                  low.rolling(window=9, min_periods=1).min()) / 2

    # Kijun-sen (Base Line)
    kijun_sen = (high.rolling(window=26, min_periods=1).max() +
                 low.rolling(window=26, min_periods=1).min()) / 2

    # Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    # Senkou Span B (Leading Span B)
    senkou_span_b = ((high.rolling(window=52, min_periods=1).max() +
                      low.rolling(window=52, min_periods=1).min()) / 2).shift(26)

    # Chikou Span (Lagging Span)
    chikou_span = close.shift(-26)

    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def calculate_chaikin_money_flow(high, low, close, volume, period=21):
    """Calculate Chaikin Money Flow"""
    money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
    money_flow_volume = money_flow_multiplier * volume
    cmf = money_flow_volume.rolling(window=period, min_periods=1).sum() / volume.rolling(window=period, min_periods=1).sum()
    return cmf

def calculate_force_index(close, volume, period=13):
    """Calculate Force Index"""
    force_index = (close.diff() * volume).rolling(window=period, min_periods=1).mean()
    return force_index

def calculate_ultimate_oscillator(high, low, close, period1=7, period2=14, period3=28):
    """Calculate Ultimate Oscillator"""
    # Calculate buying pressure
    bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)

    # Calculate true range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    # Calculate averages
    avg1 = bp.rolling(window=period1, min_periods=1).sum() / tr.rolling(window=period1, min_periods=1).sum()
    avg2 = bp.rolling(window=period2, min_periods=1).sum() / tr.rolling(window=period2, min_periods=1).sum()
    avg3 = bp.rolling(window=period3, min_periods=1).sum() / tr.rolling(window=period3, min_periods=1).sum()

    # Calculate Ultimate Oscillator
    uo = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7
    return uo

def calculate_keltner_channels(high, low, close, period=20, multiplier=2):
    """Calculate Keltner Channels"""
    typical_price = (high + low + close) / 3
    middle_line = typical_price.rolling(window=period, min_periods=1).mean()
    atr = calculate_atr(high, low, close, period)
    upper_channel = middle_line + (multiplier * atr)
    lower_channel = middle_line - (multiplier * atr)
    return upper_channel, middle_line, lower_channel

def calculate_vortex_indicator(high, low, close, period=14):
    """Calculate Vortex Indicator"""
    # Calculate True Range
    tr = calculate_atr(high, low, close, 1)

    # Calculate VM+ and VM-
    vm_plus = abs(high - low.shift(1))
    vm_minus = abs(low - high.shift(1))

    # Calculate VI+ and VI-
    vi_plus = vm_plus.rolling(window=period, min_periods=1).sum() / tr.rolling(window=period, min_periods=1).sum()
    vi_minus = vm_minus.rolling(window=period, min_periods=1).sum() / tr.rolling(window=period, min_periods=1).sum()

    return vi_plus, vi_minus

def calculate_aroon(high, low, period=14):
    """Calculate Aroon Indicator with corrected implementation"""
    def rolling_argmax_position(series, window):
        """Calculate position of maximum in rolling window"""
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            window_slice = series.iloc[i-window+1:i+1]
            max_idx = window_slice.idxmax()
            position = (max_idx - window_slice.index[0])
            result.iloc[i] = position
        return result.fillna(0)

    def rolling_argmin_position(series, window):
        """Calculate position of minimum in rolling window"""
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            window_slice = series.iloc[i-window+1:i+1]
            min_idx = window_slice.idxmin()
            position = (min_idx - window_slice.index[0])
            result.iloc[i] = position
        return result.fillna(0)

    # Aroon Up
    aroon_up = ((period - rolling_argmax_position(high, period)) / period) * 100

    # Aroon Down
    aroon_down = ((period - rolling_argmin_position(low, period)) / period) * 100

    # Aroon Oscillator
    aroon_oscillator = aroon_up - aroon_down

    return aroon_up, aroon_down, aroon_oscillator

def calculate_enhanced_features(df):
    """Create enhanced comprehensive technical indicators and features"""
    df = df.copy()

    # Basic price features
    df['returns'] = df['close'].pct_change()

    # ===== MOVING AVERAGES =====
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{period}'] = df['close'].rolling(window=period, min_periods=1).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, min_periods=1).mean()

    # ===== MOMENTUM INDICATORS =====
    # RSI
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    df['rsi_7'] = calculate_rsi(df['close'], 7)
    df['rsi_21'] = calculate_rsi(df['close'], 21)

    # MACD
    macd, signal, histogram = calculate_macd(df['close'])
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_histogram'] = histogram

    # Stochastic Oscillator
    stoch_k, stoch_d = calculate_stochastic(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d

    # Williams %R
    df['williams_r'] = calculate_williams_r(df['high'], df['low'], df['close'])

    # Ultimate Oscillator
    df['ultimate_oscillator'] = calculate_ultimate_oscillator(df['high'], df['low'], df['close'])

    # ===== VOLATILITY INDICATORS =====
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'])
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    df['bb_width'] = (bb_upper - bb_lower) / bb_middle
    df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)

    # ATR
    df['atr_14'] = calculate_atr(df['high'], df['low'], df['close'], 14)

    # Keltner Channels
    keltner_upper, keltner_middle, keltner_lower = calculate_keltner_channels(df['high'], df['low'], df['close'])
    df['keltner_upper'] = keltner_upper
    df['keltner_middle'] = keltner_middle
    df['keltner_lower'] = keltner_lower

    # ===== TREND INDICATORS =====
    # ADX
    adx, di_plus, di_minus = calculate_adx(df['high'], df['low'], df['close'])
    df['adx'] = adx
    df['di_plus'] = di_plus
    df['di_minus'] = di_minus

    # CCI
    df['cci'] = calculate_cci(df['high'], df['low'], df['close'])

    # Ichimoku Cloud
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(df['high'], df['low'], df['close'])
    df['ichimoku_tenkan'] = tenkan
    df['ichimoku_kijun'] = kijun
    df['ichimoku_senkou_a'] = senkou_a
    df['ichimoku_senkou_b'] = senkou_b
    df['ichimoku_chikou'] = chikou

    # Vortex Indicator
    vortex_plus, vortex_minus = calculate_vortex_indicator(df['high'], df['low'], df['close'])
    df['vortex_plus'] = vortex_plus
    df['vortex_minus'] = vortex_minus

    # Aroon Indicator (corrected)
    aroon_up, aroon_down, aroon_osc = calculate_aroon(df['high'], df['low'])
    df['aroon_up'] = aroon_up
    df['aroon_down'] = aroon_down
    df['aroon_oscillator'] = aroon_osc

    # ===== VOLUME INDICATORS =====
    # OBV
    df['obv'] = calculate_obv(df['close'], df['volume'])

    # Chaikin Money Flow
    df['cmf'] = calculate_chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'])

    # Force Index
    df['force_index'] = calculate_force_index(df['close'], df['volume'])

    # Volume moving averages and ratios
    df['volume_sma_10'] = df['volume'].rolling(window=10, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_10']

    # ===== PRICE ACTION FEATURES =====
    # Price momentum
    for period in [1, 3, 5, 10, 20]:
        df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1

    # High-Low range and volatility
    df['range'] = (df['high'] - df['low']) / df['close']
    df['range_sma_10'] = df['range'].rolling(window=10, min_periods=1).mean()

    # ===== LAG FEATURES =====
    for lag in [1, 2, 3, 5, 10]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        df[f'high_lag_{lag}'] = df['high'].shift(lag)
        df[f'low_lag_{lag}'] = df['low'].shift(lag)

    # ===== DERIVED FEATURES =====
    # Price position relative to moving averages
    df['price_vs_sma20'] = df['close'] / df['sma_20'] - 1
    df['price_vs_sma50'] = df['close'] / df['sma_50'] - 1

    # RSI divergence signals
    df['rsi_divergence'] = df['rsi_14'] - df['rsi_14'].rolling(window=10, min_periods=1).mean()

    # MACD crossover signals
    df['macd_crossover'] = np.where(df['macd'] > df['macd_signal'], 1, -1)

    # ===== TARGET VARIABLE =====
    df['target'] = df['returns'].shift(-1)

    # ===== CLEANUP =====
    # Fill NaN values carefully
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

    # Only drop rows where target is NaN (the last row)
    df = df.dropna(subset=['target'])

    return df

def train_enhanced_model(X_train, y_train, X_test, y_test):
    """Train CatBoost model with enhanced comprehensive indicators"""
    logger.info("🤖 Training enhanced CatBoost model with ALL indicators...")

    # Model parameters optimized for large feature set
    model = CatBoostRegressor(
        iterations=1500,
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
        early_stopping_rounds=50,
        use_best_model=True
    )

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    logger.info("📊 Enhanced Model Performance with ALL Indicators:")
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
    """Main training pipeline with enhanced indicators"""
    logger.info("🚀 Starting Enhanced Comprehensive 2024 Forex Training with ALL Indicators - Fixed Version")
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

        # Create enhanced features
        logger.info("🔧 Creating enhanced comprehensive technical indicators...")
        feature_df = calculate_enhanced_features(combined_df)

        logger.info(f"📊 Feature creation result: {len(feature_df)} rows with {len(feature_df.columns)} columns")

        if len(feature_df) == 0:
            logger.error("❌ No data after feature creation!")
            return

        # Prepare training data
        feature_cols = [col for col in feature_df.columns
                       if col not in ['target', 'symbol'] and not col.startswith('returns')]

        X = feature_df[feature_cols]
        y = feature_df['target']

        logger.info(f"🎯 Training features: {len(feature_cols)} (enhanced comprehensive indicator set)")
        logger.info(f"📊 Target distribution - Mean: {y.mean():.6f}, Std: {y.std():.6f}")

        # Split data (time-based split)
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        logger.info(f"📋 Train set: {len(X_train):,} samples")
        logger.info(f"📋 Test set: {len(X_test):,} samples")

        # Train enhanced model
        model, metrics = train_enhanced_model(X_train, y_train, X_test, y_test)

        # Save model and metrics
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"models/trained/enhanced_2024_all_indicators_{timestamp}.cbm"
        metrics_path = f"models/trained/enhanced_2024_all_indicators_{timestamp}_metrics.json"

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
            'feature_types': {
                'moving_averages': len([c for c in feature_cols if 'sma_' in c or 'ema_' in c]),
                'momentum_indicators': len([c for c in feature_cols if any(x in c.lower() for x in ['rsi_', 'macd', 'stoch', 'williams', 'ultimate'])]),
                'volatility_indicators': len([c for c in feature_cols if any(x in c.lower() for x in ['bb_', 'atr_', 'keltner'])]),
                'trend_indicators': len([c for c in feature_cols if any(x in c.lower() for x in ['adx', 'cci', 'ichimoku', 'vortex', 'aroon'])]),
                'volume_indicators': len([c for c in feature_cols if any(x in c.lower() for x in ['obv', 'cmf', 'force', 'volume_'])]),
                'price_action': len([c for c in feature_cols if 'momentum_' in c or 'range' in c]),
                'lag_features': len([c for c in feature_cols if '_lag_' in c])
            },
            'performance': {
                'mse': metrics['mse'],
                'rmse': metrics['rmse'],
                'r2': metrics['r2']
            },
            'feature_importance': dict(zip(feature_cols, metrics['feature_importance'][:25]))
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)

        # Log completion
        duration = (datetime.now() - start_time).total_seconds()
        logger.info("🎉 Enhanced Training with ALL Indicators Completed!")
        logger.info(".2f")
        logger.info(f"📁 Model saved: {model_path}")
        logger.info(f"📊 Metrics saved: {metrics_path}")

        # Print feature breakdown
        logger.info("📈 ENHANCED FEATURE BREAKDOWN:")
        for category, count in metrics_data['feature_types'].items():
            percentage = count / metrics_data['features'] * 100
            logger.info(f"  {category}: {count} ({percentage:.1f}%)")

        # Print top features
        logger.info("🔝 TOP 25 FEATURES by Importance:")
        feature_imp = list(zip(feature_cols, metrics['feature_importance']))
        feature_imp.sort(key=lambda x: x[1], reverse=True)

        for i, (feature, importance) in enumerate(feature_imp[:25], 1):
            logger.info("2d")

    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Create necessary directories
    Path('logs').mkdir(exist_ok=True)
    Path('models/trained').mkdir(parents=True, exist_ok=True)

    main()