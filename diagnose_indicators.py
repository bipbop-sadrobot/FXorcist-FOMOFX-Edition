#!/usr/bin/env python3
"""
Diagnostic script to identify problematic technical indicators
that may be producing NaN values or invalid outputs
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_sample_data():
    """Load a small sample of data for testing indicators"""
    logger.info("🔍 Loading sample data for indicator diagnosis...")

    # Load one of the processed files
    data_dir = Path('data/processed')
    files = list(data_dir.glob('*.parquet'))

    if not files:
        logger.error("❌ No processed data files found")
        return None

    # Load the first file
    df = pd.read_parquet(files[0])
    logger.info(f"📊 Loaded {len(df)} rows from {files[0].name}")

    # Take a sample for testing
    sample_df = df.head(1000).copy()
    logger.info(f"📊 Using sample of {len(sample_df)} rows for testing")

    return sample_df

def test_indicator_functions(df):
    """Test each indicator function for NaN outputs and errors"""
    logger.info("🧪 Testing individual indicator functions...")

    results = {}

    # Test RSI
    try:
        rsi_14 = calculate_rsi(df['close'], 14)
        nan_count = rsi_14.isnull().sum()
        results['RSI_14'] = {
            'nan_count': nan_count,
            'nan_percentage': nan_count / len(rsi_14) * 100,
            'valid_count': len(rsi_14) - nan_count,
            'value_range': (rsi_14.min(), rsi_14.max()) if nan_count < len(rsi_14) else (np.nan, np.nan)
        }
        logger.info(f"✅ RSI_14: {nan_count} NaN values ({nan_count/len(rsi_14)*100:.1f}%)")
    except Exception as e:
        results['RSI_14'] = {'error': str(e)}
        logger.error(f"❌ RSI_14 failed: {e}")

    # Test MACD
    try:
        macd, signal, histogram = calculate_macd(df['close'])
        for name, series in [('MACD', macd), ('MACD_Signal', signal), ('MACD_Histogram', histogram)]:
            nan_count = series.isnull().sum()
            results[name] = {
                'nan_count': nan_count,
                'nan_percentage': nan_count / len(series) * 100,
                'valid_count': len(series) - nan_count,
                'value_range': (series.min(), series.max()) if nan_count < len(series) else (np.nan, np.nan)
            }
        logger.info(f"✅ MACD: Valid across all components")
    except Exception as e:
        results['MACD'] = {'error': str(e)}
        logger.error(f"❌ MACD failed: {e}")

    # Test Bollinger Bands
    try:
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'])
        for name, series in [('BB_Upper', bb_upper), ('BB_Middle', bb_middle), ('BB_Lower', bb_lower)]:
            nan_count = series.isnull().sum()
            results[name] = {
                'nan_count': nan_count,
                'nan_percentage': nan_count / len(series) * 100,
                'valid_count': len(series) - nan_count,
                'value_range': (series.min(), series.max()) if nan_count < len(series) else (np.nan, np.nan)
            }
        logger.info(f"✅ Bollinger Bands: Valid across all components")
    except Exception as e:
        results['Bollinger_Bands'] = {'error': str(e)}
        logger.error(f"❌ Bollinger Bands failed: {e}")

    # Test Stochastic
    try:
        stoch_k, stoch_d = calculate_stochastic(df['high'], df['low'], df['close'])
        for name, series in [('Stoch_K', stoch_k), ('Stoch_D', stoch_d)]:
            nan_count = series.isnull().sum()
            results[name] = {
                'nan_count': nan_count,
                'nan_percentage': nan_count / len(series) * 100,
                'valid_count': len(series) - nan_count,
                'value_range': (series.min(), series.max()) if nan_count < len(series) else (np.nan, np.nan)
            }
        logger.info(f"✅ Stochastic: Valid across all components")
    except Exception as e:
        results['Stochastic'] = {'error': str(e)}
        logger.error(f"❌ Stochastic failed: {e}")

    # Test Williams %R
    try:
        williams_r = calculate_williams_r(df['high'], df['low'], df['close'])
        nan_count = williams_r.isnull().sum()
        results['Williams_R'] = {
            'nan_count': nan_count,
            'nan_percentage': nan_count / len(williams_r) * 100,
            'valid_count': len(williams_r) - nan_count,
            'value_range': (williams_r.min(), williams_r.max()) if nan_count < len(williams_r) else (np.nan, np.nan)
        }
        logger.info(f"✅ Williams %R: {nan_count} NaN values ({nan_count/len(williams_r)*100:.1f}%)")
    except Exception as e:
        results['Williams_R'] = {'error': str(e)}
        logger.error(f"❌ Williams %R failed: {e}")

    # Test CCI
    try:
        cci = calculate_cci(df['high'], df['low'], df['close'])
        nan_count = cci.isnull().sum()
        results['CCI'] = {
            'nan_count': nan_count,
            'nan_percentage': nan_count / len(cci) * 100,
            'valid_count': len(cci) - nan_count,
            'value_range': (cci.min(), cci.max()) if nan_count < len(cci) else (np.nan, np.nan)
        }
        logger.info(f"✅ CCI: {nan_count} NaN values ({nan_count/len(cci)*100:.1f}%)")
    except Exception as e:
        results['CCI'] = {'error': str(e)}
        logger.error(f"❌ CCI failed: {e}")

    # Test ATR
    try:
        atr = calculate_atr(df['high'], df['low'], df['close'])
        nan_count = atr.isnull().sum()
        results['ATR'] = {
            'nan_count': nan_count,
            'nan_percentage': nan_count / len(atr) * 100,
            'valid_count': len(atr) - nan_count,
            'value_range': (atr.min(), atr.max()) if nan_count < len(atr) else (np.nan, np.nan)
        }
        logger.info(f"✅ ATR: {nan_count} NaN values ({nan_count/len(atr)*100:.1f}%)")
    except Exception as e:
        results['ATR'] = {'error': str(e)}
        logger.error(f"❌ ATR failed: {e}")

    # Test ADX
    try:
        adx, di_plus, di_minus = calculate_adx(df['high'], df['low'], df['close'])
        for name, series in [('ADX', adx), ('DI_Plus', di_plus), ('DI_Minus', di_minus)]:
            nan_count = series.isnull().sum()
            results[name] = {
                'nan_count': nan_count,
                'nan_percentage': nan_count / len(series) * 100,
                'valid_count': len(series) - nan_count,
                'value_range': (series.min(), series.max()) if nan_count < len(series) else (np.nan, np.nan)
            }
        logger.info(f"✅ ADX: Valid across all components")
    except Exception as e:
        results['ADX'] = {'error': str(e)}
        logger.error(f"❌ ADX failed: {e}")

    # Test OBV
    try:
        obv = calculate_obv(df['close'], df['volume'])
        nan_count = obv.isnull().sum()
        results['OBV'] = {
            'nan_count': nan_count,
            'nan_percentage': nan_count / len(obv) * 100,
            'valid_count': len(obv) - nan_count,
            'value_range': (obv.min(), obv.max()) if nan_count < len(obv) else (np.nan, np.nan)
        }
        logger.info(f"✅ OBV: {nan_count} NaN values ({nan_count/len(obv)*100:.1f}%)")
    except Exception as e:
        results['OBV'] = {'error': str(e)}
        logger.error(f"❌ OBV failed: {e}")

    # Test Ichimoku
    try:
        tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(df['high'], df['low'], df['close'])
        for name, series in [('Ichimoku_Tenkan', tenkan), ('Ichimoku_Kijun', kijun),
                           ('Ichimoku_Senkou_A', senkou_a), ('Ichimoku_Senkou_B', senkou_b),
                           ('Ichimoku_Chikou', chikou)]:
            nan_count = series.isnull().sum()
            results[name] = {
                'nan_count': nan_count,
                'nan_percentage': nan_count / len(series) * 100,
                'valid_count': len(series) - nan_count,
                'value_range': (series.min(), series.max()) if nan_count < len(series) else (np.nan, np.nan)
            }
        logger.info(f"✅ Ichimoku: Valid across all components")
    except Exception as e:
        results['Ichimoku'] = {'error': str(e)}
        logger.error(f"❌ Ichimoku failed: {e}")

    return results

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

def main():
    """Main diagnostic function"""
    logger.info("🔍 STARTING INDICATOR DIAGNOSTICS")
    logger.info("=" * 50)

    # Load sample data
    df = load_sample_data()
    if df is None:
        return

    # Test indicators
    results = test_indicator_functions(df)

    # Analyze results
    logger.info("\\n📊 DIAGNOSTIC RESULTS SUMMARY")
    logger.info("=" * 40)

    problematic_indicators = []
    working_indicators = []

    for indicator_name, result in results.items():
        if 'error' in result:
            problematic_indicators.append((indicator_name, result['error']))
        else:
            nan_percentage = result['nan_percentage']
            if nan_percentage > 10:  # More than 10% NaN values
                problematic_indicators.append((indicator_name, f"{nan_percentage:.1f}% NaN values"))
            else:
                working_indicators.append((indicator_name, f"{nan_percentage:.1f}% NaN values"))

    logger.info(f"\\n✅ WORKING INDICATORS ({len(working_indicators)}):")
    for name, status in working_indicators:
        logger.info(f"  ✓ {name}: {status}")

    if problematic_indicators:
        logger.info(f"\\n🚨 PROBLEMATIC INDICATORS ({len(problematic_indicators)}):")
        for name, issue in problematic_indicators:
            logger.info(f"  ❌ {name}: {issue}")

        logger.info("\\n🔧 RECOMMENDATIONS:")
        logger.info("  1. Indicators with high NaN % may need longer warm-up periods")
        logger.info("  2. Check data quality for required price fields (OHLC)")
        logger.info("  3. Consider using min_periods parameter in rolling calculations")
        logger.info("  4. Implement proper NaN handling and forward/backward filling")
    else:
        logger.info("\\n🎉 ALL INDICATORS WORKING CORRECTLY!")

    logger.info("\\n✅ DIAGNOSTIC COMPLETE")

if __name__ == "__main__":
    main()