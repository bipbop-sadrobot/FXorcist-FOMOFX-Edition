import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("logs/validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
STANDARD_COLUMNS = {
    'timestamp': ['timestamp', 'date', 'time', 'datetime'],
    'open': ['open', 'Open', 'OPEN'],
    'high': ['high', 'High', 'HIGH'],
    'low': ['low', 'Low', 'LOW'],
    'close': ['close', 'Close', 'CLOSE'],
    'volume': ['volume', 'Volume', 'VOLUME']
}

def get_standard_column_name(col: str) -> Optional[str]:
    """Map various column names to standard names."""
    col = col.lower().strip()
    for std_name, variants in STANDARD_COLUMNS.items():
        if col in [v.lower() for v in variants]:
            return std_name
    return None

def validate_forex_data(
    df: pd.DataFrame,
    required_cols: List[str] = None,
    price_cols: List[str] = None,
    timestamp_col: str = None,
    volume_col: str = None,
    max_missing_perc: float = 5.0,
    z_threshold: float = 4.0,  # Increased for forex volatility
    freshness_days: Optional[int] = None,  # Optional for historical data
    holidays: List[pd.Timestamp] = None,
    is_historical: bool = False
) -> bool:
    """Enhanced validation for forex data with adaptive thresholds and historical data support."""
    # Initialize parameters with defaults if not provided
    required_cols = required_cols or ['timestamp', 'open', 'high', 'low', 'close']
    price_cols = price_cols or ['open', 'high', 'low', 'close']
    holidays = holidays or []
    issues = []
    warnings = []

    # Standardize column names
    df.columns = [get_standard_column_name(col) or col for col in df.columns]
    
    # 1. Schema & Types
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
        
    # Identify timestamp column if not provided
    if not timestamp_col:
        timestamp_candidates = [col for col in df.columns if col.lower() in STANDARD_COLUMNS['timestamp']]
        if timestamp_candidates:
            timestamp_col = timestamp_candidates[0]
        else:
            issues.append("No timestamp column found")
            return False

    # Convert timestamp
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    invalid_timestamps = df[timestamp_col].isnull()
    if invalid_timestamps.any():
        warnings.append(f"Found {invalid_timestamps.sum()} invalid timestamps")
        df = df.dropna(subset=[timestamp_col])
    
    # Set index for time-based operations
    df = df.set_index(timestamp_col).sort_index()

    # 2. Data Type Validation
    for col in price_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                warnings.append(f"Non-numeric values in {col}")
                df[col] = df[col].fillna(method='ffill')
    
    if volume_col and volume_col in df.columns:
        df[volume_col] = pd.to_numeric(df[volume_col], errors='coerce').fillna(0)

    # 3. Price Consistency
    if all(col in df.columns for col in ['high', 'low']):
        invalid_hl = df['high'] < df['low']
        if invalid_hl.any():
            warnings.append(f"Found {invalid_hl.sum()} invalid high/low pairs")
            # Swap invalid high/low pairs
            df.loc[invalid_hl, ['high', 'low']] = df.loc[invalid_hl, ['low', 'high']].values

    # 4. Completeness Check
    missing_perc = df[price_cols].isnull().mean().mean() * 100
    if missing_perc > max_missing_perc:
        warnings.append(f"Missing data: {missing_perc:.2f}% (threshold: {max_missing_perc}%)")
    
    # Check for large gaps (adaptive to data frequency)
    median_interval = df.index.to_series().diff().median()
    gap_threshold = max(median_interval * 5, timedelta(days=1))
    gaps = df.index.to_series().diff() > gap_threshold
    if gaps.any():
        warnings.append(f"Found {gaps.sum()} large gaps (>{gap_threshold})")

    # 5. Outlier Detection (adaptive thresholds)
    for col in price_cols:
        if col in df.columns:
            # Calculate rolling statistics for adaptive thresholds
            rolling_std = df[col].rolling(window=20, min_periods=1).std()
            rolling_mean = df[col].rolling(window=20, min_periods=1).mean()
            z_scores = np.abs((df[col] - rolling_mean) / rolling_std)
            extreme_outliers = z_scores > z_threshold
            
            if extreme_outliers.any():
                outlier_count = extreme_outliers.sum()
                outlier_perc = (outlier_count / len(df)) * 100
                if outlier_perc > 1:  # Allow up to 1% outliers
                    warnings.append(f"High outlier percentage in {col}: {outlier_perc:.2f}%")

    # 6. Market Microstructure Analysis
    if 'close' in df.columns:
        returns = np.log(df['close'] / df['close'].shift(1)).dropna()
        
        # Volatility clustering check
        rolling_vol = returns.rolling(window=20, min_periods=1).std()
        vol_clustering = np.corrcoef(rolling_vol[20:], rolling_vol[:-20])[0,1]
        if vol_clustering > 0.7:
            warnings.append(f"Strong volatility clustering detected: {vol_clustering:.2f}")
        
        # Kurtosis check (adaptive to market conditions)
        kurt = returns.kurtosis()
        if kurt > 30:  # Extremely fat tails
            warnings.append(f"Extreme kurtosis in returns: {kurt:.2f}")

    # 7. Stationarity Test (for non-historical data only)
    if not is_historical and 'close' in df.columns and len(df) > 20:
        try:
            adf = adfuller(df['close'])
            if adf[1] > 0.05:
                warnings.append(f"Non-stationary price series (ADF p-value: {adf[1]:.3f})")
        except Exception as e:
            logger.warning(f"Stationarity test failed: {str(e)}")

    # 8. Freshness Check (only for non-historical data)
    if not is_historical and freshness_days:
        latest = df.index.max()
        staleness = (datetime.now() - latest).days
        if staleness > freshness_days:
            warnings.append(f"Data staleness: {staleness} days (threshold: {freshness_days})")

    # 9. Trading Calendar Check
    if not is_historical:
        weekends = df.index.dayofweek.isin([5, 6])
        if weekends.any():
            warnings.append(f"Found {weekends.sum()} weekend data points")
        
        if holidays:
            holiday_data = df.index.isin(holidays)
            if holiday_data.any():
                warnings.append(f"Found {holiday_data.sum()} holiday data points")

    # Log validation results
    if issues:
        error_msg = "\n".join(issues)
        logger.error(f"Validation failed:\n{error_msg}")
        raise ValueError(error_msg)
    
    if warnings:
        warning_msg = "\n".join(warnings)
        logger.warning(f"Validation warnings:\n{warning_msg}")
    
    logger.info("Validation completed successfully")
    return True

def run_validation_tests():
    """Run comprehensive validation tests with various scenarios."""
    # Test data
    test_data = {
        'timestamp': pd.date_range(start='2025-08-01', periods=100, freq='1min'),
        'open': np.random.uniform(1.1000, 1.2000, 100),
        'high': np.random.uniform(1.1500, 1.2500, 100),
        'low': np.random.uniform(1.0500, 1.1500, 100),
        'close': np.random.uniform(1.1000, 1.2000, 100),
        'volume': np.random.randint(1000, 5000, 100)
    }
    df = pd.DataFrame(test_data)
    
    # Test cases
    test_cases = [
        ("Basic validation", df, {}),
        ("Historical data", df, {"is_historical": True}),
        ("Missing volume", df.drop('volume', axis=1), {}),
        ("Custom thresholds", df, {"z_threshold": 5.0, "max_missing_perc": 10.0}),
    ]
    
    for test_name, test_df, kwargs in test_cases:
        try:
            logger.info(f"\nRunning test: {test_name}")
            validate_forex_data(test_df, **kwargs)
            logger.info(f"Test passed: {test_name}")
        except Exception as e:
            logger.error(f"Test failed: {test_name}\nError: {str(e)}")

if __name__ == "__main__":
    run_validation_tests()
