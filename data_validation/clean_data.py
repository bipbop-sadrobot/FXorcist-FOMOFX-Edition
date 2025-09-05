import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
from scipy import stats
from scipy.signal import medfilt
from statsmodels.robust import mad

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("logs/cleaning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
CLEANING_METHODS = {
    'hampel': 'Hampel filter for outlier detection',
    'rolling_zscore': 'Rolling z-score filter',
    'median': 'Median filter',
    'ewma': 'Exponentially weighted moving average'
}

class AdaptiveThresholds:
    """Dynamic threshold calculation based on market conditions."""
    
    def __init__(self, data: pd.DataFrame, price_cols: List[str]):
        self.data = data
        self.price_cols = price_cols
        self.volatility = self._calculate_volatility()
        self.thresholds = self._set_thresholds()
    
    def _calculate_volatility(self) -> float:
        """Calculate overall market volatility."""
        if 'close' in self.data.columns:
            returns = np.log(self.data['close'] / self.data['close'].shift(1))
            return returns.std()
        return np.nan
    
    def _set_thresholds(self) -> Dict[str, float]:
        """Set adaptive thresholds based on market conditions."""
        base_threshold = 4.0  # Base z-score threshold
        vol_factor = min(2.0, max(0.5, self.volatility * 100))
        
        return {
            'zscore': base_threshold * vol_factor,
            'hampel': 3 * vol_factor,
            'mad': 3.5 * vol_factor,
            'gap': pd.Timedelta(minutes=5 if self.volatility > 0.001 else 2)
        }

def hampel_filter(series: pd.Series, window: int = 10, threshold: float = 3) -> pd.Series:
    """
    Apply Hampel filter for outlier detection and replacement.
    Uses median absolute deviation (MAD) for robust statistics.
    """
    rolling_median = series.rolling(window=window, center=True, min_periods=1).median()
    rolling_mad = mad(series.rolling(window=window, center=True, min_periods=1))
    diff = np.abs(series - rolling_median)
    outliers = diff > (threshold * rolling_mad)
    return pd.Series(np.where(outliers, rolling_median, series), index=series.index)

def adaptive_clean_forex_data(
    df: pd.DataFrame,
    price_cols: List[str] = None,
    methods: List[str] = None,
    window_size: int = 20,
    is_intraday: bool = True
) -> pd.DataFrame:
    """
    Enhanced forex data cleaning with adaptive methods and market-aware processing.
    
    Features:
    1. Adaptive thresholds based on market volatility
    2. Multiple cleaning methods (Hampel, rolling z-score, median filter)
    3. Intelligent gap filling based on market conditions
    4. Price consistency enforcement
    5. Volatility-aware outlier detection
    6. Incremental processing support
    """
    price_cols = price_cols or ['open', 'high', 'low', 'close']
    methods = methods or ['hampel', 'rolling_zscore']
    
    logger.info(f"Starting cleaning process with methods: {methods}")
    original_shape = df.shape
    
    # Initialize adaptive thresholds
    thresholds = AdaptiveThresholds(df, price_cols)
    logger.info(f"Adaptive thresholds set: {thresholds.thresholds}")
    
    try:
        # 1. Initial data preparation
        df = df.copy()
        df = df.sort_index()
        
        # 2. Handle missing values with adaptive gap filling
        for col in price_cols:
            missing_gaps = df[col].isnull()
            if missing_gaps.any():
                gap_sizes = missing_gaps.astype(int).groupby(
                    (missing_gaps.astype(int).diff() != 0).cumsum()
                ).sum()
                
                for gap_size in gap_sizes.unique():
                    if gap_size <= 5:  # Small gaps
                        df[col] = df[col].interpolate(method='linear')
                    else:  # Larger gaps
                        df[col] = df[col].fillna(method='ffill')
        
        # 3. Apply cleaning methods
        cleaned_df = df.copy()
        for col in price_cols:
            if 'hampel' in methods:
                cleaned_df[col] = hampel_filter(
                    cleaned_df[col],
                    window=window_size,
                    threshold=thresholds.thresholds['hampel']
                )
            
            if 'rolling_zscore' in methods:
                rolling_mean = cleaned_df[col].rolling(window=window_size, min_periods=1).mean()
                rolling_std = cleaned_df[col].rolling(window=window_size, min_periods=1).std()
                z_scores = np.abs((cleaned_df[col] - rolling_mean) / rolling_std)
                outliers = z_scores > thresholds.thresholds['zscore']
                cleaned_df.loc[outliers, col] = rolling_mean[outliers]
            
            if 'median' in methods:
                cleaned_df[col] = pd.Series(
                    medfilt(cleaned_df[col], kernel_size=5),
                    index=cleaned_df.index
                )
        
        # 4. Ensure price consistency
        if all(col in cleaned_df.columns for col in ['high', 'low']):
            invalid_hl = cleaned_df['high'] < cleaned_df['low']
            if invalid_hl.any():
                logger.warning(f"Correcting {invalid_hl.sum()} invalid high/low pairs")
                cleaned_df.loc[invalid_hl, ['high', 'low']] = cleaned_df.loc[invalid_hl, ['low', 'high']].values
        
        # 5. Add quality metrics
        if is_intraday:
            cleaned_df['quality_score'] = 1.0
            for col in price_cols:
                # Penalize based on number of corrections
                corrections = (df[col] != cleaned_df[col]).sum() / len(df)
                cleaned_df['quality_score'] *= (1 - corrections)
        
        # 6. Add cleaning metadata
        cleaned_df.attrs['cleaning_info'] = {
            'methods_applied': methods,
            'thresholds': thresholds.thresholds,
            'window_size': window_size,
            'original_shape': original_shape,
            'cleaned_shape': cleaned_df.shape,
            'timestamp': pd.Timestamp.now()
        }
        
        logger.info(
            f"Cleaning complete. Original shape: {original_shape}, "
            f"Cleaned shape: {cleaned_df.shape}, "
            f"Average quality score: {cleaned_df['quality_score'].mean():.3f}"
        )
        
        return cleaned_df
    
    except Exception as e:
        logger.error(f"Error during cleaning: {str(e)}", exc_info=True)
        raise

def run_cleaning_tests():
    """Run comprehensive cleaning tests with various scenarios."""
    # Generate test data
    np.random.seed(42)
    dates = pd.date_range(start='2025-08-01', periods=1000, freq='1min')
    base_price = 1.1000
    
    # Generate realistic forex price movements
    returns = np.random.normal(0, 0.0002, 1000)  # Small random changes
    prices = base_price * np.exp(np.cumsum(returns))  # Log-normal price process
    
    # Add some artificial anomalies
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.002, 1000)),
        'low': prices * (1 - np.random.uniform(0, 0.002, 1000)),
        'close': prices * (1 + np.random.normal(0, 0.001, 1000)),
        'volume': np.random.randint(1000, 5000, 1000)
    })
    
    # Add various types of anomalies
    test_data.loc[100:102, 'high'] *= 1.1  # Price spikes
    test_data.loc[300:305, 'low'] *= 0.9  # Price drops
    test_data.loc[500:510, 'close'] = np.nan  # Missing values
    test_data.loc[700, 'high'] = test_data.loc[700, 'low'] * 0.9  # Invalid high/low
    
    # Test different cleaning configurations
    test_cases = [
        ("Basic cleaning", {'methods': ['hampel']}),
        ("Multiple methods", {'methods': ['hampel', 'rolling_zscore']}),
        ("Intraday cleaning", {'methods': ['hampel', 'median'], 'is_intraday': True}),
        ("Custom window", {'methods': ['rolling_zscore'], 'window_size': 30}),
    ]
    
    for test_name, kwargs in test_cases:
        try:
            logger.info(f"\nRunning test: {test_name}")
            cleaned = adaptive_clean_forex_data(test_data.set_index('timestamp'), **kwargs)
            
            # Verify cleaning results
            logger.info(f"Test {test_name} results:")
            logger.info(f"- Missing values: {cleaned.isnull().sum().sum()}")
            logger.info(f"- Quality score: {cleaned['quality_score'].mean():.3f}")
            logger.info(f"- Methods applied: {cleaned.attrs['cleaning_info']['methods_applied']}")
            
        except Exception as e:
            logger.error(f"Test failed: {test_name}\nError: {str(e)}")

if __name__ == "__main__":
    run_cleaning_tests()
