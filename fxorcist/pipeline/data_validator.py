"""
Validates and cleans market data for backtesting.

Problems Solved:
- Missing data points
- Outliers and errors
- Inconsistent formats
- Corporate action adjustments
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ValidationConfig:
    """Configuration for data validation rules."""
    # Missing data handling
    max_gap_seconds: int = 300  # Maximum allowed gap between points
    interpolation_method: str = 'linear'  # Method for filling gaps
    max_interpolation_window: int = 5  # Maximum points to interpolate
    
    # Outlier detection
    zscore_threshold: float = 4.0  # Z-score threshold for outliers
    price_change_threshold: float = 0.05  # Maximum allowed price change (5%)
    volume_zscore_threshold: float = 5.0  # Z-score threshold for volume
    
    # Price validation
    min_price_value: float = 0.00001  # Minimum valid price
    max_spread_bps: float = 100  # Maximum allowed spread in bps
    
    # Volume validation
    min_volume: float = 0  # Minimum valid volume
    max_volume_multiplier: float = 10  # Max times average volume
    
    # Time validation
    expected_timezone: str = 'UTC'
    trading_hours: Dict[str, Tuple[str, str]] = None  # Trading hours by day
    
    # Corporate actions
    adjust_prices: bool = True  # Whether to adjust for corporate actions
    adjustment_method: str = 'backward'  # forward or backward adjustment

class DataQualityValidator:
    """Validates and cleans market data for backtesting."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.logger = logging.getLogger(__name__)
        self.validation_stats = {}
        self.cleaning_history = []
    
    def validate_and_clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data validation and cleaning."""
        original_count = len(data)
        self.validation_stats = {'original_count': original_count}
        
        try:
            # 1. Basic data structure validation
            data = self._validate_structure(data)
            
            # 2. Time series consistency
            data = self._validate_time_series(data)
            
            # 3. Handle missing values
            data = self._handle_missing_values(data)
            
            # 4. Detect and handle outliers
            data = self._handle_outliers(data)
            
            # 5. Validate price relationships
            data = self._validate_price_relationships(data)
            
            # 6. Validate volume data
            data = self._validate_volume_data(data)
            
            # 7. Check for corporate actions
            if self.config.adjust_prices:
                data = self._adjust_for_corporate_actions(data)
            
            final_count = len(data)
            self.validation_stats.update({
                'final_count': final_count,
                'data_loss_pct': (original_count - final_count) / original_count * 100
            })
            
            self._log_validation_summary()
            return data
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            raise
    
    def _validate_structure(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate basic data structure and required columns."""
        required_cols = ['timestamp', 'open', 'high', 'low', 'close']
        optional_cols = ['volume', 'bid', 'ask']
        
        # Check required columns
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Ensure correct timezone
        if data['timestamp'].dt.tz is None:
            data['timestamp'] = data['timestamp'].dt.tz_localize(self.config.expected_timezone)
        elif data['timestamp'].dt.tz.zone != self.config.expected_timezone:
            data['timestamp'] = data['timestamp'].dt.tz_convert(self.config.expected_timezone)
        
        # Ensure numeric price columns
        price_cols = [col for col in data.columns if col in required_cols + optional_cols]
        for col in price_cols:
            if not pd.api.types.is_numeric_dtype(data[col]):
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        return data.sort_values('timestamp')
    
    def _validate_time_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate time series consistency."""
        # Check for duplicates
        duplicates = data['timestamp'].duplicated()
        if duplicates.any():
            self.logger.warning(f"Found {duplicates.sum()} duplicate timestamps")
            data = data[~duplicates]
        
        # Check for gaps
        time_diffs = data['timestamp'].diff()
        gaps = time_diffs > pd.Timedelta(seconds=self.config.max_gap_seconds)
        if gaps.any():
            gap_count = gaps.sum()
            self.logger.warning(f"Found {gap_count} gaps > {self.config.max_gap_seconds}s")
            self.validation_stats['gap_count'] = gap_count
            
            # Create list of gaps for reporting
            gap_list = []
            gap_starts = data.index[gaps]
            for start in gap_starts:
                gap_list.append({
                    'start': data.loc[start-1, 'timestamp'],
                    'end': data.loc[start, 'timestamp'],
                    'duration': time_diffs.loc[start]
                })
            self.validation_stats['gaps'] = gap_list
        
        # Validate trading hours if configured
        if self.config.trading_hours:
            data = self._validate_trading_hours(data)
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Track missing value counts
        missing_counts = data.isnull().sum()
        self.validation_stats['missing_counts'] = missing_counts.to_dict()
        
        if missing_counts.any():
            self.logger.info(f"Found missing values:\n{missing_counts[missing_counts > 0]}")
            
            # Handle different types of missing data
            data = data.copy()
            
            # 1. Forward fill small gaps
            small_gaps = data.isnull() & (
                data.index.to_series().diff() <= 
                pd.Timedelta(seconds=self.config.max_gap_seconds)
            )
            if small_gaps.any().any():
                data.loc[:, small_gaps.any()] = data.loc[:, small_gaps.any()].fillna(
                    method='ffill',
                    limit=self.config.max_interpolation_window
                )
            
            # 2. Interpolate remaining gaps if possible
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].interpolate(
                method=self.config.interpolation_method,
                limit=self.config.max_interpolation_window
            )
            
            # 3. Drop any remaining rows with missing values
            missing_after = data.isnull().sum()
            if missing_after.any():
                self.logger.warning(
                    f"Dropping {len(data[data.isnull().any(axis=1)])} rows with "
                    "remaining missing values"
                )
                data = data.dropna()
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle price outliers using statistical methods."""
        price_cols = ['open', 'high', 'low', 'close']
        
        # Track outliers
        outlier_stats = {}
        data = data.copy()
        
        for col in price_cols:
            # Calculate rolling z-score
            rolling_mean = data[col].rolling(window=20).mean()
            rolling_std = data[col].rolling(window=20).std()
            z_scores = np.abs((data[col] - rolling_mean) / rolling_std)
            
            # Identify outliers
            outliers = z_scores > self.config.zscore_threshold
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                self.logger.warning(f"Found {outlier_count} outliers in {col}")
                
                # Record outlier details
                outlier_stats[col] = {
                    'count': outlier_count,
                    'timestamps': data.loc[outliers, 'timestamp'].tolist(),
                    'values': data.loc[outliers, col].tolist(),
                    'z_scores': z_scores[outliers].tolist()
                }
                
                # Replace outliers with interpolated values
                data.loc[outliers, col] = np.nan
                data[col] = data[col].interpolate(method='linear')
        
        self.validation_stats['outliers'] = outlier_stats
        return data
    
    def _validate_price_relationships(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLC relationships and bid/ask spreads."""
        violations = []
        data = data.copy()
        
        # OHLC relationship checks
        high_violations = data['high'] < np.maximum(data['open'], data['close'])
        low_violations = data['low'] > np.minimum(data['open'], data['close'])
        
        if high_violations.any() or low_violations.any():
            violation_idx = high_violations | low_violations
            violations.extend({
                'timestamp': ts,
                'type': 'OHLC_violation',
                'details': {
                    'open': o,
                    'high': h,
                    'low': l,
                    'close': c
                }
            } for ts, o, h, l, c in data.loc[
                violation_idx,
                ['timestamp', 'open', 'high', 'low', 'close']
            ].values)
            
            # Fix violations by adjusting high/low
            data.loc[high_violations, 'high'] = np.maximum(
                data.loc[high_violations, 'open'],
                data.loc[high_violations, 'close']
            )
            data.loc[low_violations, 'low'] = np.minimum(
                data.loc[low_violations, 'open'],
                data.loc[low_violations, 'close']
            )
        
        # Bid-ask spread checks if available
        if all(col in data.columns for col in ['bid', 'ask']):
            spread_violations = data['ask'] <= data['bid']
            wide_spreads = (
                (data['ask'] - data['bid']) / data['bid'] * 10000 >
                self.config.max_spread_bps
            )
            
            if spread_violations.any() or wide_spreads.any():
                violation_idx = spread_violations | wide_spreads
                violations.extend({
                    'timestamp': ts,
                    'type': 'spread_violation',
                    'details': {
                        'bid': b,
                        'ask': a,
                        'spread_bps': (a - b) / b * 10000
                    }
                } for ts, b, a in data.loc[
                    violation_idx,
                    ['timestamp', 'bid', 'ask']
                ].values)
                
                # Remove rows with invalid spreads
                data = data[~spread_violations]
        
        if violations:
            self.validation_stats['price_violations'] = violations
            self.logger.warning(f"Found {len(violations)} price relationship violations")
        
        return data
    
    def _validate_volume_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate volume data for anomalies."""
        if 'volume' not in data.columns:
            return data
        
        data = data.copy()
        volume_stats = {}
        
        # Check for negative volumes
        negative_volume = data['volume'] < 0
        if negative_volume.any():
            self.logger.warning(f"Found {negative_volume.sum()} negative volume values")
            data.loc[negative_volume, 'volume'] = 0
            volume_stats['negative_count'] = negative_volume.sum()
        
        # Check for zero volumes
        zero_volume = data['volume'] == 0
        if zero_volume.any():
            self.logger.info(f"Found {zero_volume.sum()} zero volume values")
            volume_stats['zero_count'] = zero_volume.sum()
        
        # Check for volume spikes
        rolling_mean = data['volume'].rolling(window=20).mean()
        rolling_std = data['volume'].rolling(window=20).std()
        volume_zscores = np.abs((data['volume'] - rolling_mean) / rolling_std)
        
        volume_spikes = volume_zscores > self.config.volume_zscore_threshold
        if volume_spikes.any():
            self.logger.warning(f"Found {volume_spikes.sum()} volume spikes")
            volume_stats['spike_count'] = volume_spikes.sum()
            
            # Record spike details
            volume_stats['spikes'] = [{
                'timestamp': ts,
                'volume': v,
                'zscore': z
            } for ts, v, z in data.loc[
                volume_spikes,
                ['timestamp', 'volume']
            ].join(pd.Series(volume_zscores[volume_spikes], name='zscore')).values]
            
            # Cap extreme volumes
            data.loc[volume_spikes, 'volume'] = rolling_mean[volume_spikes] + (
                rolling_std[volume_spikes] * self.config.volume_zscore_threshold
            )
        
        self.validation_stats['volume'] = volume_stats
        return data
    
    def _validate_trading_hours(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate that data points fall within expected trading hours."""
        valid_times = pd.Series(True, index=data.index)
        
        for day, (start, end) in self.config.trading_hours.items():
            # Convert day name to integer (0 = Monday, 6 = Sunday)
            day_num = pd.Timestamp(day).weekday()
            
            # Create time objects for start and end
            start_time = pd.Timestamp(start).time()
            end_time = pd.Timestamp(end).time()
            
            # Find points on this day
            day_mask = data['timestamp'].dt.weekday == day_num
            
            # Check if points are within trading hours
            time_mask = (
                (data['timestamp'].dt.time >= start_time) &
                (data['timestamp'].dt.time <= end_time)
            )
            
            valid_times &= ~day_mask | (day_mask & time_mask)
        
        invalid_count = (~valid_times).sum()
        if invalid_count > 0:
            self.logger.warning(
                f"Found {invalid_count} points outside trading hours"
            )
            self.validation_stats['outside_trading_hours'] = invalid_count
            data = data[valid_times]
        
        return data
    
    def _adjust_for_corporate_actions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adjust prices for corporate actions."""
        # This would integrate with a corporate actions database
        # For now, return unadjusted data
        return data
    
    def _log_validation_summary(self):
        """Log summary of validation results."""
        self.logger.info("\n=== Data Validation Summary ===")
        self.logger.info(f"Original rows: {self.validation_stats['original_count']}")
        self.logger.info(f"Final rows: {self.validation_stats['final_count']}")
        self.logger.info(
            f"Data loss: {self.validation_stats['data_loss_pct']:.2f}%"
        )
        
        if 'gap_count' in self.validation_stats:
            self.logger.info(f"Time gaps found: {self.validation_stats['gap_count']}")
        
        if 'outliers' in self.validation_stats:
            total_outliers = sum(
                stats['count'] for stats in self.validation_stats['outliers'].values()
            )
            self.logger.info(f"Total outliers removed: {total_outliers}")
        
        if 'price_violations' in self.validation_stats:
            self.logger.info(
                f"Price violations fixed: {len(self.validation_stats['price_violations'])}"
            )
        
        if 'volume' in self.validation_stats:
            vol_stats = self.validation_stats['volume']
            self.logger.info(
                f"Volume anomalies - Negative: {vol_stats.get('negative_count', 0)}, "
                f"Zero: {vol_stats.get('zero_count', 0)}, "
                f"Spikes: {vol_stats.get('spike_count', 0)}"
            )