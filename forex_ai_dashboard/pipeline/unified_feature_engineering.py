"""
Unified Feature Engineering Module for Forex AI
Consolidates all feature engineering logic from scattered implementations.
Provides a single, comprehensive interface for all training pipelines.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
import ta  # Technical Analysis library

logger = logging.getLogger(__name__)

class UnifiedFeatureEngineer:
    """Unified feature engineering for all forex training pipelines."""

    def __init__(self):
        self.feature_groups = {
            'basic': ['returns', 'log_returns'],
            'momentum': ['rsi', 'macd', 'stoch', 'williams_r'],
            'volatility': ['bb_upper', 'bb_lower', 'bb_middle', 'bb_pct', 'atr'],
            'trend': ['sma', 'ema', 'trend_strength'],
            'volume': ['volume_sma', 'volume_ratio', 'volume_intensity'],
            'microstructure': ['spread', 'gap_up', 'gap_down'],
            'time': ['hour', 'day_of_week', 'month']
        }

    def process_data(self,
                    df: pd.DataFrame,
                    feature_groups: Optional[List[str]] = None,
                    custom_features: Optional[Dict] = None) -> pd.DataFrame:
        """
        Process data with comprehensive feature engineering.

        Args:
            df: Input dataframe with OHLCV data
            feature_groups: List of feature groups to include
            custom_features: Custom feature configurations

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Ensure timestamp is datetime
        if 'timestamp' not in df.columns:
            timestamp_col = df.columns[0]
            df = df.rename(columns={timestamp_col: 'timestamp'})

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Default to all feature groups if none specified
        if feature_groups is None:
            feature_groups = list(self.feature_groups.keys())

        # Add features by group
        for group in feature_groups:
            if group == 'basic':
                df = self._add_basic_features(df)
            elif group == 'momentum':
                df = self._add_momentum_features(df)
            elif group == 'volatility':
                df = self._add_volatility_features(df)
            elif group == 'trend':
                df = self._add_trend_features(df)
            elif group == 'volume':
                df = self._add_volume_features(df)
            elif group == 'microstructure':
                df = self._add_microstructure_features(df)
            elif group == 'time':
                df = self._add_time_features(df)

        # Add custom features if provided
        if custom_features:
            df = self._add_custom_features(df, custom_features)

        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')

        logger.info(f"Feature engineering complete: {len(df)} rows, {len(df.columns)} columns")
        return df

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price and return features."""
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Price changes
        df['price_change'] = df['close'] - df['open']
        df['high_low_range'] = df['high'] - df['low']

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # RSI with multiple periods
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = ta.momentum.RSIIndicator(
                df['close'], window=period
            ).rsi()

        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(
            df['high'], df['low'], df['close']
        ).williams_r()

        # Momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_pct'] = bb.bollinger_pband()

        # Average True Range
        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close']
        ).average_true_range()

        # Rolling volatility
        for window in [10, 20, 30]:
            df[f'volatility_{window}'] = df['returns'].rolling(window, min_periods=1).std()

        # Price volatility
        df['price_volatility'] = (df['high'] - df['low']) / df['close']

        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators."""
        # Simple Moving Averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period, min_periods=1).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, min_periods=1).mean()

        # Trend strength (difference between short and long term trends)
        df['trend_strength'] = df['ema_10'] - df['ema_50']

        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()

        # Ichimoku Cloud (simplified)
        df['tenkan_sen'] = (df['high'].rolling(9, min_periods=1).max() + df['low'].rolling(9, min_periods=1).min()) / 2
        df['kijun_sen'] = (df['high'].rolling(26, min_periods=1).max() + df['low'].rolling(26, min_periods=1).min()) / 2

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        if 'volume' not in df.columns:
            logger.warning("Volume column not found, skipping volume features")
            return df

        # Volume moving averages
        for period in [10, 20, 30]:
            df[f'volume_sma_{period}'] = df['volume'].rolling(period, min_periods=1).mean()

        # Volume ratios
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20, min_periods=1).mean()

        # Volume intensity
        df['volume_intensity'] = (df['volume'] * np.abs(df['returns'])).rolling(20, min_periods=1).mean()

        # On-balance volume
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(
            df['close'], df['volume']
        ).on_balance_volume()

        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        # Spread (high-low range as proxy)
        df['spread'] = df['high'] - df['low']

        # Gap detection
        df['gap_up'] = (df['low'] > df['high'].shift(1)).astype(int)
        df['gap_down'] = (df['high'] < df['low'].shift(1)).astype(int)

        # Price impact
        df['price_impact'] = np.abs(df['close'] - df['open']) / df['spread']

        # Realized volatility
        df['realized_volatility'] = df['returns'].rolling(20, min_periods=1).std() * np.sqrt(252)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter

        # Business day indicator
        df['is_business_day'] = df['day_of_week'].isin([0, 1, 2, 3, 4]).astype(int)

        # Time of day categories
        df['time_of_day'] = pd.cut(
            df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening']
        )

        return df

    def _add_custom_features(self, df: pd.DataFrame, custom_features: Dict) -> pd.DataFrame:
        """Add custom features based on configuration."""
        for feature_name, config in custom_features.items():
            try:
                if config['type'] == 'rolling_mean':
                    df[feature_name] = df[config['column']].rolling(config['window'], min_periods=1).mean()
                elif config['type'] == 'rolling_std':
                    df[feature_name] = df[config['column']].rolling(config['window'], min_periods=1).std()
                elif config['type'] == 'lag':
                    df[feature_name] = df[config['column']].shift(config['periods'])
                elif config['type'] == 'ratio':
                    df[feature_name] = df[config['numerator']] / df[config['denominator']]
            except Exception as e:
                logger.warning(f"Failed to add custom feature {feature_name}: {str(e)}")

        return df

    def get_feature_list(self, feature_groups: Optional[List[str]] = None) -> List[str]:
        """Get list of features that would be generated."""
        if feature_groups is None:
            feature_groups = list(self.feature_groups.keys())

        features = []
        for group in feature_groups:
            if group in self.feature_groups:
                features.extend(self.feature_groups[group])

        return features

    def validate_features(self, df: pd.DataFrame) -> Dict[str, int]:
        """Validate generated features for quality issues."""
        validation_results = {
            'total_features': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'infinite_values': np.isinf(df.select_dtypes(include=[np.number])).sum().sum(),
            'constant_features': 0,
            'high_correlation_features': 0
        }

        # Check for constant features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].nunique() <= 1:
                validation_results['constant_features'] += 1

        # Check for highly correlated features (correlation > 0.95)
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            high_corr = (corr_matrix.abs() > 0.95).sum().sum() - len(numeric_cols)  # Subtract diagonal
            validation_results['high_correlation_features'] = high_corr // 2  # Divide by 2 for symmetry

        return validation_results