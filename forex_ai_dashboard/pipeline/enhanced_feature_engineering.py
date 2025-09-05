#!/usr/bin/env python3
"""
Enhanced Feature Engineering Module for Forex AI
Advanced indicators, feature selection, and automated feature discovery.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import ta  # Technical Analysis library
from .safe_aroon import SafeAroonIndicator

logger = logging.getLogger(__name__)

class EnhancedFeatureEngineer:
    """Enhanced feature engineering with advanced indicators and selection."""

    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.pca = None

        # Advanced feature groups
        self.advanced_feature_groups = {
            'microstructure': ['realized_volatility', 'price_impact', 'spread_analysis'],
            'advanced_momentum': ['tsi', 'uo', 'stoch_rsi', 'williams_r', 'awesome_oscillator'],
            'advanced_trend': ['vortex', 'aroon', 'chandelier_exit', 'kst'],
            'volume_advanced': ['volume_price_trend', 'ease_of_movement', 'force_index', 'negative_volume_index'],
            'cycle': ['schaff_trend_cycle', 'cycle_indicator'],
            'volatility_advanced': ['ulcer_index', 'donchian_channels', 'keltner_channels'],
            'statistical': ['zscore', 'percentile', 'rolling_skew', 'rolling_kurtosis']
        }

    def process_data(self,
                    df: pd.DataFrame,
                    feature_groups: Optional[List[str]] = None,
                    target_column: str = 'close',
                    n_features: Optional[int] = None,
                    use_pca: bool = False,
                    pca_components: Optional[int] = None) -> pd.DataFrame:
        """
        Process data with enhanced feature engineering.

        Args:
            df: Input dataframe with OHLCV data
            feature_groups: List of feature groups to include
            target_column: Target column for feature selection
            n_features: Number of features to select (if None, keep all)
            use_pca: Whether to apply PCA
            pca_components: Number of PCA components

        Returns:
            DataFrame with enhanced features
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
            feature_groups = list(self.advanced_feature_groups.keys()) + ['basic', 'momentum', 'volatility', 'trend', 'time']

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
            elif group == 'time':
                df = self._add_time_features(df)
            elif group in self.advanced_feature_groups:
                df = self._add_advanced_features(df, group)

        # Add target for feature selection
        if target_column in df.columns:
            df['temp_target'] = df[target_column].shift(-1)
            df = df.dropna(subset=['temp_target'])

            # Feature selection
            if n_features:
                df = self._select_features(df, n_features, target_column='temp_target')

            # Remove temporary target
            df = df.drop(columns=['temp_target'])

        # Apply PCA if requested
        if use_pca and pca_components:
            df = self._apply_pca(df, pca_components)

        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')

        logger.info(f"Enhanced feature engineering complete: {len(df)} rows, {len(df.columns)} columns")
        return df

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced basic features."""
        # Price changes with multiple horizons
        for lag in [1, 2, 3, 5, 10]:
            df[f'returns_{lag}'] = df['close'].pct_change(lag)
            df[f'log_returns_{lag}'] = np.log(df['close'] / df['close'].shift(lag))

        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']

        # Volume ratios (if volume exists)
        if 'volume' in df.columns:
            df['volume_change'] = df['volume'].pct_change().replace([np.inf, -np.inf], np.nan)
            df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(window=20, min_periods=1).mean().replace(0, np.nan)

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # RSI with multiple periods
        for period in [7, 14, 21, 28]:
            df[f'rsi_{period}'] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()

        # MACD with different parameters
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()

        # Momentum with different periods
        for period in [5, 10, 15, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        # Bollinger Bands with different periods
        for period in [10, 20, 30]:
            bb = ta.volatility.BollingerBands(df['close'], window=period)
            df[f'bb_upper_{period}'] = bb.bollinger_hband()
            df[f'bb_lower_{period}'] = bb.bollinger_lband()
            df[f'bb_middle_{period}'] = bb.bollinger_mavg()
            df[f'bb_pct_{period}'] = bb.bollinger_pband()

        # Average True Range
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

        # Rolling volatility
        for window in [5, 10, 20, 30]:
            df[f'volatility_{window}'] = df['returns_1'].rolling(window=window, min_periods=1).std()

        # Parkinson volatility (if high/low available)
        df['parkinson_vol'] = np.sqrt((1/(4*np.log(2))) * ((np.log(df['high']/df['low']))**2))

        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators."""
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100]:
            df[f'sma_{period}'] = df['close'].rolling(window=period, min_periods=1).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, min_periods=1).mean()

        # Trend strength indicators
        df['trend_strength_10'] = df['ema_10'] - df['ema_50']
        df['trend_strength_20'] = df['ema_20'] - df['ema_50']

        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()

        # Ichimoku Cloud
        df['tenkan_sen'] = (df['high'].rolling(window=9, min_periods=1).max() + df['low'].rolling(window=9, min_periods=1).min()) / 2
        df['kijun_sen'] = (df['high'].rolling(26, min_periods=1).max() + df['low'].rolling(26, min_periods=1).min()) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        df['senkou_span_b'] = ((df['high'].rolling(52, min_periods=1).max() + df['low'].rolling(52, min_periods=1).min()) / 2).shift(26)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced time-based features."""
        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['day_of_year'] = df['timestamp'].dt.dayofyear

        # Business day indicator
        df['is_business_day'] = df['day_of_week'].isin([0, 1, 2, 3, 4]).astype(int)

        # Time of day categories (convert to numeric codes)
        time_of_day_cat = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=['night', 'morning', 'afternoon', 'evening'])
        df['time_of_day'] = time_of_day_cat.cat.codes.astype(float)

        # Seasonal features
        df['is_month_end'] = df['timestamp'].dt.is_month_end.astype(int)
        df['is_month_start'] = df['timestamp'].dt.is_month_start.astype(int)
        df['is_quarter_end'] = df['timestamp'].dt.is_quarter_end.astype(int)
        df['is_quarter_start'] = df['timestamp'].dt.is_quarter_start.astype(int)

        return df

    def _add_advanced_features(self, df: pd.DataFrame, group: str) -> pd.DataFrame:
        """Add advanced feature groups."""
        if group == 'microstructure':
            df = self._add_microstructure_features(df)
        elif group == 'advanced_momentum':
            df = self._add_advanced_momentum_features(df)
        elif group == 'advanced_trend':
            df = self._add_advanced_trend_features(df)
        elif group == 'volume_advanced':
            df = self._add_volume_advanced_features(df)
        elif group == 'cycle':
            df = self._add_cycle_features(df)
        elif group == 'volatility_advanced':
            df = self._add_volatility_advanced_features(df)
        elif group == 'statistical':
            df = self._add_statistical_features(df)

        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        # Spread analysis (define spread first)
        df['spread'] = df['high'] - df['low']
        df['relative_spread'] = df['spread'] / df['close']

        # Realized volatility
        df['realized_volatility'] = df['returns_1'].rolling(20, min_periods=1).std() * np.sqrt(252)

        # Price impact (now spread is defined)
        spread_mean = df['spread'].rolling(window=20, min_periods=1).mean()
        df['price_impact'] = np.abs(df['close'] - df['open']) / spread_mean.replace(0, np.nan)

        # Order flow (simplified)
        df['buy_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['sell_pressure'] = 1 - df['buy_pressure']

        return df

    def _add_advanced_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced momentum indicators."""
        # True Strength Index (TSI)
        df['tsi'] = ta.momentum.TSIIndicator(df['close']).tsi()

        # Ultimate Oscillator
        df['uo'] = ta.momentum.UltimateOscillator(df['high'], df['low'], df['close']).ultimate_oscillator()

        # Stochastic RSI
        df['stoch_rsi'] = ta.momentum.StochRSIIndicator(df['close']).stochrsi()

        # Awesome Oscillator
        df['awesome_oscillator'] = ta.momentum.AwesomeOscillatorIndicator(df['high'], df['low']).awesome_oscillator()

        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = ta.momentum.ROCIndicator(df['close'], window=period).roc()

        return df

    def _add_advanced_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced trend indicators."""
        # Vortex Indicator
        vortex = ta.trend.VortexIndicator(df['high'], df['low'], df['close'])
        df['vortex_pos'] = vortex.vortex_indicator_pos()
        df['vortex_neg'] = vortex.vortex_indicator_neg()

        # Aroon Indicator (using our safe implementation)
        aroon = SafeAroonIndicator(df['high'], df['low'])
        df['aroon_up'] = aroon.aroon_up()
        df['aroon_down'] = aroon.aroon_down()
        df['aroon_indicator'] = aroon.aroon_indicator()

        # KST Oscillator
        kst = ta.trend.KSTIndicator(df['close'])
        df['kst'] = kst.kst()
        df['kst_signal'] = kst.kst_sig()

        # Detrended Price Oscillator
        df['dpo'] = ta.trend.DPOIndicator(df['close']).dpo()

        return df

    def _add_volume_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced volume indicators."""
        if 'volume' not in df.columns:
            logger.warning("Volume column not found, skipping volume features")
            return df

        # Volume Price Trend
        df['volume_price_trend'] = ta.volume.VolumePriceTrendIndicator(df['close'], df['volume']).volume_price_trend()

        # Ease of Movement
        df['ease_of_movement'] = ta.volume.EaseOfMovementIndicator(df['high'], df['low'], df['volume']).ease_of_movement()

        # Force Index
        df['force_index'] = ta.volume.ForceIndexIndicator(df['close'], df['volume']).force_index()

        # Negative Volume Index
        df['nvi'] = ta.volume.NegativeVolumeIndexIndicator(df['close'], df['volume']).negative_volume_index()

        # Accumulation/Distribution
        df['adi'] = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()

        return df

    def _add_cycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cycle indicators."""
        # Schaff Trend Cycle
        # Note: This is a simplified implementation
        cycle_period = 10
        df['cycle_indicator'] = ta.trend.EMAIndicator(df['close'], window=cycle_period).ema_indicator()

        # Hilbert Transform (simplified)
        df['ht_trendline'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()

        return df

    def _add_volatility_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced volatility indicators."""
        # Ulcer Index
        df['ulcer_index'] = np.sqrt((df['close'] - df['close'].rolling(14, min_periods=1).max())**2 / 14)

        # Donchian Channels
        df['donchian_upper'] = df['high'].rolling(20, min_periods=1).max()
        df['donchian_lower'] = df['low'].rolling(20, min_periods=1).min()
        df['donchian_middle'] = (df['donchian_upper'] + df['donchian_lower']) / 2

        # Keltner Channels
        ema_20 = df['close'].ewm(span=20).mean()
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        df['keltner_upper'] = ema_20 + (2 * atr)
        df['keltner_lower'] = ema_20 - (2 * atr)
        df['keltner_middle'] = ema_20

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        # Z-score
        rolling_mean = df['close'].rolling(20, min_periods=1).mean()
        rolling_std = df['close'].rolling(20, min_periods=1).std().replace(0, np.nan)
        df['zscore_20'] = (df['close'] - rolling_mean) / rolling_std

        # Percentile ranks
        df['percentile_20'] = df['close'].rolling(20, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        df['percentile_50'] = df['close'].rolling(50, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

        # Rolling skewness and kurtosis
        df['rolling_skew_20'] = df['returns_1'].rolling(20, min_periods=1).skew().clip(-10, 10)
        df['rolling_kurtosis_20'] = df['returns_1'].rolling(20, min_periods=1).kurt().clip(-10, 10)

        # Entropy (simplified)
        def calculate_entropy(x):
            x = x.dropna()
            if len(x) < 2 or x.nunique() <= 1:
                return 0.0
            try:
                hist, _ = np.histogram(x, bins=10, density=True)
                hist = hist[hist > 0]
                if len(hist) == 0:
                    return 0.0
                return -sum(p * np.log(p) for p in hist)
            except (ValueError, RuntimeWarning):
                return 0.0
        
        df['returns_entropy'] = df['returns_1'].rolling(20, min_periods=1).apply(calculate_entropy)

        return df

    def _select_features(self, df: pd.DataFrame, n_features: int, target_column: str) -> pd.DataFrame:
        """Select top features using multiple methods."""
        logger.info(f"Selecting top {n_features} features")

        # Get feature columns
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'symbol', target_column] and not col.startswith(('year', 'month'))]

        if len(feature_cols) <= n_features:
            logger.info("Number of features already <= requested, skipping selection")
            return df

        X = df[feature_cols]
        y = df[target_column]

        # Remove NaN values for feature selection
        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]

        if len(X_clean) == 0:
            logger.warning("No valid data for feature selection, keeping all features")
            return df

        # Method 1: Mutual Information
        mi_scores = mutual_info_regression(X_clean, y_clean)
        mi_features = pd.Series(mi_scores, index=feature_cols).nlargest(n_features).index.tolist()

        # Method 2: F-regression
        f_scores, _ = f_regression(X_clean, y_clean)
        f_features = pd.Series(f_scores, index=feature_cols).nlargest(n_features).index.tolist()

        # Method 3: Random Forest importance
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_clean, y_clean)
        rf_features = pd.Series(rf.feature_importances_, index=feature_cols).nlargest(n_features).index.tolist()

        # Combine rankings
        all_selected = list(set(mi_features + f_features + rf_features))
        feature_counts = pd.Series(all_selected).value_counts()
        final_features = feature_counts.nlargest(n_features).index.tolist()

        logger.info(f"Selected features: {final_features}")

        # Keep only selected features
        cols_to_keep = ['timestamp', 'symbol', target_column] + final_features
        df_selected = df[cols_to_keep].copy()

        return df_selected

    def _apply_pca(self, df: pd.DataFrame, n_components: int) -> pd.DataFrame:
        """Apply PCA for dimensionality reduction."""
        logger.info(f"Applying PCA with {n_components} components")

        # Get feature columns
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'symbol'] and not col.startswith(('year', 'month'))]

        if len(feature_cols) < n_components:
            logger.warning("Number of features < n_components, skipping PCA")
            return df

        # Scale features
        X = self.scaler.fit_transform(df[feature_cols])

        # Apply PCA
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)

        # Create new dataframe with PCA components
        pca_cols = [f'pca_{i}' for i in range(n_components)]
        df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)

        # Combine with original dataframe
        df_result = pd.concat([df[['timestamp', 'symbol']], df_pca], axis=1)

        logger.info(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_}")

        return df_result

    def get_feature_importance_analysis(self, df: pd.DataFrame, target_column: str) -> Dict:
        """Analyze feature importance using multiple methods."""
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'symbol', target_column]]

        X = df[feature_cols]
        y = df[target_column]

        # Remove NaN values
        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]

        analysis = {}

        # Random Forest importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_clean, y_clean)
        analysis['rf_importance'] = dict(zip(feature_cols, rf.feature_importances_))

        # Mutual Information
        mi_scores = mutual_info_regression(X_clean, y_clean)
        analysis['mutual_info'] = dict(zip(feature_cols, mi_scores))

        # F-regression
        f_scores, _ = f_regression(X_clean, y_clean)
        analysis['f_regression'] = dict(zip(feature_cols, f_scores))

        return analysis

    def validate_features(self, df: pd.DataFrame) -> Dict[str, Union[int, float]]:
        """Validate generated features for quality issues."""
        validation_results = {
            'total_features': len(df.columns),
            'total_samples': len(df),
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'infinite_values': np.isinf(df.select_dtypes(include=[np.number])).sum().sum(),
            'constant_features': 0,
            'high_correlation_features': 0
        }

        # Check for constant features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].nunique() <= 1:
                validation_results['constant_features'] += 1

        # Check for highly correlated features
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            high_corr = (corr_matrix.abs() > 0.95).sum().sum() - len(numeric_cols)
            validation_results['high_correlation_features'] = high_corr // 2

        return validation_results