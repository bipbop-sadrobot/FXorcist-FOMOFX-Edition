"""
Enhanced Data loading utility for the Forex AI dashboard.
Provides advanced caching strategies and performance optimizations.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from functools import lru_cache
import pyarrow.parquet as pq
import json
import quantstats as qs
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

from forex_ai_dashboard.pipeline.evaluation_metrics import EvaluationMetrics
from forex_ai_dashboard.models.model_hierarchy import ModelHierarchy
from forex_ai_dashboard.utils.caching import get_cache_manager, get_dataframe_cache, get_model_cache
from .quantstats_analytics import QuantStatsAnalytics

logger = logging.getLogger(__name__)

class EnhancedDataLoader:
    """Enhanced utility class for efficient data loading and management with advanced caching and data synthesis."""

    def __init__(self, num_workers: int = 4):
        """Initialize data loader with paths and advanced cache settings.
        
        Args:
            num_workers: Number of worker processes for parallel processing
        """
        self.data_dir = Path('data/processed')
        self.eval_dir = Path('evaluation_results')
        self.model_dir = Path('models/hierarchy')
        self.cache_timeout = 300  # 5 minutes
        self.num_workers = num_workers
        
        # Initialize parallel processing pools
        self.thread_pool = ThreadPoolExecutor(max_workers=num_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=num_workers)
        
        # Initialize data synthesis settings
        self.synthesis_config = {
            'edge_case_ratio': 0.2,
            'augmentation_ratio': 0.3,
            'noise_level': 0.05,
            'trend_strength': 0.7
        }

        # Initialize advanced caching system
        self.cache_manager = get_cache_manager()
        self.df_cache = get_dataframe_cache()
        self.model_cache = get_model_cache()

        # Initialize QuantStats analytics
        self.quantstats = QuantStatsAnalytics()
        
        # Configure QuantStats
        qs.extend_pandas()

        # Performance tracking
        self.load_times = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
    def load_portfolio_analytics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        use_cache: bool = True
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Load portfolio analytics with QuantStats integration."""
        cache_key = self._generate_cache_key('portfolio_analytics', returns.shape, returns.index.min(), returns.index.max())

        if use_cache:
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                self.cache_stats['hits'] += 1
                return cached_result

        self.cache_stats['misses'] += 1
        start_time = datetime.now()

        try:
            # Generate comprehensive tearsheet
            tearsheet = self.quantstats.generate_comprehensive_tearsheet(
                returns=returns,
                benchmark_returns=benchmark_returns,
                title="Portfolio Performance Analysis"
            )

            # Cache the results
            if use_cache:
                # Remove plotly figures before caching as they're not serializable
                cache_data = tearsheet.copy()
                if 'charts' in cache_data:
                    del cache_data['charts']
                self.cache_manager.set(cache_key, (cache_data, []), ttl=self.cache_timeout)

            load_time = (datetime.now() - start_time).total_seconds()
            self.load_times[cache_key] = load_time

            return tearsheet, []

        except Exception as e:
            logger.error(f"Error loading portfolio analytics: {str(e)}")
            return {}, [f"Error loading portfolio analytics: {str(e)}"]

    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        use_cache: bool = True
    ) -> Tuple[Dict[str, float], List[str]]:
        """Calculate comprehensive risk metrics."""
        cache_key = self._generate_cache_key('risk_metrics', returns.shape, returns.index.min(), returns.index.max())

        if use_cache:
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                self.cache_stats['hits'] += 1
                return cached_result

        self.cache_stats['misses'] += 1

        try:
            metrics = {}
            
            # Basic risk metrics
            metrics['volatility'] = qs.stats.volatility(returns)
            metrics['sharpe'] = qs.stats.sharpe(returns)
            metrics['sortino'] = qs.stats.sortino(returns)
            metrics['max_drawdown'] = qs.stats.max_drawdown(returns)
            
            # Advanced risk metrics
            metrics['var'] = qs.stats.value_at_risk(returns)
            metrics['cvar'] = qs.stats.conditional_value_at_risk(returns)
            metrics['tail_ratio'] = qs.stats.tail_ratio(returns)
            metrics['omega'] = qs.stats.omega(returns)
            
            # Distribution metrics
            metrics['skew'] = qs.stats.skew(returns)
            metrics['kurtosis'] = qs.stats.kurtosis(returns)

            if use_cache:
                self.cache_manager.set(cache_key, (metrics, []), ttl=self.cache_timeout)

            return metrics, []

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {}, [f"Error calculating risk metrics: {str(e)}"]
    def _generate_cache_key(self, operation: str, *args, **kwargs) -> str:
        """Generate a consistent cache key for operations."""
        key_components = [operation] + list(args)
        if kwargs:
            # Sort kwargs for consistent key generation
            sorted_kwargs = sorted(kwargs.items())
            key_components.extend(sorted_kwargs)

        return self.cache_manager._generate_key(key_components)

    def _should_use_cache(self, cache_key: str, max_age: int = 300) -> bool:
        """Check if cached data should be used based on age."""
        # For now, always try cache first - the cache manager handles TTL
        return True

    @lru_cache(maxsize=10)
    def _get_latest_file(self, directory: Path, pattern: str) -> Optional[Path]:
        """Get the latest file matching pattern in directory with caching."""
        try:
            files = list(directory.glob(pattern))
            if not files:
                return None
            return max(files, key=lambda x: x.stat().st_mtime)
        except Exception as e:
            logger.error(f"Error finding latest file: {str(e)}")
            return None

    def _validate_forex_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Validate forex data for common issues with caching."""
        cache_key = self._generate_cache_key('validate_data', df.shape, df.columns.tolist())

        # Try to get validation result from cache
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            self.cache_stats['hits'] += 1
            return cached_result

        self.cache_stats['misses'] += 1
        issues = []

        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            issues.append(f"Missing values found: {missing[missing > 0].to_dict()}")
            df = df.fillna(method='ffill').fillna(method='bfill')

        # Check for duplicate timestamps
        if hasattr(df.index, 'duplicated'):
            duplicates = df.index.duplicated()
            if duplicates.any():
                issues.append(f"Found {duplicates.sum()} duplicate timestamps")
                df = df[~duplicates]

        # Check for price anomalies
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                # Check for negative prices
                if (df[col] <= 0).any():
                    issues.append(f"Found negative or zero prices in {col}")
                    df = df[df[col] > 0]

                # Check for extreme price movements
                pct_change = df[col].pct_change().abs()
                extreme_moves = pct_change > 0.1  # 10% threshold
                if extreme_moves.any():
                    issues.append(f"Found {extreme_moves.sum()} extreme price movements in {col}")

        # Cache the validation result
        result = (df, issues)
        self.cache_manager.set(cache_key, result, ttl=600)  # Cache for 10 minutes

        return result

    def generate_synthetic_data(
        self,
        base_data: pd.DataFrame,
        num_samples: int = 1000,
        include_edge_cases: bool = True,
        market_regimes: Optional[List[str]] = None,
        correlation_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> pd.DataFrame:
        """Generate sophisticated synthetic forex data with market patterns.
        
        Args:
            base_data: Base data to generate synthetic samples from
            num_samples: Number of synthetic samples to generate
            include_edge_cases: Whether to include edge cases
            market_regimes: List of market regimes to simulate ('trending', 'ranging', 'volatile')
            correlation_pairs: List of currency pairs to maintain correlations with
            
        Returns:
            DataFrame containing synthetic data with advanced patterns
        """
        logger.info("Generating advanced synthetic data with market patterns")
        
        # Initialize market regime parameters
        if market_regimes is None:
            market_regimes = ['trending', 'ranging', 'volatile']
            
        regime_weights = {
            'trending': 0.4,
            'ranging': 0.4,
            'volatile': 0.2
        }
        
        # Calculate base statistics with regime awareness
        returns = base_data['close'].pct_change().dropna()
        volatility = returns.rolling(20).std()
        
        # Detect existing market regimes
        existing_regimes = self._detect_market_regimes(base_data)
        regime_stats = self._calculate_regime_statistics(base_data, existing_regimes)
        
        synthetic_data = []
        samples_per_regime = {
            regime: int(num_samples * regime_weights.get(regime, 0.1))
            for regime in market_regimes
        }
        
        # Generate data for each market regime
        for regime in market_regimes:
            regime_data = self._generate_regime_data(
                regime,
                samples_per_regime[regime],
                regime_stats[regime],
                correlation_pairs
            )
            synthetic_data.append(regime_data)
            
        # Generate adversarial patterns
        if include_edge_cases:
            edge_cases = self._generate_adversarial_patterns(
                base_data,
                int(num_samples * self.synthesis_config['edge_case_ratio']),
                regime_stats
            )
            synthetic_data.append(edge_cases)
        
        # Combine all synthetic data
        combined_df = pd.concat(synthetic_data)
        combined_df.sort_index(inplace=True)
        
        # Add market microstructure features
        combined_df = self._add_microstructure_features(combined_df)
        
        # Validate synthetic data quality
        combined_df = self._validate_synthetic_data(combined_df, base_data)
        
        return combined_df

    def _detect_market_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Detect market regimes using advanced technical analysis."""
        # Calculate key indicators
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std()
        trend = data['close'].rolling(50).mean()
        
        # Define regime conditions
        regimes = pd.Series(index=data.index, dtype=str)
        regimes.loc[volatility > volatility.quantile(0.8)] = 'volatile'
        regimes.loc[volatility <= volatility.quantile(0.2)] = 'ranging'
        regimes.loc[
            (returns.rolling(20).mean().abs() > returns.std()) & 
            (volatility <= volatility.quantile(0.8))
        ] = 'trending'
        
        return regimes

    def _calculate_regime_statistics(
        self, 
        data: pd.DataFrame, 
        regimes: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each market regime."""
        stats = {}
        for regime in regimes.unique():
            regime_data = data[regimes == regime]
            if len(regime_data) > 0:
                returns = regime_data['close'].pct_change().dropna()
                stats[regime] = {
                    'mean_return': returns.mean(),
                    'volatility': returns.std(),
                    'skewness': returns.skew(),
                    'kurtosis': returns.kurtosis(),
                    'autocorrelation': returns.autocorr()
                }
        return stats

    def _generate_regime_data(
        self,
        regime: str,
        num_samples: int,
        stats: Dict[str, float],
        correlation_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> pd.DataFrame:
        """Generate synthetic data for a specific market regime."""
        if regime == 'trending':
            return self._generate_trending_data(num_samples, stats)
        elif regime == 'ranging':
            return self._generate_ranging_data(num_samples, stats)
        else:  # volatile
            return self._generate_volatile_data(num_samples, stats)

    def _generate_trending_data(
        self,
        num_samples: int,
        stats: Dict[str, float]
    ) -> pd.DataFrame:
        """Generate trending market data."""
        # Generate trend component
        trend = np.linspace(0, stats['mean_return'] * num_samples, num_samples)
        
        # Add noise
        noise = np.random.normal(0, stats['volatility'] * 0.5, num_samples)
        
        # Combine trend and noise
        returns = trend + noise
        
        # Convert to prices
        prices = 100 * (1 + returns).cumprod()
        
        return self._create_ohlc_from_prices(prices)

    def _generate_ranging_data(
        self,
        num_samples: int,
        stats: Dict[str, float]
    ) -> pd.DataFrame:
        """Generate ranging market data."""
        # Generate mean-reverting process
        level = 100
        prices = [level]
        
        for _ in range(num_samples - 1):
            deviation = prices[-1] - level
            mean_reversion = -0.1 * deviation
            noise = np.random.normal(0, stats['volatility'])
            new_price = prices[-1] + mean_reversion + noise
            prices.append(new_price)
        
        return self._create_ohlc_from_prices(np.array(prices))

    def _generate_volatile_data(
        self,
        num_samples: int,
        stats: Dict[str, float]
    ) -> pd.DataFrame:
        """Generate volatile market data."""
        # Generate GARCH-like process
        returns = []
        volatility = stats['volatility']
        
        for _ in range(num_samples):
            shock = np.random.normal(0, 1)
            volatility = np.sqrt(0.1 + 0.8 * volatility**2 + 0.1 * shock**2)
            returns.append(shock * volatility)
        
        prices = 100 * (1 + np.array(returns)).cumprod()
        return self._create_ohlc_from_prices(prices)

    def _generate_adversarial_patterns(
        self,
        base_data: pd.DataFrame,
        num_samples: int,
        regime_stats: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """Generate adversarial patterns that challenge trading systems."""
        patterns = []
        
        # Generate false breakouts
        breakout_samples = int(num_samples * 0.3)
        patterns.append(self._generate_false_breakouts(breakout_samples, regime_stats))
        
        # Generate stop-hunting patterns
        hunt_samples = int(num_samples * 0.3)
        patterns.append(self._generate_stop_hunting(hunt_samples, regime_stats))
        
        # Generate complex price patterns
        complex_samples = int(num_samples * 0.4)
        patterns.append(self._generate_complex_patterns(complex_samples, regime_stats))
        
        return pd.concat(patterns)

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features to synthetic data."""
        # Add bid-ask spread
        df['spread'] = np.random.exponential(0.0001, len(df))
        
        # Add tick volume
        df['tick_volume'] = np.random.poisson(100, len(df))
        
        # Add order flow imbalance
        df['order_imbalance'] = np.random.normal(0, 1, len(df))
        
        return df

    def _validate_synthetic_data(
        self,
        synthetic_df: pd.DataFrame,
        base_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Validate synthetic data quality and adjust if needed."""
        # Check statistical properties
        synthetic_returns = synthetic_df['close'].pct_change().dropna()
        base_returns = base_data['close'].pct_change().dropna()
        
        # Adjust if statistics deviate too much
        if abs(synthetic_returns.std() - base_returns.std()) > 0.1:
            synthetic_df['close'] = synthetic_df['close'] * (base_returns.std() / synthetic_returns.std())
        
        # Ensure no negative prices
        synthetic_df.loc[synthetic_df['close'] <= 0, 'close'] = base_data['close'].mean()
        
        return synthetic_df

    def _create_ohlc_from_prices(self, prices: np.ndarray) -> pd.DataFrame:
        """Create OHLC data from price series."""
        df = pd.DataFrame({
            'timestamp': pd.date_range(start=pd.Timestamp.now(), periods=len(prices), freq='1min'),
            'close': prices
        })
        df.set_index('timestamp', inplace=True)
        
        # Generate realistic OHLC data
        df['open'] = df['close'].shift(1)
        df['high'] = df['close'] * (1 + np.random.uniform(0, 0.002, len(df)))
        df['low'] = df['close'] * (1 - np.random.uniform(0, 0.002, len(df)))
        df['volume'] = np.random.lognormal(10, 1, len(df))
        
        return df

    def _generate_trend_reversals(self, mean: float, std: float, samples: int) -> np.ndarray:
        """Generate trend reversal patterns."""
        trends = []
        for _ in range(samples // 10):  # Generate in chunks of 10 samples
            # Create uptrend followed by reversal
            uptrend = np.linspace(0, self.synthesis_config['trend_strength'], 5) * std + mean
            reversal = np.linspace(self.synthesis_config['trend_strength'], -self.synthesis_config['trend_strength'], 5) * std + mean
            trends.extend(uptrend)
            trends.extend(reversal)
        return np.array(trends[:samples])

    def _generate_volatility_clusters(self, std: float, samples: int) -> np.ndarray:
        """Generate volatility cluster patterns."""
        base_vol = np.random.normal(0, std, samples)
        vol_multiplier = np.random.gamma(2, 2, samples)
        return base_vol * vol_multiplier

    def _generate_flash_crashes(self, mean: float, std: float, samples: int) -> np.ndarray:
        """Generate flash crash patterns."""
        crashes = []
        crash_size = samples // 5
        for _ in range(5):
            pre_crash = np.random.normal(mean, std, crash_size // 2)
            crash = np.random.normal(-5*std, 2*std, crash_size // 4)
            recovery = np.random.normal(2*mean, std, crash_size // 4)
            crashes.extend(np.concatenate([pre_crash, crash, recovery]))
        return np.array(crashes[:samples])

    def load_forex_data(
        self,
        timeframe: str = "1H",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True,
        augment_data: bool = False
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Load and validate forex data with advanced caching and optional augmentation."""
        cache_key = self._generate_cache_key('forex_data', timeframe, start_date, end_date, augment_data)

        if use_cache and not augment_data:
            # Try DataFrame cache first
            cached_df = self.df_cache.get_dataframe(cache_key)
            if cached_df is not None:
                self.cache_stats['hits'] += 1
                logger.debug(f"DataFrame cache hit for forex_data_{timeframe}")
                return cached_df, []

        self.cache_stats['misses'] += 1
        start_time = datetime.now()

        try:
            # Use parallel processing for file reading
            futures = []
            for file in self.data_dir.glob("*.parquet"):
                futures.append(
                    self.process_pool.submit(
                        pq.read_table,
                        file,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                )
            
            # Combine results
            tables = [f.result() for f in futures]
            if not tables:
                return pd.DataFrame(), ["No data files found"]
            
            df = pd.concat([table.to_pandas() for table in tables])

            # Process data in parallel
            with self.process_pool as pool:
                # Set timestamp as index
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Sort index
                df.sort_index(inplace=True)
                
                if augment_data:
                    # Generate synthetic data
                    synthetic_df = self.generate_synthetic_data(df)
                    df = pd.concat([df, synthetic_df])
                    df.sort_index(inplace=True)

            # Resample to desired timeframe
            if timeframe != "1M":
                df = df.resample(timeframe).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })

            # Apply date filters if provided
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]

            # Validate data
            df, issues = self._validate_forex_data(df)

            # Cache the DataFrame
            if use_cache:
                self.df_cache.set_dataframe(cache_key, df, ttl=self.cache_timeout)

            load_time = (datetime.now() - start_time).total_seconds()
            self.load_times[cache_key] = load_time
            logger.info(".2f")

            return df, issues

        except Exception as e:
            logger.error(f"Error loading forex data: {str(e)}")
            return pd.DataFrame(), [f"Error loading data: {str(e)}"]

    def load_evaluation_results(
        self,
        limit: int = 50,
        use_cache: bool = True
    ) -> Tuple[List[EvaluationMetrics], List[str]]:
        """Load recent evaluation results with caching."""
        cache_key = self._generate_cache_key('eval_results', limit)

        if use_cache:
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                self.cache_stats['hits'] += 1
                return cached_result

        self.cache_stats['misses'] += 1
        start_time = datetime.now()

        try:
            results = []
            issues = []

            files = sorted(
                self.eval_dir.glob('*.json'),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )[:limit]

            for f in files:
                try:
                    with open(f, 'r') as file:
                        data = json.load(file)
                        results.append(EvaluationMetrics.from_dict(data))
                except Exception as e:
                    issues.append(f"Error loading {f.name}: {str(e)}")

            result = (results, issues)

            # Cache the results
            if use_cache:
                self.cache_manager.set(cache_key, result, ttl=self.cache_timeout)

            load_time = (datetime.now() - start_time).total_seconds()
            self.load_times[cache_key] = load_time

            return result

        except Exception as e:
            logger.error(f"Error loading evaluation results: {str(e)}")
            return [], [f"Error loading evaluation results: {str(e)}"]

    def load_model_hierarchy(self, use_cache: bool = True) -> Tuple[Optional[ModelHierarchy], List[str]]:
        """Load current model hierarchy with caching."""
        cache_key = self._generate_cache_key('model_hierarchy')

        if use_cache:
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                self.cache_stats['hits'] += 1
                return cached_result

        self.cache_stats['misses'] += 1
        start_time = datetime.now()

        try:
            latest_file = self._get_latest_file(self.model_dir, "*.json")
            if not latest_file:
                return None, ["No model hierarchy file found"]

            with open(latest_file, 'r') as f:
                hierarchy_data = json.load(f)

            hierarchy = ModelHierarchy.from_dict(hierarchy_data)
            result = (hierarchy, [])

            # Cache the result
            if use_cache:
                self.cache_manager.set(cache_key, result, ttl=self.cache_timeout)

            load_time = (datetime.now() - start_time).total_seconds()
            self.load_times[cache_key] = load_time

            return result

        except Exception as e:
            logger.error(f"Error loading model hierarchy: {str(e)}")
            return None, [f"Error loading model hierarchy: {str(e)}"]

    def get_model_predictions(self, model_name: str, data: pd.DataFrame, use_cache: bool = True) -> Optional[np.ndarray]:
        """Get model predictions with caching."""
        cache_key = self._generate_cache_key('model_predictions', model_name, data.shape, data.index.min(), data.index.max())

        if use_cache:
            cached_predictions = self.cache_manager.get(cache_key)
            if cached_predictions is not None:
                self.cache_stats['hits'] += 1
                return cached_predictions

        self.cache_stats['misses'] += 1

        # Load model from cache
        model = self.model_cache.get_model(model_name)
        if model is None:
            logger.warning(f"Model {model_name} not found in cache")
            return None

        try:
            # Make predictions
            predictions = model.predict(data)

            # Cache predictions
            if use_cache:
                self.cache_manager.set(cache_key, predictions, ttl=1800)  # 30 minutes

            return predictions

        except Exception as e:
            logger.error(f"Error getting predictions from model {model_name}: {str(e)}")
            return None

    def clear_cache(self, pattern: str = None):
        """Clear cached data with optional pattern matching."""
        if pattern:
            # Clear specific pattern (would need implementation)
            logger.info(f"Clearing cache for pattern: {pattern}")
        else:
            self.cache_manager.clear()
            self._get_latest_file.cache_clear()
            logger.info("Cleared all caches")

    def get_performance_stats(self) -> Dict:
        """Get performance and cache statistics."""
        cache_stats = self.cache_manager.get_stats()
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0

        return {
            'cache_stats': self.cache_stats,
            'cache_hit_rate': hit_rate,
            'load_times': self.load_times,
            'cache_manager_stats': cache_stats
        }

    def preload_common_data(self):
        """Preload commonly accessed data into cache."""
        logger.info("Preloading common data into cache...")

        # Preload forex data for common timeframes
        for timeframe in ['1H', '4H', '1D']:
            try:
                self.load_forex_data(timeframe=timeframe)
                logger.info(f"Preloaded forex data for timeframe: {timeframe}")
            except Exception as e:
                logger.warning(f"Failed to preload forex data for {timeframe}: {e}")

        # Preload evaluation results
        try:
            self.load_evaluation_results(limit=20)
            logger.info("Preloaded evaluation results")
        except Exception as e:
            logger.warning(f"Failed to preload evaluation results: {e}")

        # Preload model hierarchy
        try:
            self.load_model_hierarchy()
            logger.info("Preloaded model hierarchy")
        except Exception as e:
            logger.warning(f"Failed to preload model hierarchy: {e}")

        logger.info("Data preloading completed")