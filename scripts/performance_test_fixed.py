#!/usr/bin/env python3
"""
Performance Testing Script for Forex AI Dashboard Caching Improvements
Tests and validates the performance enhancements implemented.
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forex_ai_dashboard.utils.enhanced_data_loader import EnhancedDataLoader
from forex_ai_dashboard.pipeline.enhanced_feature_engineering import EnhancedFeatureGenerator
from forex_ai_dashboard.utils.caching import get_cache_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/performance_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerformanceTester:
    """Performance testing suite for caching improvements."""

    def __init__(self):
        self.cache_manager = get_cache_manager()
        self.data_loader = EnhancedDataLoader()
        self.feature_generator = EnhancedFeatureGenerator()

        # Test data
        self.test_data = None
        self.results = {}

    def generate_test_data(self, n_rows: int = 10000) -> pd.DataFrame:
        """Generate synthetic forex data for testing."""
        logger.info(f"Generating {n_rows} rows of test data")

        # Generate timestamps
        start_date = pd.Timestamp('2024-01-01')
        timestamps = pd.date_range(start_date, periods=n_rows, freq='1H')

        # Generate OHLC data with realistic patterns
        np.random.seed(42)

        # Base price around 1.1000
        base_price = 1.1000

        # Generate random walk with drift
        returns = np.random.normal(0.0001, 0.01, n_rows)  # Small drift, 1% volatility
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLC from close prices
        high_mult = 1 + np.abs(np.random.normal(0, 0.005, n_rows))
        low_mult = 1 - np.abs(np.random.normal(0, 0.005, n_rows))

        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices * (1 + np.random.normal(0, 0.002, n_rows)),
            'high': prices * high_mult,
            'low': prices * low_mult,
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_rows)
        })

        df.set_index('timestamp', inplace=True)
        self.test_data = df
        return df

    def test_data_loading_performance(self) -> dict:
        """Test data loading performance with and without cache."""
        logger.info("Testing data loading performance")

        results = {
            'cache_enabled': [],
            'cache_disabled': []
        }

        # Test with cache enabled
        logger.info("Testing with cache enabled...")
        start_time = time.time()
        for i in range(5):
            df, issues = self.data_loader.load_forex_data(
                timeframe="1H", use_cache=True
            )
        cache_time = (time.time() - start_time) / 5

        # Clear cache and test without cache
        self.data_loader.clear_cache()
        logger.info("Testing with cache disabled...")
        start_time = time.time()
        for i in range(5):
            df, issues = self.data_loader.load_forex_data(
                timeframe="1H", use_cache=False
            )
        no_cache_time = (time.time() - start_time) / 5

        results['cache_enabled'] = cache_time
        results['cache_disabled'] = no_cache_time
        results['speedup'] = no_cache_time / cache_time if cache_time > 0 else 1

        logger.info(".4f")
        logger.info(".4f")
        logger.info(".2f")

        return results

    def test_feature_engineering_performance(self) -> dict:
        """Test feature engineering performance with caching."""
        logger.info("Testing feature engineering performance")

        if self.test_data is None:
            self.test_data = self.generate_test_data()

        results = {
            'cache_enabled': {},
            'cache_disabled': {},
            'speedup': {}
        }

        features_to_test = ['returns', 'volatility', 'rsi_14', 'bb_20']

        # Test with cache enabled
        logger.info("Testing feature generation with cache enabled...")
        start_time = time.time()
        df_cached = self.feature_generator.generate_features_batch(
            self.test_data.copy(), features_to_test
        )
        cache_time = time.time() - start_time

        # Clear feature cache and test without cache
        self.feature_generator.clear_feature_cache()
        logger.info("Testing feature generation with cache disabled...")
        start_time = time.time()
        df_no_cache = self.feature_generator.generate_features_batch(
            self.test_data.copy(), features_to_test
        )
        no_cache_time = time.time() - start_time

        results['cache_enabled']['total_time'] = cache_time
        results['cache_disabled']['total_time'] = no_cache_time
        results['speedup']['total'] = no_cache_time / cache_time if cache_time > 0 else 1

        logger.info(".4f")
        logger.info(".4f")
        logger.info(".2f")

        return results

    def test_expensive_computations(self) -> dict:
        """Test expensive computations (Fourier, Wavelet, SHAP) with caching."""
        logger.info("Testing expensive computations with caching")

        if self.test_data is None:
            self.test_data = self.generate_test_data(5000)  # Smaller dataset for expensive ops

        results = {}

        # Test Fourier features
        logger.info("Testing Fourier features...")
        start_time = time.time()
        df_fourier_1 = self.feature_generator.compute_fourier_features(
            self.test_data.copy()
        )
        fourier_time_1 = time.time() - start_time

        start_time = time.time()
        df_fourier_2 = self.feature_generator.compute_fourier_features(
            self.test_data.copy()
        )
        fourier_time_2 = time.time() - start_time

        results['fourier'] = {
            'first_run': fourier_time_1,
            'cached_run': fourier_time_2,
            'speedup': fourier_time_1 / fourier_time_2 if fourier_time_2 > 0 else 1
        }

        # Test Wavelet features
        logger.info("Testing Wavelet features...")
        start_time = time.time()
        df_wavelet_1 = self.feature_generator.compute_wavelet_features(
            self.test_data.copy()
        )
        wavelet_time_1 = time.time() - start_time

        start_time = time.time()
        df_wavelet_2 = self.feature_generator.compute_wavelet_features(
            self.test_data.copy()
        )
        wavelet_time_2 = time.time() - start_time

        results['wavelet'] = {
            'first_run': wavelet_time_1,
            'cached_run': wavelet_time_2,
            'speedup': wavelet_time_1 / wavelet_time_2 if wavelet_time_2 > 0 else 1
        }

        logger.info("Fourier speedup: .2f")
        logger.info("Wavelet speedup: .2f")

        return results

    def test_cache_persistence(self) -> dict:
        """Test cache persistence and memory usage."""
        logger.info("Testing cache persistence and memory usage")

        # Get cache stats
        cache_stats = self.cache_manager.get_stats()
        data_loader_stats = self.data_loader.get_performance_stats()
        feature_stats = self.feature_generator.get_performance_stats()

        results = {
            'cache_manager_stats': cache_stats,
            'data_loader_stats': data_loader_stats,
            'feature_generator_stats': feature_stats,
            'memory_usage': {
                'cache_size_mb': len(self.cache_manager._cache) * 0.001,  # Rough estimate
                'feature_cache_items': feature_stats.get('total_features_cached', 0)
            }
        }

        logger.info(f"Cache hit rate: {cache_stats.get('hit_rate', 0):.2%}")
        logger.info(f"Total cached items: {len(self.cache_manager._cache)}")

        return results

    def run_full_test_suite(self) -> dict:
        """Run complete performance test suite."""
        logger.info("Starting full performance test suite")

        self.results = {
            'timestamp': datetime.now().isoformat(),
            'data_loading': self.test_data_loading_performance(),
            'feature_engineering': self.test_feature_engineering_performance(),
            'expensive_computations': self.test_expensive_computations(),
            'cache_persistence': self.test_cache_persistence()
        }

        # Save results
        self.save_results()

        logger.info("Performance test suite completed")
        return self.results

    def save_results(self, filename: str = "performance_test_results.json"):
        """Save test results to file."""
        output_path = Path("logs") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")

    def print_summary(self):
        """Print performance test summary."""
        if not self.results:
            logger.warning("No test results available")
            return

        print("\n" + "="*60)
        print("FOREX AI DASHBOARD PERFORMANCE TEST SUMMARY")
        print("="*60)

        # Data loading summary
        dl = self.results.get('data_loading', {})
        print("\n📊 DATA LOADING PERFORMANCE:")
        print(".4f")
        print(".4f")
        print(".2f")

        # Feature engineering summary
        fe = self.results.get('feature_engineering', {})
        print("\n🔧 FEATURE ENGINEERING PERFORMANCE:")
        print(".4f")
        print(".4f")
        print(".2f")

        # Expensive computations summary
        ec = self.results.get('expensive_computations', {})
        if 'fourier' in ec:
            print("\n🧮 EXPENSIVE COMPUTATIONS:")
            print(".4f")
            print(".4f")
            print(".4f")
            print(".4f")

        # Cache summary
        cp = self.results.get('cache_persistence', {})
        cache_stats = cp.get('cache_manager_stats', {})
        print("\n💾 CACHE PERFORMANCE:")
        print(".1%")
        print(f"Memory Cache Size: {cp.get('memory_usage', {}).get('cache_size_mb', 0):.2f} MB")
        print(f"Feature Cache Items: {cp.get('feature_generator_stats', {}).get('total_features_cached', 0)}")

        print("\n" + "="*60)

def main():
    """Main performance testing function."""
    try:
        tester = PerformanceTester()

        # Run full test suite
        results = tester.run_full_test_suite()

        # Print summary
        tester.print_summary()

        print("\n✅ Performance testing completed successfully!")
        print("📄 Detailed results saved to logs/performance_test_results.json")

    except Exception as e:
        logger.error(f"Performance testing failed: {str(e)}", exc_info=True)
        print(f"\n❌ Performance testing failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()