"""
Tests for enhanced training features including data synthesis and optimization.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from datetime import datetime, timedelta

from dashboard.utils.enhanced_data_loader import EnhancedDataLoader
from automated_training_pipeline import AutomatedTrainingPipeline

class TestEnhancedTraining(unittest.TestCase):
    """Test cases for enhanced training features."""

    def setUp(self):
        """Set up test environment."""
        self.data_loader = EnhancedDataLoader(num_workers=2)
        self.pipeline = AutomatedTrainingPipeline()
        
        # Create sample data
        dates = pd.date_range(start='2025-01-01', end='2025-01-10', freq='1H')
        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.normal(1.1, 0.02, len(dates)),
            'high': np.random.normal(1.15, 0.02, len(dates)),
            'low': np.random.normal(1.05, 0.02, len(dates)),
            'close': np.random.normal(1.1, 0.02, len(dates)),
            'volume': np.random.lognormal(10, 1, len(dates))
        })
        self.sample_data.set_index('timestamp', inplace=True)

    def test_data_synthesis(self):
        """Test data synthesis capabilities."""
        # Test synthetic data generation
        synthetic_data = self.data_loader.generate_synthetic_data(
            self.sample_data,
            num_samples=100,
            include_edge_cases=True
        )
        
        self.assertIsInstance(synthetic_data, pd.DataFrame)
        self.assertEqual(len(synthetic_data), 100)
        self.assertTrue(all(col in synthetic_data.columns 
                          for col in ['open', 'high', 'low', 'close', 'volume']))
        
        # Test edge case generation
        edge_case_ratio = self.data_loader.synthesis_config['edge_case_ratio']
        expected_edge_cases = int(100 * edge_case_ratio)
        
        # Verify trend reversals
        price_changes = synthetic_data['close'].pct_change()
        trend_reversals = ((price_changes > 0.01) & 
                         (price_changes.shift(-1) < -0.01)).sum()
        self.assertGreaterEqual(trend_reversals, expected_edge_cases * 0.2)

    def test_parallel_processing(self):
        """Test parallel processing optimization."""
        # Test parallel data loading
        start_time = datetime.now()
        processed_data = self.pipeline.process_data(augment_data=False)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        self.assertIsNotNone(processed_data)
        self.assertGreater(len(processed_data), 0)
        self.assertLess(processing_time, 60)  # Should process within 60 seconds

    def test_training_optimization(self):
        """Test training optimization features."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Train model with GPU acceleration
        results = self.pipeline.train_models(
            self.sample_data,
            use_gpu=True
        )
        
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # Check metrics tracking
        self.assertIn('training_time', self.pipeline.metrics)
        self.assertIn('gpu_memory_usage', self.pipeline.metrics)
        self.assertGreater(self.pipeline.metrics['training_throughput'], 0)

    def test_feature_engineering(self):
        """Test enhanced feature engineering."""
        processed_data = self.pipeline.preprocess_data(self.sample_data)
        
        # Check technical indicators
        expected_features = [
            'rsi', 'macd', 'volatility', 'bollinger_upper',
            'bollinger_lower', 'momentum', 'volume_ratio'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, processed_data.columns)
        
        # Check feature selection
        selected_features = self.pipeline._select_features(processed_data)
        self.assertGreater(len(selected_features), 5)
        self.assertLess(len(selected_features), len(processed_data.columns))

    def test_monitoring_integration(self):
        """Test monitoring system integration."""
        # Process some data
        _ = self.pipeline.process_data(augment_data=True)
        
        # Check metrics
        metrics = self.pipeline.metrics
        self.assertIn('synthetic_data_ratio', metrics)
        self.assertIn('edge_cases_generated', metrics)
        self.assertIn('cache_hit_rate', metrics)
        self.assertGreaterEqual(metrics['synthetic_data_ratio'], 0)
        self.assertGreaterEqual(metrics['edge_cases_generated'], 0)

    def test_model_convergence(self):
        """Test model convergence tracking."""
        # Train models
        results = self.pipeline.train_models(self.sample_data)
        
        # Check convergence metrics
        self.assertIn('model_convergence_rate', self.pipeline.metrics)
        convergence_rates = self.pipeline.metrics['model_convergence_rate']
        
        for model in convergence_rates:
            self.assertGreater(convergence_rates[model], 0)
            self.assertLess(convergence_rates[model], 1)

if __name__ == '__main__':
    unittest.main()