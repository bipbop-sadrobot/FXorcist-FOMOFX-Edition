#!/usr/bin/env python3
"""
Test script to isolate the min_periods error in feature engineering.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from forex_ai_dashboard.pipeline.enhanced_feature_engineering import EnhancedFeatureEngineer

def test_feature_engineering():
    """Test feature engineering with a small dataset."""
    print("Testing feature engineering...")

    # Create a small test dataset
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='H')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.random.randn(100) * 2,
        'high': 102 + np.random.randn(100) * 2,
        'low': 98 + np.random.randn(100) * 2,
        'close': 100 + np.random.randn(100) * 2,
        'volume': np.random.randint(1000, 10000, 100)
    })

    print(f"Test dataset shape: {df.shape}")
    print(f"Test dataset columns: {df.columns.tolist()}")

    # Initialize feature engineer
    engineer = EnhancedFeatureEngineer()

    try:
        # Test with basic features only
        result = engineer.process_data(
            df,
            feature_groups=['basic'],
            n_features=None
        )
        print(f"Basic features test passed: {result.shape}")
        return True
    except Exception as e:
        print(f"Error in basic features test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_feature_engineering()
    if success:
        print("✅ Feature engineering test passed!")
    else:
        print("❌ Feature engineering test failed!")
        sys.exit(1)