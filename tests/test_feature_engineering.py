import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from forex_ai_dashboard.pipeline.feature_engineering import (
    FeatureGenerator,
    FeatureSelector,
    SyntheticFeatureGenerator,
    FeatureMetadata
)

@pytest.fixture
def sample_data():
    """Create sample forex data for testing."""
    dates = pd.date_range(start='2025-01-01', periods=1000, freq='1min')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'close': np.random.normal(1.2, 0.02, 1000).cumsum(),
        'high': np.random.normal(1.21, 0.02, 1000).cumsum(),
        'low': np.random.normal(1.19, 0.02, 1000).cumsum(),
        'volume': np.random.randint(1000, 5000, 1000)
    })
    df.set_index('timestamp', inplace=True)
    return df

@pytest.fixture
def feature_generator():
    """Create a feature generator instance."""
    return FeatureGenerator(random_state=42)

def test_base_feature_generation(sample_data, feature_generator):
    """Test generation of base features."""
    df_features = feature_generator.generate_features(sample_data)
    
    # Check basic features are present
    assert 'returns' in df_features.columns
    assert 'volatility' in df_features.columns
    
    # Verify feature values
    assert not df_features['returns'].isnull().all()
    assert not df_features['volatility'].isnull().all()
    
    # Check feature metadata is registered
    assert 'returns' in feature_generator.registry.features
    assert 'volatility' in feature_generator.registry.features

def test_synthetic_feature_generation(sample_data, feature_generator):
    """Test generation of synthetic features."""
    # First generate base features
    df_features = feature_generator.generate_features(sample_data)
    
    # Generate synthetic features
    df_with_synthetic = feature_generator.generate_synthetic_features(
        df_features,
        target_col='returns',
        methods=['fourier', 'wavelet']
    )
    
    # Check Fourier features
    fourier_cols = [col for col in df_with_synthetic.columns if 'fourier' in col]
    assert len(fourier_cols) > 0
    for col in fourier_cols:
        assert not df_with_synthetic[col].isnull().all()
    
    # Check wavelet features
    wavelet_cols = [col for col in df_with_synthetic.columns if 'wavelet' in col]
    assert len(wavelet_cols) > 0
    for col in wavelet_cols:
        assert not df_with_synthetic[col].isnull().all()

def test_feature_selection(sample_data, feature_generator):
    """Test feature selection methods."""
    # Generate features first
    df_features = feature_generator.generate_features(sample_data)
    df_features = feature_generator.generate_synthetic_features(
        df_features,
        target_col='returns'
    )
    
    feature_cols = [col for col in df_features.columns if col != 'returns']
    
    # Test feature selection
    selected_features, importance_df = feature_generator.select_important_features(
        df_features,
        target_col='returns',
        feature_cols=feature_cols,
        methods=['boruta', 'shap']
    )
    
    # Verify results
    assert len(selected_features) > 0
    assert len(selected_features) <= len(feature_cols)
    assert not importance_df.empty
    assert 'importance' in importance_df.columns
    assert 'method' in importance_df.columns

def test_feature_selector():
    """Test FeatureSelector class directly."""
    selector = FeatureSelector(random_state=42)
    
    # Create synthetic data for testing
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feat_{i}' for i in range(5)])
    y = X['feat_0'] * 2 + X['feat_1'] - np.random.randn(100) * 0.1
    
    # Test Boruta selection
    selected = selector.select_features_boruta(X, y)
    assert isinstance(selected, list)
    assert len(selected) > 0
    
    # Test feature importance analysis
    importance_df = selector.analyze_feature_importance(
        pd.concat([X, y.rename('target')], axis=1),
        'target',
        X.columns.tolist(),
        method='all'
    )
    assert not importance_df.empty
    assert 'importance' in importance_df.columns
    assert 'method' in importance_df.columns

def test_synthetic_feature_generator():
    """Test SyntheticFeatureGenerator class directly."""
    generator = SyntheticFeatureGenerator()
    
    # Create synthetic time series
    t = np.linspace(0, 10, 1000)
    series = pd.Series(np.sin(t) + np.random.randn(1000) * 0.1)
    
    # Test Fourier features
    fourier_features = generator.generate_fourier_features(series)
    assert not fourier_features.empty
    assert 'fourier_amp_0' in fourier_features.columns
    
    # Test wavelet features
    wavelet_features = generator.generate_wavelet_features(series)
    assert not wavelet_features.empty
    assert 'wavelet_mean_0' in wavelet_features.columns

def test_market_regime_detection(sample_data, feature_generator):
    """Test market regime detection functionality."""
    # First generate base features
    df_features = feature_generator.generate_features(sample_data)
    
    # Test HMM regime detection
    regimes_hmm, stats_hmm = feature_generator.detect_market_regimes(
        df_features,
        n_regimes=3,
        method='hmm'
    )
    assert isinstance(regimes_hmm, np.ndarray)
    assert len(regimes_hmm) == len(df_features)
    assert len(np.unique(regimes_hmm)) <= 3
    assert isinstance(stats_hmm, pd.DataFrame)
    assert len(stats_hmm) == 3
    
    # Test k-means regime detection
    regimes_kmeans, stats_kmeans = feature_generator.detect_market_regimes(
        df_features,
        n_regimes=3,
        method='kmeans'
    )
    assert isinstance(regimes_kmeans, np.ndarray)
    assert len(regimes_kmeans) == len(df_features)
    assert len(np.unique(regimes_kmeans)) <= 3
    assert isinstance(stats_kmeans, pd.DataFrame)
    assert len(stats_kmeans) == 3

def test_market_regime_detector():
    """Test MarketRegimeDetector class directly."""
    detector = MarketRegimeDetector(n_regimes=2, random_state=42)
    
    # Create synthetic data
    n_samples = 1000
    np.random.seed(42)
    
    # Simulate two regimes: low and high volatility
    regime1 = np.random.normal(0, 0.1, n_samples // 2)
    regime2 = np.random.normal(0, 0.3, n_samples // 2)
    returns = np.concatenate([regime1, regime2])
    
    df = pd.DataFrame({
        'returns': returns,
        'close': (1 + returns).cumprod(),
        'volume': np.random.randint(1000, 5000, n_samples)
    })
    
    # Test HMM detection
    detector.method = 'hmm'
    regimes_hmm = detector.fit_predict(df)
    assert len(regimes_hmm) == n_samples
    assert len(np.unique(regimes_hmm)) <= 2
    
    # Test k-means detection
    detector.method = 'kmeans'
    regimes_kmeans = detector.fit_predict(df)
    assert len(regimes_kmeans) == n_samples
    assert len(np.unique(regimes_kmeans)) <= 2
    
    # Test regime statistics
    stats = detector.get_regime_stats(df, regimes_kmeans)
    assert isinstance(stats, pd.DataFrame)
    assert len(stats) == 2
    assert 'avg_return' in stats.columns
    assert 'volatility' in stats.columns

def test_error_handling(sample_data, feature_generator):
    """Test error handling in feature generation."""
    # Test with missing required column
    df_invalid = sample_data.drop('close', axis=1)
    df_features = feature_generator.generate_features(df_invalid)
    
    # Should not raise exception but log warning
    assert 'returns' not in df_features.columns
    
    # Test with invalid feature name
    df_features = feature_generator.generate_features(
        sample_data,
        feature_list=['nonexistent_feature']
    )
    assert 'nonexistent_feature' not in df_features.columns
    
    # Test with invalid regime detection method
    with pytest.raises(ValueError):
        feature_generator.detect_market_regimes(
            sample_data,
            method='invalid_method'
        )

if __name__ == '__main__':
    pytest.main([__file__])