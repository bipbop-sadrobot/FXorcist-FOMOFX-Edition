import pytest
import time
from prometheus_client import REGISTRY
from fxorcist.observability.metrics import PrometheusMetrics, PrometheusMetricsConfig

@pytest.fixture
def prometheus_metrics():
    """
    Fixture to create a PrometheusMetrics instance.
    """
    config = PrometheusMetricsConfig(
        port=9999,  # Use a non-standard port for testing
        namespace='test_fxorcist',
        subsystem='test_backtest'
    )
    metrics = PrometheusMetrics(config=config)
    yield metrics

def test_prometheus_metrics_initialization(prometheus_metrics):
    """
    Test PrometheusMetrics initialization.
    """
    assert prometheus_metrics.config is not None
    assert prometheus_metrics.config.namespace == 'test_fxorcist'
    assert prometheus_metrics.config.subsystem == 'test_backtest'

def test_prometheus_metrics_observe_backtest(prometheus_metrics):
    """
    Test observing a successful backtest.
    """
    strategy = "test_strategy"
    duration = 0.5
    
    # Capture initial metric values
    initial_count = REGISTRY.get_sample_value(
        'test_fxorcist_test_backtest_backtest_total', 
        {'strategy': strategy}
    ) or 0
    
    prometheus_metrics.observe_backtest(strategy, duration, success=True)
    
    # Verify count increased
    count = REGISTRY.get_sample_value(
        'test_fxorcist_test_backtest_backtest_total', 
        {'strategy': strategy}
    )
    assert count == initial_count + 1

def test_prometheus_metrics_observe_failed_backtest(prometheus_metrics):
    """
    Test observing a failed backtest.
    """
    strategy = "failed_strategy"
    duration = 0.3
    error_type = "timeout"
    
    # Capture initial metric values
    initial_error_count = REGISTRY.get_sample_value(
        'test_fxorcist_test_backtest_backtest_errors_total', 
        {'strategy': strategy, 'error_type': error_type}
    ) or 0
    
    prometheus_metrics.observe_backtest(
        strategy, 
        duration, 
        success=False, 
        error_type=error_type
    )
    
    # Verify error count increased
    error_count = REGISTRY.get_sample_value(
        'test_fxorcist_test_backtest_backtest_errors_total', 
        {'strategy': strategy, 'error_type': error_type}
    )
    assert error_count == initial_error_count + 1

def test_prometheus_metrics_track_active_backtests(prometheus_metrics):
    """
    Test tracking active backtests.
    """
    strategy = "active_strategy"
    
    # Capture initial active backtest value
    initial_active = REGISTRY.get_sample_value(
        'test_fxorcist_test_backtest_active_backtests', 
        {'strategy': strategy}
    ) or 0
    
    # Increment active backtests
    prometheus_metrics.track_active_backtests(strategy, active=True)
    
    active_count = REGISTRY.get_sample_value(
        'test_fxorcist_test_backtest_active_backtests', 
        {'strategy': strategy}
    )
    assert active_count == initial_active + 1
    
    # Decrement active backtests
    prometheus_metrics.track_active_backtests(strategy, active=False)
    
    active_count = REGISTRY.get_sample_value(
        'test_fxorcist_test_backtest_active_backtests', 
        {'strategy': strategy}
    )
    assert active_count == initial_active

def test_prometheus_metrics_duration_histogram(prometheus_metrics):
    """
    Test backtest duration histogram.
    """
    strategy = "duration_strategy"
    durations = [0.1, 0.5, 1.0, 2.0]
    
    for duration in durations:
        prometheus_metrics.observe_backtest(strategy, duration, success=True)
    
    # Verify histogram buckets
    buckets = [
        REGISTRY.get_sample_value(
            'test_fxorcist_test_backtest_backtest_duration_seconds_bucket', 
            {'strategy': strategy, 'le': str(bucket)}
        ) for bucket in [0.1, 0.5, 1.0, 2.0, float('inf')]
    ]
    
    assert all(bucket is not None for bucket in buckets)