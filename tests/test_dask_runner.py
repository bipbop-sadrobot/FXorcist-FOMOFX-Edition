import pytest
import numpy as np
from fxorcist.config import Settings
from fxorcist.distributed.dask_runner import DaskRunner

@pytest.fixture
def config():
    """Create a mock configuration for testing."""
    return Settings()

def test_dask_runner_initialization():
    """Test Dask runner initialization."""
    runner = DaskRunner(n_workers=2, threads_per_worker=1)
    runner.start()
    
    assert runner.cluster is not None
    assert runner.client is not None
    
    runner.close()

def test_dask_runner_trial_execution(config):
    """Test running multiple backtest trials in parallel."""
    runner = DaskRunner(n_workers=2, threads_per_worker=1)
    
    # Create sample trial configurations
    trial_configs = [
        {"rsi_window": 14, "rsi_overbought": 70, "rsi_oversold": 30},
        {"rsi_window": 21, "rsi_overbought": 75, "rsi_oversold": 25}
    ]
    
    results = runner.run_trials(
        strategy_name="rsi", 
        trial_configs=trial_configs, 
        config=config
    )
    
    runner.close()
    
    # Validate results
    assert len(results) == len(trial_configs)
    for result in results:
        assert "metrics" in result or "error" in result

def test_dask_runner_error_handling(config):
    """Test error handling in distributed trials."""
    runner = DaskRunner(n_workers=2, threads_per_worker=1)
    
    # Create a trial config that will likely cause an error
    trial_configs = [
        {"invalid_param": "test"}  # This should trigger an error
    ]
    
    results = runner.run_trials(
        strategy_name="rsi", 
        trial_configs=trial_configs, 
        config=config
    )
    
    runner.close()
    
    # Validate error handling
    assert len(results) == 1
    assert "error" in results[0]

def test_dask_runner_seed_reproducibility(config):
    """Test that seeds are consistently generated."""
    runner = DaskRunner(n_workers=2, threads_per_worker=1)
    
    trial_configs = [
        {"rsi_window": 14, "rsi_overbought": 70, "rsi_oversold": 30},
        {"rsi_window": 21, "rsi_overbought": 75, "rsi_oversold": 25}
    ]
    
    results1 = runner.run_trials(
        strategy_name="rsi", 
        trial_configs=trial_configs, 
        config=config
    )
    
    results2 = runner.run_trials(
        strategy_name="rsi", 
        trial_configs=trial_configs, 
        config=config
    )
    
    runner.close()
    
    # Compare seeds to ensure reproducibility
    seeds1 = [result.get("seed", None) for result in results1]
    seeds2 = [result.get("seed", None) for result in results2]
    
    assert seeds1 == seeds2