import pytest
import asyncio
from typing import Dict, Any
from fxorcist.distributed.dask_runner import DaskRunner, DaskRunnerConfig

def mock_backtest_function(strategy: str, params: Dict[str, Any], config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mock backtest function for testing Dask runner.
    
    :param strategy: Strategy name
    :param params: Strategy parameters
    :param config_dict: Configuration dictionary
    :return: Simulated backtest result
    """
    # Simulate a backtest with some basic logic
    if strategy == "test_success":
        return {
            "strategy": strategy,
            "params": params,
            "metrics": {"sharpe": 1.5, "return": 0.2},
            "equity_curve": [(0, 10000), (1, 10500)]
        }
    elif strategy == "test_failure":
        raise ValueError("Simulated backtest failure")
    else:
        return {"error": "Unknown strategy"}

@pytest.fixture
def dask_runner():
    """Fixture to create a Dask runner for testing."""
    runner = DaskRunner(
        config=DaskRunnerConfig(
            n_workers=2,
            threads_per_worker=1
        )
    )
    yield runner
    runner.close()

def test_dask_runner_successful_trials(dask_runner):
    """
    Test Dask runner with successful backtest trials.
    """
    trial_configs = [
        {
            "strategy": "test_success",
            "params": {"window": 14},
            "config_dict": {"commission": 0.001}
        },
        {
            "strategy": "test_success",
            "params": {"window": 21},
            "config_dict": {"commission": 0.002}
        }
    ]

    results = dask_runner.run_trials(trial_configs)
    
    assert len(results) == 2
    for result in results:
        assert "metrics" in result
        assert result["metrics"]["sharpe"] > 0
        assert "equity_curve" in result

def test_dask_runner_mixed_trials(dask_runner):
    """
    Test Dask runner with a mix of successful and failing trials.
    """
    trial_configs = [
        {
            "strategy": "test_success",
            "params": {"window": 14},
            "config_dict": {"commission": 0.001}
        },
        {
            "strategy": "test_failure",
            "params": {"window": 21},
            "config_dict": {"commission": 0.002}
        }
    ]

    results = dask_runner.run_trials(trial_configs, max_retries=1)
    
    assert len(results) == 2
    
    # Check successful trial
    assert "metrics" in results[0]
    assert results[0]["metrics"]["sharpe"] > 0
    
    # Check failed trial
    assert "error" in results[1]
    assert "traceback" in results[1]

def test_dask_runner_configuration():
    """
    Test Dask runner configuration options.
    """
    config = DaskRunnerConfig(
        n_workers=4,
        threads_per_worker=2,
        memory_limit="4GB"
    )
    
    runner = DaskRunner(config=config)
    
    assert runner.config.n_workers == 4
    assert runner.config.threads_per_worker == 2
    assert runner.config.memory_limit == "4GB"
    
    runner.close()

def test_dask_runner_error_handling():
    """
    Test Dask runner error handling and retry mechanism.
    """
    runner = DaskRunner(
        config=DaskRunnerConfig(
            n_workers=2,
            threads_per_worker=1
        )
    )
    
    trial_configs = [
        {
            "strategy": "test_failure",
            "params": {"window": 14},
            "config_dict": {"commission": 0.001}
        }
    ]
    
    results = runner.run_trials(trial_configs, max_retries=2)
    
    assert len(results) == 1
    assert "error" in results[0]
    assert "traceback" in results[0]
    
    runner.close()