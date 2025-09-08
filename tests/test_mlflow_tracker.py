import pytest
import pandas as pd
import numpy as np
import mlflow
from fxorcist.tracking.mlflow_tracker import MLflowTracker

@pytest.fixture
def mlflow_tracker():
    """Create a MLflow tracker for testing."""
    return MLflowTracker(experiment_name="test_experiment")

def test_log_trial(mlflow_tracker):
    """Test logging a complete trial."""
    # Simulate returns
    returns = pd.Series(np.random.normal(0.001, 0.05, 100))
    returns.index = pd.date_range(start='2024-01-01', periods=len(returns))

    # Log trial
    run_id = mlflow_tracker.log_trial(
        trial_id="test_trial_1",
        params={"window": 14, "threshold": 0.5},
        metrics={"sharpe": 1.2, "max_drawdown": -0.15},
        config={"strategy": "test_strategy"},
        returns=returns
    )

    # Verify run was logged
    assert run_id is not None
    
    # Retrieve run and verify details
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    
    assert run.data.params["window"] == "14"
    assert run.data.params["threshold"] == "0.5"
    assert "sharpe" in run.data.metrics

def test_get_best_run(mlflow_tracker):
    """Test retrieving the best run."""
    # Log multiple trials
    for i in range(5):
        returns = pd.Series(np.random.normal(0.001 * (i+1), 0.05, 100))
        returns.index = pd.date_range(start='2024-01-01', periods=len(returns))
        
        mlflow_tracker.log_trial(
            trial_id=f"trial_{i}",
            params={"window": str(14 + i)},
            metrics={"sharpe": 1.0 + 0.1 * i},
            config={"strategy": "test_strategy"},
            returns=returns
        )

    # Get best run
    best_run = mlflow_tracker.get_best_run(metric="sharpe", mode="max")
    
    assert "run_id" in best_run
    assert "params" in best_run
    assert "metrics" in best_run

def test_compare_runs(mlflow_tracker):
    """Test comparing multiple runs."""
    # Log multiple trials
    for i in range(5):
        returns = pd.Series(np.random.normal(0.001 * (i+1), 0.05, 100))
        returns.index = pd.date_range(start='2024-01-01', periods=len(returns))
        
        mlflow_tracker.log_trial(
            trial_id=f"trial_{i}",
            params={"window": str(14 + i)},
            metrics={
                "sharpe": 1.0 + 0.1 * i, 
                "max_drawdown": -0.1 * i,
                "cagr": 0.05 * (i+1)
            },
            config={"strategy": "test_strategy"},
            returns=returns
        )

    # Compare runs
    comparison_df = mlflow_tracker.compare_runs()
    
    assert not comparison_df.empty
    assert "sharpe" in comparison_df.columns
    assert "max_drawdown" in comparison_df.columns
    assert "cagr" in comparison_df.columns

def test_quantstats_report_generation(mlflow_tracker):
    """Test QuantStats report generation."""
    # Simulate returns
    returns = pd.Series(np.random.normal(0.001, 0.05, 100))
    returns.index = pd.date_range(start='2024-01-01', periods=len(returns))

    # Log trial with returns
    run_id = mlflow_tracker.log_trial(
        trial_id="quantstats_test",
        params={"window": 14},
        metrics={"sharpe": 1.2},
        config={"strategy": "test_strategy"},
        returns=returns
    )

    # Retrieve run and verify QuantStats metrics
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    
    # Check for key QuantStats metrics
    assert "cagr" in run.data.metrics
    assert "max_drawdown" in run.data.metrics
    assert "sharpe" in run.data.metrics