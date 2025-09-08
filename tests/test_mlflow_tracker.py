import pytest
import os
import tempfile
import mlflow
from fxorcist.tracking.mlflow_tracker import MLflowTracker, MLflowTrackerConfig

@pytest.fixture
def mlflow_tracker():
    """
    Fixture to create an MLflow tracker with a temporary tracking URI.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tracking_uri = f"file://{tmpdir}"
        config = MLflowTrackerConfig(
            tracking_uri=tracking_uri,
            experiment_name="test_fxorcist_backtests",
            artifact_dir=tmpdir
        )
        tracker = MLflowTracker(config=config)
        yield tracker

def test_mlflow_tracker_initialization(mlflow_tracker):
    """
    Test MLflow tracker initialization.
    """
    assert mlflow_tracker.config is not None
    assert mlflow_tracker.config.experiment_name == "test_fxorcist_backtests"

def test_mlflow_tracker_log_trial(mlflow_tracker):
    """
    Test logging a trial to MLflow.
    """
    trial_params = {
        "window": 14,
        "threshold": 70
    }
    
    metrics = {
        "sharpe": 1.5,
        "return": 0.2,
        "max_drawdown": -0.1
    }
    
    config = {
        "commission": 0.001,
        "symbol": "EURUSD"
    }
    
    equity_curve = [(0, 10000), (1, 10500), (2, 11000)]
    
    mlflow_tracker.log_trial(
        trial_id="test_trial_1",
        params=trial_params,
        metrics=metrics,
        config=config,
        equity_curve=equity_curve,
        tags={"strategy": "RSI"}
    )
    
    # Verify artifact was created
    artifact_files = os.listdir(mlflow_tracker.config.artifact_dir)
    assert any("equity_curve" in f for f in artifact_files)

def test_mlflow_tracker_error_handling(mlflow_tracker):
    """
    Test MLflow tracker error handling.
    """
    with pytest.raises(TypeError):
        mlflow_tracker.log_trial(
            trial_id=None,  # Invalid trial_id
            params={},
            metrics={},
            config={}
        )

def test_mlflow_tracker_complex_config():
    """
    Test MLflow tracker with complex configuration.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MLflowTrackerConfig(
            tracking_uri=f"file://{tmpdir}",
            experiment_name="complex_test",
            artifact_dir=tmpdir
        )
        
        tracker = MLflowTracker(config=config)
        
        assert tracker.config.experiment_name == "complex_test"
        assert tracker.config.artifact_dir == tmpdir

def test_mlflow_tracker_git_commit_tracking(mlflow_tracker):
    """
    Test that git commit is logged with each trial.
    """
    trial_params = {"window": 14}
    metrics = {"sharpe": 1.5}
    config = {"commission": 0.001}
    
    mlflow_tracker.log_trial(
        trial_id="git_commit_test",
        params=trial_params,
        metrics=metrics,
        config=config
    )
    
    # Retrieve the latest run
    client = mlflow.tracking.MlflowClient(mlflow_tracker.config.tracking_uri)
    runs = client.search_runs(
        experiment_names=["test_fxorcist_backtests"]
    )
    
    assert len(runs) > 0
    assert "git_commit" in runs[0].data.params