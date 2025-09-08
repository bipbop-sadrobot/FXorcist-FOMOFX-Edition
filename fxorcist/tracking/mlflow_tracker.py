import mlflow
import json
from typing import Dict, Any, Optional, List

class MLflowTracker:
    def __init__(self, tracking_uri: str = "http://localhost:5000", experiment_name: str = "fxorcist-backtests"):
        """
        Initialize MLflow tracking.
        
        :param tracking_uri: URI for MLflow tracking server
        :param experiment_name: Name of the experiment to log trials under
        """
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def log_trial(
        self, 
        trial_id: str, 
        params: Dict[str, Any], 
        metrics: Dict[str, float], 
        config: Dict[str, Any], 
        equity_curve: Optional[List[tuple]] = None
    ):
        """
        Log a single trial to MLflow.
        
        :param trial_id: Unique identifier for the trial
        :param params: Hyperparameters used in the trial
        :param metrics: Performance metrics from the trial
        :param config: Full configuration used
        :param equity_curve: Optional equity curve data
        """
        with mlflow.start_run(run_name=f"trial_{trial_id}"):
            # Log hyperparameters
            mlflow.log_params(params)
            
            # Log configuration parameters
            mlflow.log_params({f"config_{k}": str(v) for k, v in config.items()})
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log equity curve as artifact if provided
            if equity_curve:
                try:
                    with open("equity_curve.json", "w") as f:
                        json.dump(equity_curve, f)
                    mlflow.log_artifact("equity_curve.json")
                except Exception as e:
                    mlflow.log_text(str(e), "equity_curve_error.txt")
            
            # Log source code version
            mlflow.log_param("git_commit", _get_git_commit())

def _get_git_commit() -> str:
    """
    Get current git commit hash.
    
    :return: Git commit hash or 'unknown'
    """
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"