import mlflow
import json
import os
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import pandas as pd
import quantstats as qs

logger = logging.getLogger(__name__)

class MLflowTrackerConfig:
    """Enhanced configuration for MLflow tracking."""
    def __init__(
        self, 
        tracking_uri: str = "http://localhost:5000", 
        experiment_name: str = "fxorcist-backtests",
        artifact_dir: Optional[str] = None,
        generate_quantstats: bool = True
    ):
        """
        Initialize MLflow tracker configuration.
        
        :param tracking_uri: URI for MLflow tracking server
        :param experiment_name: Name of the experiment
        :param artifact_dir: Directory to store artifacts
        :param generate_quantstats: Auto-generate QuantStats reports
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.artifact_dir = artifact_dir or os.path.join(os.getcwd(), "mlflow_artifacts")
        self.generate_quantstats = generate_quantstats
        
        os.makedirs(self.artifact_dir, exist_ok=True)

class MLflowTracker:
    """Advanced MLflow experiment tracking with QuantStats integration."""
    
    def __init__(
        self, 
        config: Optional[MLflowTrackerConfig] = None,
        logging_level: int = logging.INFO
    ):
        """
        Initialize MLflow tracking.
        
        :param config: MLflow tracker configuration
        :param logging_level: Logging level for the tracker
        """
        logger.setLevel(logging_level)
        
        # Use default config if not provided
        self.config = config or MLflowTrackerConfig()
        
        try:
            mlflow.set_tracking_uri(self.config.tracking_uri)
            mlflow.set_experiment(self.config.experiment_name)
            logger.info(f"MLflow tracking initialized: {self.config.tracking_uri}")
        except Exception as e:
            logger.error(f"Failed to initialize MLflow tracking: {e}")
            raise

    def log_trial(
        self, 
        trial_id: Union[str, int], 
        params: Dict[str, Any], 
        metrics: Dict[str, float], 
        config: Dict[str, Any], 
        returns_series: Optional[pd.Series] = None,
        equity_curve: Optional[List[tuple]] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Log a comprehensive trial to MLflow with optional QuantStats report.
        
        :param trial_id: Unique identifier for the trial
        :param params: Hyperparameters used in the trial
        :param metrics: Performance metrics from the trial
        :param config: Full configuration used
        :param returns_series: Optional returns series for QuantStats
        :param equity_curve: Optional equity curve data
        :param tags: Optional tags for the run
        """
        try:
            with mlflow.start_run(run_name=f"trial_{trial_id}"):
                # Log hyperparameters
                mlflow.log_params(params)
                
                # Log configuration parameters
                mlflow.log_params({f"config_{k}": _sanitize_value(v) for k, v in config.items()})
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Add tags if provided
                if tags:
                    for key, value in tags.items():
                        mlflow.set_tag(key, value)
                
                # Log equity curve as artifact if provided
                if equity_curve:
                    try:
                        artifact_path = os.path.join(
                            self.config.artifact_dir, 
                            f"equity_curve_{trial_id}_{datetime.now().isoformat()}.json"
                        )
                        with open(artifact_path, "w") as f:
                            json.dump(equity_curve, f, indent=2)
                        mlflow.log_artifact(artifact_path)
                    except Exception as artifact_error:
                        logger.warning(f"Failed to log equity curve artifact: {artifact_error}")
                
                # Generate QuantStats report if returns series is provided
                if self.config.generate_quantstats and returns_series is not None:
                    try:
                        # Generate QuantStats HTML report
                        report_path = os.path.join(
                            self.config.artifact_dir, 
                            f"quantstats_report_{trial_id}_{datetime.now().isoformat()}.html"
                        )
                        qs.reports.html(returns_series, output=report_path, title=f'Strategy Tearsheet - Trial {trial_id}')
                        mlflow.log_artifact(report_path)
                    except Exception as qs_error:
                        logger.warning(f"Failed to generate QuantStats report: {qs_error}")
                
                # Log source code version
                mlflow.log_param("git_commit", _get_git_commit())
                
                # Log timestamp
                mlflow.log_param("timestamp", datetime.now().isoformat())

        except Exception as e:
            logger.error(f"Failed to log trial {trial_id}: {e}")
            raise

def _sanitize_value(value: Any) -> str:
    """
    Convert complex values to string for MLflow logging.
    
    :param value: Value to sanitize
    :return: String representation of the value
    """
    try:
        return str(value)
    except Exception:
        return repr(value)

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