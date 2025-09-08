import mlflow
import json
import pandas as pd
import quantstats as qs
from typing import Dict, Any, Optional, List
import logging
import os
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class MLflowTracker:
    """Advanced experiment tracking with MLflow and QuantStats."""

    def __init__(
        self, 
        tracking_uri: Optional[str] = None,
        experiment_name: str = "fxorcist-backtests",
        artifact_dir: Optional[str] = None
    ):
        """
        Initialize MLflow tracking with advanced configuration.

        Args:
            tracking_uri (str, optional): MLflow tracking server URI
            experiment_name (str): Name of the experiment
            artifact_dir (str, optional): Directory to store artifacts
        """
        # Set tracking URI if provided, otherwise use default
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        # Configure artifact directory
        self.artifact_dir = artifact_dir or os.path.join(os.getcwd(), "mlflow_artifacts")
        os.makedirs(self.artifact_dir, exist_ok=True)

        logger.info(f"MLflow tracking initialized. Tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"Artifact directory: {self.artifact_dir}")

    def log_trial(
        self, 
        trial_id: str, 
        params: Dict[str, Any], 
        metrics: Dict[str, float], 
        config: Dict[str, Any], 
        returns: Optional[pd.Series] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Log a complete trial with comprehensive tracking.

        Args:
            trial_id (str): Unique identifier for the trial
            params (Dict): Hyperparameters used in the trial
            metrics (Dict): Performance metrics
            config (Dict): Full configuration used
            returns (pd.Series, optional): Daily returns for detailed analysis
            tags (Dict, optional): Additional tags for the run

        Returns:
            str: MLflow run ID
        """
        with mlflow.start_run(run_name=f"trial_{trial_id}"):
            try:
                # Log parameters
                mlflow.log_params(params)

                # Log metrics
                mlflow.log_metrics(metrics)

                # Log configuration as JSON artifact
                config_path = os.path.join(self.artifact_dir, f"config_{trial_id}.json")
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                mlflow.log_artifact(config_path)

                # Generate and log QuantStats report if returns are provided
                if returns is not None and len(returns) > 1:
                    self._log_quantstats_report(returns, trial_id)

                # Log additional tags if provided
                if tags:
                    for key, value in tags.items():
                        mlflow.set_tag(key, value)

                # Log system metadata
                mlflow.set_tag("timestamp", datetime.now().isoformat())
                mlflow.set_tag("trial_id", trial_id)

                # Get current run ID
                run_id = mlflow.active_run().info.run_id
                logger.info(f"Logged trial {trial_id} to MLflow. Run ID: {run_id}")

                return run_id

            except Exception as e:
                logger.error(f"Failed to log trial {trial_id}: {e}")
                raise

    def _log_quantstats_report(self, returns: pd.Series, trial_id: str):
        """
        Generate and log QuantStats report.

        Args:
            returns (pd.Series): Daily returns series
            trial_id (str): Trial identifier
        """
        try:
            # Ensure returns have a datetime index
            if not isinstance(returns.index, pd.DatetimeIndex):
                returns.index = pd.date_range(
                    start=datetime.now() - pd.Timedelta(days=len(returns)), 
                    periods=len(returns)
                )

            # HTML report
            report_path = os.path.join(self.artifact_dir, f"quantstats_report_{trial_id}.html")
            qs.reports.html(returns, output=report_path, title=f'Trial {trial_id}')
            mlflow.log_artifact(report_path)

            # Key QuantStats metrics
            stats = qs.stats(returns)
            quantstats_metrics = {
                "cagr": stats.get("cagr", 0),
                "max_drawdown": stats.get("max_drawdown", 0),
                "sharpe": stats.get("sharpe", 0),
                "sortino": stats.get("sortino", 0),
                "win_rate": stats.get("win_rate", 0),
                "profit_factor": stats.get("profit_factor", 0)
            }
            mlflow.log_metrics(quantstats_metrics)

        except Exception as e:
            logger.warning(f"Failed to generate QuantStats report: {e}")

    def get_best_run(
        self, 
        metric: str = "sharpe", 
        mode: str = "max"
    ) -> Dict[str, Any]:
        """
        Retrieve the best run based on a specific metric.

        Args:
            metric (str): Metric to optimize
            mode (str): Optimization mode ('max' or 'min')

        Returns:
            Dict: Details of the best run
        """
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("fxorcist-backtests")
        
        runs = client.search_runs([experiment.experiment_id])
        
        if mode == "max":
            best_run = max(runs, key=lambda run: run.data.metrics.get(metric, float('-inf')))
        else:
            best_run = min(runs, key=lambda run: run.data.metrics.get(metric, float('inf')))
        
        return {
            "run_id": best_run.info.run_id,
            "params": best_run.data.params,
            "metrics": best_run.data.metrics
        }

    def compare_runs(
        self, 
        metrics: List[str] = ["sharpe", "max_drawdown", "cagr"]
    ) -> pd.DataFrame:
        """
        Compare multiple runs across specified metrics.

        Args:
            metrics (List[str]): Metrics to compare

        Returns:
            pd.DataFrame: Comparison of runs
        """
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("fxorcist-backtests")
        
        runs = client.search_runs([experiment.experiment_id])
        
        comparison_data = []
        for run in runs:
            run_data = {
                "run_id": run.info.run_id,
                **run.data.params
            }
            for metric in metrics:
                run_data[metric] = run.data.metrics.get(metric, np.nan)
            
            comparison_data.append(run_data)
        
        return pd.DataFrame(comparison_data)