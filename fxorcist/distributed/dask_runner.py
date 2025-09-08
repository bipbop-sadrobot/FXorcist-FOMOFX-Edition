from typing import List, Dict, Any
import dask
from dask.distributed import Client, LocalCluster
import logging
import json

logger = logging.getLogger(__name__)

class DaskRunner:
    def __init__(self, n_workers: int = 4, threads_per_worker: int = 1):
        """Initialize Dask LocalCluster."""
        self.cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            processes=True  # Isolate memory
        )
        self.client = Client(self.cluster)
        logger.info(f"Dask cluster started: {self.client.dashboard_link}")

    def run_trials(self, trial_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run backtests in parallel."""
        # Submit all trials
        futures = []
        for config in trial_configs:
            future = self.client.submit(
                _run_backtest_task,
                config["strategy"],
                config["params"],
                config["config_dict"]  # serializable dict
            )
            futures.append(future)

        # Gather results
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Trial failed: {e}")
                results.append({"error": str(e)})

        return results

    def close(self):
        """Close Dask client and cluster."""
        self.client.close()
        self.cluster.close()

def _run_backtest_task(strategy_name: str, params: Dict, config_dict: Dict) -> Dict:
    """Isolated task — must be serializable."""
    from fxorcist.config import Settings
    from fxorcist.backtest.engine import run_backtest

    # Reconstruct config
    config = Settings.model_validate(config_dict)
    result = run_backtest(strategy_name, config, params_file=None)
    return {
        "strategy": strategy_name,
        "params": params,
        "metrics": result.get("metrics", {}),
        "equity_curve": result.get("equity_curve", [])
    }