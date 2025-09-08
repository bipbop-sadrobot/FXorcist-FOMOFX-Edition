from typing import List, Dict, Any, Optional
from dask.distributed import Client, LocalCluster, progress
import logging
import numpy as np
import asyncio

from fxorcist.config import Settings
from fxorcist.backtest.engine import run_backtest

logger = logging.getLogger(__name__)

class DaskRunner:
    """Run backtests in parallel using Dask."""

    def __init__(
        self, 
        n_workers: int = 4, 
        threads_per_worker: int = 1, 
        memory_limit: str = '2GB'
    ):
        """
        Initialize Dask cluster for distributed backtesting.

        Args:
            n_workers (int): Number of worker processes
            threads_per_worker (int): Number of threads per worker
            memory_limit (str): Memory limit per worker
        """
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
        self.cluster = None
        self.client = None

    def start(self):
        """Start Dask cluster."""
        try:
            self.cluster = LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=self.threads_per_worker,
                processes=True,
                memory_limit=self.memory_limit
            )
            self.client = Client(self.cluster)
            logger.info(f"Dask cluster started. Dashboard: {self.client.dashboard_link}")
        except Exception as e:
            logger.error(f"Failed to start Dask cluster: {e}")
            raise

    def run_trials(
        self, 
        strategy_name: str, 
        trial_configs: List[Dict[str, Any]], 
        config: Settings
    ) -> List[Dict[str, Any]]:
        """
        Run multiple backtest trials in parallel.

        Args:
            strategy_name (str): Name of the trading strategy
            trial_configs (List[Dict]): List of parameter configurations
            config (Settings): Global configuration settings

        Returns:
            List[Dict]: Results of each trial
        """
        if not self.client:
            self.start()

        # Submit all trials
        futures = []
        for trial_config in trial_configs:
            future = self.client.submit(
                _run_backtest_task,
                strategy_name,
                trial_config,
                config.model_dump(),
                pure=False  # Don't cache results
            )
            futures.append(future)

        # Show progress
        progress(futures)

        # Gather results
        results = []
        for i, future in enumerate(futures):
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Trial {i+1}/{len(futures)} completed")
            except Exception as e:
                logger.error(f"Trial {i+1} failed: {e}")
                results.append({
                    "error": str(e), 
                    "trial_id": i, 
                    "params": trial_configs[i]
                })

        return results

    def close(self):
        """Close Dask cluster and client."""
        if self.client:
            self.client.close()
        if self.cluster:
            self.cluster.close()
        logger.info("Dask cluster closed")

def _run_backtest_task(
    strategy_name: str, 
    params: Dict, 
    config_dict: Dict
) -> Dict:
    """
    Isolated task for Dask to run a single backtest trial.

    Args:
        strategy_name (str): Name of trading strategy
        params (Dict): Strategy-specific parameters
        config_dict (Dict): Global configuration

    Returns:
        Dict: Backtest results
    """
    # Set seed for reproducibility
    seed = hash(str(params)) % (2**32)
    np.random.seed(seed)

    try:
        # Restore Settings from dict
        config = Settings.model_validate(config_dict)
        
        # Run backtest
        result = run_backtest(
            strategy_name, 
            config, 
            params_file=None,
            params=params
        )

        return {
            "strategy": strategy_name,
            "params": params,
            "metrics": result.get("metrics", {}),
            "seed": seed
        }
    except Exception as e:
        return {
            "error": str(e), 
            "params": params, 
            "seed": seed
        }