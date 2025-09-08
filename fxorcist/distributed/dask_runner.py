from typing import List, Dict, Any, Optional
import dask
from dask.distributed import Client, LocalCluster, Future
import logging
import traceback
from functools import partial
import multiprocessing

logger = logging.getLogger(__name__)

class DaskRunnerConfig:
    """Configuration for Dask distributed runner."""
    def __init__(
        self, 
        n_workers: Optional[int] = None, 
        threads_per_worker: int = 1,
        memory_limit: Optional[str] = None,
        dashboard_address: Optional[str] = None
    ):
        """
        Initialize Dask runner configuration.
        
        :param n_workers: Number of workers. Defaults to CPU count if not specified.
        :param threads_per_worker: Threads per worker
        :param memory_limit: Memory limit per worker (e.g., '4GB')
        :param dashboard_address: Custom dashboard address
        """
        self.n_workers = n_workers or multiprocessing.cpu_count()
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
        self.dashboard_address = dashboard_address

class DaskRunner:
    """Distributed backtest runner using Dask."""
    
    def __init__(
        self, 
        config: Optional[DaskRunnerConfig] = None,
        logging_level: int = logging.INFO
    ):
        """
        Initialize Dask cluster.
        
        :param config: Dask runner configuration
        :param logging_level: Logging level for the runner
        """
        logger.setLevel(logging_level)
        
        # Use default config if not provided
        self.config = config or DaskRunnerConfig()
        
        # Initialize cluster
        try:
            self.cluster = LocalCluster(
                n_workers=self.config.n_workers,
                threads_per_worker=self.config.threads_per_worker,
                processes=True,  # Isolate memory
                memory_limit=self.config.memory_limit,
                dashboard_address=self.config.dashboard_address
            )
            self.client = Client(self.cluster)
            logger.info(f"Dask cluster started: {self.client.dashboard_link}")
        except Exception as e:
            logger.error(f"Failed to initialize Dask cluster: {e}")
            raise

    def run_trials(
        self, 
        trial_configs: List[Dict[str, Any]], 
        max_retries: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Run backtests in parallel with retry mechanism.
        
        :param trial_configs: List of trial configurations
        :param max_retries: Maximum number of retries for failed trials
        :return: List of trial results
        """
        # Submit all trials
        futures = []
        for config in trial_configs:
            future = self.client.submit(
                _run_backtest_with_retry,
                config["strategy"],
                config["params"],
                config["config_dict"],
                max_retries=max_retries
            )
            futures.append(future)

        # Gather results
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Trial failed after all retries: {e}")
                results.append({
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })

        return results

    def close(self):
        """Gracefully close Dask client and cluster."""
        try:
            self.client.close()
            self.cluster.close()
            logger.info("Dask cluster shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down Dask cluster: {e}")

def _run_backtest_with_retry(
    strategy_name: str, 
    params: Dict, 
    config_dict: Dict, 
    max_retries: int = 1
) -> Dict:
    """
    Run a single backtest with retry mechanism.
    
    :param strategy_name: Name of the trading strategy
    :param params: Strategy parameters
    :param config_dict: Full configuration dictionary
    :param max_retries: Maximum number of retries
    :return: Backtest result or error information
    """
    from fxorcist.config import Settings
    from fxorcist.backtest.engine import run_backtest

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            # Reconstruct config
            config = Settings.model_validate(config_dict)
            result = run_backtest(strategy_name, config, params_file=None)
            
            return {
                "strategy": strategy_name,
                "params": params,
                "metrics": result.get("metrics", {}),
                "equity_curve": result.get("equity_curve", []),
                "attempt": attempt
            }
        except Exception as e:
            logger.warning(f"Backtest attempt {attempt} failed: {e}")
            last_exception = e
    
    # If all retries fail
    raise last_exception