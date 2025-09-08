from typing import List, Dict, Any, Optional
import dask
from dask.distributed import Client, LocalCluster, Future, progress
import logging
import traceback
import psutil
import multiprocessing
from functools import partial
import time

logger = logging.getLogger(__name__)

class DaskRunnerConfig:
    """Enhanced configuration for Dask distributed runner."""
    def __init__(
        self, 
        n_workers: Optional[int] = None, 
        threads_per_worker: int = 1,
        memory_limit: Optional[str] = None,
        dashboard_address: Optional[str] = None,
        adaptive_scaling: bool = True,
        log_resource_usage: bool = True
    ):
        """
        Initialize Dask runner configuration with advanced options.
        
        :param n_workers: Number of workers. Defaults to CPU count if not specified.
        :param threads_per_worker: Threads per worker
        :param memory_limit: Memory limit per worker (e.g., '4GB')
        :param dashboard_address: Custom dashboard address
        :param adaptive_scaling: Enable dynamic worker scaling
        :param log_resource_usage: Log CPU/memory usage per trial
        """
        self.n_workers = n_workers or multiprocessing.cpu_count()
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
        self.dashboard_address = dashboard_address
        self.adaptive_scaling = adaptive_scaling
        self.log_resource_usage = log_resource_usage

class DaskRunner:
    """Enhanced distributed backtest runner with advanced features."""
    
    def __init__(
        self, 
        config: Optional[DaskRunnerConfig] = None,
        logging_level: int = logging.INFO
    ):
        """
        Initialize Dask cluster with enhanced configuration.
        
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
                dashboard_address=self.config.dashboard_address,
                # Enable adaptive scaling if configured
                scheduler_port=0,  # Dynamically assign port
                silence_logs=logging.WARNING
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
        Run backtests in parallel with advanced features.
        
        :param trial_configs: List of trial configurations
        :param max_retries: Maximum number of retries for failed trials
        :return: List of trial results with optional resource usage
        """
        # Submit all trials
        futures = []
        for config in trial_configs:
            future = self.client.submit(
                _run_backtest_with_retry_and_logging,
                config["strategy"],
                config["params"],
                config["config_dict"],
                max_retries=max_retries,
                log_resources=self.config.log_resource_usage
            )
            futures.append(future)

        # Show progress bar
        progress(futures)

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

def _run_backtest_with_retry_and_logging(
    strategy_name: str, 
    params: Dict, 
    config_dict: Dict, 
    max_retries: int = 1,
    log_resources: bool = False
) -> Dict:
    """
    Run a single backtest with retry mechanism and optional resource logging.
    
    :param strategy_name: Name of the trading strategy
    :param params: Strategy parameters
    :param config_dict: Full configuration dictionary
    :param max_retries: Maximum number of retries
    :param log_resources: Log CPU and memory usage
    :return: Backtest result or error information
    """
    from fxorcist.config import Settings
    from fxorcist.backtest.engine import run_backtest
    import time

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            # Optional resource tracking
            start_time = time.time()
            resource_usage = {}
            
            if log_resources:
                process = psutil.Process()
                start_cpu = process.cpu_percent()
                start_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Reconstruct config
            config = Settings.model_validate(config_dict)
            result = run_backtest(strategy_name, config, params_file=None)
            
            # Capture resource usage if enabled
            if log_resources:
                end_cpu = process.cpu_percent()
                end_memory = process.memory_info().rss / (1024 * 1024)  # MB
                
                resource_usage = {
                    "cpu_usage_percent": end_cpu - start_cpu,
                    "memory_usage_mb": end_memory - start_memory,
                    "execution_time_seconds": time.time() - start_time
                }
            
            return {
                "strategy": strategy_name,
                "params": params,
                "metrics": result.get("metrics", {}),
                "equity_curve": result.get("equity_curve", []),
                "attempt": attempt,
                "resources": resource_usage
            }
        except Exception as e:
            logger.warning(f"Backtest attempt {attempt} failed: {e}")
            last_exception = e
    
    # If all retries fail
    raise last_exception