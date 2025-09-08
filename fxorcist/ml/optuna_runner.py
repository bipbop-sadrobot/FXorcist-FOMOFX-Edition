import optuna
import numpy as np
from typing import Dict, Any
import logging

from fxorcist.config import Settings
from fxorcist.backtest.engine import run_backtest
from fxorcist.distributed.dask_runner import DaskRunner
from fxorcist.tracking.mlflow_tracker import MLflowTracker

logger = logging.getLogger(__name__)

def objective(trial: optuna.Trial, config: Settings) -> float:
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.Trial): Optuna trial object
        config (Settings): Global configuration

    Returns:
        float: Negative Sharpe ratio (for minimization)
    """
    params = {
        "rsi_window": trial.suggest_int("rsi_window", 5, 30),
        "rsi_overbought": trial.suggest_int("rsi_overbought", 60, 80),
        "rsi_oversold": trial.suggest_int("rsi_oversold", 20, 40),
        "commission_pct": trial.suggest_float("commission_pct", 0.00001, 0.0001),
    }

    # Set seed for reproducibility
    seed = trial.number
    np.random.seed(seed)

    # Run backtest
    result = run_backtest(
        strategy_name="rsi", 
        config=config, 
        params_file=None, 
        params=params
    )

    # Log to MLflow
    tracker = MLflowTracker()
    tracker.log_trial(
        trial_id=str(trial.number),
        params=params,
        metrics=result.get("metrics", {}),
        config=config.model_dump(),
        returns=result.get("returns", None)
    )

    return -result["metrics"].get("sharpe", 0)

def run_optuna(
    config: Settings, 
    n_trials: int = 100, 
    distributed: bool = False,
    n_workers: int = 4
) -> optuna.Study:
    """
    Run Optuna hyperparameter optimization.

    Args:
        config (Settings): Global configuration
        n_trials (int): Number of trials to run
        distributed (bool): Whether to use distributed computing
        n_workers (int): Number of workers for distributed optimization

    Returns:
        optuna.Study: Optimization study results
    """
    if distributed:
        # Distributed optimization using Dask
        runner = DaskRunner(n_workers=n_workers)

        # Generate trial configs
        trial_configs = []
        for i in range(n_trials):
            params = {
                "rsi_window": np.random.randint(5, 30),
                "rsi_overbought": np.random.randint(60, 80),
                "rsi_oversold": np.random.randint(20, 40),
                "commission_pct": np.random.uniform(0.00001, 0.0001)
            }
            trial_configs.append(params)

        # Run distributed trials
        results = runner.run_trials(
            strategy_name="rsi", 
            trial_configs=trial_configs, 
            config=config
        )
        runner.close()

        # Create study from results
        study = optuna.create_study(direction="minimize")
        for result in results:
            if "error" not in result:
                study.add_trial(optuna.trial.create_trial(
                    params=result["params"],
                    value=-result["metrics"].get("sharpe", 0.0),
                    user_attrs={"seed": result.get("seed", 0)}
                ))
        
        return study

    else:
        # Standard Optuna optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, config), n_trials=n_trials)
        return study