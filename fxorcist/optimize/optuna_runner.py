"""
Optuna-based hyperparameter optimization for trading strategies.
"""
from typing import Dict, Any, Optional, List, Callable
import optuna
import numpy as np
import random
from rich.progress import Progress

from fxorcist.backtest.engine import run_backtest_from_events
from fxorcist.events.event_bus import InMemoryEventBus
from fxorcist.config import AppConfig

def objective(
    trial: optuna.Trial, 
    strategy_name: str, 
    config: AppConfig, 
    events: List[Any],
    strategy_factory: Optional[Callable] = None
) -> float:
    """
    Objective function for Optuna hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        strategy_name: Name of the strategy to optimize
        config: Application configuration
        events: List of market events
        strategy_factory: Optional factory to create strategy instances
    
    Returns:
        Negative Sharpe ratio (to be minimized by Optuna)
    """
    # Set random seeds for reproducibility
    random.seed(42 + trial.number)
    np.random.seed(42 + trial.number)
    
    # Define hyperparameters to optimize
    # Example: RSI strategy parameters
    if strategy_name == 'rsi':
        rsi_lower = trial.suggest_int('rsi_lower', 20, 40)
        rsi_upper = trial.suggest_int('rsi_upper', 60, 80)
        
        # Ensure upper bound is higher than lower bound
        if rsi_lower >= rsi_upper:
            trial.set_user_attr('invalid_params', True)
            return float('inf')
    
    # Add more strategy-specific parameter suggestions
    elif strategy_name == 'macd':
        fast_period = trial.suggest_int('fast_period', 5, 20)
        slow_period = trial.suggest_int('slow_period', 20, 50)
        signal_period = trial.suggest_int('signal_period', 5, 20)
    
    # Create strategy with trial parameters
    if strategy_factory:
        strategy = strategy_factory(trial)
    else:
        from fxorcist.strategies.registry import get_strategy
        strategy = get_strategy(strategy_name)(trial)
    
    # Create event bus and run backtest
    event_bus = InMemoryEventBus(events)
    
    # Configure execution model
    from fxorcist.backtest.execution import SimpleSlippageModel
    exec_model = SimpleSlippageModel(
        commission_pct=config.backtest.commission_pct,
        slippage_ticks=trial.suggest_float('slippage', 0.0, 0.001)
    )
    
    # Run backtest
    result = run_backtest_from_events(
        event_bus.replay(), 
        [strategy], 
        exec_model, 
        initial_cash=config.backtest.initial_capital
    )
    
    # Extract metrics
    metrics = result.metrics.summary()
    
    # Return negative Sharpe ratio (Optuna minimizes)
    return -metrics.get('sharpe_ratio', 0.0)

def run_optuna(
    strategy_name: str, 
    config: AppConfig, 
    n_trials: int = 100,
    events: Optional[List[Any]] = None,
    strategy_factory: Optional[Callable] = None,
    progress: Optional[Progress] = None
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter optimization.
    
    Args:
        strategy_name: Name of the strategy to optimize
        config: Application configuration
        n_trials: Number of optimization trials
        events: Optional list of market events
        strategy_factory: Optional factory to create strategy instances
        progress: Optional Rich progress bar
    
    Returns:
        Dictionary containing optimization study results
    """
    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )
    
    # Run optimization
    try:
        if progress:
            task = progress.add_task(f"Optimizing {strategy_name}", total=n_trials)
        
        study.optimize(
            lambda trial: objective(
                trial, 
                strategy_name, 
                config, 
                events or [], 
                strategy_factory
            ), 
            n_trials=n_trials,
            callbacks=[
                lambda study, trial: progress.update(task, advance=1) if progress else None
            ]
        )
    except Exception as e:
        print(f"Optimization error: {e}")
        raise
    finally:
        if progress:
            progress.update(task, completed=True)
    
    # Prepare results
    results = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'best_trial': {
            'number': study.best_trial.number,
            'params': study.best_trial.params,
            'value': study.best_trial.value
        },
        'trials': [
            {
                'number': t.number,
                'params': t.params,
                'value': t.value,
                'state': t.state.name
            } for t in study.trials
        ]
    }
    
    return results