"""
Parameter optimization engine for trading strategies using Optuna.
Supports grid search, random search, and Bayesian optimization.
"""

import optuna
import pandas as pd
from typing import Dict, Any, Callable, List, Union, Optional
from dataclasses import dataclass
import logging
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from .enhanced_backtest import backtest_strategy, BacktestConfig, TradeStats

@dataclass
class OptimizationConfig:
    """Configuration for parameter optimization."""
    param_ranges: Dict[str, Union[List, tuple]]  # Parameter ranges to explore
    n_trials: int = 50  # Number of optimization trials
    validation_window: int = 252  # Days for out-of-sample validation
    optimization_metric: str = "sharpe_ratio"  # Metric to optimize
    direction: str = "maximize"  # Optimization direction
    min_trades: int = 20  # Minimum trades required for valid evaluation

class ParameterOptimizer:
    """
    Optimizes strategy parameters using Optuna.
    
    Features:
    - Multiple optimization metrics
    - Parallel execution
    - Early pruning of poor trials
    - Hyperparameter importance analysis
    """
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize optimizer with configuration.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.study = None
        self.best_params = None
        self.param_importance = None
        
    def _create_study(self) -> optuna.Study:
        """Create Optuna study with focused settings."""
        sampler = optuna.samplers.TPESampler(seed=42)  # For reproducibility
        return optuna.create_study(
            direction=self.config.direction,
            sampler=sampler
        )
    
    def _evaluate_params(self, params: Dict[str, Any], data: pd.DataFrame,
                        strategy_generator: Callable,
                        backtest_config: BacktestConfig) -> float:
        """
        Objective function for optimization.
        
        Args:
            trial: Optuna trial
            data: Price data
            strategy_generator: Function that creates strategy with given parameters
            backtest_config: Backtest configuration
            
        Returns:
            Optimization metric value
        """
        try:
            # Create strategy with parameters
            strategy = strategy_generator(**params)
            
            # Run backtest with validation window
            trades_df, stats = backtest_strategy(
                data, strategy, backtest_config,
                validation_window=self.config.validation_window
            )
            
            # Check minimum trade requirement
            if len(trades_df) < self.config.min_trades:
                return float('-inf') if self.config.direction == "maximize" else float('inf')
            
            # Get optimization metric
            metric_value = getattr(stats, self.config.optimization_metric)
            
            # Penalize extreme drawdowns
            if stats.max_drawdown > 0.3:  # 30% max drawdown limit
                return float('-inf') if self.config.direction == "maximize" else float('inf')
            
            return metric_value
            
        except Exception as e:
            self.logger.warning(f"Parameter evaluation failed: {e}")
            return float('-inf') if self.config.direction == "maximize" else float('inf')
    
    def _objective(self, trial: optuna.Trial, data: pd.DataFrame,
                  strategy_generator: Callable,
                  backtest_config: BacktestConfig) -> float:
        """Generate parameters and evaluate them."""
        # Generate parameters for this trial
        params = {}
        for name, range_def in self.config.param_ranges.items():
            if isinstance(range_def, (list, tuple)):
                if isinstance(range_def[0], int):
                    params[name] = trial.suggest_int(name, range_def[0], range_def[1])
                else:
                    params[name] = trial.suggest_float(name, range_def[0], range_def[1])
            else:
                params[name] = trial.suggest_categorical(name, range_def)
        
        return self._evaluate_params(params, data, strategy_generator, backtest_config)

    def optimize(self, data: pd.DataFrame, strategy_generator: Callable,
                backtest_config: BacktestConfig,
                progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run parameter optimization.
        
        Args:
            data: Price data
            strategy_generator: Function that creates strategy with given parameters
            backtest_config: Backtest configuration
            
        Returns:
            Dictionary containing optimization results
        """
        self.study = self._create_study()
        
        # Create objective function with fixed arguments
        objective = lambda trial: self._objective(
            trial, data, strategy_generator, backtest_config
        )
        
        # Run optimization
        self.logger.info(f"Starting optimization with {self.config.n_trials} trials")
        
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            n_jobs=1,  # Sequential for reproducibility
            show_progress_bar=True
        )
        
        # Store results
        self.best_params = self.study.best_params
        
        # Calculate parameter importance
        try:
            self.param_importance = optuna.importance.get_param_importances(self.study)
        except Exception as e:
            self.logger.warning(f"Could not calculate parameter importance: {e}")
            self.param_importance = {}
        
        # Return results
        return {
            'best_params': self.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials),
            'param_importance': self.param_importance,
            'optimization_history': [
                {
                    'trial': t.number,
                    'value': t.value,
                    'params': t.params
                }
                for t in self.study.trials
            ]
        }
    
    def plot_optimization_history(self) -> Optional[Dict[str, Any]]:
        """
        Generate optimization history visualization.
        
        Returns:
            Dictionary containing plot data and layout
        """
        if not self.study:
            return None
        
        # Extract history data
        history = [
            {
                'trial': t.number,
                'value': t.value if t.value is not None else float('nan'),
                'datetime': t.datetime
            }
            for t in self.study.trials
        ]
        
        # Create plot data
        plot_data = {
            'data': [
                {
                    'x': [h['trial'] for h in history],
                    'y': [h['value'] for h in history],
                    'mode': 'markers+lines',
                    'name': 'Trial Value'
                },
                {
                    'x': [h['trial'] for h in history],
                    'y': [self.study.best_value] * len(history),
                    'mode': 'lines',
                    'name': 'Best Value',
                    'line': {'dash': 'dash'}
                }
            ],
            'layout': {
                'title': 'Optimization History',
                'xaxis': {'title': 'Trial Number'},
                'yaxis': {'title': self.config.optimization_metric}
            }
        }
        
        return plot_data
    
    def plot_param_importance(self) -> Optional[Dict[str, Any]]:
        """
        Generate parameter importance visualization.
        
        Returns:
            Dictionary containing plot data and layout
        """
        if not self.param_importance:
            return None
        
        # Sort parameters by importance
        sorted_params = sorted(
            self.param_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create plot data
        plot_data = {
            'data': [{
                'x': [p[0] for p in sorted_params],
                'y': [p[1] for p in sorted_params],
                'type': 'bar'
            }],
            'layout': {
                'title': 'Parameter Importance',
                'xaxis': {'title': 'Parameter'},
                'yaxis': {'title': 'Importance Score'}
            }
        }
        
        return plot_data