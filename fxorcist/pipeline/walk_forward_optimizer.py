"""
Implements walk-forward optimization to prevent over-fitting.

Problems Solved:
- Over-fitting: Parameters optimized on entire dataset
- Data snooping: Multiple testing without adjustment
- Parameter instability: Parameters don't work out-of-sample
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
import json
from pathlib import Path
import warnings

@dataclass
class OptimizationConfig:
    """Configuration for walk-forward optimization."""
    optimization_window: int = 252  # Trading days for optimization
    validation_window: int = 63    # Trading days for validation
    min_trades: int = 30          # Minimum trades for valid optimization
    parameter_stability_threshold: float = 0.3  # Max allowed parameter variation
    performance_metric: str = 'sharpe_ratio'  # Metric to optimize
    significance_level: float = 0.05  # For statistical tests

class WalkForwardOptimizer:
    """Implements walk-forward optimization to prevent over-fitting."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger(__name__)
        self.parameter_history = []
        self.optimization_results = []
        
    async def walk_forward_optimization(
        self,
        data: pd.DataFrame,
        strategy_class,
        param_ranges: Dict[str, List[Any]],
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Perform walk-forward optimization.
        
        Process:
        1. Use optimization window to find best parameters
        2. Trade forward for validation_window periods
        3. Re-optimize and repeat
        """
        results = []
        data_points = len(data)
        
        # Initialize progress tracking
        total_steps = (data_points - self.config.optimization_window) // self.config.validation_window
        current_step = 0
        
        for start_idx in range(
            self.config.optimization_window,
            data_points,
            self.config.validation_window
        ):
            # Update progress
            if progress_callback:
                progress = current_step / total_steps
                await progress_callback(progress)
            
            # Optimization period
            opt_start = start_idx - self.config.optimization_window
            opt_end = start_idx
            opt_data = data.iloc[opt_start:opt_end]
            
            # Find optimal parameters
            best_params = await self._optimize_parameters(
                opt_data, strategy_class, param_ranges
            )
            
            # Out-of-sample testing period
            test_end = min(start_idx + self.config.validation_window, data_points)
            test_data = data.iloc[start_idx:test_end]
            
            # Run backtest with optimal parameters
            test_results = await self._run_backtest_with_params(
                test_data, strategy_class, best_params
            )
            
            # Record results
            period_results = {
                'optimization_period': (opt_start, opt_end),
                'test_period': (start_idx, test_end),
                'parameters': best_params,
                'in_sample_performance': test_results['in_sample_metrics'],
                'out_of_sample_performance': test_results['out_of_sample_metrics'],
                'parameter_stability': self._calculate_parameter_stability(best_params)
            }
            
            results.append(period_results)
            self.parameter_history.append(best_params)
            
            current_step += 1
        
        # Final progress update
        if progress_callback:
            await progress_callback(1.0)
        
        # Analyze results
        analysis = self._analyze_optimization_results(results)
        
        return {
            'results': results,
            'analysis': analysis,
            'parameter_stability': self._analyze_parameter_stability(),
            'statistical_significance': self._calculate_statistical_significance(results)
        }
    
    async def _optimize_parameters(
        self,
        data: pd.DataFrame,
        strategy_class,
        param_ranges: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Optimize strategy parameters using grid search or other methods."""
        best_params = None
        best_score = float('-inf')
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_ranges)
        
        # Evaluate each parameter set
        for params in param_combinations:
            try:
                # Run mini backtest with these parameters
                score = await self._evaluate_parameter_set(
                    data, strategy_class, params
                )
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                self.logger.warning(f"Parameter evaluation failed: {e}")
                continue
        
        return best_params
    
    def _generate_param_combinations(self, param_ranges: Dict) -> List[Dict]:
        """Generate all parameter combinations for grid search."""
        import itertools
        
        keys = param_ranges.keys()
        values = param_ranges.values()
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    async def _evaluate_parameter_set(
        self,
        data: pd.DataFrame,
        strategy_class,
        params: Dict[str, Any]
    ) -> float:
        """
        Evaluate a parameter set using k-fold cross validation.
        Returns the chosen performance metric.
        """
        # Split data into k folds
        n_folds = 5
        fold_size = len(data) // n_folds
        scores = []
        
        for i in range(n_folds):
            # Create train/test split
            test_start = i * fold_size
            test_end = (i + 1) * fold_size
            
            train_data = pd.concat([
                data.iloc[:test_start],
                data.iloc[test_end:]
            ])
            test_data = data.iloc[test_start:test_end]
            
            # Run backtest on this fold
            results = await self._run_backtest_with_params(
                test_data, strategy_class, params
            )
            
            # Get score for this fold
            fold_score = results['out_of_sample_metrics'].get(
                self.config.performance_metric, 0.0
            )
            scores.append(fold_score)
        
        # Return mean score across folds
        return np.mean(scores)
    
    def _calculate_parameter_stability(self, current_params: Dict) -> float:
        """
        Calculate parameter stability score.
        Returns 1.0 for perfectly stable parameters, 0.0 for highly unstable.
        """
        if not self.parameter_history:
            return 1.0
        
        stability_scores = []
        
        # Compare with last n parameter sets
        n_lookback = min(5, len(self.parameter_history))
        recent_params = self.parameter_history[-n_lookback:]
        
        for param_name in current_params:
            param_values = [p.get(param_name) for p in recent_params]
            param_values.append(current_params[param_name])
            
            # Calculate coefficient of variation
            param_std = np.std(param_values)
            param_mean = np.mean(param_values)
            
            if param_mean != 0:
                stability = 1.0 - min(1.0, abs(param_std / param_mean))
            else:
                stability = 0.0
            
            stability_scores.append(stability)
        
        return np.mean(stability_scores)
    
    def _analyze_parameter_stability(self) -> Dict:
        """Analyze parameter stability across all optimization periods."""
        if not self.parameter_history:
            return {}
        
        analysis = {}
        param_names = self.parameter_history[0].keys()
        
        for param in param_names:
            values = [p[param] for p in self.parameter_history]
            
            analysis[param] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf,
                'trend': self._calculate_parameter_trend(values)
            }
        
        return analysis
    
    def _calculate_parameter_trend(self, values: List[float]) -> str:
        """Calculate trend in parameter values (increasing, decreasing, stable)."""
        if len(values) < 3:
            return "insufficient_data"
            
        from scipy import stats
        
        # Perform linear regression
        x = np.arange(len(values))
        slope, _, r_value, p_value, _ = stats.linregress(x, values)
        
        # Check if trend is significant
        if p_value > self.config.significance_level:
            return "stable"
        
        # Determine trend direction
        if slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def _calculate_statistical_significance(self, results: List[Dict]) -> Dict:
        """Calculate statistical significance of optimization results."""
        from scipy import stats
        
        # Extract performance metrics
        in_sample_perf = [r['in_sample_performance'][self.config.performance_metric]
                         for r in results]
        out_sample_perf = [r['out_of_sample_performance'][self.config.performance_metric]
                          for r in results]
        
        # Paired t-test between in-sample and out-of-sample performance
        t_stat, p_value = stats.ttest_rel(in_sample_perf, out_sample_perf)
        
        # Calculate effect size (Cohen's d)
        effect_size = (np.mean(out_sample_perf) - np.mean(in_sample_perf)) / \
                     np.std(in_sample_perf)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < self.config.significance_level,
            'performance_degradation': np.mean(out_sample_perf) < np.mean(in_sample_perf)
        }
    
    def _analyze_optimization_results(self, results: List[Dict]) -> Dict:
        """Analyze optimization results across all periods."""
        analysis = {
            'performance_consistency': self._analyze_performance_consistency(results),
            'parameter_convergence': self._analyze_parameter_convergence(results),
            'robustness_metrics': self._calculate_robustness_metrics(results)
        }
        
        # Add warnings for potential issues
        warnings = []
        
        if analysis['performance_consistency']['decay_ratio'] > 0.3:
            warnings.append("High performance decay from in-sample to out-of-sample")
        
        if not analysis['parameter_convergence']['converged']:
            warnings.append("Parameters failed to converge to stable values")
        
        if analysis['robustness_metrics']['failure_rate'] > 0.2:
            warnings.append("High failure rate in out-of-sample periods")
        
        analysis['warnings'] = warnings
        
        return analysis
    
    def _analyze_performance_consistency(self, results: List[Dict]) -> Dict:
        """Analyze consistency between in-sample and out-of-sample performance."""
        in_sample = [r['in_sample_performance'][self.config.performance_metric]
                    for r in results]
        out_sample = [r['out_of_sample_performance'][self.config.performance_metric]
                     for r in results]
        
        return {
            'in_sample_mean': np.mean(in_sample),
            'out_sample_mean': np.mean(out_sample),
            'decay_ratio': (np.mean(in_sample) - np.mean(out_sample)) / np.mean(in_sample)
                         if np.mean(in_sample) != 0 else np.inf,
            'correlation': np.corrcoef(in_sample, out_sample)[0, 1]
        }
    
    def _analyze_parameter_convergence(self, results: List[Dict]) -> Dict:
        """Analyze whether parameters converge to stable values."""
        if not results:
            return {'converged': False}
        
        param_names = results[0]['parameters'].keys()
        convergence_analysis = {}
        
        for param in param_names:
            values = [r['parameters'][param] for r in results]
            
            # Calculate rolling standard deviation
            rolling_std = pd.Series(values).rolling(window=5).std()
            
            # Check if standard deviation decreases
            if len(rolling_std) > 5:
                initial_std = rolling_std.iloc[5]
                final_std = rolling_std.iloc[-1]
                converged = final_std < initial_std * 0.5
            else:
                converged = False
            
            convergence_analysis[param] = {
                'converged': converged,
                'final_std': rolling_std.iloc[-1] if len(rolling_std) > 0 else None
            }
        
        # Overall convergence requires all parameters to converge
        return {
            'converged': all(v['converged'] for v in convergence_analysis.values()),
            'per_parameter': convergence_analysis
        }
    
    def _calculate_robustness_metrics(self, results: List[Dict]) -> Dict:
        """Calculate metrics for strategy robustness."""
        metric = self.config.performance_metric
        out_sample_perf = [r['out_of_sample_performance'][metric] for r in results]
        
        return {
            'failure_rate': sum(1 for x in out_sample_perf if x < 0) / len(out_sample_perf),
            'worst_performance': min(out_sample_perf),
            'performance_std': np.std(out_sample_perf),
            'consecutive_failures': self._max_consecutive_failures(out_sample_perf)
        }
    
    def _max_consecutive_failures(self, performance: List[float]) -> int:
        """Calculate maximum consecutive negative performance periods."""
        max_failures = current_failures = 0
        
        for perf in performance:
            if perf < 0:
                current_failures += 1
                max_failures = max(max_failures, current_failures)
            else:
                current_failures = 0
        
        return max_failures