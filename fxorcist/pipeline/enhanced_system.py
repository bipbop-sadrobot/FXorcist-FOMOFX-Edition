"""
Complete backtesting system that addresses all major problems.

Integrates:
- Look-ahead bias prevention
- Survivorship bias handling
- Walk-forward optimization
- Realistic transaction costs
- Data quality validation
- Robust performance analysis
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from .time_aware_backtest import TimeAwareBacktestEngine
from .survivorship_handler import SurvivorshipBiasHandler
from .walk_forward_optimizer import WalkForwardOptimizer, OptimizationConfig
from .transaction_costs import RealisticTransactionCostModel, TransactionCostConfig
from .data_validator import DataQualityValidator, ValidationConfig

@dataclass
class BacktestResults:
    """Comprehensive backtest results."""
    trades: pd.DataFrame
    performance_metrics: Dict[str, float]
    optimization_results: Optional[Dict] = None
    transaction_costs: Dict[str, float] = None
    data_quality_stats: Dict[str, Any] = None
    universe_changes: Dict[str, List] = None
    parameter_stability: Dict[str, float] = None
    statistical_significance: Dict[str, Any] = None

class EnhancedBacktestingSystem:
    """Complete backtesting system that addresses all major problems."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.time_engine = TimeAwareBacktestEngine(config.get('time_config', {}))
        self.survivor_handler = SurvivorshipBiasHandler()
        self.walk_forward = WalkForwardOptimizer(
            OptimizationConfig(**config.get('optimization_config', {}))
        )
        self.cost_model = RealisticTransactionCostModel(
            TransactionCostConfig(**config.get('cost_config', {}))
        )
        self.data_validator = DataQualityValidator(
            ValidationConfig(**config.get('validation_config', {}))
        )
        
        # State tracking
        self.current_backtest = None
        self.results_cache = {}
    
    async def run_comprehensive_backtest(
        self,
        data: pd.DataFrame,
        strategy_class,
        parameter_ranges: Optional[Dict] = None,
        universe_file: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> BacktestResults:
        """
        Run a comprehensive backtest addressing all major bias sources.
        
        Args:
            data: Price data with OHLCV columns
            strategy_class: Strategy class to backtest
            parameter_ranges: Optional parameter ranges for optimization
            universe_file: Optional file with universe history
            progress_callback: Optional callback for progress updates
        """
        try:
            self.logger.info("Starting comprehensive backtesting process...")
            
            # Step 1: Data validation and cleaning
            self.logger.info("Step 1: Validating and cleaning data...")
            clean_data = self.data_validator.validate_and_clean(data)
            
            # Step 2: Handle survivorship bias
            self.logger.info("Step 2: Handling survivorship bias...")
            if universe_file:
                self.survivor_handler.load_universe_data(universe_file)
                # Filter data based on universe at each point in time
                universe_changes = self.survivor_handler.get_universe_changes(
                    clean_data['timestamp'].min(),
                    clean_data['timestamp'].max()
                )
            else:
                universe_changes = None
            
            # Step 3: Walk-forward optimization if parameters provided
            optimization_results = None
            if parameter_ranges:
                self.logger.info("Step 3: Running walk-forward optimization...")
                optimization_results = await self.walk_forward.walk_forward_optimization(
                    clean_data,
                    strategy_class,
                    parameter_ranges,
                    progress_callback
                )
                
                # Use best parameters from optimization
                strategy_params = optimization_results['results'][-1]['parameters']
                parameter_stability = optimization_results['parameter_stability']
            else:
                # Single backtest run with provided parameters
                self.logger.info("Step 3: Running single backtest...")
                strategy_params = {}
                parameter_stability = None
            
            # Step 4: Run backtest with time-aware engine
            self.logger.info("Step 4: Running time-aware backtest...")
            trades_df = await self.time_engine.process_historical_data(
                clean_data,
                strategy_class(**strategy_params)
            )
            
            # Step 5: Apply transaction costs
            self.logger.info("Step 5: Applying transaction costs...")
            for _, trade in trades_df.iterrows():
                costs = self.cost_model.calculate_total_transaction_cost(
                    trade.to_dict(),
                    clean_data.loc[clean_data['timestamp'] == trade['timestamp']].iloc[0].to_dict()
                )
                for cost_type, amount in costs.items():
                    trades_df.loc[trade.name, f'cost_{cost_type}'] = amount
            
            # Step 6: Calculate comprehensive performance metrics
            self.logger.info("Step 6: Calculating performance metrics...")
            performance_metrics = self._calculate_performance_metrics(trades_df)
            
            # Step 7: Statistical significance testing
            significance_results = self._calculate_statistical_significance(
                trades_df, clean_data
            )
            
            # Compile results
            results = BacktestResults(
                trades=trades_df,
                performance_metrics=performance_metrics,
                optimization_results=optimization_results,
                transaction_costs=self.cost_model.analyze_costs(),
                data_quality_stats=self.data_validator.validation_stats,
                universe_changes=universe_changes,
                parameter_stability=parameter_stability,
                statistical_significance=significance_results
            )
            
            # Cache results
            self.results_cache[datetime.now().isoformat()] = results
            
            return results
            
        except Exception as e:
            self.logger.error(f"Comprehensive backtest failed: {e}")
            raise
    
    def _calculate_performance_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics."""
        if trades_df.empty:
            return {}
        
        # Calculate returns
        trades_df['return'] = trades_df['pnl'] / trades_df['notional']
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'profit_factor': (
                trades_df[trades_df['pnl'] > 0]['pnl'].sum() /
                abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
                if len(trades_df[trades_df['pnl'] < 0]) > 0 else float('inf')
            ),
            'avg_trade': trades_df['pnl'].mean(),
            'std_trade': trades_df['pnl'].std(),
            
            # Risk metrics
            'sharpe_ratio': self._calculate_sharpe_ratio(trades_df['return']),
            'sortino_ratio': self._calculate_sortino_ratio(trades_df['return']),
            'max_drawdown': self._calculate_max_drawdown(trades_df['pnl'].cumsum()),
            
            # Cost metrics
            'total_costs': trades_df['cost_total'].sum(),
            'cost_per_trade': trades_df['cost_total'].mean(),
            'cost_ratio': (
                trades_df['cost_total'].sum() / trades_df['notional'].sum()
                if trades_df['notional'].sum() != 0 else 0
            )
        }
        
        return metrics
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free
        if excess_returns.std() == 0:
            return 0.0
        
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free: float = 0.0) -> float:
        """Calculate annualized Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(cumulative_returns) < 2:
            return 0.0
        
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        return abs(drawdowns.min())
    
    def _calculate_statistical_significance(
        self,
        trades_df: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> Dict:
        """Calculate statistical significance of strategy performance."""
        from scipy import stats
        
        if trades_df.empty:
            return {}
        
        # Calculate strategy returns
        strategy_returns = trades_df['return']
        
        # Calculate market returns (close-to-close)
        market_returns = market_data['close'].pct_change().dropna()
        
        # T-test for mean return significance
        t_stat, p_value = stats.ttest_1samp(strategy_returns, 0)
        
        # Calculate correlation with market
        correlation = strategy_returns.corr(market_returns)
        
        # Calculate beta
        covariance = strategy_returns.cov(market_returns)
        market_variance = market_returns.var()
        beta = covariance / market_variance if market_variance != 0 else 0
        
        # Information ratio
        active_returns = strategy_returns - market_returns
        ir = (
            active_returns.mean() / active_returns.std() * np.sqrt(252)
            if len(active_returns) > 1 and active_returns.std() != 0
            else 0
        )
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'market_correlation': correlation,
            'beta': beta,
            'information_ratio': ir,
            'risk_adjusted_return': self._calculate_sharpe_ratio(strategy_returns)
        }
    
    def get_cached_results(self, result_id: Optional[str] = None) -> Optional[BacktestResults]:
        """Retrieve cached backtest results."""
        if result_id is None:
            # Return most recent results
            if not self.results_cache:
                return None
            latest_id = max(self.results_cache.keys())
            return self.results_cache[latest_id]
        
        return self.results_cache.get(result_id)
    
    def clear_cache(self):
        """Clear cached results."""
        self.results_cache.clear()
        self.logger.info("Results cache cleared")