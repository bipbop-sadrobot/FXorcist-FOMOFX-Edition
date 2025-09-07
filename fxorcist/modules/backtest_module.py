"""
Event-driven backtesting module for FXorcist trading platform.
Integrates enhanced backtesting with event system and visualization.
"""

import asyncio
from typing import Dict, Any, Optional, List
import pandas as pd
import logging
from datetime import datetime, timezone

from ..core.base import TradingModule
from ..core.events import (
    Event, EventType, BacktestStartEvent, BacktestUpdateEvent,
    BacktestCompleteEvent, OptimizationStartEvent, OptimizationUpdateEvent,
    OptimizationCompleteEvent
)
from ..core.dispatcher import EventDispatcher
from ..pipeline.enhanced_backtest import (
    backtest_strategy, BacktestConfig, TradeStats,
    plot_equity_curve
)
from ..pipeline.parameter_optimizer import (
    ParameterOptimizer, OptimizationConfig
)

class BacktestModule(TradingModule):
    """
    Event-driven backtesting module.
    
    Responsibilities:
    - Execute backtests with realistic market simulation
    - Manage parameter optimization workflows
    - Generate performance visualizations
    - Publish backtest and optimization events
    """
    
    def __init__(self, event_dispatcher: EventDispatcher, config: Dict[str, Any]):
        """
        Initialize the backtest module.
        
        Args:
            event_dispatcher: Central event dispatcher instance
            config: Configuration dictionary containing:
                - backtest_config: BacktestConfig settings
                - optimization_config: OptimizationConfig settings (optional)
        """
        super().__init__("BacktestModule", event_dispatcher, config)
        
        # Initialize configurations
        self.backtest_config = BacktestConfig(**config.get('backtest_config', {}))
        
        opt_config = config.get('optimization_config', {})
        if opt_config:
            self.optimization_config = OptimizationConfig(**opt_config)
            self.optimizer = ParameterOptimizer(self.optimization_config)
        else:
            self.optimization_config = None
            self.optimizer = None
        
        # State tracking
        self.current_backtest = None
        self.current_optimization = None
        self.results_cache = {}
    
    async def start(self):
        """Start the backtest module."""
        await super().start()
        
        # Subscribe to control events
        self.event_dispatcher.subscribe(EventType.SYSTEM, self.handle_event)
        self.logger.info("Backtest module started")
    
    async def handle_event(self, event: Event):
        """
        Handle system control events.
        
        Args:
            event: Event to process
        """
        if event.type == EventType.SYSTEM:
            command = event.data.get('command')
            
            if command == 'run_backtest':
                await self.run_backtest(
                    data=event.data.get('data'),
                    strategy=event.data.get('strategy'),
                    config=event.data.get('config')
                )
            elif command == 'run_optimization':
                await self.run_optimization(
                    data=event.data.get('data'),
                    strategy_generator=event.data.get('strategy_generator'),
                    config=event.data.get('config')
                )
            elif command == 'stop':
                await self.stop()
    
    async def run_backtest(self, data: pd.DataFrame, strategy: callable,
                          config: Optional[Dict[str, Any]] = None):
        """
        Run backtest with progress updates.
        
        Args:
            data: Price data
            strategy: Strategy function
            config: Optional backtest configuration override
        """
        if self.current_backtest:
            self.logger.warning("Backtest already running")
            return
        
        try:
            self.current_backtest = {
                'start_time': datetime.now(timezone.utc),
                'data': data,
                'strategy': strategy
            }
            
            # Update config if provided
            if config:
                self.backtest_config = BacktestConfig(**config)
            
            # Publish start event
            await self.publish_event(BacktestStartEvent(
                config=self.backtest_config.__dict__
            ))
            
            # Run backtest
            trades_df, stats = backtest_strategy(
                data, strategy, self.backtest_config
            )
            
            # Generate visualizations
            equity_curve = plot_equity_curve(trades_df)
            
            # Store results
            result_id = str(self.current_backtest['start_time'].timestamp())
            self.results_cache[result_id] = {
                'trades': trades_df.to_dict('records'),
                'stats': stats.__dict__,
                'plots': {
                    'equity_curve': equity_curve
                }
            }
            
            # Publish completion event
            await self.publish_event(BacktestCompleteEvent(
                stats=stats.__dict__,
                trades=trades_df.to_dict('records')
            ))
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise
        
        finally:
            self.current_backtest = None
    
    async def run_optimization(self, data: pd.DataFrame,
                             strategy_generator: callable,
                             config: Optional[Dict[str, Any]] = None):
        """
        Run parameter optimization with progress updates.
        
        Args:
            data: Price data
            strategy_generator: Function that creates strategy with parameters
            config: Optional optimization configuration override
        """
        if not self.optimizer:
            raise RuntimeError("Optimizer not configured")
        
        if self.current_optimization:
            self.logger.warning("Optimization already running")
            return
        
        try:
            self.current_optimization = {
                'start_time': datetime.now(timezone.utc),
                'data': data,
                'strategy_generator': strategy_generator
            }
            
            # Update config if provided
            if config:
                self.optimization_config = OptimizationConfig(**config)
                self.optimizer = ParameterOptimizer(self.optimization_config)
            
            # Publish start event
            await self.publish_event(OptimizationStartEvent(
                config=self.optimization_config.__dict__
            ))
            
            # Run optimization
            results = self.optimizer.optimize(
                data, strategy_generator, self.backtest_config
            )
            
            # Generate visualization
            history_plot = self.optimizer.plot_optimization_history()
            importance_plot = self.optimizer.plot_param_importance()
            
            # Store results
            result_id = str(self.current_optimization['start_time'].timestamp())
            self.results_cache[result_id] = {
                'results': results,
                'plots': {
                    'history': history_plot,
                    'importance': importance_plot
                }
            }
            
            # Publish completion event
            await self.publish_event(OptimizationCompleteEvent(
                results=results
            ))
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise
        
        finally:
            self.current_optimization = None
    
    def get_results(self, result_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached results.
        
        Args:
            result_id: Result identifier
            
        Returns:
            Dictionary containing results and visualizations
        """
        return self.results_cache.get(result_id)
    
    async def stop(self):
        """Stop the backtest module."""
        self.results_cache.clear()
        await super().stop()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the backtest module.
        
        Returns:
            Dictionary containing status information
        """
        status = super().get_status()
        status.update({
            'backtest_running': bool(self.current_backtest),
            'optimization_running': bool(self.current_optimization),
            'cached_results': len(self.results_cache)
        })
        return status