"""
Backtest pipeline integration, tying together the event bus, backtest engine, and strategy.
"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from rich.progress import Progress
import pandas as pd

from fxorcist.events.event_bus import EventBus, create_tick_event, create_bar_event
from fxorcist.backtest.engine import BacktestEngine
from fxorcist.backtest.metrics import calculate_metrics
from fxorcist.data.loader import load_symbol

def run_backtest(
    strategy_name: str,
    symbol: str,
    config: Dict[str, Any],
    params_file: Optional[str] = None,
    progress: Optional[Progress] = None,
) -> Dict[str, Any]:
    """
    Run a backtest for the given strategy and symbol.
    
    Args:
        strategy_name: Name of the strategy to use
        symbol: Trading symbol to backtest
        config: Application configuration
        params_file: Optional file with strategy parameters
        progress: Optional progress bar instance
        
    Returns:
        Dictionary of backtest results, including performance metrics.
    """
    # Load strategy
    from fxorcist.strategies.registry import get_strategy
    strategy = get_strategy(strategy_name)
    
    # Create event bus and backtest engine
    event_bus = EventBus()
    engine = BacktestEngine(event_bus, initial_capital=config.get("initial_capital", 100000))
    
    # Load market data
    start_date = config.get("backtest_start_date")
    end_date = config.get("backtest_end_date")
    
    for event in load_market_data(symbol, start_date, end_date):
        event_bus.append(event)
    
    # Run backtest
    results = engine.run(strategy, start_date, end_date, progress=progress)
    
    return results

def load_market_data(symbol: str, start_date: str, end_date: str):
    """
    Load market data for the given symbol and date range.
    
    Args:
        symbol: Trading symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Yields:
        Event objects for the market data
    """
    # Load data from storage/API and convert to events
    # This is a placeholder, the actual implementation will depend on the data source
    
    now = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    while now <= end:
        yield create_tick_event(
            timestamp=now,
            symbol=symbol,
            bid=1.1000,
            ask=1.1001,
        )
        yield create_bar_event(
            timestamp=now,
            symbol=symbol,
            open_price=1.1000,
            high=1.1002,
            low=1.0999,
            close=1.1001,
        )
        now += timedelta(minutes=1)