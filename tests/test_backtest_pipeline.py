"""
Tests for the backtest pipeline implementation.
"""
import pytest
from datetime import datetime, timedelta
import pandas as pd

from fxorcist.pipeline.backtest import run_backtest, load_market_data
from fxorcist.events.event_bus import EventBus
from fxorcist.strategies.base import BaseStrategy

class DummyStrategy(BaseStrategy):
    """
    A simple dummy strategy for testing backtest pipeline.
    """
    def __init__(self):
        self.trades = []
    
    def on_event(self, event, market_snapshot):
        """
        Dummy strategy that always generates a buy signal.
        """
        return [{'type': 'buy', 'size': 1}]
    
    def signal_to_orders(self, signals):
        """
        Convert signals to orders.
        """
        return signals

def test_load_market_data():
    """
    Test market data loading and event generation.
    """
    # Create a sample DataFrame
    dates = pd.date_range(start='2023-01-01', end='2023-01-02', freq='1H')
    df = pd.DataFrame({
        'open': [1.0] * len(dates),
        'high': [1.1] * len(dates),
        'low': [0.9] * len(dates),
        'close': [1.05] * len(dates)
    }, index=dates)
    
    # Mock load_symbol to return our sample DataFrame
    import fxorcist.data.loader
    original_load_symbol = fxorcist.data.loader.load_symbol
    fxorcist.data.loader.load_symbol = lambda *args, **kwargs: df
    
    try:
        events = list(load_market_data('EURUSD', '2023-01-01', '2023-01-02'))
        
        # Verify events
        assert len(events) == len(dates) * 2  # Tick and bar events for each timestamp
        assert all(event.type in ['tick', 'bar'] for event in events)
        assert all(event.payload['symbol'] == 'EURUSD' for event in events)
    finally:
        # Restore original load_symbol
        fxorcist.data.loader.load_symbol = original_load_symbol

def test_run_backtest():
    """
    Test full backtest pipeline execution.
    """
    # Mock strategy registry
    import fxorcist.strategies.registry
    original_get_strategy = fxorcist.strategies.registry.get_strategy
    fxorcist.strategies.registry.get_strategy = lambda name: DummyStrategy()
    
    # Mock load_symbol
    import fxorcist.data.loader
    dates = pd.date_range(start='2023-01-01', end='2023-01-02', freq='1H')
    df = pd.DataFrame({
        'open': [1.0] * len(dates),
        'high': [1.1] * len(dates),
        'low': [0.9] * len(dates),
        'close': [1.05] * len(dates)
    }, index=dates)
    original_load_symbol = fxorcist.data.loader.load_symbol
    fxorcist.data.loader.load_symbol = lambda *args, **kwargs: df
    
    try:
        config = {
            "backtest_start_date": "2023-01-01",
            "backtest_end_date": "2023-01-02",
            "initial_capital": 10000
        }
        
        results = run_backtest(
            strategy_name='dummy_strategy', 
            symbol='EURUSD', 
            config=config
        )
        
        # Verify results
        assert isinstance(results, dict)
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert results['initial_capital'] == 10000
    finally:
        # Restore original functions
        fxorcist.strategies.registry.get_strategy = original_get_strategy
        fxorcist.data.loader.load_symbol = original_load_symbol

def test_backtest_error_handling():
    """
    Test error handling in backtest pipeline.
    """
    # Test with invalid strategy
    with pytest.raises(ValueError, match="Strategy not found"):
        run_backtest(
            strategy_name='non_existent_strategy', 
            symbol='EURUSD', 
            config={}
        )
    
    # Test with invalid date range
    with pytest.raises(ValueError, match="Invalid date range"):
        run_backtest(
            strategy_name='dummy_strategy', 
            symbol='EURUSD', 
            config={
                "backtest_start_date": "2023-01-02",
                "backtest_end_date": "2023-01-01"
            }
        )