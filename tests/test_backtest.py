"""
Tests for the event-driven backtest engine and performance metrics.
"""
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import pytest
import numpy as np

from fxorcist.events.event_bus import (
    EventBus,
    create_tick_event,
    create_bar_event,
)
from fxorcist.backtest.engine import BacktestEngine, LookAheadError
from fxorcist.backtest.metrics import calculate_metrics, calculate_rolling_metrics

# Test fixtures
@pytest.fixture
def event_bus():
    """Create a clean event bus for testing."""
    return EventBus()

@pytest.fixture
def sample_events(event_bus):
    """Create a sequence of sample events."""
    now = datetime.now()
    event_bus.append(create_tick_event(
        timestamp=now,
        symbol="EURUSD",
        bid=1.1000,
        ask=1.1001,
    ))
    event_bus.append(create_tick_event(
        timestamp=now + timedelta(seconds=1),
        symbol="EURUSD",
        bid=1.1001,
        ask=1.1002,
    ))
    event_bus.append(create_bar_event(
        timestamp=now + timedelta(minutes=1),
        symbol="EURUSD",
        open_price=1.1000,
        high=1.1002,
        low=1.0999,
        close=1.1001,
    ))
    return event_bus

# Test backtest engine
def test_backtest_execution(event_bus, sample_events):
    """Test the backtest engine execution flow."""
    class MockStrategy:
        def on_event(self, event, market_snapshot):
            return []
        
        def signals_to_orders(self, signals, portfolio_snapshot, market_snapshot):
            return []
    
    engine = BacktestEngine(event_bus, initial_capital=10000)
    
    with patch("fxorcist.backtest.engine.Progress") as mock_progress:
        mock_progress_instance = MagicMock()
        mock_progress.return_value = mock_progress_instance
        
        results = engine.run(
            strategy=MockStrategy(),
            start_ts=sample_events[0].timestamp,
            end_ts=sample_events[-1].timestamp,
            progress=mock_progress_instance,
        )
        
        mock_progress.assert_called_once()
        mock_progress_instance.update.assert_called()

    assert results["final_equity"] == 10000
    assert results["total_trades"] == 0

def test_portfolio_state_updates(event_bus, sample_events):
    """Test portfolio state updates during backtest."""
    class MockStrategy:
        def on_event(self, event, market_snapshot):
            if event.type == "tick":
                return [{"symbol": event.symbol, "size": 1, "type": "market"}]
            return []
        
        def signals_to_orders(self, signals, portfolio_snapshot, market_snapshot):
            return [
                {
                    "symbol": signal["symbol"],
                    "size": signal["size"],
                    "type": signal["type"],
                }
                for signal in signals
            ]
    
    engine = BacktestEngine(event_bus, initial_capital=10000)
    results = engine.run(
        strategy=MockStrategy(),
        start_ts=sample_events[0].timestamp,
        end_ts=sample_events[-1].timestamp,
    )
    
    assert results["final_equity"] != 10000
    assert results["total_trades"] == 2
    assert len(results["trades"]) == 2

def test_look_ahead_prevention(event_bus, sample_events):
    """Test that look-ahead access is prevented."""
    class MockStrategy:
        def on_event(self, event, market_snapshot):
            if event.timestamp > market_snapshot["timestamp"]:
                raise ValueError("Attempted to access future data")
            return []
        
        def signals_to_orders(self, signals, portfolio_snapshot, market_snapshot):
            return []
    
    engine = BacktestEngine(event_bus, initial_capital=10000)
    
    with pytest.raises(LookAheadError):
        engine.run(
            strategy=MockStrategy(),
            start_ts=sample_events[0].timestamp,
            end_ts=sample_events[-1].timestamp,
        )

# Test performance metrics
def test_metrics_calculation(event_bus, sample_events):
    """Test performance metrics calculation."""
    class MockStrategy:
        def on_event(self, event, market_snapshot):
            if event.type == "tick":
                return [{"symbol": event.symbol, "size": 1, "type": "market"}]
            return []
        
        def signals_to_orders(self, signals, portfolio_snapshot, market_snapshot):
            return [
                {
                    "symbol": signal["symbol"],
                    "size": signal["size"],
                    "type": signal["type"],
                }
                for signal in signals
            ]
    
    engine = BacktestEngine(event_bus, initial_capital=10000)
    results = engine.run(
        strategy=MockStrategy(),
        start_ts=sample_events[0].timestamp,
        end_ts=sample_events[-1].timestamp,
    )
    
    assert "sharpe_ratio" in results
    assert "max_drawdown" in results
    assert "win_rate" in results
    assert "profit_factor" in results
    assert "avg_trade" in results

def test_rolling_metrics(event_bus, sample_events):
    """Test rolling performance metrics calculation."""
    class MockStrategy:
        def on_event(self, event, market_snapshot):
            if event.type == "tick":
                return [{"symbol": event.symbol, "size": 1, "type": "market"}]
            return []
        
        def signals_to_orders(self, signals, portfolio_snapshot, market_snapshot):
            return [
                {
                    "symbol": signal["symbol"],
                    "size": signal["size"],
                    "type": signal["type"],
                }
                for signal in signals
            ]
    
    engine = BacktestEngine(event_bus, initial_capital=10000)
    engine.run(
        strategy=MockStrategy(),
        start_ts=sample_events[0].timestamp,
        end_ts=sample_events[-1].timestamp,
    )
    
    rolling_metrics = calculate_rolling_metrics(engine.snapshots)
    assert "rolling_sharpe" in rolling_metrics
    assert "rolling_volatility" in rolling_metrics
    assert "rolling_returns" in rolling_metrics
    
    # Verify series lengths
    assert len(rolling_metrics["rolling_sharpe"]) == len(engine.snapshots) - 252