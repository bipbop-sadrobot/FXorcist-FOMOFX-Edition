# Event-Driven Backtest Pipeline

## Overview

The FXorcist backtest pipeline provides a robust, event-driven approach to backtesting trading strategies. It offers a flexible and extensible system for simulating trading performance across various market conditions.

## Key Features

- **Event-Driven Architecture**: Market data is processed as a stream of events, enabling precise simulation of trading conditions.
- **Anti-Bias Protection**: Prevents look-ahead bias by strictly enforcing timestamp-based event processing.
- **Flexible Strategy Integration**: Easy to plug in custom trading strategies through a standardized interface.
- **Comprehensive Metrics**: Calculates detailed performance metrics for strategy evaluation.

## Architecture

### Components

1. **Event Bus**: Manages the chronological stream of market events.
2. **Backtest Engine**: Processes events, executes strategy logic, and tracks portfolio state.
3. **Market Data Loader**: Converts historical market data into event streams.
4. **Strategy Registry**: Manages and loads trading strategy implementations.

### Event Types

- **Tick Events**: Represent individual price updates
- **Bar Events**: Represent OHLC (Open, High, Low, Close) price data

## Usage

### Running a Backtest

```python
from fxorcist.pipeline.backtest import run_backtest

config = {
    "backtest_start_date": "2023-01-01",
    "backtest_end_date": "2023-12-31",
    "initial_capital": 10000
}

results = run_backtest(
    strategy_name='my_strategy',
    symbol='EURUSD',
    config=config
)
```

### Implementing a Strategy

```python
from fxorcist.strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    def on_event(self, event, market_snapshot):
        # Generate trading signals based on event and market data
        signals = []
        # ... strategy logic ...
        return signals

    def signal_to_orders(self, signals):
        # Convert signals to executable orders
        orders = []
        # ... order generation logic ...
        return orders
```

## Performance Metrics

The backtest pipeline calculates comprehensive performance metrics:
- Total Return
- Sharpe Ratio
- Maximum Drawdown
- Win/Loss Ratio
- Trade Count
- Average Trade Duration

## Anti-Bias Protections

- Strict chronological event processing
- Timestamp-based data access controls
- Prevention of future data leakage

## Configuration Options

- `backtest_start_date`: Start date for the backtest
- `backtest_end_date`: End date for the backtest
- `initial_capital`: Starting portfolio value
- `commission_rate`: Trading commission percentage
- `slippage_model`: Slippage simulation strategy

## Best Practices

1. Use realistic market data
2. Implement robust error handling in strategies
3. Consider transaction costs and slippage
4. Validate strategy performance across multiple market conditions

## Limitations

- Backtests are historical simulations and do not guarantee future performance
- Assumes perfect order execution (real-world conditions may vary)
- Limited by available historical market data

## Future Improvements

- Machine learning-based strategy optimization
- More advanced slippage and transaction cost models
- Enhanced performance metric calculations