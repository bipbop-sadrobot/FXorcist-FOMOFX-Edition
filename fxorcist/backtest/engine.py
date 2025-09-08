"""
Event-driven backtest engine with anti-bias protection.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
from rich.progress import Progress

from fxorcist.events.event_bus import EventBus, Event, EventType, LookAheadError
from fxorcist.backtest.metrics import calculate_metrics

@dataclass
class Position:
    """Trading position state."""
    symbol: str
    size: float = 0.0
    avg_entry: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

@dataclass
class Portfolio:
    """Portfolio state with positions and metrics."""
    initial_capital: float
    cash: float = field(init=False)
    positions: Dict[str, Position] = field(default_factory=dict)
    equity: float = field(init=False)
    trades: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.cash = self.initial_capital
        self.equity = self.initial_capital

    def snapshot(self) -> Dict[str, Any]:
        """Create immutable snapshot of current state."""
        return {
            "cash": self.cash,
            "equity": self.equity,
            "positions": {
                symbol: {
                    "size": pos.size,
                    "avg_entry": pos.avg_entry,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "realized_pnl": pos.realized_pnl,
                }
                for symbol, pos in self.positions.items()
            },
            "trades_count": len(self.trades)
        }

class BacktestEngine:
    """
    Event-driven backtest engine with anti-bias protection.
    
    Features:
    - Chronological event processing
    - Look-ahead prevention
    - Portfolio state management
    - Performance metrics calculation
    """
    def __init__(
        self,
        event_bus: EventBus,
        initial_capital: float = 100_000.0,
        commission_rate: float = 0.001,
    ):
        self.event_bus = event_bus
        self.portfolio = Portfolio(initial_capital=initial_capital)
        self.commission_rate = commission_rate
        self.market_data: Dict[str, Dict[str, float]] = {}
        self.snapshots: List[Dict[str, Any]] = []

    def run(
        self,
        strategy: Any,
        start_ts: datetime,
        end_ts: datetime,
        progress: Optional[Progress] = None,
    ) -> Dict[str, Any]:
        """
        Run backtest simulation.
        
        Args:
            strategy: Strategy instance with on_event method
            start_ts: Start timestamp
            end_ts: End timestamp
            progress: Optional progress bar
            
        Returns:
            Dict with performance metrics
        """
        if progress:
            task = progress.add_task("Running backtest...", total=None)

        try:
            # Replay market data events first
            for event in self.event_bus.replay(
                start_ts,
                end_ts,
                event_types=[EventType.TICK, EventType.BAR],
            ):
                self._update_market_data(event)

            # Then replay all events for strategy
            for event in self.event_bus.replay(start_ts, end_ts):
                # Enforce anti-bias
                self.event_bus.validate_timestamp(event.timestamp)
                
                # Get strategy signals
                signals = strategy.on_event(event, self._get_market_snapshot())
                
                # Convert signals to orders
                if signals:
                    orders = strategy.signals_to_orders(
                        signals,
                        self.portfolio.snapshot(),
                        self._get_market_snapshot(),
                    )
                    
                    # Execute orders
                    for order in orders:
                        self._execute_order(order, event.timestamp)

                # Update portfolio state
                self._mark_to_market(event.timestamp)
                
                # Save snapshot
                self.snapshots.append({
                    "timestamp": event.timestamp,
                    **self.portfolio.snapshot()
                })

                if progress:
                    progress.update(task, advance=1)

        except LookAheadError as e:
            raise LookAheadError(f"Strategy attempted to look ahead: {e}")
        
        except Exception as e:
            raise RuntimeError(f"Backtest failed: {e}")

        finally:
            if progress:
                progress.update(task, completed=True)

        # Calculate metrics
        return calculate_metrics(self.snapshots)

    def _update_market_data(self, event: Event) -> None:
        """Update market data cache."""
        if event.type == EventType.TICK:
            self.market_data[event.symbol] = {
                "bid": event.data["bid"],
                "ask": event.data["ask"],
                "mid": (event.data["bid"] + event.data["ask"]) / 2,
            }
        elif event.type == EventType.BAR:
            self.market_data[event.symbol] = {
                "open": event.data["open"],
                "high": event.data["high"],
                "low": event.data["low"],
                "close": event.data["close"],
                "mid": event.data["close"],  # Use close as mid for bars
            }

    def _get_market_snapshot(self) -> Dict[str, Dict[str, float]]:
        """Get current market data snapshot."""
        return self.market_data.copy()

    def _execute_order(self, order: Dict[str, Any], timestamp: datetime) -> None:
        """
        Execute an order and update portfolio state.
        
        Args:
            order: Order details (symbol, size, type, etc.)
            timestamp: Execution timestamp
        """
        symbol = order["symbol"]
        size = order["size"]
        
        # Get execution price
        if size > 0:  # Buy at ask
            price = self.market_data[symbol]["ask"]
        else:  # Sell at bid
            price = self.market_data[symbol]["bid"]
            
        # Calculate commission
        commission = abs(size * price * self.commission_rate)
        
        # Update position
        if symbol not in self.portfolio.positions:
            self.portfolio.positions[symbol] = Position(symbol=symbol)
            
        position = self.portfolio.positions[symbol]
        
        # Calculate PnL if closing
        if (position.size > 0 and size < 0) or (position.size < 0 and size > 0):
            realized_pnl = (price - position.avg_entry) * min(abs(size), abs(position.size))
            position.realized_pnl += realized_pnl
            
        # Update position
        if position.size + size != 0:  # If not closing entirely
            position.avg_entry = (
                (position.size * position.avg_entry + size * price)
                / (position.size + size)
            )
        position.size += size
        
        # Update cash
        self.portfolio.cash -= (size * price + commission)
        
        # Record trade
        self.portfolio.trades.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "size": size,
            "price": price,
            "commission": commission,
            "realized_pnl": realized_pnl if "realized_pnl" in locals() else 0.0,
        })

    def _mark_to_market(self, timestamp: datetime) -> None:
        """
        Mark positions to market and update portfolio value.
        
        Args:
            timestamp: Current timestamp
        """
        total_value = self.portfolio.cash
        
        for position in self.portfolio.positions.values():
            if position.size != 0:
                mid_price = self.market_data[position.symbol]["mid"]
                position.unrealized_pnl = (
                    (mid_price - position.avg_entry) * position.size
                )
                total_value += position.unrealized_pnl
                
        self.portfolio.equity = total_value