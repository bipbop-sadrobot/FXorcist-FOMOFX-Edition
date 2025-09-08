"""
Portfolio Manager module for the FXorcist trading platform.
Handles portfolio state, risk management, and order generation.
"""

from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

from ..core.base import TradingModule
from ..core.events import Event, EventType, OrderEvent, SignalEvent
from ..core.dispatcher import EventDispatcher

class Position:
    """Represents a trading position."""
    
    def __init__(self, instrument: str, direction: str, units: int, 
                 entry_price: Decimal, entry_time: datetime):
        self.instrument = instrument
        self.direction = direction  # LONG or SHORT
        self.units = units
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.unrealized_pnl = Decimal('0')
        self.realized_pnl = Decimal('0')
    
    def update_pnl(self, current_price: Decimal):
        """Update unrealized P&L based on current price."""
        if self.direction == "LONG":
            self.unrealized_pnl = (current_price - self.entry_price) * self.units
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - current_price) * self.units

class PortfolioManager(TradingModule):
    """
    Manages portfolio state, position sizing, and risk controls.
    
    Responsibilities:
    - Portfolio state management
    - Risk management and position sizing
    - Order generation from signals
    - P&L tracking and portfolio metrics
    """
    
    def __init__(self, event_dispatcher: EventDispatcher, config: Dict[str, Any]):
        """
        Initialize the portfolio manager.
        
        Args:
            event_dispatcher: Central event dispatcher instance
            config: Configuration dictionary containing:
                - initial_balance: Starting account balance
                - max_risk_per_trade: Maximum risk per trade (as decimal)
                - max_positions: Maximum number of open positions
                - max_drawdown: Maximum allowed drawdown (as decimal)
        """
        super().__init__("PortfolioManager", event_dispatcher, config)
        
        # Portfolio state
        self.initial_balance = Decimal(str(config.get('initial_balance', 10000)))
        self.current_balance = self.initial_balance
        self.positions: Dict[str, Position] = {}
        self.trades_history: List[Dict] = []
        
        # Risk parameters
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)  # 2%
        self.max_positions = config.get('max_positions', 3)
        self.max_drawdown = config.get('max_drawdown', 0.20)  # 20%
        
        # Performance tracking
        self.peak_balance = self.initial_balance
        self.current_drawdown = Decimal('0')
        
        # Market data cache
        self.market_prices: Dict[str, Dict[str, Decimal]] = {}
    
    async def start(self):
        """Start portfolio management."""
        await super().start()
        
        # Subscribe to relevant events
        self.event_dispatcher.subscribe(EventType.SIGNAL, self.handle_event)
        self.event_dispatcher.subscribe(EventType.MARKET, self.handle_event)
        self.event_dispatcher.subscribe(EventType.FILL, self.handle_event)
    
    async def stop(self):
        """Stop portfolio management."""
        # Close all positions
        for instrument in list(self.positions.keys()):
            await self._close_position(instrument)
        await super().stop()
    
    async def handle_event(self, event: Event):
        """
        Handle incoming events.
        
        Args:
            event: Event to process
        """
        if event.type == EventType.SIGNAL:
            await self._process_signal(event)
        elif event.type == EventType.MARKET:
            await self._process_market_data(event)
        elif event.type == EventType.FILL:
            await self._process_fill(event)
    
    async def _process_signal(self, event: SignalEvent):
        """
        Process trading signals and generate orders.
        
        Args:
            event: Signal event to process
        """
        instrument = event.data['instrument']
        direction = event.data['direction']
        strength = event.data['strength']
        
        # Check if we can take new positions
        if not self._can_take_new_position(instrument):
            self.logger.debug(
                f"Cannot take new position in {instrument}: "
                "Position limits or risk controls reached"
            )
            return
        
        # Calculate position size
        units = self._calculate_position_size(
            instrument, direction, strength, event.data.get('price')
        )
        
        if units == 0:
            return
        
        # Generate order
        order_direction = "BUY" if direction == "LONG" else "SELL"
        order_event = OrderEvent(
            instrument=instrument,
            order_type="MARKET",
            direction=order_direction,
            units=units,
            timestamp=event.timestamp,
            signal_strength=strength,
            strategy=event.data.get('strategy')
        )
        
        await self.publish_event(order_event)
        self.logger.info(
            f"Order generated: {order_direction} {units} units of {instrument}"
        )
    
    async def _process_market_data(self, event: Event):
        """
        Process market data updates.
        
        Args:
            event: Market data event
        """
        instrument = event.data['instrument']
        
        # Update market prices cache
        self.market_prices[instrument] = {
            'bid': Decimal(str(event.data['bid'])),
            'ask': Decimal(str(event.data['ask'])),
            'mid': (Decimal(str(event.data['bid'])) + 
                   Decimal(str(event.data['ask']))) / 2
        }
        
        # Update position P&L if we have one
        if instrument in self.positions:
            self.positions[instrument].update_pnl(
                self.market_prices[instrument]['mid']
            )
            await self._check_risk_limits()
    
    async def _process_fill(self, event: Event):
        """
        Process fill events and update portfolio state.
        
        Args:
            event: Fill event to process
        """
        instrument = event.data['instrument']
        direction = event.data['direction']
        units = event.data['units']
        fill_price = Decimal(str(event.data['fill_price']))
        
        if direction == "BUY":
            await self._open_long_position(
                instrument, units, fill_price, event.timestamp
            )
        else:  # SELL
            await self._open_short_position(
                instrument, units, fill_price, event.timestamp
            )
    
    async def _open_long_position(
        self, instrument: str, units: int, 
        price: Decimal, timestamp: datetime
    ):
        """Open a long position."""
        if instrument in self.positions:
            # Update existing position
            position = self.positions[instrument]
            if position.direction == "LONG":
                # Add to long position
                total_units = position.units + units
                total_value = (position.entry_price * position.units + 
                             price * units)
                position.units = total_units
                position.entry_price = total_value / total_units
            else:
                # Close short position and open long
                await self._close_position(instrument)
                self.positions[instrument] = Position(
                    instrument, "LONG", units, price, timestamp
                )
        else:
            # Open new position
            self.positions[instrument] = Position(
                instrument, "LONG", units, price, timestamp
            )
        
        self.logger.info(
            f"Opened long position in {instrument}: "
            f"{units} units at {price}"
        )
    
    async def _open_short_position(
        self, instrument: str, units: int, 
        price: Decimal, timestamp: datetime
    ):
        """Open a short position."""
        if instrument in self.positions:
            # Update existing position
            position = self.positions[instrument]
            if position.direction == "SHORT":
                # Add to short position
                total_units = position.units + units
                total_value = (position.entry_price * position.units + 
                             price * units)
                position.units = total_units
                position.entry_price = total_value / total_units
            else:
                # Close long position and open short
                await self._close_position(instrument)
                self.positions[instrument] = Position(
                    instrument, "SHORT", units, price, timestamp
                )
        else:
            # Open new position
            self.positions[instrument] = Position(
                instrument, "SHORT", units, price, timestamp
            )
        
        self.logger.info(
            f"Opened short position in {instrument}: "
            f"{units} units at {price}"
        )
    
    async def _close_position(self, instrument: str):
        """
        Close a position and record the trade.
        
        Args:
            instrument: Instrument to close position for
        """
        if instrument not in self.positions:
            return
        
        position = self.positions[instrument]
        current_price = self.market_prices[instrument]['mid']
        
        # Calculate final P&L
        position.update_pnl(current_price)
        final_pnl = position.unrealized_pnl
        
        # Update account balance
        self.current_balance += final_pnl
        
        # Record trade
        self.trades_history.append({
            'instrument': instrument,
            'direction': position.direction,
            'units': position.units,
            'entry_price': float(position.entry_price),
            'exit_price': float(current_price),
            'entry_time': position.entry_time,
            'exit_time': datetime.now(),
            'pnl': float(final_pnl)
        })
        
        # Remove position
        del self.positions[instrument]
        
        self.logger.info(
            f"Closed position in {instrument}: "
            f"P&L: {final_pnl:.2f}"
        )
        
        # Update performance metrics
        await self._update_performance_metrics()
    
    def _calculate_position_size(
        self, instrument: str, direction: str, 
        signal_strength: float, price: Optional[float] = None
    ) -> int:
        """
        Calculate position size based on risk management rules.
        
        Args:
            instrument: Instrument to trade
            direction: Trade direction
            signal_strength: Signal strength (0-1)
            price: Optional current price
            
        Returns:
            Number of units to trade
        """
        # Risk amount for this trade
        risk_amount = (self.current_balance * 
                      Decimal(str(self.max_risk_per_trade)) * 
                      Decimal(str(signal_strength)))
        
        # Get current price if not provided
        if price is None and instrument in self.market_prices:
            price = float(self.market_prices[instrument]['mid'])
        elif price is None:
            return 0
        
        # Calculate position size (simplified)
        # In production, this would use proper position sizing based on
        # volatility, stop distance, and pip values
        base_units = int(risk_amount * 1000)  # Simplified calculation
        
        # Apply position limits
        return min(base_units, 100000)  # Cap at 100k units
    
    def _can_take_new_position(self, instrument: str) -> bool:
        """
        Check if we can take a new position.
        
        Args:
            instrument: Instrument to check
            
        Returns:
            True if new position is allowed, False otherwise
        """
        # Check number of open positions
        if (len(self.positions) >= self.max_positions and 
            instrument not in self.positions):
            return False
        
        # Check drawdown limit
        if self.current_drawdown >= self.max_drawdown:
            return False
        
        return True
    
    async def _check_risk_limits(self):
        """Check and enforce risk limits."""
        # Update performance metrics
        await self._update_performance_metrics()
        
        # Check drawdown limit
        if self.current_drawdown >= self.max_drawdown:
            self.logger.warning(
                f"Maximum drawdown reached: {float(self.current_drawdown):.1%}"
            )
            # Close all positions
            for instrument in list(self.positions.keys()):
                await self._close_position(instrument)
    
    async def _update_performance_metrics(self):
        """Update performance tracking metrics."""
        # Update peak balance
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        # Update drawdown
        if self.peak_balance > 0:
            self.current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current portfolio status.
        
        Returns:
            Dictionary containing portfolio status information
        """
        status = super().get_status()
        
        # Calculate total P&L
        unrealized_pnl = sum(
            pos.unrealized_pnl for pos in self.positions.values()
        )
        realized_pnl = self.current_balance - self.initial_balance
        
        status.update({
            'balance': float(self.current_balance),
            'unrealized_pnl': float(unrealized_pnl),
            'realized_pnl': float(realized_pnl),
            'total_equity': float(self.current_balance + unrealized_pnl),
            'open_positions': len(self.positions),
            'total_trades': len(self.trades_history),
            'current_drawdown': float(self.current_drawdown)
        })
        return status
    
    async def reset(self):
        """Reset the portfolio manager to initial state."""
        # Close all positions
        for instrument in list(self.positions.keys()):
            await self._close_position(instrument)
        
        # Reset state
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.current_drawdown = Decimal('0')
        self.positions.clear()
        self.trades_history.clear()
        self.market_prices.clear()
        
        await super().reset()