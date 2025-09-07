"""
Execution handler for simulating realistic trade execution with transaction costs.

This module handles the conversion of order events into fill events, applying:
1. Bid/Ask spreads
2. Slippage models
3. Commission costs

This provides a more realistic simulation of trading costs compared to assuming
perfect execution at the close price.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import queue

from ..models.transaction_costs import (
    SlippageModel,
    CommissionModel,
    FixedAbsoluteSlippageModel,
    OANDACommissionModel
)


@dataclass
class OrderEvent:
    """Represents a request to execute a trade."""
    timestamp: datetime
    instrument: str
    units: float  # Positive for buy, negative for sell
    direction: str  # 'BUY' or 'SELL'
    order_type: str = 'MARKET'  # Default to market orders for now
    price: Optional[float] = None  # Used for limit orders


@dataclass
class FillEvent:
    """Represents the actual execution of a trade with costs."""
    timestamp: datetime
    instrument: str
    units: float
    direction: str
    fill_price: float
    commission: float
    slippage: float = 0.0  # Track slippage for analysis


class ExecutionHandler:
    """
    Handles the execution of orders with realistic transaction costs.
    
    This class converts OrderEvents into FillEvents, applying:
    - Bid/Ask spreads (buying at ask, selling at bid)
    - Slippage based on configured model
    - Commission costs based on configured model
    """
    

    def __init__(
        self,
        data_handler,
        event_queue: queue.Queue,
        slippage_model: Optional[SlippageModel] = None,
        commission_model: Optional[CommissionModel] = None,
        max_execution_delay_ms: float = 500.0
    ):
        """
        Initialize the execution handler.
        
        Args:
            data_handler: Provides market data (prices, volatility)
            event_queue: Queue for publishing fill events
            slippage_model: Model for calculating price slippage
            commission_model: Model for calculating trade commissions
            max_execution_delay_ms: Maximum allowed delay between order and execution
        """
        from ..models.transaction_costs import TimeAwareVolatilitySlippageModel
        
        self.data_handler = data_handler
        self.event_queue = event_queue
        
        # Use default models if none provided
        self.slippage_model = slippage_model or TimeAwareVolatilitySlippageModel()
        self.commission_model = commission_model or OANDACommissionModel()
        
        # Validate and set execution constraints
        if max_execution_delay_ms <= 0:
            raise ValueError("Maximum execution delay must be positive")
        self.max_execution_delay_ms = max_execution_delay_ms
        
        # Initialize market data tracking
        self.last_market_timestamp = None

    def validate_execution_time(self, order_time: datetime) -> None:
        """
        Validates that an order's execution time is realistic and prevents look-ahead.
        
        Args:
            order_time: The timestamp of the order
            
        Raises:
            ValueError: If order timing violates execution constraints
        """
        current_time = datetime.now()
        
        # Check if order is from the future
        if order_time > current_time:
            raise ValueError("Order timestamp is in the future")
            
        # Check if order is too old
        delay_ms = (current_time - order_time).total_seconds() * 1000
        if delay_ms > self.max_execution_delay_ms:
            raise ValueError(f"Order execution delay ({delay_ms}ms) exceeds maximum allowed ({self.max_execution_delay_ms}ms)")
            
        # Ensure market data sequence
        if self.last_market_timestamp and order_time < self.last_market_timestamp:
            raise ValueError("Order timestamp precedes last known market data")

    def get_market_context(self, instrument: str, order_time: datetime) -> Dict[str, Any]:
        """
        Gather market data needed for transaction cost calculations.
        
        Args:
            instrument: The trading instrument symbol
            order_time: The timestamp of the order to prevent look-ahead
            
        Returns:
            Dictionary containing market data like volatility, bid, ask prices
        """
        # Get market data as of order time
        market_data = self.data_handler.get_market_data_at_time(instrument, order_time)
        
        # Update last known market timestamp
        self.last_market_timestamp = market_data.get('timestamp')
        
        return {
            'timestamp': market_data.get('timestamp'),
            'volatility': market_data.get('volatility'),
            'bid': market_data.get('bid'),
            'ask': market_data.get('ask'),
            'volume': market_data.get('volume'),
            'order_units': market_data.get('order_units')
        }

    def simulate_fill(self, order_event: OrderEvent) -> None:
        """
        Simulates the execution of an order with realistic constraints.
        
        This method:
        1. Validates execution timing
        2. Gets historical market data from order time
        3. Applies appropriate transaction costs
        4. Generates a fill event
        
        Args:
            order_event: The order to execute
            
        Raises:
            ValueError: If execution constraints are violated
        """
        # Validate execution timing
        self.validate_execution_time(order_event.timestamp)
        """
        Simulates the execution of an order, applying slippage and commission.
        
        This method:
        1. Gets current market prices (bid/ask)
        2. Calculates slippage based on market conditions
        3. Determines final fill price including spread and slippage
        4. Calculates commission
        5. Generates and publishes a fill event
        
        Args:
            order_event: The order to execute
        """
        # Get market context as of order time
        market_context = self.get_market_context(
            order_event.instrument,
            order_event.timestamp
        )
        
        # Add order details to market context for impact calculation
        market_context['order_units'] = abs(order_event.units)
        
        # Calculate slippage with time awareness
        slippage_amount = self.slippage_model.calculate_slippage(
            order_event.instrument,
            order_event.direction,
            market_context,
            order_event.timestamp
        )
        
        # Determine fill price based on direction and slippage
        if order_event.direction == 'BUY':
            # Buy at ask price plus slippage
            fill_price = market_context['ask'] + slippage_amount
        else:  # SELL
            # Sell at bid price minus slippage
            fill_price = market_context['bid'] - slippage_amount
            
        # Calculate commission
        commission = self.commission_model.calculate_commission(
            order_event.units,
            fill_price
        )
        
        # Create and publish fill event
        fill_event = FillEvent(
            timestamp=order_event.timestamp,
            instrument=order_event.instrument,
            units=order_event.units,
            direction=order_event.direction,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage_amount
        )
        
        # Add to event queue for processing
        self.event_queue.put(fill_event)

    def execute_order(self, order_event: OrderEvent) -> None:
        """
        Main entry point for order execution.
        
        Currently handles market orders only. Can be extended for limit orders,
        stop orders, etc.
        
        Args:
            order_event: The order to execute
        """
        if order_event.order_type != 'MARKET':
            raise ValueError(f"Unsupported order type: {order_event.order_type}")
            
        self.simulate_fill(order_event)