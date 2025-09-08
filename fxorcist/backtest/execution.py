"""
Execution model for simulating trade order execution with slippage and commission.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class Order:
    """
    Represents a trading order with key execution details.
    
    Attributes:
        symbol: Trading symbol
        side: Order side ('buy' or 'sell')
        size: Order quantity
        order_type: Type of order (market, limit, etc.)
        price: Specified price for the order
    """
    symbol: str
    side: str  # 'buy' or 'sell'
    size: float
    order_type: str = 'market'
    price: Optional[float] = None

@dataclass
class Fill:
    """
    Represents the execution details of a trade order.
    
    Attributes:
        order: Original order that was executed
        price: Actual execution price
        size: Executed quantity
        timestamp: Execution timestamp
        commission: Transaction commission
        slippage: Price difference from expected
    """
    order: Order
    price: float
    size: float
    timestamp: datetime
    commission: float
    slippage: float

class ExecutionModel:
    """
    Base class for trade execution models.
    
    Provides an interface for simulating order execution with 
    configurable slippage and commission calculations.
    """
    def __init__(
        self, 
        commission_pct: float = 0.00002,  # Default 2 basis points
        slippage_model: str = 'simple',
        latency_ms: int = 100
    ):
        """
        Initialize the execution model.
        
        Args:
            commission_pct: Percentage commission per trade
            slippage_model: Type of slippage model to use
            latency_ms: Simulated execution latency
        """
        self.commission_pct = commission_pct
        self.slippage_model = slippage_model
        self.latency_ms = latency_ms
    
    def execute(
        self, 
        orders: List[Order], 
        market_snapshot: Dict[str, Any], 
        timestamp: datetime
    ) -> List[Fill]:
        """
        Execute a list of orders based on market conditions.
        
        Args:
            orders: List of orders to execute
            market_snapshot: Current market state
            timestamp: Execution timestamp
        
        Returns:
            List of order fills
        """
        fills = []
        for order in orders:
            fill = self._execute_single_order(order, market_snapshot, timestamp)
            if fill:
                fills.append(fill)
        return fills
    
    def _execute_single_order(
        self, 
        order: Order, 
        market_snapshot: Dict[str, Any], 
        timestamp: datetime
    ) -> Optional[Fill]:
        """
        Execute a single order with slippage and commission calculation.
        
        Args:
            order: Order to execute
            market_snapshot: Current market state
            timestamp: Execution timestamp
        
        Returns:
            Filled order or None if execution is not possible
        """
        # Retrieve market price
        mid_price = market_snapshot.get(order.symbol, {}).get('mid')
        if mid_price is None:
            return None
        
        # Apply slippage based on order side
        sign = 1 if order.side == 'buy' else -1
        slippage = self._calculate_slippage(order, mid_price)
        exec_price = mid_price + sign * slippage
        
        # Calculate commission
        commission = abs(order.size * exec_price * self.commission_pct)
        
        return Fill(
            order=order,
            price=exec_price,
            size=order.size,
            timestamp=timestamp,
            commission=commission,
            slippage=slippage
        )
    
    def _calculate_slippage(self, order: Order, mid_price: float) -> float:
        """
        Calculate slippage based on the configured slippage model.
        
        Args:
            order: Order being executed
            mid_price: Current market mid price
        
        Returns:
            Slippage amount
        """
        if self.slippage_model == 'simple':
            # Simple fixed slippage model
            return 0.0001  # 1 pip
        elif self.slippage_model == 'proportional':
            # Proportional slippage based on order size
            return 0.0001 * (order.size / 1000)
        else:
            # Default to minimal slippage
            return 0.00005

class SimpleSlippageModel(ExecutionModel):
    """
    A simple execution model with configurable slippage and commission.
    """
    def __init__(
        self, 
        commission_pct: float = 0.00002,
        slippage_ticks: float = 0.0001,
        latency_ms: int = 100
    ):
        """
        Initialize the simple slippage model.
        
        Args:
            commission_pct: Percentage commission per trade
            slippage_ticks: Fixed slippage in ticks
            latency_ms: Simulated execution latency
        """
        super().__init__(
            commission_pct=commission_pct, 
            slippage_model='simple', 
            latency_ms=latency_ms
        )
        self.slippage_ticks = slippage_ticks
    
    def _calculate_slippage(self, order: Order, mid_price: float) -> float:
        """
        Override slippage calculation with a fixed tick-based model.
        
        Args:
            order: Order being executed
            mid_price: Current market mid price
        
        Returns:
            Fixed slippage amount
        """
        return self.slippage_ticks

class ProportionalSlippageModel(ExecutionModel):
    """
    An execution model with slippage proportional to order size.
    """
    def __init__(
        self, 
        commission_pct: float = 0.00002,
        base_slippage: float = 0.0001,
        slippage_factor: float = 0.00001,
        latency_ms: int = 100
    ):
        """
        Initialize the proportional slippage model.
        
        Args:
            commission_pct: Percentage commission per trade
            base_slippage: Base slippage amount
            slippage_factor: Multiplier for order size-based slippage
            latency_ms: Simulated execution latency
        """
        super().__init__(
            commission_pct=commission_pct, 
            slippage_model='proportional', 
            latency_ms=latency_ms
        )
        self.base_slippage = base_slippage
        self.slippage_factor = slippage_factor
    
    def _calculate_slippage(self, order: Order, mid_price: float) -> float:
        """
        Calculate slippage proportional to order size.
        
        Args:
            order: Order being executed
            mid_price: Current market mid price
        
        Returns:
            Proportional slippage amount
        """
        return self.base_slippage + (self.slippage_factor * order.size)