"""
Event type definitions for the FXorcist event-driven system.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
import uuid

@dataclass(frozen=True)
class Event:
    """
    Immutable event representation with strict typing.
    
    Attributes:
        id: Unique identifier for the event
        timestamp: Precise timestamp of the event
        type: Event type (e.g., 'tick', 'bar', 'order')
        payload: Event-specific data payload
        symbol: Trading symbol associated with the event
        metadata: Optional additional event metadata
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime
    type: str
    payload: Dict[str, Any]
    symbol: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

def create_tick_event(
    timestamp: datetime, 
    symbol: str, 
    bid: float, 
    ask: float, 
    **kwargs
) -> Event:
    """
    Create a tick event for market price data.
    
    Args:
        timestamp: Timestamp of the tick
        symbol: Trading symbol
        bid: Bid price
        ask: Ask price
        **kwargs: Additional metadata
    
    Returns:
        Tick event
    """
    return Event(
        timestamp=timestamp,
        type='tick',
        payload={
            'bid': bid,
            'ask': ask
        },
        symbol=symbol,
        metadata=kwargs
    )

def create_bar_event(
    timestamp: datetime, 
    symbol: str, 
    open_price: float, 
    high: float, 
    low: float, 
    close: float, 
    volume: Optional[float] = None,
    **kwargs
) -> Event:
    """
    Create a bar event for OHLC market data.
    
    Args:
        timestamp: Timestamp of the bar
        symbol: Trading symbol
        open_price: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Optional trading volume
        **kwargs: Additional metadata
    
    Returns:
        Bar event
    """
    payload = {
        'open': open_price,
        'high': high,
        'low': low,
        'close': close
    }
    
    if volume is not None:
        payload['volume'] = volume
    
    return Event(
        timestamp=timestamp,
        type='bar',
        payload=payload,
        symbol=symbol,
        metadata=kwargs
    )

def create_order_event(
    timestamp: datetime, 
    symbol: str, 
    side: str, 
    size: float, 
    price: float,
    **kwargs
) -> Event:
    """
    Create an order event for trade execution.
    
    Args:
        timestamp: Timestamp of the order
        symbol: Trading symbol
        side: Order side ('buy' or 'sell')
        size: Order size
        price: Execution price
        **kwargs: Additional metadata
    
    Returns:
        Order event
    """
    return Event(
        timestamp=timestamp,
        type='order',
        payload={
            'side': side,
            'size': size,
            'price': price
        },
        symbol=symbol,
        metadata=kwargs
    )