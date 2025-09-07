"""
Core event system implementation for FXorcist trading platform.
Defines event types and base event classes for inter-module communication.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any
from enum import Enum
from decimal import Decimal
import uuid

class EventType(Enum):
    """Enumeration of supported event types."""
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    RISK = "RISK"
    PORTFOLIO = "PORTFOLIO"
    SYSTEM = "SYSTEM"

@dataclass(frozen=True)
class Event:
    """Immutable event object for message passing between modules."""
    type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    event_id: str = None
    
    def __post_init__(self):
        if self.event_id is None:
            object.__setattr__(self, 'event_id', str(uuid.uuid4()))

@dataclass(frozen=True)
class MarketEvent(Event):
    """Market data update event."""
    def __init__(self, instrument: str, bid: Decimal, ask: Decimal, 
                 timestamp: datetime = None, volume: int = 0):
        data = {
            'instrument': instrument,
            'bid': float(bid),
            'ask': float(ask),
            'volume': volume,
            'spread': float(ask - bid)
        }
        super().__init__(
            type=EventType.MARKET,
            timestamp=timestamp or datetime.now(timezone.utc),
            data=data
        )

@dataclass(frozen=True)
class SignalEvent(Event):
    """Trading signal event."""
    def __init__(self, instrument: str, direction: str, strength: float,
                 strategy: str, timestamp: datetime = None, **kwargs):
        data = {
            'instrument': instrument,
            'direction': direction,  # LONG, SHORT, CLOSE
            'strength': strength,
            'strategy': strategy,
            **kwargs
        }
        super().__init__(
            type=EventType.SIGNAL,
            timestamp=timestamp or datetime.now(timezone.utc),
            data=data
        )

@dataclass(frozen=True)
class OrderEvent(Event):
    """Order placement event."""
    def __init__(self, instrument: str, order_type: str, direction: str,
                 units: int, timestamp: datetime = None, **kwargs):
        data = {
            'instrument': instrument,
            'order_type': order_type,  # MARKET, LIMIT, STOP
            'direction': direction,    # BUY, SELL
            'units': units,
            **kwargs
        }
        super().__init__(
            type=EventType.ORDER,
            timestamp=timestamp or datetime.now(timezone.utc),
            data=data
        )

@dataclass(frozen=True)
class FillEvent(Event):
    """Order fill confirmation event."""
    def __init__(self, instrument: str, direction: str, units: int,
                 fill_price: Decimal, timestamp: datetime = None, **kwargs):
        data = {
            'instrument': instrument,
            'direction': direction,
            'units': units,
            'fill_price': float(fill_price),
            **kwargs
        }
        super().__init__(
            type=EventType.FILL,
            timestamp=timestamp or datetime.now(timezone.utc),
            data=data
        )