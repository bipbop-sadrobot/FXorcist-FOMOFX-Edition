"""
Event bus implementation with timestamp validation for anti-bias protection.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Iterator, Optional
from enum import Enum

class EventType(Enum):
    """Types of events in the system."""
    TICK = "tick"
    BAR = "bar"
    ORDER = "order"
    FILL = "fill"
    SIGNAL = "signal"
    PORTFOLIO = "portfolio"

@dataclass
class Event:
    """Base event class with timestamp validation."""
    timestamp: datetime
    type: EventType
    symbol: str
    data: Dict[str, Any]

    def __post_init__(self):
        """Validate event timestamp."""
        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be a datetime object")
        if self.timestamp > datetime.now():
            raise ValueError("timestamp cannot be in the future")

class LookAheadError(Exception):
    """Raised when attempting to access future data."""
    pass

class EventBus:
    """
    Event bus with anti-bias protection.
    
    Features:
    - Chronological event ordering
    - Timestamp validation
    - Look-ahead prevention
    - Event filtering by type/symbol
    """
    def __init__(self):
        self._events: List[Event] = []
        self._current_time: Optional[datetime] = None

    def append(self, event: Event) -> None:
        """
        Add an event to the bus.
        
        Args:
            event: Event to add
            
        Raises:
            ValueError: If event timestamp is invalid
        """
        # Validate event
        if len(self._events) > 0:
            last_event = self._events[-1]
            if event.timestamp < last_event.timestamp:
                # Allow same timestamp but maintain order
                self._events.append(event)
                self._events.sort(key=lambda e: e.timestamp)
            else:
                self._events.append(event)
        else:
            self._events.append(event)

    def replay(
        self,
        start_ts: datetime,
        end_ts: datetime,
        event_types: Optional[List[EventType]] = None,
        symbols: Optional[List[str]] = None,
    ) -> Iterator[Event]:
        """
        Replay events in chronological order with filtering.
        
        Args:
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive) 
            event_types: Optional list of event types to include
            symbols: Optional list of symbols to include
            
        Yields:
            Event objects in chronological order
        """
        for event in self._events:
            # Time window check
            if not (start_ts <= event.timestamp <= end_ts):
                continue
                
            # Filter checks
            if event_types and event.type not in event_types:
                continue
            if symbols and event.symbol not in symbols:
                continue
                
            # Update current time for look-ahead prevention
            self._current_time = event.timestamp
            yield event

    def get_current_time(self) -> Optional[datetime]:
        """Get the timestamp of the most recently replayed event."""
        return self._current_time

    def validate_timestamp(self, ts: datetime) -> None:
        """
        Validate a timestamp against the current replay time.
        
        Args:
            ts: Timestamp to validate
            
        Raises:
            LookAheadError: If timestamp is after current replay time
        """
        if self._current_time and ts > self._current_time:
            raise LookAheadError(
                f"Attempted to access data at {ts} which is after "
                f"current replay time {self._current_time}"
            )

    def clear(self) -> None:
        """Clear all events and reset current time."""
        self._events.clear()
        self._current_time = None

    def __len__(self) -> int:
        return len(self._events)

# Factory functions for common event types
def create_tick_event(
    timestamp: datetime,
    symbol: str,
    bid: float,
    ask: float,
    volume: Optional[float] = None,
) -> Event:
    """Create a tick event."""
    return Event(
        timestamp=timestamp,
        type=EventType.TICK,
        symbol=symbol,
        data={
            "bid": bid,
            "ask": ask,
            "volume": volume,
        }
    )

def create_bar_event(
    timestamp: datetime,
    symbol: str,
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: Optional[float] = None,
) -> Event:
    """Create a bar event."""
    return Event(
        timestamp=timestamp,
        type=EventType.BAR,
        symbol=symbol,
        data={
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

def create_signal_event(
    timestamp: datetime,
    symbol: str,
    signal_type: str,
    strength: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> Event:
    """Create a signal event."""
    return Event(
        timestamp=timestamp,
        type=EventType.SIGNAL,
        symbol=symbol,
        data={
            "signal_type": signal_type,
            "strength": strength,
            **(metadata or {}),
        }
    )