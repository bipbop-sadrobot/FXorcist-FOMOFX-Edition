"""
Event bus implementation for event-driven market data processing.
"""
from typing import Iterable, List, Optional, Callable
from datetime import datetime
from .types import Event
from pydantic import ValidationError

class EventBusError(Exception):
    """Base exception for event bus errors."""
    pass

class InvalidEventError(EventBusError):
    """Raised when an event fails validation."""
    pass

class InMemoryEventBus:
    """
    In-memory event bus for storing and replaying events.
    
    Provides chronological event storage and retrieval with optional 
    timestamp-based filtering and advanced event management.
    """
    def __init__(self, events: Optional[List[Event]] = None):
        """
        Initialize the event bus.
        
        Args:
            events: Optional initial list of events to populate the bus
        
        Raises:
            InvalidEventError: If any event fails validation
        """
        try:
            # Validate and sort events
            validated_events = [Event(**event.dict()) for event in (events or [])]
            self._events = sorted(validated_events, key=lambda e: e.timestamp)
        except ValidationError as e:
            raise InvalidEventError(f"Invalid event in event bus: {e}")
    
    def append(self, event: Event) -> None:
        """
        Append an event to the bus and maintain chronological order.
        
        Args:
            event: Event to be added to the bus
        
        Raises:
            InvalidEventError: If event validation fails
        """
        try:
            validated_event = Event(**event.dict())
            self._events.append(validated_event)
            self._events.sort(key=lambda e: e.timestamp)
        except ValidationError as e:
            raise InvalidEventError(f"Invalid event: {e}")
    
    def replay(
        self, 
        start_ts: Optional[datetime] = None, 
        end_ts: Optional[datetime] = None,
        symbol: Optional[str] = None,
        event_type: Optional[str] = None
    ) -> Iterable[Event]:
        """
        Replay events with advanced filtering capabilities.
        
        Args:
            start_ts: Optional start timestamp for event replay
            end_ts: Optional end timestamp for event replay
            symbol: Optional symbol to filter events
            event_type: Optional event type to filter events
        
        Yields:
            Events matching the specified criteria
        
        Raises:
            EventBusError: If timestamp filtering is invalid
        """
        if start_ts and end_ts and start_ts > end_ts:
            raise EventBusError("Start timestamp cannot be after end timestamp")
        
        for event in self._events:
            # Check timestamp range
            if start_ts and event.timestamp < start_ts:
                continue
            if end_ts and event.timestamp > end_ts:
                break
            
            # Check symbol filter
            if symbol and event.payload.get('symbol') != symbol:
                continue
            
            # Check event type filter
            if event_type and event.type != event_type:
                continue
            
            yield event
    
    def filter_events(
        self, 
        predicate: Optional[Callable[[Event], bool]] = None
    ) -> List[Event]:
        """
        Filter events based on a custom predicate function.
        
        Args:
            predicate: Optional function to filter events
        
        Returns:
            List of filtered events
        """
        if predicate is None:
            return self._events.copy()
        
        return [event for event in self._events if predicate(event)]
    
    def clear(self) -> None:
        """
        Clear all events from the bus.
        """
        self._events.clear()
    
    def __len__(self) -> int:
        """
        Get the number of events in the bus.
        
        Returns:
            Total number of events
        """
        return len(self._events)

# Stub for future NATS JetStream adapter
class NatsJetStreamAdapter:
    """
    Placeholder for NATS JetStream event bus adapter.
    
    This will be implemented in a future version to provide 
    durable event replay using NATS JetStream.
    """
    def __init__(self, connection, stream_name):
        """
        Initialize NATS JetStream adapter.
        
        Args:
            connection: NATS connection
            stream_name: Name of the NATS stream
        """
        self._connection = connection
        self._stream_name = stream_name
    
    async def replay(self, start_ts=None, end_ts=None):
        """
        Replay events from NATS JetStream.
        
        Args:
            start_ts: Optional start timestamp
            end_ts: Optional end timestamp
        
        Yields:
            Events from the stream
        """
        # Placeholder for future implementation
        raise NotImplementedError("NATS JetStream adapter not yet implemented")