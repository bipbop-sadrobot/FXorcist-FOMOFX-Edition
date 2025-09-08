"""
Event bus implementation for event-driven market data processing.
"""
from typing import Iterable, List, Optional
from datetime import datetime
from .types import Event

class InMemoryEventBus:
    """
    In-memory event bus for storing and replaying events.
    
    Provides chronological event storage and retrieval with optional 
    timestamp-based filtering.
    """
    def __init__(self, events: Optional[List[Event]] = None):
        """
        Initialize the event bus.
        
        Args:
            events: Optional initial list of events to populate the bus
        """
        self._events = sorted(events or [], key=lambda e: e.timestamp)
    
    def append(self, event: Event) -> None:
        """
        Append an event to the bus and maintain chronological order.
        
        Args:
            event: Event to be added to the bus
        """
        self._events.append(event)
        self._events.sort(key=lambda e: e.timestamp)
    
    def replay(
        self, 
        start_ts: Optional[datetime] = None, 
        end_ts: Optional[datetime] = None,
        symbol: Optional[str] = None
    ) -> Iterable[Event]:
        """
        Replay events within the specified time range and optional symbol filter.
        
        Args:
            start_ts: Optional start timestamp for event replay
            end_ts: Optional end timestamp for event replay
            symbol: Optional symbol to filter events
        
        Yields:
            Events matching the specified criteria
        """
        for event in self._events:
            # Check timestamp range
            if start_ts and event.timestamp < start_ts:
                continue
            if end_ts and event.timestamp > end_ts:
                break
            
            # Check symbol filter
            if symbol and event.symbol != symbol:
                continue
            
            yield event
    
    def get_events(
        self, 
        event_type: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> List[Event]:
        """
        Retrieve events based on optional type and symbol filters.
        
        Args:
            event_type: Optional event type to filter
            symbol: Optional symbol to filter
        
        Returns:
            List of filtered events
        """
        return [
            event for event in self._events
            if (not event_type or event.type == event_type) and
               (not symbol or event.symbol == symbol)
        ]
    
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