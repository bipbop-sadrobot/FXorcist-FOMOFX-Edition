"""
Tests for event bus system with anti-bias protection.
"""
from datetime import datetime, timedelta
import pytest

from fxorcist.events.event_bus import (
    Event,
    EventType,
    EventBus,
    LookAheadError,
    create_tick_event,
    create_bar_event,
)

# Test fixtures
@pytest.fixture
def event_bus():
    """Create a clean event bus for testing."""
    return EventBus()

@pytest.fixture
def sample_events():
    """Create a sequence of test events."""
    now = datetime.now()
    return [
        create_tick_event(
            timestamp=now,
            symbol="EURUSD",
            bid=1.1000,
            ask=1.1001,
        ),
        create_tick_event(
            timestamp=now + timedelta(seconds=1),
            symbol="EURUSD",
            bid=1.1001,
            ask=1.1002,
        ),
        create_bar_event(
            timestamp=now + timedelta(minutes=1),
            symbol="EURUSD",
            open_price=1.1000,
            high=1.1002,
            low=1.1000,
            close=1.1001,
        ),
    ]

# Test event creation and validation
def test_event_timestamp_validation():
    """Test that events validate their timestamps."""
    future_time = datetime.now() + timedelta(days=1)
    
    with pytest.raises(ValueError, match="cannot be in the future"):
        Event(
            timestamp=future_time,
            type=EventType.TICK,
            symbol="EURUSD",
            data={"bid": 1.0, "ask": 1.0001},
        )

def test_event_factory_functions():
    """Test event factory functions."""
    now = datetime.now()
    
    tick = create_tick_event(
        timestamp=now,
        symbol="EURUSD",
        bid=1.1000,
        ask=1.1001,
        volume=1000,
    )
    assert tick.type == EventType.TICK
    assert tick.data["bid"] == 1.1000
    assert tick.data["ask"] == 1.1001
    
    bar = create_bar_event(
        timestamp=now,
        symbol="EURUSD",
        open_price=1.1000,
        high=1.1002,
        low=1.0999,
        close=1.1001,
    )
    assert bar.type == EventType.BAR
    assert bar.data["open"] == 1.1000
    assert bar.data["close"] == 1.1001

# Test event bus operations
def test_event_chronological_ordering(event_bus, sample_events):
    """Test that events are stored and replayed in chronological order."""
    # Add events in reverse order
    for event in reversed(sample_events):
        event_bus.append(event)
    
    # Verify chronological replay
    events = list(event_bus.replay(
        sample_events[0].timestamp,
        sample_events[-1].timestamp,
    ))
    
    assert len(events) == len(sample_events)
    for i in range(len(events) - 1):
        assert events[i].timestamp <= events[i + 1].timestamp

def test_event_filtering(event_bus, sample_events):
    """Test event filtering by type and symbol."""
    for event in sample_events:
        event_bus.append(event)
    
    # Filter by type
    tick_events = list(event_bus.replay(
        sample_events[0].timestamp,
        sample_events[-1].timestamp,
        event_types=[EventType.TICK],
    ))
    assert len(tick_events) == 2
    assert all(e.type == EventType.TICK for e in tick_events)
    
    # Filter by symbol
    eurusd_events = list(event_bus.replay(
        sample_events[0].timestamp,
        sample_events[-1].timestamp,
        symbols=["EURUSD"],
    ))
    assert len(eurusd_events) == len(sample_events)

# Test anti-bias protection
def test_look_ahead_prevention(event_bus, sample_events):
    """Test that look-ahead access is prevented."""
    for event in sample_events:
        event_bus.append(event)
    
    # Try to access future data
    with pytest.raises(LookAheadError):
        for event in event_bus.replay(
            sample_events[0].timestamp,
            sample_events[-1].timestamp,
        ):
            # Attempt to validate a future timestamp
            event_bus.validate_timestamp(
                event.timestamp + timedelta(seconds=1)
            )

def test_event_bus_state(event_bus, sample_events):
    """Test event bus state management."""
    # Initial state
    assert len(event_bus) == 0
    assert event_bus.get_current_time() is None
    
    # Add events
    for event in sample_events:
        event_bus.append(event)
    assert len(event_bus) == len(sample_events)
    
    # Replay updates current time
    for event in event_bus.replay(
        sample_events[0].timestamp,
        sample_events[-1].timestamp,
    ):
        assert event_bus.get_current_time() == event.timestamp
    
    # Clear state
    event_bus.clear()
    assert len(event_bus) == 0
    assert event_bus.get_current_time() is None

def test_duplicate_timestamps(event_bus):
    """Test handling of events with identical timestamps."""
    now = datetime.now()
    events = [
        create_tick_event(
            timestamp=now,
            symbol="EURUSD",
            bid=1.1000,
            ask=1.1001,
        ),
        create_tick_event(
            timestamp=now,  # Same timestamp
            symbol="GBPUSD",
            bid=1.2500,
            ask=1.2501,
        ),
    ]
    
    for event in events:
        event_bus.append(event)
    
    # Both events should be stored and retrievable
    replayed = list(event_bus.replay(now, now))
    assert len(replayed) == 2
    assert all(e.timestamp == now for e in replayed)

def test_invalid_replay_window(event_bus, sample_events):
    """Test replay with invalid time window."""
    for event in sample_events:
        event_bus.append(event)
    
    # Start after end
    events = list(event_bus.replay(
        sample_events[-1].timestamp,
        sample_events[0].timestamp,
    ))
    assert len(events) == 0
    
    # Window before all events
    early_window = sample_events[0].timestamp - timedelta(hours=1)
    events = list(event_bus.replay(
        early_window,
        early_window + timedelta(minutes=1),
    ))
    assert len(events) == 0