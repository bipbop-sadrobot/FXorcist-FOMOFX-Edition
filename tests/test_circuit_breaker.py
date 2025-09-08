"""
Unit tests for the circuit breaker implementation.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from fxorcist.dashboard.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerError
)

@pytest.fixture
def circuit_breaker():
    """Create a test circuit breaker."""
    return CircuitBreaker(
        name="test_breaker",
        failure_threshold=2,
        reset_timeout=1,
        half_open_timeout=1
    )

@pytest.mark.asyncio
async def test_initial_state(circuit_breaker):
    """Test initial circuit breaker state."""
    assert circuit_breaker.state == CircuitState.CLOSED
    assert not circuit_breaker.is_open
    assert circuit_breaker._failure_count == 0
    assert circuit_breaker._last_failure_time is None

@pytest.mark.asyncio
async def test_failure_threshold(circuit_breaker):
    """Test circuit opens after failure threshold."""
    # First failure
    circuit_breaker.record_failure()
    assert circuit_breaker.state == CircuitState.CLOSED
    
    # Second failure opens circuit
    circuit_breaker.record_failure()
    assert circuit_breaker.state == CircuitState.OPEN
    assert circuit_breaker.is_open

@pytest.mark.asyncio
async def test_success_resets_failures(circuit_breaker):
    """Test success resets failure count."""
    circuit_breaker.record_failure()
    assert circuit_breaker._failure_count == 1
    
    circuit_breaker.record_success()
    assert circuit_breaker._failure_count == 0

@pytest.mark.asyncio
async def test_reset_after_timeout(circuit_breaker):
    """Test circuit attempts reset after timeout."""
    # Open circuit
    circuit_breaker.record_failure()
    circuit_breaker.record_failure()
    assert circuit_breaker.state == CircuitState.OPEN
    
    # Wait for reset timeout
    await asyncio.sleep(1.1)
    
    # Next operation should move to half-open
    async with circuit_breaker:
        assert circuit_breaker.state == CircuitState.HALF_OPEN

@pytest.mark.asyncio
async def test_half_open_success(circuit_breaker):
    """Test successful operation in half-open state closes circuit."""
    # Set up half-open state
    circuit_breaker._state = CircuitState.HALF_OPEN
    
    async with circuit_breaker:
        pass  # Successful operation
    
    assert circuit_breaker.state == CircuitState.CLOSED

@pytest.mark.asyncio
async def test_half_open_failure(circuit_breaker):
    """Test failure in half-open state reopens circuit."""
    # Set up half-open state
    circuit_breaker._state = CircuitState.HALF_OPEN
    
    with pytest.raises(Exception):
        async with circuit_breaker:
            raise Exception("Test failure")
    
    assert circuit_breaker.state == CircuitState.OPEN

@pytest.mark.asyncio
async def test_open_circuit_raises_error(circuit_breaker):
    """Test open circuit raises error."""
    # Open circuit
    circuit_breaker._state = CircuitState.OPEN
    circuit_breaker._last_failure_time = datetime.utcnow()
    
    with pytest.raises(CircuitBreakerError):
        async with circuit_breaker:
            pass

@pytest.mark.asyncio
async def test_context_manager_success(circuit_breaker):
    """Test successful operation through context manager."""
    async with circuit_breaker:
        pass  # Successful operation
    
    assert circuit_breaker.state == CircuitState.CLOSED
    assert circuit_breaker._failure_count == 0

@pytest.mark.asyncio
async def test_context_manager_failure(circuit_breaker):
    """Test failed operation through context manager."""
    with pytest.raises(Exception):
        async with circuit_breaker:
            raise Exception("Test failure")
    
    assert circuit_breaker._failure_count == 1

@pytest.mark.asyncio
async def test_reset_timer_cancellation(circuit_breaker):
    """Test reset timer is properly cancelled."""
    # Open circuit and start reset timer
    circuit_breaker.record_failure()
    circuit_breaker.record_failure()
    assert circuit_breaker._reset_timer is not None
    
    # Successful operation in half-open state
    circuit_breaker._state = CircuitState.HALF_OPEN
    circuit_breaker.record_success()
    
    # Timer should be cancelled
    assert circuit_breaker._reset_timer is None

@pytest.mark.asyncio
async def test_state_transition_logging(circuit_breaker, caplog):
    """Test state transitions are properly logged."""
    # Open circuit
    circuit_breaker.record_failure()
    circuit_breaker.record_failure()
    assert "opened after 2 failures" in caplog.text
    
    # Reset to half-open
    circuit_breaker._state = CircuitState.HALF_OPEN
    
    # Close circuit
    circuit_breaker.record_success()
    assert "state transition: half_open -> closed" in caplog.text