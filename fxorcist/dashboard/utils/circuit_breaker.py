"""
Circuit breaker implementation for resilient service calls.

Implements the circuit breaker pattern to prevent cascading failures
and allow graceful service recovery.
"""

import asyncio
import logging
from enum import Enum
from datetime import datetime, timedelta
from typing import Optional, Callable, Any
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreakerError(Exception):
    """Raised when circuit breaker prevents operation."""
    pass

class CircuitBreaker:
    """Circuit breaker implementation."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: int = 60,
        half_open_timeout: int = 30
    ):
        """Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name for logging
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Seconds to wait before attempting reset
            half_open_timeout: Seconds to wait in half-open state
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_timeout = half_open_timeout
        
        # State
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._reset_timer: Optional[asyncio.Task] = None
        
        logger.info(
            f"Circuit breaker {self.name} initialized with threshold={failure_threshold}, "
            f"reset_timeout={reset_timeout}, half_open_timeout={half_open_timeout}"
        )
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self._state == CircuitState.OPEN
    
    def _should_reset(self) -> bool:
        """Check if circuit should attempt reset.
        
        Returns:
            bool: True if enough time has passed since last failure
        """
        if not self._last_failure_time:
            return False
        
        elapsed = (datetime.utcnow() - self._last_failure_time).total_seconds()
        should_reset = elapsed >= self.reset_timeout
        
        if should_reset:
            logger.debug(
                f"Circuit breaker {self.name} ready for reset after "
                f"{elapsed:.1f}s (threshold: {self.reset_timeout}s)"
            )
        
        return should_reset
    
    async def _schedule_reset(self) -> None:
        """Schedule automatic reset attempt."""
        if self._reset_timer:
            self._reset_timer.cancel()
        
        try:
            await asyncio.sleep(self.reset_timeout)
            await self.attempt_reset()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in reset timer for {self.name}: {e}")
        finally:
            self._reset_timer = None
    
    async def attempt_reset(self) -> None:
        """Attempt to reset circuit to half-open state."""
        if self._state == CircuitState.OPEN and self._should_reset():
            logger.info(f"Circuit breaker {self.name} attempting reset")
            self._state = CircuitState.HALF_OPEN
            self._failure_count = 0
    
    def record_failure(self) -> None:
        """Record a failure and potentially open circuit."""
        self._failure_count += 1
        self._last_failure_time = datetime.utcnow()
        
        if self._state == CircuitState.HALF_OPEN:
            logger.warning(f"Circuit breaker {self.name} failed in half-open state")
            self._state = CircuitState.OPEN
            asyncio.create_task(self._schedule_reset())
        
        elif (
            self._state == CircuitState.CLOSED and 
            self._failure_count >= self.failure_threshold
        ):
            logger.warning(
                f"Circuit breaker {self.name} opened after "
                f"{self._failure_count} failures"
            )
            self._state = CircuitState.OPEN
            asyncio.create_task(self._schedule_reset())
    
    def record_success(self) -> None:
        """Record success and potentially close circuit."""
        prev_state = self._state
        prev_failures = self._failure_count
        
        self._failure_count = 0
        
        if self._state == CircuitState.HALF_OPEN:
            logger.info(
                f"Circuit breaker {self.name} closing after successful test "
                f"(previous failures: {prev_failures})"
            )
            self._state = CircuitState.CLOSED
            self._last_failure_time = None
            
            if self._reset_timer:
                self._reset_timer.cancel()
                self._reset_timer = None
        
        if prev_state != self._state:
            logger.info(
                f"Circuit breaker {self.name} state transition: "
                f"{prev_state.value} -> {self._state.value}"
            )
    
    @asynccontextmanager
    async def __aenter__(self):
        """Enter async context, checking circuit state."""
        if self._state == CircuitState.OPEN:
            if self._should_reset():
                await self.attempt_reset()
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker {self.name} is open"
                )
        
        try:
            yield self
        except Exception as e:
            self.record_failure()
            raise
        else:
            self.record_success()
    
    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any]
    ) -> None:
        """Exit async context."""
        pass