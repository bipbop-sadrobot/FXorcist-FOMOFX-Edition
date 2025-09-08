"""
Metrics collection for dashboard monitoring.

Implements Prometheus-style metrics for monitoring dashboard performance
and circuit breaker states.
"""

from typing import Dict, Optional
from dataclasses import dataclass, field
import time

@dataclass
class Counter:
    """Simple counter metric."""
    name: str
    description: str
    labels: Dict[str, str] = field(default_factory=dict)
    _value: float = 0.0
    
    def inc(self, amount: float = 1.0) -> None:
        """Increment counter."""
        self._value += amount
    
    def get(self) -> float:
        """Get current value."""
        return self._value

@dataclass
class Gauge:
    """Simple gauge metric."""
    name: str
    description: str
    labels: Dict[str, str] = field(default_factory=dict)
    _value: float = 0.0
    
    def set(self, value: float) -> None:
        """Set gauge value."""
        self._value = value
    
    def get(self) -> float:
        """Get current value."""
        return self._value

@dataclass
class Histogram:
    """Simple histogram metric."""
    name: str
    description: str
    buckets: list[float]
    labels: Dict[str, str] = field(default_factory=dict)
    _counts: Dict[float, int] = field(default_factory=dict)
    _sum: float = 0.0
    
    def observe(self, value: float) -> None:
        """Record observation."""
        self._sum += value
        for bucket in self.buckets:
            if value <= bucket:
                self._counts[bucket] = self._counts.get(bucket, 0) + 1
    
    def get_bucket_count(self, bucket: float) -> int:
        """Get count for bucket."""
        return self._counts.get(bucket, 0)
    
    def get_sum(self) -> float:
        """Get sum of observations."""
        return self._sum

# Circuit breaker metrics
CIRCUIT_BREAKER_STATE = Gauge(
    name="circuit_breaker_state",
    description="Current state of circuit breaker (0=open, 1=closed, 2=half-open)"
)

CIRCUIT_BREAKER_FAILURES = Counter(
    name="circuit_breaker_failures_total",
    description="Total number of circuit breaker failures"
)

CIRCUIT_BREAKER_RESETS = Counter(
    name="circuit_breaker_resets_total",
    description="Total number of circuit breaker reset attempts"
)

CIRCUIT_BREAKER_SUCCESSES = Counter(
    name="circuit_breaker_successes_total",
    description="Total number of successful operations through circuit breaker"
)

# Dashboard metrics
DASHBOARD_REQUEST_DURATION = Histogram(
    name="dashboard_request_duration_seconds",
    description="Dashboard request duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

DASHBOARD_REQUESTS_TOTAL = Counter(
    name="dashboard_requests_total",
    description="Total number of dashboard requests"
)

DASHBOARD_ERRORS_TOTAL = Counter(
    name="dashboard_errors_total",
    description="Total number of dashboard errors"
)

DASHBOARD_ACTIVE_CONNECTIONS = Gauge(
    name="dashboard_active_connections",
    description="Number of active WebSocket connections"
)

DASHBOARD_CACHE_HITS = Counter(
    name="dashboard_cache_hits_total",
    description="Total number of cache hits"
)

DASHBOARD_CACHE_MISSES = Counter(
    name="dashboard_cache_misses_total",
    description="Total number of cache misses"
)

class MetricsTimer:
    """Context manager for timing operations."""
    
    def __init__(self, histogram: Histogram):
        self.histogram = histogram
        self.start_time: Optional[float] = None
    
    def __enter__(self) -> 'MetricsTimer':
        self.start_time = time.monotonic()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start_time is not None:
            duration = time.monotonic() - self.start_time
            self.histogram.observe(duration)

def get_metrics() -> Dict[str, float]:
    """Get all current metric values.
    
    Returns:
        Dict mapping metric names to current values
    """
    return {
        "circuit_breaker_state": CIRCUIT_BREAKER_STATE.get(),
        "circuit_breaker_failures": CIRCUIT_BREAKER_FAILURES.get(),
        "circuit_breaker_resets": CIRCUIT_BREAKER_RESETS.get(),
        "circuit_breaker_successes": CIRCUIT_BREAKER_SUCCESSES.get(),
        "dashboard_requests": DASHBOARD_REQUESTS_TOTAL.get(),
        "dashboard_errors": DASHBOARD_ERRORS_TOTAL.get(),
        "dashboard_active_connections": DASHBOARD_ACTIVE_CONNECTIONS.get(),
        "dashboard_cache_hits": DASHBOARD_CACHE_HITS.get(),
        "dashboard_cache_misses": DASHBOARD_CACHE_MISSES.get()
    }