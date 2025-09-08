"""
Middleware components for the dashboard API.

Implements rate limiting, request logging, and error handling middleware.
"""

import time
from typing import Callable, Dict, Optional
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp
import redis
import logging
import json
from datetime import datetime
import asyncio
from dataclasses import dataclass
from functools import partial

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    burst_limit: int = 100
    redis_url: str = "redis://localhost:6379/0"
    enabled: bool = True

class RateLimiter:
    """Redis-based rate limiter implementation."""
    
    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter with configuration."""
        self.config = config
        self.redis_client = redis.from_url(config.redis_url)
        self.window_size = 60  # 1 minute window
    
    async def is_rate_limited(self, key: str) -> bool:
        """Check if request should be rate limited."""
        if not self.config.enabled:
            return False
        
        try:
            pipe = self.redis_client.pipeline()
            now = int(time.time())
            window_key = f"{key}:{now // self.window_size}"
            
            # Clean old entries
            old_window_key = f"{key}:{(now // self.window_size) - 1}"
            pipe.delete(old_window_key)
            
            # Increment counter
            pipe.incr(window_key)
            pipe.expire(window_key, self.window_size * 2)
            
            # Get current count
            result = pipe.execute()
            current_count = result[1]
            
            return current_count > self.config.requests_per_minute
            
        except redis.RedisError as e:
            logger.error(f"Redis error in rate limiter: {e}")
            return False  # Fail open on Redis errors

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests."""
    
    def __init__(
        self,
        app: ASGIApp,
        config: Optional[RateLimitConfig] = None
    ):
        """Initialize rate limit middleware."""
        super().__init__(app)
        self.config = config or RateLimitConfig()
        self.limiter = RateLimiter(self.config)
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request with rate limiting."""
        # Skip rate limiting for certain paths
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)
        
        # Get client identifier (IP or user ID if authenticated)
        client_id = request.client.host
        if "user" in request.session:
            client_id = request.session["user"].username
        
        # Check rate limit
        if await self.limiter.is_rate_limited(client_id):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too many requests",
                    "detail": "Rate limit exceeded",
                    "retry_after": self.config.window_size
                }
            )
        
        return await call_next(request)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging request/response details."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request with logging."""
        start_time = time.time()
        
        # Extract request details
        request_id = request.headers.get("X-Request-ID", "")
        method = request.method
        url = str(request.url)
        client_ip = request.client.host
        
        # Log request
        logger.info(
            f"Request started: {method} {url} "
            f"(ID: {request_id}, IP: {client_ip})"
        )
        
        try:
            response = await call_next(request)
            
            # Calculate request duration
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                f"Request completed: {method} {url} "
                f"(ID: {request_id}, Status: {response.status_code}, "
                f"Duration: {duration:.3f}s)"
            )
            
            # Add timing header
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            return response
            
        except Exception as e:
            logger.error(
                f"Request failed: {method} {url} "
                f"(ID: {request_id}, Error: {str(e)})",
                exc_info=True
            )
            raise

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for consistent error handling."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request with error handling."""
        try:
            return await call_next(request)
            
        except Exception as e:
            logger.error(f"Unhandled error: {str(e)}", exc_info=True)
            
            # Convert exception to API error response
            status_code = getattr(e, "status_code", 500)
            error_detail = str(e)
            
            if status_code == 500:
                error_detail = "Internal server error"
            
            return JSONResponse(
                status_code=status_code,
                content={
                    "error": error_detail,
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": str(request.url),
                    "request_id": request.headers.get("X-Request-ID", "")
                }
            )

class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for monitoring API performance."""
    
    def __init__(
        self,
        app: ASGIApp,
        redis_url: str = "redis://localhost:6379/0"
    ):
        """Initialize performance monitoring."""
        super().__init__(app)
        self.redis_client = redis.from_url(redis_url)
        self.metrics_key = "api:metrics"
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request with performance monitoring."""
        start_time = time.time()
        
        response = await call_next(request)
        
        # Calculate metrics
        duration = time.time() - start_time
        endpoint = f"{request.method}:{request.url.path}"
        
        try:
            # Store metrics in Redis
            pipe = self.redis_client.pipeline()
            
            # Update average response time
            pipe.hincrbyfloat(
                f"{self.metrics_key}:avg_time",
                endpoint,
                duration
            )
            pipe.hincrby(
                f"{self.metrics_key}:count",
                endpoint,
                1
            )
            
            # Store recent response times
            pipe.lpush(
                f"{self.metrics_key}:recent:{endpoint}",
                duration
            )
            pipe.ltrim(
                f"{self.metrics_key}:recent:{endpoint}",
                0,
                99  # Keep last 100 values
            )
            
            # Track status codes
            pipe.hincrby(
                f"{self.metrics_key}:status_codes",
                f"{endpoint}:{response.status_code}",
                1
            )
            
            pipe.execute()
            
        except redis.RedisError as e:
            logger.error(f"Redis error in performance monitoring: {e}")
        
        return response

def setup_middleware(app: FastAPI, config: Dict) -> None:
    """Setup all middleware components for the application."""
    
    # Add middleware in order
    app.add_middleware(
        ErrorHandlingMiddleware
    )
    app.add_middleware(
        RequestLoggingMiddleware
    )
    app.add_middleware(
        RateLimitMiddleware,
        config=RateLimitConfig(**config.get("rate_limit", {}))
    )
    app.add_middleware(
        PerformanceMonitoringMiddleware,
        redis_url=config.get("redis_url", "redis://localhost:6379/0")
    )