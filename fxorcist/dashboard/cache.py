"""
Redis cache integration for dashboard data caching.

Implements caching strategies and performance optimization.
"""

import json
import logging
from typing import Any, Dict, Optional, Union, List, Tuple
from datetime import datetime, timedelta
import redis.asyncio as aioredis
from pydantic import BaseModel, Field
import pickle
import hashlib
from functools import wraps
import asyncio

# Setup logging
logger = logging.getLogger(__name__)

class CacheConfig(BaseModel):
    """Cache configuration settings."""
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    default_ttl: int = Field(
        default=300,  # 5 minutes
        description="Default cache TTL in seconds"
    )
    max_size: int = Field(
        default=1000,
        description="Maximum number of cache entries"
    )
    enable_stats: bool = Field(
        default=True,
        description="Enable cache statistics"
    )

class CacheStats(BaseModel):
    """Cache statistics."""
    hits: int = Field(default=0, description="Cache hits")
    misses: int = Field(default=0, description="Cache misses")
    size: int = Field(default=0, description="Current cache size")
    evictions: int = Field(default=0, description="Cache evictions")
    errors: int = Field(default=0, description="Cache errors")
    last_error: Optional[str] = Field(
        default=None,
        description="Last error message"
    )
    last_update: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )

class CacheKey:
    """Cache key generator."""
    
    @staticmethod
    def generate(prefix: str, *args: Any, **kwargs: Any) -> str:
        """Generate cache key from arguments."""
        key_parts = [prefix]
        
        # Add positional args
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                # Hash complex objects
                key_parts.append(
                    hashlib.md5(
                        str(arg).encode()
                    ).hexdigest()[:8]
                )
        
        # Add keyword args (sorted for consistency)
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        return ":".join(key_parts)

class RedisCache:
    """Redis cache implementation with monitoring."""
    
    def __init__(self, config: CacheConfig):
        """Initialize Redis cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self.redis: Optional[aioredis.Redis] = None
        self.stats = CacheStats()
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start cache service."""
        self.redis = await aioredis.from_url(self.config.redis_url)
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Redis cache started")
    
    async def stop(self):
        """Stop cache service."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.redis:
            await self.redis.close()
        
        logger.info("Redis cache stopped")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found, None otherwise
        """
        if not self.redis:
            return None
        
        try:
            value = await self.redis.get(key)
            
            if value is not None:
                self.stats.hits += 1
                return pickle.loads(value)
            
            self.stats.misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.stats.errors += 1
            self.stats.last_error = str(e)
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set cache value.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not self.redis:
            return False
        
        try:
            # Serialize value
            serialized = pickle.dumps(value)
            
            # Set with TTL
            ttl = ttl or self.config.default_ttl
            await self.redis.setex(key, ttl, serialized)
            
            self.stats.size = await self.redis.dbsize()
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            self.stats.errors += 1
            self.stats.last_error = str(e)
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False otherwise
        """
        if not self.redis:
            return False
        
        try:
            await self.redis.delete(key)
            self.stats.size = await self.redis.dbsize()
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            self.stats.errors += 1
            self.stats.last_error = str(e)
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.redis:
            return False
        
        try:
            await self.redis.flushdb()
            self.stats.size = 0
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            self.stats.errors += 1
            self.stats.last_error = str(e)
            return False
    
    async def _cleanup_loop(self):
        """Periodic cache cleanup and monitoring."""
        while True:
            try:
                # Update cache size
                if self.redis:
                    self.stats.size = await self.redis.dbsize()
                
                # Check max size
                if self.stats.size > self.config.max_size:
                    # Evict oldest entries
                    keys = await self.redis.keys("*")
                    for key in keys[self.config.max_size:]:
                        await self.delete(key)
                        self.stats.evictions += 1
                
                # Update timestamp
                self.stats.last_update = datetime.utcnow()
                
                await asyncio.sleep(60)  # Run every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    def get_stats(self) -> CacheStats:
        """Get current cache statistics.
        
        Returns:
            Current cache statistics
        """
        return self.stats

def cached(
    prefix: str,
    ttl: Optional[int] = None,
    key_builder: Optional[callable] = None
):
    """Cache decorator for async functions.
    
    Args:
        prefix: Cache key prefix
        ttl: Optional TTL override
        key_builder: Optional custom key builder
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache instance
            cache = cache_instance
            if not cache or not cache.redis:
                return await func(*args, **kwargs)
            
            # Generate cache key
            if key_builder:
                key = key_builder(*args, **kwargs)
            else:
                key = CacheKey.generate(prefix, *args, **kwargs)
            
            # Try cache
            cached_value = await cache.get(key)
            if cached_value is not None:
                return cached_value
            
            # Call function
            value = await func(*args, **kwargs)
            
            # Cache result
            await cache.set(key, value, ttl)
            
            return value
        return wrapper
    return decorator

# Global cache instance
cache_instance = RedisCache(CacheConfig())