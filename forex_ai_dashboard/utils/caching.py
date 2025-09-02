"""
Advanced caching system for Forex AI Dashboard.
Provides multiple caching strategies: memory, disk, and Redis.
"""

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union
import redis
import diskcache as dc
from cachetools import TTLCache, cached
import joblib
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)

class CacheManager:
    """Unified cache manager supporting multiple storage backends."""

    def __init__(self, cache_dir: str = "cache", redis_url: str = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize caches
        self.memory_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour TTL
        self.disk_cache = dc.Cache(str(self.cache_dir / "disk_cache"))

        # Redis cache (optional)
        self.redis_cache = None
        if redis_url:
            try:
                self.redis_cache = redis.from_url(redis_url)
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")

        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0

    def _generate_key(self, key_components: list) -> str:
        """Generate a consistent cache key from components."""
        key_string = json.dumps(key_components, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, key: str, default=None) -> Any:
        """Get value from cache with fallback strategy."""
        # Try memory cache first
        if key in self.memory_cache:
            self.cache_hits += 1
            return self.memory_cache[key]

        # Try Redis cache
        if self.redis_cache:
            try:
                cached_data = self.redis_cache.get(key)
                if cached_data is not None:
                    # Store in memory cache for faster access
                    self.memory_cache[key] = pickle.loads(cached_data)
                    self.cache_hits += 1
                    return self.memory_cache[key]
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")

        # Try disk cache
        try:
            cached_data = self.disk_cache.get(key)
            if cached_data is not None:
                # Store in memory cache
                self.memory_cache[key] = cached_data
                self.cache_hits += 1
                return cached_data
        except Exception as e:
            logger.warning(f"Disk cache error: {e}")

        self.cache_misses += 1
        return default

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set value in all available caches."""
        # Store in memory cache
        self.memory_cache[key] = value

        # Store in Redis cache
        if self.redis_cache:
            try:
                self.redis_cache.setex(key, ttl, pickle.dumps(value))
            except Exception as e:
                logger.warning(f"Redis cache set error: {e}")

        # Store in disk cache
        try:
            self.disk_cache.set(key, value, expire=ttl)
        except Exception as e:
            logger.warning(f"Disk cache set error: {e}")

    def delete(self, key: str) -> None:
        """Delete key from all caches."""
        # Delete from memory cache
        if key in self.memory_cache:
            del self.memory_cache[key]

        # Delete from Redis cache
        if self.redis_cache:
            try:
                self.redis_cache.delete(key)
            except Exception as e:
                logger.warning(f"Redis cache delete error: {e}")

        # Delete from disk cache
        try:
            self.disk_cache.delete(key)
        except Exception as e:
            logger.warning(f"Disk cache delete error: {e}")

    def clear(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()

        if self.redis_cache:
            try:
                self.redis_cache.flushdb()
            except Exception as e:
                logger.warning(f"Redis cache clear error: {e}")

        try:
            self.disk_cache.clear()
        except Exception as e:
            logger.warning(f"Disk cache clear error: {e}")

    def get_stats(self) -> Dict:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'redis_available': self.redis_cache is not None
        }

class DataFrameCache:
    """Specialized cache for pandas DataFrames with compression."""

    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager

    def get_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """Get DataFrame from cache."""
        cached_data = self.cache_manager.get(key)
        if cached_data is not None:
            try:
                return pd.read_parquet(cached_data) if isinstance(cached_data, bytes) else cached_data
            except Exception as e:
                logger.warning(f"Failed to deserialize DataFrame: {e}")
                return None
        return None

    def set_dataframe(self, key: str, df: pd.DataFrame, ttl: int = 3600) -> None:
        """Cache DataFrame with compression."""
        try:
            # Use parquet for efficient storage
            temp_path = self.cache_manager.cache_dir / f"temp_{key}.parquet"
            df.to_parquet(temp_path, compression='snappy')

            with open(temp_path, 'rb') as f:
                compressed_data = f.read()

            self.cache_manager.set(key, compressed_data, ttl)
            temp_path.unlink()  # Clean up temp file
        except Exception as e:
            logger.warning(f"Failed to cache DataFrame: {e}")

class ModelCache:
    """Cache for trained ML models."""

    def __init__(self, cache_manager: CacheManager, model_dir: str = "models/cache"):
        self.cache_manager = cache_manager
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def get_model(self, model_name: str, version: str = "latest"):
        """Get cached model."""
        key = f"model_{model_name}_{version}"
        cached_path = self.cache_manager.get(key)

        if cached_path and Path(cached_path).exists():
            try:
                return joblib.load(cached_path)
            except Exception as e:
                logger.warning(f"Failed to load cached model: {e}")

        return None

    def set_model(self, model_name: str, model: Any, version: str = "latest", ttl: int = 86400) -> None:
        """Cache trained model."""
        try:
            model_path = self.model_dir / f"{model_name}_{version}.joblib"
            joblib.dump(model, model_path)

            key = f"model_{model_name}_{version}"
            self.cache_manager.set(key, str(model_path), ttl)
        except Exception as e:
            logger.warning(f"Failed to cache model: {e}")

class ComputationCache:
    """Cache for expensive computations."""

    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager

    def cached_computation(self, func_name: str, *args, **kwargs):
        """Decorator for caching function results."""
        def decorator(func):
            def wrapper(*func_args, **func_kwargs):
                # Create cache key from function name and arguments
                key_components = [func_name, func_args, func_kwargs]
                cache_key = self.cache_manager._generate_key(key_components)

                # Try to get from cache
                cached_result = self.cache_manager.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {func_name}")
                    return cached_result

                # Compute and cache result
                logger.debug(f"Cache miss for {func_name}, computing...")
                result = func(*func_args, **func_kwargs)
                self.cache_manager.set(cache_key, result)
                return result

            return wrapper
        return decorator

# Global cache manager instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get or create global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        # Try to get Redis URL from environment
        redis_url = None
        try:
            import os
            redis_url = os.getenv('REDIS_URL')
        except:
            pass

        _cache_manager = CacheManager(redis_url=redis_url)

    return _cache_manager

def get_dataframe_cache() -> DataFrameCache:
    """Get DataFrame cache instance."""
    return DataFrameCache(get_cache_manager())

def get_model_cache() -> ModelCache:
    """Get model cache instance."""
    return ModelCache(get_cache_manager())

def get_computation_cache() -> ComputationCache:
    """Get computation cache instance."""
    return ComputationCache(get_cache_manager())