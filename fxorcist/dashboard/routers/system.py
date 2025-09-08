"""
System monitoring and management API endpoints.

Implements routes for health checks, metrics, and system management.
"""

import psutil
import logging
from fastapi import APIRouter, Depends, HTTPException, Security
from typing import Dict, List
from datetime import datetime, timedelta
import os
import json

from ..models import SystemStatus, User
from ..auth import auth_service
from ..cache import cached, cache_instance
from ..websocket import connection_manager

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/system",
    tags=["system"],
    responses={404: {"description": "Not found"}}
)

@router.get("/health")
async def health_check() -> Dict:
    """Basic health check endpoint."""
    try:
        # Check core components
        components = {
            "database": True,
            "redis": await _check_redis(),
            "websocket": connection_manager.get_stats().active_connections > 0,
            "trading_system": True  # Replace with actual check
        }
        
        # Check system resources
        resources = {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": components,
            "resources": resources
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@router.get("/status", response_model=SystemStatus)
@cached("system:status", ttl=30)
async def get_system_status(
    current_user: User = Security(
        auth_service.get_current_active_user,
        scopes=["read:system"]
    )
) -> SystemStatus:
    """Get detailed system status."""
    try:
        # Get component statuses
        components = {
            "trading_engine": "running",
            "data_handler": "running",
            "portfolio_manager": "running",
            "risk_manager": "running",
            "websocket_server": "running",
            "cache_service": "running"
        }
        
        # Get system metrics
        status = SystemStatus(
            status="operational",
            uptime=psutil.boot_time(),
            cpu_usage=psutil.cpu_percent(interval=1),
            memory_usage=psutil.Process().memory_info().rss,
            active_connections=connection_manager.get_stats().active_connections,
            last_update=datetime.utcnow(),
            components_status=components
        )
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get system status"
        )

@router.get("/metrics")
@cached("system:metrics", ttl=60)
async def get_system_metrics(
    current_user: User = Security(
        auth_service.get_current_active_user,
        scopes=["read:system"]
    )
) -> Dict:
    """Get system performance metrics."""
    try:
        # Get system metrics
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu": {
                    "usage_percent": psutil.cpu_percent(interval=1),
                    "count": psutil.cpu_count(),
                    "frequency": psutil.cpu_freq().current
                },
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "used": psutil.virtual_memory().used,
                    "percent": psutil.virtual_memory().percent
                },
                "disk": {
                    "total": psutil.disk_usage('/').total,
                    "used": psutil.disk_usage('/').used,
                    "free": psutil.disk_usage('/').free,
                    "percent": psutil.disk_usage('/').percent
                }
            },
            "application": {
                "process": {
                    "memory_usage": psutil.Process().memory_info().rss,
                    "cpu_usage": psutil.Process().cpu_percent(),
                    "threads": psutil.Process().num_threads()
                },
                "connections": {
                    "active": connection_manager.get_stats().active_connections,
                    "total": connection_manager.get_stats().total_connections
                },
                "cache": {
                    "size": cache_instance.stats.size,
                    "hits": cache_instance.stats.hits,
                    "misses": cache_instance.stats.misses
                }
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get system metrics"
        )

@router.get("/config")
async def get_system_config(
    current_user: User = Security(
        auth_service.get_current_active_user,
        scopes=["admin"]
    )
) -> Dict:
    """Get system configuration."""
    try:
        # Get configuration
        config = {
            "environment": os.getenv("FXORCIST_ENV", "development"),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "cache": {
                "enabled": True,
                "ttl": 300,
                "max_size": 1000
            },
            "websocket": {
                "heartbeat_interval": 30,
                "max_connections": 1000
            },
            "trading": {
                "execution_delay": 100,
                "max_positions": 10,
                "risk_limit": 0.02
            }
        }
        
        return config
        
    except Exception as e:
        logger.error(f"Error getting system config: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get system config"
        )

@router.get("/logs")
async def get_system_logs(
    component: str = None,
    level: str = "INFO",
    limit: int = 100,
    current_user: User = Security(
        auth_service.get_current_active_user,
        scopes=["admin"]
    )
) -> List[Dict]:
    """Get system logs."""
    try:
        # Get logs from file
        logs = []
        log_file = "logs/system.log"
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                for line in f.readlines()[-limit:]:
                    # Parse log line
                    try:
                        timestamp, log_level, message = line.split(" - ", 2)
                        if level == "ALL" or log_level == level:
                            logs.append({
                                "timestamp": timestamp,
                                "level": log_level,
                                "message": message.strip()
                            })
                    except Exception:
                        continue
        
        return logs
        
    except Exception as e:
        logger.error(f"Error getting system logs: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get system logs"
        )

async def _check_redis() -> bool:
    """Check Redis connection."""
    try:
        if cache_instance and cache_instance.redis:
            await cache_instance.redis.ping()
            return True
        return False
    except Exception:
        return False