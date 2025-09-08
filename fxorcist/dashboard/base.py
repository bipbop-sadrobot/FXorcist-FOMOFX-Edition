"""
Base dashboard module with trading system integration.

Implements core dashboard functionality and trading system integration with
circuit breaker pattern for resilience.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from ..core.base import TradingModule
from ..core.events import Event, EventType
from ..core.dispatcher import EventDispatcher
from .models import Trade, Position, OrderDirection, TradeStatus
from .cache import cache_instance
from .websocket import connection_manager, WebSocketMessage
from .utils.circuit_breaker import CircuitBreaker, CircuitBreakerError

# Default circuit breaker settings
DEFAULT_CIRCUIT_BREAKER_CONFIG = {
    'failure_threshold': 5,
    'reset_timeout': 60,
    'half_open_timeout': 30
}

class DashboardError(Exception):
    """Base exception for dashboard errors."""
    pass

class CircuitBreakerConfigError(DashboardError):
    """Error in circuit breaker configuration."""
    pass

class DashboardModule(TradingModule):
    """Base class for dashboard components with trading system integration."""
    """Base class for dashboard components with trading system integration."""
    
    def __init__(
        self,
        name: str,
        event_dispatcher: EventDispatcher,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize dashboard module.
        
        Args:
            name: Module name
            event_dispatcher: Event dispatcher instance
            config: Optional configuration dictionary with optional circuit_breaker section
        """
        super().__init__(name, event_dispatcher, config)
        
        # Module state
        self.last_update = None
        self.error_count = 0
        self.max_errors = self.config.get('max_errors', 10)
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Initialize circuit breaker
        circuit_breaker_config = self.config.get('circuit_breaker', DEFAULT_CIRCUIT_BREAKER_CONFIG)
        self.circuit_breaker = CircuitBreaker(
            name=f"{self.name}_circuit_breaker",
            failure_threshold=circuit_breaker_config.get('failure_threshold', 5),
            reset_timeout=circuit_breaker_config.get('reset_timeout', 60),
            half_open_timeout=circuit_breaker_config.get('half_open_timeout', 30)
        )
    
    async def start(self):
        """Start dashboard module."""
        await super().start()
        
        try:
            # Initialize cache
            if not cache_instance.redis:
                await cache_instance.start()
            
            # Subscribe to events
            self.event_dispatcher.subscribe(EventType.TRADE, self.handle_event)
            self.event_dispatcher.subscribe(EventType.MARKET, self.handle_event)
            self.event_dispatcher.subscribe(EventType.SIGNAL, self.handle_event)
            
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self.logger.info(f"Dashboard module {self.name} started")
            
        except Exception as e:
            self.logger.error(f"Error starting dashboard module: {e}")
            raise DashboardError(f"Failed to start module: {e}")
    
    async def stop(self):
        """Stop dashboard module."""
        try:
            # Cancel cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            await super().stop()
            self.logger.info(f"Dashboard module {self.name} stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping dashboard module: {e}")
            raise DashboardError(f"Failed to stop module: {e}")
    
    async def handle_event(self, event: Event):
        """Handle trading system events with enhanced error handling.
        
        Args:
            event: Event to handle
        """
        try:
            # Update last activity
            self.last_update = datetime.utcnow()
            
            # Process event with circuit breaker protection
            async with self.circuit_breaker:
                if event.type == EventType.TRADE:
                    await self._handle_trade_event(event)
                elif event.type == EventType.MARKET:
                    await self._handle_market_event(event)
                elif event.type == EventType.SIGNAL:
                    await self._handle_signal_event(event)
            
            # Reset error count on successful processing
            self.error_count = 0
            
        except CircuitBreakerError as e:
            self.logger.error(f"Circuit breaker open: {e}")
            await self._handle_circuit_breaker_error(event)
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error handling event: {e}", exc_info=True)
            
            # Update metrics
            DASHBOARD_ERROR_COUNTER.labels(
                module=self.name,
                event_type=event.type.value
            ).inc()
            
            if self.error_count >= self.max_errors:
                self.logger.critical("Maximum error count exceeded")
                await self.stop()
                raise DashboardError("Maximum error count exceeded")
    
    async def _handle_trade_event(self, event: Event):
        """Handle trade events.
        
        Args:
            event: Trade event
        """
        try:
            # Convert event to trade model
            trade = Trade(
                id=event.event_id,
                instrument=event.data['instrument'],
                direction=OrderDirection(event.data['direction']),
                entry_price=event.data['entry_price'],
                exit_price=event.data.get('exit_price'),
                entry_time=event.data['entry_time'],
                exit_time=event.data.get('exit_time'),
                units=event.data['units'],
                pnl=event.data.get('pnl'),
                commission=event.data['commission'],
                status=TradeStatus(event.data['status'])
            )
            
            # Broadcast trade update
            message = WebSocketMessage(
                type="trade_update",
                data=trade.dict()
            )
            await connection_manager.broadcast(message, "trading")
            
            # Update cache
            await cache_instance.delete("trading:trades:all")
            
        except Exception as e:
            self.logger.error(f"Error handling trade event: {e}")
            raise
    
    async def _handle_market_event(self, event: Event):
        """Handle market data events.
        
        Args:
            event: Market data event
        """
        try:
            # Broadcast market update
            message = WebSocketMessage(
                type="market_update",
                data={
                    "instrument": event.data['instrument'],
                    "timestamp": event.timestamp.isoformat(),
                    "bid": event.data['bid'],
                    "ask": event.data['ask'],
                    "spread": event.data['ask'] - event.data['bid'],
                    "volume": event.data.get('volume', 0)
                }
            )
            await connection_manager.broadcast(
                message,
                f"market:{event.data['instrument']}"
            )
            
            # Update cache
            cache_key = f"market:price:{event.data['instrument']}"
            await cache_instance.set(cache_key, event.data, ttl=5)
            
        except Exception as e:
            self.logger.error(f"Error handling market event: {e}")
            raise
    
    async def _handle_signal_event(self, event: Event):
        """Handle trading signal events.
        
        Args:
            event: Signal event
        """
        try:
            # Broadcast signal update
            message = WebSocketMessage(
                type="signal_update",
                data={
                    "instrument": event.data['instrument'],
                    "timestamp": event.timestamp.isoformat(),
                    "direction": event.data['direction'],
                    "strength": event.data['strength'],
                    "strategy": event.data['strategy']
                }
            )
            await connection_manager.broadcast(
                message,
                f"signals:{event.data['instrument']}"
            )
            
            # Update cache
            cache_key = f"signals:{event.data['instrument']}"
            await cache_instance.set(cache_key, event.data, ttl=60)
            
        except Exception as e:
            self.logger.error(f"Error handling signal event: {e}")
            raise
    
    async def _cleanup_loop(self):
        """Enhanced periodic cleanup and maintenance."""
        while True:
            try:
                # Check module health with metrics
                health_status = await self.health_check()
                DASHBOARD_HEALTH_GAUGE.labels(
                    module=self.name
                ).set(1 if health_status else 0)
                
                if not health_status:
                    self.logger.warning("Health check failed")
                    await self.reset()
                
                # Clean expired cache entries
                await self._cleanup_cache()
                
                # Check circuit breaker status
                if self.circuit_breaker.is_open:
                    self.logger.warning("Circuit breaker is open")
                    DASHBOARD_CIRCUIT_BREAKER_GAUGE.labels(
                        module=self.name
                    ).set(0)
                else:
                    DASHBOARD_CIRCUIT_BREAKER_GAUGE.labels(
                        module=self.name
                    ).set(1)
                
                await asyncio.sleep(60)  # Run every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}", exc_info=True)
                await asyncio.sleep(5)  # Back off on error
    
    async def _cleanup_cache(self):
        """Clean up expired cache entries."""
        try:
            # Get all cache keys
            keys = await cache_instance.keys("*")
            cleaned = 0
            
            # Check TTL for each key
            for key in keys:
                ttl = await cache_instance.ttl(key)
                if ttl < 0:  # Expired or no TTL
                    await cache_instance.delete(key)
                    cleaned += 1
                    self.logger.debug(f"Cleaned up expired cache key: {key}")
            
            if cleaned > 0:
                self.logger.info(f"Cleaned up {cleaned} expired cache entries")
            
        except Exception as e:
            self.logger.error(f"Cache cleanup error: {e}", exc_info=True)
    
    async def health_check(self) -> bool:
        """Perform health check.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check core services
            if not self.running:
                return False
            
            # Check cache connection
            if not cache_instance.redis:
                return False
            
            # Check event dispatcher
            if not self.event_dispatcher:
                return False
            
            # Check error count
            if self.error_count >= self.max_errors:
                return False
            
            # Check last update time
            if self.last_update:
                inactive_time = (datetime.utcnow() - self.last_update).total_seconds()
                if inactive_time > self.config.get('max_inactive_time', 300):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            return False
    
    async def reset(self):
        """Reset module state."""
        try:
            await super().reset()
            self.last_update = None
            self.error_count = 0
            self.logger.info("Module state reset")
            
        except Exception as e:
            self.logger.error(f"Reset error: {e}")
            raise DashboardError(f"Failed to reset module: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get module status.
        
        Returns:
            Status information dictionary
        """
        status = super().get_status()
        status.update({
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'error_count': self.error_count,
            'cache_connected': bool(cache_instance.redis),
            'websocket_connections': connection_manager.get_stats().active_connections
        })
        return status