"""
Trading system integration for the dashboard.

Implements real-time trading system state management and synchronization.
"""

import asyncio
import logging
from typing import Dict, Optional, List
from datetime import datetime
from decimal import Decimal
import json
from enum import Enum

from .models import Trade, Position, OrderDirection, TradeStatus
from .websocket import connection_manager, WebSocketMessage
from .cache import cached, cache_instance

# Setup logging
logger = logging.getLogger(__name__)

class TradingState(Enum):
    """Trading system states."""
    INITIALIZING = "initializing"
    READY = "ready"
    TRADING = "trading"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class TradingSystemManager:
    """Manages trading system integration and state."""
    
    def __init__(self):
        """Initialize trading system manager."""
        self.state = TradingState.INITIALIZING
        self.last_sync = None
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self._sync_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start trading system integration."""
        logger.info("Starting trading system integration")
        
        try:
            # Initialize connection to trading system
            await self._initialize_connection()
            
            # Start sync task
            self._sync_task = asyncio.create_task(self._sync_loop())
            
            self.state = TradingState.READY
            logger.info("Trading system integration ready")
            
        except Exception as e:
            logger.error(f"Failed to start trading system: {e}")
            self.state = TradingState.ERROR
            raise
    
    async def stop(self):
        """Stop trading system integration."""
        logger.info("Stopping trading system integration")
        
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        
        self.state = TradingState.SHUTDOWN
    
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        if self.state != TradingState.READY:
            raise RuntimeError("Trading system not ready")
        
        try:
            # Get positions from trading system
            positions = list(self.positions.values())
            
            # Update cache
            await cache_instance.set(
                "trading:positions",
                positions,
                ttl=30  # Cache for 30 seconds
            )
            
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise
    
    async def get_trades(
        self,
        status: Optional[TradeStatus] = None,
        limit: int = 100
    ) -> List[Trade]:
        """Get trade history."""
        if self.state != TradingState.READY:
            raise RuntimeError("Trading system not ready")
        
        try:
            # Filter trades by status
            trades = self.trades
            if status:
                trades = [t for t in trades if t.status == status]
            
            # Sort by entry time descending
            trades.sort(key=lambda x: x.entry_time, reverse=True)
            
            # Update cache
            cache_key = f"trading:trades:{status.value if status else 'all'}"
            await cache_instance.set(cache_key, trades[:limit], ttl=60)
            
            return trades[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            raise
    
    async def execute_trade(
        self,
        instrument: str,
        direction: OrderDirection,
        units: int,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None
    ) -> Trade:
        """Execute new trade."""
        if self.state != TradingState.READY:
            raise RuntimeError("Trading system not ready")
        
        try:
            # Execute trade in trading system
            trade = Trade(
                id=f"trade_{datetime.utcnow().timestamp()}",
                instrument=instrument,
                direction=direction,
                entry_price=Decimal("1.1850"),  # Example price
                entry_time=datetime.utcnow(),
                units=units,
                commission=Decimal("2.50"),
                status=TradeStatus.OPEN,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Update local state
            self.trades.append(trade)
            
            # Broadcast trade event
            await connection_manager.broadcast(
                WebSocketMessage(
                    type="trade_executed",
                    data=trade.dict()
                ),
                channel="trading"
            )
            
            # Invalidate relevant caches
            await cache_instance.delete("trading:positions")
            await cache_instance.delete("trading:trades:all")
            
            return trade
            
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
            raise
    
    async def close_position(self, instrument: str) -> Trade:
        """Close an open position."""
        if self.state != TradingState.READY:
            raise RuntimeError("Trading system not ready")
        
        try:
            if instrument not in self.positions:
                raise ValueError(f"No open position for {instrument}")
            
            position = self.positions[instrument]
            
            # Close position in trading system
            trade = Trade(
                id=f"trade_{datetime.utcnow().timestamp()}",
                instrument=instrument,
                direction=position.direction,
                entry_price=position.entry_price,
                exit_price=position.current_price,
                entry_time=datetime.utcnow(),
                exit_time=datetime.utcnow(),
                units=position.units,
                pnl=position.unrealized_pnl,
                commission=Decimal("2.50"),
                status=TradeStatus.CLOSED
            )
            
            # Update local state
            del self.positions[instrument]
            self.trades.append(trade)
            
            # Broadcast position update
            await connection_manager.broadcast(
                WebSocketMessage(
                    type="position_closed",
                    data=trade.dict()
                ),
                channel="trading"
            )
            
            # Invalidate relevant caches
            await cache_instance.delete("trading:positions")
            await cache_instance.delete("trading:trades:all")
            
            return trade
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            raise
    
    async def _initialize_connection(self):
        """Initialize connection to trading system."""
        # Implement actual trading system connection
        await asyncio.sleep(1)  # Simulate initialization
    
    async def _sync_loop(self):
        """Periodic state synchronization."""
        while True:
            try:
                # Sync with trading system
                await self._sync_state()
                
                # Update last sync time
                self.last_sync = datetime.utcnow()
                
                await asyncio.sleep(1)  # Sync every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync error: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def _sync_state(self):
        """Synchronize state with trading system."""
        # Implement actual state synchronization
        pass

# Global trading system manager instance
trading_manager = TradingSystemManager()