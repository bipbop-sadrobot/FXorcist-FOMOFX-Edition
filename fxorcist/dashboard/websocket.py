"""
WebSocket connection management for real-time dashboard updates.

Implements connection pooling, heartbeat, and message broadcasting.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
import redis.asyncio as aioredis
from pydantic import BaseModel, Field

# Setup logging
logger = logging.getLogger(__name__)

class ConnectionStats(BaseModel):
    """WebSocket connection statistics."""
    total_connections: int = Field(default=0, description="Total number of connections")
    active_connections: int = Field(default=0, description="Currently active connections")
    messages_sent: int = Field(default=0, description="Total messages sent")
    messages_received: int = Field(default=0, description="Total messages received")
    errors: int = Field(default=0, description="Total error count")
    last_error: Optional[str] = Field(default=None, description="Last error message")
    last_activity: Optional[datetime] = Field(default=None, description="Last activity timestamp")

class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: str = Field(..., description="Message type")
    data: Dict = Field(..., description="Message payload")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ConnectionManager:
    """Manages WebSocket connections and message broadcasting."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize connection manager.
        
        Args:
            redis_url: Redis connection URL for pub/sub
        """
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_channels: Dict[str, Set[str]] = {}
        self.stats = ConnectionStats()
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self.heartbeat_interval = 30  # seconds
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the connection manager."""
        self.redis = await aioredis.from_url(self.redis_url)
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("WebSocket connection manager started")
    
    async def stop(self):
        """Stop the connection manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.redis:
            await self.redis.close()
        
        logger.info("WebSocket connection manager stopped")
    
    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Handle new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            client_id: Unique client identifier
        """
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_channels[client_id] = set()
        self.stats.active_connections = len(self.active_connections)
        self.stats.total_connections += 1
        self.stats.last_activity = datetime.utcnow()
        
        logger.info(f"Client connected: {client_id}")
        
        # Send initial state
        await self.send_message(
            client_id,
            WebSocketMessage(
                type="connection_established",
                data={"client_id": client_id}
            )
        )
    
    async def disconnect(self, client_id: str) -> None:
        """Handle WebSocket disconnection.
        
        Args:
            client_id: Client identifier
        """
        if client_id in self.active_connections:
            # Unsubscribe from all channels
            channels = self.connection_channels.get(client_id, set())
            for channel in channels:
                await self.unsubscribe(client_id, channel)
            
            # Remove connection
            del self.active_connections[client_id]
            del self.connection_channels[client_id]
            self.stats.active_connections = len(self.active_connections)
            self.stats.last_activity = datetime.utcnow()
            
            logger.info(f"Client disconnected: {client_id}")
    
    async def send_message(
        self,
        client_id: str,
        message: WebSocketMessage
    ) -> None:
        """Send message to specific client.
        
        Args:
            client_id: Client identifier
            message: Message to send
        """
        if client_id not in self.active_connections:
            return
        
        try:
            websocket = self.active_connections[client_id]
            await websocket.send_json(message.dict())
            self.stats.messages_sent += 1
            self.stats.last_activity = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error sending message to {client_id}: {e}")
            self.stats.errors += 1
            self.stats.last_error = str(e)
            await self.disconnect(client_id)
    
    async def broadcast(
        self,
        message: WebSocketMessage,
        channel: Optional[str] = None
    ) -> None:
        """Broadcast message to all connected clients or channel subscribers.
        
        Args:
            message: Message to broadcast
            channel: Optional channel to broadcast to
        """
        if channel:
            # Broadcast to channel subscribers
            subscribers = [
                cid for cid, channels in self.connection_channels.items()
                if channel in channels
            ]
        else:
            # Broadcast to all connections
            subscribers = list(self.active_connections.keys())
        
        for client_id in subscribers:
            await self.send_message(client_id, message)
    
    async def subscribe(self, client_id: str, channel: str) -> None:
        """Subscribe client to channel.
        
        Args:
            client_id: Client identifier
            channel: Channel name
        """
        if client_id in self.connection_channels:
            self.connection_channels[client_id].add(channel)
            logger.info(f"Client {client_id} subscribed to {channel}")
    
    async def unsubscribe(self, client_id: str, channel: str) -> None:
        """Unsubscribe client from channel.
        
        Args:
            client_id: Client identifier
            channel: Channel name
        """
        if client_id in self.connection_channels:
            self.connection_channels[client_id].discard(channel)
            logger.info(f"Client {client_id} unsubscribed from {channel}")
    
    async def _cleanup_loop(self):
        """Periodic cleanup and heartbeat."""
        while True:
            try:
                # Send heartbeat to all connections
                heartbeat_msg = WebSocketMessage(
                    type="heartbeat",
                    data={"timestamp": datetime.utcnow().isoformat()}
                )
                
                dead_connections = []
                
                for client_id, websocket in self.active_connections.items():
                    try:
                        await websocket.send_json(heartbeat_msg.dict())
                    except Exception as e:
                        logger.warning(f"Connection dead for {client_id}: {e}")
                        dead_connections.append(client_id)
                
                # Clean up dead connections
                for client_id in dead_connections:
                    await self.disconnect(client_id)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    def get_stats(self) -> ConnectionStats:
        """Get current connection statistics.
        
        Returns:
            Current connection statistics
        """
        return self.stats

# Global connection manager instance
connection_manager = ConnectionManager()