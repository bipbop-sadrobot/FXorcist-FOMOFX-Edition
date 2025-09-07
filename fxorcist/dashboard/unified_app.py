"""
Unified dashboard application integrating all FXorcist components.

Implements a FastAPI backend with WebSocket support and proper trading system integration.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
from fastapi import FastAPI, WebSocket, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from ..core.base import TradingModule
from ..core.events import Event, EventType
from ..core.dispatcher import EventDispatcher
from .models import (
    Trade, Position, PortfolioSummary, PerformanceMetrics,
    MarketData, Signal, SystemStatus
)
from .auth import auth_service, User
from .cache import cache_instance
from .websocket import connection_manager
from .middleware import setup_middleware
from .routers import portfolio, market, system
from ..utils.config import config_manager, FXorcistConfig

# Setup logging
logger = logging.getLogger(__name__)

class DashboardService(TradingModule):
    """Main dashboard service integrating with trading system."""
    
    def __init__(
        self,
        event_dispatcher: EventDispatcher,
        config: FXorcistConfig
    ):
        """Initialize dashboard service.
        
        Args:
            event_dispatcher: Event dispatcher instance
            config: Configuration instance
        """
        super().__init__("dashboard", event_dispatcher)
        self.config = config
        self.app = self._create_app()
        
        # Track component states
        self.component_states: Dict[str, bool] = {
            "portfolio": False,
            "market": False,
            "system": False
        }
    
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        app = FastAPI(
            title="FXorcist Trading Dashboard",
            description="Unified trading dashboard with real-time updates",
            version="1.0.0"
        )
        
        # Configure CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately in production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Setup middleware
        setup_middleware(app, {
            "rate_limit": {
                "requests_per_minute": self.config.dashboard.rate_limit,
                "burst_limit": self.config.dashboard.burst_limit
            },
            "redis_url": self.config.system.redis_url
        })
        
        # Include routers
        app.include_router(portfolio.router)
        app.include_router(market.router)
        app.include_router(system.router)
        
        # Add authentication endpoints
        @app.post("/token")
        async def login(username: str, password: str):
            """Login endpoint."""
            user = auth_service.authenticate_user(username, password)
            if not user:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid credentials"
                )
            
            access_token = auth_service.create_access_token(
                data={"sub": user.username, "scopes": user.scopes}
            )
            
            return {
                "access_token": access_token,
                "token_type": "bearer"
            }
        
        @app.get("/users/me")
        async def read_users_me(
            current_user: User = Depends(auth_service.get_current_active_user)
        ):
            """Get current user information."""
            return current_user
        
        # Add WebSocket endpoint
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            client_id = f"client_{datetime.utcnow().timestamp()}"
            await connection_manager.connect(websocket, client_id)
            
            try:
                while True:
                    data = await websocket.receive_json()
                    
                    # Handle subscription requests
                    if data.get("type") == "subscribe":
                        channel = data.get("channel")
                        if channel:
                            await connection_manager.subscribe(client_id, channel)
                    
                    # Keep connection alive
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                await connection_manager.disconnect(client_id)
        
        return app
    
    async def start(self):
        """Start dashboard service."""
        await super().start()
        
        try:
            # Initialize cache
            await cache_instance.start()
            
            # Initialize WebSocket manager
            await connection_manager.start()
            
            # Subscribe to events
            self.event_dispatcher.subscribe(EventType.TRADE, self.handle_event)
            self.event_dispatcher.subscribe(EventType.MARKET, self.handle_event)
            self.event_dispatcher.subscribe(EventType.SIGNAL, self.handle_event)
            
            logger.info("Dashboard service started")
            
        except Exception as e:
            logger.error(f"Failed to start dashboard service: {e}")
            raise
    
    async def stop(self):
        """Stop dashboard service."""
        try:
            # Stop WebSocket manager
            await connection_manager.stop()
            
            # Stop cache
            await cache_instance.stop()
            
            await super().stop()
            logger.info("Dashboard service stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop dashboard service: {e}")
            raise
    
    async def handle_event(self, event: Event):
        """Handle trading system events.
        
        Args:
            event: Event to handle
        """
        try:
            # Process event based on type
            if event.type == EventType.TRADE:
                await self._handle_trade_event(event)
            elif event.type == EventType.MARKET:
                await self._handle_market_event(event)
            elif event.type == EventType.SIGNAL:
                await self._handle_signal_event(event)
            
        except Exception as e:
            logger.error(f"Error handling event: {e}")
            raise
    
    async def _handle_trade_event(self, event: Event):
        """Handle trade events."""
        # Convert event to trade model
        trade = Trade(
            id=event.event_id,
            instrument=event.data['instrument'],
            direction=event.data['direction'],
            entry_price=event.data['entry_price'],
            exit_price=event.data.get('exit_price'),
            entry_time=event.data['entry_time'],
            exit_time=event.data.get('exit_time'),
            units=event.data['units'],
            pnl=event.data.get('pnl'),
            commission=event.data['commission'],
            status=event.data['status']
        )
        
        # Broadcast trade update
        await connection_manager.broadcast(
            {
                "type": "trade_update",
                "data": trade.dict()
            },
            "trading"
        )
        
        # Invalidate relevant caches
        await cache_instance.delete("trading:trades:all")
    
    async def _handle_market_event(self, event: Event):
        """Handle market data events."""
        # Convert event to market data model
        market_data = MarketData(
            instrument=event.data['instrument'],
            timestamp=event.timestamp,
            bid=event.data['bid'],
            ask=event.data['ask'],
            spread=event.data['ask'] - event.data['bid'],
            volume=event.data.get('volume', 0)
        )
        
        # Broadcast market update
        await connection_manager.broadcast(
            {
                "type": "market_update",
                "data": market_data.dict()
            },
            f"market:{event.data['instrument']}"
        )
        
        # Update cache
        cache_key = f"market:price:{event.data['instrument']}"
        await cache_instance.set(cache_key, market_data.dict(), ttl=5)
    
    async def _handle_signal_event(self, event: Event):
        """Handle trading signal events."""
        # Convert event to signal model
        signal = Signal(
            instrument=event.data['instrument'],
            timestamp=event.timestamp,
            direction=event.data['direction'],
            strength=event.data['strength'],
            strategy=event.data['strategy'],
            indicators=event.data.get('indicators', {}),
            confidence=event.data.get('confidence', 0.0)
        )
        
        # Broadcast signal update
        await connection_manager.broadcast(
            {
                "type": "signal_update",
                "data": signal.dict()
            },
            f"signals:{event.data['instrument']}"
        )
        
        # Update cache
        cache_key = f"signals:{event.data['instrument']}"
        await cache_instance.set(cache_key, signal.dict(), ttl=60)
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the dashboard server.
        
        Args:
            host: Server host
            port: Server port
        """
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        server.run()

# Create dashboard service instance
config = config_manager.load_config()
dashboard_service = DashboardService(EventDispatcher(), config)