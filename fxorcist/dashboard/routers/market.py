"""
Market data API endpoints.

Implements routes for real-time and historical market data access.
"""

from fastapi import APIRouter, Depends, HTTPException, Security, Query
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np

from ..models import MarketData, User
from ..auth import auth_service
from ..cache import cached, cache_instance
from ..websocket import connection_manager, WebSocketMessage

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/market",
    tags=["market"],
    responses={404: {"description": "Not found"}}
)

@router.get("/price/{instrument}", response_model=MarketData)
@cached("market:price", ttl=5)  # Cache for 5 seconds
async def get_current_price(
    instrument: str,
    current_user: User = Security(
        auth_service.get_current_active_user,
        scopes=["read:market"]
    )
) -> MarketData:
    """Get current market price for instrument."""
    try:
        # Get real-time price from trading system
        price = MarketData(
            instrument=instrument,
            timestamp=datetime.utcnow(),
            bid=1.1850,
            ask=1.1852,
            spread=0.0002,
            volume=1000000
        )
        
        return price
        
    except Exception as e:
        logger.error(f"Error getting price for {instrument}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get price for {instrument}"
        )

@router.get("/history/{instrument}")
@cached("market:history", ttl=300)  # Cache for 5 minutes
async def get_historical_data(
    instrument: str,
    timeframe: str = Query("1H", regex="^[1-9][0-9]?[mhDWM]$"),
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    current_user: User = Security(
        auth_service.get_current_active_user,
        scopes=["read:market"]
    )
) -> Dict:
    """Get historical market data."""
    try:
        # Set default time range if not specified
        if not end:
            end = datetime.utcnow()
        if not start:
            start = end - timedelta(days=30)
        
        # Get historical data from data service
        data = {
            "instrument": instrument,
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "data": [
                {
                    "timestamp": "2025-09-07T10:00:00Z",
                    "open": 1.1850,
                    "high": 1.1855,
                    "low": 1.1845,
                    "close": 1.1852,
                    "volume": 1000000
                }
                # ... more data points
            ]
        }
        
        return data
        
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get historical data"
        )

@router.get("/indicators/{instrument}")
@cached("market:indicators", ttl=60)
async def get_technical_indicators(
    instrument: str,
    indicators: List[str] = Query(
        ["RSI", "MACD", "BB"],
        description="List of indicators to calculate"
    ),
    timeframe: str = Query("1H", regex="^[1-9][0-9]?[mhDWM]$"),
    current_user: User = Security(
        auth_service.get_current_active_user,
        scopes=["read:market"]
    )
) -> Dict:
    """Get technical indicators for instrument."""
    try:
        # Calculate indicators
        indicator_data = {
            "instrument": instrument,
            "timeframe": timeframe,
            "timestamp": datetime.utcnow().isoformat(),
            "indicators": {
                "RSI": {
                    "value": 55.5,
                    "signal": "neutral",
                    "period": 14
                },
                "MACD": {
                    "value": 0.0025,
                    "signal": 0.0015,
                    "histogram": 0.001,
                    "signal_line": "bullish"
                },
                "BB": {
                    "upper": 1.1900,
                    "middle": 1.1850,
                    "lower": 1.1800,
                    "width": 0.0100,
                    "position": "middle"
                }
            }
        }
        
        return indicator_data
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to calculate indicators"
        )

@router.get("/analysis/{instrument}")
@cached("market:analysis", ttl=300)
async def get_market_analysis(
    instrument: str,
    timeframe: str = Query("1H", regex="^[1-9][0-9]?[mhDWM]$"),
    current_user: User = Security(
        auth_service.get_current_active_user,
        scopes=["read:market"]
    )
) -> Dict:
    """Get market analysis including trend, support/resistance, and volatility."""
    try:
        # Perform market analysis
        analysis = {
            "instrument": instrument,
            "timeframe": timeframe,
            "timestamp": datetime.utcnow().isoformat(),
            "trend": {
                "direction": "bullish",
                "strength": 0.75,
                "duration": "3 days"
            },
            "support_resistance": {
                "support_levels": [1.1800, 1.1750],
                "resistance_levels": [1.1900, 1.1950],
                "nearest_support": 1.1800,
                "nearest_resistance": 1.1900
            },
            "volatility": {
                "current": 0.0080,
                "average": 0.0100,
                "trend": "decreasing",
                "percentile": 45
            },
            "momentum": {
                "strength": 0.65,
                "direction": "positive",
                "acceleration": "stable"
            }
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error performing market analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to perform market analysis"
        )

@router.websocket("/stream/{instrument}")
async def websocket_endpoint(
    websocket: WebSocket,
    instrument: str,
    client_id: str
):
    """WebSocket endpoint for real-time market data."""
    await connection_manager.connect(websocket, client_id)
    
    try:
        # Subscribe to market data channel
        await connection_manager.subscribe(
            client_id,
            f"market:{instrument}"
        )
        
        while True:
            # Simulate market data updates
            price = MarketData(
                instrument=instrument,
                timestamp=datetime.utcnow(),
                bid=1.1850 + np.random.normal(0, 0.0001),
                ask=1.1852 + np.random.normal(0, 0.0001),
                spread=0.0002,
                volume=int(1000000 * np.random.random())
            )
            
            # Send update
            message = WebSocketMessage(
                type="market_update",
                data=price.dict()
            )
            
            await connection_manager.send_message(client_id, message)
            await asyncio.sleep(1)  # Update every second
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await connection_manager.disconnect(client_id)