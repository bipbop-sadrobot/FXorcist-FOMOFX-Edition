"""
Portfolio and trading API endpoints.

Implements routes for portfolio management, trade execution, and position tracking.
"""

from fastapi import APIRouter, Depends, HTTPException, Security
from typing import List, Optional
from datetime import datetime
import logging

from ..models import (
    Trade, Position, PortfolioSummary, PerformanceMetrics,
    OrderDirection, TradeStatus
)
from ..auth import auth_service, User
from ..cache import cached, cache_instance
from ..websocket import connection_manager

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/portfolio",
    tags=["portfolio"],
    responses={404: {"description": "Not found"}}
)

@router.get("/summary", response_model=PortfolioSummary)
@cached("portfolio:summary", ttl=60)  # Cache for 1 minute
async def get_portfolio_summary(
    current_user: User = Security(
        auth_service.get_current_active_user,
        scopes=["read:portfolio"]
    )
) -> PortfolioSummary:
    """Get current portfolio summary."""
    try:
        # Get portfolio data from trading system
        summary = PortfolioSummary(
            balance=100000.0,  # Example data
            equity=105000.0,
            margin_used=20000.0,
            margin_available=80000.0,
            positions_count=2,
            open_trades_count=2,
            closed_trades_count=10,
            daily_pnl=500.0,
            total_pnl=5000.0
        )
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get portfolio summary"
        )

@router.get("/positions", response_model=List[Position])
@cached("portfolio:positions", ttl=30)  # Cache for 30 seconds
async def get_positions(
    current_user: User = Security(
        auth_service.get_current_active_user,
        scopes=["read:portfolio"]
    )
) -> List[Position]:
    """Get current open positions."""
    try:
        # Get positions from trading system
        positions = [
            Position(
                instrument="EUR_USD",
                direction=OrderDirection.LONG,
                units=100000,
                entry_price=1.1850,
                current_price=1.1860,
                unrealized_pnl=100.0,
                realized_pnl=0.0,
                stop_loss=1.1800,
                take_profit=1.1900
            ),
            Position(
                instrument="GBP_USD",
                direction=OrderDirection.SHORT,
                units=50000,
                entry_price=1.3750,
                current_price=1.3740,
                unrealized_pnl=50.0,
                realized_pnl=0.0,
                stop_loss=1.3800,
                take_profit=1.3700
            )
        ]
        
        return positions
        
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get positions"
        )

@router.get("/trades", response_model=List[Trade])
@cached("portfolio:trades", ttl=60)
async def get_trades(
    status: Optional[TradeStatus] = None,
    limit: int = 100,
    current_user: User = Security(
        auth_service.get_current_active_user,
        scopes=["read:portfolio"]
    )
) -> List[Trade]:
    """Get trade history."""
    try:
        # Get trades from trading system
        trades = [
            Trade(
                id="trade1",
                instrument="EUR_USD",
                direction=OrderDirection.LONG,
                entry_price=1.1850,
                exit_price=1.1900,
                entry_time=datetime.utcnow(),
                exit_time=datetime.utcnow(),
                units=100000,
                pnl=500.0,
                commission=10.0,
                status=TradeStatus.CLOSED,
                stop_loss=1.1800,
                take_profit=1.1900
            )
        ]
        
        # Filter by status if specified
        if status:
            trades = [t for t in trades if t.status == status]
        
        return trades[:limit]
        
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get trades"
        )

@router.get("/performance", response_model=PerformanceMetrics)
@cached("portfolio:performance", ttl=300)  # Cache for 5 minutes
async def get_performance_metrics(
    current_user: User = Security(
        auth_service.get_current_active_user,
        scopes=["read:portfolio"]
    )
) -> PerformanceMetrics:
    """Get portfolio performance metrics."""
    try:
        # Calculate performance metrics
        metrics = PerformanceMetrics(
            total_return=0.05,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=-0.02,
            win_rate=0.6,
            profit_factor=1.8,
            avg_win=100.0,
            avg_loss=-50.0,
            expectancy=40.0,
            recovery_factor=2.5
        )
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get performance metrics"
        )

@router.post("/close-position/{instrument}")
async def close_position(
    instrument: str,
    current_user: User = Security(
        auth_service.get_current_active_user,
        scopes=["write:trades"]
    )
):
    """Close an open position."""
    try:
        # Close position in trading system
        # Broadcast position update
        await connection_manager.broadcast(
            message={
                "type": "position_closed",
                "data": {
                    "instrument": instrument,
                    "timestamp": datetime.utcnow().isoformat()
                }
            },
            channel="portfolio"
        )
        
        # Invalidate relevant caches
        await cache_instance.delete("portfolio:positions")
        await cache_instance.delete("portfolio:summary")
        
        return {"status": "success", "message": f"Position closed for {instrument}"}
        
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to close position"
        )