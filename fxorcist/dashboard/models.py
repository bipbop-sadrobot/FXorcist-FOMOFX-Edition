"""
Dashboard API models and schemas.

Implements Pydantic models for request/response validation and serialization.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union
from decimal import Decimal
from pydantic import BaseModel, Field, validator
from enum import Enum

class OrderType(str, Enum):
    """Order types supported by the system."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    TAKE_PROFIT = "TAKE_PROFIT"

class OrderDirection(str, Enum):
    """Order directions."""
    LONG = "LONG"
    SHORT = "SHORT"

class OrderStatus(str, Enum):
    """Order statuses."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class TradeStatus(str, Enum):
    """Trade statuses."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"

class Trade(BaseModel):
    """Trade model with full trade information."""
    
    id: str = Field(..., description="Unique trade identifier")
    instrument: str = Field(..., description="Trading instrument")
    direction: OrderDirection
    entry_price: Decimal = Field(..., ge=0)
    exit_price: Optional[Decimal] = Field(None, ge=0)
    entry_time: datetime
    exit_time: Optional[datetime]
    units: int = Field(..., gt=0)
    pnl: Optional[Decimal]
    commission: Decimal = Field(..., ge=0)
    status: TradeStatus
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    
    @validator('exit_price')
    def validate_exit_price(cls, v, values):
        """Validate exit price based on trade direction."""
        if v is not None and 'direction' in values and 'entry_price' in values:
            if values['direction'] == OrderDirection.LONG and v < values['entry_price']:
                raise ValueError("Exit price must be higher than entry price for long trades")
            elif values['direction'] == OrderDirection.SHORT and v > values['entry_price']:
                raise ValueError("Exit price must be lower than entry price for short trades")
        return v

class Position(BaseModel):
    """Current position information."""
    
    instrument: str
    direction: OrderDirection
    units: int = Field(..., gt=0)
    entry_price: Decimal = Field(..., ge=0)
    current_price: Decimal = Field(..., ge=0)
    unrealized_pnl: Decimal
    realized_pnl: Decimal = Field(default=0)
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]

class PortfolioSummary(BaseModel):
    """Portfolio summary information."""
    
    balance: Decimal = Field(..., ge=0)
    equity: Decimal = Field(..., ge=0)
    margin_used: Decimal = Field(..., ge=0)
    margin_available: Decimal = Field(..., ge=0)
    positions_count: int = Field(..., ge=0)
    open_trades_count: int = Field(..., ge=0)
    closed_trades_count: int = Field(..., ge=0)
    daily_pnl: Decimal
    total_pnl: Decimal

class PerformanceMetrics(BaseModel):
    """Trading performance metrics."""
    
    total_return: float = Field(..., description="Total return percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    max_drawdown: float = Field(..., le=0, description="Maximum drawdown percentage")
    win_rate: float = Field(..., ge=0, le=1, description="Win rate")
    profit_factor: float = Field(..., ge=0, description="Profit factor")
    avg_win: float = Field(..., description="Average winning trade")
    avg_loss: float = Field(..., description="Average losing trade")
    expectancy: float = Field(..., description="Trade expectancy")
    recovery_factor: float = Field(..., ge=0, description="Recovery factor")

class MarketData(BaseModel):
    """Real-time market data."""
    
    instrument: str
    timestamp: datetime
    bid: Decimal = Field(..., ge=0)
    ask: Decimal = Field(..., ge=0)
    spread: Decimal = Field(..., ge=0)
    volume: int = Field(..., ge=0)

class Signal(BaseModel):
    """Trading signal information."""
    
    instrument: str
    timestamp: datetime
    direction: OrderDirection
    strength: float = Field(..., ge=0, le=1)
    strategy: str
    indicators: Dict[str, float]
    confidence: float = Field(..., ge=0, le=1)

class SystemStatus(BaseModel):
    """System health and status information."""
    
    status: str = Field(..., description="Overall system status")
    uptime: float = Field(..., description="System uptime in seconds")
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    memory_usage: float = Field(..., ge=0, description="Memory usage in bytes")
    active_connections: int = Field(..., ge=0, description="Active WebSocket connections")
    last_update: datetime = Field(..., description="Last status update time")
    components_status: Dict[str, str] = Field(..., description="Individual component statuses")

class ErrorResponse(BaseModel):
    """API error response model."""
    
    error: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[Dict] = Field(None, description="Additional error details")

class HealthCheck(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="Service version")
    checks: Dict[str, bool] = Field(..., description="Individual component health checks")
    latency: Dict[str, float] = Field(..., description="Component latencies in milliseconds")