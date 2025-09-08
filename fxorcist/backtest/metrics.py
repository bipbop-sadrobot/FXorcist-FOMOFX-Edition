"""
Performance metrics calculation for backtest results.
"""
from typing import List, Dict, Any
import numpy as np
from datetime import datetime

def calculate_returns(snapshots: List[Dict[str, Any]]) -> np.ndarray:
    """Calculate returns series from equity curve."""
    equity = np.array([s["equity"] for s in snapshots])
    returns = np.diff(equity) / equity[:-1]
    return returns

def calculate_metrics(snapshots: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate performance metrics from backtest snapshots.
    
    Args:
        snapshots: List of portfolio snapshots with timestamp and state
        
    Returns:
        Dict of metrics including:
        - Total return
        - Sharpe ratio
        - Max drawdown
        - Win rate
        - Profit factor
        etc.
    """
    if not snapshots:
        return {}

    # Extract time series
    equity = np.array([s["equity"] for s in snapshots])
    returns = calculate_returns(snapshots)
    
    # Basic metrics
    total_return = (equity[-1] - equity[0]) / equity[0]
    
    # Risk metrics
    volatility = np.std(returns) * np.sqrt(252)  # Annualized
    
    # Sharpe ratio (assuming 0% risk-free rate for simplicity)
    sharpe = np.mean(returns) * np.sqrt(252) / volatility if volatility > 0 else 0
    
    # Drawdown calculation
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_drawdown = np.min(drawdowns)
    
    # Trade metrics
    trades = []
    for s in snapshots:
        if "trades" in s and len(s["trades"]) > 0:
            trades.extend(s["trades"])
    
    if trades:
        winning_trades = sum(1 for t in trades if t["realized_pnl"] > 0)
        losing_trades = sum(1 for t in trades if t["realized_pnl"] < 0)
        
        win_rate = winning_trades / len(trades) if len(trades) > 0 else 0
        
        gross_profit = sum(t["realized_pnl"] for t in trades if t["realized_pnl"] > 0)
        gross_loss = abs(sum(t["realized_pnl"] for t in trades if t["realized_pnl"] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_trade = np.mean([t["realized_pnl"] for t in trades])
        
    else:
        win_rate = 0
        profit_factor = 0
        avg_trade = 0
    
    # Calculate time-based metrics
    timestamps = [datetime.fromisoformat(s["timestamp"].isoformat()) for s in snapshots]
    trading_days = (timestamps[-1] - timestamps[0]).days / 365.25  # Convert to years
    
    if trading_days > 0:
        annual_return = (1 + total_return) ** (1 / trading_days) - 1
        annual_volatility = volatility
        calmar_ratio = abs(annual_return / max_drawdown) if max_drawdown != 0 else 0
    else:
        annual_return = 0
        annual_volatility = 0
        calmar_ratio = 0
    
    return {
        # Return metrics
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe_ratio": sharpe,
        
        # Risk metrics
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        
        # Trade metrics
        "total_trades": len(trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade": avg_trade,
        
        # Additional metrics
        "final_equity": equity[-1],
        "peak_equity": np.max(equity),
        "trading_days": trading_days * 365.25,  # Convert back to days
    }

def calculate_rolling_metrics(
    snapshots: List[Dict[str, Any]],
    window: int = 252,  # Default to 1 year of trading days
) -> Dict[str, List[float]]:
    """
    Calculate rolling performance metrics.
    
    Args:
        snapshots: List of portfolio snapshots
        window: Rolling window size in periods
        
    Returns:
        Dict of rolling metric series
    """
    if len(snapshots) < window:
        return {}

    equity = np.array([s["equity"] for s in snapshots])
    returns = calculate_returns(snapshots)
    
    # Initialize arrays
    rolling_sharpe = np.zeros(len(snapshots) - window)
    rolling_volatility = np.zeros(len(snapshots) - window)
    rolling_returns = np.zeros(len(snapshots) - window)
    
    # Calculate rolling metrics
    for i in range(len(snapshots) - window):
        window_returns = returns[i:i+window]
        
        # Rolling Sharpe
        vol = np.std(window_returns) * np.sqrt(252)
        if vol > 0:
            rolling_sharpe[i] = np.mean(window_returns) * np.sqrt(252) / vol
            
        # Rolling volatility
        rolling_volatility[i] = vol
        
        # Rolling returns
        rolling_returns[i] = (
            equity[i + window] - equity[i]
        ) / equity[i]
    
    return {
        "rolling_sharpe": rolling_sharpe.tolist(),
        "rolling_volatility": rolling_volatility.tolist(),
        "rolling_returns": rolling_returns.tolist(),
    }