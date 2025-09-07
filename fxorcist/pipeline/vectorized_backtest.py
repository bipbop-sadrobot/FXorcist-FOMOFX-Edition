import pandas as pd
from typing import Dict

def sma_strategy_returns(df: pd.DataFrame, fast: int = 10, slow: int = 50, transaction_cost: float = 1e-4) -> pd.Series:
    """
    Calculate strategy returns using a simple moving average crossover strategy.
    
    Args:
        df: DataFrame with at least a 'Close' column
        fast: Fast moving average period
        slow: Slow moving average period
        transaction_cost: Cost per trade as a fraction
        
    Returns:
        pd.Series of strategy returns
    """
    fast_ma = df["Close"].rolling(window=fast, min_periods=1).mean()
    slow_ma = df["Close"].rolling(window=slow, min_periods=1).mean()
    signal = (fast_ma > slow_ma).astype(float)
    returns = df["Close"].pct_change().fillna(0.0)
    strat = signal.shift(1).fillna(0.0) * returns
    trades = signal.diff().abs().fillna(0.0)
    strat = strat - trades * transaction_cost
    return strat

def simple_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate key performance metrics from a series of returns.
    
    Args:
        returns: Series of period returns
        
    Returns:
        Dictionary containing:
            - sharpe: Annualized Sharpe ratio (using 252 trading days)
            - total_return: Total return as a fraction
            - max_drawdown: Maximum drawdown as a fraction
    """
    if returns is None or len(returns) == 0:
        return {"sharpe": float("nan"), "total_return": 0.0, "max_drawdown": 0.0}
    
    # Calculate metrics with explicit handling of edge cases
    avg = returns.mean()
    sd = returns.std(ddof=0) if returns.std(ddof=0) != 0 else 1e-9
    sharpe = (avg / sd) * (252 ** 0.5)  # Annualized with 252 trading days
    
    # Calculate cumulative returns and drawdown
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()
    total_ret = float(cumulative.iloc[-1] - 1.0)
    
    return {
        "sharpe": float(sharpe),
        "total_return": total_ret,
        "max_drawdown": float(max_dd)
    }