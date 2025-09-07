"""
Vectorized backtest utilities:
- SMA strategy with signal shift, transaction_cost (fixed or dynamic)
- position sizing and leverage
- risk metrics including Sortino & rolling Sharpe
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Callable

def _apply_transaction_costs(strat_returns: pd.Series, trades: pd.Series, transaction_cost: Union[float, Callable[[pd.Series], pd.Series]]):
    if callable(transaction_cost):
        tc = transaction_cost(trades)
    else:
        tc = trades * float(transaction_cost)
    return strat_returns - tc

def sma_strategy_returns(df: pd.DataFrame, fast: int = 10, slow: int = 50,
                         transaction_cost: Union[float, Callable[[pd.Series], pd.Series]] = 1e-4,
                         leverage: float = 1.0) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    fast_ma = df['Close'].rolling(window=fast, min_periods=1).mean()
    slow_ma = df['Close'].rolling(window=slow, min_periods=1).mean()
    signal = (fast_ma > slow_ma).astype(float)
    returns = df['Close'].pct_change().fillna(0.0)
    strat = signal.shift(1).fillna(0.0) * returns * leverage
    trades = signal.diff().abs().fillna(0.0)
    strat = _apply_transaction_costs(strat, trades, transaction_cost)
    return strat

def simple_metrics(returns: pd.Series) -> Dict[str, float]:
    if returns is None or len(returns) == 0:
        return {"sharpe": float('nan'), "sortino": float('nan'), "total_return": 0.0, "max_drawdown": 0.0}
    avg = returns.mean()
    sd = returns.std(ddof=0) if returns.std(ddof=0) != 0 else 1e-9
    sharpe = float(avg / sd * (252 ** 0.5))
    downside = returns[returns < 0]
    dd = downside.std(ddof=0) if len(downside) > 0 else 0.0
    sortino = float(avg / (dd if dd > 0 else 1e-9) * (252 ** 0.5))
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = float(drawdown.min())
    total_ret = float(cumulative.iloc[-1] - 1.0) if len(cumulative) > 0 else 0.0
    return {"sharpe": sharpe, "sortino": sortino, "total_return": total_ret, "max_drawdown": max_dd}
