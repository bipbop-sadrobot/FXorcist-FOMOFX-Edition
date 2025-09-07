import pandas as pd
from fxorcist.pipeline.vectorized_backtest import sma_strategy_returns, simple_metrics

def _mkdf(n=100):
    d = pd.date_range('2024-01-01', periods=n)
    p = 1 + pd.Series(range(n)) * 0.001
    df = pd.DataFrame({'Date':d,'Open':p,'High':p,'Low':p,'Close':p}).set_index('Date')
    return df

def test_vectorized_returns_and_metrics():
    df = _mkdf(120)
    r = sma_strategy_returns(df, fast=5, slow=20)
    assert len(r) == len(df)
    m = simple_metrics(r)
    assert 'sharpe' in m
