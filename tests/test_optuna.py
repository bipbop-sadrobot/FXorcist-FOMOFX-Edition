import pandas as pd
from fxorcist.ml.optuna_runner import run_optuna
from pathlib import Path

def _mkdf(n=120):
    d = pd.date_range('2024-01-01', periods=n)
    p = 1 + pd.Series(range(n)) * 0.001
    df = pd.DataFrame({'Date':d,'Open':p,'High':p,'Low':p,'Close':p}).set_index('Date')
    df['Returns'] = df['Close'].pct_change().fillna(0)
    return df

def test_optuna_quick(tmp_path):
    df = _mkdf(120)
    out = tmp_path / 'best.yaml'
    res = run_optuna(df, n_trials=2, seed=1, out_path=str(out), storage=f"sqlite:///{tmp_path/'study.db'}")
    assert 'best_params' in res
    assert out.exists()
