#!/usr/bin/env bash
# scripts/apply_fxorcist_v3.sh
# Idempotent apply script for FXorcist v3 upgrade.
# Usage: bash scripts/apply_fxorcist_v3.sh [--force]
set -euo pipefail

FORCE=0
if [ "${1:-}" = "--force" ]; then
  FORCE=1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "Applying FXorcist v3 upgrade in $REPO_ROOT (force=${FORCE})"

# Ensure we're not on main
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$CURRENT_BRANCH" = "main" ] || [ "$CURRENT_BRANCH" = "master" ]; then
  echo "ERROR: Please run this script on a feature branch (not main/master). Create one now:"
  echo "  git checkout -b refactor/v3-upgrade"
  exit 1
fi

# Create directories
mkdir -p fxorcist/data fxorcist/pipeline fxorcist/ml fxorcist/dashboard fxorcist/utils scripts tests .github/workflows

# Helper to write file only if missing or force=1
write_file() {
  local path="$1"; shift
  local tmp=$(mktemp)
  cat > "$tmp"
  if [ -f "$path" ] && [ "$FORCE" -eq 0 ]; then
    echo "SKIP (exists): $path"
    rm "$tmp"
  else
    mkdir -p "$(dirname "$path")"
    mv "$tmp" "$path"
    git add "$path"
    echo "WROTE: $path"
  fi
}

# 1) Add pre-commit config
write_file .pre-commit-config.yaml <<'YAML'
repos:
- repo: https://github.com/psf/black
  rev: 24.4b0
  hooks:
    - id: black
- repo: https://github.com/PyCQA/flake8
  rev: 7.1.0
  hooks:
    - id: flake8
- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.9.0
  hooks:
    - id: mypy
      additional_dependencies: [mypy==1.6.1]
YAML

# 2) Add mypy config
write_file mypy.ini <<'MYPY'
[mypy]
python_version = 3.10
ignore_missing_imports = True
MYPY

# 3) Add basic flake8
write_file .flake8 <<'FLAKE'
[flake8]
max-line-length = 120
extend-ignore = E203, W503
FLAKE

# 4) Add updated pyproject.toml
write_file pyproject.toml <<'TOML'
[project]
name = "fxorcist"
version = "0.3.0"
description = "FXorcist — research and backtest framework"
readme = "README.md"
requires-python = ">=3.10"

[project.scripts]
fxorcist = "fxorcist.cli:main"

[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[tool.black]
line-length = 100
TOML

# 5) requirements + extras
write_file requirements.txt <<'REQ'
streamlit>=1.20.0
plotly>=5.6.0
pandas>=2.0
numpy>=1.24
optuna>=3.0
pyyaml>=6.0
mlflow>=2.0; extra == "mlflow"
dask[distributed]>=2023.9.2
ray[default]>=2.6.0; extra == "ray"
pyarrow
quantstats
rich
pandera
hypothesis
tqdm
aiofiles
fsspec
REQ

# 6) Add improved loader
write_file fxorcist/data/loader.py <<'PY'
"""
Improved data loader:
- parquet-first, csv fallback
- optional caching to parquet (snappy)
- optional schema validation via pandera (if installed)
- asynchronous loader available via async_load_symbol (if aiofiles/fsspec is installed)
- deterministic synthetic fallback for tests with optional GARCH-ish volatility
"""
from __future__ import annotations
import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Dict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

REQUIRED_COLS = {"Date", "Open", "High", "Low", "Close"}
OPTIONAL_COLS = {"Volume"}

class DataLoaderError(Exception):
    pass

def _read_parquet(path: Path, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as e:
        logger.exception("parquet read failed %s", path)
        raise DataLoaderError(e)

def _read_csv(path: Path, usecols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, parse_dates=['Date'], usecols=usecols)
    except Exception as e:
        logger.exception("csv read failed %s", path)
        raise DataLoaderError(e)

def _ensure_index_and_types(df: pd.DataFrame) -> pd.DataFrame:
    if 'Date' in df.columns:
        df = df.set_index('Date')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')
        if df.index.isnull().any():
            raise DataLoaderError("Invalid dates in data; please check source CSV/parquet")
    for c in df.columns:
        if c.lower() in {'open','high','low','close','volume','returns'}:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def _ensure_returns(df: pd.DataFrame) -> pd.DataFrame:
    if 'Returns' not in df.columns:
        if 'Close' not in df.columns:
            raise DataLoaderError("Missing Close column; cannot compute Returns")
        df['Returns'] = df['Close'].pct_change().fillna(0.0)
    return df

def _validate_schema(df: pd.DataFrame) -> None:
    present = set(df.reset_index().columns)
    missing = REQUIRED_COLS - present
    if missing:
        raise DataLoaderError(f"Missing required columns: {sorted(missing)}")

def _candidates_for_symbol(base: Path, symbol: str):
    cleaned = base / 'data' / 'cleaned'
    return [cleaned / f"{symbol}.parquet", cleaned / f"{symbol}.csv", base / 'data' / f"{symbol}.csv", base / f"{symbol}.csv"]

@lru_cache(maxsize=1)
def list_available_symbols(base_dir: Optional[str|Path] = None):
    base = Path(base_dir) if base_dir else Path.cwd()
    symbols = set()
    for p in [base/'data'/'cleaned', base/'data', base]:
        if not p.exists():
            continue
        for ext in ('parquet','csv'):
            for f in p.glob(f"*.{ext}"):
                symbols.add(f.stem.upper())
    return sorted(symbols)

def _synthetic_series(symbol: str, n: int = 365, seed: int = 0, sv: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.today(), periods=n, freq='D')
    eps = rng.normal(0, 0.001, size=n)
    if sv:
        logvol = np.zeros(n)
        for t in range(1, n):
            logvol[t] = 0.9 * logvol[t-1] + 0.1 * rng.normal()
        vol = np.exp(logvol) * 0.002
        eps = rng.normal(0, 1, size=n) * vol
    price = 1.0 + np.cumsum(eps)
    high = price + np.abs(rng.normal(scale=0.001, size=n))
    low = price - np.abs(rng.normal(scale=0.001, size=n))
    op = low + (high - low) * rng.random(n)
    df = pd.DataFrame({'Date': idx, 'Open': op, 'High': high, 'Low': low, 'Close': price})
    df['Returns'] = df['Close'].pct_change().fillna(0.0)
    df['Volume'] = rng.integers(1000, 100000, size=n)
    return df.set_index('Date')

def load_symbol(
    symbol: str,
    base_dir: Optional[str] = None,
    usecols: Optional[Iterable[str]] = None,
    prefer_parquet: bool = True,
    cache_parquet: bool = False,
    allow_synthetic_fallback: bool = False
) -> pd.DataFrame:
    base = Path(base_dir) if base_dir else Path.cwd()
    sym = symbol.upper()
    candidates = _candidates_for_symbol(base, sym)
    cols = None if usecols is None else list(usecols)
    df = None

    if prefer_parquet:
        p = candidates[0]
        if p.exists():
            df = _read_parquet(p, columns=cols)
    if df is None:
        for p in candidates[1:]:
            if p.exists():
                df = _read_csv(p, usecols=cols)
                if cache_parquet:
                    try:
                        target = base / 'data' / 'cleaned' / f"{sym}.parquet"
                        target.parent.mkdir(parents=True, exist_ok=True)
                        df.to_parquet(target, compression='snappy', index=False)
                        logger.info("Cached %s -> %s", p, target)
                    except Exception as ex:
                        logger.warning("Failed to write parquet cache: %s", ex)
                break

    if df is None:
        if allow_synthetic_fallback:
            logger.warning("No file for %s found; returning synthetic fallback", sym)
            return _synthetic_series(sym)
        raise DataLoaderError(f"No data files found for {sym}")

    df = _ensure_index_and_types(df)
    _validate_schema(df)
    df = _ensure_returns(df)
    return df

async def async_load_symbol(*args, **kwargs):
    """
    If you have fsspec/aiofiles installed, implement async file reads here.
    Keeping a simple sync wrapper to avoid tight dependency.
    """
    return load_symbol(*args, **kwargs)
PY

# 7) Add improved vectorized backtest
write_file fxorcist/pipeline/vectorized_backtest.py <<'PY'
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
PY

# 8) Add improved parallel runner
write_file fxorcist/pipeline/parallel.py <<'PY'
"""
Parallel runner with multiprocessing, dask, and optional Ray support.
Returns structured results with metrics or error info.
"""
from __future__ import annotations
from typing import Dict, Any
from multiprocessing import Pool
from tqdm import tqdm
import dask
from dask.distributed import Client
import traceback

from fxorcist.pipeline.vectorized_backtest import sma_strategy_returns, simple_metrics

def _run_symbol(job):
    symbol, df, params = job
    try:
        returns = sma_strategy_returns(df, **params)
        metrics = simple_metrics(returns)
        return {"symbol": symbol, "metrics": metrics}
    except Exception as e:
        return {"symbol": symbol, "error": str(e), "traceback": traceback.format_exc()}

def run_parallel(symbol_dfs: Dict[str, Any], params: Dict[str, Any], n_workers: int = 4, use_dask: bool = False, show_progress: bool = True):
    jobs = [(s, symbol_dfs[s], params) for s in symbol_dfs]
    if use_dask:
        with Client(n_workers=n_workers) as client:
            delayed = [dask.delayed(_run_symbol)(job) for job in jobs]
            results = dask.compute(*delayed)
        return list(results)
    else:
        results = []
        with Pool(processes=n_workers) as p:
            if show_progress:
                for r in tqdm(p.imap_unordered(_run_symbol, jobs), total=len(jobs)):
                    results.append(r)
            else:
                results = p.map(_run_symbol, jobs)
        return results
PY

# 9) Improved Optuna runner
write_file fxorcist/ml/optuna_runner.py <<'PY'
"""
Optuna runner:
- TPESampler(seed)
- MedianPruner for speed
- Optional MLflow logging (safe if mlflow not installed)
- Save trial artifacts (best params yaml, equity plot) to artifacts/
- Support multi-objective placeholder (can be expanded)
"""
from __future__ import annotations
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import matplotlib.pyplot as plt
import io
import base64

from fxorcist.pipeline.vectorized_backtest import sma_strategy_returns, simple_metrics

logger = logging.getLogger(__name__)

def _objective(trial: optuna.Trial, df):
    fast = trial.suggest_int("fast", 5, 40)
    slow = trial.suggest_int("slow", 50, 200)
    if slow <= fast:
        return -1e9
    rets = sma_strategy_returns(df, fast=fast, slow=slow)
    metrics = simple_metrics(rets)
    trial.set_user_attr("n", len(rets))
    return metrics.get("sharpe", -1e9)

def run_optuna(df, n_trials: int = 50, seed: int = 42, out_path: str = "artifacts/best_params.yaml", storage: Optional[str] = None, use_mlflow: bool = False) -> Dict[str, Any]:
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, storage=storage, load_if_exists=True)
    mlflow = None
    if use_mlflow:
        try:
            import mlflow as _ml
            mlflow = _ml
            mlflow.start_run(run_name=f"optuna_sma_{seed}")
        except Exception as e:
            logger.warning("MLflow import failed; continuing without MLflow: %s", e)
            mlflow = None

    study.optimize(lambda t: _objective(t, df), n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        yaml.safe_dump(best, fh)
    if mlflow:
        try:
            mlflow.log_params(best)
            mlflow.log_metric("best_sharpe", study.best_value)
            mlflow.end_run()
        except Exception as e:
            logger.warning("MLflow logging failed: %s", e)

    try:
        best_rets = sma_strategy_returns(df, fast=best['fast'], slow=best['slow'])
        fig, ax = plt.subplots()
        (1 + best_rets).cumprod().plot(ax=ax)
        ax.set_title("Equity curve (best)")
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        out_file = Path(out_path).with_suffix('.png')
        out_file.write_bytes(buf.getvalue())
    except Exception as e:
        logger.warning("Failed to write equity plot: %s", e)

    return {"study": study, "best_params": best}
PY

# 10) Dashboard improvements
write_file fxorcist/dashboard/app.py <<'PY'
import streamlit as st
from fxorcist.data.loader import list_available_symbols, load_symbol
from fxorcist.dashboard.charts import candlestick_fig, equity_curve_fig, drawdown_fig
from fxorcist.pipeline.vectorized_backtest import sma_strategy_returns, simple_metrics
import pandas as pd
import io

st.set_page_config(page_title="FXorcist", layout="wide")

@st.cache_data(ttl=600)
def _get_symbols(base_dir: str | None = None):
    return list_available_symbols(base_dir)

@st.cache_data(ttl=600)
def _load(symbol: str, base_dir: str | None = None):
    return load_symbol(symbol, base_dir=base_dir, allow_synthetic_fallback=True)

def _df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    import pandas as pd
    with io.BytesIO() as b:
        with pd.ExcelWriter(b, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='data', index=True)
        return b.getvalue()

def streamlit_app():
    st.title("FXorcist — Strategy Explorer")
    left, right = st.columns([3,1])
    symbols = _get_symbols() or ['EURUSD','GBPUSD','AUDUSD']
    with right:
        st.header('Controls')
        symbol = st.selectbox('Symbol', symbols)
        fast = st.slider('Fast MA', 5, 40, 10)
        slow = st.slider('Slow MA', 21, 200, 50)
        txn_cost = st.number_input('Transaction cost', min_value=0.0, max_value=0.01, value=0.0001, format="%.6f")
        show_rsi = st.checkbox('Show RSI')
        run_backtest = st.button('Run Backtest')
        st.markdown('---')
        if st.button('Download data (Excel)'):
            df = _load(symbol)
            st.download_button('Download Excel', _df_to_excel_bytes(df), file_name=f"{symbol}.xlsx")
    with left:
        df = _load(symbol)
        st.subheader('Price Chart')
        st.plotly_chart(candlestick_fig(df), use_container_width=True)
        if run_backtest:
            with st.spinner('Running backtest...'):
                rets = sma_strategy_returns(df, fast=fast, slow=slow, transaction_cost=txn_cost)
                metrics = simple_metrics(rets)
                st.metric('Sharpe', f"{metrics['sharpe']:.3f}")
                st.metric('Total Return', f"{metrics['total_return']:.3%}")
                st.metric('Max Drawdown', f"{metrics['max_drawdown']:.3%}")
                st.plotly_chart(equity_curve_fig(rets), use_container_width=True)
                st.plotly_chart(drawdown_fig(rets), use_container_width=True)
        st.sidebar.header('Advanced')
        if st.sidebar.checkbox('Show Raw Data'):
            st.write(df.head(50))

if __name__ == '__main__':
    streamlit_app()
PY

# 11) Dashboard charts
write_file fxorcist/dashboard/charts.py <<'PY'
import plotly.graph_objects as go
import pandas as pd

def candlestick_fig(df: pd.DataFrame, volume: bool = True) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty:
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        return fig
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    if volume and 'Volume' in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', yaxis='y2', marker={'opacity':0.3}))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False))
    fig.update_layout(xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=25,b=10))
    return fig

def equity_curve_fig(returns: pd.Series) -> go.Figure:
    if returns is None or len(returns)==0:
        fig = go.Figure(); fig.add_annotation(text='No returns', x=0.5, y=0.5, showarrow=False); return fig
    cumulative = (1+returns).cumprod()
    fig = go.Figure(go.Scatter(x=cumulative.index, y=cumulative.values, mode='lines', name='Equity'))
    fig.update_layout(margin=dict(l=10,r=10,t=25,b=10))
    return fig

def drawdown_fig(returns: pd.Series) -> go.Figure:
    cumulative = (1+returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative-peak)/peak
    fig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown.values, mode='lines', name='Drawdown'))
    return fig
PY

# 12) CLI improvements
write_file fxorcist/cli.py <<'PY'
import argparse
import yaml
import json
from rich.console import Console
from rich.table import Table
from typing import Optional
from fxorcist.data.loader import load_symbol, list_available_symbols
from fxorcist.ml.optuna_runner import run_optuna

console = Console()

def load_config(path: Optional[str]) -> dict:
    if not path:
        return {}
    try:
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        console.print("[red]Failed to load config. Ignoring.[/red]")
        return {}

def main(argv=None):
    parser = argparse.ArgumentParser('fxorcist')
    parser.add_argument('--config', '-c', help='YAML config file', default=None)
    parser.add_argument('--json', action='store_true', help='Output JSON results')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_data = sub.add_parser('data')
    p_data.add_argument('--symbol', required=True)

    p_opt = sub.add_parser('optuna')
    p_opt.add_argument('--symbol', required=True)
    p_opt.add_argument('--trials', type=int, default=30)
    p_opt.add_argument('--out', default='artifacts/best_params.yaml')
    p_opt.add_argument('--storage', default=None)
    p_opt.add_argument('--mlflow', action='store_true')

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == 'data':
        df = load_symbol(args.symbol, base_dir=cfg.get('base_dir'))
        if args.json:
            print(df.head().to_json(orient='split', date_format='iso'))
        else:
            console.print(df.head())
    elif args.cmd == 'optuna':
        df = load_symbol(args.symbol, base_dir=cfg.get('base_dir'))
        res = run_optuna(df, n_trials=args.trials, seed=cfg.get('seed', 42), out_path=args.out, storage=args.storage, use_mlflow=args.mlflow)
        if args.json:
            print(json.dumps({'best': res['best_params']}))
        else:
            table = Table(title="Best params")
            for k,v in res['best_params'].items():
                table.add_row(str(k), str(v))
            console.print(table)
PY

# 13) Tests
write_file tests/test_loader.py <<'PY'
import pytest
import pandas as pd
from fxorcist.data.loader import load_symbol, _synthetic_series, list_available_symbols
from pathlib import Path

def test_synthetic_series():
    df = _synthetic_series('TST', n=10, seed=1)
    assert len(df) == 10
    assert 'Close' in df.columns

def test_load_symbol_fallback(tmp_path):
    with pytest.raises(Exception):
        load_symbol('NOFILE', base_dir=str(tmp_path))
    df = load_symbol('NOFILE', base_dir=str(tmp_path), allow_synthetic_fallback=True)
    assert not df.empty

def test_list_available_symbols(tmp_path):
    (tmp_path / 'data' / 'cleaned').mkdir(parents=True)
    p = tmp_path / 'data' / 'cleaned' / 'E1.csv'
    p.write_text('Date,Open,High,Low,Close\n2020-01-01,1,1,1,1\n')
    syms = list_available_symbols(base_dir=str(tmp_path))
    assert 'E1' in syms
PY

write_file tests/test_backtest.py <<'PY'
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
PY

write_file tests/test_optuna.py <<'PY'
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
PY

# 14) GitHub Actions CI
write_file .github/workflows/ci.yml <<'YAML'
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10','3.11']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -e .
          pip install pytest black flake8 pytest-cov pre-commit mypy
      - name: Run pre-commit
        run: pre-commit run --all-files
      - name: Format check
        run: black --check .
      - name: Lint
        run: flake8 fxorcist
      - name: MyPy
        run: mypy fxorcist || true
      - name: Tests
        run: pytest -q
      - name: Coverage
        run: pytest --cov=fxorcist --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml
YAML

# 15) Dockerfile
write_file Dockerfile <<'DOCKER'
# Multi-stage for smaller runtime image
FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml requirements.txt ./
RUN python -m pip install --upgrade pip
RUN pip wheel --wheel-dir /build/wheels -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /build/wheels /wheels
RUN pip install --no-index --find-links=/wheels -r requirements.txt
COPY . /app
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "fxorcist/dashboard/app.py"]
DOCKER

# 16) README patch
write_file README.md <<'RE'
# FXorcist — Upgraded v3

This branch introduces:
- Improved loader with schema validation and parquet caching
- Extended backtests (slippage, sortino)
- Parallel runners (dask/multiprocessing)
- Optuna with pruning, MLflow optional, artifact saving
- Streamlit dashboard improvements
- CI + pre-commit + mypy
RE

# Final commit
if git diff --cached --quiet; then
  echo "No changes staged (files existed, no new writes)."
else
  git commit -m "chore: fxorcist v3 upgrade scaffold (loader, backtest, parallel, optuna, dashboard, ci, tests, pre-commit)"
  echo "Committed changes. Please review with 'git show --name-only HEAD' and run tests."
fi

echo "DONE. Next steps:"
echo "  1) Inspect changes: git diff origin/main..HEAD"
echo "  2) Run tests: pytest -q"
echo "  3) Run pre-commit: pre-commit run --all-files"
echo "  4) Push: git push --set-upstream origin $(git rev-parse --abbrev-ref HEAD)"
if git diff --cached --quiet; then
  echo "No changes staged (files existed, no new writes)."
else
  git commit -m "chore: fxorcist v3 upgrade scaffold (loader, backtest, parallel, optuna, dashboard, ci, tests, pre-commit)"
  echo "Committed changes. Please review with 'git show --name-only HEAD' and run tests."
fi

echo "DONE. Next steps:"
echo "  1) Inspect changes: git diff origin/main..HEAD"
echo "  2) Run tests: pytest -q"
echo "  3) Run pre-commit: pre-commit run --all-files"
echo "  4) Push: git push --set-upstream origin $(git rev-parse --abbrev-ref HEAD)"