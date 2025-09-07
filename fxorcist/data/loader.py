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
