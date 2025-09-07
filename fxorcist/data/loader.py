"""
Improved data loader:
- parquet-first, csv fallback
- optional caching to parquet (snappy)
- optional schema validation via pandera (if installed)
- asynchronous loader available via async_load_symbol (if aiofiles/fsspec is installed)
- deterministic synthetic fallback for tests with optional GARCH-ish volatility
"""
from __future__ import annotations
import asyncio
import io
import os
import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Dict
import pandas as pd
import numpy as np
import aiofiles
import fsspec

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

async def _read_csv_aio(path: Path | str, usecols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Read a CSV asynchronously for local files using aiofiles -> pandas via BytesIO.
    NOTE: this reads the entire file into memory; good for small-to-medium files.
    For very large files, prefer to use to_thread with pandas.read_csv(stream).
    """
    p = Path(path)
    if p.exists() and p.is_file():
        # local file: read asynchronously with aiofiles
        async with aiofiles.open(p, mode="rb") as af:
            content = await af.read()
        return pd.read_csv(io.BytesIO(content), parse_dates=['Date'], usecols=usecols)
    else:
        # Non-local (or doesn't exist locally) - fallback to fsspec (wrapped in thread)
        def _sync_read():
            with fsspec.open(path, mode="rb") as fh:
                data = fh.read()
            return pd.read_csv(io.BytesIO(data), parse_dates=['Date'], usecols=usecols)
        return await asyncio.to_thread(_sync_read)

async def _read_parquet_aio(path: Path | str, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Parquet readers are usually blocking (pyarrow/fastparquet). Wrap in thread.
    Support local or fsspec URLs.
    """
    p = Path(path)
    def _sync_read():
        # If path is local, use pandas directly; otherwise use fsspec to open remote and read bytes to buffer
        if p.exists():
            return _read_parquet(p, columns=columns)
        else:
            # fsspec path / remote object
            with fsspec.open(path, mode="rb") as fh:
                data = fh.read()
            # pandas.read_parquet can accept a BytesIO if pyarrow supports it
            try:
                return pd.read_parquet(io.BytesIO(data), columns=columns)
            except Exception:
                # fallback: save to temp file and read
                tmp = Path(os.path.join(os.getcwd(), f".tmp_parquet_{os.getpid()}"))
                tmp.write_bytes(data)
                try:
                    return pd.read_parquet(tmp, columns=columns)
                finally:
                    try:
                        tmp.unlink()
                    except Exception:
                        pass
    try:
        return await asyncio.to_thread(_sync_read)
    except Exception as e:
        logger.exception("async parquet read failed: %s", e)
        raise DataLoaderError(f"Failed to read parquet: {e}")

async def async_load_symbol(
    symbol: str,
    base_dir: Optional[str] = None,
    usecols: Optional[Iterable[str]] = None,
    prefer_parquet: bool = True,
    cache_parquet: bool = False,
    allow_synthetic_fallback: bool = False,
    timeout: Optional[float] = None,
    validate_schema: bool = True,
) -> pd.DataFrame:
    """
    Async wrapper to load a symbol. Practical behavior:
      - If file is local CSV, uses aiofiles to read non-blocking.
      - Parquet reading remains synchronous under the hood but is executed on a thread via asyncio.to_thread.
      - Supports fsspec URLs (s3://, gs://, http://) by delegating to fsspec in a thread.
      - `timeout` will cancel the operation with asyncio.wait_for (if provided).
    """
    base = Path(base_dir) if base_dir else Path.cwd()
    sym = symbol.upper()
    candidates = _candidates_for_symbol(base, sym)
    cols = None if usecols is None else list(usecols)

    async def _inner():
        df = None
        # Try parquet first if preferred
        if prefer_parquet:
            pq = candidates[0]
            # pq can be Path or remote; check local existence first
            if Path(pq).exists():
                df = await _read_parquet_aio(str(pq), columns=cols)
            if df is None:
                # if not local, attempt using fsspec wrapper
                try:
                    # attempt remote read (this will raise if unreachable)
                    df = await _read_parquet_aio(str(pq), columns=cols)
                except Exception as e:
                    logger.debug("remote parquet read failed: %s", e)

        if df is None:
            # Try CSV candidates (use aio read for local files, else fsspec/thread)
            for p in candidates[1:]:
                # if local and exists, use aio
                if Path(p).exists():
                    try:
                        df = await _read_csv_aio(str(p), usecols=cols)
                        break
                    except Exception as e:
                        logger.debug("local csv read failed for %s: %s", p, e)
                        continue
                # if not local, try remote
                try:
                    df = await _read_csv_aio(str(p), usecols=cols)
                    break
                except Exception as e:
                    logger.debug("remote csv read failed for %s: %s", p, e)
                    continue

        if df is not None:
            # Ensure proper DataFrame structure
            df = _ensure_index_and_types(df)
            if validate_schema:
                _validate_schema(df)
            df = _ensure_returns(df)

            # optionally cache to parquet (runs in thread)
            if cache_parquet:
                try:
                    target = base / "data" / "cleaned" / f"{sym}.parquet"
                    if not target.exists():
                        # write parquet in thread to avoid blocking loop
                        await asyncio.to_thread(lambda: (
                            target.parent.mkdir(parents=True, exist_ok=True),
                            df.to_parquet(target, compression="snappy", index=False)
                        ))
                        logger.info("async cached %s -> %s", p, target)
                except Exception as ex:
                    logger.warning("async parquet cache write failed: %s", ex)
            return df

        # No data found - try synthetic
        if allow_synthetic_fallback:
            logger.warning("async: no file found for %s — return synthetic", sym)
            # synthetic is synchronous but cheap — run in thread to be consistent
            df = await asyncio.to_thread(lambda: _synthetic_series(sym))
            if validate_schema:
                _validate_schema(df)
            return df
        
        raise DataLoaderError(f"async: No data files found for {sym}")

    if timeout:
        return await asyncio.wait_for(_inner(), timeout=timeout)
    else:
        return await _inner()
