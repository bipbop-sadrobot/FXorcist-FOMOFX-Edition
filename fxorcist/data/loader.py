"""
Robust Data Loader for FXorcist.

Features:
- Prefers Parquet in data/cleaned/<symbol>.parquet (fast, columnar).
- Falls back to CSV in data/cleaned/, data/, or top-level <symbol>.csv.
- Validates schema for OHLCV columns.
- Optional caching to Parquet (opt-in).
- Optional synthetic fallback (for tests only).
"""

import os
import glob
import logging
from pathlib import Path
from functools import lru_cache
from typing import Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

class DataLoaderError(Exception):
    """Raised when data cannot be loaded properly."""

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

def _validate_schema(df: pd.DataFrame, usecols: Optional[Iterable[str]] = None) -> None:
    cols = usecols if usecols is not None else REQUIRED_COLUMNS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataLoaderError(f"Missing required columns: {missing}")

def _load_csv(path: Path, usecols: Optional[Iterable[str]]) -> pd.DataFrame:
    try:
        return pd.read_csv(path, usecols=usecols, parse_dates=["Date"])
    except Exception as e:
        raise DataLoaderError(f"Failed to load CSV {path}: {e}") from e

def _load_parquet(path: Path, usecols: Optional[Iterable[str]]) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=usecols)
    except Exception as e:
        raise DataLoaderError(f"Failed to load Parquet {path}: {e}") from e

def _synthetic_data(symbol: str, n: int = 500) -> pd.DataFrame:
    """Deterministic fallback series for tests only."""
    import numpy as np
    idx = pd.date_range("2000-01-01", periods=n, freq="D")
    close = 1.0 + np.cumsum(np.random.default_rng(42).normal(0, 0.01, size=n))
    return pd.DataFrame({
        "Date": idx,
        "Open": close,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": np.random.default_rng(42).integers(100, 1000, size=n),
    }).set_index("Date")

def load_symbol(
    symbol: str,
    base_dir: Optional[str] = None,
    usecols: Optional[Iterable[str]] = None,
    prefer_parquet: bool = True,
    cache_parquet: bool = False,
    allow_synthetic_fallback: bool = False,
) -> pd.DataFrame:
    """
    Load OHLCV data for a symbol.

    Args:
        symbol: e.g. "EURUSD"
        base_dir: optional base directory (default "data/")
        usecols: optional subset of columns
        prefer_parquet: try parquet first (default True)
        cache_parquet: if True, will write CSV loads back to parquet
        allow_synthetic_fallback: if True, returns synthetic series if files not found

    Raises:
        DataLoaderError if files missing or schema invalid (unless fallback allowed).
    """
    root = Path(base_dir or "data")
    candidates = [
        root / "cleaned" / f"{symbol}.parquet",
        root / "cleaned" / f"{symbol}.csv",
        root / f"{symbol}.csv",
        Path(f"{symbol}.csv"),
    ]
    if not prefer_parquet:
        candidates.reverse()

    for path in candidates:
        if path.suffix == ".parquet" and path.exists():
            df = _load_parquet(path, usecols)
        elif path.suffix == ".csv" and path.exists():
            df = _load_csv(path, usecols)
            if cache_parquet:
                parquet_path = path.with_suffix(".parquet")
                try:
                    parquet_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(parquet_path, index=False)
                    logger.info("Cached %s to parquet at %s", symbol, parquet_path)
                except Exception as e:
                    logger.warning("Failed to cache parquet %s: %s", symbol, e)
        else:
            continue
        _validate_schema(df, usecols)
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        return df

    if allow_synthetic_fallback:
        logger.warning("Using synthetic fallback for %s", symbol)
        return _synthetic_data(symbol)

    raise DataLoaderError(f"No valid data found for {symbol}")

@lru_cache(maxsize=None)
def list_available_symbols(base_dir: Optional[str] = None) -> list[str]:
    root = Path(base_dir or "data")
    return sorted(set(
        Path(f).stem
        for ext in ("csv", "parquet")
        for f in glob.glob(str(root / "**" / f"*.{ext}"), recursive=True)
    ))

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python -m fxorcist.data.loader <SYMBOL>")
        sys.exit(1)
    symbol = sys.argv[1]
    df = load_symbol(symbol, allow_synthetic_fallback=True)
    print(df.head())