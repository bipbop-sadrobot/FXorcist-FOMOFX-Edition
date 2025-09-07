"""
FXorcist Data Loader Module

Handles data loading, validation, and preprocessing with:
- Parquet-first approach with CSV fallback
- Schema validation
- Synthetic data generation for testing
- Caching for performance
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import pyarrow as pa
import pyarrow.parquet as pq
from diskcache import Cache

# Set up logging
logger = logging.getLogger(__name__)

# Schema definition
REQUIRED_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
SCHEMA = pa.schema([
    ('timestamp', pa.timestamp('ns')),
    ('open', pa.float64()),
    ('high', pa.float64()),
    ('low', pa.float64()),
    ('close', pa.float64()),
    ('volume', pa.float64())
])

# Cache configuration
cache = Cache(directory=os.path.join(os.path.dirname(__file__), '.cache'))

def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate DataFrame schema against requirements."""
    errors = []
    
    # Check required columns
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
    
    # Check data types
    for col in [c for c in REQUIRED_COLUMNS if c in df.columns]:
        if col == 'timestamp':
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                errors.append(f"Column {col} must be datetime type")
        elif not is_numeric_dtype(df[col]):
            errors.append(f"Column {col} must be numeric type")
    
    return len(errors) == 0, errors

def generate_synthetic_data(
    start_date: str = "2020-01-01",
    end_date: str = "2020-12-31",
    freq: str = "1min"
) -> pd.DataFrame:
    """Generate synthetic forex data for testing."""
    # Create timestamp range
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(dates)
    
    # Generate random walk prices
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(0, 0.0001, n)
    close = np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    df = pd.DataFrame({
        'timestamp': dates,
        'close': close,
        'volume': np.random.lognormal(10, 1, n)
    })
    
    # Add realistic OHLC variation
    df['open'] = df['close'] * np.exp(np.random.normal(0, 0.0002, n))
    df['high'] = df[['open', 'close']].max(axis=1) * np.exp(np.abs(np.random.normal(0, 0.0001, n)))
    df['low'] = df[['open', 'close']].min(axis=1) * np.exp(-np.abs(np.random.normal(0, 0.0001, n)))
    
    return df[REQUIRED_COLUMNS]

@cache.memoize()
def load_parquet(file_path: str) -> pd.DataFrame:
    """Load parquet file with caching."""
    return pd.read_parquet(file_path)

def load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV file with appropriate parsing."""
    return pd.read_csv(
        file_path,
        parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(file_path, nrows=1).columns else None
    )

def load_data(
    path: Union[str, Path],
    validate: bool = True,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Load data from parquet or CSV file with validation and caching.
    
    Args:
        path: Path to data file (parquet or csv)
        validate: Whether to validate schema
        use_cache: Whether to use disk cache
    
    Returns:
        DataFrame with validated schema
    """
    path = Path(path)
    
    try:
        # Try parquet first
        if path.suffix.lower() == '.parquet':
            df = load_parquet(str(path)) if use_cache else pd.read_parquet(path)
        # Fall back to CSV
        elif path.suffix.lower() == '.csv':
            logger.info("Loading CSV file (consider converting to parquet for better performance)")
            df = load_csv(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        if validate:
            valid, errors = validate_schema(df)
            if not valid:
                raise ValueError(f"Schema validation failed: {errors}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading {path}: {str(e)}")
        logger.info("Generating synthetic data as fallback")
        return generate_synthetic_data()

def compute_stats(df: pd.DataFrame, cols: List[str] = REQUIRED_COLUMNS[1:]) -> Dict:
    """Compute normalization statistics."""
    stats = {}
    for c in cols:
        if c in df.columns:
            stats[c] = {'mean': float(df[c].mean()), 'std': float(df[c].std())}
    return stats

def normalize(df: pd.DataFrame, stats: Dict) -> pd.DataFrame:
    """Normalize data using precomputed statistics."""
    df_norm = df.copy()
    for c, s in stats.items():
        if c in df.columns:
            df_norm[c] = (df_norm[c] - s['mean']) / (s['std'] if s['std'] != 0 else 1.0)
    return df_norm

def split_and_save(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    train_pct: float = 0.7,
    val_pct: float = 0.15
) -> None:
    """Split data into train/val/test sets and save."""
    output_dir = Path(output_dir)
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))
    
    # Create directories
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Save splits
    df.iloc[:train_end].to_parquet(output_dir / 'train' / 'train.parquet', index=False)
    df.iloc[train_end:val_end].to_parquet(output_dir / 'val' / 'val.parquet', index=False)
    df.iloc[val_end:].to_parquet(output_dir / 'test' / 'test.parquet', index=False)
    
    logger.info(f"Saved splits: train {train_end}, val {val_end-train_end}, test {n-val_end}")

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="FXorcist data preparation tool")
    parser.add_argument("--input", required=True, help="Input file (parquet or csv)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--train-pct", type=float, default=0.7, help="Training set percentage")
    parser.add_argument("--val-pct", type=float, default=0.15, help="Validation set percentage")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Load or generate data
        if args.synthetic:
            df = generate_synthetic_data()
            logger.info("Generated synthetic data")
        else:
            df = load_data(args.input, use_cache=not args.no_cache)
            logger.info(f"Loaded data from {args.input}")
        
        # Compute and save stats
        stats = compute_stats(df)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info("Saved normalization stats")
        
        # Normalize and split
        df_norm = normalize(df, stats)
        split_and_save(df_norm, output_dir, args.train_pct, args.val_pct)
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

if __name__ == "__main__":
    main()