"""
Tests for the async loader functionality.
Validates async file loading, synthetic fallback, error handling, and data integrity.
"""
import pytest
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from fxorcist.data.loader import async_load_symbol, DataLoaderError, REQUIRED_COLS

@pytest.mark.asyncio
async def test_async_synthetic_fallback(tmp_path):
    """Test that synthetic fallback works correctly in async mode."""
    # no files: must raise unless fallback allowed
    with pytest.raises(DataLoaderError, match="No data files found"):
        await async_load_symbol("NOFILE", base_dir=str(tmp_path), allow_synthetic_fallback=False)
    
    df = await async_load_symbol("NOFILE", base_dir=str(tmp_path), allow_synthetic_fallback=True)
    assert not df.empty
    assert all(col in df.columns for col in REQUIRED_COLS - {"Date"})
    assert "Returns" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert not df.isnull().any().any()

@pytest.mark.asyncio
async def test_async_schema_validation(tmp_path):
    """Test schema validation in async context."""
    d = tmp_path / "data" / "cleaned"
    d.mkdir(parents=True)
    
    # Missing required column
    f = d / "BAD.csv"
    f.write_text("Date,Open,High,Low\n2020-01-01,1,1,1\n")
    with pytest.raises(DataLoaderError, match="Missing required columns"):
        await async_load_symbol("BAD", base_dir=str(tmp_path))
    
    # Invalid date format
    f.write_text("Date,Open,High,Low,Close\nNOTADATE,1,1,1,1\n")
    with pytest.raises(DataLoaderError, match="Invalid dates"):
        await async_load_symbol("BAD", base_dir=str(tmp_path))

@pytest.mark.asyncio
async def test_async_local_csv(tmp_path):
    """Test loading from a local CSV file asynchronously."""
    d = tmp_path / "data" / "cleaned"
    d.mkdir(parents=True)
    f = d / "TST.csv"
    f.write_text("Date,Open,High,Low,Close\n2020-01-01,1,1,1,1\n")
    
    df = await async_load_symbol("TST", base_dir=str(tmp_path))
    assert not df.empty
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index[0] == pd.Timestamp("2020-01-01")
    assert all(col in df.columns for col in REQUIRED_COLS - {"Date"})
    assert "Returns" in df.columns
    assert df.Returns[0] == 0.0  # First return should be 0
    assert np.isclose(df.Close[0], 1.0)

@pytest.mark.asyncio
async def test_async_error_propagation(tmp_path):
    """Test error handling and propagation."""
    d = tmp_path / "data" / "cleaned"
    d.mkdir(parents=True)
    
    # Corrupt parquet file
    f = d / "CORRUPT.parquet"
    f.write_bytes(b"NOT A PARQUET FILE")
    with pytest.raises(DataLoaderError, match="Failed to read parquet"):
        await async_load_symbol("CORRUPT", base_dir=str(tmp_path))
    
    # Permission error simulation
    f = d / "NOPERM.csv"
    f.write_text("Date,Open,High,Low,Close\n2020-01-01,1,1,1,1\n")
    f.chmod(0o000)  # Remove all permissions
    with pytest.raises(DataLoaderError):
        await async_load_symbol("NOPERM", base_dir=str(tmp_path))
    f.chmod(0o644)  # Restore permissions

@pytest.mark.asyncio
async def test_async_timeout(tmp_path):
    """Test that timeout parameter works correctly."""
    with pytest.raises(asyncio.TimeoutError):
        await async_load_symbol(
            "NOFILE", 
            base_dir=str(tmp_path),
            allow_synthetic_fallback=True,
            timeout=0.0001  # Very short timeout to trigger error
        )

@pytest.mark.asyncio
async def test_async_parquet_caching(tmp_path):
    """Test that CSV data is correctly cached to parquet when requested."""
    d = tmp_path / "data" / "cleaned"
    d.mkdir(parents=True)
    f = d / "CACHE.csv"
    test_data = pd.DataFrame({
        'Date': pd.date_range('2020-01-01', periods=5),
        'Open': np.random.rand(5),
        'High': np.random.rand(5),
        'Low': np.random.rand(5),
        'Close': np.random.rand(5)
    })
    test_data.to_csv(f, index=False)
    
    # First load with caching enabled
    df1 = await async_load_symbol("CACHE", base_dir=str(tmp_path), cache_parquet=True)
    assert not df1.empty
    
    # Verify parquet file was created and has correct content
    parquet_path = d / "CACHE.parquet"
    assert parquet_path.exists()
    
    # Load again, should use parquet
    df2 = await async_load_symbol("CACHE", base_dir=str(tmp_path), prefer_parquet=True)
    assert not df2.empty
    pd.testing.assert_frame_equal(df1, df2)
    
    # Verify data integrity
    assert isinstance(df2.index, pd.DatetimeIndex)
    assert df2.index.is_monotonic_increasing
    assert not df2.isnull().any().any()
    assert "Returns" in df2.columns

@pytest.mark.asyncio
async def test_async_memory_cleanup(tmp_path):
    """Test temporary file cleanup during parquet operations."""
    d = tmp_path / "data" / "cleaned"
    d.mkdir(parents=True)
    
    # Create a large enough DataFrame to force temp file usage
    dates = pd.date_range('2020-01-01', periods=1000)
    data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.rand(1000),
        'High': np.random.rand(1000),
        'Low': np.random.rand(1000),
        'Close': np.random.rand(1000)
    })
    f = d / "LARGE.csv"
    data.to_csv(f, index=False)
    
    # Load and cache, checking for temp file cleanup
    df = await async_load_symbol("LARGE", base_dir=str(tmp_path), cache_parquet=True)
    
    # Verify no temp files left behind
    temp_files = list(Path(tmp_path).glob(".tmp_parquet_*"))
    assert len(temp_files) == 0