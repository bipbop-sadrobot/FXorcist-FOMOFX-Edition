"""
Tests for the async loader functionality.
Validates async file loading, synthetic fallback, and error cases.
"""
import pytest
import asyncio
from pathlib import Path
from fxorcist.data.loader import async_load_symbol, DataLoaderError

@pytest.mark.asyncio
async def test_async_synthetic_fallback(tmp_path):
    """Test that synthetic fallback works correctly in async mode."""
    # no files: must raise unless fallback allowed
    with pytest.raises(DataLoaderError):
        await async_load_symbol("NOFILE", base_dir=str(tmp_path), allow_synthetic_fallback=False)
    
    df = await async_load_symbol("NOFILE", base_dir=str(tmp_path), allow_synthetic_fallback=True)
    assert not df.empty
    assert "Close" in df.columns
    assert "Returns" in df.columns
    assert df.index.is_monotonic_increasing

@pytest.mark.asyncio
async def test_async_local_csv(tmp_path):
    """Test loading from a local CSV file asynchronously."""
    d = tmp_path / "data" / "cleaned"
    d.mkdir(parents=True)
    f = d / "TST.csv"
    f.write_text("Date,Open,High,Low,Close\n2020-01-01,1,1,1,1\n")
    
    df = await async_load_symbol("TST", base_dir=str(tmp_path))
    assert not df.empty
    assert "Close" in df.columns
    assert df.index[0].year == 2020
    assert df.Close[0] == 1.0

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
    f.write_text("Date,Open,High,Low,Close\n2020-01-01,1,1,1,1\n")
    
    # First load with caching enabled
    df1 = await async_load_symbol("CACHE", base_dir=str(tmp_path), cache_parquet=True)
    assert not df1.empty
    
    # Verify parquet file was created
    parquet_path = d / "CACHE.parquet"
    assert parquet_path.exists()
    
    # Load again, should use parquet
    df2 = await async_load_symbol("CACHE", base_dir=str(tmp_path), prefer_parquet=True)
    assert not df2.empty
    assert df1.equals(df2)