import tempfile
import pandas as pd
import pytest
from datetime import datetime
from fxorcist.data.loader import get_connector, prepare_data
from fxorcist.config import Settings

@pytest.mark.asyncio
async def test_csv_connector():
    # Create mock data
    with tempfile.TemporaryDirectory() as tmpdir:
        df = pd.DataFrame({
            "timestamp": [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 10, 1)],
            "bid": [1.0, 1.0001],
            "ask": [1.0001, 1.0002]
        })
        df.to_parquet(f"{tmpdir}/EURUSD.parquet")

        config = Settings()
        config.data.storage = "parquet"
        config.data.parquet_dir = tmpdir

        connector = get_connector(config)
        events = await connector.fetch("EURUSD", datetime(2024, 1, 1))
        assert len(events) == 2
        assert events[0].payload["symbol"] == "EURUSD"
        assert abs(events[0].payload["mid"] - 1.00005) < 1e-6

@pytest.mark.skip(reason="Requires live exchange connection")
@pytest.mark.asyncio
async def test_exchange_connector():
    config = Settings()
    config.data.storage = "exchange"
    config.data.exchange_id = "kraken"

    connector = get_connector(config)
    events = await connector.fetch("BTC/USDT", datetime.now() - timedelta(days=1))
    
    assert len(events) > 0
    assert all(event.type == "bar" for event in events)
    assert all("open" in event.payload for event in events)

@pytest.mark.skip(reason="Requires TimescaleDB connection")
@pytest.mark.asyncio
async def test_timescale_connector():
    config = Settings()
    config.data.storage = "timescale"
    config.data.dsn = "postgresql://user:pass@localhost:5432/fxorcist"

    connector = get_connector(config)
    events = await connector.fetch("EURUSD", datetime.now() - timedelta(days=1))
    
    assert len(events) > 0
    assert all(event.type == "tick" for event in events)
    assert all("bid" in event.payload and "ask" in event.payload for event in events)