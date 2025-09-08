import pytest
import asyncio
from datetime import datetime, timedelta
from fxorcist.data.connectors.timescale import TimescaleConnector, TimescaleConnectorConfig

# Note: These tests require a running TimescaleDB instance
# You may need to set up a test database or use a Docker container for testing

@pytest.fixture
async def timescale_connector():
    """
    Fixture to create a TimescaleDB connector.
    Assumes a test database is set up with the connection string.
    """
    config = TimescaleConnectorConfig(
        dsn="postgresql://testuser:testpass@localhost:5432/fxorcist_test",
        max_connection_pool_size=5,
        data_retention_days=30
    )
    
    connector = TimescaleConnector(config=config)
    await connector.connect()
    yield connector
    await connector.close()

@pytest.mark.asyncio
async def test_timescale_connector_initialization(timescale_connector):
    """
    Test TimescaleDB connector initialization.
    """
    assert timescale_connector.pool is not None
    assert timescale_connector.config.data_retention_days == 30

@pytest.mark.asyncio
async def test_timescale_connector_insert_and_fetch(timescale_connector):
    """
    Test inserting and fetching tick data.
    """
    # Prepare test data
    now = datetime.now()
    ticks = [
        {
            "timestamp": now,
            "symbol": "EURUSD",
            "bid": 1.0950,
            "ask": 1.0955
        },
        {
            "timestamp": now + timedelta(minutes=1),
            "symbol": "EURUSD",
            "bid": 1.0952,
            "ask": 1.0957
        }
    ]
    
    # Insert ticks
    await timescale_connector.insert_ticks(ticks)
    
    # Fetch ticks
    fetched_events = await timescale_connector.fetch(
        symbol="EURUSD", 
        start=now - timedelta(minutes=5),
        end=now + timedelta(minutes=5)
    )
    
    assert len(fetched_events) == 2
    
    # Verify fetched data
    for event in fetched_events:
        assert event.type == "tick"
        assert event.payload["symbol"] == "EURUSD"
        assert "bid" in event.payload
        assert "ask" in event.payload
        assert "mid" in event.payload

@pytest.mark.asyncio
async def test_timescale_connector_batch_insert(timescale_connector):
    """
    Test batch insertion of ticks.
    """
    now = datetime.now()
    ticks = [
        {
            "timestamp": now + timedelta(minutes=i),
            "symbol": "GBPUSD",
            "bid": 1.2500 + (i * 0.0001),
            "ask": 1.2505 + (i * 0.0001)
        }
        for i in range(100)  # Large batch of 100 ticks
    ]
    
    # Insert ticks in batches
    await timescale_connector.insert_ticks(ticks)
    
    # Fetch ticks
    fetched_events = await timescale_connector.fetch(
        symbol="GBPUSD", 
        start=now,
        end=now + timedelta(minutes=100)
    )
    
    assert len(fetched_events) == 100

@pytest.mark.asyncio
async def test_timescale_connector_error_handling(timescale_connector):
    """
    Test error handling in TimescaleDB connector.
    """
    # Attempt to fetch with invalid parameters
    with pytest.raises(Exception):
        await timescale_connector.fetch(
            symbol="INVALID_SYMBOL", 
            start=datetime.now() - timedelta(days=365),
            end=datetime.now()
        )

@pytest.mark.asyncio
async def test_timescale_connector_connection_management():
    """
    Test connection and disconnection.
    """
    config = TimescaleConnectorConfig(
        dsn="postgresql://testuser:testpass@localhost:5432/fxorcist_test"
    )
    
    connector = TimescaleConnector(config=config)
    
    # Connect
    await connector.connect()
    assert connector.pool is not None
    
    # Close connection
    await connector.close()
    # Note: In asyncpg, the pool is not set to None after closing