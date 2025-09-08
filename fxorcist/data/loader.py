from typing import List
from datetime import datetime
import asyncio

from fxorcist.config import Settings
from fxorcist.data.connectors.base import DataConnector
from fxorcist.data.connectors.csv import CSVConnector
from fxorcist.data.connectors.exchange import ExchangeConnector
# from fxorcist.data.connectors.timescale import TimescaleConnector  # Phase 4

def get_connector(config: Settings) -> DataConnector:
    """Factory: return connector based on config.data.storage."""
    storage = config.data.storage
    if storage == "parquet" or storage == "csv":
        return CSVConnector(config.data.parquet_dir)
    elif storage == "exchange":
        return ExchangeConnector()
    else:
        raise ValueError(f"Unsupported storage: {storage}")

async def prepare_data(
    symbol: str,
    config: Settings,
    start_date: str = None
) -> List:
    """Prepare data — load or download."""
    connector = get_connector(config)
    start = datetime.fromisoformat(start_date) if start_date else datetime(2020, 1, 1)
    events = await connector.fetch(symbol, start)
    print(f"Loaded {len(events)} events for {symbol}.")
    return events