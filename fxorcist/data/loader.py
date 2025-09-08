from typing import List, Optional
from datetime import datetime
import asyncio
from fxorcist.config import Settings
from fxorcist.data.connectors.base import DataConnector
from fxorcist.data.connectors.csv import CSVConnector
from fxorcist.data.connectors.exchange import ExchangeConnector
from fxorcist.data.connectors.timescale import TimescaleConnector
from fxorcist.events.event_bus import Event
from rich.console import Console

console = Console()

def get_connector(config: Settings) -> DataConnector:
    """Factory: return connector based on config."""
    storage = config.data.storage
    if storage in ["parquet", "csv"]:
        return CSVConnector(config.data.parquet_dir)
    elif storage == "exchange":
        return ExchangeConnector(
            exchange_id=config.data.get("exchange_id", "kraken"),
            api_key=config.data.get("api_key"),
            secret=config.data.get("secret")
        )
    elif storage == "timescale":
        return TimescaleConnector(config.data.get("dsn", "postgresql://user:pass@localhost:5432/fxorcist"))
    else:
        raise ValueError(f"Unsupported storage type: {storage}")

async def prepare_data(
    symbol: str,
    config: Settings,
    start_date: Optional[str] = None
) -> List[Event]:
    """Prepare data — load from configured source."""
    connector = get_connector(config)
    start = datetime.fromisoformat(start_date) if start_date else datetime(2020, 1, 1)

    try:
        events = await connector.fetch(symbol, start)
        console.log(f"[green]✓ Loaded {len(events)} events for {symbol}[/green]")
        return events
    except Exception as e:
        console.log(f"[red]✗ Failed to load data: {e}[/red]")
        raise
    finally:
        if hasattr(connector, 'close'):
            await connector.close()