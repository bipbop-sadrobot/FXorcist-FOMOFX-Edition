from typing import List, Optional
from datetime import datetime
from rich.console import Console

from fxorcist.config import Settings
from fxorcist.events.event_bus import Event
from fxorcist.data.connectors.csv import CSVConnector

console = Console()

def get_connector(config: Settings):
    # For now, default to CSVConnector
    return CSVConnector(data_dir=config.data_dir)

async def prepare_data(
    symbol: str,
    config: Settings,
    start_date: Optional[str] = None
) -> List[Event]:
    connector = get_connector(config)
    start = datetime.fromisoformat(start_date) if start_date else datetime(2020, 1, 1)
    events = await connector.fetch(symbol, start)
    console.log(f"[green]Loaded {len(events)} events for {symbol}[/green]")
    return events