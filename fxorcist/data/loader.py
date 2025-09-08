from typing import List, Optional
from datetime import datetime
from rich.console import Console
import logging
from rich.logging import RichHandler

from fxorcist.config import Settings
from fxorcist.events.event_bus import Event
from fxorcist.data.connectors.base import DataConnector
from fxorcist.data.connectors.csv import CSVConnector
from fxorcist.data.connectors.exchange import ExchangeConnector

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger("DataLoader")
console = Console()

def get_connector(config: Settings) -> DataConnector:
    """
    Factory method to return appropriate data connector based on configuration.
    
    Supports:
    - CSV/Parquet storage
    - Exchange data
    - Future extensibility
    """
    storage = config.data.storage.lower()
    
    try:
        if storage in ["parquet", "csv"]:
            return CSVConnector(
                data_dir=config.data.parquet_dir,
                timestamp_column=config.data.timestamp_column,
                validate_data=True
            )
        elif storage == "exchange":
            return ExchangeConnector(
                api_key=config.exchange.api_key,
                api_secret=config.exchange.api_secret
            )
        else:
            raise ValueError(f"Unsupported storage type: {storage}")
    
    except Exception as e:
        logger.error(f"Error creating data connector: {e}")
        raise

async def prepare_data(
    symbol: str,
    config: Settings,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[Event]:
    """
    Prepare market data with enhanced error handling and logging.
    
    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        config: System configuration
        start_date: Optional start date for data retrieval
        end_date: Optional end date for data retrieval
    
    Returns:
        List of market events
    """
    try:
        connector = get_connector(config)
        
        # Parse dates with flexible handling
        start = datetime.fromisoformat(start_date) if start_date else datetime(2020, 1, 1)
        end = datetime.fromisoformat(end_date) if end_date else None
        
        events = await connector.fetch(symbol, start, end)
        
        logger.info(f"Successfully loaded {len(events)} events for {symbol}")
        console.print(f"[green]Loaded {len(events)} events for {symbol}[/green]")
        
        return events
    
    except FileNotFoundError:
        logger.warning(f"No data found for symbol {symbol}")
        return []
    
    except ValueError as ve:
        logger.error(f"Data validation error: {ve}")
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error preparing data: {e}")
        raise