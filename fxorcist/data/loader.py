from typing import List, Optional
from datetime import datetime

from fxorcist.config import Settings
from fxorcist.events.event_bus import Event
from fxorcist.data.connectors.base import DataConnector
from fxorcist.data.connectors.csv import CSVConnector
from fxorcist.data.connectors.exchange import ExchangeConnector

def get_connector(config: Settings) -> DataConnector:
    """
    Factory method to return appropriate data connector.
    
    Args:
        config: System configuration object
    
    Returns:
        Configured DataConnector instance
    """
    storage_type = config.data.storage.lower()
    
    if storage_type in ['csv', 'parquet']:
        return CSVConnector(data_dir=config.data.parquet_dir)
    elif storage_type == 'exchange':
        return ExchangeConnector(
            api_key=config.exchange.get('api_key'),
            api_secret=config.exchange.get('api_secret')
        )
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")

async def prepare_data(
    symbol: str,
    config: Settings,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[Event]:
    """
    Prepare market data using configured connector.
    
    Args:
        symbol: Trading symbol
        config: System configuration
        start_date: Optional start date for data retrieval
        end_date: Optional end date for data retrieval
    
    Returns:
        List of market events
    """
    connector = get_connector(config)
    start = datetime.fromisoformat(start_date) if start_date else datetime(2020, 1, 1)
    end = datetime.fromisoformat(end_date) if end_date else None
    
    return await connector.fetch(symbol, start, end)