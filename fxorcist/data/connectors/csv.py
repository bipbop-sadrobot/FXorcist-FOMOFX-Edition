import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from .base import DataConnector
from fxorcist.events.event_bus import Event

class CSVConnector(DataConnector):
    """
    Minimal, production-ready CSV data connector for market data.
    
    Supports loading market data from CSV and Parquet files with 
    minimal configuration and robust error handling.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize CSVConnector with data directory.
        
        Args:
            data_dir: Absolute path to directory containing market data files
        """
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")

    async def fetch(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None
    ) -> List[Event]:
        """
        Fetch market events for a specific symbol within a date range.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            start: Start datetime for data retrieval
            end: Optional end datetime for data retrieval
        
        Returns:
            List of market events
        """
        # Attempt to load Parquet first, fallback to CSV
        parquet_path = self.data_dir / f"{symbol}.parquet"
        csv_path = self.data_dir / f"{symbol}.csv"
        
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        else:
            raise FileNotFoundError(f"No data found for {symbol}")
        
        # Filter by date range
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        mask = df['timestamp'] >= start
        if end:
            mask &= df['timestamp'] <= end
        
        df = df[mask].sort_values('timestamp')
        
        # Generate events
        events = []
        for _, row in df.iterrows():
            payload = row.to_dict()
            
            # Compute mid price if bid/ask available
            if 'bid' in payload and 'ask' in payload:
                payload['mid'] = (payload['bid'] + payload['ask']) / 2
            
            events.append(Event(
                timestamp=row['timestamp'],
                type='tick',
                payload=payload
            ))
        
        return events