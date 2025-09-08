import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Union, Dict, Any
from rich.console import Console
from rich.logging import RichHandler

import logging
from .base import DataConnector
from fxorcist.events.event_bus import Event

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger("CSVConnector")
console = Console()

class CSVConnector(DataConnector):
    """
    Advanced CSV/Parquet data connector with robust loading and validation.
    
    Supports multiple data formats:
    - Parquet files
    - CSV files
    - Multiple timestamp column formats
    - Flexible price column handling
    """
    
    SUPPORTED_PRICE_COLUMNS = [
        'mid', 'close', 'open', 'high', 'low', 
        'bid', 'ask', 'bid_price', 'ask_price'
    ]
    
    def __init__(
        self, 
        data_dir: Union[str, Path] = "data/cleaned", 
        timestamp_column: str = 'timestamp',
        validate_data: bool = True
    ):
        """
        Initialize CSVConnector with configurable parameters.
        
        Args:
            data_dir: Directory containing market data files
            timestamp_column: Name of the timestamp column
            validate_data: Enable strict data validation
        """
        self.data_dir = Path(data_dir)
        self.timestamp_column = timestamp_column
        self.validate_data = validate_data
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist.")
    
    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and preprocess DataFrame before event generation.
        
        Checks:
        - Timestamp column exists and is datetime
        - At least one price column exists
        - No missing critical data
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if self.timestamp_column not in df.columns:
            raise ValueError(f"Timestamp column '{self.timestamp_column}' not found")
        
        # Convert timestamp column to datetime if not already
        df[self.timestamp_column] = pd.to_datetime(df[self.timestamp_column])
        
        # Find first available price column
        price_column = next(
            (col for col in self.SUPPORTED_PRICE_COLUMNS if col in df.columns), 
            None
        )
        
        if price_column is None:
            raise ValueError(f"No price column found. Supported: {self.SUPPORTED_PRICE_COLUMNS}")
        
        # Handle bid/ask to mid price conversion
        if price_column == 'bid' and 'ask' in df.columns:
            df['mid'] = (df['bid'] + df['ask']) / 2
        
        # Optional data validation
        if self.validate_data:
            df.dropna(subset=[self.timestamp_column, price_column], inplace=True)
        
        return df.sort_values(self.timestamp_column)
    
    async def fetch(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None
    ) -> List[Event]:
        """
        Fetch market data for a specific symbol and date range.
        
        Supports multiple file formats and robust error handling.
        """
        try:
            # Try multiple file extensions
            file_paths = [
                self.data_dir / f"{symbol}.parquet",
                self.data_dir / f"{symbol}.csv",
                self.data_dir / f"{symbol}_1min.parquet",
                self.data_dir / f"{symbol}_1min.csv"
            ]
            
            df = None
            for path in file_paths:
                if path.exists():
                    try:
                        if path.suffix == '.parquet':
                            df = pd.read_parquet(path)
                        else:
                            df = pd.read_csv(path, parse_dates=[self.timestamp_column])
                        break
                    except Exception as e:
                        logger.warning(f"Could not read {path}: {e}")
            
            if df is None:
                raise FileNotFoundError(f"No data found for {symbol} in {self.data_dir}")
            
            # Validate and preprocess DataFrame
            df = self._validate_dataframe(df)
            
            # Filter by date range
            mask = df[self.timestamp_column] >= start
            if end:
                mask &= df[self.timestamp_column] <= end
            
            df = df[mask]
            
            logger.info(f"Loaded {len(df)} events for {symbol} from {start} to {end}")
            
            events = [
                Event(
                    timestamp=row[self.timestamp_column],
                    type="tick",
                    payload={
                        "symbol": symbol,
                        "mid": row.get('mid', np.nan),
                        **{k: v for k, v in row.items() if k not in [self.timestamp_column, 'mid']}
                    }
                )
                for _, row in df.iterrows()
            ]
            
            return events
        
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise