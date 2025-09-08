import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import asyncio
import logging

from .base import DataConnector
from fxorcist.events.event_bus import Event

logger = logging.getLogger(__name__)

class CSVConnector(DataConnector):
    """Load market data from CSV or Parquet files."""

    def __init__(self, data_dir: str = "data/cleaned"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            logger.warning(f"Data directory {data_dir} does not exist. Creating...")
            self.data_dir.mkdir(parents=True, exist_ok=True)

    async def fetch(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None
    ) -> List[Event]:
        """Load data from {symbol}.parquet or {symbol}.csv."""
        parquet_path = self.data_dir / f"{symbol}.parquet"
        csv_path = self.data_dir / f"{symbol}.csv"

        df = None
        if parquet_path.exists():
            logger.info(f"Loading {symbol} from Parquet: {parquet_path}")
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            logger.info(f"Loading {symbol} from CSV: {csv_path}")
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        else:
            raise FileNotFoundError(f"No data for {symbol} at {self.data_dir}")

        # Validate required columns
        required_cols = ["timestamp"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Filter by date range
        mask = df["timestamp"] >= start
        if end:
            mask &= df["timestamp"] <= end
        df = df[mask].sort_values("timestamp").reset_index(drop=True)

        if df.empty:
            logger.warning(f"No data found for {symbol} between {start} and {end}")
            return []

        # Ensure 'mid' price exists
        if "mid" not in df.columns:
            if "bid" in df.columns and "ask" in df.columns:
                df["mid"] = (df["bid"] + df["ask"]) / 2
            else:
                raise ValueError("DataFrame must have 'mid' or both 'bid' and 'ask' columns")

        # Convert to Events
        events = []
        for _, row in df.iterrows():
            payload = row.drop("timestamp").to_dict()
            # Ensure numeric types
            for key in ["bid", "ask", "mid"]:
                if key in payload:
                    payload[key] = float(payload[key])

            events.append(Event(
                timestamp=row["timestamp"],
                type="tick",
                payload=payload
            ))

        logger.info(f"Loaded {len(events)} events for {symbol}")
        return events

    async def close(self):
        """No-op for file-based connectors."""
        pass