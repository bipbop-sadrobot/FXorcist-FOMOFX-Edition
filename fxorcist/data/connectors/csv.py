import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from fxorcist.data.connectors.base import DataConnector
from fxorcist.events.event_bus import Event

class CSVConnector(DataConnector):
    def __init__(self, data_dir: str = "data/cleaned"):
        self.data_dir = Path(data_dir)

    async def fetch(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None
    ) -> List[Event]:
        """Load from {symbol}.parquet or {symbol}.csv."""
        parquet_path = self.data_dir / f"{symbol}.parquet"
        csv_path = self.data_dir / f"{symbol}.csv"

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        else:
            raise FileNotFoundError(f"No data for {symbol} in {self.data_dir}")

        # Filter by date
        mask = df["timestamp"] >= start
        if end:
            mask &= df["timestamp"] <= end
        df = df[mask]

        events = []
        for _, row in df.iterrows():
            events.append(Event(
                timestamp=row["timestamp"],
                type="tick",
                payload={
                    "symbol": symbol,
                    "bid": float(row["bid"]),
                    "ask": float(row["ask"]),
                    "mid": float(row["mid"]) if "mid" in row else (row["bid"] + row["ask"]) / 2
                }
            ))
        return events