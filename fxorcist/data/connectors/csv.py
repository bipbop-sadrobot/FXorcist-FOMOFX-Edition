import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from .base import DataConnector
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
        parquet_path = self.data_dir / f"{symbol}.parquet"
        csv_path = self.data_dir / f"{symbol}.csv"

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        else:
            raise FileNotFoundError(f"No data for {symbol} in {self.data_dir}")

        # Ensure 'timestamp' column exists
        if 'timestamp' not in df.columns:
            raise ValueError("CSV must have 'timestamp' column")

        # Filter by date
        mask = df["timestamp"] >= start
        if end:
            mask &= df["timestamp"] <= end
        df = df[mask].sort_values("timestamp")

        events = []
        for _, row in df.iterrows():
            payload = row.to_dict()
            # Ensure bid/ask or mid exists
            if "mid" not in payload:
                if "bid" in payload and "ask" in payload:
                    payload["mid"] = (payload["bid"] + payload["ask"]) / 2
                else:
                    raise ValueError("Row must have 'mid' or both 'bid' and 'ask'")

            events.append(Event(
                timestamp=row["timestamp"],
                type="tick",
                payload=payload
            ))
        return events