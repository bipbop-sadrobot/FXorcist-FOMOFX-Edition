import asyncpg
from datetime import datetime
from typing import List, Optional, Dict, Any
import asyncio
import logging

from .base import DataConnector
from fxorcist.events.event_bus import Event

logger = logging.getLogger(__name__)

class TimescaleConnector(DataConnector):
    """Connect to TimescaleDB for production time-series data."""

    def __init__(self, dsn: str = "postgresql://user:pass@localhost:5432/fxorcist"):
        self.dsn = dsn
        self.pool = None

    async def connect(self):
        """Initialize connection pool."""
        if self.pool is None:
            try:
                self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=10)
                logger.info("Connected to TimescaleDB")
            except Exception as e:
                logger.error(f"Failed to connect to TimescaleDB: {e}")
                raise

    async def fetch(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None
    ) -> List[Event]:
        """Fetch tick data from TimescaleDB hypertable."""
        await self.connect()

        query = """
        SELECT timestamp, bid, ask FROM ticks
        WHERE symbol = $1 AND timestamp >= $2
        ORDER BY timestamp ASC
        """
        params = [symbol, start]

        if end:
            query += " AND timestamp <= $3"
            params.append(end)

        events = []
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                for row in rows:
                    payload = {
                        "symbol": symbol,
                        "bid": float(row["bid"]),
                        "ask": float(row["ask"]),
                        "mid": (float(row["bid"]) + float(row["ask"])) / 2
                    }
                    events.append(Event(
                        timestamp=row["timestamp"],
                        type="tick",
                        payload=payload
                    ))
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

        logger.info(f"Loaded {len(events)} ticks for {symbol} from TimescaleDB")
        return events

    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Closed TimescaleDB connection")