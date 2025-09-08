import asyncpg
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

from fxorcist.events.event_bus import Event
from fxorcist.data.connectors.base import DataConnector

logger = logging.getLogger(__name__)

class TimescaleConnector(DataConnector):
    def __init__(self, dsn: str = "postgresql://user:pass@localhost:5432/fxorcist"):
        """
        Initialize TimescaleDB connector.
        
        :param dsn: Database connection string
        """
        self.dsn = dsn
        self.pool = None

    async def connect(self):
        """
        Establish connection pool to TimescaleDB.
        """
        try:
            self.pool = await asyncpg.create_pool(self.dsn)
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
        """
        Fetch tick data for a given symbol and time range.
        
        :param symbol: Trading symbol (e.g., 'EURUSD')
        :param start: Start datetime
        :param end: Optional end datetime
        :return: List of tick events
        """
        if not self.pool:
            await self.connect()

        query = """
        SELECT timestamp, bid, ask FROM ticks
        WHERE symbol = $1 AND timestamp >= $2
        """
        params = [symbol, start]
        
        if end:
            query += " AND timestamp <= $3"
            params.append(end)

        events = []
        async with self.pool.acquire() as conn:
            try:
                rows = await conn.fetch(query, *params)
                for row in rows:
                    events.append(Event(
                        timestamp=row["timestamp"],
                        type="tick",
                        payload={
                            "symbol": symbol,
                            "bid": float(row["bid"]),
                            "ask": float(row["ask"]),
                            "mid": (float(row["bid"]) + float(row["ask"])) / 2
                        }
                    ))
            except Exception as e:
                logger.error(f"Error fetching data from TimescaleDB: {e}")
                raise

        return events

    async def insert_ticks(self, ticks: List[Dict[str, Any]]):
        """
        Insert tick data into TimescaleDB.
        
        :param ticks: List of tick dictionaries
        """
        if not self.pool:
            await self.connect()

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany("""
                    INSERT INTO ticks (timestamp, symbol, bid, ask)
                    VALUES ($1, $2, $3, $4)
                """, [
                    (tick['timestamp'], tick['symbol'], tick['bid'], tick['ask'])
                    for tick in ticks
                ])

    async def close(self):
        """
        Close database connection pool.
        """
        if self.pool:
            await self.pool.close()
            logger.info("Closed TimescaleDB connection pool")