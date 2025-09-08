import asyncpg
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
import logging
import json
import os

from fxorcist.events.event_bus import Event
from fxorcist.data.connectors.base import DataConnector

logger = logging.getLogger(__name__)

class TimescaleConnectorConfig:
    """Configuration for TimescaleDB connector."""
    def __init__(
        self, 
        dsn: Optional[str] = None,
        max_connection_pool_size: int = 10,
        connection_timeout: int = 10,
        data_retention_days: Optional[int] = 365
    ):
        """
        Initialize TimescaleDB connector configuration.
        
        :param dsn: Database connection string
        :param max_connection_pool_size: Maximum number of connections in the pool
        :param connection_timeout: Connection timeout in seconds
        :param data_retention_days: Number of days to retain historical data
        """
        self.dsn = dsn or os.getenv(
            'TIMESCALEDB_DSN', 
            'postgresql://user:pass@localhost:5432/fxorcist'
        )
        self.max_connection_pool_size = max_connection_pool_size
        self.connection_timeout = connection_timeout
        self.data_retention_days = data_retention_days

class TimescaleConnector(DataConnector):
    """Advanced TimescaleDB connector for time-series forex data."""
    
    def __init__(
        self, 
        config: Optional[TimescaleConnectorConfig] = None,
        logging_level: int = logging.INFO
    ):
        """
        Initialize TimescaleDB connector.
        
        :param config: TimescaleDB connector configuration
        :param logging_level: Logging level for the connector
        """
        logger.setLevel(logging_level)
        
        # Use default config if not provided
        self.config = config or TimescaleConnectorConfig()
        self.pool = None

    async def connect(self):
        """
        Establish connection pool to TimescaleDB with advanced configuration.
        """
        try:
            self.pool = await asyncpg.create_pool(
                self.config.dsn,
                max_size=self.config.max_connection_pool_size,
                command_timeout=self.config.connection_timeout
            )
            logger.info("Connected to TimescaleDB successfully")
            
            # Setup hypertable and retention policy if not exists
            await self._initialize_database()
        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            raise

    async def _initialize_database(self):
        """
        Initialize database with hypertable and retention policy.
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Create hypertable if not exists
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS ticks (
                        timestamp TIMESTAMPTZ NOT NULL,
                        symbol TEXT NOT NULL,
                        bid NUMERIC NOT NULL,
                        ask NUMERIC NOT NULL
                    );
                    
                    SELECT create_hypertable(
                        'ticks', 
                        'timestamp', 
                        if_not_exists => true
                    );
                """)
                
                # Set retention policy if configured
                if self.config.data_retention_days:
                    await conn.execute(f"""
                        SELECT add_retention_policy(
                            'ticks', 
                            INTERVAL '{self.config.data_retention_days} days'
                        );
                    """)

    async def fetch(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Event]:
        """
        Fetch tick data for a given symbol and time range.
        
        :param symbol: Trading symbol (e.g., 'EURUSD')
        :param start: Start datetime
        :param end: Optional end datetime
        :param limit: Optional limit on number of rows
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
        
        if limit:
            query += " LIMIT $4"
            params.append(limit)

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

    async def insert_ticks(
        self, 
        ticks: List[Dict[str, Any]],
        batch_size: int = 1000
    ):
        """
        Insert tick data into TimescaleDB with batch processing.
        
        :param ticks: List of tick dictionaries
        :param batch_size: Number of ticks to insert in a single batch
        """
        if not self.pool:
            await self.connect()

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Process ticks in batches
                for i in range(0, len(ticks), batch_size):
                    batch = ticks[i:i+batch_size]
                    await conn.executemany("""
                        INSERT INTO ticks (timestamp, symbol, bid, ask)
                        VALUES ($1, $2, $3, $4)
                    """, [
                        (
                            tick.get('timestamp', datetime.now()),
                            tick['symbol'], 
                            tick['bid'], 
                            tick['ask']
                        )
                        for tick in batch
                    ])

    async def close(self):
        """
        Close database connection pool.
        """
        if self.pool:
            await self.pool.close()
            logger.info("Closed TimescaleDB connection pool")