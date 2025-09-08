import asyncpg
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
import logging
import json
import os
import pandas as pd

from fxorcist.events.event_bus import Event
from fxorcist.data.connectors.base import DataConnector

logger = logging.getLogger(__name__)

class TimescaleConnectorConfig:
    """Enhanced configuration for TimescaleDB connector."""
    def __init__(
        self, 
        dsn: Optional[str] = None,
        max_connection_pool_size: int = 10,
        connection_timeout: int = 10,
        data_retention_days: Optional[int] = 365,
        enable_compression: bool = True,
        enable_continuous_aggregates: bool = True
    ):
        """
        Initialize TimescaleDB connector configuration.
        
        :param dsn: Database connection string
        :param max_connection_pool_size: Maximum number of connections in the pool
        :param connection_timeout: Connection timeout in seconds
        :param data_retention_days: Number of days to retain historical data
        :param enable_compression: Enable TimescaleDB compression
        :param enable_continuous_aggregates: Enable continuous aggregates for OHLCV
        """
        self.dsn = dsn or os.getenv(
            'TIMESCALEDB_DSN', 
            'postgresql://user:pass@localhost:5432/fxorcist'
        )
        self.max_connection_pool_size = max_connection_pool_size
        self.connection_timeout = connection_timeout
        self.data_retention_days = data_retention_days
        self.enable_compression = enable_compression
        self.enable_continuous_aggregates = enable_continuous_aggregates

class TimescaleConnector(DataConnector):
    """Advanced TimescaleDB connector with compression and aggregation support."""
    
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
            
            # Setup hypertable, retention policy, compression, and continuous aggregates
            await self._initialize_database()
        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            raise

    async def _initialize_database(self):
        """
        Initialize database with advanced TimescaleDB features.
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
                
                # Enable compression if configured
                if self.config.enable_compression:
                    await conn.execute("""
                        ALTER TABLE ticks SET (
                            timescaledb.compress,
                            timescaledb.compress_segmentby = 'symbol'
                        );
                        
                        SELECT add_compression_policy('ticks', INTERVAL '7 days');
                    """)
                
                # Create continuous aggregates for OHLCV if configured
                if self.config.enable_continuous_aggregates:
                    await conn.execute("""
                        CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1m
                        WITH (timescaledb.continuous) AS
                        SELECT 
                            time_bucket('1 minute', timestamp) as bucket,
                            symbol,
                            FIRST(bid, timestamp) as open,
                            MAX(bid) as high,
                            MIN(bid) as low,
                            LAST(bid, timestamp) as close,
                            AVG(bid) as vwap,
                            COUNT(*) as volume
                        FROM ticks
                        GROUP BY bucket, symbol;
                        
                        -- Create an index on the continuous aggregate for faster querying
                        CREATE INDEX IF NOT EXISTS ohlcv_1m_bucket_symbol_idx 
                        ON ohlcv_1m (bucket, symbol);
                    """)

    async def fetch(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
        resolution: str = 'tick',
        limit: Optional[int] = None
    ) -> Union[List[Event], pd.DataFrame]:
        """
        Fetch tick or aggregated data for a given symbol and time range.
        
        :param symbol: Trading symbol (e.g., 'EURUSD')
        :param start: Start datetime
        :param end: Optional end datetime
        :param resolution: 'tick', '1m', '5m', '1h'
        :param limit: Optional limit on number of rows
        :return: List of tick events or OHLCV DataFrame
        """
        if not self.pool:
            await self.connect()

        # Determine query based on resolution
        if resolution == 'tick':
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
                    logger.error(f"Error fetching tick data from TimescaleDB: {e}")
                    raise

            return events
        
        else:
            # Use continuous aggregate for OHLCV data
            resolution_map = {
                '1m': '1 minute',
                '5m': '5 minutes',
                '1h': '1 hour'
            }
            
            if resolution not in resolution_map:
                raise ValueError(f"Unsupported resolution: {resolution}")
            
            query = """
            SELECT 
                bucket, symbol, open, high, low, close, vwap, volume 
            FROM ohlcv_1m
            WHERE symbol = $1 AND bucket >= $2
            """
            params = [symbol, start]
            
            if end:
                query += " AND bucket <= $3"
                params.append(end)
            
            if limit:
                query += " LIMIT $4"
                params.append(limit)
            
            async with self.pool.acquire() as conn:
                try:
                    rows = await conn.fetch(query, *params)
                    
                    # Convert to pandas DataFrame
                    df = pd.DataFrame(rows, columns=[
                        'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'vwap', 'volume'
                    ])
                    
                    # Convert numeric columns
                    numeric_cols = ['open', 'high', 'low', 'close', 'vwap', 'volume']
                    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
                    
                    return df
                except Exception as e:
                    logger.error(f"Error fetching OHLCV data from TimescaleDB: {e}")
                    raise

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