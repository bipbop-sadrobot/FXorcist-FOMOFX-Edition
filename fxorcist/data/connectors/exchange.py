import ccxt.async_support as ccxt
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import asyncio
import logging

from .base import DataConnector
from fxorcist.events.event_bus import Event

logger = logging.getLogger(__name__)

class ExchangeConnector(DataConnector):
    """Fetch live or historical data from crypto/forex exchanges via CCXT."""

    def __init__(self, exchange_id: str = "kraken", api_key: str = None, secret: str = None):
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'timeout': 30000,
        })
        self.exchange_id = exchange_id

    async def fetch(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None
    ) -> List[Event]:
        """Fetch OHLCV data from exchange."""
        logger.info(f"Fetching {symbol} data from {self.exchange_id}...")

        # Convert to milliseconds
        since = int(start.replace(tzinfo=timezone.utc).timestamp() * 1000)
        limit = 1000  # Max per request

        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, '1m', since=since, limit=limit)
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            raise

        events = []
        for entry in ohlcv:
            timestamp = datetime.fromtimestamp(entry[0] / 1000, tz=timezone.utc)
            if end and timestamp > end:
                break

            # CCXT OHLCV: [timestamp, open, high, low, close, volume]
            payload = {
                "symbol": symbol,
                "open": float(entry[1]),
                "high": float(entry[2]),
                "low": float(entry[3]),
                "close": float(entry[4]),
                "volume": float(entry[5]),
                "mid": (float(entry[1]) + float(entry[4])) / 2  # Use open/close mid
            }

            events.append(Event(
                timestamp=timestamp,
                type="bar",
                payload=payload
            ))

        logger.info(f"Fetched {len(events)} bars for {symbol}")
        return events

    async def close(self):
        """Close exchange connection."""
        if hasattr(self.exchange, 'close'):
            await self.exchange.close()
        logger.info(f"Closed {self.exchange_id} connection")