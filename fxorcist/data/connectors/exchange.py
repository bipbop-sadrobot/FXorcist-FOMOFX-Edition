import ccxt.async_support as ccxt
from datetime import datetime, timezone
from typing import List, Optional

from fxorcist.data.connectors.base import DataConnector
from fxorcist.events.event_bus import Event

class ExchangeConnector(DataConnector):
    def __init__(self, exchange_id: str = "kraken"):
        self.exchange = getattr(ccxt, exchange_id)()

    async def fetch(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None
    ) -> List[Event]:
        """Fetch OHLCV or tick data from exchange."""
        # Convert datetime to ms timestamp
        since = int(start.replace(tzinfo=timezone.utc).timestamp() * 1000)
        limit = 1000  # max per request

        ohlcv = await self.exchange.fetch_ohlcv(symbol, "1m", since=since, limit=limit)
        events = []
        for entry in ohlcv:
            timestamp = datetime.fromtimestamp(entry[0] / 1000, tz=timezone.utc)
            if end and timestamp > end:
                break
            events.append(Event(
                timestamp=timestamp,
                type="bar",
                payload={
                    "symbol": symbol,
                    "open": entry[1],
                    "high": entry[2],
                    "low": entry[3],
                    "close": entry[4],
                    "volume": entry[5]
                }
            ))
        return events

    async def close(self):
        await self.exchange.close()