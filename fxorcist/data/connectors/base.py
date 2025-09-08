from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
from fxorcist.events.event_bus import Event

class DataConnector(ABC):
    """Abstract base for all data connectors."""

    @abstractmethod
    async def fetch(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None
    ) -> List[Event]:
        """Fetch market data as list of Events."""
        pass