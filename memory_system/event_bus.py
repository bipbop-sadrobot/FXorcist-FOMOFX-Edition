from collections import defaultdict
from typing import Callable, Dict, Any, List

class EventBus:
    """Simple in-memory pub/sub event bus."""
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Any], None]]] = defaultdict(list)
        self._history: List[tuple] = [] # (event, payload)

    def subscribe(self, event: str, handler: Callable[[Any], None]):
        self._subscribers[event].append(handler)

    def publish(self, event: str, payload: Any):
        self._history.append((event, payload))
        for handler in self._subscribers.get(event, []):
            handler(payload)

    def history(self) -> List[tuple]:
        return list(self._history)

    # Hooks for persistence/backlog if desired
    def enable_persistence(self):
        pass\n