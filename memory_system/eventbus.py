from collections import defaultdict
from typing import Callable, Dict, Any, List, Optional
import threading, json, logging

class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Any], None]]] = defaultdict(list)
        self._history: List[tuple] = []

    def subscribe(self, event: str, handler: Callable[[Any], None]):
        self._subscribers[event].append(handler)

    def publish(self, event: str, payload: Any):
        self._history.append((event, payload))
        for handler in self._subscribers.get(event, []):
            try:
                handler(payload)
            except Exception as e:
                logging.exception("handler failed")

    def history(self) -> List[tuple]:
        return list(self._history)

# optional aiokafka-backed bus (async). If aiokafka not installed, users get a clear error.
try:
    import asyncio
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
    _have_aiokafka = True
except Exception:
    _have_aiokafka = False

if _have_aiokafka:
    class KafkaEventBusAsync:
        def __init__(self, bootstrap_servers: str = "localhost:9092", group_id: str = "ims-group"):
            self.bootstrap = bootstrap_servers
            self.group_id = group_id
            self._producer = None
            self._consumer = None
            self._subs = defaultdict(list)
            self._loop = asyncio.get_event_loop()
            self._running = False

        async def start(self):
            self._producer = AIOKafkaProducer(loop=self._loop, bootstrap_servers=self.bootstrap)
            await self._producer.start()
            self._consumer = AIOKafkaConsumer(loop=self._loop, bootstrap_servers=self.bootstrap, group_id=self.group_id)
            await self._consumer.start()
            self._running = True

        async def stop(self):
            if self._producer: await self._producer.stop()
            if self._consumer: await self._consumer.stop()
            self._running = False

        async def publish(self, topic: str, payload: Any):
            await self._producer.send_and_wait(topic, json.dumps(payload).encode())

        def subscribe(self, topic: str, handler: Callable[[Any], None]):
            self._subs[topic].append(handler)

        async def poll_forever(self, topics: Optional[List[str]] = None):
            if not self._consumer: return
            if topics:
                await self._consumer.subscribe(topics)
            while True:
                msg = await self._consumer.getone()
                try:
                    payload = json.loads(msg.value.decode())
                except Exception:
                    payload = msg.value
                for h in self._subs.get(msg.topic, []):
                    try: h(payload)
                    except Exception: pass
else:
    KafkaEventBusAsync = None
