import asyncio
import json
from typing import AsyncIterator, List
from datetime import datetime

from fxorcist.events.event_bus import Event

class ReplayEngine:
    """Abstract replay interface."""

    async def append(self, event: Event):
        pass

    async def replay(self, start_ts: datetime, end_ts: datetime) -> AsyncIterator[Event]:
        pass

class InMemoryReplay(ReplayEngine):
    def __init__(self):
        self.events = []

    async def append(self, event: Event):
        self.events.append(event)
        self.events.sort(key=lambda e: e.timestamp)

    async def replay(self, start_ts: datetime, end_ts: datetime) -> AsyncIterator[Event]:
        for event in self.events:
            if start_ts <= event.timestamp <= end_ts:
                yield event

class NATSReplay(ReplayEngine):
    def __init__(self, nats_url: str = "nats://localhost:4222", stream: str = "market-data"):
        self.nats_url = nats_url
        self.stream = stream
        self.nc = None
        self.js = None

    async def connect(self):
        import nats
        self.nc = await nats.connect(self.nats_url)
        self.js = self.nc.jetstream()

        # Ensure stream exists
        try:
            await self.js.stream_info(self.stream)
        except Exception:
            await self.js.add_stream(name=self.stream, subjects=[f"{self.stream}.>"])

    async def append(self, event: Event):
        if not self.js:
            await self.connect()
        subject = f"{self.stream}.{event.type}"
        payload = json.dumps({
            "timestamp": event.timestamp.isoformat(),
            "payload": event.payload
        }).encode()
        await self.js.publish(subject, payload)

    async def replay(self, start_ts: datetime, end_ts: datetime) -> AsyncIterator[Event]:
        if not self.js:
            await self.connect()

        subject = f"{self.stream}.>"
        start_seq = 1  # In real app, find by timestamp

        psub = await self.js.pull_subscribe(subject, "replay-consumer")
        while True:
            msgs = await psub.fetch(10, timeout=1)
            if not msgs:
                break
            for msg in msgs:
                data = json.loads(msg.data)
                ts = datetime.fromisoformat(data["timestamp"])
                if ts > end_ts:
                    await msg.ack()
                    return
                if ts >= start_ts:
                    yield Event(timestamp=ts, type=msg.subject.split(".")[-1], payload=data["payload"])
                await msg.ack()