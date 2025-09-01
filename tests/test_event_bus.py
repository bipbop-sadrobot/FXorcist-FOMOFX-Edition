from memory_system.event_bus import EventBus

def test_pubsub_roundtrip():
    bus = EventBus()
    seen = []
    bus.subscribe("ping", lambda p: seen.append(p))
    bus.publish("ping", {"ok": True})
    assert seen and seen[0]["ok"] is True\n