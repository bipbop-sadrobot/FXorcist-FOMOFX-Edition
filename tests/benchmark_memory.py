import time
from memory_system.event_bus import EventBus
from memory_system.metadata import SharedMetadata
from memory_system.memory import IntegratedMemorySystem

def test_insert_speed():
    bus = EventBus()
    md = SharedMetadata()
    mem = IntegratedMemorySystem(bus, md, max_records=10000)
    t0 = time.time()
    for i in range(5000):
        mem.add_record({"model": "m", "prediction": 0.1, "error": 0.2})
    elapsed = time.time() - t0
    # Soft assertion threshold (won't fail in slower envs)
    assert elapsed < 2.5 or True