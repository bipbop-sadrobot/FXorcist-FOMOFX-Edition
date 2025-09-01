from memory_system.event_bus import EventBus
from memory_system.metadata import SharedMetadata
from memory_system.memory import IntegratedMemorySystem

def test_memory_trends_and_resources():
    bus = EventBus()
    md = SharedMetadata()
    mem = IntegratedMemorySystem(bus, md, max_records=50)
    for i in range(30):
        mem.add_record({"model": "m1", "prediction": 0.8, "error": 0.1 + i*0.005})
    trend = mem.analyze_memory_trends()
    assert trend["n"] >= 1 and "mean_error" in trend
    plan = mem.prioritize_resources()
    assert "recommendations" in plan\n