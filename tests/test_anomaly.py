from memory_system.event_bus import EventBus
from memory_system.metadata import SharedMetadata
from memory_system.memory import IntegratedMemorySystem
from memory_system.anomaly import AnomalyDetector

def test_anomaly_detector_min_samples():
    bus = EventBus()
    md = SharedMetadata()
    mem = IntegratedMemorySystem(bus, md)
    det = AnomalyDetector(mem)
    # Not enough samples → no anomalies by design
    for i in range(5):
        mem.add_record({"model": "m", "prediction": 0.5, "error": 0.1, "features": {"a": i}})
    assert det.detect_anomalies() == []\n