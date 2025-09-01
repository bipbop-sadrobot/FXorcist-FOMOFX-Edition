from memory_system.event_bus import EventBus
from memory_system.metadata import SharedMetadata
from memory_system.memory import IntegratedMemorySystem
from memory_system.federated import FederatedMemory
from memory_system.anomaly import AnomalyDetector

def main():
    event_bus = EventBus()
    metadata = SharedMetadata()
    memory = IntegratedMemorySystem(event_bus, metadata)
    federated = FederatedMemory(event_bus, metadata)
    detector = AnomalyDetector(memory)

    # Subscribe to anomaly events for demo
    def on_anomaly(payload):
        print("[EVENT] anomaly_detected:", payload)
    event_bus.subscribe("anomaly_detected", on_anomaly)

    # Simulate a stream of predictions
    import random
    for t in range(60):
        err = abs(random.gauss(0.15, 0.08))
        feats = {"x1": random.random(), "x2": random.random()}
        memory.add_record({
            "ts": t, "model": "mA", "prediction": random.random(),
            "error": err, "features": feats
        })

        if memory.should_trigger_federated_round(period=15):
            federated.train_round()

        anomalies = detector.detect_anomalies()
        if anomalies:
            event_bus.publish("anomaly_detected", anomalies)

    print("Insights:", memory.generate_insights_report())
    print("Resource plan:", memory.prioritize_resources())

if __name__ == "__main__":
    main()