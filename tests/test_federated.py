from memory_system.event_bus import EventBus
from memory_system.metadata import SharedMetadata
from memory_system.federated import FederatedMemory

def test_federated_train_round():
    bus = EventBus()
    md = SharedMetadata()
    fed = FederatedMemory(bus, md, secret_key=b"test")
    model = fed.train_round()
    assert "w1" in model and "w2" in model