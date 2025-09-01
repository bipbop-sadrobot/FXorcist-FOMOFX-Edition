from memory_system.federated import FederatedMemory
from memory_system.eventbus import EventBus
from memory_system.metadata import SharedMetadata

def test_federated_dp():
    bus = EventBus()
    md = SharedMetadata()
    fed = FederatedMemory(bus, md, enable_dp=True, dp_noise_std=0.1)
    res = fed.aggregate([{'params':{'w':1.0}, 'weight':1.0, 'signature':''}], eps=0.1, delta=1e-5)
    assert 'accounting' in res
