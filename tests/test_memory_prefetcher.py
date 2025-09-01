import pytest
from datetime import datetime, timedelta
import logging
from forex_ai_dashboard.reinforcement.event_bus import EventBus
from forex_ai_dashboard.reinforcement.memory_matrix import MemoryMatrix
from forex_ai_dashboard.reinforcement.model_tracker import ModelTracker
from forex_ai_dashboard.reinforcement.integrated_memory import IntegratedMemorySystem
from forex_ai_dashboard.reinforcement.memory_prefetcher import MemoryPrefetcher
from forex_ai_dashboard.reinforcement.shared_metadata import SharedMetadata

@pytest.fixture
def setup_prefetcher():
    event_bus = EventBus()
    metadata = SharedMetadata()
    integrated_memory = IntegratedMemorySystem(event_bus, metadata)
    memory_matrix = MemoryMatrix(integrated_memory.rolling_memory)
    model_tracker = ModelTracker()  # No event_bus argument
    prefetcher = integrated_memory.prefetcher  # Use the prefetcher from integrated_memory
    return prefetcher, event_bus, integrated_memory, model_tracker

def test_prefetcher_records_access_patterns(setup_prefetcher):
    prefetcher, event_bus, integrated_memory, model_tracker = setup_prefetcher

    # Simulate a prediction event
    event_bus.publish('new_prediction', {
        'timestamp': datetime.now().isoformat(),
        'model_version': 'test_model',
        'features': {'feature1': 0.5, 'feature2': 0.75},
        'prediction': 0.6
    })

    # Verify access patterns are recorded
    assert prefetcher.access_patterns['model_usage'].get('test_model', 0) == 1
    assert prefetcher.access_patterns['feature_access'].get('feature1', 0) == 1
    assert prefetcher.access_patterns['feature_access'].get('feature2', 0) == 1
    assert len(prefetcher.access_patterns['temporal_patterns']) == 1

def test_prefetcher_predicts_future_needs(setup_prefetcher):
    prefetcher, event_bus, integrated_memory, model_tracker = setup_prefetcher

    # Setup model metadata to ensure required features are known
    integrated_memory.metadata.model_versions['test_model'] = {
        'required_features': ['feature0']
    }
    model_tracker.set_model_version('test_model')

    # Simulate multiple prediction events with consistent features
    base_time = datetime.now()
    for i in range(50):  # Reduced to 50 for faster tests; threshold is 10
        event_bus.publish('new_prediction', {
            'timestamp': (base_time + timedelta(seconds=i * 2)).isoformat(),
            'model_version': 'test_model',
            'features': {f'feature{i % 10}': 0.5 + i * 0.01},  # Cycle features
            'prediction': 0.6
        })

    # Trigger pattern analysis
    prefetcher._analyze_patterns({})

    # Verify prefetch recommendations
    recommendations = prefetcher.get_prefetch_recommendations()
    assert 'predicted_feature_count' in recommendations
    assert recommendations['model_version'] == 'test_model'
    assert recommendations['predicted_feature_count'] >= 1
    assert prefetcher.is_fitted

def test_prefetcher_triggers_prefetch_operation(setup_prefetcher, caplog):
    prefetcher, event_bus, integrated_memory, model_tracker = setup_prefetcher

    # Setup model metadata
    integrated_memory.metadata.model_versions['test_model'] = {
        'required_features': ['feature1', 'feature2']
    }
    model_tracker.set_model_version('test_model')

    # Simulate enough events to trigger analysis
    base_time = datetime.now()
    for i in range(10):
        event_bus.publish('new_prediction', {
            'timestamp': (base_time + timedelta(seconds=i)).isoformat(),
            'model_version': 'test_model',
            'features': {'feature1': 0.5, 'feature2': 0.75},
            'prediction': 0.6
        })

    # Verify prefetch operation with caplog
    caplog.clear()
    with caplog.at_level(logging.INFO, logger='forex_ai_dashboard.reinforcement.memory_prefetcher'):
        prefetcher._analyze_patterns({})

    assert "Prefetching data for model test_model" in caplog.text
    assert "Prefetching model weights for test_model" in caplog.text
    assert "Prefetching feature data: feature1" in caplog.text
    assert "Prefetching feature data: feature2" in caplog.text

def test_prefetcher_handles_insufficient_data(setup_prefetcher, caplog):
    prefetcher, event_bus, integrated_memory, model_tracker = setup_prefetcher

    # Simulate one event (below threshold)
    event_bus.publish('new_prediction', {
        'timestamp': datetime.now().isoformat(),
        'model_version': 'test_model',
        'features': {'feature1': 0.5},
        'prediction': 0.6
    })

    # Analyze (should not fit)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='forex_ai_dashboard.reinforcement.memory_prefetcher'):
        prefetcher._analyze_patterns({})
    assert "Insufficient patterns for analysis" in caplog.text
    assert not prefetcher.is_fitted

def test_prefetcher_handles_invalid_event(setup_prefetcher, caplog):
    prefetcher, event_bus, integrated_memory, model_tracker = setup_prefetcher

    # Invalid timestamp
    event_bus.publish('new_prediction', {
        'timestamp': 'Invalid timestamp in event data',
        'model_version': 'test_model',
        'features': {'feature1': 0.5},
        'prediction': 0.6
    })
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger='forex_ai_dashboard.reinforcement.memory_prefetcher'):
        event_bus.publish('new_prediction', {})  # Missing keys
    assert "Invalid timestamp in event data" in caplog.text
    assert "Invalid event data" in caplog.text
