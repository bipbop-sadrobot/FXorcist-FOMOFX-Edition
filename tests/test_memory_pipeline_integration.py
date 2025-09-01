"""
Integration tests for memory system with training pipeline.
Tests memory lifecycle, batch operations, monitoring, and dashboard integration.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
import pytest
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_system.core import MemoryManager, MemoryEntry, analyze_memory_trends
from memory_system.api import app
from memory_system.store.sqlite_store import SQLiteStore
from memory_system.adapters import FAISSAdapter
from dashboard.components.system_status import SystemMonitor
from forex_ai_dashboard.pipeline.model_training import train_model
from forex_ai_dashboard.pipeline.feature_engineering import engineer_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("logs/memory_integration_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@pytest.fixture
def memory_manager():
    """Create a test memory manager with temporary storage."""
    store = SQLiteStore(db_path=":memory:")
    vector_index = FAISSAdapter(dim=384)
    return MemoryManager(vector_index=vector_index, store=store)

@pytest.fixture
def test_data():
    """Generate test market data and predictions."""
    dates = pd.date_range(start='2025-08-01', end='2025-08-02', freq='1min')
    data = pd.DataFrame({
        'close': np.random.normal(1.1000, 0.0002, len(dates)),
        'prediction': np.random.normal(1.1000, 0.0002, len(dates)),
        'confidence': np.random.uniform(0.6, 0.9, len(dates))
    }, index=dates)
    return data

async def test_batch_operations(memory_manager, test_data):
    """Test batch storage operations and performance."""
    logger.info("Testing batch operations")
    
    # Create batch entries
    entries = [
        MemoryEntry(
            id=f"pred_{i}",
            text=f"Price: {row.close:.5f}, Pred: {row.prediction:.5f}",
            metadata={
                'timestamp': idx.timestamp(),
                'confidence': row.confidence
            }
        )
        for i, (idx, row) in enumerate(test_data.iterrows())
    ]
    
    # Test batch storage
    batch_size = 100
    start_time = datetime.now()
    
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        for entry in batch:
            await memory_manager.store_entry(entry)
    
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"Batch storage complete. Duration: {duration:.2f}s")
    
    # Verify storage
    assert len(memory_manager.tiers['wm']) > 0
    assert duration < 5.0, "Batch storage too slow"

async def test_memory_lifecycle(memory_manager, test_data):
    """Test memory system during training lifecycle."""
    logger.info("Testing memory lifecycle")
    
    # 1. Store initial predictions
    entries = [
        MemoryEntry(
            id=f"pred_{i}",
            text=f"Price: {row.close:.5f}, Pred: {row.prediction:.5f}",
            metadata={
                'timestamp': idx.timestamp(),
                'confidence': row.confidence,
                'accuracy': abs(row.close - row.prediction)
            }
        )
        for i, (idx, row) in enumerate(test_data.iterrows())
    ]
    
    for entry in entries:
        await memory_manager.store_entry(entry)
    
    # 2. Simulate training preparation
    memory_manager._consolidate_once()  # Clean old entries
    
    # 3. Verify memory state
    wm_count = len(memory_manager.tiers['wm'])
    ltm_count = len(memory_manager.tiers['ltm'])
    logger.info(f"Memory state - WM: {wm_count}, LTM: {ltm_count}")
    
    # 4. Check quarantine
    quarantined = [
        e for t in memory_manager.tiers.values()
        for e in t
        if e.metadata.get('quarantined')
    ]
    logger.info(f"Quarantined entries: {len(quarantined)}")

async def test_monitoring_integration(memory_manager, test_data):
    """Test monitoring and stats collection."""
    logger.info("Testing monitoring integration")
    
    # 1. Generate memory usage data
    for _ in range(10):
        for i, (idx, row) in enumerate(test_data.iterrows()):
            entry = MemoryEntry(
                id=f"pred_{i}_{_}",
                text=f"Price: {row.close:.5f}, Pred: {row.prediction:.5f}",
                metadata={'timestamp': idx.timestamp()}
            )
            await memory_manager.store_entry(entry)
            await asyncio.sleep(0)  # Allow other tasks
    
    # 2. Collect memory stats
    memory_data = [
        (datetime.now().timestamp(), len(memory_manager.tiers['wm']))
        for _ in range(5)
    ]
    
    # 3. Analyze trends
    trends = analyze_memory_trends(memory_data)
    assert 'average_usage' in trends
    assert 'trend_slope' in trends
    
    logger.info(f"Memory trends: {json.dumps(trends, indent=2)}")

async def test_dashboard_integration(memory_manager):
    """Test dashboard component integration."""
    logger.info("Testing dashboard integration")
    
    # 1. Initialize system monitor
    config = MagicMock()
    monitor = SystemMonitor(config)
    
    # 2. Get system metrics
    metrics = monitor._get_system_metrics()
    assert 'memory_entries' in metrics
    assert 'recall_latency' in metrics
    
    # 3. Check monitoring thresholds
    status = monitor.check_status()
    assert 'recall_latency' in status
    assert 'quarantine_ratio' in status
    
    logger.info(f"Dashboard metrics: {metrics}")
    logger.info(f"System status: {status}")

@pytest.mark.asyncio
async def test_full_integration(memory_manager, test_data):
    """Test complete memory system integration."""
    logger.info("Starting full integration test")
    
    try:
        # 1. Test batch operations
        await test_batch_operations(memory_manager, test_data)
        
        # 2. Test memory lifecycle
        await test_memory_lifecycle(memory_manager, test_data)
        
        # 3. Test monitoring
        await test_monitoring_integration(memory_manager, test_data)
        
        # 4. Test dashboard
        await test_dashboard_integration(memory_manager)
        
        logger.info("Full integration test completed successfully")
        return True
        
    except Exception as e:
        logger.error("Integration test failed", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        asyncio.run(test_full_integration(MemoryManager(), pd.DataFrame()))
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        sys.exit(1)