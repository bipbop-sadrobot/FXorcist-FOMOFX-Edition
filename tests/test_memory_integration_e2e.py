import unittest
import time
import json
from typing import List, Tuple, Dict, Any

from memory_system.core import MemoryManager, MemoryEntry, analyze_memory_trends, generate_insights_report
from memory_system.store.sqlite_store import SQLiteStore
from memory_system.eventbus import EventBus

class TestMemoryIntegrationE2E(unittest.TestCase):
    def setUp(self):
        self.store = SQLiteStore(db_path=":memory:")  # In-memory SQLite for testing
        self.event_bus = EventBus()
        self.memory_manager = MemoryManager(store=self.store, enable_embedding_hook=False)

    def tearDown(self):
        self.store.close()

    def test_e2e_memory_analysis_report(self):
        # 1. Add memory entries
        num_entries = 10
        memory_data: List[Tuple[float, float]] = []
        for i in range(num_entries):
            entry = MemoryEntry(id=f"entry_{i}", text=f"Memory entry {i}", tier="wm")
            self.memory_manager.store_entry(entry)
            memory_data.append((time.time(), i * 10.0))  # Simulate increasing memory usage

        # 2. Analyze memory trends
        trend_report = analyze_memory_trends(memory_data)

        # 3. Generate insights report
        insights_report = generate_insights_report(trend_report)
        report_dict = json.loads(insights_report)

        # 4. Assertions
        self.assertIsInstance(report_dict, dict)
        self.assertEqual(report_dict["datapoints_analyzed"], num_entries)
        self.assertIn("recommendations", report_dict)

        print(f"E2E Test Insights Report: {insights_report}")

if __name__ == '__main__':
    unittest.main()
