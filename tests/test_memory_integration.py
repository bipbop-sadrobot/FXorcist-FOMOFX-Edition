import unittest
import time
from memory_system.core import MemoryManager, MemoryEntry, analyze_memory_trends, generate_insights_report
from memory_system.store.sqlite_store import SQLiteStore
from memory_system.adapters import FAISSAdapter
from memory_system.embeddings import embed_text
from memory_system.anomaly import AnomalyDetector
import tempfile
import os

class TestMemoryIntegration(unittest.TestCase):
    def setUp(self):
        # Use a temporary directory for the SQLite store
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_memory.db")
        self.store = SQLiteStore(db_path=self.db_path)
        self.vector_index = FAISSAdapter(dim=384)  # all-MiniLM-L6-v2 has dimension 384
        self.memory_manager = MemoryManager(vector_index=self.vector_index, store=self.store, max_stm=100)
        self.anomaly_detector = AnomalyDetector(memory=self.memory_manager)

    def tearDown(self):
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_integration(self):
        # 1. Add records to memory
        from memory_system.embeddings import init_model
        init_model()
        entry1 = MemoryEntry(id="1", text="This is a test entry.", metadata={"source": "test"})
        self.memory_manager.store_entry(entry1)
        entry2 = MemoryEntry(id="2", text="This is another test entry.", metadata={"source": "test"})
        self.memory_manager.store_entry(entry2)

        # 2. Consolidate memory (optional, but good to test)
        self.memory_manager._consolidate_once()

        # 3. Simulate memory usage data
        memory_data = [(time.time(), 100), (time.time(), 110), (time.time(), 120)]

        # 4. Analyze memory trends
        trends = analyze_memory_trends(memory_data)

        # 5. Detect anomalies (this might require more setup depending on your anomaly detection)
        anomalies = self.anomaly_detector.detect_anomalies()

        # 6. Generate insights report
        report = generate_insights_report(trends)

        # Assert that the report is not empty
        self.assertTrue(report)

if __name__ == '__main__':
    unittest.main()
