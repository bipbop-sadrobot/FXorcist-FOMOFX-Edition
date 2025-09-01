import unittest
from datetime import datetime, timedelta
from forex_ai_dashboard.reinforcement.memory_schema import MemorySchema, PredictionRecord

class TestMemorySchema(unittest.TestCase):
    def setUp(self):
        self.schema = MemorySchema()
        self.test_record = PredictionRecord(
            timestamp=datetime.now(),
            model_version="test_model_v1",
            features={"feature1": 0.5, "feature2": 1.2},
            prediction=1.0,
            actual=0.9,
            error_metrics={"RMSE": 0.1, "MAE": 0.08},
            feature_importance={"feature1": 0.7, "feature2": 0.3}
        )

    def test_add_and_retrieve_record(self):
        """Test adding and retrieving a single record"""
        self.schema.add_record(self.test_record)
        records = self.schema.get_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].model_version, "test_model_v1")

    def test_filter_records_by_model(self):
        """Test filtering records by model version"""
        self.schema.add_record(self.test_record)
        self.schema.add_record(PredictionRecord(
            timestamp=datetime.now(),
            model_version="other_model",
            features={},
            prediction=0.0
        ))
        
        filtered = self.schema.get_records(model_version="test_model_v1")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].model_version, "test_model_v1")

    def test_filter_records_by_date(self):
        """Test filtering records by date range"""
        now = datetime.now()
        self.schema.add_record(PredictionRecord(
            timestamp=now - timedelta(days=2),
            model_version="test_model_v1",
            features={},
            prediction=0.0
        ))
        self.schema.add_record(self.test_record)
        
        filtered = self.schema.get_records(start_date=now - timedelta(days=1))
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].model_version, "test_model_v1")

    def test_calculate_metrics(self):
        """Test calculating aggregate metrics"""
        self.schema.add_record(self.test_record)
        self.schema.add_record(PredictionRecord(
            timestamp=datetime.now(),
            model_version="test_model_v1",
            features={},
            prediction=1.1,
            actual=1.0,
            error_metrics={"RMSE": 0.1}
        ))
        
        metrics = self.schema.calculate_meta_features()
        self.assertEqual(metrics['total_predictions'], 2)
        self.assertAlmostEqual(metrics['avg_error'], 0.1)

if __name__ == '__main__':
    unittest.main()
