import unittest
from datetime import datetime, timedelta
from forex_ai_dashboard.reinforcement.model_tracker import ModelTracker

class TestModelTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = ModelTracker()
        self.tracker.set_model_version("test_model_v1")
        self.test_features = {"feature1": 0.5, "feature2": 1.2}
        self.test_timestamp = datetime.now()

    def test_record_prediction(self):
        """Test recording a prediction"""
        result = self.tracker.record_prediction(
            features=self.test_features,
            prediction=1.0,
            timestamp=self.test_timestamp
        )
        self.assertIn("Recorded prediction", result)

    @unittest.skip("Not implemented")
    def test_record_outcome(self):
        """Test recording an outcome"""
        self.tracker.record_prediction(
            features=self.test_features,
            prediction=1.0,
            timestamp=self.test_timestamp
        )
        
        self.tracker.record_outcome(
            timestamp=self.test_timestamp,
            actual=0.9,
            error_metrics={"RMSE": 0.1},
            feature_importance={"feature1": 0.7}
        )
        
        metrics = self.tracker.get_performance_metrics()
        self.assertEqual(metrics["prediction_count"], 1)

    def test_drift_detection(self):
        """Test drift detection logic"""
        # Record good prediction
        self.tracker.record_prediction(
            features=self.test_features,
            prediction=1.0,
            timestamp=self.test_timestamp - timedelta(days=1)
        )
        self.tracker.record_outcome(
            timestamp=self.test_timestamp - timedelta(days=1),
            actual=0.95,
            error_metrics={"RMSE": 0.05}
        )
        
        # Record bad prediction that should trigger drift
        self.tracker.record_prediction(
            features=self.test_features,
            prediction=1.0,
            timestamp=self.test_timestamp
        )
        self.tracker.record_outcome(
            timestamp=self.test_timestamp,
            actual=0.5,
            error_metrics={"RMSE": 0.5}
        )
        
        self.assertTrue(self.tracker.check_for_drift(threshold=0.2))

if __name__ == '__main__':
    unittest.main()
