import unittest
from unittest.mock import MagicMock
import numpy as np
from forex_ai_dashboard.reinforcement.memory_schema import MemorySchema, PredictionRecord
from forex_ai_dashboard.reinforcement.memory_matrix import MemoryMatrix

class TestMemoryMatrix(unittest.TestCase):
    def setUp(self):
        # Setup mock memory schema with test records
        self.mock_memory = MagicMock(spec=MemorySchema)
        test_records = []
        for i in range(15):  # Generate 15 test records
            model_ver = "model_v1" if i % 2 == 0 else "model_v2"
            f1_val = 0.1 + (i * 0.02)
            f2_val = 0.2 + (i * 0.02)
            pred = 1.0 + (i * 0.01)
            actual = pred - 0.1 + (i * 0.005)
            
            test_records.append(
                PredictionRecord(
                    timestamp=None,
                    model_version=model_ver,
                    features={"f1": f1_val, "f2": f2_val},
                    prediction=pred,
                    actual=actual,
                    error_metrics={}
                )
            )
        self.mock_memory.get_records.return_value = test_records
        
        self.matrix = MemoryMatrix(self.mock_memory)

    def test_train_model_selector(self):
        """Test training the model selector"""
        self.matrix.train_model_selector()
        self.assertIsNotNone(self.matrix.meta_models['model_selector'])
        
    def test_predict_best_model(self):
        """Test model selection prediction"""
        self.matrix.train_model_selector()
        features = {"f1": 0.2, "f2": 0.3}
        best_model = self.matrix.predict_best_model(features)
        self.assertIsNotNone(best_model)

    def test_incremental_update(self):
        """Test incremental model updates"""
        self.matrix.train_model_selector()
        
        # Create new test records
        new_records = [
            PredictionRecord(
                timestamp=None,
                model_version="model_v1",
                features={"f1": 0.5, "f2": 0.6},
                prediction=1.2,
                actual=1.1,
                error_metrics={}
            ),
            PredictionRecord(
                timestamp=None,
                model_version="model_v2",
                features={"f1": 0.7, "f2": 0.8},
                prediction=1.3,
                actual=1.25,
                error_metrics={}
            )
        ]
        
        # Test incremental update
        self.matrix.incremental_update(new_records)
        # No assertion needed - just verifying no errors occur

if __name__ == '__main__':
    unittest.main()
