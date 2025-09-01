import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from forex_ai_dashboard.pipeline.rolling_validation import rolling_validation
from forex_ai_dashboard.utils.logger import logger


class TestRollingValidation(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range(start="2025-01-01", periods=100)
        self.df = pd.DataFrame(
            {
                "date": dates,
                "feature1": np.random.rand(100),
                "feature2": np.random.rand(100),
                "target": np.random.rand(100),
            }
        )
        self.model = MagicMock()
        self.model.fit.return_value = self.model
        self.model.predict.side_effect = lambda X: np.random.rand(len(X))

    def test_basic(self):
        score = rolling_validation(self.model, self.df, window_size=30, horizon=5, step=5, metric="mse")
        self.assertIsInstance(score, float)
        self.assertTrue(self.model.fit.called)
        self.assertTrue(self.model.predict.called)

    def test_custom_metric(self):
        mae = rolling_validation(self.model, self.df, 25, horizon=3, metric="mae")
        self.assertIsInstance(mae, float)

        def sse(y_true, y_pred):
            return float(np.sum((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

        cu = rolling_validation(self.model, self.df, 25, horizon=3, metric=sse)
        self.assertIsInstance(cu, float)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            rolling_validation(self.model, self.df, 0)
        with self.assertRaises(ValueError):
            rolling_validation(self.model, self.df.drop(columns=["target"]), 20)

    def test_logging(self):
        with patch.object(logger, "info") as mock_log:
            rolling_validation(self.model, self.df, 20)
            self.assertTrue(mock_log.called)


if __name__ == "__main__":
    unittest.main()
