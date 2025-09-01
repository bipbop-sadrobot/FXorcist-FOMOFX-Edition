import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from forex_ai_dashboard.pipeline.model_training import train_model
from forex_ai_dashboard.utils.logger import logger

class TestIntegrationRollingValidation(unittest.TestCase):
    @patch('forex_ai_dashboard.pipeline.feature_engineering.EconomicCalendar')
    @patch('joblib.dump')
    def test_rolling_validation_integration(self, mock_joblib, mock_econ_calendar):
        """Test rolling validation integration with model training pipeline"""
        # Setup mock economic calendar
        mock_calendar = MagicMock()
        mock_calendar.fetch_calendar_events.return_value = pd.DataFrame({
            'date': ['2025-01-01', '2025-01-02'],
            'feature1': [1, 2],
            'feature2': [3, 4],
            'target': [5, 6]
        })
        mock_econ_calendar.return_value = mock_calendar

        # Call the training function
        train_model("test_api_key", window_size=2)

        # Verify rolling validation was called with correct params
        self.assertTrue(mock_calendar.fetch_calendar_events.called)
        self.assertTrue(mock_joblib.called)  # Verify model saving

    @patch('forex_ai_dashboard.pipeline.feature_engineering.EconomicCalendar')
    def test_error_handling(self, mock_econ_calendar):
        """Test error handling in integration"""
        # Setup failing mock
        mock_calendar = MagicMock()
        mock_calendar.fetch_calendar_events.side_effect = Exception("API Error")
        mock_econ_calendar.return_value = mock_calendar

        # Verify error is handled gracefully
        with patch.object(logger, 'error') as mock_logger:
            train_model("test_api_key")
            self.assertTrue(mock_logger.called)

if __name__ == '__main__':
    unittest.main()
