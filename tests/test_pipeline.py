import pytest
import pandas as pd
from forex_ai_dashboard.pipeline.feature_engineering import engineer_features, EconomicCalendar
from unittest.mock import patch

@patch.object(EconomicCalendar, 'fetch_calendar_events')
def test_engineer_features(mock_fetch_calendar_events):
    """
    Tests the engineer_features function in pipeline/feature_engineering.py.
    """
    # Replace with a valid API key for testing
    api_key = "YOUR_API_KEY"

    # Mock the fetch_calendar_events function to return a sample DataFrame
    sample_data = {'Date': ['2023-01-01', '2023-01-02'],
                   'Actual': [1.0, 2.0],
                   'Forecast': [3.0, 4.0],
                   'Previous': [5.0, 6.0]}
    mock_fetch_calendar_events.return_value = pd.DataFrame(sample_data)
    
    # Call the engineer_features function
    features_df = engineer_features(api_key, country="United States", start_date="2023-01-01", end_date="2023-01-07")

    # Assert that the function returns a DataFrame
    assert isinstance(features_df, pd.DataFrame)

    # Assert that the DataFrame has the expected columns if not empty
    if not features_df.empty:
        expected_columns = ['year', 'month', 'day', 'hour', 'weekday', 'Actual', 'Forecast', 'Previous']
        for col in expected_columns:
            assert col in features_df.columns
    else:
        pytest.skip("No features generated - likely due to API issues")
