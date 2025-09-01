import pandas as pd
from forex_ai_dashboard.pipeline.feature_engineering import engineer_features

def test_engineer_features():
    sample_df = pd.DataFrame({'bid': [1.0, 1.01, 1.02], 'ask': [1.001, 1.011, 1.021]})
    engineered = engineer_features(sample_df)
    assert 'ma_20' in engineered.columns
    assert 'rsi' in engineered.columns
    assert 'lag_1' in engineered.columns
    assert not engineered.isnull().any().any()

if __name__ == "__main__":
    test_engineer_features()
    print("Tests passed")
