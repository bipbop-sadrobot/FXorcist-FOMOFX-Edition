# mlfinlab_tools.py
# NOTE: MlFinLab packaging changed; some premium features moved. Use open-source subset for basic triple-barrier.
import pandas as pd
import numpy as np

def apply_triple_barrier(prices: pd.Series, events: pd.DataFrame, profit_taking: float=0.02, stop_loss: float=0.01, vertical_barrier=10):
    """
    A simple triple-barrier:
    - events: DataFrame with index=event_time and a 't1' column for vertical barrier (timestamp)
    - returns labels: +1 (profit), -1 (loss), 0 (no barrier hit)
    (This is a simple adaptation; for production use mlfinlab.get_events etc.)
    """
    labels = {}
    for t0, row in events.iterrows():
        t1 = row.get("t1", None)
        if pd.isna(t1):
            t1 = t0 + pd.Timedelta(days=vertical_barrier)
        start_price = prices.loc[t0]
        series = prices.loc[t0:t1]
        # relative returns
        rel = (series / start_price) - 1.0
        # check profit taking
        if (rel >= profit_taking).any():
            labels[t0] = 1
        elif (rel <= -stop_loss).any():
            labels[t0] = -1
        else:
            labels[t0] = 0
    return pd.Series(labels)

# Purged K-Fold: remove overlapping samples between train/test
def purged_time_series_split(index, n_splits=5, purge=1):
    """
    Very small purged split: yields (train_ix, test_ix)
    purge: number of samples to drop around test indices from the train set to avoid leakage
    """
    n = len(index)
    fold_sizes = [n // n_splits + (1 if x < (n % n_splits) else 0) for x in range(n_splits)]
    current = 0
    splits = []
    for fs in fold_sizes:
        start = current
        end = current + fs
        test_ix = np.arange(start, end)
        train_ix = np.setdiff1d(np.arange(n), np.arange(max(0, start-purge), min(n, end+purge)))
        splits.append((train_ix, test_ix))
        current = end
    return splits