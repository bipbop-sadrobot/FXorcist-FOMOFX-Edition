"""
Robust walk-forward (rolling) validation for time series.

Key improvements:
- True walk-forward with configurable horizon & step
- Preserves time order (no leakage)
- Works with scikit-like estimators (fit/predict)
- Optional custom metric
"""
from typing import Callable, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from forex_ai_dashboard.utils.logger import logger


def _resolve_metric(metric: Union[str, Callable]) -> Callable:
    if isinstance(metric, str):
        m = metric.lower()
        if m == "mse":
            return mean_squared_error
        if m == "mae":
            return mean_absolute_error
        raise ValueError("Unknown metric string. Use 'mse', 'mae', or provide a callable.")
    if callable(metric):
        return metric
    raise TypeError("metric must be str or callable.")


def rolling_validation(
    model,
    data: pd.DataFrame,
    window_size: int,
    *,
    target_col: str = "target",
    horizon: int = 1,
    step: int = 1,
    metric: Union[str, Callable] = "mse",
    verbose: bool = False,
) -> float:
    """
    Walk-forward validation:
      train on [0:t), predict on [t:t+horizon), slide by 'step'

    Args:
        model: estimator with .fit(X,y) and .predict(X)
        data: time-ordered DataFrame
        window_size: number of rows used for training at each step
        target_col: name of target column
        horizon: forecast horizon (rows ahead)
        step: slide step
        metric: 'mse' | 'mae' | callable(y_true, y_pred)
        verbose: log each fold score

    Returns:
        float: average metric over all folds
    """
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if step <= 0:
        raise ValueError("step must be > 0")
    if len(data) <= window_size + horizon:
        raise ValueError("Not enough data for the given window_size and horizon.")

    metric_fn = _resolve_metric(metric)
    scores = []
    n = len(data)

    # ensure sorted by time if a 'date' column exists
    if "date" in data.columns:
        data = data.sort_values("date")

    features = data.drop(columns=[target_col])
    target = data[target_col]

    fold = 0
    for t in range(window_size, n - horizon + 1, step):
        fold += 1
        train_slice = slice(t - window_size, t)           # [t-window, t)
        test_slice = slice(t, t + horizon)                # [t, t+horizon)

        X_train = features.iloc[train_slice, :]
        y_train = target.iloc[train_slice]
        X_test = features.iloc[test_slice, :]
        y_test = target.iloc[test_slice]

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Allow scalar -> array broadcast
            if np.ndim(y_pred) == 0:
                y_pred = np.full_like(y_test.values, float(y_pred), dtype=float)

            score = float(metric_fn(y_test, y_pred))
            scores.append(score)
            if verbose:
                logger.info(f"[rolling] fold={fold:03d} {metric_fn.__name__}={score:.6f}")
        except Exception as e:
            logger.error(f"Error in fold {fold}: {e}")
            raise

    avg = float(np.mean(scores)) if scores else float("nan")
    logger.info(
        f"Completed rolling validation with {fold} folds, "
        f"avg_{metric_fn.__name__}={avg:.6f}"
    )
    return avg
