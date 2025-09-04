"""
Causal inference module for FXorcist using EconML.
Provides treatment effect estimation for market events and signals.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional, Union
from dataclasses import dataclass
import warnings

# Try to import econml, fallback to sklearn if not available
try:
    from econml.dml import LinearDML, CausalForestDML
    from econml.sklearn_extensions.linear_model import WeightedLasso
    HAS_ECONML = True
except ImportError:
    HAS_ECONML = False
    warnings.warn("econml not available, falling back to sklearn estimators")

from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

@dataclass
class TreatmentEffects:
    """Container for treatment effect estimates."""
    ate: float  # Average treatment effect
    cate: np.ndarray  # Conditional average treatment effects
    stderr: Optional[np.ndarray] = None  # Standard errors if available
    conf_intervals: Optional[np.ndarray] = None  # Confidence intervals if available

def prepare_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    standardize: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare feature matrix from dataframe.
    
    Args:
        df: Input dataframe
        feature_cols: List of feature column names. If None, uses all numeric columns
        standardize: Whether to standardize features
    
    Returns:
        Tuple of (feature matrix, feature names)
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude common non-feature columns
        exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'signal']
        feature_cols = [c for c in feature_cols if c not in exclude]
    
    X = df[feature_cols].values
    if standardize:
        X = StandardScaler().fit_transform(X)
    return X, feature_cols

def make_treatment(
    df: pd.DataFrame,
    treatment_col: str,
    threshold: Optional[float] = None,
    discrete: bool = True
) -> np.ndarray:
    """
    Convert a column into treatment vector.
    
    Args:
        df: Input dataframe
        treatment_col: Column name for treatment
        threshold: Optional threshold for discretization
        discrete: Whether to make treatment discrete (0/1)
    
    Returns:
        Treatment vector as numpy array
    """
    if treatment_col not in df.columns:
        raise ValueError(f"Treatment column {treatment_col} not found")
    
    T = df[treatment_col].values
    if discrete:
        if threshold is None:
            threshold = 0.0  # Default threshold
        T = (T > threshold).astype(int)
    return T

def make_outcome(
    df: pd.DataFrame,
    outcome_col: str = 'close',
    horizon: int = 1,
    returns: bool = True
) -> np.ndarray:
    """
    Create outcome vector, optionally as returns.
    
    Args:
        df: Input dataframe
        outcome_col: Column to use as outcome
        horizon: Forward horizon for returns
        returns: Whether to compute returns instead of raw values
    
    Returns:
        Outcome vector as numpy array
    """
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column {outcome_col} not found")
    
    if returns:
        Y = df[outcome_col].pct_change(horizon).shift(-horizon).values
    else:
        Y = df[outcome_col].shift(-horizon).values
    
    # Remove NaN from the end due to shifting
    Y = Y[:-horizon]
    return Y

def estimate_effects(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    discrete_treatment: bool = True,
    cv: int = 5,
    random_state: int = 42
) -> TreatmentEffects:
    """
    Estimate treatment effects using EconML if available, otherwise fallback.
    
    Args:
        X: Feature matrix
        T: Treatment vector
        Y: Outcome vector
        discrete_treatment: Whether treatment is discrete
        cv: Number of cross-validation folds
        random_state: Random seed
    
    Returns:
        TreatmentEffects object with estimates
    """
    if HAS_ECONML:
        # Use EconML's LinearDML
        model_t = (LogisticRegressionCV(cv=cv) if discrete_treatment 
                  else LassoCV(cv=cv))
        model_y = WeightedLasso(alpha=0.01)
        
        est = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=discrete_treatment,
            cv=cv,
            random_state=random_state
        )
        est.fit(Y=Y, T=T, X=X)
        
        # Get effects and confidence intervals
        effects = est.effect(X)
        stderr = est.effect_stderr(X)
        lb, ub = est.effect_interval(X, alpha=0.05)
        
        return TreatmentEffects(
            ate=float(effects.mean()),
            cate=effects,
            stderr=stderr,
            conf_intervals=np.column_stack([lb, ub])
        )
    else:
        # Fallback: Simple difference-in-means with random forest
        rf_t = RandomForestRegressor(n_estimators=100, random_state=random_state)
        rf_c = RandomForestRegressor(n_estimators=100, random_state=random_state)
        
        # Split by treatment
        mask_t = T == 1
        mask_c = ~mask_t
        
        # Fit separate models for treated and control
        rf_t.fit(X[mask_t], Y[mask_t])
        rf_c.fit(X[mask_c], Y[mask_c])
        
        # Estimate effects
        y1_pred = rf_t.predict(X)
        y0_pred = rf_c.predict(X)
        effects = y1_pred - y0_pred
        
        return TreatmentEffects(
            ate=float(effects.mean()),
            cate=effects,
            stderr=None,  # No stderr in simple fallback
            conf_intervals=None
        )

def summarize_effects(effects: TreatmentEffects) -> Dict[str, float]:
    """
    Create a summary dictionary of treatment effects.
    
    Args:
        effects: TreatmentEffects object
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'ate': effects.ate,
        'cate_mean': float(effects.cate.mean()),
        'cate_std': float(effects.cate.std()),
        'cate_median': float(np.median(effects.cate)),
        'cate_q25': float(np.percentile(effects.cate, 25)),
        'cate_q75': float(np.percentile(effects.cate, 75))
    }
    
    if effects.stderr is not None:
        summary.update({
            'ate_stderr': float(effects.stderr.mean()),
            'significant_effects': float((np.abs(effects.cate) > 2 * effects.stderr).mean())
        })
    
    return summary

# Example usage function
def analyze_market_event(
    df: pd.DataFrame,
    event_col: str,
    outcome_col: str = 'close',
    horizon: int = 1,
    feature_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze the causal effect of a market event.
    
    Args:
        df: DataFrame with market data
        event_col: Column indicating event occurrence
        outcome_col: Column to use as outcome
        horizon: Forward horizon for returns
        feature_cols: Optional list of feature columns
    
    Returns:
        Dictionary with analysis results
    """
    # Prepare data
    X, features = prepare_features(df, feature_cols)
    T = make_treatment(df, event_col, discrete=True)
    Y = make_outcome(df, outcome_col, horizon, returns=True)
    
    # Trim to matching lengths
    min_len = min(len(X), len(T), len(Y))
    X, T, Y = X[:min_len], T[:min_len], Y[:min_len]
    
    # Estimate effects
    effects = estimate_effects(X, T, Y)
    summary = summarize_effects(effects)
    
    return {
        'effects': effects,
        'summary': summary,
        'features': features,
        'n_samples': min_len,
        'treatment_ratio': float(T.mean())
    }

if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze market event effects')
    parser.add_argument('--csv', required=True, help='Path to CSV file')
    parser.add_argument('--event', required=True, help='Event column name')
    parser.add_argument('--outcome', default='close', help='Outcome column name')
    parser.add_argument('--horizon', type=int, default=1, help='Forward horizon')
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv)
    results = analyze_market_event(
        df,
        event_col=args.event,
        outcome_col=args.outcome,
        horizon=args.horizon
    )
    
    import json
    print(json.dumps(results['summary'], indent=2))