# estimate_causal_effects.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Optional

def estimate_causal_effects(df: pd.DataFrame,
                          signal_col: str,
                          horizon: int = 1,
                          use_econml: bool = True) -> Dict:
    """
    Estimate causal effects of trading signals on returns.
    Falls back to simple two-regressor if EconML not available.
    """
    try:
        if use_econml:
            from econml.dml import DML
            from sklearn.ensemble import RandomForestRegressor
            
            # Prepare features (exclude signal and target)
            feature_cols = [c for c in df.columns if c not in [signal_col, 'returns']]
            X = df[feature_cols]
            T = df[signal_col]
            
            # Forward returns as outcome
            y = df['returns'].shift(-horizon).fillna(0)
            
            # First stage models
            est = DML(
                model_y=RandomForestRegressor(n_estimators=100),
                model_t=RandomForestRegressor(n_estimators=100),
                cv=5  # number of folds for cross-fitting
            )
            
            est.fit(Y=y, T=T, X=X)
            tau_hat = est.effect(X)
            
            return {
                "method": "econml_dml",
                "tau": tau_hat,
                "tau_std": est.effect_stderr(X),
                "features_used": feature_cols
            }
            
    except ImportError:
        print("EconML not available, using fallback estimator")
    
    # Simple fallback: two-regressor approach
    pos_mask = df[signal_col] > 0
    neg_mask = df[signal_col] < 0
    
    # Compare mean returns
    pos_ret = df.loc[pos_mask, 'returns'].mean()
    neg_ret = df.loc[neg_mask, 'returns'].mean()
    
    return {
        "method": "two_regressor",
        "tau": pos_ret - neg_ret,
        "pos_returns": pos_ret,
        "neg_returns": neg_ret
    }

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with features, signal, returns")
    ap.add_argument("--signal", default="signal", help="signal column name")
    ap.add_argument("--horizon", type=int, default=1, help="forecast horizon")
    ap.add_argument("--save", default="integrations/artifacts/cate_bundle.joblib")
    args = ap.parse_args()
    
    df = pd.read_csv(args.csv, parse_dates=True, index_col=0)
    
    # Ensure we have required columns
    required = [args.signal, 'returns']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Estimate effects
    results = estimate_causal_effects(df, args.signal, args.horizon)
    
    # Save bundle
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, args.save)
    print(f"Saved cate bundle to {args.save}")
    print(f"Method: {results['method']}")
    print(f"Average treatment effect: {float(np.mean(results['tau'])):.4f}")