# policy_overlay.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Tuple, Dict

def load_cate_bundle(path: str = "integrations/artifacts/cate_bundle.joblib") -> Dict:
    """Load saved causal effect estimates"""
    if not Path(path).exists():
        raise FileNotFoundError(f"No cate bundle found at {path}")
    return joblib.load(path)

def apply_cate_gate(signals: pd.Series, 
                    cate_bundle: Optional[Dict] = None,
                    cate_path: str = "integrations/artifacts/cate_bundle.joblib",
                    min_tau: float = 0.0,
                    scale_by_tau: bool = False) -> Tuple[pd.Series, Dict]:
    """
    Gate trading signals based on estimated causal effects.
    
    Args:
        signals: pd.Series of raw trading signals (-1, 0, 1)
        cate_bundle: optional pre-loaded bundle (Dict from load_cate_bundle)
        cate_path: path to joblib if bundle not provided
        min_tau: minimum treatment effect to allow trade
        scale_by_tau: if True, scale signal by normalized tau
    
    Returns:
        (gated_signals, metrics)
    """
    if cate_bundle is None:
        try:
            cate_bundle = load_cate_bundle(cate_path)
        except FileNotFoundError:
            # Fallback: pass through signals unchanged
            return signals, {"warning": "No cate bundle found - using raw signals"}
    
    # Extract components (format depends on estimation method)
    tau = cate_bundle.get("tau", None)  # point estimates
    tau_std = cate_bundle.get("tau_std", None)  # uncertainty
    
    if tau is None:
        return signals, {"warning": "No tau estimates in bundle"}
    
    # Convert to series if needed
    if isinstance(tau, (float, int)):
        tau = pd.Series(tau, index=signals.index)
    
    # Gate signals where tau < threshold
    gated = signals.copy()
    gated[tau < min_tau] = 0
    
    if scale_by_tau and tau_std is not None:
        # Optional: scale by normalized tau
        tau_normalized = tau / tau_std
        gated = gated * tau_normalized
    
    metrics = {
        "n_gated": (gated == 0).sum(),
        "mean_tau": float(tau.mean()),
        "min_tau_used": min_tau
    }
    
    return gated, metrics

if __name__ == "__main__":
    # Example usage
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals_csv", required=True)
    ap.add_argument("--signal_col", default="signal")
    ap.add_argument("--cate_bundle", default="integrations/artifacts/cate_bundle.joblib")
    ap.add_argument("--min_tau", type=float, default=0.0)
    ap.add_argument("--out", default="integrations/artifacts/gated_signals.csv")
    args = ap.parse_args()
    
    df = pd.read_csv(args.signals_csv, parse_dates=True, index_col=0)
    signals = df[args.signal_col]
    gated, metrics = apply_cate_gate(signals, cate_path=args.cate_bundle, min_tau=args.min_tau)
    
    # Save gated signals
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    gated.to_csv(args.out)
    print(f"Gated {metrics['n_gated']} signals (mean tau: {metrics['mean_tau']:.4f})")