# run_decomposed_tests.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

def calculate_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calculate basic performance metrics"""
    if len(returns) < 2:
        return {"sharpe": 0.0, "hitrate": 0.0}
    
    # Annualized Sharpe (approximate)
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
    
    # Hit rate
    hitrate = (returns > 0).mean()
    
    return {
        "sharpe": float(sharpe),
        "hitrate": float(hitrate)
    }

def run_decomposed_tests(df: pd.DataFrame,
                        signal_col: str,
                        perturbations: List[Dict] = None) -> Dict[str, Dict]:
    """
    Run a series of decomposition tests to check strategy robustness.
    
    perturbations: list of dictionaries with:
        - name: test name
        - spread: additional spread to add
        - noise: std of noise to add to prices
        - delay: execution delay in periods
    """
    if perturbations is None:
        perturbations = [
            {"name": "baseline", "spread": 0.0, "noise": 0.0, "delay": 0},
            {"name": "high_spread", "spread": 0.0002, "noise": 0.0, "delay": 0},
            {"name": "noisy", "noise": 0.0001, "spread": 0.0, "delay": 0},
            {"name": "delayed", "delay": 1, "spread": 0.0, "noise": 0.0},
            {"name": "combined", "spread": 0.0002, "noise": 0.0001, "delay": 1}
        ]
    
    results = {}
    signals = df[signal_col]
    prices = df['close']
    
    for p in perturbations:
        # Apply perturbations
        adj_prices = prices.copy()
        if p["noise"] > 0:
            noise = np.random.normal(0, p["noise"], len(prices))
            adj_prices += noise
        
        # Simple returns calculation (add spread cost)
        rets = adj_prices.pct_change()
        if p["spread"] > 0:
            rets = rets - p["spread"] * (signals.diff() != 0)
        
        # Delay signals if specified
        if p["delay"] > 0:
            signals = signals.shift(p["delay"])
        
        # Calculate strategy returns
        strat_rets = signals * rets
        
        # Store metrics
        results[p["name"]] = calculate_metrics(strat_rets)
    
    return results

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with close prices and signals")
    ap.add_argument("--signal", default="signal", help="signal column name")
    ap.add_argument("--out", default="integrations/artifacts/decomp_report.csv")
    args = ap.parse_args()
    
    df = pd.read_csv(args.csv, parse_dates=True, index_col=0)
    
    # Run tests
    results = run_decomposed_tests(df, args.signal)
    
    # Save results
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).T.to_csv(args.out)
    
    # Print summary
    print("\nDecomposition Test Results:")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name:12} - Sharpe: {metrics['sharpe']:6.2f}, Hit Rate: {metrics['hitrate']:6.2%}")