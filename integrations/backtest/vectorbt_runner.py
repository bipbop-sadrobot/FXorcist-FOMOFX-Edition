# vectorbt_runner.py
import vectorbt as vbt
import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path

def ma_cross_sweep(close: pd.Series, fast_windows: List[int], slow_windows: List[int], min_diff:int = 1) -> Dict[str, Dict]:
    """
    Run a simple moving-average crossover sweep and return summary metrics.
    Returns a dict keyed by 'f{fast}-s{slow}' with {'total_return','sharpe','max_drawdown'}
    """
    results = {}
    close = close.dropna()
    for f in fast_windows:
        for s in slow_windows:
            if f + min_diff >= s:  # ensure fast < slow
                continue
            ma_f = vbt.MA.run(close, window=f).ma
            ma_s = vbt.MA.run(close, window=s).ma
            entries = ma_f > ma_s
            exits = ma_f <= ma_s
            pf = vbt.Portfolio.from_signals(close, entries, exits, fees=0.0, slippage=None, init_cash=10000)
            results[f"f{f}-s{s}"] = {
                "total_return": float(pf.total_return()),
                "sharpe": float(pf.sharpe()),
                "max_drawdown": float(pf.max_drawdown())
            }
    return results

def save_sweep_results(results: Dict, out_csv="integrations/artifacts/vectorbt_sweep.csv"):
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results).T
    df.to_csv(out_csv)
    return out_csv

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV of close prices indexed by timestamp, col 'close'")
    ap.add_argument("--fast", default="5,10,20", help="comma list")
    ap.add_argument("--slow", default="50,100,200", help="comma list")
    ap.add_argument("--out", default="integrations/artifacts/vectorbt_sweep.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.csv, parse_dates=True, index_col=0)
    close = df["close"]
    fast = [int(x) for x in args.fast.split(",")]
    slow = [int(x) for x in args.slow.split(",")]
    res = ma_cross_sweep(close, fast, slow)
    save_sweep_results(res, args.out)
    print("Wrote:", args.out)