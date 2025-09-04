# quantstats_report.py
import quantstats as qs
import pandas as pd
from pathlib import Path
from typing import Optional

def write_quantstats_report(returns: pd.Series, out_html: str, title: str = "FXorcist Report", benchmark: Optional[pd.Series] = None):
    """
    Create a QuantStats HTML tear-sheet for a strategy returns series.
    - returns: pd.Series indexed by timestamps (periodic returns, e.g., hourly/daily)
    - benchmark: optional pd.Series same index for a benchmark
    """
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    # quantstats expects returns as decimal (e.g., 0.001)
    qs.reports.html(returns, benchmark=benchmark, title=title, output=out_html)
    return out_html

if __name__ == "__main__":
    # Example CLI usage
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--returns_csv", required=True, help="CSV with timestamp and returns column (decimal)")
    ap.add_argument("--returns_col", default="returns")
    ap.add_argument("--out", default="integrations/artifacts/qstats_report.html")
    ap.add_argument("--title", default="FXorcist Report")
    args = ap.parse_args()
    df = pd.read_csv(args.returns_csv, parse_dates=True, index_col=0)
    r = df[args.returns_col].astype(float)
    write_quantstats_report(r, args.out, args.title)
    print("Wrote:", args.out)