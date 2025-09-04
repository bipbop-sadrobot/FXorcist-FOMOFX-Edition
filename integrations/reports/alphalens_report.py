# alphalens_report.py
import alphalens as al
import pandas as pd
from pathlib import Path

def alphalens_analysis(factor: pd.Series, prices: pd.Series, periods=[1,5,10], out_dir="integrations/artifacts/alphalens"):
    """
    Factor is a pd.Series indexed by timestamp (and possibly asset). For single pair,
    we convert to the format alphalens expects: MultiIndex (date, asset). For FX we can treat pair as single asset.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Build factor and pricing data for Alphalens
    # For a single asset: make a MultiIndex with asset 'FX'
    factor_df = pd.DataFrame({'factor': factor})
    factor_df.index = pd.MultiIndex.from_product([factor_df.index, ['FX']], names=['date','asset'])
    pricedf = pd.DataFrame({'price': prices})
    pricedf.index = pd.DatetimeIndex(pricedf.index)
    # Alphalens expects a price panel: columns = assets
    price_panel = pd.DataFrame(index=pricedf.index)
    price_panel['FX'] = pricedf['price']
    factor_data = al.utils.get_clean_factor_and_forward_returns(factor_df['factor'], price_panel, periods=periods)
    # Save basic metrics
    mean_ic = al.performance.mean_information_coefficient(factor_data)
    alpha_summary = {
        "mean_ic": mean_ic.to_dict() if hasattr(mean_ic, "to_dict") else float(mean_ic)
    }
    # You can also create tear sheets; here we save a simple CSV summary
    summary_csv = Path(out_dir) / "alphalens_summary.csv"
    pd.DataFrame([alpha_summary]).to_csv(summary_csv, index=False)
    return str(summary_csv)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor_csv", required=True)
    ap.add_argument("--price_csv", required=True)
    ap.add_argument("--out", default="integrations/artifacts/alphalens_summary.csv")
    args = ap.parse_args()
    fdf = pd.read_csv(args.factor_csv, parse_dates=True, index_col=0)
    pdf = pd.read_csv(args.price_csv, parse_dates=True, index_col=0)
    print(alphalens_analysis(fdf.iloc[:,0], pdf.iloc[:,0], out_dir=Path(args.out).parent))