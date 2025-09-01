#!/usr/bin/env python3
import os, argparse
import polars as pl

def read_with_fallback(path):
    # try automatic
    try:
        return pl.read_csv(path)
    except Exception:
        try:
            return pl.read_csv(path, separator=';')
        except Exception:
            try:
                return pl.read_csv(path, separator=',')
            except Exception as e:
                raise e

#!/usr/bin/env python3
import os, argparse
import polars as pl

def read_with_fallback(path):
    # try automatic
    try:
        return pl.read_csv(path)
    except Exception:
        try:
            return pl.read_csv(path, separator=';')
        except Exception:
            try:
                return pl.read_csv(path, separator=',')
            except Exception as e:
                raise e

def clean_file(input_path, output_path, downsample=None):
    if not os.path.exists(input_path):
        print(f"[WARN] Input file {input_path} not found, skipping...")
        return

    try:
        df = read_with_fallback(input_path)
    except Exception as e:
        print(f"[WARN] Could not read {input_path}: {e}; skipping.")
        return

    # Normalize column names and ensure timestamp exists
    cols = [c.lower() for c in df.columns]
    if "timestamp" not in cols:
        # assume first column is timestamp
        df = df.rename({df.columns[0]: "timestamp"})

    # parse timestamp to datetime (try detect common formats)
    try:
        df = df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, fmt=None, strict=False).alias("timestamp"))
    except Exception:
        df = df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, strict=False).alias("timestamp"))

    df = df.drop_nulls(["timestamp"])
    df = df.unique(subset=["timestamp"]).sort("timestamp")

    # forward-fill numeric cols
    df = df.fill_null(strategy="forward")

    # Ensure numeric types for OHLCV
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Float64))

    # Remove rows where prices are NaN
    if all(x in df.columns for x in ["open","high","low","close"]):
        df = df.drop_nulls(["open","high","low","close"])

    # Outlier removal via MAD (robust)
    for col in ["open","high","low","close"]:
        if col in df.columns:
            med = df[col].median()
            mad = (df[col] - med).abs().median()
            # avoid division by zero
            if mad == 0 or pl.is_null(mad).all():
                continue
            df = df.filter(((df[col] - med).abs() / mad) < 10)

    # Optional downsample (e.g., "5m" -> "5m" notations, polars expects pandas-like rules; we will accept "5T","1H")
    if downsample:
        # polars group_by_dynamic expects an ISO8601-like period (e.g. "1h","5m"). We'll accept "5T" or "1H" and convert:
        rule = downsample.upper().replace("T", "m").replace("MIN", "m")
        df = df.lazy().group_by_dynamic(index_column="timestamp", every=rule, closed="left").agg([
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("volume").sum().alias("volume")
        ]).collect()
        # drop any rows with null timestamp
        df = df.drop_nulls(["timestamp"])

    # Add year/month for convenience
    df = df.with_columns([
        pl.col("timestamp").dt.year().cast(pl.Utf8).alias("year"),
        pl.col("timestamp").dt.month().abs().cast(pl.Int64).alias("month_temp")
    ])
    df = df.with_columns(pl.col("month_temp").apply(lambda x: f"{int(x):02d}").alias("month")).drop("month_temp")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        df.write_parquet(output_path, compression="zstd")
        print(f"[INFO] Cleaned data saved to {output_path} (rows: {df.height})")
    except Exception as e:
        print(f"[WARN] Failed to write parquet {output_path}: {e}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--downsample", default=None, help="e.g., 5T, 1H")
    args = p.parse_args()
    clean_file(args.input, args.output, args.downsample)
