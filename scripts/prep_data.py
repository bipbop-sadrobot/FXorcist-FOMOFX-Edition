#!/usr/bin/env python3
"""
prep_data.py
- Input: directory of parquet files (cleaned)
- Output:
  - /data/processed/meta/stats.json
  - /data/processed/train/*.parquet, /val/, /test/
- Chronological split to avoid leakage.
"""

import os, json, argparse
import pandas as pd
import glob
import numpy as np

def load_all_parquets(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, '**', '*.parquet'), recursive=True))
    if not files:
        raise SystemExit("No parquet files found in " + input_dir)
    dfs = []
    for f in files:
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def compute_stats(df, cols=['open','high','low','close','volume']):
    stats = {}
    for c in cols:
        if c in df.columns:
            stats[c] = {'mean': float(df[c].mean()), 'std': float(df[c].std())}
    return stats

def normalise(df, stats):
    df_norm = df.copy()
    for c, s in stats.items():
        df_norm[c] = (df_norm[c] - s['mean']) / (s['std'] if s['std'] != 0 else 1.0)
    return df_norm

def split_and_save(df, output_dir, train_pct=0.7, val_pct=0.15):
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

    df.iloc[:train_end].to_parquet(os.path.join(output_dir, 'train', 'train.parquet'), index=False)
    df.iloc[train_end:val_end].to_parquet(os.path.join(output_dir, 'val', 'val.parquet'), index=False)
    df.iloc[val_end:].to_parquet(os.path.join(output_dir, 'test', 'test.parquet'), index=False)

    print(f"[INFO] Saved splits: train {train_end}, val {val_end-train_end}, test {n-val_end}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="directory to cleaned parquets")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--train_pct", type=float, default=0.7)
    p.add_argument("--val_pct", type=float, default=0.15)
    args = p.parse_args()

    print("[INFO] Loading parquet files...")
    df = load_all_parquets(args.input_dir)

    print("[INFO] Computing normalization stats...")
    stats = compute_stats(df)

    os.makedirs(os.path.join(args.output_dir, 'meta'), exist_ok=True)
    with open(os.path.join(args.output_dir, 'meta', 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    print("[INFO] Normalizing data (in-place save)...")
    df_norm = normalise(df, stats)

    print("[INFO] Splitting and saving...")
    split_and_save(df_norm, args.output_dir, args.train_pct, args.val_pct)

if __name__ == "__main__":
    import json
    main()
