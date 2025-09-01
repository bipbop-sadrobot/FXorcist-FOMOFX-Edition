#!/usr/bin/env python3
"""
dataset_analyzer.py  —  Parallel, configurable dataset analyzer

This script provides a command-line interface (CLI) and functions for analyzing datasets in various formats (CSV, Parquet, Excel, JSON).
It supports parallel processing for faster analysis and offers features such as schema validation, basic quality checks,
correlation analysis, and visualization.

Features:
- Loads CSV/Parquet/Excel/JSON (CSV can stream in chunks)
- Optional Dask backend for very large datasets
- Parallel (multiprocessing) per-column stats & anomaly detection
- Schema validation & basic quality checks
- Correlations (numeric)
- Matplotlib visualizations (correlation heatmap + histograms)
- JSON/HTML report export
- Config-driven (YAML) + CLI overrides

Usage:
  python dataset_analyzer.py /path/to/data.csv --workers 8 --plots --report-format json
  python dataset_analyzer.py /path/to/data.parquet --backend dask --sample 2_000_000

Notes:
- If Dask isn’t installed, the script auto-falls back to pandas.
- For huge CSVs, consider `--sample` to limit rows, or `--backend dask`.
"""

import os
import io
import json
import math
import yaml
import time
import argparse
import logging
import warnings
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from datetime import datetime
from multiprocessing import Pool, cpu_count, get_context

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s"
)
logger = logging.getLogger("DatasetAnalyzer")

# ---------------- Config ----------------
DEFAULT_CONFIG = {
    "features": ["mean", "std", "min", "max", "missing_pct", "correlation"],
    "visualizations": True,
    "report_format": "json",
    "zscore_threshold": 3.0,
    "plots": {
        "hist_max_columns": 30  # cap hist count for very wide datasets
    }
}

def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    if path and os.path.exists(path):
        with open(path, "r") as f:
            loaded = yaml.safe_load(f) or {}
        # Deep-ish merge
        for k, v in loaded.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
        logger.info(f"Loaded config: {path}")
    else:
        logger.info("No config provided; using defaults")
    return cfg

# ---------------- Backend helpers ----------------
def try_import_dask():
    try:
        import dask.dataframe as dd
        return dd
    except Exception:
        return None

# ---------------- Data Loading ----------------
def _read_csv_sampled(file_obj: io.BytesIO, sample: Optional[int]) -> pd.DataFrame:
    if sample is None:
        return pd.read_csv(file_obj)
    # Stream rows until reaching sample size
    chunksize = min(200_000, sample)
    chunks = []
    count = 0
    for chunk in pd.read_csv(file_obj, chunksize=chunksize):
        chunks.append(chunk)
        count += len(chunk)
        if count >= sample:
            break
    return pd.concat(chunks, axis=0).head(sample)

def load_dataset(
    file_obj: io.BytesIO,
    file_path: str,  # added file_path to retain name for report
    backend: str = "pandas",
    sample: Optional[int] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Loads a dataset from a file object.

    Args:
        file_obj (io.BytesIO): The file object containing the dataset.
        file_path (str): The path to the file (used for determining file type and reporting).
        backend (str): The backend to use for data loading ("pandas" or "dask").
            - pandas: Loads the data into a Pandas DataFrame.
            - dask: Loads the data into a Dask DataFrame (for large datasets).
        sample (Optional[int]): An optional sample size to limit the number of rows loaded.

    Returns:
        Tuple[pd.DataFrame, str]: A tuple containing the loaded DataFrame and the backend used.
    
    Raises:
        ValueError: If the file format is unsupported or the backend is invalid.
    """
    file_name = file_path
    ext = os.path.splitext(file_name)[-1].lower()
    if backend not in {"pandas", "dask"}:
        raise ValueError("--backend must be 'pandas' or 'dask'")

    if backend == "dask":
        dd = try_import_dask()
        if dd is None:
            logger.warning("Dask not installed; falling back to pandas")
            backend = "pandas"

    logger.info(f"Loading dataset: {file_name} (requested backend: {backend})")
    if backend == "dask":
        dd = try_import_dask()
        if ext == ".csv":
            ddf = dd.read_csv(file_obj, assume_missing=True, blocksize="64MB")
        elif ext == ".parquet":
            ddf = dd.read_parquet(file_obj)
        else:
            logger.warning("Dask backend only implemented for CSV/Parquet; using pandas")
            return load_dataset(file_obj, file_path, backend="pandas", sample=sample)
        if sample:
            # Random sample with Dask (approx)
            frac = sample / max(1, ddf.shape[0].compute())
            frac = min(max(frac, 0.0001), 1.0)
            df = ddf.sample(frac=frac, random_state=42).head(sample)
        else:
            # Be careful: this triggers compute
            df = ddf.compute()
        return df, "dask"

    # pandas backend
    if ext == ".csv":
        df = _read_csv_sampled(file_obj, sample)
    elif ext in {".xls", ".xlsx"}:
        df = pd.read_excel(file_obj)
        if sample:
            df = df.head(sample)
    elif ext == ".json":
        df = pd.read_json(file_obj, lines=False)
        if sample:
            df = df.head(sample)
    elif ext == ".parquet":
        df = pd.read_parquet(file_obj)
        if sample:
            df = df.head(sample)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return df, "pandas"

# ---------------- Validation ----------------
def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "missing_values": df.isna().sum().astype(int).to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()}
    }

# ---------------- Parallel column stats ----------------
def _col_stats_task(args) -> Tuple[str, Dict[str, Any]]:
    """
    Worker task: compute per-column stats + anomaly count.
    """
    col_name, series, z_thr = args
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    res = {
        "count": int(valid.size),
        "missing": int(s.isna().sum()),
        "missing_pct": float(s.isna().mean() * 100.0),
        "mean": float(valid.mean()) if valid.size else None,
        "std": float(valid.std()) if valid.size else None,
        "min": float(valid.min()) if valid.size else None,
        "max": float(valid.max()) if valid.size else None,
        "anomalies": None
    }
    if valid.size and (res["std"] not in (None, 0.0, float("nan"))):
        z = (valid - res["mean"]) / (res["std"] if res["std"] else np.nan)
        res["anomalies"] = int((np.abs(z) > z_thr).sum())
    return col_name, res

def extract_features_parallel(
    df: pd.DataFrame,
    workers: int,
    zscore_threshold: float,
    include: List[str]
) -> Dict[str, Any]:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    feats: Dict[str, Any] = {}

    if not num_cols:
        logger.warning("No numeric columns detected; skipping numeric stats")
        # Still provide missing_pct for all columns
        feats["missing_pct"] = df.isna().mean().mul(100).to_dict()
        return feats

    tasks = [(c, df[c], zscore_threshold) for c in num_cols]
    # Use spawn to be safe on macOS / notebooks
    with get_context("spawn").Pool(processes=max(1, workers)) as pool:
        results = dict(pool.map(_col_stats_task, tasks))

    # Assemble result blocks based on 'include'
    if "mean" in include:
        feats["mean"] = {c: results[c]["mean"] for c in num_cols}
    if "std" in include:
        feats["std"] = {c: results[c]["std"] for c in num_cols}
    if "min" in include:
        feats["min"] = {c: results[c]["min"] for c in num_cols}
    if "max" in include:
        feats["max"] = {c: results[c]["max"] for c in num_cols}
    if "missing_pct" in include:
        # include all columns (numeric + non-numeric)
        feats["missing_pct"] = df.isna().mean().mul(100).to_dict()
    # Always include anomalies as part of numeric inspection
    feats["anomalies"] = {c: results[c]["anomalies"] for c in num_cols}

    return feats

# ---------------- Correlations ----------------
def compute_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    num = df.select_dtypes(include=np.number)
    if num.shape[1] < 2:
        return {"correlation": {}}
    corr = num.corr(numeric_only=True)
    return {"correlation": corr.to_dict()}

# ---------------- Visualizations ----------------
def save_correlation_heatmap(
    df: pd.DataFrame,
    outdir: str
) -> Optional[str]:
    os.makedirs(outdir, exist_ok=True)
    num = df.select_dtypes(include=np.number)
    if num.shape[1] < 2:
        return None
    corr = num.corr(numeric_only=True).values
    labels = num.columns.tolist()

    fig = plt.figure(figsize=(max(6, min(18, 0.6 * len(labels))), max(5, min(16, 0.6 * len(labels)))))
    ax = fig.add_subplot(111)
    im = ax.imshow(corr, interpolation="nearest", aspect="auto")
    ax.set_title("Correlation Heatmap")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(outdir, "correlation_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def save_histograms(
    df: pd.DataFrame,
    outdir: str,
    max_columns: int = 30
) -> List[str]:
    os.makedirs(outdir, exist_ok=True)
    cols = df.select_dtypes(include=np.number).columns.tolist()[:max_columns]
    saved = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if not len(s):
            continue
        fig = plt.figure(figsize=(6, 4))
        plt.hist(s, bins="auto")
        plt.title(f"Distribution of {c}")
        plt.xlabel(c)
        plt.ylabel("Frequency")
        plt.tight_layout()
        path = os.path.join(outdir, f"{c}_hist.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)
    return saved

# ---------------- Report ----------------
def write_report(payload: Dict[str, Any], outdir: str, fmt: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if fmt == "json":
        path = os.path.join(outdir, f"report_{ts}.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path
    if fmt == "html":
        path = os.path.join(outdir, f"report_{ts}.html")
        # very simple HTML wrapper
        html = io.StringIO()
        html.write("<html><head><meta charset='utf-8'><title>Dataset Report</title></head><body>")
        html.write("<h1>Dataset Report</h1>")
        html.write("<h2>Summary (JSON)</h2><pre>")
        html.write(json.dumps(payload, indent=2))
        html.write("</pre></body></html>")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html.getvalue())
        return path
    raise ValueError("Unsupported report format (use 'json' or 'html')")

# ---------------- Main pipeline ----------------
def analyze_dataset(
    file_path: str,
    file_obj: io.BytesIO,
    config_path: Optional[str],
    backend: str,
    workers: int,
    sample: Optional[int],
    plots: bool,
    report_format: str,
    outdir: str
) -> Dict[str, Any]:
    """
    Analyzes a dataset and returns a report.

    Args:
        file_path (str): The path to the file (used for reporting).
        file_obj (io.BytesIO): The file object containing the dataset.
        config_path (Optional[str]): The path to a YAML config file (optional).
        backend (str): The backend to use for data loading ("pandas" or "dask").
        workers (int): The number of worker processes to use for parallel processing.
        sample (Optional[int]): An optional sample size to limit the number of rows loaded.
        plots (bool): Whether to generate plots.
        report_format (str): The format of the report ("json" or "html").
        outdir (str): The output directory for the report and plots.

    Returns:
        Dict[str, Any]: A dictionary containing the analysis report.
    """
    cfg = load_config(config_path)
    if report_format:
        cfg["report_format"] = report_format
    if plots is not None:
        cfg["visualizations"] = bool(plots)

    t0 = time.time()
    df, backend_used = load_dataset(file_obj, file_path, backend=backend, sample=sample)

    # Validation
    validation = validate_dataset(df)

    # Features + anomalies (parallel)
    include = cfg.get("features", [])
    zthr = float(cfg.get("zscore_threshold", 3.0))
    feats = extract_features_parallel(
        df=df,
        workers=max(1, workers),
        zscore_threshold=zthr,
        include=include
    )

    # Correlations
    cors = {}
    if "correlation" in include:
        cors = compute_correlations(df)

    # Visualizations
    plots_dir = os.path.join(outdir, "plots")
    plots_payload = {}
    if cfg.get("visualizations", True):
        heat_path = save_correlation_heatmap(df, plots_dir)
        if heat_path:
            plots_payload["correlation_heatmap"] = heat_path
        hist_paths = save_histograms(
            df, plots_dir, max_columns=int(cfg["plots"]["hist_max_columns"])
        )
        plots_payload["histograms"] = hist_paths

    # Bundle report
    payload = {
        "meta": {
            "source": os.path.abspath(file_path),
            "backend_used": backend_used,
            "rows": int(len(df)),
            "cols": int(df.shape[1]),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "duration_sec": round(time.time() - t0, 3),
            "workers": workers,
            "sample": sample
        },
        "validation": validation,
        "features": {k: v for k, v in feats.items() if k != "correlation"},
        "anomalies": feats.get("anomalies", {}),
        "correlation": cors.get("correlation", {}),
        "plots": plots_payload
    }

    report_path = write_report(payload, outdir, cfg["report_format"])
    logger.info(f"Report saved: {report_path}")
    return payload

# ---------------- CLI ----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel Dataset Analyzer")
    p.add_argument("file", help="Path to dataset (CSV/Parquet/Excel/JSON)")
    p.add_argument("--config", help="Path to YAML config", default=None)
    p.add_argument("--backend", choices=["pandas", "dask"], default="pandas",
                   help="Dataframe backend (dask requires installation)")
    p.add_argument("--workers", type=int, default=max(1, cpu_count() // 2),
                   help="Parallel worker processes for column stats")
    p.add_argument("--sample", type=int, default=None,
                   help="Limit to first N rows (fast prototyping)")
    p.add_argument("--plots", action="store_true", help="Generate plots")
    p.add_argument("--no-plots", dest="plots", action="store_false", help="Disable plots")
    p.set_defaults(plots=None)
    p.add_argument("--report-format", choices=["json", "html"], default=None,
                   help="Override report output format")
    p.add_argument("--outdir", default="reports", help="Output directory")
    return p.parse_args()

def main():
    args = parse_args()
    try:
        result = analyze_dataset(
            file_path=args.file,
            config_path=args.config,
            backend=args.backend,
            workers=args.workers,
            sample=args.sample,
            plots=args.plots,
            report_format=args.report_format,
            outdir=args.outdir
        )
        print(json.dumps(result, indent=2))
    except Exception as e:
        logger.exception("Analysis failed")
        raise SystemExit(1) from e

if __name__ == "__main__":
    main()
