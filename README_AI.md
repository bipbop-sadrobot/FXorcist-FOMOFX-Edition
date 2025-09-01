# README_AI.md — AI Client Playbook

You are an AI assistant (Cline) who must interact with a data pipeline for Forex 1-minute data.
Follow this strict, safe workflow and never deviate.

## Directory conventions
- Raw downloads: `data/raw/<source>/<symbol>/<year>/<month>.csv`
- Cleaned: `data/cleaned/<symbol>/<year>/<month>.parquet`
- Processed (train/val/test): `data/processed/`
- Meta stats: `data/processed/meta/stats.json`

## Ingestion rules (do not overload)
1. Always inspect file size and available RAM before loading.
   - Use `os.path.getsize()` and estimate memory as `size_mb * 10`.
   - If estimated memory > 2GB, use chunked reading or Dask.
2. Only load one month/file at a time for initial experiments.
3. For training, prefer streaming or batch processing:
   - `pd.read_parquet(..., columns=[...])` or `pd.read_csv(..., chunksize=10000)`
4. Normalisation:
   - Use `data/processed/meta/stats.json` for mean/std.
   - Apply `(x - mean) / std` to open/high/low/close/volume.
5. Chronological splits:
   - Train: first 70%, Val: next 15%, Test: last 15%.
   - Never shuffle across splits.

## System prompt (for the AI agent controlling train runs)


You are a conservative, resource-aware AI training agent.

Query data availability via the file-system functions provided.

Request only one dataset at a time for any heavy operation.

If RAM or disk falls below configured thresholds, pause and notify operator.

Log each step (file read, transform, model fit) with timestamp and row counts.

Use chunked training with batch sizes that keep memory < 2GB per process.


## Quick usage examples
- List available 2024 files:
  ```python
  from data_loader import list_available_data
  list_available_data('2024')


Load one file, chunked:

for chunk in pd.read_parquet('data/cleaned/EURUSD/2024/08.parquet', chunksize=10000):
    process_chunk(chunk)


---

# 9 — Best Practices, Cutting-edge & Advanced Options

I implemented conservative defaults. here's what else you can adopt when you want to scale or increase fidelity:

1. **Parquet + partitioning** (done): fast columnar IO, smaller disk, partition by year/month/symbol.
2. **Use Dask / VAEX / Polars** for lazy/distributed processing:
   - Dask: parallelize across cores for merging months & computing stats.
   - Polars: extremely fast, lower memory than pandas for many ops.
3. **Feature store**: Persist computed features (rolling means, RSI) as separate parquet partitioned tables so training reads precomputed features.
4. **Delta Lake / Iceberg** on object storage for ACID + time travel if you scale to cloud.
5. **Memory-aware training**:
   - Use PyTorch DataLoader with IterableDataset reading parquet in chunks.
   - Use mixed precision and gradient accumulation to fit larger batch-equivalent sizes.
6. **Monitoring & Logging**:
   - Integrate a small Prometheus/Statsd client or logs for disk/RAM during pipeline operations.
7. **Tests & Validation**:
   - Add unit tests for `clean_data.py` and `prep_data.py` to validate timestamp parsing, outlier removal, no duplicates.
8. **Data lineage**:
   - Save `meta/source_manifest.json` mapping source files -> cleaned parquet -> processed splits for reproducibility.
9. **Model validation**:
   - Use walk-forward validation rather than a fixed split for time-series.
10. **Backtesting safety**:
   - Ensure models only see information available at prediction time; avoid lookahead features.

---

# 10 — Prompts for your AI client (Cline) — two variants

## System prompt (to load as “system” message)


You are Cline, an expert quant data engineer and model trainer. You must be resource-aware and conservative.
Follow the repository's README_AI.md rules exactly. When asked to fetch data, use the fetch pipeline scripts; when asked to train, always check data size first and use chunked loading or Dask if necessary. Always log operations and maintain data lineage metadata. If a step would exceed resource thresholds, stop and return the reason.


## Task prompt (example to ask Cline to run a pipeline)


Task: Prepare EURUSD 2024 Jan-Aug from histdata + dukascopy and create processed train/val/test.

Steps:

Run ./scripts/run_pipeline.sh --symbol EURUSD --start-year 2024 --end-year 2024 --months 01,02,03,04,05,06,07,08 --sources histdata,dukascopy

Report any missing months, download errors, or files skipped by organization.

After processed data is created, print stats: rows in train/val/test and mean/std for open/high/low/close/volume.

If disk or memory constraints were reached, explain which month/source and suggest solution (downsample or use Dask/polars).


---

# 11 — Final notes & next steps (what I recommend you run now)

1. Add the files into your repo.
2. Make scripts executable:
   ```bash
   chmod +x scripts/*.sh
   chmod +x scripts/*.py


Run a small test fetch for one month from HistData:

./scripts/fetch_data.sh --source histdata --symbol EURUSD --year 2024 --month 08
./scripts/forganise.sh
python3 scripts/clean_data.py --input data/raw/histdata/EURUSD/2024/08.csv --output data/cleaned/EURUSD/2024/08.parquet --forward_fill --outlier_method iqr --outlier_thresh 3.0
python3 scripts/prep_data.py --input_dir data/cleaned --output_dir data/processed


Confirm data/processed/meta/stats.json exists and look at train/val/test parquet shapes.
