#!/usr/bin/env bash
set -euo pipefail

# forganise.sh
# Must be run from repo root
# Usage: ./scripts/forganise.sh

info(){ echo "[INFO]" "$@"; }
warn(){ echo "[WARN]" "$@"; }
fail(){ echo "[ERROR]" "$@"; exit 1; }

BASE="data/raw"
mkdir -p "$BASE"

# find csv files under data/raw and ensure they are at data/raw/<source>/<symbol>/<year>/<month>.csv
# We assume any top-level CSVs are legacy and move them to data/raw/legacy/
shopt -s globstar nullglob

info "Standardizing names and moving uncategorised CSVs to data/raw/legacy"
mkdir -p "$BASE/legacy"
for f in data/raw/*.csv; do
  [ -f "$f" ] || continue
  mv "$f" "$BASE/legacy/$(basename "$f")"
done

# Normalize any histdata / dukascopy files: try to parse timestamps to find symbol/year/month
info "Scanning for loose CSVs under data/raw/**"
for f in data/raw/**/**/*.csv; do
  [ -f "$f" ] || continue
  # Try to infer source/symbol/year/month from path or filename
  bn=$(basename "$f")
  # example patterns: HISTDATA_COM_ASCII_EURUSD_M1_202408.csv or EURUSD_m1_2024-08-01_...
  if [[ "$bn" =~ HISTDATA_COM_ASCII_([A-Z]+)_M1_([0-9]{6}) ]]; then
    sym="${BASH_REMATCH[1]}"
    ym="${BASH_REMATCH[2]}"
    year="${ym:0:4}"
    month="${ym:4:2}"
    mkdir -p "$BASE/histdata/$sym/$year"
    mv "$f" "$BASE/histdata/$sym/$year/$month.csv"
  elif [[ "$bn" =~ ^([A-Za-z]+)_m1_([0-9]{4})-([0-9]{2}) ]]; then
    sym="${BASH_REMATCH[1]}"
    year="${BASH_REMATCH[2]}"
    month="${BASH_REMATCH[3]}"
    mkdir -p "$BASE/dukascopy/$sym/$year"
    mv "$f" "$BASE/dukascopy/$sym/$year/$month.csv"
  else
    # copy into legacy folder with subdir
    mkdir -p "$BASE/legacy/loose"
    mv "$f" "$BASE/legacy/loose/$bn"
  fi
done

info "Deduplicating: if two files would create same destination, keep the largest file (assumed more complete)"
# Simple dedupe: within each leaf dir if multiple CSVs, keep the largest.
for dir in $(find $BASE -type d -mindepth 3 -maxdepth 3); do
  files=("$dir"/*.csv)
  if (( ${#files[@]} > 1 )); then
    largest=$(ls -S "$dir"/*.csv | head -n1)
    for ff in "$dir"/*.csv; do
      if [[ "$ff" != "$largest" ]]; then rm -f "$ff"; fi
    done
  fi
done

info "Organisation complete."
