#!/bin/bash
set -euo pipefail

YEAR="${1:-2024}"
MONTH="${2:-07}"
SYMBOLS="${3:-EURUSD,GBPUSD}"

echo "[INFO] Pipeline start for $SYMBOLS $YEAR-$MONTH"

IFS=',' read -ra SYMBOL_LIST <<< "$SYMBOLS"

# Step 1: Fetch raw data
for sym in "${SYMBOL_LIST[@]}"; do
    echo "[INFO] [Fetch] $sym $YEAR-$MONTH"
    ./scripts/fetch_data.sh --source histdata --symbols "$sym" --year "$YEAR" --month "$MONTH"
done
echo "[INFO] Fetch complete ✅"

# Step 2: Clean data
for sym in "${SYMBOL_LIST[@]}"; do
    RAW_FILE="data/raw/histdata/$sym/$YEAR/$MONTH.csv"
    CLEAN_FILE="data/cleaned/$sym/${YEAR}_${MONTH}.parquet"

    if [[ -f "$RAW_FILE" ]]; then
        echo "[INFO] Cleaning $RAW_FILE → $CLEAN_FILE"
        python3 scripts/clean_data.py --input "$RAW_FILE" --output "$CLEAN_FILE"
    else
        echo "[WARN] Missing $RAW_FILE, skipping cleaning..."
    fi
done

# Step 3: Preprocess final dataset
echo "[INFO] Running final preprocessing..."
python3 scripts/prep_data.py --input_dir "data/cleaned/$YEAR" --output_dir data/processed \
  || { echo "[ERROR] Preprocessing failed"; exit 1; }

echo "[INFO] Pipeline complete ✅"
