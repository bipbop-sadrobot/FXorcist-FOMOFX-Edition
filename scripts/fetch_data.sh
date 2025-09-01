#!/bin/bash
set -euo pipefail

SOURCE="histdata"
SYMBOLS=""
YEAR=""
MONTH=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --source) SOURCE="$2"; shift 2 ;;
    --symbols) SYMBOLS="$2"; shift 2 ;;
    --year) YEAR="$2"; shift 2 ;;
    --month) MONTH="$2"; shift 2 ;;
    *) echo "[ERROR] Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$SYMBOLS" || -z "$YEAR" || -z "$MONTH" ]]; then
    echo "[ERROR] Missing required args: --symbols, --year, --month"
    exit 1
fi

echo "[INFO] Fetching $SYMBOLS $YEAR-$MONTH from $SOURCE"

# Create directories
mkdir -p "data/raw/$SOURCE/$SYMBOLS/$YEAR"
TMPDIR=$(mktemp -d)
ZIPFILE="$TMPDIR/${SYMBOLS}_${YEAR}_${MONTH}.zip"

# Page URL (note: site accepts /7 or /07 for month, but script uses /$MONTH as-is)
URL="https://www.histdata.com/download-free-forex-data/?/ascii/1-minute-bar-quotes/$SYMBOLS/$YEAR/$MONTH"

# Download the HTML page
echo "[INFO] Fetching download page: $URL"
HTML=$(curl -sSL "$URL" -H "User-Agent: Mozilla/5.0")

# Extract the tk token from the hidden form
INPUT_TK=$(echo "$HTML" | grep -o '<input [^>]*id="tk"[^>]*>')
if [[ -z "$INPUT_TK" ]]; then
    echo "[ERROR] Failed to find input with id=tk in HTML"
    exit 1
fi
TK=$(echo "$INPUT_TK" | sed 's/.*value="\([^"]*\)".*/\1/')
if [[ -z "$TK" ]]; then
    echo "[ERROR] Failed to extract value for tk"
    exit 1
fi

# Set other form parameters (these are consistent based on the URL structure)
DATE="$YEAR"
DATEMONTH="${YEAR}${MONTH}"
PLATFORM="ASCII"
TIMEFRAME="M1"
FXPAIR=$(echo "$SYMBOLS" | tr '[:lower:]' '[:upper:]')  # Ensure uppercase

# Download the ZIP via POST to get.php
echo "[INFO] Downloading ZIP using token: $TK"
curl -sSL -X POST "https://www.histdata.com/get.php" \
  -H "User-Agent: Mozilla/5.0" \
  -H "Origin: https://www.histdata.com" \
  -H "Referer: $URL" \
  --data "tk=$TK" \
  --data "date=$DATE" \
  --data "datemonth=$DATEMONTH" \
  --data "platform=$PLATFORM" \
  --data "timeframe=$TIMEFRAME" \
  --data "fxpair=$FXPAIR" \
  -o "$ZIPFILE"

# Extract
unzip -q "$ZIPFILE" -d "$TMPDIR"

# Find the CSV inside
CSVFILE=$(find "$TMPDIR" -type f -name "*.csv" | head -n 1)
if [[ ! -f "$CSVFILE" ]]; then
    echo "[ERROR] No CSV found in archive"
    exit 1
fi

# Normalize name → $YEAR/$MONTH.csv
DEST="data/raw/$SOURCE/$SYMBOLS/$YEAR/$MONTH.csv"
mkdir -p "$(dirname "$DEST")"
mv "$CSVFILE" "$DEST"
echo "[INFO] Moved $CSVFILE → $DEST"

rm -rf "$TMPDIR"
echo "[INFO] Fetch for $SYMBOLS $YEAR-$MONTH complete ✅"