#!/usr/bin/env python3
"""
Test script to load forex data from existing files.
"""

import pandas as pd
import os
from pathlib import Path

def load_forex_data():
    """Load forex data from existing CSV files."""
    data_dir = Path("data/data/data/raw/temp_fx1min/output")
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

    all_data = []

    for symbol in symbols:
        symbol_path = data_dir / symbol.lower()
        if symbol_path.exists():
            print(f"Processing {symbol}")

            # Find all ZIP files for this symbol
            for zip_file in symbol_path.glob("*.zip"):
                try:
                    # Extract year from filename
                    filename = zip_file.name
                    if "_M1_" in filename:
                        year_part = filename.split("_M1_")[1].split(".")[0]
                        if len(year_part) == 4 and year_part.isdigit():
                            year = year_part
                            print(f"  Found {symbol} {year}")

                            # For now, just count the files
                            all_data.append(f"{symbol}_{year}")

                except Exception as e:
                    print(f"  Error processing {zip_file}: {e}")

    print(f"Found {len(all_data)} data files")
    return all_data

if __name__ == "__main__":
    load_forex_data()