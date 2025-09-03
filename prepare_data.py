#!/usr/bin/env python3
"""
Data preparation script to extract and organize forex data.
"""

import zipfile
import shutil
import tempfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_forex_data():
    """Extract and organize all forex data files."""
    source_dir = Path("data/data/data/raw/temp_fx1min/output")
    target_dir = Path("data/raw/histdata")

    if not source_dir.exists():
        logger.error("Source data directory not found")
        return False

    logger.info("Starting data preparation...")

    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
    extracted_count = 0
    error_count = 0

    for symbol in symbols:
        symbol_source = source_dir / symbol.lower()
        symbol_target = target_dir / symbol

        if not symbol_source.exists():
            logger.warning(f"Source directory for {symbol} not found")
            continue

        logger.info(f"Processing {symbol}...")

        # Create target directory
        symbol_target.mkdir(parents=True, exist_ok=True)

        # Process each ZIP file
        for zip_file in symbol_source.glob("*.zip"):
            try:
                filename = zip_file.name
                if "_M1_" in filename:
                    year_part = filename.split("_M1_")[1].split(".")[0]

                    if len(year_part) == 4 and year_part.isdigit():
                        year = year_part
                        month = None
                    elif len(year_part) == 6 and year_part.isdigit():
                        year = year_part[:4]
                        month = year_part[4:]
                    else:
                        continue

                    # Create year directory
                    year_dir = symbol_target / year
                    year_dir.mkdir(exist_ok=True)

                    # Extract to temp directory
                    with tempfile.TemporaryDirectory() as temp_dir:
                        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)

                        # Find CSV file
                        csv_files = list(Path(temp_dir).glob("*.csv"))
                        if csv_files:
                            csv_file = csv_files[0]

                            # Determine target filename
                            if month:
                                target_name = f"{month}.csv"
                            else:
                                target_name = f"{year}.csv"

                            target_path = year_dir / target_name

                            # Copy file
                            shutil.copy2(csv_file, target_path)
                            extracted_count += 1

                            if extracted_count % 10 == 0:
                                logger.info(f"  Processed {extracted_count} files...")

            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    logger.warning(f"Error processing {zip_file.name}: {e}")

    logger.info(f"Data preparation complete!")
    logger.info(f"  - Files extracted: {extracted_count}")
    logger.info(f"  - Errors: {error_count}")
    return extracted_count > 0

if __name__ == "__main__":
    success = prepare_forex_data()
    if success:
        print("✅ Data preparation completed successfully!")
    else:
        print("❌ Data preparation failed!")