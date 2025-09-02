#!/usr/bin/env python3
"""
Enhanced Data Integration Script for FXorcist-FOMOFX-Edition
Handles zip file extraction, 1-minute timeframe data processing,
and memory system integration for improved training pipeline.
"""

import pandas as pd
import numpy as np
import zipfile
import os
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from memory_system.core import MemoryManager
from memory_system.anomaly import AnomalyDetector
from memory_system.federated import FederatedMemory
from memory_system.event_bus import EventBus
from memory_system.metadata import SharedMetadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataIntegrator:
    """Enhanced data integration with memory system and federated learning."""

    def __init__(self):
        # Initialize memory system components
        self.event_bus = EventBus()
        self.metadata = SharedMetadata()
        self.memory = MemoryManager()
        self.anomaly_detector = AnomalyDetector(self.memory)
        self.federated_memory = FederatedMemory(self.event_bus, self.metadata)

        # Data directories
        self.raw_data_dir = Path("data/raw")
        self.processed_data_dir = Path("data/processed")
        self.temp_extract_dir = Path("data/temp_extracted")

        # Ensure directories exist
        self.temp_extract_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        # Track processed files
        self.processed_files = set()

    def find_m1_zip_files(self) -> List[Path]:
        """Find all M1 (1-minute) zip files in the project."""
        m1_files = []

        # Search patterns for M1 data
        patterns = [
            "**/*M1*.zip",
            "**/*_M1_*.zip",
            "DAT_ASCII_*_M1_*.zip"
        ]

        for pattern in patterns:
            # Search in current directory and subdirectories
            for zip_file in Path(".").glob(pattern):
                if zip_file.is_file() and "M1" in zip_file.name:
                    m1_files.append(zip_file)

            # Also search in data directories
            for zip_file in Path("data").glob(f"**/{pattern}"):
                if zip_file.is_file() and "M1" in zip_file.name:
                    m1_files.append(zip_file)

        # Remove duplicates
        unique_files = list(set(m1_files))
        logger.info(f"Found {len(unique_files)} M1 zip files")

        return unique_files

    def extract_zip_file(self, zip_path: Path) -> Optional[Path]:
        """Extract a zip file to temporary directory."""
        try:
            extract_dir = self.temp_extract_dir / zip_path.stem
            extract_dir.mkdir(exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            logger.info(f"Extracted {zip_path} to {extract_dir}")
            return extract_dir

        except Exception as e:
            logger.error(f"Failed to extract {zip_path}: {e}")
            return None

    def parse_m1_csv_data(self, csv_path: Path) -> Optional[pd.DataFrame]:
        """Parse M1 CSV data with proper format detection."""
        try:
            # Try different CSV formats
            formats_to_try = [
                {'delimiter': ',', 'header': None},
                {'delimiter': ';', 'header': None},
                {'delimiter': '\t', 'header': None},
                {'delimiter': ',', 'header': 0},
            ]

            df = None
            for fmt in formats_to_try:
                try:
                    df = pd.read_csv(csv_path, **fmt)
                    if len(df.columns) >= 5:  # OHLC + timestamp
                        break
                except:
                    continue

            if df is None or len(df.columns) < 5:
                logger.warning(f"Could not parse {csv_path} - insufficient columns")
                return None

            # Standardize column names
            df = self._standardize_columns(df)

            # Convert timestamp if needed
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

            # Validate data
            df = self._validate_m1_data(df)

            if len(df) > 0:
                logger.info(f"Successfully parsed {csv_path}: {len(df)} rows")
                return df
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to parse {csv_path}: {e}")
            return None

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for OHLC data."""
        # Common column name patterns
        column_mappings = {
            # Timestamp columns
            'timestamp': ['timestamp', 'time', 'date', 'datetime'],
            'open': ['open', 'o', 'open_price'],
            'high': ['high', 'h', 'high_price'],
            'low': ['low', 'l', 'low_price'],
            'close': ['close', 'c', 'close_price'],
            'volume': ['volume', 'vol', 'v', 'tick_volume']
        }

        df = df.copy()
        df.columns = df.columns.astype(str).str.lower()

        # Map columns
        for standard_name, possible_names in column_mappings.items():
            for possible_name in possible_names:
                if possible_name in df.columns:
                    df = df.rename(columns={possible_name: standard_name})
                    break

        return df

    def _validate_m1_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate M1 data quality."""
        # Remove rows with missing OHLC
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()

        # Remove rows with NaN in required columns
        df = df.dropna(subset=required_cols)

        # Validate OHLC relationships
        df = df[
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
        ]

        # Remove unrealistic price movements (more than 10% in 1 minute)
        if len(df) > 1:
            returns = df['close'].pct_change().abs()
            df = df[returns <= 0.1]  # 10% max change per minute

        return df

    def integrate_memory_system(self, df: pd.DataFrame, symbol: str):
        """Integrate processed data with memory system."""
        try:
            # Convert to memory entries
            memory_entries = []
            for _, row in df.iterrows():
                entry = {
                    "model": "data_integration",
                    "prediction": row.get('close', 0),
                    "target": row.get('close', 0),
                    "error": 0.0,  # No prediction error for raw data
                    "features": {
                        "open": row.get('open', 0),
                        "high": row.get('high', 0),
                        "low": row.get('low', 0),
                        "close": row.get('close', 0),
                        "volume": row.get('volume', 0),
                        "symbol": symbol
                    },
                    "ts": row.get('timestamp').timestamp() if pd.notna(row.get('timestamp')) else datetime.now().timestamp()
                }
                memory_entries.append(entry)

            # Add to memory system
            for entry in memory_entries:
                self.memory.add_record(entry)

            logger.info(f"Integrated {len(memory_entries)} records for {symbol} into memory system")

        except Exception as e:
            logger.error(f"Failed to integrate memory system for {symbol}: {e}")

    def process_single_zip(self, zip_path: Path) -> bool:
        """Process a single zip file."""
        if zip_path in self.processed_files:
            logger.info(f"Already processed {zip_path}")
            return True

        try:
            # Extract zip
            extract_dir = self.extract_zip_file(zip_path)
            if not extract_dir:
                return False

            # Find CSV files in extracted directory
            csv_files = list(extract_dir.glob("*.csv"))
            if not csv_files:
                logger.warning(f"No CSV files found in {extract_dir}")
                return False

            # Process each CSV file
            processed_data = []
            for csv_file in csv_files:
                # Extract symbol from filename
                symbol = self._extract_symbol_from_filename(csv_file.name)

                # Parse data
                df = self.parse_m1_csv_data(csv_file)
                if df is not None and len(df) > 0:
                    # Add symbol column
                    df['symbol'] = symbol

                    # Integrate with memory system
                    self.integrate_memory_system(df, symbol)

                    processed_data.append(df)

            # Combine all data from this zip
            if processed_data:
                combined_df = pd.concat(processed_data, ignore_index=True)

                # Save processed data
                output_file = self.processed_data_dir / f"{zip_path.stem}_processed.parquet"
                combined_df.to_parquet(output_file, index=False)

                logger.info(f"Processed {zip_path} -> {output_file} ({len(combined_df)} rows)")
                self.processed_files.add(zip_path)
                return True
            else:
                logger.warning(f"No valid data extracted from {zip_path}")
                return False

        except Exception as e:
            logger.error(f"Failed to process {zip_path}: {e}")
            return False

    def _extract_symbol_from_filename(self, filename: str) -> str:
        """Extract currency symbol from filename."""
        # Common patterns: EURUSD_M1, DAT_ASCII_EURUSD_M1, etc.
        filename = filename.upper()

        # Look for currency pairs (6-7 characters)
        import re
        pairs = re.findall(r'([A-Z]{6,7})', filename)

        for pair in pairs:
            # Validate it's a currency pair
            if len(pair) == 6 or len(pair) == 7:
                # Check if it looks like a forex pair
                if pair[:3] != pair[3:6]:  # Not same currency
                    return pair

        # Fallback to filename stem
        return Path(filename).stem.split('_')[0]

    def process_all_m1_data(self) -> Dict[str, int]:
        """Process all available M1 zip files."""
        logger.info("Starting comprehensive M1 data processing")

        # Find all M1 zip files
        m1_files = self.find_m1_zip_files()

        if not m1_files:
            logger.warning("No M1 zip files found")
            return {"processed": 0, "failed": 0, "total": 0}

        # Process each file
        processed = 0
        failed = 0

        for zip_file in m1_files:
            logger.info(f"Processing {zip_file}")
            if self.process_single_zip(zip_file):
                processed += 1
            else:
                failed += 1

        # Generate memory insights
        self._generate_memory_insights()

        result = {
            "processed": processed,
            "failed": failed,
            "total": len(m1_files)
        }

        logger.info(f"Data integration complete: {result}")
        return result

    def _generate_memory_insights(self):
        """Generate insights from integrated memory data."""
        try:
            insights = self.memory.generate_insights_report()
            logger.info(f"Memory insights: {insights}")

            # Save insights
            insights_file = self.processed_data_dir / "memory_insights.json"
            import json
            with open(insights_file, 'w') as f:
                json.dump(insights, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to generate memory insights: {e}")

    def get_memory_anomalies(self) -> Dict:
        """Get anomaly detection results from memory system."""
        try:
            return self.anomaly_detector.detect_anomalies()
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
            return {"anomalies": [], "flash_crashes": [], "regime_changes": []}

def main():
    """Main data integration function."""
    try:
        logger.info("Starting FXorcist Data Integration")

        # Initialize integrator
        integrator = DataIntegrator()

        # Process all M1 data
        results = integrator.process_all_m1_data()

        # Get anomaly detection results
        anomalies = integrator.get_memory_anomalies()

        # Print summary
        print("\n" + "="*60)
        print("FXorcist Data Integration Summary")
        print("="*60)
        print(f"Total M1 files found: {results['total']}")
        print(f"Successfully processed: {results['processed']}")
        print(f"Failed to process: {results['failed']}")
        print(f"Memory records: {len(integrator.memory.records)}")
        print(f"Anomalies detected: {len(anomalies.get('anomalies', []))}")
        print("="*60)

        logger.info("Data integration completed successfully")

    except Exception as e:
        logger.error(f"Data integration failed: {e}")
        print(f"❌ Data integration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()