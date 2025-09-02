#!/usr/bin/env python3
"""
Optimized Data Integration Script for FXorcist-FOMOFX-Edition
Addresses resource consumption issues by focusing on high-quality data,
selective processing, and efficient resource management.
"""

import pandas as pd
import numpy as np
import zipfile
import os
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import logging
from datetime import datetime, timedelta
import sys
import psutil
import time
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from memory_system.core import MemoryManager
from memory_system.anomaly import AnomalyDetector
from memory_system.federated import FederatedMemory
from memory_system.event_bus import EventBus
from memory_system.metadata import SharedMetadata
try:
    from .data_format_detector import ForexDataFormatDetector, DataQualityValidator
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from data_format_detector import ForexDataFormatDetector, DataQualityValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/optimized_data_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ResourceMonitor:
    """Monitor system resources during data processing."""

    def __init__(self):
        self.start_time = time.time()
        self.peak_memory = 0
        self.files_processed = 0
        self.data_points_processed = 0

    def update(self, memory_usage: float = None, files: int = 0, data_points: int = 0):
        """Update resource metrics."""
        if memory_usage:
            self.peak_memory = max(self.peak_memory, memory_usage)
        self.files_processed += files
        self.data_points_processed += data_points

    def get_report(self) -> Dict:
        """Get resource usage report."""
        elapsed = time.time() - self.start_time
        return {
            "elapsed_time": elapsed,
            "peak_memory_mb": self.peak_memory,
            "files_processed": self.files_processed,
            "data_points_processed": self.data_points_processed,
            "processing_rate": self.data_points_processed / elapsed if elapsed > 0 else 0
        }

class DataQualityAssessor:
    """Assess data quality before processing."""

    def __init__(self):
        # High-quality pairs and recent years
        self.preferred_pairs = {
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
            'USDCHF', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY'
        }
        self.recent_years = {2020, 2021, 2022, 2023, 2024, 2025}

    def assess_file_quality(self, zip_path: Path) -> Tuple[bool, str, Dict]:
        """Assess if a file is worth processing."""
        filename = zip_path.name.upper()

        # Extract pair and year from filename
        pair = self._extract_pair(filename)
        year = self._extract_year(filename)

        # Quality scoring
        quality_score = 0
        reasons = []

        # Prefer major pairs
        if pair in self.preferred_pairs:
            quality_score += 3
            reasons.append(f"Major pair: {pair}")
        elif pair:
            quality_score += 1
            reasons.append(f"Minor pair: {pair}")
        else:
            reasons.append("Unknown pair")

        # Prefer recent years
        if year in self.recent_years:
            quality_score += 3
            reasons.append(f"Recent year: {year}")
        elif year and year >= 2015:
            quality_score += 1
            reasons.append(f"Somewhat recent: {year}")
        elif year:
            reasons.append(f"Old data: {year}")

        # Check file size (avoid empty/corrupted files)
        file_size = zip_path.stat().st_size
        if file_size < 1000:  # Less than 1KB
            return False, "File too small", {"score": 0, "size": file_size}
        elif file_size > 100 * 1024 * 1024:  # Over 100MB
            quality_score += 1
            reasons.append("Large file (good data volume)")

        # Only process high-quality files
        should_process = quality_score >= 4

        metadata = {
            "pair": pair,
            "year": year,
            "score": quality_score,
            "size": file_size,
            "reasons": reasons
        }

        return should_process, "High quality" if should_process else f"Low quality (score: {quality_score})", metadata

    def _extract_pair(self, filename: str) -> Optional[str]:
        """Extract currency pair from filename."""
        import re
        # Look for 6-7 character currency pairs
        pairs = re.findall(r'([A-Z]{6,7})', filename)
        for pair in pairs:
            if len(pair) in [6, 7] and pair[:3] != pair[3:6]:
                return pair
        return None

    def _extract_year(self, filename: str) -> Optional[int]:
        """Extract year from filename."""
        import re
        years = re.findall(r'(\d{4})', filename)
        for year_str in years:
            year = int(year_str)
            if 2000 <= year <= 2030:  # Reasonable year range
                return year
        return None

class OptimizedDataIntegrator:
    """Optimized data integration with resource management and quality assessment."""

    def __init__(self):
        # Initialize memory system components
        self.event_bus = EventBus()
        self.metadata = SharedMetadata()
        self.memory = MemoryManager()
        self.anomaly_detector = AnomalyDetector(self.memory)
        self.federated_memory = FederatedMemory(self.event_bus, self.metadata)

        # Quality assessment and resource monitoring
        self.quality_assessor = DataQualityAssessor()
        self.resource_monitor = ResourceMonitor()
        # Advanced data format detection and validation
        self.format_detector = ForexDataFormatDetector()
        self.quality_validator = DataQualityValidator()

        # Data directories
        self.raw_data_dir = Path("data/raw")
        self.processed_data_dir = Path("data/processed")
        self.temp_extract_dir = Path("data/temp_extracted")

        # Ensure directories exist
        self.temp_extract_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        # Track processed files and quality metrics
        self.processed_files = set()
        self.quality_stats = defaultdict(int)

    def find_optimized_m1_files(self) -> List[Tuple[Path, Dict]]:
        """Find M1 files with quality assessment."""
        logger.info("Finding M1 files with quality assessment...")

        all_files = []
        patterns = [
            "**/*M1*.zip",
            "**/*_M1_*.zip",
            "DAT_ASCII_*_M1_*.zip"
        ]

        for pattern in patterns:
            # Search in current directory and subdirectories
            for zip_file in Path(".").glob(pattern):
                if zip_file.is_file() and "M1" in zip_file.name:
                    all_files.append(zip_file)

            # Also search in data directories
            for zip_file in Path("data").glob(f"**/{pattern}"):
                if zip_file.is_file() and "M1" in zip_file.name:
                    all_files.append(zip_file)

        # Remove duplicates
        unique_files = list(set(all_files))
        logger.info(f"Found {len(unique_files)} total M1 zip files")

        # Assess quality and filter
        quality_files = []
        for zip_file in unique_files:
            should_process, reason, metadata = self.quality_assessor.assess_file_quality(zip_file)
            self.quality_stats[reason] += 1

            if should_process:
                quality_files.append((zip_file, metadata))
                logger.info(f"✅ Selected: {zip_file.name} ({reason})")
            else:
                logger.info(f"❌ Skipped: {zip_file.name} ({reason})")

        logger.info(f"Selected {len(quality_files)} high-quality files for processing")
        return quality_files

    def quick_validate_zip_content(self, zip_path: Path) -> bool:
        """Quick validation of zip content without full extraction."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Check if zip has files
                file_list = zip_ref.namelist()
                if not file_list:
                    return False

                # Look for CSV files
                csv_files = [f for f in file_list if f.lower().endswith('.csv')]
                if not csv_files:
                    return False

                # Quick check of first CSV file
                first_csv = csv_files[0]
                with zip_ref.open(first_csv) as f:
                    # Read first few lines
                    lines = f.readlines()[:5]
                    if len(lines) < 2:  # Need at least header + 1 data row
                        return False

                    # Check if lines have enough columns (likely OHLC data)
                    first_line = lines[0].decode('utf-8', errors='ignore').strip()
                    data_line = lines[1].decode('utf-8', errors='ignore').strip()

                    # Try different delimiters
                    for delimiter in [',', ';', '\t', '|']:
                        if delimiter in first_line and delimiter in data_line:
                            cols = data_line.split(delimiter)
                            if len(cols) >= 5:  # timestamp + OHLC + volume
                                return True

            return False

        except Exception as e:
            logger.warning(f"Quick validation failed for {zip_path}: {e}")
            return False

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

    def parse_m1_csv_data_optimized(self, csv_path: Path) -> Optional[pd.DataFrame]:
        """Advanced M1 CSV parsing using format detector."""
        try:
            # Use advanced format detector
            df = self.format_detector.detect_and_parse(csv_path)

            if df is not None and len(df) > 10:
                # Additional quality validation
                symbol = self._extract_symbol_from_filename(csv_path.name)
                quality_metrics = self.quality_validator.validate_dataset(df, symbol)

                if quality_metrics['overall_quality'] >= 0.7:  # Good quality threshold
                    logger.info(f"✅ High-quality data: {csv_path.name} "
                              f"(Quality: {quality_metrics['overall_quality']:.2%})")
                    return df
                else:
                    logger.warning(f"❌ Low-quality data rejected: {csv_path.name} "
                                 f"(Quality: {quality_metrics['overall_quality']:.2%})")
                    return None
            else:
                logger.warning(f"No valid data from format detector: {csv_path}")
                return None

        except Exception as e:
            logger.error(f"Advanced parsing failed for {csv_path}: {e}")
            return None

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for OHLC data."""
        column_mappings = {
            'timestamp': ['timestamp', 'time', 'date', 'datetime'],
            'open': ['open', 'o', 'open_price'],
            'high': ['high', 'h', 'high_price'],
            'low': ['low', 'l', 'low_price'],
            'close': ['close', 'c', 'close_price'],
            'volume': ['volume', 'vol', 'v', 'tick_volume']
        }

        df = df.copy()
        df.columns = df.columns.astype(str).str.lower()

        for standard_name, possible_names in column_mappings.items():
            for possible_name in possible_names:
                if possible_name in df.columns:
                    df = df.rename(columns={possible_name: standard_name})
                    break

        return df

    def _validate_m1_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate M1 data quality."""
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
            df = df[returns <= 0.1]

        return df

    def integrate_memory_system_optimized(self, df: pd.DataFrame, symbol: str):
        """Optimized memory system integration with batching."""
        try:
            # Process in batches to manage memory
            batch_size = 1000
            memory_entries = []

            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size]

                for _, row in batch_df.iterrows():
                    entry = {
                        "model": "optimized_data_integration",
                        "prediction": row.get('close', 0),
                        "target": row.get('close', 0),
                        "error": 0.0,
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

                # Add batch to memory system
                for entry in memory_entries:
                    self.memory.add_record(entry)

                # Update resource monitor
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                self.resource_monitor.update(memory_usage, data_points=len(batch_df))

                memory_entries = []  # Clear for next batch

            logger.info(f"Integrated {len(df)} records for {symbol} into memory system")

        except Exception as e:
            logger.error(f"Failed to integrate memory system for {symbol}: {e}")

    def process_single_zip_optimized(self, zip_path: Path, metadata: Dict) -> bool:
        """Optimized single zip processing with quality checks."""
        if zip_path in self.processed_files:
            logger.info(f"Already processed {zip_path}")
            return True

        try:
            # Quick validation before extraction
            if not self.quick_validate_zip_content(zip_path):
                logger.warning(f"Quick validation failed for {zip_path}")
                return False

            # Extract zip
            extract_dir = self.extract_zip_file(zip_path)
            if not extract_dir:
                return False

            # Find and process CSV files
            csv_files = list(extract_dir.glob("*.csv"))
            if not csv_files:
                logger.warning(f"No CSV files found in {extract_dir}")
                return False

            processed_data = []
            for csv_file in csv_files:
                symbol = self._extract_symbol_from_filename(csv_file.name)

                # Parse data
                df = self.parse_m1_csv_data_optimized(csv_file)
                if df is not None and len(df) > 10:
                    df['symbol'] = symbol
                    processed_data.append(df)

            # Combine and save
            if processed_data:
                combined_df = pd.concat(processed_data, ignore_index=True)

                # Save processed data
                output_file = self.processed_data_dir / f"{zip_path.stem}_optimized.parquet"
                combined_df.to_parquet(output_file, index=False)

                logger.info(f"✅ Processed {zip_path} -> {output_file} ({len(combined_df)} rows)")
                self.processed_files.add(zip_path)

                # Update resource monitor
                self.resource_monitor.update(files=1, data_points=len(combined_df))

                return True
            else:
                logger.warning(f"No valid data extracted from {zip_path}")
                return False

        except Exception as e:
            logger.error(f"Failed to process {zip_path}: {e}")
            return False

    def _extract_symbol_from_filename(self, filename: str) -> str:
        """Extract currency symbol from filename."""
        filename = filename.upper()
        import re
        pairs = re.findall(r'([A-Z]{6,7})', filename)

        for pair in pairs:
            if len(pair) in [6, 7] and pair[:3] != pair[3:6]:
                return pair

        return Path(filename).stem.split('_')[0]

    def process_optimized_data(self) -> Dict[str, int]:
        """Process data with optimization and resource management."""
        logger.info("Starting optimized M1 data processing")

        # Find high-quality files
        quality_files = self.find_optimized_m1_files()

        if not quality_files:
            logger.warning("No high-quality M1 files found")
            return {"processed": 0, "skipped": 0, "total": 0}

        # Process files
        processed = 0
        skipped = 0

        for zip_file, metadata in quality_files:
            logger.info(f"Processing {zip_file.name} (Quality: {metadata['score']})")
            if self.process_single_zip_optimized(zip_file, metadata):
                processed += 1
            else:
                skipped += 1

        # Generate insights
        self._generate_optimized_insights()

        result = {
            "processed": processed,
            "skipped": skipped,
            "total": len(quality_files)
        }

        logger.info(f"Optimized data integration complete: {result}")
        return result

    def _generate_optimized_insights(self):
        """Generate insights from optimized processing."""
        try:
            insights = self.memory.generate_insights_report()
            resource_report = self.resource_monitor.get_report()

            # Combine insights
            optimized_insights = {
                "processing_stats": resource_report,
                "quality_stats": dict(self.quality_stats),
                "memory_insights": insights
            }

            # Save insights
            insights_file = self.processed_data_dir / "optimized_insights.json"
            import json
            with open(insights_file, 'w') as f:
                json.dump(optimized_insights, f, indent=2, default=str)

            logger.info(f"Optimized insights saved: {insights_file}")

        except Exception as e:
            logger.error(f"Failed to generate optimized insights: {e}")

    def get_processing_report(self) -> Dict:
        """Get comprehensive processing report."""
        return {
            "resource_usage": self.resource_monitor.get_report(),
            "quality_assessment": dict(self.quality_stats),
            "memory_stats": {
                "total_records": len(self.memory.records),
                "anomalies": len(self.anomaly_detector.detect_anomalies().get('anomalies', []))
            }
        }

def main():
    """Main optimized data integration function."""
    try:
        logger.info("Starting Optimized FXorcist Data Integration")

        # Initialize optimized integrator
        integrator = OptimizedDataIntegrator()

        # Process optimized data
        results = integrator.process_optimized_data()

        # Get processing report
        report = integrator.get_processing_report()

        # Print comprehensive summary
        print("\n" + "="*80)
        print("OPTIMIZED FXorcist Data Integration Summary")
        print("="*80)
        print(f"Total high-quality files found: {results['total']}")
        print(f"Successfully processed: {results['processed']}")
        print(f"Skipped/failed: {results['skipped']}")
        print()

        print("Resource Usage:")
        res = report['resource_usage']
        print(".2f")
        print(".1f")
        print(f"Files processed: {res['files_processed']}")
        print(".0f")
        print()

        print("Data Quality Assessment:")
        for reason, count in report['quality_assessment'].items():
            print(f"  {reason}: {count}")
        print()

        print("Memory System:")
        mem = report['memory_stats']
        print(f"  Total records: {mem['total_records']}")
        print(f"  Anomalies detected: {mem['anomalies']}")
        print("="*80)

        logger.info("Optimized data integration completed successfully")

    except Exception as e:
        logger.error(f"Optimized data integration failed: {e}")
        print(f"❌ Optimized data integration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()