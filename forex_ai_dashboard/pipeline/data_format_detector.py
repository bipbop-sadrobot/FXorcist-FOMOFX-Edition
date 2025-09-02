#!/usr/bin/env python3
"""
Advanced Data Format Detection and Cleaning for FXorcist-FOMOFX-Edition
Handles various forex data formats and ensures clean, validated data for training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import re
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForexDataFormatDetector:
    """Advanced detector for various forex data formats."""

    def __init__(self):
        # Known forex data formats
        self.known_formats = {
            'metaquotes': {
                'columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                'delimiters': ['\t', ';', ','],
                'header_rows': [0, None],  # With/without header
                'timestamp_formats': ['%Y.%m.%d %H:%M', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M']
            },
            'generic_ohlc': {
                'columns': ['date', 'time', 'open', 'high', 'low', 'close', 'volume'],
                'delimiters': [',', ';', '\t', '|'],
                'header_rows': [0, None],
                'timestamp_formats': ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%d/%m/%Y %H:%M']
            },
            'ascii_format': {
                'columns': ['timestamp', 'open', 'high', 'low', 'close', 'tick_volume'],
                'delimiters': [';', ',', '\t'],
                'header_rows': [None],  # Usually no header
                'timestamp_formats': ['%Y%m%d %H%M%S', '%Y.%m.%d %H:%M:%S']
            }
        }

        # Column name mappings
        self.column_mappings = {
            'timestamp': ['timestamp', 'time', 'date', 'datetime', 'dt'],
            'open': ['open', 'o', 'open_price', 'bid_open'],
            'high': ['high', 'h', 'high_price', 'bid_high'],
            'low': ['low', 'l', 'low_price', 'bid_low'],
            'close': ['close', 'c', 'close_price', 'bid_close', 'price'],
            'volume': ['volume', 'vol', 'v', 'tick_volume', 'real_volume']
        }

    def detect_and_parse(self, file_path: Path, sample_size: int = 1000) -> Optional[pd.DataFrame]:
        """Detect data format and parse accordingly."""
        try:
            # Quick format detection
            format_type, confidence = self._detect_format(file_path)

            if confidence < 0.3:
                logger.warning(f"Low confidence format detection for {file_path}")
                return None

            # Parse with detected format
            df = self._parse_with_format(file_path, format_type, sample_size)

            if df is not None and len(df) > 0:
                # Clean and validate
                df = self._clean_and_validate(df)
                if len(df) > 0:
                    logger.info(f"Successfully parsed {file_path}: {len(df)} rows, format: {format_type}")
                    return df

            return None

        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return None

    def _detect_format(self, file_path: Path) -> Tuple[str, float]:
        """Detect the data format with confidence score."""
        try:
            # Read sample lines
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [f.readline().strip() for _ in range(5)]

            if not lines:
                return 'unknown', 0.0

            # Analyze first line
            first_line = lines[0]
            confidence_scores = {}

            for format_name, format_spec in self.known_formats.items():
                score = self._calculate_format_confidence(first_line, format_spec)
                confidence_scores[format_name] = score

            # Return best match
            if confidence_scores:
                best_format = max(confidence_scores, key=confidence_scores.get)
                best_score = confidence_scores[best_format]
                return best_format, best_score

            return 'unknown', 0.0

        except Exception as e:
            logger.warning(f"Format detection failed for {file_path}: {e}")
            return 'unknown', 0.0

    def _calculate_format_confidence(self, line: str, format_spec: Dict) -> float:
        """Calculate confidence score for a format."""
        score = 0.0
        parts = None

        # Try different delimiters
        for delimiter in format_spec['delimiters']:
            if delimiter in line:
                parts = line.split(delimiter)
                if len(parts) >= 4:  # At least OHLC
                    score += 0.3
                    break

        if not parts:
            return 0.0

        # Check for numeric values (OHLC)
        numeric_count = 0
        for part in parts[-4:]:  # Last 4 columns likely OHLC
            try:
                float(part.strip())
                numeric_count += 1
            except:
                continue

        if numeric_count >= 2:  # At least 2 numeric columns
            score += 0.4

        # Check for timestamp-like data
        for part in parts[:2]:  # First 2 columns likely timestamp
            if self._looks_like_timestamp(part.strip()):
                score += 0.3
                break

        return min(score, 1.0)

    def _looks_like_timestamp(self, text: str) -> bool:
        """Check if text looks like a timestamp."""
        # Common timestamp patterns
        patterns = [
            r'\d{4}[-./]\d{2}[-./]\d{2}',  # YYYY-MM-DD
            r'\d{2}[-./]\d{2}[-./]\d{4}',  # DD/MM/YYYY
            r'\d{8}',  # YYYYMMDD
            r'\d{2}:\d{2}(:\d{2})?',  # HH:MM or HH:MM:SS
        ]

        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False

    def _parse_with_format(self, file_path: Path, format_type: str, sample_size: int) -> Optional[pd.DataFrame]:
        """Parse file with specific format."""
        format_spec = self.known_formats.get(format_type)
        if not format_spec:
            return None

        df = None
        best_score = 0

        # Try different combinations
        for delimiter in format_spec['delimiters']:
            for header in format_spec['header_rows']:
                try:
                    if header is None:
                        # For headerless data, explicitly specify column names
                        temp_df = pd.read_csv(
                            file_path,
                            delimiter=delimiter,
                            header=None,
                            names=range(10),  # Pre-allocate column names
                            nrows=sample_size,
                            engine='python'
                        )
                        # Trim to actual number of columns
                        actual_cols = len(temp_df.columns)
                        temp_df.columns = range(actual_cols)
                    else:
                        temp_df = pd.read_csv(
                            file_path,
                            delimiter=delimiter,
                            header=header,
                            nrows=sample_size,
                            engine='python'
                        )

                    if len(temp_df.columns) >= 4:
                        # Score this attempt
                        score = self._score_dataframe(temp_df)
                        if score > best_score:
                            df = temp_df
                            best_score = score

                except Exception:
                    continue

        return df

    def _score_dataframe(self, df: pd.DataFrame) -> float:
        """Score dataframe quality."""
        score = 0.0

        # Check column count
        if len(df.columns) >= 4:
            score += 0.2

        # Check for numeric columns
        numeric_cols = 0
        for col in df.columns[-4:]:  # Last 4 columns
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols += 1

        score += (numeric_cols / 4) * 0.4

        # Check data completeness
        completeness = 1 - df.isnull().sum().sum() / (len(df) * len(df.columns))
        score += completeness * 0.4

        return score

    def _clean_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the dataframe."""
        try:
            # Standardize column names
            df = self._standardize_columns(df)

            # Required columns
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                logger.warning(f"Missing required columns: {missing_cols}")
                return pd.DataFrame()

            # Convert to numeric
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove rows with NaN in required columns
            df = df.dropna(subset=required_cols)

            # Validate OHLC relationships
            df = df[
                (df['high'] >= df['open']) &
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) &
                (df['low'] <= df['close'])
            ]

            # Remove unrealistic price movements
            if len(df) > 1:
                returns = df['close'].pct_change().abs()
                df = df[returns <= 0.1]  # Max 10% change per minute

            # Clean volume if present
            if 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                df['volume'] = df['volume'].fillna(0)

            # Handle timestamp
            df = self._clean_timestamp(df)

            return df

        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            return pd.DataFrame()

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names."""
        df = df.copy()

        # Force correct column assignment for known forex format
        if len(df.columns) >= 5:
            # This is the standard forex format: timestamp;open;high;low;close;volume
            expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df.columns = expected_columns[:len(df.columns)]
            return df

        # Handle both string and numeric column names
        if all(isinstance(col, int) for col in df.columns):
            # Numeric columns - assign based on position for known format
            if len(df.columns) >= 5:  # timestamp + OHLC + volume
                column_names = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                df.columns = column_names[:len(df.columns)]
            else:
                # Fallback: convert to strings and try mapping
                df.columns = df.columns.astype(str).str.lower().str.strip()
        else:
            # String columns - apply standard mapping
            df.columns = df.columns.astype(str).str.lower().str.strip()

        # Apply mappings for any remaining columns
        for standard_name, possible_names in self.column_mappings.items():
            for possible_name in possible_names:
                if possible_name in df.columns:
                    df = df.rename(columns={possible_name: standard_name})
                    break

        return df

    def _clean_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize timestamp column."""
        if 'timestamp' not in df.columns:
            # Try to create from date/time columns
            if 'date' in df.columns and 'time' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(
                        df['date'] + ' ' + df['time'],
                        errors='coerce'
                    )
                except:
                    df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
            else:
                # Create sequential timestamp
                df['timestamp'] = pd.date_range(
                    start=datetime.now(),
                    periods=len(df),
                    freq='1min'
                )
        else:
            # Handle different timestamp formats
            timestamp_formats = [
                '%Y%m%d %H%M%S',      # 20200101 170000
                '%Y-%m-%d %H:%M:%S',  # 2020-01-01 17:00:00
                '%Y/%m/%d %H:%M:%S',  # 2020/01/01 17:00:00
                '%Y.%m.%d %H:%M',     # 2020.01.01 17:00
                '%Y-%m-%d %H:%M:%S.%f', # With microseconds
            ]

            # Try different formats
            for fmt in timestamp_formats:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format=fmt, errors='coerce')
                    # Check if parsing was successful
                    if not df['timestamp'].isna().all():
                        break
                except:
                    continue

            # If all formats failed, try automatic parsing
            if df['timestamp'].isna().all():
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Fill missing timestamps
        df['timestamp'] = df['timestamp'].fillna(method='ffill')

        return df

class DataQualityValidator:
    """Validate data quality and provide cleaning recommendations."""

    def __init__(self):
        self.quality_metrics = {}

    def validate_dataset(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Comprehensive data quality validation."""
        metrics = {
            'symbol': symbol,
            'total_rows': len(df),
            'valid_rows': 0,
            'completeness_score': 0.0,
            'consistency_score': 0.0,
            'realism_score': 0.0,
            'overall_quality': 0.0,
            'issues': []
        }

        if len(df) == 0:
            metrics['issues'].append('Empty dataset')
            return metrics

        # Check data completeness
        required_cols = ['open', 'high', 'low', 'close']
        completeness = 1 - df[required_cols].isnull().sum().sum() / (len(df) * len(required_cols))
        metrics['completeness_score'] = completeness

        if completeness < 0.8:
            metrics['issues'].append(f'Low completeness: {completeness:.2%}')

        # Validate OHLC consistency
        valid_ohlc = (
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
        )
        consistency = valid_ohlc.sum() / len(df)
        metrics['consistency_score'] = consistency

        if consistency < 0.9:
            metrics['issues'].append(f'OHLC inconsistency: {(1-consistency):.2%} invalid')

        # Check price realism
        if len(df) > 1:
            returns = df['close'].pct_change().abs()
            realistic = (returns <= 0.1).sum() / len(df)  # Max 10% per minute
            metrics['realism_score'] = realistic

            if realistic < 0.95:
                metrics['issues'].append(f'Unrealistic price movements: {(1-realistic):.2%}')

        # Calculate valid rows
        valid_mask = (
            df[required_cols].notna().all(axis=1) &
            valid_ohlc &
            (returns <= 0.1 if len(df) > 1 else True)
        )
        metrics['valid_rows'] = valid_mask.sum()

        # Overall quality score
        metrics['overall_quality'] = (completeness + consistency + metrics['realism_score']) / 3

        # Store metrics
        self.quality_metrics[symbol] = metrics

        return metrics

    def get_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        if not self.quality_metrics:
            return {'status': 'No data validated'}

        report = {
            'total_symbols': len(self.quality_metrics),
            'average_quality': 0.0,
            'quality_distribution': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0},
            'common_issues': {},
            'recommendations': []
        }

        qualities = []
        issues_count = {}

        for symbol, metrics in self.quality_metrics.items():
            quality = metrics['overall_quality']
            qualities.append(quality)

            # Categorize quality
            if quality >= 0.9:
                report['quality_distribution']['excellent'] += 1
            elif quality >= 0.7:
                report['quality_distribution']['good'] += 1
            elif quality >= 0.5:
                report['quality_distribution']['fair'] += 1
            else:
                report['quality_distribution']['poor'] += 1

            # Count issues
            for issue in metrics['issues']:
                issues_count[issue] = issues_count.get(issue, 0) + 1

        if qualities:
            report['average_quality'] = sum(qualities) / len(qualities)

        report['common_issues'] = dict(sorted(issues_count.items(), key=lambda x: x[1], reverse=True))

        # Generate recommendations
        if report['average_quality'] < 0.7:
            report['recommendations'].append('Overall data quality needs improvement')
        if issues_count.get('Low completeness', 0) > 0:
            report['recommendations'].append('Address missing data issues')
        if issues_count.get('OHLC inconsistency', 0) > 0:
            report['recommendations'].append('Fix OHLC relationship violations')
        if issues_count.get('Unrealistic price movements', 0) > 0:
            report['recommendations'].append('Filter out unrealistic price movements')

        return report

def main():
    """Test the data format detector."""
    detector = ForexDataFormatDetector()
    validator = DataQualityValidator()

    # Test with sample data
    test_files = [
        "data/sample_forex_data.csv",  # Add your test files here
    ]

    for file_path in test_files:
        if Path(file_path).exists():
            df = detector.detect_and_parse(Path(file_path))
            if df is not None:
                symbol = Path(file_path).stem.split('_')[0].upper()
                quality = validator.validate_dataset(df, symbol)
                print(f"Quality for {symbol}: {quality['overall_quality']:.2%}")

    # Print overall report
    report = validator.get_quality_report()
    print("\nQuality Report:")
    print(f"Average Quality: {report.get('average_quality', 0):.2%}")
    print(f"Quality Distribution: {report.get('quality_distribution', {})}")

if __name__ == "__main__":
    main()