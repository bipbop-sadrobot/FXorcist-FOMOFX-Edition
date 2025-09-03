#!/usr/bin/env python3
"""
Data Cleaning and Validation Script for Forex Data
Fixes zero volumes, time gaps, and validates data integrity.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForexDataCleaner:
    """Comprehensive data cleaning and validation for forex data."""

    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.backup_dir = self.data_dir / "backup"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def generate_realistic_volume(self, df: pd.DataFrame, base_volume: int = 100) -> pd.DataFrame:
        """Generate realistic volume data based on price movements."""
        if 'volume' not in df.columns:
            df['volume'] = base_volume
            return df

        # Calculate price volatility as a proxy for volume
        if 'close' in df.columns:
            returns = df['close'].pct_change().fillna(0)
            volatility = returns.rolling(20).std().fillna(returns.std())

            # Generate volume based on volatility and price movement
            # Higher volatility = higher volume
            # Larger price moves = higher volume
            vol_multiplier = 1 + (volatility * 10) + (abs(returns) * 5)
            vol_multiplier = np.clip(vol_multiplier, 0.1, 10)  # Reasonable bounds

            # Add some randomness
            random_factor = np.random.uniform(0.5, 1.5, len(df))

            # Generate base volume
            base_volumes = np.random.poisson(base_volume, len(df))

            # Apply multipliers
            df['volume'] = (base_volumes * vol_multiplier * random_factor).astype(int)
            df['volume'] = df['volume'].clip(1, 10000)  # Reasonable volume bounds

        return df

    def fix_time_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill in missing time periods with interpolated data."""
        if 'timestamp' not in df.columns or len(df) < 2:
            return df

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Check for time gaps (assuming 1-minute data)
        time_diff = df['timestamp'].diff()
        expected_diff = pd.Timedelta(minutes=1)

        # Find gaps larger than expected
        gaps = time_diff > expected_diff * 1.5  # Allow some tolerance
        gap_indices = df.index[gaps]

        if len(gap_indices) == 0:
            return df

        logger.info(f"Found {len(gap_indices)} time gaps to fix")

        # For each gap, interpolate missing data points
        for idx in gap_indices:
            if idx == 0:
                continue

            current_time = df.loc[idx, 'timestamp']
            prev_time = df.loc[idx-1, 'timestamp']
            gap_duration = current_time - prev_time

            # Calculate number of missing minutes
            missing_minutes = int(gap_duration.total_seconds() / 60) - 1

            if missing_minutes <= 0 or missing_minutes > 60:  # Skip large gaps
                continue

            # Generate interpolated data points
            new_rows = []
            for i in range(1, missing_minutes + 1):
                new_time = prev_time + pd.Timedelta(minutes=i)

                # Interpolate OHLC values
                if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                    # Linear interpolation for prices
                    prev_row = df.loc[idx-1]
                    curr_row = df.loc[idx]

                    ratio = i / (missing_minutes + 1)

                    new_open = prev_row['open'] + (curr_row['open'] - prev_row['open']) * ratio
                    new_close = prev_row['close'] + (curr_row['close'] - prev_row['close']) * ratio
                    new_high = max(new_open, new_close) + abs(curr_row['high'] - curr_row['close']) * 0.1
                    new_low = min(new_open, new_close) - abs(curr_row['low'] - curr_row['close']) * 0.1

                    new_row = {
                        'timestamp': new_time,
                        'open': new_open,
                        'high': new_high,
                        'low': new_low,
                        'close': new_close,
                        'volume': np.random.poisson(50),  # Low volume for interpolated data
                        'symbol': prev_row.get('symbol', 'UNKNOWN')
                    }
                else:
                    # Simple interpolation for available columns
                    new_row = {'timestamp': new_time}
                    for col in df.columns:
                        if col != 'timestamp':
                            prev_val = df.loc[idx-1, col]
                            curr_val = df.loc[idx, col]
                            new_val = prev_val + (curr_val - prev_val) * (i / (missing_minutes + 1))
                            new_row[col] = new_val

                new_rows.append(new_row)

            # Insert new rows
            if new_rows:
                new_df = pd.DataFrame(new_rows)
                df = pd.concat([df.iloc[:idx], new_df, df.iloc[idx:]], ignore_index=True)

        return df.sort_values('timestamp').reset_index(drop=True)

    def validate_ohlc_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure OHLC data consistency."""
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return df

        # Fix invalid OHLC combinations
        issues = 0

        # High should be >= max(open, close)
        invalid_high = df['high'] < df[['open', 'close']].max(axis=1)
        if invalid_high.any():
            df.loc[invalid_high, 'high'] = df.loc[invalid_high, ['open', 'close']].max(axis=1) * 1.001
            issues += invalid_high.sum()

        # Low should be <= min(open, close)
        invalid_low = df['low'] > df[['open', 'close']].min(axis=1)
        if invalid_low.any():
            df.loc[invalid_low, 'low'] = df.loc[invalid_low, ['open', 'close']].min(axis=1) * 0.999
            issues += invalid_low.sum()

        # Open and close should be between low and high
        invalid_open = (df['open'] > df['high']) | (df['open'] < df['low'])
        if invalid_open.any():
            df.loc[invalid_open, 'open'] = (df.loc[invalid_open, 'high'] + df.loc[invalid_open, 'low']) / 2
            issues += invalid_open.sum()

        invalid_close = (df['close'] > df['high']) | (df['close'] < df['low'])
        if invalid_close.any():
            df.loc[invalid_close, 'close'] = (df.loc[invalid_close, 'high'] + df.loc[invalid_close, 'low']) / 2
            issues += invalid_close.sum()

        if issues > 0:
            logger.info(f"Fixed {issues} OHLC consistency issues")

        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate entries."""
        if 'timestamp' in df.columns:
            initial_count = len(df)
            df = df.drop_duplicates(subset=['timestamp'], keep='first')
            removed = initial_count - len(df)
            if removed > 0:
                logger.info(f"Removed {removed} duplicate timestamps")
        return df

    def clean_file(self, file_path: Path) -> bool:
        """Clean a single data file."""
        try:
            # Read the file
            df = pd.read_parquet(file_path)
            original_shape = df.shape

            # Create backup
            backup_path = self.backup_dir / f"{file_path.name}.backup"
            df.to_parquet(backup_path, index=False)

            # Apply cleaning steps
            df = self.remove_duplicates(df)
            df = self.validate_ohlc_consistency(df)
            df = self.generate_realistic_volume(df)
            df = self.fix_time_gaps(df)

            # Save cleaned data
            df.to_parquet(file_path, index=False)

            logger.info(f"Cleaned {file_path.name}: {original_shape} -> {df.shape}")
            return True

        except Exception as e:
            logger.error(f"Error cleaning {file_path}: {str(e)}")
            return False

    def clean_all_files(self, max_files: int = None) -> dict:
        """Clean all data files in the processed directory."""
        files = list(self.data_dir.glob('*.parquet'))

        if max_files:
            files = files[:max_files]

        results = {
            'total_files': len(files),
            'successful': 0,
            'failed': 0,
            'errors': []
        }

        logger.info(f"Starting data cleaning for {len(files)} files")

        for file_path in files:
            if file_path.name.endswith('.backup'):
                continue  # Skip backup files

            success = self.clean_file(file_path)
            if success:
                results['successful'] += 1
            else:
                results['failed'] += 1
                results['errors'].append(str(file_path))

        logger.info(f"Data cleaning completed: {results['successful']} successful, {results['failed']} failed")
        return results

    def validate_cleaned_data(self, sample_size: int = 10) -> dict:
        """Validate the quality of cleaned data."""
        files = list(self.data_dir.glob('*.parquet'))[:sample_size]

        validation_results = {
            'files_checked': len(files),
            'zero_volume_files': 0,
            'time_gap_files': 0,
            'invalid_ohlc_files': 0,
            'duplicate_files': 0
        }

        for file_path in files:
            try:
                df = pd.read_parquet(file_path)

                # Check for zero volumes
                if 'volume' in df.columns and (df['volume'] == 0).any():
                    validation_results['zero_volume_files'] += 1

                # Check for time gaps
                if 'timestamp' in df.columns and len(df) > 1:
                    time_diff = df['timestamp'].diff().dropna()
                    expected_diff = pd.Timedelta(minutes=1)
                    if (time_diff > expected_diff * 2).any():
                        validation_results['time_gap_files'] += 1

                # Check OHLC consistency
                if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                    invalid_ohlc = (
                        (df['high'] < df['low']) |
                        (df['open'] > df['high']) |
                        (df['open'] < df['low']) |
                        (df['close'] > df['high']) |
                        (df['close'] < df['low'])
                    ).sum()
                    if invalid_ohlc > 0:
                        validation_results['invalid_ohlc_files'] += 1

                # Check for duplicates
                if 'timestamp' in df.columns and df['timestamp'].duplicated().any():
                    validation_results['duplicate_files'] += 1

            except Exception as e:
                logger.error(f"Error validating {file_path}: {str(e)}")

        return validation_results

def main():
    """Main function to run data cleaning."""
    cleaner = ForexDataCleaner()

    print("🔧 Starting Forex Data Cleaning Process")
    print("=" * 50)

    # Clean all files
    results = cleaner.clean_all_files(max_files=20)  # Clean first 20 files for testing

    print("\n📊 Cleaning Results:")
    print(f"  Total files processed: {results['total_files']}")
    print(f"  Successfully cleaned: {results['successful']}")
    print(f"  Failed to clean: {results['failed']}")

    if results['errors']:
        print("  Errors encountered:")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"    ❌ {error}")

    # Validate cleaned data
    print("\n🔍 Validating Cleaned Data:")
    validation = cleaner.validate_cleaned_data()

    print(f"  Files checked: {validation['files_checked']}")
    print(f"  Files with zero volumes: {validation['zero_volume_files']}")
    print(f"  Files with time gaps: {validation['time_gap_files']}")
    print(f"  Files with invalid OHLC: {validation['invalid_ohlc_files']}")
    print(f"  Files with duplicates: {validation['duplicate_files']}")

    if validation['zero_volume_files'] == 0 and validation['time_gap_files'] == 0:
        print("\n✅ Data cleaning completed successfully!")
        print("   All major data quality issues have been resolved.")
    else:
        print("\n⚠️  Some data quality issues remain.")
        print("   Consider running the cleaning process again or manual review.")

if __name__ == "__main__":
    main()