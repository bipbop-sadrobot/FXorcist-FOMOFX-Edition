import sys
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ingestion import ingest_all_parallel
from data_validation.validate_data import validate_forex_data
from data_validation.clean_data import adaptive_clean_forex_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("logs/integration_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_test_data():
    """Generate realistic forex test data with various anomalies."""
    # Base parameters
    start_date = '2025-08-01'
    end_date = '2025-08-17'
    dates = pd.date_range(start=start_date, end=end_date, freq='1min')
    base_price = 1.1000
    
    # Generate realistic price movements
    np.random.seed(42)
    returns = np.random.normal(0, 0.0002, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create test files
    data_dir = Path('data/test')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create multiple data sources with different formats and anomalies
    sources = []
    
    # 1. HistData format
    hist_data = pd.DataFrame({
        'datetime': dates,
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.002, len(dates))),
        'low': prices * (1 - np.random.uniform(0, 0.002, len(dates))),
        'close': prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'volume': np.random.randint(1000, 5000, len(dates))
    })
    hist_path = data_dir / 'histdata_test.csv'
    hist_data.to_csv(hist_path, index=False)
    sources.append({'type': 'histdata', 'path': str(hist_path)})
    
    # 2. Dukascopy format (with some gaps)
    duka_data = hist_data.copy()
    duka_data.loc[100:150, ['open', 'high', 'low', 'close']] = np.nan  # Add gaps
    duka_path = data_dir / 'dukascopy_test.csv'
    duka_data.to_csv(duka_path, index=False)
    sources.append({'type': 'dukascopy', 'path': str(duka_path)})
    
    # 3. EJTrader format (with some outliers)
    ej_data = hist_data.copy()
    ej_data.loc[300:302, 'high'] *= 1.1  # Add spikes
    ej_data.loc[500:505, 'low'] *= 0.9   # Add drops
    ej_path = data_dir / 'ejtrader_test.csv'
    ej_data.to_csv(ej_path, index=False)
    sources.append({'type': 'ejtrader', 'path': str(ej_path)})
    
    return sources

def test_pipeline_integration():
    """Test complete pipeline integration with various scenarios."""
    logger.info("Starting pipeline integration test")
    
    try:
        # 1. Generate and ingest test data
        sources = generate_test_data()
        logger.info(f"Generated {len(sources)} test data sources")
        
        # 2. Run ingestion
        df_ingested = ingest_all_parallel(sources)
        logger.info(
            f"Ingestion complete. Shape: {df_ingested.shape}, "
            f"Date range: {df_ingested.index.min()} to {df_ingested.index.max()}"
        )
        
        # 3. Validate raw data
        try:
            validate_forex_data(
                df_ingested,
                is_historical=True,
                z_threshold=5.0  # More permissive for test data
            )
            logger.info("Raw data validation passed")
        except Exception as e:
            logger.warning(f"Raw data validation raised: {str(e)}")
        
        # 4. Clean data with different methods
        cleaning_configs = [
            ("Basic cleaning", {'methods': ['hampel']}),
            ("Aggressive cleaning", {'methods': ['hampel', 'rolling_zscore', 'median']}),
        ]
        
        for config_name, kwargs in cleaning_configs:
            logger.info(f"\nTesting cleaning configuration: {config_name}")
            cleaned_df = adaptive_clean_forex_data(df_ingested, **kwargs)
            
            # Verify cleaning results
            logger.info(f"Cleaning results for {config_name}:")
            logger.info(f"- Original shape: {df_ingested.shape}")
            logger.info(f"- Cleaned shape: {cleaned_df.shape}")
            logger.info(f"- Quality score: {cleaned_df['quality_score'].mean():.3f}")
            logger.info(f"- Methods applied: {cleaned_df.attrs['cleaning_info']['methods_applied']}")
            
            # Validate cleaned data
            validate_forex_data(
                cleaned_df,
                is_historical=True,
                z_threshold=4.0  # Stricter for cleaned data
            )
            logger.info(f"Cleaned data validation passed for {config_name}")
            
            # Check specific improvements
            improvements = {
                'missing_values': (
                    df_ingested.isnull().sum().sum(),
                    cleaned_df.isnull().sum().sum()
                ),
                'high_low_violations': (
                    (df_ingested['high'] < df_ingested['low']).sum(),
                    (cleaned_df['high'] < cleaned_df['low']).sum()
                )
            }
            
            for metric, (before, after) in improvements.items():
                logger.info(f"- {metric}: {before} -> {after}")
        
        logger.info("\nPipeline integration test completed successfully")
        return True
        
    except Exception as e:
        logger.error("Pipeline integration test failed", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        test_pipeline_integration()
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        sys.exit(1)