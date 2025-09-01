import pandas as pd
import numpy as np
import logging

# Set up logging for tracking cleaning process
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_forex_data(df: pd.DataFrame, timestamp_col: str = 'timestamp', price_cols: list = ['open', 'high', 'low', 'close'], volume_col: str = 'volume') -> pd.DataFrame:
    """
    Comprehensive data cleaning function for forex time-series data.

    This function performs the following steps:
    1. Convert timestamp to datetime and set as index.
    2. Handle missing values: Forward-fill for prices (common in time-series to avoid gaps), drop if too many missing.
    3. Detect and remove outliers: Using IQR method for price columns.
    4. Remove duplicates: Based on timestamp.
    5. Ensure price consistency: e.g., high >= low, close between open and high/low.
    6. Derive basic features: e.g., mid-price, spread.
    7. Normalize data: Resample to consistent frequency if needed (e.g., daily).
    8. Log the process and raise errors for critical issues.

    Parameters:
    - df: Input DataFrame with forex data.
    - timestamp_col: Name of the timestamp column.
    - price_cols: List of price-related columns.
    - volume_col: Name of the volume column (optional).

    Returns:
    - Cleaned DataFrame.

    Raises:
    - ValueError: If critical issues like invalid prices or excessive missing data.
    """
    original_shape = df.shape
    logging.info(f"Starting cleaning on DataFrame of shape {original_shape}")

    # Step 1: Timestamp handling
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame.")
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    df = df.dropna(subset=[timestamp_col])  # Drop invalid timestamps
    df = df.set_index(timestamp_col).sort_index()

    # Step 2: Missing values
    missing_perc = df.isnull().mean() * 100
    logging.info(f"Missing values percentage: {missing_perc.to_dict()}")
    if any(missing_perc > 50):
        raise ValueError("Excessive missing data (>50%) in some columns.")
    df[price_cols] = df[price_cols].ffill()  # Forward-fill prices
    if volume_col in df.columns:
        df[volume_col] = df[volume_col].fillna(0)  # Volume can be 0 if missing

    # Step 3: Outlier detection (IQR for each price column)
    for col in price_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        logging.info(f"Outliers in {col}: {outliers.sum()}")
        df[col] = np.where(outliers, np.nan, df[col])  # Replace outliers with NaN
        df[col] = df[col].ffill()  # Fill with previous value

    # Step 4: Remove duplicates (already sorted by index)
    df = df[~df.index.duplicated(keep='first')]

    # Step 5: Price consistency checks
    if 'high' in price_cols and 'low' in price_cols:
        invalid_prices = df['high'] < df['low']
        if invalid_prices.any():
            logging.warning(f"Invalid prices (high < low) found in {invalid_prices.sum()} rows. Correcting by swapping.")
            df.loc[invalid_prices, ['high', 'low']] = df.loc[invalid_prices, ['low', 'high']].values
    if 'open' in price_cols and 'close' in price_cols and 'high' in price_cols and 'low' in price_cols:
        invalid_close = (df['close'] > df['high']) | (df['close'] < df['low'])
        if invalid_close.any():
            raise ValueError(f"Invalid close prices outside high/low in {invalid_close.sum()} rows.")

    # Step 6: Derive features
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        df['mid_price'] = (df['high'] + df['low']) / 2
        df['spread'] = df['high'] - df['low']

    # Step 7: Resample (optional, e.g., to daily if intraday data)
    # df = df.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()

    cleaned_shape = df.shape
    logging.info(f"Cleaning complete. Original shape: {original_shape}, Cleaned shape: {cleaned_shape}")

    return df.reset_index()  # Reset index for output

# Example usage
if __name__ == "__main__":
    # Sample DataFrame for testing
    sample_data = {
        'timestamp': pd.date_range(start='2025-08-17', periods=10, freq='T'),
        'open': np.random.uniform(1.08, 1.10, 10),
        'high': np.random.uniform(1.09, 1.11, 10),
        'low': np.random.uniform(1.07, 1.09, 10),
        'close': np.random.uniform(1.08, 1.10, 10),
        'volume': np.random.randint(100, 1000, 10)
    }
    sample_df = pd.DataFrame(sample_data)
    sample_df.loc[2, 'high'] = sample_df.loc[2, 'low'] - 0.01  # Introduce invalid price
    sample_df.loc[5, 'close'] = np.nan  # Introduce missing value

    cleaned_df = clean_forex_data(sample_df)
    print("Cleaned DataFrame:")
    print(cleaned_df.head())
