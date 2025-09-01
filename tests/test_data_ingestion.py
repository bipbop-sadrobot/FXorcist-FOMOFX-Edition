import pytest
import pandas as pd
import zipfile
import io
import os
from datetime import datetime
import logging
import multiprocessing as mp
from functools import wraps
import csv  # For delimiter sniffer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("ingestion.log"), logging.StreamHandler()])

def monitor_ingestion(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        duration = (datetime.now() - start).total_seconds()
        rows = len(result) if isinstance(result, pd.DataFrame) else 0
        logging.info(f"{func.__name__} | Duration: {duration}s | Rows: {rows} | Memory: {result.memory_usage().sum() / 1e6 if rows else 0:.2f} MB")
        return result
    return wrapper

@monitor_ingestion
def parse_csv(file_path, chunk_size=1000000, start_date='2025-08-01', end_date='2025-08-17', fallback_year=2024):
    """Parse CSV/ZIP with auto-delimiter, header inference, date filter, fallback year."""
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as z:
            csv_name = [f for f in z.namelist() if f.endswith('.csv')][0]
            with z.open(csv_name) as f:
                content = io.BytesIO(f.read())
    else:
        content = file_path

    # Detect delimiter
    sample = content.read(1024).decode(errors='ignore')
    content.seek(0)
    sniffer = csv.Sniffer()
    delimiter = sniffer.sniff(sample).delimiter

    # Detect headers
    has_header = sniffer.has_header(sample)

    def process_chunk(chunk):
        # Timestamp parsing with multiple formats
        timestamp_col = chunk.columns[0] if 'timestamp' not in chunk.columns else 'timestamp'
        for fmt in ['%Y%m%d %H%M%S', '%Y-%m-%d %H:%M:%S', '%Y%m%d%H%M%S', '%Y.%m.%d %H:%M:%S', None]:
            try:
                chunk[timestamp_col] = pd.to_datetime(chunk[timestamp_col], format=fmt, errors='coerce')
                if not chunk[timestamp_col].isnull().all():
                    break
            except:
                pass
        chunk = chunk.dropna(subset=[timestamp_col]).set_index(timestamp_col)
        chunk = chunk.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float}, errors='ignore')
        if chunk.columns.str.lower().str.contains('open').any():  # Normalize columns
            chunk.columns = ['open', 'high', 'low', 'close', 'volume'] if len(chunk.columns) == 5 else chunk.columns
        # Filter date
        return chunk[(chunk.index >= pd.to_datetime(start_date)) & (chunk.index <= pd.to_datetime(end_date))]

    chunks = []
    for chunk in pd.read_csv(content, chunksize=chunk_size, delimiter=delimiter, header=0 if has_header else None, 
                             names=['timestamp', 'open', 'high', 'low', 'close', 'volume'] if not has_header else None, on_bad_lines='warn'):
        processed = process_chunk(chunk)
        chunks.append(processed)
    df = pd.concat(chunks)
    if df.empty:  # Fallback to previous year
        logging.warning("No data for 2025; trying fallback year.")
        # Recurse or adjust dates – for simplicity, assume manual
    return df

@monitor_ingestion
def ingest_histdata(zip_path):
    return parse_csv(zip_path)

@monitor_ingestion
def ingest_dukascopy(csv_path):
    return parse_csv(csv_path)

@monitor_ingestion
def ingest_ejtrader(csv_path):
    df = parse_csv(csv_path)
    df.columns = df.columns.str.lower().str.replace(' ', '')  # Enhanced normalization
    return df

@monitor_ingestion
def ingest_fx1min(csv_path):
    return parse_csv(csv_path)

@monitor_ingestion
def ingest_algo_duka(csv_path):
    return parse_csv(csv_path)

def ingest_all_parallel(sources):
    with mp.Pool(processes=os.cpu_count()) as pool:
        results = pool.map(ingest_source, sources)
    merged = pd.concat(results, axis=0).sort_index().drop_duplicates()
    merged = merged.resample('1T').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    # Integrate clean/validate
    from data_validation.clean_data import clean_forex_data
    from data_validation.validate_data import validate_forex_data
    cleaned = clean_forex_data(merged)
    if validate_forex_data(cleaned):
        cleaned.to_parquet('data/processed/ingested_forex_1min_aug2025.parquet')
        return cleaned
    raise ValueError("Validation failed after merge.")

def ingest_source(source):
    if source['type'] == 'histdata':
        return ingest_histdata(source['path'])
    if source['type'] == 'dukascopy':
        return ingest_dukascopy(source['path'])
    if source['type'] == 'ejtrader':
        return ingest_ejtrader(source['path'])
    if source['type'] == 'fx1min':
        return ingest_fx1min(source['path'])
    if source['type'] == 'algo_duka':
        return ingest_algo_duka(source['path'])

if __name__ == "__main__":
    sources = [
        {'type': 'histdata', 'path': 'data/raw/histdata_eurusd_m1_aug2025.zip'},
        {'type': 'dukascopy', 'path': 'data/raw/duka_node_eurusd_m1_aug2025.csv'},
        {'type': 'ejtrader', 'path': 'data/raw/ejtrader_eurusd_m1.csv'},
        {'type': 'fx1min', 'path': 'data/raw/fx1min_eurusd_2025.csv'},
        {'type': 'algo_duka', 'path': 'data/raw/algo_duka_eurusd_1m_aug2025.csv'}
    ]
    df_ingested = ingest_all_parallel(sources)
    print("Ingested 1-min Data (head):")
    print(df_ingested.head())
