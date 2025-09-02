"""
Data loading utility for the Forex AI dashboard.
Handles efficient data loading, validation, and caching.
"""

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from functools import lru_cache
import pyarrow.parquet as pq
import json

from forex_ai_dashboard.pipeline.evaluation_metrics import EvaluationMetrics
from forex_ai_dashboard.models.model_hierarchy import ModelHierarchy
from forex_ai_dashboard.utils.caching import get_cache_manager, get_dataframe_cache, get_model_cache

logger = logging.getLogger(__name__)

class DataLoader:
    """Utility class for efficient data loading and management."""
    
    def __init__(self):
        """Initialize data loader with paths and cache settings."""
        self.data_dir = Path('data/processed')
        self.eval_dir = Path('evaluation_results')
        self.model_dir = Path('models/hierarchy')
        self.cache_timeout = 300  # 5 minutes
        self._last_load_time = {}
        self._cache = {}
    
    def _should_reload(self, key: str) -> bool:
        """Check if data should be reloaded based on cache timeout."""
        if key not in self._last_load_time:
            return True
        
        elapsed = (datetime.now() - self._last_load_time[key]).total_seconds()
        return elapsed > self.cache_timeout
    
    @lru_cache(maxsize=1)
    def _get_latest_file(self, directory: Path, pattern: str) -> Optional[Path]:
        """Get the latest file matching pattern in directory."""
        try:
            files = list(directory.glob(pattern))
            if not files:
                return None
            return max(files, key=lambda x: x.stat().st_mtime)
        except Exception as e:
            logger.error(f"Error finding latest file: {str(e)}")
            return None
    
    def _validate_forex_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Validate forex data for common issues."""
        issues = []
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            issues.append(f"Missing values found: {missing[missing > 0].to_dict()}")
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Check for duplicate timestamps
        duplicates = df.index.duplicated()
        if duplicates.any():
            issues.append(f"Found {duplicates.sum()} duplicate timestamps")
            df = df[~duplicates]
        
        # Check for price anomalies
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                # Check for negative prices
                if (df[col] <= 0).any():
                    issues.append(f"Found negative or zero prices in {col}")
                    df = df[df[col] > 0]
                
                # Check for extreme price movements
                pct_change = df[col].pct_change().abs()
                extreme_moves = pct_change > 0.1  # 10% threshold
                if extreme_moves.any():
                    issues.append(f"Found {extreme_moves.sum()} extreme price movements in {col}")
        
        return df, issues
    
    def load_forex_data(
        self,
        timeframe: str = "1H",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Load and validate forex data."""
        cache_key = f"forex_{timeframe}"
        
        if not self._should_reload(cache_key) and cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            latest_file = self._get_latest_file(self.data_dir, "*.parquet")
            if not latest_file:
                return pd.DataFrame(), ["No data files found"]
            
            # Read parquet file efficiently
            df = pq.read_table(
                latest_file,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            ).to_pandas()
            
            # Set timestamp as index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Resample to desired timeframe
            if timeframe != "1M":
                df = df.resample(timeframe).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })
            
            # Apply date filters if provided
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            # Validate data
            df, issues = self._validate_forex_data(df)
            
            # Update cache
            self._cache[cache_key] = (df, issues)
            self._last_load_time[cache_key] = datetime.now()
            
            return df, issues
            
        except Exception as e:
            logger.error(f"Error loading forex data: {str(e)}")
            return pd.DataFrame(), [f"Error loading data: {str(e)}"]
    
    def load_evaluation_results(
        self,
        limit: int = 50
    ) -> Tuple[List[EvaluationMetrics], List[str]]:
        """Load recent evaluation results."""
        cache_key = f"eval_results_{limit}"
        
        if not self._should_reload(cache_key) and cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            results = []
            issues = []
            
            files = sorted(
                self.eval_dir.glob('*.json'),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )[:limit]
            
            for f in files:
                try:
                    with open(f, 'r') as file:
                        data = json.load(file)
                        results.append(EvaluationMetrics.from_dict(data))
                except Exception as e:
                    issues.append(f"Error loading {f.name}: {str(e)}")
            
            # Update cache
            self._cache[cache_key] = (results, issues)
            self._last_load_time[cache_key] = datetime.now()
            
            return results, issues
            
        except Exception as e:
            logger.error(f"Error loading evaluation results: {str(e)}")
            return [], [f"Error loading evaluation results: {str(e)}"]
    
    def load_model_hierarchy(self) -> Tuple[Optional[ModelHierarchy], List[str]]:
        """Load current model hierarchy."""
        cache_key = "model_hierarchy"
        
        if not self._should_reload(cache_key) and cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            latest_file = self._get_latest_file(self.model_dir, "*.json")
            if not latest_file:
                return None, ["No model hierarchy file found"]
            
            with open(latest_file, 'r') as f:
                hierarchy_data = json.load(f)
            
            hierarchy = ModelHierarchy.from_dict(hierarchy_data)
            
            # Update cache
            self._cache[cache_key] = (hierarchy, [])
            self._last_load_time[cache_key] = datetime.now()
            
            return hierarchy, []
            
        except Exception as e:
            logger.error(f"Error loading model hierarchy: {str(e)}")
            return None, [f"Error loading model hierarchy: {str(e)}"]
    
    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        self._last_load_time.clear()
        self._get_latest_file.cache_clear()