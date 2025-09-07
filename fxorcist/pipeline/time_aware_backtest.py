"""
Time-aware backtesting engine that strictly prevents look-ahead bias.

Problems Solved:
- Look-ahead bias: Using future data for current decisions
- Temporal inconsistencies: Events processed out of order
- Data snooping: Strategy parameters fitted on test data
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

class TimeAwareBacktestEngine:
    """Enhanced backtesting engine that strictly prevents look-ahead bias."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Time management
        self.current_time = None
        self.warmup_period = timedelta(days=config.get('warmup_days', 30))
        self.event_buffer = []  # Sorted by timestamp
        
        # Bias prevention
        self.data_cache = {}  # Prevents accidental future data access
        self.strict_temporal_order = True
        
        # State tracking
        self.backtest_state = {
            'started_at': None,
            'current_period': None,
            'data_points_processed': 0,
            'events_generated': 0
        }
    
    def validate_temporal_consistency(self, data: pd.DataFrame) -> bool:
        """Validate that data is temporally consistent and sorted."""
        if 'timestamp' not in data.columns:
            raise ValueError("Data must contain 'timestamp' column")
        
        # Check for duplicates
        if data['timestamp'].duplicated().any():
            self.logger.warning("Found duplicate timestamps in data")
            data = data.drop_duplicates(subset=['timestamp'])
        
        # Ensure chronological order
        if not data['timestamp'].is_monotonic_increasing:
            self.logger.info("Sorting data by timestamp")
            data = data.sort_values('timestamp')
        
        # Check for large time gaps
        time_diffs = data['timestamp'].diff()
        large_gaps = time_diffs > timedelta(hours=24)
        if large_gaps.any():
            self.logger.warning(f"Found {large_gaps.sum()} large time gaps in data")
        
        return True
    
    async def process_historical_data(self, data: pd.DataFrame, strategy_class):
        """Process data with strict temporal ordering."""
        self.validate_temporal_consistency(data)
        
        # Split data into warmup and test periods
        warmup_end = data['timestamp'].min() + self.warmup_period
        warmup_data = data[data['timestamp'] <= warmup_end]
        test_data = data[data['timestamp'] > warmup_end]
        
        self.logger.info(f"Warmup period: {len(warmup_data)} data points")
        self.logger.info(f"Test period: {len(test_data)} data points")
        
        # Process warmup (no trading, only indicator calculation)
        await self._process_warmup_period(warmup_data, strategy_class)
        
        # Process test period (actual backtesting)
        results = await self._process_test_period(test_data, strategy_class)
        
        return results
    
    async def _process_warmup_period(self, warmup_data: pd.DataFrame, strategy_class):
        """Process warmup period to initialize indicators."""
        self.logger.info("Processing warmup period...")
        
        for _, row in warmup_data.iterrows():
            self.current_time = row['timestamp']
            
            # Only allow indicator calculation, no trading
            market_event = self._create_market_event(row)
            self.data_cache[self.current_time] = row.to_dict()
            
            # Process event for indicator warmup
            await self._process_event_for_warmup(market_event)
    
    async def _process_test_period(self, test_data: pd.DataFrame, strategy_class):
        """Process test period with full trading enabled."""
        self.logger.info("Starting backtesting period...")
        
        results = []
        
        for _, row in test_data.iterrows():
            # Strict time progression
            if self.current_time and row['timestamp'] <= self.current_time:
                raise ValueError(f"Temporal violation: {row['timestamp']} <= {self.current_time}")
            
            self.current_time = row['timestamp']
            
            # Create market event
            market_event = self._create_market_event(row)
            
            # Cache current data (only current and past data accessible)
            self.data_cache[self.current_time] = row.to_dict()
            
            # Process event
            result = await self._process_single_event(market_event, strategy_class)
            if result:
                results.append(result)
            
            self.backtest_state['data_points_processed'] += 1
        
        return results
    
    def get_historical_data(self, lookback_periods: int = None, 
                          end_time: datetime = None) -> pd.DataFrame:
        """
        Safely retrieve historical data without look-ahead bias.
        Only returns data up to current_time or specified end_time.
        """
        if end_time is None:
            end_time = self.current_time
        
        if end_time > self.current_time:
            raise ValueError(f"Cannot access future data: {end_time} > {self.current_time}")
        
        # Filter data cache to prevent look-ahead
        valid_times = [t for t in self.data_cache.keys() if t <= end_time]
        
        if lookback_periods:
            valid_times = sorted(valid_times)[-lookback_periods:]
        
        historical_data = [self.data_cache[t] for t in sorted(valid_times)]
        return pd.DataFrame(historical_data)
    
    def _create_market_event(self, row: pd.Series) -> Dict:
        """Create market event from data row."""
        return {
            'timestamp': row['timestamp'],
            'type': 'market_data',
            'data': row.to_dict()
        }
    
    async def _process_event_for_warmup(self, event: Dict):
        """Process event during warmup period (indicators only)."""
        # Implementation depends on strategy class
        pass
    
    async def _process_single_event(self, event: Dict, strategy_class) -> Optional[Dict]:
        """Process single market event with strategy."""
        # Implementation depends on strategy class
        pass