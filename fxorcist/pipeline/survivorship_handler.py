"""
Handles survivorship bias by tracking instrument universe changes.

Problems Solved:
- Survivor bias: Only using instruments that survived to present
- Selection bias: Cherry-picking best performing assets
- Corporate actions: Splits, mergers, delistings
"""

import logging
import pandas as pd
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CorporateAction:
    """Details of a corporate action event."""
    date: datetime
    action_type: str  # SPLIT, MERGER, DELISTING, etc.
    instrument: str
    details: Dict[str, Any]
    impact_factor: float = 1.0  # Price adjustment factor

class SurvivorshipBiasHandler:
    """Handles survivorship bias by tracking instrument universe changes."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.universe_history = {}  # timestamp -> list of active instruments
        self.delisting_dates = {}   # instrument -> delisting date
        self.corporate_actions = {} # instrument -> list of actions
        self.db_path = None
        
    def initialize_database(self, db_path: str):
        """Initialize SQLite database for universe history."""
        self.db_path = Path(db_path)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS universe_history (
                    timestamp TEXT,
                    instrument TEXT,
                    status TEXT,
                    PRIMARY KEY (timestamp, instrument)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS corporate_actions (
                    date TEXT,
                    instrument TEXT,
                    action_type TEXT,
                    details TEXT,
                    impact_factor REAL,
                    PRIMARY KEY (date, instrument, action_type)
                )
            """)
    
    def load_universe_data(self, universe_file: str):
        """Load historical universe data including delistings."""
        try:
            universe_df = pd.read_csv(universe_file, parse_dates=['timestamp'])
            
            # Validate data format
            required_cols = ['timestamp', 'instrument', 'status']
            if not all(col in universe_df.columns for col in required_cols):
                raise ValueError(f"Universe file must contain columns: {required_cols}")
            
            # Sort by timestamp
            universe_df = universe_df.sort_values('timestamp')
            
            # Update universe history
            for timestamp, group in universe_df.groupby('timestamp'):
                active_instruments = group[group['status'] == 'ACTIVE']['instrument'].tolist()
                self.universe_history[timestamp] = active_instruments
                
                # Track delistings
                delisted = group[group['status'] == 'DELISTED']
                for _, row in delisted.iterrows():
                    self.delisting_dates[row['instrument']] = row['timestamp']
            
            self.logger.info(f"Loaded universe data with {len(self.universe_history)} timestamps")
            
        except Exception as e:
            self.logger.error(f"Failed to load universe data: {e}")
            raise
    
    def get_active_instruments(self, timestamp: datetime) -> List[str]:
        """Get instruments active at given timestamp."""
        # Find the latest universe snapshot before or at timestamp
        valid_timestamps = [t for t in self.universe_history.keys() if t <= timestamp]
        
        if not valid_timestamps:
            return []
        
        latest_timestamp = max(valid_timestamps)
        return self.universe_history[latest_timestamp]
    
    def handle_corporate_action(self, instrument: str, action_date: datetime, 
                              action_type: str, details: Dict):
        """Handle corporate actions like splits, mergers."""
        if instrument not in self.corporate_actions:
            self.corporate_actions[instrument] = []
        
        action = CorporateAction(
            date=action_date,
            action_type=action_type,
            instrument=instrument,
            details=details
        )
        
        # Calculate impact factor based on action type
        if action_type == 'SPLIT':
            ratio = details.get('split_ratio', 1.0)
            action.impact_factor = 1.0 / ratio
        elif action_type == 'MERGER':
            action.impact_factor = details.get('exchange_ratio', 1.0)
        
        self.corporate_actions[instrument].append(action)
        
        # Store in database if configured
        if self.db_path:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO corporate_actions
                    (date, instrument, action_type, details, impact_factor)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    action_date.isoformat(),
                    instrument,
                    action_type,
                    str(details),
                    action.impact_factor
                ))
    
    def adjust_price_history(self, df: pd.DataFrame, instrument: str) -> pd.DataFrame:
        """Adjust historical prices for corporate actions."""
        if instrument not in self.corporate_actions:
            return df
        
        df = df.copy()
        price_cols = ['open', 'high', 'low', 'close']
        
        # Apply adjustments in reverse chronological order
        actions = sorted(
            self.corporate_actions[instrument],
            key=lambda x: x.date,
            reverse=True
        )
        
        cumulative_factor = 1.0
        
        for action in actions:
            # Apply adjustment to all prices before the action
            mask = df.index < action.date
            cumulative_factor *= action.impact_factor
            
            for col in price_cols:
                if col in df.columns:
                    df.loc[mask, col] *= cumulative_factor
            
            # Adjust volume in opposite direction
            if 'volume' in df.columns:
                df.loc[mask, 'volume'] /= cumulative_factor
        
        return df
    
    def validate_universe_consistency(self, instruments: List[str], 
                                    start_date: datetime, end_date: datetime) -> bool:
        """
        Validate that instruments were consistently available during period.
        Returns False if any instrument was delisted during period.
        """
        for instrument in instruments:
            if instrument in self.delisting_dates:
                delisting_date = self.delisting_dates[instrument]
                if start_date <= delisting_date <= end_date:
                    self.logger.warning(
                        f"Instrument {instrument} was delisted on {delisting_date}"
                    )
                    return False
        return True
    
    def get_universe_changes(self, start_date: datetime, end_date: datetime) -> Dict:
        """Get summary of universe changes during period."""
        changes = {
            'additions': [],
            'deletions': [],
            'corporate_actions': []
        }
        
        # Find relevant timestamps
        period_timestamps = [
            t for t in self.universe_history.keys()
            if start_date <= t <= end_date
        ]
        
        if len(period_timestamps) < 2:
            return changes
        
        # Track changes between consecutive snapshots
        for i in range(1, len(period_timestamps)):
            prev_universe = set(self.universe_history[period_timestamps[i-1]])
            curr_universe = set(self.universe_history[period_timestamps[i]])
            
            additions = curr_universe - prev_universe
            deletions = prev_universe - curr_universe
            
            if additions:
                changes['additions'].append({
                    'date': period_timestamps[i],
                    'instruments': list(additions)
                })
            
            if deletions:
                changes['deletions'].append({
                    'date': period_timestamps[i],
                    'instruments': list(deletions)
                })
        
        # Add corporate actions
        for instrument, actions in self.corporate_actions.items():
            relevant_actions = [
                action for action in actions
                if start_date <= action.date <= end_date
            ]
            if relevant_actions:
                changes['corporate_actions'].extend([{
                    'date': action.date,
                    'instrument': instrument,
                    'type': action.action_type,
                    'details': action.details
                } for action in relevant_actions])
        
        return changes