from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
from forex_ai_dashboard.utils.logger import logger

@dataclass
class PredictionRecord:
    """Represents a single prediction record in the memory matrix"""
    timestamp: datetime
    model_version: str
    features: Dict[str, float]
    prediction: float
    actual: Optional[float] = None
    error_metrics: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None

class MemorySchema:
    """Manages the schema for storing prediction history and model performance"""
    def __init__(self):
        self.records: List[PredictionRecord] = []
        
    def add_record(self, record: PredictionRecord):
        """Add a new prediction record to memory"""
        try:
            self.records.append(record)
            logger.info(f"Added record for model {record.model_version} at {record.timestamp}")
        except Exception as e:
            logger.error(f"Failed to add record: {e}")
            
    def get_records(self, model_version: str = None, start_date: datetime = None, end_date: datetime = None) -> List[PredictionRecord]:
        """Retrieve records filtered by model version and date range"""
        filtered = self.records
        
        if model_version:
            filtered = [r for r in filtered if r.model_version == model_version]
            
        if start_date:
            filtered = [r for r in filtered if r.timestamp >= start_date]
            
        if end_date:
            filtered = [r for r in filtered if r.timestamp <= end_date]
            
        return filtered
    
    def calculate_meta_features(self, model_version: str = None) -> Dict[str, float]:
        """Calculate aggregate metrics across stored records"""
        records = self.get_records(model_version)
        
        if not records:
            return {}
            
        metrics = {
            'total_predictions': len(records),
            'avg_error': None,
            'recent_performance': {}
        }
        
        # Calculate average error if actual values exist
        errors = [r.error_metrics.get('RMSE', 0) for r in records if r.error_metrics]
        if errors:
            metrics['avg_error'] = sum(errors) / len(errors)
            
        return metrics
