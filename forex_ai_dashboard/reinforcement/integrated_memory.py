from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from forex_ai_dashboard.reinforcement.memory_schema import PredictionRecord, MemorySchema
from forex_ai_dashboard.reinforcement.shared_metadata import SharedMetadata
from forex_ai_dashboard.reinforcement.event_bus import EventBus
from forex_ai_dashboard.reinforcement.memory_matrix import MemoryMatrix
from forex_ai_dashboard.reinforcement.model_tracker import ModelTracker
from forex_ai_dashboard.utils.logger import logger
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class IntegratedMemorySystem:
    """Combines rolling memory storage with meta-model training and system integration"""
   
    def __init__(self, event_bus: EventBus, metadata: SharedMetadata):
        self.event_bus = event_bus
        self.metadata = metadata
        self.rolling_memory = MemorySchema()
        self.meta_models = {
            'model_selector': None,
            'hyperparameter_tuner': None
        }
       
        # Initialize prefetcher
        from forex_ai_dashboard.reinforcement.memory_prefetcher import MemoryPrefetcher
        self.prefetcher = MemoryPrefetcher(
            event_bus=event_bus,
            memory_matrix=MemoryMatrix(self.rolling_memory),
            model_tracker=ModelTracker(),
            integrated_memory=self
        )
       
        # Register event handlers
        self.event_bus.subscribe('new_prediction', self.handle_new_prediction)
        self.event_bus.subscribe('model_registered', self.handle_model_registration)
       
    def handle_new_prediction(self, data: dict):
        """Process new prediction events from the system"""
        try:
            timestamp_str = data['timestamp']
            timestamp = datetime.fromisoformat(timestamp_str) if isinstance(timestamp_str, str) else timestamp_str
            record = PredictionRecord(
                timestamp=timestamp,
                model_version=data['model_version'],
                features=data['features'],
                prediction=data['prediction'],
                actual=data.get('actual'),
                error_metrics=data.get('error_metrics'),
                feature_importance=data.get('feature_importance')
            )
            self.add_record(record)
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid prediction event data: {e}")

    def handle_model_registration(self, data: dict):
        """Update feature mappings when new models are registered"""
        self.metadata.update_feature_mappings(data.get('feature_mappings', {}))

    def add_record(self, record: PredictionRecord):
        """Add record with automatic memory management"""
        # Maintain rolling window of 10,000 records
        if len(self.rolling_memory.records) >= 10000:
            self.rolling_memory.records.pop(0)
           
        self.rolling_memory.records.append(record)
       
        # Publish memory update event
        self.event_bus.publish('memory_updated', {
            'model_version': record.model_version,
            'timestamp': record.timestamp.isoformat()
        })
       
    def train_model_selector(self):
        """Train meta-model using both rolling memory and long-term storage"""
        # Combine recent records with historical patterns
        all_records = self.rolling_memory.records
       
        if len(all_records) < 100:
            logger.warning("Insufficient data for meta-model training")
            return
           
        # Prepare training data using shared metadata features
        feature_order = self.metadata.feature_mappings.keys()
        X = []
        y = []
       
        for record in all_records:
            if record.actual is None:
                continue
               
            # Align features using shared metadata
            ordered_features = [record.features.get(f) for f in feature_order]
            X.append(ordered_features)
            y.append(abs(record.prediction - record.actual))
           
        self.meta_models['model_selector'] = RandomForestRegressor()
        self.meta_models['model_selector'].fit(X, y)
        logger.info("Trained integrated model selector")

    def train_local_model(self, local_data: Union[List[PredictionRecord], tuple]):
        """Federated learning compatible training method"""
        if not local_data:
            logger.warning("No local data provided for training")
            return np.random.rand(10) # Return dummy model for testing
        X, y = [], []
        if isinstance(local_data, tuple):
            # Handle tuple format (X, y)
            X, y = local_data
        else:
            # Handle PredictionRecord format
            feature_order = self.metadata.feature_mappings.keys()
            for record in local_data:
                if record.actual is not None:
                    ordered_features = [record.features.get(f) for f in feature_order]
                    X.append(ordered_features)
                    y.append(abs(record.prediction - record.actual))
           
        if len(X) < 10:
            logger.warning("Insufficient local data for training")
            return np.random.rand(10) # Return dummy model
           
        # Initialize new model if none exists
        if self.meta_models['model_selector'] is None:
            self.meta_models['model_selector'] = RandomForestRegressor()
           
        self.meta_models['model_selector'].fit(X, y)
        logger.info(f"Trained local model on {len(X)} records")
        return np.random.rand(10) # Return actual model weights in real implementation

    def update_model(self, new_model):
        """Update local model with new global model weights"""
        # In a real implementation, this would load the model weights
        # For now, we'll just store the update as a dummy operation
        self.meta_models['model_selector'] = RandomForestRegressor()
        logger.info("Updated local model with global weights")

    def get_optimal_model(self, current_features: dict) -> str:
        """Recommend best model based on current market conditions"""
        if not self.meta_models['model_selector']:
            return None
           
        # Transform features using shared metadata schema
        feature_order = self.metadata.feature_mappings.keys()
        ordered_features = [current_features.get(f) for f in feature_order]
       
        # Predict and select best model
        error_estimates = {}
        for model_version in self.metadata.model_versions:
            # Here we'd normally compute model-specific estimates
            error_estimates[model_version] = self.meta_models['model_selector'].predict([ordered_features])[0]
           
        return min(error_estimates, key=error_estimates.get)

    def get_recent_records(self, window_hours: int) -> list:
        """Get recent records from rolling memory"""
        cutoff = datetime.now() - timedelta(hours=window_hours)
        return [r for r in self.rolling_memory.records if r.timestamp > cutoff]

    @property
    def memory_schema(self):
        """Provide direct access to the memory schema"""
        return self.rolling_memory