import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import hmac
import hashlib
import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

class EventBus:
    """Pub/sub event bus for system communication."""
    def __init__(self):
        self.subscribers = defaultdict(list)

    def subscribe(self, event: str, callback):
        self.subscribers[event].append(callback)

    def publish(self, event: str, data):
        for callback in self.subscribers[event]:
            callback(data)

class SharedMetadata:
    """Central metadata repository for models and features."""
    def __init__(self):
        self.models: Dict[str, str] = {}
        self.features: Dict[str, str] = {}
        self.subscribers: List = []

    def register_model(self, name: str, version: str):
        self.models[name] = version

    def add_feature(self, name: str, type_: str):
        self.features[name] = type_

    def generate_documentation(self) -> str:
        doc = "Models:\n" + "\n".join(f"{k}: {v}" for k,v in self.models.items())
        doc += "\nFeatures:\n" + "\n".join(f"{k}: {v}" for k,v in self.features.items())
        return doc

@dataclass
class PredictionRecord:
    """Represents a single prediction record in the memory matrix."""
    timestamp: datetime.datetime
    model_version: str
    features: Dict[str, float]
    prediction: float
    actual: Optional[float] = None
    error_metrics: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None

class IntegratedMemorySystem:
    """Integrated storage and analysis for prediction records."""
    def __init__(self, event_bus: EventBus, metadata: SharedMetadata, max_records: int = 1000):
        self.event_bus = event_bus
        self.metadata = metadata
        self.records: List[PredictionRecord] = []
        self.df_cache = pd.DataFrame(columns=['timestamp', 'model_version', 'prediction', 'actual', 'error_RMSE'], dtype=object)  # Cache for pandas operations
        self.df_cache['prediction'] = self.df_cache['prediction'].astype(float)
        self.df_cache['actual'] = self.df_cache['actual'].astype(float)
        self.df_cache['error_RMSE'] = self.df_cache['error_RMSE'].astype(float)
        self.meta_model = None
        self.max_records = max_records

    def add_record(self, record: PredictionRecord):
        """Add a new prediction record with validation."""
        try:
            if not isinstance(record.timestamp, datetime.datetime):
                record.timestamp = datetime.datetime.now()
            self.records.append(record)
            if len(self.records) > self.max_records:
                self.records = self.records[-self.max_records:]
            # Flatten to df_cache
            new_row = {
                'timestamp': record.timestamp,
                'model_version': record.model_version,
                'prediction': record.prediction,
                'actual': record.actual
            }
            if record.features:
                new_row.update(record.features)
            if record.error_metrics:
                new_row.update({f'error_{k}': v for k,v in record.error_metrics.items()})
            if record.feature_importance:
                new_row.update({f'importance_{k}': v for k,v in record.feature_importance.items()})
            new_df = pd.DataFrame([new_row])
            if self.df_cache.empty:
                self.df_cache = new_df
            else:
                self.df_cache = pd.concat([self.df_cache, new_df], ignore_index=True)
            self.event_bus.publish('memory_updated', record)
            self._train_meta_model()
            print(f"Added record for model {record.model_version} at {record.timestamp}")  # Logger placeholder
        except Exception as e:
            print(f"Failed to add record: {e}")

    def get_records(self, model_version: str = None, start_date: datetime.datetime = None, end_date: datetime.datetime = None) -> List[PredictionRecord]:
        """Retrieve filtered records."""
        filtered = self.records[:]
        if model_version:
            filtered = [r for r in filtered if r.model_version == model_version]
        if start_date:
            filtered = [r for r in filtered if r.timestamp >= start_date]
        if end_date:
            filtered = [r for r in filtered if r.timestamp <= end_date]
        return filtered

    def _train_meta_model(self):
        """Train meta-model for model selection."""
        if len(self.df_cache) > 50 and 'error_RMSE' in self.df_cache.columns and 'model_version' in self.df_cache.columns:
            exclude_cols = ['timestamp', 'model_version'] + [col for col in self.df_cache.columns if col.startswith('error_') or col.startswith('importance_')]
            X = self.df_cache.drop(exclude_cols, axis=1, errors='ignore').select_dtypes(include='number')
            y = self.df_cache['model_version']
            if not X.empty and not y.empty:
                self.meta_model = RandomForestClassifier(n_estimators=50, random_state=42)
                self.meta_model.fit(X, y)

    def select_best_model(self, features: Dict[str, float]) -> Optional[str]:
        """Select best model using meta-model."""
        if self.meta_model is None:
            return None
        df = pd.DataFrame([features])
        return self.meta_model.predict(df)[0]

    def analyze_memory_trends(self) -> Dict[str, float]:
        """Analyze trends in cached data."""
        if self.df_cache.empty:
            return {}
        trends = self.df_cache.mean(numeric_only=True).to_dict()
        # Forex-specific: Add volatility if bid/ask present
        if 'bid' in trends and 'ask' in trends:
            trends['volatility'] = np.std(self.df_cache['bid'] - self.df_cache['ask'])
        return trends

    def generate_insights_report(self) -> str:
        """Generate report from trends."""
        trends = self.analyze_memory_trends()
        report = "Insights:\n" + "\n".join(f"{k}: {v:.4f}" for k,v in trends.items())
        return report

    def calculate_meta_features(self, model_version: str = None) -> Dict[str, float]:
        """Calculate aggregate metrics."""
        records = self.get_records(model_version)
        if not records:
            return {}
        metrics = {
            'total_predictions': len(records),
            'avg_error': None,
            'recent_performance': {}
        }
        errors = [r.error_metrics.get('RMSE', 0) for r in records if r.error_metrics]
        if errors:
            metrics['avg_error'] = sum(errors) / len(errors)
        recent = records[-10:]
        if recent:
            recent_errors = [r.error_metrics.get('RMSE', 0) for r in recent if r.error_metrics]
            if recent_errors:
                metrics['recent_performance']['avg_error'] = sum(recent_errors) / len(recent_errors)
        return metrics

    def prioritize_resources(self) -> Dict[str, float]:
        """Prioritize system resources."""
        trends = self.analyze_memory_trends()
        num_features = len(trends)
        total_error = trends.get('error_RMSE', 0) * len(self.records)
        priorities = {
            'cpu': min(100, num_features * 5 + total_error * 10),
            'memory': min(100, len(self.records) / self.max_records * 100),
            'gpu': 50 if any('deep' in k.lower() for k in self.metadata.models.keys()) else 0
        }
        return priorities

class FederatedMemory:
    """Federated learning component."""
    def __init__(self, event_bus: EventBus, metadata: SharedMetadata, secret_key: bytes = b'secret'):
        self.event_bus = event_bus
        self.metadata = metadata
        self.local_models = []
        self.global_model = None
        self.secret_key = secret_key

    def add_local_model(self, model):
        self.local_models.append(model)

    def train_round(self):
        """Perform federated training round."""
        if not self.local_models:
            return
        model_type = type(self.local_models[0])
        if issubclass(model_type, torch.nn.Module):  # PyTorch
            with torch.no_grad():
                num_models = len(self.local_models)
                if self.global_model is None:
                    self.global_model = model_type()
                for param_g in self.global_model.parameters():
                    param_g.data.zero_()
                for model in self.local_models:
                    for param_g, param_l in zip(self.global_model.parameters(), model.parameters()):
                        param_g.data += param_l.data / num_models
        elif hasattr(self.local_models[0], 'coef_'):  # sklearn linear models
            if self.global_model is None:
                self.global_model = model_type()
            coef_avg = np.mean([m.coef_ for m in self.local_models], axis=0)
            intercept_avg = np.mean([m.intercept_ for m in self.local_models], axis=0)
            self.global_model.coef_ = coef_avg
            self.global_model.intercept_ = intercept_avg
        self.event_bus.publish('model_updated', {'global_model': self.global_model})

    def verify_update(self, update: str, signature: str) -> bool:
        """Verify update signature."""
        if not isinstance(update, bytes):
            update = str(update).encode()
        computed_sig = hmac.new(self.secret_key, update, hashlib.sha256).hexdigest()
        return hmac.compare_digest(computed_sig, signature)

class AnomalyDetector:
    """Real-time anomaly detection."""
    def __init__(self, memory: IntegratedMemorySystem):
        self.memory = memory
        self.detector = IsolationForest(contamination=0.05, random_state=42)
        self.event_bus = memory.event_bus

    def detect_anomalies(self, recent_only: bool = True) -> List[Dict]:
        """Detect anomalies, optionally on recent data."""
        df = self.memory.df_cache.tail(100) if recent_only else self.memory.df_cache
        if len(df) < 20:
            return []
        numeric_df = df.select_dtypes(include='number')
        if numeric_df.empty:
            return []
        self.detector.fit(numeric_df)
        scores = self.detector.decision_function(numeric_df)
        threshold = np.percentile(scores, 5)  # Dynamic threshold based on contamination
        anomalies = df[scores < threshold].copy()
        anomalies['anomaly_score'] = scores[scores < threshold]
        self.event_bus.publish('anomaly_detected', anomalies)
        return anomalies.to_dict('records')
