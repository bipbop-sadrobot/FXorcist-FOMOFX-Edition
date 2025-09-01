import numpy as np
from sklearn.ensemble import IsolationForest
from forex_ai_dashboard.reinforcement.integrated_memory import IntegratedMemorySystem
from forex_ai_dashboard.utils.logger import logger
from datetime import datetime, timedelta
from .event_bus import EventBus
from .shared_metadata import SharedMetadata

class AnomalyDetector:
    """Real-time anomaly detection for memory system"""
    
    def __init__(self, event_bus: EventBus, metadata: SharedMetadata):
        self.memory = IntegratedMemorySystem(event_bus, metadata)
        self.model = IsolationForest(contamination=0.01)
        self.last_trained = datetime.now()
        
    def detect_anomalies(self, window_hours: int = 24) -> list:
        """Detect anomalous records in recent memory"""
        recent_records = self.memory.get_recent_records(window_hours)
        if len(recent_records) < 100:
            return []
            
        features = self._extract_features(recent_records)
        
        # Retrain model periodically
        if (datetime.now() - self.last_trained) > timedelta(hours=1):
            self.model.fit(features)
            self.last_trained = datetime.now()
            
        predictions = self.model.predict(features)
        anomalies = [rec for rec, pred in zip(recent_records, predictions) if pred == -1]
        
        if anomalies:
            logger.warning(f"Detected {len(anomalies)} anomalies in last {window_hours}h")
            
        return anomalies
        
    def _extract_features(self, records: list) -> np.ndarray:
        """Convert records to anomaly detection features"""
        features = []
        for rec in records:
            feat = [
                rec.prediction,
                abs(rec.prediction - rec.actual) if rec.actual else 0,
                len(rec.features),
                np.std(list(rec.features.values())) if rec.features else 0
            ]
            features.append(feat)
        return np.array(features).reshape(-1, 4)  # Ensure 2D array
        
    def stream_detection(self):
        """Continuous anomaly monitoring"""
        while True:
            anomalies = self.detect_anomalies()
            if anomalies:
                self._trigger_alert(anomalies)
            time.sleep(300)  # Check every 5 minutes
            
    def _trigger_alert(self, anomalies: list):
        """Handle anomaly alerts"""
        logger.critical(f"ANOMALY ALERT: {len(anomalies)} suspicious records detected")
        # Additional alerting logic would go here
