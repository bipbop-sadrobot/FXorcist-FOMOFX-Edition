from typing import Dict, Any
from forex_ai_dashboard.utils.logger import logger

class SharedMetadata:
    """Central repository for shared system metadata"""
    
    def __init__(self):
        self.model_versions: Dict[str, Dict[str, Any]] = {}  # version: {deployment_time, features}
        self.feature_mappings: Dict[str, str] = {}  # feature_name: data_type
        self._subscribers = []

    def register_model_version(self, version: str, deployment_time: str, features: list):
        """Register a new model version with its features"""
        if version in self.model_versions:
            logger.warning(f"Overwriting existing model version: {version}")
            
        self.model_versions[version] = {
            'deployment_time': deployment_time,
            'features': features
        }
        logger.info(f"Registered model version {version} with {len(features)} features")

    def update_feature_mappings(self, mappings: Dict[str, str]):
        """Update feature data type mappings"""
        self.feature_mappings.update(mappings)
        logger.info(f"Updated {len(mappings)} feature mappings")

    def get_model_features(self, version: str) -> list:
        """Get feature list for a model version"""
        return self.model_versions.get(version, {}).get('features', [])

    def get_model_metadata(self, model_version: str) -> dict:
        """Get metadata for a specific model version"""
        return self.model_versions.get(model_version, {})

    def subscribe(self, callback):
        """Subscribe to metadata updates"""
        self._subscribers.append(callback)

    def _notify_subscribers(self, event_type: str, data: dict):
        """Notify all subscribers of metadata changes"""
        for callback in self._subscribers:
            callback(event_type, data)
            
    def generate_documentation(self) -> dict:
        """Generate comprehensive documentation of current metadata state"""
        return {
            "model_versions": list(self.model_versions.keys()),
            "feature_count": len(self.feature_mappings),
            "subscriber_count": len(self._subscribers),
            "last_updated": datetime.now().isoformat(),
            "persistence_enabled": self.persistent
        }
        
    # Database methods
    def _init_db(self):
        """Initialize database tables"""
        # Implementation would initialize database tables
        pass
            
    def _load_from_db(self):
        """Load data from database"""
        # Implementation would load data from database
        pass
            
    def save_to_db(self):
        """Persist current state to database"""
        # Implementation would persist current state to database
        pass
