from typing import Dict, Any
from forex_ai_dashboard.utils.logger import logger
import re

class SecurityValidator:
    """Provides security validation for memory system inputs"""
    
    @staticmethod
    def validate_record(record: Dict[str, Any]) -> bool:
        """Validate prediction record structure and content"""
        required = ['timestamp', 'model_version', 'features', 'prediction']
        if not all(key in record for key in required):
            logger.error("Invalid record - missing required fields")
            return False
            
        # Validate feature names
        if not SecurityValidator._validate_feature_names(record['features']):
            return False
            
        # Validate numerical values
        try:
            float(record['prediction'])
            for value in record['features'].values():
                float(value)
        except (TypeError, ValueError):
            logger.error("Invalid numerical values in record")
            return False
            
        return True
        
    @staticmethod
    def _validate_feature_names(features: Dict[str, float]) -> bool:
        """Ensure feature names match expected pattern"""
        pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,63}$")
        for name in features.keys():
            if not pattern.match(name):
                logger.error(f"Invalid feature name: {name}")
                return False
        return True
        
    @staticmethod
    def sanitize_input(input_str: str) -> str:
        """Basic input sanitization for security"""
        return re.sub(r"[;\\\"']", "", input_str)
