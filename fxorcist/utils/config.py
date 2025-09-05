"""
Configuration Management Module

Provides centralized configuration handling for the FXorcist package,
including loading, validation, and access to configuration settings.
"""

import os
import yaml
from typing import Any, Dict, Optional

class Config:
    """Central configuration manager for FXorcist."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Optional path to config file. If None, uses default locations.
        """
        self.config_path = config_path or os.getenv('FXORCIST_CONFIG', 'risk_config.yml')
        self._config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        else:
            self._config = self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Provide default configuration settings."""
        return {
            'risk': {
                'max_position_size': 0.02,  # 2% max position size
                'max_total_risk': 0.05,     # 5% max portfolio risk
                'stop_loss_pips': 50,       # Default stop loss
            },
            'trading': {
                'allowed_pairs': ['EURUSD', 'GBPUSD', 'USDJPY'],
                'trading_hours': {'start': '00:00', 'end': '23:59'},
            },
            'system': {
                'log_level': 'INFO',
                'cache_dir': 'artifacts',
                'max_memory_usage': '4G',
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (dot notation supported)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            value = self._config
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

# Global configuration instance
config = Config()