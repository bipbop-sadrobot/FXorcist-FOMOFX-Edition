"""
Enhanced Configuration Management Module

Provides robust configuration handling for the FXorcist package, including:
- Schema validation using Pydantic
- Environment-specific configurations
- Secure storage for sensitive data
- Version control and migration support
- Type validation and documentation
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, SecretStr
from cryptography.fernet import Fernet

# Setup logging
logger = logging.getLogger(__name__)

class EnvironmentType(str, Enum):
    """Trading environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

class ExecutionModel(str, Enum):
    """Trading execution models."""
    SIMULATED = "SIMULATED"
    PAPER = "PAPER"
    LIVE_OANDA = "LIVE_OANDA"

class RiskConfig(BaseModel):
    """Risk management configuration."""
    max_position_size: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Maximum position size as percentage of portfolio"
    )
    max_total_risk: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Maximum total portfolio risk"
    )
    stop_loss_pips: float = Field(
        default=50.0,
        gt=0.0,
        description="Default stop loss in pips"
    )
    take_profit_pips: float = Field(
        default=100.0,
        gt=0.0,
        description="Default take profit in pips"
    )
    max_drawdown_limit: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Maximum allowed drawdown"
    )

class TradingConfig(BaseModel):
    """Trading parameters configuration."""
    allowed_pairs: List[str] = Field(
        default=["EUR_USD", "GBP_USD", "USD_JPY"],
        description="Allowed trading pairs"
    )
    trading_hours: Dict[str, str] = Field(
        default={"start": "00:00", "end": "23:59"},
        description="Trading hours in UTC"
    )
    initial_balance: float = Field(
        default=10000.0,
        gt=0.0,
        description="Initial account balance"
    )
    max_positions: int = Field(
        default=3,
        gt=0,
        description="Maximum number of concurrent positions"
    )

class ExecutionConfig(BaseModel):
    """Execution configuration."""
    model: ExecutionModel = Field(
        default=ExecutionModel.SIMULATED,
        description="Execution model type"
    )
    delay_ms: int = Field(
        default=100,
        ge=0,
        description="Execution delay in milliseconds"
    )
    slippage_type: str = Field(
        default="fixed",
        description="Slippage model type"
    )
    slippage_pips: float = Field(
        default=0.5,
        ge=0.0,
        description="Fixed slippage in pips"
    )
    commission_type: str = Field(
        default="fixed",
        description="Commission model type"
    )
    commission_amount: float = Field(
        default=2.50,
        ge=0.0,
        description="Fixed commission amount"
    )

class APIConfig(BaseModel):
    """API credentials and settings."""
    oanda_account_id: SecretStr = Field(
        default="",
        description="OANDA account ID"
    )
    oanda_access_token: SecretStr = Field(
        default="",
        description="OANDA API access token"
    )
    oanda_environment: str = Field(
        default="practice",
        description="OANDA environment (practice/live)"
    )

class SystemConfig(BaseModel):
    """System configuration."""
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: str = Field(
        default="fxorcist.log",
        description="Log file path"
    )
    cache_dir: str = Field(
        default="artifacts",
        description="Cache directory path"
    )
    max_memory_usage: str = Field(
        default="4G",
        description="Maximum memory usage"
    )
    database_url: str = Field(
        default="sqlite:///fxorcist.db",
        description="Database connection URL"
    )

class DashboardConfig(BaseModel):
    """Dashboard configuration."""
    host: str = Field(
        default="localhost",
        description="Dashboard host"
    )
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Dashboard port"
    )
    secret_key: SecretStr = Field(
        default_factory=lambda: SecretStr(Fernet.generate_key().decode()),
        description="Dashboard secret key"
    )

class FXorcistConfig(BaseModel):
    """Main configuration schema."""
    version: str = Field(
        default="1.0.0",
        description="Configuration version"
    )
    environment: EnvironmentType = Field(
        default=EnvironmentType.DEVELOPMENT,
        description="Environment type"
    )
    risk: RiskConfig = Field(
        default_factory=RiskConfig,
        description="Risk management settings"
    )
    trading: TradingConfig = Field(
        default_factory=TradingConfig,
        description="Trading parameters"
    )
    execution: ExecutionConfig = Field(
        default_factory=ExecutionConfig,
        description="Execution settings"
    )
    api: APIConfig = Field(
        default_factory=APIConfig,
        description="API settings"
    )
    system: SystemConfig = Field(
        default_factory=SystemConfig,
        description="System settings"
    )
    dashboard: DashboardConfig = Field(
        default_factory=DashboardConfig,
        description="Dashboard settings"
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        json_encoders = {
            SecretStr: lambda v: v.get_secret_value() if v else None
        }

class ConfigManager:
    """Enhanced configuration manager with versioning and security."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_dir: Optional configuration directory path
        """
        self.config_dir = Path(config_dir or os.getenv('FXORCIST_CONFIG_DIR', 'config'))
        self.config_dir.mkdir(exist_ok=True)
        self.current_config: Optional[FXorcistConfig] = None
        self._encryption_key: Optional[bytes] = None
        self._setup_encryption()
    
    def _setup_encryption(self) -> None:
        """Setup encryption for sensitive data."""
        key_env = os.getenv('FXORCIST_ENCRYPTION_KEY')
        if key_env:
            self._encryption_key = key_env.encode()
        else:
            self._encryption_key = Fernet.generate_key()
            logger.warning("Generated new encryption key. Please set FXORCIST_ENCRYPTION_KEY environment variable.")
    
    def _encrypt_sensitive_data(self, config_dict: Dict) -> Dict:
        """Encrypt sensitive configuration data."""
        if not self._encryption_key:
            return config_dict
        
        f = Fernet(self._encryption_key)
        result = config_dict.copy()
        
        # Encrypt API credentials
        if 'api' in result:
            for key in ['oanda_access_token', 'oanda_account_id']:
                if result['api'].get(key):
                    result['api'][key] = f.encrypt(
                        result['api'][key].encode()
                    ).decode()
        
        return result
    
    def _decrypt_sensitive_data(self, config_dict: Dict) -> Dict:
        """Decrypt sensitive configuration data."""
        if not self._encryption_key:
            return config_dict
        
        f = Fernet(self._encryption_key)
        result = config_dict.copy()
        
        # Decrypt API credentials
        if 'api' in result:
            for key in ['oanda_access_token', 'oanda_account_id']:
                if result['api'].get(key):
                    try:
                        result['api'][key] = f.decrypt(
                            result['api'][key].encode()
                        ).decode()
                    except Exception as e:
                        logger.error(f"Failed to decrypt {key}: {e}")
                        result['api'][key] = ""
        
        return result
    
    def load_config(self, environment: Optional[str] = None) -> FXorcistConfig:
        """Load configuration for specified environment.
        
        Args:
            environment: Optional environment name (development/testing/production)
            
        Returns:
            Loaded configuration object
        """
        if environment is None:
            environment = os.getenv('FXORCIST_ENV', 'development')
        
        config_file = self.config_dir / f"{environment}.yaml"
        
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Decrypt sensitive data
                config_data = self._decrypt_sensitive_data(config_data)
                
                # Load environment variables
                config_data = self._load_environment_variables(config_data)
                
                # Validate and create config object
                self.current_config = FXorcistConfig(**config_data)
            else:
                logger.warning(f"Config file {config_file} not found, creating default")
                self.current_config = FXorcistConfig(environment=environment)
                self.save_config(self.current_config, environment)
            
            return self.current_config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def save_config(self, config: FXorcistConfig, environment: Optional[str] = None) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration object to save
            environment: Optional environment name
        """
        if environment is None:
            environment = config.environment.value
        
        config_file = self.config_dir / f"{environment}.yaml"
        
        try:
            # Convert to dict and encrypt sensitive data
            config_dict = config.dict()
            config_dict = self._encrypt_sensitive_data(config_dict)
            
            # Save to file
            with open(config_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def _load_environment_variables(self, config_data: Dict) -> Dict:
        """Load sensitive values from environment variables.
        
        Args:
            config_data: Configuration dictionary
            
        Returns:
            Updated configuration dictionary
        """
        env_mapping = {
            'OANDA_ACCOUNT_ID': ('api', 'oanda_account_id'),
            'OANDA_ACCESS_TOKEN': ('api', 'oanda_access_token'),
            'DATABASE_URL': ('system', 'database_url'),
            'LOG_LEVEL': ('system', 'log_level'),
        }
        
        result = config_data.copy()
        
        for env_key, config_path in env_mapping.items():
            env_value = os.getenv(env_key)
            if env_value:
                # Create nested dict path if it doesn't exist
                current = result
                for part in config_path[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[config_path[-1]] = env_value
        
        return result
    
    def get_version_history(self) -> List[Dict]:
        """Get configuration version history.
        
        Returns:
            List of historical configuration versions
        """
        history_file = self.config_dir / "version_history.json"
        if not history_file.exists():
            return []
        
        with open(history_file, 'r') as f:
            return json.load(f)
    
    def save_version(self, config: FXorcistConfig, description: str) -> None:
        """Save configuration version to history.
        
        Args:
            config: Configuration object
            description: Version description
        """
        history_file = self.config_dir / "version_history.json"
        history = self.get_version_history()
        
        version_entry = {
            "version": config.version,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "environment": config.environment.value
        }
        
        history.append(version_entry)
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

# Global configuration instance
config_manager = ConfigManager()