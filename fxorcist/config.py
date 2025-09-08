from typing import Optional, Dict, Any, Union
from pathlib import Path
import yaml
import os
from functools import lru_cache
from pydantic import BaseModel, Field, ValidationError, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

class BaseConfigModel(BaseModel):
    """Enhanced base configuration model with advanced validation."""
    model_config = ConfigDict(
        extra='forbid',  # Prevent unexpected keys
        validate_assignment=True,  # Validate on attribute modification
        str_strip_whitespace=True  # Automatically strip whitespace
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseConfigModel':
        """Create instance from dictionary with robust error handling."""
        try:
            return cls.model_validate(data)
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}") from e

class DataConfig(BaseConfigModel):
    default_symbol: str = Field(default="EURUSD", pattern=r'^[A-Z]{6}$')
    storage: str = Field(default="parquet", pattern=r'^(parquet|csv|timescale)$')
    parquet_dir: Path = Field(default_factory=lambda: Path("data/cleaned"))

    @field_validator('parquet_dir', mode='before')
    @classmethod
    def ensure_parquet_dir(cls, v: Union[str, Path]) -> Path:
        """Ensure parquet directory exists and is a valid path."""
        path = Path(v).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

class BacktestConfig(BaseConfigModel):
    commission_pct: float = Field(default=0.00002, ge=0.0, le=0.01)
    slippage_model: str = Field(default="simple", pattern=r'^(simple|impact|historical)$')
    latency_ms: int = Field(default=100, ge=0, le=5000)

class ServerConfig(BaseConfigModel):
    port: int = Field(default=8080, ge=1024, le=65535)
    host: str = Field(default="0.0.0.0")

class OptimConfig(BaseConfigModel):
    engine: str = Field(default="optuna", pattern=r'^(optuna|grid|ga)$')
    n_trials: int = Field(default=200, ge=1, le=10000)

class Settings(BaseModel):
    """Comprehensive configuration settings with advanced features."""
    data: DataConfig = Field(default_factory=DataConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)

    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True
    )

    @classmethod
    @lru_cache(maxsize=1)
    def load(cls, config_path: Optional[str] = None) -> 'Settings':
        """
        Load configuration with multiple fallback mechanisms.
        
        Priority:
        1. Explicitly provided config file
        2. Environment variable pointing to config
        3. Default config.yaml in project root
        4. Default settings
        """
        # Try paths in order of priority
        possible_paths = [
            config_path,
            os.environ.get('FXORCIST_CONFIG'),
            Path.cwd() / 'config.yaml',
            Path.home() / '.fxorcist' / 'config.yaml'
        ]

        for path in possible_paths:
            if path and Path(path).is_file():
                try:
                    with open(path, 'r') as f:
                        config_dict = yaml.safe_load(f) or {}
                    return cls.model_validate(config_dict)
                except (IOError, ValidationError) as e:
                    print(f"Error loading config from {path}: {e}")

        # Fallback to default settings
        return cls()

    def save(self, path: Union[str, Path] = 'config.yaml'):
        """Save current configuration to a YAML file."""
        config_dict = self.model_dump()
        with open(path, 'w') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)

def get_config(config_path: Optional[str] = None) -> Settings:
    """Convenience function to get configuration."""
    return Settings.load(config_path)