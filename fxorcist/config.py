from typing import Optional
from pathlib import Path
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator

class DataConfig(BaseSettings):
    default_symbol: str = "EURUSD"
    storage: str = "parquet"  # "parquet", "csv", "timescale"
    parquet_dir: str = "data/cleaned"
    model_config = SettingsConfigDict(env_prefix="DATA_")

class BacktestConfig(BaseSettings):
    commission_pct: float = Field(0.00002, ge=0.0, le=0.01)
    slippage_model: str = "simple"  # "simple", "impact", "historical"
    latency_ms: int = Field(100, ge=0, le=5000)
    model_config = SettingsConfigDict(env_prefix="BACKTEST_")

class ServerConfig(BaseSettings):
    port: int = Field(8080, ge=1024, le=65535)
    model_config = SettingsConfigDict(env_prefix="SERVER_")

class OptimConfig(BaseSettings):
    engine: str = "optuna"  # "optuna", "grid", "ga"
    n_trials: int = Field(200, ge=1, le=10000)
    model_config = SettingsConfigDict(env_prefix="OPTIM_")

class Settings(BaseSettings):
    data: DataConfig = DataConfig()
    backtest: BacktestConfig = BacktestConfig()
    server: ServerConfig = ServerConfig()
    optim: OptimConfig = OptimConfig()
    model_config = SettingsConfigDict(env_nested_delimiter="__")

    @classmethod
    def from_yaml(cls, path: str) -> "Settings":
        """Load settings from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}
        return cls.model_validate(config_dict)

    @model_validator(mode="after")
    def validate_storage_path(self) -> "Settings":
        """Ensure parquet_dir exists if storage is parquet."""
        if self.data.storage == "parquet":
            parquet_path = Path(self.data.parquet_dir)
            if not parquet_path.exists():
                parquet_path.mkdir(parents=True, exist_ok=True)
        return self

def load(config_path: str = "config.yaml") -> Settings:
    """Load and validate config."""
    return Settings.from_yaml(config_path)