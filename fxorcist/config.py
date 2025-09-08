"""
Configuration management for FXorcist using Pydantic.
"""
from pydantic import BaseModel, Field
from typing import Optional
import yaml

class DataConfig(BaseModel):
    default_symbol: str = "EURUSD"
    storage: str = "parquet"
    parquet_dir: str = "data/cleaned"

class BacktestConfig(BaseModel):
    commission_pct: float = 0.00002
    slippage_model: str = "simple"
    latency_ms: int = 100
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class OptimConfig(BaseModel):
    engine: str = "optuna"
    n_trials: int = 100

class AppConfig(BaseModel):
    data: DataConfig = DataConfig()
    backtest: BacktestConfig = BacktestConfig()
    optim: OptimConfig = OptimConfig()

def load_config(path: str = "config.yaml") -> AppConfig:
    """
    Load configuration from a YAML file.
    
    Args:
        path: Path to the configuration YAML file
    
    Returns:
        Validated AppConfig instance
    """
    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return AppConfig(**raw)
    except FileNotFoundError:
        # Return default configuration if file not found
        return AppConfig()