import os
import tempfile
from pathlib import Path
import pytest
from pydantic import ValidationError

from fxorcist.config import Settings, get_config, DataConfig, BacktestConfig

def test_default_config():
    """Test default configuration settings."""
    config = Settings()
    assert config.data.default_symbol == "EURUSD"
    assert config.backtest.commission_pct == 0.00002
    assert config.server.port == 8080

def test_config_validation():
    """Test configuration validation."""
    with pytest.raises(ValidationError):
        BacktestConfig(commission_pct=2.0)  # Out of range

    with pytest.raises(ValidationError):
        DataConfig(default_symbol="INVALID")  # Invalid symbol format

def test_config_loading_with_yaml():
    """Test loading configuration from YAML file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_config:
        temp_config.write("""
        data:
          default_symbol: GBPUSD
          storage: csv
        backtest:
          commission_pct: 0.00003
        server:
          port: 9090
        """)
        temp_config.close()

    try:
        config = Settings.load(temp_config.name)
        assert config.data.default_symbol == "GBPUSD"
        assert config.data.storage == "csv"
        assert config.backtest.commission_pct == 0.00003
        assert config.server.port == 9090
    finally:
        os.unlink(temp_config.name)

def test_env_var_config_override(monkeypatch):
    """Test environment variable configuration override."""
    monkeypatch.setenv('FXORCIST_CONFIG', 'non_existent_config.yaml')
    monkeypatch.setenv('DATA__DEFAULT_SYMBOL', 'AUDUSD')
    monkeypatch.setenv('BACKTEST__COMMISSION_PCT', '0.0001')

    config = get_config()
    assert config.data.default_symbol == 'AUDUSD'
    assert config.backtest.commission_pct == 0.0001

def test_config_save_and_load():
    """Test saving and reloading configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'test_config.yaml'
        
        original_config = Settings()
        original_config.save(config_path)

        loaded_config = Settings.load(config_path)
        assert loaded_config.model_dump() == original_config.model_dump()

def test_parquet_dir_creation():
    """Test automatic parquet directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_path = Path(tmpdir) / 'custom_parquet'
        config = DataConfig(parquet_dir=custom_path)
        
        assert config.parquet_dir.exists()
        assert config.parquet_dir == custom_path