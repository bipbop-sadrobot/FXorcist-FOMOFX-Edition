import tempfile
import os
import pytest
from fxorcist.config import Settings, load

def test_load_valid_config():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
data:
  default_symbol: GBPUSD
backtest:
  commission_pct: 0.00003
server:
  port: 9090
""")
        temp_path = f.name

    try:
        settings = load(temp_path)
        assert settings.data.default_symbol == "GBPUSD"
        assert settings.backtest.commission_pct == 0.00003
        assert settings.server.port == 9090
    finally:
        os.unlink(temp_path)

def test_invalid_commission():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
backtest:
  commission_pct: 1.5  # too high
""")
        temp_path = f.name

    try:
        with pytest.raises(Exception) as excinfo:
            load(temp_path)
        assert "commission_pct" in str(excinfo.value)
    finally:
        os.unlink(temp_path)

def test_env_var_override(monkeypatch):
    # Test environment variable override
    monkeypatch.setenv("DATA__DEFAULT_SYMBOL", "AUDUSD")
    monkeypatch.setenv("BACKTEST__COMMISSION_PCT", "0.0001")
    
    settings = Settings()
    assert settings.data.default_symbol == "AUDUSD"
    assert settings.backtest.commission_pct == 0.0001