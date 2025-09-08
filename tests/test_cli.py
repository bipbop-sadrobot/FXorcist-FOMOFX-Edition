"""
Tests for FXorcist CLI commands and functionality.
"""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from typer.testing import CliRunner
import pandas as pd

from fxorcist.cli import app
from fxorcist.utils.config import load_config

# Test fixtures
@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    return CliRunner()

@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return {
        "base_dir": "test_data",
        "models_dir": "test_models",
        "logs_dir": "test_logs",
        "seed": 42,
        "batch_size": 500,
    }

@pytest.fixture
def config_file(tmp_path, mock_config):
    """Create a temporary config file."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(mock_config))
    return config_path

# Test CLI basics
def test_version(runner):
    """Test version display."""
    with patch("fxorcist.__version__", "1.0.0"):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output

def test_help(runner):
    """Test help output."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "FXorcist" in result.output
    for cmd in ["prepare", "backtest", "optimize", "serve"]:
        assert cmd in result.output

# Test prepare command
def test_prepare_command(runner, config_file):
    """Test data preparation command."""
    with patch("fxorcist.data.loader.load_symbol") as mock_load:
        mock_load.return_value = pd.DataFrame({
            "open": [1.0, 1.1],
            "close": [1.1, 1.2]
        }, index=pd.date_range("2025-01-01", periods=2))
        
        result = runner.invoke(app, [
            "--config", str(config_file),
            "prepare",
            "EURUSD",
            "--start-date", "2025-01-01"
        ])
        
        assert result.exit_code == 0
        mock_load.assert_called_once()
        assert "Loaded" in result.output

def test_prepare_error(runner, config_file):
    """Test data preparation error handling."""
    with patch("fxorcist.data.loader.load_symbol") as mock_load:
        mock_load.side_effect = Exception("Data error")
        
        result = runner.invoke(app, [
            "--config", str(config_file),
            "prepare",
            "INVALID"
        ])
        
        assert result.exit_code == 1
        assert "Error" in result.output

# Test backtest command
def test_backtest_command(runner, config_file):
    """Test backtest execution."""
    with patch("fxorcist.backtest.engine.run_backtest") as mock_run:
        mock_run.return_value = {
            "sharpe": 1.5,
            "max_drawdown": -0.1,
            "total_return": 0.25
        }
        
        result = runner.invoke(app, [
            "--config", str(config_file),
            "backtest",
            "rsi",
            "--symbol", "EURUSD"
        ])
        
        assert result.exit_code == 0
        mock_run.assert_called_once()
        assert "Backtest Results" in result.output
        assert "1.5" in result.output  # Sharpe ratio

def test_backtest_with_report(runner, config_file):
    """Test backtest with report generation."""
    with patch("fxorcist.backtest.engine.run_backtest") as mock_run:
        mock_results = {
            "sharpe": 1.5,
            "report": MagicMock()
        }
        mock_run.return_value = mock_results
        
        result = runner.invoke(app, [
            "--config", str(config_file),
            "backtest",
            "rsi",
            "--symbol", "EURUSD",
            "--report"
        ])
        
        assert result.exit_code == 0
        mock_results["report"].save.assert_called_once()
        assert "Report saved" in result.output

# Test optimize command
def test_optimize_command(runner, config_file):
    """Test optimization execution."""
    with patch("fxorcist.ml.optuna_runner.run_optuna") as mock_run:
        mock_run.return_value = {
            "best_params": {
                "window": 14,
                "threshold": 70
            }
        }
        
        result = runner.invoke(app, [
            "--config", str(config_file),
            "optimize",
            "rsi",
            "--symbol", "EURUSD",
            "--trials", "10"
        ])
        
        assert result.exit_code == 0
        mock_run.assert_called_once()
        assert "Best Parameters" in result.output
        assert "window" in result.output

def test_optimize_with_mlflow(runner, config_file):
    """Test optimization with MLflow tracking."""
    with patch("fxorcist.ml.optuna_runner.run_optuna") as mock_run:
        result = runner.invoke(app, [
            "--config", str(config_file),
            "optimize",
            "rsi",
            "--symbol", "EURUSD",
            "--mlflow"
        ])
        
        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            mock_run.call_args[0][0],  # df
            n_trials=30,  # default
            seed=42,  # from mock config
            use_mlflow=True,
            progress=mock_run.call_args[1]["progress"]
        )

# Test serve command
def test_serve_command(runner, config_file):
    """Test dashboard server startup."""
    with patch("fxorcist.dashboard.app.run_dashboard") as mock_run:
        result = runner.invoke(app, [
            "--config", str(config_file),
            "serve",
            "--port", "8888",
            "--reload"
        ])
        
        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            host="127.0.0.1",
            port=8888,
            reload=True,
            config=mock_run.call_args[1]["config"]
        )

def test_serve_missing_dependencies(runner, config_file):
    """Test dashboard server with missing dependencies."""
    with patch("fxorcist.dashboard.app.run_dashboard", side_effect=ImportError):
        result = runner.invoke(app, [
            "--config", str(config_file),
            "serve"
        ])
        
        assert result.exit_code == 1
        assert "dependencies not installed" in result.output

# Test JSON output
def test_json_output(runner, config_file):
    """Test JSON output format."""
    with patch("fxorcist.backtest.engine.run_backtest") as mock_run:
        mock_run.return_value = {"sharpe": 1.5}
        
        result = runner.invoke(app, [
            "--config", str(config_file),
            "--json",
            "backtest",
            "rsi",
            "--symbol", "EURUSD"
        ])
        
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["sharpe"] == 1.5

# Test config loading
def test_invalid_config(runner, tmp_path):
    """Test handling of invalid config file."""
    bad_config = tmp_path / "bad_config.yaml"
    bad_config.write_text("invalid: yaml: content")
    
    result = runner.invoke(app, [
        "--config", str(bad_config),
        "prepare",
        "EURUSD"
    ])
    
    assert result.exit_code == 1
    assert "Error" in result.output