"""
Tests for FXorcist CLI commands and functionality.
"""

import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from click.testing import CliRunner

from fxorcist_cli import cli, Config

# Test fixtures
@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    return CliRunner()

@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return {
        "data_dir": "test_data",
        "models_dir": "test_models",
        "logs_dir": "test_logs",
        "dashboard_port": 9999,
        "auto_backup": True,
        "quality_threshold": 0.8,
        "batch_size": 500
    }

@pytest.fixture
def config_file(tmp_path, mock_config):
    """Create a temporary config file."""
    config_path = tmp_path / "config" / "cli_config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(mock_config))
    return config_path

# Test Config class
def test_config_load_defaults():
    """Test loading default configuration."""
    config = Config()
    assert "data_dir" in config.config
    assert "models_dir" in config.config
    assert "dashboard_port" in config.config

def test_config_save(tmp_path):
    """Test saving configuration."""
    config = Config()
    config.config_file = tmp_path / "config" / "cli_config.json"
    config.save_config()
    assert config.config_file.exists()
    saved_config = json.loads(config.config_file.read_text())
    assert saved_config == config.config

# Test CLI commands
def test_cli_help(runner):
    """Test CLI help output."""
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'FXorcist AI Dashboard' in result.output

def test_data_integrate(runner):
    """Test data integration command."""
    with patch('fxorcist_cli.OptimizedDataIntegrator') as mock_integrator:
        mock_instance = MagicMock()
        mock_instance.process_optimized_data.return_value = {"processed": 10}
        mock_integrator.return_value = mock_instance
        
        result = runner.invoke(cli, ['data', 'integrate'])
        assert result.exit_code == 0
        mock_instance.process_optimized_data.assert_called_once()

def test_data_validate(runner):
    """Test data validation command."""
    with patch('fxorcist_cli.logger') as mock_logger:
        test_file = "test.csv"
        result = runner.invoke(cli, ['data', 'validate', test_file])
        assert result.exit_code == 1  # Should fail as file doesn't exist
        mock_logger.error.assert_called()

def test_train_start(runner):
    """Test training start command."""
    with patch('fxorcist_cli.EnhancedTrainingPipeline') as mock_pipeline:
        mock_instance = MagicMock()
        mock_pipeline.return_value = mock_instance
        
        result = runner.invoke(cli, ['train', 'start', '--quick'])
        assert result.exit_code == 0
        mock_pipeline.assert_called_once()

def test_dashboard_start(runner):
    """Test dashboard start command."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = runner.invoke(cli, ['dashboard', 'start', '--port', '8888'])
        assert result.exit_code == 0
        mock_run.assert_called_once()
        assert '--server.port 8888' in mock_run.call_args[0][0]

def test_memory_stats(runner):
    """Test memory stats command."""
    with patch('fxorcist_cli.MemoryManager') as mock_manager:
        mock_instance = MagicMock()
        mock_instance.get_statistics.return_value = {"records": 100, "size": "1MB"}
        mock_manager.return_value = mock_instance
        
        result = runner.invoke(cli, ['memory', 'stats'])
        assert result.exit_code == 0
        mock_instance.get_statistics.assert_called_once()

def test_memory_clear(runner):
    """Test memory clear command."""
    with patch('fxorcist_cli.MemoryManager') as mock_manager:
        mock_instance = MagicMock()
        mock_manager.return_value = mock_instance
        
        result = runner.invoke(cli, ['memory', 'clear'])
        assert result.exit_code == 0
        mock_instance.clear_cache.assert_called_once()

def test_config_view(runner):
    """Test config view command."""
    result = runner.invoke(cli, ['config', 'view'])
    assert result.exit_code == 0
    assert 'Current Configuration' in result.output

def test_config_set(runner):
    """Test config set command."""
    with patch('fxorcist_cli.Config.save_config') as mock_save:
        inputs = ['test_key', 'test_value']
        result = runner.invoke(cli, ['config', 'set'], input='\n'.join(inputs))
        assert result.exit_code == 0
        mock_save.assert_called_once()

def test_config_reset(runner):
    """Test config reset command."""
    with patch('fxorcist_cli.Config.save_config') as mock_save:
        result = runner.invoke(cli, ['config', 'reset'], input='y\n')
        assert result.exit_code == 0
        mock_save.assert_called_once()

# Test error handling
def test_data_integrate_error(runner):
    """Test error handling in data integration."""
    with patch('fxorcist_cli.OptimizedDataIntegrator') as mock_integrator:
        mock_instance = MagicMock()
        mock_instance.process_optimized_data.side_effect = Exception("Test error")
        mock_integrator.return_value = mock_instance
        
        result = runner.invoke(cli, ['data', 'integrate'])
        assert result.exit_code == 1
        assert "Test error" in str(result.output)

def test_dashboard_start_error(runner):
    """Test error handling in dashboard start."""
    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, "test")
        result = runner.invoke(cli, ['dashboard', 'start'])
        assert result.exit_code == 1
        assert "failed to start" in str(result.output).lower()

# Test type conversion in config
def test_config_set_type_conversion(runner):
    """Test type conversion in config set command."""
    test_cases = [
        ('int_key', '42', int),
        ('float_key', '3.14', float),
        ('bool_key', 'true', bool),
        ('str_key', 'hello', str)
    ]
    
    for key, value, expected_type in test_cases:
        with patch('fxorcist_cli.Config.save_config'):
            inputs = [key, value]
            result = runner.invoke(cli, ['config', 'set'], input='\n'.join(inputs))
            assert result.exit_code == 0

# Test debug mode
def test_debug_mode(runner):
    """Test debug mode enabling."""
    with patch('logging.getLogger') as mock_logger:
        result = runner.invoke(cli, ['--debug', 'config', 'view'])
        assert result.exit_code == 0
        mock_logger.assert_called_with("fxorcist")