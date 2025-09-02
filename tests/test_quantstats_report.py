"""
Integration tests for QuantStats HTML report generation functionality.
Tests both QuantStatsTearsheet and QuantStatsPortfolio components.
"""

import pytest
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from dashboard.components import QuantStatsTearsheet, QuantStatsPortfolio
from dashboard.components import ComponentConfig

@pytest.fixture
def sample_returns():
    """Generate sample returns data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
    return {'returns': returns}

@pytest.fixture
def component_config():
    """Create component configuration for testing."""
    return ComponentConfig(
        title="Test Component",
        description="Test component for quantstats analysis"
    )

def test_tearsheet_report_generation(sample_returns, component_config):
    """Test HTML report generation in QuantStatsTearsheet component."""
    # Initialize component
    tearsheet = QuantStatsTearsheet(component_config)
    
    # Generate report
    tearsheet.generate_html_report(sample_returns)
    
    # Verify report was generated
    assert tearsheet.report_path is not None
    assert os.path.exists(tearsheet.report_path)
    assert tearsheet.report_path.endswith('.html')
    
    # Verify report content
    with open(tearsheet.report_path, 'r') as f:
        content = f.read()
        assert 'Portfolio Analysis Report' in content
        assert 'Sharpe Ratio' in content
        assert 'Drawdown' in content
    
    # Cleanup
    os.unlink(tearsheet.report_path)

def test_portfolio_report_generation(sample_returns, component_config):
    """Test HTML report generation in QuantStatsPortfolio component."""
    # Initialize component
    portfolio = QuantStatsPortfolio(component_config)
    
    # Generate report
    portfolio.generate_html_report(sample_returns)
    
    # Verify report was generated
    assert portfolio.report_path is not None
    assert os.path.exists(portfolio.report_path)
    assert portfolio.report_path.endswith('.html')
    
    # Verify report content
    with open(portfolio.report_path, 'r') as f:
        content = f.read()
        assert 'Portfolio Analysis Report' in content
        assert 'Sharpe Ratio' in content
        assert 'Drawdown' in content
    
    # Cleanup
    os.unlink(portfolio.report_path)

def test_report_with_benchmark(sample_returns, component_config):
    """Test HTML report generation with benchmark comparison."""
    # Initialize component
    tearsheet = QuantStatsTearsheet(component_config)
    tearsheet.benchmark_enabled = True
    
    # Generate report
    tearsheet.generate_html_report(sample_returns)
    
    # Verify report was generated with benchmark
    assert tearsheet.report_path is not None
    with open(tearsheet.report_path, 'r') as f:
        content = f.read()
        assert 'Benchmark' in content
        assert 'SPY' in content
    
    # Cleanup
    os.unlink(tearsheet.report_path)

def test_report_without_benchmark(sample_returns, component_config):
    """Test HTML report generation without benchmark comparison."""
    # Initialize component
    portfolio = QuantStatsPortfolio(component_config)
    portfolio.benchmark_enabled = False
    
    # Generate report
    portfolio.generate_html_report(sample_returns)
    
    # Verify report was generated without benchmark
    assert portfolio.report_path is not None
    with open(portfolio.report_path, 'r') as f:
        content = f.read()
        assert 'SPY' not in content
    
    # Cleanup
    os.unlink(portfolio.report_path)

def test_report_error_handling(component_config):
    """Test error handling in report generation."""
    # Initialize component
    tearsheet = QuantStatsTearsheet(component_config)
    
    # Test with invalid data
    invalid_data = {'returns': pd.Series([])}
    tearsheet.generate_html_report(invalid_data)
    assert tearsheet.report_path is None
    
    # Test with missing data
    missing_data = {}
    tearsheet.generate_html_report(missing_data)
    assert tearsheet.report_path is None

def test_report_cleanup(sample_returns, component_config):
    """Test proper cleanup of report files."""
    # Initialize component
    portfolio = QuantStatsPortfolio(component_config)
    
    # Generate first report
    portfolio.generate_html_report(sample_returns)
    first_report_path = portfolio.report_path
    assert os.path.exists(first_report_path)
    
    # Generate second report
    portfolio.generate_html_report(sample_returns)
    second_report_path = portfolio.report_path
    
    # Verify first report was cleaned up
    assert not os.path.exists(first_report_path)
    assert os.path.exists(second_report_path)
    
    # Cleanup
    os.unlink(second_report_path)

def test_component_state_reset(sample_returns, component_config):
    """Test proper state reset when updating data."""
    # Initialize component
    tearsheet = QuantStatsTearsheet(component_config)
    
    # Generate report
    tearsheet.generate_html_report(sample_returns)
    assert tearsheet.report_path is not None
    
    # Update component
    tearsheet.update({'returns': pd.Series([])})
    assert tearsheet.report_path is None