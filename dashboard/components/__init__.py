"""
Dashboard components module.
Contains modular components for the Forex AI dashboard visualization and analysis.
Provides base classes and interfaces for consistent component implementation.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Callable
from dataclasses import dataclass
from functools import lru_cache
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta

from forex_ai_dashboard.pipeline.evaluation_metrics import EvaluationMetrics

@dataclass
class ComponentConfig:
    """Configuration for dashboard components."""
    title: str
    description: Optional[str] = None
    height: int = 400
    width: Optional[int] = None
    cache_ttl: int = 300  # Cache timeout in seconds

class DashboardComponent(ABC):
    """Base class for all dashboard components."""
    
    def __init__(self, config: ComponentConfig):
        """Initialize component with configuration."""
        self.config = config
        self._cache = {}
    
    @abstractmethod
    def render(self) -> None:
        """Render the component in the dashboard."""
        pass
    
    @abstractmethod
    def update(self, data: Any) -> None:
        """Update component with new data."""
        pass
    
    def clear_cache(self) -> None:
        """Clear component cache."""
        self._cache.clear()

class VisualizationComponent(DashboardComponent):
    """Base class for visualization components."""
    
    @abstractmethod
    def create_figure(self, data: Any) -> go.Figure:
        """Create plotly figure for visualization."""
        pass
    
    def render(self) -> None:
        """Render visualization in dashboard."""
        fig = self.create_figure(self._cache.get('data'))
        if fig:
            st.plotly_chart(fig, use_container_width=True)

class MetricsComponent(DashboardComponent):
    """Base class for metrics display components."""
    
    @abstractmethod
    def calculate_metrics(self, data: Any) -> dict[str, float]:
        """Calculate metrics from data."""
        pass
    
    def render(self) -> None:
        """Render metrics in dashboard."""
        metrics = self.calculate_metrics(self._cache.get('data'))
        if metrics:
            cols = st.columns(len(metrics))
            for col, (name, value) in zip(cols, metrics.items()):
                col.metric(name, f"{value:.4f}")

class PerformanceComponent(VisualizationComponent):
    """Base class for performance analysis components."""
    
    @abstractmethod
    def calculate_performance_metrics(self, data: Any) -> dict[str, pd.Series]:
        """Calculate performance metrics over time."""
        pass
    
    @abstractmethod
    def analyze_drawdowns(self, returns: pd.Series) -> dict[str, Any]:
        """Analyze drawdowns from return series."""
        pass

class SystemStatusComponent(DashboardComponent):
    """Base class for system status components."""
    
    @abstractmethod
    def check_status(self) -> dict[str, bool]:
        """Check system component status."""
        pass
    
    def render(self) -> None:
        """Render system status indicators."""
        status = self.check_status()
        cols = st.columns(len(status))
        for col, (name, healthy) in zip(cols, status.items()):
            col.metric(
                name,
                "Healthy" if healthy else "Error",
                "✓" if healthy else "⚠️"
            )

def cache_data(ttl_seconds: int = 300) -> Callable:
    """Decorator for caching component data."""
    def decorator(func: Callable) -> Callable:
        @lru_cache(maxsize=1)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Common utility functions
def calculate_rolling_metrics(returns: pd.Series, window: int = 252) -> dict[str, pd.Series]:
    """Calculate rolling financial metrics."""
    return {
        'rolling_sharpe': returns.rolling(window, min_periods=1).mean() / returns.rolling(window, min_periods=1).std() * np.sqrt(252),
        'rolling_volatility': returns.rolling(window, min_periods=1).std() * np.sqrt(252),
        'rolling_returns': (1 + returns).rolling(window, min_periods=1).prod() - 1
    }

def calculate_drawdowns(returns: pd.Series) -> pd.Series:
    """Calculate drawdown series from returns."""
    wealth_index = (1 + returns).cumprod()
    previous_peaks = wealth_index.expanding(min_periods=1).max()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return drawdowns