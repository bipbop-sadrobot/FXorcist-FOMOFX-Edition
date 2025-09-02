"""
Performance metrics visualization component for the Forex AI dashboard.
Handles PnL analysis, drawdowns, Sharpe ratios, and risk metrics.
"""

from typing import List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta

from . import (
    PerformanceComponent,
    ComponentConfig,
    cache_data,
    calculate_rolling_metrics,
    calculate_drawdowns
)

class PerformanceMetrics(PerformanceComponent):
    """Component for visualizing trading performance metrics."""
    
    def __init__(self, config: ComponentConfig):
        """Initialize performance metrics component."""
        super().__init__(config)
        self.window_sizes = {
            "1D": 24,
            "1W": 24 * 5,
            "1M": 24 * 20,
            "3M": 24 * 60,
            "6M": 24 * 120,
            "1Y": 24 * 252
        }
        self.selected_window = "1M"
        self.show_drawdowns = True
        self.show_distributions = True
    
    @st.cache_data(ttl=timedelta(seconds=300))
    def calculate_performance_metrics(self, data: dict) -> dict[str, pd.Series]:
        """Calculate comprehensive performance metrics."""
        if not data or 'returns' not in data:
            return {}
            
        returns = data['returns']
        window = self.window_sizes[self.selected_window]
        
        # Calculate rolling metrics
        metrics = calculate_rolling_metrics(returns, window)
        
        # Add drawdown series
        metrics['drawdowns'] = calculate_drawdowns(returns)
        
        # Calculate cumulative PnL
        metrics['cumulative_pnl'] = (1 + returns).cumprod() - 1
        
        # Calculate additional risk metrics
        metrics['rolling_var'] = returns.rolling(window).quantile(0.05)
        metrics['rolling_sortino'] = (
            returns.rolling(window).mean() / 
            returns[returns < 0].rolling(window).std() * 
            np.sqrt(252)
        )
        
        return metrics
    
    @st.cache_data(ttl=timedelta(seconds=300))
    def analyze_drawdowns(self, returns: pd.Series) -> dict[str, Any]:
        """Perform detailed drawdown analysis."""
        drawdowns = calculate_drawdowns(returns)
        
        # Calculate drawdown statistics
        max_drawdown = drawdowns.min()
        avg_drawdown = drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0.0
        drawdown_duration = pd.Timedelta(days=0)  # Placeholder, would need proper calculation
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'avg_duration': drawdown_duration,
            'drawdown_series': drawdowns
        }
    
    def create_figure(self, data: dict) -> go.Figure:
        """Create performance visualization figure."""
        if not data or 'returns' not in data:
            return None
            
        # Calculate metrics
        metrics = self.calculate_performance_metrics(data)
        
        # Create subplot figure
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                'Cumulative PnL',
                'Rolling Sharpe Ratio',
                'Drawdowns'
            ),
            vertical_spacing=0.1
        )
        
        # Plot cumulative PnL
        fig.add_trace(
            go.Scatter(
                x=metrics['cumulative_pnl'].index,
                y=metrics['cumulative_pnl'].values * 100,
                name='Cumulative PnL (%)',
                line=dict(color='green', width=2)
            ),
            row=1,
            col=1
        )
        
        # Plot rolling Sharpe ratio
        fig.add_trace(
            go.Scatter(
                x=metrics['rolling_sharpe'].index,
                y=metrics['rolling_sharpe'].values,
                name='Rolling Sharpe',
                line=dict(color='blue', width=2)
            ),
            row=2,
            col=1
        )
        
        # Add Sharpe ratio threshold lines
        fig.add_hline(
            y=2,
            line_dash="dash",
            line_color="green",
            row=2,
            col=1
        )
        fig.add_hline(
            y=1,
            line_dash="dash",
            line_color="yellow",
            row=2,
            col=1
        )
        
        # Plot drawdowns
        fig.add_trace(
            go.Scatter(
                x=metrics['drawdowns'].index,
                y=metrics['drawdowns'].values * 100,
                name='Drawdowns (%)',
                line=dict(color='red', width=2),
                fill='tozeroy'
            ),
            row=3,
            col=1
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text="Trading Performance Analysis",
            hovermode='x unified'
        )
        
        return fig
    
    def create_distribution_figure(self, data: dict) -> go.Figure:
        """Create return distribution visualization."""
        if not data or 'returns' not in data:
            return None
            
        returns = data['returns']
        
        fig = go.Figure()
        
        # Add returns histogram
        fig.add_trace(go.Histogram(
            x=returns.values * 100,
            name='Returns Distribution',
            nbinsx=50,
            histnorm='probability'
        ))
        
        # Add normal distribution for comparison
        x = np.linspace(returns.min() * 100, returns.max() * 100, 100)
        fig.add_trace(go.Scatter(
            x=x,
            y=np.exp(-(x - returns.mean() * 100)**2 / (2 * returns.std() * 100)**2) / 
              (returns.std() * 100 * np.sqrt(2 * np.pi)),
            name='Normal Distribution',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Returns Distribution Analysis',
            xaxis_title='Returns (%)',
            yaxis_title='Probability',
            height=400,
            showlegend=True
        )
        
        return fig
    
    def render(self) -> None:
        """Render performance metrics component."""
        st.subheader(self.config.title)
        
        # Controls
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            self.selected_window = st.selectbox(
                "Analysis Window",
                list(self.window_sizes.keys()),
                index=list(self.window_sizes.keys()).index(self.selected_window)
            )
        
        with col2:
            self.show_drawdowns = st.checkbox(
                "Show Drawdown Analysis",
                value=self.show_drawdowns
            )
        
        with col3:
            self.show_distributions = st.checkbox(
                "Show Return Distributions",
                value=self.show_distributions
            )
        
        # Main performance plots
        fig = self.create_figure(self._cache.get('data'))
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown analysis
        if self.show_drawdowns and 'returns' in self._cache.get('data', {}):
            drawdown_stats = self.analyze_drawdowns(
                self._cache['data']['returns']
            )
            
            st.subheader("Drawdown Analysis")
            cols = st.columns(3)
            cols[0].metric(
                "Maximum Drawdown",
                f"{drawdown_stats['max_drawdown']*100:.2f}%"
            )
            cols[1].metric(
                "Average Drawdown",
                f"{drawdown_stats['avg_drawdown']*100:.2f}%"
            )
            cols[2].metric(
                "Avg Recovery Time",
                f"{drawdown_stats['avg_duration'].days:.1f} days"
            )
        
        # Distribution analysis
        if self.show_distributions:
            fig_dist = self.create_distribution_figure(self._cache.get('data'))
            if fig_dist:
                st.plotly_chart(fig_dist, use_container_width=True)
    
    def update(self, data: dict) -> None:
        """Update component with new data."""
        self._cache['data'] = data
        self.clear_cache()  # Clear cached calculations