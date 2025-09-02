"""
QuantStats Comprehensive Tearsheet Component
Provides full quantstats tearsheet functionality with all core metrics and visualizations.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from . import (
    PerformanceComponent,
    ComponentConfig,
    cache_data
)
from utils.quantstats_analytics import QuantStatsAnalytics

class QuantStatsTearsheet(PerformanceComponent):
    """Component for comprehensive quantstats tearsheet analysis."""

    def __init__(self, config: ComponentConfig):
        """Initialize quantstats tearsheet component."""
        super().__init__(config)
        self.analytics = QuantStatsAnalytics()

        # Component state
        self.benchmark_enabled = True
        self.analysis_period = "1Y"
        self.report_path = None  # Store path to generated report

    @st.cache_data(ttl=timedelta(seconds=300))
    def generate_tearsheet(self, returns_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive quantstats tearsheet."""
        if not returns_data or 'returns' not in returns_data:
            return {}

        returns = returns_data['returns']

        # Prepare benchmark data
        benchmark_returns = None
        if self.benchmark_enabled:
            try:
                import quantstats as qs
                benchmark_returns = qs.utils.download_returns('SPY')
            except Exception as e:
                st.warning(f"Could not load benchmark data: {e}")

        # Generate comprehensive tearsheet
        tearsheet = self.analytics.generate_comprehensive_tearsheet(
            returns=returns,
            benchmark_returns=benchmark_returns,
            title="Portfolio Performance Tearsheet"
        )

        return tearsheet

    def display_overview_metrics(self, metrics: Dict[str, Any]) -> None:
        """Display key metrics overview at the top of tearsheet."""
        st.subheader("📊 Key Performance Metrics")

        # Primary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{metrics.get('total_return', 0):.1%}")
        with col2:
            st.metric("Annual Return", f"{metrics.get('annual_return', 0):.1%}")
        with col3:
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        with col4:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.1%}")

        # Secondary metrics
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Volatility", f"{metrics.get('volatility', 0):.1%}")
        with col6:
            st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1%}")
        with col7:
            st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}")
        with col8:
            st.metric("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.2f}")

    def render_summary_tab(self, tearsheet: Dict[str, Any]) -> None:
        """Render summary tab with performance table."""
        st.subheader("Performance Summary")

        if 'summary_table' in tearsheet:
            st.dataframe(tearsheet['summary_table'], use_container_width=True)

        # Analysis period info
        if 'analysis_period' in tearsheet:
            period = tearsheet['analysis_period']
            st.write(f"**Analysis Period:** {period['start']} to {period['end']}")
            st.write(f"**Total Trading Days:** {period['total_days']}")

    def render_returns_tab(self, tearsheet: Dict[str, Any]) -> None:
        """Render returns analysis tab."""
        st.subheader("Returns Analysis")

        if 'charts' in tearsheet and 'cumulative_returns' in tearsheet['charts']:
            st.plotly_chart(tearsheet['charts']['cumulative_returns'], use_container_width=True)

        # Returns statistics
        if 'metrics' in tearsheet:
            metrics = tearsheet['metrics']
            col1, col2, col3 = st.columns(3)
            with col1:
                daily_returns = metrics.get('daily_returns', pd.Series())
                if not daily_returns.empty:
                    st.metric("Best Day", f"{daily_returns.max():.2%}")
            with col2:
                if not daily_returns.empty:
                    st.metric("Worst Day", f"{daily_returns.min():.2%}")
            with col3:
                if not daily_returns.empty:
                    st.metric("Avg Daily Return", f"{daily_returns.mean():.2%}")

    def render_risk_tab(self, tearsheet: Dict[str, Any]) -> None:
        """Render risk analysis tab."""
        st.subheader("Risk Analysis")

        # Risk metrics gauges
        if 'charts' in tearsheet and 'risk_metrics' in tearsheet['charts']:
            st.plotly_chart(tearsheet['charts']['risk_metrics'], use_container_width=True)

        # Drawdown chart
        if 'charts' in tearsheet and 'drawdowns' in tearsheet['charts']:
            st.subheader("Drawdown Analysis")
            st.plotly_chart(tearsheet['charts']['drawdowns'], use_container_width=True)

    def render_monthly_tab(self, tearsheet: Dict[str, Any]) -> None:
        """Render monthly performance tab."""
        st.subheader("Monthly Performance")

        if 'charts' in tearsheet and 'monthly_heatmap' in tearsheet['charts']:
            st.plotly_chart(tearsheet['charts']['monthly_heatmap'], use_container_width=True)

        # Monthly returns table
        if 'metrics' in tearsheet and 'monthly_returns' in tearsheet['metrics']:
            monthly_returns = tearsheet['metrics']['monthly_returns']
            if not monthly_returns.empty:
                st.subheader("Monthly Returns Table")
                monthly_pct = monthly_returns * 100
                st.dataframe(monthly_pct.round(2), use_container_width=True)

    def render_rolling_tab(self, tearsheet: Dict[str, Any]) -> None:
        """Render rolling statistics tab."""
        st.subheader("Rolling Statistics")

        if 'charts' in tearsheet and 'rolling_statistics' in tearsheet['charts']:
            st.plotly_chart(tearsheet['charts']['rolling_statistics'], use_container_width=True)

    def render_distribution_tab(self, tearsheet: Dict[str, Any]) -> None:
        """Render distribution analysis tab."""
        st.subheader("Returns Distribution Analysis")

        if 'charts' in tearsheet and 'distribution_analysis' in tearsheet['charts']:
            st.plotly_chart(tearsheet['charts']['distribution_analysis'], use_container_width=True)

        # Distribution statistics
        if 'metrics' in tearsheet:
            metrics = tearsheet['metrics']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Skewness", f"{metrics.get('skewness', 0):.2f}")
            with col2:
                st.metric("Kurtosis", f"{metrics.get('kurtosis', 0):.2f}")
            with col3:
                st.metric("VaR (95%)", f"{metrics.get('var_95', 0):.1%}")
            with col4:
                st.metric("CVaR (95%)", f"{metrics.get('cvar_95', 0):.1%}")

    def render_detailed_tab(self, tearsheet: Dict[str, Any]) -> None:
        """Render detailed metrics tab."""
        st.subheader("Detailed Metrics")

        if 'metrics' in tearsheet:
            metrics = tearsheet['metrics']

            # Create detailed metrics table
            detailed_data = []
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if any(term in key.lower() for term in ['return', 'drawdown', 'var', 'cvar']):
                        formatted_value = f"{value:.1%}"
                    elif any(term in key.lower() for term in ['ratio', 'rate']):
                        formatted_value = f"{value:.2f}"
                    elif 'volatility' in key.lower():
                        formatted_value = f"{value:.1%}"
                    else:
                        formatted_value = f"{value:.4f}"
                    detailed_data.append({'Metric': key.replace('_', ' ').title(), 'Value': formatted_value})

            if detailed_data:
                detailed_df = pd.DataFrame(detailed_data)
                st.dataframe(detailed_df, use_container_width=True)

        # Export functionality
        if st.button("Export Tearsheet Data", help="Download complete analysis results"):
            import json
            tearsheet_copy = tearsheet.copy()
            # Remove non-serializable objects
            if 'charts' in tearsheet_copy:
                del tearsheet_copy['charts']
            if 'generated_at' in tearsheet_copy:
                tearsheet_copy['generated_at'] = tearsheet_copy['generated_at'].isoformat()

            st.download_button(
                label="Download JSON",
                data=json.dumps(tearsheet_copy, indent=2, default=str),
                file_name="quantstats_tearsheet.json",
                mime="application/json"
            )

    def render(self) -> None:
        """Render the comprehensive quantstats tearsheet."""
        st.subheader(self.config.title)

        # Sidebar controls
        with st.sidebar:
            st.subheader("QuantStats Tearsheet Settings")

            # Analysis period selection
            self.analysis_period = st.selectbox(
                "Analysis Period",
                ["1M", "3M", "6M", "1Y", "2Y", "5Y"],
                index=3,
                help="Select the time period for analysis"
            )

            # Benchmark toggle
            self.benchmark_enabled = st.checkbox(
                "Include Benchmark (SPY)",
                value=True,
                help="Compare portfolio against S&P 500 benchmark"
            )

            # Report generation section
            st.divider()
            st.subheader("HTML Report")
            if st.button("📊 Generate Full Report", help="Generate comprehensive HTML report"):
                with st.spinner("Generating HTML report..."):
                    self.generate_html_report(data)
                
            if self.report_path and os.path.exists(self.report_path):
                with open(self.report_path, 'r') as f:
                    report_html = f.read()
                    st.download_button(
                        label="📥 Download HTML Report",
                        data=report_html,
                        file_name="quantstats_report.html",
                        mime="text/html"
                    )

            # Refresh button
            if st.button("🔄 Refresh Tearsheet", help="Recalculate all metrics"):
                self.clear_cache()
                self.report_path = None  # Clear old report

        # Get data from cache
        data = self._cache.get('data')

        if not data or 'returns' not in data:
            st.info("No portfolio data available. Please ensure returns data is loaded.")
            return

        # Generate tearsheet
        with st.spinner("Generating comprehensive QuantStats tearsheet..."):
            tearsheet = self.generate_tearsheet(data)

        if not tearsheet:
            st.error("Failed to generate tearsheet")
            return

        # Display overview metrics
        if 'metrics' in tearsheet:
            self.display_overview_metrics(tearsheet['metrics'])

        st.divider()

        # Main tearsheet tabs
        tabs = st.tabs([
            "📊 Summary",
            "📈 Returns Analysis",
            "📉 Risk Analysis",
            "📅 Monthly Performance",
            "📊 Rolling Statistics",
            "📈 Distribution Analysis",
            "🔍 Detailed Metrics"
        ])

        with tabs[0]:  # Summary
            self.render_summary_tab(tearsheet)

        with tabs[1]:  # Returns Analysis
            self.render_returns_tab(tearsheet)

        with tabs[2]:  # Risk Analysis
            self.render_risk_tab(tearsheet)

        with tabs[3]:  # Monthly Performance
            self.render_monthly_tab(tearsheet)

        with tabs[4]:  # Rolling Statistics
            self.render_rolling_tab(tearsheet)

        with tabs[5]:  # Distribution Analysis
            self.render_distribution_tab(tearsheet)

        with tabs[6]:  # Detailed Metrics
            self.render_detailed_tab(tearsheet)

    def calculate_performance_metrics(self, data: Any) -> dict[str, pd.Series]:
        """Calculate performance metrics over time."""
        if not data or 'returns' not in data:
            return {}

        returns = data['returns']
        metrics = {}

        # Basic performance metrics
        metrics['cumulative_returns'] = (1 + returns).cumprod() - 1
        metrics['rolling_sharpe'] = returns.rolling(window=30).mean() / returns.rolling(window=30).std() * np.sqrt(252)
        metrics['rolling_volatility'] = returns.rolling(window=30).std() * np.sqrt(252)

        return metrics

    def analyze_drawdowns(self, returns: pd.Series) -> dict[str, Any]:
        """Analyze drawdowns from return series."""
        if returns.empty:
            return {}

        # Calculate drawdowns
        wealth_index = (1 + returns).cumprod()
        previous_peaks = wealth_index.expanding(min_periods=1).max()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks

        return {
            'drawdowns': drawdowns,
            'max_drawdown': drawdowns.min(),
            'current_drawdown': drawdowns.iloc[-1],
            'drawdown_duration': (drawdowns < 0).astype(int).groupby((drawdowns >= 0).cumsum()).cumsum()
        }

    def create_figure(self, data: Any) -> go.Figure:
        """Create plotly figure for visualization."""
        if not data or 'returns' not in data:
            return go.Figure()

        returns = data['returns']
        cumulative_returns = (1 + returns).cumprod() - 1

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values * 100,
            name='Cumulative Returns',
            line=dict(color='blue', width=2)
        ))

        fig.update_layout(
            title='Portfolio Performance',
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            height=400
        )

        return fig

    def generate_html_report(self, data: Dict[str, Any]) -> None:
        """Generate comprehensive HTML report using quantstats."""
        if not data or 'returns' not in data:
            st.error("No data available for report generation")
            return

        try:
            import quantstats as qs
            import tempfile
            import os

            returns = data['returns']
            
            # Prepare benchmark data if enabled
            benchmark_returns = None
            if self.benchmark_enabled:
                try:
                    benchmark_returns = qs.utils.download_returns('SPY')
                except Exception as e:
                    st.warning(f"Could not load benchmark data: {e}")

            # Create temporary file for the report
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                # Generate the report
                qs.reports.html(
                    returns=returns,
                    benchmark=benchmark_returns,
                    output=tmp.name,
                    title=f"Portfolio Analysis Report - {datetime.now().strftime('%Y-%m-%d')}",
                    rf=0.02  # Risk-free rate
                )
                self.report_path = tmp.name

            st.success("HTML report generated successfully!")
            
        except Exception as e:
            st.error(f"Error generating HTML report: {e}")
            self.report_path = None

    def update(self, data: Dict[str, Any]) -> None:
        """Update component with new data."""
        self._cache['data'] = data
        self.clear_cache()
        self.report_path = None  # Clear old report