"""
QuantStats Portfolio Analytics Component
Provides comprehensive portfolio analysis using quantstats library.
Integrates with the existing dashboard framework.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

from . import (
    PerformanceComponent,
    ComponentConfig,
    cache_data
)
from dashboard.utils.quantstats_analytics import QuantStatsAnalytics

class QuantStatsPortfolio(PerformanceComponent):
    """Component for quantstats-based portfolio analytics."""

    def __init__(self, config: ComponentConfig):
        """Initialize quantstats portfolio component."""
        super().__init__(config)
        self.analytics = QuantStatsAnalytics()

        # Component state
        self.selected_portfolio = None
        self.benchmark_enabled = True
        self.show_comparison = False
        self.analysis_period = "1Y"
        self.report_path = None  # Store path to generated HTML report

    @st.cache_data(ttl=timedelta(seconds=300))
    def analyze_portfolio(self, returns_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio using quantstats."""
        if not returns_data or 'returns' not in returns_data:
            return {}

        returns = returns_data['returns']

        # Prepare benchmark data
        benchmark_returns = None
        if self.benchmark_enabled:
            try:
                # Use SPY as benchmark
                import quantstats as qs
                benchmark_returns = qs.utils.download_returns('SPY')
            except Exception as e:
                st.warning(f"Could not load benchmark data: {e}")

        # Generate comprehensive report
        report = self.analytics.generate_portfolio_report(
            returns=returns,
            benchmark_returns=benchmark_returns,
            title="Portfolio Performance Analysis"
        )

        return report

    def create_metrics_summary(self, report: Dict[str, Any]) -> None:
        """Create metrics summary display."""
        if not report or 'risk_metrics' not in report:
            st.warning("No risk metrics available")
            return

        metrics = report['risk_metrics']

        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            sharpe = metrics.get('sharpe_ratio', 0)
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.2f}",
                help="Risk-adjusted return measure. Higher is better."
            )

        with col2:
            sortino = metrics.get('sortino_ratio', 0)
            st.metric(
                "Sortino Ratio",
                f"{sortino:.2f}",
                help="Downside risk-adjusted return. Higher is better."
            )

        with col3:
            max_dd = metrics.get('max_drawdown', 0)
            st.metric(
                "Max Drawdown",
                f"{max_dd:.1%}",
                help="Largest peak-to-trough decline. Lower is better."
            )

        with col4:
            volatility = metrics.get('volatility', 0)
            st.metric(
                "Volatility",
                f"{volatility:.1%}",
                help="Standard deviation of returns."
            )

        # Additional metrics
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            calmar = metrics.get('calmar_ratio', 0)
            st.metric(
                "Calmar Ratio",
                f"{calmar:.2f}",
                help="Annual return divided by max drawdown. Higher is better."
            )

        with col6:
            omega = metrics.get('omega_ratio', 0)
            st.metric(
                "Omega Ratio",
                f"{omega:.2f}",
                help="Probability-weighted ratio of gains vs losses. Higher is better."
            )

        with col7:
            var_95 = metrics.get('var_95', 0)
            st.metric(
                "VaR (95%)",
                f"{var_95:.1%}",
                help="Value at Risk at 95% confidence level."
            )

        with col8:
            cvar_95 = metrics.get('cvar_95', 0)
            st.metric(
                "CVaR (95%)",
                f"{cvar_95:.1%}",
                help="Conditional Value at Risk at 95% confidence level."
            )

    def create_portfolio_comparison(self, portfolio_data: Dict[str, Any]) -> None:
        """Create portfolio comparison interface."""
        st.subheader("Portfolio Comparison")

        # Portfolio selection
        available_portfolios = ["Portfolio 1", "Portfolio 2", "Portfolio 3"]  # Placeholder
        selected_portfolios = st.multiselect(
            "Select Portfolios to Compare",
            available_portfolios,
            default=["Portfolio 1"],
            help="Choose portfolios to include in comparison"
        )

        if len(selected_portfolios) > 1:
            # Generate sample comparison data
            comparison_data = self.generate_sample_comparison_data(selected_portfolios)

            if comparison_data:
                # Display comparison chart
                st.plotly_chart(
                    comparison_data['comparison_chart'],
                    use_container_width=True,
                    help="Compare cumulative returns across selected portfolios"
                )

                # Display metrics comparison table
                st.subheader("Risk Metrics Comparison")
                st.dataframe(
                    comparison_data['metrics_comparison'],
                    use_container_width=True,
                    help="Compare key risk metrics across portfolios"
                )
        else:
            st.info("Select multiple portfolios to enable comparison")

    def generate_sample_comparison_data(self, portfolio_names: List[str]) -> Optional[Dict[str, Any]]:
        """Generate sample comparison data for demonstration."""
        try:
            # Create sample returns for each portfolio
            dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
            np.random.seed(42)  # For reproducible results

            portfolio_returns = {}
            for name in portfolio_names:
                # Generate random returns with different characteristics
                if "Portfolio 1" in name:
                    returns = np.random.normal(0.0005, 0.02, len(dates))
                elif "Portfolio 2" in name:
                    returns = np.random.normal(0.0003, 0.015, len(dates))
                else:
                    returns = np.random.normal(0.0007, 0.025, len(dates))

                portfolio_returns[name] = pd.Series(returns, index=dates)

            # Use analytics module for comparison
            comparison_results = self.analytics.compare_portfolios(portfolio_returns)

            return comparison_results

        except Exception as e:
            st.error(f"Error generating comparison data: {e}")
            return None

    def render_analysis_tabs(self, report: Dict[str, Any]) -> None:
        """Render analysis results in tabs."""
        if not report or 'charts' not in report:
            st.warning("No analysis data available")
            return

        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Returns Analysis",
            "📊 Risk Metrics",
            "📉 Drawdown Analysis",
            "🔍 Detailed Report"
        ])

        with tab1:
            st.subheader("Cumulative Returns")
            if 'charts' in report and 'cumulative_returns' in report['charts']:
                st.plotly_chart(
                    report['charts']['cumulative_returns'],
                    use_container_width=True
                )

            # Returns summary
            if 'cumulative_returns' in report:
                cum_data = report['cumulative_returns']
                if 'portfolio_total_return' in cum_data:
                    st.metric(
                        "Total Return",
                        f"{cum_data['portfolio_total_return']:.1%}",
                        help="Total cumulative return over the analysis period"
                    )

        with tab2:
            st.subheader("Risk Metrics Dashboard")
            if 'charts' in report and 'risk_metrics' in report['charts']:
                st.plotly_chart(
                    report['charts']['risk_metrics'],
                    use_container_width=True
                )

        with tab3:
            st.subheader("Drawdown Analysis")
            if 'charts' in report and 'drawdowns' in report['charts']:
                st.plotly_chart(
                    report['charts']['drawdowns'],
                    use_container_width=True
                )

            # Drawdown statistics
            if 'drawdown_analysis' in report:
                dd_analysis = report['drawdown_analysis']
                if 'max_drawdown' in dd_analysis:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Max Drawdown", f"{dd_analysis['max_drawdown']:.1%}")
                    with col2:
                        st.metric("Avg Drawdown", f"{dd_analysis.get('avg_drawdown', 0):.1%}")
                    with col3:
                        avg_recovery = dd_analysis.get('avg_recovery_days', 0)
                        st.metric("Avg Recovery (Days)", f"{avg_recovery:.1f}")

        with tab4:
            st.subheader("Detailed Analysis Report")

            # Analysis period info
            if 'analysis_period' in report:
                period = report['analysis_period']
                st.write(f"**Analysis Period:** {period['start']} to {period['end']}")
                st.write(f"**Total Trading Days:** {period['total_days']}")

            # Raw metrics data
            if 'risk_metrics' in report:
                st.subheader("Raw Risk Metrics")
                metrics_df = pd.DataFrame(
                    list(report['risk_metrics'].items()),
                    columns=['Metric', 'Value']
                )
                st.dataframe(metrics_df, use_container_width=True)

            # Export options
            if st.button("Export Report Data", help="Download analysis results as JSON"):
                import json
                report_copy = report.copy()
                # Remove non-serializable objects
                if 'charts' in report_copy:
                    del report_copy['charts']
                if 'generated_at' in report_copy:
                    report_copy['generated_at'] = report_copy['generated_at'].isoformat()

                st.download_button(
                    label="Download JSON",
                    data=json.dumps(report_copy, indent=2, default=str),
                    file_name="portfolio_analysis_report.json",
                    mime="application/json"
                )

    def render(self) -> None:
        """Render the quantstats portfolio component."""
        st.subheader(self.config.title)

        # Sidebar controls
        with st.sidebar:
            st.subheader("QuantStats Analytics Settings")

            # Analysis period selection
            self.analysis_period = st.selectbox(
                "Analysis Period",
                ["1M", "3M", "6M", "1Y", "2Y", "5Y"],
                index=3,  # Default to 1Y
                help="Select the time period for analysis"
            )

            # Benchmark toggle
            self.benchmark_enabled = st.checkbox(
                "Include Benchmark (SPY)",
                value=True,
                help="Compare portfolio against S&P 500 benchmark"
            )

            # Portfolio comparison toggle
            self.show_comparison = st.checkbox(
                "Enable Portfolio Comparison",
                value=False,
                help="Compare multiple portfolios side by side"
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
                        file_name="portfolio_analysis_report.html",
                        mime="text/html"
                    )

            # Refresh button
            if st.button("🔄 Refresh Analysis", help="Recalculate all metrics"):
                self.clear_cache()
                self.report_path = None  # Clear old report

        # Get data from cache
        data = self._cache.get('data')

        if not data or 'returns' not in data:
            st.info("No portfolio data available. Please ensure returns data is loaded.")
            return

        # Analyze portfolio
        with st.spinner("Analyzing portfolio with QuantStats..."):
            report = self.analyze_portfolio(data)

        if not report:
            st.error("Failed to generate portfolio analysis")
            return

        # Display metrics summary
        self.create_metrics_summary(report)

        st.divider()

        # Main analysis tabs
        self.render_analysis_tabs(report)

        # Portfolio comparison section
        if self.show_comparison:
            st.divider()
            self.create_portfolio_comparison(data)

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
        self.clear_cache()  # Clear cached calculations
        self.report_path = None  # Clear old report