"""
QuantStats Analytics Integration Module - Final Version
Provides comprehensive portfolio analytics using the quantstats library.
Handles cumulative returns, Sharpe ratio, maximum drawdown, and risk metrics.
"""

import pandas as pd
import numpy as np
import quantstats as qs
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class QuantStatsAnalytics:
    """Main class for quantstats-based portfolio analytics."""

    def __init__(self):
        """Initialize quantstats analytics with default settings."""
        # Configure quantstats
        qs.extend_pandas()

        # Default settings
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.compounding = True
        self.periods_per_year = 252  # Trading days

    def prepare_returns_data(self, returns: Union[pd.Series, pd.DataFrame],
                            benchmark_returns: Optional[Union[pd.Series, pd.DataFrame]] = None) -> Tuple[pd.Series, Optional[pd.Series]]:
        """
        Prepare returns data for quantstats analysis.

        Args:
            returns: Portfolio returns (Series for single portfolio, DataFrame for multiple)
            benchmark_returns: Optional benchmark returns

        Returns:
            Tuple of (portfolio_returns, benchmark_returns)
        """
        # Ensure returns are in the correct format
        if isinstance(returns, pd.DataFrame):
            # For multiple portfolios, use the first column as primary
            portfolio_returns = returns.iloc[:, 0] if not returns.empty else pd.Series()
        else:
            portfolio_returns = returns

        # Clean and prepare benchmark data
        if benchmark_returns is not None:
            if isinstance(benchmark_returns, pd.DataFrame):
                benchmark_returns = benchmark_returns.iloc[:, 0] if not benchmark_returns.empty else None
        else:
            # Use S&P 500 as default benchmark if available
            try:
                benchmark_returns = qs.utils.download_returns('SPY')
            except Exception as e:
                logger.warning(f"Could not download benchmark data: {e}")
                benchmark_returns = None

        return portfolio_returns, benchmark_returns

    def calculate_comprehensive_metrics(self, returns: pd.Series,
                                      benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Union[float, pd.Series, pd.DataFrame]]:
        """
        Calculate comprehensive metrics similar to quantstats tearsheet.

        Args:
            returns: Portfolio returns
            benchmark_returns: Optional benchmark returns

        Returns:
            Dictionary with comprehensive metrics
        """
        try:
            metrics = {}

            # Basic return metrics
            metrics['total_return'] = qs.stats.comp(returns)
            metrics['annual_return'] = qs.stats.cagr(returns)
            metrics['daily_returns'] = returns
            # Calculate cumulative returns manually since cum_returns doesn't exist
            metrics['cumulative_returns'] = (1 + returns).cumprod() - 1

            # Risk metrics
            metrics['volatility'] = qs.stats.volatility(returns, periods=self.periods_per_year)
            metrics['sharpe_ratio'] = qs.stats.sharpe(returns, rf=self.risk_free_rate, periods=self.periods_per_year)
            metrics['sortino_ratio'] = qs.stats.sortino(returns, rf=self.risk_free_rate, periods=self.periods_per_year)
            metrics['calmar_ratio'] = qs.stats.calmar(returns)
            metrics['max_drawdown'] = qs.stats.max_drawdown(returns)
            metrics['var_95'] = qs.stats.value_at_risk(returns, sigma=2.0)
            metrics['cvar_95'] = qs.stats.conditional_value_at_risk(returns, sigma=2.0)

            # Additional risk metrics
            metrics['skewness'] = qs.stats.skew(returns)
            metrics['kurtosis'] = qs.stats.kurtosis(returns)
            metrics['tail_ratio'] = qs.stats.tail_ratio(returns)
            metrics['common_sense_ratio'] = qs.stats.common_sense_ratio(returns)

            # Win rate and performance metrics
            metrics['win_rate'] = qs.stats.win_rate(returns)
            metrics['avg_win'] = qs.stats.avg_win(returns)
            metrics['avg_loss'] = qs.stats.avg_loss(returns)
            metrics['profit_factor'] = qs.stats.profit_factor(returns)
            metrics['payoff_ratio'] = qs.stats.payoff_ratio(returns)

            # Rolling metrics (simplified without periods parameter)
            try:
                metrics['rolling_sharpe'] = qs.stats.rolling_sharpe(returns, rf=self.risk_free_rate)
                metrics['rolling_volatility'] = qs.stats.rolling_volatility(returns)
                metrics['rolling_sortino'] = qs.stats.rolling_sortino(returns, rf=self.risk_free_rate)
            except Exception as e:
                logger.warning(f"Rolling metrics calculation failed: {e}")
                # Fallback to simple rolling calculations
                window = 30  # 30-day rolling window
                metrics['rolling_sharpe'] = returns.rolling(window, min_periods=1).mean() / returns.rolling(window, min_periods=1).std() * np.sqrt(252)
                metrics['rolling_volatility'] = returns.rolling(window, min_periods=1).std() * np.sqrt(252)
                metrics['rolling_sortino'] = returns.rolling(window, min_periods=1).mean() / returns[returns < 0].rolling(window, min_periods=1).std() * np.sqrt(252)

            # Monthly returns
            try:
                monthly_returns = qs.stats.monthly_returns(returns)
                # Ensure it has proper datetime index
                if isinstance(monthly_returns.index, pd.DatetimeIndex):
                    metrics['monthly_returns'] = monthly_returns
                else:
                    # Create proper datetime index for monthly returns
                    monthly_idx = pd.date_range(start=returns.index.min(), end=returns.index.max(), freq='M')
                    monthly_data = []
                    for i, date in enumerate(monthly_idx):
                        if i < len(monthly_returns):
                            monthly_data.append(monthly_returns.iloc[i])
                        else:
                            monthly_data.append(0.0)
                    metrics['monthly_returns'] = pd.Series(monthly_data, index=monthly_idx)
            except Exception as e:
                logger.warning(f"Monthly returns calculation failed: {e}")
                # Fallback: calculate monthly returns manually
                monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                metrics['monthly_returns'] = monthly_returns

            # Drawdown analysis
            try:
                metrics['drawdowns'] = qs.stats.to_drawdown_series(returns)
                metrics['drawdown_details'] = qs.stats.drawdown_details(metrics['drawdowns'])
            except Exception as e:
                logger.warning(f"Drawdown analysis failed: {e}")
                # Fallback drawdown calculation
                wealth_index = (1 + returns).cumprod()
                previous_peaks = wealth_index.expanding(min_periods=1).max()
                metrics['drawdowns'] = (wealth_index - previous_peaks) / previous_peaks
                metrics['drawdown_details'] = pd.DataFrame()  # Empty fallback

            # Benchmark-relative metrics
            if benchmark_returns is not None:
                try:
                    metrics['beta'] = qs.stats.beta(returns, benchmark_returns)
                    metrics['alpha'] = qs.stats.alpha(returns, benchmark_returns, rf=self.risk_free_rate, periods=self.periods_per_year)
                    metrics['r_squared'] = qs.stats.r_squared(returns, benchmark_returns)
                    metrics['tracking_error'] = qs.stats.tracking_error(returns, benchmark_returns)
                    metrics['information_ratio'] = qs.stats.information_ratio(returns, benchmark_returns)
                except Exception as e:
                    logger.warning(f"Benchmark metrics calculation failed: {e}")

            return metrics

        except Exception as e:
            logger.error(f"Error calculating comprehensive metrics: {e}")
            return {}

    def create_cumulative_returns_chart(self, cumulative_data: Dict[str, pd.Series]) -> go.Figure:
        """
        Create interactive cumulative returns chart.

        Args:
            cumulative_data: Dictionary with cumulative returns data

        Returns:
            Plotly figure object
        """
        fig = go.Figure()

        # Add portfolio cumulative returns
        if 'portfolio_cumulative' in cumulative_data:
            portfolio_data = cumulative_data['portfolio_cumulative']
            fig.add_trace(go.Scatter(
                x=portfolio_data.index,
                y=portfolio_data.values * 100,
                name='Portfolio',
                line=dict(color='blue', width=2),
                hovertemplate='Date: %{x}<br>Cumulative Return: %{y:.2f}%<extra></extra>'
            ))

        # Add benchmark cumulative returns
        if 'benchmark_cumulative' in cumulative_data:
            benchmark_data = cumulative_data['benchmark_cumulative']
            fig.add_trace(go.Scatter(
                x=benchmark_data.index,
                y=benchmark_data.values * 100,
                name='Benchmark (SPY)',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>Cumulative Return: %{y:.2f}%<extra></extra>'
            ))

        # Update layout
        fig.update_layout(
            title='Cumulative Returns',
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            hovermode='x unified',
            showlegend=True,
            height=500
        )

        return fig

    def create_drawdown_chart(self, drawdown_data: Dict[str, Union[float, pd.Series]]) -> go.Figure:
        """
        Create drawdown visualization chart.

        Args:
            drawdown_data: Dictionary with drawdown analysis data

        Returns:
            Plotly figure object
        """
        fig = go.Figure()

        if 'drawdown_series' in drawdown_data:
            drawdown_series = drawdown_data['drawdown_series']
            fig.add_trace(go.Scatter(
                x=drawdown_series.index,
                y=drawdown_series.values * 100,
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red', width=1),
                hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
            ))

        fig.update_layout(
            title='Portfolio Drawdowns',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            showlegend=False,
            height=400
        )

        return fig

    def create_monthly_returns_heatmap(self, monthly_returns: Union[pd.Series, pd.DataFrame]) -> go.Figure:
        """
        Create monthly returns heatmap visualization.

        Args:
            monthly_returns: Monthly returns data

        Returns:
            Plotly figure object
        """
        try:
            # Handle different formats of monthly returns
            if isinstance(monthly_returns, pd.DataFrame):
                # If it's a DataFrame, use the first column
                monthly_series = monthly_returns.iloc[:, 0] if not monthly_returns.empty else pd.Series()
            else:
                monthly_series = monthly_returns

            # Ensure we have a proper datetime index
            if not isinstance(monthly_series.index, pd.DatetimeIndex):
                # Create a proper datetime index
                start_date = pd.Timestamp('2023-01-01') if monthly_series.empty else pd.Timestamp('2023-01-01')
                monthly_series.index = pd.date_range(start=start_date, periods=len(monthly_series), freq='M')

            # Convert to DataFrame with year/month structure
            monthly_df = monthly_series.to_frame('returns')
            monthly_df['year'] = monthly_df.index.year
            monthly_df['month'] = monthly_df.index.month
            monthly_df['month_name'] = monthly_df.index.strftime('%b')

            # Pivot to create heatmap data
            heatmap_data = monthly_df.pivot_table(
                values='returns',
                index='year',
                columns='month_name',
                aggfunc='first'
            )

            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='RdYlGn',
                text=np.round(heatmap_data.values * 100, 2),
                texttemplate='%{text}%',
                textfont={"size": 10},
                hoverongaps=False
            ))

            fig.update_layout(
                title='Monthly Returns Heatmap',
                xaxis_title='Month',
                yaxis_title='Year',
                height=400
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating monthly returns heatmap: {e}")
            return go.Figure()

    def create_rolling_statistics_chart(self, rolling_data: Dict[str, pd.Series]) -> go.Figure:
        """
        Create rolling statistics visualization.

        Args:
            rolling_data: Dictionary with rolling statistics

        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Rolling Sharpe Ratio', 'Rolling Volatility', 'Rolling Sortino Ratio'),
            vertical_spacing=0.1
        )

        # Rolling Sharpe
        if 'rolling_sharpe' in rolling_data and not rolling_data['rolling_sharpe'].empty:
            sharpe_data = rolling_data['rolling_sharpe']
            fig.add_trace(
                go.Scatter(
                    x=sharpe_data.index,
                    y=sharpe_data.values,
                    name='Rolling Sharpe',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )

        # Rolling Volatility
        if 'rolling_volatility' in rolling_data and not rolling_data['rolling_volatility'].empty:
            vol_data = rolling_data['rolling_volatility']
            fig.add_trace(
                go.Scatter(
                    x=vol_data.index,
                    y=vol_data.values,
                    name='Rolling Volatility',
                    line=dict(color='red')
                ),
                row=2, col=1
            )

        # Rolling Sortino
        if 'rolling_sortino' in rolling_data and not rolling_data['rolling_sortino'].empty:
            sortino_data = rolling_data['rolling_sortino']
            fig.add_trace(
                go.Scatter(
                    x=sortino_data.index,
                    y=sortino_data.values,
                    name='Rolling Sortino',
                    line=dict(color='green')
                ),
                row=3, col=1
            )

        fig.update_layout(height=800, showlegend=False)
        return fig

    def create_returns_distribution_chart(self, returns: pd.Series) -> go.Figure:
        """
        Create returns distribution analysis chart.

        Args:
            returns: Daily returns data

        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Returns Distribution', 'Q-Q Plot'),
            horizontal_spacing=0.1
        )

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=returns.values,
                nbinsx=50,
                name='Returns',
                showlegend=False
            ),
            row=1, col=1
        )

        # Add normal distribution overlay
        x_range = np.linspace(returns.min(), returns.max(), 100)
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=np.exp(-(x_range - returns.mean())**2 / (2 * returns.std()**2)) /
                   (returns.std() * np.sqrt(2 * np.pi)) * len(returns) * (returns.max() - returns.min()) / 50,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )

        # Q-Q plot
        from scipy import stats
        probplot_data = stats.probplot(returns.values, dist="norm")
        fig.add_trace(
            go.Scatter(
                x=probplot_data[0][0],
                y=probplot_data[0][1],
                mode='markers',
                name='Q-Q Plot',
                showlegend=False
            ),
            row=1, col=2
        )

        # Add reference line
        fig.add_trace(
            go.Scatter(
                x=probplot_data[0][0],
                y=probplot_data[0][0] * probplot_data[1][0] + probplot_data[1][1],
                mode='lines',
                name='Reference Line',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )

        fig.update_layout(height=400)
        return fig

    def create_performance_summary_table(self, metrics: Dict[str, Union[float, pd.Series]]) -> pd.DataFrame:
        """
        Create performance summary table.

        Args:
            metrics: Dictionary of performance metrics

        Returns:
            DataFrame with formatted performance summary
        """
        summary_data = []

        # Returns section
        monthly_returns = metrics.get('monthly_returns', pd.Series())
        summary_data.extend([
            {'Category': 'Returns', 'Metric': 'Total Return', 'Value': f"{metrics.get('total_return', 0):.1%}"},
            {'Category': 'Returns', 'Metric': 'Annual Return', 'Value': f"{metrics.get('annual_return', 0):.1%}"},
            {'Category': 'Returns', 'Metric': 'Best Month', 'Value': f"{monthly_returns.max() if not monthly_returns.empty else 0:.1%}"},
            {'Category': 'Returns', 'Metric': 'Worst Month', 'Value': f"{monthly_returns.min() if not monthly_returns.empty else 0:.1%}"},
        ])

        # Risk metrics section
        summary_data.extend([
            {'Category': 'Risk Metrics', 'Metric': 'Volatility', 'Value': f"{metrics.get('volatility', 0):.1%}"},
            {'Category': 'Risk Metrics', 'Metric': 'Sharpe Ratio', 'Value': f"{metrics.get('sharpe_ratio', 0):.2f}"},
            {'Category': 'Risk Metrics', 'Metric': 'Sortino Ratio', 'Value': f"{metrics.get('sortino_ratio', 0):.2f}"},
            {'Category': 'Risk Metrics', 'Metric': 'Max Drawdown', 'Value': f"{metrics.get('max_drawdown', 0):.1%}"},
            {'Category': 'Risk Metrics', 'Metric': 'VaR (95%)', 'Value': f"{metrics.get('var_95', 0):.1%}"},
            {'Category': 'Risk Metrics', 'Metric': 'CVaR (95%)', 'Value': f"{metrics.get('cvar_95', 0):.1%}"},
        ])

        # Win rate section
        summary_data.extend([
            {'Category': 'Win Rate', 'Metric': 'Win Rate', 'Value': f"{metrics.get('win_rate', 0):.1%}"},
            {'Category': 'Win Rate', 'Metric': 'Avg Win', 'Value': f"{metrics.get('avg_win', 0):.2%}"},
            {'Category': 'Win Rate', 'Metric': 'Avg Loss', 'Value': f"{metrics.get('avg_loss', 0):.2%}"},
            {'Category': 'Win Rate', 'Metric': 'Profit Factor', 'Value': f"{metrics.get('profit_factor', 0):.2f}"},
        ])

        return pd.DataFrame(summary_data)

    def generate_comprehensive_tearsheet(self, returns: pd.Series,
                                       benchmark_returns: Optional[pd.Series] = None,
                                       title: str = "Portfolio Tearsheet") -> Dict[str, Union[go.Figure, pd.DataFrame, Dict]]:
        """
        Generate comprehensive tearsheet similar to quantstats.

        Args:
            returns: Portfolio returns
            benchmark_returns: Optional benchmark returns
            title: Report title

        Returns:
            Dictionary containing all tearsheet components
        """
        try:
            tearsheet = {
                'title': title,
                'generated_at': datetime.now(),
                'analysis_period': {
                    'start': returns.index.min(),
                    'end': returns.index.max(),
                    'total_days': len(returns)
                }
            }

            # Calculate comprehensive metrics
            metrics = self.calculate_comprehensive_metrics(returns, benchmark_returns)

            # Create visualizations
            cumulative_chart = self.create_cumulative_returns_chart({
                'portfolio_cumulative': metrics.get('cumulative_returns', pd.Series())
            })

            monthly_heatmap = self.create_monthly_returns_heatmap(metrics.get('monthly_returns', pd.Series()))
            rolling_chart = self.create_rolling_statistics_chart({
                'rolling_sharpe': metrics.get('rolling_sharpe', pd.Series()),
                'rolling_volatility': metrics.get('rolling_volatility', pd.Series()),
                'rolling_sortino': metrics.get('rolling_sortino', pd.Series())
            })

            distribution_chart = self.create_returns_distribution_chart(returns)
            drawdown_chart = self.create_drawdown_chart({
                'drawdown_series': metrics.get('drawdowns', pd.Series())
            })

            # Create summary table
            summary_table = self.create_performance_summary_table(metrics)

            # Add to tearsheet
            tearsheet.update({
                'metrics': metrics,
                'summary_table': summary_table,
                'charts': {
                    'cumulative_returns': cumulative_chart,
                    'monthly_heatmap': monthly_heatmap,
                    'rolling_statistics': rolling_chart,
                    'distribution_analysis': distribution_chart,
                    'drawdowns': drawdown_chart
                }
            })

            return tearsheet

        except Exception as e:
            logger.error(f"Error generating comprehensive tearsheet: {e}")
            return {'error': str(e)}