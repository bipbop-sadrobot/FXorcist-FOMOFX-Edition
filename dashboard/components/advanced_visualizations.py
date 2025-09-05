"""
Advanced visualization components for the Forex AI dashboard.
Includes interactive charts, heatmaps, 3D visualizations, and performance optimizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

class InteractivePriceChart:
    """Advanced interactive price chart with multiple overlays."""

    def __init__(self, height: int = 600):
        self.height = height
        self.chart_cache = {}

    @st.cache_data(ttl=timedelta(seconds=300))
    def create_multi_timeframe_chart(self,
                                   data: pd.DataFrame,
                                   symbol: str,
                                   timeframes: List[str] = None) -> go.Figure:
        """Create multi-timeframe price chart."""
        if timeframes is None:
            timeframes = ['1H', '4H', '1D']

        # Create subplots
        fig = make_subplots(
            rows=len(timeframes),
            cols=1,
            shared_xaxes=True,
            subplot_titles=[f'{symbol} - {tf}' for tf in timeframes],
            vertical_spacing=0.05
        )

        colors = ['#2563eb', '#10b981', '#f59e0b']

        for i, timeframe in enumerate(timeframes):
            # Resample data to timeframe
            if timeframe != '1H':
                resampled = self._resample_data(data, timeframe)
            else:
                resampled = data

            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=resampled.index,
                    open=resampled['open'],
                    high=resampled['high'],
                    low=resampled['low'],
                    close=resampled['close'],
                    name=f'Price {timeframe}',
                    increasing_line_color=colors[i % len(colors)],
                    decreasing_line_color='#ef4444'
                ),
                row=i+1, col=1
            )

            # Add volume bars
            if 'volume' in resampled.columns:
                fig.add_trace(
                    go.Bar(
                        x=resampled.index,
                        y=resampled['volume'],
                        name=f'Volume {timeframe}',
                        marker_color='rgba(100,100,100,0.3)',
                        showlegend=False
                    ),
                    row=i+1, col=1,
                    secondary_y=True
                )

        # Update layout
        fig.update_layout(
            height=self.height,
            xaxis_rangeslider_visible=False,
            showlegend=True
        )

        # Update y-axes
        for i in range(len(timeframes)):
            fig.update_yaxes(title_text="Price", row=i+1, col=1)

        return fig

    def _resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to specified timeframe."""
        rules = {
            '4H': '4H',
            '1D': 'D',
            '1W': 'W',
            '1M': 'M'
        }

        rule = rules.get(timeframe, 'D')

        resampled = data.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum' if 'volume' in data.columns else 'count'
        }).dropna()

        return resampled

    @st.cache_data(ttl=timedelta(seconds=300))
    def create_prediction_confidence_chart(self,
                                         actual_data: pd.DataFrame,
                                         predictions: pd.Series,
                                         confidence_intervals: Tuple[pd.Series, pd.Series],
                                         symbol: str) -> go.Figure:
        """Create prediction confidence visualization."""
        fig = go.Figure()

        # Actual prices
        fig.add_trace(go.Scatter(
            x=actual_data.index,
            y=actual_data['close'],
            mode='lines',
            name='Actual Price',
            line=dict(color='#64748b', width=2)
        ))

        # Predictions
        fig.add_trace(go.Scatter(
            x=predictions.index,
            y=predictions.values,
            mode='lines',
            name='Predicted Price',
            line=dict(color='#2563eb', width=3)
        ))

        # Confidence intervals
        lower_bound, upper_bound = confidence_intervals

        fig.add_trace(go.Scatter(
            x=lower_bound.index,
            y=lower_bound.values,
            mode='lines',
            name='Lower Bound',
            line=dict(color='#10b981', width=1, dash='dash'),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=upper_bound.index,
            y=upper_bound.values,
            mode='lines',
            name='Upper Bound',
            line=dict(color='#ef4444', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(16, 185, 129, 0.1)',
            showlegend=False
        ))

        # Add prediction accuracy indicators
        accuracy = self._calculate_prediction_accuracy(actual_data['close'], predictions)
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=f"Prediction Accuracy: {accuracy:.2f}%",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#2563eb",
            borderwidth=1
        )

        fig.update_layout(
            title=f'{symbol} Price Prediction with Confidence Intervals',
            xaxis_title='Time',
            yaxis_title='Price',
            height=self.height,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig

    def _calculate_prediction_accuracy(self, actual: pd.Series, predicted: pd.Series) -> float:
        """Calculate prediction accuracy percentage."""
        try:
            # Calculate directional accuracy
            actual_direction = np.sign(actual.diff().dropna())
            predicted_direction = np.sign(predicted.diff().dropna())

            # Align the series
            min_len = min(len(actual_direction), len(predicted_direction))
            accuracy = (actual_direction[:min_len] == predicted_direction[:min_len]).mean() * 100

            return accuracy
        except Exception:
            return 0.0

class CorrelationHeatmap:
    """Interactive correlation heatmap for feature analysis."""

    def __init__(self, height: int = 600):
        self.height = height

    @st.cache_data(ttl=timedelta(seconds=300))
    def create_feature_correlation_heatmap(self,
                                         data: pd.DataFrame,
                                         features: List[str] = None,
                                         method: str = 'spearman') -> go.Figure:
        """Create interactive correlation heatmap."""
        if features is None:
            # Select numeric features
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            features = [col for col in numeric_cols if col not in ['timestamp', 'volume']][:20]

        # Calculate correlation matrix
        corr_matrix = data[features].corr(method=method)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))

        # Add correlation strength annotations
        annotations = []
        for i, row in enumerate(corr_matrix.values):
            for j, val in enumerate(row):
                if abs(val) > 0.7 and i != j:  # Strong correlations
                    strength = "Strong" if abs(val) > 0.8 else "Moderate"
                    direction = "Positive" if val > 0 else "Negative"
                    annotations.append(
                        f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {strength} {direction} ({val:.2f})"
                    )

        # Update layout
        fig.update_layout(
            title=f'Feature Correlation Matrix ({method.title()})',
            height=self.height,
            xaxis=dict(tickangle=45),
            yaxis=dict(tickangle=0),
            margin=dict(l=100, r=100, t=100, b=100)
        )

        # Add correlation insights
        if annotations:
            fig.add_annotation(
                x=0.02, y=0.02,
                xref="paper", yref="paper",
                text=f"🔍 Key Insights:<br>" + "<br>".join(annotations[:3]),
                showarrow=False,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#2563eb",
                borderwidth=1,
                align="left"
            )

        return fig

    @st.cache_data(ttl=timedelta(seconds=300))
    def create_rolling_correlation_chart(self,
                                       data: pd.DataFrame,
                                       feature1: str,
                                       feature2: str,
                                       window: int = 30) -> go.Figure:
        """Create rolling correlation chart."""
        # Calculate rolling correlation
        rolling_corr = data[feature1].rolling(window=window, min_periods=1).corr(data[feature2])

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data.index,
            y=rolling_corr,
            mode='lines',
            name=f'Rolling Correlation ({window} periods)',
            line=dict(color='#2563eb', width=2)
        ))

        # Add correlation zones
        fig.add_hline(y=0.8, line_dash="dash", line_color="#10b981",
                     annotation_text="Strong Positive")
        fig.add_hline(y=-0.8, line_dash="dash", line_color="#ef4444",
                     annotation_text="Strong Negative")
        fig.add_hline(y=0, line_dash="solid", line_color="#64748b")

        fig.update_layout(
            title=f'Rolling Correlation: {feature1} vs {feature2}',
            xaxis_title='Time',
            yaxis_title='Correlation',
            height=self.height,
            yaxis_range=[-1, 1]
        )

        return fig

class ThreeDimensionalVisualizer:
    """3D visualization components for multi-dimensional data analysis."""

    def __init__(self, height: int = 600):
        self.height = height

    @st.cache_data(ttl=timedelta(seconds=300))
    def create_3d_feature_scatter(self,
                                data: pd.DataFrame,
                                x_feature: str,
                                y_feature: str,
                                z_feature: str,
                                color_feature: str = None,
                                symbol: str = None) -> go.Figure:
        """Create 3D scatter plot for feature analysis."""
        fig = go.Figure()

        # Prepare data
        plot_data = data.dropna(subset=[x_feature, y_feature, z_feature])

        if len(plot_data) == 0:
            # Return empty figure with message
            fig.add_annotation(
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                text="Insufficient data for 3D visualization",
                showarrow=False
            )
            return fig

        # Create 3D scatter
        scatter = go.Scatter3d(
            x=plot_data[x_feature],
            y=plot_data[y_feature],
            z=plot_data[z_feature],
            mode='markers',
            marker=dict(
                size=4,
                color=plot_data[color_feature] if color_feature else None,
                colorscale='Viridis' if color_feature else None,
                showscale=True if color_feature else False,
                opacity=0.7
            ),
            text=plot_data.index.strftime('%Y-%m-%d %H:%M') if hasattr(plot_data.index, 'strftime') else None,
            hovertemplate=(
                f"{x_feature}: %{{x}}<br>"
                f"{y_feature}: %{{y}}<br>"
                f"{z_feature}: %{{z}}<br>"
                f"{'Time' if hasattr(plot_data.index, 'strftime') else 'Index'}: %{{text}}"
            )
        )

        fig.add_trace(scatter)

        # Update layout
        fig.update_layout(
            title=f'3D Feature Analysis: {x_feature} × {y_feature} × {z_feature}',
            scene=dict(
                xaxis_title=x_feature,
                yaxis_title=y_feature,
                zaxis_title=z_feature,
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=self.height,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        return fig

    @st.cache_data(ttl=timedelta(seconds=300))
    def create_3d_surface_plot(self,
                             data: pd.DataFrame,
                             x_feature: str,
                             y_feature: str,
                             z_feature: str) -> go.Figure:
        """Create 3D surface plot for relationship visualization."""
        # Create grid for surface plot
        x_unique = np.sort(data[x_feature].unique())
        y_unique = np.sort(data[y_feature].unique())

        if len(x_unique) < 3 or len(y_unique) < 3:
            # Not enough data for surface plot
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                text="Insufficient data points for surface plot",
                showarrow=False
            )
            return fig

        # Create meshgrid
        X, Y = np.meshgrid(x_unique, y_unique)

        # Interpolate Z values
        from scipy.interpolate import griddata
        points = data[[x_feature, y_feature]].values
        values = data[z_feature].values

        Z = griddata(points, values, (X, Y), method='linear')

        # Create surface plot
        fig = go.Figure(data=[go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale='Viridis',
            showscale=True
        )])

        fig.update_layout(
            title=f'3D Surface: {z_feature} vs {x_feature} × {y_feature}',
            scene=dict(
                xaxis_title=x_feature,
                yaxis_title=y_feature,
                zaxis_title=z_feature
            ),
            height=self.height
        )

        return fig

class PerformanceVisualizer:
    """Advanced performance visualization with risk metrics."""

    def __init__(self, height: int = 600):
        self.height = height

    @st.cache_data(ttl=timedelta(seconds=300))
    def create_risk_parity_chart(self,
                               returns: pd.Series,
                               weights: Dict[str, float] = None) -> go.Figure:
        """Create risk parity visualization."""
        # Calculate risk metrics
        volatility = returns.rolling(30, min_periods=1).std() * np.sqrt(252)
        var_95 = returns.rolling(30, min_periods=1).quantile(0.05)
        cvar_95 = returns[returns <= var_95].rolling(30, min_periods=1).mean()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Volatility', 'Value at Risk (95%)', 'Conditional VaR (95%)', 'Risk Contribution'),
            vertical_spacing=0.1
        )

        # Volatility plot
        fig.add_trace(
            go.Scatter(
                x=volatility.index,
                y=volatility.values * 100,
                mode='lines',
                name='Volatility',
                line=dict(color='#ef4444')
            ),
            row=1, col=1
        )

        # VaR plot
        fig.add_trace(
            go.Scatter(
                x=var_95.index,
                y=var_95.values * 100,
                mode='lines',
                name='VaR 95%',
                line=dict(color='#f59e0b'),
                fill='tozeroy'
            ),
            row=1, col=2
        )

        # CVaR plot
        fig.add_trace(
            go.Scatter(
                x=cvar_95.index,
                y=cvar_95.values * 100,
                mode='lines',
                name='CVaR 95%',
                line=dict(color='#10b981'),
                fill='tozeroy'
            ),
            row=2, col=1
        )

        # Risk contribution (placeholder)
        if weights:
            risk_contrib = {asset: weight * volatility.iloc[-1] for asset, weight in weights.items()}
            fig.add_trace(
                go.Bar(
                    x=list(risk_contrib.keys()),
                    y=list(risk_contrib.values()),
                    name='Risk Contribution',
                    marker_color='#2563eb'
                ),
                row=2, col=2
            )

        fig.update_layout(
            height=self.height,
            showlegend=False,
            title_text="Risk Analysis Dashboard"
        )

        return fig

    @st.cache_data(ttl=timedelta(seconds=300))
    def create_performance_attribution_chart(self,
                                           portfolio_returns: pd.Series,
                                           benchmark_returns: pd.Series,
                                           factors: Dict[str, pd.Series] = None) -> go.Figure:
        """Create performance attribution visualization."""
        # Calculate attribution metrics
        excess_returns = portfolio_returns - benchmark_returns

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Excess Returns', 'Performance Attribution'),
            vertical_spacing=0.1
        )

        # Excess returns
        fig.add_trace(
            go.Scatter(
                x=excess_returns.index,
                y=excess_returns.cumsum() * 100,
                mode='lines',
                name='Cumulative Excess Return',
                line=dict(color='#2563eb', width=2)
            ),
            row=1, col=1
        )

        # Performance attribution (simplified)
        if factors:
            attribution_data = []
            for factor_name, factor_returns in factors.items():
                # Simple factor attribution
                beta = np.cov(portfolio_returns, factor_returns)[0, 1] / np.var(factor_returns)
                attribution = beta * factor_returns
                attribution_data.append((factor_name, attribution.sum()))

            factor_names, attributions = zip(*attribution_data)

            fig.add_trace(
                go.Bar(
                    x=factor_names,
                    y=attributions,
                    name='Factor Attribution',
                    marker_color='#10b981'
                ),
                row=2, col=1
            )

        fig.update_layout(
            height=self.height,
            title_text="Performance Attribution Analysis"
        )

        return fig

class AdvancedVisualizationComponent:
    """Main component for advanced visualizations."""

    def __init__(self):
        self.price_chart = InteractivePriceChart()
        self.correlation_heatmap = CorrelationHeatmap()
        self.three_d_visualizer = ThreeDimensionalVisualizer()
        self.performance_visualizer = PerformanceVisualizer()

    def render_advanced_charts(self, data: pd.DataFrame, symbol: str = "EURUSD"):
        """Render advanced visualization dashboard."""
        st.markdown("## 📊 Advanced Visualizations")

        # Chart type selector
        chart_types = [
            "Multi-Timeframe Analysis",
            "Prediction Confidence",
            "Feature Correlation",
            "3D Feature Analysis",
            "Risk Analysis",
            "Performance Attribution"
        ]

        selected_chart = st.selectbox(
            "Select Visualization Type",
            chart_types,
            key="advanced_chart_selector"
        )

        if selected_chart == "Multi-Timeframe Analysis":
            self._render_multi_timeframe_chart(data, symbol)

        elif selected_chart == "Prediction Confidence":
            self._render_prediction_confidence_chart(data, symbol)

        elif selected_chart == "Feature Correlation":
            self._render_correlation_analysis(data)

        elif selected_chart == "3D Feature Analysis":
            self._render_3d_analysis(data)

        elif selected_chart == "Risk Analysis":
            self._render_risk_analysis(data)

        elif selected_chart == "Performance Attribution":
            self._render_performance_attribution(data)

    def _render_multi_timeframe_chart(self, data: pd.DataFrame, symbol: str):
        """Render multi-timeframe analysis."""
        st.markdown("### 📈 Multi-Timeframe Price Analysis")

        timeframes = st.multiselect(
            "Select Timeframes",
            ["1H", "4H", "1D", "1W"],
            default=["1H", "4H", "1D"],
            key="timeframe_selector"
        )

        if timeframes:
            fig = self.price_chart.create_multi_timeframe_chart(data, symbol, timeframes)
            st.plotly_chart(fig, use_container_width=True, key="multi_timeframe_chart")

            st.markdown("""
            **Analysis Tips:**
            - Compare price action across different timeframes
            - Look for confluence in support/resistance levels
            - Identify trending vs ranging market conditions
            """)

    def _render_prediction_confidence_chart(self, data: pd.DataFrame, symbol: str):
        """Render prediction confidence analysis."""
        st.markdown("### 🎯 Prediction Confidence Analysis")

        # Generate sample predictions (replace with real model predictions)
        predictions = data['close'].shift(-1) * (1 + np.random.normal(0, 0.01, len(data)))
        predictions = predictions.dropna()

        # Generate confidence intervals
        std_dev = data['close'].rolling(20, min_periods=1).std()
        lower_bound = predictions * (1 - 2 * std_dev / data['close'])
        upper_bound = predictions * (1 + 2 * std_dev / data['close'])

        # Align data
        common_index = predictions.index.intersection(data.index)
        aligned_data = data.loc[common_index]
        aligned_predictions = predictions.loc[common_index]
        aligned_lower = lower_bound.loc[common_index]
        aligned_upper = upper_bound.loc[common_index]

        fig = self.price_chart.create_prediction_confidence_chart(
            aligned_data, aligned_predictions,
            (aligned_lower, aligned_upper), symbol
        )
        st.plotly_chart(fig, use_container_width=True, key="prediction_confidence_chart")

    def _render_correlation_analysis(self, data: pd.DataFrame):
        """Render correlation analysis."""
        st.markdown("### 🔗 Feature Correlation Analysis")

        col1, col2 = st.columns([2, 1])

        with col1:
            method = st.selectbox(
                "Correlation Method",
                ["spearman", "pearson", "kendall"],
                key="correlation_method"
            )

            fig = self.correlation_heatmap.create_feature_correlation_heatmap(data, method=method)
            st.plotly_chart(fig, use_container_width=True, key="correlation_heatmap")

        with col2:
            st.markdown("### Correlation Insights")
            st.markdown("""
            **Strong Correlations (>0.8):**
            - May indicate multicollinearity
            - Consider feature selection
            - Useful for portfolio diversification

            **Weak Correlations (<0.3):**
            - Independent features
            - Good for ensemble models
            - May miss important relationships
            """)

            # Feature pair analysis
            if st.button("Analyze Feature Pairs", key="feature_pair_analysis"):
                self._analyze_feature_pairs(data)

    def _render_3d_analysis(self, data: pd.DataFrame):
        """Render 3D feature analysis."""
        st.markdown("### 🔍 3D Feature Analysis")

        # Feature selectors
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['timestamp']]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            x_feature = st.selectbox("X-Axis", numeric_cols, index=0, key="x_feature")

        with col2:
            y_feature = st.selectbox("Y-Axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="y_feature")

        with col3:
            z_feature = st.selectbox("Z-Axis", numeric_cols, index=2 if len(numeric_cols) > 2 else 0, key="z_feature")

        with col4:
            color_feature = st.selectbox("Color", ["None"] + numeric_cols, key="color_feature")
            color_feature = None if color_feature == "None" else color_feature

        if x_feature and y_feature and z_feature:
            fig = self.three_d_visualizer.create_3d_feature_scatter(
                data, x_feature, y_feature, z_feature, color_feature
            )
            st.plotly_chart(fig, use_container_width=True, key="3d_scatter_chart")

            # Analysis tips
            st.markdown("""
            **3D Analysis Tips:**
            - Rotate the plot to explore different angles
            - Look for clusters and patterns
            - Use color coding to identify relationships
            - Hover over points for detailed information
            """)

    def _render_risk_analysis(self, data: pd.DataFrame):
        """Render risk analysis."""
        st.markdown("### ⚠️ Risk Analysis Dashboard")

        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()

            fig = self.performance_visualizer.create_risk_parity_chart(returns)
            st.plotly_chart(fig, use_container_width=True, key="risk_analysis_chart")

            # Risk metrics summary
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                volatility = returns.std() * np.sqrt(252) * 100
                st.metric("Annual Volatility", f"{volatility:.1f}%")

            with col2:
                var_95 = returns.quantile(0.05) * 100
                st.metric("VaR (95%)", f"{var_95:.2f}%")

            with col3:
                sharpe = returns.mean() / returns.std() * np.sqrt(252)
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")

            with col4:
                max_drawdown = (returns.cumsum() - returns.cumsum().expanding().max()).min() * 100
                st.metric("Max Drawdown", f"{max_drawdown:.2f}%")

    def _render_performance_attribution(self, data: pd.DataFrame):
        """Render performance attribution analysis."""
        st.markdown("### 📊 Performance Attribution")

        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()

            # Create benchmark (simple moving average strategy)
            benchmark_returns = (data['close'] > data['close'].rolling(20, min_periods=1).mean()).shift(1) * returns
            benchmark_returns = benchmark_returns.fillna(0)

            fig = self.performance_visualizer.create_performance_attribution_chart(
                returns, benchmark_returns
            )
            st.plotly_chart(fig, use_container_width=True, key="performance_attribution_chart")

    def _analyze_feature_pairs(self, data: pd.DataFrame):
        """Analyze feature pair relationships."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['timestamp']]

        if len(numeric_cols) >= 2:
            # Find highest correlation pairs
            corr_matrix = data[numeric_cols].corr()

            # Get upper triangle
            upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool))

            # Find top correlations
            top_pairs = []
            for i in range(len(upper.columns)):
                for j in range(i+1, len(upper.columns)):
                    if not np.isnan(upper.iloc[i, j]):
                        top_pairs.append((
                            upper.columns[i],
                            upper.columns[j],
                            upper.iloc[i, j]
                        ))

            top_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

            st.markdown("### Top Feature Correlations")
            for feat1, feat2, corr in top_pairs[:5]:
                strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.5 else "Weak"
                st.write(f"• {feat1} ↔ {feat2}: {corr:.3f} ({strength})")