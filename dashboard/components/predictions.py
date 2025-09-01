"""
Predictions visualization component for the Forex AI dashboard.
Handles price predictions, feature importance, and related visualizations.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta

from . import VisualizationComponent, ComponentConfig, cache_data
from forex_ai_dashboard.pipeline.evaluation_metrics import EvaluationMetrics
from forex_ai_dashboard.models.model_hierarchy import ModelHierarchy

class PredictionsVisualization(VisualizationComponent):
    """Component for visualizing model predictions and feature importance."""
    
    def __init__(self, config: ComponentConfig):
        """Initialize predictions visualization component."""
        super().__init__(config)
        self.timeframes = ["1H", "4H", "1D"]
        self.selected_timeframe = "1H"
        self.show_confidence = True
        self.show_features = True
    
    @cache_data(ttl_seconds=300)
    def _calculate_confidence_intervals(
        self,
        predictions: pd.Series,
        std_dev: pd.Series,
        confidence_level: float = 0.95
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate confidence intervals for predictions."""
        z_score = 1.96  # 95% confidence level
        lower_bound = predictions - z_score * std_dev
        upper_bound = predictions + z_score * std_dev
        return lower_bound, upper_bound
    
    @cache_data(ttl_seconds=300)
    def _calculate_feature_importance(
        self,
        model: ModelHierarchy,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate and format feature importance scores."""
        importance_scores = model.get_feature_importance()
        
        # Create DataFrame with feature importance
        importance_df = pd.DataFrame({
            'Feature': list(importance_scores.keys()),
            'Importance': list(importance_scores.values())
        }).sort_values('Importance', ascending=True)
        
        return importance_df
    
    def create_figure(self, data: Dict) -> go.Figure:
        """Create prediction visualization figure."""
        if not data or 'df' not in data or 'predictions' not in data:
            return None
            
        df = data['df']
        predictions = data['predictions']
        
        # Create main figure
        fig = go.Figure()
        
        # Add actual price
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['close'],
            name='Actual Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add predictions
        fig.add_trace(go.Scatter(
            x=df.index,
            y=predictions,
            name='Predicted Price',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Add confidence intervals if enabled
        if self.show_confidence and 'std_dev' in data:
            lower_bound, upper_bound = self._calculate_confidence_intervals(
                predictions,
                data['std_dev']
            )
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=upper_bound,
                name='Upper Bound',
                line=dict(color='rgba(255,0,0,0.2)', width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=lower_bound,
                name='Lower Bound',
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)',
                line=dict(color='rgba(255,0,0,0.2)', width=0),
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Price Predictions ({self.selected_timeframe})',
            xaxis_title='Time',
            yaxis_title='Price',
            height=self.config.height,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def create_feature_importance_figure(self, data: Dict) -> go.Figure:
        """Create feature importance visualization."""
        if not data or 'model' not in data or 'features' not in data:
            return None
            
        importance_df = self._calculate_feature_importance(
            data['model'],
            data['features']
        )
        
        fig = go.Figure(go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h'
        ))
        
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            height=min(400, len(importance_df) * 20 + 100),
            showlegend=False
        )
        
        return fig
    
    def render(self) -> None:
        """Render predictions visualization component."""
        st.subheader(self.config.title)
        
        # Controls
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            self.selected_timeframe = st.selectbox(
                "Timeframe",
                self.timeframes,
                index=self.timeframes.index(self.selected_timeframe)
            )
        
        with col2:
            self.show_confidence = st.checkbox(
                "Show Confidence Intervals",
                value=self.show_confidence
            )
        
        with col3:
            self.show_features = st.checkbox(
                "Show Feature Importance",
                value=self.show_features
            )
        
        # Main predictions plot
        fig = self.create_figure(self._cache.get('data'))
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance plot
        if self.show_features:
            fig_importance = self.create_feature_importance_figure(self._cache.get('data'))
            if fig_importance:
                st.plotly_chart(fig_importance, use_container_width=True)
    
    def update(self, data: Dict) -> None:
        """Update component with new data."""
        self._cache['data'] = data
        self.clear_cache()  # Clear cached calculations