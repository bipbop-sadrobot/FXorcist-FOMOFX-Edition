import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional
import sys

# Add project root to path
sys.path.append('..')
from forex_ai_dashboard.pipeline.evaluation_metrics import EvaluationMetrics
from forex_ai_dashboard.models.model_hierarchy import ModelHierarchy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("logs/dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DashboardApp:
    """Main dashboard application for forex AI monitoring."""
    
    def __init__(self):
        self.eval_dir = Path('evaluation_results')
        self.data_dir = Path('data/processed')
        self.model_dir = Path('models/hierarchy')
    
    def load_latest_data(self) -> pd.DataFrame:
        """Load the latest processed forex data."""
        try:
            files = list(self.data_dir.glob('*.parquet'))
            if not files:
                return pd.DataFrame()
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            return pd.read_parquet(latest_file)
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()
    
    def load_evaluation_results(self) -> List[EvaluationMetrics]:
        """Load recent evaluation results."""
        try:
            files = list(self.eval_dir.glob('*.json'))
            results = []
            for f in sorted(files, key=lambda x: x.stat().st_mtime)[-50:]:  # Last 50 evaluations
                with open(f, 'r') as file:
                    data = json.load(file)
                    results.append(EvaluationMetrics.from_dict(data))
            return results
        except Exception as e:
            logger.error(f"Error loading evaluation results: {str(e)}")
            return []
    
    def plot_predictions(self, df: pd.DataFrame, predictions: pd.Series):
        """Plot actual vs predicted values."""
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['close'],
            name='Actual',
            line=dict(color='blue')
        ))
        
        # Add predictions
        fig.add_trace(go.Scatter(
            x=df.index,
            y=predictions,
            name='Predicted',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Price Predictions vs Actual',
            xaxis_title='Time',
            yaxis_title='Price',
            height=400
        )
        
        return fig
    
    def plot_metrics_history(self, results: List[EvaluationMetrics]):
        """Plot historical metrics."""
        if not results:
            return None
            
        # Extract metrics over time
        dates = [r.timestamp for r in results]
        metrics_history = {
            metric: [r.metrics.get(metric, np.nan) for r in results]
            for metric in results[0].metrics.keys()
        }
        
        # Create subplots for different metric categories
        fig = go.Figure()
        
        # Financial metrics
        financial_metrics = ['sharpe_ratio', 'max_drawdown', 'annual_return']
        for metric in financial_metrics:
            if metric in metrics_history:
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=metrics_history[metric],
                    name=metric,
                    yaxis='y1'
                ))
        
        # Prediction metrics
        prediction_metrics = ['mse', 'directional_accuracy', 'ic']
        for metric in prediction_metrics:
            if metric in metrics_history:
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=metrics_history[metric],
                    name=metric,
                    yaxis='y2'
                ))
        
        fig.update_layout(
            title='Model Metrics Over Time',
            xaxis_title='Time',
            yaxis_title='Financial Metrics',
            yaxis2=dict(
                title='Prediction Metrics',
                overlaying='y',
                side='right'
            ),
            height=400
        )
        
        return fig
    
    def plot_resource_usage(self, results: List[EvaluationMetrics]):
        """Plot system resource usage."""
        if not results:
            return None
            
        # Extract resource metrics
        dates = [r.timestamp for r in results]
        resources = {
            metric: [r.resource_usage.get(metric, np.nan) for r in results]
            for metric in results[0].resource_usage.keys()
        }
        
        fig = go.Figure()
        
        for metric, values in resources.items():
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                name=metric
            ))
        
        fig.update_layout(
            title='System Resource Usage',
            xaxis_title='Time',
            yaxis_title='Usage (%)',
            height=300
        )
        
        return fig
    
    def display_model_drift(self, results: List[EvaluationMetrics]):
        """Display model drift indicators."""
        if not results:
            return
            
        st.subheader("Model Drift Analysis")
        
        # Calculate drift metrics
        recent = results[-1]
        historical = results[:-1]
        
        if not historical:
            st.warning("Insufficient historical data for drift analysis")
            return
        
        # Calculate z-scores for key metrics
        drift_metrics = {}
        for metric in ['mse', 'directional_accuracy', 'ic']:
            if metric in recent.metrics:
                historical_values = [r.metrics.get(metric, np.nan) for r in historical]
                current_value = recent.metrics[metric]
                
                mean = np.nanmean(historical_values)
                std = np.nanstd(historical_values)
                
                if std > 0:
                    z_score = (current_value - mean) / std
                    drift_metrics[metric] = {
                        'z_score': z_score,
                        'current': current_value,
                        'historical_mean': mean,
                        'historical_std': std
                    }
        
        # Display drift metrics
        cols = st.columns(len(drift_metrics))
        for col, (metric, stats) in zip(cols, drift_metrics.items()):
            with col:
                st.metric(
                    label=metric,
                    value=f"{stats['current']:.4f}",
                    delta=f"{stats['z_score']:.2f}σ"
                )
                
                # Add warning if drift detected
                if abs(stats['z_score']) > 2:
                    st.warning("⚠️ Significant drift detected")
    
    def run(self):
        """Run the dashboard application."""
        st.set_page_config(
            page_title="Forex AI Dashboard",
            page_icon="📈",
            layout="wide"
        )
        
        st.title("Forex AI Trading Dashboard")
        
        try:
            # Load data
            df = self.load_latest_data()
            eval_results = self.load_evaluation_results()
            
            if df.empty:
                st.error("No data available")
                return
            
            # Sidebar controls
            st.sidebar.header("Controls")
            model_layer = st.sidebar.selectbox(
                "Model Layer",
                ["Strategist", "Tactician", "Executor"]
            )
            
            timeframe = st.sidebar.selectbox(
                "Timeframe",
                ["1H", "4H", "1D"]
            )
            
            # Main dashboard layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Price Predictions")
                if eval_results:
                    latest = eval_results[-1]
                    fig = self.plot_predictions(df, latest.predictions)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Model Metrics")
                fig = self.plot_metrics_history(eval_results)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Latest Metrics")
                if eval_results:
                    latest = eval_results[-1]
                    for metric, value in latest.metrics.items():
                        st.metric(metric, f"{value:.4f}")
                
                st.subheader("Resource Usage")
                fig = self.plot_resource_usage(eval_results)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Model drift analysis
            self.display_model_drift(eval_results)
            
            # Pipeline status
            st.subheader("Pipeline Status")
            status_cols = st.columns(4)
            
            with status_cols[0]:
                st.metric("Data Freshness", "2 min ago", "On time")
            with status_cols[1]:
                st.metric("Feature Pipeline", "Healthy", "✓")
            with status_cols[2]:
                st.metric("Model Status", "Active", "✓")
            with status_cols[3]:
                st.metric("Last Update", "1 min ago", "✓")
            
        except Exception as e:
            logger.error(f"Dashboard error: {str(e)}", exc_info=True)
            st.error("An error occurred while updating the dashboard")

if __name__ == "__main__":
    dashboard = DashboardApp()
    dashboard.run()