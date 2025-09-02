#!/usr/bin/env python3
"""
Unified Forex AI Dashboard
Combines features from both dashboard versions:
- Comprehensive analytics and visualization
- Enhanced performance and caching
- Advanced risk metrics and portfolio analysis
- Optimized UI and user experience
"""

import streamlit as st
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.append('..')

from forex_ai_dashboard.pipeline.evaluation_metrics import EvaluationMetrics
from forex_ai_dashboard.models.model_hierarchy import ModelHierarchy

from components import ComponentConfig
from components.predictions import PredictionsVisualization
from components.performance import PerformanceMetrics
from components.system_status import SystemMonitor
from components.pipeline_monitor import PipelineMonitor
from components.quantstats_tearsheet import QuantStatsTearsheet
from components.training_monitor import TrainingMonitor
from utils.enhanced_data_loader import EnhancedDataLoader

# Ensure logs directory exists
Path("logs").mkdir(parents=True, exist_ok=True)

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

class UnifiedDashboardApp:
    """Unified dashboard combining comprehensive analytics with enhanced performance."""

    def __init__(self):
        """Initialize dashboard with all components and enhanced features."""
        self.data_loader = EnhancedDataLoader()

        # Initialize components
        self.predictions = PredictionsVisualization(
            ComponentConfig(
                title="📈 Price Predictions & Feature Analysis",
                description="Model predictions and feature importance analysis",
                height=600
            )
        )

        self.performance = PerformanceMetrics(
            ComponentConfig(
                title="📊 Trading Performance",
                description="Comprehensive performance metrics and analysis",
                height=800
            )
        )

        self.quantstats_portfolio = QuantStatsTearsheet(
            ComponentConfig(
                title="📈 Advanced Portfolio Analytics",
                description="Comprehensive portfolio analysis with QuantStats",
                height=900
            )
        )

        self.system = SystemMonitor(
            ComponentConfig(
                title="💻 System Health & Resources",
                description="System monitoring and resource usage",
                height=600
            )
        )

        self.pipeline_monitor = PipelineMonitor(
            ComponentConfig(
                title="🔄 Pipeline Monitoring",
                description="Real-time monitoring of training pipelines and system status",
                height=800
            )
        )

        self.training_monitor = TrainingMonitor(
            ComponentConfig(
                title="🎯 Training System",
                description="Enhanced training system with model comparison",
                height=800
            )
        )

        # Track data refresh
        self.last_refresh = None
        self.refresh_interval = 300  # 5 minutes

        # Preload common data
        self._preload_data()

    def _preload_data(self):
        """Preload commonly accessed data into cache."""
        try:
            logger.info("Preloading common data for enhanced performance...")
            self.data_loader.preload_common_data()
            logger.info("Data preloading completed")
        except Exception as e:
            logger.warning(f"Failed to preload data: {e}")

    def _should_refresh(self) -> bool:
        """Check if data should be refreshed."""
        if not self.last_refresh:
            return True

        elapsed = (datetime.now() - self.last_refresh).total_seconds()
        return elapsed > self.refresh_interval

    def _load_data(self) -> None:
        """Load and distribute data to components with enhanced caching."""
        try:
            # Load forex data
            df, issues = self.data_loader.load_forex_data(
                timeframe=st.session_state.get('timeframe', '1H')
            )

            if issues:
                for issue in issues:
                    st.warning(f"Data issue: {issue}")

            # Load evaluation results
            eval_results, eval_issues = self.data_loader.load_evaluation_results()

            if eval_issues:
                for issue in eval_issues:
                    st.warning(f"Evaluation issue: {issue}")

            # Load model hierarchy
            model, model_issues = self.data_loader.load_model_hierarchy()

            if model_issues:
                for issue in model_issues:
                    st.warning(f"Model issue: {issue}")

            # Update components with new data
            if not df.empty:
                # Update predictions component
                self.predictions.update({
                    'df': df,
                    'predictions': eval_results[-1].predictions if eval_results else None,
                    'std_dev': eval_results[-1].prediction_std if eval_results else None,
                    'model': model,
                    'features': df
                })

                # Update quantstats portfolio component
                self.quantstats_portfolio.update({
                    'returns': df['close'].pct_change(),
                    'portfolio_name': 'Forex Trading Portfolio'
                })

                # Update performance component
                self.performance.update({
                    'returns': df['close'].pct_change(),
                    'predictions': eval_results[-1].predictions if eval_results else None,
                    'metrics': [r.metrics for r in eval_results] if eval_results else []
                })

                # Store returns for risk metrics
                st.session_state.returns = df['close'].pct_change()

            self.last_refresh = datetime.now()

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", exc_info=True)
            st.error("An error occurred while loading data")

    def run(self):
        """Run the unified dashboard application."""
        st.set_page_config(
            page_title="Forex AI Dashboard (Unified)",
            page_icon="🚀",
            layout="wide"
        )

        # Header with performance stats
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("🚀 Forex AI Trading Dashboard")
        with col2:
            if st.button("📊 Performance Stats"):
                stats = self.data_loader.get_performance_stats()
                st.json(stats)

        try:
            # Sidebar controls
            st.sidebar.header("⚙️ Settings")

            # Timeframe selection
            timeframe = st.sidebar.selectbox(
                "Timeframe",
                ["1M", "5M", "15M", "1H", "4H", "1D"],
                index=3  # Default to 1H
            )

            if 'timeframe' not in st.session_state or st.session_state.timeframe != timeframe:
                st.session_state.timeframe = timeframe
                self.data_loader.clear_cache()

            # Auto-refresh toggle
            auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)

            # Manual refresh button
            if st.sidebar.button("🔄 Refresh Data") or (auto_refresh and self._should_refresh()):
                self.data_loader.clear_cache()
                self._load_data()

            # Load initial data if needed
            if self._should_refresh():
                self._load_data()

            # Main content tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📈 Predictions & Features",
                "📊 Performance Analysis",
                "📈 Portfolio Analytics",
                "🎯 Training System",
                "💻 System Monitor",
                "🔄 Pipeline Monitor"
            ])

            # Render components in tabs
            with tab1:
                self.predictions.render()

            with tab2:
                # Performance metrics with enhanced analytics
                col1, col2 = st.columns([2, 1])
                with col1:
                    self.performance.render()
                with col2:
                    if 'returns' in st.session_state:
                        metrics, issues = self.data_loader.calculate_risk_metrics(
                            st.session_state.returns
                        )
                        if metrics and not issues:
                            st.subheader("📊 Risk Metrics")
                            for key, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    st.metric(
                                        key.replace('_', ' ').title(),
                                        f"{value:.2%}"
                                    )

            with tab3:
                self.quantstats_portfolio.render()

            with tab4:
                self.training_monitor.render()

            with tab5:
                self.system.render()

            with tab6:
                self.pipeline_monitor.render()

        except Exception as e:
            logger.error(f"Dashboard error: {str(e)}", exc_info=True)
            st.error(
                "An error occurred while updating the dashboard. "
                "Please check the logs for details."
            )

if __name__ == "__main__":
    dashboard = UnifiedDashboardApp()
    dashboard.run()