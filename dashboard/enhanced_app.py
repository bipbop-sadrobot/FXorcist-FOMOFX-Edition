"""
Enhanced Main dashboard application for the Forex AI monitoring system.
Implements modular components with advanced caching strategies.
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

class EnhancedDashboardApp:
    """Enhanced dashboard application with advanced caching and performance optimizations."""

    def __init__(self):
        """Initialize dashboard with components and enhanced data loader."""
        self.data_loader = EnhancedDataLoader()

        # Initialize components
        self.predictions = PredictionsVisualization(
            ComponentConfig(
                title="Price Predictions & Feature Analysis",
                description="Model predictions and feature importance analysis",
                height=600
            )
        )

        self.performance = PerformanceMetrics(
            ComponentConfig(
                title="Trading Performance",
                description="Comprehensive performance metrics and analysis",
                height=800
            )
        )

        self.system = SystemMonitor(
            ComponentConfig(
                title="System Health & Resources",
                description="System monitoring and resource usage",
                height=600
            )
        )

        self.pipeline_monitor = PipelineMonitor(
            ComponentConfig(
                title="Pipeline Monitoring",
                description="Real-time monitoring of training pipelines and system status",
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
        """Load and distribute data to components with caching."""
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

                # Update performance component
                self.performance.update({
                    'returns': df['close'].pct_change(),
                    'predictions': eval_results[-1].predictions if eval_results else None,
                    'metrics': [r.metrics for r in eval_results] if eval_results else []
                })

            self.last_refresh = datetime.now()

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", exc_info=True)
            st.error("An error occurred while loading data")

    def run(self):
        """Run the enhanced dashboard application."""
        st.set_page_config(
            page_title="Forex AI Dashboard (Enhanced)",
            page_icon="🚀",
            layout="wide"
        )

        # Header with performance info
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("🚀 Forex AI Dashboard (Enhanced)")
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
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Predictions & Features",
                "📊 Performance Analysis",
                "🖥️ System Monitor",
                "🔧 Pipeline Monitor"
            ])

            # Render components in tabs
            with tab1:
                self.predictions.render()

            with tab2:
                self.performance.render()

            with tab3:
                self.system.render()

            with tab4:
                self.pipeline_monitor.render()

        except Exception as e:
            logger.error(f"Dashboard error: {str(e)}", exc_info=True)
            st.error(
                "An error occurred while updating the dashboard. "
                "Please check the logs for details."
            )

if __name__ == "__main__":
    dashboard = EnhancedDashboardApp()
    dashboard.run()