#!/usr/bin/env python3
"""
Advanced System Dashboard for FXorcist

This dashboard provides a comprehensive web interface for monitoring,
managing, and optimizing the FXorcist AI system with advanced analytics,
auto-scaling, and real-time performance insights.

Features:
- Real-time system monitoring and health checks
- Advanced analytics and trend analysis
- Auto-scaling management and recommendations
- Performance optimization suggestions
- Alert management and notifications
- Configuration management interface
- Historical data analysis and forecasting
- Interactive charts and visualizations

Author: FXorcist Development Team
Version: 2.0
Date: September 2, 2025
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging

# Data science and visualization
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# Local imports
sys.path.append(str(Path(__file__).parent))
from forex_ai_dashboard.utils.health_checker import HealthChecker
from forex_ai_dashboard.utils.config_manager import ConfigurationManager
from forex_ai_dashboard.utils.advanced_auto_scaler import AdvancedAutoScaler, ScalingMetrics
from forex_ai_dashboard.utils.advanced_monitor import AdvancedMonitor, PerformanceMetrics
from scripts.interactive_dashboard_launcher import InteractiveDashboardLauncher
from scripts.enhanced_dashboard_manager import EnhancedDashboardManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit
st.set_page_config(
    page_title="FXorcist Advanced System Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.fxorcist.com',
        'Report a bug': 'https://github.com/fxorcist/issues',
        'About': 'FXorcist Advanced System Dashboard v2.0'
    }
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(45deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .alert-card {
        border-left: 4px solid #ff6b6b;
        background-color: #fff5f5;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .success-card {
        border-left: 4px solid #51cf66;
        background-color: #f8fff8;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .warning-card {
        border-left: 4px solid #ffd43b;
        background-color: #fffef8;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .info-card {
        border-left: 4px solid #74c0fc;
        background-color: #f8f9ff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #1e3c72;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedSystemDashboard:
    """Advanced system dashboard with comprehensive monitoring and control"""

    def __init__(self):
        # Initialize core components
        self.health_checker = HealthChecker()
        self.config_manager = ConfigurationManager()
        self.monitor = AdvancedMonitor("system")
        self.auto_scaler = AdvancedAutoScaler("dashboard_main")
        self.launcher = InteractiveDashboardLauncher()
        self.manager = EnhancedDashboardManager()

        # Dashboard state
        self.refresh_interval = 30  # seconds
        self.last_refresh = datetime.min

        # Initialize session state
        if 'dashboard_initialized' not in st.session_state:
            st.session_state.dashboard_initialized = False
            st.session_state.selected_instance = "dashboard_main"
            st.session_state.alert_filters = {"critical": True, "warning": True, "info": True}

    def initialize_system(self):
        """Initialize all system components"""
        try:
            # Start monitoring systems
            self.monitor.start_monitoring()
            self.auto_scaler.start_auto_scaling()

            # Initialize with some sample data for demonstration
            self._initialize_sample_data()

            st.session_state.dashboard_initialized = True
            logger.info("Advanced system dashboard initialized successfully")

        except Exception as e:
            st.error(f"Failed to initialize system: {e}")
            logger.error(f"System initialization error: {e}")

    def _initialize_sample_data(self):
        """Initialize with sample data for demonstration"""
        # Add some sample metrics to the systems
        for i in range(50):
            # Create sample scaling metrics
            scaling_metrics = ScalingMetrics(
                timestamp=datetime.now() - timedelta(minutes=50-i),
                cpu_percent=40 + 30 * np.sin(i * 0.2) + np.random.normal(0, 5),
                memory_percent=50 + 25 * np.cos(i * 0.15) + np.random.normal(0, 3),
                disk_percent=45 + np.random.normal(0, 2),
                network_connections=15 + np.random.normal(0, 3),
                response_time_ms=1200 + 400 * np.sin(i * 0.1) + np.random.normal(0, 50),
                error_rate=2.0 + np.random.normal(0, 0.5),
                throughput=45 + 15 * np.cos(i * 0.12) + np.random.normal(0, 5),
                active_users=20 + 10 * np.sin(i * 0.08) + np.random.normal(0, 2)
            )
            self.auto_scaler.add_metrics(scaling_metrics)

            # Create sample performance metrics
            perf_metrics = PerformanceMetrics(
                timestamp=datetime.now() - timedelta(minutes=50-i),
                system_metrics={
                    'cpu_percent': scaling_metrics.cpu_percent,
                    'memory_percent': scaling_metrics.memory_percent,
                    'disk_percent': scaling_metrics.disk_percent,
                    'network_connections': scaling_metrics.network_connections,
                    'fxorcist_processes': 3 + np.random.randint(0, 3)
                },
                application_metrics={
                    'health_score': 0.85 + 0.1 * np.sin(i * 0.1) + np.random.normal(0, 0.05),
                    'dashboard_response_time': scaling_metrics.response_time_ms,
                    'dashboard_active_users': scaling_metrics.active_users
                },
                business_metrics={
                    'data_files_count': 150 + np.random.randint(-10, 10),
                    'model_files_count': 5 + np.random.randint(0, 3),
                    'error_rate_percent': scaling_metrics.error_rate
                }
            )
            self.monitor.metrics_history.append(perf_metrics)

    def render_header(self):
        """Render the main dashboard header"""
        st.markdown('<h1 class="main-header">🚀 FXorcist Advanced System Dashboard</h1>', unsafe_allow_html=True)

        # System status overview
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            health_status = self.health_checker.check_all_components()
            status_color = "🟢" if health_status['status'] == 'healthy' else "🔴"
            st.metric("System Health", f"{status_color} {health_status['status'].title()}")

        with col2:
            instance_count = len(self.launcher.instances)
            running_count = sum(1 for i in self.launcher.instances.values() if i['status'] == 'running')
            st.metric("Active Instances", f"{running_count}/{instance_count}")

        with col3:
            if self.monitor.metrics_history:
                latest = self.monitor.metrics_history[-1]
                cpu_usage = latest.system_metrics.get('cpu_percent', 0)
                st.metric("CPU Usage", ".1f")

        with col4:
            active_alerts = len([a for a in self.monitor.alerts.values() if not a.resolved_at])
            st.metric("Active Alerts", active_alerts)

        # Quick actions
        st.markdown("---")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                self.last_refresh = datetime.min  # Force refresh
                st.rerun()

        with col2:
            if st.button("🚀 Start All", use_container_width=True):
                self._start_all_instances()
                st.success("Starting all instances...")

        with col3:
            if st.button("⏹️ Stop All", use_container_width=True):
                self._stop_all_instances()
                st.success("Stopping all instances...")

        with col4:
            if st.button("⚡ Optimize", use_container_width=True):
                recommendations = self.monitor.generate_optimization_recommendations()
                if recommendations:
                    st.session_state.optimization_recommendations = recommendations
                    st.success(f"Generated {len(recommendations)} optimization recommendations")

        with col5:
            if st.button("📊 Analytics", use_container_width=True):
                st.session_state.show_analytics = True

    def _start_all_instances(self):
        """Start all dashboard instances"""
        for instance_id in list(self.launcher.instances.keys()):
            if self.launcher.instances[instance_id]['status'] != 'running':
                self.launcher.start_instance(instance_id)

    def _stop_all_instances(self):
        """Stop all dashboard instances"""
        for instance_id in list(self.launcher.instances.keys()):
            if self.launcher.instances[instance_id]['status'] == 'running':
                self.launcher.stop_instance(instance_id)

    def render_system_overview_tab(self):
        """Render system overview tab"""
        st.header("📊 System Overview")

        # Real-time metrics
        st.subheader("Real-time Metrics")

        if self.monitor.metrics_history:
            latest = self.monitor.metrics_history[-1]

            # System metrics cards
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                cpu = latest.system_metrics.get('cpu_percent', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>CPU Usage</h3>
                    <h2>{cpu:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                memory = latest.system_metrics.get('memory_percent', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Memory Usage</h3>
                    <h2>{memory:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                disk = latest.system_metrics.get('disk_percent', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Disk Usage</h3>
                    <h2>{disk:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                processes = latest.system_metrics.get('fxorcist_processes', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Active Processes</h3>
                    <h2>{processes}</h2>
                </div>
                """, unsafe_allow_html=True)

        # System health components
        st.subheader("Component Health Status")

        health_status = self.health_checker.check_all_components()
        components = health_status.get('components', {})

        cols = st.columns(len(components))
        for i, (component, status) in enumerate(components.items()):
            with cols[i]:
                if status['status'] == 'healthy':
                    st.success(f"{component.title()}")
                elif status['status'] == 'warning':
                    st.warning(f"{component.title()}")
                else:
                    st.error(f"{component.title()}")

        # Performance trends
        st.subheader("Performance Trends (Last 24 Hours)")

        if len(self.monitor.metrics_history) > 10:
            # Prepare data for plotting
            timestamps = []
            cpu_values = []
            memory_values = []
            response_times = []

            for metric in list(self.monitor.metrics_history)[-100:]:  # Last 100 data points
                timestamps.append(metric.timestamp)
                cpu_values.append(metric.system_metrics.get('cpu_percent', 0))
                memory_values.append(metric.system_metrics.get('memory_percent', 0))
                response_times.append(metric.application_metrics.get('dashboard_response_time', 0))

            # Create subplot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('CPU Usage', 'Memory Usage', 'Response Time', 'System Health'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": True}]]
            )

            # CPU usage
            fig.add_trace(
                go.Scatter(x=timestamps, y=cpu_values, name='CPU %', line=dict(color='#1f77b4')),
                row=1, col=1
            )

            # Memory usage
            fig.add_trace(
                go.Scatter(x=timestamps, y=memory_values, name='Memory %', line=dict(color='#ff7f0e')),
                row=1, col=2
            )

            # Response time
            fig.add_trace(
                go.Scatter(x=timestamps, y=response_times, name='Response Time (ms)', line=dict(color='#2ca02c')),
                row=2, col=1
            )

            # System health score
            health_scores = [m.application_metrics.get('health_score', 1.0) for m in list(self.monitor.metrics_history)[-100:]]
            fig.add_trace(
                go.Scatter(x=timestamps, y=health_scores, name='Health Score', line=dict(color='#d62728')),
                row=2, col=2
            )

            # Update layout
            fig.update_layout(height=600, showlegend=False)
            fig.update_xaxes(title_text="Time")
            fig.update_yaxes(title_text="Percentage", row=1, col=1)
            fig.update_yaxes(title_text="Percentage", row=1, col=2)
            fig.update_yaxes(title_text="Milliseconds", row=2, col=1)
            fig.update_yaxes(title_text="Score", row=2, col=2)

            st.plotly_chart(fig, use_container_width=True)

    def render_instances_tab(self):
        """Render instances management tab"""
        st.header("🎛️ Instance Management")

        # Instance control panel
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Dashboard Instances")

            if not self.launcher.instances:
                st.info("No dashboard instances configured. Create one below.")
            else:
                for instance_id, instance in self.launcher.instances.items():
                    with st.expander(f"{instance['name']} ({instance['type']})", expanded=True):
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                        with col1:
                            status_color = {
                                'running': '🟢',
                                'stopped': '🔴',
                                'starting': '🟡',
                                'error': '🔴'
                            }.get(instance['status'], '⚪')

                            st.write(f"**Status:** {status_color} {instance['status'].title()}")
                            st.write(f"**Port:** {instance['port']}")

                            if instance.get('start_time'):
                                runtime = datetime.now() - instance['start_time']
                                st.write(f"**Runtime:** {str(runtime).split('.')[0]}")

                        with col2:
                            if instance['status'] == 'running':
                                if st.button("🛑 Stop", key=f"stop_{instance_id}"):
                                    if self.launcher.stop_instance(instance_id):
                                        st.success("Instance stopped!")
                                        time.sleep(1)
                                        st.rerun()
                            else:
                                if st.button("▶️ Start", key=f"start_{instance_id}"):
                                    if self.launcher.start_instance(instance_id):
                                        st.success("Instance started!")
                                        time.sleep(2)
                                        st.rerun()

                        with col3:
                            if instance['status'] == 'running':
                                st.markdown(f"[🌐 Open Dashboard](http://localhost:{instance['port']})")

                        with col4:
                            if st.button("🗑️ Delete", key=f"delete_{instance_id}"):
                                if instance['status'] == 'running':
                                    self.launcher.stop_instance(instance_id)
                                del self.launcher.instances[instance_id]
                                self.launcher.save_config()
                                st.success("Instance deleted!")
                                st.rerun()

        with col2:
            st.subheader("Quick Stats")

            total_instances = len(self.launcher.instances)
            running_instances = sum(1 for i in self.launcher.instances.values() if i['status'] == 'running')
            error_instances = sum(1 for i in self.launcher.instances.values() if i['status'] == 'error')

            st.metric("Total Instances", total_instances)
            st.metric("Running", running_instances)
            st.metric("Errors", error_instances)

            # System resources
            if self.monitor.metrics_history:
                latest = self.monitor.metrics_history[-1]
                st.metric("CPU Usage", ".1f")
                st.metric("Memory Usage", ".1f")

        # Create new instance
        st.markdown("---")
        st.subheader("Create New Instance")

        with st.form("create_instance"):
            col1, col2 = st.columns(2)

            with col1:
                name = st.text_input("Instance Name", placeholder="My Dashboard")
                dashboard_type = st.selectbox("Type", ["main", "training", "custom"])

            with col2:
                port = st.number_input("Port", min_value=1024, max_value=65535, value=8501)

            submitted = st.form_submit_button("Create Instance", use_container_width=True)

            if submitted and name:
                instance_id = self.launcher.create_instance(name, dashboard_type, port)
                st.success(f"Created instance '{name}' with ID: {instance_id}")
                time.sleep(1)
                st.rerun()

    def render_auto_scaling_tab(self):
        """Render auto-scaling management tab"""
        st.header("⚡ Auto-Scaling Management")

        # Current scaling status
        st.subheader("Scaling Status")

        scaling_decision = self.auto_scaler.make_scaling_decision()
        analytics = self.auto_scaler.get_scaling_analytics()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Current Instances", analytics['current_instances'])

        with col2:
            st.metric("Target Instances", scaling_decision.recommended_instances)

        with col3:
            confidence_color = "🟢" if scaling_decision.confidence > 0.7 else "🟡" if scaling_decision.confidence > 0.5 else "🔴"
            st.metric("Confidence", f"{confidence_color} {scaling_decision.confidence:.2f}")

        with col4:
            action_color = {
                'scale_up': '🟢',
                'scale_down': '🟡',
                'no_action': '⚪'
            }.get(scaling_decision.action, '⚪')
            st.metric("Recommended Action", f"{action_color} {scaling_decision.action.replace('_', ' ').title()}")

        # Scaling decision reasoning
        if scaling_decision.reasoning:
            st.subheader("Decision Reasoning")
            for reason in scaling_decision.reasoning:
                st.info(reason)

        # Scaling metrics visualization
        st.subheader("Scaling Metrics (Last Hour)")

        if len(self.auto_scaler.metrics_history) > 5:
            recent_metrics = list(self.auto_scaler.metrics_history)[-60:]  # Last hour (assuming 1 min intervals)

            timestamps = [m.timestamp for m in recent_metrics]
            cpu_values = [m.cpu_percent for m in recent_metrics]
            memory_values = [m.memory_percent for m in recent_metrics]
            response_times = [m.response_time_ms for m in recent_metrics]
            load_values = [self.auto_scaler.calculate_load_from_metric(m) for m in recent_metrics]

            # Create metrics chart
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('CPU Usage', 'Memory Usage', 'Response Time', 'System Load'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )

            fig.add_trace(go.Scatter(x=timestamps, y=cpu_values, name='CPU %', line=dict(color='#1f77b4')), row=1, col=1)
            fig.add_trace(go.Scatter(x=timestamps, y=memory_values, name='Memory %', line=dict(color='#ff7f0e')), row=1, col=2)
            fig.add_trace(go.Scatter(x=timestamps, y=response_times, name='Response Time (ms)', line=dict(color='#2ca02c')), row=2, col=1)
            fig.add_trace(go.Scatter(x=timestamps, y=load_values, name='System Load', line=dict(color='#d62728')), row=2, col=2)

            # Add threshold lines
            fig.add_hline(y=self.auto_scaler.thresholds.cpu_high, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=self.auto_scaler.thresholds.memory_high, line_dash="dash", line_color="red", row=1, col=2)
            fig.add_hline(y=self.auto_scaler.thresholds.response_time_high, line_dash="dash", line_color="red", row=2, col=1)

            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Scaling history
        st.subheader("Scaling History")

        if self.auto_scaler.scaling_decisions:
            history_data = []
            for decision in list(self.auto_scaler.scaling_decisions)[-10:]:  # Last 10 decisions
                history_data.append({
                    'Time': decision.timestamp.strftime('%H:%M:%S'),
                    'Action': decision.action.replace('_', ' ').title(),
                    'Confidence': ".2f",
                    'Instances': decision.recommended_instances,
                    'Reasoning': '; '.join(decision.reasoning[:2])  # First 2 reasons
                })

            st.dataframe(pd.DataFrame(history_data))

        # Scaling configuration
        st.subheader("Scaling Configuration")

        with st.expander("Advanced Settings"):
            col1, col2 = st.columns(2)

            with col1:
                st.number_input("Min Instances", min_value=1, value=self.auto_scaler.min_instances, key="min_instances")
                st.number_input("Max Instances", min_value=1, value=self.auto_scaler.max_instances, key="max_instances")

            with col2:
                st.slider("CPU High Threshold", 50, 95, int(self.auto_scaler.thresholds.cpu_high), key="cpu_high")
                st.slider("Memory High Threshold", 60, 95, int(self.auto_scaler.thresholds.memory_high), key="memory_high")

            if st.button("Update Configuration"):
                # Update auto-scaler configuration
                self.auto_scaler.min_instances = st.session_state.min_instances
                self.auto_scaler.max_instances = st.session_state.max_instances
                self.auto_scaler.thresholds.cpu_high = st.session_state.cpu_high
                self.auto_scaler.thresholds.memory_high = st.session_state.memory_high
                self.auto_scaler.save_config()
                st.success("Configuration updated!")

    def render_monitoring_tab(self):
        """Render monitoring and analytics tab"""
        st.header("📈 Monitoring & Analytics")

        # Alert management
        st.subheader("Active Alerts")

        alerts = self.monitor.check_alerts()
        active_alerts = [a for a in self.monitor.alerts.values() if not a.resolved_at]

        if active_alerts:
            for alert in active_alerts[-5:]:  # Show last 5 alerts
                alert_class = {
                    'critical': 'alert-card',
                    'warning': 'warning-card',
                    'info': 'info-card'
                }.get(alert.severity, 'info-card')

                st.markdown(f"""
                <div class="{alert_class}">
                    <strong>{alert.severity.upper()}: {alert.condition_name}</strong><br>
                    {alert.message}<br>
                    <small>Triggered: {alert.triggered_at.strftime('%H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No active alerts")

        # Performance trends
        st.subheader("Performance Trends Analysis")

        key_metrics = [
            'system.cpu_percent',
            'system.memory_percent',
            'application.health_score',
            'application.dashboard_response_time'
        ]

        trend_data = []
        for metric in key_metrics:
            trend = self.monitor.analyze_trends(metric, 24)
            trend_data.append({
                'Metric': metric,
                'Trend': trend.trend_direction.title(),
                'Strength': ".2f",
                'Confidence': ".2f",
                'Forecast (1h)': ".2f",
                'Forecast (24h)': ".2f"
            })

        if trend_data:
            st.dataframe(pd.DataFrame(trend_data))

        # Optimization recommendations
        st.subheader("Optimization Recommendations")

        if 'optimization_recommendations' in st.session_state:
            recommendations = st.session_state.optimization_recommendations
        else:
            recommendations = self.monitor.generate_optimization_recommendations()

        if recommendations:
            for rec in recommendations[:5]:  # Show top 5
                priority_color = {
                    'high': '🔴',
                    'medium': '🟡',
                    'low': '🟢'
                }.get(rec.priority, '⚪')

                with st.expander(f"{priority_color} {rec.title} ({rec.priority.title()})", expanded=False):
                    st.write(f"**Description:** {rec.description}")
                    st.write(f"**Impact Score:** {rec.impact_score:.2f}")
                    st.write(f"**Effort:** {rec.effort_estimate.title()}")

                    if rec.expected_benefits:
                        st.write("**Expected Benefits:**")
                        for benefit, value in rec.expected_benefits.items():
                            st.write(f"- {benefit.replace('_', ' ').title()}: {value:.1f}%")

                    if rec.implementation_steps:
                        st.write("**Implementation Steps:**")
                        for i, step in enumerate(rec.implementation_steps, 1):
                            st.write(f"{i}. {step}")
        else:
            st.info("No optimization recommendations available")

        # System health dashboard
        st.subheader("System Health Dashboard")

        dashboard_data = self.monitor.create_system_health_dashboard()

        if dashboard_data:
            # Health score
            health_score = dashboard_data['overview']['overall_health_score']
            health_color = "🟢" if health_score > 0.8 else "🟡" if health_score > 0.6 else "🔴"

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Health Score", ".1f")

            with col2:
                efficiency = dashboard_data['resource_summary']['efficiency_score']
                st.metric("Efficiency Score", ".1f")

            with col3:
                active_alerts = dashboard_data['alerts']['active']
                st.metric("Active Alerts", active_alerts)

    def render_configuration_tab(self):
        """Render configuration management tab"""
        st.header("⚙️ Configuration Management")

        # Configuration sections
        tab1, tab2, tab3 = st.tabs(["System Config", "Auto-Scaling", "Monitoring"])

        with tab1:
            st.subheader("System Configuration")

            # CLI Configuration
            st.markdown("**CLI Configuration**")
            cli_config = self.config_manager.get_configuration('cli')

            if cli_config:
                with st.expander("View/Edit CLI Configuration"):
                    st.json(cli_config)

                    # Edit form
                    with st.form("edit_cli_config"):
                        col1, col2 = st.columns(2)

                        with col1:
                            batch_size = st.number_input("Batch Size", value=cli_config.get('batch_size', 1000))
                            quality_threshold = st.slider("Quality Threshold", 0.0, 1.0, cli_config.get('quality_threshold', 0.7))

                        with col2:
                            log_level = st.selectbox("Log Level", ['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                                                   index=['DEBUG', 'INFO', 'WARNING', 'ERROR'].index(cli_config.get('log_level', 'INFO')))
                            cache_enabled = st.checkbox("Enable Caching", value=cli_config.get('cache_enabled', True))

                        if st.form_submit_button("Update CLI Config"):
                            updates = {
                                'batch_size': batch_size,
                                'quality_threshold': quality_threshold,
                                'log_level': log_level,
                                'cache_enabled': cache_enabled
                            }
                            if self.config_manager.update_configuration('cli', updates):
                                st.success("CLI configuration updated!")
                            else:
                                st.error("Failed to update configuration")

        with tab2:
            st.subheader("Auto-Scaling Configuration")

            scaling_config = {
                'min_instances': self.auto_scaler.min_instances,
                'max_instances': self.auto_scaler.max_instances,
                'cpu_high_threshold': self.auto_scaler.thresholds.cpu_high,
                'memory_high_threshold': self.auto_scaler.thresholds.memory_high,
                'scale_up_cooldown': self.auto_scaler.scale_up_cooldown,
                'scale_down_cooldown': self.auto_scaler.scale_down_cooldown
            }

            with st.expander("Auto-Scaling Settings"):
                st.json(scaling_config)

                with st.form("edit_scaling_config"):
                    col1, col2 = st.columns(2)

                    with col1:
                        min_inst = st.number_input("Min Instances", 1, 10, scaling_config['min_instances'])
                        max_inst = st.number_input("Max Instances", 1, 20, scaling_config['max_instances'])

                    with col2:
                        cpu_thresh = st.slider("CPU High Threshold", 50, 95, int(scaling_config['cpu_high_threshold']))
                        mem_thresh = st.slider("Memory High Threshold", 60, 95, int(scaling_config['memory_high_threshold']))

                    if st.form_submit_button("Update Scaling Config"):
                        self.auto_scaler.min_instances = min_inst
                        self.auto_scaler.max_instances = max_inst
                        self.auto_scaler.thresholds.cpu_high = cpu_thresh
                        self.auto_scaler.thresholds.memory_high = mem_thresh
                        self.auto_scaler.save_config()
                        st.success("Auto-scaling configuration updated!")

        with tab3:
            st.subheader("Monitoring Configuration")

            # Alert conditions
            st.markdown("**Alert Conditions**")

            if self.monitor.alert_conditions:
                alert_data = []
                for name, condition in self.monitor.alert_conditions.items():
                    alert_data.append({
                        'Name': name,
                        'Metric': condition.metric,
                        'Operator': condition.operator,
                        'Threshold': condition.threshold,
                        'Severity': condition.severity,
                        'Enabled': condition.enabled
                    })

                st.dataframe(pd.DataFrame(alert_data))
            else:
                st.info("No alert conditions configured")

            # Add new alert condition
            with st.expander("Add Alert Condition"):
                with st.form("add_alert"):
                    col1, col2 = st.columns(2)

                    with col1:
                        alert_name = st.text_input("Alert Name")
                        metric = st.text_input("Metric (e.g., system.cpu_percent)")
                        operator = st.selectbox("Operator", ['>', '<', '>=', '<=', '==', '!='])

                    with col2:
                        threshold = st.number_input("Threshold", value=80.0)
                        severity = st.selectbox("Severity", ['critical', 'warning', 'info'])
                        duration = st.number_input("Duration (minutes)", min_value=1, value=5)

                    if st.form_submit_button("Add Alert Condition"):
                        from forex_ai_dashboard.utils.advanced_monitor import AlertCondition
                        condition = AlertCondition(
                            name=alert_name,
                            metric=metric,
                            operator=operator,
                            threshold=threshold,
                            duration_minutes=duration,
                            severity=severity
                        )
                        self.monitor.alert_conditions[alert_name] = condition
                        self.monitor.save_config()
                        st.success("Alert condition added!")

    def run_dashboard(self):
        """Run the main dashboard"""
        # Initialize system if not already done
        if not st.session_state.dashboard_initialized:
            with st.spinner("Initializing Advanced System Dashboard..."):
                self.initialize_system()

        # Render header
        self.render_header()

        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 System Overview",
            "🎛️ Instances",
            "⚡ Auto-Scaling",
            "📈 Monitoring",
            "⚙️ Configuration"
        ])

        with tab1:
            self.render_system_overview_tab()

        with tab2:
            self.render_instances_tab()

        with tab3:
            self.render_auto_scaling_tab()

        with tab4:
            self.render_monitoring_tab()

        with tab5:
            self.render_configuration_tab()

        # Auto-refresh
        if (datetime.now() - self.last_refresh).seconds >= self.refresh_interval:
            self.last_refresh = datetime.now()
            time.sleep(0.1)  # Small delay to prevent immediate rerun
            st.rerun()

def main():
    """Main entry point"""
    dashboard = AdvancedSystemDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()