"""
Pipeline monitoring component for the Forex AI dashboard.
Provides real-time monitoring of training pipelines and system status.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import psutil
import subprocess
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import asyncio
import threading
import time

from . import ComponentConfig, DashboardComponent

logger = logging.getLogger(__name__)

class PipelineMonitor(DashboardComponent):
    """Component for monitoring training pipelines and system resources."""

    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.project_root = Path(__file__).parent.parent.parent
        self.logs_dir = self.project_root / "logs"
        self.models_dir = self.project_root / "models" / "trained"
        self.data_dir = self.project_root / "data"
        self.refresh_interval = 5  # seconds
        self._last_update = None
        self._pipeline_status = {}
        self._system_metrics = {}

    def _get_running_processes(self) -> List[Dict]:
        """Get information about running training processes."""
        processes = []

        try:
            # Check for Python processes running training scripts
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent', 'create_time']):
                try:
                    if proc.info['name'] == 'python' or proc.info['name'] == 'python3':
                        cmdline = proc.info['cmdline']
                        if cmdline and any(script in ' '.join(cmdline) for script in [
                            'training_pipeline', 'focused_training', 'automated_training',
                            'continuous_training', 'simple_train'
                        ]):
                            processes.append({
                                'pid': proc.info['pid'],
                                'name': ' '.join(cmdline),
                                'cpu_percent': proc.info['cpu_percent'],
                                'memory_percent': proc.info['memory_percent'],
                                'start_time': datetime.fromtimestamp(proc.info['create_time']),
                                'status': 'running'
                            })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.error(f"Error getting process info: {str(e)}")

        return processes

    def _get_pipeline_logs(self) -> Dict[str, List[Dict]]:
        """Get recent logs from training pipelines."""
        logs = {}

        try:
            # Check for log files
            log_files = {
                'automated_training': self.logs_dir / 'automated_training.log',
                'focused_training': self.logs_dir / 'focused_training.log',
                'continuous_training': self.logs_dir / 'continuous_training.log',
                'simple_training': self.logs_dir / 'simple_train.log'
            }

            for pipeline_name, log_file in log_files.items():
                if log_file.exists():
                    logs[pipeline_name] = self._parse_log_file(log_file)
                else:
                    logs[pipeline_name] = []

        except Exception as e:
            logger.error(f"Error reading pipeline logs: {str(e)}")

        return logs

    def _parse_log_file(self, log_file: Path, max_lines: int = 50) -> List[Dict]:
        """Parse log file and extract recent entries."""
        entries = []

        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()[-max_lines:]

            for line in lines:
                if line.strip():
                    try:
                        # Parse log line (assuming format: timestamp - level - message)
                        parts = line.split(' - ', 2)
                        if len(parts) >= 3:
                            timestamp_str, level, message = parts
                            timestamp = pd.to_datetime(timestamp_str.strip())

                            entries.append({
                                'timestamp': timestamp,
                                'level': level.strip(),
                                'message': message.strip(),
                                'pipeline': log_file.stem
                            })
                    except Exception:
                        # If parsing fails, add as raw message
                        entries.append({
                            'timestamp': datetime.now(),
                            'level': 'UNKNOWN',
                            'message': line.strip(),
                            'pipeline': log_file.stem
                        })

        except Exception as e:
            logger.error(f"Error parsing log file {log_file}: {str(e)}")

        return entries

    def _get_system_metrics(self) -> Dict:
        """Get current system resource usage."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return {}

    def _get_training_progress(self) -> Dict[str, Dict]:
        """Get training progress from log files and model files."""
        progress = {}

        try:
            # Check for recent model files
            if self.models_dir.exists():
                model_files = list(self.models_dir.glob('*.cbm')) + list(self.models_dir.glob('*.json'))
                recent_models = []

                for model_file in model_files:
                    try:
                        mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
                        if datetime.now() - mtime < timedelta(hours=24):  # Last 24 hours
                            recent_models.append({
                                'name': model_file.name,
                                'size_mb': model_file.stat().st_size / (1024 * 1024),
                                'created': mtime
                            })
                    except Exception:
                        continue

                progress['recent_models'] = recent_models

            # Check data processing status
            processed_data_dir = self.data_dir / "processed"
            if processed_data_dir.exists():
                parquet_files = list(processed_data_dir.glob('*.parquet'))
                data_files = []

                for file in parquet_files:
                    try:
                        mtime = datetime.fromtimestamp(file.stat().st_mtime)
                        data_files.append({
                            'name': file.name,
                            'size_mb': file.stat().st_size / (1024 * 1024),
                            'created': mtime
                        })
                    except Exception:
                        continue

                progress['processed_data'] = data_files

        except Exception as e:
            logger.error(f"Error getting training progress: {str(e)}")

        return progress

    def _create_pipeline_status_figure(self, processes: List[Dict], logs: Dict) -> go.Figure:
        """Create pipeline status visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Running Processes', 'Recent Activity', 'System Resources', 'Training Progress'),
            specs=[[{'type': 'table'}, {'type': 'scatter'}],
                   [{'type': 'indicator'}, {'type': 'bar'}]]
        )

        # Running Processes Table
        if processes:
            process_data = []
            for proc in processes:
                process_data.append([
                    proc['pid'],
                    proc['name'][:50] + '...' if len(proc['name']) > 50 else proc['name'],
                    f"{proc['cpu_percent']:.1f}%",
                    f"{proc['memory_percent']:.1f}%",
                    proc['start_time'].strftime('%H:%M:%S')
                ])

            fig.add_trace(
                go.Table(
                    header=dict(values=['PID', 'Process', 'CPU', 'Memory', 'Start Time']),
                    cells=dict(values=list(zip(*process_data)))
                ),
                row=1, col=1
            )
        else:
            fig.add_trace(
                go.Table(
                    header=dict(values=['Status']),
                    cells=dict(values=[['No training pipelines running']])
                ),
                row=1, col=1
            )

        # Recent Activity Scatter
        all_logs = []
        for pipeline_logs in logs.values():
            all_logs.extend(pipeline_logs[-20:])  # Last 20 entries

        if all_logs:
            df_logs = pd.DataFrame(all_logs)
            df_logs = df_logs.sort_values('timestamp')

            # Color by log level
            color_map = {'INFO': 'blue', 'WARNING': 'orange', 'ERROR': 'red', 'DEBUG': 'gray'}

            fig.add_trace(
                go.Scatter(
                    x=df_logs['timestamp'],
                    y=[1] * len(df_logs),  # All on same line
                    mode='markers+text',
                    marker=dict(
                        color=[color_map.get(level, 'gray') for level in df_logs['level']],
                        size=8
                    ),
                    text=df_logs['message'].str[:30] + '...',
                    textposition="top center",
                    name='Log Activity'
                ),
                row=1, col=2
            )

        # System Resources Indicator
        system_metrics = self._get_system_metrics()
        if system_metrics:
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=system_metrics.get('cpu_percent', 0),
                    title={'text': "CPU Usage"},
                    gauge={'axis': {'range': [0, 100]}},
                    domain={'row': 1, 'column': 1}
                ),
                row=2, col=1
            )

        # Training Progress Bar
        progress = self._get_training_progress()
        if 'recent_models' in progress and progress['recent_models']:
            model_names = [m['name'][:20] for m in progress['recent_models']]
            model_sizes = [m['size_mb'] for m in progress['recent_models']]

            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=model_sizes,
                    name='Model Sizes (MB)',
                    marker_color='lightblue'
                ),
                row=2, col=2
            )

        fig.update_layout(height=800, showlegend=False)
        return fig

    def _create_resource_monitor_figure(self) -> go.Figure:
        """Create system resource monitoring figure."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU Usage', 'Memory Usage', 'Disk Usage', 'Network I/O'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'indicator'}, {'type': 'scatter'}]]
        )

        # CPU Usage over time
        if hasattr(self, '_cpu_history') and self._cpu_history:
            timestamps, cpu_values = zip(*self._cpu_history)
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=cpu_values,
                    mode='lines',
                    name='CPU %',
                    line=dict(color='red')
                ),
                row=1, col=1
            )

        # Memory Usage over time
        if hasattr(self, '_memory_history') and self._memory_history:
            timestamps, mem_values = zip(*self._memory_history)
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=mem_values,
                    mode='lines',
                    name='Memory %',
                    line=dict(color='blue')
                ),
                row=1, col=2
            )

        # Disk Usage Indicator
        disk_usage = psutil.disk_usage('/').percent
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=disk_usage,
                title={'text': "Disk Usage"},
                gauge={'axis': {'range': [0, 100]}},
                domain={'row': 1, 'column': 1}
            ),
            row=2, col=1
        )

        # Network I/O (placeholder)
        fig.add_trace(
            go.Scatter(
                x=[datetime.now()],
                y=[0],
                mode='lines',
                name='Network I/O',
                line=dict(color='green')
            ),
            row=2, col=2
        )

        fig.update_layout(height=600, showlegend=True)
        return fig

    def _update_resource_history(self):
        """Update resource usage history for trending."""
        if not hasattr(self, '_cpu_history'):
            self._cpu_history = []
            self._memory_history = []

        current_time = datetime.now()
        system_metrics = self._get_system_metrics()

        # Keep only last 60 data points (5 minutes at 5-second intervals)
        self._cpu_history.append((current_time, system_metrics.get('cpu_percent', 0)))
        self._memory_history.append((current_time, system_metrics.get('memory_percent', 0)))

        if len(self._cpu_history) > 60:
            self._cpu_history.pop(0)
            self._memory_history.pop(0)

    def render(self) -> None:
        """Render the pipeline monitoring component."""
        st.subheader(self.config.title)

        # Auto-refresh toggle
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            auto_refresh = st.checkbox("Auto-refresh", value=True, key="pipeline_auto_refresh")

        with col2:
            refresh_rate = st.selectbox(
                "Refresh Rate",
                [1, 5, 10, 30],
                index=1,
                key="pipeline_refresh_rate"
            )

        with col3:
            if st.button("🔄 Refresh Now", key="pipeline_refresh_now"):
                st.rerun()

        # Update resource history
        self._update_resource_history()

        # Get current data
        processes = self._get_running_processes()
        logs = self._get_pipeline_logs()
        progress = self._get_training_progress()

        # Pipeline Status Section
        st.subheader("🚀 Pipeline Status")

        if processes:
            st.success(f"✅ {len(processes)} training pipeline(s) currently running")

            # Process details
            for proc in processes:
                with st.expander(f"Process {proc['pid']}: {proc['name'][:30]}..."):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("CPU Usage", f"{proc['cpu_percent']:.1f}%")
                    col2.metric("Memory Usage", f"{proc['memory_percent']:.1f}%")
                    col3.metric("Runtime", str(datetime.now() - proc['start_time']).split('.')[0])
        else:
            st.info("ℹ️ No training pipelines currently running")

        # Pipeline Status Figure
        fig_status = self._create_pipeline_status_figure(processes, logs)
        st.plotly_chart(fig_status, use_container_width=True)

        # Resource Monitoring Section
        st.subheader("📊 System Resources")

        # Current metrics
        system_metrics = self._get_system_metrics()
        if system_metrics:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("CPU Usage", f"{system_metrics.get('cpu_percent', 0):.1f}%")
            col2.metric("Memory Usage", f"{system_metrics.get('memory_percent', 0):.1f}%")
            col3.metric("Disk Usage", f"{system_metrics.get('disk_usage', 0):.1f}%")
            col4.metric("Active Processes", len(processes))

        # Resource monitoring figure
        fig_resources = self._create_resource_monitor_figure()
        st.plotly_chart(fig_resources, use_container_width=True)

        # Recent Activity Section
        st.subheader("📝 Recent Activity")

        # Show recent logs
        all_recent_logs = []
        for pipeline_logs in logs.values():
            all_recent_logs.extend(pipeline_logs[-10:])  # Last 10 from each pipeline

        if all_recent_logs:
            df_recent = pd.DataFrame(all_recent_logs)
            df_recent = df_recent.sort_values('timestamp', ascending=False)

            for _, log_entry in df_recent.head(10).iterrows():
                if log_entry['level'] == 'ERROR':
                    st.error(f"❌ {log_entry['pipeline']}: {log_entry['message']}")
                elif log_entry['level'] == 'WARNING':
                    st.warning(f"⚠️ {log_entry['pipeline']}: {log_entry['message']}")
                elif log_entry['level'] == 'INFO':
                    st.info(f"ℹ️ {log_entry['pipeline']}: {log_entry['message']}")
                else:
                    st.text(f"{log_entry['pipeline']}: {log_entry['message']}")
        else:
            st.info("No recent activity found")

        # Training Progress Section
        st.subheader("🎯 Training Progress")

        if 'recent_models' in progress and progress['recent_models']:
            st.success(f"✅ {len(progress['recent_models'])} model(s) trained recently")

            for model in progress['recent_models'][:5]:  # Show top 5
                with st.expander(f"📁 {model['name']}"):
                    col1, col2 = st.columns(2)
                    col1.metric("Size", f"{model['size_mb']:.2f} MB")
                    col2.metric("Created", model['created'].strftime('%Y-%m-%d %H:%M'))
        else:
            st.info("No recent models found")

        # Auto-refresh logic
        if auto_refresh:
            time.sleep(refresh_rate)
            st.rerun()

    def update(self, data: dict) -> None:
        """Update component with new data."""
        self._cache['data'] = data
        self.clear_cache()