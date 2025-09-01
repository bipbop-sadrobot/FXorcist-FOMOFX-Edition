"""
System monitoring component for the Forex AI dashboard.
Tracks resource usage, system health, and pipeline status.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import psutil
import logging
from pathlib import Path

from . import SystemStatusComponent, ComponentConfig, cache_data

logger = logging.getLogger(__name__)

class SystemMonitor(SystemStatusComponent):
    """Component for monitoring system health and resource usage."""
    
    def __init__(self, config: ComponentConfig):
        """Initialize system monitoring component."""
        super().__init__(config)
        self.alert_thresholds = {
            'cpu_usage': 80.0,  # Percentage
            'memory_usage': 85.0,  # Percentage
            'disk_usage': 90.0,  # Percentage
            'data_latency': 300,  # Seconds
            'pipeline_latency': 60  # Seconds
        }
        self.history_length = 100  # Number of historical points to keep
        self._initialize_metrics_history()
    
    def _initialize_metrics_history(self):
        """Initialize historical metrics storage."""
        self.metrics_history = {
            'timestamps': [],
            'cpu_usage': [],
            'memory_usage': [],
            'disk_usage': [],
            'data_latency': [],
            'pipeline_latency': []
        }
    
    @cache_data(ttl_seconds=60)
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system resource metrics."""
        try:
            return {
                'cpu_usage': psutil.cpu_percent(interval=1),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return {}
    
    def _get_pipeline_metrics(self) -> Dict[str, float]:
        """Get pipeline performance metrics."""
        try:
            # Check data freshness
            data_path = Path('data/processed')
            latest_file = max(data_path.glob('*.parquet'), key=lambda x: x.stat().st_mtime)
            data_latency = (datetime.now() - datetime.fromtimestamp(latest_file.stat().st_mtime)).total_seconds()
            
            # Check pipeline logs
            log_path = Path('logs/pipeline.log')
            if log_path.exists():
                with open(log_path, 'r') as f:
                    last_lines = f.readlines()[-100:]  # Last 100 lines
                    pipeline_timestamps = [
                        datetime.strptime(line.split(' - ')[0], '%Y-%m-%d %H:%M:%S,%f')
                        for line in last_lines
                        if 'Pipeline completed' in line
                    ]
                    if pipeline_timestamps:
                        pipeline_latency = (datetime.now() - max(pipeline_timestamps)).total_seconds()
                    else:
                        pipeline_latency = float('inf')
            else:
                pipeline_latency = float('inf')
            
            return {
                'data_latency': data_latency,
                'pipeline_latency': pipeline_latency
            }
        except Exception as e:
            logger.error(f"Error getting pipeline metrics: {str(e)}")
            return {}
    
    def _update_metrics_history(self, metrics: Dict[str, float]):
        """Update historical metrics storage."""
        current_time = datetime.now()
        self.metrics_history['timestamps'].append(current_time)
        
        for metric_name in self.metrics_history.keys():
            if metric_name != 'timestamps':
                self.metrics_history[metric_name].append(
                    metrics.get(metric_name, float('nan'))
                )
        
        # Maintain history length
        if len(self.metrics_history['timestamps']) > self.history_length:
            for key in self.metrics_history:
                self.metrics_history[key] = self.metrics_history[key][-self.history_length:]
    
    def check_status(self) -> Dict[str, bool]:
        """Check system component status."""
        metrics = {
            **self._get_system_metrics(),
            **self._get_pipeline_metrics()
        }
        
        self._update_metrics_history(metrics)
        
        return {
            name: metrics.get(name, float('inf')) < threshold
            for name, threshold in self.alert_thresholds.items()
        }
    
    def create_resource_figure(self) -> go.Figure:
        """Create resource usage visualization."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                'CPU Usage',
                'Memory Usage',
                'Disk Usage',
                'Pipeline Latency'
            )
        )
        
        # CPU Usage
        fig.add_trace(
            go.Scatter(
                x=self.metrics_history['timestamps'],
                y=self.metrics_history['cpu_usage'],
                name='CPU',
                line=dict(color='blue')
            ),
            row=1,
            col=1
        )
        
        # Memory Usage
        fig.add_trace(
            go.Scatter(
                x=self.metrics_history['timestamps'],
                y=self.metrics_history['memory_usage'],
                name='Memory',
                line=dict(color='green')
            ),
            row=1,
            col=2
        )
        
        # Disk Usage
        fig.add_trace(
            go.Scatter(
                x=self.metrics_history['timestamps'],
                y=self.metrics_history['disk_usage'],
                name='Disk',
                line=dict(color='red')
            ),
            row=2,
            col=1
        )
        
        # Pipeline Latency
        fig.add_trace(
            go.Scatter(
                x=self.metrics_history['timestamps'],
                y=self.metrics_history['pipeline_latency'],
                name='Pipeline',
                line=dict(color='purple')
            ),
            row=2,
            col=2
        )
        
        # Add threshold lines
        for row, col, metric in [
            (1, 1, 'cpu_usage'),
            (1, 2, 'memory_usage'),
            (2, 1, 'disk_usage'),
            (2, 2, 'pipeline_latency')
        ]:
            fig.add_hline(
                y=self.alert_thresholds[metric],
                line_dash="dash",
                line_color="red",
                row=row,
                col=col
            )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text="System Resource Usage",
            hovermode='x unified'
        )
        
        return fig
    
    def render(self) -> None:
        """Render system monitoring component."""
        st.subheader(self.config.title)
        
        # Check current status
        status = self.check_status()
        
        # Display status indicators
        cols = st.columns(len(status))
        for col, (name, healthy) in zip(cols, status.items()):
            col.metric(
                name.replace('_', ' ').title(),
                f"{self.metrics_history[name][-1]:.1f}" if self.metrics_history[name] else "N/A",
                "✓" if healthy else "⚠️"
            )
        
        # Display resource usage plots
        fig = self.create_resource_figure()
        st.plotly_chart(fig, use_container_width=True)
        
        # Display alerts if any
        alerts = [
            f"⚠️ {name.replace('_', ' ').title()} above threshold"
            for name, healthy in status.items()
            if not healthy
        ]
        
        if alerts:
            st.warning(
                "System Alerts:\n" + "\n".join(alerts)
            )
    
    def update(self, data: Dict) -> None:
        """Update component with new data."""
        # System monitoring doesn't need external data updates
        pass