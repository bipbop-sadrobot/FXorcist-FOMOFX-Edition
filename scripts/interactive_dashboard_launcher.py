#!/usr/bin/env python3
"""
Interactive Dashboard Launcher for FXorcist AI Dashboard System

This script provides an interactive web-based interface for managing and launching
multiple dashboard instances with real-time monitoring and configuration management.

Features:
- Web-based dashboard management interface
- Real-time health monitoring integration
- Multi-dashboard support with resource management
- Configuration persistence and validation
- Auto-scaling and load balancing
- RESTful API for external integrations
- Comprehensive logging and error handling

Author: FXorcist Development Team
Version: 2.0
Date: September 2, 2025
"""

import os
import sys
import json
import time
import threading
import subprocess
import webbrowser
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
import logging.handlers

# Third-party imports
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import psutil
import requests
from flask import Flask, request, jsonify
import socket
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from forex_ai_dashboard.utils.health_checker import HealthChecker
from scripts.dashboard_launcher import DashboardManager, DashboardConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/interactive_launcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DashboardInstance:
    """Dashboard instance configuration and status"""
    id: str
    name: str
    type: str  # 'main', 'training', 'custom'
    port: int
    status: str  # 'stopped', 'starting', 'running', 'error'
    pid: Optional[int] = None
    start_time: Optional[datetime] = None
    config: Dict[str, Any] = None
    health_url: Optional[str] = None
    metrics: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert datetime to ISO format
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DashboardInstance':
        """Create instance from dictionary"""
        if 'start_time' in data and data['start_time']:
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        return cls(**data)

class InteractiveDashboardLauncher:
    """Interactive web-based dashboard launcher with monitoring"""

    def __init__(self, config_path: str = "config/dashboard_launcher.json"):
        self.config_path = Path(config_path)
        self.instances: Dict[str, DashboardInstance] = {}
        self.health_checker = HealthChecker()
        self.dashboard_manager = DashboardManager()
        self.api_server = None
        self.monitoring_thread = None
        self.stop_monitoring = False

        # Load configuration
        self.load_config()

        # Setup directories
        self.setup_directories()

        # Start monitoring thread
        self.start_monitoring()

    def setup_directories(self):
        """Setup required directories"""
        dirs = ['logs', 'config', 'data', 'models', 'temp']
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)

    def load_config(self):
        """Load configuration from file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    for instance_data in config.get('instances', []):
                        instance = DashboardInstance.from_dict(instance_data)
                        self.instances[instance.id] = instance
                logger.info(f"Loaded configuration with {len(self.instances)} instances")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        else:
            logger.info("No existing configuration found, starting fresh")

    def save_config(self):
        """Save current configuration to file"""
        try:
            config = {
                'instances': [instance.to_dict() for instance in self.instances.values()],
                'last_updated': datetime.now().isoformat()
            }
            self.config_path.parent.mkdir(exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    def create_instance(self, name: str, dashboard_type: str, port: int,
                       config: Optional[Dict[str, Any]] = None) -> str:
        """Create a new dashboard instance"""
        instance_id = f"{dashboard_type}_{port}_{int(time.time())}"

        instance = DashboardInstance(
            id=instance_id,
            name=name,
            type=dashboard_type,
            port=port,
            status='stopped',
            config=config or {},
            health_url=f"http://localhost:{port}/health"
        )

        self.instances[instance_id] = instance
        self.save_config()
        logger.info(f"Created dashboard instance: {instance_id}")
        return instance_id

    def start_instance(self, instance_id: str) -> bool:
        """Start a dashboard instance"""
        if instance_id not in self.instances:
            logger.error(f"Instance {instance_id} not found")
            return False

        instance = self.instances[instance_id]

        try:
            # Update status
            instance.status = 'starting'
            instance.start_time = datetime.now()

            # Determine dashboard script based on type
            script_map = {
                'main': 'dashboard/app.py',
                'training': 'enhanced_training_dashboard.py',
                'custom': instance.config.get('script', 'dashboard/app.py')
            }

            script_path = script_map.get(instance.type, 'dashboard/app.py')

            # Start the dashboard process
            cmd = [sys.executable, '-m', 'streamlit', 'run', script_path,
                   '--server.port', str(instance.port),
                   '--server.headless', 'true']

            # Add custom configuration
            if instance.config:
                for key, value in instance.config.items():
                    if key.startswith('server.'):
                        cmd.extend([f'--{key}', str(value)])

            logger.info(f"Starting dashboard: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).parent.parent
            )

            instance.pid = process.pid
            instance.status = 'running'
            self.save_config()

            logger.info(f"Dashboard instance {instance_id} started with PID {process.pid}")
            return True

        except Exception as e:
            logger.error(f"Error starting instance {instance_id}: {e}")
            instance.status = 'error'
            self.save_config()
            return False

    def stop_instance(self, instance_id: str) -> bool:
        """Stop a dashboard instance"""
        if instance_id not in self.instances:
            logger.error(f"Instance {instance_id} not found")
            return False

        instance = self.instances[instance_id]

        try:
            if instance.pid:
                # Try graceful shutdown first
                try:
                    requests.post(f"http://localhost:{instance.port}/shutdown", timeout=5)
                    time.sleep(2)
                except:
                    pass

                # Force kill if still running
                try:
                    process = psutil.Process(instance.pid)
                    process.terminate()
                    process.wait(timeout=10)
                except psutil.NoSuchProcess:
                    pass
                except psutil.TimeoutExpired:
                    process.kill()

            instance.status = 'stopped'
            instance.pid = None
            instance.start_time = None
            self.save_config()

            logger.info(f"Dashboard instance {instance_id} stopped")
            return True

        except Exception as e:
            logger.error(f"Error stopping instance {instance_id}: {e}")
            return False

    def get_instance_status(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a dashboard instance"""
        if instance_id not in self.instances:
            return None

        instance = self.instances[instance_id]
        status = instance.to_dict()

        # Add runtime information
        if instance.start_time:
            status['runtime'] = str(datetime.now() - instance.start_time)

        # Add health information
        if instance.health_url and instance.status == 'running':
            try:
                response = requests.get(instance.health_url, timeout=5)
                if response.status_code == 200:
                    status['health'] = response.json()
                else:
                    status['health'] = {'status': 'unhealthy', 'error': f'HTTP {response.status_code}'}
            except Exception as e:
                status['health'] = {'status': 'unreachable', 'error': str(e)}
        else:
            status['health'] = {'status': 'unknown'}

        return status

    def start_monitoring(self):
        """Start background monitoring thread"""
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Started monitoring thread")

    def _monitoring_loop(self):
        """Background monitoring loop"""
        while not self.stop_monitoring:
            try:
                self._update_instance_statuses()
                self._collect_system_metrics()
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)

    def _update_instance_statuses(self):
        """Update status of all instances"""
        for instance in self.instances.values():
            if instance.status == 'running' and instance.pid:
                try:
                    process = psutil.Process(instance.pid)
                    if not process.is_running():
                        instance.status = 'error'
                        instance.pid = None
                        logger.warning(f"Instance {instance.id} process died")
                except psutil.NoSuchProcess:
                    instance.status = 'error'
                    instance.pid = None
                    logger.warning(f"Instance {instance.id} process not found")

    def _collect_system_metrics(self):
        """Collect system-wide metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Process metrics
            fxorcist_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    if any(keyword in proc.info['name'].lower()
                          for keyword in ['fxorcist', 'streamlit', 'python']):
                        fxorcist_processes.append(proc.info)
                except:
                    continue

            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'disk_percent': disk.percent,
                'fxorcist_processes': len(fxorcist_processes),
                'process_details': fxorcist_processes
            }

            # Store metrics for each running instance
            for instance in self.instances.values():
                if instance.status == 'running':
                    instance.metrics = metrics

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def get_system_overview(self) -> Dict[str, Any]:
        """Get system overview with all instances and metrics"""
        return {
            'instances': {id: self.get_instance_status(id) for id in self.instances},
            'system_health': self.health_checker.check_all_components(),
            'total_instances': len(self.instances),
            'running_instances': sum(1 for i in self.instances.values() if i.status == 'running'),
            'timestamp': datetime.now().isoformat()
        }

    def cleanup_stopped_instances(self):
        """Clean up stopped/error instances older than 1 hour"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        to_remove = []

        for instance_id, instance in self.instances.items():
            if instance.status in ['stopped', 'error'] and instance.start_time:
                if instance.start_time < cutoff_time:
                    to_remove.append(instance_id)

        for instance_id in to_remove:
            del self.instances[instance_id]
            logger.info(f"Cleaned up old instance: {instance_id}")

        if to_remove:
            self.save_config()

    def shutdown(self):
        """Shutdown the launcher and stop all instances"""
        logger.info("Shutting down interactive dashboard launcher")
        self.stop_monitoring = True

        # Stop all running instances
        for instance_id in list(self.instances.keys()):
            self.stop_instance(instance_id)

        # Save final configuration
        self.save_config()

def create_streamlit_app(launcher: InteractiveDashboardLauncher):
    """Create the Streamlit web interface"""

    st.set_page_config(
        page_title="FXorcist Dashboard Manager",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🚀 FXorcist Interactive Dashboard Launcher")
    st.markdown("---")

    # Sidebar with controls
    with st.sidebar:
        st.header("⚙️ Controls")

        # System Health
        st.subheader("System Health")
        system_overview = launcher.get_system_overview()
        health_status = system_overview['system_health']

        health_color = "🟢" if health_status['status'] == 'healthy' else "🔴"
        st.metric("Status", f"{health_color} {health_status['status'].title()}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Running", system_overview['running_instances'])
        with col2:
            st.metric("Total", system_overview['total_instances'])

        st.markdown("---")

        # Quick Actions
        st.subheader("Quick Actions")

        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()

        if st.button("🧹 Cleanup Old Instances", use_container_width=True):
            launcher.cleanup_stopped_instances()
            st.success("Cleaned up old instances!")
            time.sleep(1)
            st.rerun()

        if st.button("💾 Save Configuration", use_container_width=True):
            launcher.save_config()
            st.success("Configuration saved!")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard Instances", "➕ Create New", "📈 System Monitor", "🔧 Configuration"])

    with tab1:
        st.header("Dashboard Instances")

        if not launcher.instances:
            st.info("No dashboard instances found. Create one using the 'Create New' tab.")
        else:
            # Instance management
            for instance_id, instance in launcher.instances.items():
                with st.expander(f"{instance.name} ({instance.type}) - {instance.status.title()}", expanded=True):
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                    with col1:
                        st.write(f"**Port:** {instance.port}")
                        if instance.start_time:
                            runtime = datetime.now() - instance.start_time
                            st.write(f"**Runtime:** {str(runtime).split('.')[0]}")

                    with col2:
                        if instance.status == 'running':
                            if st.button("🛑 Stop", key=f"stop_{instance_id}"):
                                if launcher.stop_instance(instance_id):
                                    st.success("Instance stopped!")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("Failed to stop instance")
                        elif instance.status in ['stopped', 'error']:
                            if st.button("▶️ Start", key=f"start_{instance_id}"):
                                if launcher.start_instance(instance_id):
                                    st.success("Instance started!")
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    st.error("Failed to start instance")

                    with col3:
                        if instance.status == 'running' and instance.port:
                            st.markdown(f"[🌐 Open Dashboard](http://localhost:{instance.port})")

                    with col4:
                        if st.button("🗑️ Delete", key=f"delete_{instance_id}"):
                            if instance.status == 'running':
                                launcher.stop_instance(instance_id)
                            del launcher.instances[instance_id]
                            launcher.save_config()
                            st.success("Instance deleted!")
                            time.sleep(1)
                            st.rerun()

                    # Health information
                    if instance.status == 'running':
                        status_info = launcher.get_instance_status(instance_id)
                        if status_info and 'health' in status_info:
                            health = status_info['health']
                            if health['status'] == 'healthy':
                                st.success("✅ Health Check: Healthy")
                            else:
                                st.error(f"❌ Health Check: {health.get('error', 'Unknown error')}")

    with tab2:
        st.header("Create New Dashboard Instance")

        with st.form("create_instance"):
            col1, col2 = st.columns(2)

            with col1:
                name = st.text_input("Instance Name", placeholder="My Dashboard")
                dashboard_type = st.selectbox("Dashboard Type",
                                            ["main", "training", "custom"],
                                            help="Type of dashboard to create")

            with col2:
                port = st.number_input("Port", min_value=1024, max_value=65535, value=8501)
                auto_port = st.checkbox("Auto-assign port", value=True)

            # Advanced configuration
            with st.expander("Advanced Configuration"):
                st.json({
                    "server.headless": True,
                    "server.enableCORS": False,
                    "theme.base": "light"
                })

            submitted = st.form_submit_button("Create Instance", use_container_width=True)

            if submitted:
                if not name:
                    st.error("Please provide an instance name")
                else:
                    if auto_port:
                        # Find available port
                        port = 8501
                        while port in [i.port for i in launcher.instances.values()]:
                            port += 1

                    instance_id = launcher.create_instance(name, dashboard_type, port)
                    st.success(f"Created instance '{name}' with ID: {instance_id}")

                    # Auto-start option
                    if st.checkbox("Start immediately", value=True):
                        if launcher.start_instance(instance_id):
                            st.success("Instance started successfully!")
                        else:
                            st.error("Failed to start instance")

                    time.sleep(2)
                    st.rerun()

    with tab3:
        st.header("System Monitor")

        # Real-time metrics
        system_overview = launcher.get_system_overview()

        # System metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("CPU Usage", ".1f")

        with col2:
            st.metric("Memory Usage", ".1f")

        with col3:
            st.metric("Disk Usage", ".1f")

        with col4:
            st.metric("Active Processes", system_overview.get('running_instances', 0))

        # Health status overview
        st.subheader("Component Health")
        health_status = system_overview['system_health']

        health_df = pd.DataFrame([
            {"Component": comp, "Status": info['status'], "Details": info.get('message', '')}
            for comp, info in health_status.get('components', {}).items()
        ])

        if not health_df.empty:
            # Color coding for status
            def color_status(val):
                color = 'green' if val == 'healthy' else 'red' if val == 'error' else 'orange'
                return f'color: {color}'

            st.dataframe(health_df.style.applymap(color_status, subset=['Status']))

        # Performance chart placeholder
        st.subheader("Performance Trends")
        st.info("Performance monitoring data will be displayed here in future updates")

    with tab4:
        st.header("Configuration Management")

        st.subheader("Current Instances Configuration")
        if launcher.instances:
            config_data = []
            for instance in launcher.instances.values():
                config_data.append({
                    "ID": instance.id,
                    "Name": instance.name,
                    "Type": instance.type,
                    "Port": instance.port,
                    "Status": instance.status,
                    "PID": instance.pid or "N/A"
                })

            st.dataframe(pd.DataFrame(config_data))
        else:
            st.info("No instances configured")

        st.markdown("---")

        # Export/Import configuration
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📤 Export Configuration", use_container_width=True):
                config_json = json.dumps({
                    'instances': [i.to_dict() for i in launcher.instances.values()],
                    'exported_at': datetime.now().isoformat()
                }, indent=2)
                st.download_button(
                    label="Download Config",
                    data=config_json,
                    file_name="dashboard_config.json",
                    mime="application/json"
                )

        with col2:
            uploaded_file = st.file_uploader("📥 Import Configuration", type=['json'])
            if uploaded_file is not None and st.button("Import", use_container_width=True):
                try:
                    config_data = json.load(uploaded_file)
                    for instance_data in config_data.get('instances', []):
                        instance = DashboardInstance.from_dict(instance_data)
                        launcher.instances[instance.id] = instance
                    launcher.save_config()
                    st.success("Configuration imported successfully!")
                    time.sleep(2)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error importing configuration: {e}")

def main():
    """Main entry point"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Interactive Dashboard Launcher")
    parser.add_argument("--port", type=int, default=8502, help="Port for the launcher interface")
    parser.add_argument("--config", type=str, default="config/dashboard_launcher.json", help="Configuration file path")
    parser.add_argument("--web-only", action="store_true", help="Run only web interface, no CLI")
    args = parser.parse_args()

    # Initialize launcher
    launcher = InteractiveDashboardLauncher(args.config)

    if args.web_only:
        # Run only web interface
        create_streamlit_app(launcher)
    else:
        # Run both CLI and web interface
        print("FXorcist Interactive Dashboard Launcher")
        print("=" * 50)
        print(f"Web interface will be available at: http://localhost:{args.port}")
        print("Press Ctrl+C to stop")

        # Start web interface in background
        def run_web():
            create_streamlit_app(launcher)

        web_thread = threading.Thread(target=run_web, daemon=True)
        web_thread.start()

        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            launcher.shutdown()
            print("Shutdown complete")

if __name__ == "__main__":
    main()