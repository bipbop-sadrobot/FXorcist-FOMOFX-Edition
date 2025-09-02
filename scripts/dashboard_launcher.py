#!/usr/bin/env python3
"""
Advanced Dashboard Launcher for FXorcist-FOMOFX-Edition
Unified dashboard management with configuration, monitoring, and automation.
"""

import sys
import os
import subprocess
import signal
import time
import json
import threading
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from forex_ai_dashboard.pipeline.optimized_data_integration import OptimizedDataIntegrator

@dataclass
class DashboardConfig:
    """Configuration for dashboard instances."""
    name: str
    script_path: str
    port: int
    title: str
    description: str
    category: str
    auto_start: bool = False
    health_check_url: Optional[str] = None
    environment_vars: Optional[Dict[str, str]] = None

@dataclass
class DashboardInstance:
    """Running dashboard instance information."""
    config: DashboardConfig
    process: Optional[subprocess.Popen] = None
    start_time: Optional[datetime] = None
    port: Optional[int] = None
    health_status: str = "stopped"
    last_health_check: Optional[datetime] = None
    restart_count: int = 0

class DashboardManager:
    """Advanced dashboard management system."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config_file = self.project_root / "config" / "dashboard_config.json"
        self.instances: Dict[str, DashboardInstance] = {}
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Default dashboard configurations
        self.default_configs = {
            "main": DashboardConfig(
                name="main",
                script_path="dashboard/app.py",
                port=8501,
                title="FXorcist Main Dashboard",
                description="Complete system overview and monitoring",
                category="main",
                auto_start=True,
                health_check_url="http://localhost:8501/health"
            ),
            "training": DashboardConfig(
                name="training",
                script_path="enhanced_training_dashboard.py",
                port=8502,
                title="Training Dashboard",
                description="Model training and evaluation monitoring",
                category="training",
                auto_start=False,
                health_check_url="http://localhost:8502/health"
            ),
            "memory": DashboardConfig(
                name="memory",
                script_path="dashboard/memory_dashboard.py",
                port=8503,
                title="Memory System Dashboard",
                description="Pattern analysis and memory insights",
                category="memory",
                auto_start=False,
                health_check_url="http://localhost:8503/health"
            ),
            "analytics": DashboardConfig(
                name="analytics",
                script_path="dashboard/analytics_dashboard.py",
                port=8504,
                title="Analytics Dashboard",
                description="Advanced analytics and reporting",
                category="analytics",
                auto_start=False,
                health_check_url="http://localhost:8504/health"
            )
        }

        self.load_config()

    def load_config(self):
        """Load dashboard configuration."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)

                # Update default configs with loaded data
                for name, config_dict in config_data.get('dashboards', {}).items():
                    if name in self.default_configs:
                        # Update existing config
                        config = self.default_configs[name]
                        for key, value in config_dict.items():
                            if hasattr(config, key):
                                setattr(config, key, value)
                    else:
                        # Add new config
                        self.default_configs[name] = DashboardConfig(**config_dict)

            except Exception as e:
                print(f"⚠️  Failed to load dashboard config: {e}")

        # Initialize instances
        for name, config in self.default_configs.items():
            self.instances[name] = DashboardInstance(config=config)

    def save_config(self):
        """Save current dashboard configuration."""
        self.config_file.parent.mkdir(exist_ok=True)

        config_data = {
            'dashboards': {},
            'last_updated': datetime.now().isoformat(),
            'version': '1.0'
        }

        for name, config in self.default_configs.items():
            config_data['dashboards'][name] = {
                'name': config.name,
                'script_path': config.script_path,
                'port': config.port,
                'title': config.title,
                'description': config.description,
                'category': config.category,
                'auto_start': config.auto_start,
                'health_check_url': config.health_check_url,
                'environment_vars': config.environment_vars
            }

        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=2)

        print(f"✅ Dashboard configuration saved to {self.config_file}")

    def start_dashboard(self, name: str, port: Optional[int] = None) -> bool:
        """Start a specific dashboard."""
        if name not in self.instances:
            print(f"❌ Dashboard '{name}' not found")
            return False

        instance = self.instances[name]

        if instance.process and instance.process.poll() is None:
            print(f"⚠️  Dashboard '{name}' is already running on port {instance.port}")
            return True

        # Use specified port or config port
        dashboard_port = port or instance.config.port

        try:
            # Prepare environment
            env = os.environ.copy()
            if instance.config.environment_vars:
                env.update(instance.config.environment_vars)

            # Set port environment variable
            env['STREAMLIT_SERVER_PORT'] = str(dashboard_port)
            env['STREAMLIT_SERVER_HEADLESS'] = 'true'
            env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

            # Start dashboard process
            script_path = self.project_root / instance.config.script_path
            if not script_path.exists():
                print(f"❌ Dashboard script not found: {script_path}")
                return False

            cmd = [
                sys.executable, "-m", "streamlit", "run",
                str(script_path),
                "--server.port", str(dashboard_port),
                "--server.headless", "true"
            ]

            print(f"🚀 Starting {instance.config.title} on port {dashboard_port}...")
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )

            # Update instance info
            instance.process = process
            instance.start_time = datetime.now()
            instance.port = dashboard_port
            instance.health_status = "starting"
            instance.restart_count = 0

            # Wait for startup
            time.sleep(3)

            # Check if process is still running
            if process.poll() is None:
                instance.health_status = "running"
                print(f"✅ {instance.config.title} started successfully")
                print(f"   📱 Access at: http://localhost:{dashboard_port}")
                return True
            else:
                instance.health_status = "failed"
                stdout, stderr = process.communicate()
                print(f"❌ Failed to start {instance.config.title}")
                if stderr:
                    print(f"   Error: {stderr.decode()[:200]}...")
                return False

        except Exception as e:
            print(f"❌ Error starting dashboard '{name}': {e}")
            instance.health_status = "error"
            return False

    def stop_dashboard(self, name: str) -> bool:
        """Stop a specific dashboard."""
        if name not in self.instances:
            print(f"❌ Dashboard '{name}' not found")
            return False

        instance = self.instances[name]

        if not instance.process or instance.process.poll() is not None:
            print(f"⚠️  Dashboard '{name}' is not running")
            return True

        try:
            if os.name == 'nt':  # Windows
                instance.process.terminate()
            else:  # Unix-like
                os.killpg(os.getpgid(instance.process.pid), signal.SIGTERM)

            # Wait for graceful shutdown
            try:
                instance.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                if os.name == 'nt':
                    instance.process.kill()
                else:
                    os.killpg(os.getpgid(instance.process.pid), signal.SIGKILL)
                instance.process.wait()

            instance.process = None
            instance.health_status = "stopped"
            instance.port = None

            print(f"✅ {instance.config.title} stopped successfully")
            return True

        except Exception as e:
            print(f"❌ Error stopping dashboard '{name}': {e}")
            return False

    def restart_dashboard(self, name: str) -> bool:
        """Restart a specific dashboard."""
        print(f"🔄 Restarting dashboard '{name}'...")

        if not self.stop_dashboard(name):
            print(f"❌ Failed to stop dashboard '{name}'")
            return False

        time.sleep(2)  # Brief pause

        if self.start_dashboard(name):
            instance = self.instances[name]
            instance.restart_count += 1
            print(f"✅ Dashboard '{name}' restarted successfully")
            return True
        else:
            print(f"❌ Failed to restart dashboard '{name}'")
            return False

    def start_all_dashboards(self) -> Dict[str, bool]:
        """Start all configured dashboards."""
        results = {}

        print("🚀 Starting all dashboards...")
        for name, instance in self.instances.items():
            if instance.config.auto_start:
                results[name] = self.start_dashboard(name)
            else:
                results[name] = False
                print(f"⏭️  Skipping {name} (auto_start=False)")

        return results

    def stop_all_dashboards(self) -> Dict[str, bool]:
        """Stop all running dashboards."""
        results = {}

        print("🛑 Stopping all dashboards...")
        for name, instance in self.instances.items():
            if instance.process and instance.process.poll() is None:
                results[name] = self.stop_dashboard(name)
            else:
                results[name] = True  # Already stopped

        return results

    def get_dashboard_status(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get status of dashboard(s)."""
        if name:
            if name not in self.instances:
                return {"error": f"Dashboard '{name}' not found"}

            instance = self.instances[name]
            return {
                "name": name,
                "title": instance.config.title,
                "status": instance.health_status,
                "port": instance.port,
                "start_time": instance.start_time.isoformat() if instance.start_time else None,
                "uptime": str(datetime.now() - instance.start_time) if instance.start_time else None,
                "restarts": instance.restart_count,
                "process_id": instance.process.pid if instance.process else None
            }

        # Return status for all dashboards
        status = {}
        for name, instance in self.instances.items():
            status[name] = self.get_dashboard_status(name)

        return status

    def check_dashboard_health(self, name: str) -> str:
        """Check health of a specific dashboard."""
        if name not in self.instances:
            return "not_found"

        instance = self.instances[name]

        if not instance.process or instance.process.poll() is not None:
            instance.health_status = "stopped"
            return "stopped"

        # Check if port is accessible
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('localhost', instance.port or 0))
            sock.close()

            if result == 0:
                instance.health_status = "running"
                instance.last_health_check = datetime.now()
                return "running"
            else:
                instance.health_status = "unreachable"
                return "unreachable"

        except Exception:
            instance.health_status = "error"
            return "error"

    def start_monitoring(self):
        """Start background monitoring of dashboards."""
        if self.monitoring_active:
            print("⚠️  Monitoring is already active")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        print("👁️  Dashboard monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        print("⏹️  Dashboard monitoring stopped")

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                for name, instance in self.instances.items():
                    if instance.process and instance.process.poll() is None:
                        # Check health
                        health = self.check_dashboard_health(name)

                        # Auto-restart if unhealthy
                        if health in ["stopped", "unreachable"] and instance.restart_count < 3:
                            print(f"🔄 Auto-restarting unhealthy dashboard: {name}")
                            self.restart_dashboard(name)

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")
                time.sleep(10)

    def get_system_resources(self) -> Dict[str, Any]:
        """Get system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024**3),
                "disk_total_gb": disk.total / (1024**3)
            }
        except Exception as e:
            return {"error": str(e)}

    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_resources": self.get_system_resources(),
            "dashboards": {},
            "overall_status": "healthy"
        }

        unhealthy_count = 0

        for name, instance in self.instances.items():
            status = self.get_dashboard_status(name)
            report["dashboards"][name] = status

            if status.get("status") not in ["running"]:
                unhealthy_count += 1

        if unhealthy_count > 0:
            report["overall_status"] = "degraded"
        if unhealthy_count == len(self.instances):
            report["overall_status"] = "critical"

        return report

def main():
    """Main dashboard launcher function."""
    import argparse

    parser = argparse.ArgumentParser(description="FXorcist Dashboard Launcher")
    parser.add_argument('action', choices=[
        'start', 'stop', 'restart', 'status', 'list',
        'start-all', 'stop-all', 'monitor', 'health', 'config'
    ], help='Action to perform')
    parser.add_argument('--dashboard', '-d', help='Dashboard name')
    parser.add_argument('--port', '-p', type=int, help='Port number')
    parser.add_argument('--config', help='Save configuration')

    args = parser.parse_args()

    manager = DashboardManager()

    if args.action == 'list':
        print("Available dashboards:")
        for name, config in manager.default_configs.items():
            status = manager.get_dashboard_status(name)
            print(f"  {name}: {config.title} (Port: {config.port}) - {status.get('status', 'unknown')}")

    elif args.action == 'start':
        if not args.dashboard:
            print("❌ Please specify dashboard name with --dashboard")
            sys.exit(1)
        success = manager.start_dashboard(args.dashboard, args.port)
        sys.exit(0 if success else 1)

    elif args.action == 'stop':
        if not args.dashboard:
            print("❌ Please specify dashboard name with --dashboard")
            sys.exit(1)
        success = manager.stop_dashboard(args.dashboard)
        sys.exit(0 if success else 1)

    elif args.action == 'restart':
        if not args.dashboard:
            print("❌ Please specify dashboard name with --dashboard")
            sys.exit(1)
        success = manager.restart_dashboard(args.dashboard)
        sys.exit(0 if success else 1)

    elif args.action == 'start-all':
        results = manager.start_all_dashboards()
        failed = [name for name, success in results.items() if not success]
        if failed:
            print(f"❌ Failed to start: {', '.join(failed)}")
            sys.exit(1)
        else:
            print("✅ All dashboards started successfully")

    elif args.action == 'stop-all':
        results = manager.stop_all_dashboards()
        failed = [name for name, success in results.items() if not success]
        if failed:
            print(f"❌ Failed to stop: {', '.join(failed)}")
            sys.exit(1)
        else:
            print("✅ All dashboards stopped successfully")

    elif args.action == 'status':
        if args.dashboard:
            status = manager.get_dashboard_status(args.dashboard)
            print(json.dumps(status, indent=2, default=str))
        else:
            status = manager.get_dashboard_status()
            print(json.dumps(status, indent=2, default=str))

    elif args.action == 'monitor':
        print("👁️  Starting dashboard monitoring...")
        manager.start_monitoring()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping monitoring...")
            manager.stop_monitoring()

    elif args.action == 'health':
        report = manager.generate_health_report()
        print(json.dumps(report, indent=2, default=str))

    elif args.action == 'config':
        if args.config == 'save':
            manager.save_config()
        else:
            print("Current configuration:")
            print(json.dumps({
                name: {
                    'port': config.port,
                    'auto_start': config.auto_start,
                    'description': config.description
                }
                for name, config in manager.default_configs.items()
            }, indent=2))

if __name__ == "__main__":
    main()