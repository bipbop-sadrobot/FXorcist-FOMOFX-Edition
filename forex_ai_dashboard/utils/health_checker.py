#!/usr/bin/env python3
"""
FXorcist Health Check System
Comprehensive monitoring and health assessment for all system components.
"""

import sys
import os
import time
import json
import psutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import threading
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from forex_ai_dashboard.pipeline.optimized_data_integration import OptimizedDataIntegrator
from forex_ai_dashboard.pipeline.unified_training_pipeline import UnifiedTrainingPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthStatus:
    """Health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class ComponentHealth:
    """Health information for a system component."""

    def __init__(self, name: str, component_type: str):
        self.name = name
        self.component_type = component_type
        self.status = HealthStatus.UNKNOWN
        self.last_check = None
        self.response_time = None
        self.error_message = None
        self.metrics = {}
        self.details = {}

    def update_health(self, status: str, response_time: Optional[float] = None,
                     error_message: Optional[str] = None, **metrics):
        """Update component health information."""
        self.status = status
        self.last_check = datetime.now()
        self.response_time = response_time
        self.error_message = error_message
        self.metrics.update(metrics)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "type": self.component_type,
            "status": self.status,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "response_time": self.response_time,
            "error_message": self.error_message,
            "metrics": self.metrics,
            "details": self.details
        }

class HealthChecker:
    """Comprehensive health monitoring system."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.components: Dict[str, ComponentHealth] = {}
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.check_interval = 60  # seconds

        # Initialize component monitoring
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all system components for monitoring."""
        # Core system components
        self.components["system"] = ComponentHealth("system", "infrastructure")
        self.components["python"] = ComponentHealth("python", "runtime")
        self.components["memory_system"] = ComponentHealth("memory_system", "core")
        self.components["data_pipeline"] = ComponentHealth("data_pipeline", "processing")
        self.components["training_pipeline"] = ComponentHealth("training_pipeline", "ml")

        # Dashboard components
        dashboard_ports = [8501, 8502, 8503, 8504]  # main, training, memory, analytics
        for i, port in enumerate(dashboard_ports):
            names = ["main_dashboard", "training_dashboard", "memory_dashboard", "analytics_dashboard"]
            self.components[names[i]] = ComponentHealth(names[i], "dashboard")

        # Data components
        self.components["data_integrity"] = ComponentHealth("data_integrity", "data")
        self.components["model_storage"] = ComponentHealth("model_storage", "storage")

        logger.info(f"Initialized health monitoring for {len(self.components)} components")

    def start_monitoring(self, interval: int = 60):
        """Start background health monitoring."""
        if self.monitoring_active:
            logger.warning("Health monitoring is already active")
            return

        self.monitoring_active = True
        self.check_interval = interval
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        logger.info(f"Health monitoring started (interval: {interval}s)")

    def stop_monitoring(self):
        """Stop background health monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        logger.info("Health monitoring stopped")

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                self.check_all_components()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)

    def check_all_components(self) -> Dict[str, Any]:
        """Check health of all components."""
        results = {}

        # System-level checks
        results["system"] = self.check_system_health()
        results["python"] = self.check_python_health()

        # Core component checks
        results["memory_system"] = self.check_memory_system_health()
        results["data_pipeline"] = self.check_data_pipeline_health()
        results["training_pipeline"] = self.check_training_pipeline_health()

        # Dashboard checks
        for name in ["main_dashboard", "training_dashboard", "memory_dashboard", "analytics_dashboard"]:
            results[name] = self.check_dashboard_health(name)

        # Data checks
        results["data_integrity"] = self.check_data_integrity_health()
        results["model_storage"] = self.check_model_storage_health()

        # Update component statuses
        for name, result in results.items():
            if name in self.components:
                component = self.components[name]
                component.update_health(
                    status=result.get("status", HealthStatus.UNKNOWN),
                    response_time=result.get("response_time"),
                    error_message=result.get("error_message"),
                    **result.get("metrics", {})
                )

        return results

    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            # Network connectivity (basic)
            network_status = self._check_network_connectivity()

            # Determine overall status
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 95:
                status = HealthStatus.CRITICAL
            elif cpu_percent > 70 or memory_percent > 80 or disk_percent > 85:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY

            return {
                "status": status,
                "response_time": 1.0,
                "metrics": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent,
                    "network_status": network_status
                }
            }

        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "error_message": str(e),
                "response_time": None
            }

    def check_python_health(self) -> Dict[str, Any]:
        """Check Python runtime health."""
        try:
            import sys
            import platform

            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            platform_info = platform.platform()

            # Check for required modules
            required_modules = ["pandas", "numpy", "streamlit", "catboost"]
            missing_modules = []

            for module in required_modules:
                try:
                    __import__(module)
                except ImportError:
                    missing_modules.append(module)

            if missing_modules:
                return {
                    "status": HealthStatus.CRITICAL,
                    "error_message": f"Missing modules: {', '.join(missing_modules)}",
                    "metrics": {
                        "python_version": python_version,
                        "platform": platform_info,
                        "missing_modules": missing_modules
                    }
                }

            return {
                "status": HealthStatus.HEALTHY,
                "response_time": 0.1,
                "metrics": {
                    "python_version": python_version,
                    "platform": platform_info,
                    "modules_loaded": len(required_modules)
                }
            }

        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "error_message": str(e)
            }

    def check_memory_system_health(self) -> Dict[str, Any]:
        """Check memory system health."""
        try:
            from memory_system.core import MemoryManager

            start_time = time.time()
            memory_manager = MemoryManager()
            response_time = time.time() - start_time

            # Get memory statistics
            record_count = len(memory_manager.records) if hasattr(memory_manager, 'records') else 0

            # Check memory performance
            if response_time > 5.0:
                status = HealthStatus.WARNING
            elif response_time > 10.0:
                status = HealthStatus.CRITICAL
            else:
                status = HealthStatus.HEALTHY

            return {
                "status": status,
                "response_time": response_time,
                "metrics": {
                    "record_count": record_count,
                    "response_time": response_time
                }
            }

        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "error_message": str(e),
                "response_time": None
            }

    def check_data_pipeline_health(self) -> Dict[str, Any]:
        """Check data pipeline health."""
        try:
            start_time = time.time()

            # Check if data directories exist and have content
            data_dir = self.project_root / "data"
            processed_dir = data_dir / "processed"

            if not processed_dir.exists():
                return {
                    "status": HealthStatus.CRITICAL,
                    "error_message": "Processed data directory not found",
                    "response_time": time.time() - start_time
                }

            # Count processed files
            processed_files = list(processed_dir.glob("*.parquet"))
            file_count = len(processed_files)

            # Check recent file activity
            recent_files = 0
            cutoff_time = datetime.now() - timedelta(hours=24)

            for file_path in processed_files:
                if file_path.stat().st_mtime > cutoff_time.timestamp():
                    recent_files += 1

            response_time = time.time() - start_time

            # Determine status
            if file_count == 0:
                status = HealthStatus.CRITICAL
                error_msg = "No processed data files found"
            elif recent_files == 0:
                status = HealthStatus.WARNING
                error_msg = "No recent data processing activity"
            else:
                status = HealthStatus.HEALTHY
                error_msg = None

            return {
                "status": status,
                "response_time": response_time,
                "error_message": error_msg,
                "metrics": {
                    "processed_files": file_count,
                    "recent_files": recent_files,
                    "data_freshness_hours": 24 if recent_files > 0 else None
                }
            }

        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "error_message": str(e),
                "response_time": None
            }

    def check_training_pipeline_health(self) -> Dict[str, Any]:
        """Check training pipeline health."""
        try:
            start_time = time.time()

            # Check if models directory exists and has content
            models_dir = self.project_root / "models"
            if not models_dir.exists():
                return {
                    "status": HealthStatus.WARNING,
                    "error_message": "Models directory not found",
                    "response_time": time.time() - start_time
                }

            # Count model files
            model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.joblib"))
            model_count = len(model_files)

            # Check for recent training activity
            logs_dir = self.project_root / "logs"
            recent_training = False

            if logs_dir.exists():
                training_logs = list(logs_dir.glob("*training*.log"))
                cutoff_time = datetime.now() - timedelta(hours=24)

                for log_file in training_logs:
                    if log_file.stat().st_mtime > cutoff_time.timestamp():
                        recent_training = True
                        break

            response_time = time.time() - start_time

            # Determine status
            if model_count == 0:
                status = HealthStatus.WARNING
                error_msg = "No trained models found"
            elif not recent_training:
                status = HealthStatus.WARNING
                error_msg = "No recent training activity"
            else:
                status = HealthStatus.HEALTHY
                error_msg = None

            return {
                "status": status,
                "response_time": response_time,
                "error_message": error_msg,
                "metrics": {
                    "model_count": model_count,
                    "recent_training": recent_training
                }
            }

        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "error_message": str(e),
                "response_time": None
            }

    def check_dashboard_health(self, dashboard_name: str) -> Dict[str, Any]:
        """Check dashboard health by attempting connection."""
        try:
            import socket

            # Map dashboard names to ports
            port_mapping = {
                "main_dashboard": 8501,
                "training_dashboard": 8502,
                "memory_dashboard": 8503,
                "analytics_dashboard": 8504
            }

            port = port_mapping.get(dashboard_name)
            if not port:
                return {
                    "status": HealthStatus.UNKNOWN,
                    "error_message": f"Unknown dashboard: {dashboard_name}"
                }

            start_time = time.time()

            # Try to connect to dashboard port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('localhost', port))
            sock.close()

            response_time = time.time() - start_time

            if result == 0:
                return {
                    "status": HealthStatus.HEALTHY,
                    "response_time": response_time,
                    "metrics": {"port": port}
                }
            else:
                return {
                    "status": HealthStatus.CRITICAL,
                    "error_message": f"Dashboard not accessible on port {port}",
                    "response_time": response_time,
                    "metrics": {"port": port}
                }

        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "error_message": str(e),
                "response_time": None
            }

    def check_data_integrity_health(self) -> Dict[str, Any]:
        """Check data integrity and quality."""
        try:
            start_time = time.time()

            data_dir = self.project_root / "data" / "processed"
            if not data_dir.exists():
                return {
                    "status": HealthStatus.CRITICAL,
                    "error_message": "Data directory not found",
                    "response_time": time.time() - start_time
                }

            # Check for data files
            data_files = list(data_dir.glob("*.parquet")) + list(data_dir.glob("*.csv"))
            file_count = len(data_files)

            if file_count == 0:
                return {
                    "status": HealthStatus.WARNING,
                    "error_message": "No data files found",
                    "response_time": time.time() - start_time
                }

            # Check file sizes and ages
            total_size = 0
            recent_files = 0
            cutoff_time = datetime.now() - timedelta(hours=24)

            for file_path in data_files:
                stat = file_path.stat()
                total_size += stat.st_size

                if stat.st_mtime > cutoff_time.timestamp():
                    recent_files += 1

            response_time = time.time() - start_time

            return {
                "status": HealthStatus.HEALTHY,
                "response_time": response_time,
                "metrics": {
                    "file_count": file_count,
                    "total_size_mb": total_size / (1024 * 1024),
                    "recent_files": recent_files
                }
            }

        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "error_message": str(e),
                "response_time": None
            }

    def check_model_storage_health(self) -> Dict[str, Any]:
        """Check model storage health."""
        try:
            start_time = time.time()

            models_dir = self.project_root / "models"
            if not models_dir.exists():
                return {
                    "status": HealthStatus.WARNING,
                    "error_message": "Models directory not found",
                    "response_time": time.time() - start_time
                }

            # Check for model files
            model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.joblib"))
            model_count = len(model_files)

            # Check model file sizes and ages
            total_size = 0
            recent_models = 0
            cutoff_time = datetime.now() - timedelta(hours=24)

            for file_path in model_files:
                stat = file_path.stat()
                total_size += stat.st_size

                if stat.st_mtime > cutoff_time.timestamp():
                    recent_models += 1

            response_time = time.time() - start_time

            if model_count == 0:
                status = HealthStatus.WARNING
                error_msg = "No model files found"
            else:
                status = HealthStatus.HEALTHY
                error_msg = None

            return {
                "status": status,
                "response_time": response_time,
                "error_message": error_msg,
                "metrics": {
                    "model_count": model_count,
                    "total_size_mb": total_size / (1024 * 1024),
                    "recent_models": recent_models
                }
            }

        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "error_message": str(e),
                "response_time": None
            }

    def _check_network_connectivity(self) -> str:
        """Check basic network connectivity."""
        try:
            import urllib.request
            urllib.request.urlopen('http://www.google.com', timeout=5)
            return "connected"
        except:
            return "disconnected"

    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        logger.info("Generating comprehensive health report...")

        # Run all health checks
        component_results = self.check_all_components()

        # Calculate overall system health
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 0,
            HealthStatus.CRITICAL: 0,
            HealthStatus.UNKNOWN: 0
        }

        for result in component_results.values():
            status = result.get("status", HealthStatus.UNKNOWN)
            status_counts[status] += 1

        # Determine overall status
        if status_counts[HealthStatus.CRITICAL] > 0:
            overall_status = HealthStatus.CRITICAL
        elif status_counts[HealthStatus.WARNING] > 0:
            overall_status = HealthStatus.WARNING
        elif status_counts[HealthStatus.HEALTHY] > 0:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN

        # Generate recommendations
        recommendations = self._generate_recommendations(component_results)

        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "component_summary": status_counts,
            "components": component_results,
            "recommendations": recommendations,
            "system_info": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform,
                "project_root": str(self.project_root)
            }
        }

        return report

    def _generate_recommendations(self, component_results: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on component results."""
        recommendations = []

        # Check system resources
        system_result = component_results.get("system", {})
        if system_result.get("status") == HealthStatus.CRITICAL:
            recommendations.append("Critical: High system resource usage detected. Consider scaling resources.")
        elif system_result.get("status") == HealthStatus.WARNING:
            recommendations.append("Warning: Elevated system resource usage. Monitor closely.")

        # Check data pipeline
        data_result = component_results.get("data_pipeline", {})
        if data_result.get("status") == HealthStatus.CRITICAL:
            recommendations.append("Critical: Data pipeline is not functioning. Check data sources and processing.")
        elif data_result.get("status") == HealthStatus.WARNING:
            recommendations.append("Warning: Data pipeline issues detected. Review recent processing activity.")

        # Check training pipeline
        training_result = component_results.get("training_pipeline", {})
        if training_result.get("status") == HealthStatus.CRITICAL:
            recommendations.append("Critical: Training pipeline failure. Check model training configuration.")
        elif training_result.get("status") == HealthStatus.WARNING:
            recommendations.append("Warning: Training pipeline issues. Review recent training activity.")

        # Check dashboards
        dashboard_issues = []
        for name in ["main_dashboard", "training_dashboard", "memory_dashboard", "analytics_dashboard"]:
            result = component_results.get(name, {})
            if result.get("status") != HealthStatus.HEALTHY:
                dashboard_issues.append(name.replace("_", " ").title())

        if dashboard_issues:
            recommendations.append(f"Dashboard issues detected: {', '.join(dashboard_issues)}")

        # General recommendations
        if not recommendations:
            recommendations.append("All systems healthy. Continue monitoring.")

        return recommendations

    def export_health_report(self, output_path: Optional[Path] = None) -> str:
        """Export health report to file."""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.project_root / "logs" / f"health_report_{timestamp}.json"

        output_path.parent.mkdir(exist_ok=True)

        report = self.generate_health_report()

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Health report exported to: {output_path}")
        return str(output_path)

def main():
    """Main health check function."""
    import argparse

    parser = argparse.ArgumentParser(description="FXorcist Health Check System")
    parser.add_argument('action', choices=[
        'check', 'monitor', 'report', 'export', 'status'
    ], help='Action to perform')
    parser.add_argument('--component', '-c', help='Specific component to check')
    parser.add_argument('--interval', '-i', type=int, default=60,
                       help='Monitoring interval in seconds')
    parser.add_argument('--output', '-o', help='Output file path for reports')

    args = parser.parse_args()

    health_checker = HealthChecker()

    if args.action == 'check':
        if args.component:
            # Check specific component
            if args.component in health_checker.components:
                result = health_checker.check_all_components()
                component_result = result.get(args.component, {})
                print(json.dumps(component_result, indent=2))
            else:
                print(f"❌ Component '{args.component}' not found")
                print("Available components:")
                for name in health_checker.components.keys():
                    print(f"  - {name}")
        else:
            # Check all components
            results = health_checker.check_all_components()
            print(json.dumps(results, indent=2))

    elif args.action == 'monitor':
        print(f"👁️  Starting health monitoring (interval: {args.interval}s)...")
        health_checker.start_monitoring(args.interval)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping monitoring...")
            health_checker.stop_monitoring()

    elif args.action == 'report':
        report = health_checker.generate_health_report()
        print(json.dumps(report, indent=2))

    elif args.action == 'export':
        output_path = health_checker.export_health_report(
            Path(args.output) if args.output else None
        )
        print(f"✅ Health report exported to: {output_path}")

    elif args.action == 'status':
        # Quick status overview
        results = health_checker.check_all_components()

        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 0,
            HealthStatus.CRITICAL: 0,
            HealthStatus.UNKNOWN: 0
        }

        for result in results.values():
            status = result.get("status", HealthStatus.UNKNOWN)
            status_counts[status] += 1

        print("🏥 FXorcist Health Status")
        print("=" * 30)
        print(f"Healthy:   {status_counts[HealthStatus.HEALTHY]}")
        print(f"Warning:   {status_counts[HealthStatus.WARNING]}")
        print(f"Critical:  {status_counts[HealthStatus.CRITICAL]}")
        print(f"Unknown:   {status_counts[HealthStatus.UNKNOWN]}")

        if status_counts[HealthStatus.CRITICAL] > 0:
            print("\n🔴 Critical Issues Detected!")
        elif status_counts[HealthStatus.WARNING] > 0:
            print("\n🟡 Warning Issues Detected!")
        else:
            print("\n🟢 All Systems Healthy!")

if __name__ == "__main__":
    main()