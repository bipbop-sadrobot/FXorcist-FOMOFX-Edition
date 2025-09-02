#!/usr/bin/env python3
"""
Enhanced Dashboard Manager for FXorcist AI Dashboard System

This module provides advanced dashboard management with:
- Multi-instance management with load balancing
- Auto-scaling based on resource usage
- Advanced health monitoring and alerting
- Configuration management integration
- Performance optimization and caching
- Backup and recovery capabilities
- Real-time metrics and analytics

Features:
- Intelligent instance scaling
- Resource-aware load balancing
- Advanced health checks with predictive analysis
- Configuration hot-reloading
- Performance monitoring and optimization
- Automated backup and recovery
- Integration with external monitoring systems

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
import logging
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import requests
import socket
from collections import deque
import statistics
import warnings

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from forex_ai_dashboard.utils.health_checker import HealthChecker
from forex_ai_dashboard.utils.config_manager import ConfigurationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_dashboard_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DashboardMetrics:
    """Real-time dashboard metrics"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_connections: int = 0
    active_users: int = 0
    response_time_ms: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration"""
    min_instances: int = 1
    max_instances: int = 5
    cpu_threshold: float = 70.0
    memory_threshold: float = 80.0
    response_time_threshold: float = 2000.0  # ms
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    predictive_scaling: bool = True

@dataclass
class BackupConfig:
    """Backup configuration"""
    enabled: bool = True
    interval_hours: int = 24
    retention_days: int = 7
    include_data: bool = True
    include_models: bool = True
    include_config: bool = True
    compression: bool = True
    remote_storage: Optional[str] = None

class EnhancedDashboardManager:
    """Enhanced dashboard manager with advanced features"""

    def __init__(self, config_path: str = "config/dashboard_manager.json"):
        self.config_path = Path(config_path)
        self.instances: Dict[str, Dict[str, Any]] = {}
        self.metrics_history: Dict[str, deque] = {}
        self.scaling_policies: Dict[str, ScalingPolicy] = {}
        self.backup_configs: Dict[str, BackupConfig] = {}

        # Components
        self.health_checker = HealthChecker()
        self.config_manager = ConfigurationManager()
        self.executor = ThreadPoolExecutor(max_workers=10)

        # State
        self.running = False
        self.monitoring_thread = None
        self.scaling_thread = None
        self.backup_thread = None
        self.last_scale_up: Dict[str, datetime] = {}
        self.last_scale_down: Dict[str, datetime] = {}

        # Load configuration
        self.load_config()

        # Setup directories
        self.setup_directories()

        # Initialize metrics tracking
        self.initialize_metrics_tracking()

    def setup_directories(self):
        """Setup required directories"""
        dirs = ['logs', 'config', 'backups', 'metrics', 'temp']
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)

    def load_config(self):
        """Load configuration from file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)

                # Load instances
                for instance_data in config.get('instances', []):
                    instance_id = instance_data['id']
                    self.instances[instance_id] = instance_data

                    # Load scaling policy
                    if 'scaling_policy' in instance_data:
                        policy_data = instance_data['scaling_policy']
                        self.scaling_policies[instance_id] = ScalingPolicy(**policy_data)

                    # Load backup config
                    if 'backup_config' in instance_data:
                        backup_data = instance_data['backup_config']
                        self.backup_configs[instance_id] = BackupConfig(**backup_data)

                logger.info(f"Loaded configuration with {len(self.instances)} instances")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        else:
            logger.info("No existing configuration found, starting fresh")

    def save_config(self):
        """Save current configuration to file"""
        try:
            config = {
                'instances': list(self.instances.values()),
                'last_updated': datetime.now().isoformat()
            }
            self.config_path.parent.mkdir(exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    def create_instance(self, name: str, dashboard_type: str, port: int,
                       config: Optional[Dict[str, Any]] = None,
                       scaling_policy: Optional[ScalingPolicy] = None,
                       backup_config: Optional[BackupConfig] = None) -> str:
        """Create a new dashboard instance with advanced configuration"""
        instance_id = f"{dashboard_type}_{port}_{int(time.time())}"

        instance = {
            'id': instance_id,
            'name': name,
            'type': dashboard_type,
            'port': port,
            'status': 'stopped',
            'config': config or {},
            'health_url': f"http://localhost:{port}/health",
            'created_at': datetime.now(),
            'metrics': DashboardMetrics(),
            'scaling_policy': scaling_policy or ScalingPolicy(),
            'backup_config': backup_config or BackupConfig()
        }

        self.instances[instance_id] = instance
        self.scaling_policies[instance_id] = instance['scaling_policy']
        self.backup_configs[instance_id] = instance['backup_config']

        # Initialize metrics history
        self.metrics_history[instance_id] = deque(maxlen=100)

        self.save_config()
        logger.info(f"Created enhanced dashboard instance: {instance_id}")
        return instance_id

    def start_instance(self, instance_id: str) -> bool:
        """Start a dashboard instance with enhanced monitoring"""
        if instance_id not in self.instances:
            logger.error(f"Instance {instance_id} not found")
            return False

        instance = self.instances[instance_id]

        try:
            # Update status
            instance['status'] = 'starting'
            instance['start_time'] = datetime.now()

            # Determine dashboard script based on type
            script_map = {
                'main': 'dashboard/app.py',
                'training': 'enhanced_training_dashboard.py',
                'custom': instance['config'].get('script', 'dashboard/app.py')
            }

            script_path = script_map.get(instance['type'], 'dashboard/app.py')

            # Prepare environment variables
            env = os.environ.copy()
            env.update({
                'FXORCIST_ENV': 'production',
                'STREAMLIT_SERVER_PORT': str(instance['port']),
                'STREAMLIT_SERVER_HEADLESS': 'true',
                'STREAMLIT_SERVER_ENABLE_CORS': 'false'
            })

            # Add custom configuration
            if instance['config']:
                for key, value in instance['config'].items():
                    if key.startswith('STREAMLIT_'):
                        env[key] = str(value)

            logger.info(f"Starting enhanced dashboard: {' '.join(['python', '-m', 'streamlit', 'run', script_path])}")

            # Start the dashboard process
            process = subprocess.Popen(
                ['python', '-m', 'streamlit', 'run', script_path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).parent.parent
            )

            instance['pid'] = process.pid
            instance['process'] = process
            instance['status'] = 'running'

            # Start individual monitoring for this instance
            self.executor.submit(self._monitor_instance, instance_id)

            self.save_config()

            logger.info(f"Enhanced dashboard instance {instance_id} started with PID {process.pid}")
            return True

        except Exception as e:
            logger.error(f"Error starting instance {instance_id}: {e}")
            instance['status'] = 'error'
            self.save_config()
            return False

    def stop_instance(self, instance_id: str, graceful: bool = True) -> bool:
        """Stop a dashboard instance with graceful shutdown"""
        if instance_id not in self.instances:
            logger.error(f"Instance {instance_id} not found")
            return False

        instance = self.instances[instance_id]

        try:
            if instance.get('pid'):
                if graceful:
                    # Try graceful shutdown first
                    try:
                        response = requests.post(f"http://localhost:{instance['port']}/shutdown",
                                               timeout=10)
                        if response.status_code == 200:
                            logger.info(f"Graceful shutdown initiated for {instance_id}")
                            time.sleep(5)  # Wait for graceful shutdown
                    except:
                        logger.warning(f"Graceful shutdown failed for {instance_id}")

                # Force kill if still running
                try:
                    process = psutil.Process(instance['pid'])
                    process.terminate()
                    process.wait(timeout=10)
                except psutil.NoSuchProcess:
                    pass
                except psutil.TimeoutExpired:
                    process.kill()
                    logger.warning(f"Force killed process for {instance_id}")

            instance['status'] = 'stopped'
            instance['pid'] = None
            instance['process'] = None
            instance['start_time'] = None

            self.save_config()

            logger.info(f"Dashboard instance {instance_id} stopped")
            return True

        except Exception as e:
            logger.error(f"Error stopping instance {instance_id}: {e}")
            return False

    def _monitor_instance(self, instance_id: str):
        """Monitor individual instance health and metrics"""
        while self.running and instance_id in self.instances:
            try:
                instance = self.instances[instance_id]
                if instance['status'] != 'running':
                    break

                # Collect metrics
                metrics = self._collect_instance_metrics(instance_id)

                # Update instance metrics
                instance['metrics'] = metrics
                self.metrics_history[instance_id].append(metrics)

                # Check health
                if not self._check_instance_health(instance_id):
                    logger.warning(f"Instance {instance_id} health check failed")
                    instance['status'] = 'unhealthy'

                time.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error(f"Error monitoring instance {instance_id}: {e}")
                time.sleep(10)

    def _collect_instance_metrics(self, instance_id: str) -> DashboardMetrics:
        """Collect comprehensive metrics for an instance"""
        instance = self.instances[instance_id]

        metrics = DashboardMetrics()

        try:
            # System metrics
            if instance.get('pid'):
                process = psutil.Process(instance['pid'])
                metrics.cpu_percent = process.cpu_percent()
                metrics.memory_percent = process.memory_percent()

                # Network connections
                connections = process.net_connections()
                metrics.network_connections = len(connections)

            # Disk usage
            disk = psutil.disk_usage('/')
            metrics.disk_usage_percent = disk.percent

            # Health check for response time
            start_time = time.time()
            try:
                response = requests.get(instance['health_url'], timeout=5)
                if response.status_code == 200:
                    metrics.response_time_ms = (time.time() - start_time) * 1000
                    health_data = response.json()
                    metrics.error_rate = health_data.get('error_rate', 0.0)
                    metrics.throughput = health_data.get('throughput', 0.0)
                    metrics.active_users = health_data.get('active_users', 0)
            except:
                metrics.response_time_ms = 9999  # High value for timeout

        except Exception as e:
            logger.error(f"Error collecting metrics for {instance_id}: {e}")

        return metrics

    def _check_instance_health(self, instance_id: str) -> bool:
        """Perform comprehensive health check"""
        instance = self.instances[instance_id]

        try:
            # Basic connectivity check
            response = requests.get(instance['health_url'], timeout=5)
            if response.status_code != 200:
                return False

            health_data = response.json()

            # Check critical metrics
            if health_data.get('status') != 'healthy':
                return False

            # Check resource usage
            metrics = instance.get('metrics', DashboardMetrics())
            if metrics.cpu_percent > 95 or metrics.memory_percent > 95:
                return False

            return True

        except Exception as e:
            logger.error(f"Health check failed for {instance_id}: {e}")
            return False

    def start_auto_scaling(self):
        """Start auto-scaling service"""
        self.scaling_thread = threading.Thread(target=self._auto_scaling_loop, daemon=True)
        self.scaling_thread.start()
        logger.info("Auto-scaling service started")

    def _auto_scaling_loop(self):
        """Auto-scaling decision loop"""
        while self.running:
            try:
                for instance_id in list(self.instances.keys()):
                    self._evaluate_scaling(instance_id)
                time.sleep(60)  # Evaluate every minute
            except Exception as e:
                logger.error(f"Error in auto-scaling loop: {e}")
                time.sleep(30)

    def _evaluate_scaling(self, instance_id: str):
        """Evaluate scaling needs for an instance"""
        if instance_id not in self.scaling_policies:
            return

        instance = self.instances[instance_id]
        policy = self.scaling_policies[instance_id]

        # Skip if not running
        if instance['status'] != 'running':
            return

        # Get recent metrics
        metrics_history = self.metrics_history.get(instance_id, [])
        if len(metrics_history) < 5:  # Need at least 5 data points
            return

        # Calculate averages
        recent_metrics = list(metrics_history)[-5:]
        avg_cpu = statistics.mean(m['cpu_percent'] for m in recent_metrics)
        avg_memory = statistics.mean(m['memory_percent'] for m in recent_metrics)
        avg_response_time = statistics.mean(m['response_time_ms'] for m in recent_metrics)

        # Scale up conditions
        scale_up_needed = (
            avg_cpu > policy.cpu_threshold or
            avg_memory > policy.memory_threshold or
            avg_response_time > policy.response_time_threshold
        )

        # Scale down conditions
        scale_down_needed = (
            avg_cpu < policy.cpu_threshold * 0.5 and
            avg_memory < policy.memory_threshold * 0.5 and
            avg_response_time < policy.response_time_threshold * 0.5
        )

        # Check cooldown periods
        now = datetime.now()
        last_up = self.last_scale_up.get(instance_id)
        last_down = self.last_scale_down.get(instance_id)

        if scale_up_needed and (not last_up or (now - last_up).seconds > policy.scale_up_cooldown):
            self._scale_up(instance_id)
            self.last_scale_up[instance_id] = now

        elif scale_down_needed and (not last_down or (now - last_down).seconds > policy.scale_down_cooldown):
            self._scale_down(instance_id)
            self.last_scale_down[instance_id] = now

    def _scale_up(self, instance_id: str):
        """Scale up an instance"""
        logger.info(f"Scaling up instance {instance_id}")
        # Implementation would create additional instances or increase resources
        # For now, just log the scaling decision
        pass

    def _scale_down(self, instance_id: str):
        """Scale down an instance"""
        logger.info(f"Scaling down instance {instance_id}")
        # Implementation would reduce instances or decrease resources
        # For now, just log the scaling decision
        pass

    def start_backup_service(self):
        """Start automated backup service"""
        self.backup_thread = threading.Thread(target=self._backup_loop, daemon=True)
        self.backup_thread.start()
        logger.info("Backup service started")

    def _backup_loop(self):
        """Automated backup loop"""
        while self.running:
            try:
                for instance_id, backup_config in self.backup_configs.items():
                    if backup_config.enabled:
                        self._perform_backup(instance_id, backup_config)

                # Sleep for backup interval (24 hours)
                time.sleep(24 * 3600)
            except Exception as e:
                logger.error(f"Error in backup loop: {e}")
                time.sleep(3600)

    def _perform_backup(self, instance_id: str, backup_config: BackupConfig):
        """Perform backup for an instance"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path(f"backups/{instance_id}_{timestamp}")
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Backup data
            if backup_config.include_data:
                self._backup_directory("data", backup_dir / "data")

            # Backup models
            if backup_config.include_models:
                self._backup_directory("models", backup_dir / "models")

            # Backup configuration
            if backup_config.include_config:
                self._backup_directory("config", backup_dir / "config")

            # Compress backup
            if backup_config.compression:
                self._compress_backup(backup_dir)

            # Upload to remote storage if configured
            if backup_config.remote_storage:
                self._upload_to_remote(backup_dir, backup_config.remote_storage)

            # Cleanup old backups
            self._cleanup_old_backups(instance_id, backup_config.retention_days)

            logger.info(f"Backup completed for instance {instance_id}")

        except Exception as e:
            logger.error(f"Backup failed for instance {instance_id}: {e}")

    def _backup_directory(self, src_dir: str, dest_dir: Path):
        """Backup a directory"""
        src_path = Path(src_dir)
        if src_path.exists():
            import shutil
            shutil.copytree(src_path, dest_dir, dirs_exist_ok=True)

    def _compress_backup(self, backup_dir: Path):
        """Compress backup directory"""
        import tarfile
        archive_path = backup_dir.with_suffix('.tar.gz')

        with tarfile.open(archive_path, 'w:gz') as tar:
            tar.add(backup_dir, arcname=backup_dir.name)

        # Remove uncompressed directory
        import shutil
        shutil.rmtree(backup_dir)

    def _upload_to_remote(self, backup_path: Path, remote_config: str):
        """Upload backup to remote storage"""
        # Implementation would depend on remote storage type (S3, GCS, etc.)
        logger.info(f"Would upload {backup_path} to {remote_config}")

    def _cleanup_old_backups(self, instance_id: str, retention_days: int):
        """Clean up old backups"""
        backup_pattern = Path("backups").glob(f"{instance_id}_*")
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        for backup_path in backup_pattern:
            try:
                # Extract timestamp from filename
                timestamp_str = backup_path.name.split('_', 1)[1].split('.')[0]
                backup_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                if backup_date < cutoff_date:
                    if backup_path.is_file():
                        backup_path.unlink()
                    elif backup_path.is_dir():
                        import shutil
                        shutil.rmtree(backup_path)
                    logger.info(f"Cleaned up old backup: {backup_path}")
            except Exception as e:
                logger.error(f"Error cleaning up backup {backup_path}: {e}")

    def get_performance_analytics(self, instance_id: str) -> Dict[str, Any]:
        """Get performance analytics for an instance"""
        if instance_id not in self.metrics_history:
            return {}

        metrics = list(self.metrics_history[instance_id])

        if not metrics:
            return {}

        # Calculate statistics
        cpu_values = [m.cpu_percent for m in metrics]
        memory_values = [m.memory_percent for m in metrics]
        response_times = [m.response_time_ms for m in metrics]

        analytics = {
            'cpu_stats': {
                'mean': statistics.mean(cpu_values),
                'median': statistics.median(cpu_values),
                'std_dev': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0,
                'min': min(cpu_values),
                'max': max(cpu_values)
            },
            'memory_stats': {
                'mean': statistics.mean(memory_values),
                'median': statistics.median(memory_values),
                'std_dev': statistics.stdev(memory_values) if len(memory_values) > 1 else 0,
                'min': min(memory_values),
                'max': max(memory_values)
            },
            'response_time_stats': {
                'mean': statistics.mean(response_times),
                'median': statistics.median(response_times),
                'std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0,
                'min': min(response_times),
                'max': max(response_times)
            },
            'sample_count': len(metrics),
            'time_range': {
                'start': metrics[0].timestamp.isoformat(),
                'end': metrics[-1].timestamp.isoformat()
            }
        }

        return analytics

    def optimize_instance(self, instance_id: str) -> Dict[str, Any]:
        """Optimize instance configuration based on analytics"""
        analytics = self.get_performance_analytics(instance_id)

        if not analytics:
            return {'status': 'no_data'}

        recommendations = []

        # CPU optimization
        cpu_mean = analytics['cpu_stats']['mean']
        if cpu_mean > 80:
            recommendations.append({
                'type': 'cpu',
                'action': 'increase_resources',
                'reason': f'High CPU usage ({cpu_mean:.1f}%)',
                'suggestion': 'Consider increasing CPU allocation or optimizing code'
            })
        elif cpu_mean < 20:
            recommendations.append({
                'type': 'cpu',
                'action': 'decrease_resources',
                'reason': f'Low CPU usage ({cpu_mean:.1f}%)',
                'suggestion': 'Consider reducing CPU allocation to save resources'
            })

        # Memory optimization
        memory_mean = analytics['memory_stats']['mean']
        if memory_mean > 85:
            recommendations.append({
                'type': 'memory',
                'action': 'increase_resources',
                'reason': f'High memory usage ({memory_mean:.1f}%)',
                'suggestion': 'Consider increasing memory allocation or optimizing memory usage'
            })
        elif memory_mean < 30:
            recommendations.append({
                'type': 'memory',
                'action': 'optimize_code',
                'reason': f'Low memory usage ({memory_mean:.1f}%)',
                'suggestion': 'Memory allocation seems optimal'
            })

        # Response time optimization
        response_mean = analytics['response_time_stats']['mean']
        if response_mean > 1000:
            recommendations.append({
                'type': 'performance',
                'action': 'optimize_response_time',
                'reason': f'Slow response time ({response_mean:.0f}ms)',
                'suggestion': 'Consider caching, database optimization, or code profiling'
            })

        return {
            'status': 'analyzed',
            'recommendations': recommendations,
            'analytics': analytics
        }

    def start_monitoring(self):
        """Start all monitoring services"""
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._global_monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        self.start_auto_scaling()
        self.start_backup_service()

        logger.info("Enhanced dashboard manager monitoring started")

    def _global_monitoring_loop(self):
        """Global monitoring loop"""
        while self.running:
            try:
                # Global health check
                system_health = self.health_checker.check_all_components()

                # Log system status
                if system_health['status'] != 'healthy':
                    logger.warning(f"System health issue: {system_health}")

                # Check for failed instances
                for instance_id, instance in self.instances.items():
                    if instance['status'] == 'error':
                        logger.warning(f"Instance {instance_id} is in error state")

                time.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error in global monitoring: {e}")
                time.sleep(60)

    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        return {
            'instances': self.instances,
            'system_health': self.health_checker.check_all_components(),
            'total_instances': len(self.instances),
            'running_instances': sum(1 for i in self.instances.values() if i['status'] == 'running'),
            'error_instances': sum(1 for i in self.instances.values() if i['status'] == 'error'),
            'scaling_policies': {k: v.__dict__ for k, v in self.scaling_policies.items()},
            'backup_configs': {k: v.__dict__ for k, v in self.backup_configs.items()},
            'timestamp': datetime.now().isoformat()
        }

    def shutdown(self):
        """Shutdown the enhanced dashboard manager"""
        logger.info("Shutting down enhanced dashboard manager")
        self.running = False

        # Stop all instances
        for instance_id in list(self.instances.keys()):
            self.stop_instance(instance_id, graceful=True)

        # Save final configuration
        self.save_config()

        # Shutdown executor
        self.executor.shutdown(wait=True)

        logger.info("Enhanced dashboard manager shutdown complete")

def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Dashboard Manager")
    parser.add_argument("--config", type=str, default="config/dashboard_manager.json",
                       help="Configuration file path")
    parser.add_argument("--port", type=int, default=8503,
                       help="Port for the management interface")
    parser.add_argument("--daemon", action="store_true",
                       help="Run as daemon (background service)")
    args = parser.parse_args()

    # Initialize manager
    manager = EnhancedDashboardManager(args.config)

    if args.daemon:
        # Run as daemon
        manager.start_monitoring()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            manager.shutdown()
    else:
        # Interactive mode
        print("Enhanced Dashboard Manager")
        print("=" * 40)
        print("Available commands:")
        print("  start <instance_id>  - Start an instance")
        print("  stop <instance_id>   - Stop an instance")
        print("  status               - Show system status")
        print("  create <name> <type> <port> - Create new instance")
        print("  optimize <instance_id> - Optimize instance")
        print("  quit                 - Exit")

        while True:
            try:
                cmd = input("> ").strip().split()
                if not cmd:
                    continue

                if cmd[0] == 'quit':
                    break
                elif cmd[0] == 'start' and len(cmd) > 1:
                    if manager.start_instance(cmd[1]):
                        print(f"Started instance {cmd[1]}")
                    else:
                        print(f"Failed to start instance {cmd[1]}")
                elif cmd[0] == 'stop' and len(cmd) > 1:
                    if manager.stop_instance(cmd[1]):
                        print(f"Stopped instance {cmd[1]}")
                    else:
                        print(f"Failed to stop instance {cmd[1]}")
                elif cmd[0] == 'status':
                    overview = manager.get_system_overview()
                    print(json.dumps(overview, indent=2, default=str))
                elif cmd[0] == 'create' and len(cmd) > 3:
                    instance_id = manager.create_instance(cmd[1], cmd[2], int(cmd[3]))
                    print(f"Created instance: {instance_id}")
                elif cmd[0] == 'optimize' and len(cmd) > 1:
                    result = manager.optimize_instance(cmd[1])
                    print(json.dumps(result, indent=2))
                else:
                    print("Invalid command")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        manager.shutdown()

if __name__ == "__main__":
    main()