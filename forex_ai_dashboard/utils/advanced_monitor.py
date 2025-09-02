#!/usr/bin/env python3
"""
Advanced Monitoring and Analytics System for FXorcist

This module provides comprehensive monitoring and analytics with:
- Real-time performance metrics collection
- Historical data analysis and trend detection
- Automated performance optimization recommendations
- Comprehensive system health dashboards
- Predictive analytics and anomaly detection
- Custom metrics and alerting system

Author: FXorcist Development Team
Version: 2.0
Date: September 2, 2025
"""

import os
import sys
import json
import time
import threading
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import statistics
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import psutil
import requests
import warnings

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from forex_ai_dashboard.utils.health_checker import HealthChecker
from forex_ai_dashboard.utils.advanced_auto_scaler import ScalingMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    timestamp: datetime
    system_metrics: Dict[str, float]
    application_metrics: Dict[str, float]
    business_metrics: Dict[str, float]
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrendAnalysis:
    """Trend analysis results"""
    metric_name: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # 0-1
    confidence: float
    seasonality_detected: bool
    anomaly_score: float
    forecast_next_hour: float
    forecast_next_day: float

@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""
    category: str  # 'system', 'application', 'infrastructure'
    priority: str  # 'high', 'medium', 'low'
    title: str
    description: str
    impact_score: float  # 0-1
    effort_estimate: str  # 'low', 'medium', 'high'
    implementation_steps: List[str]
    expected_benefits: Dict[str, float]
    prerequisites: List[str] = field(default_factory=list)

@dataclass
class AlertCondition:
    """Alert condition configuration"""
    name: str
    metric: str
    operator: str  # '>', '<', '>=', '<=', '==', '!='
    threshold: float
    duration_minutes: int
    severity: str  # 'critical', 'warning', 'info'
    enabled: bool = True
    cooldown_minutes: int = 5

@dataclass
class Alert:
    """Active alert"""
    id: str
    condition_name: str
    severity: str
    message: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    value: float
    threshold: float
    duration_minutes: int

class AdvancedMonitor:
    """Advanced monitoring and analytics system"""

    def __init__(self, instance_id: str = "system", config_path: str = "config/monitor.json"):
        self.instance_id = instance_id
        self.config_path = Path(config_path)

        # Metrics storage
        self.metrics_history: deque = deque(maxlen=5000)  # Store last 5000 data points
        self.alerts: Dict[str, Alert] = {}
        self.alert_conditions: Dict[str, AlertCondition] = {}

        # Analytics models
        self.trend_models: Dict[str, Any] = {}
        self.forecasting_models: Dict[str, Any] = {}
        self.anomaly_detectors: Dict[str, Any] = {}

        # Performance baselines
        self.baselines: Dict[str, Dict[str, float]] = {}

        # Threading
        self.running = False
        self.collection_thread = None
        self.analysis_thread = None
        self.alert_thread = None

        # Health checker integration
        self.health_checker = HealthChecker()

        # Load configuration
        self.load_config()

        # Initialize analytics
        self.initialize_analytics()

    def load_config(self):
        """Load monitoring configuration"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)

                # Load alert conditions
                for condition_data in config.get('alert_conditions', []):
                    condition = AlertCondition(**condition_data)
                    self.alert_conditions[condition.name] = condition

                # Load baselines
                self.baselines = config.get('baselines', {})

                logger.info(f"Loaded monitoring configuration with {len(self.alert_conditions)} alert conditions")

            except Exception as e:
                logger.error(f"Error loading monitoring configuration: {e}")

    def save_config(self):
        """Save current configuration"""
        try:
            config = {
                'alert_conditions': [c.__dict__ for c in self.alert_conditions.values()],
                'baselines': self.baselines,
                'last_updated': datetime.now().isoformat()
            }

            self.config_path.parent.mkdir(exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving monitoring configuration: {e}")

    def collect_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics"""
        timestamp = datetime.now()

        # System metrics
        system_metrics = self._collect_system_metrics()

        # Application metrics
        application_metrics = self._collect_application_metrics()

        # Business metrics
        business_metrics = self._collect_business_metrics()

        metrics = PerformanceMetrics(
            timestamp=timestamp,
            system_metrics=system_metrics,
            application_metrics=application_metrics,
            business_metrics=business_metrics
        )

        # Store metrics
        self.metrics_history.append(metrics)

        return metrics

    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()

            # Memory metrics
            memory = psutil.virtual_memory()

            # Disk metrics
            disk = psutil.disk_usage('/')

            # Network metrics
            network = psutil.net_io_counters()

            # Process metrics
            fxorcist_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    if any(keyword in proc.info['name'].lower()
                          for keyword in ['fxorcist', 'streamlit', 'python']):
                        fxorcist_processes.append(proc.info)
                except:
                    continue

            return {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'cpu_freq_current': cpu_freq.current if cpu_freq else 0,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'fxorcist_processes': len(fxorcist_processes),
                'total_processes': len(fxorcist_processes)
            }

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}

    def _collect_application_metrics(self) -> Dict[str, float]:
        """Collect application-specific metrics"""
        try:
            metrics = {}

            # Dashboard health metrics
            health_status = self.health_checker.check_all_components()
            metrics['health_score'] = health_status.get('overall_score', 0.0)

            # Try to get metrics from running dashboards
            try:
                # Check main dashboard
                response = requests.get('http://localhost:8501/health', timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    metrics.update({
                        'dashboard_response_time': health_data.get('response_time', 0),
                        'dashboard_active_users': health_data.get('active_users', 0),
                        'dashboard_memory_usage': health_data.get('memory_usage', 0)
                    })
            except:
                pass

            # Database connection metrics (if applicable)
            # Add database-specific metrics here

            # Cache metrics (if applicable)
            # Add cache hit/miss ratios here

            return metrics

        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
            return {}

    def _collect_business_metrics(self) -> Dict[str, float]:
        """Collect business-level metrics"""
        try:
            metrics = {}

            # Data processing metrics
            data_dir = Path('data')
            if data_dir.exists():
                total_files = len(list(data_dir.rglob('*')))
                total_size = sum(f.stat().st_size for f in data_dir.rglob('*') if f.is_file())
                metrics['data_files_count'] = total_files
                metrics['data_total_size_gb'] = total_size / (1024**3)

            # Model metrics
            models_dir = Path('models')
            if models_dir.exists():
                model_files = list(models_dir.rglob('*.pkl')) + list(models_dir.rglob('*.joblib'))
                metrics['model_files_count'] = len(model_files)

            # Log analysis metrics
            logs_dir = Path('logs')
            if logs_dir.exists():
                log_files = list(logs_dir.rglob('*.log'))
                total_log_size = sum(f.stat().st_size for f in log_files)
                metrics['log_files_count'] = len(log_files)
                metrics['log_total_size_mb'] = total_log_size / (1024**2)

                # Error rate from recent logs
                error_count = 0
                total_lines = 0
                for log_file in log_files[-5:]:  # Check last 5 log files
                    try:
                        with open(log_file, 'r') as f:
                            lines = f.readlines()[-1000:]  # Last 1000 lines
                            total_lines += len(lines)
                            error_count += sum(1 for line in lines if 'ERROR' in line.upper())
                    except:
                        pass

                if total_lines > 0:
                    metrics['error_rate_percent'] = (error_count / total_lines) * 100

            return metrics

        except Exception as e:
            logger.error(f"Error collecting business metrics: {e}")
            return {}

    def analyze_trends(self, metric_name: str, hours: int = 24) -> TrendAnalysis:
        """Analyze trends for a specific metric"""
        if not self.metrics_history:
            return TrendAnalysis(
                metric_name=metric_name,
                trend_direction='unknown',
                trend_strength=0.0,
                confidence=0.0,
                seasonality_detected=False,
                anomaly_score=0.0,
                forecast_next_hour=0.0,
                forecast_next_day=0.0
            )

        # Extract metric values
        values = []
        timestamps = []

        cutoff_time = datetime.now() - timedelta(hours=hours)
        for metric in self.metrics_history:
            if metric.timestamp >= cutoff_time:
                # Extract metric value based on category
                if '.' in metric_name:
                    category, key = metric_name.split('.', 1)
                    if category == 'system' and key in metric.system_metrics:
                        values.append(metric.system_metrics[key])
                        timestamps.append(metric.timestamp)
                    elif category == 'application' and key in metric.application_metrics:
                        values.append(metric.application_metrics[key])
                        timestamps.append(metric.timestamp)
                    elif category == 'business' and key in metric.business_metrics:
                        values.append(metric.business_metrics[key])
                        timestamps.append(metric.timestamp)
                else:
                    # Search all categories
                    if metric_name in metric.system_metrics:
                        values.append(metric.system_metrics[metric_name])
                        timestamps.append(metric.timestamp)
                    elif metric_name in metric.application_metrics:
                        values.append(metric.application_metrics[metric_name])
                        timestamps.append(metric.timestamp)
                    elif metric_name in metric.business_metrics:
                        values.append(metric.business_metrics[metric_name])
                        timestamps.append(metric.timestamp)

        if len(values) < 10:
            return TrendAnalysis(
                metric_name=metric_name,
                trend_direction='insufficient_data',
                trend_strength=0.0,
                confidence=0.0,
                seasonality_detected=False,
                anomaly_score=0.0,
                forecast_next_hour=statistics.mean(values) if values else 0.0,
                forecast_next_day=statistics.mean(values) if values else 0.0
            )

        # Calculate trend using linear regression
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)

        # Determine trend direction
        if abs(slope) < 0.01:
            trend_direction = 'stable'
            trend_strength = 0.0
        elif slope > 0:
            trend_direction = 'increasing'
            trend_strength = min(abs(slope) / statistics.mean(values), 1.0)
        else:
            trend_direction = 'decreasing'
            trend_strength = min(abs(slope) / statistics.mean(values), 1.0)

        # Calculate confidence
        confidence = max(0.0, min(1.0, r_value**2))

        # Detect seasonality (simple check for periodic patterns)
        seasonality_detected = self._detect_seasonality(values)

        # Calculate anomaly score (simple z-score based)
        if len(values) > 1:
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values)
            current_val = values[-1]
            anomaly_score = abs(current_val - mean_val) / std_val if std_val > 0 else 0.0
        else:
            anomaly_score = 0.0

        # Simple forecasting
        forecast_next_hour = values[-1] + slope  # Linear extrapolation
        forecast_next_day = values[-1] + slope * 24

        return TrendAnalysis(
            metric_name=metric_name,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            confidence=confidence,
            seasonality_detected=seasonality_detected,
            anomaly_score=anomaly_score,
            forecast_next_hour=forecast_next_hour,
            forecast_next_day=forecast_next_day
        )

    def _detect_seasonality(self, values: List[float], min_period: int = 6) -> bool:
        """Simple seasonality detection using autocorrelation"""
        if len(values) < min_period * 2:
            return False

        # Calculate autocorrelation for different lags
        max_corr = 0.0
        for lag in range(min_period, min(len(values)//2, 24)):  # Check up to 24-hour patterns
            if lag >= len(values):
                break

            corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
            max_corr = max(max_corr, abs(corr))

        return max_corr > 0.6  # Threshold for seasonality detection

    def generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate automated performance optimization recommendations"""
        recommendations = []

        if not self.metrics_history:
            return recommendations

        # Analyze system metrics for optimization opportunities
        system_analysis = self._analyze_system_performance()
        recommendations.extend(system_analysis)

        # Analyze application metrics
        app_analysis = self._analyze_application_performance()
        recommendations.extend(app_analysis)

        # Analyze business metrics
        business_analysis = self._analyze_business_performance()
        recommendations.extend(business_analysis)

        # Sort by impact score
        recommendations.sort(key=lambda x: x.impact_score, reverse=True)

        return recommendations[:10]  # Return top 10 recommendations

    def _analyze_system_performance(self) -> List[OptimizationRecommendation]:
        """Analyze system performance and generate recommendations"""
        recommendations = []

        if len(self.metrics_history) < 10:
            return recommendations

        # Get recent system metrics
        recent_metrics = list(self.metrics_history)[-10:]
        avg_cpu = statistics.mean(m.system_metrics.get('cpu_percent', 0) for m in recent_metrics)
        avg_memory = statistics.mean(m.system_metrics.get('memory_percent', 0) for m in recent_metrics)

        # CPU optimization
        if avg_cpu > 80:
            recommendations.append(OptimizationRecommendation(
                category='system',
                priority='high',
                title='High CPU Usage Detected',
                description=f'Average CPU usage is {avg_cpu:.1f}%. Consider optimizing CPU-intensive operations.',
                impact_score=0.8,
                effort_estimate='medium',
                implementation_steps=[
                    'Profile CPU usage with performance monitoring tools',
                    'Optimize database queries and data processing',
                    'Consider horizontal scaling if load is consistent',
                    'Review and optimize background processes'
                ],
                expected_benefits={
                    'cpu_reduction': 20.0,
                    'response_time_improvement': 15.0
                }
            ))

        # Memory optimization
        if avg_memory > 85:
            recommendations.append(OptimizationRecommendation(
                category='system',
                priority='high',
                title='High Memory Usage Detected',
                description=f'Average memory usage is {avg_memory:.1f}%. Memory optimization needed.',
                impact_score=0.9,
                effort_estimate='medium',
                implementation_steps=[
                    'Implement memory-efficient data processing',
                    'Add memory monitoring and garbage collection optimization',
                    'Consider increasing system memory',
                    'Review memory leaks in application code'
                ],
                expected_benefits={
                    'memory_reduction': 25.0,
                    'stability_improvement': 30.0
                }
            ))

        # Disk optimization
        avg_disk = statistics.mean(m.system_metrics.get('disk_percent', 0) for m in recent_metrics)
        if avg_disk > 90:
            recommendations.append(OptimizationRecommendation(
                category='infrastructure',
                priority='medium',
                title='High Disk Usage',
                description=f'Disk usage is {avg_disk:.1f}%. Storage optimization recommended.',
                impact_score=0.6,
                effort_estimate='low',
                implementation_steps=[
                    'Implement log rotation and cleanup',
                    'Archive old data files',
                    'Add disk space monitoring alerts',
                    'Consider storage expansion'
                ],
                expected_benefits={
                    'disk_space_free': 30.0,
                    'system_stability': 20.0
                }
            ))

        return recommendations

    def _analyze_application_performance(self) -> List[OptimizationRecommendation]:
        """Analyze application performance"""
        recommendations = []

        if len(self.metrics_history) < 10:
            return recommendations

        recent_metrics = list(self.metrics_history)[-10:]

        # Response time analysis
        response_times = [m.application_metrics.get('dashboard_response_time', 0)
                         for m in recent_metrics if m.application_metrics.get('dashboard_response_time', 0) > 0]

        if response_times:
            avg_response_time = statistics.mean(response_times)
            if avg_response_time > 2000:  # 2 seconds
                recommendations.append(OptimizationRecommendation(
                    category='application',
                    priority='high',
                    title='Slow Response Times',
                    description=f'Average response time is {avg_response_time:.0f}ms. Performance optimization needed.',
                    impact_score=0.85,
                    effort_estimate='medium',
                    implementation_steps=[
                        'Implement caching for frequently accessed data',
                        'Optimize database queries and indexes',
                        'Add response time monitoring and profiling',
                        'Consider CDN for static assets'
                    ],
                    expected_benefits={
                        'response_time_improvement': 40.0,
                        'user_experience': 35.0
                    }
                ))

        # Health score analysis
        health_scores = [m.application_metrics.get('health_score', 1.0) for m in recent_metrics]
        if health_scores:
            avg_health = statistics.mean(health_scores)
            if avg_health < 0.8:
                recommendations.append(OptimizationRecommendation(
                    category='application',
                    priority='medium',
                    title='Application Health Issues',
                    description=f'Average health score is {avg_health:.2f}. Review application components.',
                    impact_score=0.7,
                    effort_estimate='medium',
                    implementation_steps=[
                        'Review error logs and exception handling',
                        'Implement comprehensive health checks',
                        'Add circuit breakers for external dependencies',
                        'Improve error recovery mechanisms'
                    ],
                    expected_benefits={
                        'system_reliability': 30.0,
                        'error_reduction': 25.0
                    }
                ))

        return recommendations

    def _analyze_business_performance(self) -> List[OptimizationRecommendation]:
        """Analyze business-level performance"""
        recommendations = []

        if len(self.metrics_history) < 10:
            return recommendations

        recent_metrics = list(self.metrics_history)[-10:]

        # Error rate analysis
        error_rates = [m.business_metrics.get('error_rate_percent', 0) for m in recent_metrics]
        if error_rates:
            avg_error_rate = statistics.mean(error_rates)
            if avg_error_rate > 5.0:
                recommendations.append(OptimizationRecommendation(
                    category='application',
                    priority='high',
                    title='High Error Rate',
                    description=f'Average error rate is {avg_error_rate:.2f}%. Error handling needs improvement.',
                    impact_score=0.9,
                    effort_estimate='high',
                    implementation_steps=[
                        'Implement comprehensive error logging and monitoring',
                        'Add input validation and sanitization',
                        'Improve exception handling throughout application',
                        'Add automated error recovery mechanisms'
                    ],
                    expected_benefits={
                        'error_reduction': 50.0,
                        'system_stability': 40.0
                    }
                ))

        # Data processing analysis
        data_files = [m.business_metrics.get('data_files_count', 0) for m in recent_metrics]
        if data_files and statistics.mean(data_files) > 1000:
            recommendations.append(OptimizationRecommendation(
                category='infrastructure',
                priority='medium',
                title='Large Data Volume',
                description='System is processing large volumes of data. Consider optimization.',
                impact_score=0.6,
                effort_estimate='medium',
                implementation_steps=[
                    'Implement data partitioning and archiving',
                    'Add data compression and efficient storage',
                    'Optimize data processing pipelines',
                    'Consider distributed processing for large datasets'
                ],
                expected_benefits={
                    'processing_speed': 30.0,
                    'storage_efficiency': 25.0
                }
            ))

        return recommendations

    def check_alerts(self) -> List[Alert]:
        """Check alert conditions and return triggered alerts"""
        triggered_alerts = []

        if not self.metrics_history:
            return triggered_alerts

        latest_metrics = self.metrics_history[-1]

        for condition in self.alert_conditions.values():
            if not condition.enabled:
                continue

            # Get metric value
            metric_value = self._get_metric_value(latest_metrics, condition.metric)

            if metric_value is None:
                continue

            # Check condition
            triggered = self._evaluate_condition(metric_value, condition.operator, condition.threshold)

            if triggered:
                # Check if alert already exists and is not in cooldown
                alert_key = f"{condition.name}_{condition.metric}"
                existing_alert = self.alerts.get(alert_key)

                if existing_alert and not existing_alert.resolved_at:
                    # Alert already active, check duration
                    duration = (datetime.now() - existing_alert.triggered_at).total_seconds() / 60
                    if duration >= condition.duration_minutes:
                        # Alert condition met for required duration
                        existing_alert.value = metric_value
                        triggered_alerts.append(existing_alert)
                elif not existing_alert or self._check_cooldown(existing_alert, condition.cooldown_minutes):
                    # Create new alert
                    alert = Alert(
                        id=alert_key,
                        condition_name=condition.name,
                        severity=condition.severity,
                        message=f"{condition.metric} {condition.operator} {condition.threshold} (current: {metric_value:.2f})",
                        triggered_at=datetime.now(),
                        value=metric_value,
                        threshold=condition.threshold,
                        duration_minutes=condition.duration_minutes
                    )
                    self.alerts[alert_key] = alert
                    triggered_alerts.append(alert)

        return triggered_alerts

    def _get_metric_value(self, metrics: PerformanceMetrics, metric_path: str) -> Optional[float]:
        """Get metric value from metrics object using dot notation"""
        try:
            if '.' in metric_path:
                category, key = metric_path.split('.', 1)
                if category == 'system':
                    return metrics.system_metrics.get(key)
                elif category == 'application':
                    return metrics.application_metrics.get(key)
                elif category == 'business':
                    return metrics.business_metrics.get(key)
            else:
                # Search all categories
                for category_metrics in [metrics.system_metrics, metrics.application_metrics, metrics.business_metrics]:
                    if metric_path in category_metrics:
                        return category_metrics[metric_path]
            return None
        except:
            return None

    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return abs(value - threshold) < 0.01  # Small tolerance for floating point
        elif operator == '!=':
            return abs(value - threshold) >= 0.01
        return False

    def _check_cooldown(self, alert: Alert, cooldown_minutes: int) -> bool:
        """Check if alert is out of cooldown period"""
        if not alert.resolved_at:
            return False

        cooldown_end = alert.resolved_at + timedelta(minutes=cooldown_minutes)
        return datetime.now() >= cooldown_end

    def create_system_health_dashboard(self) -> Dict[str, Any]:
        """Create comprehensive system health dashboard data"""
        if not self.metrics_history:
            return {}

        # Get recent metrics (last 24 hours)
        recent_metrics = []
        cutoff_time = datetime.now() - timedelta(hours=24)

        for metric in self.metrics_history:
            if metric.timestamp >= cutoff_time:
                recent_metrics.append(metric)

        if not recent_metrics:
            return {}

        # System health overview
        latest = recent_metrics[-1]
        health_status = self.health_checker.check_all_components()

        dashboard_data = {
            'overview': {
                'timestamp': datetime.now().isoformat(),
                'system_status': health_status.get('status', 'unknown'),
                'overall_health_score': health_status.get('overall_score', 0.0),
                'total_metrics_collected': len(recent_metrics),
                'monitoring_duration_hours': 24
            },
            'current_metrics': {
                'cpu_percent': latest.system_metrics.get('cpu_percent', 0),
                'memory_percent': latest.system_metrics.get('memory_percent', 0),
                'disk_percent': latest.system_metrics.get('disk_percent', 0),
                'network_connections': latest.system_metrics.get('fxorcist_processes', 0)
            },
            'performance_trends': {},
            'alerts': {
                'active': len([a for a in self.alerts.values() if not a.resolved_at]),
                'total_today': len([a for a in self.alerts.values()
                                  if a.triggered_at.date() == datetime.now().date()])
            },
            'recommendations': [r.__dict__ for r in self.generate_optimization_recommendations()[:5]]
        }

        # Calculate trends for key metrics
        key_metrics = [
            'system.cpu_percent',
            'system.memory_percent',
            'system.disk_percent',
            'application.health_score'
        ]

        for metric in key_metrics:
            trend = self.analyze_trends(metric, 24)
            dashboard_data['performance_trends'][metric] = {
                'direction': trend.trend_direction,
                'strength': trend.trend_strength,
                'confidence': trend.confidence,
                'forecast_1h': trend.forecast_next_hour,
                'forecast_24h': trend.forecast_next_day
            }

        # Resource utilization summary
        cpu_values = [m.system_metrics.get('cpu_percent', 0) for m in recent_metrics]
        memory_values = [m.system_metrics.get('memory_percent', 0) for m in recent_metrics]

        dashboard_data['resource_summary'] = {
            'cpu_avg': statistics.mean(cpu_values) if cpu_values else 0,
            'cpu_peak': max(cpu_values) if cpu_values else 0,
            'memory_avg': statistics.mean(memory_values) if memory_values else 0,
            'memory_peak': max(memory_values) if memory_values else 0,
            'efficiency_score': self._calculate_efficiency_score(recent_metrics)
        }

        return dashboard_data

    def _calculate_efficiency_score(self, metrics: List[PerformanceMetrics]) -> float:
        """Calculate system efficiency score"""
        if not metrics:
            return 0.0

        # Efficiency based on resource usage vs performance
        cpu_usage = statistics.mean(m.system_metrics.get('cpu_percent', 0) for m in metrics)
        memory_usage = statistics.mean(m.system_metrics.get('memory_percent', 0) for m in metrics)
        health_score = statistics.mean(m.application_metrics.get('health_score', 1.0) for m in metrics)

        # Higher efficiency = lower resource usage + higher health score
        resource_efficiency = max(0, 100 - (cpu_usage + memory_usage) / 2)
        performance_efficiency = health_score * 100

        return (resource_efficiency + performance_efficiency) / 2

    def start_monitoring(self):
        """Start the monitoring system"""
        self.running = True

        # Start collection thread
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()

        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()

        # Start alert thread
        self.alert_thread = threading.Thread(target=self._alert_loop, daemon=True)
        self.alert_thread.start()

        logger.info("Advanced monitoring system started")

    def _collection_loop(self):
        """Metrics collection loop"""
        while self.running:
            try:
                self.collect_metrics()
                time.sleep(60)  # Collect every minute
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                time.sleep(30)

    def _analysis_loop(self):
        """Analysis and optimization loop"""
        while self.running:
            try:
                # Update baselines periodically
                self._update_baselines()

                # Analyze trends for key metrics
                key_metrics = [
                    'system.cpu_percent',
                    'system.memory_percent',
                    'application.health_score',
                    'business.error_rate_percent'
                ]

                for metric in key_metrics:
                    trend = self.analyze_trends(metric, 24)
                    # Store trend analysis results for dashboard

                time.sleep(300)  # Analyze every 5 minutes
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                time.sleep(60)

    def _alert_loop(self):
        """Alert monitoring loop"""
        while self.running:
            try:
                triggered_alerts = self.check_alerts()

                # Log triggered alerts
                for alert in triggered_alerts:
                    logger.warning(f"Alert triggered: {alert.condition_name} - {alert.message}")

                time.sleep(60)  # Check alerts every minute
            except Exception as e:
                logger.error(f"Error in alert loop: {e}")
                time.sleep(30)

    def _update_baselines(self):
        """Update performance baselines"""
        if len(self.metrics_history) < 100:
            return

        # Calculate baselines for key metrics over last 7 days
        baseline_metrics = [
            'system.cpu_percent',
            'system.memory_percent',
            'application.health_score'
        ]

        for metric in baseline_metrics:
            values = []
            for m in self.metrics_history:
                value = self._get_metric_value(m, metric)
                if value is not None:
                    values.append(value)

            if len(values) >= 50:  # Need sufficient data
                self.baselines[metric] = {
                    'mean': statistics.mean(values),
                    'std_dev': statistics.stdev(values),
                    'min': min(values),
                    'max': max(values),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99),
                    'last_updated': datetime.now().isoformat()
                }

    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.running = False

        # Wait for threads to finish
        threads = [self.collection_thread, self.analysis_thread, self.alert_thread]
        for thread in threads:
            if thread:
                thread.join(timeout=5)

        logger.info("Advanced monitoring system stopped")

# Global monitor instance
_monitor_instance = None

def get_monitor(instance_id: str = "system") -> AdvancedMonitor:
    """Get global monitor instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = AdvancedMonitor(instance_id)
    return _monitor_instance

def start_monitoring(instance_id: str = "system"):
    """Start monitoring for instance"""
    monitor = get_monitor(instance_id)
    monitor.start_monitoring()

def stop_monitoring():
    """Stop monitoring"""
    global _monitor_instance
    if _monitor_instance:
        _monitor_instance.stop_monitoring()

if __name__ == "__main__":
    # Example usage
    monitor = AdvancedMonitor("system")

    # Collect some metrics
    for i in range(10):
        metrics = monitor.collect_metrics()
        print(f"Collected metrics: CPU={metrics.system_metrics.get('cpu_percent', 0):.1f}%")

        # Add some mock application metrics
        metrics.application_metrics['health_score'] = 0.9
        metrics.application_metrics['dashboard_response_time'] = 1500 + (i * 10)

        time.sleep(2)

    # Analyze trends
    cpu_trend = monitor.analyze_trends('system.cpu_percent', 1)
    print(f"CPU Trend: {cpu_trend.trend_direction} (strength: {cpu_trend.trend_strength:.2f})")

    # Generate recommendations
    recommendations = monitor.generate_optimization_recommendations()
    print(f"Generated {len(recommendations)} optimization recommendations")

    # Create dashboard data
    dashboard = monitor.create_system_health_dashboard()
    print(f"Dashboard created with {len(dashboard)} sections")