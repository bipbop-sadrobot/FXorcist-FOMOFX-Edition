#!/usr/bin/env python3
"""
Advanced Auto-Scaling Engine for FXorcist

This module provides intelligent auto-scaling capabilities with:
- Machine learning-based predictive scaling
- Multi-dimensional resource analysis
- Advanced cooldown and hysteresis mechanisms
- Cost-aware scaling decisions
- Anomaly detection and adaptive thresholds
- Integration with cloud provider auto-scaling APIs

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
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from forex_ai_dashboard.utils.config_manager import ConfigurationManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScalingMetrics:
    """Comprehensive scaling metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_connections: int
    response_time_ms: float
    error_rate: float
    throughput: float
    active_users: int
    queue_length: int = 0
    custom_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class PredictiveModel:
    """Predictive scaling model"""
    model_type: str
    trained_at: datetime
    accuracy_score: float
    feature_importance: Dict[str, float]
    scaler: StandardScaler = None
    model: Any = None

@dataclass
class ScalingDecision:
    """Auto-scaling decision with reasoning"""
    timestamp: datetime
    action: str  # 'scale_up', 'scale_down', 'no_action'
    confidence: float
    reasoning: List[str]
    predicted_load: float
    current_load: float
    recommended_instances: int
    cost_impact: float
    cooldown_remaining: int

@dataclass
class AdaptiveThresholds:
    """Adaptive threshold configuration"""
    cpu_high: float = 70.0
    cpu_low: float = 30.0
    memory_high: float = 80.0
    memory_low: float = 40.0
    response_time_high: float = 2000.0
    response_time_low: float = 500.0
    error_rate_high: float = 5.0
    throughput_low: float = 10.0
    hysteresis_factor: float = 0.1
    adaptation_rate: float = 0.05

class AdvancedAutoScaler:
    """Advanced auto-scaling engine with ML-based predictions"""

    def __init__(self, instance_id: str, config_path: str = "config/auto_scaler.json"):
        self.instance_id = instance_id
        self.config_path = Path(config_path)
        self.config_manager = ConfigurationManager()

        # Metrics storage
        self.metrics_history: deque = deque(maxlen=1000)
        self.scaling_decisions: deque = deque(maxlen=100)

        # Scaling state
        self.current_instances = 1
        self.min_instances = 1
        self.max_instances = 10
        self.cooldown_period = 300  # 5 minutes
        self.last_scale_time = datetime.min
        self.scale_up_cooldown = 180  # 3 minutes
        self.scale_down_cooldown = 600  # 10 minutes

        # ML models
        self.predictive_models: Dict[str, PredictiveModel] = {}
        self.anomaly_detector: IsolationForest = None

        # Adaptive thresholds
        self.thresholds = AdaptiveThresholds()

        # Cost tracking
        self.cost_per_instance_hour = 0.10  # Default cost
        self.cost_history: deque = deque(maxlen=168)  # 1 week of hourly costs

        # Threading
        self.running = False
        self.monitoring_thread = None
        self.scaling_thread = None
        self.learning_thread = None

        # Load configuration
        self.load_config()

        # Initialize ML components
        self.initialize_ml_components()

    def load_config(self):
        """Load auto-scaler configuration"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)

                instance_config = config.get(self.instance_id, {})

                self.min_instances = instance_config.get('min_instances', 1)
                self.max_instances = instance_config.get('max_instances', 10)
                self.cooldown_period = instance_config.get('cooldown_period', 300)
                self.scale_up_cooldown = instance_config.get('scale_up_cooldown', 180)
                self.scale_down_cooldown = instance_config.get('scale_down_cooldown', 600)
                self.cost_per_instance_hour = instance_config.get('cost_per_instance_hour', 0.10)

                # Load thresholds
                if 'thresholds' in instance_config:
                    threshold_data = instance_config['thresholds']
                    self.thresholds = AdaptiveThresholds(**threshold_data)

                logger.info(f"Loaded configuration for instance {self.instance_id}")

            except Exception as e:
                logger.error(f"Error loading configuration: {e}")

    def save_config(self):
        """Save current configuration"""
        try:
            config = {}
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)

            config[self.instance_id] = {
                'min_instances': self.min_instances,
                'max_instances': self.max_instances,
                'cooldown_period': self.cooldown_period,
                'scale_up_cooldown': self.scale_up_cooldown,
                'scale_down_cooldown': self.scale_down_cooldown,
                'cost_per_instance_hour': self.cost_per_instance_hour,
                'thresholds': self.thresholds.__dict__,
                'last_updated': datetime.now().isoformat()
            }

            self.config_path.parent.mkdir(exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    def initialize_ml_components(self):
        """Initialize machine learning components"""
        try:
            # Initialize anomaly detector
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )

            # Load existing models if available
            self.load_predictive_models()

        except Exception as e:
            logger.error(f"Error initializing ML components: {e}")

    def load_predictive_models(self):
        """Load existing predictive models"""
        model_dir = Path("models/scaling")
        model_dir.mkdir(exist_ok=True)

        for model_file in model_dir.glob(f"{self.instance_id}_*.pkl"):
            try:
                model_name = model_file.stem.split('_', 1)[1]
                model_data = joblib.load(model_file)

                self.predictive_models[model_name] = PredictiveModel(
                    model_type=model_name,
                    trained_at=datetime.fromisoformat(model_data['trained_at']),
                    accuracy_score=model_data['accuracy_score'],
                    feature_importance=model_data['feature_importance'],
                    scaler=model_data['scaler'],
                    model=model_data['model']
                )

                logger.info(f"Loaded predictive model: {model_name}")

            except Exception as e:
                logger.error(f"Error loading model {model_file}: {e}")

    def add_metrics(self, metrics: ScalingMetrics):
        """Add new metrics data point"""
        with threading.Lock():
            self.metrics_history.append(metrics)

            # Keep only recent data (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            while self.metrics_history and self.metrics_history[0].timestamp < cutoff_time:
                self.metrics_history.popleft()

    def get_recent_metrics(self, hours: int = 1) -> List[ScalingMetrics]:
        """Get metrics from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]

    def calculate_current_load(self) -> float:
        """Calculate current system load based on multiple metrics"""
        if not self.metrics_history:
            return 0.0

        recent_metrics = self.get_recent_metrics(1)  # Last hour
        if not recent_metrics:
            return 0.0

        # Weighted load calculation
        weights = {
            'cpu': 0.25,
            'memory': 0.25,
            'response_time': 0.20,
            'error_rate': 0.15,
            'throughput': 0.10,
            'active_users': 0.05
        }

        load_components = []

        # CPU load (normalized 0-1)
        cpu_values = [m.cpu_percent / 100.0 for m in recent_metrics]
        load_components.append(weights['cpu'] * statistics.mean(cpu_values))

        # Memory load (normalized 0-1)
        memory_values = [m.memory_percent / 100.0 for m in recent_metrics]
        load_components.append(weights['memory'] * statistics.mean(memory_values))

        # Response time load (normalized 0-1, inverted)
        response_values = [min(m.response_time_ms / 5000.0, 1.0) for m in recent_metrics]
        load_components.append(weights['response_time'] * statistics.mean(response_values))

        # Error rate load (normalized 0-1)
        error_values = [min(m.error_rate / 10.0, 1.0) for m in recent_metrics]
        load_components.append(weights['error_rate'] * statistics.mean(error_values))

        # Throughput load (normalized 0-1, inverted for low throughput)
        throughput_values = [max(0, 1.0 - (m.throughput / 100.0)) for m in recent_metrics]
        load_components.append(weights['throughput'] * statistics.mean(throughput_values))

        # Active users load (normalized 0-1)
        user_values = [min(m.active_users / 1000.0, 1.0) for m in recent_metrics]
        load_components.append(weights['active_users'] * statistics.mean(user_values))

        return sum(load_components)

    def predict_future_load(self, hours_ahead: int = 1) -> Tuple[float, float]:
        """Predict future load using ML models"""
        if 'load_prediction' not in self.predictive_models or len(self.metrics_history) < 24:
            # Fallback to simple trend analysis
            return self.predict_load_trend(hours_ahead)

        model = self.predictive_models['load_prediction']

        try:
            # Prepare features from recent data
            recent_data = list(self.metrics_history)[-24:]  # Last 24 data points

            features = []
            for i, metric in enumerate(recent_data):
                feature_vector = [
                    metric.cpu_percent,
                    metric.memory_percent,
                    metric.response_time_ms,
                    metric.error_rate,
                    metric.throughput,
                    metric.active_users,
                    i  # Time index
                ]
                features.append(feature_vector)

            # Scale features
            features_scaled = model.scaler.transform(features)

            # Make prediction
            prediction = model.model.predict(features_scaled[-1].reshape(1, -1))[0]

            # Calculate confidence based on model accuracy
            confidence = model.accuracy_score

            return max(0.0, min(1.0, prediction)), confidence

        except Exception as e:
            logger.error(f"Error in load prediction: {e}")
            return self.predict_load_trend(hours_ahead)

    def predict_load_trend(self, hours_ahead: int = 1) -> Tuple[float, float]:
        """Simple trend-based load prediction"""
        if len(self.metrics_history) < 6:
            return self.calculate_current_load(), 0.5

        # Use linear regression on recent trend
        recent_metrics = list(self.metrics_history)[-6:]
        x = np.arange(len(recent_metrics)).reshape(-1, 1)
        y = np.array([self.calculate_load_from_metric(m) for m in recent_metrics])

        if len(np.unique(y)) < 2:
            return y[-1], 0.6

        model = LinearRegression()
        model.fit(x, y)

        # Predict next value
        next_x = np.array([[len(recent_metrics)]])
        prediction = model.predict(next_x)[0]

        # Calculate trend confidence
        r_squared = model.score(x, y)
        confidence = max(0.3, min(0.9, r_squared))

        return max(0.0, min(1.0, prediction)), confidence

    def calculate_load_from_metric(self, metric: ScalingMetrics) -> float:
        """Calculate load from a single metric"""
        cpu_load = metric.cpu_percent / 100.0
        memory_load = metric.memory_percent / 100.0
        response_load = min(metric.response_time_ms / 2000.0, 1.0)
        error_load = min(metric.error_rate / 5.0, 1.0)

        return (cpu_load + memory_load + response_load + error_load) / 4.0

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics using ML"""
        if not self.anomaly_detector or len(self.metrics_history) < 50:
            return []

        try:
            # Prepare data for anomaly detection
            data = []
            for metric in self.metrics_history:
                data.append([
                    metric.cpu_percent,
                    metric.memory_percent,
                    metric.response_time_ms,
                    metric.error_rate,
                    metric.throughput,
                    metric.active_users
                ])

            # Fit and predict anomalies
            self.anomaly_detector.fit(data)
            anomaly_scores = self.anomaly_detector.decision_function(data)
            predictions = self.anomaly_detector.predict(data)

            # Find recent anomalies
            anomalies = []
            recent_metrics = list(self.metrics_history)[-len(predictions):]

            for i, (pred, score, metric) in enumerate(zip(predictions, anomaly_scores, recent_metrics)):
                if pred == -1:  # Anomaly detected
                    anomalies.append({
                        'timestamp': metric.timestamp,
                        'severity': 'high' if score < -0.5 else 'medium',
                        'metrics': {
                            'cpu_percent': metric.cpu_percent,
                            'memory_percent': metric.memory_percent,
                            'response_time_ms': metric.response_time_ms,
                            'error_rate': metric.error_rate
                        },
                        'anomaly_score': score
                    })

            return anomalies[-10:]  # Return last 10 anomalies

        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return []

    def make_scaling_decision(self) -> ScalingDecision:
        """Make intelligent scaling decision"""
        current_time = datetime.now()
        current_load = self.calculate_current_load()
        predicted_load, prediction_confidence = self.predict_future_load(1)

        # Check cooldown periods
        time_since_last_scale = (current_time - self.last_scale_time).total_seconds()
        cooldown_remaining = max(0, self.cooldown_period - time_since_last_scale)

        if cooldown_remaining > 0:
            return ScalingDecision(
                timestamp=current_time,
                action='no_action',
                confidence=1.0,
                reasoning=['In cooldown period after recent scaling action'],
                predicted_load=predicted_load,
                current_load=current_load,
                recommended_instances=self.current_instances,
                cost_impact=0.0,
                cooldown_remaining=int(cooldown_remaining)
            )

        # Adaptive threshold adjustment
        self.adapt_thresholds()

        # Multi-metric evaluation
        scale_up_signals = []
        scale_down_signals = []

        # CPU-based decision
        if current_load > self.thresholds.cpu_high / 100.0:
            scale_up_signals.append(f"High CPU usage: {current_load * 100:.1f}% > {self.thresholds.cpu_high}%")
        elif current_load < self.thresholds.cpu_low / 100.0 and self.current_instances > self.min_instances:
            scale_down_signals.append(f"Low CPU usage: {current_load * 100:.1f}% < {self.thresholds.cpu_low}%")

        # Memory-based decision
        recent_metrics = self.get_recent_metrics(1)
        if recent_metrics:
            avg_memory = statistics.mean(m.memory_percent for m in recent_metrics)
            if avg_memory > self.thresholds.memory_high:
                scale_up_signals.append(f"High memory usage: {avg_memory:.1f}% > {self.thresholds.memory_high}%")
            elif avg_memory < self.thresholds.memory_low and self.current_instances > self.min_instances:
                scale_down_signals.append(f"Low memory usage: {avg_memory:.1f}% < {self.thresholds.memory_low}%")

        # Response time-based decision
        if recent_metrics:
            avg_response_time = statistics.mean(m.response_time_ms for m in recent_metrics)
            if avg_response_time > self.thresholds.response_time_high:
                scale_up_signals.append(f"Slow response time: {avg_response_time:.0f}ms > {self.thresholds.response_time_high}ms")
            elif avg_response_time < self.thresholds.response_time_low and self.current_instances > self.min_instances:
                scale_down_signals.append(f"Fast response time: {avg_response_time:.0f}ms < {self.thresholds.response_time_low}ms")

        # Predictive scaling
        if prediction_confidence > 0.7:
            if predicted_load > 0.8 and self.current_instances < self.max_instances:
                scale_up_signals.append(f"Predicted high load: {predicted_load:.2f} (confidence: {prediction_confidence:.2f})")
            elif predicted_load < 0.3 and self.current_instances > self.min_instances:
                scale_down_signals.append(f"Predicted low load: {predicted_load:.2f} (confidence: {prediction_confidence:.2f})")

        # Anomaly detection
        anomalies = self.detect_anomalies()
        recent_anomalies = [a for a in anomalies if (current_time - a['timestamp']).total_seconds() < 3600]
        if recent_anomalies:
            scale_up_signals.append(f"Detected {len(recent_anomalies)} recent anomalies")

        # Make final decision
        if scale_up_signals and self.current_instances < self.max_instances:
            new_instances = min(self.current_instances + 1, self.max_instances)
            cost_impact = (new_instances - self.current_instances) * self.cost_per_instance_hour

            return ScalingDecision(
                timestamp=current_time,
                action='scale_up',
                confidence=min(0.9, len(scale_up_signals) * 0.2),
                reasoning=scale_up_signals,
                predicted_load=predicted_load,
                current_load=current_load,
                recommended_instances=new_instances,
                cost_impact=cost_impact,
                cooldown_remaining=0
            )

        elif scale_down_signals and self.current_instances > self.min_instances:
            new_instances = max(self.current_instances - 1, self.min_instances)
            cost_impact = (new_instances - self.current_instances) * self.cost_per_instance_hour

            return ScalingDecision(
                timestamp=current_time,
                action='scale_down',
                confidence=min(0.8, len(scale_down_signals) * 0.15),
                reasoning=scale_down_signals,
                predicted_load=predicted_load,
                current_load=current_load,
                recommended_instances=new_instances,
                cost_impact=cost_impact,
                cooldown_remaining=0
            )

        else:
            return ScalingDecision(
                timestamp=current_time,
                action='no_action',
                confidence=1.0,
                reasoning=['All metrics within acceptable ranges'],
                predicted_load=predicted_load,
                current_load=current_load,
                recommended_instances=self.current_instances,
                cost_impact=0.0,
                cooldown_remaining=0
            )

    def adapt_thresholds(self):
        """Adapt thresholds based on historical performance"""
        if len(self.metrics_history) < 50:
            return

        recent_metrics = self.get_recent_metrics(24)  # Last 24 hours
        if not recent_metrics:
            return

        # Calculate optimal thresholds based on percentiles
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        response_values = [m.response_time_ms for m in recent_metrics]

        # Adaptive CPU thresholds
        cpu_p75 = np.percentile(cpu_values, 75)
        cpu_p25 = np.percentile(cpu_values, 25)
        self.thresholds.cpu_high = min(90, cpu_p75 + self.thresholds.hysteresis_factor * (cpu_p75 - cpu_p25))
        self.thresholds.cpu_low = max(10, cpu_p25 - self.thresholds.hysteresis_factor * (cpu_p75 - cpu_p25))

        # Adaptive memory thresholds
        memory_p75 = np.percentile(memory_values, 75)
        memory_p25 = np.percentile(memory_values, 25)
        self.thresholds.memory_high = min(95, memory_p75 + self.thresholds.hysteresis_factor * (memory_p75 - memory_p25))
        self.thresholds.memory_low = max(20, memory_p25 - self.thresholds.hysteresis_factor * (memory_p75 - memory_p25))

        # Adaptive response time thresholds
        response_p75 = np.percentile(response_values, 75)
        response_p25 = np.percentile(response_values, 25)
        self.thresholds.response_time_high = response_p75 + self.thresholds.hysteresis_factor * (response_p75 - response_p25)
        self.thresholds.response_time_low = max(100, response_p25 - self.thresholds.hysteresis_factor * (response_p75 - response_p25))

    def execute_scaling_action(self, decision: ScalingDecision) -> bool:
        """Execute the scaling decision"""
        if decision.action == 'no_action':
            return True

        try:
            if decision.action == 'scale_up':
                success = self.scale_up(decision.recommended_instances - self.current_instances)
            elif decision.action == 'scale_down':
                success = self.scale_down(self.current_instances - decision.recommended_instances)
            else:
                return False

            if success:
                self.current_instances = decision.recommended_instances
                self.last_scale_time = decision.timestamp
                self.scaling_decisions.append(decision)
                self.save_config()

                logger.info(f"Successfully executed scaling action: {decision.action} to {self.current_instances} instances")
                return True
            else:
                logger.error(f"Failed to execute scaling action: {decision.action}")
                return False

        except Exception as e:
            logger.error(f"Error executing scaling action: {e}")
            return False

    def scale_up(self, instances: int) -> bool:
        """Scale up by adding instances"""
        logger.info(f"Scaling up by {instances} instances")
        # Implementation would depend on the deployment environment
        # For now, just log the scaling action
        return True

    def scale_down(self, instances: int) -> bool:
        """Scale down by removing instances"""
        logger.info(f"Scaling down by {instances} instances")
        # Implementation would depend on the deployment environment
        # For now, just log the scaling action
        return True

    def train_predictive_models(self):
        """Train predictive models using historical data"""
        if len(self.metrics_history) < 100:
            logger.info("Not enough data for training predictive models")
            return

        try:
            # Prepare training data
            data = []
            targets = []

            for i in range(24, len(self.metrics_history)):  # Use 24-hour windows
                window_data = list(self.metrics_history)[i-24:i+1]

                # Features from the 24-hour window
                features = []
                for metric in window_data[:-1]:  # All but the last point
                    features.extend([
                        metric.cpu_percent,
                        metric.memory_percent,
                        metric.response_time_ms,
                        metric.error_rate,
                        metric.throughput,
                        metric.active_users
                    ])

                # Target is the load 1 hour ahead
                target_metric = window_data[-1]
                target_load = self.calculate_load_from_metric(target_metric)

                data.append(features)
                targets.append(target_load)

            # Train load prediction model
            X = np.array(data)
            y = np.array(targets)

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train model
            model = LinearRegression()
            model.fit(X_scaled, y)

            # Calculate accuracy
            predictions = model.predict(X_scaled)
            mse = np.mean((predictions - y) ** 2)
            r_squared = model.score(X_scaled, y)

            # Feature importance (using coefficients)
            feature_importance = {}
            feature_names = []
            for hour in range(24):
                for metric in ['cpu', 'memory', 'response_time', 'error_rate', 'throughput', 'active_users']:
                    feature_names.append(f"{metric}_h{hour}")

            for name, coef in zip(feature_names, model.coef_):
                feature_importance[name] = abs(coef)

            # Save model
            model_data = {
                'model': model,
                'scaler': scaler,
                'trained_at': datetime.now().isoformat(),
                'accuracy_score': r_squared,
                'feature_importance': feature_importance
            }

            model_dir = Path("models/scaling")
            model_dir.mkdir(exist_ok=True)
            model_file = model_dir / f"{self.instance_id}_load_prediction.pkl"
            joblib.dump(model_data, model_file)

            # Update in-memory model
            self.predictive_models['load_prediction'] = PredictiveModel(
                model_type='load_prediction',
                trained_at=datetime.now(),
                accuracy_score=r_squared,
                feature_importance=feature_importance,
                scaler=scaler,
                model=model
            )

            logger.info(f"Trained predictive model with R² = {r_squared:.3f}")

        except Exception as e:
            logger.error(f"Error training predictive models: {e}")

    def get_scaling_analytics(self) -> Dict[str, Any]:
        """Get comprehensive scaling analytics"""
        if not self.metrics_history:
            return {}

        recent_metrics = self.get_recent_metrics(24)

        analytics = {
            'current_instances': self.current_instances,
            'min_instances': self.min_instances,
            'max_instances': self.max_instances,
            'scaling_decisions': list(self.scaling_decisions),
            'thresholds': self.thresholds.__dict__,
            'anomalies': self.detect_anomalies(),
            'cost_analysis': self.analyze_costs(),
            'performance_metrics': {
                'avg_cpu': statistics.mean(m.cpu_percent for m in recent_metrics) if recent_metrics else 0,
                'avg_memory': statistics.mean(m.memory_percent for m in recent_metrics) if recent_metrics else 0,
                'avg_response_time': statistics.mean(m.response_time_ms for m in recent_metrics) if recent_metrics else 0,
                'avg_error_rate': statistics.mean(m.error_rate for m in recent_metrics) if recent_metrics else 0,
                'avg_throughput': statistics.mean(m.throughput for m in recent_metrics) if recent_metrics else 0
            },
            'predictive_accuracy': {
                model_name: model.accuracy_score
                for model_name, model in self.predictive_models.items()
            }
        }

        return analytics

    def analyze_costs(self) -> Dict[str, Any]:
        """Analyze scaling costs"""
        if not self.scaling_decisions:
            return {}

        # Calculate costs based on scaling decisions
        total_cost = 0
        scale_up_events = 0
        scale_down_events = 0

        for decision in self.scaling_decisions:
            if decision.action in ['scale_up', 'scale_down']:
                total_cost += abs(decision.cost_impact)
                if decision.action == 'scale_up':
                    scale_up_events += 1
                else:
                    scale_down_events += 1

        return {
            'total_cost': total_cost,
            'scale_up_events': scale_up_events,
            'scale_down_events': scale_down_events,
            'avg_cost_per_scaling_event': total_cost / max(1, len(self.scaling_decisions)),
            'cost_per_instance_hour': self.cost_per_instance_hour,
            'estimated_monthly_cost': self.current_instances * self.cost_per_instance_hour * 24 * 30
        }

    def start_auto_scaling(self):
        """Start the auto-scaling service"""
        self.running = True

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        # Start scaling thread
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()

        # Start learning thread
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()

        logger.info("Advanced auto-scaling service started")

    def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while self.running:
            try:
                # This would be called with real metrics from the system
                # For now, simulate metrics collection
                time.sleep(60)  # Collect metrics every minute
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)

    def _scaling_loop(self):
        """Auto-scaling decision loop"""
        while self.running:
            try:
                decision = self.make_scaling_decision()
                if decision.action != 'no_action':
                    self.execute_scaling_action(decision)
                time.sleep(300)  # Make scaling decisions every 5 minutes
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                time.sleep(60)

    def _learning_loop(self):
        """Continuous learning loop"""
        while self.running:
            try:
                # Retrain models periodically
                if len(self.metrics_history) >= 100:
                    self.train_predictive_models()
                time.sleep(3600)  # Retrain every hour
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                time.sleep(300)

    def stop_auto_scaling(self):
        """Stop the auto-scaling service"""
        self.running = False

        # Wait for threads to finish
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5)
        if self.learning_thread:
            self.learning_thread.join(timeout=5)

        logger.info("Advanced auto-scaling service stopped")

# Global auto-scaler registry
_auto_scalers: Dict[str, AdvancedAutoScaler] = {}

def get_auto_scaler(instance_id: str) -> AdvancedAutoScaler:
    """Get or create auto-scaler for instance"""
    if instance_id not in _auto_scalers:
        _auto_scalers[instance_id] = AdvancedAutoScaler(instance_id)
    return _auto_scalers[instance_id]

def start_all_auto_scalers():
    """Start auto-scaling for all registered instances"""
    for scaler in _auto_scalers.values():
        scaler.start_auto_scaling()

def stop_all_auto_scalers():
    """Stop auto-scaling for all instances"""
    for scaler in _auto_scalers.values():
        scaler.stop_auto_scaling()

if __name__ == "__main__":
    # Example usage
    scaler = get_auto_scaler("dashboard_main")

    # Add some sample metrics
    for i in range(100):
        metrics = ScalingMetrics(
            timestamp=datetime.now() - timedelta(minutes=100-i),
            cpu_percent=50 + 20 * np.sin(i * 0.1),
            memory_percent=60 + 15 * np.cos(i * 0.1),
            disk_percent=30,
            network_connections=10,
            response_time_ms=800 + 200 * np.sin(i * 0.05),
            error_rate=1.0,
            throughput=50,
            active_users=25
        )
        scaler.add_metrics(metrics)

    # Train predictive models
    scaler.train_predictive_models()

    # Make scaling decision
    decision = scaler.make_scaling_decision()
    print(f"Scaling decision: {decision.action}")
    print(f"Reasoning: {decision.reasoning}")
    print(f"Recommended instances: {decision.recommended_instances}")

    # Get analytics
    analytics = scaler.get_scaling_analytics()
    print(f"Current instances: {analytics['current_instances']}")
    print(f"Average CPU: {analytics['performance_metrics']['avg_cpu']:.1f}%")