# FXorcist Examples Guide

## Overview

This guide provides practical examples and code snippets for common use cases with the FXorcist AI Dashboard system. Each example includes setup instructions, code samples, and expected outputs.

## Table of Contents

- [Quick Start Examples](#quick-start-examples)
- [Data Processing Examples](#data-processing-examples)
- [Training Pipeline Examples](#training-pipeline-examples)
- [Dashboard Customization Examples](#dashboard-customization-examples)
- [API Integration Examples](#api-integration-examples)
- [Advanced Configuration Examples](#advanced-configuration-examples)
- [Troubleshooting Examples](#troubleshooting-examples)

## Quick Start Examples

### Basic System Setup

```bash
# 1. Clone and setup
git clone https://github.com/your-org/FXorcist-FOMOFX-Edition.git
cd FXorcist-FOMOFX-Edition

# 2. Create virtual environment
python3 -m venv fxorcist_env
source fxorcist_env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run setup
python setup.py

# 5. Start dashboard
python fxorcist_cli.py --dashboard
```

### Interactive CLI Session

```bash
# Start interactive mode
python fxorcist_cli.py --interactive

# In interactive mode:
# 1. Select "Data Integration"
# 2. Choose "Process All Data"
# 3. Select "Training Pipeline"
# 4. Choose "Quick Training"
# 5. Select "Dashboard"
# 6. Choose "Start Main Dashboard"
```

### One-Command Workflow

```bash
# Complete workflow in one command
python fxorcist_cli.py --command "workflow complete"
```

## Data Processing Examples

### Basic Data Integration

```python
from fxorcist_cli import FXorcistCLI

# Initialize CLI
cli = FXorcistCLI()

# Process all available data
results = cli.run_data_integration()
print(f"Processed: {results['processed']} files")
print(f"Skipped: {results['skipped']} files")
print(f"Total: {results['total']} files")
```

### Advanced Data Processing with Quality Assessment

```python
from forex_ai_dashboard.pipeline.optimized_data_integrator import OptimizedDataIntegrator
from forex_ai_dashboard.pipeline.data_quality_validator import DataQualityValidator

# Initialize components
integrator = OptimizedDataIntegrator()
validator = DataQualityValidator()

# Find and process high-quality M1 files
m1_files = integrator.find_optimized_m1_files()
print(f"Found {len(m1_files)} M1 files")

for file_path, metadata in m1_files:
    # Validate data quality
    df = integrator.detect_and_parse(file_path)
    if df is not None:
        quality_score = validator.validate_dataset(df, "EURUSD")
        if quality_score['overall_score'] > 0.8:
            print(f"High quality data: {file_path}")
            print(f"Quality score: {quality_score['overall_score']:.2f}")
```

### Custom Data Format Detection

```python
from forex_ai_dashboard.pipeline.data_ingestion import ForexDataFormatDetector
import pandas as pd

# Initialize detector
detector = ForexDataFormatDetector()

# Process multiple files
data_files = [
    "data/EURUSD_2023.csv",
    "data/GBPUSD_M1_2023.zip",
    "data/USDJPY_historical.json"
]

processed_data = {}
for file_path in data_files:
    df = detector.detect_and_parse(file_path)
    if df is not None:
        processed_data[file_path] = df
        print(f"Processed {file_path}: {len(df)} rows")
    else:
        print(f"Failed to process {file_path}")
```

### Data Quality Analysis

```python
from forex_ai_dashboard.pipeline.data_quality_validator import DataQualityValidator
import matplotlib.pyplot as plt

# Initialize validator
validator = DataQualityValidator()

# Analyze data quality
df = pd.read_csv("data/EURUSD_M1_2023.csv")
quality_report = validator.validate_dataset(df, "EURUSD")

# Print quality metrics
print("Data Quality Report:")
print(f"Completeness: {quality_report['completeness']:.2%}")
print(f"Accuracy: {quality_report['accuracy']:.2%}")
print(f"Consistency: {quality_report['consistency']:.2%}")
print(f"Timeliness: {quality_report['timeliness']:.2%}")
print(f"Overall Score: {quality_report['overall_score']:.2%}")

# Visualize quality issues
validator.plot_quality_issues(df, "EURUSD")
plt.show()
```

## Training Pipeline Examples

### Basic Model Training

```python
from forex_ai_dashboard.pipeline.unified_training_pipeline import UnifiedTrainingPipeline

# Initialize pipeline
pipeline = UnifiedTrainingPipeline()

# Run complete training
results = pipeline.run_complete_training_pipeline()
print("Training Results:")
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  Train Score: {metrics['train_score']:.4f}")
    print(f"  Test Score: {metrics['test_score']:.4f}")
    print(f"  MAE: {metrics['mae']:.6f}")
```

### Advanced Training with Hyperparameter Optimization

```python
from forex_ai_dashboard.pipeline.enhanced_training_pipeline import EnhancedTrainingPipeline
import pandas as pd

# Load data
data = pd.read_csv("data/processed/EURUSD_features.csv")
X = data.drop('target', axis=1)
y = data['target']

# Initialize enhanced pipeline
pipeline = EnhancedTrainingPipeline({
    'hyperparameter_optimization': True,
    'n_trials': 50,
    'cross_validation_folds': 5
})

# Train with optimization
model, metrics = pipeline.train_with_optimization(X, y)
print(f"Best Model: {type(model).__name__}")
print(f"Best Score: {metrics['best_score']:.4f}")
print(f"Best Parameters: {metrics['best_params']}")
```

### Ensemble Model Training

```python
from forex_ai_dashboard.pipeline.unified_training_pipeline import UnifiedTrainingPipeline
from sklearn.ensemble import VotingClassifier
import pandas as pd

# Load data
data = pd.read_csv("data/processed/EURUSD_features.csv")
X = data.drop('target', axis=1)
y = data['target']

# Initialize pipeline
pipeline = UnifiedTrainingPipeline({
    'ensemble_methods': True,
    'models': ['catboost', 'lightgbm', 'xgboost']
})

# Train ensemble
ensemble_results = pipeline.run_complete_training_pipeline()
print("Ensemble Results:")
print(f"Ensemble Score: {ensemble_results['ensemble']['test_score']:.4f}")
print(f"Individual Scores: {ensemble_results['individual_scores']}")
```

### Custom Feature Engineering

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from forex_ai_dashboard.pipeline.feature_engineering import FeatureEngineer

# Load raw data
df = pd.read_csv("data/EURUSD_M1_2023.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Initialize feature engineer
feature_engineer = FeatureEngineer()

# Add technical indicators
df = feature_engineer.add_technical_indicators(df, 'close')

# Add time-based features
df = feature_engineer.add_time_features(df, 'timestamp')

# Add statistical features
df = feature_engineer.add_statistical_features(df, 'close', window=20)

# Scale features
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print(f"Original features: {len(df.columns)}")
print(f"Engineered features: {len([col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close']])}")
```

## Dashboard Customization Examples

### Custom Dashboard Layout

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from forex_ai_dashboard.utils.health_checker import HealthChecker

# Initialize health checker
health_checker = HealthChecker()

def create_custom_dashboard():
    st.title("FXorcist Custom Dashboard")

    # Sidebar with controls
    with st.sidebar:
        st.header("Controls")
        symbol = st.selectbox("Symbol", ["EURUSD", "GBPUSD", "USDJPY"])
        timeframe = st.selectbox("Timeframe", ["M1", "M5", "H1", "D1"])
        model_type = st.selectbox("Model", ["catboost", "lightgbm", "xgboost"])

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Price Chart")
        # Add your price chart here
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close']
        ))
        st.plotly_chart(fig)

    with col2:
        st.subheader("System Health")
        health_status = health_checker.check_all_components()
        for component, status in health_status.items():
            if status['status'] == 'healthy':
                st.success(f"{component}: {status['status']}")
            else:
                st.error(f"{component}: {status['status']}")

        st.subheader("Model Performance")
        # Add model performance metrics here

if __name__ == "__main__":
    create_custom_dashboard()
```

### Real-time Data Monitoring

```python
import streamlit as st
import time
from forex_ai_dashboard.utils.data_monitor import DataMonitor

# Initialize monitor
monitor = DataMonitor()

def real_time_dashboard():
    st.title("Real-time FXorcist Monitor")

    # Create placeholders
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()

    while True:
        # Update system status
        with status_placeholder.container():
            st.subheader("System Status")
            status = monitor.get_system_status()
            col1, col2, col3 = st.columns(3)
            col1.metric("CPU Usage", f"{status['cpu_percent']:.1f}%")
            col2.metric("Memory Usage", f"{status['memory_percent']:.1f}%")
            col3.metric("Disk Usage", f"{status['disk_percent']:.1f}%")

        # Update metrics
        with metrics_placeholder.container():
            st.subheader("Data Pipeline Metrics")
            metrics = monitor.get_pipeline_metrics()
            st.json(metrics)

        # Update chart
        with chart_placeholder.container():
            st.subheader("Live Data Flow")
            chart_data = monitor.get_live_data()
            st.line_chart(chart_data)

        # Wait before next update
        time.sleep(5)

if __name__ == "__main__":
    real_time_dashboard()
```

### Custom Alert System

```python
import streamlit as st
from forex_ai_dashboard.utils.alert_system import AlertSystem
import smtplib
from email.mime.text import MIMEText

class EmailAlertSystem(AlertSystem):
    def __init__(self, smtp_server, smtp_port, username, password):
        super().__init__()
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password

    def send_alert(self, alert_type, message, severity):
        # Create email message
        msg = MIMEText(message)
        msg['Subject'] = f"FXorcist Alert: {alert_type} ({severity})"
        msg['From'] = self.username
        msg['To'] = self.username  # Send to self for demo

        # Send email
        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.sendmail(self.username, self.username, msg.as_string())
            server.quit()
            return True
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False

def alert_dashboard():
    st.title("FXorcist Alert System")

    # Initialize alert system
    alert_system = EmailAlertSystem(
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        username="your-email@gmail.com",
        password="your-app-password"
    )

    # Alert configuration
    st.subheader("Alert Configuration")
    col1, col2 = st.columns(2)

    with col1:
        cpu_threshold = st.slider("CPU Threshold (%)", 0, 100, 80)
        memory_threshold = st.slider("Memory Threshold (%)", 0, 100, 85)

    with col2:
        disk_threshold = st.slider("Disk Threshold (%)", 0, 100, 90)
        error_threshold = st.slider("Error Rate Threshold (%)", 0, 100, 5)

    # Manual alert test
    if st.button("Test Alert"):
        alert_system.send_alert(
            "TEST",
            "This is a test alert from FXorcist dashboard",
            "INFO"
        )
        st.success("Test alert sent!")

    # Alert history
    st.subheader("Recent Alerts")
    alerts = alert_system.get_recent_alerts()
    if alerts:
        for alert in alerts[-10:]:  # Show last 10 alerts
            if alert['severity'] == 'CRITICAL':
                st.error(f"{alert['timestamp']}: {alert['message']}")
            elif alert['severity'] == 'WARNING':
                st.warning(f"{alert['timestamp']}: {alert['message']}")
            else:
                st.info(f"{alert['timestamp']}: {alert['message']}")
    else:
        st.info("No recent alerts")

if __name__ == "__main__":
    alert_dashboard()
```

## API Integration Examples

### REST API Client

```python
import requests
import json
from typing import Dict, Any

class FXorcistAPIClient:
    def __init__(self, base_url: str = "http://localhost:8501"):
        self.base_url = base_url
        self.session = requests.Session()

    def health_check(self) -> Dict[str, Any]:
        """Check system health"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()

    def get_data_status(self) -> Dict[str, Any]:
        """Get data processing status"""
        response = self.session.get(f"{self.base_url}/api/data/status")
        return response.json()

    def start_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start model training"""
        response = self.session.post(
            f"{self.base_url}/api/training/start",
            json=config
        )
        return response.json()

    def get_training_status(self, training_id: str) -> Dict[str, Any]:
        """Get training status"""
        response = self.session.get(f"{self.base_url}/api/training/{training_id}/status")
        return response.json()

    def get_predictions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get model predictions"""
        response = self.session.post(
            f"{self.base_url}/api/predict",
            json=data
        )
        return response.json()

# Usage example
client = FXorcistAPIClient()

# Check health
health = client.health_check()
print(f"System health: {health['status']}")

# Start training
training_config = {
    "model_type": "catboost",
    "data_source": "EURUSD_M1_2023",
    "hyperparameter_optimization": True
}
training_response = client.start_training(training_config)
training_id = training_response['training_id']

# Monitor training
import time
while True:
    status = client.get_training_status(training_id)
    print(f"Training status: {status['status']}")
    if status['status'] == 'completed':
        break
    time.sleep(10)

# Get predictions
test_data = {
    "features": [1.234, 0.567, 0.890, 0.123]
}
predictions = client.get_predictions(test_data)
print(f"Predictions: {predictions}")
```

### Webhook Integration

```python
from flask import Flask, request, jsonify
from forex_ai_dashboard.utils.webhook_handler import WebhookHandler
import logging

app = Flask(__name__)
webhook_handler = WebhookHandler()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/webhook/fxorcist', methods=['POST'])
def fxorcist_webhook():
    """Handle FXorcist webhooks"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Process webhook based on type
        webhook_type = data.get('type')

        if webhook_type == 'training_completed':
            result = webhook_handler.handle_training_completion(data)
        elif webhook_type == 'prediction_request':
            result = webhook_handler.handle_prediction_request(data)
        elif webhook_type == 'alert_triggered':
            result = webhook_handler.handle_alert(data)
        else:
            return jsonify({"error": f"Unknown webhook type: {webhook_type}"}), 400

        logger.info(f"Processed webhook: {webhook_type}")
        return jsonify({"status": "processed", "result": result}), 200

    except Exception as e:
        logger.error(f"Webhook processing error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/webhook/status', methods=['GET'])
def webhook_status():
    """Get webhook processing status"""
    status = webhook_handler.get_status()
    return jsonify(status)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
```

### Database Integration

```python
import sqlite3
from datetime import datetime
from typing import List, Dict, Any
from forex_ai_dashboard.utils.database_connector import DatabaseConnector

class FXorcistDatabase:
    def __init__(self, db_path: str = "fxorcist.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY,
                    model_type TEXT NOT NULL,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    status TEXT,
                    accuracy REAL,
                    config TEXT
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    symbol TEXT,
                    prediction REAL,
                    confidence REAL,
                    actual REAL
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    cpu_percent REAL,
                    memory_percent REAL,
                    disk_percent REAL,
                    active_processes INTEGER
                )
            ''')

            conn.commit()

    def save_training_run(self, model_type: str, accuracy: float, config: Dict[str, Any]) -> int:
        """Save training run results"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO training_runs (model_type, start_time, end_time, status, accuracy, config)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                model_type,
                datetime.now(),
                datetime.now(),
                'completed',
                accuracy,
                json.dumps(config)
            ))
            return cursor.lastrowid

    def get_training_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get training history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM training_runs
                ORDER BY start_time DESC
                LIMIT ?
            ''', (limit,))

            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def save_prediction(self, symbol: str, prediction: float, confidence: float, actual: float = None) -> int:
        """Save prediction result"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (timestamp, symbol, prediction, confidence, actual)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                symbol,
                prediction,
                confidence,
                actual
            ))
            return cursor.lastrowid

    def save_system_metrics(self, cpu_percent: float, memory_percent: float, disk_percent: float, active_processes: int):
        """Save system metrics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO system_metrics (timestamp, cpu_percent, memory_percent, disk_percent, active_processes)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                cpu_percent,
                memory_percent,
                disk_percent,
                active_processes
            ))

# Usage example
db = FXorcistDatabase()

# Save training results
training_id = db.save_training_run(
    model_type="catboost",
    accuracy=0.95,
    config={"iterations": 1000, "learning_rate": 0.1}
)

# Get training history
history = db.get_training_history()
for run in history:
    print(f"Model: {run['model_type']}, Accuracy: {run['accuracy']:.4f}")

# Save predictions
prediction_id = db.save_prediction(
    symbol="EURUSD",
    prediction=1.0850,
    confidence=0.85,
    actual=1.0820
)
```

## Advanced Configuration Examples

### Multi-Environment Configuration

```python
import os
from pathlib import Path
from forex_ai_dashboard.utils.config_manager import ConfigManager

class MultiEnvironmentConfig:
    def __init__(self):
        self.env = os.getenv('FXORCIST_ENV', 'development')
        self.config_manager = ConfigManager()
        self.load_environment_config()

    def load_environment_config(self):
        """Load configuration based on environment"""
        config_dir = Path("config")
        base_config = self.load_config_file(config_dir / "base.json")

        if self.env == 'development':
            env_config = self.load_config_file(config_dir / "development.json")
        elif self.env == 'staging':
            env_config = self.load_config_file(config_dir / "staging.json")
        elif self.env == 'production':
            env_config = self.load_config_file(config_dir / "production.json")
        else:
            raise ValueError(f"Unknown environment: {self.env}")

        # Merge configurations
        self.config = self.merge_configs(base_config, env_config)
        self.apply_environment_variables()

    def load_config_file(self, config_path: Path) -> dict:
        """Load configuration from JSON file"""
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}

    def merge_configs(self, base: dict, env: dict) -> dict:
        """Merge base and environment configurations"""
        merged = base.copy()
        for key, value in env.items():
            if isinstance(value, dict) and key in merged:
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged

    def apply_environment_variables(self):
        """Apply environment variable overrides"""
        env_vars = {
            'FXORCIST_LOG_LEVEL': 'log_level',
            'FXORCIST_DATA_DIR': 'data_dir',
            'FXORCIST_MODEL_DIR': 'model_dir',
            'FXORCIST_DB_HOST': 'database.host',
            'FXORCIST_DB_PORT': 'database.port'
        }

        for env_var, config_key in env_vars.items():
            value = os.getenv(env_var)
            if value:
                self.set_nested_config(self.config, config_key.split('.'), value)

    def set_nested_config(self, config: dict, keys: list, value):
        """Set nested configuration value"""
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        config[keys[-1]] = value

    def get(self, key: str, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

# Usage example
config = MultiEnvironmentConfig()

# Access configuration
data_dir = config.get('data_dir', './data')
log_level = config.get('log_level', 'INFO')
db_host = config.get('database.host', 'localhost')

print(f"Environment: {config.env}")
print(f"Data directory: {data_dir}")
print(f"Log level: {log_level}")
print(f"Database host: {db_host}")
```

### Custom Logging Configuration

```python
import logging
import logging.handlers
from pathlib import Path
import json
from typing import Dict, Any

class AdvancedLogger:
    def __init__(self, config_path: str = "config/logging.json"):
        self.config_path = Path(config_path)
        self.setup_logging()

    def setup_logging(self):
        """Setup advanced logging configuration"""
        # Load logging configuration
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        else:
            config = self.get_default_config()

        # Configure logging
        logging.config.dictConfig(config)

        # Add custom handlers
        self.add_performance_handler()
        self.add_error_handler()

    def get_default_config(self) -> Dict[str, Any]:
        """Get default logging configuration"""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
                },
                "simple": {
                    "format": "%(levelname)s - %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "simple"
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "filename": "logs/fxorcist.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5
                }
            },
            "root": {
                "level": "DEBUG",
                "handlers": ["console", "file"]
            }
        }

    def add_performance_handler(self):
        """Add performance monitoring handler"""
        logger = logging.getLogger('performance')
        handler = logging.handlers.TimedRotatingFileHandler(
            'logs/performance.log',
            when='midnight',
            interval=1,
            backupCount=30
        )
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(message)s'
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def add_error_handler(self):
        """Add error notification handler"""
        logger = logging.getLogger('errors')
        handler = logging.handlers.SMTPHandler(
            mailhost=('smtp.gmail.com', 587),
            fromaddr='fxorcist@yourdomain.com',
            toaddrs=['admin@yourdomain.com'],
            subject='FXorcist Error Alert',
            credentials=('username', 'password'),
            secure=()
        )
        handler.setLevel(logging.ERROR)
        logger.addHandler(handler)

    def log_performance(self, operation: str, duration: float, metadata: Dict[str, Any] = None):
        """Log performance metrics"""
        logger = logging.getLogger('performance')
        message = f"{operation} completed in {duration:.3f}s"
        if metadata:
            message += f" - {metadata}"
        logger.info(message)

    def log_error_with_context(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with additional context"""
        logger = logging.getLogger('errors')
        error_msg = f"{type(error).__name__}: {str(error)}"
        if context:
            error_msg += f" - Context: {context}"
        logger.error(error_msg, exc_info=True)

# Usage example
logger = AdvancedLogger()

# Regular logging
logging.info("FXorcist system started")
logging.debug("Loading configuration...")

# Performance logging
import time
start_time = time.time()
# ... some operation ...
duration = time.time() - start_time
logger.log_performance("data_processing", duration, {"records": 10000})

# Error logging
try:
    # ... some operation that might fail ...
    pass
except Exception as e:
    logger.log_error_with_context(e, {"operation": "data_processing", "file": "EURUSD.csv"})
```

## Troubleshooting Examples

### Debug Data Processing Issues

```python
import pandas as pd
from forex_ai_dashboard.pipeline.data_ingestion import ForexDataFormatDetector
from forex_ai_dashboard.pipeline.data_quality_validator import DataQualityValidator
import logging

logging.basicConfig(level=logging.DEBUG)

def debug_data_processing(file_path: str):
    """Debug data processing issues"""
    print(f"Debugging file: {file_path}")

    # Step 1: Check file existence and basic info
    try:
        file_size = Path(file_path).stat().st_size
        print(f"File size: {file_size} bytes")
    except FileNotFoundError:
        print("ERROR: File not found")
        return

    # Step 2: Detect format
    detector = ForexDataFormatDetector()
    try:
        format_type, confidence = detector._detect_format(file_path)
        print(f"Detected format: {format_type} (confidence: {confidence:.2f})")
    except Exception as e:
        print(f"ERROR in format detection: {e}")
        return

    # Step 3: Parse data
    try:
        df = detector.detect_and_parse(file_path)
        if df is None:
            print("ERROR: Failed to parse data")
            return
        print(f"Parsed data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
    except Exception as e:
        print(f"ERROR in data parsing: {e}")
        return

    # Step 4: Validate data quality
    validator = DataQualityValidator()
    try:
        quality_report = validator.validate_dataset(df, "EURUSD")
        print("Quality Report:")
        for key, value in quality_report.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"ERROR in quality validation: {e}")
        return

    # Step 5: Sample data inspection
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())
    print("\nSummary statistics:")
    print(df.describe())

# Usage
debug_data_processing("data/EURUSD_M1_2023.csv")
```

### Debug Training Issues

```python
from forex_ai_dashboard.pipeline.unified_training_pipeline import UnifiedTrainingPipeline
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)

def debug_training_pipeline(data_path: str):
    """Debug training pipeline issues"""
    print("Debugging training pipeline...")

    # Step 1: Load data
    try:
        data = pd.read_csv(data_path)
        print(f"Data shape: {data.shape}")
        print(f"Target distribution:\n{data['target'].value_counts()}")
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return

    # Step 2: Check data quality
    print("\nChecking data quality...")
    null_counts = data.isnull().sum()
    if null_counts.sum() > 0:
        print(f"Null values found:\n{null_counts[null_counts > 0]}")
    else:
        print("No null values found")

    # Step 3: Check feature correlations
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    correlations = data[numeric_cols].corr()['target'].abs().sort_values(ascending=False)
    print(f"\nTop feature correlations with target:\n{correlations.head(10)}")

    # Step 4: Initialize pipeline
    try:
        pipeline = UnifiedTrainingPipeline({
            'debug': True,
            'log_level': 'DEBUG'
        })
        print("Pipeline initialized successfully")
    except Exception as e:
        print(f"ERROR initializing pipeline: {e}")
        return

    # Step 5: Run training with debugging
    try:
        print("\nStarting training...")
        results = pipeline.run_complete_training_pipeline(data_source=data_path)
        print("Training completed successfully")
        print(f"Results: {results}")
    except Exception as e:
        print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return

# Usage
debug_training_pipeline("data/processed/EURUSD_features.csv")
```

### System Health Monitoring

```python
import psutil
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

def monitor_system_health(duration_minutes: int = 5):
    """Monitor system health for specified duration"""
    print(f"Monitoring system health for {duration_minutes} minutes...")

    metrics = []
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)

    while time.time() < end_time:
        # Collect system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Collect process information
        fxorcist_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                if 'fxorcist' in proc.info['name'].lower() or 'streamlit' in proc.info['name'].lower():
                    fxorcist_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Record metrics
        metric = {
            'timestamp': datetime.now(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / (1024**3),
            'fxorcist_processes': len(fxorcist_processes),
            'process_details': fxorcist_processes
        }
        metrics.append(metric)

        # Log current status
        logging.info(f"CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%, "
                    f"Disk: {disk.percent:.1f}%, Processes: {len(fxorcist_processes)}")

        # Check for issues
        if cpu_percent > 90:
            logging.warning(f"High CPU usage: {cpu_percent:.1f}%")
        if memory.percent > 85:
            logging.warning(f"High memory usage: {memory.percent:.1f}%")
        if disk.percent > 90:
            logging.warning(f"Low disk space: {disk.percent:.1f}% free")

        time.sleep(10)  # Wait 10 seconds before next measurement

    # Generate summary report
    print("\n" + "="*50)
    print("SYSTEM HEALTH MONITORING REPORT")
    print("="*50)

    if metrics:
        avg_cpu = sum(m['cpu_percent'] for m in metrics) / len(metrics)
        avg_memory = sum(m['memory_percent'] for m in metrics) / len(metrics)
        avg_disk = sum(m['disk_percent'] for m in metrics) / len(metrics)
        max_processes = max(m['fxorcist_processes'] for m in metrics)

        print(f"Average CPU Usage: {avg_cpu:.1f}%")
        print(f"Average Memory Usage: {avg_memory:.1f}%")
        print(f"Average Disk Usage: {avg_disk:.1f}%")
        print(f"Max FXorcist Processes: {max_processes}")

        # Check for concerning patterns
        high_cpu_periods = sum(1 for m in metrics if m['cpu_percent'] > 80)
        high_memory_periods = sum(1 for m in metrics if m['memory_percent'] > 80)

        if high_cpu_periods > len(metrics) * 0.5:
            print("WARNING: High CPU usage detected for more than 50% of monitoring period")
        if high_memory_periods > len(metrics) * 0.5:
            print("WARNING: High memory usage detected for more than 50% of monitoring period")

    print(f"\nMonitoring completed. Collected {len(metrics)} data points.")

# Usage
monitor_system_health(duration_minutes=2)
```

---

*Examples Guide Version: 2.0*
*Last Updated: September 2, 2025*