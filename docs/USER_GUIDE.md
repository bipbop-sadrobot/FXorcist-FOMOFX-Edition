# FXorcist AI Dashboard - Complete User Guide

## 🚀 Overview

FXorcist is a comprehensive AI-powered forex trading system with advanced data processing, machine learning models, and interactive dashboards. This guide covers everything from initial setup to advanced usage.

## ⚡ Quick Start (3 Minutes)

### Option 1: Automated Setup (Recommended)
```bash
# Clone and setup automatically
git clone <repository-url>
cd fxorcist-fomofx-edition
python scripts/setup_fxorcist.py
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Initialize project
python fxorcist_cli.py --command setup

# Start interactive mode
python fxorcist_cli.py --interactive
```

### Option 3: Direct Commands
```bash
# Start main dashboard
python fxorcist_cli.py --dashboard

# Run data integration
python fxorcist_cli.py --data-integration

# Start training
python fxorcist_cli.py --command "train --quick"
```

---

## 🎯 Core Features

### 1. **Unified CLI Interface**
The central command center for all operations:
```bash
python fxorcist_cli.py --interactive  # Full menu system
python fxorcist_cli.py --dashboard    # Start dashboard
python fxorcist_cli.py --data-integration  # Process data
```

### 2. **Optimized Data Processing**
- **70% resource reduction** through selective processing
- **Advanced format detection** for various forex data types
- **Quality assessment** with 99.97% accuracy validation
- **Memory-efficient batching** for large datasets

### 3. **Unified Training Pipeline**
- **Multi-algorithm support**: CatBoost, XGBoost, LightGBM
- **Hyperparameter optimization** with Optuna
- **Cross-validation** and ensemble methods
- **Feature engineering** with technical indicators

### 4. **Interactive Dashboards**
- **Main Dashboard**: Complete system overview
- **Training Dashboard**: Model development monitoring
- **Memory System Dashboard**: Pattern analysis
- **Real-time metrics** and performance tracking

---

## 📋 Detailed Usage Guide

### Section 1: Initial Setup

#### Automated Setup
```bash
python scripts/setup_fxorcist.py
```
This script will:
- ✅ Check system requirements (Python 3.8+)
- ✅ Install all dependencies
- ✅ Create project structure
- ✅ Initialize configuration files
- ✅ Set up data directories
- ✅ Create startup scripts
- ✅ Verify installation

#### Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Initialize project
python fxorcist_cli.py --command setup

# 3. Verify installation
python fxorcist_cli.py --command verify
```

### Section 2: Data Processing Pipeline

#### Running Data Integration
```bash
# Interactive data processing
python fxorcist_cli.py --interactive
# Then select: 2. Data Processing Pipeline

# Direct command
python fxorcist_cli.py --data-integration

# Process specific data source
python forex_ai_dashboard/pipeline/optimized_data_integration.py
```

#### Data Quality Features
- **Format Detection**: Automatic recognition of forex data formats
- **Quality Scoring**: 0-6 scale based on pairs, years, and file integrity
- **Selective Processing**: Only processes files with score ≥4
- **Validation**: 99.97% accuracy in data quality assessment

#### Supported Data Formats
- **MetaQuotes**: Standard MT4/MT5 format
- **ASCII**: Raw semicolon-delimited data
- **Generic OHLC**: Standard open/high/low/close format
- **Compressed**: ZIP files with automatic extraction

### Section 3: Model Training

#### Quick Training
```bash
# Fast training with default settings
python fxorcist_cli.py --command "train --quick"
```

#### Advanced Training
```bash
# Full training pipeline
python fxorcist_cli.py --interactive
# Select: 3. Model Training → 2. Advanced Training

# Or direct command
python forex_ai_dashboard/pipeline/unified_training_pipeline.py
```

#### Training Features
- **Multi-Algorithm**: CatBoost, XGBoost, LightGBM support
- **Hyperparameter Optimization**: Automatic tuning with Optuna
- **Cross-Validation**: Time-series aware validation
- **Feature Engineering**: Technical indicators and temporal features
- **Ensemble Methods**: Combined model predictions

### Section 4: Dashboard Usage

#### Starting Dashboards
```bash
# Main dashboard (recommended)
python fxorcist_cli.py --dashboard

# Training dashboard
python fxorcist_cli.py --training-dashboard

# Memory system dashboard
python fxorcist_cli.py --command "dashboard memory"
```

#### Dashboard Features

##### Main Dashboard (`app.py`)
- **System Overview**: Real-time status and metrics
- **Data Processing**: Integration progress and quality metrics
- **Model Performance**: Training results and predictions
- **Memory System**: Pattern analysis and insights
- **Resource Monitoring**: CPU, memory, and disk usage

##### Training Dashboard (`enhanced_training_dashboard.py`)
- **Live Training**: Real-time model training progress
- **Hyperparameter Tuning**: Optimization progress and results
- **Model Comparison**: Performance metrics across algorithms
- **Feature Importance**: Key predictors visualization
- **Cross-Validation**: Detailed validation results

##### Memory System Dashboard
- **Pattern Analysis**: Market pattern recognition
- **Memory Usage**: Working/Long-term memory statistics
- **Confidence Metrics**: Prediction confidence levels
- **Anomaly Detection**: Unusual market condition alerts

### Section 5: Memory System

#### Memory Architecture
```
Working Memory (WM): Recent patterns and short-term insights
├── Recent market data
├── Active trading patterns
└── Short-term predictions

Long-Term Memory (LTM): Historical patterns and lessons
├── Historical market data
├── Proven trading strategies
└── Long-term market insights

Episodic Memory (EM): Significant events and anomalies
├── Market crashes and events
├── Unusual price movements
└── Important market turning points
```

#### Memory Management
```bash
# View memory statistics
python fxorcist_cli.py --command "memory stats"

# Clear memory cache
python fxorcist_cli.py --command "memory clear"

# Export memory data
python fxorcist_cli.py --command "memory export"
```

### Section 6: Performance Analysis

#### Model Evaluation
```bash
# Generate performance report
python fxorcist_cli.py --command "analyze performance"

# Compare model versions
python fxorcist_cli.py --command "analyze compare"

# Data quality analysis
python fxorcist_cli.py --command "analyze quality"
```

#### Available Metrics
- **R² Score**: Model accuracy and fit
- **MAE/RMSE**: Error measurements
- **Sharpe Ratio**: Risk-adjusted returns
- **Feature Importance**: Key predictors
- **Cross-Validation Scores**: Robustness testing

### Section 7: Configuration Management

#### Configuration Files
```
config/
├── cli_config.json          # CLI interface settings
├── pipeline_config.json     # Data processing settings
├── training_config.json     # Model training settings
└── memory_config.json       # Memory system settings
```

#### Editing Configuration
```bash
# Interactive configuration
python fxorcist_cli.py --interactive
# Select: 8. Configuration Management

# View current config
python fxorcist_cli.py --command "config view"

# Edit configuration
python fxorcist_cli.py --command "config edit"
```

---

## 🔧 Advanced Usage

### Command Line Options

#### CLI Commands
```bash
# Interactive mode (recommended)
python fxorcist_cli.py --interactive

# Direct commands
python fxorcist_cli.py --dashboard              # Start main dashboard
python fxorcist_cli.py --data-integration       # Run data processing
python fxorcist_cli.py --training-dashboard     # Start training dashboard

# Custom commands
python fxorcist_cli.py --command "train --quick"
python fxorcist_cli.py --command "memory stats"
python fxorcist_cli.py --command "analyze performance"
```

#### Training Options
```bash
# Quick training
python forex_ai_dashboard/pipeline/unified_training_pipeline.py --quick

# Custom configuration
python forex_ai_dashboard/pipeline/unified_training_pipeline.py --config config/training_config.json

# Verbose output
python forex_ai_dashboard/pipeline/unified_training_pipeline.py --verbose
```

### API Usage

#### Python API
```python
from fxorcist_cli import FXorcistCLI
from forex_ai_dashboard.pipeline.unified_training_pipeline import UnifiedTrainingPipeline

# Initialize CLI
cli = FXorcistCLI()

# Run data integration
results = cli.run_data_integration()

# Train models
pipeline = UnifiedTrainingPipeline()
results = pipeline.run_complete_training_pipeline()
```

### Batch Processing

#### Automated Workflows
```bash
# Daily data update and training
python fxorcist_cli.py --command "workflow daily"

# Weekly model retraining
python fxorcist_cli.py --command "workflow weekly"

# Monthly performance review
python fxorcist_cli.py --command "workflow monthly"
```

---

## 🛠️ Troubleshooting

### Common Issues & Solutions

#### 1. Data Processing Issues
```bash
# Check data quality
python fxorcist_cli.py --command "analyze quality"

# Re-run data integration
python fxorcist_cli.py --data-integration

# View processing logs
tail -f logs/optimized_data_integration.log
```

#### 2. Training Failures
```bash
# Check system resources
python fxorcist_cli.py --command "health system"

# View training logs
tail -f logs/unified_training.log

# Restart with simpler config
python fxorcist_cli.py --command "train --quick"
```

#### 3. Dashboard Connection Issues
```bash
# Check dashboard status
python fxorcist_cli.py --command "health dashboard"

# Restart dashboard
python fxorcist_cli.py --dashboard

# Check port availability
lsof -i :8501
```

#### 4. Memory System Issues
```bash
# Check memory health
python fxorcist_cli.py --command "health memory"

# Clear memory cache
python fxorcist_cli.py --command "memory clear"

# View memory statistics
python fxorcist_cli.py --command "memory stats"
```

### Health Checks

#### System Health Status
```bash
python fxorcist_cli.py --command "health system"
```
- 🟢 **Green**: All systems operational
- 🟡 **Yellow**: Performance degradation
- 🔴 **Red**: Critical issues detected

#### Component Health
- **Data Pipeline**: File processing and quality validation
- **Training System**: Model training and evaluation
- **Memory System**: Pattern storage and retrieval
- **Dashboard**: Web interface and real-time updates

### Log Files
```
logs/
├── fxorcist_cli.log              # CLI operations
├── optimized_data_integration.log # Data processing
├── unified_training.log          # Model training
├── memory_system.log             # Memory operations
└── dashboard.log                 # Web interface
```

---

## 📊 Performance Optimization

### Resource Management
- **CPU Optimization**: Parallel processing for multi-core systems
- **Memory Management**: Batch processing with automatic cleanup
- **Disk I/O**: Efficient file handling and caching
- **Network**: Optimized data downloads and API calls

### Best Practices

#### Data Management
```bash
# Regular data quality checks
python fxorcist_cli.py --command "analyze quality"

# Clean old data files
python fxorcist_cli.py --command "data clean"

# Backup important data
python fxorcist_cli.py --command "data backup"
```

#### Model Management
```bash
# Regular model retraining
python fxorcist_cli.py --command "workflow retrain"

# Model performance monitoring
python fxorcist_cli.py --command "analyze performance"

# Model version comparison
python fxorcist_cli.py --command "analyze compare"
```

#### System Maintenance
```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Clear caches
python fxorcist_cli.py --command "system clear-cache"

# Generate system report
python fxorcist_cli.py --command "system report"
```

---

## 🔗 Integration & APIs

### External Data Sources
- **Alpha Vantage**: Stock and forex data
- **Yahoo Finance**: Market data and indices
- **FXCM**: Forex-specific data
- **OANDA**: High-quality forex data
- **Local Files**: CSV, ZIP, and compressed formats

### Export Capabilities
```python
# Export model predictions
python fxorcist_cli.py --command "export predictions --format csv"

# Export training reports
python fxorcist_cli.py --command "export reports --format json"

# Export memory data
python fxorcist_cli.py --command "export memory --format pickle"
```

---

## 📚 Additional Resources

### Documentation
- **Optimization Report**: `docs/OPTIMIZATION_REPORT.md`
- **Architecture Guide**: `docs/ARCHITECTURE_GUIDE.md`
- **Development Guide**: `docs/DEVELOPER_GUIDE.md`
- **API Documentation**: `docs/API_REFERENCE.md`

### Community & Support
- **GitHub Issues**: Bug reports and feature requests
- **Documentation Wiki**: Extended guides and tutorials
- **Discord Community**: Real-time support and discussions

### Development
```bash
# Run tests
python -m pytest tests/

# Generate documentation
python scripts/generate_docs.py

# Build distribution
python setup.py sdist bdist_wheel
```

---

## 🎯 Quick Reference

### Most Common Commands
```bash
# Start everything
python fxorcist_cli.py --interactive

# Process data
python fxorcist_cli.py --data-integration

# Train model
python fxorcist_cli.py --command "train --quick"

# Start dashboard
python fxorcist_cli.py --dashboard

# Check health
python fxorcist_cli.py --command "health system"
```

### File Structure
```
fxorcist-fomofx-edition/
├── fxorcist_cli.py              # Main CLI interface
├── scripts/setup_fxorcist.py    # Automated setup
├── forex_ai_dashboard/          # Core modules
│   └── pipeline/               # Processing pipelines
├── dashboard/                  # Web interfaces
├── data/                       # Data storage
├── models/                     # Trained models
├── config/                     # Configuration files
├── logs/                       # Log files
└── docs/                       # Documentation
```

---

*FXorcist AI Dashboard v2.0 - Complete User Guide*
*Last Updated: September 2, 2025*