# FXorcist API Reference

## Overview

This document provides comprehensive API reference for the FXorcist AI Dashboard system, including all classes, methods, and configuration options.

## Table of Contents

- [Core Classes](#core-classes)
- [Pipeline Components](#pipeline-components)
- [CLI Interface](#cli-interface)
- [Configuration](#configuration)
- [Error Handling](#error-handling)

## Core Classes

### FXorcistCLI

Main command-line interface for the system.

#### Methods

##### `__init__()`
Initialize the CLI interface.
```python
cli = FXorcistCLI()
```

##### `interactive_menu()`
Launch interactive menu system.
```python
cli.interactive_menu()
```

##### `run_data_integration()`
Execute optimized data integration pipeline.
```python
results = cli.run_data_integration()
# Returns: {"processed": int, "skipped": int, "total": int}
```

##### `start_dashboard(dashboard_name: str)`
Start a specific dashboard.
```python
cli.start_dashboard("main")  # Start main dashboard
cli.start_dashboard("training")  # Start training dashboard
```

### OptimizedDataIntegrator

Advanced data integration with quality assessment.

#### Methods

##### `__init__()`
Initialize the data integrator.
```python
integrator = OptimizedDataIntegrator()
```

##### `process_optimized_data()`
Run complete data integration pipeline.
```python
results = integrator.process_optimized_data()
# Returns: {"processed": int, "skipped": int, "total": int}
```

##### `find_optimized_m1_files()`
Find high-quality M1 files with assessment.
```python
files = integrator.find_optimized_m1_files()
# Returns: List[Tuple[Path, Dict]] - (file_path, metadata)
```

##### `quick_validate_zip_content(zip_path: Path)`
Quick validation of zip file content.
```python
is_valid = integrator.quick_validate_zip_content(zip_path)
# Returns: bool
```

### ForexDataFormatDetector

Advanced data format detection and parsing.

#### Methods

##### `__init__()`
Initialize the format detector.
```python
detector = ForexDataFormatDetector()
```

##### `detect_and_parse(file_path: Path, sample_size: int = 1000)`
Detect format and parse data.
```python
df = detector.detect_and_parse("data.csv", sample_size=1000)
# Returns: pd.DataFrame or None
```

##### `_detect_format(file_path: Path)`
Detect data format with confidence score.
```python
format_type, confidence = detector._detect_format(file_path)
# Returns: Tuple[str, float] - (format_name, confidence_score)
```

##### `_clean_and_validate(df: pd.DataFrame)`
Clean and validate parsed data.
```python
clean_df = detector._clean_and_validate(raw_df)
# Returns: pd.DataFrame
```

### DataQualityValidator

Data quality assessment and validation.

#### Methods

##### `__init__()`
Initialize quality validator.
```python
validator = DataQualityValidator()
```

##### `validate_dataset(df: pd.DataFrame, symbol: str)`
Comprehensive data quality validation.
```python
quality = validator.validate_dataset(df, "EURUSD")
# Returns: Dict with quality metrics
```

##### `get_quality_report()`
Generate overall quality report.
```python
report = validator.get_quality_report()
# Returns: Dict with quality statistics
```

## Pipeline Components

### UnifiedTrainingPipeline

Consolidated training pipeline with multiple algorithms.

#### Methods

##### `__init__(config: Optional[Dict] = None)`
Initialize training pipeline.
```python
pipeline = UnifiedTrainingPipeline(config)
```

##### `run_complete_training_pipeline(data_source: Optional[str] = None)`
Run complete training pipeline.
```python
results = pipeline.run_complete_training_pipeline()
# Returns: Dict with training results
```

##### `_train_models(data: pd.DataFrame)`
Train multiple models with optimization.
```python
training_results = pipeline._train_models(feature_data)
# Returns: Dict with model results
```

##### `_evaluate_models()`
Evaluate trained models.
```python
evaluation = pipeline._evaluate_models()
# Returns: Dict with evaluation metrics
```

### EnhancedTrainingPipeline

Advanced training with hyperparameter optimization.

#### Methods

##### `__init__(config: Optional[Dict] = None)`
Initialize enhanced training pipeline.
```python
pipeline = EnhancedTrainingPipeline(config)
```

##### `train_with_optimization(data: pd.DataFrame, target: pd.Series)`
Train with hyperparameter optimization.
```python
model, metrics = pipeline.train_with_optimization(X, y)
# Returns: Tuple[model, Dict]
```

##### `cross_validate_model(model, X: pd.DataFrame, y: pd.Series)`
Perform cross-validation.
```python
scores = pipeline.cross_validate_model(model, X, y)
# Returns: Dict with CV scores
```

## CLI Interface

### Command Structure

```bash
python fxorcist_cli.py [command] [options]
```

### Available Commands

#### Interactive Mode
```bash
python fxorcist_cli.py --interactive
```
Launches full menu-driven interface.

#### Data Integration
```bash
python fxorcist_cli.py --data-integration
python fxorcist_cli.py --command "data process"
```

#### Dashboard Management
```bash
python fxorcist_cli.py --dashboard
python fxorcist_cli.py --training-dashboard
python fxorcist_cli.py --command "dashboard start main"
python fxorcist_cli.py --command "dashboard stop all"
```

#### Training
```bash
python fxorcist_cli.py --command "train --quick"
python fxorcist_cli.py --command "train --advanced"
python fxorcist_cli.py --command "train --hyperopt"
```

#### Health Checks
```bash
python fxorcist_cli.py --command "health system"
python fxorcist_cli.py --command "health check"
python fxorcist_cli.py --command "health monitor"
```

#### Configuration
```bash
python fxorcist_cli.py --command "config view"
python fxorcist_cli.py --command "config edit"
python fxorcist_cli.py --command "config reset"
```

## Configuration

### Configuration Files

#### CLI Configuration (`config/cli_config.json`)
```json
{
  "data_dir": "data",
  "models_dir": "models",
  "logs_dir": "logs",
  "dashboard_port": 8501,
  "auto_backup": true,
  "quality_threshold": 0.7,
  "batch_size": 1000,
  "log_level": "INFO"
}
```

#### Pipeline Configuration (`config/pipeline_config.json`)
```json
{
  "data_quality_threshold": 0.7,
  "batch_size": 1000,
  "memory_efficient": true,
  "parallel_processing": true,
  "cache_enabled": true,
  "auto_cleanup": true
}
```

#### Training Configuration (`config/training_config.json`)
```json
{
  "default_model": "catboost",
  "cross_validation_folds": 5,
  "hyperparameter_optimization": true,
  "n_trials": 50,
  "early_stopping": true,
  "feature_selection": true,
  "ensemble_methods": true
}
```

### Environment Variables

#### Dashboard Configuration
```bash
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
FXORCIST_ENV=production
FXORCIST_LOG_LEVEL=INFO
```

#### Training Configuration
```bash
FXORCIST_MODEL_DIR=models/
FXORCIST_DATA_DIR=data/
FXORCIST_MEMORY_LIMIT=0.8
FXORCIST_CPU_CORES=4
```

## Error Handling

### Exception Types

#### DataProcessingError
Raised when data processing fails.
```python
try:
    results = integrator.process_optimized_data()
except DataProcessingError as e:
    logger.error(f"Data processing failed: {e}")
    # Handle error appropriately
```

#### ModelTrainingError
Raised when model training fails.
```python
try:
    model = pipeline.train_model(X, y)
except ModelTrainingError as e:
    logger.error(f"Training failed: {e}")
    # Fallback to alternative approach
```

#### ConfigurationError
Raised when configuration is invalid.
```python
try:
    config = load_config()
except ConfigurationError as e:
    logger.error(f"Configuration error: {e}")
    # Use default configuration
```

### Error Codes

| Code | Description | Action |
|------|-------------|--------|
| 1001 | Data file not found | Check file paths and permissions |
| 1002 | Invalid data format | Verify data format and structure |
| 1003 | Memory limit exceeded | Reduce batch size or increase memory |
| 2001 | Model training failed | Check data quality and parameters |
| 2002 | Hyperparameter optimization failed | Review optimization settings |
| 3001 | Dashboard startup failed | Check port availability and dependencies |
| 3002 | Health check failed | Review system resources and logs |

## Data Structures

### Data Integration Results
```python
{
    "processed": 150,
    "skipped": 25,
    "total": 175,
    "quality_score": 0.97,
    "processing_time": 45.2,
    "errors": []
}
```

### Training Results
```python
{
    "model_type": "catboost",
    "train_score": 0.95,
    "test_score": 0.89,
    "mae": 0.012,
    "rmse": 0.015,
    "feature_importance": {"feature1": 0.25, "feature2": 0.18},
    "training_time": 120.5,
    "best_params": {"iterations": 1000, "learning_rate": 0.1}
}
```

### Health Check Results
```python
{
    "status": "healthy",
    "response_time": 1.2,
    "metrics": {
        "cpu_percent": 45.2,
        "memory_percent": 67.8,
        "disk_percent": 23.4
    },
    "components": {
        "system": "healthy",
        "data_pipeline": "healthy",
        "training_pipeline": "warning"
    }
}
```

## Performance Optimization

### Memory Management
```python
# Configure memory limits
config = {
    "memory_limit": 0.8,  # 80% of available memory
    "batch_size": 1000,   # Process in batches
    "cache_enabled": True,
    "auto_cleanup": True
}
```

### Parallel Processing
```python
# Enable parallel processing
config = {
    "parallel_processing": True,
    "max_workers": 4,
    "thread_pool_size": 8
}
```

### Caching Strategies
```python
# Configure caching
config = {
    "cache_enabled": True,
    "cache_dir": "cache/",
    "cache_ttl": 3600,  # 1 hour
    "memory_cache_size": 1000
}
```

## Integration Examples

### Python API Usage
```python
from fxorcist_cli import FXorcistCLI
from forex_ai_dashboard.pipeline.unified_training_pipeline import UnifiedTrainingPipeline

# Initialize components
cli = FXorcistCLI()
pipeline = UnifiedTrainingPipeline()

# Run data integration
data_results = cli.run_data_integration()

# Train models
training_results = pipeline.run_complete_training_pipeline()

# Start dashboard
cli.start_dashboard("main")
```

### Command Line Automation
```bash
# Complete workflow
python fxorcist_cli.py --data-integration
python fxorcist_cli.py --command "train --quick"
python fxorcist_cli.py --dashboard

# Batch processing
python fxorcist_cli.py --command "workflow daily"
python fxorcist_cli.py --command "workflow weekly"
```

### Configuration Management
```python
from forex_ai_dashboard.utils.config_manager import ConfigManager

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config("production")

# Update settings
config["batch_size"] = 2000
config_manager.save_config()
```

---

*API Reference Version: 2.0*
*Last Updated: September 2, 2025*