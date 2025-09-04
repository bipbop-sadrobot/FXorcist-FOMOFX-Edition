# FXorcist CLI Guide

## Overview

The FXorcist CLI provides a comprehensive command-line interface for managing forex data processing, model training, and system monitoring. Built with Click, it offers intuitive commands with rich output formatting and proper error handling.

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# Verify installation
fxorcist_cli.py --help
```

## Command Structure

The CLI is organized into logical command groups:

- `data`: Data processing and management
- `train`: Model training and optimization
- `dashboard`: Dashboard and visualization
- `memory`: Memory system management
- `config`: Configuration management

## Global Options

- `--debug/--no-debug`: Enable debug logging
- `--help`: Show help message

## Data Commands

### Data Integration

Process and integrate forex data files:

```bash
# Run data integration with default settings
fxorcist_cli.py data integrate

# Specify input directory
fxorcist_cli.py data integrate --input-dir /path/to/data

# Force reprocessing of existing data
fxorcist_cli.py data integrate --force
```

### Data Validation

Validate data quality:

```bash
# Validate specific file
fxorcist_cli.py data validate /path/to/file.csv
```

## Training Commands

### Start Training

```bash
# Quick training mode
fxorcist_cli.py train start --quick

# Full training with config file
fxorcist_cli.py train start --config training_config.yaml
```

### Hyperparameter Optimization

```bash
# Run hyperparameter optimization
fxorcist_cli.py train optimize
```

## Dashboard Commands

### Main Dashboard

```bash
# Start main dashboard on default port
fxorcist_cli.py dashboard start

# Specify custom port
fxorcist_cli.py dashboard start --port 8888
```

### Training Dashboard

```bash
# Start training visualization dashboard
fxorcist_cli.py dashboard training
```

## Memory System Commands

### Memory Statistics

```bash
# View memory system stats
fxorcist_cli.py memory stats
```

### Cache Management

```bash
# Clear memory cache
fxorcist_cli.py memory clear
```

## Configuration Commands

### View Configuration

```bash
# Display current configuration
fxorcist_cli.py config view
```

### Modify Configuration

```bash
# Set configuration value
fxorcist_cli.py config set --key dashboard_port --value 8501

# Reset to defaults
fxorcist_cli.py config reset
```

## Error Handling

The CLI provides clear error messages with proper exit codes:

- Exit code 0: Success
- Exit code 1: Error (with detailed message)

Example error output:
```
❌ Error: Data integration failed: Invalid file format
```

## Configuration File

The CLI uses a configuration file located at `config/cli_config.json`. Default settings:

```json
{
    "data_dir": "data",
    "models_dir": "models",
    "logs_dir": "logs",
    "dashboard_port": 8501,
    "auto_backup": true,
    "quality_threshold": 0.7,
    "batch_size": 1000
}
```

## Logging

Logs are written to the configured logs directory with rich formatting:

- Info level: Normal operations
- Warning level: Non-critical issues
- Error level: Critical failures
- Debug level: Detailed debugging (when --debug is enabled)

## Best Practices

1. **Use Help**: Always check command help with `--help` for detailed usage information.
2. **Configuration**: Use the config commands to manage settings rather than editing files directly.
3. **Validation**: Validate data files before processing to catch issues early.
4. **Monitoring**: Use the dashboard commands to monitor long-running operations.
5. **Error Handling**: Check error messages and logs for troubleshooting.

## Examples

### Complete Training Workflow

```bash
# 1. Validate input data
fxorcist_cli.py data validate input.csv

# 2. Run data integration
fxorcist_cli.py data integrate --input-dir data/raw

# 3. Start training
fxorcist_cli.py train start --config my_config.yaml

# 4. Monitor progress
fxorcist_cli.py dashboard training
```

### System Maintenance

```bash
# 1. Check memory stats
fxorcist_cli.py memory stats

# 2. Clear cache if needed
fxorcist_cli.py memory clear

# 3. Verify system health
fxorcist_cli.py config view
```

## Troubleshooting

Common issues and solutions:

1. **Command not found**
   - Ensure the CLI script is in your PATH
   - Verify Python environment is activated

2. **Dashboard won't start**
   - Check if port is in use
   - Verify streamlit installation

3. **Data processing errors**
   - Validate input file format
   - Check disk space
   - Enable debug logging

4. **Memory system issues**
   - Clear cache
   - Check system resources
   - Review memory stats

## Support

For additional help:
- Check the full documentation in the `docs` directory
- Review the logs in the configured logs directory
- Use `--debug` for detailed logging