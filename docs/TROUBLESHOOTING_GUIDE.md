# FXorcist Troubleshooting Guide

## Overview

This guide provides solutions to common issues and problems that may occur when using the FXorcist AI Dashboard system. Each section includes diagnostic steps, solutions, and preventive measures.

## Table of Contents

- [Quick Diagnosis](#quick-diagnosis)
- [Installation Issues](#installation-issues)
- [Data Processing Problems](#data-processing-problems)
- [Training Pipeline Issues](#training-pipeline-issues)
- [Dashboard Problems](#dashboard-problems)
- [Performance Issues](#performance-issues)
- [Memory and Resource Problems](#memory-and-resource-problems)
- [Network and Connectivity Issues](#network-and-connectivity-issues)
- [Configuration Problems](#configuration-problems)
- [Advanced Debugging](#advanced-debugging)

## Quick Diagnosis

### System Health Check

Run the comprehensive health check command:
```bash
python fxorcist_cli.py --command "health system"
```

This will check:
- System resources (CPU, memory, disk)
- Python environment and dependencies
- Data pipeline status
- Training pipeline status
- Dashboard availability
- Configuration validity

### Generate Diagnostic Report

```bash
python fxorcist_cli.py --command "diagnostics"
```

This creates a detailed report including:
- System information
- Configuration status
- Log analysis
- Performance metrics
- Error summaries

### Quick Status Check

```bash
# Check running processes
ps aux | grep -E "(streamlit|python.*fxorcist)"

# Check port usage
lsof -i :8501

# Check disk space
df -h

# Check memory usage
free -h
```

## Installation Issues

### Python Version Problems

**Error**: `Python version 3.8+ required`
```bash
# Check Python version
python --version

# Install Python 3.8+ (Ubuntu/Debian)
sudo apt update
sudo apt install python3.8 python3.8-venv

# Install Python 3.8+ (macOS with Homebrew)
brew install python@3.8

# Create virtual environment
python3.8 -m venv fxorcist_env
source fxorcist_env/bin/activate
```

**Error**: `ModuleNotFoundError` for core packages
```bash
# Reinstall requirements
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Or install specific package
pip install streamlit==1.25.0 pandas==2.0.3
```

### Dependency Conflicts

**Error**: `ImportError: cannot import name`
```bash
# Check package versions
pip list | grep -E "(streamlit|pandas|numpy|scikit)"

# Update conflicting packages
pip install --upgrade streamlit pandas numpy

# Use specific versions
pip install streamlit==1.25.0 pandas==2.0.3 numpy==1.24.3
```

### Virtual Environment Issues

**Error**: `virtualenv: command not found`
```bash
# Install virtualenv
pip install virtualenv

# Or use built-in venv
python3 -m venv fxorcist_env

# Activate environment
source fxorcist_env/bin/activate  # Linux/macOS
# fxorcist_env\Scripts\activate   # Windows
```

## Data Processing Problems

### File Not Found Errors

**Error**: `FileNotFoundError: No such file or directory`
```bash
# Check file existence
ls -la data/

# Check permissions
ls -ld data/

# Fix permissions
chmod -R 755 data/

# Check file paths in configuration
python fxorcist_cli.py --command "config view"
```

### Data Format Issues

**Error**: `ValueError: Unknown data format`
```bash
# Detect file format
python -c "
from forex_ai_dashboard.pipeline.data_ingestion import ForexDataFormatDetector
detector = ForexDataFormatDetector()
print(detector.detect_format('data/your_file.csv'))
"

# Convert data format
python fxorcist_cli.py --command "data convert --input your_file.csv --output your_file_converted.csv"
```

### Memory Issues During Data Processing

**Error**: `MemoryError` or `Killed` process
```bash
# Reduce batch size
python fxorcist_cli.py --command "config edit --key batch_size --value 500"

# Enable memory efficient mode
python fxorcist_cli.py --command "config edit --key memory_efficient --value true"

# Process in chunks
python fxorcist_cli.py --command "data process --chunk_size 1000"
```

### Data Quality Problems

**Error**: `DataQualityError: Low quality score`
```bash
# Check data quality
python fxorcist_cli.py --command "data validate"

# Generate quality report
python fxorcist_cli.py --command "data quality --report"

# Clean data
python fxorcist_cli.py --command "data clean --aggressive"
```

## Training Pipeline Issues

### Model Training Failures

**Error**: `ModelTrainingError: Training failed`
```bash
# Check training data
python fxorcist_cli.py --command "data validate --training"

# Verify model configuration
python fxorcist_cli.py --command "config view --section training"

# Start with simple model
python fxorcist_cli.py --command "train --model linear --quick"
```

### Hyperparameter Optimization Issues

**Error**: `OptimizationError: No improvement found`
```bash
# Reduce optimization trials
python fxorcist_cli.py --command "config edit --key n_trials --value 25"

# Use different optimization method
python fxorcist_cli.py --command "train --opt_method random"

# Skip optimization for quick training
python fxorcist_cli.py --command "train --no-opt"
```

### GPU/CUDA Problems

**Error**: `RuntimeError: CUDA out of memory`
```bash
# Check GPU memory
nvidia-smi

# Reduce batch size
python fxorcist_cli.py --command "config edit --key batch_size --value 256"

# Disable GPU
export CUDA_VISIBLE_DEVICES=""

# Use CPU only
python fxorcist_cli.py --command "train --device cpu"
```

### Cross-Validation Errors

**Error**: `ValueError: Not enough data for cross-validation`
```bash
# Check data size
python fxorcist_cli.py --command "data info"

# Reduce CV folds
python fxorcist_cli.py --command "config edit --key cross_validation_folds --value 3"

# Skip cross-validation
python fxorcist_cli.py --command "train --no-cv"
```

## Dashboard Problems

### Dashboard Won't Start

**Error**: `StreamlitConnectionError` or port issues
```bash
# Check port availability
lsof -i :8501

# Kill conflicting process
kill -9 $(lsof -ti :8501)

# Use different port
export STREAMLIT_SERVER_PORT=8502
python fxorcist_cli.py --dashboard
```

### Dashboard Loading Issues

**Error**: `StreamlitAPIException: Session not found`
```bash
# Clear Streamlit cache
rm -rf ~/.streamlit/

# Restart dashboard
python fxorcist_cli.py --command "dashboard restart"

# Check browser cache
# Clear browser cache and cookies for localhost:8501
```

### Widget Interaction Problems

**Error**: `DuplicateWidgetID` or interaction failures
```bash
# Clear session state
python fxorcist_cli.py --command "dashboard clear-session"

# Restart with clean state
python fxorcist_cli.py --command "dashboard restart --clean"
```

### Performance Issues with Large Datasets

**Error**: Dashboard becomes unresponsive
```bash
# Enable data caching
python fxorcist_cli.py --command "config edit --key cache_enabled --value true"

# Reduce data sample size
python fxorcist_cli.py --command "dashboard --sample 10000"

# Use pagination
python fxorcist_cli.py --command "config edit --key pagination_enabled --value true"
```

## Performance Issues

### Slow Data Processing

**Symptoms**: Data processing takes too long
```bash
# Enable parallel processing
python fxorcist_cli.py --command "config edit --key parallel_processing --value true"

# Increase worker count
python fxorcist_cli.py --command "config edit --key max_workers --value 8"

# Optimize data types
python fxorcist_cli.py --command "data optimize --types"
```

### Slow Model Training

**Symptoms**: Training takes excessive time
```bash
# Use faster algorithm
python fxorcist_cli.py --command "train --model lightgbm"

# Reduce training data size
python fxorcist_cli.py --command "train --sample 50000"

# Enable early stopping
python fxorcist_cli.py --command "config edit --key early_stopping --value true"
```

### High CPU Usage

**Symptoms**: System becomes unresponsive
```bash
# Check CPU usage
top -p $(pgrep -f fxorcist)

# Reduce worker threads
python fxorcist_cli.py --command "config edit --key max_workers --value 2"

# Enable CPU affinity
taskset -c 0-3 python fxorcist_cli.py --dashboard
```

### Database Query Performance

**Symptoms**: Slow dashboard queries
```bash
# Check query performance
python fxorcist_cli.py --command "db analyze"

# Add database indexes
python fxorcist_cli.py --command "db optimize"

# Enable query caching
python fxorcist_cli.py --command "config edit --key query_cache --value true"
```

## Memory and Resource Problems

### Out of Memory Errors

**Error**: `MemoryError` or system OOM killer
```bash
# Check memory usage
free -h

# Reduce memory usage
python fxorcist_cli.py --command "config edit --key memory_limit --value 0.6"

# Enable memory monitoring
python fxorcist_cli.py --command "monitor memory --alert"

# Use memory efficient algorithms
python fxorcist_cli.py --command "config edit --key memory_efficient --value true"
```

### Disk Space Issues

**Error**: `No space left on device`
```bash
# Check disk usage
df -h

# Clean up temporary files
python fxorcist_cli.py --command "cleanup temp"

# Compress old logs
python fxorcist_cli.py --command "logs compress"

# Move data to external storage
python fxorcist_cli.py --command "data archive --external"
```

### Resource Leaks

**Symptoms**: Memory/CPU usage increases over time
```bash
# Monitor resource usage
python fxorcist_cli.py --command "monitor resources"

# Enable garbage collection
python fxorcist_cli.py --command "config edit --key gc_enabled --value true"

# Restart services periodically
python fxorcist_cli.py --command "services restart --schedule"
```

## Network and Connectivity Issues

### Connection Refused

**Error**: `Connection refused` when accessing dashboard
```bash
# Check if service is running
ps aux | grep streamlit

# Check firewall
sudo ufw status
sudo ufw allow 8501

# Check network interfaces
ip addr show

# Bind to all interfaces
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Slow Network Performance

**Symptoms**: Slow dashboard loading or data transfer
```bash
# Check network speed
speedtest-cli

# Enable compression
python fxorcist_cli.py --command "config edit --key compression --value true"

# Reduce data transfer
python fxorcist_cli.py --command "dashboard --lightweight"
```

### SSL/HTTPS Issues

**Error**: SSL certificate problems
```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

# Configure SSL
export STREAMLIT_SERVER_CERT_FILE=cert.pem
export STREAMLIT_SERVER_KEY_FILE=key.pem
export STREAMLIT_SERVER_ENABLE_CORS=false
```

## Configuration Problems

### Invalid Configuration

**Error**: `ConfigurationError: Invalid config`
```bash
# Validate configuration
python fxorcist_cli.py --command "config validate"

# Reset to defaults
python fxorcist_cli.py --command "config reset"

# Edit configuration
python fxorcist_cli.py --command "config edit"
```

### Environment Variable Issues

**Error**: Environment variables not recognized
```bash
# Check environment variables
env | grep FXORCIST

# Export variables
export FXORCIST_ENV=production
export FXORCIST_LOG_LEVEL=INFO

# Add to shell profile
echo 'export FXORCIST_ENV=production' >> ~/.bashrc
source ~/.bashrc
```

### Path Configuration Problems

**Error**: `Path not found` or directory issues
```bash
# Check paths
python fxorcist_cli.py --command "config paths"

# Create missing directories
python fxorcist_cli.py --command "setup directories"

# Fix permissions
sudo chown -R $USER:$USER /path/to/fxorcist
```

## Advanced Debugging

### Log Analysis

```bash
# View recent logs
python fxorcist_cli.py --command "logs tail"

# Search for errors
python fxorcist_cli.py --command "logs grep ERROR"

# Analyze log patterns
python fxorcist_cli.py --command "logs analyze"
```

### Debug Mode

```bash
# Enable debug logging
export FXORCIST_LOG_LEVEL=DEBUG
export STREAMLIT_LOG_LEVEL=DEBUG

# Run with debug flags
python -m pdb fxorcist_cli.py --dashboard

# Enable verbose output
python fxorcist_cli.py --verbose --debug
```

### Performance Profiling

```bash
# Profile application
python -m cProfile -s time fxorcist_cli.py --command "data process" > profile.txt

# Memory profiling
python -m memory_profiler fxorcist_cli.py --command "train --quick"

# CPU profiling
python -c "
import cProfile
cProfile.run('from fxorcist_cli import main; main()', 'profile.prof')
"
```

### System Tracing

```bash
# Trace system calls
strace -p $(pgrep -f fxorcist) -o trace.log

# Network tracing
tcpdump -i any port 8501 -w network_trace.pcap

# Process monitoring
perf record -p $(pgrep -f fxorcist) -o perf.data
```

### Database Debugging

```bash
# Check database connections
python fxorcist_cli.py --command "db status"

# Analyze slow queries
python fxorcist_cli.py --command "db slow-queries"

# Check database locks
python fxorcist_cli.py --command "db locks"
```

### Memory Debugging

```bash
# Check memory leaks
python -c "
import tracemalloc
tracemalloc.start()
# Run your code here
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
"

# Heap analysis
python -m guppy fxorcist_cli.py --command "data process"
```

## Common Error Codes

| Error Code | Description | Solution |
|------------|-------------|----------|
| 1001 | Data file not found | Check file paths and permissions |
| 1002 | Invalid data format | Verify data format and use converter |
| 1003 | Memory limit exceeded | Reduce batch size or increase memory |
| 2001 | Model training failed | Check data quality and reduce complexity |
| 2002 | Hyperparameter optimization failed | Reduce trials or use simpler method |
| 3001 | Dashboard startup failed | Check port availability and dependencies |
| 3002 | Widget interaction failed | Clear session and restart |
| 4001 | Configuration invalid | Validate and reset configuration |
| 4002 | Environment not set | Export required environment variables |
| 5001 | Network connection failed | Check firewall and network settings |
| 5002 | SSL certificate error | Generate or configure SSL certificates |

## Preventive Maintenance

### Regular Health Checks

```bash
# Daily health check
python fxorcist_cli.py --command "health daily"

# Weekly system maintenance
python fxorcist_cli.py --command "maintenance weekly"

# Monthly cleanup
python fxorcist_cli.py --command "cleanup monthly"
```

### Backup Strategy

```bash
# Automated backups
python fxorcist_cli.py --command "backup create"

# Verify backups
python fxorcist_cli.py --command "backup verify"

# Restore from backup
python fxorcist_cli.py --command "backup restore --file backup_20231201.tar.gz"
```

### Performance Monitoring

```bash
# Enable monitoring
python fxorcist_cli.py --command "monitor enable"

# Set up alerts
python fxorcist_cli.py --command "alerts configure --email admin@example.com"

# Generate performance reports
python fxorcist_cli.py --command "reports performance --weekly"
```

### Log Rotation

```bash
# Configure log rotation
python fxorcist_cli.py --command "logs rotate --size 10MB --count 5"

# Compress old logs
python fxorcist_cli.py --command "logs compress --older 30d"

# Archive logs
python fxorcist_cli.py --command "logs archive --destination /backup/logs"
```

## Getting Help

### Support Resources

1. **Documentation**
   - User Guide: `docs/USER_GUIDE.md`
   - API Reference: `docs/API_REFERENCE.md`
   - Deployment Guide: `docs/DEPLOYMENT_GUIDE.md`

2. **Community Support**
   - GitHub Issues: Report bugs and request features
   - Community Forum: Ask questions and share solutions
   - Stack Overflow: Technical questions with `fxorcist` tag

3. **Professional Support**
   - Enterprise support: support@fxorcist.com
   - Training and consulting: training@fxorcist.com

### Diagnostic Information to Provide

When seeking help, include:

```bash
# System information
python fxorcist_cli.py --command "system info"

# Configuration dump
python fxorcist_cli.py --command "config dump"

# Recent logs
python fxorcist_cli.py --command "logs recent --lines 100"

# Error traceback
python fxorcist_cli.py --command "errors traceback"
```

---

*Troubleshooting Guide Version: 2.0*
*Last Updated: September 2, 2025*