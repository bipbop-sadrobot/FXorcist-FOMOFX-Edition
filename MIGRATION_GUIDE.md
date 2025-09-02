# Migration Guide: Unified Forex AI System

This guide helps you transition to the unified versions of the training pipeline and dashboard.

## Overview

We have merged multiple versions of our components into unified, feature-complete implementations:

1. Training Pipeline: `unified_training_pipeline.py`
   - Combines features from automated, comprehensive, and focused versions
   - Includes all advanced features while maintaining efficiency
   - Preserves all existing functionality

2. Dashboard: `dashboard/unified_app.py`
   - Merges features from both dashboard versions
   - Enhanced performance with comprehensive analytics
   - Improved user interface and experience

## Key Features Preserved

### Training Pipeline

- **Distributed Training**
  - GPU support
  - Resource management
  - Parallel processing

- **Advanced ML Features**
  - Hyperparameter optimization
  - Model interpretability
  - Ensemble methods
  - Feature engineering

- **Optimizations**
  - Forex-specific parameters
  - Efficient data handling
  - Resource monitoring

### Dashboard

- **Enhanced Data Loading**
  - Preloading mechanism
  - Advanced caching
  - Optimized refresh strategies

- **Comprehensive Analytics**
  - QuantStats portfolio analysis
  - Advanced risk metrics
  - Performance tracking

- **Improved UI**
  - All visualization components
  - Performance statistics
  - Enhanced layout and navigation

## Migration Steps

1. Update Your Imports
   ```python
   # Old imports
   from automated_training_pipeline import AutomatedTrainingPipeline
   from comprehensive_training_pipeline import ComprehensiveTrainingPipeline
   
   # New import
   from unified_training_pipeline import UnifiedTrainingPipeline
   ```

2. Update Dashboard Usage
   ```python
   # Old usage
   from dashboard.app import DashboardApp
   # or
   from dashboard.enhanced_app import EnhancedDashboardApp
   
   # New usage
   from dashboard.unified_app import UnifiedDashboardApp
   ```

3. Update Configuration
   - All existing configuration options are supported
   - New options are available but optional
   - Previous settings will work without modification

## New Features

### Training Pipeline

1. Unified Configuration
   ```python
   pipeline = UnifiedTrainingPipeline()
   
   # Run with all features
   await pipeline.run_unified_pipeline(
       comprehensive_data=True,
       optimize_hyperparams=True,
       use_ensemble=True,
       enable_interpretability=True
   )
   
   # Or use focused mode
   await pipeline.run_unified_pipeline(
       comprehensive_data=False,
       optimize_hyperparams=False
   )
   ```

2. Enhanced Monitoring
   ```python
   # Access comprehensive metrics
   metrics = pipeline.metrics
   
   # Get specific performance aspects
   training_efficiency = metrics['training_throughput']
   resource_usage = metrics['peak_memory_usage']
   ```

### Dashboard

1. Advanced Analytics
   ```python
   dashboard = UnifiedDashboardApp()
   
   # Access enhanced features
   stats = dashboard.data_loader.get_performance_stats()
   risk_metrics = dashboard.data_loader.calculate_risk_metrics(returns)
   ```

2. Customization
   ```python
   # Configure components
   dashboard.predictions.update_config(
       height=800,
       show_confidence=True
   )
   
   # Enable advanced features
   dashboard.run(
       enable_performance_stats=True,
       advanced_analytics=True
   )
   ```

## Best Practices

1. **Data Management**
   - Use the unified data loading mechanisms
   - Take advantage of the preloading feature
   - Configure caching based on your needs

2. **Resource Optimization**
   - Monitor resource usage through the dashboard
   - Adjust batch sizes and worker counts as needed
   - Use GPU acceleration when available

3. **Performance Monitoring**
   - Check the system monitor tab regularly
   - Review pipeline metrics after training
   - Use the performance stats feature

## Support

The unified versions maintain backward compatibility while providing new features. If you encounter any issues during migration, please:

1. Check the logs for detailed error messages
2. Review the configuration settings
3. Ensure all dependencies are up to date
4. Contact support if issues persist

## Deprecation Notice

The following files are now deprecated and will be removed in future versions:

- `automated_training_pipeline.py`
- `comprehensive_training_pipeline.py`
- `focused_training_pipeline.py`
- `dashboard/app.py`
- `dashboard/enhanced_app.py`

Please migrate to the unified versions as soon as possible.