# Forex AI Dashboard

A comprehensive monitoring and analysis dashboard for the Forex AI trading system.

## Architecture

The dashboard implements a modular component architecture for maintainability and extensibility:

### Core Components

- **Predictions Visualization**: Price predictions, feature importance, and confidence intervals
- **Performance Metrics**: PnL analysis, drawdowns, and risk metrics
- **System Monitor**: Resource usage and health monitoring

### Data Management

- Efficient data loading with caching
- Automatic validation and cleaning
- Support for multiple timeframes
- Real-time updates

## Running the Dashboard

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the dashboard:
```bash
cd dashboard
streamlit run app.py
```

3. Access at http://localhost:8501

## Component Structure

### Base Components

All components inherit from base classes in `components/__init__.py`:
- `DashboardComponent`: Base interface for all components
- `VisualizationComponent`: For plot-based components
- `MetricsComponent`: For metric displays
- `PerformanceComponent`: For trading analysis
- `SystemStatusComponent`: For system monitoring

### Configuration

Components use `ComponentConfig` for consistent configuration:
```python
config = ComponentConfig(
    title="Component Title",
    description="Component description",
    height=400,
    cache_ttl=300  # Cache timeout in seconds
)
```

## Data Loading

The `DataLoader` utility (`utils/data_loader.py`) provides:
- Efficient data access with caching
- Data validation and cleaning
- Support for multiple data formats
- Automatic updates

### Usage Example:
```python
loader = DataLoader()
df, issues = loader.load_forex_data(timeframe="1H")
results, issues = loader.load_evaluation_results()
```

## Adding New Components

1. Create a new component class inheriting from appropriate base
2. Implement required methods:
   - `render()`: Display component
   - `update(data)`: Update component data
   - `create_figure()`: For visualization components

Example:
```python
class NewComponent(VisualizationComponent):
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        
    def render(self) -> None:
        # Implement rendering logic
        
    def update(self, data: Dict) -> None:
        # Handle data updates
```

## Performance Optimization

The dashboard implements several optimization strategies:

1. **Data Caching**:
   - In-memory caching with TTL
   - Efficient data loading with parquet
   - Selective updates

2. **Resource Management**:
   - Automatic resource monitoring
   - Cache clearing on memory pressure
   - Efficient data structures

3. **Update Strategies**:
   - Configurable refresh intervals
   - Smart update triggers
   - Background data loading

## Error Handling

The dashboard implements comprehensive error handling:

1. **Data Validation**:
   - Missing value detection
   - Anomaly detection
   - Data quality metrics

2. **Component Errors**:
   - Graceful degradation
   - User feedback
   - Error logging

3. **System Monitoring**:
   - Resource usage alerts
   - Pipeline health checks
   - Performance monitoring

## Extending the System

### Adding Features

1. Create new component in `components/`
2. Add to main app tabs
3. Implement data loading if needed
4. Update documentation

### Customizing Visualizations

1. Override `create_figure()` in component
2. Use Plotly for consistent styling
3. Implement interactive features
4. Add control parameters

## Best Practices

1. **Code Organization**:
   - Follow component structure
   - Use type hints
   - Document public interfaces
   - Maintain test coverage

2. **Performance**:
   - Use caching appropriately
   - Optimize data loading
   - Monitor resource usage
   - Profile critical paths

3. **User Experience**:
   - Provide clear feedback
   - Implement loading states
   - Handle errors gracefully
   - Maintain responsive UI

## Troubleshooting

Common issues and solutions:

1. **Data Loading Errors**:
   - Check file permissions
   - Verify data format
   - Check network connectivity
   - Review validation logs

2. **Performance Issues**:
   - Clear cache
   - Adjust refresh intervals
   - Monitor resource usage
   - Check data volume

3. **Visualization Errors**:
   - Verify data format
   - Check component config
   - Review error logs
   - Test in isolation

## Contributing

1. Follow coding standards
2. Add tests for new features
3. Update documentation
4. Submit pull request

## License

MIT License - See LICENSE file for details