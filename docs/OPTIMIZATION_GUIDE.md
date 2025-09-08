# Hyperparameter Optimization Guide

## Overview

FXorcist provides a powerful, flexible hyperparameter optimization framework built on Optuna, enabling systematic strategy improvement through intelligent parameter tuning.

## Key Features

- **Advanced Optimization**: Leverages Optuna's state-of-the-art hyperparameter search algorithms
- **Reproducible Trials**: Consistent random seeding for reliable results
- **Flexible Strategy Support**: Works with custom and predefined trading strategies
- **Rich Logging**: Comprehensive trial tracking and performance metrics

## Basic Usage

```python
from fxorcist.optimize import run_optuna
from fxorcist.config import load_config

# Load configuration
config = load_config('config.yaml')

# Run optimization
results = run_optuna(
    strategy_name='rsi',
    config=config,
    n_trials=100
)

print(results['best_params'])
```

## Advanced Configuration

### Custom Strategy Factories

```python
def custom_strategy_factory(trial):
    # Create strategy with Optuna-suggested parameters
    return MyCustomStrategy(
        lower_threshold=trial.suggest_int('lower', 20, 40),
        upper_threshold=trial.suggest_int('upper', 60, 80)
    )

results = run_optuna(
    strategy_name='custom',
    config=config,
    strategy_factory=custom_strategy_factory
)
```

### Persistent Storage

```python
results = run_optuna(
    strategy_name='macd',
    config=config,
    storage='sqlite:///optimization_study.db',
    study_name='macd_optimization'
)
```

## Optimization Strategies

### Pruning

FXorcist uses Optuna's MedianPruner to efficiently explore the parameter space:

- Automatically stops unpromising trials
- Reduces computational overhead
- Focuses on most promising parameter combinations

### Metrics Optimization

The optimization objective is typically the negative Sharpe ratio, allowing Optuna to maximize strategy performance.

## Best Practices

1. Start with a reasonable parameter range
2. Use domain knowledge to constrain search space
3. Monitor trial logs for insights
4. Validate results with out-of-sample testing

## Troubleshooting

- **Invalid Parameters**: Ensure parameter constraints make sense
- **Performance Issues**: Adjust trial count and pruning settings
- **Reproducibility**: Use consistent random seeds

## Extending the Framework

- Implement custom objective functions
- Create strategy-specific parameter suggestion methods
- Integrate with MLflow for experiment tracking

## Example: Multi-Strategy Optimization

```python
strategies = ['rsi', 'macd', 'bollinger']
for strategy in strategies:
    results = run_optuna(
        strategy_name=strategy,
        config=config,
        n_trials=50
    )
    print(f"{strategy} Best Params: {results['best_params']}")
```

## Performance Considerations

- Larger `n_trials` increases optimization quality
- Use parallel execution for faster results
- Monitor memory usage with large event datasets

## Visualization

Optuna provides built-in visualization tools:

```python
import optuna
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_parallel_coordinate(study)
```

## Logging and Tracking

Comprehensive logging captures:
- Trial parameters
- Performance metrics
- Execution timestamps
- Trial states (completed, pruned)

## Contributing

Help improve FXorcist's optimization framework:
- Report issues
- Suggest parameter search strategies
- Contribute strategy-specific optimization techniques