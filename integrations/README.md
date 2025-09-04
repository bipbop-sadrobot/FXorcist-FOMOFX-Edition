# FXorcist ML Integration Components

Advanced machine learning components for FXorcist-FOMOFX-Edition, providing causal inference, robust testing, and enhanced classification capabilities.

## Components

### 1. Causal Inference (econml_effects.py)
Estimates the causal impact of market events on price movements using EconML's double machine learning approach.

```python
from fxorcist_integration.causal.econml_effects import analyze_market_event

results = analyze_market_event(
    df,
    event_col='news_event',
    outcome_col='close',
    horizon=1
)
print(f"Average Treatment Effect: {results['summary']['ate']}")
```

### 2. Decomposed Tests (decomposed_tests.py)
Validates models under various market conditions using component-wise classifiers and perturbation analysis.

```python
from fxorcist_integration.tests.decomposed_tests import run_decomposed_analysis

results = run_decomposed_analysis(df)
print("Performance across regimes:")
print(results.groupby('regime')['auc'].mean())
```

### 3. Model Zoo (model_zoo.py)
Collection of classification models optimized for forex prediction, including Random Forests and Gradient Boosting.

```python
from fxorcist_integration.models.model_zoo import ModelZoo

zoo = ModelZoo()
results = zoo.fit(df, label_col='label')
print(f"Best model: {results['best_model']}")
print(f"AUC: {results['auc']:.4f}")
```

## Quick Start

1. Install dependencies:
```bash
pip install -r integrations/requirements.txt
```

2. Run the complete integration example:
```bash
python integrations/scripts/run_integration.py --data example
```

This will:
- Create example forex data
- Run causal analysis
- Perform decomposed testing
- Train and evaluate models
- Save results to ./results/

## Integration with FXorcist

### Vector Adapters
The components work directly with FXorcist's vector adapters:

```python
# In your consolidation worker
from fxorcist_integration.models.model_zoo import ModelZoo

def process_vectors(vectors):
    zoo = ModelZoo.load('path/to/model.joblib')
    predictions = zoo.predict(vectors)
    return predictions
```

### Eventbus Integration
For real-time causal analysis:

```python
# Subscribe to market events
from fxorcist_integration.causal.econml_effects import analyze_market_event

def on_market_event(event_data):
    effects = analyze_market_event(
        event_data,
        event_col='event_type',
        outcome_col='price'
    )
    publish_results(effects['summary'])
```

### Validation Pipeline
Add decomposed testing to your validation:

```python
from fxorcist_integration.tests.decomposed_tests import run_decomposed_analysis

def validate_model(model, validation_data):
    results = run_decomposed_analysis(validation_data)
    if results['high_vol']['auc'].mean() < 0.55:
        raise ValidationError("Poor performance in high volatility")
```

## Data Format

Required columns for full functionality:

- timestamp: Datetime
- close: Float (price)
- feat_*: Float (technical indicators/features)
- event: Int (0/1 for event occurrence)
- signal: Int (-1/0/1 for trade signals)
- label: Int (0/1 for classification)

Example:
```python
df = pd.DataFrame({
    'timestamp': pd.date_range('2025-01-01', periods=1000, freq='H'),
    'close': [...],
    'feat_rsi': [...],
    'feat_ma': [...],
    'event': [...],
    'signal': [...],
    'label': [...]
})
```

## Advanced Usage

### Custom Market Regimes
Define custom market regimes for decomposed testing:

```python
from fxorcist_integration.tests.decomposed_tests import MarketRegime

regimes = [
    MarketRegime('low_vol', (0.0, 0.001)),
    MarketRegime('high_vol', (0.001, float('inf')))
]
results = run_decomposed_analysis(df, regimes=regimes)
```

### Causal Effect Thresholds
Use treatment effects for position sizing:

```python
effects = analyze_market_event(df, event_col='news_impact')
significant_effects = effects['cate'] > 2 * effects['stderr']
position_sizes = np.where(significant_effects, 1.0, 0.5)
```

### Model Ensembling
Combine multiple models from the zoo:

```python
zoo = ModelZoo()
zoo.fit(df, label_col='label')
predictions = zoo.predict_proba(new_data)
```

## Troubleshooting

### Common Issues

1. EconML Import Errors
```
If econml fails to import, the system falls back to sklearn estimators.
Check your numpy and scipy versions match econml requirements.
```

2. Memory Usage
```
For large datasets, use batch processing:
df_generator = pd.read_csv('large_file.csv', chunksize=10000)
for chunk in df_generator:
    results = analyze_chunk(chunk)
```

3. Performance Issues
```
- Enable parallel processing in ModelZoo
- Use smaller CV splits for faster iteration
- Profile memory usage with decomposed tests
```

### Best Practices

1. Data Quality
- Remove outliers before analysis
- Handle missing values appropriately
- Normalize features for better convergence

2. Model Selection
- Start with simpler models (LogisticRegression)
- Move to ensemble methods if needed
- Use time-series CV for proper validation

3. Production Deployment
- Save models with joblib for consistency
- Monitor treatment effects over time
- Implement fallback strategies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

Same as FXorcist-FOMOFX-Edition main license.