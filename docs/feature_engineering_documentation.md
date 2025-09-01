# Feature Engineering Documentation

## Overview

The feature engineering pipeline provides comprehensive capabilities for generating, selecting, and managing features for forex trading. It includes automated feature selection, synthetic feature generation, market regime detection, and experiment tracking.

## Key Components

### 1. Feature Generation

The `FeatureGenerator` class manages the overall feature engineering process, including:

- Basic price features (returns, volatility)
- Technical indicators (RSI, Bollinger Bands)
- Volume-based features
- Market microstructure features (spreads, gaps)
- Synthetic features (Fourier, wavelet transforms)
- Market regime labels

### 2. Automated Feature Selection

Multiple methods are available for feature selection:

#### Boruta Algorithm
```python
selected_features, _ = generator.select_important_features(
    df, target_col='returns', methods=['boruta']
)
```

#### SHAP-based Selection
```python
selected_features, importance_df = generator.select_important_features(
    df, target_col='returns', methods=['shap']
)
```

### 3. Synthetic Feature Generation

Two types of transforms are supported:

#### Fourier Transform Features
- Captures frequency components
- Useful for identifying cycles
```python
df_features = generator.generate_synthetic_features(
    df, target_col='returns', methods=['fourier']
)
```

#### Wavelet Transform Features
- Multi-scale decomposition
- Captures both frequency and time information
```python
df_features = generator.generate_synthetic_features(
    df, target_col='returns', methods=['wavelet']
)
```

### 4. Market Regime Detection

Two clustering methods are available:

#### Hidden Markov Model (HMM)
```python
regimes, stats = generator.detect_market_regimes(
    df, n_regimes=3, method='hmm'
)
```

#### K-means Clustering
```python
regimes, stats = generator.detect_market_regimes(
    df, n_regimes=3, method='kmeans'
)
```

### 5. Feature Registry

The `FeatureRegistry` maintains metadata about features:
- Dependencies
- Version history
- Statistics
- Parameters

### 6. Experiment Tracking

MLflow integration provides:
- Parameter logging
- Metric tracking
- Nested runs for different components
- Feature importance visualization

## Usage Examples

### Complete Pipeline

```python
# Initialize generator
generator = FeatureGenerator(
    random_state=42,
    experiment_name="forex_feature_engineering"
)

# Generate base features
df_features = generator.generate_features(df)

# Detect market regimes
regimes, regime_stats = generator.detect_market_regimes(
    df_features,
    n_regimes=3,
    method='hmm'
)
df_features['market_regime'] = regimes

# Generate synthetic features
df_features = generator.generate_synthetic_features(
    df_features,
    target_col='returns',
    methods=['fourier', 'wavelet']
)

# Select important features
feature_cols = [col for col in df_features.columns if col != 'returns']
selected_features, importance_df = generator.select_important_features(
    df_features,
    target_col='returns',
    feature_cols=feature_cols,
    methods=['boruta', 'shap']
)
```

## Best Practices

1. **Feature Selection**
   - Use multiple methods (Boruta + SHAP)
   - Review feature importance scores
   - Consider domain knowledge

2. **Synthetic Features**
   - Start with basic transforms
   - Monitor for overfitting
   - Validate on out-of-sample data

3. **Market Regimes**
   - Test different numbers of regimes
   - Compare HMM vs k-means results
   - Validate regime transitions

4. **Experiment Tracking**
   - Use meaningful experiment names
   - Log all relevant parameters
   - Track feature importance over time

## Monitoring and Maintenance

1. **Feature Statistics**
   - Monitor for distribution shifts
   - Track missing values
   - Review feature correlations

2. **Regime Changes**
   - Monitor regime transition frequencies
   - Review regime characteristics
   - Validate regime persistence

3. **Performance Metrics**
   - Track feature importance stability
   - Monitor synthetic feature quality
   - Review selection consistency

## Future Enhancements

1. Additional feature selection methods
2. More synthetic feature types
3. Advanced regime detection algorithms
4. Enhanced visualization capabilities
5. Automated feature validation

## Troubleshooting

Common issues and solutions:

1. **Missing Data**
   - Check data preprocessing
   - Review feature dependencies
   - Validate data quality

2. **Feature Selection**
   - Verify input data quality
   - Check for multicollinearity
   - Review selection parameters

3. **Market Regimes**
   - Validate regime stability
   - Check feature preparation
   - Review model parameters

4. **Experiment Tracking**
   - Verify MLflow connection
   - Check logging permissions
   - Review run hierarchy