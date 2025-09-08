# FXorcist — Event-Driven Forex Research Platform (v3)

## 🚀 Key Features

### 🔍 Advanced Backtesting
- **Event-Driven Architecture**: Precise market simulation with chronological event processing
- **Anti-Bias Protection**: Prevents look-ahead bias in strategy evaluation
- **Flexible Strategy Integration**: Easy to develop and test custom trading strategies

### 💻 Modern CLI
- **Rich Interactive Interface**: Powered by Typer and Rich
- **Comprehensive Commands**: Prepare data, run backtests, optimize strategies
- **Type-Hinted Configuration**: Automatic validation and help generation

### 🤖 Machine Learning
- Integrated optimization with Optuna
- Strategy parameter tuning
- Advanced performance metric analysis

## 📊 Highlights

- Comprehensive Forex market data handling
- Modular, extensible architecture
- Advanced performance metrics calculation
- Supports multiple trading strategies

## 🛠 Core Improvements

- Improved data loader with schema validation
- Parallel processing capabilities
- Enhanced backtesting with slippage and transaction cost models
- MLflow integration for experiment tracking

## Quick Start

```bash
# Install FXorcist
pip install fxorcist

# Prepare market data
fxorcist prepare --symbol EURUSD

# Run a backtest
fxorcist backtest --strategy rsi --symbol EURUSD
```

## Documentation

- [Backtest Pipeline Overview](/docs/BACKTEST_PIPELINE.md)
- [CLI Usage Guide](/docs/CLI_GUIDE.md)
- [Development Roadmap](/docs/ROADMAP.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.