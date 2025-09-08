# FXorcist — Event-Driven Forex Research Platform (v3)

## 🚀 Key Features

### 🔍 Advanced Backtesting
- **Event-Driven Architecture**: Precise market simulation with chronological event processing
- **Anti-Bias Protection**: Prevents look-ahead bias in strategy evaluation
- **Flexible Strategy Integration**: Easy to develop and test custom trading strategies
- **Immutable Event Bus**: Robust event handling with advanced filtering capabilities

### 💻 Modern CLI
- **Rich Interactive Interface**: Powered by Typer and Rich
- **Comprehensive Commands**: Prepare data, run backtests, optimize strategies
- **Type-Hinted Configuration**: Automatic validation and help generation
- **Progress Tracking**: Real-time feedback during long-running operations

### 🤖 Hyperparameter Optimization
- **Optuna-Powered Search**: Intelligent parameter tuning
- **Reproducible Trials**: Consistent random seeding
- **Advanced Pruning**: Efficient exploration of parameter spaces
- **Flexible Strategy Support**: Custom strategy parameter optimization

## 📊 Technical Highlights

- Comprehensive Forex market data handling
- Modular, extensible event-driven architecture
- Advanced performance metrics calculation
- Multi-strategy support with strategy registry
- Configurable execution models (slippage, commissions)

## 🛠 Core Improvements

- Pydantic-based configuration management
- Parallel processing capabilities
- Enhanced backtesting with transaction cost models
- MLflow experiment tracking integration
- Comprehensive logging and error handling

## Quick Start

```bash
# Install FXorcist
pip install fxorcist

# Prepare market data
fxorcist prepare --symbol EURUSD

# Run a backtest
fxorcist backtest --strategy rsi --symbol EURUSD

# Optimize strategy parameters
fxorcist optimize --strategy macd --trials 100
```

## Documentation

- [Optimization Guide](/docs/OPTIMIZATION_GUIDE.md)
- [Backtest Pipeline Overview](/docs/BACKTEST_PIPELINE.md)
- [CLI Usage Guide](/docs/CLI_GUIDE.md)
- [Development Roadmap](/docs/ROADMAP.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.