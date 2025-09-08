# FXorcist Reinforcement Learning Module

## Overview

The FXorcist RL module provides an advanced, adaptive trading agent using Reinforcement Learning techniques. It supports multiple RL algorithms and offers a sophisticated trading environment with comprehensive state representation and reward calculation.

## Key Features

### Advanced Trading Environment
- 10-dimensional state space
- Technical indicators (SMA, RSI)
- Portfolio metrics
- Market context features
- Realistic transaction cost simulation

### Supported RL Algorithms
- Proximal Policy Optimization (PPO)
- Soft Actor-Critic (SAC)

### State Representation
The state includes:
1. Technical Indicators
   - Price to SMA ratio
   - Relative Strength Index (RSI)
   - Close price
   - Short-term Moving Average
   - Long-term Moving Average

2. Portfolio Metrics
   - Portfolio return
   - Position ratio
   - Trade frequency

3. Market Context
   - Volatility
   - Trend strength

### Action Space
- Strong Sell
- Sell
- Hold
- Buy
- Strong Buy

## Training Options

### CLI Training
```bash
# Basic training
fxorcist train-rl --symbol EURUSD

# Advanced configuration
fxorcist train-rl \
    --symbol EURUSD \
    --initial-capital 10000 \
    --max-steps 1000 \
    --iterations 100 \
    --algorithm ppo \
    --data-path /path/to/market/data.csv
```

### Programmatic Training
```python
from fxorcist.rl.train import train_rl

result = train_rl(
    symbol="EURUSD",
    initial_capital=10000,
    max_steps=1000,
    iterations=100,
    algorithm="ppo",
    data_path=None
)
```

## Reward Calculation

The reward function considers:
- Portfolio value change
- Trade frequency penalty
- Drawdown management
- Trend alignment bonus

## Customization

### Extending the Environment
- Modify `fxorcist/rl/env.py` to add custom indicators
- Adjust reward calculation in `_calculate_reward()`
- Add more sophisticated market data loading

## Performance Monitoring

- Tracks episode rewards
- Monitors trade performance
- Supports hyperparameter tuning via Population Based Training

## Future Roadmap
- Multi-asset training
- More advanced reward functions
- Enhanced market data integration
- Support for more RL algorithms

## Dependencies
- Ray RLlib
- Gymnasium
- NumPy
- Pandas

## Experimental Status
This module is experimental and under active development. Expect frequent updates and improvements.