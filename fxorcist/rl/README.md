# FXorcist Reinforcement Learning Module

## Overview

This module implements a Reinforcement Learning (RL) trading environment and training pipeline using Ray RLlib. The goal is to create an adaptive trading agent that can learn optimal trading policies through trial and error.

## Key Components

- `env.py`: Gymnasium-compatible trading environment
- `train.py`: Ray RLlib training script for RL agents

## Environment Specification

### State Space
- Price
- RSI (Relative Strength Index)
- Portfolio Value
- Current Position Size

### Action Space
- 0: Sell
- 1: Hold
- 2: Buy

### Reward Function
The reward is calculated based on:
- Portfolio value change
- Small penalty for position changes to discourage unnecessary trading

## Training

To train an RL agent:

```bash
fxorcist train-rl --symbol EURUSD --initial-capital 10000 --max-steps 1000 --iterations 100
```

## Key Considerations

- The environment is a simplified simulation
- Real-world trading requires more complex state representations
- Reward function can be customized based on specific trading objectives

## Future Improvements

- Multi-asset training
- More sophisticated state representation
- Advanced reward functions
- Integration with live market data