import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple

class TradingEnv(gym.Env):
    """Gymnasium environment for RL trading agent."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.symbol = config.get("symbol", "EURUSD")
        self.initial_capital = config.get("initial_capital", 10000)
        self.max_steps = config.get("max_steps", 1000)

        # State: [price, rsi, portfolio_value, position_size]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        # Action: [position_change] (-1=sell, 0=hold, 1=buy) — discrete
        self.action_space = gym.spaces.Discrete(3)

        self.current_step = 0
        self.state = None
        self.portfolio_value = self.initial_capital
        self.position_size = 0

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.position_size = 0
        
        # Mock initial state generation
        self.state = self._generate_initial_state()
        return self.state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action, return (next_state, reward, terminated, truncated, info)."""
        # Map action to position change
        position_map = {0: -1, 1: 0, 2: 1}  # sell, hold, buy
        position_change = position_map[action]

        # Update position and simulate price movement
        self._update_position(position_change)
        
        # Advance step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        
        # Generate next state
        self.state = self._generate_next_state()
        
        # Calculate reward
        reward = self._calculate_reward(position_change)
        
        return self.state, reward, terminated, False, {}

    def _generate_initial_state(self) -> np.ndarray:
        """Generate initial state with some randomness."""
        return np.array([
            1.0,  # initial price
            50.0,  # initial RSI
            self.initial_capital,  # initial portfolio value
            0.0  # initial position size
        ], dtype=np.float32)

    def _generate_next_state(self) -> np.ndarray:
        """Generate next state with simulated price and portfolio changes."""
        # Simulate price movement
        price = self.state[0] * (1 + np.random.normal(0, 0.01))
        
        # Simulate RSI (simplified)
        rsi = np.clip(self.state[1] + np.random.normal(0, 5), 0, 100)
        
        # Update portfolio value based on price movement and position
        portfolio_value = self.portfolio_value * (1 + (price - self.state[0]) / self.state[0] * self.position_size)
        
        return np.array([
            price,
            rsi,
            portfolio_value,
            self.position_size
        ], dtype=np.float32)

    def _update_position(self, position_change: int):
        """Update position based on action."""
        # Simplified position sizing (1 unit per action)
        self.position_size += position_change
        
        # Simulate transaction costs or slippage
        transaction_cost = 0.001  # 0.1% transaction cost
        self.portfolio_value *= (1 - transaction_cost * abs(position_change))

    def _calculate_reward(self, position_change: int) -> float:
        """Calculate reward based on portfolio performance."""
        # Reward is the change in portfolio value
        reward = (self.state[2] - self.portfolio_value) / self.initial_capital
        
        # Add a small penalty for changing positions to discourage unnecessary trading
        position_change_penalty = abs(position_change) * 0.001
        
        return reward - position_change_penalty

    def render(self):
        """Optional render method for visualization."""
        print(f"Step: {self.current_step}")
        print(f"Portfolio Value: {self.portfolio_value}")
        print(f"Position Size: {self.position_size}")
        print(f"State: {self.state}")