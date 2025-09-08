import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

class TradingEnv(gym.Env):
    """Advanced Gymnasium environment for adaptive trading agent."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        # Configuration parameters
        self.symbol = config.get("symbol", "EURUSD")
        self.initial_capital = config.get("initial_capital", 10000)
        self.max_steps = config.get("max_steps", 1000)
        self.risk_tolerance = config.get("risk_tolerance", 0.02)  # 2% risk per trade
        
        # Advanced state space
        self._load_market_data(config.get("data_path"))
        
        # State includes: 
        # 1-5: Technical indicators
        # 6-8: Portfolio metrics
        # 9-10: Market context
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        # Actions: 
        # 0: Strong Sell
        # 1: Sell
        # 2: Hold
        # 3: Buy
        # 4: Strong Buy
        self.action_space = gym.spaces.Discrete(5)

        # Trading parameters
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.position_size = 0
        self.trade_history = []

    def _load_market_data(self, data_path: Optional[str] = None):
        """Load and preprocess market data."""
        if data_path:
            try:
                self.market_data = pd.read_csv(data_path)
                # Compute technical indicators
                self.market_data['SMA_50'] = self.market_data['close'].rolling(window=50).mean()
                self.market_data['SMA_200'] = self.market_data['close'].rolling(window=200).mean()
                self.market_data['RSI'] = self._calculate_rsi(self.market_data['close'])
            except Exception as e:
                print(f"Warning: Could not load market data. Using mock data. Error: {e}")
                self._generate_mock_data()
        else:
            self._generate_mock_data()

    def _generate_mock_data(self):
        """Generate synthetic market data for testing."""
        np.random.seed(42)
        steps = self.max_steps
        close_prices = np.cumsum(np.random.normal(0, 0.01, steps)) + 1.0
        self.market_data = pd.DataFrame({
            'close': close_prices,
            'SMA_50': pd.Series(close_prices).rolling(window=50).mean(),
            'SMA_200': pd.Series(close_prices).rolling(window=200).mean(),
            'RSI': self._calculate_rsi(close_prices)
        })

    def _calculate_rsi(self, prices, periods=14):
        """Calculate Relative Strength Index."""
        delta = np.diff(prices)
        gain = (delta > 0) * delta
        loss = (delta < 0) * -delta

        avg_gain = pd.Series(gain).rolling(window=periods).mean()
        avg_loss = pd.Series(loss).rolling(window=periods).mean()

        relative_strength = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + relative_strength))
        return np.pad(rsi, (1, 0), mode='constant', constant_values=50)

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the trading environment."""
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.position_size = 0
        self.trade_history = []

        initial_state = self._get_state()
        return initial_state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute trading action and return next state."""
        # Map action to position change
        position_changes = {
            0: -2,  # Strong Sell
            1: -1,  # Sell
            2: 0,   # Hold
            3: 1,   # Buy
            4: 2    # Strong Buy
        }
        position_change = position_changes[action]

        # Update position and simulate execution
        current_price = self.market_data.loc[self.current_step, 'close']
        self._update_position(position_change, current_price)

        # Advance step
        self.current_step += 1
        terminated = self.current_step >= len(self.market_data) - 1

        # Get next state
        next_state = self._get_state()

        # Calculate reward
        reward = self._calculate_reward(current_price, position_change)

        return next_state, reward, terminated, False, {}

    def _update_position(self, position_change: int, current_price: float):
        """Update portfolio position based on action."""
        trade_size = self.portfolio_value * 0.1  # 10% of portfolio per trade
        new_position = position_change * trade_size / current_price

        # Transaction costs simulation
        transaction_cost_rate = 0.001  # 0.1% transaction cost
        transaction_cost = abs(new_position) * current_price * transaction_cost_rate

        # Update portfolio
        self.position_size += new_position
        self.portfolio_value -= transaction_cost

        # Record trade
        self.trade_history.append({
            'step': self.current_step,
            'position_change': position_change,
            'price': current_price,
            'portfolio_value': self.portfolio_value
        })

    def _get_state(self) -> np.ndarray:
        """Generate comprehensive state representation."""
        if self.current_step >= len(self.market_data):
            # Return last known state if out of data
            return np.zeros(10, dtype=np.float32)

        row = self.market_data.iloc[self.current_step]
        
        # Technical Indicators (normalized)
        close_price = row['close']
        sma_50 = row['SMA_50']
        sma_200 = row['SMA_200']
        rsi = row['RSI']
        price_sma_ratio = close_price / ((sma_50 + sma_200) / 2)

        # Portfolio Metrics
        portfolio_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        position_ratio = self.position_size * close_price / self.portfolio_value
        trade_frequency = len(self.trade_history) / (self.current_step + 1)

        # Market Context
        volatility = self.market_data['close'].iloc[max(0, self.current_step-50):self.current_step].std()
        trend_strength = abs(sma_50 - sma_200) / close_price

        return np.array([
            price_sma_ratio,  # Technical Indicator 1
            rsi / 100.0,      # Technical Indicator 2
            close_price,      # Technical Indicator 3
            sma_50,           # Technical Indicator 4
            sma_200,          # Technical Indicator 5
            portfolio_return, # Portfolio Metric 1
            position_ratio,   # Portfolio Metric 2
            trade_frequency,  # Portfolio Metric 3
            volatility,       # Market Context 1
            trend_strength    # Market Context 2
        ], dtype=np.float32)

    def _calculate_reward(self, current_price: float, position_change: int) -> float:
        """Advanced reward calculation considering multiple factors."""
        # Base reward: portfolio value change
        portfolio_return = (self.portfolio_value - self.initial_capital) / self.initial_capital

        # Penalize unnecessary trading
        trade_penalty = abs(position_change) * 0.001

        # Risk management: penalize large drawdowns
        max_drawdown_penalty = max(0, -portfolio_return * 10) if portfolio_return < 0 else 0

        # Trend alignment bonus/penalty
        row = self.market_data.iloc[self.current_step]
        trend_alignment_bonus = (1 if row['SMA_50'] > row['SMA_200'] and position_change > 0 else 
                                 -1 if row['SMA_50'] < row['SMA_200'] and position_change < 0 else 0) * 0.005

        return portfolio_return - trade_penalty - max_drawdown_penalty + trend_alignment_bonus

    def render(self, mode='human'):
        """Optional rendering method for environment visualization."""
        print(f"Step: {self.current_step}")
        print(f"Portfolio Value: {self.portfolio_value}")
        print(f"Position Size: {self.position_size}")
        print(f"Current State: {self._get_state()}")