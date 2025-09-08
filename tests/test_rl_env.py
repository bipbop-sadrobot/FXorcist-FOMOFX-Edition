import numpy as np
import gymnasium as gym
import pytest
from fxorcist.rl.env import TradingEnv

def test_rl_env_creation():
    """Test environment creation with various configurations."""
    configs = [
        {"symbol": "EURUSD", "initial_capital": 10000, "max_steps": 100},
        {"symbol": "GBPUSD", "initial_capital": 5000, "max_steps": 500},
    ]
    
    for config in configs:
        env = TradingEnv(config)
        
        assert isinstance(env, gym.Env)
        assert env.symbol == config["symbol"]
        assert env.initial_capital == config["initial_capital"]
        assert env.max_steps == config["max_steps"]

def test_rl_env_observation_space():
    """Test observation space characteristics."""
    env = TradingEnv({"symbol": "EURUSD"})
    
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (10,)
    assert env.observation_space.dtype == np.float32

def test_rl_env_action_space():
    """Test action space characteristics."""
    env = TradingEnv({"symbol": "EURUSD"})
    
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 5  # Strong Sell, Sell, Hold, Buy, Strong Buy

def test_rl_env_reset():
    """Comprehensive reset method test."""
    env = TradingEnv({"symbol": "EURUSD", "initial_capital": 10000, "max_steps": 100})
    
    obs, info = env.reset()
    
    # Check observation
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (10,)
    assert obs.dtype == np.float32
    
    # Check initial state values
    assert env.portfolio_value == 10000
    assert env.position_size == 0
    assert env.current_step == 0
    assert len(env.trade_history) == 0

def test_rl_env_step():
    """Comprehensive step method test."""
    env = TradingEnv({"symbol": "EURUSD", "initial_capital": 10000, "max_steps": 100})
    
    obs, _ = env.reset()
    
    # Test multiple steps with different actions
    for action in range(5):  # Test all actions
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Observation checks
        assert isinstance(next_obs, np.ndarray)
        assert next_obs.shape == (10,)
        assert np.isfinite(next_obs).all()
        
        # Reward checks
        assert isinstance(reward, float)
        assert np.isfinite(reward)
        
        # Termination checks
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        
        # Info checks
        assert isinstance(info, dict)

def test_rl_env_episode_termination():
    """Test environment episode termination."""
    max_steps = 10
    env = TradingEnv({
        "symbol": "EURUSD",
        "initial_capital": 10000,
        "max_steps": max_steps
    })
    
    obs, _ = env.reset()
    
    # Run through all steps
    for _ in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(2)  # Hold action
        assert not terminated
    
    # Next step should terminate
    obs, reward, terminated, truncated, info = env.step(2)
    assert terminated

def test_rl_env_reward_calculation():
    """Test reward calculation logic."""
    env = TradingEnv({"symbol": "EURUSD", "initial_capital": 10000})
    
    obs, _ = env.reset()
    
    # Test multiple actions and check reward characteristics
    for action in range(5):
        _, reward, _, _, _ = env.step(action)
        
        # Reward should be a float
        assert isinstance(reward, float)
        
        # Reward should be reasonable (not extreme)
        assert -10 < reward < 10, f"Unreasonable reward for action {action}: {reward}"

def test_rl_env_market_data():
    """Test market data generation and loading."""
    # Test with mock data generation
    env = TradingEnv({"symbol": "EURUSD"})
    assert hasattr(env, 'market_data')
    assert len(env.market_data) >= env.max_steps

    # Optional: Test with custom data path if you have a test CSV
    # Uncomment and modify path as needed
    # env_with_data = TradingEnv({
    #     "symbol": "EURUSD", 
    #     "data_path": "path/to/test/market_data.csv"
    # })
    # assert len(env_with_data.market_data) > 0