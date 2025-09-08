import numpy as np
import gymnasium as gym
from fxorcist.rl.env import TradingEnv

def test_rl_env_creation():
    """Test environment creation."""
    env = TradingEnv({
        "symbol": "EURUSD",
        "initial_capital": 10000,
        "max_steps": 100
    })
    
    assert isinstance(env, gym.Env)
    assert env.symbol == "EURUSD"
    assert env.initial_capital == 10000
    assert env.max_steps == 100

def test_rl_env_reset():
    """Test environment reset."""
    env = TradingEnv({
        "symbol": "EURUSD",
        "initial_capital": 10000,
        "max_steps": 100
    })
    
    obs, info = env.reset()
    
    # Check observation shape and type
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)
    assert obs.dtype == np.float32
    
    # Check initial state values
    assert obs[2] == 10000  # Initial portfolio value
    assert obs[3] == 0  # Initial position size

def test_rl_env_step():
    """Test environment step method."""
    env = TradingEnv({
        "symbol": "EURUSD",
        "initial_capital": 10000,
        "max_steps": 100
    })
    
    obs, _ = env.reset()
    
    # Test multiple steps with different actions
    for action in [0, 1, 2]:  # sell, hold, buy
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Check next observation
        assert isinstance(next_obs, np.ndarray)
        assert next_obs.shape == (4,)
        
        # Check reward
        assert isinstance(reward, float)
        
        # Check termination flags
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        
        # Check info
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
        obs, reward, terminated, truncated, info = env.step(1)  # hold action
        assert not terminated
    
    # Next step should terminate
    obs, reward, terminated, truncated, info = env.step(1)
    assert terminated