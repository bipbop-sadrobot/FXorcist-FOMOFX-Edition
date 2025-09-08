import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from fxorcist.rl.env import TradingEnv

def env_creator(config):
    """Create environment for Ray."""
    return TradingEnv(config)

def train_rl(
    symbol: str = "EURUSD", 
    initial_capital: float = 10000, 
    max_steps: int = 1000, 
    iterations: int = 100
):
    """Train RL agent using Ray RLlib."""
    # Ensure results directory exists
    os.makedirs("./ray_results", exist_ok=True)
    os.makedirs("./models/rl_agent", exist_ok=True)

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Register environment
    env_name = "TradingEnv-v0"
    register_env(env_name, env_creator)

    # Configure PPO algorithm
    config = (
        PPOConfig()
        .environment(
            env=env_name, 
            env_config={
                "symbol": symbol,
                "initial_capital": initial_capital,
                "max_steps": max_steps
            }
        )
        .rollouts(num_rollout_workers=2, num_envs_per_worker=1)
        .resources(num_gpus=0)  # Set to 1 if GPU available
        .training(
            train_batch_size=4000,
            lr=5e-5,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01
        )
        .reporting(min_sample_batch_size=100)
    )

    # Create Tuner
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=tune.RunConfig(
            stop={"training_iteration": iterations},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=10,
                checkpoint_at_end=True
            ),
            storage_path="./ray_results"
        )
    )

    # Run training
    results = tuner.fit()

    # Find best result
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    print(f"Best reward: {best_result.metrics.get('episode_reward_mean', 'N/A')}")

    # Save the best model
    if best_result.checkpoint:
        # Copy checkpoint to models directory
        import shutil
        shutil.copytree(
            best_result.checkpoint.path, 
            "./models/rl_agent", 
            dirs_exist_ok=True
        )
        print(f"Best model saved to ./models/rl_agent")

    # Shutdown Ray
    ray.shutdown()

    return best_result

if __name__ == "__main__":
    train_rl()