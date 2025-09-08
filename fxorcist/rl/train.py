import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.schedulers import PopulationBasedTraining

from fxorcist.rl.env import TradingEnv
from fxorcist.utils.logging import get_logger

logger = get_logger(__name__)

def env_creator(config):
    """Create environment for Ray."""
    return TradingEnv(config)

def train_rl(
    symbol: str = "EURUSD", 
    initial_capital: float = 10000, 
    max_steps: int = 1000, 
    iterations: int = 100,
    algorithm: str = "ppo",
    data_path: str = None
):
    """
    Advanced RL training with multiple algorithm support and configuration.
    
    Args:
        symbol (str): Trading symbol
        initial_capital (float): Starting capital for trading
        max_steps (int): Maximum steps per episode
        iterations (int): Number of training iterations
        algorithm (str): RL algorithm to use (ppo, sac)
        data_path (str, optional): Path to market data for training
    """
    # Ensure results and model directories exist
    os.makedirs("./ray_results", exist_ok=True)
    os.makedirs("./models/rl_agent", exist_ok=True)

    # Initialize Ray
    ray.init(ignore_reinit_error=True, logging_level="ERROR")

    # Register environment
    env_name = "TradingEnv-v0"
    register_env(env_name, env_creator)

    # Environment configuration
    env_config = {
        "symbol": symbol,
        "initial_capital": initial_capital,
        "max_steps": max_steps,
        "data_path": data_path
    }

    # Algorithm selection and configuration
    if algorithm.lower() == "sac":
        config = (
            SACConfig()
            .environment(env=env_name, env_config=env_config)
            .rollouts(num_rollout_workers=2, num_envs_per_worker=1)
            .resources(num_gpus=0)
            .training(
                target_entropy="auto",
                tau=0.005,
                train_batch_size=256,
                learning_rate=3e-4
            )
        )
    else:  # Default to PPO
        config = (
            PPOConfig()
            .environment(env=env_name, env_config=env_config)
            .rollouts(num_rollout_workers=2, num_envs_per_worker=1)
            .resources(num_gpus=0)
            .training(
                train_batch_size=4000,
                lr=5e-5,
                gamma=0.99,
                lambda_=0.95,
                clip_param=0.2,
                vf_clip_param=10.0,
                entropy_coeff=0.01
            )
        )

    # Population Based Training for hyperparameter optimization
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=10,
        hyperparam_mutations={
            "lr": [1e-3, 5e-4, 1e-4, 5e-5],
            "gamma": [0.9, 0.95, 0.99],
            "entropy_coeff": [0.001, 0.01, 0.1]
        }
    )

    # Tuner configuration
    tuner = tune.Tuner(
        algorithm.upper(),
        param_space=config.to_dict(),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=3,  # Number of parallel trials
        ),
        run_config=tune.RunConfig(
            stop={"training_iteration": iterations},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=10,
                checkpoint_at_end=True
            ),
            storage_path="./ray_results",
            verbose=1,
            progress_reporter=tune.CLIReporter(
                metric_columns={
                    "episode_reward_mean": "Avg Reward", 
                    "episode_len_mean": "Avg Episode Length"
                }
            )
        )
    )

    # Run training
    results = tuner.fit()

    # Find and save best result
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    
    if best_result:
        # Log training results
        logger.info(f"Best Training Result:")
        logger.info(f"Average Reward: {best_result.metrics.get('episode_reward_mean', 'N/A')}")
        logger.info(f"Average Episode Length: {best_result.metrics.get('episode_len_mean', 'N/A')}")

        # Save the best model
        if best_result.checkpoint:
            import shutil
            shutil.copytree(
                best_result.checkpoint.path, 
                "./models/rl_agent", 
                dirs_exist_ok=True
            )
            logger.info("Best model saved to ./models/rl_agent")
    else:
        logger.warning("No successful training results found.")

    # Shutdown Ray
    ray.shutdown()

    return best_result

if __name__ == "__main__":
    train_rl()