import optuna
from optuna.samplers import TPESampler
import yaml
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from ..pipeline.vectorized_backtest import sma_strategy_returns, simple_metrics

logger = logging.getLogger(__name__)

def run_optuna(
    df,
    n_trials: int = 50,
    seed: int = 42,
    out_path: Optional[str] = "artifacts/best_params.yaml",
    use_mlflow: bool = False
) -> Tuple[optuna.Study, Dict[str, Any]]:
    """
    Run Optuna optimization for SMA strategy parameters with optional MLflow tracking.
    
    Args:
        df: DataFrame with OHLC data
        n_trials: Number of optimization trials
        seed: Random seed for reproducibility
        out_path: Path to save best parameters (None to skip saving)
        use_mlflow: Whether to log results to MLflow
        
    Returns:
        Tuple of (optuna study object, dict of best parameters)
    """
    # Use seeded sampler for reproducibility
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    def objective(trial):
        # Suggest parameters with constraints
        fast = trial.suggest_int("fast", 5, 20)
        slow = trial.suggest_int("slow", 21, 200)
        
        # Invalid parameter combination
        if slow <= fast:
            return float("-inf")
            
        # Calculate strategy returns and metrics
        returns = sma_strategy_returns(df, fast=fast, slow=slow)
        metrics = simple_metrics(returns)
        return metrics.get("sharpe", float("-inf"))
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    
    # Save parameters if path provided
    if out_path:
        try:
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(yaml.safe_dump(best))
        except Exception as e:
            logger.warning(f"Failed to save parameters to {out_path}: {e}")
    
    # Optional MLflow logging
    if use_mlflow:
        try:
            import mlflow
            with mlflow.start_run(run_name=f"optuna_sma_{seed}"):
                mlflow.log_params(best)
                mlflow.log_metric("best_sharpe", study.best_value)
        except ImportError:
            logger.warning("MLflow not installed, skipping logging")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
    
    return study, best