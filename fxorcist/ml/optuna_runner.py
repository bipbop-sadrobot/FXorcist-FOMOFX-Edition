"""
Optuna runner:
- TPESampler(seed)
- MedianPruner for speed
- Optional MLflow logging (safe if mlflow not installed)
- Save trial artifacts (best params yaml, equity plot) to artifacts/
- Support multi-objective placeholder (can be expanded)
"""
from __future__ import annotations
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import matplotlib.pyplot as plt
import io
import base64

from fxorcist.pipeline.vectorized_backtest import sma_strategy_returns, simple_metrics

logger = logging.getLogger(__name__)

def _objective(trial: optuna.Trial, df):
    fast = trial.suggest_int("fast", 5, 40)
    slow = trial.suggest_int("slow", 50, 200)
    if slow <= fast:
        return -1e9
    rets = sma_strategy_returns(df, fast=fast, slow=slow)
    metrics = simple_metrics(rets)
    trial.set_user_attr("n", len(rets))
    return metrics.get("sharpe", -1e9)

def run_optuna(df, n_trials: int = 50, seed: int = 42, out_path: str = "artifacts/best_params.yaml", storage: Optional[str] = None, use_mlflow: bool = False) -> Dict[str, Any]:
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, storage=storage, load_if_exists=True)
    mlflow = None
    if use_mlflow:
        try:
            import mlflow as _ml
            mlflow = _ml
            mlflow.start_run(run_name=f"optuna_sma_{seed}")
        except Exception as e:
            logger.warning("MLflow import failed; continuing without MLflow: %s", e)
            mlflow = None

    study.optimize(lambda t: _objective(t, df), n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        yaml.safe_dump(best, fh)
    if mlflow:
        try:
            mlflow.log_params(best)
            mlflow.log_metric("best_sharpe", study.best_value)
            mlflow.end_run()
        except Exception as e:
            logger.warning("MLflow logging failed: %s", e)

    try:
        best_rets = sma_strategy_returns(df, fast=best['fast'], slow=best['slow'])
        fig, ax = plt.subplots()
        (1 + best_rets).cumprod().plot(ax=ax)
        ax.set_title("Equity curve (best)")
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        out_file = Path(out_path).with_suffix('.png')
        out_file.write_bytes(buf.getvalue())
    except Exception as e:
        logger.warning("Failed to write equity plot: %s", e)

    return {"study": study, "best_params": best}
