import typer
from rich.console import Console
from typing import Optional

from fxorcist.config import load_config
from fxorcist.ml.optuna_runner import run_optuna

app = typer.Typer()
console = Console()

@app.command()
def optimize(
    algo: str = typer.Option("optuna", help="Optimization algorithm"),
    n_trials: int = typer.Option(100, help="Number of trials"),
    distributed: bool = typer.Option(False, help="Run trials in parallel with Dask"),
    workers: int = typer.Option(4, help="Number of workers for distributed optimization")
):
    """Run parameter optimization."""
    cfg = load_config()
    
    console.log(f"[bold magenta]Optimizing with {algo}, {n_trials} trials (distributed={distributed})[/bold magenta]")
    
    study = run_optuna(
        config=cfg, 
        n_trials=n_trials, 
        distributed=distributed,
        n_workers=workers
    )
    
    console.log(f"Best value: {-study.best_value:.4f}")
    console.log(f"Best params: {study.best_params}")

if __name__ == "__main__":
    app()