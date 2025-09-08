import typer
from rich.console import Console
from rich.table import Table
from typing import Optional

from fxorcist.config import load_config
from fxorcist.ml.optuna_runner import run_optuna, analyze_optimization_results

app = typer.Typer()
console = Console()

@app.command()
def optimize(
    algo: str = typer.Option("optuna", help="Optimization algorithm"),
    n_trials: int = typer.Option(100, help="Number of trials"),
    distributed: bool = typer.Option(False, help="Run trials in parallel with Dask"),
    workers: int = typer.Option(4, help="Number of workers for distributed optimization"),
    analyze: bool = typer.Option(False, help="Generate detailed optimization analysis")
):
    """Run parameter optimization with MLflow tracking."""
    cfg = load_config()
    
    console.log(f"[bold magenta]Optimizing with {algo}, {n_trials} trials (distributed={distributed})[/bold magenta]")
    
    # Run optimization
    study = run_optuna(
        config=cfg, 
        n_trials=n_trials, 
        distributed=distributed,
        n_workers=workers
    )
    
    # Display basic results
    console.log(f"Best value: {-study.best_value:.4f}")
    console.log(f"Best params: {study.best_params}")
    
    # Optional detailed analysis
    if analyze:
        results = analyze_optimization_results(study)
        
        # Create rich table for best parameters
        params_table = Table(title="Best Parameters")
        params_table.add_column("Parameter", style="cyan")
        params_table.add_column("Value", style="magenta")
        
        for param, value in results['best_params'].items():
            params_table.add_row(param, str(value))
        
        console.print(params_table)
        
        # Log additional insights
        console.log(f"[green]Best Run ID: {results['best_run_id']}")
        console.log(f"[green]Best Sharpe Ratio: {results['best_value']:.4f}")

if __name__ == "__main__":
    app()