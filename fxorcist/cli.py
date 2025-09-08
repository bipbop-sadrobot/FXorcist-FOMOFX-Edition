"""
FXorcist CLI - Event-Driven Forex Backtesting & Optimization
"""
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import yaml
import json

from fxorcist.data.loader import load_symbol, list_available_symbols
from fxorcist.ml.optuna_runner import run_optuna
from fxorcist.backtest.engine import run_backtest
from fxorcist.utils.config import load_config

app = typer.Typer(
    help="FXorcist — Event-Driven Forex Backtesting & Optimization",
    add_completion=False,
)
console = Console()

def version_callback(value: bool):
    """Print version and exit."""
    if value:
        from fxorcist import __version__
        console.print(f"FXorcist v{__version__}")
        raise typer.Exit()

@app.callback()
def main(
    config: Path = typer.Option(
        None,
        "--config", "-c",
        help="Path to YAML config file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    version: bool = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
    ),
):
    """
    FXorcist CLI - Event-Driven Forex Backtesting & Optimization

    Use --help with any command for detailed usage information.
    """
    # Store common options in state
    app.state.config = load_config(config) if config else {}
    app.state.json_output = json_output

@app.command()
def prepare(
    symbol: str = typer.Argument(..., help="Trading symbol (e.g., EURUSD)"),
    start_date: str = typer.Option(None, help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(None, help="End date (YYYY-MM-DD)"),
    force: bool = typer.Option(False, help="Force redownload even if data exists"),
):
    """Prepare market data for backtesting."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Preparing {symbol} data...", total=None)
        try:
            df = load_symbol(
                symbol,
                base_dir=app.state.config.get('base_dir'),
                start_date=start_date,
                end_date=end_date,
                force=force,
            )
            progress.update(task, completed=True)
            
            if app.state.json_output:
                console.print_json(json.dumps({
                    'symbol': symbol,
                    'rows': len(df),
                    'start': df.index[0].isoformat(),
                    'end': df.index[-1].isoformat(),
                }))
            else:
                console.print(f"[green]✓[/green] Loaded {len(df):,} rows")
                console.print(f"   Period: {df.index[0]} → {df.index[-1]}")
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            raise typer.Exit(1)

@app.command()
def backtest(
    strategy: str = typer.Argument(..., help="Strategy name"),
    symbol: str = typer.Option(..., help="Trading symbol"),
    params_file: Optional[Path] = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Strategy parameters YAML/JSON file",
    ),
    report: bool = typer.Option(False, help="Generate HTML report"),
):
    """Run event-driven backtest."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running backtest...", total=None)
        try:
            results = run_backtest(
                strategy_name=strategy,
                symbol=symbol,
                params_file=params_file,
                config=app.state.config,
                progress=progress,
            )
            progress.update(task, completed=True)
            
            if app.state.json_output:
                console.print_json(json.dumps(results))
            else:
                table = Table(title=f"Backtest Results - {strategy}")
                table.add_column("Metric")
                table.add_column("Value")
                for k, v in results.items():
                    table.add_row(k, f"{v:,.4f}" if isinstance(v, float) else str(v))
                console.print(table)
                
            if report:
                report_path = f"reports/{strategy}_{symbol}.html"
                results['report'].save(report_path)
                console.print(f"\nReport saved to: {report_path}")
                
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            raise typer.Exit(1)

@app.command()
def optimize(
    strategy: str = typer.Argument(..., help="Strategy to optimize"),
    symbol: str = typer.Option(..., help="Trading symbol"),
    trials: int = typer.Option(30, help="Number of optimization trials"),
    out: Path = typer.Option(
        "artifacts/best_params.yaml",
        help="Output file for best parameters",
    ),
    storage: Optional[str] = typer.Option(
        None, help="Optuna storage URL"
    ),
    mlflow: bool = typer.Option(False, help="Log to MLflow"),
):
    """Run hyperparameter optimization."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Optimizing {strategy}...", total=trials)
        try:
            df = load_symbol(symbol, base_dir=app.state.config.get('base_dir'))
            results = run_optuna(
                df,
                n_trials=trials,
                seed=app.state.config.get('seed', 42),
                out_path=str(out),
                storage=storage,
                use_mlflow=mlflow,
                progress=progress,
            )
            progress.update(task, completed=True)
            
            if app.state.json_output:
                console.print_json(json.dumps({'best': results['best_params']}))
            else:
                table = Table(title="Best Parameters")
                table.add_column("Parameter")
                table.add_column("Value")
                for k, v in results['best_params'].items():
                    table.add_row(str(k), str(v))
                console.print(table)
                
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            raise typer.Exit(1)

@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Dashboard host"),
    port: int = typer.Option(8000, help="Dashboard port"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
):
    """Start the web dashboard."""
    try:
        from fxorcist.dashboard.app import run_dashboard
        console.print(f"Starting dashboard on http://{host}:{port}")
        run_dashboard(
            host=host,
            port=port,
            reload=reload,
            config=app.state.config,
        )
    except ImportError:
        console.print("[red]Error:[/red] Dashboard dependencies not installed.")
        console.print("Run: pip install fxorcist[dashboard]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()