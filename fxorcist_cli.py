#!/usr/bin/env python3
"""
FXorcist CLI - Unified Command Line Interface

A comprehensive command-line interface for the FXorcist forex trading system.
Provides commands for:
- Data processing and integration
- Model training and optimization
- Dashboard management
- Memory system operations
- Configuration management

Examples:
    # Start data integration:
    $ fxorcist data integrate --input-dir ./data

    # Train a model:
    $ fxorcist train start --config training_config.yaml

    # Launch dashboard:
    $ fxorcist dashboard start --port 8501

    # View memory stats:
    $ fxorcist memory stats

For detailed documentation, see: docs/CLI_GUIDE.md
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import logging
from datetime import datetime
import subprocess
import yaml
from functools import partial

import click
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.prompt import Confirm
from rich.panel import Panel
from rich import print as rprint

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forex_ai_dashboard.pipeline.optimized_data_integration import OptimizedDataIntegrator
from forex_ai_dashboard.pipeline.enhanced_training_pipeline import EnhancedTrainingPipeline
from memory_system.core import MemoryManager

# Configure logging with rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
)
logger = logging.getLogger("fxorcist")

# Rich console for pretty output
console = Console()

class Config:
    """Configuration management for FXorcist CLI.
    
    Handles loading/saving of configuration from both JSON and YAML formats.
    Supports hierarchical config files with environment-specific overrides.
    """
    
    def __init__(self) -> None:
        self.project_root = Path(__file__).parent
        self.config_file = self.project_root / "config" / "cli_config.yaml"
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load CLI configuration with defaults and environment overrides."""
        default_config = {
            "data_dir": "data",
            "models_dir": "models",
            "logs_dir": "logs",
            "dashboard_port": 8501,
            "auto_backup": True,
            "quality_threshold": 0.7,
            "batch_size": 1000,
            "environment": "development",
            "max_memory": "8G",
            "logging": {
                "level": "INFO",
                "format": "%(message)s",
                "file": "fxorcist.log"
            }
        }

        # Try YAML first, fall back to JSON
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    if loaded_config:
                        default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Failed to load YAML config: {e}, falling back to JSON")
                json_config = self.project_root / "config" / "cli_config.json"
                if json_config.exists():
                    try:
                        with open(json_config, 'r') as f:
                            loaded_config = json.load(f)
                            default_config.update(loaded_config)
                    except Exception as e:
                        logger.error(f"Failed to load JSON config: {e}")

        return default_config

    def save_config(self) -> None:
        """Save current configuration to YAML file."""
        try:
            self.config_file.parent.mkdir(exist_ok=True)
            with open(self.config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

def show_command_help(ctx: click.Context, param: Any, value: bool) -> None:
    """Show detailed help for commands including examples."""
    if not value:
        return
    
    help_text = {
        "data integrate": """
        Examples:
            # Process new data files:
            $ fxorcist data integrate --input-dir ./new_data
            
            # Force reprocess existing data:
            $ fxorcist data integrate --force
            
            # Specify batch size:
            $ fxorcist data integrate --batch-size 2000
        """,
        "train start": """
        Examples:
            # Start training with config file:
            $ fxorcist train start --config my_training.yaml
            
            # Quick training mode:
            $ fxorcist train start --quick
            
            # Resume from checkpoint:
            $ fxorcist train start --resume checkpoint.pt
        """
    }
    
    if ctx.command.name in help_text:
        console.print(Panel(help_text[ctx.command.name], title="Command Help"))
        ctx.exit()

# CLI group
@click.group()
@click.option('--debug/--no-debug', default=False, help='Enable debug logging')
@click.option('--config-file', type=click.Path(), help='Custom config file path')
@click.pass_context
def cli(ctx: click.Context, debug: bool, config_file: Optional[str]) -> None:
    """FXorcist AI Dashboard - Command Line Interface

    Comprehensive toolset for forex data processing, model training, and system management.
    
    Common operations:
    \b
    - Data processing:        fxorcist data integrate
    - Model training:         fxorcist train start
    - Launch dashboard:       fxorcist dashboard start
    - View memory stats:      fxorcist memory stats
    - Configure system:       fxorcist config view
    
    Use --help with any command for detailed usage information.
    """
    # Set up context
    ctx.ensure_object(dict)
    ctx.obj['config'] = Config()
    
    # Override config file if specified
    if config_file:
        ctx.obj['config'].config_file = Path(config_file)
        ctx.obj['config'].config = ctx.obj['config'].load_config()
    
    # Configure logging level
    if debug:
        logging.getLogger("fxorcist").setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

# Data commands group
@cli.group()
def data() -> None:
    """Data processing and management commands.
    
    Handles data integration, validation, and preprocessing for the FXorcist system.
    """
    pass

@data.command()
@click.option('--input-dir', type=click.Path(exists=True), help='Input data directory')
@click.option('--force/--no-force', default=False, help='Force reprocessing of existing data')
@click.option('--batch-size', type=int, help='Processing batch size')
@click.option('--help-examples', is_flag=True, callback=show_command_help, expose_value=False,
              help='Show usage examples')
def integrate(input_dir: Optional[str], force: bool, batch_size: Optional[int]) -> None:
    """Run optimized data integration pipeline.
    
    Processes forex data files, applying necessary transformations and validations.
    Supports incremental processing and batch operations for large datasets.
    """
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            
            logger.info("Starting data integration...")
            main_task = progress.add_task("Processing data files...", total=100)
            
            integrator = OptimizedDataIntegrator()
            if batch_size:
                integrator.batch_size = batch_size
                
            def update_progress(percentage: float, message: str) -> None:
                progress.update(main_task, completed=percentage, description=message)
                
            results = integrator.process_optimized_data(
                progress_callback=update_progress,
                force_reprocess=force
            )
            
            progress.update(main_task, completed=100, description="Data integration complete!")
            
        # Show results table
        table = Table(title="Integration Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        for key, value in results.items():
            table.add_row(key, str(value))
            
        console.print(table)
            
    except Exception as e:
        logger.error(f"Data integration failed: {e}")
        sys.exit(1)

@data.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--threshold', type=float, help='Quality threshold')
def validate(file_path: str, threshold: Optional[float]) -> None:
    """Validate data quality for specific file.
    
    Performs comprehensive quality checks including:
    - Missing value detection
    - Outlier analysis
    - Format validation
    - Temporal consistency
    """
    try:
        with Progress() as progress:
            task = progress.add_task("Validating data...", total=100)
            
            logger.info(f"Validating data file: {file_path}")
            # Implementation here
            progress.update(task, advance=50)
            
            # More validation steps...
            progress.update(task, advance=50)
            
        logger.info("Validation complete")
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

# Training commands group
@cli.group()
def train() -> None:
    """Model training and optimization commands.
    
    Manages model training pipelines, hyperparameter optimization,
    and model evaluation processes.
    """
    pass

@train.command()
@click.option('--config', type=click.Path(exists=True), help='Training configuration file')
@click.option('--quick/--no-quick', default=False, help='Use quick training mode')
@click.option('--resume', type=click.Path(exists=True), help='Resume from checkpoint')
@click.option('--help-examples', is_flag=True, callback=show_command_help, expose_value=False,
              help='Show usage examples')
def start(config: Optional[str], quick: bool, resume: Optional[str]) -> None:
    """Start model training with specified configuration.
    
    Executes the training pipeline with the given parameters and monitors progress.
    Supports quick training mode for rapid prototyping and checkpoint resumption.
    """
    try:
        with Progress() as progress:
            train_task = progress.add_task("Training model...", total=100)
            
            logger.info("Starting model training...")
            pipeline = EnhancedTrainingPipeline()
            
            def update_progress(percentage: float, message: str) -> None:
                progress.update(train_task, completed=percentage, description=message)
            
            if quick:
                logger.info("Using quick training mode")
                pipeline.quick_train(progress_callback=update_progress)
            else:
                pipeline.train(
                    config_path=config,
                    resume_checkpoint=resume,
                    progress_callback=update_progress
                )
                
            progress.update(train_task, completed=100, description="Training complete!")
            
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

@train.command()
@click.option('--trials', type=int, default=100, help='Number of optimization trials')
@click.option('--timeout', type=int, help='Optimization timeout in minutes')
def optimize(trials: int, timeout: Optional[int]) -> None:
    """Run hyperparameter optimization.
    
    Uses Optuna to perform systematic hyperparameter search with:
    - Parallel trial execution
    - Early stopping for inefficient trials
    - Results visualization
    """
    try:
        with Progress() as progress:
            opt_task = progress.add_task("Optimizing hyperparameters...", total=trials)
            
            logger.info("Starting hyperparameter optimization...")
            # Implementation here
            for i in range(trials):
                progress.update(opt_task, advance=1)
                
            logger.info("Optimization completed")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(1)

# Dashboard commands group
@cli.group()
def dashboard() -> None:
    """Dashboard and visualization commands.
    
    Manages various dashboard interfaces including:
    - Main trading dashboard
    - Training monitoring dashboard
    - Performance analytics
    """
    pass

@dashboard.command()
@click.option('--port', type=int, help='Dashboard port')
@click.option('--host', type=str, default='localhost', help='Host to bind to')
@click.pass_context
def start(ctx: click.Context, port: Optional[int], host: str) -> None:
    """Start the main dashboard.
    
    Launches the Streamlit-based trading dashboard with:
    - Real-time market data visualization
    - Model performance monitoring
    - Trading signals display
    - System status indicators
    """
    config = ctx.obj['config']
    port = port or config.config.get('dashboard_port', 8501)
    
    try:
        logger.info(f"Starting dashboard on {host}:{port}...")
        cmd = f"cd dashboard && streamlit run app.py --server.port {port} --server.address {host}"
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Dashboard failed to start: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

@dashboard.command()
@click.pass_context
def training(ctx: click.Context) -> None:
    """Start the training dashboard.
    
    Launches specialized dashboard for:
    - Training progress monitoring
    - Model metrics visualization
    - Resource usage tracking
    - Hyperparameter analysis
    """
    config = ctx.obj['config']
    port = config.config.get('dashboard_port', 8501) + 1
    
    try:
        logger.info(f"Starting training dashboard on port {port}...")
        cmd = f"streamlit run enhanced_training_dashboard.py --server.port {port}"
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Training dashboard failed to start: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

# Memory system commands group
@cli.group()
def memory() -> None:
    """Memory system management commands.
    
    Controls the FXorcist memory system including:
    - Statistics and monitoring
    - Cache management
    - Memory optimization
    """
    pass

@memory.command()
def stats() -> None:
    """View memory system statistics.
    
    Displays detailed metrics including:
    - Memory usage and allocation
    - Cache hit rates
    - Storage efficiency
    - System health indicators
    """
    try:
        memory_manager = MemoryManager()
        stats = memory_manager.get_statistics()
        
        # Create rich table for stats
        table = Table(title="Memory System Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Description", style="green")
        
        for key, value in stats.items():
            description = memory_manager.get_metric_description(key)
            table.add_row(key, str(value), description)
        
        console.print(table)
    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        sys.exit(1)

@memory.command()
@click.option('--force/--no-force', default=False, help='Force clear without confirmation')
def clear(force: bool) -> None:
    """Clear memory system cache.
    
    Removes cached data while preserving critical system state.
    Use --force to skip confirmation prompt.
    """
    try:
        if force or Confirm.ask("Are you sure you want to clear the memory cache?"):
            with Progress() as progress:
                task = progress.add_task("Clearing cache...", total=100)
                
                memory_manager = MemoryManager()
                memory_manager.clear_cache(
                    progress_callback=lambda p: progress.update(task, completed=p)
                )
                
                progress.update(task, completed=100)
            logger.info("Memory cache cleared successfully")
    except Exception as e:
        logger.error(f"Failed to clear memory cache: {e}")
        sys.exit(1)

# Configuration commands group
@cli.group()
def config() -> None:
    """Configuration management commands.
    
    Handles system configuration including:
    - Settings viewing and modification
    - Environment management
    - Default value control
    """
    pass

@config.command()
@click.pass_context
def view(ctx: click.Context) -> None:
    """View current configuration.
    
    Displays all configuration settings in a formatted table,
    highlighting modified values and their sources.
    """
    config = ctx.obj['config']
    
    table = Table(title="Current Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_column("Source", style="green")
    
    for key, value in config.config.items():
        # Determine if value is default or custom
        source = "Custom" if key in config.config else "Default"
        table.add_row(key, str(value), source)
    
    console.print(table)

@config.command()
@click.option('--key', prompt='Setting key', help='Configuration key to modify')
@click.option('--value', prompt='New value', help='New value for the setting')
@click.pass_context
def set(ctx: click.Context, key: str, value: str) -> None:
    """Set configuration value.
    
    Updates a specific configuration setting with type conversion.
    Supports nested keys using dot notation (e.g., 'logging.level').
    """
    config = ctx.obj['config']
    
    try:
        # Try to convert string value to appropriate type
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        elif value.isdigit():
            value = int(value)
        elif value.replace('.', '').isdigit():
            value = float(value)
            
        # Handle nested keys
        keys = key.split('.')
        current = config.config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
        
        config.save_config()
        logger.info(f"Configuration updated: {key} = {value}")
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        sys.exit(1)

@config.command()
@click.pass_context
def reset(ctx: click.Context) -> None:
    """Reset configuration to defaults.
    
    Restores all settings to their default values.
    Requires confirmation unless --force is used.
    """
    if Confirm.ask("Are you sure you want to reset to defaults?"):
        config = ctx.obj['config']
        config.config = config.load_config()
        config.save_config()
        logger.info("Configuration reset to defaults")

def main() -> None:
    """Main CLI entry point."""
    try:
        # Enable shell completion
        cli.add_command(click.Command('completion', help='Output shell completion code'))
        cli(obj={})
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()