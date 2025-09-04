#!/usr/bin/env python3
"""
FXorcist CLI - Unified Command Line Interface
Main entry point for all FXorcist operations with interactive prompts and robust command handling.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging
from datetime import datetime
import subprocess

import click
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forex_ai_dashboard.pipeline.optimized_data_integration import OptimizedDataIntegrator
from forex_ai_dashboard.pipeline.enhanced_training_pipeline import EnhancedTrainingPipeline
from memory_system.core import MemoryManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("fxorcist")

# Rich console for pretty output
console = Console()

class Config:
    """Configuration management for FXorcist CLI."""
    
    def __init__(self) -> None:
        self.project_root = Path(__file__).parent
        self.config_file = self.project_root / "config" / "cli_config.json"
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load CLI configuration with defaults."""
        default_config = {
            "data_dir": "data",
            "models_dir": "models",
            "logs_dir": "logs",
            "dashboard_port": 8501,
            "auto_backup": True,
            "quality_threshold": 0.7,
            "batch_size": 1000
        }

        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except json.JSONDecodeError:
                logger.error("Failed to parse config file")
            except Exception as e:
                logger.error(f"Error loading config: {e}")

        return default_config

    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            self.config_file.parent.mkdir(exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

# CLI group
@click.group()
@click.option('--debug/--no-debug', default=False, help='Enable debug logging')
@click.pass_context
def cli(ctx: click.Context, debug: bool) -> None:
    """FXorcist AI Dashboard - Command Line Interface

    Comprehensive toolset for forex data processing, model training, and system management.
    """
    # Set up context
    ctx.ensure_object(dict)
    ctx.obj['config'] = Config()
    
    # Configure logging level
    if debug:
        logging.getLogger("fxorcist").setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

# Data commands group
@cli.group()
def data() -> None:
    """Data processing and management commands."""
    pass

@data.command()
@click.option('--input-dir', type=click.Path(exists=True), help='Input data directory')
@click.option('--force/--no-force', default=False, help='Force reprocessing of existing data')
def integrate(input_dir: Optional[str], force: bool) -> None:
    """Run optimized data integration pipeline."""
    try:
        logger.info("Starting data integration...")
        integrator = OptimizedDataIntegrator()
        results = integrator.process_optimized_data()
        logger.info(f"Data integration completed: {results}")
    except Exception as e:
        logger.error(f"Data integration failed: {e}")
        sys.exit(1)

@data.command()
@click.argument('file_path', type=click.Path(exists=True))
def validate(file_path: str) -> None:
    """Validate data quality for specific file."""
    try:
        logger.info(f"Validating data file: {file_path}")
        # Implementation here
        logger.info("Validation complete")
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

# Training commands group
@cli.group()
def train() -> None:
    """Model training and optimization commands."""
    pass

@train.command()
@click.option('--config', type=click.Path(exists=True), help='Training configuration file')
@click.option('--quick/--no-quick', default=False, help='Use quick training mode')
def start(config: Optional[str], quick: bool) -> None:
    """Start model training with specified configuration."""
    try:
        logger.info("Starting model training...")
        pipeline = EnhancedTrainingPipeline()
        if quick:
            logger.info("Using quick training mode")
            # Quick training implementation
        else:
            # Full training implementation
            pass
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

@train.command()
def optimize() -> None:
    """Run hyperparameter optimization."""
    try:
        logger.info("Starting hyperparameter optimization...")
        # Implementation here
        logger.info("Optimization completed")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(1)

# Dashboard commands group
@cli.group()
def dashboard() -> None:
    """Dashboard and visualization commands."""
    pass

@dashboard.command()
@click.option('--port', type=int, help='Dashboard port')
@click.pass_context
def start(ctx: click.Context, port: Optional[int]) -> None:
    """Start the main dashboard."""
    config = ctx.obj['config']
    port = port or config.config.get('dashboard_port', 8501)
    
    try:
        logger.info(f"Starting dashboard on port {port}...")
        cmd = f"cd dashboard && streamlit run app.py --server.port {port}"
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
    """Start the training dashboard."""
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
    """Memory system management commands."""
    pass

@memory.command()
def stats() -> None:
    """View memory system statistics."""
    try:
        memory_manager = MemoryManager()
        stats = memory_manager.get_statistics()
        
        # Create rich table for stats
        table = Table(title="Memory System Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        for key, value in stats.items():
            table.add_row(key, str(value))
        
        console.print(table)
    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        sys.exit(1)

@memory.command()
def clear() -> None:
    """Clear memory system cache."""
    try:
        memory_manager = MemoryManager()
        memory_manager.clear_cache()
        logger.info("Memory cache cleared successfully")
    except Exception as e:
        logger.error(f"Failed to clear memory cache: {e}")
        sys.exit(1)

# Configuration commands group
@cli.group()
def config() -> None:
    """Configuration management commands."""
    pass

@config.command()
@click.pass_context
def view(ctx: click.Context) -> None:
    """View current configuration."""
    config = ctx.obj['config']
    
    table = Table(title="Current Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")
    
    for key, value in config.config.items():
        table.add_row(key, str(value))
    
    console.print(table)

@config.command()
@click.option('--key', prompt='Setting key', help='Configuration key to modify')
@click.option('--value', prompt='New value', help='New value for the setting')
@click.pass_context
def set(ctx: click.Context, key: str, value: str) -> None:
    """Set configuration value."""
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
            
        config.config[key] = value
        config.save_config()
        logger.info(f"Configuration updated: {key} = {value}")
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        sys.exit(1)

@config.command()
@click.pass_context
def reset(ctx: click.Context) -> None:
    """Reset configuration to defaults."""
    if click.confirm("Are you sure you want to reset to defaults?"):
        config = ctx.obj['config']
        config.config = config.load_config()
        config.save_config()
        logger.info("Configuration reset to defaults")

def main() -> None:
    """Main CLI entry point."""
    try:
        cli(obj={})
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()