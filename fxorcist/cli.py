"""
FXorcist CLI Module

Provides command-line interface with subcommands for data operations,
model training, backtesting, and dashboard launching.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# Install Rich traceback handler
install(show_locals=True)

# Set up Rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("fxorcist")

def setup_data_parser(subparsers):
    """Set up data subcommand parser."""
    parser = subparsers.add_parser(
        "data",
        help="Data operations (download, validate, transform)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Currency pair symbol (e.g., EURUSD)",
        required=True
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)",
        required=True
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)",
        required=True
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory",
        default="data"
    )
    return parser

def setup_train_parser(subparsers):
    """Set up training subcommand parser."""
    parser = subparsers.add_parser(
        "train",
        help="Train models with Optuna optimization",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Training configuration file",
        required=True
    )
    parser.add_argument(
        "--trials",
        type=int,
        help="Number of Optuna trials",
        default=100
    )
    return parser

def setup_backtest_parser(subparsers):
    """Set up backtest subcommand parser."""
    parser = subparsers.add_parser(
        "backtest",
        help="Run vectorized backtests",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Data file or directory",
        required=True
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model file",
        required=True
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing"
    )
    return parser

def setup_dashboard_parser(subparsers):
    """Set up dashboard subcommand parser."""
    parser = subparsers.add_parser(
        "dashboard",
        help="Launch Streamlit dashboard",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port number",
        default=8501
    )
    return parser

def get_parser():
    """Create main argument parser."""
    parser = argparse.ArgumentParser(
        description="FXorcist: AI-driven FX trading system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add global arguments
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        help="Enable safe mode (no live trading)"
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    setup_data_parser(subparsers)
    setup_train_parser(subparsers)
    setup_backtest_parser(subparsers)
    setup_dashboard_parser(subparsers)
    
    return parser

def main(args: Optional[list] = None):
    """Main entry point for the CLI."""
    parser = get_parser()
    args = parser.parse_args(args)
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "data":
            from fxorcist.data.loader import process_data
            process_data(args)
        elif args.command == "train":
            from fxorcist.ml.optuna_runner import run_optimization
            run_optimization(args)
        elif args.command == "backtest":
            from fxorcist.pipeline.vectorized_backtest import run_backtest
            run_backtest(args)
        elif args.command == "dashboard":
            from fxorcist.dashboard.app import run_dashboard
            run_dashboard(args)
    except Exception as e:
        logger.error(f"Error executing {args.command}: {str(e)}")
        if args.verbose:
            raise
        sys.exit(1)

if __name__ == "__main__":
    main()