import argparse
import yaml
import json
from rich.console import Console
from rich.table import Table
from typing import Optional
from fxorcist.data.loader import load_symbol, list_available_symbols
from fxorcist.ml.optuna_runner import run_optuna

console = Console()

def load_config(path: Optional[str]) -> dict:
    if not path:
        return {}
    try:
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        console.print("[red]Failed to load config. Ignoring.[/red]")
        return {}

def main(argv=None):
    parser = argparse.ArgumentParser('fxorcist')
    parser.add_argument('--config', '-c', help='YAML config file', default=None)
    parser.add_argument('--json', action='store_true', help='Output JSON results')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_data = sub.add_parser('data')
    p_data.add_argument('--symbol', required=True)

    p_opt = sub.add_parser('optuna')
    p_opt.add_argument('--symbol', required=True)
    p_opt.add_argument('--trials', type=int, default=30)
    p_opt.add_argument('--out', default='artifacts/best_params.yaml')
    p_opt.add_argument('--storage', default=None)
    p_opt.add_argument('--mlflow', action='store_true')

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == 'data':
        df = load_symbol(args.symbol, base_dir=cfg.get('base_dir'))
        if args.json:
            print(df.head().to_json(orient='split', date_format='iso'))
        else:
            console.print(df.head())
    elif args.cmd == 'optuna':
        df = load_symbol(args.symbol, base_dir=cfg.get('base_dir'))
        res = run_optuna(df, n_trials=args.trials, seed=cfg.get('seed', 42), out_path=args.out, storage=args.storage, use_mlflow=args.mlflow)
        if args.json:
            print(json.dumps({'best': res['best_params']}))
        else:
            table = Table(title="Best params")
            for k,v in res['best_params'].items():
                table.add_row(str(k), str(v))
            console.print(table)
