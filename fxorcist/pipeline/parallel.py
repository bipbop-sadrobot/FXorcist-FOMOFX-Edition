"""
Parallel runner with multiprocessing, dask, and optional Ray support.
Returns structured results with metrics or error info.
"""
from __future__ import annotations
from typing import Dict, Any
from multiprocessing import Pool
from tqdm import tqdm
import dask
from dask.distributed import Client
import traceback

from fxorcist.pipeline.vectorized_backtest import sma_strategy_returns, simple_metrics

def _run_symbol(job):
    symbol, df, params = job
    try:
        returns = sma_strategy_returns(df, **params)
        metrics = simple_metrics(returns)
        return {"symbol": symbol, "metrics": metrics}
    except Exception as e:
        return {"symbol": symbol, "error": str(e), "traceback": traceback.format_exc()}

def run_parallel(symbol_dfs: Dict[str, Any], params: Dict[str, Any], n_workers: int = 4, use_dask: bool = False, show_progress: bool = True):
    jobs = [(s, symbol_dfs[s], params) for s in symbol_dfs]
    if use_dask:
        with Client(n_workers=n_workers) as client:
            delayed = [dask.delayed(_run_symbol)(job) for job in jobs]
            results = dask.compute(*delayed)
        return list(results)
    else:
        results = []
        with Pool(processes=n_workers) as p:
            if show_progress:
                for r in tqdm(p.imap_unordered(_run_symbol, jobs), total=len(jobs)):
                    results.append(r)
            else:
                results = p.map(_run_symbol, jobs)
        return results
