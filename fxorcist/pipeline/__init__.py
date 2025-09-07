"""
FXorcist Pipeline Module

Provides infrastructure for:
- Vectorized backtesting
- Parallel operations
- Pipeline execution management
"""

from . import vectorized_backtest  # type: ignore
from . import parallel  # type: ignore

__all__ = ['vectorized_backtest', 'parallel']