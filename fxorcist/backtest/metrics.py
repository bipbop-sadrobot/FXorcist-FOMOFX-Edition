import numpy as np
from typing import List, Dict, Any
from datetime import datetime

class MetricsCalculator:
    def __init__(self):
        self.equity_curve = []  # (timestamp, value)
        self.trades = []

    def update(self, portfolio, timestamp: datetime):
        """Incremental update — store snapshot."""
        self.equity_curve.append((timestamp, portfolio.value))
        # In real app, append trades if filled

    def get_results(self) -> Dict[str, float]:
        """Calculate final metrics."""
        if len(self.equity_curve) < 2:
            return {"cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

        # Vectorized with numpy
        values = np.array([v for _, v in self.equity_curve])
        timestamps = [t for t, _ in self.equity_curve]

        # CAGR
        years = (timestamps[-1] - timestamps[0]).days / 365.25
        cagr = (values[-1] / values[0]) ** (1/years) - 1 if years > 0 else 0.0

        # Daily returns
        daily_values = self._resample_to_daily(values, timestamps)
        if len(daily_values) < 2:
            sharpe = 0.0
        else:
            daily_returns = np.diff(daily_values) / daily_values[:-1]
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)

        # Max Drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        max_drawdown = np.max(drawdown)

        return {
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "total_trades": len(self.trades),
            "win_rate": self._calculate_win_rate()
        }

    def _resample_to_daily(self, values, timestamps):
        """Simple daily resample (first value per day)."""
        daily = {}
        for ts, v in zip(timestamps, values):
            day = ts.date()
            if day not in daily:
                daily[day] = v
        return np.array(list(daily.values()))

    def _calculate_win_rate(self):
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t["pnl"] > 0)
        return wins / len(self.trades)