"""
Performance metrics calculation for trading strategy backtests.
"""
from typing import List, Dict, Any
from datetime import datetime
import numpy as np
import pandas as pd

class MetricsCollector:
    """
    Comprehensive metrics collection for trading strategy performance.
    
    Tracks portfolio equity, trades, and calculates various performance metrics.
    """
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize metrics collector.
        
        Args:
            initial_capital: Starting portfolio value
        """
        self.initial_capital = initial_capital
        self.equity_curve: List[Dict[str, Any]] = [
            {'timestamp': datetime.now(), 'equity': initial_capital}
        ]
        self.trades: List[Dict[str, Any]] = []
    
    def record_trade(
        self, 
        symbol: str, 
        side: str, 
        entry_price: float, 
        exit_price: float, 
        size: float, 
        entry_time: datetime, 
        exit_time: datetime
    ) -> None:
        """
        Record a completed trade.
        
        Args:
            symbol: Trading symbol
            side: Trade side ('buy' or 'sell')
            entry_price: Price at trade entry
            exit_price: Price at trade exit
            size: Trade size
            entry_time: Trade entry timestamp
            exit_time: Trade exit timestamp
        """
        trade_return = (exit_price - entry_price) / entry_price if side == 'buy' else (entry_price - exit_price) / entry_price
        trade_profit = trade_return * size * entry_price
        
        self.trades.append({
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'return': trade_return,
            'profit': trade_profit
        })
    
    def update_equity(self, timestamp: datetime, equity: float) -> None:
        """
        Update the equity curve with a new data point.
        
        Args:
            timestamp: Current timestamp
            equity: Current portfolio value
        """
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity
        })
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive trading performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        # Convert equity curve to DataFrame
        df = pd.DataFrame(self.equity_curve)
        df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        df['returns'] = df['equity'].pct_change()
        
        # Basic metrics
        total_return = (df['equity'].iloc[-1] / self.initial_capital) - 1
        
        # Annualized return (CAGR)
        trading_days = (df.index[-1] - df.index[0]).days
        years = trading_days / 365
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Sharpe Ratio (assuming risk-free rate = 0)
        returns = df['returns'].dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Maximum Drawdown
        cumulative_max = df['equity'].cummax()
        drawdown = (df['equity'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        
        # Trade-level metrics
        trade_df = pd.DataFrame(self.trades)
        win_rate = (trade_df['profit'] > 0).mean() if len(trade_df) > 0 else 0
        avg_trade_return = trade_df['return'].mean() if len(trade_df) > 0 else 0
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return,
            'total_trades': len(self.trades),
            'initial_capital': self.initial_capital,
            'final_equity': df['equity'].iloc[-1]
        }
    
    def summary(self) -> str:
        """
        Generate a human-readable summary of performance metrics.
        
        Returns:
            Formatted performance summary string
        """
        metrics = self.calculate_metrics()
        summary_lines = [
            "Performance Summary:",
            f"Total Return: {metrics['total_return']:.2%}",
            f"CAGR: {metrics['cagr']:.2%}",
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}",
            f"Max Drawdown: {metrics['max_drawdown']:.2%}",
            f"Win Rate: {metrics['win_rate']:.2%}",
            f"Avg Trade Return: {metrics['avg_trade_return']:.2%}",
            f"Total Trades: {metrics['total_trades']}",
            f"Initial Capital: ${metrics['initial_capital']:,.2f}",
            f"Final Equity: ${metrics['final_equity']:,.2f}"
        ]
        return "\n".join(summary_lines)

def calculate_metrics(
    equity_curve: List[Dict[str, Any]], 
    initial_capital: float = 100000.0
) -> Dict[str, float]:
    """
    Standalone function to calculate metrics from an equity curve.
    
    Args:
        equity_curve: List of equity points
        initial_capital: Starting portfolio value
    
    Returns:
        Dictionary of performance metrics
    """
    collector = MetricsCollector(initial_capital)
    for point in equity_curve:
        collector.update_equity(point['timestamp'], point['equity'])
    
    return collector.calculate_metrics()