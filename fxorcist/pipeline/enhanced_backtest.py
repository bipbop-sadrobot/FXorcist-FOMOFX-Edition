"""
Enhanced backtesting module with realistic market simulation features.
Implements slippage modeling, commission structures, and latency effects.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Callable, List, Tuple
from dataclasses import dataclass
import plotly.graph_objects as go
from datetime import datetime, timedelta

@dataclass
class BacktestConfig:
    """Configuration for backtest parameters."""
    slippage_model: str = "fixed"  # fixed, percentage, or impact
    slippage_value: float = 0.0001  # 1 pip default fixed slippage
    market_impact: float = 0.2  # Impact factor for volume-based slippage
    commission_type: str = "fixed"  # fixed or percentage
    commission_value: float = 0.0001  # 1 pip default commission
    latency_ms: int = 100  # Simulated latency in milliseconds
    initial_capital: float = 100000.0
    leverage: float = 1.0

@dataclass
class TradeStats:
    """Detailed trade statistics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: timedelta
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: timedelta
    total_return: float
    annual_return: float

def calculate_slippage(price: float, volume: float, direction: int,
                      config: BacktestConfig) -> float:
    """
    Calculate slippage based on configured model.
    
    Args:
        price: Current price
        volume: Trade volume
        direction: Trade direction (1 for buy, -1 for sell)
        config: Backtest configuration
        
    Returns:
        Price adjustment due to slippage
    """
    if config.slippage_model == "fixed":
        return config.slippage_value * direction
    elif config.slippage_model == "percentage":
        return price * config.slippage_value * direction
    elif config.slippage_model == "impact":
        # Square root model for price impact
        impact = config.market_impact * np.sqrt(volume) / 10000
        return price * impact * direction
    return 0.0

def calculate_commission(price: float, volume: float, config: BacktestConfig) -> float:
    """
    Calculate commission based on configured structure.
    
    Args:
        price: Trade price
        volume: Trade volume
        config: Backtest configuration
        
    Returns:
        Commission amount
    """
    if config.commission_type == "fixed":
        return config.commission_value
    elif config.commission_type == "percentage":
        return price * volume * config.commission_value
    return 0.0

def apply_latency(timestamp: pd.Timestamp, config: BacktestConfig) -> pd.Timestamp:
    """
    Apply simulated latency to order execution.
    
    Args:
        timestamp: Original timestamp
        config: Backtest configuration
        
    Returns:
        Adjusted timestamp
    """
    return timestamp + pd.Timedelta(milliseconds=config.latency_ms)

def calculate_trade_stats(trades_df: pd.DataFrame) -> TradeStats:
    """
    Calculate comprehensive trade statistics.
    
    Args:
        trades_df: DataFrame containing trade history
        
    Returns:
        TradeStats object with calculated metrics
    """
    if trades_df.empty:
        return TradeStats(
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, avg_win=0.0, avg_loss=0.0,
            largest_win=0.0, largest_loss=0.0,
            avg_trade_duration=timedelta(0),
            profit_factor=0.0, sharpe_ratio=0.0,
            sortino_ratio=0.0, max_drawdown=0.0,
            max_drawdown_duration=timedelta(0),
            total_return=0.0, annual_return=0.0
        )
    
    # Basic trade metrics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    
    # Profit metrics
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0.0
    avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0.0
    largest_win = trades_df['pnl'].max()
    largest_loss = abs(trades_df['pnl'].min())
    
    # Duration metrics
    trades_df['duration'] = trades_df['exit_time'] - trades_df['entry_time']
    avg_duration = trades_df['duration'].mean()
    
    # Risk metrics
    returns = trades_df['pnl'].values
    if len(returns) > 1:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
        downside_returns = returns[returns < 0]
        sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
    else:
        sharpe = sortino = 0.0
    
    # Drawdown analysis
    cumulative = (1 + trades_df['pnl'].cumsum())
    rolling_max = cumulative.cummax()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_drawdown = abs(drawdowns.min()) if not drawdowns.empty else 0.0
    
    # Calculate drawdown duration
    is_drawdown = cumulative < rolling_max
    drawdown_starts = is_drawdown.ne(is_drawdown.shift()).cumsum()
    drawdown_duration = trades_df.groupby(drawdown_starts)['duration'].sum()
    max_drawdown_duration = drawdown_duration.max() if not drawdown_duration.empty else timedelta(0)
    
    # Return metrics
    total_return = cumulative.iloc[-1] - 1 if not cumulative.empty else 0.0
    days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
    annual_return = ((1 + total_return) ** (365 / days) - 1) if days > 0 else 0.0
    
    # Profit factor
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    return TradeStats(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        largest_win=largest_win,
        largest_loss=largest_loss,
        avg_trade_duration=avg_duration,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_drawdown_duration,
        total_return=total_return,
        annual_return=annual_return
    )

def plot_equity_curve(trades_df: pd.DataFrame, title: str = "Equity Curve") -> go.Figure:
    """
    Generate interactive equity curve visualization.
    
    Args:
        trades_df: DataFrame containing trade history
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    if trades_df.empty:
        return go.Figure()
    
    cumulative = (1 + trades_df['pnl'].cumsum())
    rolling_max = cumulative.cummax()
    drawdowns = (cumulative - rolling_max) / rolling_max
    
    fig = go.Figure()
    
    # Equity curve
    fig.add_trace(go.Scatter(
        x=trades_df['exit_time'],
        y=cumulative,
        name='Equity',
        line=dict(color='blue')
    ))
    
    # Drawdown overlay
    fig.add_trace(go.Scatter(
        x=trades_df['exit_time'],
        y=drawdowns,
        name='Drawdown',
        line=dict(color='red'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Equity",
        yaxis2=dict(
            title="Drawdown",
            overlaying="y",
            side="right",
            range=[min(drawdowns) * 1.1, 0]
        ),
        showlegend=True
    )
    
    return fig

def backtest_strategy(df: pd.DataFrame, strategy_func: Callable,
                     config: BacktestConfig = None,
                     validation_window: int = 252) -> Tuple[pd.DataFrame, TradeStats]:
    """
    Run backtest with realistic market simulation.
    
    Args:
        df: Price data with OHLCV columns
        strategy_func: Strategy function that generates signals
        config: Backtest configuration
        
    Returns:
        Tuple of (trades DataFrame, trade statistics)
    """
    """
    Run backtest with realistic market simulation and look-ahead prevention.
    
    Args:
        df: Price data with OHLCV columns
        strategy_func: Strategy function that generates signals
        config: Backtest configuration
        validation_window: Number of days for out-of-sample validation
        
    Returns:
        Tuple of (trades DataFrame, trade statistics)
    """
    if config is None:
        config = BacktestConfig()
    
    positions = []
    trades = []
    current_position = 0
    entry_price = 0
    entry_time = None
    capital = config.initial_capital
    
    # Split data into training and validation sets
    train_size = len(df) - validation_window
    train_df = df.iloc[:train_size].copy()
    validation_df = df.iloc[train_size:].copy()
    
    # Generate signals using only training data for parameter fitting
    signals = pd.Series(index=df.index, dtype=float)
    signals.iloc[:train_size] = strategy_func(train_df)
    
    # Generate validation signals using only current and past data
    for i in range(train_size, len(df)):
        # Create historical window up to current timestamp
        historical_data = df.iloc[:i+1].copy()
        signals.iloc[i] = strategy_func(historical_data).iloc[-1]
    
    # Process signals with execution simulation
    for i, row in df.iterrows():
        # Validate timestamp sequence
        if i > 0 and (row.name - df.index[i-1]).total_seconds() < 0:
            raise ValueError(f"Non-chronological timestamps detected at {row.name}")
        # Apply latency to signal processing
        signal_time = apply_latency(i, config)
        
        # Process entry signals
        if signals.iloc[i] != 0 and current_position == 0:
            direction = signals.iloc[i]
            
            # Calculate execution costs
            spread = (row['Ask'] - row['Bid']) / 2 if 'Ask' in row and 'Bid' in row else row['Close'] * 0.0001
            slippage = calculate_slippage(row['Close'], row['Volume'], direction, config)
            commission = calculate_commission(row['Close'], row['Volume'], config)
            
            # Calculate entry price with spread and slippage
            base_price = row['Ask'] if direction > 0 else row['Bid'] if 'Ask' in row and 'Bid' in row else row['Close']
            entry_price = base_price + slippage
            entry_time = signal_time
            current_position = direction
            
            # Update capital for commission
            capital -= commission
            
        # Process exit signals
        elif (signals.iloc[i] == 0 or signals.iloc[i] == -current_position) and current_position != 0:
            # Calculate exit price with spread
            base_price = row['Bid'] if current_position > 0 else row['Ask'] if 'Ask' in row and 'Bid' in row else row['Close']
            exit_price = base_price
            exit_time = signal_time
            
            # Calculate exit slippage and commission
            slippage = calculate_slippage(exit_price, row['Volume'], -current_position, config)
            commission = calculate_commission(exit_price, row['Volume'], config)
            
            # Adjust exit price for slippage
            exit_price += slippage
            
            # Calculate trade P&L
            pnl = (exit_price - entry_price) * current_position * config.leverage
            pnl -= commission  # Include exit commission
            
            # Record trade
            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': current_position,
                'pnl': pnl,
                'commission': commission,
                'slippage': slippage
            })
            
            # Update capital
            capital += pnl
            current_position = 0
        
        # Record position for equity curve
        positions.append({
            'timestamp': i,
            'position': current_position,
            'capital': capital
        })
    
    # Convert to DataFrames
    trades_df = pd.DataFrame(trades)
    positions_df = pd.DataFrame(positions)
    
    # Calculate trade statistics
    stats = calculate_trade_stats(trades_df)
    
    return trades_df, stats