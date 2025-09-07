"""
Strategy Engine module for the FXorcist trading platform.
Processes market data and generates trading signals based on configured strategies.
"""

import asyncio
from collections import deque
from datetime import datetime
from typing import Dict, List, Any, Deque
import numpy as np
import logging

from ..core.base import TradingModule
from ..core.events import Event, EventType, SignalEvent
from ..core.dispatcher import EventDispatcher

class StrategyEngine(TradingModule):
    """
    Processes market events and generates trading signals.
    
    Responsibilities:
    - Market data processing and analysis
    - Technical indicator calculation
    - Signal generation based on strategies
    - Multi-instrument strategy management
    """
    
    def __init__(self, event_dispatcher: EventDispatcher, config: Dict[str, Any]):
        """
        Initialize the strategy engine.
        
        Args:
            event_dispatcher: Central event dispatcher instance
            config: Configuration dictionary containing:
                - instruments: List of instruments to monitor
                - lookback_period: Number of price points to store
                - strategies: List of strategy configurations
        """
        super().__init__("StrategyEngine", event_dispatcher, config)
        
        self.lookback_period = config.get('lookback_period', 100)
        self.market_data: Dict[str, Deque[Dict]] = {}
        self.indicators: Dict[str, Dict] = {}
        self.strategies = config.get('strategies', ['bollinger_rsi'])
        
        # Strategy parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2.0)
        self.signal_threshold = config.get('signal_threshold', 0.6)
    
    async def start(self):
        """Start the strategy engine."""
        await super().start()
        
        # Subscribe to market events
        self.event_dispatcher.subscribe(EventType.MARKET, self.handle_event)
        self.logger.info(f"Monitoring strategies: {', '.join(self.strategies)}")
    
    async def stop(self):
        """Stop the strategy engine."""
        self.market_data.clear()
        self.indicators.clear()
        await super().stop()
    
    async def handle_event(self, event: Event):
        """
        Handle market data events.
        
        Args:
            event: Market event to process
        """
        if event.type == EventType.MARKET:
            await self._process_market_data(event)
    
    async def _process_market_data(self, event: Event):
        """
        Process market data and generate signals.
        
        Args:
            event: Market event containing price data
        """
        instrument = event.data['instrument']
        
        # Initialize data storage for instrument
        if instrument not in self.market_data:
            self.market_data[instrument] = deque(maxlen=self.lookback_period)
            self.indicators[instrument] = {}
        
        # Calculate mid price and store data point
        mid_price = (event.data['bid'] + event.data['ask']) / 2
        self.market_data[instrument].append({
            'timestamp': event.timestamp,
            'price': mid_price,
            'volume': event.data['volume']
        })
        
        # Generate signals if we have enough data
        if len(self.market_data[instrument]) >= self.bb_period:
            await self._generate_signals(instrument, event.timestamp)
    
    async def _generate_signals(self, instrument: str, timestamp: datetime):
        """
        Generate trading signals using configured strategies.
        
        Args:
            instrument: Instrument to analyze
            timestamp: Current timestamp
        """
        data = list(self.market_data[instrument])
        prices = np.array([d['price'] for d in data])
        
        # Calculate indicators
        rsi = self._calculate_rsi(prices)
        sma, upper_band, lower_band = self._calculate_bollinger_bands(prices)
        
        # Store indicators
        current_price = prices[-1]
        self.indicators[instrument].update({
            'rsi': rsi,
            'sma': sma,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'current_price': current_price
        })
        
        # Apply strategies
        for strategy in self.strategies:
            if strategy == 'bollinger_rsi':
                await self._apply_bollinger_rsi_strategy(
                    instrument, timestamp, current_price, rsi, 
                    upper_band, lower_band
                )
            # Add additional strategy implementations here
    
    async def _apply_bollinger_rsi_strategy(
        self, instrument: str, timestamp: datetime, 
        current_price: float, rsi: float,
        upper_band: float, lower_band: float
    ):
        """
        Apply Bollinger Bands + RSI strategy.
        
        Args:
            instrument: Instrument being analyzed
            timestamp: Current timestamp
            current_price: Current price
            rsi: Current RSI value
            upper_band: Upper Bollinger Band
            lower_band: Lower Bollinger Band
        """
        signal_direction = None
        signal_strength = 0.0
        
        # Long signal conditions
        if current_price <= lower_band and rsi < 30:
            signal_direction = "LONG"
            # Calculate signal strength based on distance from band and RSI
            band_factor = (lower_band - current_price) / lower_band
            rsi_factor = (30 - rsi) / 30
            signal_strength = min(1.0, (band_factor + rsi_factor) / 2)
        
        # Short signal conditions
        elif current_price >= upper_band and rsi > 70:
            signal_direction = "SHORT"
            # Calculate signal strength based on distance from band and RSI
            band_factor = (current_price - upper_band) / upper_band
            rsi_factor = (rsi - 70) / 30
            signal_strength = min(1.0, (band_factor + rsi_factor) / 2)
        
        # Generate signal if strong enough
        if signal_direction and signal_strength > self.signal_threshold:
            signal_event = SignalEvent(
                instrument=instrument,
                direction=signal_direction,
                strength=signal_strength,
                strategy="bollinger_rsi",
                timestamp=timestamp,
                rsi=rsi,
                price=current_price,
                upper_band=upper_band,
                lower_band=lower_band
            )
            
            await self.publish_event(signal_event)
            self.logger.info(
                f"Signal generated: {signal_direction} {instrument} "
                f"(strength: {signal_strength:.2f}, RSI: {rsi:.1f})"
            )
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = None) -> float:
        """
        Calculate RSI indicator.
        
        Args:
            prices: Array of prices
            period: RSI period (optional, uses configured period if not specified)
            
        Returns:
            Current RSI value
        """
        period = period or self.rsi_period
        
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI when not enough data
        
        # Calculate price changes
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    
    def _calculate_bollinger_bands(
        self, prices: np.ndarray, 
        period: int = None, 
        std_dev: float = None
    ) -> tuple:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Array of prices
            period: Moving average period (optional)
            std_dev: Standard deviation multiplier (optional)
            
        Returns:
            Tuple of (SMA, Upper Band, Lower Band)
        """
        period = period or self.bb_period
        std_dev = std_dev or self.bb_std
        
        if len(prices) < period:
            current_price = prices[-1]
            return (
                current_price,
                current_price * 1.01,
                current_price * 0.99
            )
        
        # Calculate bands
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return float(sma), float(upper_band), float(lower_band)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the strategy engine.
        
        Returns:
            Dictionary containing status information
        """
        status = super().get_status()
        status.update({
            'active_instruments': len(self.market_data),
            'strategies': self.strategies,
            'lookback_period': self.lookback_period
        })
        return status
    
    async def reset(self):
        """Reset the strategy engine to initial state."""
        self.market_data.clear()
        self.indicators.clear()
        await super().reset()