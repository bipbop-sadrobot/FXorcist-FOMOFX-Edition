"""
Data Handler module for the FXorcist trading platform.
Manages market data ingestion, preprocessing, and distribution.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any
import logging
import random  # For demo data generation

from ..core.base import TradingModule
from ..core.events import MarketEvent, EventType, Event
from ..core.dispatcher import EventDispatcher

class DataHandler(TradingModule):
    """
    Handles market data ingestion from multiple sources.
    
    Responsibilities:
    - Market data ingestion and normalization
    - Data feed management for multiple instruments
    - Real-time and historical data handling
    - Data preprocessing and validation
    """
    
    def __init__(self, event_dispatcher: EventDispatcher, config: Dict[str, Any]):
        """
        Initialize the data handler.
        
        Args:
            event_dispatcher: Central event dispatcher instance
            config: Configuration dictionary containing:
                - instruments: List of instruments to track
                - data_sources: Dict of data source configurations
                - tick_interval: Interval between market updates (seconds)
        """
        super().__init__("DataHandler", event_dispatcher, config)
        
        self.instruments = config.get('instruments', ['EUR_USD'])
        self.data_sources = {}
        self.tick_interval = config.get('tick_interval', 1.0)
        self.feed_tasks = {}
        self.last_prices = {}
        
        # Data validation thresholds
        self.max_spread = config.get('max_spread', 0.0020)  # 20 pips
        self.max_price_change = config.get('max_price_change', 0.0050)  # 50 pips
    
    async def start(self):
        """Start data ingestion for all configured instruments."""
        await super().start()
        
        # Subscribe to system events
        self.event_dispatcher.subscribe(EventType.SYSTEM, self.handle_event)
        
        # Start data feeds for each instrument
        for instrument in self.instruments:
            self.feed_tasks[instrument] = asyncio.create_task(
                self._market_data_feed(instrument)
            )
            self.logger.info(f"Started data feed for {instrument}")
    
    async def stop(self):
        """Stop all data feeds and cleanup."""
        # Cancel all feed tasks
        for instrument, task in self.feed_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            self.logger.info(f"Stopped data feed for {instrument}")
        
        self.feed_tasks.clear()
        await super().stop()
    
    async def handle_event(self, event: Event):
        """
        Handle system events.
        
        Args:
            event: Incoming event to process
        """
        if event.type == EventType.SYSTEM:
            command = event.data.get('command')
            
            if command == 'stop':
                await self.stop()
            elif command == 'add_instrument':
                instrument = event.data.get('instrument')
                if instrument and instrument not in self.instruments:
                    self.instruments.append(instrument)
                    self.feed_tasks[instrument] = asyncio.create_task(
                        self._market_data_feed(instrument)
                    )
                    self.logger.info(f"Added data feed for {instrument}")
            elif command == 'remove_instrument':
                instrument = event.data.get('instrument')
                if instrument in self.feed_tasks:
                    self.feed_tasks[instrument].cancel()
                    self.instruments.remove(instrument)
                    self.logger.info(f"Removed data feed for {instrument}")
    
    def _validate_price(self, instrument: str, bid: Decimal, ask: Decimal) -> bool:
        """
        Validate price updates against thresholds.
        
        Args:
            instrument: Instrument being validated
            bid: New bid price
            ask: New ask price
            
        Returns:
            bool: True if price is valid, False otherwise
        """
        # Check spread
        spread = ask - bid
        if spread > self.max_spread:
            self.logger.warning(
                f"Invalid spread for {instrument}: {float(spread):.5f}"
            )
            return False
        
        # Check price change if we have a previous price
        if instrument in self.last_prices:
            last_mid = self.last_prices[instrument]
            new_mid = (bid + ask) / 2
            change = abs(new_mid - last_mid)
            
            if change > self.max_price_change:
                self.logger.warning(
                    f"Excessive price change for {instrument}: {float(change):.5f}"
                )
                return False
        
        return True
    
    async def _market_data_feed(self, instrument: str):
        """
        Simulated market data feed - replace with actual broker API integration.
        
        Args:
            instrument: Instrument to generate data for
        """
        # Initialize price based on instrument
        if instrument == 'EUR_USD':
            base_price = Decimal('1.0750')
        elif instrument == 'GBP_USD':
            base_price = Decimal('1.2650')
        elif instrument == 'USD_JPY':
            base_price = Decimal('147.50')
        else:
            base_price = Decimal('1.0000')
        
        self.last_prices[instrument] = base_price
        
        while self.running:
            try:
                # Simulate price movement
                change = Decimal(str(random.uniform(-0.0005, 0.0005)))
                bid = base_price + change
                ask = bid + Decimal('0.0002')  # 2 pip spread
                
                # Validate price update
                if not self._validate_price(instrument, bid, ask):
                    await asyncio.sleep(self.tick_interval)
                    continue
                
                # Create and publish market event
                market_event = MarketEvent(
                    instrument=instrument,
                    bid=bid,
                    ask=ask,
                    volume=random.randint(100000, 1000000),
                    timestamp=datetime.now(timezone.utc)
                )
                
                await self.publish_event(market_event)
                
                # Update state
                base_price = bid
                self.last_prices[instrument] = (bid + ask) / 2
                
                # Wait for next tick
                await asyncio.sleep(self.tick_interval)
                
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error(f"Error in market data feed for {instrument}: {e}")
                await asyncio.sleep(self.tick_interval)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the data handler.
        
        Returns:
            Dictionary containing status information
        """
        status = super().get_status()
        status.update({
            'active_feeds': len(self.feed_tasks),
            'instruments': self.instruments,
            'tick_interval': self.tick_interval
        })
        return status
    
    async def reset(self):
        """Reset the data handler to initial state."""
        await self.stop()
        self.last_prices.clear()
        self.data_sources.clear()
        await super().reset()