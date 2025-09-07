"""
Main entry point for the FXorcist trading platform.
Configures and runs the event-driven trading system.
"""

import asyncio
import logging
from datetime import datetime, timezone
import signal
import sys
from typing import Dict, Any, List

from .core.dispatcher import EventDispatcher
from .core.events import Event, EventType
from .modules.data_handler import DataHandler
from .modules.strategy_engine import StrategyEngine
from .modules.portfolio_manager import PortfolioManager

class TradingSystem:
    """
    Main trading system coordinator.
    
    Responsibilities:
    - System configuration and initialization
    - Component lifecycle management
    - Clean shutdown handling
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trading system.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.modules: List[Any] = []
        
        # Initialize event dispatcher
        self.dispatcher = EventDispatcher(
            max_queue_size=config.get('max_queue_size', 10000)
        )
        
        # Initialize modules
        self.data_handler = DataHandler(self.dispatcher, config)
        self.strategy_engine = StrategyEngine(self.dispatcher, config)
        self.portfolio_manager = PortfolioManager(self.dispatcher, config)
        
        self.modules = [
            self.data_handler,
            self.strategy_engine,
            self.portfolio_manager
        ]
    
    async def start(self):
        """Start all system components."""
        self.running = True
        self.logger.info("Starting trading system...")
        
        # Start all modules
        for module in self.modules:
            await module.start()
        
        # Start event dispatcher
        self.dispatcher_task = asyncio.create_task(self.dispatcher.start())
        
        self.logger.info("Trading system started")
    
    async def stop(self):
        """Stop all system components."""
        self.running = False
        self.logger.info("Stopping trading system...")
        
        # Stop all modules in reverse order
        for module in reversed(self.modules):
            await module.stop()
        
        # Drain and stop dispatcher
        await self.dispatcher.drain()
        self.dispatcher.stop()
        
        try:
            await self.dispatcher_task
        except asyncio.CancelledError:
            pass
        
        self.logger.info("Trading system stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status.
        
        Returns:
            Dictionary containing system status information
        """
        return {
            'running': self.running,
            'data_handler': self.data_handler.get_status(),
            'strategy_engine': self.strategy_engine.get_status(),
            'portfolio': self.portfolio_manager.get_status(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

async def main():
    """Main entry point for the trading system."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # System configuration
    config = {
        'instruments': ['EUR_USD', 'GBP_USD', 'USD_JPY'],
        'initial_balance': 10000,
        'max_risk_per_trade': 0.02,
        'max_positions': 3,
        'max_drawdown': 0.20,
        'lookback_period': 100,
        'strategies': ['bollinger_rsi'],
        'tick_interval': 1.0  # 1 second for demo
    }
    
    # Create and start trading system
    system = TradingSystem(config)
    
    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    
    def signal_handler():
        """Handle shutdown signals."""
        logging.info("Shutdown signal received")
        asyncio.create_task(system.stop())
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        # Start system
        await system.start()
        
        # Run until stopped
        while system.running:
            await asyncio.sleep(1)
            
            # Periodically log system status
            if system.running:
                status = system.get_system_status()
                portfolio = status['portfolio']
                logging.info(
                    f"Portfolio Balance: ${portfolio['balance']:.2f}, "
                    f"Open Positions: {portfolio['open_positions']}, "
                    f"Total Trades: {portfolio['total_trades']}"
                )
    
    except Exception as e:
        logging.error(f"System error: {e}")
        raise
    
    finally:
        # Ensure clean shutdown
        await system.stop()
        
        # Final status report
        status = system.get_system_status()
        portfolio = status['portfolio']
        
        print("\nFinal Portfolio Summary:")
        print(f"Balance: ${portfolio['balance']:.2f}")
        print(f"Realized P&L: ${portfolio['realized_pnl']:.2f}")
        print(f"Total Trades: {portfolio['total_trades']}")
        print(f"Max Drawdown: {portfolio['current_drawdown']:.1%}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")
        sys.exit(0)