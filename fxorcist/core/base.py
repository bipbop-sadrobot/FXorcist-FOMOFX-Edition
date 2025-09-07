"""
Abstract base class for trading modules in the FXorcist platform.
Defines the common interface and basic functionality for all trading components.
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, Optional
from .dispatcher import EventDispatcher
from .events import Event

class TradingModule(ABC):
    """
    Abstract base class for all trading modules.
    
    Provides common functionality and enforces consistent interface across
    all trading components in the system.
    """
    
    def __init__(self, name: str, event_dispatcher: EventDispatcher, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a trading module.
        
        Args:
            name: Unique identifier for the module
            event_dispatcher: Central event dispatcher instance
            config: Optional configuration dictionary
        """
        self.name = name
        self.event_dispatcher = event_dispatcher
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.running = False
        
        # Set up module-specific logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure module-specific logging."""
        # Add module name to log format
        formatter = logging.Formatter(
            f'%(asctime)s - {self.name} - %(levelname)s - %(message)s'
        )
        
        # Ensure all handlers use this formatter
        for handler in self.logger.handlers:
            handler.setFormatter(formatter)
    
    @abstractmethod
    async def start(self):
        """
        Start the module's operation.
        
        This method should:
        1. Set up any necessary resources
        2. Subscribe to relevant events
        3. Initialize internal state
        4. Begin any continuous processing tasks
        """
        self.running = True
        self.logger.info(f"Module {self.name} started")
    
    @abstractmethod
    async def stop(self):
        """
        Stop the module's operation.
        
        This method should:
        1. Clean up resources
        2. Cancel any ongoing tasks
        3. Ensure proper shutdown
        """
        self.running = False
        self.logger.info(f"Module {self.name} stopped")
    
    @abstractmethod
    async def handle_event(self, event: Event):
        """
        Handle an incoming event.
        
        Args:
            event: The event to process
            
        This method should:
        1. Process the event according to module logic
        2. Update internal state as needed
        3. Generate and publish any resulting events
        """
        pass
    
    async def publish_event(self, event: Event):
        """
        Convenience method to publish an event.
        
        Args:
            event: The event to publish
        """
        try:
            await self.event_dispatcher.publish(event)
            self.logger.debug(f"Published event: {event.type} - {event.event_id}")
        except Exception as e:
            self.logger.error(f"Error publishing event: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the module.
        
        Returns:
            Dictionary containing module status information
        """
        return {
            'name': self.name,
            'running': self.running,
            'type': self.__class__.__name__
        }
    
    async def health_check(self) -> bool:
        """
        Perform a health check of the module.
        
        Returns:
            True if module is healthy, False otherwise
        """
        return self.running
    
    def configure(self, config: Dict[str, Any]):
        """
        Update module configuration.
        
        Args:
            config: New configuration dictionary to merge with existing config
        """
        self.config.update(config)
        self.logger.info("Configuration updated")
    
    async def reset(self):
        """
        Reset the module to its initial state.
        
        This is a basic implementation that can be overridden by subclasses
        to provide more specific reset behavior.
        """
        self.running = False
        self.config = {}
        self.logger.info("Module reset to initial state")