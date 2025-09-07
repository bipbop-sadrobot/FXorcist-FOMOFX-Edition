"""
Event dispatcher implementation for FXorcist trading platform.
Provides central event routing and async event processing capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Callable
from .events import Event, EventType

class EventDispatcher:
    """Central event dispatcher with async queue and routing."""
    
    def __init__(self, max_queue_size: int = 10000):
        """
        Initialize the event dispatcher.
        
        Args:
            max_queue_size: Maximum number of events to queue before blocking
        """
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.handlers: Dict[EventType, List[Callable]] = {}
        self.running = False
        self.processed_events = 0
        self.logger = logging.getLogger(__name__)
        
    def subscribe(self, event_type: EventType, handler: Callable):
        """
        Subscribe a handler to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Async callback function to handle events
        """
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        self.logger.info(f"Handler {handler.__name__} subscribed to {event_type}")
    
    async def publish(self, event: Event):
        """
        Publish an event to the queue.
        
        Args:
            event: Event object to publish
            
        Raises:
            asyncio.QueueFull: If queue is full and can't accept more events
        """
        try:
            await self.queue.put(event)
            self.logger.debug(f"Event published: {event.type} - {event.event_id}")
        except asyncio.QueueFull:
            self.logger.error(f"Event queue full! Dropping event: {event.type}")
            raise
    
    async def start(self):
        """Start the event processing loop."""
        self.running = True
        self.logger.info("Event dispatcher started")
        
        while self.running:
            try:
                # Get event with timeout to allow periodic checks
                event = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                await self._process_event(event)
                self.processed_events += 1
                
            except asyncio.TimeoutError:
                continue  # Normal timeout, check if still running
            except Exception as e:
                self.logger.error(f"Error processing event: {e}")
    
    async def _process_event(self, event: Event):
        """
        Process an event by calling all registered handlers.
        
        Args:
            event: Event object to process
        """
        handlers = self.handlers.get(event.type, [])
        
        if not handlers:
            self.logger.debug(f"No handlers for event type: {event.type}")
            return
        
        # Call all handlers concurrently
        tasks = [handler(event) for handler in handlers]
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            self.logger.error(f"Error in event handlers: {e}")
    
    def stop(self):
        """Stop the event dispatcher."""
        self.running = False
        self.logger.info(f"Event dispatcher stopped. Processed {self.processed_events} events.")

    async def drain(self):
        """
        Drain the event queue by processing remaining events.
        Should be called before stopping if clean shutdown is needed.
        """
        self.logger.info("Draining event queue...")
        while not self.queue.empty():
            try:
                event = self.queue.get_nowait()
                await self._process_event(event)
                self.processed_events += 1
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                self.logger.error(f"Error draining queue: {e}")
        
        self.logger.info("Event queue drained")