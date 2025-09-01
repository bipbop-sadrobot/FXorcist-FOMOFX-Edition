from typing import Callable, Dict, Any
from forex_ai_dashboard.utils.logger import logger

class EventBus:
    """Pub/sub event bus for cross-component communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, list] = {}
        
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to specific event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        logger.info(f"New subscriber for {event_type} events")
        
    def publish(self, event_type: str, data: Dict[str, Any]):
        """Publish event to all subscribers"""
        if event_type not in self.subscribers:
            return
            
        for callback in self.subscribers[event_type]:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Event callback failed: {e}")
                
    def get_registered_events(self) -> list:
        """Get list of event types with subscribers"""
        return list(self.subscribers.keys())
