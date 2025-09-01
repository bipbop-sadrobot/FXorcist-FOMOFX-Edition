from loguru import logger
import sys
import os
from pathlib import Path
from datetime import datetime

# Configure log directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Configure log file path with timestamp
LOG_FILE = LOG_DIR / f"forex_ai_{datetime.now().strftime('%Y%m%d')}.log"

# Remove default logger configuration
logger.remove()

# Add file logging with rotation
logger.add(
    LOG_FILE,
    rotation="10 MB",  # Rotate when file reaches 10MB
    retention="30 days",  # Keep logs for 30 days
    compression="zip",  # Compress rotated logs
    serialize=True,  # Output as JSON
    level="DEBUG",  # Log level
)

# Add console logging
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# Custom exception handler
def log_exception(exc_type, exc_value, exc_traceback):
    logger.opt(exception=(exc_type, exc_value, exc_traceback)).error("Unhandled exception occurred")

# Set exception handler
sys.excepthook = log_exception

# Export logger for use in other modules
__all__ = ["logger"]
