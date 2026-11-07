"""
Centralized logging configuration for the entire application.
All modules import from here to maintain consistent log formatting.
"""

import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Create and return a named logger with console handler.
    
    Args:
        name: Usually __name__ from the calling module
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Console handler with structured format
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger