"""Logger configuration for the ML library."""

from typing import Dict, Optional, Union
import sys

from loguru import logger


def setup_logger(
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    rotation: Optional[str] = "10 MB",
    retention: Optional[str] = "1 week",
    format: Optional[str] = None,
) -> None:
    """Configure the logger for the ML library.
    
    Args:
        level: Log level to use. Can be a string or int.
        log_file: Path to the log file. If None, logs to stderr only.
        rotation: When to rotate the log file. Default is "10 MB".
        retention: How long to keep log files. Default is "1 week".
        format: Log format string. If None, uses the default format.
    """
    # Remove default handlers
    logger.remove()
    
    # Default format if not specified
    if format is None:
        format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Add stderr handler
    logger.add(sys.stderr, level=level, format=format)
    
    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            level=level,
            format=format,
            rotation=rotation,
            retention=retention,
        )
    
    logger.debug("Logger initialized")


def get_logger():
    """Get the configured logger instance."""
    return logger
