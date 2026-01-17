"""Logging configuration and utilities for Connectomix."""

import logging
import sys
import time
from contextlib import contextmanager
from typing import Optional
from colorama import Fore, Style, init


# Initialize colorama for cross-platform color support
init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    COLORS = {
        'DEBUG': Fore.BLUE,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record):
        """Format log record with color.
        
        Args:
            record: Log record to format
        
        Returns:
            Formatted string with color codes
        """
        # Save original levelname
        original_levelname = record.levelname
        
        # Add color to levelname
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        
        # Format the message
        result = super().format(record)
        
        # Restore original levelname
        record.levelname = original_levelname
        
        return result


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging with color support.
    
    Args:
        verbose: If True, set log level to DEBUG, otherwise INFO
        log_file: Optional path to log file
    
    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create logger
    logger = logging.getLogger('connectomix')
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create console handler with color
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = ColoredFormatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add console handler
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        # File handler uses plain formatter (no colors)
        plain_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(plain_formatter)
        
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


@contextmanager
def timer(logger: logging.Logger, message: str):
    """Context manager for timing operations.
    
    Args:
        logger: Logger instance
        message: Description of the operation being timed
    
    Yields:
        None
    
    Example:
        >>> logger = setup_logging()
        >>> with timer(logger, "Processing data"):
        ...     process_data()
        INFO - Starting: Processing data
        INFO - Completed: Processing data (2.34s)
    """
    start = time.time()
    logger.info(f"Starting: {message}")
    
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"Completed: {message} ({elapsed:.2f}s)")


def log_config(logger: logging.Logger, config: dict, title: str = "Configuration") -> None:
    """Log configuration parameters in a formatted way.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
        title: Title for the configuration section
    """
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)
    
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for subkey, subvalue in value.items():
                logger.info(f"  {subkey}: {subvalue}")
        else:
            logger.info(f"{key}: {value}")
    
    logger.info("=" * 60)


def log_section(logger: logging.Logger, title: str) -> None:
    """Log a section header.
    
    Args:
        logger: Logger instance
        title: Section title
    """
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)


def log_warning_box(logger: logging.Logger, message: str) -> None:
    """Log a warning message in a box.
    
    Args:
        logger: Logger instance
        message: Warning message
    """
    logger.warning("┌" + "─" * 58 + "┐")
    logger.warning(f"│ {message:<56} │")
    logger.warning("└" + "─" * 58 + "┘")


def log_error_box(logger: logging.Logger, message: str) -> None:
    """Log an error message in a box.
    
    Args:
        logger: Logger instance
        message: Error message
    """
    logger.error("┌" + "─" * 58 + "┐")
    logger.error(f"│ {message:<56} │")
    logger.error("└" + "─" * 58 + "┘")
