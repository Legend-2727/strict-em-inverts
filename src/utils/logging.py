"""
Centralized logging configuration for the dynamix project.

Usage:
    from src.utils.logging import get_logger, setup_logging

    # Get a logger for your module
    logger = get_logger(__name__)

    # Or configure logging globally (typically in main script)
    setup_logging(level="DEBUG", log_file="logs/experiment.log")
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


# Default format strings
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
SIMPLE_FORMAT = "%(levelname)-8s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: The name for the logger, typically __name__ of the calling module.

    Returns:
        A configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return logging.getLogger(name)


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    log_dir: Optional[Union[str, Path]] = None,
    format_string: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    console: bool = True,
    file_level: Optional[Union[str, int]] = None,
    reset: bool = False,
) -> logging.Logger:
    """
    Configure the root logger with console and optional file handlers.

    Args:
        level: The logging level for console output. Can be string ("DEBUG", "INFO", etc.)
               or int (logging.DEBUG, logging.INFO, etc.). Defaults to "INFO".
        log_file: Path to the log file. If provided, logs will also be written to this file.
        log_dir: Directory for log files. If log_file is not provided but log_dir is,
                 a timestamped log file will be created in this directory.
        format_string: The format string for log messages.
        date_format: The date format for timestamps.
        console: Whether to output logs to console. Defaults to True.
        file_level: Separate logging level for file output. If None, uses the same as `level`.
        reset: If True, remove all existing handlers before configuring.

    Returns:
        The configured root logger.

    Example:
        >>> setup_logging(level="DEBUG", log_file="logs/debug.log")
        >>> setup_logging(level="INFO", log_dir="logs/")  # Creates timestamped file
    """
    # Convert string level to int if necessary
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    if file_level is None:
        file_level = level
    elif isinstance(file_level, str):
        file_level = getattr(logging, file_level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(min(level, file_level))

    # Remove existing handlers if reset is True
    if reset:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt=date_format)

    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Determine log file path
    if log_file is None and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"dynamix_{timestamp}.log"

    # Add file handler if log file is specified
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def set_level(level: Union[str, int], logger_name: Optional[str] = None) -> None:
    """
    Change the logging level for a specific logger or the root logger.

    Args:
        level: The new logging level.
        logger_name: The name of the logger to modify. If None, modifies the root logger.

    Example:
        >>> set_level("DEBUG")  # Set root logger to DEBUG
        >>> set_level("WARNING", "src.models")  # Set specific logger to WARNING
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)


def suppress_external_loggers(
    loggers: Optional[list[str]] = None, level: Union[str, int] = "WARNING"
) -> None:
    """
    Suppress verbose logging from external libraries.

    Args:
        loggers: List of logger names to suppress. If None, suppresses common verbose loggers.
        level: The level to set for suppressed loggers.

    Example:
        >>> suppress_external_loggers()  # Suppress default list
        >>> suppress_external_loggers(["urllib3", "requests"], level="ERROR")
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.WARNING)

    default_loggers = [
        "transformers",
        "transformers.modeling_utils",
        "transformers.configuration_utils",
        "transformers.tokenization_utils_base",
        "datasets",
        "urllib3",
        "PIL",
        "httpx",
        "httpcore",
        "accelerate",
        "filelock",
    ]

    loggers_to_suppress = loggers if loggers is not None else default_loggers

    for logger_name in loggers_to_suppress:
        logging.getLogger(logger_name).setLevel(level)


class LogContext:
    """
    Context manager for temporarily changing the log level.

    Example:
        >>> logger = get_logger(__name__)
        >>> with LogContext("DEBUG"):
        ...     logger.debug("This will be shown")
        >>> logger.debug("This might not be shown")
    """

    def __init__(
        self,
        level: Union[str, int],
        logger_name: Optional[str] = None,
    ):
        """
        Initialize the log context.

        Args:
            level: The temporary logging level.
            logger_name: The logger to modify. If None, modifies the root logger.
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)

        self.level = level
        self.logger = logging.getLogger(logger_name)
        self.original_level = self.logger.level

    def __enter__(self):
        self.logger.setLevel(self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)
        return False


# Initialize default logging on import (can be reconfigured later)
_initialized = False


def _ensure_initialized():
    """Ensure basic logging is configured."""
    global _initialized
    if not _initialized:
        # Check if root logger has handlers (already configured elsewhere)
        if not logging.getLogger().handlers:
            setup_logging(level="INFO", console=True)
        _initialized = True


# Convenience aliases for log levels
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL
