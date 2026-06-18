"""
Logging configuration for Remastered FastMPNNdesign.

Provides centralized logging setup with configurable verbosity levels.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict

from remastered_fastmpnn.constants import (
    LOG_FORMAT_VERBOSE,
    LOG_FORMAT_SIMPLE,
    LOG_DATE_FORMAT,
)

# Module-level logger registry
_loggers: Dict[str, logging.Logger] = {}

# Root logger name for the package
ROOT_LOGGER_NAME = "remastered_fastmpnn"


def setup_logging(
    verbose: bool = False,
    debug: bool = False,
    log_file: Optional[Path] = None,
    level: Optional[int] = None,
) -> logging.Logger:
    """
    Set up logging for the remastered_fastmpnn package.

    Args:
        verbose: Enable INFO level logging with verbose format
        debug: Enable DEBUG level logging (overrides verbose)
        log_file: Optional path to write logs to file
        level: Override log level (if provided, ignores verbose/debug)

    Returns:
        The root logger for the package
    """
    # Determine log level
    if level is not None:
        log_level = level
    elif debug:
        log_level = logging.DEBUG
    elif verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    # Get or create root logger
    root_logger = logging.getLogger(ROOT_LOGGER_NAME)
    root_logger.setLevel(log_level)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Select format based on verbosity
    if debug or verbose:
        fmt = LOG_FORMAT_VERBOSE
    else:
        fmt = LOG_FORMAT_SIMPLE

    formatter = logging.Formatter(fmt, datefmt=LOG_DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if requested)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(log_level)
        # Always use verbose format for file logging
        file_handler.setFormatter(
            logging.Formatter(LOG_FORMAT_VERBOSE, datefmt=LOG_DATE_FORMAT)
        )
        root_logger.addHandler(file_handler)

    # Prevent propagation to root logger
    root_logger.propagate = False

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Module name (will be prefixed with package name if needed)

    Returns:
        Logger instance for the module
    """
    # Ensure name is under our package namespace
    if not name.startswith(ROOT_LOGGER_NAME):
        full_name = f"{ROOT_LOGGER_NAME}.{name}"
    else:
        full_name = name

    if full_name not in _loggers:
        _loggers[full_name] = logging.getLogger(full_name)

    return _loggers[full_name]


def log_section(logger: logging.Logger, title: str, char: str = "=", width: int = 60) -> None:
    """
    Log a section header for visual separation.

    Args:
        logger: Logger to use
        title: Section title
        char: Character to use for the border
        width: Total width of the section header
    """
    border = char * width
    logger.info(border)
    logger.info(title.center(width))
    logger.info(border)


def log_key_value(
    logger: logging.Logger,
    key: str,
    value: str,
    key_width: int = 25,
    level: int = logging.INFO
) -> None:
    """
    Log a key-value pair with consistent formatting.

    Args:
        logger: Logger to use
        key: Key/label
        value: Value to display
        key_width: Width for key column
        level: Log level
    """
    logger.log(level, f"  {key:<{key_width}}: {value}")


class LogContext:
    """
    Context manager for logging with automatic enter/exit messages.

    Usage:
        with LogContext(logger, "Processing residues"):
            # ... do work ...
    """

    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        level: int = logging.INFO
    ):
        self.logger = logger
        self.operation = operation
        self.level = level

    def __enter__(self):
        self.logger.log(self.level, f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.logger.log(self.level, f"Completed: {self.operation}")
        else:
            self.logger.error(f"Failed: {self.operation} - {exc_val}")
        return False  # Don't suppress exceptions
