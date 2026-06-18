"""
Logging configuration for enhanced_fastmpnndesign.

Provides consistent logging setup across all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# Global logger cache
_loggers: dict = {}

# Default format
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
VERBOSE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(funcName)-20s | %(message)s"
SIMPLE_FORMAT = "%(levelname)-8s | %(message)s"


def setup_logging(
    verbose: bool = True,
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging for the entire package.

    Args:
        verbose: If True, use detailed format; otherwise simple format
        log_file: Optional path to write logs to file
        level: Logging level (default INFO)

    Returns:
        Root logger for the package
    """
    # Get the package root logger
    root_logger = logging.getLogger("fastmpnndesign")
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Choose format
    fmt = VERBOSE_FORMAT if verbose else SIMPLE_FORMAT

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    root_logger.addHandler(console_handler)

    # File handler if requested
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(VERBOSE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
        root_logger.addHandler(file_handler)

    # Prevent propagation to root logger
    root_logger.propagate = False

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Module name (will be prefixed with 'fastmpnndesign.')

    Returns:
        Logger instance
    """
    full_name = f"fastmpnndesign.{name}" if not name.startswith("fastmpnndesign") else name

    if full_name not in _loggers:
        _loggers[full_name] = logging.getLogger(full_name)

    return _loggers[full_name]


def set_level(level: int) -> None:
    """Set logging level for all package loggers."""
    root_logger = logging.getLogger("fastmpnndesign")
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)


def set_quiet() -> None:
    """Set logging to WARNING level (quiet mode)."""
    set_level(logging.WARNING)


def set_debug() -> None:
    """Set logging to DEBUG level."""
    set_level(logging.DEBUG)
