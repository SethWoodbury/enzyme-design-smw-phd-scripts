"""
Logging configuration for fastmpnndesign.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Package-wide logger
LOGGER_NAME = "fastmpnndesign"


def setup_logging(
    verbose: bool = True,
    log_file: Optional[Path] = None,
    logger_name: str = LOGGER_NAME
) -> logging.Logger:
    """
    Configure logging for the fastmpnndesign package.

    Args:
        verbose: If True, set level to DEBUG; otherwise INFO.
        log_file: Optional path to write logs to file.
        logger_name: Name for the logger.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(logger_name)

    # Clear any existing handlers
    logger.handlers.clear()

    # Set level based on verbosity
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        fmt='%(levelname)-8s | %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter if not verbose else detailed_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always detailed in file
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Sub-logger name. If None, returns the package logger.

    Returns:
        Logger instance.
    """
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


def quiet_external_loggers() -> None:
    """Reduce verbosity of external library loggers."""
    for logger_name in ['pyrosetta', 'rosetta', 'numpy', 'matplotlib']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
