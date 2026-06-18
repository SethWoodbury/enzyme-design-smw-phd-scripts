"""Logging configuration utilities."""

import logging
import sys
from typing import Optional

from .constants import DATE_FORMAT, LOG_FORMAT

# Global logger for pipeline-wide verbose output
_verbose_mode = False


def configure_logging(verbosity: int) -> None:
    """Configure logging based on verbosity level.

    Args:
        verbosity: 0=WARNING, 1=INFO, 2+=DEBUG
    """
    global _verbose_mode
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
        _verbose_mode = True
    elif verbosity >= 2:
        level = logging.DEBUG
        _verbose_mode = True

    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT, force=True)


def is_verbose() -> bool:
    """Check if verbose mode is enabled."""
    return _verbose_mode


def print_section_header(title: str, char: str = "=") -> None:
    """Print a section header for clear visual separation."""
    width = 80
    print()
    print(char * width)
    print(f" {title}")
    print(char * width)


def print_subsection_header(title: str) -> None:
    """Print a subsection header."""
    print()
    print(f"--- {title} ---")


def print_key_value(key: str, value: str, indent: int = 2) -> None:
    """Print a key-value pair with consistent formatting."""
    prefix = " " * indent
    print(f"{prefix}{key}: {value}")


def print_list_item(item: str, indent: int = 4) -> None:
    """Print a list item with bullet point."""
    prefix = " " * indent
    print(f"{prefix}• {item}")


def print_dict_summary(data: dict, title: Optional[str] = None, indent: int = 2) -> None:
    """Print a dictionary as a formatted summary."""
    if title:
        print_subsection_header(title)
    prefix = " " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            for k, v in value.items():
                print(f"{prefix}  {k}: {v}")
        elif isinstance(value, list):
            print(f"{prefix}{key}: [{len(value)} items]")
        else:
            print(f"{prefix}{key}: {value}")
