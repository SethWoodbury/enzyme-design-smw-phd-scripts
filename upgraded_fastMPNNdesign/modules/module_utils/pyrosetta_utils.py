"""Helpers for importing PyRosetta with fallback paths and version checks."""
import logging
import os
import sys
import re
from typing import Iterable, Optional, Tuple

from .constants import (
    DEFAULT_PYROSETTA_FALLBACK_PATHS,
    DEFAULT_PYROSETTA_REQUIRED_VERSION,
    DEFAULT_PYROSETTA_MIN_VERSION,
    DEFAULT_PYROSETTA_DIR,
)

LOGGER = logging.getLogger(__name__)


def _expand_paths(paths: Iterable[str]) -> Iterable[str]:
    for path in paths:
        if not path:
            continue
        yield path
        if os.path.isdir(path):
            yield os.path.join(path, "setup")
            yield os.path.join(path, "setup", "pyrosetta")


def _add_fallback_paths(paths: Optional[Iterable[str]] = None) -> None:
    """Add fallback paths to sys.path if they exist."""
    for path in _expand_paths(paths or []):
        if not path:
            continue
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            LOGGER.debug(f"Added PyRosetta fallback path: {path}")


def _parse_version(version: str) -> Tuple[int, ...]:
    parts = re.findall(r"\d+", version or "")
    return tuple(int(p) for p in parts) if parts else tuple()


def _get_pyrosetta_version() -> str:
    try:
        import pyrosetta  # noqa: F401
        if hasattr(pyrosetta, "__version__"):
            return str(pyrosetta.__version__)
        if hasattr(pyrosetta, "rosetta_version"):
            return str(pyrosetta.rosetta_version())
        if hasattr(pyrosetta, "version"):
            return str(pyrosetta.version())
    except Exception:
        pass
    return ""


def log_pyrosetta_version(context: str = "") -> None:
    """Log PyRosetta version if available."""
    version = _get_pyrosetta_version()
    if version:
        prefix = f"{context}: " if context else ""
        LOGGER.info(f"{prefix}PyRosetta version: {version}")


def _check_version(required: str, minimum: str) -> bool:
    if not required and not minimum:
        return True
    version = _get_pyrosetta_version()
    if not version:
        LOGGER.warning("Could not determine PyRosetta version")
        return False
    if required:
        if required not in version:
            LOGGER.warning(f"PyRosetta version mismatch: required '{required}', got '{version}'")
            return False
    if minimum:
        try:
            if _parse_version(version) < _parse_version(minimum):
                LOGGER.warning(f"PyRosetta version too old: minimum '{minimum}', got '{version}'")
                return False
        except Exception:
            LOGGER.warning(f"PyRosetta version compare failed: '{version}' vs minimum '{minimum}'")
            return False
    return True


def try_import_pyrosetta(
    fallback_paths: Optional[Iterable[str]] = None,
    required_version: Optional[str] = None,
    min_version: Optional[str] = None,
) -> bool:
    """Try importing PyRosetta; if it fails, add fallback paths and retry.

    Supports optional version checks via environment variables or constants:
      - PYROSETTA_REQUIRED_VERSION (exact substring match)
      - PYROSETTA_MIN_VERSION (numeric >=)
    """
    required_version = required_version or os.environ.get("PYROSETTA_REQUIRED_VERSION") or DEFAULT_PYROSETTA_REQUIRED_VERSION
    min_version = min_version or os.environ.get("PYROSETTA_MIN_VERSION") or DEFAULT_PYROSETTA_MIN_VERSION

    try:
        import pyrosetta  # noqa: F401
        ok = _check_version(required_version, min_version)
        if ok:
            log_pyrosetta_version("Import")
        return ok
    except Exception:
        env_path = os.environ.get("PYROSETTA_DIR")
        paths = list(fallback_paths or DEFAULT_PYROSETTA_FALLBACK_PATHS)
        if DEFAULT_PYROSETTA_DIR and DEFAULT_PYROSETTA_DIR not in paths:
            paths.insert(0, DEFAULT_PYROSETTA_DIR)
        if env_path:
            paths.insert(0, env_path)
        _add_fallback_paths(paths)
        try:
            import pyrosetta  # noqa: F401
            ok = _check_version(required_version, min_version)
            if ok:
                log_pyrosetta_version("Fallback import")
            return ok
        except Exception as e:
            LOGGER.debug(f"PyRosetta import failed after fallback: {e}")
            return False
