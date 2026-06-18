"""
PyRosetta initialization for enhanced_fastmpnndesign.

Handles PyRosetta initialization with proper multithreading,
scorefunction selection (including beta_jan25), and params loading.
"""

import sys
import os
from pathlib import Path
from typing import List, Optional

# Add package directory to path for standalone execution
_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from constants import (
    DEFAULT_SCOREFUNCTION, DEFAULT_DALPHABALL, BETA_SCOREFUNCTIONS
)
from utils import get_nproc
from logging_config import get_logger

logger = get_logger("rosetta_init")

# Global flag to track initialization
_PYROSETTA_INITIALIZED = False


def initialize_pyrosetta(
    params: Optional[List[Path]] = None,
    scorefunction: str = DEFAULT_SCOREFUNCTION,
    dalphaball: Optional[Path] = None,
    multithreading: bool = True,
    preserve_header: bool = True,
    extra_options: Optional[str] = None,
    quiet: bool = False
) -> None:
    """
    Initialize PyRosetta with appropriate settings.

    Handles:
    - Loading params files for ligands/NCAAs
    - Setting up multithreading based on environment
    - Configuring beta scorefunctions (beta_jan25, beta_nov16, etc.)
    - DAlphaBall for hole detection

    Args:
        params: List of paths to params files
        scorefunction: Scorefunction name (default: beta_jan25)
        dalphaball: Path to DAlphaBall executable
        multithreading: Enable multithreading
        preserve_header: Preserve REMARK lines in output PDBs
        extra_options: Additional PyRosetta options
        quiet: Suppress PyRosetta output
    """
    global _PYROSETTA_INITIALIZED

    if _PYROSETTA_INITIALIZED:
        logger.debug("PyRosetta already initialized")
        return

    import pyrosetta

    options = []

    # Load params files
    if params:
        params_str = " ".join([str(p) for p in params])
        options.append(f"-extra_res_fa {params_str}")
        logger.info(f"Loading params files: {params_str}")

    # DAlphaBall for hole detection
    if dalphaball is None:
        dalphaball = Path(DEFAULT_DALPHABALL)

    if dalphaball.exists():
        options.append(f"-dalphaball {dalphaball}")
    else:
        logger.warning(f"DAlphaBall not found at {dalphaball}")

    # Beta scorefunction handling
    # Extract base scorefunction name (remove _cart, _cst suffixes)
    sf_base = scorefunction.split('_cart')[0].split('_cst')[0]
    if sf_base in BETA_SCOREFUNCTIONS:
        options.append(f"-{sf_base}")
        logger.info(f"Using beta scorefunction: {sf_base}")

    # Preserve header (REMARK lines)
    if preserve_header:
        options.append("-run:preserve_header")

    # Multithreading
    if multithreading:
        nproc = get_nproc()
        if nproc > 1:
            options.append("-multithreading true")
            options.append(f"-multithreading:total_threads {nproc}")
            options.append(f"-multithreading:interaction_graph_threads {nproc}")
            logger.info(f"Multithreading enabled with {nproc} threads")

    # Additional options
    if extra_options:
        options.append(extra_options)

    # Mute output if quiet
    if quiet:
        options.append("-mute all")

    # Build options string
    options_str = " ".join(options)
    logger.debug(f"PyRosetta options: {options_str}")

    # Initialize
    pyrosetta.init(options_str)
    _PYROSETTA_INITIALIZED = True

    logger.info("PyRosetta initialized successfully")


def is_initialized() -> bool:
    """Check if PyRosetta has been initialized."""
    return _PYROSETTA_INITIALIZED


def get_init_options(
    params: Optional[List[Path]] = None,
    scorefunction: str = DEFAULT_SCOREFUNCTION,
    dalphaball: Optional[Path] = None,
    multithreading: bool = True,
    preserve_header: bool = True
) -> str:
    """
    Build PyRosetta init options string without initializing.

    Useful for logging or container execution.

    Args:
        params: List of paths to params files
        scorefunction: Scorefunction name
        dalphaball: Path to DAlphaBall executable
        multithreading: Enable multithreading
        preserve_header: Preserve REMARK lines

    Returns:
        Options string for pyrosetta.init()
    """
    options = []

    if params:
        params_str = " ".join([str(p) for p in params])
        options.append(f"-extra_res_fa {params_str}")

    if dalphaball is None:
        dalphaball = Path(DEFAULT_DALPHABALL)
    if dalphaball.exists():
        options.append(f"-dalphaball {dalphaball}")

    sf_base = scorefunction.split('_cart')[0].split('_cst')[0]
    if sf_base in BETA_SCOREFUNCTIONS:
        options.append(f"-{sf_base}")

    if preserve_header:
        options.append("-run:preserve_header")

    if multithreading:
        nproc = get_nproc()
        if nproc > 1:
            options.append("-multithreading true")
            options.append(f"-multithreading:total_threads {nproc}")
            options.append(f"-multithreading:interaction_graph_threads {nproc}")

    return " ".join(options)


def reset_initialization():
    """Reset initialization flag (for testing)."""
    global _PYROSETTA_INITIALIZED
    _PYROSETTA_INITIALIZED = False
