"""
Scoring interface for enhanced_fastmpnndesign.

Handles loading and using custom scoring modules.
"""

import sys
import os
import importlib.util
from pathlib import Path
from typing import Any, Optional, List, Callable
import pandas as pd

# Add package directory to path for standalone execution
_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from logging_config import get_logger
from utils import dump_scorefile

logger = get_logger("scoring")


class ScoringModule:
    """
    Wrapper for a custom scoring module.

    The scoring module must implement:
    - score_design(pose, scorefxn, catres_seqpos) -> DataFrame
    - filter_scores(df) -> DataFrame
    - filters: dict of filter criteria
    """

    def __init__(self, module: Any):
        """
        Initialize with a loaded module.

        Args:
            module: Python module with scoring functions
        """
        self._module = module

        # Validate required functions
        if not hasattr(module, 'score_design'):
            raise ValueError("Scoring module must have 'score_design' function")
        if not hasattr(module, 'filter_scores'):
            raise ValueError("Scoring module must have 'filter_scores' function")
        if not hasattr(module, 'filters'):
            raise ValueError("Scoring module must have 'filters' attribute")

        self.score_design: Callable = module.score_design
        self.filter_scores: Callable = module.filter_scores
        self.filters: dict = module.filters

    @property
    def module(self) -> Any:
        """Get the underlying module."""
        return self._module


def load_scoring_module(script_path: Path) -> Optional[ScoringModule]:
    """
    Load a custom scoring module from a Python script.

    Args:
        script_path: Path to scoring script

    Returns:
        ScoringModule wrapper or None if loading fails
    """
    script_path = Path(script_path)

    if not script_path.exists():
        logger.error(f"Scoring script not found: {script_path}")
        return None

    try:
        # Add directory to path
        script_dir = str(script_path.parent)
        if script_dir not in sys.path:
            sys.path.append(script_dir)

        # Load module
        module_name = script_path.stem
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        scoring = ScoringModule(module)
        logger.info(f"Loaded scoring module from {script_path}")
        return scoring

    except Exception as e:
        logger.error(f"Failed to load scoring module: {e}")
        return None


def score_design(
    pose: Any,
    scorefxn: Any,
    catres_seqpos: List[int],
    scoring_module: Optional[ScoringModule] = None
) -> pd.DataFrame:
    """
    Score a design using the scoring module or basic scoring.

    Args:
        pose: PyRosetta Pose object
        scorefxn: ScoreFunction to use
        catres_seqpos: List of catalytic residue positions
        scoring_module: Optional custom scoring module

    Returns:
        DataFrame with scores
    """
    if scoring_module is not None:
        return scoring_module.score_design(pose, scorefxn, catres_seqpos)

    # Basic scoring if no custom module
    total_score = scorefxn(pose)

    scores = {
        'total_score': [total_score],
    }

    # Add constraint scores if present
    for term in ['atom_pair_constraint', 'angle_constraint', 'dihedral_constraint']:
        if term in pose.scores:
            scores[term] = [pose.scores[term]]

    return pd.DataFrame(scores)


def filter_design(
    scores_df: pd.DataFrame,
    scoring_module: Optional[ScoringModule] = None
) -> bool:
    """
    Check if a design passes filters.

    Args:
        scores_df: DataFrame with scores
        scoring_module: Optional custom scoring module

    Returns:
        True if design passes filters
    """
    if scoring_module is None:
        return True

    filtered = scoring_module.filter_scores(scores_df)
    return len(filtered) > 0


def save_scores(
    scores_df: pd.DataFrame,
    description: str,
    scorefile_path: Path
) -> None:
    """
    Save scores to a scorefile.

    Args:
        scores_df: DataFrame with scores
        description: Design description (added to DataFrame)
        scorefile_path: Path to scorefile
    """
    scores_df = scores_df.copy()
    scores_df['description'] = description
    dump_scorefile(scores_df, str(scorefile_path))
