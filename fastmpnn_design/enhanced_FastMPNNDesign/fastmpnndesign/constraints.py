"""
Constraint handling for enhanced_fastmpnndesign.

Provides the CSTs class for enzyme design constraint management.
"""

import sys
from pathlib import Path
from typing import Optional, Any

# Add package directory to path for standalone execution
_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from logging_config import get_logger

logger = get_logger("constraints")


class CSTs:
    """
    Wrapper for enzyme design constraints from Matcher/Enzdes.

    This matches the CSTs class from the original script (lines 25-48).
    Handles reading constraint files and applying/removing constraints to poses.
    """

    def __init__(self, cstfile: Path, scorefxn: Any):
        """
        Initialize constraint manager.

        Args:
            cstfile: Path to matcher/enzdes constraint file
            scorefxn: PyRosetta ScoreFunction (for constraint scoring)
        """
        import pyrosetta.rosetta.protocols.enzdes
        import pyrosetta.rosetta.protocols.toolbox.match_enzdes_util
        import pyrosetta.rosetta.core.chemical

        self._scorefxn = scorefxn
        self._cstfile = Path(cstfile)

        if not self._cstfile.exists():
            raise FileNotFoundError(f"Constraint file not found: {cstfile}")

        # Initialize movers and managers
        self._addcst_mover = pyrosetta.rosetta.protocols.enzdes.AddOrRemoveMatchCsts()

        chem_manager = pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance()
        residue_type_set = chem_manager.residue_type_set("fa_standard")

        self._cst_io = pyrosetta.rosetta.protocols.toolbox.match_enzdes_util.EnzConstraintIO(
            residue_type_set
        )
        self._cst_io.read_enzyme_cstfile(str(cstfile))

        logger.info(f"Loaded enzyme constraints from {cstfile}")

    def add_cst(self, pose: Any) -> None:
        """
        Add constraints to a pose.

        Args:
            pose: PyRosetta Pose object
        """
        self._cst_io.add_constraints_to_pose(pose, self._scorefxn, True)
        logger.debug("Added constraints to pose")

    def remove_cst(self, pose: Any) -> None:
        """
        Remove constraints from a pose.

        Args:
            pose: PyRosetta Pose object
        """
        self._cst_io.remove_constraints_from_pose(pose, True, True)
        logger.debug("Removed constraints from pose")

    def cst_io(self) -> Any:
        """
        Get the internal EnzConstraintIO object.

        Returns:
            EnzConstraintIO object for advanced operations
        """
        return self._cst_io

    def cst_score(self, pose: Any) -> float:
        """
        Calculate total constraint score for a pose.

        Args:
            pose: PyRosetta Pose object

        Returns:
            Sum of all constraint terms
        """
        # Score the pose
        self._scorefxn(pose)

        # Sum constraint terms
        total = 0.0
        for term in ['atom_pair_constraint', 'angle_constraint', 'dihedral_constraint']:
            if term in pose.scores:
                total += pose.scores[term]

        return total

    @property
    def cstfile(self) -> Path:
        """Get path to constraint file."""
        return self._cstfile


def setup_constraints(
    cstfile: Optional[Path],
    scorefxn: Any
) -> Optional[CSTs]:
    """
    Setup constraint manager if cstfile provided.

    Args:
        cstfile: Path to constraint file (or None)
        scorefxn: PyRosetta ScoreFunction

    Returns:
        CSTs object or None if no cstfile
    """
    if cstfile is None:
        return None

    cstfile = Path(cstfile)
    if not cstfile.exists():
        logger.warning(f"Constraint file not found: {cstfile}")
        return None

    return CSTs(cstfile, scorefxn)
