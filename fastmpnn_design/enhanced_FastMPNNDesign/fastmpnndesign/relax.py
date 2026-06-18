"""
Pre-relaxation for enhanced_fastmpnndesign.

Handles Cartesian FastRelax with constraints for structure preparation.
"""

import sys
from pathlib import Path
from typing import List, Any, Optional

# Add package directory to path for standalone execution
_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import RelaxConfig
from constraints import CSTs
from scorefunction import setup_cartesian_scorefunction
from constants import DESIGN_UTILS_PATH
from logging_config import get_logger

logger = get_logger("relax")


def find_clashing_residues(
    pose: Any,
    ligand_seqpos: int,
    keep_pos: List[int]
) -> List[int]:
    """
    Find residues with sidechains clashing with the ligand.

    This wraps design_utils.find_clashes_between_target_and_sidechains.

    Args:
        pose: PyRosetta Pose object
        ligand_seqpos: Sequence position of ligand
        keep_pos: Positions to exclude from clash check

    Returns:
        List of clashing residue positions
    """
    if DESIGN_UTILS_PATH not in sys.path:
        sys.path.append(DESIGN_UTILS_PATH)

    import design_utils

    ligand = pose.residue(ligand_seqpos)

    # Get non-hydrogen atom indices
    target_atoms = [
        n for n in range(1, ligand.natoms() + 1)
        if not ligand.atom_is_hydrogen(n)
    ]

    # Find clashing residues
    clashes = design_utils.find_clashes_between_target_and_sidechains(
        pose, ligand_seqpos,
        target_atoms=target_atoms,
        residues=[n for n in range(1, pose.size()) if n not in keep_pos]
    )

    # Filter out GLY, ALA, PRO and keep_pos
    clashes = [
        x for x in clashes
        if pose.residue(x).name3() not in ["ALA", "GLY", "PRO"]
        and x not in keep_pos
    ]

    if clashes:
        logger.info(f"Found {len(clashes)} clashing residues")

    return clashes


def mutate_residues(pose: Any, positions: List[int], aa: str = "ALA") -> Any:
    """
    Mutate specified residues to a given amino acid.

    This wraps design_utils.mutate_residues.

    Args:
        pose: PyRosetta Pose object
        positions: List of positions to mutate
        aa: Target amino acid (default ALA)

    Returns:
        Mutated pose
    """
    if DESIGN_UTILS_PATH not in sys.path:
        sys.path.append(DESIGN_UTILS_PATH)

    import design_utils
    return design_utils.mutate_residues(pose, positions, aa)


def setup_fastrelax(
    scorefxn: Any,
    crude: bool = True,
    disable_min_resons: Optional[List[int]] = None
) -> Any:
    """
    Setup FastRelax mover.

    This wraps design_utils.setup_fastrelax.

    Args:
        scorefxn: PyRosetta ScoreFunction
        crude: Use crude (fast) settings
        disable_min_resons: Residues to exclude from minimization

    Returns:
        FastRelax mover
    """
    if DESIGN_UTILS_PATH not in sys.path:
        sys.path.append(DESIGN_UTILS_PATH)

    import design_utils
    return design_utils.setup_fastrelax(
        scorefxn,
        crude=crude,
        disable_min_resons=disable_min_resons or []
    )


def run_pre_relaxation(
    pose: Any,
    scorefxn: Any,
    cst_mover: Optional[CSTs],
    keep_pos: List[int],
    config: RelaxConfig,
    ligand_seqpos: Optional[int] = None
) -> Any:
    """
    Run pre-relaxation with constraints.

    This replicates the pre-relaxation logic from original script lines 234-261.

    Args:
        pose: PyRosetta Pose object
        scorefxn: Base scorefunction
        cst_mover: Constraint manager (optional)
        keep_pos: Positions to keep fixed
        config: Relaxation configuration
        ligand_seqpos: Ligand position (defaults to last residue)

    Returns:
        Pre-relaxed pose
    """
    if ligand_seqpos is None:
        ligand_seqpos = pose.size()

    logger.info(f"Running pre-relaxation (cartesian={config.cartesian})")

    # Setup Cartesian scorefunction if needed
    if config.cartesian:
        sfx_cart = setup_cartesian_scorefunction(
            scorefxn,
            cart_bonded_weight=config.cart_bonded_weight,
            pro_close_weight=config.pro_close_weight
        )
    else:
        sfx_cart = scorefxn.clone()

    # Setup FastRelax
    fastRelax = setup_fastrelax(
        sfx_cart,
        crude=config.crude,
        disable_min_resons=[ligand_seqpos]
    )
    fastRelax.cartesian(config.cartesian)

    # Find and mutate clashing residues to ALA
    clashes = find_clashing_residues(pose, ligand_seqpos, keep_pos)
    if clashes:
        logger.info(f"Mutating {len(clashes)} clashing residues to ALA")
        pose = mutate_residues(pose, clashes, "ALA")

    # Add constraints if available
    if cst_mover is not None:
        cst_mover.cst_io().add_constraints_to_pose(pose, sfx_cart, True)

    # Run FastRelax
    fastRelax.apply(pose)
    sfx_cart(pose)

    # Log constraint score
    if cst_mover is not None:
        cst_score = cst_mover.cst_score(pose)
        logger.info(f"CST score after pre-relaxation: {cst_score:.2f}")

    return pose


def repack_sidechains(pose: Any, scorefxn: Any) -> Any:
    """
    Repack sidechains.

    This wraps design_utils.repack.

    Args:
        pose: PyRosetta Pose object
        scorefxn: ScoreFunction for packing

    Returns:
        Repacked pose
    """
    if DESIGN_UTILS_PATH not in sys.path:
        sys.path.append(DESIGN_UTILS_PATH)

    import design_utils
    return design_utils.repack(pose, scorefxn)


def thread_sequence_to_pose(pose: Any, sequence: str) -> Any:
    """
    Thread a new sequence onto a pose backbone.

    This wraps design_utils.thread_seq_to_pose.

    Args:
        pose: PyRosetta Pose object
        sequence: New sequence string

    Returns:
        Pose with new sequence
    """
    if DESIGN_UTILS_PATH not in sys.path:
        sys.path.append(DESIGN_UTILS_PATH)

    import design_utils
    return design_utils.thread_seq_to_pose(pose, sequence)


def fix_catalytic_residue_rotamers(
    pose: Any,
    reference_pose: Any,
    catres_seqpos: List[int]
) -> Any:
    """
    Fix catalytic residue rotamers to match reference.

    This wraps design_utils.fix_catalytic_residue_rotamers.

    Args:
        pose: Target pose
        reference_pose: Reference pose with correct rotamers
        catres_seqpos: List of catalytic residue positions (or dict)

    Returns:
        Pose with fixed rotamers
    """
    if DESIGN_UTILS_PATH not in sys.path:
        sys.path.append(DESIGN_UTILS_PATH)

    import design_utils
    return design_utils.fix_catalytic_residue_rotamers(pose, reference_pose, catres_seqpos)
