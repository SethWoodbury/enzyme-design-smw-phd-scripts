"""
Scorefunction setup for enhanced_fastmpnndesign.

Handles scorefunction creation and constraint weight configuration.
"""

import sys
from pathlib import Path
from typing import Optional, Any

# Add package directory to path for standalone execution
_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from constants import (
    DEFAULT_SCOREFUNCTION,
    CONSTRAINT_WEIGHT_ATOM_PAIR,
    CONSTRAINT_WEIGHT_ANGLE,
    CONSTRAINT_WEIGHT_DIHEDRAL,
    CART_BONDED_WEIGHT,
    PRO_CLOSE_WEIGHT
)
from logging_config import get_logger

logger = get_logger("scorefunction")


def create_scorefunction(
    name: str = DEFAULT_SCOREFUNCTION,
    with_constraints: bool = False
) -> Any:
    """
    Create a PyRosetta scorefunction.

    Args:
        name: Scorefunction name (e.g., 'beta_jan25', 'ref2015')
        with_constraints: If True, set constraint weights

    Returns:
        PyRosetta ScoreFunction object
    """
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import score_type_from_name

    # Create scorefunction
    if name.startswith('beta'):
        # For beta scorefunctions, use get_fa_scorefxn() after init with -beta flag
        sfx = pyrosetta.get_fa_scorefxn()
    else:
        sfx = pyrosetta.create_score_function(name)

    logger.info(f"Created scorefunction: {name}")

    # Add constraint weights if requested
    if with_constraints:
        sfx.set_weight(score_type_from_name("atom_pair_constraint"), CONSTRAINT_WEIGHT_ATOM_PAIR)
        sfx.set_weight(score_type_from_name("angle_constraint"), CONSTRAINT_WEIGHT_ANGLE)
        sfx.set_weight(score_type_from_name("dihedral_constraint"), CONSTRAINT_WEIGHT_DIHEDRAL)
        logger.debug(f"Added constraint weights: atom_pair={CONSTRAINT_WEIGHT_ATOM_PAIR}, "
                    f"angle={CONSTRAINT_WEIGHT_ANGLE}, dihedral={CONSTRAINT_WEIGHT_DIHEDRAL}")

    return sfx


def setup_scorefunction(
    name: str = DEFAULT_SCOREFUNCTION,
    cstfile: Optional[Path] = None
) -> Any:
    """
    Setup scorefunction with constraint weights if cstfile provided.

    This matches the behavior from original script lines 149-155.

    Args:
        name: Scorefunction name
        cstfile: Path to constraint file (triggers constraint weight setup)

    Returns:
        PyRosetta ScoreFunction object
    """
    with_constraints = cstfile is not None
    return create_scorefunction(name, with_constraints=with_constraints)


def setup_cartesian_scorefunction(
    base_sfx: Any,
    cart_bonded_weight: float = CART_BONDED_WEIGHT,
    pro_close_weight: float = PRO_CLOSE_WEIGHT
) -> Any:
    """
    Setup scorefunction for Cartesian relaxation.

    Creates a clone of the base scorefunction with Cartesian-specific weights.
    This matches the behavior from original script lines 240-243.

    Args:
        base_sfx: Base scorefunction to clone
        cart_bonded_weight: Weight for cart_bonded term
        pro_close_weight: Weight for pro_close term

    Returns:
        Cloned scorefunction with Cartesian weights
    """
    from pyrosetta.rosetta.core.scoring import score_type_from_name

    sfx_cart = base_sfx.clone()
    sfx_cart.set_weight(score_type_from_name("cart_bonded"), cart_bonded_weight)
    sfx_cart.set_weight(score_type_from_name("pro_close"), pro_close_weight)

    logger.debug(f"Setup Cartesian scorefunction: cart_bonded={cart_bonded_weight}, "
                f"pro_close={pro_close_weight}")

    return sfx_cart


def get_constraint_score(pose: Any, sfx: Any) -> float:
    """
    Calculate total constraint score for a pose.

    Args:
        pose: PyRosetta Pose object
        sfx: ScoreFunction to use

    Returns:
        Sum of all constraint terms
    """
    # Score the pose first
    sfx(pose)

    # Sum constraint terms
    constraint_terms = ['atom_pair_constraint', 'angle_constraint', 'dihedral_constraint',
                       'coordinate_constraint']

    total = 0.0
    for term in constraint_terms:
        if term in pose.scores:
            total += pose.scores[term]

    return total


def log_score_breakdown(pose: Any, sfx: Any, prefix: str = "") -> None:
    """
    Log a breakdown of scorefunction terms for a pose.

    Args:
        pose: PyRosetta Pose object
        sfx: ScoreFunction to use
        prefix: Optional prefix for log messages
    """
    # Score the pose
    total = sfx(pose)

    logger.info(f"{prefix}Total score: {total:.2f}")

    # Log key terms
    key_terms = [
        'fa_atr', 'fa_rep', 'fa_sol', 'fa_elec',
        'hbond_sr_bb', 'hbond_lr_bb', 'hbond_bb_sc', 'hbond_sc',
        'atom_pair_constraint', 'angle_constraint', 'dihedral_constraint',
        'coordinate_constraint', 'cart_bonded'
    ]

    for term in key_terms:
        if term in pose.scores:
            value = pose.scores[term]
            if abs(value) > 0.01:  # Only log non-zero terms
                logger.debug(f"{prefix}  {term}: {value:.2f}")


def scale_weight(sfx: Any, term_name: str, scale: float) -> None:
    """
    Scale a scorefunction term weight.

    Args:
        sfx: ScoreFunction to modify
        term_name: Name of the term to scale
        scale: Scale factor (new_weight = original_weight * scale)
    """
    from pyrosetta.rosetta.core.scoring import score_type_from_name

    try:
        score_type = score_type_from_name(term_name)
        current_weight = sfx.get_weight(score_type)
        new_weight = current_weight * scale if current_weight > 0 else scale
        sfx.set_weight(score_type, new_weight)
        logger.debug(f"Scaled {term_name}: {current_weight:.3f} -> {new_weight:.3f}")
    except Exception as e:
        logger.warning(f"Could not scale weight for {term_name}: {e}")


def set_weight(sfx: Any, term_name: str, weight: float) -> None:
    """
    Set a scorefunction term weight.

    Args:
        sfx: ScoreFunction to modify
        term_name: Name of the term
        weight: New weight value
    """
    from pyrosetta.rosetta.core.scoring import score_type_from_name

    try:
        score_type = score_type_from_name(term_name)
        sfx.set_weight(score_type, weight)
        logger.debug(f"Set {term_name} weight to {weight:.3f}")
    except Exception as e:
        logger.warning(f"Could not set weight for {term_name}: {e}")
