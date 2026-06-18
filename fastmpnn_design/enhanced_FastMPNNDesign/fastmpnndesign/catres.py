"""
Catalytic residue handling for enhanced_fastmpnndesign.

Provides utilities for managing catalytic residues throughout the design pipeline.
"""

import sys
from pathlib import Path
from typing import List, Optional, Set, Dict, Any, Tuple

# Add package directory to path for standalone execution
_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import CatalyticResidue, CatresConfig
from logging_config import get_logger

logger = get_logger("catres")


def get_fixed_catres(
    catres_list: List[CatalyticResidue],
    config: CatresConfig
) -> Tuple[List[CatalyticResidue], List[CatalyticResidue]]:
    """
    Determine which catalytic residues should be fixed during design.

    Based on config settings:
    - By default, ALL catres are fixed (fix_all_catres=True)
    - If catres_subset is specified, only subset gets tight geometry constraints
    - If redesign_non_subset_catres=True, non-subset catres can be redesigned

    Args:
        catres_list: Full list of catalytic residues
        config: Catalytic residue configuration

    Returns:
        Tuple of (fixed_catres, designable_catres)
    """
    # Split into subset and non-subset
    if config.catres_subset is None:
        # All catres in subset (tight constraints), none designable
        subset = catres_list.copy()
        non_subset = []
    else:
        subset = [cr for cr in catres_list if cr.catres_index in config.catres_subset]
        non_subset = [cr for cr in catres_list if cr.catres_index not in config.catres_subset]

    # Determine what's fixed vs designable
    if config.redesign_non_subset_catres:
        # Non-subset catres can be redesigned
        fixed = subset
        designable = non_subset
    else:
        # All catres are fixed (default behavior)
        fixed = catres_list.copy()
        designable = []

    logger.info(f"Catres handling: {len(fixed)} fixed, {len(designable)} designable")
    if fixed:
        logger.debug(f"  Fixed catres: {[cr.pdb_resid for cr in fixed]}")
    if designable:
        logger.debug(f"  Designable catres: {[cr.pdb_resid for cr in designable]}")

    return fixed, designable


def get_catres_seqpos_list(
    catres_list: List[CatalyticResidue],
    pose=None
) -> List[int]:
    """
    Get list of sequence positions for catalytic residues.

    Args:
        catres_list: List of catalytic residues
        pose: Optional PyRosetta Pose for mapping if seqpos not set

    Returns:
        List of sequence positions
    """
    seqpos_list = []

    for cr in catres_list:
        if cr.seqpos is not None:
            seqpos_list.append(cr.seqpos)
        elif pose is not None:
            # Try to find seqpos in pose
            pdb_info = pose.pdb_info()
            for i in range(1, pose.size() + 1):
                if (pdb_info.chain(i) == cr.chain and
                    pdb_info.number(i) == cr.resnum):
                    cr.seqpos = i
                    seqpos_list.append(i)
                    break
            else:
                logger.warning(f"Could not find seqpos for catres {cr.pdb_resid}")
        else:
            logger.warning(f"Catres {cr.pdb_resid} has no seqpos and no pose provided")

    return seqpos_list


def validate_catres_types(catres_list: List[CatalyticResidue], pose) -> bool:
    """
    Validate that catalytic residue types in pose match REMARK 666 definitions.

    Args:
        catres_list: List of catalytic residues from REMARK 666
        pose: PyRosetta Pose object

    Returns:
        True if all match, False otherwise
    """
    all_valid = True

    for cr in catres_list:
        if cr.seqpos is None:
            logger.warning(f"Catres {cr.pdb_resid} has no seqpos assigned")
            all_valid = False
            continue

        pose_resname = pose.residue(cr.seqpos).name3()
        if pose_resname != cr.resname:
            logger.warning(
                f"Catres type mismatch at {cr.pdb_resid}: "
                f"REMARK 666 says {cr.resname}, pose has {pose_resname}"
            )
            all_valid = False

    return all_valid


def get_catres_for_hbond_keeper(
    catres_list: List[CatalyticResidue],
    ligand_seqpos: int
) -> List[int]:
    """
    Get list of target residues for HBondKeeper task operation.

    Includes ligand and all catalytic residues.

    Args:
        catres_list: List of catalytic residues
        ligand_seqpos: Sequence position of ligand

    Returns:
        List of sequence positions [ligand_seqpos, catres1, catres2, ...]
    """
    targets = [ligand_seqpos]

    for cr in catres_list:
        if cr.seqpos is not None:
            targets.append(cr.seqpos)

    return targets


def get_catres_by_index(
    catres_list: List[CatalyticResidue],
    indices: List[int]
) -> List[CatalyticResidue]:
    """
    Get subset of catalytic residues by their indices.

    Args:
        catres_list: Full list of catalytic residues
        indices: List of catres indices to select

    Returns:
        Subset of catalytic residues
    """
    return [cr for cr in catres_list if cr.catres_index in indices]


def catres_summary(catres_list: List[CatalyticResidue]) -> str:
    """
    Generate a summary string of catalytic residues.

    Args:
        catres_list: List of catalytic residues

    Returns:
        Formatted summary string
    """
    if not catres_list:
        return "No catalytic residues defined"

    lines = ["Catalytic Residues:"]
    for cr in sorted(catres_list, key=lambda x: x.catres_index):
        seqpos_str = f" (seqpos {cr.seqpos})" if cr.seqpos else ""
        lines.append(
            f"  {cr.catres_index:2d}: {cr.chain}{cr.resnum:4d} {cr.resname}{seqpos_str}"
        )

    return "\n".join(lines)


def build_keep_pos_list(
    catres_list: List[CatalyticResidue],
    user_keep_pos: Optional[List[int]] = None,
    pose=None
) -> List[int]:
    """
    Build complete list of positions to keep fixed.

    Combines:
    - User-specified keep positions
    - All catalytic residue positions
    - Motif-labeled positions (if pose provided)

    Args:
        catres_list: List of catalytic residues
        user_keep_pos: User-specified positions to keep
        pose: Optional PyRosetta Pose for motif label detection

    Returns:
        Combined list of positions to keep (no duplicates)
    """
    keep_pos = set()

    # Add user-specified positions
    if user_keep_pos:
        keep_pos.update(user_keep_pos)

    # Add catres positions
    for cr in catres_list:
        if cr.seqpos is not None:
            keep_pos.add(cr.seqpos)

    # Add motif-labeled positions from pose
    if pose is not None:
        try:
            import pyrosetta.rosetta.core.select.residue_selector as selectors
            import pyrosetta.rosetta.core.select as core_select

            motif_sel = selectors.ResiduePDBInfoHasLabelSelector(label_str="motif")
            motif_subset = motif_sel.apply(pose)
            motif_residues = list(core_select.get_residue_set_from_subset(motif_subset))
            keep_pos.update(motif_residues)

            if motif_residues:
                logger.debug(f"Found {len(motif_residues)} motif-labeled residues")

        except Exception as e:
            logger.debug(f"Could not detect motif labels: {e}")

    keep_pos_list = sorted(list(keep_pos))
    logger.info(f"Keep positions: {len(keep_pos_list)} residues")

    return keep_pos_list
