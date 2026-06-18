"""
Design layer detection for enhanced_fastmpnndesign.

Handles detection of design/repack/fixed residue layers based on distance from ligand.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional, Any, Set

# Add package directory to path for standalone execution
_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from constants import (
    DEFAULT_LAYER_DIST_BB, DEFAULT_LAYER_DIST_SC, DEFAULT_LAYER_CUTS,
    CLOSE_SC_CUTOFF, DESIGN_UTILS_PATH
)
from logging_config import get_logger
from utils import format_residue_list

logger = get_logger("layer_detection")


def get_layer_selections(
    pose: Any,
    keep_pos: List[int],
    design_pos: List[int],
    ligand_seqpos: int,
    heavyatoms: List[str],
    cuts: Tuple[float, ...] = DEFAULT_LAYER_CUTS,
    design_GP: bool = False
) -> Tuple[Any, Any, Any, List[List[int]]]:
    """
    Get residue layer selections based on distance from ligand.

    This wraps design_utils.get_layer_selections from the original script.

    Args:
        pose: PyRosetta Pose object
        keep_pos: Positions to keep fixed
        design_pos: Positions explicitly designated for design
        ligand_seqpos: Sequence position of ligand
        heavyatoms: List of ligand heavy atom names for distance calculation
        cuts: Distance cutoffs for layers
        design_GP: Whether to design Gly/Pro

    Returns:
        Tuple of (SEL_mutate, SEL_repack, SEL_do_not_repack, residues_by_layer)
    """
    # Import design_utils
    if DESIGN_UTILS_PATH not in sys.path:
        sys.path.append(DESIGN_UTILS_PATH)

    import design_utils

    return design_utils.get_layer_selections(
        pose, keep_pos, design_pos, ligand_seqpos, heavyatoms,
        cuts=list(cuts), design_GP=design_GP
    )


def get_ligand_heavyatoms(pose: Any) -> List[str]:
    """
    Get list of heavy atom names from the ligand.

    This wraps design_utils.get_ligand_heavyatoms from the original script.

    Args:
        pose: PyRosetta Pose object (ligand assumed to be last residue)

    Returns:
        List of heavy atom names
    """
    if DESIGN_UTILS_PATH not in sys.path:
        sys.path.append(DESIGN_UTILS_PATH)

    import design_utils
    return design_utils.get_ligand_heavyatoms(pose)


def get_residues_with_close_sc(
    pose: Any,
    ref_atoms: List[str],
    residues: List[int],
    exclude_residues: List[int],
    cutoff: float = CLOSE_SC_CUTOFF
) -> List[int]:
    """
    Get residues with sidechains close to reference atoms.

    This wraps design_utils.get_residues_with_close_sc from the original script.

    Args:
        pose: PyRosetta Pose object
        ref_atoms: Reference atom names on ligand
        residues: Candidate residues to check
        exclude_residues: Residues to exclude from result
        cutoff: Distance cutoff

    Returns:
        List of residues with close sidechains
    """
    if DESIGN_UTILS_PATH not in sys.path:
        sys.path.append(DESIGN_UTILS_PATH)

    import design_utils
    return design_utils.get_residues_with_close_sc(
        pose, ref_atoms, residues, exclude_residues, cutoff
    )


def detect_design_residues(
    pose: Any,
    keep_pos: List[int],
    design_pos: Optional[List[int]],
    ligand_seqpos: int,
    detect_pocket: bool = False
) -> Tuple[List[int], List[int], List[int]]:
    """
    Detect which residues should be designed, repacked, or left untouched.

    This replicates the logic from original script lines 172-227.

    Args:
        pose: PyRosetta Pose object
        keep_pos: Positions to keep fixed (catalytic residues, etc.)
        design_pos: Explicitly specified design positions (or None for auto)
        ligand_seqpos: Sequence position of ligand
        detect_pocket: Whether to auto-detect pocket residues

    Returns:
        Tuple of (design_residues, repack_residues, do_not_touch_residues)
    """
    heavyatoms = get_ligand_heavyatoms(pose)
    ligand = pose.residue(ligand_seqpos)

    # Get layer selections
    SEL_mutate, SEL_repack, SEL_do_not_repack, residues = get_layer_selections(
        pose, keep_pos, design_pos or [], ligand_seqpos, heavyatoms
    )

    if not detect_pocket:
        # Simple case: design everything not in keep_pos
        if design_pos is not None:
            design_residues = design_pos
        else:
            design_residues = [
                res.seqpos() for res in pose.residues
                if not res.is_ligand() and res.seqpos() not in keep_pos
            ]
            repack_residues = list(set(
                keep_pos + [res.seqpos() for res in pose.residues
                           if res.seqpos() not in design_residues]
            ))
            do_not_touch_residues = []
    else:
        # Auto-detect pocket: more sophisticated layer detection
        # Using substrate atoms (excluding metal and water-like atoms)
        substrate_atoms_ref = [
            ligand.atom_name(n).strip()
            for n in range(1, ligand.natoms() + 1)
            if ligand.atom_name(n).strip() not in ["ZN1", "O1"]
        ]

        _, _, _, residues_substrate = get_layer_selections(
            pose, keep_pos, design_pos or [], ligand_seqpos, substrate_atoms_ref,
            cuts=(7.0, 9.0, 11.0, 13.0)
        )

        design_residues = list(residues_substrate[0] + residues_substrate[1])

        # Add residues with close sidechains
        close_residues = get_residues_with_close_sc(
            pose, substrate_atoms_ref,
            residues_substrate[2] + residues_substrate[3],
            keep_pos, CLOSE_SC_CUTOFF
        )
        design_residues += close_residues
        design_residues = list(set(design_residues))

    # Build repack list
    repack_residues = residues[2] + residues[3] + residues[4] + [ligand_seqpos]
    do_not_touch_residues = []

    # Add non-design layer 0/1 residues to repack
    for res in residues[0] + residues[1]:
        if res not in design_residues:
            repack_residues.append(res)

    repack_residues = [x for x in repack_residues if x not in design_residues]

    # Verify all residues are classified
    all_classified = set(design_residues) | set(repack_residues) | set(do_not_touch_residues)
    unclassified = [
        res.seqpos() for res in pose.residues
        if res.seqpos() not in all_classified
    ]

    if unclassified:
        logger.warning(f"Unclassified residues: {format_residue_list(unclassified)}")

    logger.info(f"Design residues: {format_residue_list(design_residues)}")
    logger.debug(f"Repack residues: {format_residue_list(repack_residues)}")

    return design_residues, repack_residues, do_not_touch_residues


def get_2nd_layer_fixed_pos(
    pose: Any,
    ligand_seqpos: int,
    heavyatoms: List[str],
    keep_pos: List[int],
    dist_bb: float = DEFAULT_LAYER_DIST_BB,
    dist_sc: float = DEFAULT_LAYER_DIST_SC
) -> List[str]:
    """
    Get fixed positions for 2nd layer MPNN.

    This replicates the get_2nd_layer_fixed_pos function from original script (lines 51-74).

    Args:
        pose: PyRosetta Pose object
        ligand_seqpos: Sequence position of ligand
        heavyatoms: List of ligand heavy atom names
        keep_pos: Positions to keep fixed (catalytic residues)
        dist_bb: Backbone distance cutoff
        dist_sc: Sidechain distance cutoff

    Returns:
        List of fixed residue IDs in format ['A150', 'A152', ...]
    """
    import pyrosetta.rosetta.core.select.residue_selector as selectors
    import pyrosetta.rosetta.core.select as core_select

    # Get H-bond labeled residues
    motif_label_sel = selectors.ResiduePDBInfoHasLabelSelector(
        label_str="keep_hbonds_to_ligand_and_catres"
    )
    motif_subset = motif_label_sel.apply(pose)
    pocket_positions = keep_pos + list(core_select.get_residue_set_from_subset(motif_subset))
    pocket_positions = list(set(pocket_positions))

    # Get layer selections around pocket
    _, _, _, residues = get_layer_selections(
        pose, pocket_positions, [], ligand_seqpos, heavyatoms,
        cuts=(dist_bb, dist_bb + 2.0, dist_bb + 4.0, dist_bb + 6.0),
        design_GP=True
    )

    # Get residues with close sidechains
    close_ones = get_residues_with_close_sc(
        pose, heavyatoms, residues[1] + residues[2],
        exclude_residues=pocket_positions, cutoff=dist_sc
    )

    pocket_positions += residues[0] + close_ones
    pocket_positions = list(set(pocket_positions))

    # Identify design residues (not in pocket)
    design_residues = [
        x for x in residues[0] + residues[1] + residues[2] + residues[3]
        if x not in pocket_positions
    ]

    # Also include all ALA not in pocket
    ala_positions = [
        res.seqpos() for res in pose.residues
        if res.seqpos() not in pocket_positions + design_residues
        and res.name3() == "ALA"
    ]
    if ala_positions:
        logger.debug(f"ALA positions: {format_residue_list(ala_positions)}")
    design_residues += ala_positions

    # Fixed residues are everything NOT being designed
    pdb_info = pose.pdb_info()
    fixed_residues = [
        f"{pdb_info.chain(res.seqpos())}{pdb_info.number(res.seqpos())}"
        for res in pose.residues
        if res.seqpos() not in design_residues and res.is_protein()
    ]

    return fixed_residues
