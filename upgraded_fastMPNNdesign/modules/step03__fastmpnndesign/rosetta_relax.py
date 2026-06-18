#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rosetta relaxation module for step03 FastMPNN design.

This module provides standalone Rosetta relaxation functionality:
- Cartesian FastRelax for bond geometry optimization
- Torsional FastRelax for faster sampling
- Minimization for fine-tuning
- Sidechain repacking
- Hydrogen-only dump ("hydrate" mode) to add hydrogens without moving atoms

Designed to be run in a PyRosetta container via subprocess.

Usage (standalone):
    python rosetta_relax.py --pdb input.pdb --params LIG.params \\
        --mode cartesian --output output.pdb --constraints_json cst.json

Usage (as module):
    from rosetta_relax import run_relaxation
    result = run_relaxation(pdb_path, params, mode="cartesian", constraints=cst_dict)
"""
import os
import sys
import json
import argparse
import logging
import time
import math
from typing import Dict, List, Optional, Set, Tuple

# Add module_utils to path for standalone execution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, MODULE_DIR)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def init_pyrosetta(params_files: List[str], extra_options: str = "") -> None:
    """Initialize PyRosetta with params files."""
    try:
        import pyrosetta as pyr
    except ImportError:
        from module_utils.pyrosetta_utils import try_import_pyrosetta
        if not try_import_pyrosetta():
            raise
        import pyrosetta as pyr

    # Base options
    options = "-ignore_unrecognized_res false -ignore_zero_occupancy false"
    # Enable beta_jan25 scorefunction (improved LJ parameters, merged Dec 2025)
    # Note: -beta_jan25 is the simplified flag (equivalent to -corrections:beta_jan25)
    options += " -beta_jan25"
    if params_files:
        params_str = " ".join(params_files)
        options += f" -extra_res_fa {params_str}"
    if extra_options:
        options += f" {extra_options}"

    LOGGER.info(f"Initializing PyRosetta: {options[:100]}...")
    pyr.init(options)


def compute_scaled_stdev(
    desired_weight: float,
    desired_stdev: float,
    scorefxn_coord_cst_weight: float,
) -> Optional[float]:
    """Scale stdev to achieve an effective constraint weight.

    Rosetta uses a single coordinate_constraint score term. To apply different
    effective weights to different constraint groups, we scale stdev by
    sqrt(scorefxn_weight / desired_weight).
    """
    if desired_weight <= 0:
        return None
    if scorefxn_coord_cst_weight <= 0:
        return desired_stdev
    if math.isclose(desired_weight, scorefxn_coord_cst_weight):
        return desired_stdev
    scale = math.sqrt(scorefxn_coord_cst_weight / desired_weight)
    return desired_stdev * scale


def load_pose(pdb_path: str):
    """Load a pose from PDB file."""
    import pyrosetta as pyr
    LOGGER.info(f"Loading pose from {pdb_path}")
    return pyr.pose_from_pdb(pdb_path)


def fix_his_tautomers(pose, his_tautomer_map: Dict[Tuple[str, int], str]) -> int:
    """Fix HIS tautomers in pose using MutateResidue.

    After PyRosetta adds hydrogens, it uses default HIS tautomer states.
    This function corrects them to match the reference PDB tautomers.

    Args:
        pose: PyRosetta pose object
        his_tautomer_map: Dict mapping (chain, resno) -> "HIS" or "HIS_D"
                         HIS_D = delta tautomer (proton on ND1, has HD1)
                         HIS = epsilon tautomer (proton on NE2, has HE2)

    Returns:
        Number of HIS residues fixed
    """
    import pyrosetta as pyr

    if not his_tautomer_map:
        return 0

    fixed_count = 0
    pdb_info = pose.pdb_info()

    for (chain, resno), target_type in his_tautomer_map.items():
        # Get pose index for this residue
        seqpos = pdb_info.pdb2pose(chain, resno)
        if seqpos == 0:
            LOGGER.warning(f"Could not find HIS {chain}:{resno} in pose")
            continue

        # Check if this is actually a HIS residue
        current_name = pose.residue(seqpos).name3()
        if current_name not in ("HIS", "HIS_D"):
            LOGGER.debug(f"Skipping {chain}:{resno} - not HIS (is {current_name})")
            continue

        # Skip if already correct
        if current_name == target_type:
            LOGGER.debug(f"HIS {chain}:{resno} already correct ({target_type})")
            continue

        # Apply MutateResidue to fix tautomer
        try:
            mutres = pyr.rosetta.protocols.simple_moves.MutateResidue()
            mutres.set_res_name(target_type)
            mutres.set_target(seqpos)
            mutres.set_preserve_atom_coords(True)
            mutres.apply(pose)

            LOGGER.debug(f"Fixed HIS {chain}:{resno}: {current_name} -> {target_type}")
            fixed_count += 1
        except Exception as e:
            LOGGER.warning(f"Failed to fix HIS {chain}:{resno}: {e}")

    if fixed_count > 0:
        LOGGER.info(f"Fixed {fixed_count} HIS tautomers")

    return fixed_count


def add_coordinate_constraints(
    pose,
    constraints: Dict[str, List[str]],
    weight: float = 750.0,
    stdev: float = 0.01,
    global_weight: float = 0.0,
    global_stdev: float = 0.5,
    scorefxn_coord_cst_weight: Optional[float] = None,
) -> int:
    """Add coordinate constraints to pose.

    Args:
        pose: PyRosetta pose
        constraints: Dict mapping "chain:resno" -> list of atom names, "ALL_HEAVY", or "ALL_ATOMS"
                     ALL_ATOMS includes hydrogens (use for ligands)
                     ALL_HEAVY excludes hydrogens (use for protein residues)
        weight: Desired constraint weight for catalytic residues
        stdev: HarmonicFunc standard deviation in Angstroms for catalytic residues
        global_weight: Desired constraint weight for ALL protein atoms (0 = disabled)
        global_stdev: Stdev for global constraints (looser than catres constraints)
        scorefxn_coord_cst_weight: Actual coordinate_constraint weight used in the scorefunction.
            If this differs from desired weights, stdevs are scaled so each group
            matches its requested effective stiffness.

    Returns:
        Number of constraints added
    """
    from pyrosetta.rosetta.core.scoring.constraints import CoordinateConstraint
    from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
    from pyrosetta.rosetta.core.id import AtomID

    if scorefxn_coord_cst_weight is None:
        scorefxn_coord_cst_weight = max(weight, global_weight)

    LOGGER.info(f"Adding coordinate constraints (catres weight={weight}, stdev={stdev})...")
    if global_weight > 0:
        LOGGER.info(f"  Global constraints enabled (weight={global_weight}, stdev={global_stdev})")

    catres_stdev_eff = compute_scaled_stdev(weight, stdev, scorefxn_coord_cst_weight)
    if catres_stdev_eff is None:
        LOGGER.info("  Catres constraints disabled (weight <= 0)")
    elif not math.isclose(catres_stdev_eff, stdev):
        LOGGER.info(f"  Catres stdev scaled to {catres_stdev_eff:.4f} for effective weight")

    global_stdev_eff = compute_scaled_stdev(global_weight, global_stdev, scorefxn_coord_cst_weight)
    if global_weight > 0 and global_stdev_eff is None:
        LOGGER.info("  Global constraints disabled (weight <= 0)")
    elif global_weight > 0 and not math.isclose(global_stdev_eff, global_stdev):
        LOGGER.info(f"  Global stdev scaled to {global_stdev_eff:.4f} for effective weight")

    constraint_count = 0

    # Add global constraints to all protein backbone atoms if requested
    if global_weight > 0 and global_stdev_eff is not None:
        from pyrosetta.rosetta.core.scoring.constraints import CoordinateConstraint
        from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
        from pyrosetta.rosetta.core.id import AtomID

        LOGGER.info(f"  Adding global backbone constraints to all protein residues...")
        for pose_idx in range(1, pose.size() + 1):
            residue = pose.residue(pose_idx)

            # Only constrain protein residues (skip ligands, waters, etc.)
            if not residue.is_protein():
                continue

            # Constrain backbone heavy atoms (N, CA, C, O)
            backbone_atoms = ["N", "CA", "C", "O"]
            for atom_name in backbone_atoms:
                if not residue.has(atom_name):
                    continue

                atom_idx = residue.atom_index(atom_name)
                xyz = residue.xyz(atom_name)

                func = HarmonicFunc(0.0, global_stdev_eff)
                cst = CoordinateConstraint(
                    AtomID(atom_idx, pose_idx),
                    AtomID(1, 1),  # Anchor to first atom
                    xyz,
                    func
                )
                pose.add_constraint(cst)
                constraint_count += 1

        LOGGER.info(f"  Added {constraint_count} global backbone constraints")

    # Add specific constraints for catalytic residues
    catres_count = 0
    if catres_stdev_eff is None:
        return constraint_count

    for res_key, atom_spec in constraints.items():
        # Parse residue key (format: "A:123" or "A123")
        if ":" in res_key:
            chain, resno = res_key.split(":")
            resno = int(resno)
        else:
            chain = res_key[0]
            resno = int(res_key[1:])

        pose_idx = pose.pdb_info().pdb2pose(chain, resno)
        if pose_idx == 0:
            LOGGER.warning(f"  Could not find {chain}{resno} in pose")
            continue

        residue = pose.residue(pose_idx)

        # Determine which atoms to constrain
        if atom_spec == "ALL_ATOMS" or (isinstance(atom_spec, list) and "ALL_ATOMS" in atom_spec):
            # ALL_ATOMS includes hydrogens - use for ligands
            atom_names = [
                residue.atom_name(i).strip()
                for i in range(1, residue.natoms() + 1)
            ]
        elif atom_spec == "ALL_HEAVY" or (isinstance(atom_spec, list) and "ALL_HEAVY" in atom_spec):
            # ALL_HEAVY excludes hydrogens
            atom_names = [
                residue.atom_name(i).strip()
                for i in range(1, residue.natoms() + 1)
                if not residue.atom_is_hydrogen(i)
            ]
        else:
            atom_names = atom_spec if isinstance(atom_spec, list) else [atom_spec]

        for atom_name in atom_names:
            if not residue.has(atom_name):
                continue

            atom_idx = residue.atom_index(atom_name)
            xyz = residue.xyz(atom_name)

            func = HarmonicFunc(0.0, catres_stdev_eff)
            cst = CoordinateConstraint(
                AtomID(atom_idx, pose_idx),
                AtomID(1, 1),  # Anchor to first atom
                xyz,
                func
            )
            pose.add_constraint(cst)
            catres_count += 1

    LOGGER.info(f"  Added {catres_count} catalytic residue constraints")
    LOGGER.info(f"  Total constraints: {constraint_count + catres_count}")
    return constraint_count + catres_count


def setup_scorefunction(
    mode: str,
    coord_cst_weight: float = 750.0,
    cart_bonded_weight: float = 2.0,
    scorefunction_name: Optional[str] = None,
    fa_rep_weight: Optional[float] = None,
    score_term_weights: Optional[Dict[str, float]] = None,
    global_coord_cst_weight: Optional[float] = None,
):
    """Create scorefunction for relaxation.

    Args:
        mode: "cartesian" or "torsional"
        coord_cst_weight: Weight for coordinate constraints (effective scorefunction weight)
        cart_bonded_weight: Weight for cart_bonded term (cartesian only)
        scorefunction_name: Override scorefunction name
        fa_rep_weight: Override fa_rep weight (default 0.55, try 0.3-1.0 for clash tolerance)
        global_coord_cst_weight: Optional global constraint weight (for logging only)

    Returns:
        PyRosetta ScoreFunction
    """
    from pyrosetta.rosetta.core.scoring import ScoreType, ScoreFunctionFactory

    if scorefunction_name is None:
        if mode == "cartesian":
            scorefunction_name = "ref2015_cart"
        else:
            scorefunction_name = "beta_jan25"

    LOGGER.info(f"Setting up scorefunction: {scorefunction_name}")
    sfxn = ScoreFunctionFactory.create_score_function(scorefunction_name)

    if global_coord_cst_weight is not None and global_coord_cst_weight > coord_cst_weight:
        LOGGER.info(
            f"  Using coord_cst weight {global_coord_cst_weight} to satisfy global constraints"
        )
        coord_cst_weight = global_coord_cst_weight

    sfxn.set_weight(ScoreType.coordinate_constraint, coord_cst_weight)

    if mode == "cartesian":
        sfxn.set_weight(ScoreType.cart_bonded, cart_bonded_weight)
        sfxn.set_weight(ScoreType.pro_close, 0.0)  # Incompatible with cart_bonded

    # Override fa_rep weight if specified
    if fa_rep_weight is not None:
        LOGGER.info(f"  Setting fa_rep weight to {fa_rep_weight}")
        sfxn.set_weight(ScoreType.fa_rep, fa_rep_weight)

    # Apply arbitrary score term weights
    if score_term_weights:
        from pyrosetta.rosetta.core.scoring import score_type_from_name
        for term_name, weight in score_term_weights.items():
            try:
                score_type = score_type_from_name(term_name)
                sfxn.set_weight(score_type, float(weight))
                LOGGER.info(f"  Setting {term_name} weight to {weight}")
            except Exception as e:
                LOGGER.warning(f"  Could not set score term {term_name}: {e}")

    return sfxn


def setup_movemap(
    pose,
    mobile_residues: Optional[Set[int]] = None,
    cartesian: bool = True,
    chi: bool = True,
    bb: bool = True,
    jump: bool = False,
):
    """Create movemap for relaxation.

    Args:
        pose: PyRosetta pose
        mobile_residues: Set of pose indices that can move (None = all)
        cartesian: Whether this is for cartesian relaxation
        chi: Allow chi (sidechain) movement
        bb: Allow backbone movement
        jump: Allow jump (inter-chain) movement

    Returns:
        PyRosetta MoveMap
    """
    from pyrosetta.rosetta.core.kinematics import MoveMap

    mm = MoveMap()

    if mobile_residues is None:
        # All residues mobile
        mm.set_chi(chi)
        mm.set_bb(bb)
        mm.set_jump(jump)
    else:
        # Only specified residues mobile
        mm.set_chi(False)
        mm.set_bb(False)
        mm.set_jump(False)

        for pose_idx in mobile_residues:
            if pose_idx >= 1 and pose_idx <= pose.size():
                mm.set_chi(pose_idx, chi)
                mm.set_bb(pose_idx, bb)

    return mm


def run_fastrelax(
    pose,
    sfxn,
    movemap,
    repeats: int = 2,
    ramp_stages: int = 3,
    cartesian: bool = True,
    enable_bond_geometry_min: bool = True,
    relax_rounds: int = 5,
    relax_inner_cycles: Optional[int] = None,
) -> float:
    """Run FastRelax protocol.

    Args:
        pose: PyRosetta pose
        sfxn: ScoreFunction
        movemap: MoveMap
        repeats: Number of FastRelax repeats (outer loop)
        ramp_stages: Number of ramping stages
        cartesian: Use cartesian minimization
        enable_bond_geometry_min: Enable bond length/angle minimization
        relax_rounds: Number of FastRelax rounds/outer cycles (default 5)
        relax_inner_cycles: Number of inner cycles (default varies by mode)

    Returns:
        Final score
    """
    from pyrosetta.rosetta.protocols.relax import FastRelax

    LOGGER.info(f"Running FastRelax (repeats={repeats}, stages={ramp_stages}, cartesian={cartesian})")
    LOGGER.info(f"  FastRelax rounds: {relax_rounds}, inner_cycles: {relax_inner_cycles or 'default'}")
    LOGGER.info(f"  Bond geometry minimization: {enable_bond_geometry_min}")

    relax = FastRelax()
    relax.set_scorefxn(sfxn)
    relax.set_movemap(movemap)
    relax.cartesian(cartesian)
    relax.min_type("lbfgs_armijo_nonmonotone")

    # Enable bond geometry minimization using direct method calls
    # This is more reliable than script keywords and affects ALL minimization cycles
    if cartesian and enable_bond_geometry_min:
        relax.minimize_bond_angles(True)
        relax.minimize_bond_lengths(True)
        LOGGER.info("  Bond angle/length minimization enabled via FastRelax methods")

    # Build relax script and convert to Rosetta vector
    script_lines = build_relax_script(
        ramp_stages=ramp_stages,
        cartesian=cartesian,
        enable_bond_geometry_min=False,  # Don't use script keywords, use method calls above
        relax_rounds=relax_rounds,
    )
    # Convert Python list to Rosetta std.vector_std_string
    from pyrosetta.rosetta.std import vector_std_string
    rosetta_lines = vector_std_string()
    for line in script_lines:
        rosetta_lines.append(line)
    relax.set_script_from_lines(rosetta_lines)

    # Set max_iter if inner_cycles is specified
    if relax_inner_cycles is not None:
        relax.max_iter(relax_inner_cycles)

    start_score = sfxn(pose)
    LOGGER.info(f"  Start score: {start_score:.1f}")

    for i in range(repeats):
        relax.apply(pose)
        current_score = sfxn(pose)
        LOGGER.info(f"  Round {i+1}/{repeats} score: {current_score:.1f}")

    final_score = sfxn(pose)
    LOGGER.info(f"  Final score: {final_score:.1f}")

    return final_score


def build_relax_script(
    ramp_stages: int = 3,
    cartesian: bool = True,
    enable_bond_geometry_min: bool = True,
    relax_rounds: int = 5,
) -> List[str]:
    """Build FastRelax script lines.

    Args:
        ramp_stages: Number of fa_rep ramping stages
        cartesian: Whether this is for cartesian relax
        enable_bond_geometry_min: Enable bond geometry minimization
        relax_rounds: Number of FastRelax rounds/outer cycles

    Returns:
        List of script lines
    """
    lines = [f"repeat {relax_rounds}"]

    # Ramping schedule for fa_rep
    if ramp_stages <= 1:
        ramp_values = [1.0]
    elif ramp_stages == 2:
        ramp_values = [0.5, 1.0]
    elif ramp_stages == 3:
        ramp_values = [0.1, 0.5, 1.0]
    elif ramp_stages == 4:
        ramp_values = [0.02, 0.1, 0.5, 1.0]
    else:
        ramp_values = [0.02, 0.1, 0.33, 0.67, 1.0]

    for stage, rep_val in enumerate(ramp_values, 1):
        lines.append(f"scale:fa_rep {rep_val:.3f}")

        if cartesian:
            lines.append("repack")
            if enable_bond_geometry_min:
                lines.append("min bondangle_min bondlength_min cartesian 0.01")
            else:
                lines.append("min cartesian 0.01")
        else:
            lines.append("repack")
            lines.append("min dfpmin_armijo_nonmonotone 0.01")

    lines.append("accept_to_best")
    return lines


def run_minimization(
    pose,
    sfxn,
    movemap,
    tolerance: float = 0.01,
    max_iter: int = 200,
    cartesian: bool = False,
) -> float:
    """Run energy minimization.

    Args:
        pose: PyRosetta pose
        sfxn: ScoreFunction
        movemap: MoveMap
        tolerance: Convergence tolerance
        max_iter: Maximum iterations
        cartesian: Use cartesian minimization

    Returns:
        Final score
    """
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover

    LOGGER.info(f"Running minimization (tol={tolerance}, max_iter={max_iter}, cartesian={cartesian})")

    min_type = "lbfgs_armijo_nonmonotone" if cartesian else "dfpmin_armijo_nonmonotone"

    minmover = MinMover()
    minmover.score_function(sfxn)
    minmover.movemap(movemap)
    minmover.min_type(min_type)
    minmover.tolerance(tolerance)
    minmover.max_iter(max_iter)
    minmover.cartesian(cartesian)

    start_score = sfxn(pose)
    minmover.apply(pose)
    final_score = sfxn(pose)

    LOGGER.info(f"  Minimization: {start_score:.1f} -> {final_score:.1f}")
    return final_score


def run_repack(
    pose,
    sfxn,
    repack_residues: Optional[Set[int]] = None,
) -> float:
    """Run sidechain repacking.

    Args:
        pose: PyRosetta pose
        sfxn: ScoreFunction
        repack_residues: Set of pose indices to repack (None = all)

    Returns:
        Final score
    """
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import (
        IncludeCurrent,
        InitializeFromCommandline,
        NoRepackDisulfides,
        OperateOnResidueSubset,
        PreventRepackingRLT,
        RestrictToRepacking,
    )
    from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector
    from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover

    LOGGER.info("Running sidechain repacking...")

    tf = TaskFactory()
    # Legacy-style task ops
    tf.push_back(InitializeFromCommandline())
    tf.push_back(IncludeCurrent())
    tf.push_back(NoRepackDisulfides())
    tf.push_back(RestrictToRepacking())

    if repack_residues is not None:
        # Prevent repacking outside specified residues
        all_indices = set(range(1, pose.size() + 1))
        prevent_indices = all_indices - repack_residues

        if prevent_indices:
            prevent_selector = ResidueIndexSelector(','.join(map(str, prevent_indices)))
            prevent_op = OperateOnResidueSubset(PreventRepackingRLT(), prevent_selector)
            tf.push_back(prevent_op)

    packer = PackRotamersMover(sfxn)
    packer.task_factory(tf)

    start_score = sfxn(pose)
    packer.apply(pose)
    final_score = sfxn(pose)

    LOGGER.info(f"  Repack: {start_score:.1f} -> {final_score:.1f}")
    return final_score


def _parse_mobile_residues_arg(
    mobile_residues: Optional[List[str]],
    pose,
) -> Optional[Set[int]]:
    """Parse mobile residue identifiers into pose indices.

    Accepts identifiers like:
    - "A:123" or "A123" (chain/resno)
    - "123" (pose index)
    """
    if not mobile_residues:
        return None

    indices: Set[int] = set()
    pdb_info = pose.pdb_info()
    for item in mobile_residues:
        if item is None:
            continue
        token = str(item).strip()
        if not token:
            continue

        if ":" in token:
            chain, resno = token.split(":")
            try:
                pose_idx = pdb_info.pdb2pose(chain, int(resno))
            except ValueError:
                pose_idx = 0
        elif token[0].isalpha() and token[1:].isdigit():
            chain = token[0]
            resno = token[1:]
            try:
                pose_idx = pdb_info.pdb2pose(chain, int(resno))
            except ValueError:
                pose_idx = 0
        elif token.isdigit():
            pose_idx = int(token)
        else:
            pose_idx = 0

        if pose_idx > 0:
            indices.add(pose_idx)

    return indices if indices else None


def run_relaxation(
    pdb_path: str,
    params_files: List[str],
    output_path: str,
    mode: str = "torsional",
    constraints: Optional[Dict[str, List[str]]] = None,
    coord_cst_weight: float = 750.0,
    coord_cst_stdev: float = 0.01,
    global_coord_cst_weight: float = 0.0,
    global_coord_cst_stdev: float = 0.5,
    cart_bonded_weight: float = 2.0,
    repeats: int = 2,
    ramp_stages: int = 3,
    mobile_residues: Optional[Set[int]] = None,
    enable_bond_geometry_min: bool = True,
    scorefunction: Optional[str] = None,
    fa_rep_weight: Optional[float] = None,
    score_term_weights: Optional[Dict[str, float]] = None,
    relax_rounds: int = 5,
    relax_inner_cycles: Optional[int] = None,
    nstruct: int = 1,
    min_tolerance: float = 0.01,
    min_max_iter: int = 200,
    min_cartesian: Optional[bool] = None,
    min_backbone_rmsd_cutoff: Optional[float] = None,
    his_tautomer_map: Optional[Dict[Tuple[str, int], str]] = None,
) -> Dict:
    """Run complete relaxation protocol.

    Args:
        pdb_path: Input PDB file
        params_files: List of ligand .params files
        output_path: Output PDB file
        mode: "cartesian", "torsional", "minimize", "repack", or "hydrate"
        constraints: Dict mapping "chain:resno" -> atom names
        coord_cst_weight: Coordinate constraint weight for catalytic residues
        coord_cst_stdev: Coordinate constraint stdev for catalytic residues
        global_coord_cst_weight: Coordinate constraint weight for ALL protein atoms (0 = disabled)
        global_coord_cst_stdev: Stdev for global constraints (looser than catres)
        cart_bonded_weight: Cart bonded weight (cartesian only)
        repeats: FastRelax repeats
        ramp_stages: FastRelax ramping stages
        mobile_residues: Set of mobile pose indices
        enable_bond_geometry_min: Enable bond geometry minimization
        scorefunction: Override scorefunction name (ref2015, beta_nov16, beta_nov16_cart, etc.)
        fa_rep_weight: Override fa_rep weight (default 0.55, try 0.3-1.0)
        relax_rounds: Number of FastRelax rounds/outer cycles (default 5)
        relax_inner_cycles: Number of inner cycles (default varies by mode)
        nstruct: Number of structures to generate per input (default 1)
        min_tolerance: Minimization tolerance (minimize mode only)
        min_max_iter: Minimization max iterations (minimize mode only)
        his_tautomer_map: Dict mapping (chain, resno) -> "HIS" or "HIS_D" for tautomer fixing

    Returns:
        Dict with relaxation results
    """
    start_time = time.time()

    # Initialize PyRosetta
    init_pyrosetta(params_files)

    # Load pose
    pose = load_pose(pdb_path)
    LOGGER.info(f"Loaded pose with {pose.size()} residues")

    # Fix HIS tautomers if map provided
    if his_tautomer_map:
        fix_his_tautomers(pose, his_tautomer_map)

    # Map mobile residues (if provided as chain/resno strings)
    if mobile_residues:
        if isinstance(mobile_residues, (list, tuple)):
            mobile_residues = _parse_mobile_residues_arg(list(mobile_residues), pose)
        elif isinstance(mobile_residues, set):
            # Assume pose indices already
            mobile_residues = set(int(x) for x in mobile_residues)

    # Add constraints
    num_cst = 0
    effective_coord_cst_weight = max(coord_cst_weight, global_coord_cst_weight)
    if constraints or global_coord_cst_weight > 0:
        num_cst = add_coordinate_constraints(
            pose, constraints or {},
            weight=coord_cst_weight,
            stdev=coord_cst_stdev,
            global_weight=global_coord_cst_weight,
            global_stdev=global_coord_cst_stdev,
            scorefxn_coord_cst_weight=effective_coord_cst_weight,
        )

    # Setup scorefunction
    cartesian = mode == "cartesian"
    if mode == "minimize" and min_cartesian is not None:
        cartesian = bool(min_cartesian)
    sfxn = setup_scorefunction(
        mode=mode,
        coord_cst_weight=effective_coord_cst_weight,
        cart_bonded_weight=cart_bonded_weight,
        scorefunction_name=scorefunction,
        fa_rep_weight=fa_rep_weight,
        score_term_weights=score_term_weights,
        global_coord_cst_weight=global_coord_cst_weight,
    )

    start_score = sfxn(pose)

    # Setup movemap
    mm = setup_movemap(
        pose,
        mobile_residues=mobile_residues,
        cartesian=cartesian,
    )

    # Run protocol
    if mode == "hydrate":
        # No movement; just dump to add hydrogens via PyRosetta
        final_score = sfxn(pose)
    elif mode == "cartesian":
        final_score = run_fastrelax(
            pose, sfxn, mm,
            repeats=repeats,
            ramp_stages=ramp_stages,
            cartesian=True,
            enable_bond_geometry_min=enable_bond_geometry_min,
            relax_rounds=relax_rounds,
            relax_inner_cycles=relax_inner_cycles,
        )
    elif mode == "torsional":
        final_score = run_fastrelax(
            pose, sfxn, mm,
            repeats=repeats,
            ramp_stages=ramp_stages,
            cartesian=False,
            enable_bond_geometry_min=False,
            relax_rounds=relax_rounds,
            relax_inner_cycles=relax_inner_cycles,
        )
    elif mode == "minimize":
        pose_before = pose.clone()
        final_score = run_minimization(
            pose, sfxn, mm,
            cartesian=cartesian,
            tolerance=min_tolerance,
            max_iter=min_max_iter,
        )
        if min_backbone_rmsd_cutoff is not None:
            try:
                from pyrosetta.rosetta.utility import vector1_unsigned_long
                from pyrosetta.rosetta.protocols.toolbox.pose_manipulation import superimpose_pose_on_subset_CA
                overlay = vector1_unsigned_long()
                for n in range(1, pose.size() + 1):
                    if pose.residue(n).is_protein():
                        overlay.append(n)
                rmsd = superimpose_pose_on_subset_CA(pose_before, pose, overlay, 0)
                LOGGER.info(f"  Minimize backbone RMSD: {rmsd:.3f} (cutoff {min_backbone_rmsd_cutoff})")
                if rmsd > min_backbone_rmsd_cutoff:
                    LOGGER.warning("  Minimize RMSD cutoff exceeded; reverting to pre-min pose")
                    pose.assign(pose_before)
                    final_score = sfxn(pose)
            except Exception as e:
                LOGGER.warning(f"  Failed RMSD cutoff check: {e}")
    elif mode == "repack":
        final_score = run_repack(
            pose, sfxn,
            repack_residues=mobile_residues,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Save output
    LOGGER.info(f"Saving output to {output_path}")
    pose.dump_pdb(output_path)

    elapsed = time.time() - start_time

    result = {
        "input_pdb": pdb_path,
        "output_pdb": output_path,
        "mode": mode,
        "start_score": start_score,
        "final_score": final_score,
        "num_constraints": num_cst,
        "repeats": repeats,
        "ramp_stages": ramp_stages,
        "runtime_seconds": elapsed,
    }

    LOGGER.info(f"Relaxation complete in {elapsed:.1f}s")
    LOGGER.info(f"  Score: {start_score:.1f} -> {final_score:.1f}")

    return result


def main():
    """CLI entry point for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Run Rosetta relaxation on a PDB structure"
    )
    parser.add_argument("--pdb", required=True, help="Input PDB file")
    parser.add_argument("--params", nargs="+", default=[], help="Ligand .params files")
    parser.add_argument("--output", required=True, help="Output PDB file")
    parser.add_argument("--mode", default="torsional",
                       choices=["cartesian", "torsional", "minimize", "repack", "hydrate"],
                       help="Relaxation mode")
    parser.add_argument("--constraints_json", help="JSON file with constraints")
    parser.add_argument("--coord_cst_weight", type=float, default=750.0)
    parser.add_argument("--coord_cst_stdev", type=float, default=0.01)
    parser.add_argument("--cart_bonded_weight", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--ramp_stages", type=int, default=3)
    parser.add_argument("--scorefunction", default=None,
                       help="Scorefunction name (beta_jan25, ref2015, beta_nov16, ref2015_cart, beta_nov16_cart, etc.)")
    parser.add_argument("--enable_bond_geometry_min", action="store_true", default=False,
                       help="Enable bond length/angle minimization (cartesian mode)")
    parser.add_argument("--disable_bond_geometry_min", action="store_true", default=False,
                       help="Disable bond length/angle minimization (cartesian mode)")
    parser.add_argument("--fa_rep_weight", type=float, default=None,
                       help="Override fa_rep weight (default 0.55, try 0.3-1.0 for clash tolerance)")
    parser.add_argument("--relax_rounds", type=int, default=5,
                       help="Number of FastRelax rounds/outer cycles (default 5)")
    parser.add_argument("--relax_inner_cycles", type=int, default=None,
                       help="Number of inner cycles (default varies by mode)")
    parser.add_argument("--min_tolerance", type=float, default=0.01,
                       help="Minimization tolerance (minimize mode only)")
    parser.add_argument("--min_max_iter", type=int, default=200,
                       help="Minimization max iterations (minimize mode only)")
    parser.add_argument("--min_cartesian", action="store_true", default=False,
                       help="Use cartesian minimization when mode=minimize")
    parser.add_argument("--min_backbone_rmsd_cutoff", type=float, default=None,
                       help="Reject minimize if backbone RMSD exceeds this cutoff")
    parser.add_argument("--mobile_residues", default=None,
                       help="Comma-separated residues allowed to move/repack (A:10,A:11 or A10,B5 or pose indices)")
    parser.add_argument("--global_coord_cst_weight", type=float, default=0.0,
                       help="Coordinate constraint weight for ALL protein atoms (default 0, meaning no global constraints)")
    parser.add_argument("--global_coord_cst_stdev", type=float, default=0.5,
                       help="Stdev for global constraints (default 0.5, looser than catres constraints)")
    parser.add_argument("--nstruct", type=int, default=1,
                       help="Number of structures to generate per input (default 1)")
    parser.add_argument("--output_json", help="Output results JSON file")
    parser.add_argument("--his_tautomer_json", help="JSON file with HIS tautomer map")
    parser.add_argument("--score_term_weights_json", help="JSON file with score term weights")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load constraints
    constraints = None
    if args.constraints_json:
        with open(args.constraints_json, "r") as f:
            constraints = json.load(f)

    # Load HIS tautomer map
    # JSON format: {"A:13": "HIS_D", "A:15": "HIS_D", "A:176": "HIS", ...}
    # Convert to dict of (chain, resno) -> tautomer
    his_tautomer_map = None
    if args.his_tautomer_json:
        with open(args.his_tautomer_json, "r") as f:
            raw_map = json.load(f)
        his_tautomer_map = {}
        for key, tautomer in raw_map.items():
            # Parse "A:13" format
            parts = key.split(":")
            if len(parts) == 2:
                chain = parts[0]
                resno = int(parts[1])
                his_tautomer_map[(chain, resno)] = tautomer

    # Load score term weights
    score_term_weights = None
    if args.score_term_weights_json:
        try:
            with open(args.score_term_weights_json, "r") as f:
                score_term_weights = json.load(f)
        except Exception as e:
            LOGGER.warning(f"Failed to load score term weights JSON: {e}")
            score_term_weights = None

    # Parse mobile residues arg
    mobile_residues = None
    if args.mobile_residues:
        mobile_residues = [x.strip() for x in args.mobile_residues.split(",") if x.strip()]

    # Determine bond geometry minimization flag
    enable_bond_geometry_min = True
    if args.enable_bond_geometry_min:
        enable_bond_geometry_min = True
    if args.disable_bond_geometry_min:
        enable_bond_geometry_min = False

    # Run relaxation
    result = run_relaxation(
        pdb_path=args.pdb,
        params_files=args.params,
        output_path=args.output,
        mode=args.mode,
        constraints=constraints,
        coord_cst_weight=args.coord_cst_weight,
        coord_cst_stdev=args.coord_cst_stdev,
        global_coord_cst_weight=args.global_coord_cst_weight,
        global_coord_cst_stdev=args.global_coord_cst_stdev,
        cart_bonded_weight=args.cart_bonded_weight,
        repeats=args.repeats,
        ramp_stages=args.ramp_stages,
        enable_bond_geometry_min=enable_bond_geometry_min,
        scorefunction=args.scorefunction,
        fa_rep_weight=args.fa_rep_weight,
        score_term_weights=score_term_weights,
        relax_rounds=args.relax_rounds,
        relax_inner_cycles=args.relax_inner_cycles,
        mobile_residues=mobile_residues,
        nstruct=args.nstruct,
        min_tolerance=args.min_tolerance,
        min_max_iter=args.min_max_iter,
        min_cartesian=args.min_cartesian,
        min_backbone_rmsd_cutoff=args.min_backbone_rmsd_cutoff,
        his_tautomer_map=his_tautomer_map,
    )

    # Save results JSON
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        LOGGER.info(f"Results saved to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
