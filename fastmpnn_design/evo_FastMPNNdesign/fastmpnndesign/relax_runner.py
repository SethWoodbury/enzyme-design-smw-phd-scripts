"""
PyRosetta-based constrained relaxation runner.

Implements Cartesian FastRelax with coordinate and distance constraints
to preserve ligand and catalytic residue geometry.

Supports two execution modes:
1. Direct PyRosetta execution (when PyRosetta is available in environment)
2. Container-based execution via pyrosetta.sif (default, for beta_jan25 scorefunction)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import sys
import json
import subprocess
import tempfile

from fastmpnndesign.config import RelaxConfig, CatalyticResidue
from fastmpnndesign.constraints import ConstraintSet, CoordinateConstraint, DistanceConstraint
from fastmpnndesign.ligand import detect_ligands_from_pdb, detect_metals_from_pdb, Ligand
from fastmpnndesign.utils import ensure_dir, calculate_distance
from fastmpnndesign.logging_config import get_logger
from fastmpnndesign.constants import BETA_SCOREFUNCTIONS
from fastmpnndesign.protonation import HistidineProtonation

logger = get_logger("relax_runner")

# Global flag for PyRosetta initialization
_PYROSETTA_INITIALIZED = False


@dataclass
class RelaxResult:
    """Result from a relaxation run."""
    success: bool
    input_pdb: Path
    output_pdb: Optional[Path] = None
    total_score: Optional[float] = None
    constraint_score: Optional[float] = None
    cart_bonded_score: Optional[float] = None
    score_terms: Dict[str, float] = field(default_factory=dict)
    displacements: Dict[str, float] = field(default_factory=dict)
    mean_displacement: Optional[float] = None
    max_displacement: Optional[float] = None
    error_message: str = ""
    ring_flips_tried: int = 0
    ring_flips_accepted: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'input_pdb': str(self.input_pdb),
            'output_pdb': str(self.output_pdb) if self.output_pdb else None,
            'total_score': self.total_score,
            'constraint_score': self.constraint_score,
            'cart_bonded_score': self.cart_bonded_score,
            'score_terms': self.score_terms,
            'mean_displacement': self.mean_displacement,
            'max_displacement': self.max_displacement,
            'error_message': self.error_message,
            'ring_flips_tried': self.ring_flips_tried,
            'ring_flips_accepted': self.ring_flips_accepted
        }


def init_pyrosetta(
    params_files: Optional[List[Path]] = None,
    extra_options: Optional[str] = None,
    quiet: bool = True,
    scorefunction: Optional[str] = None
) -> None:
    """
    Initialize PyRosetta with params files.

    Args:
        params_files: List of ligand params files.
        extra_options: Additional Rosetta options.
        quiet: Suppress PyRosetta output.
        scorefunction: Scorefunction name (for beta scorefunctions, adds -corrections flag).
    """
    global _PYROSETTA_INITIALIZED

    if _PYROSETTA_INITIALIZED:
        logger.debug("PyRosetta already initialized")
        return

    try:
        import pyrosetta
    except ImportError:
        raise ImportError(
            "PyRosetta not found. Please install PyRosetta or add it to PYTHONPATH."
        )

    # Build options string
    options = []

    if quiet:
        options.append("-mute all")

    # Add corrections flag for beta scorefunctions
    if scorefunction:
        # Extract base scorefunction name (e.g., "beta_jan25" from "beta_jan25_cart")
        sf_base = scorefunction.split('_cart')[0].split('_cst')[0]
        if sf_base in BETA_SCOREFUNCTIONS:
            options.append(f"-corrections:{sf_base}")
            logger.info(f"Adding -corrections:{sf_base} flag for beta scorefunction")

    # Add params files
    if params_files:
        params_str = " ".join(str(p) for p in params_files if p.exists())
        if params_str:
            options.append(f"-extra_res_fa {params_str}")

    if extra_options:
        options.append(extra_options)

    options_str = " ".join(options)
    logger.info(f"Initializing PyRosetta with: {options_str}")

    pyrosetta.init(options_str)
    _PYROSETTA_INITIALIZED = True

    logger.info("PyRosetta initialized successfully")


def create_scorefunction(config: RelaxConfig) -> Any:
    """
    Create and configure scorefunction for relaxation.

    Args:
        config: Relax configuration.

    Returns:
        PyRosetta ScoreFunction object.
    """
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import ScoreType

    # Get base scorefunction
    sfxn = pyrosetta.create_score_function(config.scorefunction)

    # Set weights for constrained relax
    sfxn.set_weight(ScoreType.cart_bonded, config.cart_bonded_weight)
    sfxn.set_weight(ScoreType.coordinate_constraint, config.coord_cst_weight)
    sfxn.set_weight(ScoreType.atom_pair_constraint, 50.0)  # High weight for distance constraints

    # Disable pro_close for Cartesian relax
    sfxn.set_weight(ScoreType.pro_close, 0.0)

    logger.debug(f"Created scorefunction {config.scorefunction} with cart_bonded={config.cart_bonded_weight}")

    return sfxn


def create_movemap(
    pose: Any,
    ligands: List[Ligand],
    metals: List[Ligand],
    mobile_center: Tuple[float, float, float],
    mobile_radius: float,
    freeze_catres: bool = False,
    catres_residues: Optional[Set[Tuple[str, int]]] = None,
    allow_catres_bb: bool = True
) -> Tuple[Any, Set[int]]:
    """
    Create MoveMap for relaxation.

    Enables backbone and chi angles for residues within mobile_radius
    of mobile_center. Completely freezes ligands/metals (no BB, no chi).
    Catres can have chi movement and BY DEFAULT also have BB movement allowed.

    Args:
        pose: PyRosetta Pose object.
        ligands: List of ligand molecules.
        metals: List of metal ions.
        mobile_center: Center point for mobile region (x, y, z).
        mobile_radius: Radius around center for mobile residues.
        freeze_catres: If True, freeze catalytic residue backbone (overrides allow_catres_bb).
        catres_residues: Set of (chain, resnum) for catres.
        allow_catres_bb: If True, allow backbone movement for catres (default True).

    Returns:
        Tuple of (PyRosetta MoveMap object, set of frozen ligand residue numbers).
    """
    from pyrosetta.rosetta.core.kinematics import MoveMap

    mm = MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)
    mm.set_jump(False)

    pdb_info = pose.pdb_info()

    # Build set of ligand/metal residue numbers in pose (these will be COMPLETELY frozen)
    ligand_resnums = set()

    for lig in ligands + metals:
        for res_i in range(1, pose.total_residue() + 1):
            if (pdb_info.chain(res_i) == lig.chain and
                pdb_info.number(res_i) == lig.resnum):
                ligand_resnums.add(res_i)
                logger.debug(f"Identified ligand/metal residue: {lig.chain}{lig.resnum} -> pose residue {res_i}")
                break

    # Get catres pose residue numbers
    catres_resnums = set()
    if catres_residues:
        for res_i in range(1, pose.total_residue() + 1):
            key = (pdb_info.chain(res_i), pdb_info.number(res_i))
            if key in catres_residues:
                catres_resnums.add(res_i)
                logger.debug(f"Identified catres residue: {key[0]}{key[1]} -> pose residue {res_i}")

    # Enable movement for residues within mobile_radius
    n_mobile_bb = 0
    n_mobile_chi = 0
    n_frozen = 0

    for res_i in range(1, pose.total_residue() + 1):
        # COMPLETELY freeze ligand/metal residues - no BB, no chi
        if res_i in ligand_resnums:
            mm.set_bb(res_i, False)
            mm.set_chi(res_i, False)
            n_frozen += 1
            continue

        # Check if residue is within mobile radius
        try:
            if pose.residue(res_i).has("CA"):
                xyz = pose.residue(res_i).xyz("CA")
            else:
                xyz = pose.residue(res_i).xyz(1)

            dist = calculate_distance(
                mobile_center[0], mobile_center[1], mobile_center[2],
                xyz.x, xyz.y, xyz.z
            )

            if dist <= mobile_radius:
                # Handle catres specially - allow chi but restrict BB unless explicitly allowed
                if res_i in catres_resnums:
                    mm.set_chi(res_i, True)
                    n_mobile_chi += 1
                    if allow_catres_bb and not freeze_catres:
                        mm.set_bb(res_i, True)
                        n_mobile_bb += 1
                        logger.debug(f"Catres {pdb_info.chain(res_i)}{pdb_info.number(res_i)}: BB=True, Chi=True")
                    else:
                        mm.set_bb(res_i, False)
                        logger.debug(f"Catres {pdb_info.chain(res_i)}{pdb_info.number(res_i)}: BB=False, Chi=True")
                else:
                    # Regular protein residue in mobile region
                    mm.set_bb(res_i, True)
                    mm.set_chi(res_i, True)
                    n_mobile_bb += 1
                    n_mobile_chi += 1
        except Exception:
            pass

    logger.info(f"MoveMap: {n_mobile_bb} BB-mobile, {n_mobile_chi} Chi-mobile, {n_frozen} frozen (ligand/metal)")

    return mm, ligand_resnums


def apply_coordinate_constraints(
    pose: Any,
    constraints: List[CoordinateConstraint],
    ligand_resnums: Optional[Set[int]] = None,
    ligand_cst_stdev: float = 0.001
) -> Tuple[int, int, Dict[str, Tuple[float, float, float]]]:
    """
    Apply coordinate constraints to pose.

    Also applies very tight constraints to all ligand atoms to freeze them.

    Args:
        pose: PyRosetta Pose object.
        constraints: List of CoordinateConstraint objects.
        ligand_resnums: Set of pose residue numbers for ligands (will be frozen).
        ligand_cst_stdev: Standard deviation for ligand constraints (smaller = tighter).

    Returns:
        Tuple of (number of user constraints applied, number of ligand constraints,
                  dict of ligand original coordinates).
    """
    from pyrosetta.rosetta.core.scoring.constraints import (
        CoordinateConstraint as RosettaCoordCst
    )
    from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.numeric import xyzVector_double_t

    pdb_info = pose.pdb_info()
    n_applied = 0
    n_ligand_cst = 0
    ligand_original_coords = {}

    # Find anchor atom (CA of first protein residue)
    anchor_atom_id = AtomID(1, 1)  # Default
    for res_i in range(1, pose.total_residue() + 1):
        if pose.residue(res_i).is_protein() and pose.residue(res_i).has("CA"):
            anchor_atom_id = AtomID(
                pose.residue(res_i).atom_index("CA"),
                res_i
            )
            break

    # Apply user-specified coordinate constraints (catres, etc.)
    for cst in constraints:
        # Find residue in pose
        target_res = None
        for res_i in range(1, pose.total_residue() + 1):
            if (pdb_info.chain(res_i) == cst.chain and
                pdb_info.number(res_i) == cst.resnum):
                target_res = res_i
                break

        if target_res is None:
            logger.warning(f"Residue not found: {cst.chain}{cst.resnum}")
            continue

        # Find atom in residue
        residue = pose.residue(target_res)
        if not residue.has(cst.atom_name):
            logger.warning(f"Atom not found: {cst.atom_name} in {cst.chain}{cst.resnum}")
            continue

        atom_idx = residue.atom_index(cst.atom_name)
        target_atom_id = AtomID(atom_idx, target_res)

        # Create constraint
        xyz = xyzVector_double_t(cst.x, cst.y, cst.z)
        func = HarmonicFunc(0.0, cst.stdev)

        coord_cst = RosettaCoordCst(
            target_atom_id,
            anchor_atom_id,
            xyz,
            func
        )

        pose.add_constraint(coord_cst)
        n_applied += 1

    logger.info(f"Applied {n_applied} user coordinate constraints")

    # Apply VERY tight coordinate constraints to ALL ligand atoms (complete freeze)
    if ligand_resnums:
        for res_i in ligand_resnums:
            residue = pose.residue(res_i)
            chain = pdb_info.chain(res_i)
            resnum = pdb_info.number(res_i)

            for atom_i in range(1, residue.natoms() + 1):
                atom_name = residue.atom_name(atom_i).strip()
                xyz = residue.xyz(atom_i)

                # Store original coordinates
                key = f"{chain}_{resnum}_{atom_name}"
                ligand_original_coords[key] = (xyz.x, xyz.y, xyz.z)

                # Create very tight constraint
                target_atom_id = AtomID(atom_i, res_i)
                xyz_vec = xyzVector_double_t(xyz.x, xyz.y, xyz.z)
                func = HarmonicFunc(0.0, ligand_cst_stdev)

                coord_cst = RosettaCoordCst(
                    target_atom_id,
                    anchor_atom_id,
                    xyz_vec,
                    func
                )

                pose.add_constraint(coord_cst)
                n_ligand_cst += 1

        logger.info(f"Applied {n_ligand_cst} ligand coordinate constraints (frozen, stdev={ligand_cst_stdev})")

    return n_applied, n_ligand_cst, ligand_original_coords


def apply_distance_constraints(
    pose: Any,
    distance_constraints: List[DistanceConstraint],
    pdb_info: Any
) -> int:
    """
    Apply distance constraints to pose as AtomPairConstraints.

    Args:
        pose: PyRosetta Pose object.
        distance_constraints: List of DistanceConstraint objects.
        pdb_info: PDBInfo object from the pose.

    Returns:
        Number of distance constraints successfully applied.
    """
    from pyrosetta.rosetta.core.scoring.constraints import AtomPairConstraint
    from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
    from pyrosetta.rosetta.core.id import AtomID

    n_applied = 0

    for dcst in distance_constraints:
        # Find the first residue in pose
        res1 = None
        for res_i in range(1, pose.total_residue() + 1):
            if (pdb_info.chain(res_i) == dcst.chain1 and
                pdb_info.number(res_i) == dcst.resnum1):
                res1 = res_i
                break

        if res1 is None:
            logger.warning(f"Residue not found for distance constraint: {dcst.chain1}{dcst.resnum1}")
            continue

        # Find the second residue in pose
        res2 = None
        for res_i in range(1, pose.total_residue() + 1):
            if (pdb_info.chain(res_i) == dcst.chain2 and
                pdb_info.number(res_i) == dcst.resnum2):
                res2 = res_i
                break

        if res2 is None:
            logger.warning(f"Residue not found for distance constraint: {dcst.chain2}{dcst.resnum2}")
            continue

        # Find the atoms in each residue
        residue1 = pose.residue(res1)
        residue2 = pose.residue(res2)

        if not residue1.has(dcst.atom1):
            logger.warning(f"Atom not found: {dcst.atom1} in {dcst.chain1}{dcst.resnum1}")
            continue

        if not residue2.has(dcst.atom2):
            logger.warning(f"Atom not found: {dcst.atom2} in {dcst.chain2}{dcst.resnum2}")
            continue

        atom_idx1 = residue1.atom_index(dcst.atom1)
        atom_idx2 = residue2.atom_index(dcst.atom2)

        atom_id1 = AtomID(atom_idx1, res1)
        atom_id2 = AtomID(atom_idx2, res2)

        # Create AtomPairConstraint with HarmonicFunc
        func = HarmonicFunc(dcst.distance, dcst.stdev)
        atom_pair_cst = AtomPairConstraint(atom_id1, atom_id2, func)

        pose.add_constraint(atom_pair_cst)
        n_applied += 1

    logger.info(f"Applied {n_applied} distance constraints")
    return n_applied


def run_cartesian_fastrelax(
    pose: Any,
    sfxn: Any,
    movemap: Any,
    n_cycles: int = 2
) -> Dict[str, Any]:
    """
    Run Cartesian FastRelax on pose with per-cycle logging.

    Args:
        pose: PyRosetta Pose object.
        sfxn: ScoreFunction object.
        movemap: MoveMap object.
        n_cycles: Number of FastRelax cycles.

    Returns:
        Dict with per-cycle score information.
    """
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.core.scoring import ScoreType

    relax = FastRelax()
    relax.set_scorefxn(sfxn)
    relax.set_movemap(movemap)
    relax.cartesian(True)
    relax.min_type("lbfgs_armijo_nonmonotone")

    # Set number of repeats
    relax.max_iter(200)

    # Score initial pose
    initial_score = sfxn(pose)
    logger.info(f"Initial total score: {initial_score:.2f}")

    try:
        initial_cart_bonded = pose.energies().total_energies()[ScoreType.cart_bonded]
        logger.info(f"Initial cart_bonded score: {initial_cart_bonded:.2f}")
    except Exception:
        initial_cart_bonded = None

    cycle_scores = []

    logger.info(f"Running Cartesian FastRelax with {n_cycles} cycles")
    for i in range(n_cycles):
        logger.info(f"FastRelax cycle {i+1}/{n_cycles}...")
        relax.apply(pose)

        # Score after this cycle
        cycle_score = sfxn(pose)
        cycle_data = {'cycle': i + 1, 'total_score': cycle_score}

        try:
            cycle_cart_bonded = pose.energies().total_energies()[ScoreType.cart_bonded]
            cycle_coord_cst = pose.energies().total_energies()[ScoreType.coordinate_constraint]
            cycle_data['cart_bonded'] = cycle_cart_bonded
            cycle_data['coordinate_constraint'] = cycle_coord_cst
            logger.info(f"  Cycle {i+1} complete: total={cycle_score:.2f}, cart_bonded={cycle_cart_bonded:.2f}, coord_cst={cycle_coord_cst:.2f}")
        except Exception:
            logger.info(f"  Cycle {i+1} complete: total={cycle_score:.2f}")

        cycle_scores.append(cycle_data)

    return {
        'initial_score': initial_score,
        'initial_cart_bonded': initial_cart_bonded,
        'cycle_scores': cycle_scores
    }


def run_multistage_relax(
    pose: Any,
    base_sfxn: Any,
    movemap: Any,
    initial_coord_cst_weight: float = 1000.0,
    final_coord_cst_weight: float = 100.0,
    initial_fa_rep_scale: float = 0.15,
    n_stages: int = 3
) -> Dict[str, Any]:
    """
    Multi-stage relaxation with ramped constraints.

    Philosophy: Start with high constraint weights and low repulsion to satisfy
    constraints first, then gradually increase repulsion and decrease constraint
    weight to optimize geometry and packing. This prevents sidechains from
    detaching/stretching when backbone can move to accommodate them.

    Stage 1: High constraint weight (1000), Low fa_rep (0.15)
        - Satisfy constraints first
        - Allow backbone to move freely to accommodate sidechains
        - Minimize bond distortions

    Stage 2: Medium constraint (500), Medium fa_rep (0.5)
        - Balance constraints and packing
        - Continue optimizing geometry

    Stage 3: Low constraint (100), Full fa_rep (1.0)
        - Final refinement
        - Let geometry optimize with full energy function

    Args:
        pose: PyRosetta Pose object.
        base_sfxn: Base ScoreFunction object (will be cloned and modified).
        movemap: MoveMap object.
        initial_coord_cst_weight: Starting constraint weight (default 1000).
        final_coord_cst_weight: Final constraint weight (default 100).
        initial_fa_rep_scale: Starting fa_rep scale (default 0.15).
        n_stages: Number of relaxation stages (default 3).

    Returns:
        Dict with per-stage score information.
    """
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.protocols.simple_moves import MinMover
    from pyrosetta.rosetta.core.scoring import ScoreType
    import pyrosetta

    logger.info("=" * 80)
    logger.info("MULTI-STAGE RELAXATION PROTOCOL")
    logger.info("=" * 80)
    logger.info("Philosophy: High constraints + low repulsion → Low constraints + full repulsion")
    logger.info("This allows backbone to adjust before optimizing packing")
    logger.info(f"Stages: {n_stages}, Constraint ramp: {initial_coord_cst_weight} → {final_coord_cst_weight}")
    logger.info(f"fa_rep ramp: {initial_fa_rep_scale} → 1.0")
    logger.info("=" * 80)

    # Score initial pose with base scorefunction
    initial_score = base_sfxn(pose)
    logger.info(f"Initial total score: {initial_score:.2f}")

    try:
        initial_cart_bonded = pose.energies().total_energies()[ScoreType.cart_bonded]
        logger.info(f"Initial cart_bonded score: {initial_cart_bonded:.2f}")
    except Exception:
        initial_cart_bonded = None

    stage_results = []

    # Calculate constraint and fa_rep weights for each stage
    for stage_idx in range(n_stages):
        # Linear interpolation from initial to final values
        progress = stage_idx / max(1, n_stages - 1)

        coord_cst_weight = initial_coord_cst_weight + progress * (final_coord_cst_weight - initial_coord_cst_weight)
        fa_rep_scale = initial_fa_rep_scale + progress * (1.0 - initial_fa_rep_scale)

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"STAGE {stage_idx + 1}/{n_stages}")
        logger.info(f"  coordinate_constraint weight: {coord_cst_weight:.1f}")
        logger.info(f"  fa_rep scale: {fa_rep_scale:.3f}")
        logger.info("=" * 80)

        # Clone and configure scorefunction for this stage
        sfxn_stage = pyrosetta.rosetta.core.scoring.ScoreFunction(base_sfxn)
        sfxn_stage.set_weight(ScoreType.coordinate_constraint, coord_cst_weight)
        sfxn_stage.set_weight(ScoreType.fa_rep, base_sfxn.get_weight(ScoreType.fa_rep) * fa_rep_scale)

        # Stage 1: Use minimization first to gently satisfy constraints
        if stage_idx == 0:
            logger.info("  Running initial minimization to satisfy constraints...")
            min_mover = MinMover()
            min_mover.movemap(movemap)
            min_mover.score_function(sfxn_stage)
            min_mover.cartesian(True)
            min_mover.tolerance(0.001)
            min_mover.max_iter(200)
            min_mover.apply(pose)

            # Score after minimization
            min_score = sfxn_stage(pose)
            try:
                min_cart_bonded = pose.energies().total_energies()[ScoreType.cart_bonded]
                min_coord_cst = pose.energies().total_energies()[ScoreType.coordinate_constraint]
                logger.info(f"  After minimization: total={min_score:.2f}, cart_bonded={min_cart_bonded:.2f}, coord_cst={min_coord_cst:.2f}")
            except Exception:
                logger.info(f"  After minimization: total={min_score:.2f}")

        # Run FastRelax for this stage
        logger.info("  Running FastRelax...")
        relax = FastRelax()
        relax.set_scorefxn(sfxn_stage)
        relax.set_movemap(movemap)
        relax.cartesian(True)
        relax.min_type("lbfgs_armijo_nonmonotone")
        relax.max_iter(200)
        relax.apply(pose)

        # Score after this stage
        stage_score = sfxn_stage(pose)
        stage_data = {
            'stage': stage_idx + 1,
            'total_score': stage_score,
            'coord_cst_weight': coord_cst_weight,
            'fa_rep_scale': fa_rep_scale
        }

        try:
            stage_cart_bonded = pose.energies().total_energies()[ScoreType.cart_bonded]
            stage_coord_cst = pose.energies().total_energies()[ScoreType.coordinate_constraint]
            stage_fa_rep = pose.energies().total_energies()[ScoreType.fa_rep]
            stage_data['cart_bonded'] = stage_cart_bonded
            stage_data['coordinate_constraint'] = stage_coord_cst
            stage_data['fa_rep'] = stage_fa_rep
            logger.info(f"  Stage {stage_idx + 1} complete:")
            logger.info(f"    total={stage_score:.2f}, cart_bonded={stage_cart_bonded:.2f}")
            logger.info(f"    coord_cst={stage_coord_cst:.2f}, fa_rep={stage_fa_rep:.2f}")
        except Exception:
            logger.info(f"  Stage {stage_idx + 1} complete: total={stage_score:.2f}")

        stage_results.append(stage_data)

    # Final minimization with the last scorefunction to polish
    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL MINIMIZATION")
    logger.info("=" * 80)
    sfxn_final = pyrosetta.rosetta.core.scoring.ScoreFunction(base_sfxn)
    sfxn_final.set_weight(ScoreType.coordinate_constraint, final_coord_cst_weight)

    min_final = MinMover()
    min_final.movemap(movemap)
    min_final.score_function(sfxn_final)
    min_final.cartesian(True)
    min_final.tolerance(0.00001)
    min_final.max_iter(500)
    min_final.apply(pose)

    final_score = sfxn_final(pose)
    try:
        final_cart_bonded = pose.energies().total_energies()[ScoreType.cart_bonded]
        final_coord_cst = pose.energies().total_energies()[ScoreType.coordinate_constraint]
        logger.info(f"Final minimization complete:")
        logger.info(f"  total={final_score:.2f}, cart_bonded={final_cart_bonded:.2f}, coord_cst={final_coord_cst:.2f}")
    except Exception:
        logger.info(f"Final minimization complete: total={final_score:.2f}")

    logger.info("=" * 80)
    logger.info("MULTI-STAGE RELAXATION COMPLETE")
    logger.info("=" * 80)

    return {
        'initial_score': initial_score,
        'initial_cart_bonded': initial_cart_bonded,
        'stage_results': stage_results,
        'protocol': 'multistage'
    }


def compute_displacements(
    pose: Any,
    original_coords: Dict[str, Tuple[float, float, float]]
) -> Dict[str, float]:
    """
    Compute displacement of constrained atoms from original positions.

    Args:
        pose: PyRosetta Pose after relaxation.
        original_coords: Dict mapping atom IDs to original (x, y, z).

    Returns:
        Dict mapping atom IDs to displacement in Angstroms.
    """
    pdb_info = pose.pdb_info()
    displacements = {}

    for atom_id, (orig_x, orig_y, orig_z) in original_coords.items():
        # Parse atom_id: "chain_resnum_atomname"
        parts = atom_id.split('_')
        if len(parts) < 3:
            continue
        chain = parts[0]
        resnum = int(parts[1])
        atom_name = '_'.join(parts[2:])

        # Find in pose
        for res_i in range(1, pose.total_residue() + 1):
            if (pdb_info.chain(res_i) == chain and
                pdb_info.number(res_i) == resnum):
                residue = pose.residue(res_i)
                if residue.has(atom_name):
                    xyz = residue.xyz(atom_name)
                    dist = calculate_distance(
                        orig_x, orig_y, orig_z,
                        xyz.x, xyz.y, xyz.z
                    )
                    displacements[atom_id] = dist
                break

    return displacements


def sample_ring_flips(
    pose: Any,
    residue_list: List[Tuple[str, int]],
    sfxn: Any
) -> Dict[str, Any]:
    """
    For residues with planar symmetry, try 180° flip and keep better conformation.

    Applies to: HIS (chi2), PHE (chi2), TYR (chi2)

    For HIS: flip imidazole ring by rotating chi2 by 180°
    For PHE/TYR: flip aromatic ring by rotating chi2 by 180°

    Args:
        pose: PyRosetta Pose object (will be modified in place)
        residue_list: List of (chain, resnum) tuples to try flipping
        sfxn: ScoreFunction for energy evaluation

    Returns:
        Dict with:
        - n_flipped: int - number of residues that improved with flip
        - n_tried: int - number of residues tried
        - details: List[Dict] - per-residue flip results
            Each dict has: chain, resnum, resname, original_chi2, flipped_chi2,
                          original_score, flipped_score, kept_flip
    """
    pdb_info = pose.pdb_info()

    # Residues with chi2 ring symmetry
    ring_flip_residues = {'HIS', 'PHE', 'TYR'}

    results = {
        'n_flipped': 0,
        'n_tried': 0,
        'details': []
    }

    for chain, resnum in residue_list:
        # Find residue in pose
        target_res = None
        for res_i in range(1, pose.total_residue() + 1):
            if (pdb_info.chain(res_i) == chain and
                pdb_info.number(res_i) == resnum):
                target_res = res_i
                break

        if target_res is None:
            continue

        residue = pose.residue(target_res)
        resname = residue.name3()

        # Handle HIS variants (HIS, HIS_D)
        base_resname = resname[:3] if resname.startswith('HIS') else resname

        if base_resname not in ring_flip_residues:
            continue

        # Check if residue has chi2
        if residue.nchi() < 2:
            continue

        results['n_tried'] += 1

        # Get current chi2 and score
        original_chi2 = pose.chi(2, target_res)
        score_before = sfxn(pose)

        # Flip 180°
        pose.set_chi(2, target_res, original_chi2 + 180.0)
        score_after = sfxn(pose)

        detail = {
            'chain': chain,
            'resnum': resnum,
            'resname': resname,
            'original_chi2': original_chi2,
            'flipped_chi2': original_chi2 + 180.0,
            'original_score': score_before,
            'flipped_score': score_after,
            'kept_flip': False
        }

        # Keep the better conformation
        if score_after < score_before:
            # Flip is better, keep it
            detail['kept_flip'] = True
            results['n_flipped'] += 1
        else:
            # Revert to original
            pose.set_chi(2, target_res, original_chi2)

        results['details'].append(detail)

    return results


def relax_structure(
    input_pdb: Path,
    output_pdb: Path,
    constraint_set: ConstraintSet,
    params_files: List[Path],
    config: RelaxConfig,
    ligands: Optional[List[Ligand]] = None,
    metals: Optional[List[Ligand]] = None,
    catres_list: Optional[List[CatalyticResidue]] = None,
    protonation_states: Optional[Dict[Tuple[str, int, str], HistidineProtonation]] = None,
    dry_run: bool = False
) -> RelaxResult:
    """
    Run constrained relaxation on a structure.

    Features:
    - Completely frozen ligand residues (no movement at all)
    - Increased backbone mobility for protein near active site
    - Heavy geometry penalty via cart_bonded weight
    - Per-cycle progress logging

    Args:
        input_pdb: Path to input PDB file.
        output_pdb: Path for output relaxed PDB.
        constraint_set: Constraints to apply.
        params_files: Ligand params files.
        config: Relaxation configuration.
        ligands: Pre-detected ligands (optional, will detect if not provided).
        metals: Pre-detected metals (optional).
        catres_list: Catalytic residues (optional).
        protonation_states: Histidine protonation states to enforce (optional).
        dry_run: If True, skip actual relaxation.

    Returns:
        RelaxResult with relaxation details.
    """
    input_pdb = Path(input_pdb)
    output_pdb = Path(output_pdb)
    ensure_dir(output_pdb.parent)

    # Detect ligands/metals early (needed for container mode too)
    if ligands is None:
        ligands = detect_ligands_from_pdb(input_pdb)
    if metals is None:
        metals = detect_metals_from_pdb(input_pdb)

    # Check if we should use container-based execution
    if config.use_pyrosetta_image:
        logger.info(f"Using PyRosetta container: {config.pyrosetta_image}")
        return relax_structure_in_container(
            input_pdb=input_pdb,
            output_pdb=output_pdb,
            constraint_set=constraint_set,
            params_files=params_files,
            config=config,
            ligands=ligands,
            metals=metals,
            catres_list=catres_list,
            protonation_states=protonation_states,
            dry_run=dry_run
        )

    # Direct execution mode
    if dry_run:
        logger.info(f"Dry run: would relax {input_pdb} -> {output_pdb}")
        return RelaxResult(
            success=True,
            input_pdb=input_pdb,
            output_pdb=output_pdb
        )

    try:
        import pyrosetta
        from pyrosetta.rosetta.core.scoring import ScoreType
    except ImportError:
        return RelaxResult(
            success=False,
            input_pdb=input_pdb,
            error_message="PyRosetta not available"
        )

    # Initialize PyRosetta
    init_pyrosetta(params_files, quiet=True, scorefunction=config.scorefunction)

    logger.info(f"Detected {len(ligands)} ligands, {len(metals)} metals")

    # Load pose
    try:
        pose = pyrosetta.pose_from_pdb(str(input_pdb))
    except Exception as e:
        return RelaxResult(
            success=False,
            input_pdb=input_pdb,
            error_message=f"Failed to load PDB: {e}"
        )

    # Store original coordinates for displacement calculation
    original_coords = {}
    for cst in constraint_set.coordinate_constraints:
        key = f"{cst.chain}_{cst.resnum}_{cst.atom_name}"
        original_coords[key] = (cst.x, cst.y, cst.z)

    # Create scorefunction
    sfxn = create_scorefunction(config)
    logger.info(f"Scorefunction: {config.scorefunction}, cart_bonded weight: {config.cart_bonded_weight}")

    # Determine mobile center (ligand center of mass)
    if ligands:
        mobile_center = ligands[0].center_of_mass
    elif metals:
        mobile_center = (metals[0].atoms[0].x, metals[0].atoms[0].y, metals[0].atoms[0].z)
    else:
        # Use geometric center of constraints
        if constraint_set.coordinate_constraints:
            xs = [c.x for c in constraint_set.coordinate_constraints]
            ys = [c.y for c in constraint_set.coordinate_constraints]
            zs = [c.z for c in constraint_set.coordinate_constraints]
            mobile_center = (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))
        else:
            mobile_center = (0, 0, 0)

    # Build catres set
    catres_set = None
    if catres_list:
        catres_set = {(cr.chain, cr.resnum) for cr in catres_list}

    # Create movemap with frozen ligands
    movemap, ligand_resnums = create_movemap(
        pose, ligands, metals,
        mobile_center=mobile_center,
        mobile_radius=config.mobile_radius,
        freeze_catres=False,
        catres_residues=catres_set,
        allow_catres_bb=getattr(config, 'allow_catres_bb', True)
    )

    # Apply constraints including ligand freezing constraints
    n_user_cst, n_ligand_cst, ligand_original_coords = apply_coordinate_constraints(
        pose,
        constraint_set.coordinate_constraints,
        ligand_resnums=ligand_resnums,
        ligand_cst_stdev=0.001  # Very tight - essentially frozen
    )

    if n_user_cst == 0:
        logger.warning("No user constraints applied")

    # Apply distance constraints
    n_dist_cst = apply_distance_constraints(
        pose,
        constraint_set.distance_constraints,
        pose.pdb_info()
    )

    if n_dist_cst > 0:
        logger.info(f"Applied {n_dist_cst} distance constraints to pose")

    # Run relaxation - choose protocol based on config
    try:
        if getattr(config, 'use_multistage_relax', True):
            logger.info("Using multi-stage relaxation protocol")
            relax_info = run_multistage_relax(
                pose,
                sfxn,
                movemap,
                initial_coord_cst_weight=getattr(config, 'initial_coord_cst_weight', 1000.0),
                final_coord_cst_weight=getattr(config, 'final_coord_cst_weight', 100.0),
                initial_fa_rep_scale=getattr(config, 'initial_fa_rep_scale', 0.15),
                n_stages=getattr(config, 'n_relax_stages', 3)
            )
        else:
            logger.info("Using standard FastRelax protocol")
            relax_info = run_cartesian_fastrelax(pose, sfxn, movemap, n_cycles=config.fastrelax_cycles)
    except Exception as e:
        return RelaxResult(
            success=False,
            input_pdb=input_pdb,
            error_message=f"Relaxation failed: {e}"
        )

    # Try ring flips for catres after relax converges
    ring_flips_tried = 0
    ring_flips_accepted = 0
    if catres_list:
        try:
            catres_tuples = [(cr.chain, cr.resnum) for cr in catres_list]
            flip_results = sample_ring_flips(pose, catres_tuples, sfxn)
            ring_flips_tried = flip_results['n_tried']
            ring_flips_accepted = flip_results['n_flipped']
            logger.info(f"Ring flip sampling: tried {ring_flips_tried}, flipped {ring_flips_accepted}")
            for detail in flip_results['details']:
                if detail['kept_flip']:
                    logger.info(f"  Flipped {detail['chain']}{detail['resnum']} ({detail['resname']}): "
                               f"chi2 {detail['original_chi2']:.1f} -> {detail['flipped_chi2']:.1f}, "
                               f"score {detail['original_score']:.1f} -> {detail['flipped_score']:.1f}")
        except Exception as e:
            logger.warning(f"Ring flip sampling failed: {e}")

    # Score final pose
    total_score = sfxn(pose)

    # Get individual score terms
    score_terms = {}
    for score_type in [ScoreType.total_score, ScoreType.cart_bonded,
                       ScoreType.coordinate_constraint, ScoreType.atom_pair_constraint,
                       ScoreType.fa_atr, ScoreType.fa_rep, ScoreType.fa_elec]:
        try:
            score_terms[str(score_type)] = pose.energies().total_energies()[score_type]
        except Exception:
            pass

    logger.info("Final score breakdown:")
    for term, value in score_terms.items():
        logger.info(f"  {term}: {value:.2f}")

    # Compute displacements for user constraints
    displacements = compute_displacements(pose, original_coords)
    mean_disp = None
    max_disp = None
    if displacements:
        disp_values = list(displacements.values())
        mean_disp = sum(disp_values) / len(disp_values)
        max_disp = max(disp_values)

    # Compute ligand displacements to verify freezing
    ligand_displacements = compute_displacements(pose, ligand_original_coords)
    if ligand_displacements:
        mean_lig_disp = sum(ligand_displacements.values()) / len(ligand_displacements)
        max_lig_disp = max(ligand_displacements.values())
        logger.info(f"Ligand displacement check (should be ~0):")
        logger.info(f"  Mean ligand displacement: {mean_lig_disp:.4f} A")
        logger.info(f"  Max ligand displacement: {max_lig_disp:.4f} A")
        if max_lig_disp > 0.01:
            logger.warning(f"Ligand moved more than 0.01 A!")

    # Save output
    pose.dump_pdb(str(output_pdb))
    logger.info(f"Saved relaxed structure to {output_pdb}")

    return RelaxResult(
        success=True,
        input_pdb=input_pdb,
        output_pdb=output_pdb,
        total_score=total_score,
        constraint_score=score_terms.get('ScoreType.coordinate_constraint'),
        cart_bonded_score=score_terms.get('ScoreType.cart_bonded'),
        score_terms=score_terms,
        displacements=displacements,
        mean_displacement=mean_disp,
        max_displacement=max_disp,
        ring_flips_tried=ring_flips_tried,
        ring_flips_accepted=ring_flips_accepted
    )


def validate_pyrosetta_setup(config: RelaxConfig) -> Tuple[bool, List[str]]:
    """
    Validate PyRosetta availability.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors = []

    try:
        import pyrosetta
    except ImportError:
        errors.append("PyRosetta not installed or not in PYTHONPATH")

    return len(errors) == 0, errors


# Standalone relax script template for container execution
RELAX_SCRIPT_TEMPLATE = '''#!/usr/bin/env python
"""
Standalone relax script for container execution.
Generated by fastmpnndesign.relax_runner

Features:
- Completely frozen ligand residues (no BB, chi, or coordinate movement)
- Increased backbone mobility for protein residues near active site
- Heavy geometry penalty via cart_bonded weight
- Ramped constraint weight schedule during relaxation
- Per-cycle progress logging with score breakdowns
"""
import json
import sys
from pathlib import Path

def main():
    # Load parameters from JSON
    params_file = sys.argv[1]
    with open(params_file, 'r') as f:
        params = json.load(f)

    input_pdb = Path(params['input_pdb'])
    output_pdb = Path(params['output_pdb'])
    params_files = [Path(p) for p in params['params_files']]
    scorefunction = params['scorefunction']
    fastrelax_cycles = params['fastrelax_cycles']
    mobile_radius = params['mobile_radius']
    cart_bonded_weight = params['cart_bonded_weight']
    coord_cst_weight = params.get('coord_cst_weight', 200.0)  # Coordinate constraint weight
    coord_constraints = params['coord_constraints']
    distance_constraints = params.get('distance_constraints', [])  # List of distance constraints
    ligand_residues = params.get('ligand_residues', [])  # List of {chain, resnum, resname}
    catres_residues = params.get('catres_residues', [])  # List of {chain, resnum}
    allow_catres_bb = params.get('allow_catres_bb', True)  # Allow BB movement for catres (default True)
    ligand_coord_cst_weight = params.get('ligand_coord_cst_weight', 1000.0)  # Extra weight for ligand
    protonation_states = params.get('protonation_states', [])  # List of {chain, resnum, protonation_state}
    # Multi-stage relax parameters
    use_multistage_relax = params.get('use_multistage_relax', True)
    initial_coord_cst_weight = params.get('initial_coord_cst_weight', 1000.0)
    final_coord_cst_weight = params.get('final_coord_cst_weight', 100.0)
    initial_fa_rep_scale = params.get('initial_fa_rep_scale', 0.15)
    n_relax_stages = params.get('n_relax_stages', 3)

    import pyrosetta
    from pyrosetta.rosetta.core.scoring import ScoreType
    from pyrosetta.rosetta.core.scoring.constraints import CoordinateConstraint as RosettaCoordCst
    from pyrosetta.rosetta.core.scoring.constraints import AtomPairConstraint
    from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.numeric import xyzVector_double_t
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue, MinMover

    # Build init options
    options = ["-mute all"]

    # Add packing flags for better side chain handling
    options.append("-packing:use_input_sc")

    # Add corrections flag for beta scorefunctions
    beta_scorefunctions = {'beta_jan25', 'beta_nov16', 'beta_july15', 'beta_nov15', 'beta'}
    sf_base = scorefunction.split('_cart')[0].split('_cst')[0]
    if sf_base in beta_scorefunctions:
        options.append(f"-corrections:{sf_base}")

    # Add params files
    if params_files:
        params_str = " ".join(str(p) for p in params_files if p.exists())
        if params_str:
            options.append(f"-extra_res_fa {params_str}")

    options_str = " ".join(options)
    print(f"Initializing PyRosetta with: {options_str}", file=sys.stderr)
    pyrosetta.init(options_str)

    # Load pose
    pose = pyrosetta.pose_from_pdb(str(input_pdb))
    pdb_info = pose.pdb_info()

    # Apply histidine protonation states using MutateResidue
    # This enforces the correct tautomer (HIS vs HIS_D) for metal coordination
    if protonation_states:
        print(f"Applying {len(protonation_states)} histidine protonation state(s)...", file=sys.stderr)
        for prot in protonation_states:
            chain = prot['chain']
            resnum = prot['resnum']
            target_resname = prot['protonation_state']  # 'HIS_D' for delta-protonated

            # Find the residue in the pose
            target_res = None
            for res_i in range(1, pose.total_residue() + 1):
                if (pdb_info.chain(res_i) == chain and
                    pdb_info.number(res_i) == resnum):
                    target_res = res_i
                    break

            if target_res is None:
                print(f"  Warning: Could not find HIS at {chain}{resnum}", file=sys.stderr)
                continue

            current_resname = pose.residue(target_res).name3()

            # Only mutate if needed (HIS to HIS_D or vice versa)
            if current_resname.startswith('HIS') and target_resname == 'HIS_D':
                print(f"  Mutating {chain}{resnum} from {current_resname} to HIS_D", file=sys.stderr)
                mutate = MutateResidue(target_res, 'HIS_D')
                mutate.apply(pose)
                print(f"    -> Now: {pose.residue(target_res).name3()}", file=sys.stderr)
            elif target_resname == 'HIS_D':
                print(f"  {chain}{resnum} already has protonation {current_resname}", file=sys.stderr)
            else:
                print(f"  Skipping {chain}{resnum}: target={target_resname}, current={current_resname}", file=sys.stderr)

    # Update pdb_info after any mutations
    pdb_info = pose.pdb_info()

    # Build sets for ligand and catres residue numbers in pose
    ligand_resnums = set()
    for lig in ligand_residues:
        for res_i in range(1, pose.total_residue() + 1):
            if (pdb_info.chain(res_i) == lig['chain'] and
                pdb_info.number(res_i) == lig['resnum']):
                ligand_resnums.add(res_i)
                print(f"Identified ligand residue: {lig['chain']}{lig['resnum']} -> pose residue {res_i}", file=sys.stderr)
                break

    catres_resnums = set()
    for cr in catres_residues:
        for res_i in range(1, pose.total_residue() + 1):
            if (pdb_info.chain(res_i) == cr['chain'] and
                pdb_info.number(res_i) == cr['resnum']):
                catres_resnums.add(res_i)
                print(f"Identified catres residue: {cr['chain']}{cr['resnum']} -> pose residue {res_i}", file=sys.stderr)
                break

    print(f"Total ligand residues to freeze: {len(ligand_resnums)}", file=sys.stderr)
    print(f"Total catres residues: {len(catres_resnums)}", file=sys.stderr)

    # Store original coordinates for displacement calculation
    original_coords = {}
    for cst in coord_constraints:
        key = f"{cst['chain']}_{cst['resnum']}_{cst['atom_name']}"
        original_coords[key] = (cst['x'], cst['y'], cst['z'])

    # Store original ligand coordinates for frozen constraints
    ligand_original_coords = {}
    for res_i in ligand_resnums:
        residue = pose.residue(res_i)
        chain = pdb_info.chain(res_i)
        resnum = pdb_info.number(res_i)
        for atom_i in range(1, residue.natoms() + 1):
            atom_name = residue.atom_name(atom_i).strip()
            xyz = residue.xyz(atom_i)
            key = f"{chain}_{resnum}_{atom_name}"
            ligand_original_coords[key] = (xyz.x, xyz.y, xyz.z)

    print(f"Stored {len(ligand_original_coords)} ligand atom coordinates for freezing", file=sys.stderr)

    # Apply coordinate constraints
    # Find anchor atom (CA of first protein residue)
    anchor_atom_id = AtomID(1, 1)
    for res_i in range(1, pose.total_residue() + 1):
        if pose.residue(res_i).is_protein() and pose.residue(res_i).has("CA"):
            anchor_atom_id = AtomID(pose.residue(res_i).atom_index("CA"), res_i)
            break

    n_applied = 0
    n_ligand_cst = 0

    # Apply user-specified coordinate constraints (catres, etc.)
    for cst in coord_constraints:
        target_res = None
        for res_i in range(1, pose.total_residue() + 1):
            if (pdb_info.chain(res_i) == cst['chain'] and
                pdb_info.number(res_i) == cst['resnum']):
                target_res = res_i
                break

        if target_res is None:
            print(f"Warning: Residue not found: {cst['chain']}{cst['resnum']}", file=sys.stderr)
            continue

        residue = pose.residue(target_res)
        if not residue.has(cst['atom_name']):
            print(f"Warning: Atom not found: {cst['atom_name']} in {cst['chain']}{cst['resnum']}", file=sys.stderr)
            continue

        atom_idx = residue.atom_index(cst['atom_name'])
        target_atom_id = AtomID(atom_idx, target_res)

        xyz = xyzVector_double_t(cst['x'], cst['y'], cst['z'])
        func = HarmonicFunc(0.0, cst['stdev'])
        coord_cst = RosettaCoordCst(target_atom_id, anchor_atom_id, xyz, func)
        pose.add_constraint(coord_cst)
        n_applied += 1

    print(f"Applied {n_applied} user coordinate constraints", file=sys.stderr)

    # Apply VERY tight coordinate constraints to ALL ligand atoms (complete freeze)
    # Using extremely small stdev (0.001 A) and very high weight handled by score function
    ligand_cst_stdev = 0.001  # 0.001 Angstrom - essentially frozen
    for res_i in ligand_resnums:
        residue = pose.residue(res_i)
        chain = pdb_info.chain(res_i)
        resnum = pdb_info.number(res_i)
        for atom_i in range(1, residue.natoms() + 1):
            atom_name = residue.atom_name(atom_i).strip()
            key = f"{chain}_{resnum}_{atom_name}"
            if key in ligand_original_coords:
                orig_x, orig_y, orig_z = ligand_original_coords[key]
                target_atom_id = AtomID(atom_i, res_i)
                xyz = xyzVector_double_t(orig_x, orig_y, orig_z)
                func = HarmonicFunc(0.0, ligand_cst_stdev)
                coord_cst = RosettaCoordCst(target_atom_id, anchor_atom_id, xyz, func)
                pose.add_constraint(coord_cst)
                n_ligand_cst += 1

    print(f"Applied {n_ligand_cst} ligand coordinate constraints (frozen)", file=sys.stderr)

    # Apply distance constraints
    n_distance_cst = 0
    for dcst in distance_constraints:
        # Find the first residue in pose
        res1 = None
        for res_i in range(1, pose.total_residue() + 1):
            if (pdb_info.chain(res_i) == dcst['chain1'] and
                pdb_info.number(res_i) == dcst['resnum1']):
                res1 = res_i
                break

        if res1 is None:
            print(f"Warning: Residue not found for distance constraint: {dcst['chain1']}{dcst['resnum1']}", file=sys.stderr)
            continue

        # Find the second residue in pose
        res2 = None
        for res_i in range(1, pose.total_residue() + 1):
            if (pdb_info.chain(res_i) == dcst['chain2'] and
                pdb_info.number(res_i) == dcst['resnum2']):
                res2 = res_i
                break

        if res2 is None:
            print(f"Warning: Residue not found for distance constraint: {dcst['chain2']}{dcst['resnum2']}", file=sys.stderr)
            continue

        # Find atoms in each residue
        residue1 = pose.residue(res1)
        residue2 = pose.residue(res2)

        if not residue1.has(dcst['atom1']):
            print(f"Warning: Atom not found: {dcst['atom1']} in {dcst['chain1']}{dcst['resnum1']}", file=sys.stderr)
            continue

        if not residue2.has(dcst['atom2']):
            print(f"Warning: Atom not found: {dcst['atom2']} in {dcst['chain2']}{dcst['resnum2']}", file=sys.stderr)
            continue

        atom_idx1 = residue1.atom_index(dcst['atom1'])
        atom_idx2 = residue2.atom_index(dcst['atom2'])

        atom_id1 = AtomID(atom_idx1, res1)
        atom_id2 = AtomID(atom_idx2, res2)

        # Create AtomPairConstraint with HarmonicFunc
        func = HarmonicFunc(dcst['distance'], dcst['stdev'])
        atom_pair_cst = AtomPairConstraint(atom_id1, atom_id2, func)

        pose.add_constraint(atom_pair_cst)
        n_distance_cst += 1

    print(f"Applied {n_distance_cst} distance constraints", file=sys.stderr)

    # Create scorefunction with geometry penalties
    sfxn = pyrosetta.create_score_function(scorefunction)
    sfxn.set_weight(ScoreType.cart_bonded, cart_bonded_weight)
    sfxn.set_weight(ScoreType.coordinate_constraint, coord_cst_weight)
    sfxn.set_weight(ScoreType.atom_pair_constraint, 50.0)  # High weight for distance constraints
    sfxn.set_weight(ScoreType.pro_close, 0.0)

    print(f"Scorefunction: {scorefunction}", file=sys.stderr)
    print(f"  cart_bonded weight: {cart_bonded_weight}", file=sys.stderr)
    print(f"  coordinate_constraint weight: {coord_cst_weight}", file=sys.stderr)
    print(f"  atom_pair_constraint weight: 50.0", file=sys.stderr)

    # Determine mobile center from constraint center of mass
    if coord_constraints:
        xs = [c['x'] for c in coord_constraints]
        ys = [c['y'] for c in coord_constraints]
        zs = [c['z'] for c in coord_constraints]
        mobile_center = (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))
    else:
        mobile_center = (0, 0, 0)

    # Create movemap with frozen ligands and mobile protein
    mm = MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)
    mm.set_jump(False)

    n_mobile_bb = 0
    n_mobile_chi = 0
    n_frozen = 0

    for res_i in range(1, pose.total_residue() + 1):
        # COMPLETELY freeze ligand residues - no BB, no chi, nothing
        if res_i in ligand_resnums:
            mm.set_bb(res_i, False)
            mm.set_chi(res_i, False)
            n_frozen += 1
            continue

        try:
            if pose.residue(res_i).has("CA"):
                xyz = pose.residue(res_i).xyz("CA")
            else:
                xyz = pose.residue(res_i).xyz(1)

            dist = ((mobile_center[0] - xyz.x)**2 +
                    (mobile_center[1] - xyz.y)**2 +
                    (mobile_center[2] - xyz.z)**2)**0.5

            if dist <= mobile_radius:
                # For catres, allow chi but optionally restrict BB
                if res_i in catres_resnums:
                    mm.set_chi(res_i, True)
                    n_mobile_chi += 1
                    if allow_catres_bb:
                        mm.set_bb(res_i, True)
                        n_mobile_bb += 1
                        print(f"  Catres {pdb_info.chain(res_i)}{pdb_info.number(res_i)}: BB=True, Chi=True", file=sys.stderr)
                    else:
                        mm.set_bb(res_i, False)
                        print(f"  Catres {pdb_info.chain(res_i)}{pdb_info.number(res_i)}: BB=False, Chi=True", file=sys.stderr)
                else:
                    # Regular protein residue in mobile region
                    mm.set_bb(res_i, True)
                    mm.set_chi(res_i, True)
                    n_mobile_bb += 1
                    n_mobile_chi += 1
        except Exception:
            pass

    print(f"MoveMap summary:", file=sys.stderr)
    print(f"  Frozen residues (ligand): {n_frozen}", file=sys.stderr)
    print(f"  Mobile BB residues: {n_mobile_bb}", file=sys.stderr)
    print(f"  Mobile Chi residues: {n_mobile_chi}", file=sys.stderr)

    # Score initial pose
    initial_score = sfxn(pose)
    print(f"Initial total score: {initial_score:.2f}", file=sys.stderr)

    # Get initial cart_bonded score
    try:
        initial_cart_bonded = pose.energies().total_energies()[ScoreType.cart_bonded]
        print(f"Initial cart_bonded score: {initial_cart_bonded:.2f}", file=sys.stderr)
    except Exception:
        initial_cart_bonded = None

    # Choose relaxation protocol
    if use_multistage_relax:
        print("=" * 80, file=sys.stderr)
        print("MULTI-STAGE RELAXATION PROTOCOL", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print("Philosophy: High constraints + low repulsion → Low constraints + full repulsion", file=sys.stderr)
        print("This allows backbone to adjust before optimizing packing", file=sys.stderr)
        print(f"Stages: {n_relax_stages}, Constraint ramp: {initial_coord_cst_weight} → {final_coord_cst_weight}", file=sys.stderr)
        print(f"fa_rep ramp: {initial_fa_rep_scale} → 1.0", file=sys.stderr)
        print("=" * 80, file=sys.stderr)

        # Get original fa_rep weight
        original_fa_rep = sfxn.get_weight(ScoreType.fa_rep)

        # Run multi-stage relaxation
        for stage_idx in range(n_relax_stages):
            # Linear interpolation from initial to final values
            progress = stage_idx / max(1, n_relax_stages - 1)
            coord_cst_weight_stage = initial_coord_cst_weight + progress * (final_coord_cst_weight - initial_coord_cst_weight)
            fa_rep_scale = initial_fa_rep_scale + progress * (1.0 - initial_fa_rep_scale)

            print("", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            print(f"STAGE {stage_idx + 1}/{n_relax_stages}", file=sys.stderr)
            print(f"  coordinate_constraint weight: {coord_cst_weight_stage:.1f}", file=sys.stderr)
            print(f"  fa_rep scale: {fa_rep_scale:.3f}", file=sys.stderr)
            print("=" * 80, file=sys.stderr)

            # Configure scorefunction for this stage
            sfxn_stage = pyrosetta.rosetta.core.scoring.ScoreFunction(sfxn)
            sfxn_stage.set_weight(ScoreType.coordinate_constraint, coord_cst_weight_stage)
            sfxn_stage.set_weight(ScoreType.fa_rep, original_fa_rep * fa_rep_scale)

            # Stage 1: Use minimization first
            if stage_idx == 0:
                print("  Running initial minimization to satisfy constraints...", file=sys.stderr)
                min_mover = MinMover()
                min_mover.movemap(mm)
                min_mover.score_function(sfxn_stage)
                min_mover.cartesian(True)
                min_mover.tolerance(0.001)
                min_mover.max_iter(200)
                min_mover.apply(pose)

                # Score after minimization
                min_score = sfxn_stage(pose)
                try:
                    min_cart_bonded = pose.energies().total_energies()[ScoreType.cart_bonded]
                    min_coord_cst = pose.energies().total_energies()[ScoreType.coordinate_constraint]
                    print(f"  After minimization: total={min_score:.2f}, cart_bonded={min_cart_bonded:.2f}, coord_cst={min_coord_cst:.2f}", file=sys.stderr)
                except Exception:
                    print(f"  After minimization: total={min_score:.2f}", file=sys.stderr)

            # Run FastRelax for this stage
            print("  Running FastRelax...", file=sys.stderr)
            relax = FastRelax()
            relax.set_scorefxn(sfxn_stage)
            relax.set_movemap(mm)
            relax.cartesian(True)
            relax.min_type("lbfgs_armijo_nonmonotone")
            relax.max_iter(200)
            relax.apply(pose)

            # Score after this stage
            stage_score = sfxn_stage(pose)
            try:
                stage_cart_bonded = pose.energies().total_energies()[ScoreType.cart_bonded]
                stage_coord_cst = pose.energies().total_energies()[ScoreType.coordinate_constraint]
                stage_fa_rep = pose.energies().total_energies()[ScoreType.fa_rep]
                print(f"  Stage {stage_idx + 1} complete:", file=sys.stderr)
                print(f"    total={stage_score:.2f}, cart_bonded={stage_cart_bonded:.2f}", file=sys.stderr)
                print(f"    coord_cst={stage_coord_cst:.2f}, fa_rep={stage_fa_rep:.2f}", file=sys.stderr)
            except Exception:
                print(f"  Stage {stage_idx + 1} complete: total={stage_score:.2f}", file=sys.stderr)

        # Final minimization
        print("", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print("FINAL MINIMIZATION", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        sfxn_final = pyrosetta.rosetta.core.scoring.ScoreFunction(sfxn)
        sfxn_final.set_weight(ScoreType.coordinate_constraint, final_coord_cst_weight)

        min_final = MinMover()
        min_final.movemap(mm)
        min_final.score_function(sfxn_final)
        min_final.cartesian(True)
        min_final.tolerance(0.00001)
        min_final.max_iter(500)
        min_final.apply(pose)

        final_score = sfxn_final(pose)
        try:
            final_cart_bonded = pose.energies().total_energies()[ScoreType.cart_bonded]
            final_coord_cst = pose.energies().total_energies()[ScoreType.coordinate_constraint]
            print(f"Final minimization complete:", file=sys.stderr)
            print(f"  total={final_score:.2f}, cart_bonded={final_cart_bonded:.2f}, coord_cst={final_coord_cst:.2f}", file=sys.stderr)
        except Exception:
            print(f"Final minimization complete: total={final_score:.2f}", file=sys.stderr)

        print("=" * 80, file=sys.stderr)
        print("MULTI-STAGE RELAXATION COMPLETE", file=sys.stderr)
        print("=" * 80, file=sys.stderr)

    else:
        # Standard FastRelax protocol
        print(f"Running standard Cartesian FastRelax with {fastrelax_cycles} cycles", file=sys.stderr)
        relax = FastRelax()
        relax.set_scorefxn(sfxn)
        relax.set_movemap(mm)
        relax.cartesian(True)
        relax.min_type("lbfgs_armijo_nonmonotone")
        relax.max_iter(200)

        for i in range(fastrelax_cycles):
            print(f"FastRelax cycle {i+1}/{fastrelax_cycles}...", file=sys.stderr)
            relax.apply(pose)

            # Score after this cycle
            cycle_score = sfxn(pose)
            try:
                cycle_cart_bonded = pose.energies().total_energies()[ScoreType.cart_bonded]
                cycle_coord_cst = pose.energies().total_energies()[ScoreType.coordinate_constraint]
                print(f"  Cycle {i+1} complete: total={cycle_score:.2f}, cart_bonded={cycle_cart_bonded:.2f}, coord_cst={cycle_coord_cst:.2f}", file=sys.stderr)
            except Exception:
                print(f"  Cycle {i+1} complete: total={cycle_score:.2f}", file=sys.stderr)

    # Ring flip sampling for aromatic/imidazole residues
    ring_flip_residues = {'HIS', 'PHE', 'TYR'}
    n_ring_flips_tried = 0
    n_ring_flips_accepted = 0

    if catres_residues:
        for cr in catres_residues:
            chain = cr['chain']
            resnum = cr['resnum']

            # Find in pose
            target_res = None
            for res_i in range(1, pose.total_residue() + 1):
                if (pdb_info.chain(res_i) == chain and
                    pdb_info.number(res_i) == resnum):
                    target_res = res_i
                    break

            if target_res is None:
                continue

            residue = pose.residue(target_res)
            resname = residue.name3()
            base_resname = resname[:3] if resname.startswith('HIS') else resname

            if base_resname not in ring_flip_residues:
                continue

            if residue.nchi() < 2:
                continue

            n_ring_flips_tried += 1
            original_chi2 = pose.chi(2, target_res)
            score_before = sfxn(pose)

            pose.set_chi(2, target_res, original_chi2 + 180.0)
            score_after = sfxn(pose)

            if score_after < score_before:
                print(f"  Flipped {chain}{resnum} ({base_resname}): chi2 {original_chi2:.1f} -> {original_chi2 + 180.0:.1f}, score {score_before:.1f} -> {score_after:.1f}", file=sys.stderr)
                n_ring_flips_accepted += 1
            else:
                pose.set_chi(2, target_res, original_chi2)  # Revert

    print(f"Ring flip sampling: tried {n_ring_flips_tried}, flipped {n_ring_flips_accepted}", file=sys.stderr)

    # Score final pose
    total_score = sfxn(pose)

    # Get detailed score breakdown
    score_terms = {}
    for score_type_name in ['cart_bonded', 'coordinate_constraint', 'atom_pair_constraint',
                            'fa_atr', 'fa_rep', 'fa_elec', 'fa_sol', 'hbond_sc', 'hbond_bb_sc']:
        try:
            st = getattr(ScoreType, score_type_name)
            score_terms[score_type_name] = pose.energies().total_energies()[st]
        except Exception:
            pass

    print(f"Final score breakdown:", file=sys.stderr)
    for term, value in score_terms.items():
        print(f"  {term}: {value:.2f}", file=sys.stderr)

    # Compute displacements for user constraints
    displacements = {}
    for atom_id, (orig_x, orig_y, orig_z) in original_coords.items():
        parts = atom_id.split('_')
        if len(parts) < 3:
            continue
        chain = parts[0]
        resnum = int(parts[1])
        atom_name = '_'.join(parts[2:])

        for res_i in range(1, pose.total_residue() + 1):
            if (pdb_info.chain(res_i) == chain and pdb_info.number(res_i) == resnum):
                residue = pose.residue(res_i)
                if residue.has(atom_name):
                    xyz = residue.xyz(atom_name)
                    dist = ((orig_x - xyz.x)**2 + (orig_y - xyz.y)**2 + (orig_z - xyz.z)**2)**0.5
                    displacements[atom_id] = dist
                break

    # Compute ligand displacements to verify freezing
    ligand_displacements = {}
    max_ligand_disp = 0.0
    for res_i in ligand_resnums:
        residue = pose.residue(res_i)
        chain = pdb_info.chain(res_i)
        resnum = pdb_info.number(res_i)
        for atom_i in range(1, residue.natoms() + 1):
            atom_name = residue.atom_name(atom_i).strip()
            key = f"{chain}_{resnum}_{atom_name}"
            if key in ligand_original_coords:
                orig_x, orig_y, orig_z = ligand_original_coords[key]
                xyz = residue.xyz(atom_i)
                dist = ((orig_x - xyz.x)**2 + (orig_y - xyz.y)**2 + (orig_z - xyz.z)**2)**0.5
                ligand_displacements[key] = dist
                if dist > max_ligand_disp:
                    max_ligand_disp = dist

    if ligand_displacements:
        mean_ligand_disp = sum(ligand_displacements.values()) / len(ligand_displacements)
        print(f"Ligand displacement check (should be ~0):", file=sys.stderr)
        print(f"  Mean ligand displacement: {mean_ligand_disp:.4f} A", file=sys.stderr)
        print(f"  Max ligand displacement: {max_ligand_disp:.4f} A", file=sys.stderr)
        if max_ligand_disp > 0.01:
            print(f"  WARNING: Ligand moved more than 0.01 A!", file=sys.stderr)

    # Save output
    pose.dump_pdb(str(output_pdb))
    print(f"Saved relaxed structure to {output_pdb}", file=sys.stderr)

    # Write results JSON
    result = {
        'success': True,
        'total_score': total_score,
        'score_terms': score_terms,
        'cart_bonded_score': score_terms.get('cart_bonded'),
        'displacements': displacements,
        'mean_displacement': sum(displacements.values()) / len(displacements) if displacements else None,
        'max_displacement': max(displacements.values()) if displacements else None,
        'ligand_displacements': ligand_displacements,
        'mean_ligand_displacement': sum(ligand_displacements.values()) / len(ligand_displacements) if ligand_displacements else None,
        'max_ligand_displacement': max_ligand_disp if ligand_displacements else None,
        'ring_flips_tried': n_ring_flips_tried,
        'ring_flips_accepted': n_ring_flips_accepted
    }

    results_file = str(output_pdb).replace('.pdb', '_relax_result.json')
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results written to {results_file}", file=sys.stderr)

if __name__ == '__main__':
    main()
'''


def relax_structure_in_container(
    input_pdb: Path,
    output_pdb: Path,
    constraint_set: ConstraintSet,
    params_files: List[Path],
    config: RelaxConfig,
    ligands: Optional[List[Ligand]] = None,
    metals: Optional[List[Ligand]] = None,
    catres_list: Optional[List[CatalyticResidue]] = None,
    protonation_states: Optional[Dict[Tuple[str, int, str], HistidineProtonation]] = None,
    dry_run: bool = False
) -> RelaxResult:
    """
    Run constrained relaxation inside the PyRosetta container.

    This allows using newer PyRosetta versions (e.g., with beta_jan25 scorefunction)
    even when the host environment has an older version.

    Features:
    - Completely frozen ligand residues (no movement at all)
    - Increased backbone mobility for protein near active site
    - Heavy geometry penalty via cart_bonded weight
    - Per-cycle progress logging

    Args:
        input_pdb: Path to input PDB file.
        output_pdb: Path for output relaxed PDB.
        constraint_set: Constraints to apply.
        params_files: Ligand params files.
        config: Relaxation configuration.
        ligands: Pre-detected ligands (for freezing).
        metals: Pre-detected metals (for freezing).
        catres_list: Catalytic residues (for special handling).
        protonation_states: Histidine protonation states to enforce (optional).
        dry_run: If True, skip actual relaxation.

    Returns:
        RelaxResult with relaxation details.
    """
    logger.info(f"Starting container-based relax for {input_pdb}")

    input_pdb = Path(input_pdb)
    output_pdb = Path(output_pdb)
    ensure_dir(output_pdb.parent)

    pyrosetta_image = config.pyrosetta_image
    logger.debug(f"Checking for PyRosetta container at: {pyrosetta_image}")
    if not pyrosetta_image.exists():
        error_msg = f"PyRosetta container not found: {pyrosetta_image}"
        logger.error(error_msg)
        return RelaxResult(
            success=False,
            input_pdb=input_pdb,
            error_message=error_msg
        )

    # Check if apptainer is available (might not be if running inside a container)
    import shutil
    if shutil.which('apptainer') is None:
        error_msg = (
            "apptainer not found - cannot run PyRosetta in container.\n"
            "This usually means you're running inside a container (e.g., universal.sif).\n"
            "Solutions:\n"
            "  1. Run the CLI from the HOST with --use_apptainer for MPNN\n"
            "  2. Or use --no_pyrosetta_image to use the container's built-in PyRosetta (can't use beta_jan25)\n"
            "\nExample from host:\n"
            "  python cli.py --pdb input.pdb --params ligand.params --use_apptainer"
        )
        logger.error(error_msg)
        return RelaxResult(
            success=False,
            input_pdb=input_pdb,
            error_message=error_msg
        )

    # Prepare constraint data for JSON serialization
    logger.debug(f"Preparing {len(constraint_set.coordinate_constraints)} coordinate constraints")
    coord_constraints_data = []
    for cst in constraint_set.coordinate_constraints:
        coord_constraints_data.append({
            'chain': cst.chain,
            'resnum': cst.resnum,
            'atom_name': cst.atom_name,
            'x': cst.x,
            'y': cst.y,
            'z': cst.z,
            'stdev': cst.stdev
        })

    # Prepare distance constraint data for JSON serialization
    logger.debug(f"Preparing {len(constraint_set.distance_constraints)} distance constraints")
    distance_constraints_data = []
    for dcst in constraint_set.distance_constraints:
        distance_constraints_data.append({
            'chain1': dcst.chain1,
            'resnum1': dcst.resnum1,
            'resname1': dcst.resname1,
            'atom1': dcst.atom1,
            'chain2': dcst.chain2,
            'resnum2': dcst.resnum2,
            'resname2': dcst.resname2,
            'atom2': dcst.atom2,
            'distance': dcst.distance,
            'stdev': dcst.stdev,
            'constraint_type': dcst.constraint_type
        })
    if distance_constraints_data:
        logger.info(f"  Distance constraints to apply: {len(distance_constraints_data)}")

    # Prepare ligand residue data for freezing
    ligand_residues_data = []
    if ligands:
        for lig in ligands:
            ligand_residues_data.append({
                'chain': lig.chain,
                'resnum': lig.resnum,
                'resname': lig.resname
            })
    if metals:
        for metal in metals:
            ligand_residues_data.append({
                'chain': metal.chain,
                'resnum': metal.resnum,
                'resname': metal.resname
            })
    if ligand_residues_data:
        logger.info(f"  Ligand/metal residues to freeze: {len(ligand_residues_data)}")
        for lr in ligand_residues_data:
            logger.info(f"    {lr['chain']}{lr['resnum']} ({lr['resname']})")

    # Prepare catres residue data
    catres_residues_data = []
    if catres_list:
        for cr in catres_list:
            catres_residues_data.append({
                'chain': cr.chain,
                'resnum': cr.resnum
            })
        logger.info(f"  Catalytic residues: {len(catres_residues_data)}")
        for cr in catres_residues_data:
            logger.info(f"    {cr['chain']}{cr['resnum']}")

    # Prepare protonation states data for JSON serialization
    protonation_data = []
    if protonation_states:
        for (chain, resnum, icode), prot in protonation_states.items():
            if prot.protonation_state == 'HIS_D':
                protonation_data.append({
                    'chain': chain,
                    'resnum': resnum,
                    'icode': icode,
                    'protonation_state': prot.protonation_state,
                    'reason': prot.reason
                })
        if protonation_data:
            logger.info(f"  Protonation states to enforce: {len(protonation_data)} HIS_D residues")
            for p in protonation_data:
                logger.info(f"    {p['chain']}{p['resnum']}: HIS_D")

    # Create parameters JSON
    params_data = {
        'input_pdb': str(input_pdb.resolve()),
        'output_pdb': str(output_pdb.resolve()),
        'params_files': [str(p.resolve()) for p in params_files if p.exists()],
        'scorefunction': config.scorefunction,
        'fastrelax_cycles': config.fastrelax_cycles,
        'mobile_radius': config.mobile_radius,
        'cart_bonded_weight': config.cart_bonded_weight,
        'coord_cst_weight': config.coord_cst_weight,
        'coord_constraints': coord_constraints_data,
        'distance_constraints': distance_constraints_data,
        'ligand_residues': ligand_residues_data,
        'catres_residues': catres_residues_data,
        'allow_catres_bb': getattr(config, 'allow_catres_bb', True),
        'protonation_states': protonation_data,
        # Multi-stage relaxation parameters
        'use_multistage_relax': getattr(config, 'use_multistage_relax', True),
        'initial_coord_cst_weight': getattr(config, 'initial_coord_cst_weight', 1000.0),
        'final_coord_cst_weight': getattr(config, 'final_coord_cst_weight', 100.0),
        'initial_fa_rep_scale': getattr(config, 'initial_fa_rep_scale', 0.15),
        'n_relax_stages': getattr(config, 'n_relax_stages', 3)
    }

    # Write parameters and script to temporary files
    logger.debug("Writing temporary parameter and script files")
    with tempfile.NamedTemporaryFile(mode='w', suffix='_params.json', delete=False) as f:
        json.dump(params_data, f, indent=2)
        params_file = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='_relax.py', delete=False) as f:
        f.write(RELAX_SCRIPT_TEMPLATE)
        script_file = f.name

    logger.debug(f"Params file: {params_file}")
    logger.debug(f"Script file: {script_file}")

    if dry_run:
        logger.info(f"Dry run: would run relax in container {pyrosetta_image}")
        logger.info(f"  Input: {input_pdb}")
        logger.info(f"  Output: {output_pdb}")
        return RelaxResult(
            success=True,
            input_pdb=input_pdb,
            output_pdb=output_pdb
        )

    # Build apptainer command
    cmd = [
        "apptainer", "exec",
        str(pyrosetta_image),
        "python", script_file, params_file
    ]

    logger.info(f"Running relax in container: {' '.join(cmd)}")
    logger.info(f"  Input PDB: {input_pdb}")
    logger.info(f"  Output PDB: {output_pdb}")
    logger.info(f"  Scorefunction: {config.scorefunction}")
    logger.info(f"  FastRelax cycles: {config.fastrelax_cycles}")

    try:
        logger.debug("Executing subprocess...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        # Always log stdout and stderr for debugging
        if result.stdout:
            logger.debug(f"Container stdout:\n{result.stdout}")
        if result.stderr:
            # stderr contains PyRosetta progress messages, log at debug level for success
            if result.returncode == 0:
                logger.debug(f"Container stderr:\n{result.stderr}")
            else:
                logger.error(f"Container stderr:\n{result.stderr}")

        logger.debug(f"Container exit code: {result.returncode}")

        if result.returncode != 0:
            error_msg = f"Container execution failed with exit code {result.returncode}"
            logger.error(error_msg)
            if result.stdout:
                logger.error(f"STDOUT: {result.stdout}")
            if result.stderr:
                logger.error(f"STDERR: {result.stderr}")
            return RelaxResult(
                success=False,
                input_pdb=input_pdb,
                error_message=f"{error_msg}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )

        # Read results JSON
        results_file = str(output_pdb).replace('.pdb', '_relax_result.json')
        logger.debug(f"Looking for results file: {results_file}")

        if Path(results_file).exists():
            logger.debug(f"Reading results from {results_file}")
            with open(results_file, 'r') as f:
                results_data = json.load(f)

            logger.info(f"Relax completed successfully for {input_pdb}")
            logger.info(f"  Total score: {results_data.get('total_score')}")
            logger.info(f"  Mean displacement: {results_data.get('mean_displacement')}")

            return RelaxResult(
                success=results_data['success'],
                input_pdb=input_pdb,
                output_pdb=output_pdb,
                total_score=results_data.get('total_score'),
                displacements=results_data.get('displacements', {}),
                mean_displacement=results_data.get('mean_displacement'),
                max_displacement=results_data.get('max_displacement')
            )
        else:
            logger.warning(f"Results file not found: {results_file}")
            # Check if output PDB was created
            if output_pdb.exists():
                logger.info(f"Output PDB exists at {output_pdb}, but no results JSON found")
                return RelaxResult(
                    success=True,
                    input_pdb=input_pdb,
                    output_pdb=output_pdb
                )
            else:
                error_msg = f"Output PDB not created at {output_pdb}"
                logger.error(error_msg)
                # Log container output for debugging even though return code was 0
                logger.error(f"Container returned 0 but output not created")
                if result.stdout:
                    logger.error(f"STDOUT: {result.stdout}")
                if result.stderr:
                    logger.error(f"STDERR: {result.stderr}")
                return RelaxResult(
                    success=False,
                    input_pdb=input_pdb,
                    error_message=f"{error_msg}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                )

    except subprocess.TimeoutExpired as e:
        error_msg = "Relax timed out after 1 hour"
        logger.error(error_msg)
        # Try to capture any partial output
        if hasattr(e, 'stdout') and e.stdout:
            logger.error(f"Partial STDOUT: {e.stdout}")
        if hasattr(e, 'stderr') and e.stderr:
            logger.error(f"Partial STDERR: {e.stderr}")
        return RelaxResult(
            success=False,
            input_pdb=input_pdb,
            error_message=error_msg
        )
    except Exception as e:
        error_msg = f"Container execution error: {e}"
        logger.error(error_msg)
        logger.exception("Full traceback:")
        return RelaxResult(
            success=False,
            input_pdb=input_pdb,
            error_message=error_msg
        )
    finally:
        # Clean up temp files
        import os
        logger.debug(f"Cleaning up temporary files: {params_file}, {script_file}")
        try:
            os.unlink(params_file)
            os.unlink(script_file)
        except Exception as cleanup_error:
            logger.debug(f"Failed to clean up temp files: {cleanup_error}")
