#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced FastMPNN Design Script for Enzyme Design

This script performs protein design using FastMPNN with PyRosetta for enzyme design.
It supports constraint handling, position biasing, and multi-layer design protocols.

Original author: ikalvet
Refactored for clarity and modularity.
"""

import os
import sys
import glob
import argparse
import subprocess

# Path setup - local modules are in same directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# =============================================================================
# EARLY CONTAINER CHECK - Must happen before heavy imports like pyrosetta
# =============================================================================

# Import only what we need for container detection
from constants import (
    DEFAULT_APPTAINER_IMAGE, CONTAINER_ENV_VAR,
    BETA_SCOREFUNCTIONS_REQUIRE_CONTAINER, DEFAULT_PYROSETTA_IMAGE
)


def _is_inside_container() -> bool:
    """Detect if we're running inside an Apptainer/Singularity container."""
    if os.environ.get(CONTAINER_ENV_VAR):
        return True
    if os.environ.get("SINGULARITY_CONTAINER") or os.environ.get("APPTAINER_CONTAINER"):
        return True
    if os.path.exists("/.singularity.d"):
        return True
    return False


def _early_reexec_in_container(container_image: str = DEFAULT_APPTAINER_IMAGE):
    """
    Re-execute the current script inside an Apptainer container.
    Called early, before heavy imports, when --container flag is detected.
    """
    script_path = os.path.abspath(sys.argv[0])
    args = [a for a in sys.argv[1:] if a != "--container"]  # Remove --container to avoid loop

    # Build bind paths
    bind_paths = set()
    bind_paths.add(os.getcwd())
    bind_paths.add(os.path.dirname(script_path))

    # Always bind fused_mpnn directory
    fused_mpnn_dir = "/net/software/lab/fused_mpnn"
    if os.path.exists(fused_mpnn_dir):
        bind_paths.add(fused_mpnn_dir)

    # Bind common software paths
    for path in ["/net/software", "/software", "/databases", "/projects"]:
        if os.path.exists(path):
            bind_paths.add(path)

    # Extract paths from arguments
    for i, arg in enumerate(args):
        if arg.startswith("--") and i + 1 < len(args):
            next_arg = args[i + 1]
            if not next_arg.startswith("--") and os.path.exists(next_arg):
                bind_paths.add(os.path.dirname(os.path.abspath(next_arg)))
        elif os.path.exists(arg):
            bind_paths.add(os.path.dirname(os.path.abspath(arg)))

    bind_mounts = ",".join(sorted(bind_paths))

    cmd = [
        "apptainer", "exec",
        "--bind", bind_mounts,
        "--env", f"{CONTAINER_ENV_VAR}=1",
        container_image,
        "python", script_path
    ] + args

    print("=" * 70)
    print("[Container] Re-executing inside Apptainer container")
    print(f"[Container] Image: {container_image}")
    print(f"[Container] Bind mounts: {bind_mounts}")
    print("=" * 70)
    print("")

    os.execvp("apptainer", cmd)


# Check for --container flag BEFORE importing pyrosetta
if "--container" in sys.argv and not _is_inside_container():
    _early_reexec_in_container(DEFAULT_APPTAINER_IMAGE)

# =============================================================================
# HEAVY IMPORTS - Only reached if we're inside container or not using --container
# =============================================================================

import pyrosetta as pyr
import pyrosetta.rosetta
import pyrosetta.distributed.io

# Local imports (from this directory)
import design_protocol
import rosetta_utils
import ref_pdb_utils
import pipeline_tracker
from hbond_selectors import SelectHBondsToResidue
import time
from constants import (
    DEFAULT_PARAMS, DEFAULT_DALPHABALL, DEFAULT_BIAS_AAS, DEFAULT_POSITION_BIAS,
    DEFAULT_PROTOCOL, DEFAULT_MPNN_RUNNER, DEFAULT_MODEL_TYPE, DEFAULT_OMIT_AA,
    DEFAULT_2ND_LAYER_TEMPS, DEFAULT_2ND_LAYER_BATCH_SIZE, DEFAULT_HBOND_ACCEPT_PROB,
    DEFAULT_LIGAND_MPNN_USE_SC_CONTEXT, DEFAULT_REPACK_EVERYTHING,
    DEFAULT_PACK_SIDE_CHAINS, DEFAULT_LIGAND_RIGIDITY, LIGAND_RIGIDITY_MODES,
    DEFAULT_SCOREFUNCTION, SCOREFUNCTION_PRESETS, DEFAULT_CONSTRAINT_WEIGHTS,
    DEFAULT_CART_BONDED_WEIGHT, DEFAULT_PRO_CLOSE_WEIGHT_CART, DEFAULT_COORD_CST_WEIGHT,
    BETA_SCOREFUNCTIONS, DEFAULT_INIT_CORRECTION,
    DEFAULT_CST_DEVIATION_THRESHOLD, DEFAULT_COORD_CST_NEIGHBOR_WINDOW,
    DEFAULT_REDUCED_COORD_CST_WEIGHT,
    DEFAULT_LAYER_CUTS, DEFAULT_POCKET_CUTS,
    DEFAULT_HBOND_KEEPER_ENABLED, KEEP_BEST_MODES, DEFAULT_KEEP_BEST_MODE,
    DEFAULT_CST_COMPARABLE_THRESHOLD, DEFAULT_ENHANCE_MODEL
)
from mpnn_runner import MPNNConfig, run_mpnn_from_pose


# =============================================================================
# Container Detection and Re-execution (for scorefunction-based container needs)
# =============================================================================

def is_inside_container() -> bool:
    """Detect if we're running inside an Apptainer/Singularity container."""
    return _is_inside_container()


def scorefunction_requires_container(scorefunction_name: str) -> bool:
    """Check if the specified scorefunction requires running in a container."""
    return scorefunction_name in BETA_SCOREFUNCTIONS_REQUIRE_CONTAINER


def reexec_in_container(container_image: str = DEFAULT_APPTAINER_IMAGE, reason: str = ""):
    """
    Re-execute the current script inside an Apptainer container.

    This function does not return - it replaces the current process with the containerized one.
    """
    import subprocess

    # Get the current command line arguments
    script_path = os.path.abspath(sys.argv[0])
    args = sys.argv[1:]

    # Remove --container flag to avoid infinite loop
    args = [a for a in args if a != "--container"]

    # Build the Apptainer command
    # We need to bind the current directory and any paths referenced in args
    bind_paths = set()
    bind_paths.add(os.getcwd())
    bind_paths.add(os.path.dirname(script_path))

    # Always bind the fused_mpnn directory
    fused_mpnn_dir = "/net/software/lab/fused_mpnn"
    if os.path.exists(fused_mpnn_dir):
        bind_paths.add(fused_mpnn_dir)

    # Bind common software paths
    for path in ["/net/software", "/software"]:
        if os.path.exists(path):
            bind_paths.add(path)

    # Try to extract paths from arguments (for --pdb, --params, --cstfile, etc.)
    for i, arg in enumerate(args):
        if arg.startswith("--") and i + 1 < len(args):
            next_arg = args[i + 1]
            if os.path.exists(next_arg):
                bind_paths.add(os.path.dirname(os.path.abspath(next_arg)))
        elif os.path.exists(arg):
            bind_paths.add(os.path.dirname(os.path.abspath(arg)))

    # Build bind mount string
    bind_mounts = ",".join(sorted(bind_paths))

    # Build the command
    cmd = [
        "apptainer", "exec",
        "--bind", bind_mounts,
        "--env", f"{CONTAINER_ENV_VAR}=1",
        container_image,
        "python", script_path
    ] + args

    print_header("Container Re-execution")
    if reason:
        print(f"  Reason: {reason}")
    print(f"  Container: {container_image}")
    print(f"  Re-executing inside Apptainer container...")
    print(f"  Bind mounts: {bind_mounts}")
    print(f"  Command: apptainer exec ... python {os.path.basename(script_path)} ...")
    print("")

    # Execute and replace current process
    try:
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print("\n" + "="*80)
        print("ERROR: Apptainer not found!")
        print("="*80)
        print(f"""
The scorefunction you specified requires running inside a container,
but 'apptainer' command was not found on your system.

SOLUTIONS:

  1. LOAD APPTAINER MODULE (if on HPC):
     module load apptainer
     # or
     module load singularity

  2. INSTALL APPTAINER:
     See: https://apptainer.org/docs/admin/main/installation.html

  3. USE A DIFFERENT SCOREFUNCTION:
     Use the default scorefunction which doesn't require a container:
     python enzyme_design.py --pdb input.pdb --scorefunction ref2015

  4. RUN MANUALLY INSIDE CONTAINER:
     If you have access to the container but apptainer isn't in PATH:
     /path/to/apptainer exec {container_image} python {script_path} {' '.join(args)}
""")
        print("="*80)
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Failed to execute in container: {e}")
        sys.exit(1)


# =============================================================================
# Debug Mode Utilities
# =============================================================================
class TeeOutput:
    """Write to both stdout and log file."""

    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        if self.log_file:
            self.log_file.close()


# =============================================================================
# Classes
# =============================================================================
class ConstraintManager:
    """Manages enzyme design constraints for PyRosetta poses."""

    def __init__(self, cstfile, scorefxn):
        self._scorefxn = scorefxn
        self._addcst_mover = pyrosetta.rosetta.protocols.enzdes.AddOrRemoveMatchCsts()
        chem_manager = pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance()
        residue_type_set = chem_manager.residue_type_set("fa_standard")
        self._cst_io = pyrosetta.rosetta.protocols.toolbox.match_enzdes_util.EnzConstraintIO(residue_type_set)
        self._cst_io.read_enzyme_cstfile(cstfile)

    def add_constraints(self, pose, force=False):
        """
        Add constraints to pose.

        Arguments:
            pose: PyRosetta Pose
            force: If True, remove existing constraints first. If False (default),
                   skip adding if pose already has enzyme constraints to avoid
                   conflicts with existing covalent connections/pseudobonds.
        """
        # Check if pose already has enzyme constraints to avoid pseudobond conflicts
        # This can happen when the pose already has covalent connections established
        # via reestablish_covalent_connections() in setup_pose_from_mpnn_output()
        if pose.constraint_set().has_constraints() and not force:
            # Already has constraints, skip to avoid "Unable to handle change in
            # the number of residue connections in the presence of pseudobonds" error
            return

        try:
            self._cst_io.add_constraints_to_pose(pose, self._scorefxn, True)
        except RuntimeError as e:
            # Handle the case where covalent connections conflict with existing pseudobonds
            error_msg = str(e)
            if "pseudobonds" in error_msg.lower() or "residue connections" in error_msg.lower():
                print(f"  [Constraints] Skipping constraint addition - pose already has covalent connections")
            else:
                raise  # Re-raise if it's a different error

    def remove_constraints(self, pose):
        """Remove constraints from pose."""
        self._cst_io.remove_constraints_from_pose(pose, True, True)

    def cst_io(self):
        """Return the constraint IO object."""
        return self._cst_io


# =============================================================================
# Helper Functions
# =============================================================================
def print_header(title, char="="):
    """Print a formatted section header."""
    width = 80
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_positions(label, positions):
    """Print a list of positions in a readable format."""
    print(f"{label}: {'+'.join(str(x) for x in sorted(positions))}")


def get_cpu_count():
    """Determine number of CPUs to use from environment."""
    nproc = os.cpu_count()
    if "OMP_NUM_THREADS" in os.environ:
        nproc = int(os.environ["OMP_NUM_THREADS"])
    if "SLURM_CPUS_ON_NODE" in os.environ:
        nproc = int(os.environ["SLURM_CPUS_ON_NODE"])
    return nproc


def estimate_expected_sequences(protocol_text, nstruct):
    """
    Estimate the expected number of output sequences from the protocol.

    Arguments:
        protocol_text: Protocol string or file path
        nstruct: Number of design iterations

    Returns:
        tuple: (estimated_sequences, breakdown_info)
    """
    if os.path.exists(protocol_text):
        with open(protocol_text, 'r') as f:
            lines = f.readlines()
    else:
        lines = protocol_text.strip().split('\n')

    # Parse protocol to track sequence counts
    current_count = 1  # Start with 1 pose
    mpnn_steps = []
    keep_best_steps = []
    shell_mpnn_steps = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cmd = parts[0]

        if cmd == 'mpnn':
            temp = float(parts[1]) if len(parts) > 1 else 0.1
            n_seq = int(parts[2]) if len(parts) > 2 else 10
            mpnn_steps.append((temp, n_seq))
            current_count *= n_seq

        elif cmd == 'keep_best':
            n_keep = int(parts[1]) if len(parts) > 1 else 5
            keep_best_steps.append(n_keep)
            current_count = min(current_count, n_keep)

        elif cmd == '2nd_shell_mpnn':
            temp = float(parts[1]) if len(parts) > 1 else 0.1
            n_seq = int(parts[2]) if len(parts) > 2 else 2
            shell_mpnn_steps.append((temp, n_seq))
            current_count *= n_seq

    # Final count per nstruct
    per_iter = current_count

    breakdown = {
        'mpnn_steps': mpnn_steps,
        'keep_best_steps': keep_best_steps,
        '2nd_shell_mpnn_steps': shell_mpnn_steps,
        'per_iteration': per_iter,
        'total': per_iter * nstruct
    }

    return per_iter * nstruct, breakdown


def format_duration(seconds):
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{int(mins)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{int(hours)}h {int(mins)}m"


def setup_pyrosetta(params_files, dalphaball_path, nproc, scorefunction_name=None):
    """
    Initialize PyRosetta with appropriate settings.

    Arguments:
        params_files: List of ligand/NCAA params files
        dalphaball_path: Path to DAlphaBall executable
        nproc: Number of CPU threads
        scorefunction_name: Scorefunction name (used to determine -corrections flag)
            If a beta scorefunction (e.g., beta_jan25) is specified, the appropriate
            -corrections: flag will be added to the init string.
    """
    extra_res_fa = "-extra_res_fa " + " ".join(params_files)
    multithreading = ""
    if nproc > 1:
        multithreading = f"-multithreading true -multithreading:total_threads {nproc} -multithreading:interaction_graph_threads {nproc}"

    # Determine the appropriate correction flag based on scorefunction
    # Beta scorefunctions (beta_jan25, beta_july15, etc.) require specific -corrections: flags
    correction_flag = DEFAULT_INIT_CORRECTION  # Default: -beta_nov16
    if scorefunction_name is not None and scorefunction_name in BETA_SCOREFUNCTIONS:
        correction_flag = BETA_SCOREFUNCTIONS[scorefunction_name]

    print_header("Initializing PyRosetta")
    print(f"  Params files: {', '.join(params_files)}")
    print(f"  CPU threads: {nproc}")
    print(f"  Correction flag: {correction_flag}")
    if scorefunction_name in BETA_SCOREFUNCTIONS:
        print(f"  NOTE: Using beta scorefunction '{scorefunction_name}' - requires {correction_flag}")

    pyr.init(f"{extra_res_fa} -dalphaball {dalphaball_path} {correction_flag} -run:preserve_header {multithreading}")


def setup_scorefxn_with_constraints(cstfile=None, scorefunction_name="ref2015",
                                     constraint_weights=None, verbose=True):
    """
    Create scorefunction and optionally add constraint weights.

    Arguments:
        cstfile: Path to enzyme constraint file (.cst)
        scorefunction_name: Name of scorefunction to use (default: ref2015)
        constraint_weights: Dict with constraint weights, or None for defaults
        verbose: Print scorefunction info

    Returns:
        tuple: (ScoreFunction, ConstraintManager or None)
    """
    print_header("Scorefunction Setup", "-")

    # Create base scorefunction
    if scorefunction_name == "ref2015" or scorefunction_name is None:
        # Use get_fa_scorefxn() which respects -beta_nov16 flag from init
        sfx = pyr.get_fa_scorefxn()
        if verbose:
            print(f"  Base scorefunction: REF2015 (with beta_nov16 from init)")
    else:
        sfx = rosetta_utils.create_scorefunction(scorefunction_name, verbose=verbose)

    cst_manager = None

    if cstfile is not None:
        # Use default constraint weights if not specified
        if constraint_weights is None:
            constraint_weights = DEFAULT_CONSTRAINT_WEIGHTS

        rosetta_utils.add_constraint_weights(
            sfx,
            atom_pair=constraint_weights.get("atom_pair_constraint", 1.0),
            angle=constraint_weights.get("angle_constraint", 1.0),
            dihedral=constraint_weights.get("dihedral_constraint", 1.0),
            verbose=verbose
        )
        cst_manager = ConstraintManager(cstfile, sfx)

    return sfx, cst_manager


def identify_design_layers(pose, ligand_seqpos, keep_pos, design_pos, heavyatoms,
                           detect_pocket=False, layer_cuts=None):
    """
    Identify which residues belong to design, repack, and fixed layers.

    Arguments:
        pose: PyRosetta Pose
        ligand_seqpos: Ligand sequence position
        keep_pos: Positions to keep fixed (repack only)
        design_pos: Positions to force into design
        heavyatoms: Ligand heavy atom names
        detect_pocket: Auto-detect pocket residues
        layer_cuts: [inner_design, outer_design, inner_repack, outer_repack] in Angstroms
                   Default: [6.0, 8.0, 10.0, 12.0]
    """
    if layer_cuts is None:
        layer_cuts = DEFAULT_LAYER_CUTS

    print_header("Layer Selection", "-")
    print(f"  Layer cutoffs: design≤{layer_cuts[0]}Å, design≤{layer_cuts[1]}Å(CB), repack≤{layer_cuts[2]}Å, repack≤{layer_cuts[3]}Å(CB)")

    ligand = pose.residue(ligand_seqpos)

    SEL_mutate, SEL_repack, SEL_do_not_repack, residues = rosetta_utils.get_layer_selections(
        pose, keep_pos, design_pos, ligand_seqpos, heavyatoms, cuts=layer_cuts
    )

    if not detect_pocket:
        if design_pos:
            design_residues = design_pos
        else:
            design_residues = [res.seqpos() for res in pose.residues if not res.is_ligand() and res.seqpos() not in keep_pos]
    else:
        substrate_atoms_ref = [ligand.atom_name(n).strip() for n in range(1, ligand.natoms()+1)
                              if ligand.atom_name(n).strip() not in ["ZN1", "O1"]]
        # Use pocket cuts which are slightly larger than layer cuts
        pocket_cuts = [c + 1.0 for c in layer_cuts]
        _, _, _, residues_substrate = rosetta_utils.get_layer_selections(
            pose, keep_pos, design_pos, ligand_seqpos, substrate_atoms_ref, cuts=pocket_cuts
        )
        design_residues = list(residues_substrate[0] + residues_substrate[1])
        design_residues += rosetta_utils.get_residues_with_close_sc(
            pose, substrate_atoms_ref, residues_substrate[2] + residues_substrate[3], keep_pos, 8.0
        )
        design_residues = list(set(design_residues))

    repack_residues = residues[2] + residues[3] + residues[4] + [ligand_seqpos]
    for res in residues[0] + residues[1]:
        if res not in design_residues:
            repack_residues.append(res)
    repack_residues = [x for x in repack_residues if x not in design_residues]

    do_not_touch_residues = []
    unclassified = [res.seqpos() for res in pose.residues
                    if res.seqpos() not in design_residues + repack_residues + do_not_touch_residues]
    assert len(unclassified) == 0, f"Unclassified residues found: {unclassified}"

    print_positions("  Design positions", design_residues)
    print(f"  Repack positions: {len(repack_residues)} residues")

    return design_residues, repack_residues, do_not_touch_residues


def prerelax_structure(pose, sfx, cst_manager, keep_pos, ligand_seqpos, use_cartesian=True,
                       ligand_rigidity="fixed", cart_bonded_weight=DEFAULT_CART_BONDED_WEIGHT,
                       coord_cst_weights=None):
    """
    Perform constrained FastRelax on the input structure.

    Arguments:
        pose: PyRosetta Pose
        sfx: ScoreFunction
        cst_manager: ConstraintManager or None
        keep_pos: List of positions to keep fixed
        ligand_seqpos: Ligand sequence position
        use_cartesian: Use Cartesian minimization (default: True)
        ligand_rigidity: Ligand rigidity mode
        cart_bonded_weight: Weight for cart_bonded term in Cartesian mode
        coord_cst_weights: Per-residue coordinate constraint weights dict, or None for uniform

    Returns:
        Pose: Relaxed pose
    """
    print_header("Pre-relaxation", "-")
    print(f"  Cartesian mode: {use_cartesian}")
    print(f"  Ligand rigidity: {ligand_rigidity}")
    if coord_cst_weights is not None:
        print(f"  Adaptive coord constraints: Enabled (per-residue weights)")
    else:
        print(f"  Adaptive coord constraints: Disabled (uniform weights)")

    sfx_relax = sfx.clone()
    if use_cartesian:
        rosetta_utils.configure_for_cartesian(
            sfx_relax,
            cart_bonded=cart_bonded_weight,
            pro_close=DEFAULT_PRO_CLOSE_WEIGHT_CART,
            verbose=True
        )

    # Use adaptive FastRelax setup if per-residue weights provided
    if coord_cst_weights is not None:
        fastRelax = rosetta_utils.setup_fastrelax_adaptive(
            sfx_relax, coord_cst_weights=coord_cst_weights, crude=True,
            disable_min_resons=[ligand_seqpos], pose=pose, ligand_rigidity=ligand_rigidity
        )
    else:
        fastRelax = rosetta_utils.setup_fastrelax(
            sfx_relax, crude=True, disable_min_resons=[ligand_seqpos],
            pose=pose, ligand_rigidity=ligand_rigidity
        )
    fastRelax.cartesian(use_cartesian)

    ligand = pose.residue(ligand_seqpos)
    target_atoms = [n for n in range(1, ligand.natoms()+1) if not ligand.atom_is_hydrogen(n)]
    residue_range = [n for n in range(1, pose.size()) if n not in keep_pos]

    clashes = rosetta_utils.find_clashes_between_target_and_sidechains(
        pose, pose.size(), target_atoms=target_atoms, residues=residue_range
    )
    clashes = [x for x in clashes if pose.residue(x).name3() not in ["ALA", "GLY", "PRO"] and x not in keep_pos]

    if clashes:
        print(f"  Mutating {len(clashes)} clashing residues to ALA")

    relaxed_pose = rosetta_utils.mutate_residues(pose, clashes, "ALA")

    if cst_manager is not None:
        cst_manager.cst_io().add_constraints_to_pose(relaxed_pose, sfx_relax, True)

    # Apply per-residue coordinate constraints if adaptive mode
    if coord_cst_weights is not None:
        rosetta_utils.apply_coordinate_constraints_with_weights(
            relaxed_pose, coord_cst_weights, reference_pose=pose
        )

    fastRelax.apply(relaxed_pose)
    sfx_relax(relaxed_pose)

    if cst_manager is not None:
        cst_score = sum([relaxed_pose.scores[s] for s in relaxed_pose.scores if "constraint" in s])
        print(f"  Constraint score after relax: {cst_score:.2f}")

    return relaxed_pose


def compute_bias_positions(pose, ligand_seqpos, keep_pos, design_pos, bias_atoms, bias_aas, position_bias):
    """Compute per-position biases for specified atoms."""
    print_header("Position Bias Configuration", "-")

    assert all([pose.residue(ligand_seqpos).has(a) for a in bias_atoms]), "Invalid --bias_atoms atom names."

    _, _, _, residues_bias = rosetta_utils.get_layer_selections(
        pose, keep_pos, design_pos, ligand_seqpos, bias_atoms, cuts=[5.0, 7.0, 9.0, 11.0]
    )
    bias_positions = list(set(residues_bias[0] + residues_bias[1]))

    # Exclude positions closer to O1 than to bias atoms
    if "O1" not in bias_atoms and "H1" in bias_atoms:
        filtered_positions = []
        for pos in bias_positions:
            dist_nbr = (pose.residue(pos).xyz("CA") - pose.residue(ligand_seqpos).xyz("O1")).norm()
            dists = [(pose.residue(pos).xyz("CA") - pose.residue(ligand_seqpos).xyz(a)).norm() for a in bias_atoms]
            if min(dists) < dist_nbr:
                filtered_positions.append(pos)
        bias_positions = filtered_positions

    bias_dict = {}
    for pos in bias_positions:
        chain = pose.pdb_info().chain(pos)
        bias_dict[f"{chain}{pos}"] = {a: position_bias for a in bias_aas}

    print(f"  Bias AAs: {bias_aas}")
    print(f"  Bias value: {position_bias}")
    print_positions("  Biased positions", bias_positions)

    return bias_dict


def get_2nd_layer_fixed_positions(pose, ligand_seqpos, heavyatoms, keep_pos):
    """Determine fixed positions for 2nd layer MPNN design."""
    dist_bb = 6.0
    dist_sc = 5.0

    motif_sel = pyrosetta.rosetta.core.select.residue_selector.ResiduePDBInfoHasLabelSelector(
        label_str="keep_hbonds_to_ligand_and_catres"
    )
    pocket_positions = keep_pos + list(pyrosetta.rosetta.core.select.get_residue_set_from_subset(motif_sel.apply(pose)))
    pocket_positions = list(set(pocket_positions))

    _, _, _, residues = rosetta_utils.get_layer_selections(
        pose, repack_only_pos=pocket_positions, design_pos=[], ref_resno=ligand_seqpos,
        heavyatoms=heavyatoms, cuts=[dist_bb, dist_bb+2.0, dist_bb+4.0, dist_bb+6.0], design_GP=True
    )

    close_ones = rosetta_utils.get_residues_with_close_sc(
        pose, heavyatoms, residues[1] + residues[2], exclude_residues=pocket_positions, cutoff=dist_sc
    )
    pocket_positions += residues[0] + close_ones
    pocket_positions = list(set(pocket_positions))

    design_residues = [x for x in residues[0] + residues[1] + residues[2] + residues[3] if x not in pocket_positions]

    # Include alanines not in pocket
    ala_positions = [res.seqpos() for res in pose.residues
                     if res.seqpos() not in pocket_positions + design_residues and res.name3() == "ALA"]
    if ala_positions:
        print(f"  Including ALA positions: {'+'.join(str(x) for x in ala_positions)}")
    design_residues += ala_positions

    fixed_residues = [f"{pose.pdb_info().chain(r.seqpos())}{pose.pdb_info().number(r.seqpos())}"
                      for r in pose.residues if r.seqpos() not in design_residues and r.is_protein()]
    return fixed_residues


def run_secondary_mpnn(pose, pdbname, suffix, n_iter, i, ligand_seqpos, heavyatoms, keep_pos,
                       catres, cst_manager, sfx, mpnn_config, temperatures=None):
    """Run secondary MPNN design on successful outputs using mpnn_runner."""
    if temperatures is None:
        temperatures = DEFAULT_2ND_LAYER_TEMPS

    fixed_residues = get_2nd_layer_fixed_positions(pose, ligand_seqpos, heavyatoms, keep_pos)

    for T in temperatures:
        # Format temperature with underscore instead of period (T0.1 -> T0_1)
        T_str = str(T).replace(".", "_")
        name = f"{pdbname}{suffix}_{n_iter}_{i}_T{T_str}"
        result = run_mpnn_from_pose(pose, mpnn_config, "seqs/", name, temperature=T,
                                    batch_size=DEFAULT_2ND_LAYER_BATCH_SIZE, num_batches=1,
                                    fixed_residues=fixed_residues)
        if not result["success"]:
            continue

        # Process generated structures - thread onto pose and repack with Rosetta
        for seqid, output_file in enumerate(result["output_files"]):
            try:
                mpnn_pose = pyr.pose_from_file(output_file)
                threaded_pose = rosetta_utils.thread_seq_to_pose(pose, mpnn_pose.sequence())
                threaded_pose = rosetta_utils.fix_catalytic_residue_rotamers(threaded_pose, pose, catres)
                if cst_manager is not None:
                    cst_manager.add_constraints(threaded_pose)
                packed_pose = rosetta_utils.repack(threaded_pose, sfx)
                sfx(packed_pose)
                packed_pose.dump_pdb(f"seqs/{pdbname}{suffix}_{n_iter}_{i}_T{T_str}_{seqid}.pdb")
            except Exception as e:
                print(f"  [2nd Layer MPNN] WARNING: Failed to process {output_file}: {e}")


def run_design_iteration(pose, fmd, cst_manager, scoring_module, pdbname, suffix, n_iter,
                         catres, apply_filter=False, run_mpnn=False, mpnn_config=None,
                         ligand_seqpos=None, heavyatoms=None, keep_pos=None, sfx=None,
                         scorefilename=None, input_pose=None, iteration_start_time=None,
                         ref_pose=None, ref_catres=None):
    """Run a single design iteration and process outputs."""
    poses = fmd.apply(pose)

    # Remove duplicate sequences
    unique_poses = []
    seen_sequences = set()
    for p in poses:
        seq = p.sequence()
        if seq not in seen_sequences:
            seen_sequences.add(seq)
            unique_poses.append(p)
        else:
            print(f"  [DUPLICATE] Removing duplicate sequence")

    if len(unique_poses) < len(poses):
        print(f"  [DUPLICATE] Removed {len(poses) - len(unique_poses)} duplicate sequences, {len(unique_poses)} unique remain")

    poses = unique_poses
    saved_count = 0

    for i, p in enumerate(poses):
        output_name = f"{pdbname}{suffix}_{n_iter}_{i}"

        if cst_manager is not None:
            cst_manager.add_constraints(p)

        scores_df = scoring_module.score_design(p, pyr.get_fa_scorefxn(), list(catres.keys()))

        # Add RMSD and sequence identity metrics
        if input_pose is not None:
            # CA RMSD
            ca_rmsd = rosetta_utils.calculate_ca_rmsd(p, input_pose)
            scores_df.at[0, "CA_rmsd"] = ca_rmsd

            # CA RMSD converged (removing outliers)
            ca_rmsd_conv, n_used, n_outliers = rosetta_utils.calculate_ca_rmsd_converged(p, input_pose)
            scores_df.at[0, "CA_rmsd_converge"] = ca_rmsd_conv
            scores_df.at[0, "CA_rmsd_n_outliers"] = n_outliers

            # Sequence identity
            seq_id = rosetta_utils.calculate_sequence_identity(p, input_pose)
            scores_df.at[0, "sequence_identity"] = seq_id

        # Add ref_pdb metrics if reference pose available
        if ref_pose is not None and ref_catres is not None:
            try:
                ref_metrics = ref_pdb_utils.calculate_ref_pdb_metrics(
                    p, ref_pose, ref_catres, input_pose=input_pose, verbose=False
                )
                for metric_name, value in ref_metrics.items():
                    scores_df.at[0, metric_name] = value
            except Exception as e:
                print(f"  [WARNING] Failed to calculate ref_pdb metrics: {e}")

        # Add timing info if available
        if iteration_start_time is not None:
            elapsed = time.time() - iteration_start_time
            scores_df.at[0, "design_time_sec"] = elapsed

        if apply_filter and len(scoring_module.filter_scores(scores_df)) == 0:
            print(f"  [FILTERED] {output_name} - did not pass filters")
            continue

        scores_df.at[0, "description"] = output_name
        rosetta_utils.dump_scorefile(scores_df, scorefilename)
        p.dump_pdb(f"{output_name}.pdb")
        saved_count += 1

        # Print save message with metrics if available
        if input_pose is not None and ref_pose is not None:
            lig_rmsd_ref = scores_df.at[0, "lig_rmsd_to_refpdb"] if "lig_rmsd_to_refpdb" in scores_df.columns else float('nan')
            print(f"  [SAVED] {output_name}.pdb (CA_rmsd={ca_rmsd:.2f}Å, seq_id={seq_id:.1%}, lig_rmsd_ref={lig_rmsd_ref:.3f}Å)")
        elif input_pose is not None:
            print(f"  [SAVED] {output_name}.pdb (CA_rmsd={ca_rmsd:.2f}Å, seq_id={seq_id:.1%})")
        else:
            print(f"  [SAVED] {output_name}.pdb")

        if run_mpnn and mpnn_config is not None:
            run_secondary_mpnn(p, pdbname, suffix, n_iter, i, ligand_seqpos, heavyatoms,
                               keep_pos, catres, cst_manager, sfx, mpnn_config)

    return saved_count


# =============================================================================
# Argument Parsing
# =============================================================================
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced FastMPNN Design - Enzyme design with constraints and biasing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument("--pdb", type=str, required=True, help="Input PDB file with ligand and matcher CST lines in header")

    # Output control
    parser.add_argument("--nstruct", type=int, default=1, help="Number of design iterations (default: 1)")
    parser.add_argument("--suffix", type=str, default="", help="Suffix added to output filenames")

    # Structure parameters
    parser.add_argument("--params", type=str, nargs="+", help="Ligand and NCAA params file(s)")
    parser.add_argument("--cstfile", type=str, help="Matcher/enzdes constraint file")

    # Design positions
    parser.add_argument("--design_pos", type=int, nargs="+", help="Positions to redesign (if not specified, uses pocket detection)")
    parser.add_argument("--keep_pos", type=int, nargs="+", help="Positions to keep fixed (repack allowed)")
    parser.add_argument("--global_seq_redesign", action="store_true", default=False, help="Redesign entire sequence globally instead of just pocket residues")

    # Layer selection distances (distance from CA to closest ligand heavy atom)
    parser.add_argument("--layer_design_inner", type=float, default=DEFAULT_LAYER_CUTS[0], help=f"Inner design layer cutoff in Angstroms (default: {DEFAULT_LAYER_CUTS[0]})")
    parser.add_argument("--layer_design_outer", type=float, default=DEFAULT_LAYER_CUTS[1], help=f"Outer design layer cutoff in Angstroms (default: {DEFAULT_LAYER_CUTS[1]})")
    parser.add_argument("--layer_repack_inner", type=float, default=DEFAULT_LAYER_CUTS[2], help=f"Inner repack layer cutoff in Angstroms (default: {DEFAULT_LAYER_CUTS[2]})")
    parser.add_argument("--layer_repack_outer", type=float, default=DEFAULT_LAYER_CUTS[3], help=f"Outer repack layer cutoff in Angstroms (default: {DEFAULT_LAYER_CUTS[3]})")

    # Protocol and scoring
    parser.add_argument("--protocol", type=str, help="Text file defining FastMPNNDesign protocol")
    parser.add_argument("--scoring", type=str, help="Scoring script for custom scores and filtering")

    # Biasing options
    parser.add_argument("--position_bias", type=float, default=DEFAULT_POSITION_BIAS, help=f"Bias for polar AAs near --bias_atoms (default: {DEFAULT_POSITION_BIAS})")
    parser.add_argument("--bias_atoms", nargs="+", type=str, help="Ligand atom names for position biasing")
    parser.add_argument("--bias_AAs", type=str, default=DEFAULT_BIAS_AAS, help=f"AAs to bias near --bias_atoms (default: {DEFAULT_BIAS_AAS})")

    # Processing options
    parser.add_argument("--filter", action="store_true", default=False, help="Only output designs passing filter criteria")
    parser.add_argument("--2nd-shell-mpnn-seqs", dest="second_shell_mpnn_seqs", action="store_true", default=False, help="Run additional 2nd shell MPNN and save to seqs/ folder")

    # H-bond keeper options
    parser.add_argument("--hbond_accept_prob", type=float, default=DEFAULT_HBOND_ACCEPT_PROB, help=f"Probability to fix each H-bonding residue (default: {DEFAULT_HBOND_ACCEPT_PROB})")
    parser.add_argument("--disable_hbond_keeper", action="store_true", default=False, help="Disable automatic fixing of H-bonding residues")

    # Keep best scoring options
    parser.add_argument("--keep_best_mode", type=str, default=DEFAULT_KEEP_BEST_MODE, choices=KEEP_BEST_MODES, help=f"Scoring mode for keep_best: cst_priority (prioritize constraints), total_score (original), cst_only (default: {DEFAULT_KEEP_BEST_MODE})")
    parser.add_argument("--cst_comparable_threshold", type=float, default=DEFAULT_CST_COMPARABLE_THRESHOLD, help=f"Threshold for 'comparable' constraint scores in cst_priority mode (default: {DEFAULT_CST_COMPARABLE_THRESHOLD} REU)")

    # Ligand rigidity options
    parser.add_argument("--ligand_rigidity", type=str, default=DEFAULT_LIGAND_RIGIDITY, choices=LIGAND_RIGIDITY_MODES, help=f"Ligand rigidity mode: 'fixed' (no movement, DEFAULT), 'rigid_body' (rigid-body only), 'flexible' (full flexibility)")

    # Scorefunction options
    parser.add_argument("--scorefunction", type=str, default=DEFAULT_SCOREFUNCTION, help=f"Scorefunction name or weights file (default: {DEFAULT_SCOREFUNCTION}). Options: ref2015, ref2015_cart, beta_nov16, talaris2014, or beta scorefunctions (beta_jan25, beta_july15) which require special init flags")
    parser.add_argument("--constraint_weight", type=float, default=1.0, help="Weight for all constraint terms (default: 1.0)")
    parser.add_argument("--cart_bonded_weight", type=float, default=DEFAULT_CART_BONDED_WEIGHT, help=f"Weight for cart_bonded term in Cartesian relax (default: {DEFAULT_CART_BONDED_WEIGHT})")
    parser.add_argument("--no_cartesian_relax", action="store_true", default=False, help="Use torsion-space relaxation instead of Cartesian")

    # Adaptive coordinate constraint options
    parser.add_argument("--adaptive_coord_cst", action="store_true", default=False, help="Enable adaptive coordinate constraints - catalytic residues far from their constraints get reduced coord cst weights")
    parser.add_argument("--cst_deviation_threshold", type=float, default=DEFAULT_CST_DEVIATION_THRESHOLD, help=f"Constraint score threshold above which a residue is 'far off' (default: {DEFAULT_CST_DEVIATION_THRESHOLD} REU)")
    parser.add_argument("--coord_cst_neighbor_window", type=int, default=DEFAULT_COORD_CST_NEIGHBOR_WINDOW, help=f"Number of neighboring residues (±N) to also reduce constraints for (default: {DEFAULT_COORD_CST_NEIGHBOR_WINDOW})")
    parser.add_argument("--reduced_coord_cst_weight", type=float, default=DEFAULT_REDUCED_COORD_CST_WEIGHT, help=f"Reduced coordinate constraint weight for far-off residues (default: {DEFAULT_REDUCED_COORD_CST_WEIGHT})")

    # Reference PDB options
    parser.add_argument("--ref_pdb", type=str, default=None, help="Reference PDB with ideal catalytic residue and ligand positioning. Will be aligned to input PDB by ligand atoms.")
    parser.add_argument("--catres_cst_subset", type=str, default=None, help="Comma-separated list of REMARK 666 line numbers to consider (1-indexed). E.g., '1,3,5'. Default: use all.")
    parser.add_argument("--ref_coord_cst", action="store_true", default=False, help="Include 3D coordinate constraints for functional groups from ref_pdb (in addition to 2D constraints)")

    # MPNN configuration
    parser.add_argument("--mpnn_runner", type=str, default=DEFAULT_MPNN_RUNNER, help=f"Path to MPNN run.py (default: {DEFAULT_MPNN_RUNNER})")
    parser.add_argument("--mpnn_model_type", type=str, default=DEFAULT_MODEL_TYPE, help=f"Model type (default: {DEFAULT_MODEL_TYPE})")
    parser.add_argument("--mpnn_omit_aa", type=str, default=DEFAULT_OMIT_AA, help=f"AAs to omit (default: {DEFAULT_OMIT_AA})")
    parser.add_argument("--apptainer_image", type=str, default=DEFAULT_APPTAINER_IMAGE, help=f"Apptainer image (default: {DEFAULT_APPTAINER_IMAGE})")
    parser.add_argument("--mpnn_use_sc_context", type=int, default=DEFAULT_LIGAND_MPNN_USE_SC_CONTEXT, help=f"Use SC context (default: {DEFAULT_LIGAND_MPNN_USE_SC_CONTEXT})")
    parser.add_argument("--mpnn_pack_side_chains", type=int, default=DEFAULT_PACK_SIDE_CHAINS, help=f"Pack designed SCs (default: {DEFAULT_PACK_SIDE_CHAINS})")
    parser.add_argument("--mpnn_repack_everything", type=int, default=DEFAULT_REPACK_EVERYTHING, help=f"Repack all SCs (default: {DEFAULT_REPACK_EVERYTHING})")

    # MPNN model caching (default: enabled - loads models once, reuses for all MPNN calls)
    parser.add_argument("--no_mpnn_cache", action="store_true", default=False,
                        help="Disable model caching (slower - reloads models each MPNN call)")

    # MPNN external server mode (optional - for connecting to a separate server process)
    parser.add_argument("--mpnn_server", action="store_true", default=False,
                        help="Connect to external MPNN server instead of local execution")
    parser.add_argument("--mpnn_server_host", type=str, default="localhost",
                        help="External MPNN server hostname (default: localhost)")
    parser.add_argument("--mpnn_server_port", type=int, default=5000,
                        help="External MPNN server port (default: 5000)")

    # Container execution
    parser.add_argument("--container", action="store_true", default=False, help="Run inside Apptainer container (auto-binds necessary paths)")

    # Debug mode
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug mode: log output to full_log.txt and save intermediate structures")

    # Quick test mode - minimal pipeline for fast debugging
    parser.add_argument("--quick_test", action="store_true", default=False,
                        help="Run minimal pipeline for fast testing (~30s instead of ~500s)")

    return parser.parse_args()


# =============================================================================
# Main Execution
# =============================================================================
def main():
    """Main execution function."""
    args = parse_arguments()
    pipeline_start_time = time.time()

    # Note: --container flag is handled early (before imports) at the top of this file
    # Here we only check if the scorefunction requires a container
    if scorefunction_requires_container(args.scorefunction) and not is_inside_container():
        # Re-execute inside the container - this function does not return
        reexec_in_container(DEFAULT_PYROSETTA_IMAGE, reason=f"scorefunction '{args.scorefunction}' requires container")

    # If we get here, we're either:
    # 1. Not using a container-requiring scorefunction, or
    # 2. Already running inside the container

    print_header("FastMPNN Zinc Esterase Design")
    print(f"  Input PDB: {args.pdb}")
    print(f"  N structures: {args.nstruct}")
    print(f"  Ligand rigidity: {args.ligand_rigidity}")
    print(f"  Scorefunction: {args.scorefunction}")
    print(f"  Cartesian relax: {not args.no_cartesian_relax}")
    if is_inside_container():
        print(f"  Running in container: Yes")

    # Estimate expected sequences from protocol
    protocol_text = args.protocol if args.protocol else DEFAULT_PROTOCOL
    expected_seqs, breakdown = estimate_expected_sequences(protocol_text, args.nstruct)
    print(f"\n  === Expected Output Summary ===")
    print(f"  MPNN steps: {len(breakdown['mpnn_steps'])} ({', '.join(f'T{t} x{n}' for t, n in breakdown['mpnn_steps'])})")
    print(f"  Keep best steps: {len(breakdown['keep_best_steps'])} ({', '.join(str(k) for k in breakdown['keep_best_steps'])})")
    if breakdown['2nd_shell_mpnn_steps']:
        print(f"  2nd shell MPNN: {len(breakdown['2nd_shell_mpnn_steps'])} ({', '.join(f'T{t} x{n}' for t, n in breakdown['2nd_shell_mpnn_steps'])})")
    print(f"  Per iteration: ~{breakdown['per_iteration']} sequences")
    print(f"  Total expected: ~{expected_seqs} sequences (from {args.nstruct} iterations)")

    # Setup paths and names
    pdbname = os.path.basename(args.pdb).replace(".pdb", "")
    suffix = f"_{args.suffix}" if args.suffix else ""

    # Setup debug mode if enabled
    debug_dir = None
    debug_structures_dir = None
    if args.debug:
        debug_dir = f"debug_output_{pdbname}{suffix}"
        debug_structures_dir = os.path.join(debug_dir, "structures")
        os.makedirs(debug_dir, exist_ok=True)
        os.makedirs(debug_structures_dir, exist_ok=True)
        log_path = os.path.join(debug_dir, "full_log.txt")
        tee = TeeOutput(log_path)
        sys.stdout = tee
        sys.stderr = tee
        print(f"[DEBUG] Debug mode enabled")
        print(f"[DEBUG] Log file: {log_path}")
        print(f"[DEBUG] Structures directory: {debug_structures_dir}")

    # Setup directories
    os.makedirs("scores/", exist_ok=True)
    scorefilename = f"scores/{pdbname}{suffix}.sc"
    if args.second_shell_mpnn_seqs:
        os.makedirs("seqs/", exist_ok=True)

    # Load scoring module
    if args.scoring:
        scoring_path = args.scoring
        sys.path.append(os.path.dirname(scoring_path))
        scoring_module = __import__(os.path.basename(scoring_path.replace(".py", "")))
        print(f"  Scoring module: {scoring_path} (custom)")
    else:
        import default_scoring as scoring_module
        scoring_path = os.path.join(SCRIPT_DIR, "default_scoring.py")
        print(f"  Scoring module: {scoring_path} (default)")

    assert hasattr(scoring_module, "score_design"), "Scoring module must have score_design function"
    assert hasattr(scoring_module, "filter_scores"), "Scoring module must have filter_scores function"
    assert hasattr(scoring_module, "filters"), "Scoring module must have filters attribute"

    # Setup params
    params_files = args.params if args.params else DEFAULT_PARAMS

    # Initialize PyRosetta
    nproc = get_cpu_count()
    setup_pyrosetta(params_files, DEFAULT_DALPHABALL, nproc, scorefunction_name=args.scorefunction)

    # Load pose
    print_header("Loading Structure", "-")

    # First, get ALL catalytic residues from the original PDB (for metrics tracking)
    all_catres_original = rosetta_utils.get_matcher_residues(args.pdb)
    print(f"  Original PDB has {len(all_catres_original)} REMARK 666 lines (catalytic residues)")

    # If a CST file is provided, filter the PDB to remove REMARK 666 lines
    # that reference constraint blocks that don't exist in the CST file
    pdb_to_load = args.pdb
    if args.cstfile:
        num_cst_blocks = rosetta_utils.count_cst_blocks(args.cstfile)
        print(f"  CST file has {num_cst_blocks} constraint blocks")
        pdb_to_load = rosetta_utils.filter_pdb_remark666_by_cst_blocks(
            args.pdb, args.cstfile, verbose=True
        )
        if pdb_to_load != args.pdb:
            print(f"  Using filtered PDB (removed invalid REMARK 666 lines)")

    pose = pyr.pose_from_file(pdb_to_load)

    # Clean up temp filtered PDB if created
    if pdb_to_load != args.pdb and os.path.exists(pdb_to_load):
        os.remove(pdb_to_load)

    ligand_seqpos = pose.size()
    assert pose.residue(ligand_seqpos).is_ligand(), f"Expected ligand at position {ligand_seqpos}"
    print(f"  Pose loaded: {pose.size()} residues")
    print(f"  Ligand at position: {ligand_seqpos}")

    # Identify catalytic residues that have valid constraint blocks
    catres = rosetta_utils.get_matcher_residues(pose)
    if len(catres) < len(all_catres_original):
        print(f"  Catalytic residues with constraints: {list(catres.keys())} ({len(catres)} of {len(all_catres_original)})")
        unconstrained = set(all_catres_original.keys()) - set(catres.keys())
        print(f"  Catalytic residues WITHOUT constraints (no CST block): {sorted(unconstrained)}")
    else:
        print(f"  Catalytic residues: {list(catres.keys())}")

    # Handle reference PDB if provided
    ref_pose = None
    ref_catres = None
    input_catres = None

    if args.ref_pdb:
        print_header("Reference PDB Processing", "-")
        ref_result = ref_pdb_utils.process_ref_pdb(
            pose, args.ref_pdb,
            catres_cst_subset=args.catres_cst_subset,
            include_coord_cst=args.ref_coord_cst,
            derive_cst=(args.cstfile is None),  # Only derive if no cstfile provided
            verbose=True
        )
        ref_pose = ref_result['ref_pose']
        ref_catres = ref_result['ref_catres']
        input_catres = ref_result['input_catres']

        # Use derived constraints if no cstfile provided
        if args.cstfile is None and ref_result['cst_file_path']:
            args.cstfile = ref_result['cst_file_path']
            print(f"  Using derived constraints from ref_pdb: {args.cstfile}")

        # Update catres to use subset if specified
        if args.catres_cst_subset:
            catres = input_catres
            print(f"  Using catres subset: {list(catres.keys())}")

    # Setup scorefunction and constraints
    constraint_weights = {
        "atom_pair_constraint": args.constraint_weight,
        "angle_constraint": args.constraint_weight,
        "dihedral_constraint": args.constraint_weight,
    }
    sfx, cst_manager = setup_scorefxn_with_constraints(
        cstfile=args.cstfile,
        scorefunction_name=args.scorefunction,
        constraint_weights=constraint_weights if args.cstfile else None
    )


    keep_pos = list(args.keep_pos) if args.keep_pos else []
    keep_pos += list(catres.keys())

    motif_sel = pyrosetta.rosetta.core.select.residue_selector.ResiduePDBInfoHasLabelSelector(label_str="motif")
    motif_pos = list(pyrosetta.rosetta.core.select.get_residue_set_from_subset(motif_sel.apply(pose)))
    keep_pos += motif_pos
    keep_pos = list(set(keep_pos))

    if motif_pos:
        print(f"  Motif residues: {motif_pos}")

    # Get ligand heavy atoms
    heavyatoms = rosetta_utils.get_ligand_heavyatoms(pose)

    # Identify design layers
    design_pos = list(args.design_pos) if args.design_pos else []
    layer_cuts = [args.layer_design_inner, args.layer_design_outer,
                  args.layer_repack_inner, args.layer_repack_outer]
    design_residues, repack_residues, do_not_touch = identify_design_layers(
        pose, ligand_seqpos, keep_pos, design_pos, heavyatoms,
        detect_pocket=not args.global_seq_redesign, layer_cuts=layer_cuts
    )

    # Evaluate and setup adaptive coordinate constraints if enabled
    coord_cst_weights = None
    if args.adaptive_coord_cst and args.cstfile and catres:
        print_header("Adaptive Coordinate Constraints", "-")

        # Add constraints to pose for evaluation
        if cst_manager is not None:
            cst_manager.add_constraints(pose)

        # Evaluate constraint scores for catalytic residues
        cat_scores = rosetta_utils.evaluate_catalytic_constraint_scores(
            pose, sfx, catres, verbose=True
        )

        # Identify residues needing freedom
        residues_to_reduce, far_off_residues = rosetta_utils.identify_residues_needing_freedom(
            pose, cat_scores,
            threshold=args.cst_deviation_threshold,
            neighbor_window=args.coord_cst_neighbor_window,
            verbose=True
        )

        # Create per-residue weights if any residues need freedom
        if residues_to_reduce:
            coord_cst_weights = rosetta_utils.create_per_residue_coord_cst_weights(
                pose, residues_to_reduce,
                normal_weight=1.0,
                reduced_weight=args.reduced_coord_cst_weight,
                verbose=True
            )

        # Remove constraints after evaluation (will be re-added during relax)
        if cst_manager is not None:
            cst_manager.remove_constraints(pose)

    elif args.adaptive_coord_cst and not args.cstfile:
        print("  [WARNING] --adaptive_coord_cst requires --cstfile to evaluate constraint satisfaction")

    # Detect backbone/CB clashes with ligand and free those residues + neighbors
    print_header("Backbone/CB Clash Detection", "-")
    bb_clashes, bb_residues_to_free = rosetta_utils.find_backbone_clashes_with_ligand(
        pose, ligand_seqpos, clash_dist=2.5, neighbor_window=3
    )
    if bb_clashes:
        print(f"  [BB/CB Clash] Found {len(bb_clashes)} residues with backbone/CB clashing with ligand:")
        print(f"    Clashing: {'+'.join(str(r) for r in bb_clashes)}")
        print(f"    Freeing (incl. neighbors): {'+'.join(str(r) for r in bb_residues_to_free)}")

        # Add to coord_cst_weights or create new if not already set
        if coord_cst_weights is None:
            coord_cst_weights = rosetta_utils.create_per_residue_coord_cst_weights(
                pose, bb_residues_to_free,
                normal_weight=DEFAULT_COORD_CST_WEIGHT,
                reduced_weight=args.reduced_coord_cst_weight,
                verbose=True
            )
        else:
            # Merge with existing weights - use reduced weight for clashing residues
            for resno in bb_residues_to_free:
                if resno in coord_cst_weights:
                    coord_cst_weights[resno] = min(coord_cst_weights[resno], args.reduced_coord_cst_weight)
                else:
                    coord_cst_weights[resno] = args.reduced_coord_cst_weight
            print(f"  [BB/CB Clash] Updated coordinate constraint weights for {len(bb_residues_to_free)} residues")
    else:
        print(f"  [BB/CB Clash] No backbone/CB clashes detected with ligand")

    # Pre-relax structure
    relaxed_pose = prerelax_structure(
        pose, sfx, cst_manager, keep_pos, ligand_seqpos,
        use_cartesian=not args.no_cartesian_relax,
        ligand_rigidity=args.ligand_rigidity,
        cart_bonded_weight=args.cart_bonded_weight,
        coord_cst_weights=coord_cst_weights
    )

    # Setup position biasing
    bias_dict = {}
    if args.bias_atoms:
        bias_dict = compute_bias_positions(
            relaxed_pose, ligand_seqpos, keep_pos, design_pos,
            args.bias_atoms, args.bias_AAs, args.position_bias
        )

    # Load protocol
    if args.quick_test:
        # Minimal protocol for fast testing - single MPNN + repack + min
        protocol = """
mpnn 0.1 2
repack
min 0.01
keep_best 1
"""
        print(f"\n  [QUICK TEST] Using minimal protocol for fast debugging")
    else:
        protocol = args.protocol if args.protocol else DEFAULT_PROTOCOL

    # Setup H-bond keeper task operation (if enabled)
    ligand_hbond_keeper = None
    if not args.disable_hbond_keeper:
        ligand_hbond_keeper = SelectHBondsToResidue(name="keep_hbonds_to_ligand_and_catres")
        ligand_hbond_keeper.target([ligand_seqpos] + list(catres))
        ligand_hbond_keeper.allow_updating(True)
        ligand_hbond_keeper.accept_probability(args.hbond_accept_prob)
        print(f"  H-bond keeper: Enabled (accept_prob={args.hbond_accept_prob})")
    else:
        print(f"  H-bond keeper: Disabled")

    # Print keep_best scoring mode
    print(f"  Keep best mode: {args.keep_best_mode} (cst_threshold={args.cst_comparable_threshold})")

    # Setup MPNN configuration if needed
    mpnn_config = None
    if args.second_shell_mpnn_seqs:
        mpnn_config = MPNNConfig.from_args(args)
        print(f"  MPNN runner: {mpnn_config.runner}")
        print(f"  MPNN model: {mpnn_config.model_type}")
        print(f"  MPNN SC context: {mpnn_config.use_sc_context} | Pack SC: {mpnn_config.pack_side_chains}")

    # Main design loop
    print_header("Running Design Iterations")

    total_saved = 0
    for n_iter in range(args.nstruct):
        existing = glob.glob(f"{pdbname}{suffix}_{n_iter}_*.pdb")
        if existing:
            print(f"  [SKIP] Iteration {n_iter}: outputs already exist")
            continue

        iteration_start = time.time()
        print(f"\n  [ITER {n_iter + 1}/{args.nstruct}]")

        pose_iter = relaxed_pose.clone()

        if cst_manager is not None:
            cst_manager.remove_constraints(pose_iter)
            cst_manager.add_constraints(pose_iter)

        fmd = design_protocol.FastMPNNdesign(
            model_type="ligand_mpnn",
            enhance_model=DEFAULT_ENHANCE_MODEL,
            params=params_files,
            scorefxn=sfx,
            script_file=protocol,
            design_positions=design_residues,
            repack_positions=repack_residues,
            do_not_repack_positions=do_not_touch,
            cst_io=cst_manager.cst_io() if cst_manager else None,
            omit_AA="CM",
            ligand_rigidity=args.ligand_rigidity,
            keep_best_mode=args.keep_best_mode,
            cst_comparable_threshold=args.cst_comparable_threshold,
            debug=args.debug,
            debug_output_dir=debug_structures_dir,
            cache_mpnn_models=not args.no_mpnn_cache,
            use_mpnn_server=args.mpnn_server,
            mpnn_server_host=args.mpnn_server_host,
            mpnn_server_port=args.mpnn_server_port
        )

        # Configure 2nd shell MPNN (used by 2nd_shell_mpnn protocol command)
        fmd.set_2nd_shell_config(ligand_seqpos, heavyatoms, keep_pos)

        # Configure covalent connection restoration from --ref_pdb
        # Only if ref_pose is provided (from --ref_pdb argument)
        # This ensures covalent bonds are only restored from trusted reference structures
        if ref_pose is not None and ref_catres is not None:
            catres_seqpos = list(ref_catres.keys())
            fmd.set_covalent_reference(ref_pose, catres_seqpos)

        # Set input PDB path for HIS tautomer preservation
        # This detects HIS tautomers (HD1 vs HE2) from the input PDB and re-applies them
        # after MPNN design to prevent PyRosetta from guessing different tautomers
        fmd.set_input_pdb_path(args.pdb)

        # Enable pipeline tracker for detailed debugging output when --debug is set
        if args.debug:
            debug_output_dir = os.path.join(os.path.dirname(args.pdb), f"debug_{pdbname}_{n_iter}")
            os.makedirs(debug_output_dir, exist_ok=True)
            tracker = pipeline_tracker.PipelineTracker(
                output_dir=debug_output_dir,
                verbose=True
            )
            fmd.set_tracker(tracker)
            fmd.set_debug_output_dir(debug_output_dir)
            print(f"  [DEBUG] Pipeline tracker enabled, output dir: {debug_output_dir}")

        if bias_dict:
            fmd.set_mpnn_bias_per_residue(bias_dict)

        if ligand_hbond_keeper is not None:
            fmd.add_task_operation(ligand_hbond_keeper)

        saved = run_design_iteration(
            pose_iter, fmd, cst_manager, scoring_module, pdbname, suffix, n_iter,
            catres, apply_filter=args.filter, run_mpnn=args.second_shell_mpnn_seqs, mpnn_config=mpnn_config,
            ligand_seqpos=ligand_seqpos, heavyatoms=heavyatoms, keep_pos=keep_pos,
            sfx=sfx, scorefilename=scorefilename, input_pose=pose,
            iteration_start_time=iteration_start,
            ref_pose=ref_pose, ref_catres=ref_catres
        )
        total_saved += saved if saved else 0

        iteration_time = time.time() - iteration_start
        print(f"  [ITER {n_iter + 1}] Completed in {format_duration(iteration_time)}")

    # Final summary
    pipeline_time = time.time() - pipeline_start_time
    print_header("Design Complete")
    print(f"  Total designs saved: {total_saved}")
    print(f"  Scores written to: {scorefilename}")
    if args.second_shell_mpnn_seqs:
        print(f"  Secondary sequences in: seqs/")
    print(f"  Total pipeline time: {format_duration(pipeline_time)}")


if __name__ == "__main__":
    main()
