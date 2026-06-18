#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constants and default values for Enhanced FastMPNN Design.

This module centralizes all configurable defaults and paths.
"""

# =============================================================================
# MPNN Defaults
# =============================================================================

DEFAULT_MPNN_RUNNER: str = "/net/software/lab/fused_mpnn/seth_temp/run.py"
DEFAULT_MODEL_TYPE: str = "ligand_mpnn"
DEFAULT_ENHANCE_MODEL: str = "plddt_3_20240930-f9c9ea0f"
DEFAULT_OMIT_AA: str = "CM"  # Omit Cysteine and Methionine
DEFAULT_APPTAINER_IMAGE: str = "/software/containers/universal.sif"

# MPNN Side-Chain and Repacking Settings
DEFAULT_LIGAND_MPNN_USE_SC_CONTEXT: int = 1  # Use side-chain context for ligand MPNN
DEFAULT_REPACK_EVERYTHING: int = 0           # Don't repack fixed residues (e.g., catalytic)
DEFAULT_PACK_SIDE_CHAINS: int = 1            # Pack side chains for designed residues
DEFAULT_SC_DENOISING_STEPS: int = 3          # Side-chain denoising steps

# MPNN Server Settings
DEFAULT_MPNN_SERVER_PORT: int = 5000
DEFAULT_MPNN_SERVER_HOST: str = "localhost"

# =============================================================================
# PyRosetta Defaults
# =============================================================================

DEFAULT_DALPHABALL: str = "/net/software/lab/scripts/enzyme_design/DAlphaBall.gcc"
DEFAULT_PARAMS: list = ["/home/ikalvet/projects/Organophosphate/Esterase/theozyme/ZRE/ZRE.params"]

# =============================================================================
# Design Defaults
# =============================================================================

DEFAULT_BIAS_AAS: str = "KREDYQWSTH"
DEFAULT_POSITION_BIAS: float = -1.0
DEFAULT_HBOND_ACCEPT_PROB: float = 0.75
DEFAULT_HBOND_KEEPER_ENABLED: bool = True

# =============================================================================
# Keep Best Scoring Modes
# =============================================================================

# Available modes for keep_best selection:
#   "cst_priority"  - Prioritize constraint score, fall back to total if comparable (DEFAULT)
#   "total_score"   - Use total Rosetta score only (original behavior)
#   "cst_only"      - Use constraint score only
KEEP_BEST_MODES = ["cst_priority", "total_score", "cst_only"]
DEFAULT_KEEP_BEST_MODE: str = "cst_priority"

# Threshold for "comparable" constraint scores in cst_priority mode
# If all poses have cst_score < this, or max-min difference < this, use total_score
DEFAULT_CST_COMPARABLE_THRESHOLD: float = 2.0  # REU

# =============================================================================
# Layer Selection Distances (Angstroms)
# =============================================================================

DEFAULT_LAYER_CUTS: list = [6.0, 8.0, 10.0, 12.0]
DEFAULT_POCKET_CUTS: list = [7.0, 9.0, 11.0, 13.0]
DEFAULT_BIAS_CUTS: list = [5.0, 7.0, 9.0, 11.0]

# =============================================================================
# 2nd Layer MPNN Defaults
# =============================================================================

DEFAULT_2ND_LAYER_TEMPS: list = [0.1, 0.2]
DEFAULT_2ND_LAYER_BATCH_SIZE: int = 2

# =============================================================================
# Scorefunction Configuration
# =============================================================================

# Available scorefunction presets
# These correspond to Rosetta scorefunction weights files
SCOREFUNCTION_PRESETS = {
    "ref2015": "ref2015",           # REF2015 (requires -beta_nov16 flag for full effect)
    "ref2015_cart": "ref2015_cart", # REF2015 optimized for Cartesian minimization
    "beta_nov16": "beta_nov16",     # Explicit beta_nov16 weights
    "talaris2014": "talaris2014",   # Older Talaris scorefunction
    "score12": "score12",           # Legacy score12
}
DEFAULT_SCOREFUNCTION: str = "ref2015"  # Uses REF2015 with beta_nov16 weights

# Beta scorefunctions that require special -corrections: flags during pyrosetta.init()
# These are newer scorefunctions with improved parameters
# Format: {scorefunction_name: init_flag}
BETA_SCOREFUNCTIONS = {
    "beta_jan25": "-corrections:beta_jan25",
    "beta_nov16": "-beta_nov16",  # Can also use -corrections:beta_nov16
    "beta_july15": "-corrections:beta_july15",
}

# Default initialization flag (used when scorefunction is ref2015 or not specified)
DEFAULT_INIT_CORRECTION: str = "-beta_nov16"

# Beta scorefunctions that require running inside the PyRosetta container
# (newer scorefunctions not available in standard PyRosetta installations)
BETA_SCOREFUNCTIONS_REQUIRE_CONTAINER: set = {"beta_jan25"}

# PyRosetta container for newer scorefunctions
DEFAULT_PYROSETTA_IMAGE: str = "/software/containers/pyrosetta.sif"

# Environment variable set when running inside a container (for detection)
CONTAINER_ENV_VAR: str = "FASTMPNN_IN_CONTAINER"

# Constraint weights (for enzyme design)
DEFAULT_CONSTRAINT_WEIGHTS = {
    "atom_pair_constraint": 1.0,
    "angle_constraint": 1.0,
    "dihedral_constraint": 1.0,
}

# Cartesian-specific weights
DEFAULT_CART_BONDED_WEIGHT: float = 0.75
DEFAULT_PRO_CLOSE_WEIGHT_CART: float = 0.0  # Must be 0 for Cartesian

# Coordinate constraint weight for relaxation
DEFAULT_COORD_CST_WEIGHT: float = 0.9

# =============================================================================
# Adaptive Coordinate Constraints for Catalytic Residues
# =============================================================================

# When enabled, catalytic residues with poor constraint satisfaction (and their
# neighbors) get reduced coordinate constraints, giving them backbone freedom
# to satisfy enzyme constraints.

# Constraint score threshold - residues with scores above this are "far off"
# Higher score = worse constraint satisfaction
DEFAULT_CST_DEVIATION_THRESHOLD: float = 5.0  # REU (Rosetta Energy Units)

# Number of neighboring residues (N-terminal and C-terminal) to also free
DEFAULT_COORD_CST_NEIGHBOR_WINDOW: int = 3  # ±3 residues

# Reduced coordinate constraint weight for "far off" residues and neighbors
# 0.0 = completely free, 1.0 = fully constrained
DEFAULT_REDUCED_COORD_CST_WEIGHT: float = 0.2

# =============================================================================
# Ligand Rigidity Options
# =============================================================================

# Ligand rigidity modes:
#   "fixed"       - Ligands completely rigid (no internal changes, no rigid-body movement) [DEFAULT]
#   "rigid_body"  - Ligands internally rigid, but can move independently as rigid bodies
#   "flexible"    - Full ligand flexibility (internal torsions + rigid-body movement)
LIGAND_RIGIDITY_MODES = ["fixed", "rigid_body", "flexible"]
DEFAULT_LIGAND_RIGIDITY: str = "fixed"

# =============================================================================
# Default Scoring Filters
# =============================================================================

DEFAULT_SCORE_FILTERS: dict = {
    "L_SASA": [0.25, "<="],           # Relative ligand SASA
    "corrected_ddg": [-25.0, "<="],   # Interaction energy
    "sc": [0.55, ">="],               # Shape complementarity
    "nlr_totrms": [1.0, "<="],        # No-ligand-repack RMSD
}

# Polar elements for H-bond auto-detection
DEFAULT_POLAR_ELEMENTS: set = {"O", "N", "S"}

# Atoms to exclude from automatic polar detection (metals, etc.)
DEFAULT_EXCLUDE_ATOMS: set = {"ZN", "ZN1", "ZN2", "MG", "CA", "FE", "MN", "CU", "NI", "CO"}

# =============================================================================
# Default Design Protocol
# =============================================================================

DEFAULT_PROTOCOL: str = """
scale:coordinate_constraint 1.0
scale:fa_rep 0.150
mpnn 0.3 10
repack
scale:fa_rep 0.200
min 0.01
keep_best 5
task_operation keep_hbonds_to_ligand_and_catres
scale:coordinate_constraint 0.5
scale:fa_rep 0.365
mpnn 0.2 2
repack
keep_best 5
scale:fa_rep 0.480
min 0.01
task_operation keep_hbonds_to_ligand_and_catres
scale:coordinate_constraint 0.0
scale:fa_rep 0.659
mpnn 0.1 2
repack
keep_best 5
scale:fa_rep 0.750
min 0.01
task_operation keep_hbonds_to_ligand_and_catres
scale:coordinate_constraint 0.0
scale:fa_rep 1
mpnn 0.1 2
repack
min 0.00001
keep_best 8
2nd_shell_mpnn 0.1 2
2nd_shell_mpnn 0.2 2
"""

# =============================================================================
# Reference PDB Constraint Derivation Defaults
# =============================================================================

# Atom preference for distance constraints (prefer O, N, S over C)
CONSTRAINT_ATOM_PREFERENCE: list = ["O", "N", "S", "C"]

# Default tolerances for derived constraints (Rosetta .cst file format)
DEFAULT_DERIVED_DISTANCE_TOLERANCE: float = 0.5   # Angstroms
DEFAULT_DERIVED_ANGLE_TOLERANCE: float = 15.0     # Degrees
DEFAULT_DERIVED_DIHEDRAL_TOLERANCE: float = 20.0  # Degrees

# Force constants for derived constraints
DEFAULT_DERIVED_DISTANCE_FORCE: float = 100.0
DEFAULT_DERIVED_ANGLE_FORCE: float = 50.0
DEFAULT_DERIVED_DIHEDRAL_FORCE: float = 50.0

# Ligand RMSD threshold for alignment warning
DEFAULT_LIGAND_RMSD_WARNING_THRESHOLD: float = 0.5  # Angstroms
