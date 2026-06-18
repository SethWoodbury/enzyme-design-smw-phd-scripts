"""
Constants for enhanced_fastmpnndesign package.

Centralizes all default values, paths, and configuration constants.
Values are extracted from the original fastmpnn_ZnEsterase_SETH_LINKED.py script.
"""

from typing import FrozenSet, Dict
from pathlib import Path


# =============================================================================
# MPNN Defaults
# =============================================================================

DEFAULT_MPNN_RUNNER: str = "/net/software/lab/fused_mpnn/seth_temp/run.py"
DEFAULT_MODEL_TYPE: str = "ligand_mpnn"
DEFAULT_ENHANCE_MODEL: str = "plddt_3_20240930-f9c9ea0f"
DEFAULT_TEMPERATURE: float = 0.3  # Protocol default, lower temps (0.1, 0.2) used in later stages
DEFAULT_BATCHES: int = 10
DEFAULT_BATCH_SIZE: int = 1
DEFAULT_OMIT_AA: str = "CM"  # Omit Cysteine and Methionine
DEFAULT_SC_DENOISING_STEPS: int = 3
DEFAULT_APPTAINER_IMAGE: str = "/software/containers/universal.sif"

# 2nd layer MPNN temperatures (from original lines 404)
SECOND_LAYER_TEMPERATURES: tuple = (0.1, 0.2)
SECOND_LAYER_BATCHES: int = 1
SECOND_LAYER_BATCH_SIZE: int = 2


# =============================================================================
# PyRosetta / Rosetta Defaults
# =============================================================================

DEFAULT_SCOREFUNCTION: str = "beta_jan25"  # Updated from beta_nov16 to newest
DEFAULT_DALPHABALL: str = "/net/software/lab/scripts/enzyme_design/DAlphaBall.gcc"
DEFAULT_PARAMS_PATH: str = "/home/ikalvet/projects/Organophosphate/Esterase/theozyme/ZRE/ZRE.params"

# Beta scorefunctions that require special initialization
BETA_SCOREFUNCTIONS: FrozenSet[str] = frozenset({
    'beta_jan25', 'beta_nov16', 'beta_july15', 'beta_nov15', 'beta'
})


# =============================================================================
# Layer Detection Defaults (from original lines 52-53, 210)
# =============================================================================

# 2nd layer fixed position detection
DEFAULT_LAYER_DIST_BB: float = 6.0
DEFAULT_LAYER_DIST_SC: float = 5.0

# Layer cuts for design pocket detection (from original lines 195-196)
DEFAULT_LAYER_CUTS: tuple = (7.0, 9.0, 11.0, 13.0)

# Close sidechain cutoff (from original line 212)
CLOSE_SC_CUTOFF: float = 8.0


# =============================================================================
# Pre-Relaxation Defaults (from original lines 241-243)
# =============================================================================

CART_BONDED_WEIGHT: float = 0.5
PRO_CLOSE_WEIGHT: float = 0.0
USE_CARTESIAN_PRERELAX: bool = True


# =============================================================================
# Constraint Weights (from original lines 152-154)
# =============================================================================

CONSTRAINT_WEIGHT_ATOM_PAIR: float = 1.0
CONSTRAINT_WEIGHT_ANGLE: float = 1.0
CONSTRAINT_WEIGHT_DIHEDRAL: float = 1.0


# =============================================================================
# MPNN Bias Defaults (from original lines 92-94, 270)
# =============================================================================

DEFAULT_BIAS_VALUE: float = -1.0
DEFAULT_BIAS_AAS: str = "KREDYQWSTH"
DEFAULT_BIAS_CUTS: tuple = (5.0, 7.0, 9.0, 11.0)


# =============================================================================
# HBond Keeper Defaults (from original line 359)
# =============================================================================

DEFAULT_HBOND_ACCEPT_PROBABILITY: float = 0.75


# =============================================================================
# Default Protocol (from original lines 310-341)
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
"""


# =============================================================================
# Amino Acid Classifications
# =============================================================================

STANDARD_AA: FrozenSet[str] = frozenset({
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
})

NONSTANDARD_AA: FrozenSet[str] = frozenset({
    'MSE', 'SEC', 'PYL', 'HYP', 'SEP', 'TPO', 'PTR', 'CSO', 'CSS',
    'CME', 'MLY', 'ALY', 'M3L', 'OCS', 'CSD', 'CAS', 'CGU'
})

# Amino acid code mappings
AA_3TO1: Dict[str, str] = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M', 'SEC': 'U', 'PYL': 'O'
}

AA_1TO3: Dict[str, str] = {v: k for k, v in AA_3TO1.items() if k in STANDARD_AA}

# Polar amino acids (for bias)
POLAR_AA: FrozenSet[str] = frozenset({'K', 'R', 'E', 'D', 'Y', 'Q', 'W', 'S', 'T', 'H'})


# =============================================================================
# Element Classifications
# =============================================================================

METALS: FrozenSet[str] = frozenset({
    'ZN', 'FE', 'MG', 'CA', 'MN', 'CO', 'NI', 'CU', 'MO', 'W'
})

BACKBONE_ATOMS: FrozenSet[str] = frozenset({'N', 'CA', 'C', 'O', 'H', 'HA'})


# =============================================================================
# Path Constants
# =============================================================================

# FastMPNNDesign module path (used for importing)
FASTMPNN_DESIGN_PATH: str = "/home/woodbuse/special_scripts/fastmpnn_design/FastMPNNDesign"

# Design utilities path
DESIGN_UTILS_PATH: str = "/software/scripts/enzyme_design/utils"
