"""Shared constants for the upgraded_fastMPNNdesign pipeline.

This module provides default values for:
- MPNN execution settings
- Rosetta scorefunction selection
- Sphere/layer distance cutoffs
- Geometry tolerances and thresholds
- Interaction detection parameters
"""
import logging
from typing import List, Set

LOGGER = logging.getLogger(__name__)

# =============================================================================
# MPNN Defaults
# =============================================================================
DEFAULT_MPNN_RUNNER: str = "/net/software/lab/fused_mpnn/seth_temp/run.py"
DEFAULT_MODEL_TYPE: str = "ligand_mpnn"
DEFAULT_ENHANCE_MODEL: str = "plddt_3_20240930-f9c9ea0f"
DEFAULT_OMIT_AA: str = "CM"  # Omit Cysteine and Methionine

# Container paths
# universal.sif contains both PyRosetta (2026.03) and ProteinMPNN
# It has beta_jan25 scorefunction, multi-threading, and serialization support
# Using a single container for everything simplifies execution
DEFAULT_APPTAINER_IMAGE: str = "/net/software/containers/universal.sif"
DEFAULT_PYROSETTA_IMAGE: str = "/net/software/containers/universal.sif"  # Now uses universal.sif by default
DEFAULT_PYROSETTA_FALLBACK_PATHS: List[str] = [
    "/software/pyrosetta/latest",
    "/software/pyrosetta/latest/setup",
    "/software/pyrosetta/latest/setup/pyrosetta",
]
DEFAULT_PYROSETTA_DIR: str = "/software/pyrosetta/latest"
DEFAULT_PYROSETTA_REQUIRED_VERSION: str = ""
DEFAULT_PYROSETTA_MIN_VERSION: str = ""

# MPNN Side-Chain and Repacking Settings
DEFAULT_LIGAND_MPNN_USE_SC_CONTEXT: int = 1  # Use side-chain context for ligand MPNN
DEFAULT_REPACK_EVERYTHING: int = 0           # Don't repack fixed residues (e.g., catalytic)
DEFAULT_PACK_SIDE_CHAINS: int = 1            # Pack side chains for designed residues
DEFAULT_SC_DENOISING_STEPS: int = 3          # Side-chain denoising steps

# MPNN Diversity Settings (higher batches + lower batch_size = more diversity)
DEFAULT_MPNN_BATCH_SIZE: int = 1
DEFAULT_MPNN_NUMBER_OF_BATCHES: int = 8
DEFAULT_MPNN_TEMPERATURE: float = 0.1
DEFAULT_MPNN_NUM_DESIGNS_AFTER_FIRST: int = 1

# MPNN Server Settings
DEFAULT_MPNN_SERVER_HOST: str = "localhost"
DEFAULT_MPNN_SERVER_PORT: int = 5000
DEFAULT_MPNN_SERVER_TIMEOUT: int = 600  # Request timeout (seconds)
DEFAULT_MPNN_SERVER_STARTUP_TIMEOUT: int = 120  # Server startup timeout (seconds)

# =============================================================================
# Rosetta Scorefunctions
# =============================================================================
# For cartesian relax (validated in step02)
SCOREFUNCTION_CART: str = "ref2015_cart"

# For torsional relax/repack (faster, good for sampling)
SCOREFUNCTION_TORSIONAL: str = "beta_jan25"

# All available scorefunctions
SCOREFUNCTION_OPTIONS_CART: List[str] = [
    "ref2015_cart",
    "beta_nov16_cart",
]

SCOREFUNCTION_OPTIONS_TORSIONAL: List[str] = [
    "beta_jan25",
    "ref2015",
    "beta_nov16",
]

SCOREFUNCTION_OPTIONS: List[str] = SCOREFUNCTION_OPTIONS_CART + SCOREFUNCTION_OPTIONS_TORSIONAL

# =============================================================================
# Sphere/Layer Definitions
# =============================================================================
# Distance cutoffs for layer classification (Angstroms)
# Format: [core_cutoff, shell_cutoff, flex_cutoff]
DEFAULT_LAYER_CUTS: List[float] = [6.0, 8.0, 12.0]
#
# DESIGN_CORE   (0 - 6Å)   : Residues redesigned by MPNN
# DESIGN_SHELL  (6 - 8Å)   : Residues redesigned by MPNN (with CB orientation check)
# FLEX          (8 - 12Å)  : Sidechains can repack, sequence stays fixed
# FROZEN        (> 12Å)    : Completely fixed, nothing changes
#
# Design scope options:
#   design_core       : Redesign DESIGN_CORE only (0-6Å)
#   design_core_shell : Redesign DESIGN_CORE + DESIGN_SHELL (0-8Å)
#   design_shell_only : Redesign DESIGN_SHELL only, freeze core (6-8Å)
#   design_global     : Redesign everything non-fixed
#
# Repack scope options:
#   repack_core           : Repack 0-6Å only
#   repack_core_shell     : Repack 0-8Å
#   repack_core_shell_flex: Repack 0-12Å (default)
#   repack_global         : Repack everything non-fixed

# Residues protected from design by default
PROTECTED_RESIDUES: Set[str] = {"GLY", "PRO"}

# =============================================================================
# Geometry Thresholds (for convergence checking)
# =============================================================================
DEFAULT_BOND_LENGTH_TOLERANCE: float = 0.05   # Angstroms (unconstrained bonds)
DEFAULT_BOND_ANGLE_TOLERANCE: float = 10.0    # Degrees (unconstrained angles)

# Severity thresholds for geometry offenders
SEVERE_BOND_THRESHOLD: float = 0.2     # Angstroms
MODERATE_BOND_THRESHOLD: float = 0.1   # Angstroms
SEVERE_ANGLE_THRESHOLD: float = 15.0   # Degrees
MODERATE_ANGLE_THRESHOLD: float = 10.0 # Degrees

# Catres-specific tolerances
DEFAULT_CATRES_BOND_TOLERANCE: float = 0.05   # Angstroms
DEFAULT_CATRES_ANGLE_TOLERANCE: float = 10.0  # Degrees

# =============================================================================
# Coordinate Constraint Settings
# =============================================================================
DEFAULT_COORD_CST_WEIGHT: float = 750.0   # Very tight constraint
DEFAULT_COORD_CST_STDEV: float = 0.01     # Angstroms (harmonic std dev)
# Energy penalty: weight * 0.5 * (displacement / stdev)^2
# At 0.01A displacement: penalty = 375
# At 0.1A displacement: penalty = 37,500

# =============================================================================
# Cartesian Bonded Settings
# =============================================================================
DEFAULT_CART_BONDED_WEIGHT: float = 2.0   # Weight for cart_bonded score term
DEFAULT_CART_BONDED_MAX: float = 3.0      # Maximum weight during adaptive scaling
DEFAULT_CART_BONDED_SCALE_FACTOR: float = 1.5  # Multiplier per round if not converging

# =============================================================================
# FastRelax Settings
# =============================================================================
DEFAULT_FASTRELAX_REPEATS: int = 3
DEFAULT_FASTRELAX_RAMP_STAGES: int = 5

# =============================================================================
# Default protocol string (step03)
# =============================================================================
DEFAULT_PROTOCOL_STRING: str = (
    "mpnn:T0.2:N8 -> torsional_relax:R1S2 -> "
    "mpnn:T0.1:N16 -> torsional_relax:R2S3"
)

# Mobile region settings
DEFAULT_MOBILE_RADIUS: float = 10.0        # Angstroms around ligand/catres
DEFAULT_SEQUENCE_NEIGHBOR_BUFFER: int = 5  # +/- residues from catres

# =============================================================================
# Interaction Detection Thresholds (from step01/align_catres.py)
# =============================================================================
# H-bond detection
HBOND_DIST_MAX: float = 3.5              # Angstroms (heavy atom distance)
HBOND_DONOR_ANGLE_MIN: float = 120.0     # Degrees (D-H...A angle)
HBOND_ACCEPTOR_ANGLE_MIN: float = 100.0  # Degrees (H...A-B angle)
H_BOND_DIST_MAX: float = 1.4             # Angstroms (H to donor heavy atom)
HEAVY_BOND_DIST_MAX: float = 1.8         # Angstroms (heavy atom bond)

# Pi-stacking detection
PI_CENTROID_DIST_MIN: float = 3.3        # Angstroms
PI_CENTROID_DIST_MAX: float = 6.0        # Angstroms
PI_PARALLEL_ANGLE_MAX: float = 30.0      # Degrees (for parallel stacking)
PI_TSHAPE_ANGLE_MIN: float = 60.0        # Degrees (for T-shaped)
PI_TSHAPE_ANGLE_MAX: float = 90.0        # Degrees (for T-shaped)
PI_PERP_SEPARATION_MAX: float = 4.0      # Angstroms
PI_OFFSET_FACE_TO_FACE: float = 2.0      # Angstroms (lateral offset)
PI_OFFSET_DISPLACED_MAX: float = 5.0     # Angstroms
PI_TSHAPE_CONTACT_MAX: float = 4.5       # Angstroms

# Hydrophobic contacts
HYDROPHOBIC_DIST_MAX: float = 4.5        # Angstroms

# Charged/ionic interactions
CHARGED_DIST_MAX: float = 4.5            # Angstroms

# Metal coordination
METAL_COORD_DIST_MAX: float = 2.8        # Angstroms

# Covalent/PTM detection
COVALENT_DIST_MAX: float = 2.2           # Angstroms

# Acid-base catalysis
ACID_BASE_DIST_MAX: float = 1.5          # Angstroms

# =============================================================================
# Residue Classifications
# =============================================================================
# Standard amino acids (1-letter codes)
STANDARD_AA_1: str = "ACDEFGHIKLMNPQRSTVWY"

# Standard amino acids (3-letter codes)
STANDARD_AA_3: Set[str] = {
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"
}

# Aromatic residues (for pi-stacking)
AROMATIC_RESIDUES: Set[str] = {"PHE", "TYR", "TRP", "HIS"}

# Aromatic ring atoms by residue
AROMATIC_RING_ATOMS: dict = {
    "PHE": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
    "TYR": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
    "TRP": [
        ["CG", "CD1", "CD2", "NE1", "CE2"],  # 5-membered pyrrole
        ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"]  # 6-membered benzene
    ],
    "HIS": [["CG", "ND1", "CD2", "CE1", "NE2"]],
}

# Hydrophobic residues
HYDROPHOBIC_RESIDUES: Set[str] = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}

# Nonpolar sidechain residues
NONPOLAR_SIDECHAIN_RESIDUES: Set[str] = {"ALA", "VAL", "LEU", "ILE", "PHE", "PRO", "GLY"}

# Charged residues
CHARGED_RESIDUES: Set[str] = {"ASP", "GLU", "LYS", "ARG", "HIS"}

# Polar atoms by element
POLAR_ELEMENTS: Set[str] = {"N", "O", "S"}

# Metal atoms
METAL_ATOMS: Set[str] = {"ZN", "MG", "CA", "FE", "MN", "CU", "CO", "NI", "NA", "K"}

# Charged atoms by residue (sidechain only)
CHARGED_ATOMS_BY_RESIDUE: dict = {
    "ASP": ["OD1", "OD2"],
    "GLU": ["OE1", "OE2"],
    "LYS": ["NZ"],
    "ARG": ["NE", "NH1", "NH2"],
    "HIS": ["ND1", "NE2"],
}

# Acid-base catalytic atoms
ACID_BASE_RESIDUES: Set[str] = {"ASP", "GLU", "HIS", "LYS", "ARG", "CYS", "TYR", "SER"}

ACID_BASE_ATOMS_BY_RESIDUE: dict = {
    "ASP": ["OD1", "OD2"],
    "GLU": ["OE1", "OE2"],
    "HIS": ["ND1", "NE2"],
    "LYS": ["NZ"],
    "ARG": ["NE", "NH1", "NH2"],
    "CYS": ["SG"],
    "TYR": ["OH"],
    "SER": ["OG"],
}

# =============================================================================
# 3-letter to 1-letter amino acid mapping
# =============================================================================
AA_3_TO_1: dict = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    # HIS tautomers
    "HIS_D": "H", "HIP": "H", "HIE": "H", "HID": "H",
}

AA_1_TO_3: dict = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}

# =============================================================================
# Runtime Settings
# =============================================================================
DEFAULT_MAX_RUNTIME: int = 7200          # 2 hours
DEFAULT_NUM_FINAL_DESIGNS: int = 10

# Conservation settings
DEFAULT_CONSERVATION_PROBABILITY: float = 0.5  # Probability to conserve favorable interactions
DEFAULT_HBOND_ACCEPT_PROBABILITY: float = 0.5  # H-bond keeper probability
