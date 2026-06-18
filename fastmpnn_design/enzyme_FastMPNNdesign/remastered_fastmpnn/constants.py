"""
Constants and default values for Remastered FastMPNNdesign.

This module centralizes all configurable defaults, element classifications,
distance cutoffs, and atom definitions used throughout the pipeline.
"""

from typing import FrozenSet, Dict, Set

# =============================================================================
# Element Classifications
# =============================================================================

METALS: FrozenSet[str] = frozenset({
    'ZN', 'FE', 'MG', 'CA', 'MN', 'CO', 'NI', 'CU', 'MO', 'W'
})

HETEROATOMS: FrozenSet[str] = frozenset({'N', 'O', 'S'})

# Hydrogen atom names (for filtering)
HYDROGEN_NAMES: FrozenSet[str] = frozenset({
    'H', '1H', '2H', '3H', 'HA', 'HA2', 'HA3',
    'HB', 'HB1', 'HB2', 'HB3',
    'HG', 'HG1', 'HG2', 'HG3', 'HG11', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23',
    'HD1', 'HD2', 'HD11', 'HD12', 'HD13', 'HD21', 'HD22', 'HD23',
    'HE', 'HE1', 'HE2', 'HE3', 'HE21', 'HE22',
    'HZ', 'HZ1', 'HZ2', 'HZ3',
    'HH', 'HH11', 'HH12', 'HH21', 'HH22',
    'HN', 'HXT'
})

# =============================================================================
# Backbone Atom Definitions
# =============================================================================

# Core backbone atoms (always present)
BACKBONE_ATOMS: FrozenSet[str] = frozenset({'N', 'CA', 'C', 'O'})

# Extended backbone atoms (including hydrogens)
BACKBONE_ATOMS_WITH_H: FrozenSet[str] = frozenset({'N', 'CA', 'C', 'O', 'H', 'HA', 'HA2', 'HA3', 'OXT'})

# Backbone amide atoms for H-bond detection
BACKBONE_AMIDE_DONOR: FrozenSet[str] = frozenset({'N', 'H'})  # H-bond donors
BACKBONE_AMIDE_ACCEPTOR: FrozenSet[str] = frozenset({'O', 'OXT'})  # H-bond acceptors

# =============================================================================
# Standard Amino Acids
# =============================================================================

STANDARD_AA: FrozenSet[str] = frozenset({
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
})

# Non-standard amino acids (treat as protein)
NONSTANDARD_AA: FrozenSet[str] = frozenset({
    'MSE', 'SEC', 'PYL', 'HYP', 'SEP', 'TPO', 'PTR', 'CSO', 'CSS',
    'CME', 'MLY', 'ALY', 'M3L', 'OCS', 'CSD', 'CAS', 'CGU',
    'HIS_D', 'HID', 'HIE', 'HIP', 'HSE', 'HSD', 'HSP'  # Histidine variants
})

# All protein residue names
PROTEIN_RESIDUES: FrozenSet[str] = STANDARD_AA | NONSTANDARD_AA

# Common solvents to ignore
SOLVENTS: FrozenSet[str] = frozenset({
    'HOH', 'WAT', 'SOL', 'DOD', 'D2O', 'TIP', 'TIP3', 'TIP4', 'SPC'
})

# Common buffers/ions to ignore
BUFFERS: FrozenSet[str] = frozenset({
    'SO4', 'PO4', 'GOL', 'EDO', 'PEG', 'MPD', 'ACT', 'ACY', 'FMT', 'TRS',
    'CL', 'NA', 'K', 'IOD', 'BR', 'BME', 'DMS'
})

# 3-letter to 1-letter code mapping
AA_3TO1: Dict[str, str] = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M', 'SEC': 'U', 'PYL': 'O'
}

AA_1TO3: Dict[str, str] = {v: k for k, v in AA_3TO1.items() if k in STANDARD_AA}

# =============================================================================
# Distance Cutoffs (Angstroms)
# =============================================================================

# Metal coordination
METAL_COORDINATION_CUTOFF: float = 2.6

# Hydrogen bonds
HBOND_CUTOFF: float = 3.5

# Covalent/PTM detection
COVALENT_BOND_CUTOFF: float = 1.8

# Acid-base modification (polar atom to hydrogen)
ACID_BASE_CUTOFF: float = 2.5

# Salt bridge
SALT_BRIDGE_CUTOFF: float = 4.0

# Pi-stacking
PI_STACK_CUTOFF: float = 5.5
PI_STACK_ANGLE_CUTOFF: float = 30.0  # Max angle deviation from parallel (degrees)

# Hydrophobic contact
HYDROPHOBIC_CUTOFF: float = 4.5

# General contact cutoffs
PRIMARY_CONTACT_CUTOFF: float = 3.6
SECONDARY_CONTACT_CUTOFF: float = 4.2

# Ligand alignment RMSD warning threshold
LIGAND_RMSD_WARNING: float = 0.1  # Warn if RMSD > 0.1 A

# =============================================================================
# Residue-Specific Atom Classifications
# =============================================================================

# H-bond donors by residue (sidechain atoms that donate H-bonds)
SIDECHAIN_HBOND_DONORS: Dict[str, FrozenSet[str]] = {
    'ARG': frozenset({'NE', 'NH1', 'NH2'}),
    'ASN': frozenset({'ND2'}),
    'GLN': frozenset({'NE2'}),
    'HIS': frozenset({'ND1', 'NE2'}),
    'LYS': frozenset({'NZ'}),
    'SER': frozenset({'OG'}),
    'THR': frozenset({'OG1'}),
    'TRP': frozenset({'NE1'}),
    'TYR': frozenset({'OH'}),
    'CYS': frozenset({'SG'}),
}

# H-bond acceptors by residue (sidechain atoms that accept H-bonds)
SIDECHAIN_HBOND_ACCEPTORS: Dict[str, FrozenSet[str]] = {
    'ASN': frozenset({'OD1'}),
    'ASP': frozenset({'OD1', 'OD2'}),
    'GLN': frozenset({'OE1'}),
    'GLU': frozenset({'OE1', 'OE2'}),
    'HIS': frozenset({'ND1', 'NE2'}),
    'SER': frozenset({'OG'}),
    'THR': frozenset({'OG1'}),
    'TYR': frozenset({'OH'}),
    'MET': frozenset({'SD'}),
    'CYS': frozenset({'SG'}),
}

# Metal-coordinating atoms by residue
METAL_COORDINATING_ATOMS: Dict[str, FrozenSet[str]] = {
    'HIS': frozenset({'ND1', 'NE2'}),
    'CYS': frozenset({'SG'}),
    'ASP': frozenset({'OD1', 'OD2'}),
    'GLU': frozenset({'OE1', 'OE2'}),
    'MET': frozenset({'SD'}),
    'SER': frozenset({'OG'}),
    'THR': frozenset({'OG1'}),
}

# Charged residue classifications
POSITIVELY_CHARGED_AA: FrozenSet[str] = frozenset({'LYS', 'ARG', 'HIS'})
NEGATIVELY_CHARGED_AA: FrozenSet[str] = frozenset({'ASP', 'GLU'})

# Charged sidechain atoms
POSITIVE_CHARGE_ATOMS: Dict[str, FrozenSet[str]] = {
    'LYS': frozenset({'NZ'}),
    'ARG': frozenset({'NE', 'NH1', 'NH2'}),
    'HIS': frozenset({'ND1', 'NE2'}),
}

NEGATIVE_CHARGE_ATOMS: Dict[str, FrozenSet[str]] = {
    'ASP': frozenset({'OD1', 'OD2'}),
    'GLU': frozenset({'OE1', 'OE2'}),
}

# Aromatic residues
AROMATIC_AA: FrozenSet[str] = frozenset({'PHE', 'TYR', 'TRP', 'HIS'})

# Aromatic ring atoms for pi-stacking calculations
AROMATIC_RING_ATOMS: Dict[str, FrozenSet[str]] = {
    'PHE': frozenset({'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'}),
    'TYR': frozenset({'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'}),
    'TRP': frozenset({'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'}),
    'HIS': frozenset({'CG', 'ND1', 'CD2', 'CE1', 'NE2'}),
}

# Hydrophobic residues
HYDROPHOBIC_AA: FrozenSet[str] = frozenset({
    'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO'
})

# =============================================================================
# Interaction Priority Levels (higher = more important)
# =============================================================================

PRIORITY_METAL_COORDINATION: int = 100
PRIORITY_COVALENT_BOND: int = 95
PRIORITY_SALT_BRIDGE: int = 90
PRIORITY_HBOND_BACKBONE: int = 85
PRIORITY_HBOND_SIDECHAIN: int = 80
PRIORITY_CATRES_HBOND: int = 70
PRIORITY_PI_STACK: int = 60
PRIORITY_HYDROPHOBIC: int = 40

# =============================================================================
# Logging Configuration
# =============================================================================

LOG_FORMAT_VERBOSE: str = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
LOG_FORMAT_SIMPLE: str = "%(levelname)-8s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# =============================================================================
# Output Defaults
# =============================================================================

DEFAULT_OUTPUT_DIR: str = "./step1_output"
DEFAULT_OUTPUT_PDB: str = "step1_output.pdb"
DEFAULT_OUTPUT_JSON: str = "residue_registry.json"

# =============================================================================
# Important Component Decision Rules
# =============================================================================

# When no interactions are detected, default to sidechain
DEFAULT_IMPORTANT_COMPONENT: str = "sidechain"

# Minimum number of interactions to consider component "important"
MIN_INTERACTIONS_THRESHOLD: int = 1
