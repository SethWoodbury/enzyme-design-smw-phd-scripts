"""
Constants for fastmpnndesign package.

Defines element sets, distance cutoffs, default paths, and other constants.
"""

from typing import FrozenSet, Dict

# Element classifications
METALS: FrozenSet[str] = frozenset({
    'ZN', 'FE', 'MG', 'CA', 'MN', 'CO', 'NI', 'CU', 'MO', 'W'
})

HETEROATOMS: FrozenSet[str] = frozenset({'N', 'O', 'S'})

HYDROGEN_NAMES: FrozenSet[str] = frozenset({
    'H', '1H', '2H', '3H', 'HA', 'HB', 'HB1', 'HB2', 'HB3',
    'HG', 'HG1', 'HG2', 'HG3', 'HD1', 'HD2', 'HE', 'HE1', 'HE2',
    'HZ', 'HZ1', 'HZ2', 'HZ3', 'HH', 'HH11', 'HH12', 'HH21', 'HH22'
})

# Standard amino acid 3-letter codes
STANDARD_AA: FrozenSet[str] = frozenset({
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
})

# Non-standard amino acids that should be treated as protein
NONSTANDARD_AA: FrozenSet[str] = frozenset({
    'MSE', 'SEC', 'PYL', 'HYP', 'SEP', 'TPO', 'PTR', 'CSO', 'CSS',
    'CME', 'MLY', 'ALY', 'M3L', 'OCS', 'CSD', 'CAS', 'CGU'
})

# Common solvent/buffer molecules (not ligands)
SOLVENTS: FrozenSet[str] = frozenset({
    'HOH', 'WAT', 'SOL', 'DOD', 'D2O', 'TIP'
})

BUFFERS: FrozenSet[str] = frozenset({
    'SO4', 'PO4', 'GOL', 'EDO', 'PEG', 'MPD', 'ACT', 'ACY', 'FMT', 'TRS',
    'CL', 'NA', 'MG', 'CA', 'ZN', 'K', 'IOD', 'BR', 'BME', 'DMS'
})

# Distance cutoffs (in Angstroms)
METAL_CONTACT_CUTOFF: float = 2.6
PRIMARY_CONTACT_CUTOFF: float = 3.6
SECONDARY_CONTACT_CUTOFF: float = 4.2
MOBILE_RADIUS: float = 10.0

# Catres-catres interaction cutoffs (in Angstroms)
HBOND_CUTOFF: float = 3.5  # Hydrogen bond distance cutoff (donor-acceptor)
SALT_BRIDGE_CUTOFF: float = 4.0  # Salt bridge distance cutoff
PI_STACK_CUTOFF: float = 5.5  # Pi-stacking centroid distance cutoff
PI_STACK_ANGLE_CUTOFF: float = 30.0  # Max angle deviation from parallel/perpendicular (degrees)

# Residue classifications for catres-catres interactions
POSITIVELY_CHARGED_AA: FrozenSet[str] = frozenset({'LYS', 'ARG', 'HIS'})
NEGATIVELY_CHARGED_AA: FrozenSet[str] = frozenset({'ASP', 'GLU'})
AROMATIC_AA: FrozenSet[str] = frozenset({'PHE', 'TYR', 'TRP', 'HIS'})

# Hydrogen bond donor atoms by residue
HBOND_DONORS: Dict[str, FrozenSet[str]] = {
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

# Hydrogen bond acceptor atoms by residue
HBOND_ACCEPTORS: Dict[str, FrozenSet[str]] = {
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

# Charged atoms for salt bridges
POSITIVE_CHARGE_ATOMS: Dict[str, FrozenSet[str]] = {
    'LYS': frozenset({'NZ'}),
    'ARG': frozenset({'NE', 'NH1', 'NH2'}),
    'HIS': frozenset({'ND1', 'NE2'}),
}

NEGATIVE_CHARGE_ATOMS: Dict[str, FrozenSet[str]] = {
    'ASP': frozenset({'OD1', 'OD2'}),
    'GLU': frozenset({'OE1', 'OE2'}),
}

# Aromatic ring atoms for pi-stacking calculations
AROMATIC_RING_ATOMS: Dict[str, FrozenSet[str]] = {
    'PHE': frozenset({'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'}),
    'TYR': frozenset({'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'}),
    'TRP': frozenset({'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'}),
    'HIS': frozenset({'CG', 'ND1', 'CD2', 'CE1', 'NE2'}),
}

# Contact priorities
PRIORITY_METAL: int = 100
PRIORITY_PRIMARY_HETERO: int = 50
PRIORITY_PRIMARY_OTHER: int = 30
PRIORITY_SECONDARY_HETERO: int = 10
PRIORITY_SECONDARY_OTHER: int = 5
PRIORITY_CARBON_CARBON: int = 5

# Catres-catres interaction priorities
PRIORITY_CATRES_HBOND: int = 80
PRIORITY_CATRES_SALT_BRIDGE: int = 90
PRIORITY_CATRES_PI_STACK: int = 40

# Constraint parameters
# Note: stdev values are in Angstroms. Looser values prevent excessive penalties.
COORD_CST_WEIGHT: float = 200.0  # Increased from 100.0 for better constraint enforcement
COORD_CST_STDEV: float = 0.1    # Relaxed from 0.01 to avoid massive penalties
METAL_CST_STDEV: float = 0.05   # Relaxed from 0.005 for realistic metal coordination
PRIMARY_CST_STDEV: float = 0.1  # Relaxed from 0.01 for primary contacts
SECONDARY_CST_STDEV: float = 0.2  # Relaxed from 0.05 for secondary contacts

# Catres-catres constraint parameters (tight tolerances)
HBOND_CST_STDEV: float = 0.1  # Tight tolerance for hydrogen bonds
SALT_BRIDGE_CST_STDEV: float = 0.15  # Slightly looser for salt bridges
PI_STACK_CST_STDEV: float = 0.3  # More flexible for pi-stacking

# FastRelax parameters
CART_BONDED_WEIGHT: float = 1.0  # Increased to penalize bad geometry more heavily
FASTRELAX_CYCLES: int = 3  # Increased for better convergence
LIGAND_CST_STDEV: float = 0.001  # Very tight constraints for ligand freezing (0.001 A)
ALLOW_CATRES_BB: bool = True  # Whether to allow backbone movement for catres (default True)

# Multi-stage relaxation parameters
USE_MULTISTAGE_RELAX: bool = True  # Use multi-stage relaxation protocol
INITIAL_COORD_CST_WEIGHT: float = 1000.0  # Stage 1: high constraint weight
FINAL_COORD_CST_WEIGHT: float = 100.0  # Final stage: low constraint weight
INITIAL_FA_REP_SCALE: float = 0.15  # Stage 1: low repulsion to allow movement
N_RELAX_STAGES: int = 3  # Number of relaxation stages

# MPNN defaults
DEFAULT_MPNN_RUNNER: str = "/net/software/lab/fused_mpnn/seth_temp/run.py"
DEFAULT_MODEL_TYPE: str = "ligand_mpnn"
DEFAULT_ENHANCE_MODEL: str = "plddt_3_20240930-f9c9ea0f"
DEFAULT_TEMPERATURE: float = 0.1
DEFAULT_BATCHES: int = 10
DEFAULT_BATCH_SIZE: int = 1
DEFAULT_OMIT_AA: str = "CM"
DEFAULT_SC_DENOISING_STEPS: int = 3
DEFAULT_APPTAINER_IMAGE: str = "/software/containers/universal.sif"

# Rosetta defaults
DEFAULT_ROSETTA_PATH: str = "/software/rosetta/latest"
DEFAULT_PYROSETTA_PATH: str = "/software/pyrosetta/latest"
DEFAULT_SCOREFUNCTION: str = "beta_jan25"
DEFAULT_PYROSETTA_IMAGE: str = "/software/containers/pyrosetta.sif"

# Scorefunctions that require special initialization flags
BETA_SCOREFUNCTIONS: FrozenSet[str] = frozenset({
    'beta_jan25', 'beta_nov16', 'beta_july15', 'beta_nov15', 'beta'
})

# Pipeline defaults
DEFAULT_N_CYCLES: int = 3
DEFAULT_N_CANDIDATES: int = 10
DEFAULT_N_KEEP: int = 2
DEFAULT_N_FINAL: int = 10

# Slurm defaults
DEFAULT_SLURM_TIME: str = "4:00:00"
DEFAULT_SLURM_CPUS: int = 8
DEFAULT_SLURM_MEM: str = "16G"

# Backbone atom names by residue
BACKBONE_ATOMS: FrozenSet[str] = frozenset({'N', 'CA', 'C', 'O', 'H', 'HA'})

# Map of 3-letter to 1-letter amino acid codes
AA_3TO1: Dict[str, str] = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M', 'SEC': 'U', 'PYL': 'O'
}

AA_1TO3: Dict[str, str] = {v: k for k, v in AA_3TO1.items() if k in STANDARD_AA}

# Histidine protonation states
# HIS: standard histidine, proton on NE2 (epsilon nitrogen)
# HIS_D: delta-protonated histidine, proton on ND1 (delta nitrogen)
#        Used when NE2 coordinates a metal ion
HIS_STANDARD_RESNAME: str = "HIS"
HIS_DELTA_RESNAME: str = "HIS_D"

# Histidine tautomer residue names used in various force fields
HIS_TAUTOMER_NAMES: FrozenSet[str] = frozenset({
    'HIS',    # Standard histidine (may be ambiguous)
    'HIS_D',  # Rosetta delta-protonated (H on ND1)
    'HID',    # AMBER delta-protonated (H on ND1)
    'HIE',    # AMBER epsilon-protonated (H on NE2)
    'HIP',    # AMBER doubly protonated
    'HSE',    # CHARMM epsilon-protonated
    'HSD',    # CHARMM delta-protonated
    'HSP',    # CHARMM doubly protonated
})

# Metal coordination distance cutoff for histidine-metal bonds
METAL_COORDINATION_CUTOFF: float = 2.5
