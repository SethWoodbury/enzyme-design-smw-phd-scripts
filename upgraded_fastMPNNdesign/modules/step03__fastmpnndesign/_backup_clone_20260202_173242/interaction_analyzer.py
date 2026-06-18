"""Interaction analysis for conserving favorable mutations in step03 FastMPNN design.

This module provides:
- Analysis of new interactions between designed residues and catalytic residues
- Analysis of interactions between designed residues and ligand
- Detection of beneficial mutations worth conserving
- Generation of MPNN biases based on interaction analysis
- Configurable interaction types: H-bond, pi-stacking, metal coordination,
  hydrophobic contacts, charged/ionic, cation-pi, halogen bonds

The key concept is "H-bond keeper" style conservation: when a mutation creates
a favorable interaction (H-bond, pi-stacking, etc.) with a catalytic residue or
ligand, we probabilistically fix that residue in subsequent MPNN rounds.
"""
import argparse
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..module_utils.pdb_utils import read_pdb_atoms, get_residue_atoms
from ..module_utils.sequence_utils import get_sequence_with_positions
from ..module_utils.interaction_utils import (
    analyze_residue_interactions,
    count_favorable_interactions,
    has_strong_interactions,
    detect_hbonds,
    detect_pi_stacking,
    detect_hydrophobic_contacts,
    detect_metal_coordination,
    detect_charged_interactions,
    get_ring_atoms,
    compute_ring_geometry,
    distance,
    get_coords,
)
from ..module_utils.constants import (
    DEFAULT_CONSERVATION_PROBABILITY,
    DEFAULT_HBOND_ACCEPT_PROBABILITY,
    HBOND_DIST_MAX,
    PI_CENTROID_DIST_MAX,
    HYDROPHOBIC_DIST_MAX,
    CHARGED_DIST_MAX,
    METAL_COORD_DIST_MAX,
    AROMATIC_RESIDUES,
    AROMATIC_RING_ATOMS,
    HYDROPHOBIC_RESIDUES,
    CHARGED_RESIDUES,
    POLAR_ELEMENTS,
)

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Interaction Type Enumeration
# =============================================================================

class InteractionType(Enum):
    """Enumeration of supported interaction types."""
    HBOND = "hbond"
    PI_STACK = "pi_stack"
    METAL = "metal"
    HYDROPHOBIC = "hydrophobic"
    CHARGED = "charged"
    CATION_PI = "cation_pi"
    HALOGEN = "halogen"

    @classmethod
    def from_string(cls, s: str) -> "InteractionType":
        """Parse interaction type from string."""
        mapping = {
            "hbond": cls.HBOND,
            "h_bond": cls.HBOND,
            "hydrogen_bond": cls.HBOND,
            "pi_stack": cls.PI_STACK,
            "pi_stacking": cls.PI_STACK,
            "pi": cls.PI_STACK,
            "metal": cls.METAL,
            "metal_coord": cls.METAL,
            "metal_coordination": cls.METAL,
            "spodium": cls.METAL,  # Spodium bonds are a type of metal coordination
            "hydrophobic": cls.HYDROPHOBIC,
            "hydro": cls.HYDROPHOBIC,
            "charged": cls.CHARGED,
            "ionic": cls.CHARGED,
            "salt_bridge": cls.CHARGED,
            "cation_pi": cls.CATION_PI,
            "cation-pi": cls.CATION_PI,
            "halogen": cls.HALOGEN,
            "halogen_bond": cls.HALOGEN,
            "xbond": cls.HALOGEN,
        }
        key = s.lower().strip()
        if key not in mapping:
            raise ValueError(f"Unknown interaction type: {s}. Valid types: {list(mapping.keys())}")
        return mapping[key]

    @classmethod
    def all_types(cls) -> List["InteractionType"]:
        """Return all interaction types."""
        return list(cls)

    @classmethod
    def default_types(cls) -> List["InteractionType"]:
        """Return default interaction types (H-bond, pi-stacking, metal, charged)."""
        return [cls.HBOND, cls.PI_STACK, cls.METAL, cls.CHARGED]


# =============================================================================
# Interaction Configuration
# =============================================================================

@dataclass
class InteractionConfig:
    """Configuration for interaction detection and conservation.

    This class holds all settings for controlling which interactions to detect,
    their distance/angle thresholds, and how they contribute to conservation scoring.

    Attributes:
        interaction_types: List of interaction types to detect
        include_ligand_interactions: Whether to analyze ligand interactions
        include_catres_interactions: Whether to analyze catalytic residue interactions

        # Distance thresholds (Angstroms)
        hbond_dist_max: Maximum H...A distance for H-bonds
        pi_stack_dist_max: Maximum centroid distance for pi-stacking
        metal_coord_dist_max: Maximum distance for metal coordination
        hydrophobic_dist_max: Maximum C-C distance for hydrophobic contacts
        charged_dist_max: Maximum distance for charged interactions
        cation_pi_dist_max: Maximum cation-to-ring-centroid distance
        halogen_dist_max: Maximum X...A distance for halogen bonds

        # MPNN biases for each interaction type
        hbond_bias: MPNN bias for H-bond forming amino acids
        pi_stack_bias: MPNN bias for pi-stacking amino acids
        metal_bias: MPNN bias for metal-coordinating amino acids
        hydrophobic_bias: MPNN bias for hydrophobic contact amino acids
        charged_bias: MPNN bias for charged interaction amino acids
        cation_pi_bias: MPNN bias for cation-pi interaction amino acids
        halogen_bias: MPNN bias for halogen bond amino acids

        # Conservation scoring weights
        hbond_score_weight: Weight for H-bonds in conservation score
        pi_stack_score_weight: Weight for pi-stacking in conservation score
        metal_score_weight: Weight for metal coordination in conservation score
        hydrophobic_score_weight: Weight for hydrophobic contacts
        charged_score_weight: Weight for charged interactions
        cation_pi_score_weight: Weight for cation-pi interactions
        halogen_score_weight: Weight for halogen bonds

        # Conservation probability settings
        conservation_probability: Base probability to conserve a favorable mutation
        hbond_accept_probability: Probability boost for H-bond forming mutations
        strong_interaction_types: Types considered "strong" for conservation decisions
    """
    # Interaction types to detect
    interaction_types: List[InteractionType] = field(
        default_factory=lambda: InteractionType.default_types()
    )
    include_ligand_interactions: bool = False
    include_catres_interactions: bool = True

    # Distance thresholds (Angstroms)
    hbond_dist_max: float = HBOND_DIST_MAX
    pi_stack_dist_max: float = PI_CENTROID_DIST_MAX
    metal_coord_dist_max: float = METAL_COORD_DIST_MAX
    hydrophobic_dist_max: float = HYDROPHOBIC_DIST_MAX
    charged_dist_max: float = CHARGED_DIST_MAX
    cation_pi_dist_max: float = 6.0  # Typical cation-pi distance
    halogen_dist_max: float = 3.5    # Typical halogen bond distance

    # MPNN biases for each interaction type
    hbond_bias: float = 2.0
    pi_stack_bias: float = 1.5
    metal_bias: float = 2.5
    hydrophobic_bias: float = 0.5
    charged_bias: float = 1.5
    cation_pi_bias: float = 1.5
    halogen_bias: float = 1.0

    # Conservation scoring weights (how much each interaction contributes to score)
    hbond_score_weight: float = 0.5
    pi_stack_score_weight: float = 0.4
    metal_score_weight: float = 0.6
    hydrophobic_score_weight: float = 0.1
    charged_score_weight: float = 0.4
    cation_pi_score_weight: float = 0.3
    halogen_score_weight: float = 0.2

    # Conservation probability settings
    conservation_probability: float = DEFAULT_CONSERVATION_PROBABILITY
    hbond_accept_probability: float = DEFAULT_HBOND_ACCEPT_PROBABILITY

    # Which interaction types are considered "strong" for conservation
    strong_interaction_types: List[InteractionType] = field(
        default_factory=lambda: [InteractionType.HBOND, InteractionType.PI_STACK, InteractionType.METAL]
    )

    # Atom scope filters
    mutator_atoms: str = "either"  # sidechain|backbone|either
    target_atoms: str = "either"   # sidechain|backbone|either

    def has_interaction_type(self, itype: InteractionType) -> bool:
        """Check if an interaction type is enabled."""
        return itype in self.interaction_types

    def get_bias_for_type(self, itype: InteractionType) -> float:
        """Get MPNN bias for an interaction type."""
        bias_map = {
            InteractionType.HBOND: self.hbond_bias,
            InteractionType.PI_STACK: self.pi_stack_bias,
            InteractionType.METAL: self.metal_bias,
            InteractionType.HYDROPHOBIC: self.hydrophobic_bias,
            InteractionType.CHARGED: self.charged_bias,
            InteractionType.CATION_PI: self.cation_pi_bias,
            InteractionType.HALOGEN: self.halogen_bias,
        }
        return bias_map.get(itype, 0.0)

    def get_score_weight_for_type(self, itype: InteractionType) -> float:
        """Get conservation score weight for an interaction type."""
        weight_map = {
            InteractionType.HBOND: self.hbond_score_weight,
            InteractionType.PI_STACK: self.pi_stack_score_weight,
            InteractionType.METAL: self.metal_score_weight,
            InteractionType.HYDROPHOBIC: self.hydrophobic_score_weight,
            InteractionType.CHARGED: self.charged_score_weight,
            InteractionType.CATION_PI: self.cation_pi_score_weight,
            InteractionType.HALOGEN: self.halogen_score_weight,
        }
        return weight_map.get(itype, 0.0)

    def is_strong_interaction(self, itype: InteractionType) -> bool:
        """Check if an interaction type is considered 'strong'."""
        return itype in self.strong_interaction_types

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "InteractionConfig":
        """Create InteractionConfig from CLI arguments.

        Args:
            args: Parsed CLI arguments with interaction-related options

        Returns:
            Configured InteractionConfig instance
        """
        config = cls()

        # Parse interaction types if provided
        if hasattr(args, 'interaction_types') and args.interaction_types:
            type_strs = args.interaction_types.split(',')
            config.interaction_types = [InteractionType.from_string(t) for t in type_strs]

        # Ligand interactions flag
        if hasattr(args, 'include_ligand_interactions'):
            config.include_ligand_interactions = args.include_ligand_interactions

        # Distance thresholds
        if hasattr(args, 'hbond_dist') and args.hbond_dist is not None:
            config.hbond_dist_max = args.hbond_dist
        if hasattr(args, 'pi_stack_dist') and args.pi_stack_dist is not None:
            config.pi_stack_dist_max = args.pi_stack_dist
        if hasattr(args, 'metal_dist') and args.metal_dist is not None:
            config.metal_coord_dist_max = args.metal_dist
        if hasattr(args, 'hydrophobic_dist') and args.hydrophobic_dist is not None:
            config.hydrophobic_dist_max = args.hydrophobic_dist
        if hasattr(args, 'charged_dist') and args.charged_dist is not None:
            config.charged_dist_max = args.charged_dist
        if hasattr(args, 'cation_pi_dist') and args.cation_pi_dist is not None:
            config.cation_pi_dist_max = args.cation_pi_dist
        if hasattr(args, 'halogen_dist') and args.halogen_dist is not None:
            config.halogen_dist_max = args.halogen_dist

        # MPNN biases
        if hasattr(args, 'hbond_bias') and args.hbond_bias is not None:
            config.hbond_bias = args.hbond_bias
        if hasattr(args, 'pi_stack_bias') and args.pi_stack_bias is not None:
            config.pi_stack_bias = args.pi_stack_bias
        if hasattr(args, 'metal_bias') and args.metal_bias is not None:
            config.metal_bias = args.metal_bias

        # Conservation settings
        if hasattr(args, 'conservation_probability') and args.conservation_probability is not None:
            config.conservation_probability = args.conservation_probability

        return config

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add CLI arguments for interaction configuration.

        Args:
            parser: ArgumentParser to add arguments to
        """
        group = parser.add_argument_group("Interaction Analysis Options")

        group.add_argument(
            "--interaction_types",
            type=str,
            default="hbond,pi_stack,metal,charged",
            help="Comma-separated list of interaction types to detect. "
                 "Options: hbond, pi_stack, metal, hydrophobic, charged, cation_pi, halogen. "
                 "Example: --interaction_types hbond,pi_stack,metal,hydrophobic"
        )

        group.add_argument(
            "--include_ligand_interactions",
            action="store_true",
            default=False,
            help="Include interactions with ligand in addition to catalytic residues"
        )

        group.add_argument(
            "--no_catres_interactions",
            action="store_true",
            default=False,
            help="Disable interactions with catalytic residues (only analyze ligand)"
        )

        # Distance thresholds
        group.add_argument(
            "--hbond_dist", type=float, default=None,
            help=f"Maximum H-bond distance in Angstroms (default: {HBOND_DIST_MAX})"
        )
        group.add_argument(
            "--pi_stack_dist", type=float, default=None,
            help=f"Maximum pi-stacking centroid distance (default: {PI_CENTROID_DIST_MAX})"
        )
        group.add_argument(
            "--metal_dist", type=float, default=None,
            help=f"Maximum metal coordination distance (default: {METAL_COORD_DIST_MAX})"
        )
        group.add_argument(
            "--hydrophobic_dist", type=float, default=None,
            help=f"Maximum hydrophobic contact distance (default: {HYDROPHOBIC_DIST_MAX})"
        )
        group.add_argument(
            "--charged_dist", type=float, default=None,
            help=f"Maximum charged interaction distance (default: {CHARGED_DIST_MAX})"
        )
        group.add_argument(
            "--cation_pi_dist", type=float, default=None,
            help="Maximum cation-pi distance (default: 6.0)"
        )
        group.add_argument(
            "--halogen_dist", type=float, default=None,
            help="Maximum halogen bond distance (default: 3.5)"
        )

        # MPNN biases
        group.add_argument(
            "--hbond_bias", type=float, default=None,
            help="MPNN bias for H-bond forming residues (default: 2.0)"
        )
        group.add_argument(
            "--pi_stack_bias", type=float, default=None,
            help="MPNN bias for pi-stacking residues (default: 1.5)"
        )
        group.add_argument(
            "--metal_bias", type=float, default=None,
            help="MPNN bias for metal-coordinating residues (default: 2.5)"
        )

        # Conservation
        group.add_argument(
            "--conservation_probability", type=float, default=None,
            help=f"Base probability to conserve favorable mutations (default: {DEFAULT_CONSERVATION_PROBABILITY})"
        )


# =============================================================================
# Additional Interaction Detection Functions (Cation-Pi, Halogen)
# These are implemented here since they're not in interaction_utils.py
# =============================================================================

# Cation-containing residues and their cationic atoms
CATION_RESIDUES: Set[str] = {"LYS", "ARG"}
CATION_ATOMS_BY_RESIDUE: Dict[str, List[str]] = {
    "LYS": ["NZ"],
    "ARG": ["CZ"],  # Guanidinium center
}

# Halogen atoms (for halogen bond detection)
HALOGEN_ATOMS: Set[str] = {"F", "CL", "BR", "I", "FL", "BR1", "CL1", "I1"}


def detect_cation_pi_interactions(
    residue_atoms: List[Dict],
    resname: str,
    target_atoms: List[Dict],
    target_resname: Optional[str] = None,
    dist_max: float = 6.0,
) -> List[Dict]:
    """Detect cation-pi interactions between a residue and target.

    Cation-pi interactions occur between:
    - Positively charged groups (LYS NZ, ARG guanidinium) and aromatic rings
    - Works in both directions (residue can be cation or aromatic)

    Args:
        residue_atoms: Atoms of the residue
        resname: Three-letter residue code
        target_atoms: Atoms of target (ligand or another residue)
        target_resname: Three-letter code of target residue (None for ligand)
        dist_max: Maximum cation-to-centroid distance

    Returns:
        List of cation-pi interaction dicts
    """
    interactions = []

    # Case 1: Residue is cation, target is aromatic
    if resname in CATION_RESIDUES:
        cation_atom_names = CATION_ATOMS_BY_RESIDUE.get(resname, [])
        cation_atoms = [a for a in residue_atoms if a["atom_name"] in cation_atom_names]

        # Get target rings (if aromatic residue or ligand)
        target_rings = []
        if target_resname and target_resname in AROMATIC_RESIDUES:
            target_rings = get_ring_atoms(target_atoms, target_resname)
        else:
            # For ligand, try to find aromatic rings by looking for ring patterns
            # This is a simplified heuristic - check for 6-membered carbon rings
            target_rings = _find_ligand_aromatic_rings(target_atoms)

        for cat_atom in cation_atoms:
            cat_coords = get_coords(cat_atom)
            for ring in target_rings:
                if len(ring) < 3:
                    continue
                centroid, _ = compute_ring_geometry(ring)
                d = np.linalg.norm(cat_coords - centroid)
                if d <= dist_max:
                    interactions.append({
                        "type": "cation_pi",
                        "cation_atom": cat_atom["atom_name"],
                        "cation_residue": resname,
                        "distance": round(d, 2),
                        "role": "cation",
                    })
                    break  # One interaction per cation atom

    # Case 2: Residue is aromatic, target has cation
    if resname in AROMATIC_RESIDUES:
        rings = get_ring_atoms(residue_atoms, resname)

        # Find cation atoms in target
        target_cation_atoms = []
        if target_resname in CATION_RESIDUES:
            cation_names = CATION_ATOMS_BY_RESIDUE.get(target_resname, [])
            target_cation_atoms = [a for a in target_atoms if a["atom_name"] in cation_names]
        else:
            # For ligand, look for nitrogen atoms that might be cationic
            for a in target_atoms:
                elem = a.get("element", a["atom_name"][0])
                if elem.upper() == "N":
                    target_cation_atoms.append(a)

        for ring in rings:
            if len(ring) < 3:
                continue
            centroid, _ = compute_ring_geometry(ring)
            for cat_atom in target_cation_atoms:
                cat_coords = get_coords(cat_atom)
                d = np.linalg.norm(cat_coords - centroid)
                if d <= dist_max:
                    interactions.append({
                        "type": "cation_pi",
                        "aromatic_residue": resname,
                        "cation_atom": cat_atom["atom_name"],
                        "distance": round(d, 2),
                        "role": "aromatic",
                    })
                    break  # One interaction per ring

    return interactions


def _find_ligand_aromatic_rings(atoms: List[Dict]) -> List[List[Dict]]:
    """Attempt to find aromatic rings in a ligand.

    This is a simplified heuristic that looks for groups of 5-6 carbon atoms
    that are close together and roughly planar.

    Args:
        atoms: All ligand atoms

    Returns:
        List of potential ring atom lists
    """
    # Simple approach: look for carbon atoms close together
    # This is a placeholder - real aromatic detection would need connectivity info
    carbon_atoms = [a for a in atoms if a.get("element", a["atom_name"][0]).upper() == "C"]

    if len(carbon_atoms) < 5:
        return []

    # Try to find clusters of 5-6 carbons within 2.5A of each other
    rings = []
    used = set()

    for i, center in enumerate(carbon_atoms):
        if i in used:
            continue

        cluster = [center]
        center_coords = get_coords(center)

        for j, other in enumerate(carbon_atoms):
            if i == j or j in used:
                continue
            other_coords = get_coords(other)
            d = np.linalg.norm(center_coords - other_coords)
            if d < 2.5:  # Typical C-C aromatic bond distance is ~1.4A
                cluster.append(other)

        if 5 <= len(cluster) <= 6:
            # Check if roughly planar (simplified check)
            rings.append(cluster)
            for atom in cluster:
                idx = carbon_atoms.index(atom)
                used.add(idx)

    return rings


def detect_halogen_bonds(
    residue_atoms: List[Dict],
    target_atoms: List[Dict],
    dist_max: float = 3.5,
) -> List[Dict]:
    """Detect halogen bonds between residue and target.

    Halogen bonds (X-bonds) occur between halogen atoms (Cl, Br, I) and
    electronegative atoms (O, N, S). The halogen acts as electron acceptor
    via its sigma-hole.

    Args:
        residue_atoms: Atoms of the residue
        target_atoms: Atoms of target (ligand or another residue)
        dist_max: Maximum X...A distance

    Returns:
        List of halogen bond dicts
    """
    interactions = []

    # Find halogen atoms in target (typically ligands have halogens)
    halogen_atoms = []
    for a in target_atoms:
        elem = a.get("element", a["atom_name"][0]).upper()
        atom_name = a["atom_name"].upper()
        if elem in HALOGEN_ATOMS or atom_name in HALOGEN_ATOMS:
            halogen_atoms.append(a)

    if not halogen_atoms:
        return interactions

    # Find electronegative atoms in residue (N, O, S)
    acceptor_atoms = []
    for a in residue_atoms:
        elem = a.get("element", a["atom_name"][0]).upper()
        if elem in {"N", "O", "S"}:
            acceptor_atoms.append(a)

    # Check for halogen bonds
    for hal in halogen_atoms:
        hal_coords = get_coords(hal)
        for acc in acceptor_atoms:
            acc_coords = get_coords(acc)
            d = np.linalg.norm(hal_coords - acc_coords)
            if d <= dist_max:
                interactions.append({
                    "type": "halogen_bond",
                    "halogen_atom": hal["atom_name"],
                    "acceptor_atom": acc["atom_name"],
                    "distance": round(d, 2),
                })

    return interactions


# =============================================================================
# Mutation Interaction Dataclass
# =============================================================================

@dataclass
class MutationInteraction:
    """Record of a mutation and its interactions with catalytic residues and/or ligand.

    Attributes:
        chain: Chain ID
        resno: Residue number
        original_aa: Original amino acid (1-letter code)
        new_aa: New amino acid (1-letter code)
        resname_3: New amino acid (3-letter code)

        # Interactions with catalytic residues
        hbonds_to_catres: List of H-bond interactions with catalytic residues
        pi_stacks_to_catres: List of pi-stacking interactions with catalytic residues
        hydrophobic_to_catres: List of hydrophobic contacts with catalytic residues
        metal_to_catres: List of metal coordination interactions with catalytic residues
        charged_to_catres: List of charged/ionic interactions with catalytic residues
        cation_pi_to_catres: List of cation-pi interactions with catalytic residues
        halogen_to_catres: List of halogen bonds with catalytic residues

        # Interactions with ligand
        hbonds_to_ligand: List of H-bond interactions with ligand
        pi_stacks_to_ligand: List of pi-stacking interactions with ligand
        hydrophobic_to_ligand: List of hydrophobic contacts with ligand
        metal_to_ligand: List of metal coordination interactions with ligand
        charged_to_ligand: List of charged/ionic interactions with ligand
        cation_pi_to_ligand: List of cation-pi interactions with ligand
        halogen_to_ligand: List of halogen bonds with ligand

        # Conservation metadata
        should_conserve: Whether this mutation should be conserved
        conservation_score: Score indicating importance of conservation (0-1)
        mpnn_bias: Recommended MPNN bias for this amino acid at this position
        contributing_interactions: Dict of interaction types that contributed to score
    """
    chain: str
    resno: int
    original_aa: str
    new_aa: str
    resname_3: str

    # Interactions with catalytic residues
    hbonds_to_catres: List[Dict] = field(default_factory=list)
    pi_stacks_to_catres: List[Dict] = field(default_factory=list)
    hydrophobic_to_catres: List[Dict] = field(default_factory=list)
    metal_to_catres: List[Dict] = field(default_factory=list)
    charged_to_catres: List[Dict] = field(default_factory=list)
    cation_pi_to_catres: List[Dict] = field(default_factory=list)
    halogen_to_catres: List[Dict] = field(default_factory=list)

    # Interactions with ligand
    hbonds_to_ligand: List[Dict] = field(default_factory=list)
    pi_stacks_to_ligand: List[Dict] = field(default_factory=list)
    hydrophobic_to_ligand: List[Dict] = field(default_factory=list)
    metal_to_ligand: List[Dict] = field(default_factory=list)
    charged_to_ligand: List[Dict] = field(default_factory=list)
    cation_pi_to_ligand: List[Dict] = field(default_factory=list)
    halogen_to_ligand: List[Dict] = field(default_factory=list)

    # Conservation metadata
    should_conserve: bool = False
    conservation_score: float = 0.0
    mpnn_bias: float = 0.0
    contributing_interactions: Dict[str, int] = field(default_factory=dict)

    @property
    def mutation_string(self) -> str:
        """Get mutation string like 'A10V'."""
        return f"{self.original_aa}{self.chain}{self.resno}{self.new_aa}"

    @property
    def residue_id(self) -> str:
        """Get residue ID like 'A10'."""
        return f"{self.chain}{self.resno}"

    @property
    def total_catres_interactions(self) -> int:
        """Total number of interactions with catalytic residues."""
        return (
            len(self.hbonds_to_catres) +
            len(self.pi_stacks_to_catres) +
            len(self.hydrophobic_to_catres) +
            len(self.metal_to_catres) +
            len(self.charged_to_catres) +
            len(self.cation_pi_to_catres) +
            len(self.halogen_to_catres)
        )

    @property
    def total_ligand_interactions(self) -> int:
        """Total number of interactions with ligand."""
        return (
            len(self.hbonds_to_ligand) +
            len(self.pi_stacks_to_ligand) +
            len(self.hydrophobic_to_ligand) +
            len(self.metal_to_ligand) +
            len(self.charged_to_ligand) +
            len(self.cation_pi_to_ligand) +
            len(self.halogen_to_ligand)
        )

    @property
    def total_interactions(self) -> int:
        """Total number of all interactions."""
        return self.total_catres_interactions + self.total_ligand_interactions

    def get_interactions_by_type(self, itype: InteractionType, target: str = "all") -> List[Dict]:
        """Get interactions of a specific type.

        Args:
            itype: Interaction type
            target: "catres", "ligand", or "all"

        Returns:
            List of interaction dicts
        """
        interactions = []

        type_to_catres = {
            InteractionType.HBOND: self.hbonds_to_catres,
            InteractionType.PI_STACK: self.pi_stacks_to_catres,
            InteractionType.HYDROPHOBIC: self.hydrophobic_to_catres,
            InteractionType.METAL: self.metal_to_catres,
            InteractionType.CHARGED: self.charged_to_catres,
            InteractionType.CATION_PI: self.cation_pi_to_catres,
            InteractionType.HALOGEN: self.halogen_to_catres,
        }

        type_to_ligand = {
            InteractionType.HBOND: self.hbonds_to_ligand,
            InteractionType.PI_STACK: self.pi_stacks_to_ligand,
            InteractionType.HYDROPHOBIC: self.hydrophobic_to_ligand,
            InteractionType.METAL: self.metal_to_ligand,
            InteractionType.CHARGED: self.charged_to_ligand,
            InteractionType.CATION_PI: self.cation_pi_to_ligand,
            InteractionType.HALOGEN: self.halogen_to_ligand,
        }

        if target in ("catres", "all"):
            interactions.extend(type_to_catres.get(itype, []))
        if target in ("ligand", "all"):
            interactions.extend(type_to_ligand.get(itype, []))

        return interactions

    def has_interaction_type(self, itype: InteractionType, target: str = "all") -> bool:
        """Check if mutation has a specific interaction type."""
        return len(self.get_interactions_by_type(itype, target)) > 0


@dataclass
class InteractionAnalysisResult:
    """Results from analyzing interactions in a designed structure.

    Attributes:
        mutations: List of MutationInteraction objects for each mutation
        conserved_positions: Set of (chain, resno) tuples to conserve
        fixed_residue_ids: List of residue IDs to add to fixed list (e.g., ["A10"])
        per_residue_bias: Dict mapping residue ID to amino acid biases
        summary: Summary statistics
        config: The InteractionConfig used for analysis
    """
    mutations: List[MutationInteraction] = field(default_factory=list)
    conserved_positions: Set[Tuple[str, int]] = field(default_factory=set)
    fixed_residue_ids: List[str] = field(default_factory=list)
    per_residue_bias: Dict[str, Dict[str, float]] = field(default_factory=dict)
    summary: Dict = field(default_factory=dict)
    config: Optional[InteractionConfig] = None


# =============================================================================
# Main Interaction Analyzer Class
# =============================================================================

class InteractionAnalyzer:
    """Analyze interactions between designed residues and catalytic residues/ligand.

    This class:
    1. Identifies mutations between original and designed sequences
    2. Detects configurable favorable interactions with catres and/or ligand
    3. Determines which mutations should be conserved
    4. Generates MPNN biases for subsequent rounds
    """

    # Mapping of 1-letter to 3-letter codes
    AA_1_TO_3: Dict[str, str] = {
        "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
        "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
        "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
        "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
    }

    BACKBONE_ATOMS: Set[str] = {"N", "CA", "C", "O", "H", "HA", "HA2", "HA3"}

    @classmethod
    def _filter_atoms_by_scope(cls, atoms: List[Dict], scope: str) -> List[Dict]:
        scope = (scope or "either").lower()
        if scope == "either":
            return atoms
        if scope == "backbone":
            return [a for a in atoms if a.get("atom_name", "").strip() in cls.BACKBONE_ATOMS]
        if scope == "sidechain":
            return [a for a in atoms if a.get("atom_name", "").strip() not in cls.BACKBONE_ATOMS]
        return atoms

    def __init__(
        self,
        config: Optional[InteractionConfig] = None,
        # Legacy parameters for backward compatibility
        conservation_probability: Optional[float] = None,
        hbond_accept_probability: Optional[float] = None,
        hbond_bias: Optional[float] = None,
        pi_stack_bias: Optional[float] = None,
        hydrophobic_bias: Optional[float] = None,
    ):
        """Initialize interaction analyzer.

        Args:
            config: InteractionConfig object with all settings (recommended)
            conservation_probability: Legacy - base probability to conserve
            hbond_accept_probability: Legacy - probability boost for H-bonds
            hbond_bias: Legacy - MPNN bias for H-bond forming amino acids
            pi_stack_bias: Legacy - MPNN bias for pi-stacking amino acids
            hydrophobic_bias: Legacy - MPNN bias for hydrophobic contact amino acids
        """
        # Use config if provided, otherwise create default
        self.config = config if config is not None else InteractionConfig()

        # Apply legacy parameters if provided (override config)
        if conservation_probability is not None:
            self.config.conservation_probability = conservation_probability
        if hbond_accept_probability is not None:
            self.config.hbond_accept_probability = hbond_accept_probability
        if hbond_bias is not None:
            self.config.hbond_bias = hbond_bias
        if pi_stack_bias is not None:
            self.config.pi_stack_bias = pi_stack_bias
        if hydrophobic_bias is not None:
            self.config.hydrophobic_bias = hydrophobic_bias

        # Legacy aliases for backward compatibility
        self.conservation_probability = self.config.conservation_probability
        self.hbond_accept_probability = self.config.hbond_accept_probability
        self.hbond_bias = self.config.hbond_bias
        self.pi_stack_bias = self.config.pi_stack_bias
        self.hydrophobic_bias = self.config.hydrophobic_bias

    def _detect_all_interactions(
        self,
        mut_atoms: List[Dict],
        resname_3: str,
        target_atoms: List[Dict],
        target_resname: Optional[str] = None,
        target_type: str = "residue",
    ) -> Dict[InteractionType, List[Dict]]:
        """Detect all enabled interaction types between residue and target.

        Args:
            mut_atoms: Atoms of the mutated residue
            resname_3: Three-letter code of the mutated residue
            target_atoms: Atoms of the target (catres or ligand)
            target_resname: Three-letter code of target residue (None for ligand)
            target_type: "residue" or "ligand"

        Returns:
            Dict mapping InteractionType to list of detected interactions
        """
        results: Dict[InteractionType, List[Dict]] = {itype: [] for itype in InteractionType}

        # Use interaction_utils.py functions for most types
        if self.config.has_interaction_type(InteractionType.HBOND):
            # H-bonds as donor
            hbonds_donor = detect_hbonds(mut_atoms, target_atoms, self.config.hbond_dist_max)
            for hb in hbonds_donor:
                hb["role"] = "donor"
            # H-bonds as acceptor
            hbonds_acceptor = detect_hbonds(target_atoms, mut_atoms, self.config.hbond_dist_max)
            for hb in hbonds_acceptor:
                hb["role"] = "acceptor"
            results[InteractionType.HBOND] = hbonds_donor + hbonds_acceptor

        if self.config.has_interaction_type(InteractionType.PI_STACK):
            if resname_3 in AROMATIC_RESIDUES:
                if target_type == "ligand":
                    # For ligand, use analyze_residue_interactions which handles ligand pi
                    interaction_results = analyze_residue_interactions(
                        mut_atoms, resname_3, target_atoms, target_resname, target_type
                    )
                    results[InteractionType.PI_STACK] = interaction_results.get("pi_interactions", [])
                elif target_resname and target_resname in AROMATIC_RESIDUES:
                    pi_stacks = detect_pi_stacking(mut_atoms, resname_3, target_atoms, target_resname)
                    results[InteractionType.PI_STACK] = pi_stacks

        if self.config.has_interaction_type(InteractionType.HYDROPHOBIC):
            if resname_3 in HYDROPHOBIC_RESIDUES:
                hydro = detect_hydrophobic_contacts(
                    mut_atoms, target_atoms, self.config.hydrophobic_dist_max
                )
                results[InteractionType.HYDROPHOBIC] = hydro

        if self.config.has_interaction_type(InteractionType.METAL):
            metal = detect_metal_coordination(
                mut_atoms, target_atoms, self.config.metal_coord_dist_max
            )
            results[InteractionType.METAL] = metal

        if self.config.has_interaction_type(InteractionType.CHARGED):
            if resname_3 in CHARGED_RESIDUES:
                charged = detect_charged_interactions(
                    mut_atoms, resname_3, target_atoms, self.config.charged_dist_max
                )
                results[InteractionType.CHARGED] = charged

        if self.config.has_interaction_type(InteractionType.CATION_PI):
            cation_pi = detect_cation_pi_interactions(
                mut_atoms, resname_3, target_atoms, target_resname, self.config.cation_pi_dist_max
            )
            results[InteractionType.CATION_PI] = cation_pi

        if self.config.has_interaction_type(InteractionType.HALOGEN):
            halogen = detect_halogen_bonds(
                mut_atoms, target_atoms, self.config.halogen_dist_max
            )
            results[InteractionType.HALOGEN] = halogen

        return results

    def _calculate_conservation_score(
        self,
        interactions_by_type: Dict[InteractionType, int],
    ) -> float:
        """Calculate conservation score based on interaction counts.

        Args:
            interactions_by_type: Dict mapping InteractionType to count

        Returns:
            Conservation score (0-1)
        """
        score = 0.0
        for itype, count in interactions_by_type.items():
            if count > 0:
                weight = self.config.get_score_weight_for_type(itype)
                score += weight * count

        return min(1.0, score)  # Cap at 1.0

    def _calculate_mpnn_bias(
        self,
        interactions_by_type: Dict[InteractionType, int],
    ) -> float:
        """Calculate MPNN bias based on detected interactions.

        Args:
            interactions_by_type: Dict mapping InteractionType to count

        Returns:
            MPNN bias value
        """
        bias = 0.0
        for itype, count in interactions_by_type.items():
            if count > 0:
                bias += self.config.get_bias_for_type(itype)

        return bias

    def _should_conserve(
        self,
        interactions_by_type: Dict[InteractionType, int],
    ) -> bool:
        """Determine if mutation should be conserved based on interactions.

        Args:
            interactions_by_type: Dict mapping InteractionType to count

        Returns:
            True if mutation should be conserved
        """
        for itype, count in interactions_by_type.items():
            if count > 0 and self.config.is_strong_interaction(itype):
                return True
        return False

    def analyze_mutations(
        self,
        designed_pdb: str,
        original_pdb: str,
        catres_positions: List[Tuple[str, int]],
        ligand_chain: Optional[str] = None,
        ligand_resname: Optional[str] = None,
    ) -> InteractionAnalysisResult:
        """Analyze interactions created by mutations.

        Args:
            designed_pdb: Path to designed PDB
            original_pdb: Path to original (reference) PDB
            catres_positions: List of (chain, resno) for catalytic residues
            ligand_chain: Chain ID for ligand (if analyzing ligand interactions)
            ligand_resname: Residue name of ligand (e.g., "LIG", "SUB")

        Returns:
            InteractionAnalysisResult with mutation analysis
        """
        LOGGER.info(f"Analyzing mutation interactions in {designed_pdb}")
        LOGGER.info(f"  Enabled interaction types: {[t.value for t in self.config.interaction_types]}")
        LOGGER.info(f"  Include ligand interactions: {self.config.include_ligand_interactions}")

        # Read structures
        _, designed_atoms = read_pdb_atoms(designed_pdb)
        _, original_atoms = read_pdb_atoms(original_pdb)

        # Get sequences
        designed_seq = get_sequence_with_positions(designed_atoms)
        original_seq = get_sequence_with_positions(original_atoms)

        # Get catres atoms dict
        catres_atoms_dict = {}
        if self.config.include_catres_interactions:
            for chain, resno in catres_positions:
                atoms = get_residue_atoms(designed_atoms, chain, resno)
                atoms = self._filter_atoms_by_scope(atoms, self.config.target_atoms)
                if atoms:
                    catres_atoms_dict[(chain, resno)] = atoms

        # Get ligand atoms
        ligand_atoms = []
        if self.config.include_ligand_interactions and ligand_chain:
            for atom in designed_atoms:
                if atom.get("chain") == ligand_chain:
                    if ligand_resname is None or atom.get("resname") == ligand_resname:
                        ligand_atoms.append(atom)
            ligand_atoms = self._filter_atoms_by_scope(ligand_atoms, self.config.target_atoms)
            LOGGER.info(f"  Found {len(ligand_atoms)} ligand atoms in chain {ligand_chain}")

        # Find mutations and analyze their interactions
        mutations = []
        for pos, new_aa in designed_seq.items():
            orig_aa = original_seq.get(pos)
            if orig_aa is None or orig_aa == new_aa:
                continue

            chain, resno = pos
            resname_3 = self.AA_1_TO_3.get(new_aa, "UNK")

            # Get atoms of the mutated residue
            mut_atoms = get_residue_atoms(designed_atoms, chain, resno)
            mut_atoms = self._filter_atoms_by_scope(mut_atoms, self.config.mutator_atoms)
            if not mut_atoms:
                continue

            # Initialize interaction lists
            catres_interactions: Dict[InteractionType, List[Dict]] = {
                itype: [] for itype in InteractionType
            }
            ligand_interactions: Dict[InteractionType, List[Dict]] = {
                itype: [] for itype in InteractionType
            }

            # Analyze interactions with each catres
            for cat_pos, cat_atoms in catres_atoms_dict.items():
                if cat_pos == pos:
                    continue  # Don't check against self

                cat_resname = cat_atoms[0].get("resname", "UNK") if cat_atoms else "UNK"

                detected = self._detect_all_interactions(
                    mut_atoms, resname_3, cat_atoms, cat_resname, "residue"
                )

                # Add catres identifier to each interaction
                for itype, interaction_list in detected.items():
                    for interaction in interaction_list:
                        interaction["catres"] = f"{cat_pos[0]}{cat_pos[1]}"
                    catres_interactions[itype].extend(interaction_list)

            # Analyze interactions with ligand
            if ligand_atoms:
                detected = self._detect_all_interactions(
                    mut_atoms, resname_3, ligand_atoms, None, "ligand"
                )
                for itype, interaction_list in detected.items():
                    for interaction in interaction_list:
                        interaction["target"] = "ligand"
                    ligand_interactions[itype].extend(interaction_list)

            # Count interactions for scoring
            all_interaction_counts: Dict[InteractionType, int] = {}
            for itype in InteractionType:
                count = len(catres_interactions[itype]) + len(ligand_interactions[itype])
                if count > 0:
                    all_interaction_counts[itype] = count

            # Calculate scores and biases
            conservation_score = self._calculate_conservation_score(all_interaction_counts)
            mpnn_bias = self._calculate_mpnn_bias(all_interaction_counts)
            should_conserve = self._should_conserve(all_interaction_counts)

            # Create MutationInteraction
            mutation = MutationInteraction(
                chain=chain,
                resno=resno,
                original_aa=orig_aa,
                new_aa=new_aa,
                resname_3=resname_3,
                # Catres interactions
                hbonds_to_catres=catres_interactions[InteractionType.HBOND],
                pi_stacks_to_catres=catres_interactions[InteractionType.PI_STACK],
                hydrophobic_to_catres=catres_interactions[InteractionType.HYDROPHOBIC],
                metal_to_catres=catres_interactions[InteractionType.METAL],
                charged_to_catres=catres_interactions[InteractionType.CHARGED],
                cation_pi_to_catres=catres_interactions[InteractionType.CATION_PI],
                halogen_to_catres=catres_interactions[InteractionType.HALOGEN],
                # Ligand interactions
                hbonds_to_ligand=ligand_interactions[InteractionType.HBOND],
                pi_stacks_to_ligand=ligand_interactions[InteractionType.PI_STACK],
                hydrophobic_to_ligand=ligand_interactions[InteractionType.HYDROPHOBIC],
                metal_to_ligand=ligand_interactions[InteractionType.METAL],
                charged_to_ligand=ligand_interactions[InteractionType.CHARGED],
                cation_pi_to_ligand=ligand_interactions[InteractionType.CATION_PI],
                halogen_to_ligand=ligand_interactions[InteractionType.HALOGEN],
                # Metadata
                should_conserve=should_conserve,
                conservation_score=conservation_score,
                mpnn_bias=mpnn_bias,
                contributing_interactions={
                    itype.value: count for itype, count in all_interaction_counts.items()
                },
            )
            mutations.append(mutation)

        # Determine which to actually conserve (probabilistic)
        conserved_positions = set()
        fixed_residue_ids = []
        per_residue_bias = {}

        for mut in mutations:
            if mut.should_conserve:
                # Probabilistic conservation
                prob = self.config.conservation_probability
                if mut.hbonds_to_catres or mut.hbonds_to_ligand:
                    prob = max(prob, self.config.hbond_accept_probability)

                if random.random() < prob:
                    conserved_positions.add((mut.chain, mut.resno))
                    fixed_residue_ids.append(mut.residue_id)
                    LOGGER.info(
                        f"  Conserving {mut.mutation_string}: "
                        f"catres={mut.total_catres_interactions}, "
                        f"ligand={mut.total_ligand_interactions}"
                    )
                else:
                    # Not conserved but add bias
                    per_residue_bias[mut.residue_id] = {mut.new_aa: mut.mpnn_bias}
            elif mut.mpnn_bias > 0:
                # Add bias even if not conserving
                per_residue_bias[mut.residue_id] = {mut.new_aa: mut.mpnn_bias}

        # Summary statistics
        summary = {
            "total_mutations": len(mutations),
            "mutations_with_hbonds": sum(
                1 for m in mutations if m.hbonds_to_catres or m.hbonds_to_ligand
            ),
            "mutations_with_pi_stacks": sum(
                1 for m in mutations if m.pi_stacks_to_catres or m.pi_stacks_to_ligand
            ),
            "mutations_with_metal": sum(
                1 for m in mutations if m.metal_to_catres or m.metal_to_ligand
            ),
            "mutations_with_hydrophobic": sum(
                1 for m in mutations if m.hydrophobic_to_catres or m.hydrophobic_to_ligand
            ),
            "mutations_with_charged": sum(
                1 for m in mutations if m.charged_to_catres or m.charged_to_ligand
            ),
            "mutations_with_cation_pi": sum(
                1 for m in mutations if m.cation_pi_to_catres or m.cation_pi_to_ligand
            ),
            "mutations_with_halogen": sum(
                1 for m in mutations if m.halogen_to_catres or m.halogen_to_ligand
            ),
            "mutations_conserved": len(conserved_positions),
            "positions_biased": len(per_residue_bias),
            "interaction_types_analyzed": [t.value for t in self.config.interaction_types],
            "include_ligand": self.config.include_ligand_interactions,
        }

        LOGGER.info(
            f"Mutation analysis: {summary['total_mutations']} total, "
            f"{summary['mutations_with_hbonds']} H-bond, "
            f"{summary['mutations_with_pi_stacks']} pi-stack, "
            f"{summary['mutations_with_metal']} metal, "
            f"{summary['mutations_conserved']} conserved"
        )

        return InteractionAnalysisResult(
            mutations=mutations,
            conserved_positions=conserved_positions,
            fixed_residue_ids=fixed_residue_ids,
            per_residue_bias=per_residue_bias,
            summary=summary,
            config=self.config,
        )

    def update_fixed_residues(
        self,
        current_fixed: List[str],
        analysis_result: InteractionAnalysisResult,
    ) -> List[str]:
        """Update fixed residue list with conserved positions.

        Args:
            current_fixed: Current list of fixed residue IDs
            analysis_result: Result from analyze_mutations

        Returns:
            Updated list of fixed residue IDs
        """
        new_fixed = list(current_fixed)
        for res_id in analysis_result.fixed_residue_ids:
            if res_id not in new_fixed:
                new_fixed.append(res_id)
                LOGGER.debug(f"  Added {res_id} to fixed residues")

        return new_fixed

    def get_mpnn_bias(
        self,
        analysis_result: InteractionAnalysisResult,
    ) -> Dict[str, Dict[str, float]]:
        """Get per-residue MPNN biases from analysis result.

        Args:
            analysis_result: Result from analyze_mutations

        Returns:
            Dict mapping residue ID -> {amino_acid: bias}
        """
        return analysis_result.per_residue_bias


def analyze_structure_interactions(
    pdb_path: str,
    catres_positions: List[Tuple[str, int]],
    design_positions: List[Tuple[str, int]],
    config: Optional[InteractionConfig] = None,
    ligand_chain: Optional[str] = None,
    ligand_resname: Optional[str] = None,
) -> Dict[Tuple[str, int], Dict]:
    """Analyze all interactions in a structure (not just mutations).

    Useful for understanding the interaction network before/after design.

    Args:
        pdb_path: Path to PDB structure
        catres_positions: List of (chain, resno) for catalytic residues
        design_positions: List of (chain, resno) for designed positions
        config: Optional InteractionConfig for customization
        ligand_chain: Chain ID for ligand
        ligand_resname: Residue name of ligand

    Returns:
        Dict mapping design position -> interaction summary
    """
    if config is None:
        config = InteractionConfig()

    _, atoms = read_pdb_atoms(pdb_path)

    # Get catres atoms
    catres_atoms = []
    for chain, resno in catres_positions:
        catres_atoms.extend(get_residue_atoms(atoms, chain, resno))

    # Get ligand atoms
    ligand_atoms = []
    if config.include_ligand_interactions and ligand_chain:
        for atom in atoms:
            if atom.get("chain") == ligand_chain:
                if ligand_resname is None or atom.get("resname") == ligand_resname:
                    ligand_atoms.append(atom)

    results = {}
    for chain, resno in design_positions:
        res_atoms = get_residue_atoms(atoms, chain, resno)
        if not res_atoms:
            continue

        resname = res_atoms[0].get("resname", "UNK")

        # Analyze catres interactions
        catres_interactions = analyze_residue_interactions(
            residue_atoms=res_atoms,
            resname=resname,
            target_atoms=catres_atoms,
            target_type="residue",
        )

        # Analyze ligand interactions
        ligand_interaction_results = {}
        if ligand_atoms:
            ligand_interaction_results = analyze_residue_interactions(
                residue_atoms=res_atoms,
                resname=resname,
                target_atoms=ligand_atoms,
                target_type="ligand",
            )

        results[(chain, resno)] = {
            "resname": resname,
            "catres_interactions": catres_interactions,
            "ligand_interactions": ligand_interaction_results,
            "num_hbonds_catres": (
                len(catres_interactions.get("hbonds_as_donor", [])) +
                len(catres_interactions.get("hbonds_as_acceptor", []))
            ),
            "num_hbonds_ligand": (
                len(ligand_interaction_results.get("hbonds_as_donor", [])) +
                len(ligand_interaction_results.get("hbonds_as_acceptor", []))
            ),
            "num_pi_catres": len(catres_interactions.get("pi_interactions", [])),
            "num_pi_ligand": len(ligand_interaction_results.get("pi_interactions", [])),
            "num_hydrophobic_catres": len(catres_interactions.get("hydrophobic", [])),
            "num_hydrophobic_ligand": len(ligand_interaction_results.get("hydrophobic", [])),
            "num_metal_catres": len(catres_interactions.get("metal_coord", [])),
            "num_metal_ligand": len(ligand_interaction_results.get("metal_coord", [])),
            "num_charged_catres": len(catres_interactions.get("charged", [])),
            "num_charged_ligand": len(ligand_interaction_results.get("charged", [])),
        }

    return results


def identify_hbond_keepers(
    analysis_result: InteractionAnalysisResult,
    min_hbonds: int = 1,
    include_ligand: bool = False,
) -> List[str]:
    """Identify positions that form H-bonds and should be kept fixed.

    Similar to the H-bond keeper concept in modern_FastMPNNdesign.

    Args:
        analysis_result: Result from analyze_mutations
        min_hbonds: Minimum number of H-bonds to be considered a keeper
        include_ligand: Whether to count ligand H-bonds

    Returns:
        List of residue IDs to keep fixed
    """
    keepers = []
    for mut in analysis_result.mutations:
        hbond_count = len(mut.hbonds_to_catres)
        if include_ligand:
            hbond_count += len(mut.hbonds_to_ligand)

        if hbond_count >= min_hbonds:
            keepers.append(mut.residue_id)

    return keepers


def identify_interaction_keepers(
    analysis_result: InteractionAnalysisResult,
    interaction_types: Optional[List[InteractionType]] = None,
    min_interactions: int = 1,
    include_ligand: bool = False,
) -> List[str]:
    """Identify positions with specific interaction types that should be kept fixed.

    Generalized version of identify_hbond_keepers.

    Args:
        analysis_result: Result from analyze_mutations
        interaction_types: List of interaction types to consider (default: strong types)
        min_interactions: Minimum number of interactions to be considered a keeper
        include_ligand: Whether to count ligand interactions

    Returns:
        List of residue IDs to keep fixed
    """
    if interaction_types is None:
        interaction_types = [InteractionType.HBOND, InteractionType.PI_STACK, InteractionType.METAL]

    keepers = []
    for mut in analysis_result.mutations:
        total_count = 0

        for itype in interaction_types:
            target = "all" if include_ligand else "catres"
            total_count += len(mut.get_interactions_by_type(itype, target))

        if total_count >= min_interactions:
            keepers.append(mut.residue_id)

    return keepers


def parse_interaction_types_string(types_str: str) -> List[InteractionType]:
    """Parse a comma-separated string of interaction types.

    Args:
        types_str: Comma-separated string like "hbond,pi_stack,metal"

    Returns:
        List of InteractionType enums
    """
    if not types_str:
        return InteractionType.default_types()

    types = []
    for t in types_str.split(","):
        t = t.strip()
        if t:
            types.append(InteractionType.from_string(t))

    return types if types else InteractionType.default_types()
