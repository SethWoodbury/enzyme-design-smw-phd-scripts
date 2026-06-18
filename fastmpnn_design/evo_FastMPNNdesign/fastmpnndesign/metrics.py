"""
Quality metrics computation for design candidates.

Computes geometry quality, sequence metrics, and scoring metrics
for evaluating and ranking design candidates.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import math

from fastmpnndesign.config import DesignCandidate, CatalyticResidue
from fastmpnndesign.constraints import ConstraintSet, CoordinateConstraint
from fastmpnndesign.ligand import Ligand, detect_ligands_from_pdb
from fastmpnndesign.utils import (
    iter_pdb_atoms, calculate_distance, get_pdb_sequence
)
from fastmpnndesign.constants import AA_3TO1, BACKBONE_ATOMS
from fastmpnndesign.logging_config import get_logger

logger = get_logger("metrics")


def calculate_sequence_distance(input_pdb: Path, output_pdb: Path, chain: str = "A") -> dict:
    """
    Calculate sequence distance between input and output PDB.

    Extracts sequences by reading ATOM records and getting residue names
    from CA atoms, then compares them position by position.

    Args:
        input_pdb: Path to the input/reference PDB file.
        output_pdb: Path to the output/designed PDB file.
        chain: Chain ID to compare (default "A").

    Returns:
        dict with:
        - n_mutations: int - number of positions that changed
        - pct_mutated: float - percentage of sequence mutated
        - mutations: list - list of "X123Y" style mutation strings
        - sequence_identity: float - percentage identical
        - input_sequence: str - sequence from input PDB
        - output_sequence: str - sequence from output PDB
        - sequence_length: int - length of the sequence
    """
    input_pdb = Path(input_pdb)
    output_pdb = Path(output_pdb)

    # Extract sequences using CA atoms
    input_residues = _extract_sequence_from_ca(input_pdb, chain)
    output_residues = _extract_sequence_from_ca(output_pdb, chain)

    # Build sequences aligned by residue number
    all_resnums = sorted(set(input_residues.keys()) | set(output_residues.keys()))

    if not all_resnums:
        return {
            'n_mutations': 0,
            'pct_mutated': 0.0,
            'mutations': [],
            'sequence_identity': 100.0,
            'input_sequence': '',
            'output_sequence': '',
            'sequence_length': 0
        }

    # Compare sequences at each position
    mutations = []
    n_identical = 0
    n_compared = 0
    input_seq_chars = []
    output_seq_chars = []

    for resnum in all_resnums:
        input_aa = input_residues.get(resnum, '-')
        output_aa = output_residues.get(resnum, '-')

        input_seq_chars.append(input_aa)
        output_seq_chars.append(output_aa)

        # Only count positions present in both
        if input_aa != '-' and output_aa != '-':
            n_compared += 1
            if input_aa == output_aa:
                n_identical += 1
            else:
                mutations.append(f"{input_aa}{resnum}{output_aa}")

    n_mutations = len(mutations)

    if n_compared > 0:
        pct_mutated = 100.0 * n_mutations / n_compared
        sequence_identity = 100.0 * n_identical / n_compared
    else:
        pct_mutated = 0.0
        sequence_identity = 100.0

    return {
        'n_mutations': n_mutations,
        'pct_mutated': round(pct_mutated, 2),
        'mutations': mutations,
        'sequence_identity': round(sequence_identity, 2),
        'input_sequence': ''.join(input_seq_chars),
        'output_sequence': ''.join(output_seq_chars),
        'sequence_length': n_compared
    }


def _extract_sequence_from_ca(pdb_path: Path, chain: str) -> Dict[int, str]:
    """
    Extract sequence from PDB by reading CA atoms from ATOM records.

    Args:
        pdb_path: Path to PDB file.
        chain: Chain ID to extract.

    Returns:
        Dictionary mapping residue number to single-letter amino acid code.
    """
    residues = {}

    for atom in iter_pdb_atoms(pdb_path):
        # Only ATOM records (not HETATM)
        if atom['record_type'] != 'ATOM':
            continue

        # Only CA atoms
        if atom['name'] != 'CA':
            continue

        # Only specified chain (handle empty chain as matching any if chain is empty)
        atom_chain = atom['chain'] or 'A'
        if chain and atom_chain != chain:
            continue

        resnum = atom['resnum']
        resname = atom['resname']

        # Convert to single letter code
        aa_1letter = AA_3TO1.get(resname, 'X')

        # Store (first occurrence wins for alt conformers)
        if resnum not in residues:
            residues[resnum] = aa_1letter

    return residues


@dataclass
class GeometryMetrics:
    """Geometry quality metrics for a structure."""
    mean_displacement: float = 0.0
    max_displacement: float = 0.0
    pct_within_0_1A: float = 0.0  # Percentage within 0.1 A tolerance
    pct_within_0_5A: float = 0.0  # Percentage within 0.5 A tolerance
    n_constrained_atoms: int = 0
    displacements: Dict[str, float] = field(default_factory=dict)
    ligand_rmsd: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mean_displacement': self.mean_displacement,
            'max_displacement': self.max_displacement,
            'pct_within_0_1A': self.pct_within_0_1A,
            'pct_within_0_5A': self.pct_within_0_5A,
            'n_constrained_atoms': self.n_constrained_atoms,
            'ligand_rmsd': self.ligand_rmsd
        }


@dataclass
class SequenceMetrics:
    """Sequence-level metrics for a design."""
    sequence: str = ""
    length: int = 0
    identity_to_native: float = 0.0
    n_mutations: int = 0
    mutations: List[str] = field(default_factory=list)
    active_site_sequence: str = ""
    active_site_identity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sequence': self.sequence,
            'length': self.length,
            'identity_to_native': self.identity_to_native,
            'n_mutations': self.n_mutations,
            'mutations': self.mutations,
            'active_site_sequence': self.active_site_sequence,
            'active_site_identity': self.active_site_identity
        }


@dataclass
class ScoringMetrics:
    """Rosetta scoring metrics."""
    total_score: float = 0.0
    constraint_score: float = 0.0
    cart_bonded_score: float = 0.0
    fa_atr: float = 0.0
    fa_rep: float = 0.0
    fa_elec: float = 0.0
    per_residue_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_score': self.total_score,
            'constraint_score': self.constraint_score,
            'cart_bonded_score': self.cart_bonded_score,
            'fa_atr': self.fa_atr,
            'fa_rep': self.fa_rep,
            'fa_elec': self.fa_elec
        }


@dataclass
class SequenceDistanceMetrics:
    """Sequence mutation distance metrics between input and output PDB."""
    n_mutations: int = 0
    pct_mutated: float = 0.0
    mutations: List[str] = field(default_factory=list)
    sequence_identity: float = 100.0
    input_sequence: str = ""
    output_sequence: str = ""
    sequence_length: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_mutations': self.n_mutations,
            'pct_mutated': self.pct_mutated,
            'mutations': self.mutations,
            'sequence_identity': self.sequence_identity,
            'input_sequence': self.input_sequence,
            'output_sequence': self.output_sequence,
            'sequence_length': self.sequence_length
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SequenceDistanceMetrics':
        """Create SequenceDistanceMetrics from dictionary."""
        return cls(
            n_mutations=data.get('n_mutations', 0),
            pct_mutated=data.get('pct_mutated', 0.0),
            mutations=data.get('mutations', []),
            sequence_identity=data.get('sequence_identity', 100.0),
            input_sequence=data.get('input_sequence', ''),
            output_sequence=data.get('output_sequence', ''),
            sequence_length=data.get('sequence_length', 0)
        )


@dataclass
class BondGeometryMetrics:
    """Bond geometry quality metrics from cart_bonded scoring."""
    cart_bonded_score: float = 0.0
    per_residue_cart_bonded: Dict[str, float] = field(default_factory=dict)
    bond_length_deviations: List[Dict[str, Any]] = field(default_factory=list)
    bond_angle_deviations: List[Dict[str, Any]] = field(default_factory=list)
    n_residues_evaluated: int = 0
    # NEW FIELDS FOR DETAILED GEOMETRY ANALYSIS
    mean_bond_length_deviation: float = 0.0
    max_bond_length_deviation: float = 0.0
    mean_bond_angle_deviation: float = 0.0
    max_bond_angle_deviation: float = 0.0
    n_critical_bonds: int = 0  # bonds with deviation > 0.1 Å
    n_critical_angles: int = 0  # angles with deviation > 10°
    per_residue_bond_length_max: Dict[str, float] = field(default_factory=dict)
    per_residue_bond_angle_max: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cart_bonded_score': self.cart_bonded_score,
            'per_residue_cart_bonded': self.per_residue_cart_bonded,
            'bond_length_deviations': self.bond_length_deviations,
            'bond_angle_deviations': self.bond_angle_deviations,
            'n_residues_evaluated': self.n_residues_evaluated,
            'mean_bond_length_deviation': self.mean_bond_length_deviation,
            'max_bond_length_deviation': self.max_bond_length_deviation,
            'mean_bond_angle_deviation': self.mean_bond_angle_deviation,
            'max_bond_angle_deviation': self.max_bond_angle_deviation,
            'n_critical_bonds': self.n_critical_bonds,
            'n_critical_angles': self.n_critical_angles,
            'per_residue_bond_length_max': self.per_residue_bond_length_max,
            'per_residue_bond_angle_max': self.per_residue_bond_angle_max
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'BondGeometryMetrics':
        """Create BondGeometryMetrics from dictionary."""
        return cls(
            cart_bonded_score=data.get('cart_bonded_score', 0.0),
            per_residue_cart_bonded=data.get('per_residue_cart_bonded', {}),
            bond_length_deviations=data.get('bond_length_deviations', []),
            bond_angle_deviations=data.get('bond_angle_deviations', []),
            n_residues_evaluated=data.get('n_residues_evaluated', 0),
            mean_bond_length_deviation=data.get('mean_bond_length_deviation', 0.0),
            max_bond_length_deviation=data.get('max_bond_length_deviation', 0.0),
            mean_bond_angle_deviation=data.get('mean_bond_angle_deviation', 0.0),
            max_bond_angle_deviation=data.get('max_bond_angle_deviation', 0.0),
            n_critical_bonds=data.get('n_critical_bonds', 0),
            n_critical_angles=data.get('n_critical_angles', 0),
            per_residue_bond_length_max=data.get('per_residue_bond_length_max', {}),
            per_residue_bond_angle_max=data.get('per_residue_bond_angle_max', {})
        )


@dataclass
class RotamerMetrics:
    """Dunbrack rotamer quality metrics."""
    per_residue_rotamer: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    mean_rotamer_prob: Optional[float] = None
    n_residues_evaluated: int = 0
    n_favorable_rotamers: int = 0  # Rotamers with prob > 0.01
    n_unfavorable_rotamers: int = 0  # Rotamers with prob < 0.01

    def to_dict(self) -> Dict[str, Any]:
        return {
            'per_residue_rotamer': self.per_residue_rotamer,
            'mean_rotamer_prob': self.mean_rotamer_prob,
            'n_residues_evaluated': self.n_residues_evaluated,
            'n_favorable_rotamers': self.n_favorable_rotamers,
            'n_unfavorable_rotamers': self.n_unfavorable_rotamers
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RotamerMetrics':
        """Create RotamerMetrics from dictionary."""
        return cls(
            per_residue_rotamer=data.get('per_residue_rotamer', {}),
            mean_rotamer_prob=data.get('mean_rotamer_prob'),
            n_residues_evaluated=data.get('n_residues_evaluated', 0),
            n_favorable_rotamers=data.get('n_favorable_rotamers', 0),
            n_unfavorable_rotamers=data.get('n_unfavorable_rotamers', 0)
        )


@dataclass
class CatresRMSDMetrics:
    """Catalytic residue RMSD metrics between output and reference structures."""
    catres_all_atom_rmsd: Optional[float] = None
    catres_sidechain_rmsd: Optional[float] = None
    catres_backbone_rmsd: Optional[float] = None
    per_residue_rmsd: Dict[str, Dict[str, float]] = field(default_factory=dict)
    n_residues_compared: int = 0
    n_atoms_compared: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'catres_all_atom_rmsd': self.catres_all_atom_rmsd,
            'catres_sidechain_rmsd': self.catres_sidechain_rmsd,
            'catres_backbone_rmsd': self.catres_backbone_rmsd,
            'per_residue_rmsd': self.per_residue_rmsd,
            'n_residues_compared': self.n_residues_compared,
            'n_atoms_compared': self.n_atoms_compared
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CatresRMSDMetrics':
        """Create CatresRMSDMetrics from dictionary."""
        return cls(
            catres_all_atom_rmsd=data.get('catres_all_atom_rmsd'),
            catres_sidechain_rmsd=data.get('catres_sidechain_rmsd'),
            catres_backbone_rmsd=data.get('catres_backbone_rmsd'),
            per_residue_rmsd=data.get('per_residue_rmsd', {}),
            n_residues_compared=data.get('n_residues_compared', 0),
            n_atoms_compared=data.get('n_atoms_compared', 0)
        )


@dataclass
class CandidateMetrics:
    """Complete metrics for a design candidate."""
    pdb_path: Path
    geometry: GeometryMetrics
    sequence: SequenceMetrics
    scoring: ScoringMetrics
    sequence_distance: Optional[SequenceDistanceMetrics] = None
    bond_geometry: Optional[BondGeometryMetrics] = None
    rotamer_quality: Optional[RotamerMetrics] = None
    catres_rmsd: Optional[CatresRMSDMetrics] = None
    mpnn_score: Optional[float] = None
    cycle: int = 0
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'pdb_path': str(self.pdb_path),
            'geometry': self.geometry.to_dict(),
            'sequence': self.sequence.to_dict(),
            'scoring': self.scoring.to_dict(),
            'mpnn_score': self.mpnn_score,
            'cycle': self.cycle,
            'rank': self.rank
        }
        if self.sequence_distance is not None:
            result['sequence_distance'] = self.sequence_distance.to_dict()
        if self.bond_geometry is not None:
            result['bond_geometry'] = self.bond_geometry.to_dict()
        if self.rotamer_quality is not None:
            result['rotamer_quality'] = self.rotamer_quality.to_dict()
        if self.catres_rmsd is not None:
            result['catres_rmsd'] = self.catres_rmsd.to_dict()
        return result


def compute_geometry_metrics(
    pdb_path: Path,
    constraint_set: ConstraintSet,
    reference_pdb: Optional[Path] = None
) -> GeometryMetrics:
    """
    Compute geometry quality metrics by measuring displacement from constraints.

    Args:
        pdb_path: Path to PDB file to evaluate.
        constraint_set: Constraints with target coordinates.
        reference_pdb: Optional reference PDB for ligand RMSD.

    Returns:
        GeometryMetrics object.
    """
    pdb_path = Path(pdb_path)

    # Build lookup of current atom coordinates
    atom_coords = {}
    for atom in iter_pdb_atoms(pdb_path):
        key = f"{atom['chain']}_{atom['resnum']}_{atom['name']}"
        atom_coords[key] = (atom['x'], atom['y'], atom['z'])

    # Compute displacements from constraint targets
    displacements = {}
    for cst in constraint_set.coordinate_constraints:
        key = f"{cst.chain}_{cst.resnum}_{cst.atom_name}"
        if key in atom_coords:
            x, y, z = atom_coords[key]
            dist = calculate_distance(x, y, z, cst.x, cst.y, cst.z)
            displacements[key] = dist

    if not displacements:
        return GeometryMetrics()

    disp_values = list(displacements.values())
    mean_disp = sum(disp_values) / len(disp_values)
    max_disp = max(disp_values)

    # Count atoms within tolerances
    n_within_0_1 = sum(1 for d in disp_values if d <= 0.1)
    n_within_0_5 = sum(1 for d in disp_values if d <= 0.5)

    pct_0_1 = 100.0 * n_within_0_1 / len(disp_values)
    pct_0_5 = 100.0 * n_within_0_5 / len(disp_values)

    # Compute ligand RMSD if reference provided
    ligand_rmsd = None
    if reference_pdb and reference_pdb.exists():
        ligand_rmsd = compute_ligand_rmsd(pdb_path, reference_pdb)

    return GeometryMetrics(
        mean_displacement=mean_disp,
        max_displacement=max_disp,
        pct_within_0_1A=pct_0_1,
        pct_within_0_5A=pct_0_5,
        n_constrained_atoms=len(displacements),
        displacements=displacements,
        ligand_rmsd=ligand_rmsd
    )


def compute_ligand_rmsd(
    pdb_path: Path,
    reference_pdb: Path
) -> Optional[float]:
    """
    Compute RMSD of ligand heavy atoms between two structures.

    Args:
        pdb_path: Path to target PDB.
        reference_pdb: Path to reference PDB.

    Returns:
        RMSD in Angstroms or None if computation fails.
    """
    try:
        # Get ligand atoms from both structures
        target_ligands = detect_ligands_from_pdb(pdb_path)
        ref_ligands = detect_ligands_from_pdb(reference_pdb)

        if not target_ligands or not ref_ligands:
            return None

        # Match by residue name
        total_sq_dist = 0.0
        n_atoms = 0

        for ref_lig in ref_ligands:
            # Find matching ligand in target
            target_lig = None
            for tl in target_ligands:
                if tl.resname == ref_lig.resname:
                    target_lig = tl
                    break

            if not target_lig:
                continue

            # Match atoms by name
            ref_atoms = {a.name: a for a in ref_lig.heavy_atoms}
            for target_atom in target_lig.heavy_atoms:
                if target_atom.name in ref_atoms:
                    ref_atom = ref_atoms[target_atom.name]
                    sq_dist = (
                        (target_atom.x - ref_atom.x)**2 +
                        (target_atom.y - ref_atom.y)**2 +
                        (target_atom.z - ref_atom.z)**2
                    )
                    total_sq_dist += sq_dist
                    n_atoms += 1

        if n_atoms == 0:
            return None

        return (total_sq_dist / n_atoms) ** 0.5

    except Exception as e:
        logger.warning(f"Could not compute ligand RMSD: {e}")
        return None


def compute_sequence_metrics(
    pdb_path: Path,
    native_pdb: Optional[Path] = None,
    active_site_residues: Optional[List[Tuple[str, int]]] = None
) -> SequenceMetrics:
    """
    Compute sequence-level metrics.

    Args:
        pdb_path: Path to design PDB.
        native_pdb: Path to native/reference PDB for identity calculation.
        active_site_residues: List of (chain, resnum) defining active site.

    Returns:
        SequenceMetrics object.
    """
    pdb_path = Path(pdb_path)

    # Get sequence from PDB
    sequences = get_pdb_sequence(pdb_path)
    if not sequences:
        return SequenceMetrics()

    # Concatenate all chains
    full_sequence = ''.join(sequences.values())

    metrics = SequenceMetrics(
        sequence=full_sequence,
        length=len(full_sequence)
    )

    # Compute identity to native if provided
    if native_pdb and native_pdb.exists():
        native_seqs = get_pdb_sequence(native_pdb)
        native_full = ''.join(native_seqs.values())

        if len(full_sequence) == len(native_full):
            n_identical = sum(
                1 for a, b in zip(full_sequence, native_full) if a == b
            )
            metrics.identity_to_native = 100.0 * n_identical / len(full_sequence)
            metrics.n_mutations = len(full_sequence) - n_identical

            # Get mutations
            mutations = []
            for i, (native_aa, design_aa) in enumerate(zip(native_full, full_sequence)):
                if native_aa != design_aa:
                    mutations.append(f"{native_aa}{i+1}{design_aa}")
            metrics.mutations = mutations

    # Active site sequence if residues specified
    if active_site_residues:
        as_seq = extract_active_site_sequence(pdb_path, active_site_residues)
        metrics.active_site_sequence = as_seq

        if native_pdb and native_pdb.exists():
            native_as_seq = extract_active_site_sequence(native_pdb, active_site_residues)
            if as_seq and native_as_seq and len(as_seq) == len(native_as_seq):
                n_id = sum(1 for a, b in zip(as_seq, native_as_seq) if a == b)
                metrics.active_site_identity = 100.0 * n_id / len(as_seq)

    return metrics


def extract_active_site_sequence(
    pdb_path: Path,
    residues: List[Tuple[str, int]]
) -> str:
    """
    Extract sequence for specific residues from PDB.

    Args:
        pdb_path: Path to PDB file.
        residues: List of (chain, resnum) tuples.

    Returns:
        Sequence string for specified residues.
    """
    residue_set = set(residues)
    residue_aa = {}

    for atom in iter_pdb_atoms(pdb_path):
        if atom['record_type'] != 'ATOM':
            continue

        key = (atom['chain'], atom['resnum'])
        if key in residue_set and key not in residue_aa:
            aa = AA_3TO1.get(atom['resname'], 'X')
            residue_aa[key] = aa

    # Build sequence in order of input residues
    seq = []
    for res in residues:
        seq.append(residue_aa.get(res, 'X'))

    return ''.join(seq)


def compute_scoring_metrics(
    pdb_path: Path,
    params_files: Optional[List[Path]] = None
) -> ScoringMetrics:
    """
    Compute Rosetta scoring metrics using PyRosetta.

    Args:
        pdb_path: Path to PDB file.
        params_files: Ligand params files.

    Returns:
        ScoringMetrics object.
    """
    try:
        from fastmpnndesign.relax_runner import init_pyrosetta
        import pyrosetta
        from pyrosetta.rosetta.core.scoring import ScoreType
    except ImportError:
        logger.warning("PyRosetta not available, returning empty scoring metrics")
        return ScoringMetrics()

    # Initialize PyRosetta
    init_pyrosetta(params_files, quiet=True)

    try:
        pose = pyrosetta.pose_from_pdb(str(pdb_path))
        sfxn = pyrosetta.create_score_function("beta_jan25")
        total_score = sfxn(pose)

        energies = pose.energies().total_energies()

        return ScoringMetrics(
            total_score=total_score,
            fa_atr=energies[ScoreType.fa_atr] if ScoreType.fa_atr else 0.0,
            fa_rep=energies[ScoreType.fa_rep] if ScoreType.fa_rep else 0.0,
            fa_elec=energies[ScoreType.fa_elec] if ScoreType.fa_elec else 0.0
        )

    except Exception as e:
        logger.warning(f"Could not compute scoring metrics: {e}")
        return ScoringMetrics()


def compute_bond_length_deviations(
    pdb_path: Path,
    residue_list: List[Tuple[str, int]],
    params_files: Optional[List[Path]] = None
) -> Dict[str, Any]:
    """
    Compare actual bond lengths to ideal values from residue parameters.

    Evaluates bond lengths only for heavy atoms in the specified residues.

    Args:
        pdb_path: Path to PDB file.
        residue_list: List of (chain, resnum) tuples to evaluate.
        params_files: Optional ligand params files.

    Returns:
        Dict with:
        - per_residue_max_deviation: Dict[str, float] - max deviation per residue
        - per_residue_mean_deviation: Dict[str, float] - mean deviation per residue
        - worst_bonds: List[Dict] - bonds with deviation > 0.02 Å
        - mean_deviation: float - overall mean
        - max_deviation: float - overall max
        - n_bonds_evaluated: int
        - critical_bonds: int - bonds with deviation > 0.1 Å
    """
    try:
        from fastmpnndesign.relax_runner import init_pyrosetta
        import pyrosetta
    except ImportError:
        logger.warning("PyRosetta not available, returning empty bond length deviations")
        return {
            'per_residue_max_deviation': {},
            'per_residue_mean_deviation': {},
            'worst_bonds': [],
            'mean_deviation': 0.0,
            'max_deviation': 0.0,
            'n_bonds_evaluated': 0,
            'critical_bonds': 0
        }

    pdb_path = Path(pdb_path)
    residue_set = set(residue_list)

    if not residue_set:
        return {
            'per_residue_max_deviation': {},
            'per_residue_mean_deviation': {},
            'worst_bonds': [],
            'mean_deviation': 0.0,
            'max_deviation': 0.0,
            'n_bonds_evaluated': 0,
            'critical_bonds': 0
        }

    # Initialize PyRosetta
    init_pyrosetta(params_files, quiet=True)

    try:
        pose = pyrosetta.pose_from_pdb(str(pdb_path))
        pdb_info = pose.pdb_info()

        all_deviations = []
        per_residue_deviations = {}
        worst_bonds = []
        n_critical = 0

        for res_i in range(1, pose.total_residue() + 1):
            chain = pdb_info.chain(res_i)
            resnum = pdb_info.number(res_i)
            key = (chain, resnum)

            if key not in residue_set:
                continue

            residue = pose.residue(res_i)
            resname = residue.name3()
            res_key = f"{chain}_{resnum}"

            residue_deviations = []

            # Analyze bond lengths for this residue
            for bond_idx in range(1, residue.natoms() + 1):
                atom_name = residue.atom_name(bond_idx).strip()
                xyz = residue.xyz(bond_idx)

                # Skip hydrogens
                atom_type = residue.atom_type(bond_idx).element()
                if str(atom_type) == 'H':
                    continue

                # Get bonded atoms
                try:
                    bonded_atoms = residue.bonded_neighbor(bond_idx)
                except Exception:
                    continue

                for bonded_idx in bonded_atoms:
                    if bonded_idx > bond_idx:  # Avoid double counting
                        bonded_name = residue.atom_name(bonded_idx).strip()
                        bonded_xyz = residue.xyz(bonded_idx)

                        # Skip if bonded atom is hydrogen
                        bonded_type = residue.atom_type(bonded_idx).element()
                        if str(bonded_type) == 'H':
                            continue

                        # Calculate actual bond length
                        actual_length = math.sqrt(
                            (xyz.x - bonded_xyz.x)**2 +
                            (xyz.y - bonded_xyz.y)**2 +
                            (xyz.z - bonded_xyz.z)**2
                        )

                        # Get ideal bond length from params
                        try:
                            ideal_length = residue.type().bond_length(bond_idx, bonded_idx)
                            deviation = abs(actual_length - ideal_length)

                            bond_info = {
                                'atom1': atom_name,
                                'atom2': bonded_name,
                                'ideal': round(ideal_length, 4),
                                'actual': round(actual_length, 4),
                                'deviation': round(deviation, 4),
                                'chain': chain,
                                'resnum': resnum,
                                'resname': resname
                            }

                            all_deviations.append(deviation)
                            residue_deviations.append(deviation)

                            # Record significant deviations
                            if deviation > 0.02:
                                worst_bonds.append(bond_info)

                            # Count critical deviations (> 0.1 Å)
                            if deviation > 0.1:
                                n_critical += 1

                        except Exception:
                            pass  # Some bonds may not have ideal values defined

            # Per-residue statistics
            if residue_deviations:
                per_residue_deviations[res_key] = {
                    'max': round(max(residue_deviations), 4),
                    'mean': round(sum(residue_deviations) / len(residue_deviations), 4),
                    'n_bonds': len(residue_deviations)
                }

        # Sort worst bonds by deviation
        worst_bonds.sort(key=lambda x: x['deviation'], reverse=True)

        # Overall statistics
        mean_dev = 0.0
        max_dev = 0.0
        if all_deviations:
            mean_dev = sum(all_deviations) / len(all_deviations)
            max_dev = max(all_deviations)

        return {
            'per_residue_max_deviation': {k: v['max'] for k, v in per_residue_deviations.items()},
            'per_residue_mean_deviation': {k: v['mean'] for k, v in per_residue_deviations.items()},
            'worst_bonds': worst_bonds,
            'mean_deviation': round(mean_dev, 4),
            'max_deviation': round(max_dev, 4),
            'n_bonds_evaluated': len(all_deviations),
            'critical_bonds': n_critical
        }

    except Exception as e:
        logger.warning(f"Could not compute bond length deviations: {e}")
        return {
            'per_residue_max_deviation': {},
            'per_residue_mean_deviation': {},
            'worst_bonds': [],
            'mean_deviation': 0.0,
            'max_deviation': 0.0,
            'n_bonds_evaluated': 0,
            'critical_bonds': 0
        }


def compute_bond_angle_deviations(
    pdb_path: Path,
    residue_list: List[Tuple[str, int]],
    params_files: Optional[List[Path]] = None
) -> Dict[str, Any]:
    """
    Compare actual bond angles to ideal values from residue parameters.

    Evaluates angles only for heavy atoms in the specified residues.

    Args:
        pdb_path: Path to PDB file.
        residue_list: List of (chain, resnum) tuples to evaluate.
        params_files: Optional ligand params files.

    Returns:
        Dict with:
        - per_residue_max_deviation: Dict[str, float] - max deviation per residue (degrees)
        - per_residue_mean_deviation: Dict[str, float]
        - worst_angles: List[Dict] - angles with deviation > 3°
        - mean_deviation: float - overall mean
        - max_deviation: float - overall max
        - n_angles_evaluated: int
        - critical_angles: int - angles with deviation > 10°
    """
    try:
        from fastmpnndesign.relax_runner import init_pyrosetta
        import pyrosetta
    except ImportError:
        logger.warning("PyRosetta not available, returning empty bond angle deviations")
        return {
            'per_residue_max_deviation': {},
            'per_residue_mean_deviation': {},
            'worst_angles': [],
            'mean_deviation': 0.0,
            'max_deviation': 0.0,
            'n_angles_evaluated': 0,
            'critical_angles': 0
        }

    pdb_path = Path(pdb_path)
    residue_set = set(residue_list)

    if not residue_set:
        return {
            'per_residue_max_deviation': {},
            'per_residue_mean_deviation': {},
            'worst_angles': [],
            'mean_deviation': 0.0,
            'max_deviation': 0.0,
            'n_angles_evaluated': 0,
            'critical_angles': 0
        }

    # Initialize PyRosetta
    init_pyrosetta(params_files, quiet=True)

    try:
        pose = pyrosetta.pose_from_pdb(str(pdb_path))
        pdb_info = pose.pdb_info()

        all_deviations = []
        per_residue_deviations = {}
        worst_angles = []
        n_critical = 0

        for res_i in range(1, pose.total_residue() + 1):
            chain = pdb_info.chain(res_i)
            resnum = pdb_info.number(res_i)
            key = (chain, resnum)

            if key not in residue_set:
                continue

            residue = pose.residue(res_i)
            resname = residue.name3()
            res_key = f"{chain}_{resnum}"

            residue_deviations = []

            # Analyze bond angles for this residue
            for atom_j in range(1, residue.natoms() + 1):
                atom_j_name = residue.atom_name(atom_j).strip()
                xyz_j = residue.xyz(atom_j)

                # Skip hydrogens
                atom_j_type = residue.atom_type(atom_j).element()
                if str(atom_j_type) == 'H':
                    continue

                # Get bonded atoms (which can be central atom in angle)
                try:
                    bonded_to_j = residue.bonded_neighbor(atom_j)
                except Exception:
                    continue

                # For each pair of atoms bonded to j, compute angle
                bonded_list = list(bonded_to_j)
                for i in range(len(bonded_list)):
                    atom_i = bonded_list[i]
                    for k in range(i + 1, len(bonded_list)):
                        atom_k = bonded_list[k]

                        # Skip hydrogens
                        atom_i_type = residue.atom_type(atom_i).element()
                        atom_k_type = residue.atom_type(atom_k).element()
                        if str(atom_i_type) == 'H' or str(atom_k_type) == 'H':
                            continue

                        atom_i_name = residue.atom_name(atom_i).strip()
                        atom_k_name = residue.atom_name(atom_k).strip()

                        xyz_i = residue.xyz(atom_i)
                        xyz_k = residue.xyz(atom_k)

                        # Calculate actual bond angle
                        # Vector from j to i
                        v1 = (xyz_i.x - xyz_j.x, xyz_i.y - xyz_j.y, xyz_i.z - xyz_j.z)
                        # Vector from j to k
                        v2 = (xyz_k.x - xyz_j.x, xyz_k.y - xyz_j.y, xyz_k.z - xyz_j.z)

                        # Dot product and magnitudes
                        dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
                        mag1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
                        mag2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)

                        if mag1 > 0 and mag2 > 0:
                            cos_angle = dot / (mag1 * mag2)
                            # Clamp to [-1, 1] to avoid numerical errors
                            cos_angle = max(-1.0, min(1.0, cos_angle))
                            actual_angle = math.degrees(math.acos(cos_angle))

                            # Get ideal bond angle from params
                            try:
                                ideal_angle = residue.type().bond_angle(atom_i, atom_j, atom_k)
                                deviation = abs(actual_angle - ideal_angle)

                                angle_info = {
                                    'atom1': atom_i_name,
                                    'atom2': atom_j_name,
                                    'atom3': atom_k_name,
                                    'ideal': round(ideal_angle, 2),
                                    'actual': round(actual_angle, 2),
                                    'deviation': round(deviation, 2),
                                    'chain': chain,
                                    'resnum': resnum,
                                    'resname': resname
                                }

                                all_deviations.append(deviation)
                                residue_deviations.append(deviation)

                                # Record significant deviations
                                if deviation > 3.0:
                                    worst_angles.append(angle_info)

                                # Count critical deviations (> 10°)
                                if deviation > 10.0:
                                    n_critical += 1

                            except Exception:
                                pass  # Some angles may not have ideal values defined

            # Per-residue statistics
            if residue_deviations:
                per_residue_deviations[res_key] = {
                    'max': round(max(residue_deviations), 2),
                    'mean': round(sum(residue_deviations) / len(residue_deviations), 2),
                    'n_angles': len(residue_deviations)
                }

        # Sort worst angles by deviation
        worst_angles.sort(key=lambda x: x['deviation'], reverse=True)

        # Overall statistics
        mean_dev = 0.0
        max_dev = 0.0
        if all_deviations:
            mean_dev = sum(all_deviations) / len(all_deviations)
            max_dev = max(all_deviations)

        return {
            'per_residue_max_deviation': {k: v['max'] for k, v in per_residue_deviations.items()},
            'per_residue_mean_deviation': {k: v['mean'] for k, v in per_residue_deviations.items()},
            'worst_angles': worst_angles,
            'mean_deviation': round(mean_dev, 2),
            'max_deviation': round(max_dev, 2),
            'n_angles_evaluated': len(all_deviations),
            'critical_angles': n_critical
        }

    except Exception as e:
        logger.warning(f"Could not compute bond angle deviations: {e}")
        return {
            'per_residue_max_deviation': {},
            'per_residue_mean_deviation': {},
            'worst_angles': [],
            'mean_deviation': 0.0,
            'max_deviation': 0.0,
            'n_angles_evaluated': 0,
            'critical_angles': 0
        }


def get_geometry_quality_grade(metrics: BondGeometryMetrics) -> str:
    """
    Return a quality grade based on geometry metrics.

    Scoring:
    - GOOD: cart_bonded < 50, max_bond_length < 0.03 Å, max_bond_angle < 5°
    - ACCEPTABLE: cart_bonded < 200, max_bond_length < 0.06 Å, max_bond_angle < 10°
    - POOR: cart_bonded < 500, max_bond_length < 0.1 Å, max_bond_angle < 20°
    - CRITICAL: Any metric exceeds POOR thresholds

    Args:
        metrics: BondGeometryMetrics object.

    Returns:
        Grade string: "GOOD", "ACCEPTABLE", "POOR", or "CRITICAL"
    """
    # Check critical conditions
    if (metrics.cart_bonded_score > 500 or
        metrics.max_bond_length_deviation > 0.1 or
        metrics.max_bond_angle_deviation > 20.0 or
        metrics.n_critical_bonds > 0 or
        metrics.n_critical_angles > 0):
        return "CRITICAL"

    # Check poor conditions
    if (metrics.cart_bonded_score > 200 or
        metrics.max_bond_length_deviation > 0.06 or
        metrics.max_bond_angle_deviation > 10.0):
        return "POOR"

    # Check acceptable conditions
    if (metrics.cart_bonded_score > 50 or
        metrics.max_bond_length_deviation > 0.03 or
        metrics.max_bond_angle_deviation > 5.0):
        return "ACCEPTABLE"

    # Otherwise good
    return "GOOD"


def compute_geometry_quality(
    pdb_path: Path,
    residues: List[Tuple[str, int]],
    params_files: Optional[List[Path]] = None
) -> BondGeometryMetrics:
    """
    Compute geometry quality metrics for specified residues.

    Uses PyRosetta's cart_bonded scoring to evaluate bond length and angle
    deviations from ideal geometry. Combines cart_bonded scoring with detailed
    bond length and angle deviation analysis.

    Args:
        pdb_path: Path to PDB file.
        residues: List of (chain, resnum) tuples to evaluate.
        params_files: Ligand params files.

    Returns:
        BondGeometryMetrics with complete geometry analysis including:
        - cart_bonded_score: Total cart_bonded energy
        - per_residue_cart_bonded: Dict of per-residue scores
        - bond_length_deviations: List of deviations > 0.02 Å
        - bond_angle_deviations: List of deviations > 3°
        - mean/max bond length and angle deviations
        - counts of critical deviations
    """
    try:
        from fastmpnndesign.relax_runner import init_pyrosetta
        import pyrosetta
        from pyrosetta.rosetta.core.scoring import ScoreType
    except ImportError:
        logger.warning("PyRosetta not available, returning empty bond geometry metrics")
        return BondGeometryMetrics()

    pdb_path = Path(pdb_path)
    residue_set = set(residues)

    if not residue_set:
        return BondGeometryMetrics()

    # Initialize PyRosetta
    init_pyrosetta(params_files, quiet=True)

    try:
        pose = pyrosetta.pose_from_pdb(str(pdb_path))
        pdb_info = pose.pdb_info()

        # Create scorefunction with cart_bonded
        sfxn = pyrosetta.create_score_function("beta_jan25")
        sfxn.set_weight(ScoreType.cart_bonded, 0.5)

        # Score the pose to populate energies
        sfxn(pose)

        # Get per-residue cart_bonded scores
        per_residue_cart_bonded = {}
        total_cart_bonded = 0.0
        n_evaluated = 0

        for res_i in range(1, pose.total_residue() + 1):
            chain = pdb_info.chain(res_i)
            resnum = pdb_info.number(res_i)
            key = (chain, resnum)

            if key in residue_set:
                # Get residue energy for cart_bonded
                residue_energies = pose.energies().residue_total_energies(res_i)
                cart_score = residue_energies[ScoreType.cart_bonded]

                res_key = f"{chain}_{resnum}"
                per_residue_cart_bonded[res_key] = cart_score
                total_cart_bonded += cart_score
                n_evaluated += 1

        # Compute detailed bond length deviations
        bond_length_result = compute_bond_length_deviations(pdb_path, residues, params_files)

        # Compute detailed bond angle deviations
        bond_angle_result = compute_bond_angle_deviations(pdb_path, residues, params_files)

        # Extract per-residue maxima
        per_residue_bond_length_max = bond_length_result['per_residue_max_deviation']
        per_residue_bond_angle_max = bond_angle_result['per_residue_max_deviation']

        return BondGeometryMetrics(
            cart_bonded_score=total_cart_bonded,
            per_residue_cart_bonded=per_residue_cart_bonded,
            bond_length_deviations=bond_length_result['worst_bonds'],
            bond_angle_deviations=bond_angle_result['worst_angles'],
            n_residues_evaluated=n_evaluated,
            mean_bond_length_deviation=bond_length_result['mean_deviation'],
            max_bond_length_deviation=bond_length_result['max_deviation'],
            mean_bond_angle_deviation=bond_angle_result['mean_deviation'],
            max_bond_angle_deviation=bond_angle_result['max_deviation'],
            n_critical_bonds=bond_length_result['critical_bonds'],
            n_critical_angles=bond_angle_result['critical_angles'],
            per_residue_bond_length_max=per_residue_bond_length_max,
            per_residue_bond_angle_max=per_residue_bond_angle_max
        )

    except Exception as e:
        logger.warning(f"Could not compute geometry quality metrics: {e}")
        return BondGeometryMetrics()


def compute_rotamer_quality(
    pdb_path: Path,
    residues: List[Tuple[str, int]],
    params_files: Optional[List[Path]] = None
) -> RotamerMetrics:
    """
    Compute Dunbrack rotamer quality for specified residues.

    Uses PyRosetta's RotamerLibrary to evaluate how well the current
    chi angles match canonical rotamers.

    Args:
        pdb_path: Path to PDB file.
        residues: List of (chain, resnum) tuples to evaluate.
        params_files: Ligand params files.

    Returns:
        RotamerMetrics with:
        - per_residue_rotamer: Dict mapping (chain, resnum) to {
            'chi_angles': [chi1, chi2, ...],
            'rotamer_prob': float (probability of this rotamer),
            'nearest_rotamer': str (name of nearest ideal rotamer),
            'rotamer_deviation': float (angular deviation from nearest)
          }
        - mean_rotamer_prob: Average rotamer probability across residues
        - n_favorable_rotamers: Number with prob > 0.01
        - n_unfavorable_rotamers: Number with prob < 0.01
    """
    try:
        from fastmpnndesign.relax_runner import init_pyrosetta
        import pyrosetta
        from pyrosetta.rosetta.core.pack.dunbrack import (
            RotamerLibrary, RotamerLibraryScratchSpace
        )
        from pyrosetta.rosetta.core.scoring import ScoreType
    except ImportError:
        logger.warning("PyRosetta not available, returning empty rotamer metrics")
        return RotamerMetrics()

    pdb_path = Path(pdb_path)
    residue_set = set(residues)

    if not residue_set:
        return RotamerMetrics()

    # Initialize PyRosetta
    init_pyrosetta(params_files, quiet=True)

    try:
        pose = pyrosetta.pose_from_pdb(str(pdb_path))
        pdb_info = pose.pdb_info()

        # Get the rotamer library
        rotlib = RotamerLibrary.get_instance()
        scratch = RotamerLibraryScratchSpace()

        per_residue_rotamer = {}
        rotamer_probs = []
        n_favorable = 0
        n_unfavorable = 0
        n_evaluated = 0

        for res_i in range(1, pose.total_residue() + 1):
            chain = pdb_info.chain(res_i)
            resnum = pdb_info.number(res_i)
            key = (chain, resnum)

            if key not in residue_set:
                continue

            residue = pose.residue(res_i)

            # Skip non-protein residues
            if not residue.is_protein():
                continue

            # Skip residues without chi angles (GLY, ALA)
            n_chi = residue.nchi()
            if n_chi == 0:
                continue

            # Get current chi angles
            chi_angles = []
            for chi_idx in range(1, n_chi + 1):
                try:
                    chi_val = residue.chi(chi_idx)
                    chi_angles.append(round(chi_val, 1))
                except Exception:
                    break

            if not chi_angles:
                continue

            # Get rotamer probability
            try:
                # Get the rotamer probability from Dunbrack library
                rot_prob = rotlib.rotamer_energy(residue, scratch)
                # Convert energy to probability (Dunbrack returns -ln(prob))
                if rot_prob < 50:  # Avoid overflow
                    prob = math.exp(-rot_prob)
                else:
                    prob = 0.0
            except Exception:
                prob = None

            # Determine rotamer name based on chi angles
            rotamer_name = _get_rotamer_name(residue.name3(), chi_angles)

            res_key = f"{chain}_{resnum}"
            per_residue_rotamer[res_key] = {
                'resname': residue.name3(),
                'chi_angles': chi_angles,
                'rotamer_prob': round(prob, 4) if prob is not None else None,
                'nearest_rotamer': rotamer_name,
                'n_chi': n_chi
            }

            if prob is not None:
                rotamer_probs.append(prob)
                if prob > 0.01:
                    n_favorable += 1
                else:
                    n_unfavorable += 1

            n_evaluated += 1

        # Compute mean probability
        mean_prob = None
        if rotamer_probs:
            mean_prob = round(sum(rotamer_probs) / len(rotamer_probs), 4)

        return RotamerMetrics(
            per_residue_rotamer=per_residue_rotamer,
            mean_rotamer_prob=mean_prob,
            n_residues_evaluated=n_evaluated,
            n_favorable_rotamers=n_favorable,
            n_unfavorable_rotamers=n_unfavorable
        )

    except Exception as e:
        logger.warning(f"Could not compute rotamer quality metrics: {e}")
        return RotamerMetrics()


def _get_rotamer_name(resname: str, chi_angles: List[float]) -> str:
    """
    Get a descriptive rotamer name based on chi angles.

    Uses standard nomenclature:
    - g+ (gauche+): -120 to 0 degrees
    - t (trans): -180 to -120 or 120 to 180 degrees
    - g- (gauche-): 0 to 120 degrees

    Args:
        resname: Three-letter residue name.
        chi_angles: List of chi angles in degrees.

    Returns:
        Rotamer name string (e.g., "g+t" for chi1=g+, chi2=t).
    """
    if not chi_angles:
        return "none"

    def chi_to_name(angle: float) -> str:
        """Convert chi angle to rotamer name."""
        # Normalize to -180 to 180
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360

        if -120 <= angle < 0:
            return "g+"
        elif 0 <= angle < 120:
            return "g-"
        else:
            return "t"

    names = [chi_to_name(chi) for chi in chi_angles]
    return "".join(names)


def compute_catres_rmsd(
    output_pdb: Path,
    ref_pdb: Path,
    catres: List[CatalyticResidue]
) -> CatresRMSDMetrics:
    """
    Compute RMSD of catres between output and reference structures.

    The reference PDB should already be ligand-aligned with the output.

    Args:
        output_pdb: Path to output/designed PDB.
        ref_pdb: Path to reference PDB (already ligand-aligned).
        catres: List of CatalyticResidue objects.

    Returns:
        CatresRMSDMetrics with:
        - catres_all_atom_rmsd: float
        - catres_sidechain_rmsd: float (excluding backbone atoms)
        - catres_backbone_rmsd: float
        - per_residue_rmsd: Dict mapping (chain, resnum) to all_atom and sidechain RMSD
    """
    output_pdb = Path(output_pdb)
    ref_pdb = Path(ref_pdb)

    if not catres:
        return CatresRMSDMetrics()

    if not output_pdb.exists() or not ref_pdb.exists():
        logger.warning(f"PDB files not found for RMSD calculation")
        return CatresRMSDMetrics()

    # Build set of catres to compare
    catres_set = {(cr.chain, cr.resnum) for cr in catres}

    # Extract atom coordinates from both structures
    output_atoms = _get_residue_atoms(output_pdb, catres_set)
    ref_atoms = _get_residue_atoms(ref_pdb, catres_set)

    if not output_atoms or not ref_atoms:
        logger.warning("No atoms found for RMSD calculation")
        return CatresRMSDMetrics()

    # Calculate RMSDs
    all_atom_sq_dist = []
    sidechain_sq_dist = []
    backbone_sq_dist = []
    per_residue_data = {}
    n_atoms_total = 0

    for res_key in catres_set:
        chain, resnum = res_key
        res_str = f"{chain}_{resnum}"

        if res_str not in output_atoms or res_str not in ref_atoms:
            continue

        out_res = output_atoms[res_str]
        ref_res = ref_atoms[res_str]

        res_all_sq = []
        res_sc_sq = []
        res_bb_sq = []

        # Match atoms by name
        for atom_name, out_coord in out_res.items():
            if atom_name not in ref_res:
                continue

            ref_coord = ref_res[atom_name]
            sq_dist = (
                (out_coord[0] - ref_coord[0])**2 +
                (out_coord[1] - ref_coord[1])**2 +
                (out_coord[2] - ref_coord[2])**2
            )

            all_atom_sq_dist.append(sq_dist)
            res_all_sq.append(sq_dist)
            n_atoms_total += 1

            # Determine if backbone or sidechain
            if atom_name in BACKBONE_ATOMS:
                backbone_sq_dist.append(sq_dist)
                res_bb_sq.append(sq_dist)
            else:
                sidechain_sq_dist.append(sq_dist)
                res_sc_sq.append(sq_dist)

        # Per-residue RMSDs
        per_residue_data[res_str] = {
            'all_atom_rmsd': round(math.sqrt(sum(res_all_sq) / len(res_all_sq)), 3) if res_all_sq else None,
            'sidechain_rmsd': round(math.sqrt(sum(res_sc_sq) / len(res_sc_sq)), 3) if res_sc_sq else None,
            'backbone_rmsd': round(math.sqrt(sum(res_bb_sq) / len(res_bb_sq)), 3) if res_bb_sq else None,
            'n_atoms': len(res_all_sq)
        }

    # Calculate overall RMSDs
    all_atom_rmsd = None
    sidechain_rmsd = None
    backbone_rmsd = None

    if all_atom_sq_dist:
        all_atom_rmsd = round(math.sqrt(sum(all_atom_sq_dist) / len(all_atom_sq_dist)), 3)

    if sidechain_sq_dist:
        sidechain_rmsd = round(math.sqrt(sum(sidechain_sq_dist) / len(sidechain_sq_dist)), 3)

    if backbone_sq_dist:
        backbone_rmsd = round(math.sqrt(sum(backbone_sq_dist) / len(backbone_sq_dist)), 3)

    return CatresRMSDMetrics(
        catres_all_atom_rmsd=all_atom_rmsd,
        catres_sidechain_rmsd=sidechain_rmsd,
        catres_backbone_rmsd=backbone_rmsd,
        per_residue_rmsd=per_residue_data,
        n_residues_compared=len(per_residue_data),
        n_atoms_compared=n_atoms_total
    )


def _get_residue_atoms(
    pdb_path: Path,
    residue_set: Set[Tuple[str, int]]
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """
    Extract atom coordinates for specified residues from a PDB file.

    Args:
        pdb_path: Path to PDB file.
        residue_set: Set of (chain, resnum) tuples.

    Returns:
        Dict mapping "chain_resnum" to dict of atom_name -> (x, y, z).
    """
    residue_atoms = {}

    for atom in iter_pdb_atoms(pdb_path):
        if atom['record_type'] != 'ATOM':
            continue

        chain = atom['chain'] or 'A'
        resnum = atom['resnum']
        key = (chain, resnum)

        if key not in residue_set:
            continue

        # Skip hydrogens
        element = atom.get('element', '')
        atom_name = atom['name']
        if element == 'H' or atom_name.startswith('H') or atom_name[0].isdigit():
            continue

        res_str = f"{chain}_{resnum}"
        if res_str not in residue_atoms:
            residue_atoms[res_str] = {}

        residue_atoms[res_str][atom_name] = (atom['x'], atom['y'], atom['z'])

    return residue_atoms


def compute_all_metrics(
    pdb_path: Path,
    constraint_set: Optional[ConstraintSet] = None,
    native_pdb: Optional[Path] = None,
    catres_list: Optional[List[CatalyticResidue]] = None,
    params_files: Optional[List[Path]] = None,
    mpnn_score: Optional[float] = None,
    cycle: int = 0,
    input_pdb: Optional[Path] = None,
    chain: str = "A"
) -> CandidateMetrics:
    """
    Compute all metrics for a design candidate.

    Args:
        pdb_path: Path to design PDB.
        constraint_set: Constraints for geometry evaluation.
        native_pdb: Native PDB for sequence comparison.
        catres_list: Catalytic residues for active site definition.
        params_files: Ligand params files for scoring.
        mpnn_score: MPNN sequence score.
        cycle: Design cycle number.
        input_pdb: Input PDB for sequence distance calculation (defaults to native_pdb).
        chain: Chain ID for sequence distance calculation (default "A").

    Returns:
        CandidateMetrics object.
    """
    pdb_path = Path(pdb_path)

    # Geometry metrics
    if constraint_set:
        geometry = compute_geometry_metrics(pdb_path, constraint_set, native_pdb)
    else:
        geometry = GeometryMetrics()

    # Active site residues from catres
    active_site = None
    if catres_list:
        active_site = [(cr.chain, cr.resnum) for cr in catres_list]

    # Sequence metrics
    sequence = compute_sequence_metrics(pdb_path, native_pdb, active_site)

    # Scoring metrics (optional, requires PyRosetta)
    try:
        scoring = compute_scoring_metrics(pdb_path, params_files)
    except Exception:
        scoring = ScoringMetrics()

    # Sequence distance metrics
    sequence_distance = None
    reference_pdb = input_pdb if input_pdb else native_pdb
    if reference_pdb and Path(reference_pdb).exists():
        try:
            dist_dict = calculate_sequence_distance(reference_pdb, pdb_path, chain)
            sequence_distance = SequenceDistanceMetrics.from_dict(dist_dict)
        except Exception as e:
            logger.warning(f"Could not compute sequence distance metrics: {e}")

    # Bond geometry metrics (for catres)
    bond_geometry = None
    if active_site:
        try:
            bond_geometry = compute_geometry_quality(pdb_path, active_site, params_files)
        except Exception as e:
            logger.warning(f"Could not compute bond geometry metrics: {e}")

    # Rotamer quality metrics (for catres)
    rotamer_quality = None
    if active_site:
        try:
            rotamer_quality = compute_rotamer_quality(pdb_path, active_site, params_files)
        except Exception as e:
            logger.warning(f"Could not compute rotamer quality metrics: {e}")

    # Catres RMSD metrics
    catres_rmsd = None
    if catres_list and reference_pdb and Path(reference_pdb).exists():
        try:
            catres_rmsd = compute_catres_rmsd(pdb_path, reference_pdb, catres_list)
        except Exception as e:
            logger.warning(f"Could not compute catres RMSD metrics: {e}")

    return CandidateMetrics(
        pdb_path=pdb_path,
        geometry=geometry,
        sequence=sequence,
        scoring=scoring,
        sequence_distance=sequence_distance,
        bond_geometry=bond_geometry,
        rotamer_quality=rotamer_quality,
        catres_rmsd=catres_rmsd,
        mpnn_score=mpnn_score,
        cycle=cycle
    )


def summarize_metrics(
    candidates: List[CandidateMetrics]
) -> Dict[str, Any]:
    """
    Generate summary statistics across multiple candidates.

    Args:
        candidates: List of CandidateMetrics.

    Returns:
        Summary dictionary.
    """
    if not candidates:
        return {}

    geom_disps = [c.geometry.mean_displacement for c in candidates
                  if c.geometry.mean_displacement is not None]
    scores = [c.scoring.total_score for c in candidates
              if c.scoring.total_score != 0]
    mpnn_scores = [c.mpnn_score for c in candidates if c.mpnn_score is not None]

    # Sequence distance statistics
    n_mutations_list = [c.sequence_distance.n_mutations for c in candidates
                        if c.sequence_distance is not None]
    pct_mutated_list = [c.sequence_distance.pct_mutated for c in candidates
                        if c.sequence_distance is not None]
    seq_identity_list = [c.sequence_distance.sequence_identity for c in candidates
                         if c.sequence_distance is not None]

    result = {
        'n_candidates': len(candidates),
        'geometry': {
            'mean_displacement_avg': sum(geom_disps) / len(geom_disps) if geom_disps else None,
            'mean_displacement_min': min(geom_disps) if geom_disps else None,
            'mean_displacement_max': max(geom_disps) if geom_disps else None,
        },
        'scoring': {
            'total_score_avg': sum(scores) / len(scores) if scores else None,
            'total_score_min': min(scores) if scores else None,
            'total_score_max': max(scores) if scores else None,
        },
        'mpnn': {
            'score_avg': sum(mpnn_scores) / len(mpnn_scores) if mpnn_scores else None,
            'score_min': min(mpnn_scores) if mpnn_scores else None,
            'score_max': max(mpnn_scores) if mpnn_scores else None,
        }
    }

    # Add sequence distance summary if available
    if n_mutations_list:
        result['sequence_distance'] = {
            'n_mutations_avg': sum(n_mutations_list) / len(n_mutations_list),
            'n_mutations_min': min(n_mutations_list),
            'n_mutations_max': max(n_mutations_list),
            'pct_mutated_avg': sum(pct_mutated_list) / len(pct_mutated_list) if pct_mutated_list else None,
            'pct_mutated_min': min(pct_mutated_list) if pct_mutated_list else None,
            'pct_mutated_max': max(pct_mutated_list) if pct_mutated_list else None,
            'sequence_identity_avg': sum(seq_identity_list) / len(seq_identity_list) if seq_identity_list else None,
            'sequence_identity_min': min(seq_identity_list) if seq_identity_list else None,
            'sequence_identity_max': max(seq_identity_list) if seq_identity_list else None,
        }

    # Add bond geometry summary if available
    cart_bonded_scores = [c.bond_geometry.cart_bonded_score for c in candidates
                          if c.bond_geometry is not None]
    if cart_bonded_scores:
        result['bond_geometry'] = {
            'cart_bonded_avg': round(sum(cart_bonded_scores) / len(cart_bonded_scores), 3),
            'cart_bonded_min': round(min(cart_bonded_scores), 3),
            'cart_bonded_max': round(max(cart_bonded_scores), 3),
            'n_evaluated': len(cart_bonded_scores)
        }

    # Add rotamer quality summary if available
    rotamer_probs = [c.rotamer_quality.mean_rotamer_prob for c in candidates
                     if c.rotamer_quality is not None and c.rotamer_quality.mean_rotamer_prob is not None]
    favorable_counts = [c.rotamer_quality.n_favorable_rotamers for c in candidates
                        if c.rotamer_quality is not None]
    unfavorable_counts = [c.rotamer_quality.n_unfavorable_rotamers for c in candidates
                          if c.rotamer_quality is not None]
    if rotamer_probs:
        result['rotamer_quality'] = {
            'mean_prob_avg': round(sum(rotamer_probs) / len(rotamer_probs), 4),
            'mean_prob_min': round(min(rotamer_probs), 4),
            'mean_prob_max': round(max(rotamer_probs), 4),
            'n_favorable_avg': round(sum(favorable_counts) / len(favorable_counts), 1) if favorable_counts else None,
            'n_unfavorable_avg': round(sum(unfavorable_counts) / len(unfavorable_counts), 1) if unfavorable_counts else None,
            'n_evaluated': len(rotamer_probs)
        }

    # Add catres RMSD summary if available
    catres_all_atom_rmsds = [c.catres_rmsd.catres_all_atom_rmsd for c in candidates
                             if c.catres_rmsd is not None and c.catres_rmsd.catres_all_atom_rmsd is not None]
    catres_sidechain_rmsds = [c.catres_rmsd.catres_sidechain_rmsd for c in candidates
                               if c.catres_rmsd is not None and c.catres_rmsd.catres_sidechain_rmsd is not None]
    catres_backbone_rmsds = [c.catres_rmsd.catres_backbone_rmsd for c in candidates
                              if c.catres_rmsd is not None and c.catres_rmsd.catres_backbone_rmsd is not None]
    if catres_all_atom_rmsds:
        result['catres_rmsd'] = {
            'all_atom_rmsd_avg': round(sum(catres_all_atom_rmsds) / len(catres_all_atom_rmsds), 3),
            'all_atom_rmsd_min': round(min(catres_all_atom_rmsds), 3),
            'all_atom_rmsd_max': round(max(catres_all_atom_rmsds), 3),
            'sidechain_rmsd_avg': round(sum(catres_sidechain_rmsds) / len(catres_sidechain_rmsds), 3) if catres_sidechain_rmsds else None,
            'sidechain_rmsd_min': round(min(catres_sidechain_rmsds), 3) if catres_sidechain_rmsds else None,
            'sidechain_rmsd_max': round(max(catres_sidechain_rmsds), 3) if catres_sidechain_rmsds else None,
            'backbone_rmsd_avg': round(sum(catres_backbone_rmsds) / len(catres_backbone_rmsds), 3) if catres_backbone_rmsds else None,
            'backbone_rmsd_min': round(min(catres_backbone_rmsds), 3) if catres_backbone_rmsds else None,
            'backbone_rmsd_max': round(max(catres_backbone_rmsds), 3) if catres_backbone_rmsds else None,
            'n_evaluated': len(catres_all_atom_rmsds)
        }

    return result
