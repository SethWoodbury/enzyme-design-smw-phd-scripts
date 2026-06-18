"""Sequence manipulation utilities for the upgraded_fastMPNNdesign pipeline.

This module provides functions for:
- Extracting sequences from PDB atom lists
- Calculating sequence identity and mutations
- Handling duplicate sequences
- Sequence masking for design regions
"""
import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import OrderedDict

from .constants import AA_3_TO_1, STANDARD_AA_3

LOGGER = logging.getLogger(__name__)


def get_sequence_from_atoms(atoms: List[Dict], chain: Optional[str] = None) -> str:
    """Extract protein sequence from a list of atom dictionaries.

    Extracts one amino acid per unique (chain, resno) pair, using CA atoms
    to ensure correct ordering. Only includes standard amino acids.

    Args:
        atoms: List of atom dicts from read_pdb_atoms()
        chain: If provided, only extract sequence for this chain

    Returns:
        One-letter amino acid sequence string
    """
    # Group by (chain, resno) and get resname
    residues = OrderedDict()  # Maintain order

    for atom in atoms:
        if atom["record_type"] != "ATOM":
            continue
        if chain is not None and atom["chain"] != chain:
            continue

        key = (atom["chain"], atom["resno"])
        if key not in residues:
            resname = atom["resname"]
            # Handle non-standard names
            if resname in AA_3_TO_1:
                residues[key] = AA_3_TO_1[resname]
            elif resname in STANDARD_AA_3:
                residues[key] = AA_3_TO_1.get(resname, "X")
            # Skip non-standard residues (ligands, etc.)

    # Sort by chain then resno
    sorted_keys = sorted(residues.keys(), key=lambda x: (x[0], x[1]))
    sequence = "".join(residues[k] for k in sorted_keys)

    return sequence


def get_sequence_with_positions(atoms: List[Dict], chain: Optional[str] = None) -> Dict[Tuple[str, int], str]:
    """Extract protein sequence with position mapping.

    Args:
        atoms: List of atom dicts from read_pdb_atoms()
        chain: If provided, only extract sequence for this chain

    Returns:
        Dict mapping (chain, resno) -> 1-letter amino acid
    """
    residues = OrderedDict()

    for atom in atoms:
        if atom["record_type"] != "ATOM":
            continue
        if chain is not None and atom["chain"] != chain:
            continue

        key = (atom["chain"], atom["resno"])
        if key not in residues:
            resname = atom["resname"]
            if resname in AA_3_TO_1:
                residues[key] = AA_3_TO_1[resname]
            elif resname in STANDARD_AA_3:
                residues[key] = AA_3_TO_1.get(resname, "X")

    return residues


def calculate_sequence_identity(seq1: str, seq2: str) -> float:
    """Calculate sequence identity between two sequences.

    Args:
        seq1: First sequence (1-letter codes)
        seq2: Second sequence (1-letter codes)

    Returns:
        Fraction of identical positions (0.0 to 1.0)
    """
    if len(seq1) != len(seq2):
        LOGGER.warning(f"Sequence length mismatch: {len(seq1)} vs {len(seq2)}")
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]

    if len(seq1) == 0:
        return 0.0

    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1)


def calculate_sequence_identity_from_maps(
    seq_map_a: Dict[Tuple[str, int], str],
    seq_map_b: Dict[Tuple[str, int], str],
) -> float:
    """Calculate sequence identity from (chain, resno) -> aa maps.

    Uses the intersection of positions present in both maps.
    """
    keys = sorted(set(seq_map_a.keys()) & set(seq_map_b.keys()))
    if not keys:
        return 0.0
    matches = sum(1 for k in keys if seq_map_a[k] == seq_map_b[k])
    return matches / len(keys)


def get_mutations_list(
    original_seq: str,
    designed_seq: str,
    start_resno: int = 1,
    chain: str = "A"
) -> List[str]:
    """Get list of mutations between original and designed sequences.

    Args:
        original_seq: Original sequence (1-letter codes)
        designed_seq: Designed sequence (1-letter codes)
        start_resno: Starting residue number for mutation notation
        chain: Chain identifier for mutation notation

    Returns:
        List of mutation strings like ["A10V", "L15F", ...]
    """
    if len(original_seq) != len(designed_seq):
        LOGGER.warning(f"Sequence length mismatch: {len(original_seq)} vs {len(designed_seq)}")
        min_len = min(len(original_seq), len(designed_seq))
        original_seq = original_seq[:min_len]
        designed_seq = designed_seq[:min_len]

    mutations = []
    for i, (orig, new) in enumerate(zip(original_seq, designed_seq)):
        if orig != new:
            resno = start_resno + i
            # Format: S(A17)A means residue 17 on chain A mutated from S to A
            mutations.append(f"{orig}({chain}{resno}){new}")

    return mutations


def get_mutations_list_from_maps(
    original_map: Dict[Tuple[str, int], str],
    designed_map: Dict[Tuple[str, int], str],
) -> List[str]:
    """Get mutations list using explicit (chain, resno) positions.

    Returns strings like "S(A13)V" (oldAA + (chain + resno) + newAA).
    """
    mutations = []
    keys = sorted(set(original_map.keys()) & set(designed_map.keys()))
    for (chain, resno) in keys:
        orig = original_map[(chain, resno)]
        new = designed_map[(chain, resno)]
        if orig != new:
            # Format: S(A13)V means residue 13 on chain A mutated from S to V
            mutations.append(f"{orig}({chain}{resno}){new}")
    return mutations


def get_mutations_dict(
    original_seq: str,
    designed_seq: str,
    start_resno: int = 1,
) -> Dict[int, Tuple[str, str]]:
    """Get dictionary of mutations with position -> (original, new).

    Args:
        original_seq: Original sequence (1-letter codes)
        designed_seq: Designed sequence (1-letter codes)
        start_resno: Starting residue number

    Returns:
        Dict mapping resno -> (original_aa, new_aa)
    """
    if len(original_seq) != len(designed_seq):
        min_len = min(len(original_seq), len(designed_seq))
        original_seq = original_seq[:min_len]
        designed_seq = designed_seq[:min_len]

    mutations = {}
    for i, (orig, new) in enumerate(zip(original_seq, designed_seq)):
        if orig != new:
            resno = start_resno + i
            mutations[resno] = (orig, new)

    return mutations


def remove_duplicate_sequences(
    sequences: List[str],
    pdbs: List[str],
    metrics: Optional[List[Dict]] = None,
    keep_best_geometry: bool = True,
) -> Tuple[List[str], List[str], Optional[List[Dict]]]:
    """Remove duplicate sequences, keeping the one with best geometry.

    Args:
        sequences: List of sequences (1-letter codes)
        pdbs: List of corresponding PDB file paths
        metrics: Optional list of metrics dicts (must have same length)
        keep_best_geometry: If True and metrics provided, keep structure with
                           lowest max_bond_dev; otherwise keep first occurrence

    Returns:
        Tuple of (unique_sequences, unique_pdbs, unique_metrics or None)
    """
    if metrics is not None and len(metrics) != len(sequences):
        raise ValueError("metrics list must have same length as sequences")

    seen: Dict[str, Tuple[str, Optional[Dict], int]] = {}  # seq -> (pdb, metrics, index)

    for i, seq in enumerate(sequences):
        pdb = pdbs[i]
        met = metrics[i] if metrics is not None else None

        if seq not in seen:
            seen[seq] = (pdb, met, i)
        elif keep_best_geometry and metrics is not None:
            # Compare geometry and keep better one
            existing_met = seen[seq][1]

            def _get_geom_block(m: Optional[Dict], key: str) -> Dict:
                if not m:
                    return {}
                if "bond_geometry" in m:
                    return m.get("bond_geometry", {}).get(key, {})
                return m.get(key, {})

            def _get_max_dev(geom: Dict) -> float:
                if "unconstrained_only" in geom:
                    val = geom["unconstrained_only"].get("max")
                    if val is None:
                        val = geom["unconstrained_only"].get("max_deviation", float("inf"))
                    return val if val is not None else float("inf")
                if "all" in geom:
                    val = geom["all"].get("max")
                    if val is None:
                        val = geom["all"].get("max_deviation", float("inf"))
                    return val if val is not None else float("inf")
                if "max" in geom or "max_deviation" in geom:
                    val = geom.get("max")
                    if val is None:
                        val = geom.get("max_deviation", float("inf"))
                    return val if val is not None else float("inf")
                return float("inf")

            def get_bond_angle_pair(m: Optional[Dict]) -> Tuple[float, float]:
                if m is None:
                    return float("inf"), float("inf")
                bond_geom = _get_geom_block(m, "bond_length_geometry")
                angle_geom = _get_geom_block(m, "bond_angle_geometry")
                bond_max = _get_max_dev(bond_geom)
                angle_max = _get_max_dev(angle_geom)
                # Fall back to legacy keys if present
                bond_max = min(bond_max, m.get("max_bond_dev", m.get("bond_max", float("inf"))))
                angle_max = min(angle_max, m.get("max_angle_dev", m.get("angle_max", float("inf"))))
                return bond_max, angle_max

            existing_bond, existing_angle = get_bond_angle_pair(existing_met)
            new_bond, new_angle = get_bond_angle_pair(met)

            if (new_bond < existing_bond) or (new_bond == existing_bond and new_angle < existing_angle):
                seen[seq] = (pdb, met, i)
                LOGGER.debug(
                    f"Replacing duplicate seq (idx {seen[seq][2]} -> {i}): "
                    f"bond_dev {existing_bond:.4f} -> {new_bond:.4f}, "
                    f"angle_dev {existing_angle:.2f} -> {new_angle:.2f}"
                )

    # Extract unique entries, maintaining original order by first occurrence
    sorted_items = sorted(seen.items(), key=lambda x: x[1][2])

    unique_seqs = [item[0] for item in sorted_items]
    unique_pdbs = [item[1][0] for item in sorted_items]
    unique_metrics = [item[1][1] for item in sorted_items] if metrics is not None else None

    num_removed = len(sequences) - len(unique_seqs)
    if num_removed > 0:
        LOGGER.info(f"Removed {num_removed} duplicate sequences, {len(unique_seqs)} unique remain")

    return unique_seqs, unique_pdbs, unique_metrics


def create_fixed_residue_mask(
    sequence_length: int,
    fixed_positions: Set[int],
    design_char: str = "X",
    fixed_char: str = "-",
) -> str:
    """Create a design mask string for MPNN-style fixed/design specification.

    Args:
        sequence_length: Total length of sequence
        fixed_positions: Set of 1-indexed positions to fix (not design)
        design_char: Character for designable positions (default "X")
        fixed_char: Character for fixed positions (default "-")

    Returns:
        Mask string of length sequence_length
    """
    mask = []
    for i in range(1, sequence_length + 1):
        if i in fixed_positions:
            mask.append(fixed_char)
        else:
            mask.append(design_char)
    return "".join(mask)


def get_residue_positions_by_chain(atoms: List[Dict]) -> Dict[str, List[int]]:
    """Get list of residue numbers for each chain.

    Args:
        atoms: List of atom dicts from read_pdb_atoms()

    Returns:
        Dict mapping chain -> sorted list of residue numbers
    """
    chain_residues: Dict[str, Set[int]] = {}

    for atom in atoms:
        if atom["record_type"] != "ATOM":
            continue
        chain = atom["chain"]
        resno = atom["resno"]

        if chain not in chain_residues:
            chain_residues[chain] = set()
        chain_residues[chain].add(resno)

    return {chain: sorted(resnos) for chain, resnos in chain_residues.items()}


def format_residue_id(chain: str, resno: int) -> str:
    """Format a residue identifier for MPNN (e.g., 'A45').

    Args:
        chain: Chain identifier
        resno: Residue number

    Returns:
        Formatted residue ID string
    """
    return f"{chain}{resno}"


def parse_residue_id(res_id: str) -> Tuple[str, int]:
    """Parse a residue identifier from MPNN format (e.g., 'A45').

    Args:
        res_id: Residue ID string like 'A45'

    Returns:
        Tuple of (chain, resno)
    """
    chain = res_id[0]
    resno = int(res_id[1:])
    return chain, resno


def residue_ids_to_set(res_ids: List[str]) -> Set[Tuple[str, int]]:
    """Convert list of residue IDs to set of (chain, resno) tuples.

    Args:
        res_ids: List of residue IDs like ['A45', 'A46', 'B30']

    Returns:
        Set of (chain, resno) tuples
    """
    return {parse_residue_id(r) for r in res_ids}


def filter_mutations_by_positions(
    mutations: List[str],
    allowed_positions: Set[Tuple[str, int]],
) -> List[str]:
    """Filter mutations to only include those at allowed positions.

    Args:
        mutations: List of mutation strings like ['A10V', 'L15F']
        allowed_positions: Set of (chain, resno) tuples

    Returns:
        Filtered list of mutations
    """
    filtered = []
    for mut in mutations:
        # Parse mutation string: e.g., "VA10L" -> chain=A, resno=10
        # Format is: orig_aa + chain + resno + new_aa
        if len(mut) < 4:
            continue
        try:
            chain = mut[1]
            # Find where the digits end
            resno_str = ""
            for i, c in enumerate(mut[2:], start=2):
                if c.isdigit():
                    resno_str += c
                else:
                    break
            if resno_str:
                resno = int(resno_str)
                if (chain, resno) in allowed_positions:
                    filtered.append(mut)
        except (ValueError, IndexError):
            continue

    return filtered
