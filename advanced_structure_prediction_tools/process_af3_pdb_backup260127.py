#!/usr/bin/env python3
"""
AlphaFold3 PDB Processing Script

This script processes AlphaFold3 (AF3) structure predictions, comparing them against
a reference PDB structure to compute structural metrics including:
- RMSD (Root Mean Square Deviation) for Calpha atoms, catalytic residues, and ligands
- pLDDT (predicted Local Distance Difference Test) confidence scores
- PAE (Predicted Aligned Error) between chain pairs
- Support for non-canonical amino acids (ncAAs) and symmetric ligand atoms

Usage:
    python process_af3_pdb.py --af3_dir <path> --ref_pdb <path> --outscr <path> \\
        --ligand_groups_json '<json>' [--verbose] [--calculate_avg_pae_in_addition_to_pair]

Author: Woodbuse Lab
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONSTANTS
# ==============================================================================

from Bio import PDB
from datetime import datetime
from itertools import combinations, product
from numpy.linalg import norm
from typing import Dict, List, Tuple, Optional, Any
import argparse
import glob
import json
import numpy as np
import os
import pandas as pd
import sys

# PDB format column positions (0-indexed)
PDB_RECORD_TYPE_COLS = (0, 6)
PDB_ATOM_NAME_COLS = (12, 16)
PDB_RESIDUE_NAME_COLS = (17, 21)
PDB_CHAIN_ID_COLS = (21, 23)
PDB_RESIDUE_NUM_COLS = (23, 27)
PDB_PLDDT_COLS = (61, 67)

# Canonical backbone atom ordering for consistent RMSD comparisons
CANONICAL_BACKBONE_ORDER = ["N", "CA", "C", "O"]


# ==============================================================================
# SECTION 2: LOGGING UTILITIES
# ==============================================================================

def log(msg: str, level: str = "INFO") -> None:
    """
    Print a timestamped log message to stdout.

    Args:
        msg: The message to log
        level: Log level (INFO, WARN, ERROR, DEBUG)
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level}] {msg}", flush=True)


def vlog(verbose: bool, msg: str, level: str = "DEBUG") -> None:
    """
    Conditional verbose logging - only prints if verbose is True.

    Args:
        verbose: Whether to print the message
        msg: The message to log
        level: Log level (default: DEBUG)
    """
    if verbose:
        log(msg, level)


# ==============================================================================
# SECTION 3: PDB FILE I/O
# ==============================================================================

def load_pdb(file_path: str) -> PDB.Structure.Structure:
    """
    Load a PDB file using Biopython's PDBParser.

    Args:
        file_path: Path to the PDB file

    Returns:
        Biopython Structure object
    """
    parser = PDB.PDBParser(QUIET=True)
    structure_id = os.path.basename(file_path)
    return parser.get_structure(structure_id, file_path)


def save_pdb(structure: PDB.Structure.Structure, output_file: str) -> None:
    """
    Save a Biopython structure to a PDB file.

    Args:
        structure: Biopython Structure object
        output_file: Output file path
    """
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_file)


def rapid_skip_if_sc_present(af3_dir: str, enabled: bool) -> None:
    """
    Early exit if output score files already exist in the directory.

    This function enables a "rapid mode" where processing is skipped if
    any .sc (score) files are already present, avoiding redundant computation.

    Args:
        af3_dir: Directory containing AF3 predictions
        enabled: Whether rapid skip mode is enabled
    """
    if not enabled:
        return
    if not af3_dir or not os.path.isdir(af3_dir):
        return

    sc_hits = glob.glob(os.path.join(af3_dir, "*.sc"))
    if sc_hits:
        log(f"[RAPID] Found existing .sc file(s) in {af3_dir}:", "INFO")
        for p in sc_hits[:5]:
            print(f"        - {os.path.basename(p)}")
        if len(sc_hits) > 5:
            print(f"        ... (+{len(sc_hits) - 5} more)")
        log("[RAPID] Skipping processing and exiting cleanly (--rapid_mode_skip_sc_if_found).", "INFO")
        sys.exit(0)


# ==============================================================================
# SECTION 4: COORDINATE EXTRACTION
# ==============================================================================

def extract_ca_coords(structure: PDB.Structure.Structure) -> np.ndarray:
    """
    Extract coordinates of all Calpha (CA) atoms for structural alignment.

    Args:
        structure: Biopython Structure object

    Returns:
        NumPy array of shape (N, 3) containing CA coordinates
    """
    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_atom = residue['CA']
                    ca_coords.append(ca_atom.get_coord())
    return np.array(ca_coords)


def extract_coords_atom(
    structure: PDB.Structure.Structure,
    chain_id: str,
    residue_name: str,
    atom_name: str
) -> np.ndarray:
    """
    Extract coordinates of a specific atom within residues of a given name.

    Args:
        structure: Biopython Structure object
        chain_id: Chain identifier (e.g., 'A', 'B')
        residue_name: 3-letter residue name (e.g., 'ALA', 'YYE')
        atom_name: Atom name (e.g., 'CA', 'P1')

    Returns:
        NumPy array of matching atom coordinates
    """
    coords = []
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.get_resname() == residue_name:
                    for atom in residue:
                        if atom.get_name() == atom_name:
                            coords.append(atom.get_coord())
    return np.array(coords)


def extract_coords_res(
    structure: PDB.Structure.Structure,
    chain_id: str,
    residue_num: int
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract heavy-atom coordinates and names for a specific residue.

    Excludes hydrogen atoms for more robust RMSD comparisons.

    Args:
        structure: Biopython Structure object
        chain_id: Chain identifier
        residue_num: Residue number

    Returns:
        Tuple of (coordinates array, list of atom names)
    """
    coords, names = [], []
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.get_id()[1] == residue_num:
                    for atom in residue:
                        if atom.element == "H":
                            continue
                        coords.append(atom.get_coord())
                        names.append(atom.get_name())
    return np.array(coords), names


def get_catalytic_residues(pdb_file: str, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Parse catalytic residue information from REMARK 666 lines in a PDB file.

    REMARK 666 is a Rosetta/enzyme design convention for marking catalytic residues.
    Format: REMARK 666 MATCH ... <chain> <resname> <resnum> ...

    Args:
        pdb_file: Path to the PDB file
        verbose: Whether to print extracted residues

    Returns:
        List of dicts with keys: 'res_num', 'name3', 'chain'
    """
    catalytic_residues = []

    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                break  # REMARK lines come before ATOM records
            if "REMARK 666" in line:
                parts = line.split()
                chain = parts[9]
                residue_name = parts[10]
                residue_number = int(parts[11])
                catalytic_residues.append({
                    'res_num': residue_number,
                    'name3': residue_name,
                    'chain': chain
                })

    if verbose or catalytic_residues:
        vlog(True, f"Catalytic residues from {os.path.basename(pdb_file)}: {catalytic_residues}", "INFO")

    return catalytic_residues


# ==============================================================================
# SECTION 5: STRUCTURAL ALIGNMENT (KABSCH ALGORITHM)
# ==============================================================================

def kabsch_alignment(
    coords1: np.ndarray,
    coords2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute optimal rotation matrix to align coords1 onto coords2 using Kabsch algorithm.

    The Kabsch algorithm finds the rotation matrix that minimizes RMSD between
    two paired sets of points. It uses Singular Value Decomposition (SVD) to
    find the optimal rotation.

    Args:
        coords1: Source coordinates, shape (N, 3)
        coords2: Target coordinates, shape (N, 3)

    Returns:
        Tuple of (rotation_matrix, centroid1, centroid2)
        - rotation_matrix: 3x3 rotation matrix
        - centroid1: Center of mass of coords1
        - centroid2: Center of mass of coords2
    """
    # Center the coordinates by subtracting the mean
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)

    coords1_centered = coords1 - centroid1
    coords2_centered = coords2 - centroid2

    # Compute the covariance matrix
    H = np.dot(coords1_centered.T, coords2_centered)

    # Perform Singular Value Decomposition
    V, S, W = np.linalg.svd(H)

    # Ensure a proper rotation matrix (determinant should be +1, not -1)
    d = np.sign(np.linalg.det(V @ W))
    V[:, -1] *= d
    rotation_matrix = V @ W

    return rotation_matrix, centroid1, centroid2


def apply_rotation(
    coords: np.ndarray,
    rotation_matrix: np.ndarray,
    centroid1: np.ndarray,
    centroid2: np.ndarray
) -> np.ndarray:
    """
    Apply rotation matrix to coordinates, transforming from frame 1 to frame 2.

    Args:
        coords: Coordinates to transform, shape (N, 3)
        rotation_matrix: 3x3 rotation matrix from kabsch_alignment
        centroid1: Original centroid (subtracted before rotation)
        centroid2: Target centroid (added after rotation)

    Returns:
        Transformed coordinates
    """
    coords_centered = coords - centroid1
    coords_rotated = np.dot(coords_centered, rotation_matrix) + centroid2
    return coords_rotated


def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Calculate Root Mean Square Deviation between two coordinate sets.

    RMSD = sqrt(mean(sum((coords1 - coords2)^2)))

    Args:
        coords1: First coordinate set, shape (N, 3)
        coords2: Second coordinate set, shape (N, 3)

    Returns:
        RMSD value in Angstroms
    """
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


# ==============================================================================
# SECTION 6: NON-CANONICAL AMINO ACID (ncAA) SUPPORT
# ==============================================================================

def order_common_atoms(names1: List[str], names2: List[str]) -> List[str]:
    """
    Return a canonical ordering of atoms present in both residues.

    Prioritizes backbone atoms (N, CA, C, O) first, then remaining atoms
    alphabetically. This ensures consistent RMSD calculations when comparing
    residues with different atom compositions (e.g., KCX vs LYS).

    Args:
        names1: Atom names from structure 1
        names2: Atom names from structure 2

    Returns:
        Ordered list of common atom names
    """
    s1, s2 = set(names1), set(names2)
    inter = list(s1 & s2)
    if not inter:
        return []

    # Backbone atoms first (if present), then the rest sorted alphabetically
    back = [a for a in CANONICAL_BACKBONE_ORDER if a in inter]
    rest = sorted([a for a in inter if a not in CANONICAL_BACKBONE_ORDER])
    return back + rest


def rmsd_on_common_atoms(
    coords1: np.ndarray,
    names1: List[str],
    coords2: np.ndarray,
    names2: List[str]
) -> Tuple[float, List[str], List[str], List[str]]:
    """
    Compute RMSD using only atoms present in both structures.

    Useful for comparing ncAAs or modified residues where atom sets differ.

    Args:
        coords1: Coordinates from structure 1
        names1: Atom names from structure 1
        coords2: Coordinates from structure 2
        names2: Atom names from structure 2

    Returns:
        Tuple of:
        - rmsd_value: RMSD computed on common atoms (or NaN if no overlap)
        - af3_only: Atoms only in structure 1
        - ref_only: Atoms only in structure 2
        - common: List of common atom names used
    """
    common = order_common_atoms(names1, names2)
    s1, s2 = set(names1), set(names2)
    af3_only = sorted(list(s1 - s2))
    ref_only = sorted(list(s2 - s1))

    if not common:
        return (float("nan"), af3_only, ref_only, [])

    idx1 = [names1.index(n) for n in common]
    idx2 = [names2.index(n) for n in common]
    return (calculate_rmsd(coords1[idx1], coords2[idx2]), af3_only, ref_only, common)


def get_resname_by_chain_resnum(
    structure: PDB.Structure.Structure,
    chain_id: str,
    resnum: int
) -> Optional[str]:
    """
    Look up the 3-letter residue name at a specific position.

    Args:
        structure: Biopython Structure object
        chain_id: Chain identifier
        resnum: Residue number

    Returns:
        3-letter residue name or None if not found
    """
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.get_id()[1] == resnum:
                    return residue.get_resname()
    return None


# ==============================================================================
# SECTION 7: SYMMETRIC ATOM HANDLING
# ==============================================================================

def permute_symmetric_atoms(
    coords1: np.ndarray,
    coords2: np.ndarray,
    atom_names1: List[str],
    atom_names2: List[str],
    symmetric_groups: List[Dict]
) -> float:
    """
    Calculate minimum RMSD considering symmetric atom permutations.

    Many ligands have chemically equivalent atoms that can be permuted without
    changing the molecule's identity (e.g., the two oxygens in a carboxylate).
    This function tries all valid permutations and returns the minimum RMSD.

    Args:
        coords1: Coordinates from structure 1
        coords2: Coordinates from structure 2
        atom_names1: Atom names from structure 1
        atom_names2: Atom names from structure 2
        symmetric_groups: List of dicts defining symmetric atom groups:
            {
                "lig1": [["atom1", "atom2"], ...],  # Subgroups in ligand 1
                "lig2": [[["perm1_atom1", "perm1_atom2"], ["perm2_atom1", "perm2_atom2"]], ...]
            }

    Returns:
        Minimum RMSD across all valid permutations
    """
    name_to_idx1 = {name: i for i, name in enumerate(atom_names1)}
    name_to_idx2 = {name: i for i, name in enumerate(atom_names2)}
    min_rmsd = float("inf")

    for group in symmetric_groups:
        atoms1_subgroups = group["lig1"]
        atoms2_permutations_per_subgroup = group["lig2"]

        if len(atoms1_subgroups) != len(atoms2_permutations_per_subgroup):
            raise ValueError("Mismatch in number of lig1 subgroups and lig2 permutation sets")

        # Create all combinations of one permutation per subgroup
        subgroup_perm_combos = product(*atoms2_permutations_per_subgroup)

        for atoms2_perm_combo in subgroup_perm_combos:
            coords1_concat = []
            coords2_concat = []

            for atoms1_subgroup, atoms2_perm in zip(atoms1_subgroups, atoms2_perm_combo):
                idxs1 = [name_to_idx1[a] for a in atoms1_subgroup]
                idxs2 = [name_to_idx2[a] for a in atoms2_perm]
                coords1_concat.extend(coords1[idxs1])
                coords2_concat.extend(coords2[idxs2])

            coords1_array = np.array(coords1_concat)
            coords2_array = np.array(coords2_concat)

            if coords1_array.shape != coords2_array.shape:
                raise ValueError("Mismatch in coords shape during symmetric RMSD calculation")

            rmsd = calculate_rmsd(coords1_array, coords2_array)
            min_rmsd = min(min_rmsd, rmsd)

    return min_rmsd


# ==============================================================================
# SECTION 8: pLDDT CONFIDENCE METRICS
# ==============================================================================

def get_plddt_per_residue_dic(af3_pdb: str) -> Dict[Tuple[str, int], float]:
    """
    Parse per-residue average pLDDT from an AF3 PDB file.

    pLDDT (predicted Local Distance Difference Test) is stored in the B-factor
    column of AF3 PDB files. Values range from 0-100:
    - >90: Very high confidence
    - 70-90: High confidence
    - 50-70: Low confidence
    - <50: Very low confidence (often disordered regions)

    Args:
        af3_pdb: Path to AF3 PDB file

    Returns:
        Dict mapping (chain_id, residue_num) to average pLDDT
    """
    plddt_dic = {}

    with open(af3_pdb, 'r') as f:
        for line in f:
            record_type = line[PDB_RECORD_TYPE_COLS[0]:PDB_RECORD_TYPE_COLS[1]]
            if record_type not in ["ATOM  ", "HETATM"]:
                continue
            if len(line.strip()) < PDB_PLDDT_COLS[1]:
                continue

            ch_id = line[PDB_CHAIN_ID_COLS[0]:PDB_CHAIN_ID_COLS[1]].strip()
            res_num = int(line[PDB_RESIDUE_NUM_COLS[0]:PDB_RESIDUE_NUM_COLS[1]].strip())
            atom_name = line[PDB_ATOM_NAME_COLS[0]:PDB_ATOM_NAME_COLS[1]].strip()
            plddt = float(line[PDB_PLDDT_COLS[0]:PDB_PLDDT_COLS[1]])

            key = (ch_id, res_num)
            if key not in plddt_dic:
                plddt_dic[key] = {}
            plddt_dic[key][atom_name] = plddt

    # Average pLDDT across all atoms in each residue
    plddt_per_residue = {key: np.average(list(plddt_dic[key].values())) for key in plddt_dic}
    return plddt_per_residue


def get_avr_plddt(af3_pdb: str, ligand_groups_json: List[Dict]) -> List[float]:
    """
    Calculate average pLDDT for each ligand group.

    Args:
        af3_pdb: Path to AF3 PDB file
        ligand_groups_json: List of ligand group definitions

    Returns:
        List of average pLDDT values, one per ligand group
    """
    avr_plddt_list = []

    with open(af3_pdb, 'r') as f:
        pdb_lines = f.readlines()

    for ligand in ligand_groups_json:
        plddt_list = []
        for line in pdb_lines:
            record_type = line[PDB_RECORD_TYPE_COLS[0]:PDB_RECORD_TYPE_COLS[1]]
            if record_type not in ["ATOM  ", "HETATM"]:
                continue
            if len(line.strip()) < PDB_PLDDT_COLS[1]:
                continue

            atom = line[PDB_ATOM_NAME_COLS[0]:PDB_ATOM_NAME_COLS[1]].strip()
            name3 = line[PDB_RESIDUE_NAME_COLS[0]:PDB_RESIDUE_NAME_COLS[1]].strip()
            chain = line[PDB_CHAIN_ID_COLS[0]:PDB_CHAIN_ID_COLS[1]].strip()
            plddt = float(line[PDB_PLDDT_COLS[0]:PDB_PLDDT_COLS[1]])

            if (name3 == ligand["name3"][0] and
                atom in ligand["atoms"][0] and
                chain == ligand["chain"][0]):
                plddt_list.append(plddt)

        avr_plddt_list.append(np.mean(plddt_list) if plddt_list else float("nan"))

    return avr_plddt_list


def get_plddt_for_specific_atoms(
    pdb_path: str,
    chain_id: str,
    resnum: int,
    atom_names: List[str]
) -> float:
    """
    Get average pLDDT for specific atoms in a residue.

    Args:
        pdb_path: Path to PDB file
        chain_id: Chain identifier
        resnum: Residue number
        atom_names: List of atom names to average

    Returns:
        Average pLDDT or NaN if no atoms found
    """
    if not atom_names:
        return float("nan")

    plddts = []
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if len(line.strip()) < PDB_PLDDT_COLS[1]:
                continue

            ch = line[PDB_CHAIN_ID_COLS[0]:PDB_CHAIN_ID_COLS[1]].strip()
            try:
                rn = int(line[PDB_RESIDUE_NUM_COLS[0]:PDB_RESIDUE_NUM_COLS[1]].strip())
            except ValueError:
                continue

            if ch != chain_id or rn != resnum:
                continue

            atom = line[PDB_ATOM_NAME_COLS[0]:PDB_ATOM_NAME_COLS[1]].strip()
            if atom in atom_names:
                try:
                    plddts.append(float(line[PDB_PLDDT_COLS[0]:PDB_PLDDT_COLS[1]]))
                except ValueError:
                    pass

    return float(np.mean(plddts)) if plddts else float("nan")


# ==============================================================================
# SECTION 9: PAE (PREDICTED ALIGNED ERROR) ANALYSIS
# ==============================================================================

def compute_interchain_pae_means_from_conf(
    conf_json_path: str,
    valid_chains: Optional[List[str]] = None,
    verbose: bool = False,
    verbose_matrix: bool = False,
    label: Optional[str] = None
) -> Dict[Tuple[str, str], float]:
    """
    Compute mean interchain PAE from AF3's token-level PAE matrix.

    This function reads the NxN PAE matrix from *_confidences.json where N is
    the total number of tokens (residues/atoms) across all chains. For each
    unique chain pair (A, B), it computes the mean PAE across all token pairs
    where one token belongs to chain A and the other to chain B.

    PAE Interpretation:
    - PAE[i][j] = predicted error in position of token j when aligned using
      the frame of token i (in Angstroms)
    - Lower values indicate higher confidence in relative positioning
    - Typically: PAE < 5A suggests confident interaction
                 PAE > 20A suggests no interaction

    Algorithm:
    1. Load the NxN PAE matrix and token_chain_ids list
    2. For each unique chain pair (A, B) where A < B alphabetically:
       - Create boolean mask selecting A->B and B->A entries
       - Extract PAE values at masked positions
       - Compute mean (ignoring NaN values)

    Args:
        conf_json_path: Path to AF3's *_confidences.json file
        valid_chains: Optional list of chain IDs to include (default: all chains)
        verbose: If True, print compact PAE summaries per chain pair
        verbose_matrix: If True, print full NxN PAE matrices (very large output)
        label: Optional label for verbose output

    Returns:
        Dict mapping (chain_a, chain_b) tuples to mean PAE values,
        where chain_a < chain_b alphabetically. Empty dict on error.
    """
    try:
        with open(conf_json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}

    pae_list = data.get("pae")
    token_chains_list = data.get("token_chain_ids")
    if pae_list is None or token_chains_list is None:
        return {}

    # Build arrays (None -> NaN for safety)
    pae = np.array(
        [[(float(x) if x is not None else np.nan) for x in row] for row in pae_list],
        dtype=float,
    )
    token_chains = np.array(token_chains_list, dtype=str)

    # Validate shapes
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1] or len(token_chains) != pae.shape[0]:
        return {}

    # Determine chain IDs to consider
    chain_set = sorted(set(token_chains.tolist()))
    if valid_chains is not None:
        valid = set(map(str, valid_chains))
        chain_list = [c for c in chain_set if c in valid]
    else:
        chain_list = chain_set

    # Labels for verbose output
    labels = [f"{i}:{c}" for i, c in enumerate(token_chains)]

    out = {}
    with np.errstate(all="ignore"):
        for a, b in combinations(chain_list, 2):
            # Create boolean masks for chain pair selection
            # rows_a: True where token belongs to chain A (Nx1 for broadcasting)
            # cols_b: True where token belongs to chain B (1xN for broadcasting)
            rows_a = (token_chains == a)[:, None]  # Nx1
            cols_b = (token_chains == b)[None, :]  # 1xN
            rows_b = (token_chains == b)[:, None]
            cols_a = (token_chains == a)[None, :]

            # Mask selects both A->B and B->A entries (symmetric)
            mask = (rows_a & cols_b) | (rows_b & cols_a)
            vals = pae[mask]
            finite = np.isfinite(vals)
            count = int(np.count_nonzero(finite))
            sum_val = float(np.nansum(vals)) if vals.size > 0 else float("nan")
            mean_val = float(np.nanmean(vals)) if count > 0 else float("nan")

            out[(a, b)] = mean_val

            if verbose or verbose_matrix:
                _print_verbose_pae_analysis(
                    pae, token_chains, labels, mask, a, b,
                    sum_val, count, mean_val, label, conf_json_path,
                    print_full_matrix=verbose_matrix
                )

    return out


def _print_verbose_pae_analysis(
    pae: np.ndarray,
    token_chains: np.ndarray,
    labels: List[str],
    mask: np.ndarray,
    chain_a: str,
    chain_b: str,
    sum_val: float,
    count: int,
    mean_val: float,
    label: Optional[str],
    conf_json_path: str,
    print_full_matrix: bool = False
) -> None:
    """
    Helper function to print verbose PAE analysis output.

    Args:
        print_full_matrix: If True, prints the full NxN matrices (can be huge).
                          If False, prints only a compact summary.
    """
    if print_full_matrix:
        # Full matrix output (only when explicitly requested)
        try:
            with pd.option_context(
                "display.max_rows", None,
                "display.max_columns", None,
                "display.width", 500,
                "display.float_format", lambda x: f"{x:6.2f}"
            ):
                print("\n" + "=" * 80)
                print(f"[MATRIX] Token-level PAE  |  Pair {chain_a}-{chain_b}  |  "
                      f"{label or os.path.basename(conf_json_path)}")
                print("- token_chain_ids:")
                print(" ".join(token_chains.tolist()))

                df_pae = pd.DataFrame(pae, index=labels, columns=labels)
                print("\n- PAE matrix:")
                print(df_pae)

                df_mask = pd.DataFrame(mask.astype(int), index=labels, columns=labels)
                print("\n- Boolean mask (1==selected AB/BA):")
                print(df_mask)

                sliced = np.full_like(pae, np.nan, dtype=float)
                sliced[mask] = pae[mask]
                df_sliced = pd.DataFrame(sliced, index=labels, columns=labels).fillna("")
                print("\n- Sliced matrix (only AB/BA shown; others blank):")
                print(df_sliced)

                print(f"\n- Sum over mask: {sum_val:.6g}  |  Count: {count}  |  Mean: {mean_val:.6g}")
                print("=" * 80)
        except Exception:
            print(f"[MATRIX] Pair {chain_a}-{chain_b}: Sum={sum_val:.4f}, Count={count}, Mean={mean_val:.4f}")
    else:
        # Compact summary output (default for --verbose)
        vals = pae[mask]
        min_val = float(np.nanmin(vals)) if vals.size > 0 else float("nan")
        max_val = float(np.nanmax(vals)) if vals.size > 0 else float("nan")
        vlog(True, f"    {chain_a}<->{chain_b}: mean={mean_val:.2f}, min={min_val:.2f}, max={max_val:.2f}, n={count}", "DEBUG")


def compute_interchain_pae_min_from_conf(
    conf_json_path: str,
    valid_chains: Optional[List[str]] = None,
    verbose: bool = False,
    label: Optional[str] = None
) -> Dict[Tuple[str, str], float]:
    """
    Compute minimum interchain PAE from AF3's token-level PAE matrix.

    This function computes PAE min directly from the raw NxN PAE matrix,
    which can be used to validate AF3's pre-computed chain_pair_pae_min values.

    PAE Min Definition:
    - For chain pair (A, B): min(PAE[i][j]) for all tokens i in chain A, j in chain B
    - This is ASYMMETRIC: PAE_min(A→B) may differ from PAE_min(B→A)

    Args:
        conf_json_path: Path to AF3's *_confidences.json file
        valid_chains: Optional list of chain IDs to include (default: all chains)
        verbose: If True, print detailed PAE min calculations
        label: Optional label for verbose output

    Returns:
        Dict mapping (chain_a, chain_b) tuples to min PAE values.
        Note: Returns BOTH directions, so (A,B) and (B,A) are separate keys.
        Empty dict on error.
    """
    try:
        with open(conf_json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}

    pae_list = data.get("pae")
    token_chains_list = data.get("token_chain_ids")
    if pae_list is None or token_chains_list is None:
        return {}

    # Build arrays (None -> NaN for safety)
    pae = np.array(
        [[(float(x) if x is not None else np.nan) for x in row] for row in pae_list],
        dtype=float,
    )
    token_chains = np.array(token_chains_list, dtype=str)

    # Validate shapes
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1] or len(token_chains) != pae.shape[0]:
        return {}

    # Determine chain IDs to consider
    chain_set = sorted(set(token_chains.tolist()))
    if valid_chains is not None:
        valid = set(map(str, valid_chains))
        chain_list = [c for c in chain_set if c in valid]
    else:
        chain_list = chain_set

    out = {}
    with np.errstate(all="ignore"):
        # Compute for ALL ordered pairs (including A→A diagonal)
        for a in chain_list:
            for b in chain_list:
                # Create mask for A→B: rows from chain A, columns from chain B
                rows_a = (token_chains == a)[:, None]  # Nx1
                cols_b = (token_chains == b)[None, :]  # 1xN
                mask = rows_a & cols_b

                vals = pae[mask]
                if vals.size > 0:
                    min_val = float(np.nanmin(vals))
                else:
                    min_val = float("nan")

                out[(a, b)] = min_val

                if verbose:
                    count = int(np.count_nonzero(np.isfinite(vals)))
                    vlog(True, f"  PAE min {a}→{b}: {min_val:.4f} (from {count} token pairs)", "DEBUG")

    return out


def validate_pae_min_against_summary(
    conf_json_path: str,
    summary_conf_path: str,
    chain_ids: List[str],
    tolerance: float = 0.01,
    verbose: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate AF3's pre-computed chain_pair_pae_min against raw matrix calculation.

    This function computes PAE min from the raw NxN matrix and compares it
    to the pre-computed values in summary_confidences.json.

    Args:
        conf_json_path: Path to *_confidences.json (contains raw PAE matrix)
        summary_conf_path: Path to *_summary_confidences.json (contains pre-computed PAE min)
        chain_ids: List of chain IDs in order matching the summary matrix
        tolerance: Maximum allowed difference (default: 0.01 Angstroms)
        verbose: If True, print comparison details

    Returns:
        Tuple of (all_match: bool, discrepancies: List[str])
        - all_match: True if all values match within tolerance
        - discrepancies: List of human-readable discrepancy descriptions
    """
    discrepancies = []

    # Load summary confidences
    try:
        with open(summary_conf_path, "r") as f:
            summary_data = json.load(f)
        summary_pae_min = summary_data.get("chain_pair_pae_min", [])
    except Exception as e:
        return False, [f"Failed to load summary confidences: {e}"]

    # Compute PAE min from raw matrix
    computed_pae_min = compute_interchain_pae_min_from_conf(
        conf_json_path, valid_chains=chain_ids, verbose=False
    )

    if not computed_pae_min:
        return False, ["Failed to compute PAE min from raw matrix"]

    if verbose:
        log("Validating PAE min: computed from raw matrix vs AF3 summary", "INFO")

    # Compare each chain pair
    all_match = True
    for i, ch_i in enumerate(chain_ids):
        for j, ch_j in enumerate(chain_ids):
            # Get AF3's pre-computed value
            try:
                af3_val = summary_pae_min[i][j]
                if af3_val is None:
                    af3_val = float("nan")
            except (IndexError, TypeError):
                af3_val = float("nan")

            # Get our computed value
            computed_val = computed_pae_min.get((ch_i, ch_j), float("nan"))

            # Compare
            if np.isnan(af3_val) and np.isnan(computed_val):
                match = True
            elif np.isnan(af3_val) or np.isnan(computed_val):
                match = False
            else:
                match = abs(af3_val - computed_val) <= tolerance

            if verbose:
                status = "OK" if match else "MISMATCH"
                vlog(True, f"  {ch_i}→{ch_j}: AF3={af3_val:.4f}, computed={computed_val:.4f} [{status}]", "DEBUG")

            if not match:
                all_match = False
                discrepancies.append(
                    f"{ch_i}→{ch_j}: AF3={af3_val:.4f} vs computed={computed_val:.4f} "
                    f"(diff={abs(af3_val - computed_val):.4f})"
                )

    if verbose:
        if all_match:
            log("PAE min validation PASSED: All values match within tolerance", "INFO")
        else:
            log(f"PAE min validation FAILED: {len(discrepancies)} discrepancies found", "WARN")

    return all_match, discrepancies


# ==============================================================================
# SECTION 10: MAIN RMSD CALCULATION PIPELINE
# ==============================================================================

def calculate_rmsd_after_alignment(
    structure1: PDB.Structure.Structure,
    structure2: PDB.Structure.Structure,
    catres_list: List[Dict],
    ligand_atom_mapping_list: List[Dict],
    verbose: bool = False
) -> Tuple[float, List[float], List[float], List[Dict]]:
    """
    Align two structures and calculate various RMSD metrics.

    This is the main RMSD calculation pipeline that:
    1. Aligns structure1 onto structure2 using Kabsch algorithm on CA atoms
    2. Calculates CA RMSD between aligned structures
    3. Calculates RMSD for each catalytic residue (ncAA-aware)
    4. Calculates RMSD for each ligand group (with symmetric atom handling)

    Args:
        structure1: AF3 predicted structure (will be modified in-place during alignment)
        structure2: Reference structure
        catres_list: List of catalytic residue definitions
        ligand_atom_mapping_list: List of ligand group definitions
        verbose: Whether to print detailed progress

    Returns:
        Tuple of:
        - ca_rmsd: CA RMSD value
        - catres_rmsds: List of RMSD values for each catalytic residue
        - ligand_rmsds: List of RMSD values for each ligand group
        - ncaa_meta: List of metadata dicts for ncAA handling
    """
    vlog(verbose, "Performing Kabsch alignment on CA atoms...")

    # Extract CA coordinates for alignment
    ca_coords1 = extract_ca_coords(structure1)
    ca_coords2 = extract_ca_coords(structure2)

    # Perform Kabsch alignment of CA atoms
    rotation_matrix, centroid1, centroid2 = kabsch_alignment(ca_coords1, ca_coords2)

    # Apply rotation to all atoms in structure1
    for model in structure1:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    new_coord = apply_rotation(
                        np.array([atom.get_coord()]),
                        rotation_matrix, centroid1, centroid2
                    )[0]
                    atom.set_coord(new_coord)

    # Calculate CA RMSD after alignment
    ca_coords1 = extract_ca_coords(structure1)
    ca_rmsd_value = calculate_rmsd(ca_coords1, ca_coords2)
    vlog(verbose, f"CA RMSD after alignment: {ca_rmsd_value:.4f} A")

    # Calculate RMSD for catalytic residues (ncAA-aware)
    vlog(verbose, f"Calculating RMSD for {len(catres_list)} catalytic residues...")
    catres_rmsds = []
    ncaa_meta = []

    for catres in catres_list:
        catres_coords1, catres_atom_name1 = extract_coords_res(
            structure1, catres["chain"], catres["res_num"]
        )
        catres_coords2, catres_atom_name2 = extract_coords_res(
            structure2, catres["chain"], catres["res_num"]
        )

        rmsd_val, af3_only_atoms, ref_only_atoms, common_used = rmsd_on_common_atoms(
            catres_coords1, catres_atom_name1, catres_coords2, catres_atom_name2
        )

        if not common_used:
            rmsd_val = float("nan")

        catres_rmsds.append(rmsd_val)

        # Capture ncAA / overlap metadata per residue
        af3_resname = get_resname_by_chain_resnum(structure1, catres["chain"], catres["res_num"])
        ref_resname = get_resname_by_chain_resnum(structure2, catres["chain"], catres["res_num"])
        ncaa_meta.append({
            "chain": catres["chain"],
            "resnum": catres["res_num"],
            "ref_resname": ref_resname,
            "af3_resname": af3_resname,
            "af3_only_atoms": af3_only_atoms,
            "ref_only_atoms": ref_only_atoms,
            "common_atoms_used": common_used
        })

    # Calculate RMSD for ligand atoms
    vlog(verbose, f"Calculating RMSD for {len(ligand_atom_mapping_list)} ligand groups...")
    ligand_rmsd_values = []

    for ligand_atom_mapping in ligand_atom_mapping_list:
        label = ligand_atom_mapping["label"]
        chain = ligand_atom_mapping["chain"]
        name3 = ligand_atom_mapping["name3"]
        atoms = ligand_atom_mapping["atoms"]

        # Extract coordinates for specified atoms
        ligand_coords1, ligand_coords2 = [], []
        for atom1, atom2 in zip(atoms[0], atoms[1]):
            res1, res2 = name3[0], name3[1]
            ch1, ch2 = chain[0], chain[1]
            ligand_coords1.append(extract_coords_atom(structure1, ch1, res1, atom1))
            ligand_coords2.append(extract_coords_atom(structure2, ch2, res2, atom2))

        # Concatenate and validate coordinates
        try:
            ligand_coords1 = np.concatenate(ligand_coords1)
        except ValueError:
            error_atoms = [atoms[0][i] for i, el in enumerate(ligand_coords1) if len(el) == 0]
            raise KeyError(
                f"[{label}] Missing atoms in AF3({chain[0]}:{name3[0]}): "
                f"{', '.join(error_atoms)} in file mapped to {atoms[0]}"
            )

        try:
            ligand_coords2 = np.concatenate(ligand_coords2)
        except ValueError:
            error_atoms = [atoms[1][i] for i, el in enumerate(ligand_coords2) if len(el) == 0]
            raise KeyError(
                f"[{label}] Missing atoms in REF({chain[1]}:{name3[1]}): "
                f"{', '.join(error_atoms)} in file mapped to {atoms[1]}"
            )

        # Calculate RMSD (with symmetric handling if specified)
        if "symmetric_atom_groups" in ligand_atom_mapping:
            min_rmsd = permute_symmetric_atoms(
                ligand_coords1, ligand_coords2,
                atoms[0], atoms[1],
                ligand_atom_mapping["symmetric_atom_groups"]
            )
            ligand_rmsd_values.append(min_rmsd)
            vlog(verbose, f"  {label}: {min_rmsd:.4f} A (symmetric)")
        else:
            rmsd = calculate_rmsd(ligand_coords1, ligand_coords2)
            ligand_rmsd_values.append(rmsd)
            vlog(verbose, f"  {label}: {rmsd:.4f} A")

    return ca_rmsd_value, catres_rmsds, ligand_rmsd_values, ncaa_meta


# ==============================================================================
# SECTION 11: SINGLE PDB PROCESSING
# ==============================================================================

def process_single_af3_pdb(
    af3_pdb: str,
    af3_pdb_num: int,
    ref_structure: PDB.Structure.Structure,
    catres_list: List[Dict],
    ligand_groups_json: List[Dict],
    calculate_avg_pae: bool,
    validate_pae_min: bool,
    use_af3_summary_pae_min: bool,
    verbose: bool,
    verbose_pae_matrix: bool = False
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Process a single AF3 PDB prediction and extract all metrics.

    This function handles all metric extraction for one AF3 prediction:
    1. Loads AF3 structure and confidence files
    2. Extracts global metrics (ipTM, pTM)
    3. Extracts per-chain pLDDT
    4. Extracts chain-pair PAE metrics (min and mean)
    5. Optionally validates PAE min against raw matrix calculation
    6. Calculates structural RMSD metrics
    7. Extracts per-ligand and per-catalytic-residue metrics

    Args:
        af3_pdb: Path to AF3 PDB file
        af3_pdb_num: Index of this prediction (0-based)
        ref_structure: Reference Biopython structure
        catres_list: Catalytic residue definitions
        ligand_groups_json: Ligand group definitions
        calculate_avg_pae: Whether to compute mean interchain PAE
        validate_pae_min: Whether to validate PAE min against raw matrix
        use_af3_summary_pae_min: If True, use AF3's pre-computed PAE min (may have NaN gaps);
                                  if False (default), compute from raw NxN matrix
        verbose: Whether to print detailed progress
        verbose_pae_matrix: Whether to print full NxN PAE matrices

    Returns:
        Tuple of:
        - Dict with all extracted metrics, keyed by metric_name_idx_{af3_pdb_num}
        - List of warning/issue messages for summary
    """
    metrics = {}
    warnings = []  # Track issues for summary
    idx_suffix = f"idx_{af3_pdb_num}"

    vlog(verbose, f"Loading AF3 structure from {os.path.basename(af3_pdb)}")
    af3_structure = load_pdb(af3_pdb)
    af3_ch_ids = [chain.id for model in af3_structure for chain in model]
    vlog(verbose, f"Found {len(af3_ch_ids)} chains: {af3_ch_ids}")

    # Load confidence files
    vlog(verbose, "Loading confidence files...")
    summary_conf_path = af3_pdb.replace("_model.pdb", "_summary_confidences.json")
    conf_path = af3_pdb.replace("_model.pdb", "_confidences.json")

    with open(summary_conf_path, "r") as f:
        summary_confidences = json.load(f)
    with open(conf_path, "r") as f:
        confidences = json.load(f)

    # Extract global metrics
    iptm = summary_confidences["iptm"]
    ptm = summary_confidences["ptm"]
    metrics[f"iptm_{idx_suffix}"] = iptm
    metrics[f"ptm_{idx_suffix}"] = ptm
    vlog(verbose, f"Global metrics: ipTM={iptm:.4f}, pTM={ptm:.4f}")

    # Per-chain pLDDT
    vlog(verbose, "Computing per-chain pLDDT scores...")
    for ch_id in af3_ch_ids:
        chain_plddts = [
            el[1] for el in zip(confidences["atom_chain_ids"], confidences["atom_plddts"])
            if el[0] == ch_id
        ]
        metrics[f"chain{ch_id}_plddt_{idx_suffix}"] = np.mean(chain_plddts)

    # Chain-pair PAE minimums
    if use_af3_summary_pae_min:
        # Use AF3's pre-computed values from summary file (may have NaN gaps for single-atom chains)
        vlog(verbose, "Using AF3 summary PAE min values (--use_af3_summary_pae_min)...")
        for i, ch_id in enumerate(af3_ch_ids):
            for j, ch_id2 in enumerate(af3_ch_ids):
                val = summary_confidences["chain_pair_pae_min"][i][j]
                if val is None:
                    val = float("nan")
                metrics[f"chain{ch_id}_chain{ch_id2}_pair_pae_min_{idx_suffix}"] = val
    else:
        # Default: compute from raw NxN matrix for complete coverage
        # (AF3's summary_confidences.json often has NaN for non-protein chains)
        vlog(verbose, "Computing chain-pair PAE minimums from raw NxN matrix...")
        computed_pae_min = compute_interchain_pae_min_from_conf(
            conf_path, valid_chains=af3_ch_ids, verbose=False
        )

        # Store computed values (complete coverage, no NaN gaps)
        for i, ch_i in enumerate(af3_ch_ids):
            for j, ch_j in enumerate(af3_ch_ids):
                computed_val = computed_pae_min.get((ch_i, ch_j), float("nan"))
                metrics[f"chain{ch_i}_chain{ch_j}_pair_pae_min_{idx_suffix}"] = computed_val

        # Also store AF3's pre-computed values for comparison (may have NaN gaps)
        vlog(verbose, "Storing AF3 summary PAE min values (for reference)...")
        for i, ch_id in enumerate(af3_ch_ids):
            for j, ch_id2 in enumerate(af3_ch_ids):
                val = summary_confidences["chain_pair_pae_min"][i][j]
                if val is None:
                    val = float("nan")
                metrics[f"chain{ch_id}_chain{ch_id2}_pair_pae_min_af3summary_{idx_suffix}"] = val

    # Symmetric PAE min averages (off-diagonal only)
    vlog(verbose, "Computing symmetric PAE min averages...")
    with np.errstate(all="ignore"):
        for i, ch_i in enumerate(af3_ch_ids):
            for j, ch_j in enumerate(af3_ch_ids):
                if j <= i:
                    continue  # Skip diagonal and reverse duplicates

                key_ij = f"chain{ch_i}_chain{ch_j}_pair_pae_min_{idx_suffix}"
                key_ji = f"chain{ch_j}_chain{ch_i}_pair_pae_min_{idx_suffix}"

                v_ij = metrics.get(key_ij, float("nan"))
                v_ji = metrics.get(key_ji, float("nan"))

                avg_val = np.nanmean([v_ij, v_ji])

                # Canonical name: lower/earlier chain ID first (alphabetical)
                a, b = sorted([str(ch_i), str(ch_j)])
                avg_key = f"chain{a}_chain{b}_pair_pae_min_avg_{idx_suffix}"
                metrics[avg_key] = float(avg_val) if np.isfinite(avg_val) else float("nan")

    # Validate PAE min against raw matrix calculation (optional)
    if validate_pae_min:
        vlog(verbose, "Validating computed PAE min against AF3 summary...")
        all_match, discrepancies = validate_pae_min_against_summary(
            conf_path, summary_conf_path, af3_ch_ids,
            tolerance=0.1, verbose=verbose  # Increased tolerance for rounding differences
        )
        if not all_match:
            # Filter to show only significant discrepancies (not just rounding or NaN fill-ins)
            significant_discrepancies = [d for d in discrepancies if "AF3=nan" not in d]
            if significant_discrepancies:
                warn_msg = f"idx_{af3_pdb_num}: PAE min has {len(significant_discrepancies)} significant discrepancies (beyond rounding)"
                warnings.append(warn_msg)
                log(warn_msg, "WARN")
                for disc in significant_discrepancies[:3]:
                    log(f"    {disc}", "WARN")
                if len(significant_discrepancies) > 3:
                    log(f"    ... and {len(significant_discrepancies) - 3} more", "WARN")
            else:
                vlog(verbose, f"PAE min validation: all discrepancies are rounding/NaN fill-ins (OK)")
        else:
            vlog(verbose, f"PAE min validation passed for idx {af3_pdb_num}")

    # Mean interchain PAE from token-level PAE matrix (optional)
    if calculate_avg_pae:
        vlog(verbose, "Computing token-level mean PAE (interchain)...")
        try:
            if os.path.isfile(conf_path) and not conf_path.endswith("summary_confidences.json"):
                pair_means = compute_interchain_pae_means_from_conf(
                    conf_path,
                    valid_chains=af3_ch_ids,
                    verbose=verbose,
                    verbose_matrix=verbose_pae_matrix,
                    label=f"idx {af3_pdb_num}"
                )
                if pair_means:
                    for (a, b), mean_val in pair_means.items():
                        x, y = sorted([str(a), str(b)])
                        key = f"chain{x}_chain{y}_pair_pae_mean_{idx_suffix}"
                        metrics[key] = float(mean_val) if np.isfinite(mean_val) else float("nan")
        except Exception as e:
            warn_msg = f"idx_{af3_pdb_num}: Failed to compute token-level PAE - {e}"
            warnings.append(warn_msg)
            log(warn_msg, "WARN")

    # Structural alignment and RMSD calculations
    ca_rmsd, catres_rmsds, ligand_rmsds, ncaa_meta = calculate_rmsd_after_alignment(
        af3_structure, ref_structure, catres_list, ligand_groups_json, verbose
    )

    # Ligand pLDDT scores
    ligand_plddts = get_avr_plddt(af3_pdb, ligand_groups_json)

    # Store CA RMSD
    metrics[f"ca_rmsd_{idx_suffix}"] = ca_rmsd

    # Store ligand metrics
    vlog(verbose, "Storing ligand RMSD and pLDDT metrics...")
    for i, (ligand_rmsd, ligand_plddt) in enumerate(zip(ligand_rmsds, ligand_plddts)):
        label = ligand_groups_json[i]['label']
        metrics[f"{label}_rmsd_{idx_suffix}"] = ligand_rmsd
        metrics[f"{label}_plddt_{idx_suffix}"] = ligand_plddt

    # Store catalytic residue RMSD metrics
    vlog(verbose, "Storing catalytic residue RMSD metrics...")
    for i, catres in enumerate(catres_list):
        metrics[f"{catres['name3']}{i+1}_rmsd_{idx_suffix}"] = catres_rmsds[i]
    metrics[f"catres_rmsd_{idx_suffix}"] = np.mean(catres_rmsds)

    # Store catalytic residue pLDDT metrics
    vlog(verbose, "Extracting catalytic residue pLDDT scores...")
    plddt_per_residue_dic = get_plddt_per_residue_dic(af3_pdb)
    cat_res_plddts = []
    for i, catres in enumerate(catres_list):
        cat_res_plddt = np.mean([
            plddt_per_residue_dic[el] for el in plddt_per_residue_dic.keys()
            if el == (catres["chain"], catres["res_num"])
        ])
        metrics[f"{catres['name3']}{i+1}_plddt_{idx_suffix}"] = cat_res_plddt
        cat_res_plddts.append(cat_res_plddt)
    metrics[f"catres_plddt_{idx_suffix}"] = np.nanmean(cat_res_plddts)

    # Store ncAA (non-canonical amino acid) metadata
    for i, meta in enumerate(ncaa_meta):
        if not meta["af3_only_atoms"]:
            continue
        label_core = (meta["af3_resname"] or catres_list[i]["name3"]) + str(i + 1)
        metrics[f"{label_core}_extra_atoms_count_{idx_suffix}"] = len(meta["af3_only_atoms"])
        avg_plddt_ncaa = get_plddt_for_specific_atoms(
            af3_pdb, meta["chain"], meta["resnum"], meta["af3_only_atoms"]
        )
        metrics[f"{label_core}_extra_atoms_plddt_{idx_suffix}"] = avg_plddt_ncaa
        metrics[f"{label_core}_common_atoms_used_{idx_suffix}"] = (
            "|".join(meta["common_atoms_used"]) if meta["common_atoms_used"] else ""
        )

    return metrics, warnings


# ==============================================================================
# SECTION 12: MAIN ENTRY POINT
# ==============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed argument namespace
    """
    parser = argparse.ArgumentParser(
        description="Process AlphaFold3 PDB predictions and compute structural metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage:
    python process_af3_pdb.py --af3_dir ./af3_output --ref_pdb reference.pdb \\
        --outscr scores.sc --ligand_groups_json '[...]'

  With verbose output:
    python process_af3_pdb.py --af3_dir ./af3_output --ref_pdb reference.pdb \\
        --outscr scores.sc --ligand_groups_json '[...]' --verbose

  With mean PAE calculation:
    python process_af3_pdb.py --af3_dir ./af3_output --ref_pdb reference.pdb \\
        --outscr scores.sc --ligand_groups_json '[...]' --calculate_avg_pae_in_addition_to_pair
        """
    )

    parser.add_argument(
        "--ref_pdb",
        required=True,
        help="Path to reference PDB file with REMARK 666 catalytic residue annotations"
    )
    parser.add_argument(
        "--af3_dir",
        required=True,
        help="Directory containing AF3 predictions (*_model.pdb and *_confidences.json files)"
    )
    parser.add_argument(
        "--outscr",
        type=str,
        required=True,
        help="Output score file path (.sc or .csv)"
    )
    parser.add_argument(
        "--ligand_groups_json",
        type=str,
        required=True,
        help="JSON string defining ligand atom groups and labels for RMSD calculation"
    )
    parser.add_argument(
        "--calculate_avg_pae_in_addition_to_pair",
        action="store_true",
        help="Also compute mean interchain PAE from token-level PAE matrix"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress and debug information"
    )
    parser.add_argument(
        "--rapid_mode_skip_sc_if_found",
        action="store_true",
        help="Skip processing if any .sc file already exists in --af3_dir"
    )
    parser.add_argument(
        "--validate_pae_min",
        action="store_true",
        help="Validate AF3's pre-computed PAE min values against raw matrix calculation"
    )
    parser.add_argument(
        "--use_af3_summary_pae_min",
        action="store_true",
        help="Use AF3's pre-computed PAE min from summary file instead of computing from raw NxN matrix (may have NaN gaps for single-atom chains)"
    )
    parser.add_argument(
        "--verbose_pae_matrix",
        action="store_true",
        help="Print full NxN PAE matrices (WARNING: very large output, use for debugging only)"
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point for AF3 PDB processing.

    Workflow:
    1. Parse arguments and check for rapid skip
    2. Load reference structure and extract catalytic residues
    3. Process each AF3 PDB prediction
    4. Aggregate metrics and write output CSV
    5. Print comprehensive summary
    """
    args = parse_arguments()

    # Log startup information
    log("=" * 70)
    log("AF3 PDB PROCESSING - STARTED")
    log("=" * 70)
    log(f"  Reference PDB: {args.ref_pdb}")
    log(f"  AF3 directory: {args.af3_dir}")
    log(f"  Output score file: {args.outscr}")
    log(f"  Options: verbose={args.verbose}, validate_pae_min={args.validate_pae_min}, "
        f"calculate_avg_pae={args.calculate_avg_pae_in_addition_to_pair}")

    # Check for rapid skip mode
    rapid_skip_if_sc_present(args.af3_dir, args.rapid_mode_skip_sc_if_found)

    # Parse ligand groups JSON
    args.ligand_groups_json = json.loads(args.ligand_groups_json)
    log(f"  Ligand groups defined: {len(args.ligand_groups_json)}")

    # Initialize score dictionary and tracking
    score_dic = {"description": args.af3_dir}
    all_warnings = []
    per_pdb_summary = []

    # Load reference structure
    vlog(args.verbose, f"Loading reference structure from {args.ref_pdb}")
    ref_structure = load_pdb(args.ref_pdb)

    # Extract catalytic residues from reference
    catres_list = get_catalytic_residues(args.ref_pdb, args.verbose)
    log(f"  Catalytic residues: {len(catres_list)}")

    # Find all AF3 PDB files
    af3_pdbs = sorted(glob.glob(os.path.join(args.af3_dir, "*.pdb")))
    total_pdbs = len(af3_pdbs)
    log(f"  AF3 PDB files found: {total_pdbs}")
    log("-" * 70)

    # Process each AF3 prediction
    for af3_pdb_num, af3_pdb in enumerate(af3_pdbs):
        log(f"Processing [{af3_pdb_num + 1}/{total_pdbs}]: {os.path.basename(af3_pdb)}")

        metrics, warnings = process_single_af3_pdb(
            af3_pdb=af3_pdb,
            af3_pdb_num=af3_pdb_num,
            ref_structure=ref_structure,
            catres_list=catres_list,
            ligand_groups_json=args.ligand_groups_json,
            calculate_avg_pae=args.calculate_avg_pae_in_addition_to_pair,
            validate_pae_min=args.validate_pae_min,
            use_af3_summary_pae_min=args.use_af3_summary_pae_min,
            verbose=args.verbose,
            verbose_pae_matrix=args.verbose_pae_matrix
        )

        # Merge metrics into main score dictionary
        score_dic.update(metrics)
        all_warnings.extend(warnings)

        # Collect per-PDB summary
        idx = f"idx_{af3_pdb_num}"
        pdb_summary = {
            "pdb": os.path.basename(af3_pdb),
            "iptm": metrics.get(f"iptm_{idx}", float("nan")),
            "ptm": metrics.get(f"ptm_{idx}", float("nan")),
            "ca_rmsd": metrics.get(f"ca_rmsd_{idx}", float("nan")),
            "catres_rmsd": metrics.get(f"catres_rmsd_{idx}", float("nan")),
            "warnings": len(warnings)
        }
        per_pdb_summary.append(pdb_summary)

    # Write output
    log("-" * 70)
    log(f"Writing results to {args.outscr}")
    score_df = pd.DataFrame([score_dic])
    score_df.to_csv(args.outscr, index=False)

    # =========================================================================
    # SUMMARY SECTION
    # =========================================================================
    print("\n")
    log("=" * 70)
    log("PROCESSING SUMMARY")
    log("=" * 70)

    # Per-PDB metrics summary
    print(f"\n{'PDB File':<50} {'ipTM':>8} {'pTM':>8} {'CA RMSD':>10} {'CatRes RMSD':>12} {'Warns':>6}")
    print("-" * 96)
    for s in per_pdb_summary:
        print(f"{s['pdb']:<50} {s['iptm']:>8.4f} {s['ptm']:>8.4f} {s['ca_rmsd']:>10.4f} {s['catres_rmsd']:>12.4f} {s['warnings']:>6}")

    # Overall statistics
    print("\n" + "-" * 70)
    if per_pdb_summary:
        avg_iptm = np.nanmean([s['iptm'] for s in per_pdb_summary])
        avg_ptm = np.nanmean([s['ptm'] for s in per_pdb_summary])
        avg_ca_rmsd = np.nanmean([s['ca_rmsd'] for s in per_pdb_summary])
        avg_catres_rmsd = np.nanmean([s['catres_rmsd'] for s in per_pdb_summary])
        best_iptm_idx = np.nanargmax([s['iptm'] for s in per_pdb_summary])
        best_ca_rmsd_idx = np.nanargmin([s['ca_rmsd'] for s in per_pdb_summary])

        print(f"AVERAGES:  ipTM={avg_iptm:.4f}  pTM={avg_ptm:.4f}  CA_RMSD={avg_ca_rmsd:.4f}A  CatRes_RMSD={avg_catres_rmsd:.4f}A")
        print(f"BEST ipTM: idx_{best_iptm_idx} ({per_pdb_summary[best_iptm_idx]['pdb']}) = {per_pdb_summary[best_iptm_idx]['iptm']:.4f}")
        print(f"BEST CA_RMSD: idx_{best_ca_rmsd_idx} ({per_pdb_summary[best_ca_rmsd_idx]['pdb']}) = {per_pdb_summary[best_ca_rmsd_idx]['ca_rmsd']:.4f}A")

    # Warnings summary
    print("\n" + "-" * 70)
    if all_warnings:
        log(f"WARNINGS: {len(all_warnings)} issue(s) detected", "WARN")
        for w in all_warnings:
            print(f"  - {w}")
    else:
        log("WARNINGS: None - all validations passed", "INFO")

    # Final status
    print("\n" + "=" * 70)
    status = "COMPLETED WITH WARNINGS" if all_warnings else "COMPLETED SUCCESSFULLY"
    log(f"STATUS: {status}")
    log(f"Output: {args.outscr}")
    log(f"Total metrics columns: {len(score_dic)}")
    log("=" * 70)


if __name__ == "__main__":
    main()
