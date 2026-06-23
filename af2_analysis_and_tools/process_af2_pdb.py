#!/usr/bin/env python3
"""
================================================================================
process_af2_pdb.py  --  AlphaFold2 (superfold) prediction structural analysis
================================================================================

Processes AlphaFold2 / superfold monomer predictions against a reference PDB
(carrying Rosetta REMARK 666 catalytic-residue annotations) and writes one
Rosetta-style ``.sc`` scorefile per predicted PDB.

This is the AF2 analogue of ``advanced_structure_prediction_tools/process_af3_pdb.py``.
The catalytic-residue / structural metrics use the SAME engine (Biopython +
biotite + numpy), but ALL ligand-matching machinery is dropped: AF2 superfold
output is a pure-protein monomer (chain A is the norm). The default path imports
and runs with NO pyrosetta installed (universal.sif). The legacy pyrosetta path
(``--legacy_atom_groups``) lazy-imports pyrosetta inside its own function.

--------------------------------------------------------------------------------
OUTPUT FORMAT  (IMPORTANT -- the concatenator relies on this)
--------------------------------------------------------------------------------
Each predicted PDB produces exactly ONE ``.sc`` file. A ``.sc`` file is a CSV
with EXACTLY TWO lines: a header line and EXACTLY ONE data row. ``description``
(the full prediction basename) is ALWAYS the first column. Writes are atomic
(``<out>.sc.tmp.<pid>`` -> ``os.replace``) so SLURM-array re-runs never see a
half-written file. ``--skip_if_sc_present`` makes re-runs idempotent.

Column policy (applied as a pure POST-BUILD filter; computed values are never
changed and the ``--verbose`` REPORT always reflects the FULL computation):

  Catres-pair PAE aggregates report mean/max/MIN of the off-diagonal symmetric
  catres x catres PAE: ``catres_pair_pae_mean/_max/_min`` (and the
  ``catres_subset_pair_pae_*`` mirror when --catres_subset is given). The _min
  stat is emitted ALWAYS and is kept by --lean.

  AUTO-DROP (in ALL modes, incl. default/full -- these columns carry no info):
    * Monomer (n_protein_chains == 1): af2_mean_pae_intra_chain_A,
      af2_mean_pae_interaction, n_protein_chains are dropped. (>1 keeps them.)
    * No terminal trim (N=0 and C=0): terminal_ignore_N, terminal_ignore_C are
      dropped. (Either --N/C_terminus_tag_length_to_ignore > 0 keeps both.)
    With --verbose the detection is PRINTED before the row is written.

  Two opt-in flags further trim columns:
  --no_per_catres  drop ONLY the per-catalytic-residue columns
                   ({name3}{i}_rmsd/_bb_rmsd/_plddt/_pae and any --pae_full
                   pairwise {name3a}{i}_{name3b}{j}_pae columns); ALL catres_*
                   aggregates and everything else (status/error, metadata, paths)
                   are kept. The auto-drops above still apply.
  --lean           emit ONLY the high-signal core keep-list (those present after
                   the auto-drops): description, pdb_path, af2_json_path, ref_path,
                   catres_signature, catres_count, af2_mean_plddt, af2_ptm_score,
                   af2_mean_pae, af2_mean_pae_intra_chain, af2_rmsd_to_input,
                   af2_tol, ca_rmsd, tm_score, and the catres_* aggregates
                   (catres_rmsd/_bb_rmsd/_lddt/_plddt/_pae_to_all_mean/
                   _pair_pae_mean/_pair_pae_max/_pair_pae_min) PLUS the matching
                   catres_subset_* aggregates (incl. _pair_pae_min) only when
                   --catres_subset is given. --lean DROPS status, error, all
                   per-catres columns, the --pae_full pairs, af2_model/_type/_seed/
                   _recycles/_elapsed_time/_pae_length and ca_rmsd_TMalign. --lean
                   wins over --no_per_catres.

--------------------------------------------------------------------------------
AF2 / superfold output facts (flat directory; one model per design is the norm)
--------------------------------------------------------------------------------
Per prediction (in one flat dir):
  <base>_unrelaxed.pdb              per-residue pLDDT in the B-factor column
  <base>_prediction_results.json   AF2 self-reported scalars + full LxL PAE
where  <base> = <design>_model_<N>_ptm_seed_<S>.

JSON keys captured (ALL emitted under an ``af2_`` prefix to distinguish AF2's
self-reported values from our recomputed-vs-reference values):
  mean_plddt, pTMscore, mean_pae, mean_pae_intra_chain, mean_pae_intra_chain_A,
  mean_pae_interaction, rmsd_to_input, tol, recycles, model, type, seed,
  elapsed_time, pae (full LxL residue x residue matrix; directional; diag ~0.25).
Some JSON files have trailing garbage after the object -> handled by safe_loads.

--------------------------------------------------------------------------------
RESIDUE-NUMBERING POLICY  (P0 #1)
--------------------------------------------------------------------------------
REMARK 666 gives (chain, name3, res_num, cst_block). AF2/superfold preserves the
INPUT residue numbering, so res_num is treated as a literal PDB resseq and the
residue is looked up directly in BOTH ref and prediction via the Bio.PDB residue
id (' ', res_num, icode). The residue NAME is validated (ref name3 == pred
resname). On any mismatch / not-found / missing common atoms, that catres' metrics
become NaN and a diagnostic is appended to the row's ``error`` field -- we never
silently emit a wrong number.

Terminal-tag trimming (``--N/C_terminus_tag_length_to_ignore``) only affects the
GLOBAL CA Kabsch alignment and the PAE row map (chain A only). It does NOT renumber
catres -- those are always resolved by literal resseq.

--------------------------------------------------------------------------------
PAE METRICS  (P0 #2, #3)
--------------------------------------------------------------------------------
The PAE matrix is 0-based over the UNTRIMMED predicted residue order. After the
(N,C) terminal-ignore fallback has chosen the actual pair, an effective->raw
ordinal map (chain,resseq,icode)->pae_row is built with the SAME effective mapping
the RMSD/pLDDT path uses. catres PAE metrics index the matrix through that map.
PAE is directional: pairwise catres metrics use the symmetric average
0.5*(pae[i][j]+pae[j][i]); the diagonal is excluded from pair aggregates. The
off-diagonal pair PAE is aggregated as mean / max / MIN (catres_pair_pae_mean,
_max, _min, plus the catres_subset_pair_pae_* mirror).

Author: Woodbuse Lab
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONSTANTS
# ==============================================================================

from Bio import PDB
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import argparse
import copy
import csv
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

# Biotite imports for TM-score and lDDT calculations
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile

# PDB format column positions (0-indexed; standard fixed-width PDB columns)
PDB_RECORD_TYPE_COLS = (0, 6)
PDB_ATOM_NAME_COLS = (12, 16)
PDB_RESIDUE_NAME_COLS = (17, 20)
PDB_CHAIN_ID_COLS = (21, 22)
PDB_RESIDUE_NUM_COLS = (22, 26)
PDB_ICODE_COL = (26, 27)
PDB_PLDDT_COLS = (60, 66)

# Canonical backbone atom ordering for consistent RMSD comparisons
CANONICAL_BACKBONE_ORDER = ["N", "CA", "C", "O"]

# Backbone atoms used for the symmetry-immune catres_bb_rmsd
BACKBONE_ATOMS = ("N", "CA", "C", "O")

# Standard amino acids (used for protein vs ligand distinction)
STANDARD_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
}

# Modified/non-canonical amino acids treated as protein residues (have a CA and
# participate in the chain). May appear as HETATM 'H_XXX' in some PDBs.
MODIFIED_AMINO_ACIDS = {
    "KCX",  # Carboxylated lysine
    "MSE",  # Selenomethionine
    "SEC",  # Selenocysteine
    "PYL",  # Pyrrolysine
}

PROTEIN_AMINO_ACIDS = STANDARD_AMINO_ACIDS | MODIFIED_AMINO_ACIDS

# Symmetric atom pairs (chemically equivalent due to rotational symmetry):
#   PHE/TYR 180 ring flip; ASP/GLU carboxylate swap; ARG guanidinium nitrogens;
#   LEU/VAL branched aliphatic carbons.
SYMMETRIC_ATOM_PAIRS = {
    "PHE": [("CD1", "CD2"), ("CE1", "CE2")],
    "TYR": [("CD1", "CD2"), ("CE1", "CE2")],
    "ASP": [("OD1", "OD2")],
    "GLU": [("OE1", "OE2")],
    "ARG": [("NH1", "NH2")],
    "LEU": [("CD1", "CD2")],
    "VAL": [("CG1", "CG2")],
}

# Monomer assumption: chain A is the designed protein and the only chain trimmed.
DEFAULT_TRIM_CHAINS = frozenset({"A"})

# AF2 PDB / JSON filename conventions.
PRED_PDB_SUFFIX = "_unrelaxed.pdb"
PRED_JSON_SUFFIX = "_prediction_results.json"


# ==============================================================================
# SECTION 2: LOGGING UTILITIES
# ==============================================================================

def log(msg: str, level: str = "INFO") -> None:
    """Print a timestamped log message to stdout."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level}] {msg}", flush=True)


def vlog(verbose: bool, msg: str, level: str = "DEBUG") -> None:
    """Conditional verbose logging - only prints if verbose is True."""
    if verbose:
        log(msg, level)


# ==============================================================================
# SECTION 3: PDB FILE I/O  (ported from process_af3_pdb.py: load_pdb, save_pdb)
# ==============================================================================

def load_pdb(file_path: str) -> PDB.Structure.Structure:
    """Load a PDB file using Biopython's PDBParser."""
    parser = PDB.PDBParser(QUIET=True)
    structure_id = os.path.basename(file_path)
    return parser.get_structure(structure_id, file_path)


def save_pdb(structure: PDB.Structure.Structure, output_file: str) -> None:
    """Save a Biopython structure to a PDB file."""
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_file)


# ==============================================================================
# SECTION 4: COORDINATE EXTRACTION  (ported; ligand branches dropped)
# ==============================================================================

def is_protein_residue(residue) -> bool:
    """
    True if a Biopython residue is a protein residue (standard or modified AA).

    Handles standard AAs (hetero flag ' ') and modified AAs like KCX (which may
    carry an 'H_KCX' hetero flag).
    """
    resname = residue.get_resname()
    hetero_flag = residue.get_id()[0]
    if hetero_flag == ' ' and resname in STANDARD_AMINO_ACIDS:
        return True
    if resname in MODIFIED_AMINO_ACIDS:
        return True
    return False


def _effective_ignore(
    chain_id: str,
    n_ignore: int,
    c_ignore: int,
    n_trim_chains: Optional[set] = None,
    c_trim_chains: Optional[set] = None
) -> Tuple[int, int]:
    """Return (effective_n_ignore, effective_c_ignore) for one chain.

    Trimming is applied to a chain only if it appears in the relevant trim set
    (default {'A'}); peptide / ligand chains are never trimmed.
    """
    n_set = DEFAULT_TRIM_CHAINS if n_trim_chains is None else n_trim_chains
    c_set = DEFAULT_TRIM_CHAINS if c_trim_chains is None else c_trim_chains
    eff_n = n_ignore if chain_id in n_set else 0
    eff_c = c_ignore if chain_id in c_set else 0
    return eff_n, eff_c


def extract_ca_coords(
    structure: PDB.Structure.Structure,
    n_ignore: int = 0,
    c_ignore: int = 0,
    n_trim_chains: Optional[set] = None,
    c_trim_chains: Optional[set] = None
) -> np.ndarray:
    """
    Extract CA coordinates for structural alignment (per-chain terminal ignore).

    The first n_ignore and last c_ignore protein residues are skipped ONLY on
    chains listed in n_trim_chains / c_trim_chains (default {'A'}).
    """
    ca_coords = []
    for model in structure:
        for chain in model:
            eff_n, eff_c = _effective_ignore(
                chain.id, n_ignore, c_ignore, n_trim_chains, c_trim_chains
            )
            protein_residues = [r for r in chain.get_residues() if is_protein_residue(r)]
            if eff_c > 0:
                effective_residues = protein_residues[eff_n:len(protein_residues) - eff_c]
            else:
                effective_residues = protein_residues[eff_n:]
            for residue in effective_residues:
                if 'CA' in residue:
                    ca_coords.append(residue['CA'].get_coord())
    return np.array(ca_coords)


def _residue_heavy_atoms(residue) -> Tuple[np.ndarray, List[str]]:
    """Return (coords, names) of heavy (non-H) atoms for a Biopython residue."""
    coords, names = [], []
    for atom in residue:
        if atom.element == "H":
            continue
        coords.append(atom.get_coord())
        names.append(atom.get_name())
    return np.array(coords), names


def find_residue_by_resseq(
    structure: PDB.Structure.Structure,
    chain_id: str,
    res_num: int,
    icode: str = ' '
):
    """
    Look up a residue by literal PDB resseq (P0 #1).

    AF2/superfold preserves input numbering, so REMARK-666 res_num maps directly
    onto the prediction and reference via the Bio.PDB residue id (' ', res_num,
    icode). Returns the Biopython Residue or None if not found.
    """
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            target_id = (' ', int(res_num), icode)
            if target_id in chain:
                return chain[target_id]
            # Fall back to a manual scan (covers HETATM-flagged modified AAs).
            for residue in chain:
                hetflag, seq, ic = residue.get_id()
                if seq == int(res_num) and ic == icode:
                    return residue
    return None


def get_catalytic_residues(pdb_file: str, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Parse REMARK 666 catalytic residues (ported from process_af3_pdb.py).

    Format: REMARK 666 MATCH TEMPLATE <tch> <tname> <tresi> MATCH MOTIF
            <chain> <name3> <resnum> <cst_block> <cst_idx>

    Returns a list (in REMARK order) of dicts: res_num, name3, chain, icode,
    cst_block. A trailing insertion code on the residue number (e.g. "42A") is
    parsed into ``res_num`` (int) and ``icode`` (str, default ' ') so the catres
    is not silently dropped.
    """
    catalytic_residues: List[Dict[str, Any]] = []
    with open(pdb_file, 'r') as fh:
        for line in fh:
            if line.startswith("ATOM"):
                break  # REMARK lines precede ATOM records
            if line.startswith("REMARK 666") and "MATCH MOTIF" in line:
                parts = line.split()
                try:
                    chain = parts[9]
                    residue_name = parts[10]
                    resnum_tok = parts[11]
                    m = re.match(r'^(-?\d+)', resnum_tok)
                    if not m:
                        raise ValueError(f"unparsable resnum token: {resnum_tok!r}")
                    digits = m.group(1)
                    residue_number = int(digits)
                    icode = resnum_tok[len(digits):].strip() or ' '
                    cst_block = int(parts[12]) if len(parts) > 12 else 0
                except (IndexError, ValueError):
                    continue
                catalytic_residues.append({
                    'res_num': residue_number,
                    'name3': residue_name,
                    'chain': chain,
                    'icode': icode,
                    'cst_block': cst_block,
                })
    if verbose and catalytic_residues:
        vlog(True, f"Catalytic residues from {os.path.basename(pdb_file)}: {catalytic_residues}", "INFO")
    return catalytic_residues


# ==============================================================================
# SECTION 5: STRUCTURAL ALIGNMENT (KABSCH)  (ported faithfully)
# ==============================================================================

def kabsch_alignment(
    coords1: np.ndarray,
    coords2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimal rotation aligning coords1 onto coords2 (Kabsch via SVD)."""
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)
    coords1_centered = coords1 - centroid1
    coords2_centered = coords2 - centroid2
    H = np.dot(coords1_centered.T, coords2_centered)
    V, S, W = np.linalg.svd(H)
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
    """Apply a Kabsch rotation, transforming from frame 1 to frame 2."""
    coords_centered = coords - centroid1
    coords_rotated = np.dot(coords_centered, rotation_matrix) + centroid2
    return coords_rotated


def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """RMSD between two equally-sized coordinate sets (Angstroms)."""
    diff = coords1 - coords2
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


# ==============================================================================
# SECTION 6: COMMON-ATOM + SYMMETRY-AWARE RMSD  (ported faithfully)
# ==============================================================================

def order_common_atoms(names1: List[str], names2: List[str]) -> List[str]:
    """Canonical ordering of atoms present in both residues (backbone first)."""
    s1, s2 = set(names1), set(names2)
    inter = list(s1 & s2)
    if not inter:
        return []
    back = [a for a in CANONICAL_BACKBONE_ORDER if a in inter]
    rest = sorted([a for a in inter if a not in CANONICAL_BACKBONE_ORDER])
    return back + rest


def rmsd_on_common_atoms(
    coords1: np.ndarray,
    names1: List[str],
    coords2: np.ndarray,
    names2: List[str]
) -> Tuple[float, List[str], List[str], List[str]]:
    """RMSD over atoms present in both structures."""
    common = order_common_atoms(names1, names2)
    s1, s2 = set(names1), set(names2)
    pred_only = sorted(list(s1 - s2))
    ref_only = sorted(list(s2 - s1))
    if not common:
        return (float("nan"), pred_only, ref_only, [])
    idx1 = [names1.index(n) for n in common]
    idx2 = [names2.index(n) for n in common]
    return (calculate_rmsd(coords1[idx1], coords2[idx2]), pred_only, ref_only, common)


def rmsd_on_common_atoms_with_symmetry(
    coords1: np.ndarray,
    names1: List[str],
    coords2: np.ndarray,
    names2: List[str],
    resname: str
) -> Tuple[float, List[str], List[str], List[str], bool]:
    """
    Common-atom RMSD with symmetric-atom handling (PHE/TYR/ASP/GLU/ARG/LEU/VAL).

    Aligns on non-symmetric atoms, then keeps the lower-RMSD symmetric assignment.
    Returns (rmsd, pred_only, ref_only, common_used, swapped).
    """
    if resname not in SYMMETRIC_ATOM_PAIRS:
        rmsd_val, pred_only, ref_only, common = rmsd_on_common_atoms(
            coords1, names1, coords2, names2
        )
        return rmsd_val, pred_only, ref_only, common, False

    symmetric_atoms = set()
    for atom1, atom2 in SYMMETRIC_ATOM_PAIRS[resname]:
        symmetric_atoms.add(atom1)
        symmetric_atoms.add(atom2)

    name_to_coord1 = {names1[i]: coords1[i] for i in range(len(names1))}
    name_to_coord2 = {names2[i]: coords2[i] for i in range(len(names2))}

    s1, s2 = set(names1), set(names2)
    common_all = s1 & s2
    pred_only = sorted(list(s1 - s2))
    ref_only = sorted(list(s2 - s1))

    if not common_all:
        return float("nan"), pred_only, ref_only, [], False

    common_non_symmetric = [n for n in common_all if n not in symmetric_atoms]
    common_symmetric = [n for n in common_all if n in symmetric_atoms]

    if len(common_non_symmetric) < 3 or not common_symmetric:
        rmsd_val, pred_only, ref_only, common = rmsd_on_common_atoms(
            coords1, names1, coords2, names2
        )
        return rmsd_val, pred_only, ref_only, common, False

    align_coords1 = np.array([name_to_coord1[n] for n in sorted(common_non_symmetric)])
    align_coords2 = np.array([name_to_coord2[n] for n in sorted(common_non_symmetric)])

    center1 = np.mean(align_coords1, axis=0)
    center2 = np.mean(align_coords2, axis=0)
    centered1 = align_coords1 - center1
    centered2 = align_coords2 - center2

    H = centered1.T @ centered2
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    def transform(coord):
        return R @ (coord - center1) + center2

    def swap_name(name):
        for a1, a2 in SYMMETRIC_ATOM_PAIRS[resname]:
            if name == a1:
                return a2
            elif name == a2:
                return a1
        return name

    rmsd_sym_orig = 0.0
    for name in common_symmetric:
        pred_transformed = transform(name_to_coord1[name])
        rmsd_sym_orig += np.sum((pred_transformed - name_to_coord2[name]) ** 2)

    rmsd_sym_swap = 0.0
    for name in common_symmetric:
        swapped = swap_name(name)
        if swapped in name_to_coord2:
            pred_transformed = transform(name_to_coord1[name])
            rmsd_sym_swap += np.sum((pred_transformed - name_to_coord2[swapped]) ** 2)
        else:
            rmsd_sym_swap += np.sum((transform(name_to_coord1[name]) - name_to_coord2[name]) ** 2)

    use_swap = rmsd_sym_swap < rmsd_sym_orig

    common_ordered = order_common_atoms(names1, names2)
    if use_swap:
        swapped_names1 = [swap_name(n) if n in symmetric_atoms else n for n in names1]
        common_ordered = order_common_atoms(swapped_names1, names2)
        idx1 = [swapped_names1.index(n) for n in common_ordered]
    else:
        idx1 = [names1.index(n) for n in common_ordered]

    idx2 = [names2.index(n) for n in common_ordered]
    rmsd_val = calculate_rmsd(coords1[idx1], coords2[idx2])
    return rmsd_val, pred_only, ref_only, common_ordered, use_swap


def backbone_rmsd_for_residue(
    pred_coords: np.ndarray,
    pred_names: List[str],
    ref_coords: np.ndarray,
    ref_names: List[str]
) -> float:
    """
    Symmetry-immune backbone RMSD over (N, CA, C, O) atoms common to both.

    Returns NaN if fewer than one common backbone atom is found.
    """
    common = [a for a in BACKBONE_ATOMS if a in set(pred_names) & set(ref_names)]
    if not common:
        return float("nan")
    idx1 = [pred_names.index(n) for n in common]
    idx2 = [ref_names.index(n) for n in common]
    return calculate_rmsd(pred_coords[idx1], ref_coords[idx2])


# ==============================================================================
# SECTION 7: BIOTITE TM-SCORE AND lDDT  (ported; symmetric helpers retained)
# ==============================================================================

def load_biotite_structure(pdb_path: str) -> Optional[struc.AtomArray]:
    """Load a PDB file as a Biotite AtomArray (model 1), or None on failure."""
    try:
        pdb_file = PDBFile.read(pdb_path)
        return pdb_file.get_structure(model=1)
    except Exception:
        return None


def _build_chain_residue_order(atoms: struc.AtomArray) -> Dict[str, List[int]]:
    """{chain_id: [res_id, ...]} with first-appearance order, de-duped per chain."""
    result: Dict[str, List[int]] = {}
    seen: Dict[str, set] = {}
    for ch, ri in zip(atoms.chain_id, atoms.res_id):
        ch_s = str(ch)
        ri_i = int(ri)
        if ch_s not in seen:
            seen[ch_s] = set()
            result[ch_s] = []
        if ri_i not in seen[ch_s]:
            seen[ch_s].add(ri_i)
            result[ch_s].append(ri_i)
    return result


def filter_biotite_protein_residues(
    atoms: struc.AtomArray,
    n_ignore: int = 0,
    c_ignore: int = 0,
    n_trim_chains: Optional[set] = None,
    c_trim_chains: Optional[set] = None
) -> struc.AtomArray:
    """Filter to protein residues with per-chain terminal ignoring."""
    aa_mask = np.isin(atoms.res_name, list(PROTEIN_AMINO_ACIDS))
    protein_atoms = atoms[aa_mask]
    if len(protein_atoms) == 0:
        return protein_atoms
    if n_ignore == 0 and c_ignore == 0:
        return protein_atoms

    chain_res_order = _build_chain_residue_order(protein_atoms)
    keep_by_chain: Dict[str, set] = {}
    for ch_s, res_list in chain_res_order.items():
        eff_n, eff_c = _effective_ignore(
            ch_s, n_ignore, c_ignore, n_trim_chains, c_trim_chains
        )
        if eff_c > 0:
            kept = res_list[eff_n:len(res_list) - eff_c]
        else:
            kept = res_list[eff_n:]
        keep_by_chain[ch_s] = set(kept)

    keep_mask = np.zeros(len(protein_atoms), dtype=bool)
    for ch_s, kept_ids in keep_by_chain.items():
        if not kept_ids:
            continue
        chain_mask = (protein_atoms.chain_id == ch_s)
        res_mask = np.isin(protein_atoms.res_id, list(kept_ids))
        keep_mask |= (chain_mask & res_mask)
    return protein_atoms[keep_mask]


def calculate_tm_score(
    pred_pdb_path: str,
    ref_pdb_path: str,
    n_ignore: int = 0,
    c_ignore: int = 0,
    verbose: bool = False,
    n_trim_chains: Optional[set] = None,
    c_trim_chains: Optional[set] = None
) -> float:
    """
    biotite TM-score between prediction and reference (length-normalized).

    TM > 0.5 generally indicates the same fold; TM > 0.17 better than random.
    """
    pred_atoms = load_biotite_structure(pred_pdb_path)
    ref_atoms = load_biotite_structure(ref_pdb_path)
    if pred_atoms is None or ref_atoms is None:
        vlog(verbose, "TM-score: failed to load structures with biotite", "WARN")
        return float("nan")

    pred_protein = filter_biotite_protein_residues(
        pred_atoms, n_ignore, c_ignore, n_trim_chains, c_trim_chains
    )
    ref_protein = filter_biotite_protein_residues(ref_atoms, 0, 0)

    pred_ca = pred_protein[pred_protein.atom_name == "CA"]
    ref_ca = ref_protein[ref_protein.atom_name == "CA"]

    n_pred = len(pred_ca)
    n_ref = len(ref_ca)
    if n_pred == 0 or n_ref == 0:
        vlog(verbose, "TM-score: no CA atoms found", "WARN")
        return float("nan")

    n_atoms = min(n_pred, n_ref)
    if n_pred != n_ref:
        vlog(verbose, f"TM-score: truncating to {n_atoms} CA (pred={n_pred}, ref={n_ref})", "WARN")
    pred_ca = pred_ca[:n_atoms]
    ref_ca = ref_ca[:n_atoms]
    indices = np.arange(n_atoms)

    try:
        pred_ca_superimposed, _ = struc.superimpose(ref_ca, pred_ca)
        tm_val = struc.tm_score(
            ref_ca, pred_ca_superimposed, indices, indices, reference_length=n_ref
        )
        vlog(verbose, f"TM-score: {tm_val:.4f} (n_CA={n_atoms})")
        return float(tm_val)
    except Exception as e:
        vlog(verbose, f"TM-score calculation failed: {e}", "WARN")
        return float("nan")


def _get_symmetric_atom_names(resname: str) -> set:
    """Set of all symmetric atom names for a residue type."""
    if resname not in SYMMETRIC_ATOM_PAIRS:
        return set()
    symmetric = set()
    for atom1, atom2 in SYMMETRIC_ATOM_PAIRS[resname]:
        symmetric.add(atom1)
        symmetric.add(atom2)
    return symmetric


def _apply_swap_to_name(name: str, resname: str) -> str:
    """Apply symmetric swap to an atom name."""
    if resname not in SYMMETRIC_ATOM_PAIRS:
        return name
    for atom1, atom2 in SYMMETRIC_ATOM_PAIRS[resname]:
        if name == atom1:
            return atom2
        elif name == atom2:
            return atom1
    return name


def _determine_best_swap(
    pred_res_atoms,
    ref_res_atoms,
    resname: str,
    verbose: bool = False
) -> bool:
    """Whether swapping symmetric atoms gives better correspondence."""
    if resname not in SYMMETRIC_ATOM_PAIRS:
        return False
    symmetric_atoms = _get_symmetric_atom_names(resname)

    pred_name_to_coord = {pred_res_atoms.atom_name[i]: pred_res_atoms.coord[i]
                          for i in range(len(pred_res_atoms))}
    ref_name_to_coord = {ref_res_atoms.atom_name[i]: ref_res_atoms.coord[i]
                         for i in range(len(ref_res_atoms))}

    common_names = set(pred_name_to_coord.keys()) & set(ref_name_to_coord.keys())
    non_symmetric_common = [n for n in common_names if n not in symmetric_atoms]
    if len(non_symmetric_common) < 3:
        return False

    pred_align_coords = np.array([pred_name_to_coord[n] for n in sorted(non_symmetric_common)])
    ref_align_coords = np.array([ref_name_to_coord[n] for n in sorted(non_symmetric_common)])

    pred_center = np.mean(pred_align_coords, axis=0)
    ref_center = np.mean(ref_align_coords, axis=0)
    pred_centered = pred_align_coords - pred_center
    ref_centered = ref_align_coords - ref_center

    H = pred_centered.T @ ref_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    def transform(coord):
        return R @ (coord - pred_center) + ref_center

    symmetric_common = [n for n in common_names if n in symmetric_atoms]
    if not symmetric_common:
        return False

    rmsd_orig = 0.0
    for name in symmetric_common:
        pred_transformed = transform(pred_name_to_coord[name])
        rmsd_orig += np.sum((pred_transformed - ref_name_to_coord[name]) ** 2)
    rmsd_orig = np.sqrt(rmsd_orig / len(symmetric_common))

    rmsd_swap = 0.0
    for name in symmetric_common:
        swapped_name = _apply_swap_to_name(name, resname)
        if swapped_name in ref_name_to_coord:
            pred_transformed = transform(pred_name_to_coord[name])
            rmsd_swap += np.sum((pred_transformed - ref_name_to_coord[swapped_name]) ** 2)
        else:
            rmsd_swap += np.sum((transform(pred_name_to_coord[name]) - ref_name_to_coord[name]) ** 2)
    rmsd_swap = np.sqrt(rmsd_swap / len(symmetric_common))
    return rmsd_swap < rmsd_orig


def _get_residue_coords_with_swap(
    pred_res_atoms,
    ref_res_atoms,
    resname: str,
    swap: bool = False
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """Matched coords between prediction and reference residue (optional swap)."""
    pred_name_to_coord = {}
    for i, name in enumerate(pred_res_atoms.atom_name):
        pred_name_to_coord[name] = pred_res_atoms.coord[i]

    if swap and resname in SYMMETRIC_ATOM_PAIRS:
        swapped_mapping = {}
        for name, coord in pred_name_to_coord.items():
            swapped_name = _apply_swap_to_name(name, resname)
            swapped_mapping[swapped_name] = coord
        pred_name_to_coord = swapped_mapping

    ref_name_to_coord = {}
    for i, name in enumerate(ref_res_atoms.atom_name):
        ref_name_to_coord[name] = ref_res_atoms.coord[i]

    common_names = set(pred_name_to_coord.keys()) & set(ref_name_to_coord.keys())
    pred_coords, ref_coords, atom_names = [], [], []
    for name in sorted(common_names):
        pred_coords.append(pred_name_to_coord[name])
        ref_coords.append(ref_name_to_coord[name])
        atom_names.append(name)
    return pred_coords, ref_coords, atom_names


def _compute_simple_lddt(
    ref_coords: np.ndarray,
    pred_coords: np.ndarray,
    inclusion_radius: float = 15.0,
    thresholds: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
) -> float:
    """Simplified lDDT between reference and predicted coordinate sets (0-1)."""
    n_atoms = len(ref_coords)
    if n_atoms < 2:
        return float("nan")

    ref_dists = np.zeros((n_atoms, n_atoms))
    pred_dists = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            ref_dists[i, j] = np.linalg.norm(ref_coords[i] - ref_coords[j])
            ref_dists[j, i] = ref_dists[i, j]
            pred_dists[i, j] = np.linalg.norm(pred_coords[i] - pred_coords[j])
            pred_dists[j, i] = pred_dists[i, j]

    mask = (ref_dists > 0) & (ref_dists < inclusion_radius)
    n_pairs = np.sum(mask) // 2
    if n_pairs == 0:
        return float("nan")

    deviations = np.abs(ref_dists - pred_dists)
    scores = []
    for thresh in thresholds:
        preserved = np.sum((deviations < thresh) & mask) // 2
        scores.append(preserved / n_pairs)
    return float(np.mean(scores))


def calculate_catres_lddt(
    pred_pdb_path: str,
    ref_pdb_path: str,
    catres_list: List[Dict],
    verbose: bool = False
) -> float:
    """
    biotite-loaded lDDT over heavy atoms of all catalytic residues together (AF3
    semantics). Catres are resolved by literal resseq (AF2 numbering policy).

    Symmetric residues use the best (swapped/unswapped) atom assignment.
    """
    if not catres_list:
        return float("nan")
    pred_atoms = load_biotite_structure(pred_pdb_path)
    ref_atoms = load_biotite_structure(ref_pdb_path)
    if pred_atoms is None or ref_atoms is None:
        vlog(verbose, "lDDT: failed to load structures with biotite", "WARN")
        return float("nan")

    all_pred_coords = []
    all_ref_coords = []
    for catres in catres_list:
        res_num = catres["res_num"]
        resname = catres["name3"]
        ch_id = catres["chain"]
        icode = catres.get("icode", ' ')

        pred_res_mask = ((pred_atoms.chain_id == ch_id) &
                         (pred_atoms.res_id == int(res_num)) &
                         (pred_atoms.element != "H"))
        ref_res_mask = ((ref_atoms.chain_id == ch_id) &
                        (ref_atoms.res_id == int(res_num)) &
                        (ref_atoms.element != "H"))
        if icode != ' ' and hasattr(pred_atoms, "ins_code"):
            pred_res_mask = pred_res_mask & (pred_atoms.ins_code == icode)
            ref_res_mask = ref_res_mask & (ref_atoms.ins_code == icode)
        pred_res_atoms = pred_atoms[pred_res_mask]
        ref_res_atoms = ref_atoms[ref_res_mask]
        if len(pred_res_atoms) == 0 or len(ref_res_atoms) == 0:
            vlog(verbose, f"  Catres {ch_id}:{res_num}: no atoms found, skipping", "WARN")
            continue

        use_swapped = False
        if resname in SYMMETRIC_ATOM_PAIRS:
            use_swapped = _determine_best_swap(pred_res_atoms, ref_res_atoms, resname, verbose)

        pred_coords, ref_coords, names = _get_residue_coords_with_swap(
            pred_res_atoms, ref_res_atoms, resname, swap=use_swapped
        )
        for pc, rc in zip(pred_coords, ref_coords):
            all_pred_coords.append(pc)
            all_ref_coords.append(rc)

    if len(all_pred_coords) < 2:
        vlog(verbose, "lDDT: not enough atoms across catalytic residues", "WARN")
        return float("nan")

    try:
        lddt_val = _compute_simple_lddt(np.array(all_ref_coords), np.array(all_pred_coords))
        vlog(verbose, f"Catres lDDT: {lddt_val:.4f} ({len(all_pred_coords)} atoms)")
        return lddt_val
    except Exception as e:
        vlog(verbose, f"Catres lDDT calculation failed: {e}", "WARN")
        return float("nan")


# ==============================================================================
# SECTION 8: pLDDT (B-FACTOR) PER-RESIDUE READ  (our read of the pred PDB)
# ==============================================================================

def get_plddt_per_residue_dic(pred_pdb: str) -> Dict[Tuple[str, int, str], float]:
    """
    Per-residue mean pLDDT from the B-factor column, keyed by
    (chain, raw_resseq, icode).

    AF2/superfold preserves input numbering, so catres pLDDT is looked up by
    literal resseq (with insertion code) -- no effective-index remapping is
    needed here. The blank insertion code is normalized to a single space.
    """
    plddt_dic: Dict[Tuple[str, int, str], Dict[str, float]] = {}
    with open(pred_pdb, 'r') as fh:
        for line in fh:
            record_type = line[PDB_RECORD_TYPE_COLS[0]:PDB_RECORD_TYPE_COLS[1]]
            if record_type not in ("ATOM  ", "HETATM"):
                continue
            if len(line.rstrip("\n")) < PDB_PLDDT_COLS[1]:
                continue
            ch_id = line[PDB_CHAIN_ID_COLS[0]:PDB_CHAIN_ID_COLS[1]].strip()
            icode = line[PDB_ICODE_COL[0]:PDB_ICODE_COL[1]].strip() or ' '
            try:
                raw_res_num = int(line[PDB_RESIDUE_NUM_COLS[0]:PDB_RESIDUE_NUM_COLS[1]].strip())
                plddt = float(line[PDB_PLDDT_COLS[0]:PDB_PLDDT_COLS[1]])
            except ValueError:
                continue
            atom_name = line[PDB_ATOM_NAME_COLS[0]:PDB_ATOM_NAME_COLS[1]].strip()
            key = (ch_id, raw_res_num, icode)
            plddt_dic.setdefault(key, {})[atom_name] = plddt

    return {key: float(np.average(list(vals.values()))) for key, vals in plddt_dic.items()}


# ==============================================================================
# SECTION 9: AF2 JSON PARSING  (safe_loads reused from af2_parse_processing_multi)
# ==============================================================================

def safe_loads(json_str: str):
    """
    Parse a string as JSON, tolerating trailing garbage after a valid object.

    Reused verbatim from af2_parse_processing_multi.py. Returns (obj, extra).
    """
    try:
        return json.loads(json_str), None
    except json.JSONDecodeError as e:
        if e.pos < len(json_str):
            return json.loads(json_str[:e.pos]), json_str[e.pos:]
        raise


def _reduce_scalar(value, reducer):
    """Reduce a (possibly list-valued) AF2 JSON field to a single float."""
    if isinstance(value, list):
        if not value:
            return float("nan")
        try:
            return float(reducer(value))
        except (TypeError, ValueError):
            return float("nan")
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_af2_json(json_path: str, want_pae: bool = True) -> Dict[str, Any]:
    """
    Centralized AF2 JSON parser.

    Captures every documented field under an ``af2_`` name. List-valued numeric
    fields are reduced consistently: plddt -> max, paes/rmsd/tol/elapsed -> min,
    ptm -> max. Returns a dict including ``af2_pae`` (np.ndarray or None) and
    ``af2_pae_length``. ``want_pae=False`` skips parsing the LxL matrix for speed.

    On any failure returns {"_error": "<reason>"} plus NaN/"" placeholders.
    """
    out: Dict[str, Any] = {}
    try:
        with open(json_path, "r") as fh:
            data, extra = safe_loads(fh.read())
    except Exception as e:
        return {"_error": f"json_parse_fail:{type(e).__name__}"}

    if extra:
        out["_warn_extra_json"] = True

    out["af2_mean_plddt"] = _reduce_scalar(data.get("mean_plddt"), max)
    out["af2_ptm_score"] = _reduce_scalar(data.get("pTMscore"), max)
    out["af2_mean_pae"] = _reduce_scalar(data.get("mean_pae"), min)
    out["af2_mean_pae_intra_chain"] = _reduce_scalar(data.get("mean_pae_intra_chain"), min)
    out["af2_mean_pae_intra_chain_A"] = _reduce_scalar(data.get("mean_pae_intra_chain_A"), min)
    out["af2_mean_pae_interaction"] = _reduce_scalar(data.get("mean_pae_interaction"), min)
    out["af2_rmsd_to_input"] = _reduce_scalar(data.get("rmsd_to_input"), min)
    out["af2_tol"] = _reduce_scalar(data.get("tol"), min)
    out["af2_elapsed_time"] = _reduce_scalar(data.get("elapsed_time"), min)

    recycles = data.get("recycles")
    if isinstance(recycles, list):
        recycles = max(recycles) if recycles else None
    try:
        out["af2_recycles"] = int(recycles) if recycles is not None else ""
    except (TypeError, ValueError):
        out["af2_recycles"] = ""

    # String-valued metadata.
    for json_key, col in (("model", "af2_model"), ("type", "af2_type"), ("seed", "af2_seed")):
        val = data.get(json_key)
        out[col] = "" if val is None else str(val)

    # Full LxL PAE matrix (optional).
    pae = None
    pae_len = 0
    if want_pae:
        pae_list = data.get("pae")
        if pae_list is not None:
            try:
                pae = np.array(
                    [[(float(x) if x is not None else np.nan) for x in row] for row in pae_list],
                    dtype=float,
                )
                if pae.ndim == 2 and pae.shape[0] == pae.shape[1]:
                    pae_len = int(pae.shape[0])
                else:
                    pae = None
                    pae_len = 0
            except Exception:
                pae = None
                pae_len = 0
    else:
        pae_list = data.get("pae")
        if isinstance(pae_list, list):
            pae_len = len(pae_list)

    out["af2_pae"] = pae
    out["af2_pae_length"] = pae_len
    return out


# ==============================================================================
# SECTION 10: PAE ROW MAP + PAE METRICS  (P0 #2, #3)
# ==============================================================================

def build_pae_row_map(
    structure: PDB.Structure.Structure,
    n_ignore: int,
    c_ignore: int,
    n_trim_chains: Optional[set] = None,
    c_trim_chains: Optional[set] = None
) -> Dict[Tuple[str, int, str], int]:
    """
    Map (chain, resseq, icode) -> 0-based PAE row index.

    The PAE matrix is indexed by the UNTRIMMED predicted protein residue order
    (first-appearance), so the ordinal is assigned BEFORE terminal trimming. The
    (n_ignore, c_ignore) effective trim (chain A only by default) is applied only
    to decide which residues map to a row at all -- but the row index itself is
    the position in the full, untrimmed protein-residue ordering, matching how
    AF2 emits the LxL matrix over every predicted residue.
    """
    row_map: Dict[Tuple[str, int, str], int] = {}
    ordinal = 0
    for model in structure:
        for chain in model:
            eff_n, eff_c = _effective_ignore(
                chain.id, n_ignore, c_ignore, n_trim_chains, c_trim_chains
            )
            protein_residues = [r for r in chain.get_residues() if is_protein_residue(r)]
            n_res = len(protein_residues)
            for i, residue in enumerate(protein_residues):
                # Skip trimmed terminal residues for THIS chain (mirrors the
                # effective mapping used by the RMSD/pLDDT/CA path).
                if i < eff_n:
                    ordinal += 1
                    continue
                if eff_c > 0 and i >= n_res - eff_c:
                    ordinal += 1
                    continue
                hetflag, resseq, icode = residue.get_id()
                row_map[(chain.id, int(resseq), icode)] = ordinal
                ordinal += 1
        break  # model 1 only
    return row_map


def pae_row_for_catres(
    catres: Dict[str, Any],
    row_map: Dict[Tuple[str, int, str], int]
) -> Optional[int]:
    """Resolve the PAE row index for a catres (literal resseq + icode)."""
    return row_map.get(
        (catres["chain"], int(catres["res_num"]), catres.get("icode", ' '))
    )


def catres_row_mean_pae(pae: np.ndarray, row_idx: int) -> float:
    """Mean PAE of one catres' row over ALL residues (global active-site error)."""
    if pae is None or row_idx is None or row_idx >= pae.shape[0]:
        return float("nan")
    row = pae[row_idx, :]
    finite = np.isfinite(row)
    return float(np.mean(row[finite])) if np.any(finite) else float("nan")


def symmetric_pair_pae(pae: np.ndarray, i: int, j: int) -> float:
    """Symmetric-averaged PAE for a residue pair: 0.5*(pae[i][j]+pae[j][i])."""
    if pae is None or i is None or j is None:
        return float("nan")
    if i >= pae.shape[0] or j >= pae.shape[0]:
        return float("nan")
    a = pae[i, j]
    b = pae[j, i]
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float(0.5 * (a + b))


# ==============================================================================
# SECTION 11: RESOLVED JOB + REFERENCE RESOLVER
# ==============================================================================

@dataclass
class ResolvedJob:
    """A fully-resolved unit of work: one predicted PDB paired with its ref."""
    pred_pdb: str
    json_path: str
    ref_pdb: Optional[str]
    out_sc: str
    description: str
    error: str = ""


AUTO_STRIP_RE = re.compile(r"_model_\d+_ptm_seed_\d+(_unrelaxed)?$")


def derive_json_from_pdb(pred_pdb: str) -> str:
    """Derive the AF2 JSON path from the predicted PDB path (P0 #5)."""
    if pred_pdb.endswith(PRED_PDB_SUFFIX):
        return pred_pdb[:-len(PRED_PDB_SUFFIX)] + PRED_JSON_SUFFIX
    base, _ = os.path.splitext(pred_pdb)
    return base + PRED_JSON_SUFFIX


def pred_basename(pred_pdb: str) -> str:
    """Full prediction basename (no extension, no _unrelaxed) for ``description``."""
    name = os.path.basename(pred_pdb)
    if name.endswith(PRED_PDB_SUFFIX):
        return name[:-len(PRED_PDB_SUFFIX)]
    return os.path.splitext(name)[0]


class ReferenceResolver:
    """
    Resolve a reference PDB for each predicted PDB. Four modes:

      explicit       --ref_pdb FILE                (one ref for everything)
      suffix_strip   --ref_dir + --af2_suffix STR  (strip EXACT suffix, then
                                                     <ref_dir>/<stripped>.pdb)
      auto_strip     --ref_dir                     (strip the AF2 model/seed tail
                                                     via anchored regex, then
                                                     exact <stripped>.pdb)
      scorefile      handled by the scorefile driver per-row.

    ``resolve(pred_pdb)`` returns (ref_path_or_None, error_str). Anchored known
    suffix is tried FIRST, then exact ``<stripped>.pdb``. If multiple candidate
    refs could match (collision), an error string is returned (codex #4).
    """

    def __init__(
        self,
        mode: str,
        ref_pdb: Optional[str] = None,
        ref_dir: Optional[str] = None,
        af2_suffix: Optional[str] = None,
    ):
        self.mode = mode
        self.ref_pdb = ref_pdb
        self.ref_dir = ref_dir
        self.af2_suffix = af2_suffix
        self._ref_index: Optional[Dict[str, List[str]]] = None

    def _index_ref_dir(self) -> Dict[str, List[str]]:
        """Build {basename_no_ext: [paths...]} for collision detection."""
        if self._ref_index is not None:
            return self._ref_index
        index: Dict[str, List[str]] = {}
        if self.ref_dir:
            for p in glob.glob(os.path.join(self.ref_dir, "*.pdb")):
                stem = os.path.splitext(os.path.basename(p))[0]
                index.setdefault(stem, []).append(p)
        self._ref_index = index
        return index

    def _lookup_stem(self, stem: str) -> Tuple[Optional[str], str]:
        """Resolve a stripped stem to a unique ref path in ref_dir."""
        index = self._index_ref_dir()
        hits = index.get(stem, [])
        if len(hits) == 1:
            return hits[0], ""
        if len(hits) > 1:
            return None, f"ref_collision:{stem}:{len(hits)}_candidates"
        # Fall back to a direct path probe (handles refs added after indexing).
        candidate = os.path.join(self.ref_dir, f"{stem}.pdb")
        if os.path.isfile(candidate):
            return candidate, ""
        return None, f"missing_ref:{stem}.pdb"

    def resolve(self, pred_pdb: str) -> Tuple[Optional[str], str]:
        if self.mode == "explicit":
            if self.ref_pdb and os.path.isfile(self.ref_pdb):
                return self.ref_pdb, ""
            return None, "missing_ref:explicit"

        base = pred_basename(pred_pdb)

        if self.mode == "suffix_strip":
            suffix = self.af2_suffix or ""
            if suffix and base.endswith(suffix):
                stem = base[:-len(suffix)]
            elif suffix:
                return None, f"suffix_not_found:{suffix}"
            else:
                stem = base
            return self._lookup_stem(stem)

        if self.mode == "auto_strip":
            stem = AUTO_STRIP_RE.sub("", base)
            if stem == base:
                # Regex didn't match the tail; try the raw base as a last resort.
                stem = base
            return self._lookup_stem(stem)

        return None, f"unknown_resolver_mode:{self.mode}"


# ==============================================================================
# SECTION 12: PURE WORKER  --  analyze_prediction
# ==============================================================================

def analyze_prediction(
    pred_pdb: str,
    ref_pdb: Optional[str],
    json_path: str,
    opts: Dict[str, Any],
    ref_cache: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    PURE worker: compute the full row dict for one prediction. No file writes, no
    argparse, no prints except via the verbose flag in ``opts``.

    opts keys: verbose, tmalign, pae_full, no_pae, n_ignore, c_ignore,
               catres_subset (list[int] or None), no_per_catres, lean.
    ref_cache: optional per-worker dict {ref_path: {"structure", "catres"}}.

    The optional ``no_per_catres`` / ``lean`` keys are applied ONLY as a final
    post-build filter on the returned row dict (see ``_apply_column_trim``); the
    full computation is performed regardless so the verbose REPORT (and any
    aggregates) always reflect every value. ``lean`` wins over ``no_per_catres``.

    Returns (row_dict, warnings). row_dict["status"] is "ok" or "error".
    """
    verbose = opts.get("verbose", False)
    want_pae = not opts.get("no_pae", False)
    description = pred_basename(pred_pdb)

    row: Dict[str, Any] = {
        "description": description,
        "pdb_path": pred_pdb,
        "af2_json_path": json_path,
        "ref_path": ref_pdb or "",
        "status": "ok",
        "error": "",
        "catres_signature": "",
    }
    warnings: List[str] = []
    errors: List[str] = []

    def fail(reason: str) -> Tuple[Dict[str, Any], List[str]]:
        row["status"] = "error"
        row["error"] = reason
        if opts.get("verbose", False):
            _emit_verbose_report(row, [], [], opts, warnings)
        return _apply_column_trim(row, [], opts), warnings

    # --- Input validation -----------------------------------------------------
    if not os.path.isfile(pred_pdb):
        return fail("missing_pred_pdb")
    if not os.path.isfile(json_path):
        return fail("missing_json")
    if not ref_pdb or not os.path.isfile(ref_pdb):
        return fail("missing_ref")

    # --- AF2 JSON scalars + PAE ----------------------------------------------
    af2 = parse_af2_json(json_path, want_pae=want_pae)
    if "_error" in af2:
        return fail(af2["_error"])
    if af2.pop("_warn_extra_json", False):
        warnings.append("extra_json_data")
    pae = af2.pop("af2_pae", None)
    pae_length = af2.get("af2_pae_length", 0)
    for k, v in af2.items():
        row[k] = v

    # --- Load structures (cache the reference) --------------------------------
    try:
        pred_structure_orig = load_pdb(pred_pdb)
    except Exception as e:
        return fail(f"pred_load_fail:{type(e).__name__}")

    if ref_cache is not None and ref_pdb in ref_cache:
        ref_structure = ref_cache[ref_pdb]["structure"]
        catres_list = ref_cache[ref_pdb]["catres"]
    else:
        try:
            ref_structure = load_pdb(ref_pdb)
        except Exception as e:
            return fail(f"ref_load_fail:{type(e).__name__}")
        catres_list = get_catalytic_residues(ref_pdb, verbose=verbose)
        if ref_cache is not None:
            ref_cache[ref_pdb] = {"structure": ref_structure, "catres": catres_list}

    if verbose:
        n_pred_res = sum(
            1 for model in pred_structure_orig for chain in model
            for r in chain.get_residues() if is_protein_residue(r)
        )
        n_ref_res = sum(
            1 for model in ref_structure for chain in model
            for r in chain.get_residues() if is_protein_residue(r)
        )
        vlog(True, f"  loaded pred={n_pred_res} residues, ref={n_ref_res} residues "
                   f"(pred={os.path.basename(pred_pdb)}, ref={os.path.basename(ref_pdb)})")

    # --- Multichain warn + proceed (P0 #6) -----------------------------------
    pred_chains = [chain.id for model in pred_structure_orig for chain in model
                   if any(is_protein_residue(r) for r in chain.get_residues())]
    n_protein_chains = len(set(pred_chains))
    row["n_protein_chains"] = n_protein_chains
    if n_protein_chains > 1:
        warnings.append(f"multichain:{n_protein_chains}")
        log(f"[{description}] WARNING: {n_protein_chains} protein chains detected; "
            f"global CA align over all chains, catres on their own chains.", "WARN")

    # --- Catres signature (audit) --------------------------------------------
    sig_parts = []
    for catres in catres_list:
        sig_parts.append(f"{catres['chain']}/{catres['name3']}/{catres['res_num']}/cst{catres['cst_block']}")
    row["catres_signature"] = "|".join(sig_parts)
    catres_count = len(catres_list)
    row["catres_count"] = catres_count

    n_ignore = int(opts.get("n_ignore", 0))
    c_ignore = int(opts.get("c_ignore", 0))

    # --- Global CA Kabsch with terminal-ignore fallback (P0 #4: copy per attempt) ---
    ref_ca = extract_ca_coords(ref_structure, 0, 0)
    if len(ref_ca) == 0:
        return fail("ref_no_ca")

    fallback_pairs = [(n_ignore, c_ignore)]
    if n_ignore > 0 or c_ignore > 0:
        fallback_pairs.append((0, 0))
        if n_ignore > 0 and c_ignore > 0 and n_ignore != c_ignore:
            fallback_pairs.append((n_ignore, 0))
            fallback_pairs.append((0, c_ignore))

    aligned_structure = None
    alignment_ok = False
    used_n, used_c = n_ignore, c_ignore
    ca_rmsd = float("nan")
    for attempt_n, attempt_c in fallback_pairs:
        # Deep-copy so a failed/partial alignment can't contaminate later attempts.
        attempt_struct = copy.deepcopy(pred_structure_orig)
        pred_ca = extract_ca_coords(attempt_struct, attempt_n, attempt_c)
        if len(pred_ca) == 0 or len(pred_ca) != len(ref_ca):
            vlog(verbose, f"  CA mismatch at (N={attempt_n}, C={attempt_c}): "
                          f"pred={len(pred_ca)} ref={len(ref_ca)}", "DEBUG")
            continue
        rot, c1, c2 = kabsch_alignment(pred_ca, ref_ca)
        for model in attempt_struct:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atom.set_coord(
                            apply_rotation(np.array([atom.get_coord()]), rot, c1, c2)[0]
                        )
        pred_ca_aligned = extract_ca_coords(attempt_struct, attempt_n, attempt_c)
        ca_rmsd = calculate_rmsd(pred_ca_aligned, ref_ca)
        aligned_structure = attempt_struct
        alignment_ok = True
        used_n, used_c = attempt_n, attempt_c
        if (attempt_n, attempt_c) != (n_ignore, c_ignore):
            warnings.append(f"terminal_fallback:N={attempt_n},C={attempt_c}")
        break

    row["terminal_ignore_N"] = used_n
    row["terminal_ignore_C"] = used_c
    row["ca_rmsd"] = ca_rmsd

    if alignment_ok:
        vlog(verbose, f"  CA Kabsch aligned on {len(ref_ca)} CA atoms; "
                      f"ca_rmsd={ca_rmsd:.3f} (used N={used_n},C={used_c})")
    else:
        vlog(verbose, f"  CA-count mismatch pred=? ref={len(ref_ca)} across all "
                      f"fallbacks -> per-catres RMSDs = NaN")

    if aligned_structure is None:
        # CA count never matched -> alignment-dependent metrics are NaN, but we
        # still emit AF2 scalars and PAE so the row is informative.
        errors.append(f"ca_count_mismatch(pred!=ref={len(ref_ca)})")
        aligned_structure = copy.deepcopy(pred_structure_orig)

    # --- Catres find + name validation ONCE up front (FIX 2) -----------------
    # A catres is "valid" only if BOTH the pred and the ref residue are present
    # at its (chain, resseq, icode) AND both carry the expected REMARK name3.
    # ALL per-catres metrics (incl. plddt/pae and lDDT) are gated on this so a
    # missing / mutated / wrong residue can never leak a misleading number.
    catres_valid: List[bool] = []
    catres_pred_res: List[Any] = []
    catres_ref_res: List[Any] = []
    for catres in catres_list:
        name3 = catres["name3"]
        ch_id = catres["chain"]
        res_num = catres["res_num"]
        icode = catres.get("icode", ' ')
        pred_res = find_residue_by_resseq(aligned_structure, ch_id, res_num, icode)
        ref_res = find_residue_by_resseq(ref_structure, ch_id, res_num, icode)
        valid = (
            pred_res is not None and ref_res is not None
            and pred_res.get_resname() == name3
            and ref_res.get_resname() == name3
        )
        catres_valid.append(valid)
        catres_pred_res.append(pred_res)
        catres_ref_res.append(ref_res)
        if verbose:
            tag = f"{name3}{len(catres_valid)} ({ch_id}/{name3}/{res_num})"
            if valid:
                vlog(True, f"  catres {tag}: valid")
            elif pred_res is None or ref_res is None:
                vlog(True, f"  catres {tag}: not_found "
                           f"(pred={'-' if pred_res is None else pred_res.get_resname()},"
                           f"ref={'-' if ref_res is None else ref_res.get_resname()})")
            else:
                vlog(True, f"  catres {tag}: name_mismatch "
                           f"(expected={name3},ref={ref_res.get_resname()},"
                           f"pred={pred_res.get_resname()})")

    # Only validated catres enter lDDT (catres_lddt + catres_subset_lddt) so a
    # mutated/wrong residue cannot contaminate the lDDT metric (FIX 2).
    validated_catres = [c for c, v in zip(catres_list, catres_valid) if v]

    # --- TM-score (length-normalized fold metric) ----------------------------
    import tempfile
    tmp_pred_path = None
    tm_score_val = float("nan")
    ca_rmsd_tmalign = float("nan")
    catres_lddt_val = float("nan")
    catres_subset_lddt_val = float("nan")
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmpf:
            tmp_pred_path = tmpf.name
        save_pdb(aligned_structure, tmp_pred_path)

        tm_score_val = calculate_tm_score(tmp_pred_path, ref_pdb, used_n, used_c, verbose)
        # Only validated catres (present + name-matched in both) enter lDDT.
        catres_lddt_val = calculate_catres_lddt(tmp_pred_path, ref_pdb, validated_catres, verbose)

        subset_blocks = opts.get("catres_subset")
        subset_catres = None
        if subset_blocks:
            subset_catres = [c for c in validated_catres if c.get("cst_block", 0) in subset_blocks]
            if subset_catres:
                catres_subset_lddt_val = calculate_catres_lddt(tmp_pred_path, ref_pdb, subset_catres, verbose)
            else:
                catres_subset_lddt_val = float("nan")
        else:
            catres_subset_lddt_val = catres_lddt_val

        if opts.get("tmalign", False):
            ca_rmsd_tmalign = _tmalign_ca_rmsd(tmp_pred_path, ref_pdb, used_n, used_c, verbose)
    except Exception as e:
        vlog(verbose, f"TM-score/lDDT block failed: {e}", "WARN")
    finally:
        if tmp_pred_path and os.path.exists(tmp_pred_path):
            os.unlink(tmp_pred_path)

    row["tm_score"] = tm_score_val
    if opts.get("tmalign", False):
        row["ca_rmsd_TMalign"] = ca_rmsd_tmalign

    # --- Per-catres RMSD / bb-RMSD / pLDDT / PAE (literal resseq, P0 #1) ------
    plddt_dic = get_plddt_per_residue_dic(pred_pdb)
    row_map = build_pae_row_map(aligned_structure, used_n, used_c) if pae is not None else {}

    catres_heavy_rmsds: List[float] = []
    catres_bb_rmsds: List[float] = []
    catres_plddts: List[float] = []
    catres_pae_rows: List[float] = []
    catres_row_idx: List[Optional[int]] = []

    for i, catres in enumerate(catres_list):
        idx1 = i + 1
        name3 = catres["name3"]
        ch_id = catres["chain"]
        res_num = catres["res_num"]
        icode = catres.get("icode", ' ')
        col_prefix = f"{name3}{idx1}"

        valid = catres_valid[i]
        pred_res = catres_pred_res[i]
        ref_res = catres_ref_res[i]

        # pLDDT / PAE are alignment-invariant: emitted for any VALID catres even
        # when the global CA alignment failed. They are NaN only when the catres
        # itself is invalid (missing / name-mismatch). (FIX 2, FIX 3)
        cat_plddt = float("nan")
        cat_pae = float("nan")
        # Invalid catres contribute to NO PAE metric (row-mean OR pair); keep
        # their row index None so pair aggregates skip them too. (FIX 2)
        ridx = None
        if valid:
            ridx = pae_row_for_catres(catres, row_map) if pae is not None else None
            cat_plddt = plddt_dic.get((ch_id, int(res_num), icode), float("nan"))
            cat_pae = catres_row_mean_pae(pae, ridx) if pae is not None else float("nan")
        catres_row_idx.append(ridx)

        heavy_rmsd = float("nan")
        bb_rmsd = float("nan")
        # Per-catres RMSDs require BOTH a valid catres AND a successful global
        # superposition (they live in the Kabsch frame). (FIX 2, FIX 3)
        if not valid:
            if pred_res is None or ref_res is None:
                errors.append(f"{col_prefix}:not_found")
            else:
                errors.append(
                    f"{col_prefix}:name_mismatch(expected={name3},"
                    f"ref={ref_res.get_resname()},pred={pred_res.get_resname()})"
                )
        elif not alignment_ok:
            # ca_count_mismatch already recorded once globally; RMSDs stay NaN.
            pass
        else:
            pred_coords, pred_names = _residue_heavy_atoms(pred_res)
            ref_coords, ref_names = _residue_heavy_atoms(ref_res)
            if len(pred_coords) == 0 or len(ref_coords) == 0:
                errors.append(f"{col_prefix}:no_heavy_atoms")
            else:
                rmsd_val, pred_only, ref_only, common_used, _ = rmsd_on_common_atoms_with_symmetry(
                    pred_coords, pred_names, ref_coords, ref_names, name3
                )
                if not common_used:
                    errors.append(f"{col_prefix}:no_common_atoms")
                else:
                    heavy_rmsd = rmsd_val
                # Surface (but tolerate) missing atoms; RMSD stays on the common
                # atom set, matching AF3 behavior. (FIX 6b)
                if pred_only or ref_only:
                    warnings.append(
                        f"{col_prefix}:missing_atoms("
                        f"pred_only={len(pred_only)},ref_only={len(ref_only)})"
                    )
                bb_rmsd = backbone_rmsd_for_residue(pred_coords, pred_names, ref_coords, ref_names)

        row[f"{col_prefix}_rmsd"] = heavy_rmsd
        row[f"{col_prefix}_bb_rmsd"] = bb_rmsd
        row[f"{col_prefix}_plddt"] = cat_plddt
        row[f"{col_prefix}_pae"] = cat_pae

        catres_heavy_rmsds.append(heavy_rmsd)
        catres_bb_rmsds.append(bb_rmsd)
        catres_plddts.append(cat_plddt)
        catres_pae_rows.append(cat_pae)

    # --- Aggregate catres metrics (equal-per-residue avg; AF3 semantics) ------
    row["catres_rmsd"] = _nanmean(catres_heavy_rmsds)
    row["catres_bb_rmsd"] = _nanmean(catres_bb_rmsds)
    row["catres_lddt"] = catres_lddt_val
    row["catres_plddt"] = _nanmean(catres_plddts)
    row["catres_pae_to_all_mean"] = _nanmean(catres_pae_rows)

    pair_mean, pair_max, pair_min = _catres_pair_pae(pae, catres_row_idx)
    row["catres_pair_pae_mean"] = pair_mean
    row["catres_pair_pae_max"] = pair_max
    row["catres_pair_pae_min"] = pair_min

    # --- Optional full pairwise PAE (i<j; symmetric average) ------------------
    if opts.get("pae_full", False) and pae is not None:
        for a in range(len(catres_list)):
            for b in range(a + 1, len(catres_list)):
                ia, ib = catres_row_idx[a], catres_row_idx[b]
                val = symmetric_pair_pae(pae, ia, ib)
                na = f"{catres_list[a]['name3']}{a + 1}"
                nb = f"{catres_list[b]['name3']}{b + 1}"
                row[f"{na}_{nb}_pae"] = val

    # --- Catres subset metrics (only if --catres_subset) ----------------------
    subset_blocks = opts.get("catres_subset")
    if subset_blocks:
        subset_idx = [i for i, c in enumerate(catres_list)
                      if c.get("cst_block", 0) in subset_blocks]
        row["catres_subset_count"] = len(subset_idx)
        if subset_idx:
            row["catres_subset_rmsd"] = _nanmean([catres_heavy_rmsds[i] for i in subset_idx])
            row["catres_subset_bb_rmsd"] = _nanmean([catres_bb_rmsds[i] for i in subset_idx])
            row["catres_subset_lddt"] = catres_subset_lddt_val
            row["catres_subset_plddt"] = _nanmean([catres_plddts[i] for i in subset_idx])
            row["catres_subset_pae_to_all_mean"] = _nanmean([catres_pae_rows[i] for i in subset_idx])
            sub_rows = [catres_row_idx[i] for i in subset_idx]
            sub_mean, sub_max, sub_min = _catres_pair_pae(pae, sub_rows)
            row["catres_subset_pair_pae_mean"] = sub_mean
            row["catres_subset_pair_pae_max"] = sub_max
            row["catres_subset_pair_pae_min"] = sub_min
        else:
            for col in ("catres_subset_rmsd", "catres_subset_bb_rmsd", "catres_subset_lddt",
                        "catres_subset_plddt", "catres_subset_pae_to_all_mean",
                        "catres_subset_pair_pae_mean", "catres_subset_pair_pae_max",
                        "catres_subset_pair_pae_min"):
                row[col] = float("nan")

    # --- Finalize error string ------------------------------------------------
    if errors:
        # We still emit the row; per-catres NaNs are already in place. Mark a
        # status note but keep status="ok" if the global alignment succeeded so
        # the AF2 scalars/global metrics aren't lost. Only a fatal absence flips
        # status to "error" (handled by fail()).
        row["error"] = ";".join(errors[:8]) + ("..." if len(errors) > 8 else "")

    # --- Verbose REPORT (Change 1): reflects the FULL row, BEFORE any trim -----
    if verbose:
        _emit_verbose_report(row, catres_list, catres_valid, opts, warnings)

    # --- Column trim (Change 2): post-build filter, computed values unchanged --
    row = _apply_column_trim(row, catres_list, opts)
    return row, warnings


# ------------------------------------------------------------------------------
# Verbose REPORT + column-trim helpers (additive: see CHANGE 1 / CHANGE 2)
# ------------------------------------------------------------------------------

def _fmt_num(v: Any, ndigits: int = 3) -> str:
    """Format a scalar to ~3-4 sig figs; 'NaN' for nan/None, str() otherwise."""
    if v is None or v == "":
        return "NaN"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "NaN"
    return f"{f:.{ndigits}f}"


def _per_catres_keys(catres_list: List[Dict[str, Any]]) -> List[str]:
    """Per-catalytic-residue column names (the {name3}{i}_* family) in row order.

    Reconstructs the exact key names analyze_prediction assigned for each catres
    i (1-based): {name3}{i}_rmsd, _bb_rmsd, _plddt, _pae. Used by the column-trim
    filter to drop ONLY these (not the catres_* aggregates).
    """
    keys: List[str] = []
    for i, catres in enumerate(catres_list):
        prefix = f"{catres['name3']}{i + 1}"
        keys.extend([f"{prefix}_rmsd", f"{prefix}_bb_rmsd",
                     f"{prefix}_plddt", f"{prefix}_pae"])
    return keys


def _pae_full_pair_keys(catres_list: List[Dict[str, Any]]) -> List[str]:
    """The --pae_full pairwise column names {name3a}{i}_{name3b}{j}_pae (i<j)."""
    keys: List[str] = []
    n = len(catres_list)
    for a in range(n):
        for b in range(a + 1, n):
            na = f"{catres_list[a]['name3']}{a + 1}"
            nb = f"{catres_list[b]['name3']}{b + 1}"
            keys.append(f"{na}_{nb}_pae")
    return keys


# Lean core column set (Change C). Order is preserved; only keys present in the
# row (after the Section-B auto-drops) are emitted. The subset aggregates are
# appended only when --catres_subset. Per the column policy, --lean KEEPS the
# pdb_path / af2_json_path / ref_path / af2_tol / af2_mean_pae_intra_chain columns
# but DROPS status/error (those stay only in full/no_per_catres output).
_LEAN_CORE_COLUMNS = [
    "description", "pdb_path", "af2_json_path", "ref_path",
    "catres_signature", "catres_count",
    "af2_mean_plddt", "af2_ptm_score", "af2_mean_pae", "af2_mean_pae_intra_chain",
    "af2_rmsd_to_input", "af2_tol",
    "ca_rmsd", "tm_score",
    "catres_rmsd", "catres_bb_rmsd", "catres_lddt", "catres_plddt",
    "catres_pae_to_all_mean",
    "catres_pair_pae_mean", "catres_pair_pae_max", "catres_pair_pae_min",
]
_LEAN_SUBSET_COLUMNS = [
    "catres_subset_count", "catres_subset_rmsd", "catres_subset_bb_rmsd",
    "catres_subset_lddt", "catres_subset_plddt", "catres_subset_pae_to_all_mean",
    "catres_subset_pair_pae_mean", "catres_subset_pair_pae_max",
    "catres_subset_pair_pae_min",
]

# Section B: columns dropped in ALL modes when they carry no information.
_MONOMER_REDUNDANT_COLUMNS = (
    "af2_mean_pae_intra_chain_A", "af2_mean_pae_interaction", "n_protein_chains",
)
_NO_TRIM_COLUMNS = ("terminal_ignore_N", "terminal_ignore_C")


def _apply_auto_drops(
    row: Dict[str, Any],
    opts: Dict[str, Any],
) -> Dict[str, Any]:
    """Section B: drop redundant columns in ALL modes (incl. default/full).

    - Monomer (n_protein_chains == 1): drop af2_mean_pae_intra_chain_A,
      af2_mean_pae_interaction, n_protein_chains.  (>1 keeps all three.)
    - No terminal trim (opts n_ignore == 0 AND c_ignore == 0): drop
      terminal_ignore_N, terminal_ignore_C.  (either > 0 keeps both.)

    Detection is computed BEFORE dropping n_protein_chains (its value is still in
    the row) and PRINTED when verbose. Returns a new dict with the keys removed.
    """
    verbose = opts.get("verbose", False)
    drop: set = set()

    # Monomer detection uses the still-present n_protein_chains value.
    n_chains = row.get("n_protein_chains")
    is_monomer = False
    try:
        is_monomer = int(n_chains) == 1
    except (TypeError, ValueError):
        is_monomer = False
    if is_monomer:
        present = [c for c in _MONOMER_REDUNDANT_COLUMNS if c in row]
        drop.update(present)
        if verbose and present:
            log(f"[system: monomer (1 protein chain)] auto-dropped redundant "
                f"columns: {', '.join(present)}")

    n_ignore = int(opts.get("n_ignore", 0))
    c_ignore = int(opts.get("c_ignore", 0))
    if n_ignore == 0 and c_ignore == 0:
        present = [c for c in _NO_TRIM_COLUMNS if c in row]
        drop.update(present)
        if verbose and present:
            log(f"[no terminal trim (N=0,C=0)] auto-dropped: {', '.join(present)}")

    if not drop:
        return row
    return {k: v for k, v in row.items() if k not in drop}


def _apply_column_trim(
    row: Dict[str, Any],
    catres_list: List[Dict[str, Any]],
    opts: Dict[str, Any],
) -> Dict[str, Any]:
    """Post-build column filter for the WRITTEN .sc (Sections B-E).

    Selects/drops keys only -- never touches computed values. Order of operations
    (spec section E):
      1. (row already fully assembled by the caller)
      2. Section-B auto-drops (ALL modes), printing the detection when verbose.
      3. --lean keeps only the C lean set (those present); elif --no_per_catres
         drops the per-catres + --pae_full pair columns. --lean wins over
         --no_per_catres. With neither flag, only the B auto-drops are applied.
    The verbose REPORT runs BEFORE this and sees the full row, so trimming never
    affects what is reported.
    """
    # Step 2: Section-B auto-drops apply in every mode.
    row = _apply_auto_drops(row, opts)

    # Step 3: explicit column-trim flags.
    if opts.get("lean", False):
        wanted = list(_LEAN_CORE_COLUMNS)
        if opts.get("catres_subset"):
            wanted += _LEAN_SUBSET_COLUMNS
        # Keep only lean keys that actually exist in the row, preserving order.
        return {k: row[k] for k in wanted if k in row}

    if opts.get("no_per_catres", False):
        drop = set(_per_catres_keys(catres_list)) | set(_pae_full_pair_keys(catres_list))
        return {k: v for k, v in row.items() if k not in drop}

    return row


def _emit_verbose_report(
    row: Dict[str, Any],
    catres_list: List[Dict[str, Any]],
    catres_valid: List[bool],
    opts: Dict[str, Any],
    warnings: List[str],
) -> None:
    """Print a grouped per-prediction REPORT (Change 1).

    Reflects the FULL computation (called BEFORE any column trim). Uses log() so
    the report is timestamped like every other line. Tolerant of a partial row
    (e.g. an early failure) -- absent keys render as 'NaN'/'-'.
    """
    g = row.get  # local alias

    desc = g("description", "?")
    log("=" * 16 + f" [{desc}] " + "=" * 16)

    pred_b = os.path.basename(str(g("pdb_path", "") or ""))
    ref_b = os.path.basename(str(g("ref_path", "") or ""))
    log(f"  inputs : pred={pred_b or '-'}  ref={ref_b or '-'}")

    status = g("status", "?")
    err = g("error", "") or ""
    if status == "error":
        log(f"  status : error: {err or '(unspecified)'}")
    else:
        log(f"  status : ok" + (f"   (notes: {err})" if err else ""))

    log(f"  chains : {g('n_protein_chains', '-')} protein chain(s)")
    log(f"  catres : {g('catres_count', 0)}  ->  {g('catres_signature', '') or '-'}")

    align = f"  align  : ca_rmsd={_fmt_num(g('ca_rmsd'))}  tm_score={_fmt_num(g('tm_score'))}"
    if opts.get("tmalign", False):
        align += f"  ca_rmsd_TMalign={_fmt_num(g('ca_rmsd_TMalign'))}"
    align += f"  (N_ignore={g('terminal_ignore_N', '-')} C_ignore={g('terminal_ignore_C', '-')})"
    log(align)

    log(f"  AF2    : mean_plddt={_fmt_num(g('af2_mean_plddt'))} ptm={_fmt_num(g('af2_ptm_score'))} "
        f"mean_pae={_fmt_num(g('af2_mean_pae'))} rmsd_to_input={_fmt_num(g('af2_rmsd_to_input'))} "
        f"recycles={g('af2_recycles', '-')} tol={_fmt_num(g('af2_tol'))} "
        f"pae_len={g('af2_pae_length', '-')}")

    if catres_list:
        log("  catres detail:")
        for i, catres in enumerate(catres_list):
            prefix = f"{catres['name3']}{i + 1}"
            label = f"{catres['name3']}{catres['res_num']}"
            valid = catres_valid[i] if i < len(catres_valid) else False
            if valid:
                detail = (f"rmsd={_fmt_num(g(prefix + '_rmsd'))}  "
                          f"bb={_fmt_num(g(prefix + '_bb_rmsd'))}  "
                          f"plddt={_fmt_num(g(prefix + '_plddt'))}  "
                          f"pae={_fmt_num(g(prefix + '_pae'))}")
            else:
                detail = "NaN (invalid/unaligned)"
            log(f"     #{i + 1} {label:<8} {detail}")

    log(f"  aggregate: catres_rmsd={_fmt_num(g('catres_rmsd'))} bb={_fmt_num(g('catres_bb_rmsd'))} "
        f"lddt={_fmt_num(g('catres_lddt'), 4)} plddt={_fmt_num(g('catres_plddt'))} "
        f"pae_to_all={_fmt_num(g('catres_pae_to_all_mean'))} "
        f"pair_mean={_fmt_num(g('catres_pair_pae_mean'))} "
        f"pair_max={_fmt_num(g('catres_pair_pae_max'))} "
        f"pair_min={_fmt_num(g('catres_pair_pae_min'))}")

    if opts.get("catres_subset"):
        log(f"  subset : count={g('catres_subset_count', '-')} "
            f"catres_subset_rmsd={_fmt_num(g('catres_subset_rmsd'))} "
            f"bb={_fmt_num(g('catres_subset_bb_rmsd'))} "
            f"lddt={_fmt_num(g('catres_subset_lddt'), 4)} "
            f"plddt={_fmt_num(g('catres_subset_plddt'))} "
            f"pae_to_all={_fmt_num(g('catres_subset_pae_to_all_mean'))} "
            f"pair_mean={_fmt_num(g('catres_subset_pair_pae_mean'))} "
            f"pair_max={_fmt_num(g('catres_subset_pair_pae_max'))} "
            f"pair_min={_fmt_num(g('catres_subset_pair_pae_min'))}")

    log(f"  warnings: {', '.join(warnings) if warnings else 'none'}")
    log("=" * 48)


def _nanmean(values: List[float]) -> float:
    """Mean ignoring NaN; NaN if empty/all-NaN (numpy-2.0 clean, no warnings)."""
    if not values:
        return float("nan")
    arr = np.array(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size > 0 else float("nan")


def _catres_pair_pae(
    pae: Optional[np.ndarray], row_indices: List[Optional[int]]
) -> Tuple[float, float, float]:
    """Mean, max and min of off-diagonal catres x catres symmetric-averaged PAE.

    Uses symmetric_pair_pae for each i<j pair (the diagonal is excluded). Returns
    (nan, nan, nan) when fewer than two valid row indices contribute a finite pair.
    """
    if pae is None:
        return float("nan"), float("nan"), float("nan")
    vals = []
    n = len(row_indices)
    for a in range(n):
        for b in range(a + 1, n):
            v = symmetric_pair_pae(pae, row_indices[a], row_indices[b])
            if np.isfinite(v):
                vals.append(v)
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(vals)), float(np.max(vals)), float(np.min(vals))


def _tmalign_ca_rmsd(
    pred_pdb_path: str,
    ref_pdb_path: str,
    n_ignore: int,
    c_ignore: int,
    verbose: bool
) -> float:
    """CA RMSD after biotite TM-align-style superposition (only if --tmalign)."""
    pred_atoms = load_biotite_structure(pred_pdb_path)
    ref_atoms = load_biotite_structure(ref_pdb_path)
    if pred_atoms is None or ref_atoms is None:
        return float("nan")
    pred_protein = filter_biotite_protein_residues(pred_atoms, n_ignore, c_ignore)
    ref_protein = filter_biotite_protein_residues(ref_atoms, 0, 0)
    if len(pred_protein) == 0 or len(ref_protein) == 0:
        return float("nan")
    try:
        aligned_pred, transform, fix_idx, mob_idx = struc.superimpose_structural_homologs(
            ref_protein, pred_protein, max_iterations=10
        )
        return float(struc.rmsd(ref_protein[fix_idx], aligned_pred[mob_idx]))
    except Exception as e:
        vlog(verbose, f"TM-align superposition failed: {e}", "WARN")
        return float("nan")


# ==============================================================================
# SECTION 13: .sc WRITER  (atomic; CSV header + one row)
# ==============================================================================

def write_sc_atomic(row: Dict[str, Any], out_sc: str) -> None:
    """
    Write a one-data-row CSV ``.sc`` ATOMICALLY (tmp file -> os.replace).

    ``description`` is forced first; remaining columns keep insertion order.
    """
    columns = ["description"] + [k for k in row.keys() if k != "description"]
    out_dir = os.path.dirname(os.path.abspath(out_sc))
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = f"{out_sc}.tmp.{os.getpid()}"
    with open(tmp_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerow({c: _csv_value(row.get(c, "")) for c in columns})
    os.replace(tmp_path, out_sc)


def _csv_value(v: Any) -> Any:
    """Render NaN floats as 'nan' string; pass everything else through."""
    if isinstance(v, float) and np.isnan(v):
        return "nan"
    return v


def process_job_write_sc(args: Tuple[ResolvedJob, Dict[str, Any]]):
    """
    Top-level worker for the Pool: analyze one job and write its ``.sc``.

    Honors --skip_if_sc_present. Returns (description, status, warnings). Never
    raises into the Pool: any exception becomes a status=error row.
    """
    job, opts = args
    # Module-level ref cache, one per worker process (P1 LRU; bounded by maxsize).
    ref_cache = _get_worker_ref_cache()

    if opts.get("skip_if_sc_present", False) and os.path.isfile(job.out_sc):
        return job.description, "skipped", []

    # Resolver miss recorded up front -> emit an error row, don't silently drop.
    if job.error:
        row = {
            "description": job.description,
            "pdb_path": job.pred_pdb,
            "af2_json_path": job.json_path,
            "ref_path": job.ref_pdb or "",
            "status": "error",
            "error": job.error,
            "catres_signature": "",
        }
        try:
            write_sc_atomic(row, job.out_sc)
        except Exception as e:
            return job.description, "error", [f"write_fail:{e}"]
        return job.description, "error", [job.error]

    try:
        row, warnings = analyze_prediction(
            job.pred_pdb, job.ref_pdb, job.json_path, opts, ref_cache=ref_cache
        )
    except Exception as e:
        row = {
            "description": job.description,
            "pdb_path": job.pred_pdb,
            "af2_json_path": job.json_path,
            "ref_path": job.ref_pdb or "",
            "status": "error",
            "error": f"worker_exception:{type(e).__name__}:{e}",
            "catres_signature": "",
        }
        warnings = [f"exception:{type(e).__name__}"]

    try:
        write_sc_atomic(row, job.out_sc)
    except Exception as e:
        return job.description, "error", [f"write_fail:{e}"]
    vlog(opts.get("verbose", False), f"wrote {job.out_sc}")
    return job.description, row.get("status", "ok"), warnings


# Per-process ref cache (bounded). Populated lazily inside each Pool worker.
_WORKER_REF_CACHE: Optional[Dict[str, Any]] = None
_WORKER_REF_CACHE_MAX = 8


def _get_worker_ref_cache() -> Dict[str, Any]:
    """Return this worker's bounded ref cache, evicting oldest when over cap."""
    global _WORKER_REF_CACHE
    if _WORKER_REF_CACHE is None:
        _WORKER_REF_CACHE = {}
    if len(_WORKER_REF_CACHE) > _WORKER_REF_CACHE_MAX:
        # Simple FIFO eviction (dicts preserve insertion order).
        oldest = next(iter(_WORKER_REF_CACHE))
        del _WORKER_REF_CACHE[oldest]
    return _WORKER_REF_CACHE


# ==============================================================================
# SECTION 14: DRIVERS
# ==============================================================================

def _default_worker_count() -> int:
    """min(cpu_count()-5, 16), floored at 1 (codex #11)."""
    from multiprocessing import cpu_count
    try:
        cores = cpu_count()
    except NotImplementedError:
        cores = 1
    return max(1, min(cores - 5, 16))


def _out_sc_for(pred_pdb: str, out_dir: Optional[str], explicit_out: Optional[str]) -> str:
    """Where this prediction's .sc goes."""
    if explicit_out:
        return explicit_out
    base = pred_basename(pred_pdb)
    target_dir = out_dir if out_dir else os.path.dirname(os.path.abspath(pred_pdb))
    return os.path.join(target_dir, f"{base}.sc")


def _run_pool(jobs: List[ResolvedJob], opts: Dict[str, Any], cpus: int,
              combined_csv: Optional[str]) -> None:
    """Run jobs through a Pool and (optionally) emit a combined CSV."""
    from multiprocessing import Pool
    import time

    n = len(jobs)
    log(f"Processing {n} job(s) with {cpus} worker(s)...")
    rows_for_combined: List[Dict[str, Any]] = []
    statuses = {"ok": 0, "error": 0, "skipped": 0}
    start = time.time()

    task_args = [(job, opts) for job in jobs]
    if cpus <= 1 or n <= 1:
        results = (process_job_write_sc(a) for a in task_args)
        results = list(results)
    else:
        with Pool(processes=cpus) as pool:
            results = []
            for i, res in enumerate(pool.imap_unordered(process_job_write_sc, task_args), 1):
                results.append(res)
                if i % 200 == 0 or i == n:
                    elapsed = time.time() - start
                    log(f"  {i}/{n} done ({elapsed:.1f}s)")

    for desc, status, warns in results:
        statuses[status] = statuses.get(status, 0) + 1
        if warns:
            for w in warns:
                vlog(opts.get("verbose", False), f"  [{desc}] {w}", "WARN")

    if combined_csv:
        # Re-read the .sc files (bounded RAM not a concern for combined opt-in).
        for job in jobs:
            if os.path.isfile(job.out_sc):
                try:
                    df = pd.read_csv(job.out_sc)
                    if len(df) > 0:
                        rows_for_combined.append(df.iloc[0].to_dict())
                except Exception:
                    pass
        if rows_for_combined:
            pd.DataFrame.from_records(rows_for_combined).to_csv(combined_csv, index=False)
            log(f"Combined CSV written: {combined_csv} ({len(rows_for_combined)} rows)")

    # Report where the .sc files were written (unique output dirs).
    out_dirs = sorted({os.path.dirname(os.path.abspath(job.out_sc)) for job in jobs})
    if len(jobs) == 1:
        out_loc = os.path.abspath(jobs[0].out_sc)
    elif len(out_dirs) == 1:
        out_loc = out_dirs[0] + os.sep
    else:
        out_loc = f"{len(out_dirs)} dirs: " + ", ".join(out_dirs[:5]) + (
            f" (+{len(out_dirs) - 5} more)" if len(out_dirs) > 5 else "")
    log(f"Done. ok={statuses.get('ok', 0)} error={statuses.get('error', 0)} "
        f"skipped={statuses.get('skipped', 0)} ({time.time() - start:.1f}s); output -> {out_loc}")


def run_single(args, opts: Dict[str, Any]) -> None:
    """Single predicted PDB + a ref source (explicit --ref_pdb OR --ref_dir).

    --pred_pdb may pair with --ref_dir [+--af2_suffix] (auto-strip / suffix-strip)
    just like directory mode; an unresolved ref yields a status=error row (FIX 7),
    never a crash.
    """
    pred_pdb = args.pred_pdb
    json_path = derive_json_from_pdb(pred_pdb)
    if args.ref_pdb:
        resolver = ReferenceResolver("explicit", ref_pdb=args.ref_pdb)
    else:
        resolver = _build_resolver(args)
    ref_pdb, err = resolver.resolve(pred_pdb)
    out_sc = _out_sc_for(pred_pdb, args.out_dir, args.out)
    job = ResolvedJob(pred_pdb, json_path, ref_pdb, out_sc, pred_basename(pred_pdb), err)
    _print_manifest([job])
    _run_pool([job], opts, 1, args.combined_csv)


def run_directory(args, opts: Dict[str, Any]) -> None:
    """All *_unrelaxed.pdb in a flat dir + a ref source."""
    pred_pdbs = sorted(glob.glob(os.path.join(args.pred_dir, f"*{PRED_PDB_SUFFIX}")))
    if not pred_pdbs:
        log(f"No '*{PRED_PDB_SUFFIX}' files found in {args.pred_dir}", "ERROR")
        sys.exit(1)

    resolver = _build_resolver(args)
    jobs: List[ResolvedJob] = []
    for pred_pdb in pred_pdbs:
        json_path = derive_json_from_pdb(pred_pdb)
        ref_pdb, err = resolver.resolve(pred_pdb)
        out_sc = _out_sc_for(pred_pdb, args.out_dir, None)
        jobs.append(ResolvedJob(pred_pdb, json_path, ref_pdb, out_sc, pred_basename(pred_pdb), err))

    _print_manifest(jobs)
    cpus = args.cpus if args.cpus else _default_worker_count()
    _run_pool(jobs, opts, cpus, args.combined_csv)


def run_scorefile(args, opts: Dict[str, Any]) -> None:
    """Scorefile-driven: one row per pdb_path column entry."""
    df = pd.read_csv(args.scorefile)
    if args.pdb_path_col not in df.columns:
        log(f"Column '{args.pdb_path_col}' not in scorefile {args.scorefile}. "
            f"Available columns: {list(df.columns)}", "ERROR")
        sys.exit(2)

    explicit_ref_col = args.scorefile_ref_col
    resolver = None
    if explicit_ref_col:
        # Validate the ref column exists BEFORE building any jobs, so a typo
        # gives a clean error instead of crashing later in .resolve()/NaN paths.
        if explicit_ref_col not in df.columns:
            log(f"--scorefile_ref_col '{explicit_ref_col}' not in scorefile "
                f"{args.scorefile}. Available columns: {list(df.columns)}", "ERROR")
            sys.exit(2)
    else:
        resolver = _build_resolver(args)

    jobs: List[ResolvedJob] = []
    for _, srow in df.iterrows():
        pred_pdb = str(srow[args.pdb_path_col])
        json_path = derive_json_from_pdb(pred_pdb)
        if explicit_ref_col:
            ref_pdb = str(srow[explicit_ref_col])
            err = "" if (ref_pdb and os.path.isfile(ref_pdb)) else "missing_ref:scorefile_col"
            if err:
                ref_pdb = None
        else:
            ref_pdb, err = resolver.resolve(pred_pdb)
        out_sc = _out_sc_for(pred_pdb, args.out_dir, None)
        jobs.append(ResolvedJob(pred_pdb, json_path, ref_pdb, out_sc, pred_basename(pred_pdb), err))

    _print_manifest(jobs)
    cpus = args.cpus if args.cpus else _default_worker_count()
    _run_pool(jobs, opts, cpus, args.combined_csv)


def _build_resolver(args) -> ReferenceResolver:
    """Pick the ref resolver for dir/scorefile modes."""
    if args.ref_pdb:
        return ReferenceResolver("explicit", ref_pdb=args.ref_pdb)
    if args.ref_dir and args.af2_suffix:
        return ReferenceResolver("suffix_strip", ref_dir=args.ref_dir, af2_suffix=args.af2_suffix)
    if args.ref_dir:
        return ReferenceResolver("auto_strip", ref_dir=args.ref_dir)
    log("No compatible ref source for dir/scorefile mode "
        "(need --ref_pdb, --ref_dir [+--af2_suffix], or --scorefile_ref_col).", "ERROR")
    sys.exit(1)


def _print_manifest(jobs: List[ResolvedJob]) -> None:
    """Report resolved jobs and all reference misses up front (codex #5)."""
    misses = [j for j in jobs if j.error]
    log(f"Resolved {len(jobs)} job(s); {len(jobs) - len(misses)} with refs, "
        f"{len(misses)} miss(es).")
    if misses:
        log("Reference misses (each still gets a status=error .sc row):", "WARN")
        for j in misses[:20]:
            log(f"  MISS {j.description}: {j.error}", "WARN")
        if len(misses) > 20:
            log(f"  ... (+{len(misses) - 20} more)", "WARN")


# ==============================================================================
# SECTION 15: LEGACY PYROSETTA MODE  (lazy import; near-verbatim v2 port)
# ==============================================================================

def _run_legacy_atom_groups(args) -> None:
    """
    Legacy pyrosetta path (near-verbatim port of
    sidechain_rmsd_and_info_af2_matching_res_v2.py). Lazy-imports pyrosetta so
    the default biotite path never depends on it. Emits cat_{name3}{i}_{label}
    columns and reuses the new ReferenceResolver for matching.
    """
    import pyrosetta  # lazy: only here, only in a pyrosetta container
    import time
    from multiprocessing import Pool, cpu_count

    # --- init_pyrosetta (v2) --------------------------------------------------
    expanded_params = []
    if args.params:
        for param in args.params:
            expanded_params.extend(glob.glob(param))
    if expanded_params:
        options = f"-extra_res_fa {' '.join(expanded_params)} -mute all"
    else:
        options = "-mute all -beta_nov16"
    pyrosetta.init(options)
    log(f"PyRosetta initialized with options: {options}")

    if not args.atom_groups:
        log("Legacy mode requires --atom_groups JSON.", "ERROR")
        sys.exit(1)
    atom_groups = json.loads(args.atom_groups)

    # --- Build (pred, ref) match list via ReferenceResolver -------------------
    if args.scorefile:
        df = pd.read_csv(args.scorefile)
        pred_paths = df[args.pdb_path_col].tolist()
    elif args.pred_dir:
        pred_paths = sorted(glob.glob(os.path.join(args.pred_dir, f"*{PRED_PDB_SUFFIX}")))
        df = None
    elif args.pred_pdb:
        pred_paths = [args.pred_pdb]
        df = None
    else:
        log("Legacy mode needs --scorefile, --pred_dir, or --pred_pdb.", "ERROR")
        sys.exit(1)

    resolver = _build_resolver(args)
    matches = []
    for pred in pred_paths:
        ref, err = resolver.resolve(pred)
        if err or not ref:
            log(f"  Legacy MISS {os.path.basename(pred)}: {err}", "WARN")
            continue
        matches.append((ref, pred))

    log(f"Legacy matched {len(matches)}/{len(pred_paths)} predictions.")

    # --- detailed_rmsd_calculation (v2), preserving REMARK order --------------
    def _legacy_catres(pdb_file):
        out = []
        idx = 0
        with open(pdb_file) as fh:
            for line in fh:
                if line.startswith("ATOM"):
                    break
                if "REMARK 666" in line and "MATCH MOTIF" in line:
                    parts = line.split()
                    try:
                        idx += 1
                        out.append({"i": idx, "resnum": int(parts[11]), "name3": parts[10]})
                    except (IndexError, ValueError):
                        idx -= 1
        return out

    def _detailed_rmsd(ref_pose, af2_pose, catres, agroups):
        results = {}
        for info in catres:
            i = info["i"]
            rn = info["resnum"]
            name3 = info["name3"]
            if name3 not in agroups:
                continue
            for config in agroups[name3]:
                devs = []
                for atom_name in config["atoms"]:
                    if (rn <= ref_pose.size() and ref_pose.residue(rn).has(atom_name)
                            and rn <= af2_pose.size() and af2_pose.residue(rn).has(atom_name)):
                        devs.append((ref_pose.residue(rn).xyz(atom_name)
                                     - af2_pose.residue(rn).xyz(atom_name)).norm())
                if devs:
                    results[f"cat_{name3}{i}_{config['label']}"] = float(
                        np.sqrt(np.mean(np.square(devs)))
                    )
        return results

    def _process(pair):
        ref_pdb, af2_pdb = pair
        ref_pose = pyrosetta.rosetta.core.import_pose.pose_from_file(ref_pdb)
        af2_pose = pyrosetta.rosetta.core.import_pose.pose_from_file(af2_pdb)
        catres = _legacy_catres(ref_pdb)
        rmsd_dict = _detailed_rmsd(ref_pose, af2_pose, catres, atom_groups)
        return {args.pdb_path_col: af2_pdb, **rmsd_dict}

    start = time.time()
    cpus = args.cpus if args.cpus else max(1, cpu_count() - 5)
    results = []
    if matches:
        if cpus <= 1 or len(matches) <= 1:
            results = [_process(p) for p in matches]
        else:
            with Pool(processes=cpus) as pool:
                for i, r in enumerate(pool.imap(_process, matches), 1):
                    results.append(r)
                    if i % 50 == 0 or i == len(matches):
                        log(f"  {i}/{len(matches)} legacy RMSDs done ({time.time() - start:.1f}s)")

    out_path = args.out or args.combined_csv or "scores_updated.sc"
    if df is not None:
        updates = pd.DataFrame(results)
        merged = pd.merge(df, updates, on=args.pdb_path_col, how="left") if not updates.empty else df
        merged.to_csv(out_path, index=False)
    else:
        pd.DataFrame.from_records(results).to_csv(out_path, index=False)
    log(f"Legacy output saved: {out_path}")


# ==============================================================================
# SECTION 16: CLI
# ==============================================================================

def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process AlphaFold2 (superfold) predictions vs a reference PDB "
                    "and write per-prediction Rosetta-style .sc scorefiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single prediction + explicit ref:
    python process_af2_pdb.py --pred_pdb pred_unrelaxed.pdb --ref_pdb design.pdb \\
        --out out.sc --verbose

  Whole directory, refs by auto-strip of the AF2 model/seed tail:
    python process_af2_pdb.py --pred_dir ./af2_out --ref_dir ./refs --out_dir ./sc

  Whole directory, refs by stripping an exact suffix:
    python process_af2_pdb.py --pred_dir ./af2_out --ref_dir ./refs \\
        --af2_suffix _model_4_ptm_seed_0 --out_dir ./sc

  Scorefile-driven with an explicit ref column:
    python process_af2_pdb.py --scorefile in.csv --pdb_path_col pdb_path \\
        --scorefile_ref_col ref_path --out_dir ./sc

Each .sc is a 2-line CSV (header + one row); 'description' is the first column.
""",
    )

    # Input (exactly one).
    ginput = parser.add_mutually_exclusive_group(required=True)
    ginput.add_argument("--pred_pdb", help="Single predicted *_unrelaxed.pdb")
    ginput.add_argument("--pred_dir", help="Directory of predicted *_unrelaxed.pdb files")
    ginput.add_argument("--scorefile", help="CSV with a column of predicted PDB paths")

    # Reference.
    parser.add_argument("--ref_pdb", help="Single explicit reference PDB (REMARK 666)")
    parser.add_argument("--ref_dir", help="Directory of reference PDBs (suffix/auto strip)")
    parser.add_argument("--af2_suffix", help="Exact suffix to strip from the AF2 basename "
                                             "before looking up <stripped>.pdb in --ref_dir")
    parser.add_argument("--scorefile_ref_col", help="Scorefile column holding ref PDB paths")
    parser.add_argument("--pdb_path_col", default="pdb_path",
                        help="Scorefile column with predicted PDB paths (default: pdb_path)")

    # Metrics.
    parser.add_argument("--catres_subset", help="Comma-separated REMARK-666 cst_block numbers "
                                                "for an extra subset of catres metrics")
    parser.add_argument("--tmalign", action="store_true",
                        help="Also emit ca_rmsd_TMalign (biotite TM-align superposition)")
    parser.add_argument("--pae_full", action="store_true",
                        help="Also emit per-pair catres PAE columns {AA}{i}_{AA}{j}_pae (i<j)")
    parser.add_argument("--no_pae", action="store_true",
                        help="Skip parsing the LxL PAE matrix entirely (faster)")
    parser.add_argument("--no_per_catres", action="store_true",
                        help="Drop ONLY the per-catalytic-residue columns from the written .sc "
                             "({AA}{i}_rmsd/_bb_rmsd/_plddt/_pae and any --pae_full pair columns); "
                             "all catres_* aggregates, status/error, metadata and paths are kept. "
                             "Post-build filter: computed values are unchanged and --verbose still "
                             "reports the full computation. (Monomer/no-trim auto-drops still apply.)")
    parser.add_argument("--lean", action="store_true",
                        help="Emit ONLY the high-signal keep-list: description, pdb_path, "
                             "af2_json_path, ref_path, catres_signature/_count, af2_mean_plddt, "
                             "af2_ptm_score, af2_mean_pae, af2_mean_pae_intra_chain, "
                             "af2_rmsd_to_input, af2_tol, ca_rmsd, tm_score and all catres_* "
                             "aggregates incl. catres_pair_pae_min (+ the catres_subset_* mirror "
                             "incl. _min when --catres_subset). Drops status/error, per-catres, "
                             "folding metadata and ca_rmsd_TMalign. Wins over --no_per_catres. "
                             "Post-build filter only; --verbose still reports the full computation.")
    parser.add_argument("--N_terminus_tag_length_to_ignore", type=int, default=0,
                        help="N-terminal protein residues to ignore in global CA align (chain A)")
    parser.add_argument("--C_terminus_tag_length_to_ignore", type=int, default=0,
                        help="C-terminal protein residues to ignore in global CA align (chain A)")

    # Output.
    parser.add_argument("--out", help="Output .sc path (single-prediction mode)")
    parser.add_argument("--out_dir", help="Output dir for per-prediction .sc (dir/scorefile mode)")
    parser.add_argument("--combined_csv", help="Also write one combined CSV of all rows")
    parser.add_argument("--skip_if_sc_present", action="store_true",
                        help="Skip a prediction if its .sc already exists (idempotent re-runs)")

    # Exec.
    parser.add_argument("--cpus", type=int, default=None,
                        help="Worker count (default: min(cpu_count()-5, 16))")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    # Legacy.
    parser.add_argument("--legacy_atom_groups", action="store_true",
                        help="Run the legacy pyrosetta atom-groups path (lazy pyrosetta import)")
    parser.add_argument("--atom_groups", help="JSON string of atom groups (legacy mode)")
    parser.add_argument("--params", nargs="+", help="PyRosetta params files (legacy mode)")

    return parser.parse_args(argv)


def _validate_args(args) -> None:
    """Validate input/ref compatibility (P0 #25 surface)."""
    if args.legacy_atom_groups:
        if not args.atom_groups:
            log("--legacy_atom_groups requires --atom_groups JSON.", "ERROR")
            sys.exit(1)
        return

    # Single mode needs an output target and a ref source. --pred_pdb may pair
    # with --ref_pdb OR --ref_dir [+--af2_suffix] (FIX 7).
    if args.pred_pdb:
        if not (args.out or args.out_dir):
            log("Single-prediction mode needs --out or --out_dir.", "ERROR")
            sys.exit(1)
        if not (args.ref_pdb or args.ref_dir):
            log("Single-prediction mode needs a ref source "
                "(--ref_pdb | --ref_dir [+--af2_suffix]).", "ERROR")
            sys.exit(1)

    # Dir / scorefile modes need a ref source.
    if args.pred_dir or args.scorefile:
        has_ref = bool(args.ref_pdb or args.ref_dir or args.scorefile_ref_col)
        if not has_ref:
            log("Directory/scorefile mode needs a ref source "
                "(--ref_pdb | --ref_dir [+--af2_suffix] | --scorefile_ref_col).", "ERROR")
            sys.exit(1)


def main(argv=None) -> None:
    args = parse_arguments(argv)

    if args.legacy_atom_groups:
        _run_legacy_atom_groups(args)
        return

    _validate_args(args)

    catres_subset = None
    if args.catres_subset:
        try:
            catres_subset = [int(x) for x in str(args.catres_subset).split(",") if x.strip() != ""]
        except ValueError:
            log(f"Could not parse --catres_subset '{args.catres_subset}'", "ERROR")
            sys.exit(1)

    opts = {
        "verbose": args.verbose,
        "tmalign": args.tmalign,
        "pae_full": args.pae_full,
        "no_pae": args.no_pae,
        "n_ignore": args.N_terminus_tag_length_to_ignore,
        "c_ignore": args.C_terminus_tag_length_to_ignore,
        "catres_subset": catres_subset,
        "skip_if_sc_present": args.skip_if_sc_present,
        "no_per_catres": args.no_per_catres,
        "lean": args.lean,
    }

    if args.pred_pdb:
        run_single(args, opts)
    elif args.pred_dir:
        run_directory(args, opts)
    elif args.scorefile:
        run_scorefile(args, opts)


if __name__ == "__main__":
    main()
