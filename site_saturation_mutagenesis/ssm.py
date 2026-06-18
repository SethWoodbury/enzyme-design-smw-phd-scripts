#!/usr/bin/env python3
"""
Site Saturation Mutagenesis (SSM)

Generates single-point mutant PDB files for specified residue positions using
the MPNN neural sidechain packer to place mutant sidechains. All other atoms
(backbone, other sidechains, ligands, ions) remain identical to the input.

Usage:
    python ssm.py --pdb_path input.pdb --residues "A138-142, A162" --omit "MC"
"""

import argparse
import sys
import os
import copy
import re
import time
import types
from concurrent.futures import ThreadPoolExecutor

# ============================================================================
# Constants
# ============================================================================

MPNN_PATH = "/net/software/lab/fused_mpnn/seth_temp"
DEFAULT_CHECKPOINT = "/projects/ml/struc2seq/ligandMPNN_models/b_v1/s_300756.pt"

RESTYPE_STR_TO_INT = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
    'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15,
    'T': 16, 'V': 17, 'W': 18, 'Y': 19
}
RESTYPE_INT_TO_STR = {v: k for k, v in RESTYPE_STR_TO_INT.items()}
RESTYPE_1TO3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
}
ALL_AAS = set(RESTYPE_STR_TO_INT.keys())


# ============================================================================
# Helpers
# ============================================================================

def parse_residue_ranges(range_str):
    """
    Parse residue specification string into (chain, resnum) tuples.

    Examples:
        "A138-142"       -> [("A",138), ("A",139), ("A",140), ("A",141), ("A",142)]
        "A138-142, A162" -> [("A",138), ..., ("A",142), ("A",162)]
        "A78-81, A132, A119-120" -> 7 positions
    """
    positions = []
    seen = set()
    segments = [s.strip() for s in range_str.split(',')]
    pattern = re.compile(r'^([A-Za-z])(\d+)(?:-(\d+))?$')

    for seg in segments:
        m = pattern.match(seg)
        if not m:
            print(f"  WARNING: Cannot parse residue spec '{seg}', skipping")
            continue
        chain = m.group(1)
        start = int(m.group(2))
        end = int(m.group(3)) if m.group(3) else start
        if end < start:
            start, end = end, start
        for resnum in range(start, end + 1):
            key = (chain, resnum)
            if key not in seen:
                positions.append(key)
                seen.add(key)

    return positions


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.0f}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m {s:.0f}s"


def get_system_info():
    """Auto-detect system resources."""
    import torch

    info = {}
    info['cpu_count'] = os.cpu_count() or 1

    # Memory via /proc/meminfo (Linux)
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    info['total_memory_gb'] = int(line.split()[1]) / (1024 ** 2)
                elif line.startswith('MemAvailable'):
                    info['available_memory_gb'] = int(line.split()[1]) / (1024 ** 2)
    except Exception:
        info['total_memory_gb'] = 0
        info['available_memory_gb'] = 0

    # GPU
    info['gpu_available'] = torch.cuda.is_available()
    if info['gpu_available']:
        info['gpu_name'] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info['gpu_memory_gb'] = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024 ** 3)

    return info


RESTYPE_INT_TO_STR_FULL = {
    0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I',
    8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S',
    16: 'T', 17: 'V', 18: 'W', 19: 'Y', 20: 'X'
}

RESTYPE_NAME_TO_ATOM14 = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', ''],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', '', '', ''],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', '', '', '', '', '', ''],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', '', '', '', '', '', ''],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG', '', '', '', '', '', '', '', ''],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2', '', '', '', '', ''],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', '', '', '', '', ''],
    'GLY': ['N', 'CA', 'C', 'O', '', '', '', '', '', '', '', '', '', ''],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', '', '', '', ''],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '', '', '', '', '', ''],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', '', '', '', '', '', ''],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', '', '', '', '', ''],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', '', '', '', '', '', ''],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', '', '', ''],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', '', '', '', '', '', '', ''],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG', '', '', '', '', '', '', '', ''],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '', '', '', '', '', '', ''],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'NE1', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', '', ''],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '', '', '', '', '', '', ''],
    'UNK': ['', '', '', '', '', '', '', '', '', '', '', '', '', ''],
}


def extract_pdb_header(pdb_path):
    """
    Extract all header lines from a PDB file (everything before the first
    ATOM/HETATM/MODEL record).  Returns a single string including newlines.
    """
    header_lines = []
    with open(pdb_path) as fh:
        for line in fh:
            record = line[:6].strip()
            if record in ("ATOM", "HETATM", "MODEL"):
                break
            header_lines.append(line)
    return "".join(header_lines)


def write_pdb(save_path, X, X_m, b_factors, R_idx, chain_letters, S,
              other_atoms=None, icodes=None, header=""):
    """
    Write a PDB file from atom14 coordinates using biotite.

    Args:
        save_path: output file path
        X: atom14 coordinates [L, 14, 3]
        X_m: atom14 mask [L, 14]
        b_factors: per-atom b-factors [L, 14]
        R_idx: residue indices [L]
        chain_letters: chain IDs [L]
        S: integer sequence [L]
        other_atoms: biotite AtomArray of ligands/ions (appended as-is)
        icodes: insertion codes [L]
        header: string to prepend (HEADER, REMARK, LINK lines etc.)
    """
    import numpy as np
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdbb

    S_str = [RESTYPE_1TO3.get(RESTYPE_INT_TO_STR_FULL.get(int(aa), 'X'), 'UNK')
             for aa in S]

    X_list, bfac_list, aname_list = [], [], []
    ename_list, rname_list, rnum_list, cid_list, ic_list = [], [], [], [], []

    for i, aa_3 in enumerate(S_str):
        sel = X_m[i].astype(np.int32) == 1
        total = int(np.sum(sel))
        atom14_names = np.array(RESTYPE_NAME_TO_ATOM14.get(aa_3, RESTYPE_NAME_TO_ATOM14['UNK']))
        tmp_names = atom14_names[sel]

        X_list.append(X[i][sel])
        bfac_list.append(b_factors[i][sel])
        aname_list.append(tmp_names)
        ename_list += [n[:1] for n in tmp_names]
        rname_list += total * [aa_3]
        rnum_list += total * [R_idx[i]]
        cid_list += total * [chain_letters[i]]
        ic_list += total * [icodes[i] if icodes is not None else '']

    X_stack = np.concatenate(X_list, 0)
    bfac_stack = np.concatenate(bfac_list, 0)
    aname_stack = np.concatenate(aname_list, 0)

    n_atoms = X_stack.shape[0]
    atom_array = struc.AtomArray(n_atoms)
    atom_array.coord = X_stack
    atom_array.atom_name = aname_stack
    atom_array.res_id = np.array(rnum_list, dtype=int)
    atom_array.res_name = rname_list
    atom_array.chain_id = cid_list
    atom_array.element = ename_list
    atom_array.b_factor = bfac_stack
    atom_array.occupancy = np.ones(n_atoms)
    atom_array.ins_code = ic_list
    atom_array.hetero = np.zeros(n_atoms, dtype=bool)

    if other_atoms is not None and len(other_atoms) > 0:
        atom_array = atom_array + other_atoms

    pdb_file = pdbb.PDBFile()
    pdb_file.set_structure(atom_array)

    if header:
        # Prepend original header lines before biotite's ATOM/HETATM records
        with open(save_path, "w") as fh:
            fh.write(header)
            for line in pdb_file.lines:
                fh.write(line + "\n")
    else:
        pdb_file.write(save_path)


def build_residue_mapping(protein_dict, icodes):
    """
    Build mapping from "chain_letter + resnum + icode" strings to array indices.

    Returns:
        encoded_residues: list of strings like ["A78", "A79", ...]
        encoded_residue_dict: dict mapping string -> index
    """
    R_idx_list = list(protein_dict["R_idx"].cpu().numpy())
    chain_letters_list = list(protein_dict["chain_letters"])
    encoded_residues = []
    for i in range(len(R_idx_list)):
        encoded_residues.append(
            str(chain_letters_list[i]) + str(R_idx_list[i]) + icodes[i]
        )
    encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
    return encoded_residues, encoded_residue_dict


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Site Saturation Mutagenesis: generate single-point mutant PDBs "
                    "with MPNN neural sidechain packing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pdb_path", required=True, type=str,
                        help="Path to input PDB file")
    parser.add_argument("--residues", required=True, type=str,
                        help='Residue ranges to mutate, e.g. "A138-142, A162, A170-175"')
    parser.add_argument("--omit", type=str, default="",
                        help='1-letter amino acids to omit from mutations, e.g. "MC"')
    parser.add_argument("--output_dir", type=str, default="./ssm_output",
                        help="Output directory for mutant PDBs")
    parser.add_argument("--num_processes", type=int, default=0,
                        help="CPU threads for PyTorch (0 = auto-detect, uses all available)")
    parser.add_argument("--max_batch_size", type=int, default=0,
                        help="Max mutations per packing batch (0 = auto, packs all mutations "
                             "for a position in one call)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--sc_num_denoising_steps", type=int, default=3,
                        help="Denoising steps for sidechain packing")
    parser.add_argument("--sc_num_samples", type=int, default=16,
                        help="Sidechain samples to evaluate per denoising step")
    parser.add_argument("--checkpoint_path_sc", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to sidechain packer model checkpoint")
    parser.add_argument("--parse_these_chains_only", type=str, default="",
                        help="Restrict PDB parsing to these chains, e.g. 'AB'")
    args = parser.parse_args()

    t_total_start = time.time()

    # ========================================================================
    # Banner
    # ========================================================================
    print()
    print("=" * 72)
    print("  SITE SATURATION MUTAGENESIS (SSM)")
    print("  MPNN Neural Sidechain Packer")
    print("=" * 72)
    print()

    # ========================================================================
    # Imports (MPNN codebase)
    # ========================================================================
    # Mock cifutils (used by data_utils but not available in most envs)
    # We write PDBs with biotite directly instead.
    _mock_io = types.ModuleType('cifutils.utils.io_utils')
    _mock_io.to_pdb_buffer = None
    _mock_io.to_cif_file = None
    _mock_utils = types.ModuleType('cifutils.utils')
    _mock_utils.io_utils = _mock_io
    _mock_top = types.ModuleType('cifutils')
    _mock_top.utils = _mock_utils
    sys.modules['cifutils'] = _mock_top
    sys.modules['cifutils.utils'] = _mock_utils
    sys.modules['cifutils.utils.io_utils'] = _mock_io

    sys.path.insert(0, MPNN_PATH)
    import torch
    import numpy as np
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdbb
    from data_utils import parse_PDB, featurize
    from sc_utils import Packer, pack_side_chains

    # ========================================================================
    # Seeds
    # ========================================================================
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    # ========================================================================
    # System info & device
    # ========================================================================
    sys_info = get_system_info()
    device = torch.device("cuda:0" if sys_info['gpu_available'] else "cpu")

    # Set torch threads
    if args.num_processes > 0:
        num_threads = args.num_processes
    else:
        num_threads = sys_info['cpu_count']
    torch.set_num_threads(num_threads)

    print("System:")
    if sys_info['gpu_available']:
        print(f"  Device:     CUDA ({sys_info['gpu_name']}, "
              f"{sys_info['gpu_memory_gb']:.1f} GB)")
    else:
        print(f"  Device:     CPU")
        print(f"  Threads:    {num_threads} (of {sys_info['cpu_count']} available)")
    if sys_info.get('total_memory_gb'):
        print(f"  Memory:     {sys_info.get('available_memory_gb', 0):.1f} GB available / "
              f"{sys_info.get('total_memory_gb', 0):.1f} GB total")
    print()

    # ========================================================================
    # Validate input PDB
    # ========================================================================
    if not os.path.exists(args.pdb_path):
        print(f"ERROR: PDB file not found: {args.pdb_path}")
        sys.exit(1)

    pdb_size_kb = os.path.getsize(args.pdb_path) / 1024
    pdb_name = os.path.basename(args.pdb_path)
    if pdb_name.endswith((".pdb", ".cif")):
        pdb_name = pdb_name[:pdb_name.rfind('.')]

    print(f"Input PDB:    {args.pdb_path}")
    print(f"  Base name:  {pdb_name}")
    print(f"  File size:  {pdb_size_kb:.1f} KB")
    print()

    # ========================================================================
    # Output directory
    # ========================================================================
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output dir:   {os.path.abspath(args.output_dir)}")
    print()

    # ========================================================================
    # Parse PDB
    # ========================================================================
    print("Parsing PDB structure...")
    t0 = time.time()

    protein_dict, _, other_atoms, icodes, _ = parse_PDB(
        args.pdb_path,
        device=device,
        atom_context_num=16,
        chains=list(args.parse_these_chains_only) if args.parse_these_chains_only else [],
        parse_all_atoms=True,
    )

    L = protein_dict["S"].shape[0]
    chain_set = sorted(set(protein_dict["chain_letters"]))
    n_chains = len(chain_set)
    has_other = other_atoms is not None and len(other_atoms) > 0

    print(f"  Residues:          {L}")
    print(f"  Chains:            {n_chains} ({', '.join(chain_set)})")
    if has_other:
        # Identify unique ligand/ion residue names
        other_resnames = sorted(set(other_atoms.res_name))
        print(f"  Non-protein atoms: {len(other_atoms)} "
              f"({', '.join(other_resnames)})")
    print(f"  Parse time:        {time.time() - t0:.2f}s")
    print()

    # ========================================================================
    # Extract PDB header (REMARK 666, LINK, HETNAM, etc.)
    # ========================================================================
    pdb_header = extract_pdb_header(args.pdb_path)
    if pdb_header.strip():
        header_line_count = len(pdb_header.strip().splitlines())
        print(f"  PDB header:        {header_line_count} lines preserved "
              f"(REMARK, LINK, HETNAM, etc.)")
        print()

    # ========================================================================
    # Build residue mapping
    # ========================================================================
    encoded_residues, encoded_residue_dict = build_residue_mapping(
        protein_dict, icodes
    )

    # ========================================================================
    # Parse and validate target positions
    # ========================================================================
    print("Parsing target residues...")
    target_positions = parse_residue_ranges(args.residues)
    omit_aas = set(args.omit.upper()) if args.omit else set()

    valid_positions = []
    for chain, resnum in target_positions:
        key = f"{chain}{resnum}"
        # Try exact match first (with empty icode)
        if key in encoded_residue_dict:
            idx = encoded_residue_dict[key]
            valid_positions.append((chain, resnum, idx, key))
        else:
            # Try matching with any icode
            found = False
            for enc in encoded_residues:
                if enc.startswith(key):
                    idx = encoded_residue_dict[enc]
                    valid_positions.append((chain, resnum, idx, enc))
                    found = True
                    break
            if not found:
                print(f"  WARNING: Residue {key} not found in PDB, skipping")

    if not valid_positions:
        print("\nERROR: No valid target positions found in PDB!")
        sys.exit(1)

    # ========================================================================
    # Compute mutation details per position
    # ========================================================================
    print()
    print("Target positions:")
    print("-" * 60)

    total_mutations = 0
    position_details = []

    for chain, resnum, idx, key in valid_positions:
        wt_int = protein_dict["S"][idx].item()
        if wt_int > 19:
            print(f"  {key}: non-standard residue (index {wt_int}), skipping")
            continue
        wt_aa = RESTYPE_INT_TO_STR[wt_int]
        wt_3 = RESTYPE_1TO3.get(wt_aa, "UNK")

        # Mutations = all 20 AAs minus WT minus omitted
        mut_aas = sorted(ALL_AAS - {wt_aa} - omit_aas)
        n_mut = len(mut_aas)
        total_mutations += n_mut

        position_details.append({
            'chain': chain, 'resnum': resnum, 'idx': idx, 'key': key,
            'wt_int': wt_int, 'wt_aa': wt_aa, 'wt_3': wt_3,
            'mut_aas': mut_aas, 'n_mut': n_mut,
        })

        note = ""
        if wt_aa in omit_aas:
            note = "  (WT is in omit list, so 20 - omitted)"
        print(f"  {key:>6s}:  {wt_aa} ({wt_3})  ->  {n_mut} mutations{note}")

    print()
    if omit_aas:
        omit_parts = [f"{aa} ({RESTYPE_1TO3.get(aa, '?')})" for aa in sorted(omit_aas)]
        print(f"Omitted AAs:     {', '.join(omit_parts)}")
    else:
        print(f"Omitted AAs:     (none)")

    print()
    print(f"Positions:       {len(position_details)}")
    print(f"Mutant PDBs:     {total_mutations}")
    print(f"WT control PDB:  1")
    print(f"Total files:     {total_mutations + 1}")
    print()

    if total_mutations == 0:
        print("Nothing to do (all mutations omitted or no valid positions).")
        sys.exit(0)

    # ========================================================================
    # Load sidechain packer model
    # ========================================================================
    print("Loading MPNN sidechain packer model...")
    t0 = time.time()

    model_sc = Packer(
        node_features=128, edge_features=128,
        num_positional_embeddings=16, num_chain_embeddings=16,
        num_rbf=16, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3,
        atom_context_num=16,
        lower_bound=0.0, upper_bound=20.0,
        top_k=32, dropout=0.0, augment_eps=0.0,
        atom37_order=False, device=device, num_mix=3,
    )
    checkpoint = torch.load(args.checkpoint_path_sc, map_location=device, weights_only=False)
    model_sc.load_state_dict(checkpoint['model_state_dict'])
    model_sc.to(device)
    model_sc.eval()

    print(f"  Checkpoint:  {args.checkpoint_path_sc}")
    print(f"  Load time:   {time.time() - t0:.2f}s")
    print()

    # ========================================================================
    # Featurize structure (once)
    # ========================================================================
    print("Featurizing structure...")
    t0 = time.time()

    # All positions fixed by default; we selectively unmask per mutation batch
    protein_dict["chain_mask"] = torch.zeros(L, device=device)
    protein_dict["side_chain_mask"] = torch.zeros(L, device=device)
    protein_dict["pssm"] = torch.zeros([L, 20], device=device)

    feature_dict = featurize(
        protein_dict,
        cutoff_for_score=8.0,
        use_atom_context=True,
        number_of_ligand_atoms=16,
        model_type="protein_mpnn",
    )

    # Fix atom37 ordering: parse_PDB stores xyz_37 with CB at index 3, O at
    # index 4 (OpenFold order), but make_torsion_features in sc_utils.py
    # expects O at 3, CB at 4 and internally swaps them. Pre-swap here so the
    # double-swap lands on the correct OpenFold order for fixed-residue
    # coordinate preservation.
    xyz37_fixed = feature_dict["xyz_37"].clone()
    feature_dict["xyz_37"] = torch.empty_like(xyz37_fixed)
    feature_dict["xyz_37"][:] = xyz37_fixed
    feature_dict["xyz_37"][:, :, 3, :] = xyz37_fixed[:, :, 4, :]
    feature_dict["xyz_37"][:, :, 4, :] = xyz37_fixed[:, :, 3, :]

    print(f"  Done ({time.time() - t0:.2f}s)")
    print()

    # ========================================================================
    # Write WT control PDB (direct from parsed coords — no packer needed)
    # ========================================================================
    print("Generating wild-type control PDB...")
    t0 = time.time()

    from openfold_dependencies.data_transforms import make_atom14_masks

    # Map MPNN seq integers to AlphaFold2 ordering for atom14 mask computation
    map_mpnn_to_af2 = torch.tensor([
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    ], device=device, dtype=torch.float32)

    with torch.inference_mode():
        S_wt = feature_dict["S"][0].long()
        S_af2 = torch.argmax(
            torch.nn.functional.one_hot(S_wt, 21).float() @ map_mpnn_to_af2, -1
        )
        masks14 = make_atom14_masks({"aatype": S_af2[None]})
        residx_a14_to_a37 = masks14["residx_atom14_to_atom37"][0].long()
        atom14_exists = masks14["atom14_atom_exists"][0]

        # Gather atom14 coords from xyz_37 (already pre-swapped to MPNN order,
        # so re-swap to OpenFold order for correct atom14 mapping)
        xyz37_of = feature_dict["xyz_37"][0].clone()
        xyz37_of[:, 3, :], xyz37_of[:, 4, :] = (
            feature_dict["xyz_37"][0, :, 4, :].clone(),
            feature_dict["xyz_37"][0, :, 3, :].clone(),
        )
        idx_expand = residx_a14_to_a37[..., None].expand(-1, -1, 3)
        wt_X14 = torch.gather(xyz37_of, 1, idx_expand) * atom14_exists[..., None]
        wt_X14_m = atom14_exists * feature_dict["mask"][0, :, None]

    wt_path = os.path.join(args.output_dir, f"{pdb_name}_WT_control.pdb")
    R_idx_orig_np = feature_dict["R_idx_original"][0].cpu().numpy()
    S_wt_np = feature_dict["S"][0].cpu().numpy()
    write_pdb(
        wt_path,
        wt_X14.cpu().numpy(),
        wt_X14_m.cpu().numpy(),
        np.zeros_like(wt_X14_m.cpu().numpy()),
        R_idx_orig_np,
        protein_dict["chain_letters"],
        S_wt_np,
        other_atoms=other_atoms,
        icodes=icodes,
        header=pdb_header,
    )

    print(f"  Wrote: {os.path.basename(wt_path)}  ({time.time() - t0:.2f}s)")
    print()

    # ========================================================================
    # Generate mutant structures (per-position batching + threaded writes)
    # ========================================================================
    print("=" * 72)
    print("  GENERATING MUTANT STRUCTURES")
    print("=" * 72)
    print()
    print(f"  Strategy: {len(position_details)} positions, batching all mutations "
          f"per position (B~{position_details[0]['n_mut'] if position_details else 0})")
    print()

    files_written = 1  # WT already written
    position_times = []
    t_packing_total = time.time()

    # Pre-compute shared numpy arrays (avoid repeated .cpu().numpy())
    R_idx_orig_np = feature_dict["R_idx_original"][0].cpu().numpy()
    chain_letters_list = protein_dict["chain_letters"]

    # Thread pool for parallel PDB writing (overlaps I/O with packing compute)
    write_futures = []
    writer_pool = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))

    with torch.inference_mode():
        for pos_i, pinfo in enumerate(position_details):
            t_pos = time.time()

            idx = pinfo['idx']
            key = pinfo['key']
            wt_aa = pinfo['wt_aa']
            mut_aas = pinfo['mut_aas']
            n_mut = pinfo['n_mut']

            if n_mut == 0:
                print(f"  [{pos_i+1}/{len(position_details)}] {key} ({wt_aa}): "
                      f"skipped (no mutations after omissions)")
                continue

            mut_aa_ints = [RESTYPE_STR_TO_INT[aa] for aa in mut_aas]
            B = n_mut

            print(f"  [{pos_i+1}/{len(position_details)}] {key} ({wt_aa}): "
                  f"{n_mut} mutations ... ", end="", flush=True)

            # Build batch: expand feature_dict to B copies, set S & chain_mask
            sc_fd = {}
            for k, v in feature_dict.items():
                if k == "S":
                    continue
                try:
                    ndim = len(v.shape)
                    if ndim >= 2:
                        sc_fd[k] = v.repeat(B, *([1] * (ndim - 1)))
                    else:
                        sc_fd[k] = v
                except Exception:
                    sc_fd[k] = v

            chain_mask_batch = torch.zeros(B, L, device=device)
            chain_mask_batch[:, idx] = 1.0
            sc_fd["chain_mask"] = chain_mask_batch
            sc_fd["side_chain_mask"] = chain_mask_batch

            S_batch = feature_dict["S"].long().repeat(B, 1)
            for b, mut_int in enumerate(mut_aa_ints):
                S_batch[b, idx] = mut_int
            sc_fd["S"] = S_batch

            # Pack sidechains
            sc_result = pack_side_chains(
                sc_fd, model_sc,
                args.sc_num_denoising_steps, args.sc_num_samples,
                repack_everything=False,
            )

            # Move results to CPU once (not per-file)
            X_cpu = sc_result["X"].cpu().numpy()
            X_m_cpu = sc_result["X_m"].cpu().numpy()
            bfac_cpu = sc_result["b_factors"].cpu().numpy()
            S_cpu = S_batch.cpu().numpy()

            elapsed_pos = time.time() - t_pos

            # Submit PDB writes to thread pool (overlaps with next position's packing)
            for b, (mut_aa, mut_int) in enumerate(zip(mut_aas, mut_aa_ints)):
                filename = f"{pdb_name}_{wt_aa}{pinfo['resnum']}{mut_aa}.pdb"
                filepath = os.path.join(args.output_dir, filename)
                fut = writer_pool.submit(
                    write_pdb,
                    filepath, X_cpu[b], X_m_cpu[b], bfac_cpu[b],
                    R_idx_orig_np, chain_letters_list, S_cpu[b],
                    other_atoms=other_atoms, icodes=icodes,
                    header=pdb_header,
                )
                write_futures.append(fut)
                files_written += 1

            position_times.append(elapsed_pos)

            # ETA
            avg_per_pos = sum(position_times) / len(position_times)
            remaining_pos = len(position_details) - (pos_i + 1)
            eta = avg_per_pos * remaining_pos

            if remaining_pos > 0:
                print(f"done ({elapsed_pos:.1f}s)  "
                      f"[ETA remaining: {format_time(eta)}]")
            else:
                print(f"done ({elapsed_pos:.1f}s)")

    # Wait for all PDB writes to finish
    print(f"\n  Flushing {len(write_futures)} PDB writes ... ", end="", flush=True)
    t_write = time.time()
    for fut in write_futures:
        fut.result()  # propagate any exceptions
    writer_pool.shutdown(wait=True)
    print(f"done ({time.time() - t_write:.1f}s)")

    # ========================================================================
    # Summary
    # ========================================================================
    total_time = time.time() - t_total_start
    packing_time = sum(position_times)

    print()
    print("=" * 72)
    print("  COMPLETE")
    print("=" * 72)
    print(f"  Files written:    {files_written} "
          f"({files_written - 1} mutants + 1 WT control)")
    print(f"  Output directory: {os.path.abspath(args.output_dir)}")
    print(f"  Packing time:     {format_time(packing_time)}")
    print(f"  Total wall time:  {format_time(total_time)}")
    if position_times:
        print(f"  Throughput:       {(files_written-1)/packing_time:.1f} mutations/s "
              f"({packing_time/(files_written-1):.2f}s per mutation)")
    print()


if __name__ == "__main__":
    main()
