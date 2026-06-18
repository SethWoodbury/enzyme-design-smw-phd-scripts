#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Bio import PDB
import os, glob, json, argparse
import numpy as np
import pandas as pd

# =========================
# I/O helpers
# =========================
def load_pdb(file_path):
    """Load a PDB file using Biopython."""
    parser = PDB.PDBParser(QUIET=True)
    return parser.get_structure(os.path.basename(file_path), file_path)

def save_pdb(structure, output_file):
    """Save the structure to a PDB file."""
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_file)

# =========================
# Header parsing
# =========================
def get_catalytic_residues(pdb_file):
    """
    Parse catalytic residues from REMARK 666 lines in the header.
    Assumes tokens like: ... REMARK 666 ... <chain> <resname> <resnum> ...
    Returns: [{'res_num': int, 'name3': str, 'chain': str}, ...]
    """
    catalytic_residues = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                break
            if "REMARK 666" in line:
                parts = line.split()
                chain = parts[9]
                residue_name = parts[10]
                residue_number = int(parts[11])
                catalytic_residues.append({'res_num': residue_number, 'name3': residue_name, 'chain': chain})
    print(f"Catalytic residues identified in {pdb_file}: {catalytic_residues}")
    return catalytic_residues

# =========================
# pLDDT parsing (0–100 scale)
# =========================
def parse_plddt_indices(af3_pdb_path):
    """
    Parse AF3 PDB once and build:
      - per_residue_plddt: {(chain, resseq): avg_plddt(0–100)}
      - per_atom_plddt   : {(chain, resseq, atom): plddt(0–100)}
    """
    per_atom = {}
    tmp_res = {}  # {(chain, resseq): {atom: plddt}}

    with open(af3_pdb_path, 'r') as f:
        for line in f:
            if line[:6] not in ("ATOM  ", "HETATM"):
                continue
            if len(line) < 67:
                continue
            chain = line[21:23].strip()
            res_s = line[23:27].strip()
            if not res_s.isdigit():
                continue
            resi = int(res_s)
            atom = line[12:16].strip()  # standard atom-name slice
            plddt = float(line[61:67])  # 0–100

            per_atom[(chain, resi, atom)] = plddt
            key = (chain, resi)
            if key not in tmp_res:
                tmp_res[key] = {}
            tmp_res[key][atom] = plddt

    per_residue = {k: float(np.mean(list(v.values()))) for k, v in tmp_res.items()} if tmp_res else {}
    return per_residue, per_atom

# =========================
# Geometry helpers
# =========================
def extract_ca_coords(structure, chain_id="A"):
    """Extract Cα coordinates from a given chain (default: A)."""
    ca = []
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if 'CA' in residue:
                    ca.append(residue['CA'].get_coord())
    return np.array(ca)

def extract_residue_coords_noH(structure, chain_id, residue_num):
    """Extract non-H atom coords+names for (chain,resnum)."""
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

def extract_named_atoms(structure, chain_id, residue_num, atom_names):
    """
    Return Nx3 coords in the same order as atom_names, and a list of missing atom names (if any).
    No hydrogen filtering: uses exactly the requested names.
    """
    name_to_coord = {}
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.get_id()[1] == residue_num:
                    for atom in residue:
                        name_to_coord[atom.get_name()] = atom.get_coord()
                    break
    coords = []
    missing = []
    for nm in atom_names:
        if nm in name_to_coord:
            coords.append(name_to_coord[nm])
        else:
            missing.append(nm)
    return (np.array(coords) if coords else np.zeros((0, 3))), missing

def kabsch_alignment(coords1, coords2):
    """
    Return R, c1, c2 such that (coords1 - c1) @ R + c2 ≈ coords2
    """
    c1 = np.mean(coords1, axis=0)
    c2 = np.mean(coords2, axis=0)
    A = coords1 - c1
    B = coords2 - c2
    U, S, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R, c1, c2

def apply_rotation(coords, R, c1, c2):
    """Apply affine transform defined by R, c1, c2."""
    return (coords - c1) @ R + c2

def calculate_rmsd(coords1, coords2):
    """RMSD between two Nx3 arrays."""
    if coords1.shape != coords2.shape:
        raise ValueError(f"RMSD shape mismatch: {coords1.shape} vs {coords2.shape}")
    diff = coords1 - coords2
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

# =========================
# Core alignment/scoring
# =========================
def align_and_score(af3_structure, ref_structure, catres_list, chain_id="A"):
    """
    Align AF3 to REF using CA atoms from chain_id.
    Compute CA RMSD and per-catalytic-residue RMSD (non-H), chain-aware.
    """
    ca1 = extract_ca_coords(af3_structure, chain_id)
    ca2 = extract_ca_coords(ref_structure, chain_id)
    if ca1.size == 0 or ca2.size == 0:
        raise ValueError(f"No CA atoms found on chain {chain_id} in one of the structures.")
    if ca1.shape != ca2.shape:
        raise ValueError(f"CA count mismatch on chain {chain_id}: AF3 {ca1.shape[0]} vs REF {ca2.shape[0]}.")

    R, c1, c2 = kabsch_alignment(ca1, ca2)

    # Transform AF3 in-place
    for model in af3_structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom.set_coord(apply_rotation(np.array([atom.get_coord()]), R, c1, c2)[0])

    # RMSD after alignment
    ca1_aligned = extract_ca_coords(af3_structure, chain_id)
    ca_rmsd = calculate_rmsd(ca1_aligned, ca2)

    # Catalytic residue RMSDs
    catres_rmsds = []
    for cat in catres_list:
        ch = cat["chain"]
        resn = cat["res_num"]
        c1r, n1 = extract_residue_coords_noH(af3_structure, ch, resn)
        c2r, n2 = extract_residue_coords_noH(ref_structure,  ch, resn)
        if len(n1) == 0 or len(n2) == 0:
            raise ValueError(f"Residue not found or empty: {ch}{resn} (AF3 atoms: {len(n1)}, REF atoms: {len(n2)})")
        if n1 != n2:
            raise ValueError(f"Atom name mismatch for {ch}{resn}: AF3={n1} vs REF={n2}")
        catres_rmsds.append(calculate_rmsd(c1r, c2r))

    return ca_rmsd, catres_rmsds

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_pdb", required=True, help="Path to reference PDB (protein only, chain A).")
    parser.add_argument("--af3_dir", required=True, help="Directory containing AF3 PDB predictions.")
    parser.add_argument("--outscr", required=True, type=str, help="Output CSV path.")
    parser.add_argument("--chain_id", default="A", help="Protein chain ID for alignment (default: A).")
    parser.add_argument("--atom_groups_json", type=str, default=None,
                        help="JSON mapping residue name (e.g., 'HIS') to [{'atoms':[...], 'label':'...'}, ...].")
    args = parser.parse_args()

    atom_groups = json.loads(args.atom_groups_json) if args.atom_groups_json else None

    # Load reference and catalytic residues
    ref_structure = load_pdb(args.ref_pdb)
    catres_list  = get_catalytic_residues(args.ref_pdb)

    updated = {"description": args.af3_dir}
    af3_pdbs = sorted(glob.glob(os.path.join(args.af3_dir, "*.pdb")))

    for idx, af3_pdb in enumerate(af3_pdbs):
        af3_structure = load_pdb(af3_pdb)

        # Optional AF3 confidences
        try:
            with open(af3_pdb.replace("_model.pdb", "_summary_confidences.json"), "r") as f:
                summary_conf = json.load(f)
            with open(af3_pdb.replace("_model.pdb", "_confidences.json"), "r") as f:
                confidences = json.load(f)

            updated[f"iptm_idx_{idx}"] = summary_conf.get("iptm", np.nan)
            updated[f"ptm_idx_{idx}"]  = summary_conf.get("ptm",  np.nan)

            ch_ids = [chain.id for model in af3_structure for chain in model]
            for i, ch in enumerate(ch_ids):
                if "atom_chain_ids" in confidences and "atom_plddts" in confidences:
                    vals = [pl for ch_id, pl in zip(confidences["atom_chain_ids"], confidences["atom_plddts"]) if ch_id == ch]
                    updated[f"chain{ch}_plddt_idx_{idx}"] = float(np.mean(vals)) if vals else np.nan
                if "chain_pair_pae_min" in summary_conf:
                    for j, ch2 in enumerate(ch_ids):
                        updated[f"chain{ch}_chain{ch2}_pae_idx_{idx}"] = summary_conf["chain_pair_pae_min"][i][j]
        except FileNotFoundError:
            updated[f"iptm_idx_{idx}"] = np.nan
            updated[f"ptm_idx_{idx}"]  = np.nan

        # Align & base scores
        ca_rmsd, catres_rmsds = align_and_score(af3_structure, ref_structure, catres_list, chain_id=args.chain_id)
        updated[f"ca_rmsd_idx_{idx}"] = ca_rmsd
        for i, cat in enumerate(catres_list):
            updated[f"{cat['name3']}{i+1}_rmsd_idx_{idx}"] = catres_rmsds[i]
        updated[f"catres_rmsd_idx_{idx}"] = float(np.mean(catres_rmsds)) if catres_rmsds else np.nan

        # pLDDT indices from AF3 PDB (0–100 scale, parsed once)
        per_res_plddt, per_atom_plddt = parse_plddt_indices(af3_pdb)

        # Per-residue catalytic pLDDT
        cat_pls = []
        for i, cat in enumerate(catres_list):
            key = (cat["chain"], cat["res_num"])
            val = per_res_plddt.get(key, np.nan)
            updated[f"{cat['name3']}{i+1}_plddt_idx_{idx}"] = val
            cat_pls.append(val)
        updated[f"catres_plddt_idx_{idx}"] = float(np.nanmean(cat_pls)) if cat_pls else np.nan

        # === NEW: per-atom-group RMSD + pLDDT (0–100) ===
        if atom_groups and catres_list:
            for r_i, cat in enumerate(catres_list):
                resname = cat["name3"]
                chain   = cat["chain"]
                resnum  = cat["res_num"]
                groups = atom_groups.get(resname, [])
                for grp in groups:
                    atom_list = grp["atoms"]
                    label     = grp["label"]

                    # group RMSD (AF3 aligned vs REF) in the same atom order
                    af3_coords, miss_af3 = extract_named_atoms(af3_structure, chain, resnum, atom_list)
                    ref_coords, miss_ref = extract_named_atoms(ref_structure,  chain, resnum, atom_list)
                    if len(miss_af3) == 0 and len(miss_ref) == 0 and af3_coords.size > 0:
                        grp_rmsd = calculate_rmsd(af3_coords, ref_coords)
                    else:
                        grp_rmsd = np.nan
                    updated[f"{resname}{r_i+1}_{label}_idx_{idx}"] = grp_rmsd

                    # group pLDDT (mean over requested atoms) from per_atom_plddt (0–100)
                    vals = []
                    for nm in atom_list:
                        v = per_atom_plddt.get((chain, resnum, nm), np.nan)
                        if not np.isnan(v):
                            vals.append(v)
                    grp_plddt = float(np.mean(vals)) if vals else np.nan
                    updated[f"{resname}{r_i+1}_{label}_plddt_idx_{idx}"] = grp_plddt

    # Dump CSV
    df = pd.DataFrame([updated])
    df.to_csv(args.outscr, index=False)
    print(df)

if __name__ == "__main__":
    main()
