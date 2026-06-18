#!/usr/bin/env python3

import os
import subprocess
import argparse
from collections import Counter
from pathlib import Path
_HERE = Path(__file__).resolve().parent

# --- locate repo root + shared external paths ---
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "repo_paths.py").is_file():
        _sys.path.insert(0, str(_anc)); break
import repo_paths

DEBUGGING = False

# Define the SMILES library for amino acids with the updated structures
amino_acid_data = {
    "ALA": {"smiles": "C[C@H](N)C=O", "features": {"C": 3, "N": 1, "O": 1}},
    "CYS": {"smiles": "N[C@H](C=O)CS", "features": {"C": 3, "N": 1, "O": 1, "S": 1}},
    "ASP": {"smiles": "N[C@H](C=O)CC(=O)O", "features": {"C": 4, "N": 1, "O": 3}},
    "GLU": {"smiles": "N[C@H](C=O)CCC(=O)O", "features": {"C": 5, "N": 1, "O": 3}},
    "PHE": {"smiles": "N[C@H](C=O)Cc1ccccc1", "features": {"C": 9, "N": 1, "O": 1}},
    "GLY": {"smiles": "NCC=O", "features": {"C": 2, "N": 1, "O": 1}},
    "HIS": {"smiles": "N[C@H](C=O)Cc1c[nH]cn1", "features": {"C": 6, "N": 3, "O": 1}},
    "ILE": {"smiles": "CC[C@H](C)[C@H](N)C=O", "features": {"C": 6, "N": 1, "O": 1}},
    "LYS": {"smiles": "NCCCC[C@H](N)C=O", "features": {"C": 6, "N": 2, "O": 1}},
    "LEU": {"smiles": "CC(C)C[C@H](N)C=O", "features": {"C": 6, "N": 1, "O": 1}},
    "MET": {"smiles": "CSCC[C@H](N)C=O", "features": {"C": 5, "N": 1, "O": 1, "S": 1}},
    "ASN": {"smiles": "NC(=O)C[C@H](N)C=O", "features": {"C": 4, "N": 2, "O": 2}},
    "PRO": {"smiles": "O=C[C@@H]1CCCN1", "features": {"C": 5, "N": 1, "O": 1}},
    "GLN": {"smiles": "NC(=O)CC[C@H](N)C=O", "features": {"C": 5, "N": 2, "O": 2}},
    "ARG": {"smiles": "N=C(N)NCCC[C@H](N)C=O", "features": {"C": 6, "N": 4, "O": 1}},
    "SER": {"smiles": "N[C@H](C=O)CO", "features": {"C": 3, "N": 1, "O": 2}},
    "THR": {"smiles": "C[C@@H](O)[C@H](N)C=O", "features": {"C": 4, "N": 1, "O": 2}},
    "VAL": {"smiles": "CC(C)[C@H](N)C=O", "features": {"C": 5, "N": 1, "O": 1}},
    "TRP": {"smiles": "N[C@H](C=O)Cc1c[nH]c2ccccc12", "features": {"C": 11, "N": 2, "O": 1}},
    "TYR": {"smiles": "N[C@H](C=O)Cc1ccc(O)cc1", "features": {"C": 9, "N": 1, "O": 2}}
}

def parse_pdb(input_pdb):
    residues = {}
    current_residue = None
    current_residue_lines = []
    residue_index = 1

    with open(input_pdb, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                residue_num = int(line[22:26].strip())
                if residue_num != current_residue:
                    if current_residue is not None:
                        residues[residue_index] = current_residue_lines
                        residue_index += 1
                    current_residue = residue_num
                    current_residue_lines = []
                current_residue_lines.append(line)
        
        if current_residue_lines:
            residues[residue_index] = current_residue_lines

    return residues

def write_residue_pdbs(residues):
    obabel_path = repo_paths.OBABEL
    
    print("### RESIDUE FILE PROCESSING ###")
    for index, lines in residues.items():
        pdb_filename = f"pdb{index}_TEMP.pdb"
        smiles_filename = f"pdb{index}_TEMP_smiles.smi"

        with open(pdb_filename, 'w') as f:
            f.writelines(lines)
        print(f"Created {pdb_filename}")

        try:
            subprocess.run([obabel_path, "-ipdb", pdb_filename, "-ocan", "-O", smiles_filename], check=True)
            print(f"Converted {pdb_filename} to SMILES as {smiles_filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {pdb_filename} to SMILES: {e}")

def count_atoms(smiles):
    atom_counts = Counter(char.upper() for char in smiles if char.upper() in "CNOS")
    return dict(atom_counts)

def match_residue_smiles(tip_atom_residues_3letter=None):
    residue_map = {}

    # Convert tip_atom_residues_3letter entries to uppercase for case-insensitivity
    if tip_atom_residues_3letter is not None:
        tip_atom_residues_3letter = [aa.upper() for aa in tip_atom_residues_3letter]

    print("### RESIDUE MATCHING ###")
    for index in range(1, len(os.listdir()) + 1):
        smi_filename = f"pdb{index}_TEMP_smiles.smi"
        if os.path.exists(smi_filename):
            with open(smi_filename, 'r') as f:
                smiles = f.readline().split()[0].strip()

            fragment_atom_counts = count_atoms(smiles)
            print(f"\nFragment {smi_filename} atom counts: {fragment_atom_counts}")
            
            candidates = []
            candidate_data = {}

            for aa, data in amino_acid_data.items():
                aa_atom_counts = data["features"]
                if all(fragment_atom_counts.get(atom, 0) <= aa_atom_counts.get(atom, 0) for atom in fragment_atom_counts):
                    if tip_atom_residues_3letter is None or aa in tip_atom_residues_3letter:
                        candidates.append(aa)
                        candidate_data[aa] = data["smiles"]

            print(f"Candidates for {smi_filename} after filtering: {candidates}")

            if candidates:
                # Prepare arguments for substructure matching script
                candidate_args = [smiles] + [item for aa in candidates for item in (aa, amino_acid_data[aa]["smiles"])]
                subprocess_args = [
                    "python3",
                    str(_HERE / "identify_residues_based_on_substructure.py")
                ] + candidate_args

                # If DEBUGGING is True, let output print directly; otherwise, capture it
                if DEBUGGING:
                    print("\n### Running substructure match script with debug output ###")
                    result = subprocess.run(subprocess_args)  # Debugging prints to terminal
                    best_match = "Check output above for best match."
                else:
                    # Capture only the final match when debugging is off
                    result = subprocess.run(subprocess_args, capture_output=True, text=True)
                    best_match = result.stdout.strip().splitlines()[-1] if result.stdout else "unknown"
            else:
                best_match = "unknown"

            residue_map[f"pdb{index}_TEMP_smiles"] = best_match
            print(f"Matched {smi_filename} to {best_match}")

    with open("residue_map.txt", "w") as f:
        for pdb_name, aa in residue_map.items():
            f.write(f"{pdb_name} = {aa}\n")
    print("\n### RESIDUE MAP CREATED ###")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse an input PDB file, create separate PDB files for each residue, convert to SMILES, and match against amino acid SMILES.")
    parser.add_argument("-input_pdb", required=True, help="Input PDB file")
    parser.add_argument("-tip_atom_residues_3letter", nargs="+", help="Optional 3-letter codes for residues with specified side chains")

    args = parser.parse_args()

    print("### STARTING RESIDUE IDENTIFICATION ###")
    residues = parse_pdb(args.input_pdb)
    write_residue_pdbs(residues)
    match_residue_smiles(args.tip_atom_residues_3letter)
    print("### RESIDUE IDENTIFICATION COMPLETE ###")
