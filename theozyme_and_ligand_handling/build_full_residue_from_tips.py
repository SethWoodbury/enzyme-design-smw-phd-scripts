#!/usr/bin/env python

import os
import argparse
import numpy as np
import sys
import itertools

# Add the directory containing the helper script to sys.path
sys.path.append('/home/woodbuse/special_scripts/theozyme_and_ligand_handling')  # Update this path as needed

# Import the functions from the helper script
from build_full_residue_from_tips_INITIAL_FUNCTIONS import (
    read_pdb_residues,
    generate_residue_pdbs,
    extract_residue_from_input_pdb,
    modify_pdb_file,
    read_pdb_atoms,
    apply_transformation,
    update_pdb_coordinates
)

def calculate_superimposition(matched_atoms):
    # Prepare coordinate arrays
    P = np.array([pair[0]['coords'] for pair in matched_atoms])  # Standard residue atoms
    Q = np.array([pair[1]['coords'] for pair in matched_atoms])  # Input residue atoms
    # Center the coordinates
    P_mean = P.mean(axis=0)
    Q_mean = Q.mean(axis=0)
    P_centered = P - P_mean
    Q_centered = Q - Q_mean
    # Compute covariance matrix
    C = np.dot(P_centered.T, Q_centered)
    # Singular Value Decomposition
    V, S, Wt = np.linalg.svd(C)
    # Compute rotation matrix
    d = np.sign(np.linalg.det(np.dot(Wt.T, V.T)))
    D = np.diag([1, 1, d])
    U = np.dot(np.dot(Wt.T, D), V.T)
    # Compute translation vector
    translation = Q_mean - np.dot(P_mean, U)
    # Compute RMSD
    P_transformed = np.dot(P_centered, U)
    diff = P_transformed - Q_centered
    rmsd = np.sqrt((diff ** 2).sum() / len(P))
    return U, translation, rmsd

def superimpose_residues(residues_info, debug=False):
    import itertools
    for res_info in residues_info:
        res_name = res_info['resname']
        res_num = res_info['resnum']
        chain_id = res_info['chain']
        # Filenames for the standard and input residues
        standard_pdb = f"{res_name}{res_num}_rosetta_TEMP.pdb"
        input_pdb = f"{res_name}{res_num}_inputpdb_TEMP.pdb"
        if not os.path.isfile(standard_pdb):
            print(f"Standard PDB file {standard_pdb} not found for residue {res_name}{res_num}. Skipping.")
            continue
        if not os.path.isfile(input_pdb):
            print(f"Input PDB file {input_pdb} not found for residue {res_name}{res_num}. Skipping.")
            continue
        # Read atoms from both PDB files
        standard_atoms = read_pdb_atoms(standard_pdb, include_hydrogens=False)   # Exclude hydrogens
        input_atoms = read_pdb_atoms(input_pdb, include_hydrogens=False)         # Exclude hydrogens
        if len(input_atoms) < 3:
            print(f"Not enough atoms in input residue {res_name}{res_num} for superimposition.")
            continue
        # Build element-wise atom lists
        matched_atoms = []
        for element in set(atom['element'] for atom in input_atoms):
            std_atoms = [atom for atom in standard_atoms if atom['element'] == element]
            inp_atoms = [atom for atom in input_atoms if atom['element'] == element]
            N_s = len(std_atoms)
            N_i = len(inp_atoms)
            if N_s == 0 or N_i == 0:
                continue  # No matching atoms
            if N_s < N_i:
                print(f"Not enough standard atoms for element {element} in residue {res_name}{res_num}. Skipping element.")
                continue
            if debug:
                print(f"Element {element}: {N_s} standard atoms, {N_i} input atoms")
            # Generate all possible combinations of standard atoms
            std_combinations = itertools.combinations(std_atoms, N_i)
            best_rmsd = float('inf')
            best_mapping = None
            # For each combination, generate all permutations
            for std_combo in std_combinations:
                permutations = itertools.permutations(std_combo)
                for perm in permutations:
                    pairs = list(zip(perm, inp_atoms))
                    # Calculate RMSD for this pairing
                    rotation_matrix, translation_vector, rmsd = calculate_superimposition(pairs)
                    if rmsd < best_rmsd:
                        best_rmsd = rmsd
                        best_mapping = pairs
            if best_mapping is not None:
                matched_atoms.extend(best_mapping)
                if debug:
                    print(f"Best RMSD for element {element} in residue {res_name}{res_num}: {best_rmsd:.4f} Å")
                    for std_atom, inp_atom in best_mapping:
                        print(f"Matched standard atom {std_atom['atom_name']} to input atom {inp_atom['atom_name']}")
            else:
                print(f"No valid mappings found for element {element} in residue {res_name}{res_num}.")
        if len(matched_atoms) < 3:
            print(f"Not enough matched atoms for residue {res_name}{res_num}. Skipping superimposition.")
            continue
        # Perform superimposition
        rotation_matrix, translation_vector, rmsd = calculate_superimposition(matched_atoms)
        # Apply transformation to standard atoms (excluding hydrogens)
        best_transformed_atoms = apply_transformation(standard_atoms, rotation_matrix, translation_vector)
        # Write updated coordinates back to the standard PDB file
        update_pdb_coordinates(standard_pdb, best_transformed_atoms)
        print(f"Superimposed residue {res_name}{res_num} and updated coordinates in {standard_pdb}.")
        print(f"RMSD after superimposition: {rmsd:.4f} Å")

def main(input_pdb, debug=False):
    residues_info = read_pdb_residues(input_pdb)
    if not residues_info:
        print("No residues found in the input PDB file.")
        return
    generate_residue_pdbs(residues_info)
    extract_residue_from_input_pdb(input_pdb, residues_info)
    superimpose_residues(residues_info, debug=debug)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate standard residue PDBs, extract residues from input PDB, and superimpose them.")
    parser.add_argument("-input_pdb", required=True, help="Input PDB file")
    parser.add_argument("-debug", action='store_true', help="Enable debug mode")
    args = parser.parse_args()
    main(args.input_pdb, debug=args.debug)
