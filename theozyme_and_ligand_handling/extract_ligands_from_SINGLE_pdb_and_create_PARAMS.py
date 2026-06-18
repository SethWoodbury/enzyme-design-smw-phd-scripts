#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract ligands from a PDB, write to XYZ, then generate mol2 and params.

This script takes:
  --input_single_pdb                    Path to the input PDB file.
  --ligands_to_extract_via_3letter_code Optional list of 3‑letter codes to restrict extraction (e.g. ZN1 LIG).
                                        If omitted, all HETATM residues are extracted.
  --output_dir_for_params_stuff         Directory where the XYZ, mol2, and params files will be created.
  --desired_ligand_3letter_code         3‑letter code used in naming the output files.
  --stop_after_XYZ_is_made              If set, only write the XYZ and print downstream commands without executing.

Workflow:
 1. Read the input PDB and collect all HETATM lines.
 2. Filter by the provided 3‑letter codes (if any).
 3. Write an XYZ file containing only the extracted ligand atoms:
       saved_xyz_cropped_ligand_fromSINGLEpdb__lig_{desired_code}.xyz
 4. Change into the output directory.
 5. If --stop_after_XYZ_is_made: print the two downstream commands and exit.
 6. Otherwise, run the fully bonded mol2 generator, then the mol2→params converter.
"""

import argparse
import os
import sys
import subprocess
from collections import defaultdict

# Base directory for the helper scripts
scripts_directory = "/home/woodbuse/special_scripts/theozyme_and_ligand_handling/"

def parse_args():
    p = argparse.ArgumentParser(description="Extract ligand and generate mol2 & params")
    p.add_argument('--input_single_pdb', required=True, help="Path to input PDB file")
    p.add_argument('--ligands_to_extract_via_3letter_code',
                   nargs='*', default=None,
                   help="Optional list of 3-letter ligand codes to extract (e.g. ZN1 LIG)")
    p.add_argument('--output_dir_for_params_stuff', required=True,
                   help="Directory for XYZ, mol2, and params files")
    p.add_argument('--desired_ligand_3letter_code', required=True,
                   help="3-letter code used in naming output files")
    p.add_argument('--stop_after_XYZ_is_made', action='store_true',
                   help="Only write the XYZ and print downstream commands without executing them")
    return p.parse_args()

def read_pdb_hetatm(pdb_path):
    het_atoms = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('HETATM'):
                resname = line[17:20].strip()
                element = line[76:78].strip() or line[12:14].strip()[0]
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                het_atoms.append((resname, element, x, y, z))
    return het_atoms

def write_xyz(atoms, xyz_path):
    with open(xyz_path, 'w') as out:
        out.write(f"{len(atoms)}\n")
        out.write(f"Extracted ligand XYZ\n")
        for elem, x, y, z in atoms:
            out.write(f"{elem:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")

def main():
    args = parse_args()

    print(f"[INFO] Reading PDB: {args.input_single_pdb}")
    het_atoms = read_pdb_hetatm(args.input_single_pdb)
    print(f"[INFO] Found {len(het_atoms)} HETATM entries")

    # Filter by 3-letter codes if provided
    if args.ligands_to_extract_via_3letter_code:
        desired = set(args.ligands_to_extract_via_3letter_code)
        filtered = [atom for atom in het_atoms if atom[0] in desired]
        print(f"[INFO] Filtering to codes {desired}: {len(filtered)} atoms remain")
    else:
        filtered = het_atoms
        print(f"[INFO] No filter codes given, extracting all HETATM atoms")

    # Prepare output directory and XYZ filename
    os.makedirs(args.output_dir_for_params_stuff, exist_ok=True)
    xyz_fname = f"saved_xyz_cropped_ligand_fromSINGLEpdb__lig_{args.desired_ligand_3letter_code}.xyz"
    xyz_path = os.path.join(args.output_dir_for_params_stuff, xyz_fname)

    # Prepare XYZ atom list: discard resname, keep element+coords
    xyz_atoms = [(elem, x, y, z) for (_, elem, x, y, z) in filtered]

    print(f"[INFO] Writing XYZ to: {xyz_path}")
    write_xyz(xyz_atoms, xyz_path)

    # Change into output directory for follow-up commands
    print(f"[INFO] Changing directory to: {args.output_dir_for_params_stuff}")
    os.chdir(args.output_dir_for_params_stuff)

    # Build downstream commands
    mol2_script = os.path.join(scripts_directory,
        "make_FullyBonded_mol2_file_from_singleXYZ_ThatCanHave_multipleXYZinside.py")
    full_xyz = os.path.join(args.output_dir_for_params_stuff, xyz_fname)
    cmd1 = (
        f"python {mol2_script} "
        f"--input_xyz_that_may_contain_multipleXYZinside {full_xyz} "
        f"--output_mol2_file_basename {args.desired_ligand_3letter_code}"
    )

    params_script = os.path.join(scripts_directory, "mol2_with_confs_to_params.py")
    mol2_file = f"{args.desired_ligand_3letter_code}.mol2"
    cmd2 = (
        f"python {params_script} "
        f"--lig_3letter_code {args.desired_ligand_3letter_code} "
        f"--input_mol2_with_confs {mol2_file}"
    )

    if args.stop_after_XYZ_is_made:
        print("\n[INFO] --stop_after_XYZ_is_made set; not executing downstream steps.")
        print(f"[CMD] {cmd1}")
        print(f"[CMD] {cmd2}")
        sys.exit(0)

    # Execute downstream commands
    print("############################################## COMMAND 1 EXECUTION ##############################################")
    print(f"[CMD] Running:")
    print(f"{cmd1}")
    print("")
    subprocess.run(cmd1.split(), check=True)

    print("")
    print("############################################## COMMAND 2 EXECUTION ##############################################")
    print(f"[CMD] Running:")
    print(f"{cmd2}")
    print("")
    subprocess.run(cmd2.split(), check=True)

    print("")
    print("############################################## DONE ##############################################")
    print("[INFO] Ligand extraction and parameter generation complete.")

if __name__ == "__main__":
    main()
