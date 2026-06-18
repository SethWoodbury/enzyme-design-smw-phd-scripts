#!/usr/bin/env python

"""
Author: Seth M Woodbury
Date: 2025-01-03

Description:
    This script wraps Rosetta's `molfile_to_params.py` in order to:
     1) Generate params files from a single .mol2 file containing multiple conformations.
     2) Merge the generated {LIG}.pdb contents on TOP of {LIG}_conformers.pdb contents.

Usage (example command):
    python /home/woodbuse/special_scripts/theozyme_and_ligand_handling/mol2_with_confs_to_params.py \
        --lig_3letter_code LIG \
        --input_mol2_with_confs /path/to/multiple_conformers.mol2
"""

import os
import argparse
import subprocess

def main(lig_code, input_mol2):
    """
    Execute the Rosetta 'molfile_to_params.py' script with the user-provided arguments.
    Then merge {lig_code}.pdb on top of {lig_code}_conformers.pdb.
    """
    # Path to the 'molfile_to_params.py' script - adjust if needed
    molfile_to_params_script = "/software/rosetta/main/source/scripts/python/public/molfile_to_params.py"
    
    # 1) Build the command
    cmd = [
        "python", 
        molfile_to_params_script,
        "--name", lig_code,
        "--conformers-in-one-file", input_mol2,
        "--root_atom=1",
        "--clobber",
    ]

    # 2) Print out command for debugging
    print("[mol2_with_confs_to_params] Running command:")
    print(" ".join(cmd))

    # 3) Execute the command
    subprocess.run(cmd, check=True)

    # 4) After successful run, you should have:
    #    - {lig_code}.pdb
    #    - {lig_code}_conformers.pdb
    #
    # Now we place the contents of {lig_code}.pdb on top of the lines in {lig_code}_conformers.pdb.

    main_pdb_path = f"{lig_code}.pdb"
    conf_pdb_path = f"{lig_code}_conformers.pdb"

    # Safety check: ensure both files exist
    if not os.path.isfile(main_pdb_path):
        print(f"WARNING: File {main_pdb_path} not found. Cannot merge.")
        return
    if not os.path.isfile(conf_pdb_path):
        print(f"WARNING: File {conf_pdb_path} not found. Cannot merge.")
        return

    # Read the lines from both files
    with open(main_pdb_path, "r") as f_main:
        main_lines = f_main.readlines()

    with open(conf_pdb_path, "r") as f_conf:
        conf_lines = f_conf.readlines()

    # Write them back to {lig_code}_conformers.pdb with main file's lines on top
    with open(conf_pdb_path, "w") as f_conf_out:
        f_conf_out.writelines(main_lines)
        f_conf_out.writelines(conf_lines)

    print(f"[mol2_with_confs_to_params] Merged contents of '{main_pdb_path}' on top of '{conf_pdb_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wrap Rosetta's molfile_to_params.py for a single .mol2 with multiple conformations, then merge PDBs."
    )
    parser.add_argument(
        "--lig_3letter_code", 
        type=str, 
        required=True, 
        help="The 3-letter code for the ligand (will produce {code}.pdb and {code}_conformers.pdb)."
    )
    parser.add_argument(
        "--input_mol2_with_confs", 
        type=str, 
        required=True, 
        help="Path to the .mol2 file containing multiple conformers."
    )

    args = parser.parse_args()
    main(lig_code=args.lig_3letter_code, input_mol2=args.input_mol2_with_confs)
