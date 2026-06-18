#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:
    Seth M. Woodbury, David Baker Lab, University of Washington

Email:
    woodbuse@uw.edu

Date:
    2025-05-14

Script: prepare_PDB_structure_into_theozyme__MAIN.py

Purpose:
    Wrap and execute the cleanup (step1) and reorder REMARK666 (step2) scripts by constructing
    and running the appropriate Python commands.

Requirements:
    - Python3
    - The step1 and step2 scripts at the hardcoded paths below

Usage:
    python prepare_PDB_structure_into_theozyme__MAIN.py \
        --input_pdb /path/to/input.pdb \
        --output_pdb_path /path/to/output.pdb \
        --ligand_complex_3_letter_name LIG \
        [--remark666_residue_front_order A244 A199] \
        [--remark666_residue_back_order A207 A143]
"""

import os
import argparse
import subprocess
import sys

# Paths to the step1 and step2 scripts
STEP1_SCRIPT = "/home/woodbuse/special_scripts/theozyme_and_ligand_handling/prepare_PDB_structure_into_theozyme__STEP1__cleanPDB_and_addREMARK666.py"
STEP2_SCRIPT = "/home/woodbuse/special_scripts/theozyme_and_ligand_handling/prepare_PDB_structure_into_theozyme__STEP2__reorder_REMARK666_lines.py"


def parse_args():
    p = argparse.ArgumentParser(
        description="Wrapper to run step1 cleanup and step2 reorder of PDB"
    )
    p.add_argument(
        '--input_pdb', required=True,
        help='Path to the input PDB file'
    )
    p.add_argument(
        '--output_pdb_path', required=True,
        help='Destination path for the cleaned PDB file'
    )
    p.add_argument(
        '--ligand_complex_3_letter_name', required=True,
        help='Three-letter code to group all HETATM entries'
    )
    p.add_argument(
        '--remark666_residue_front_order', nargs='*', default=[],
        help='Optional list of ChainResidue entries (e.g. A57) to place first'
    )
    p.add_argument(
        '--remark666_residue_back_order', nargs='*', default=[],
        help='Optional list of ChainResidue entries (e.g. A207) to place last'
    )
    return p.parse_args()


def run_script(cmd, description):
    print(f"[INFO] Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)


def main():
    args = parse_args()

    # make sure output folder exists
    out_dir = os.path.dirname(args.output_pdb_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Step 1: cleanup + REMARK666
    cmd1 = [
        sys.executable,
        STEP1_SCRIPT,
        '--input_pdb', args.input_pdb,
        '--output_pdb_path', args.output_pdb_path,
        '--ligand_complex_3_letter_name', args.ligand_complex_3_letter_name
    ]
    run_script(cmd1, 'Step1 clean-and-add-REMARK666')

    # Step 2: reorder REMARK666
    cmd2 = [
        sys.executable,
        STEP2_SCRIPT,
        '--input_pdb', args.output_pdb_path
    ]
    if args.remark666_residue_front_order:
        cmd2 += ['--remark666_residue_front_order'] + args.remark666_residue_front_order
    if args.remark666_residue_back_order:
        cmd2 += ['--remark666_residue_back_order'] + args.remark666_residue_back_order

    run_script(cmd2, 'Step2 reorder-REMARK666')

    print("[INFO] All steps complete.")

if __name__ == '__main__':
    main()
