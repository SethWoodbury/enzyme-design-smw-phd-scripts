#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:
    Seth M. Woodbury, David Baker Lab, University of Washington

Email:
    woodbuse@uw.edu

Date:
    2025-04-29

Clean Pregrafted Motif PDB Script
---------------------------------

Purpose:
    Automates the “step 1” cleanup of a pregrafted motif PDB by:
      1. Copying the input motif PDB to a new file with suffix “__STEP1_CLEANED.pdb”
      2. Extracting HEADER and existing REMARK 666 lines from a reference DFT theozyme PDB
      3. Parsing existing REMARK 666 entries to avoid duplication
      4. Identifying all unique ATOM residues in the pregrafted motif PDB
      5. Generating and appending new REMARK 666 entries (with sequential indices)
      6. Filtering the final file to only include HEADER, REMARK 666, ATOM, HETATM, and TER records

Usage:
    python clean_pregrafted_motif.py \
        --ungrafted_dft_theozyme_pdb_ref <path/to/reference_DFT_TS.pdb> \
        --pregrafted_motif_pdb_file <path/to/grafted_motif.pdb>

Example:
    python /home/woodbuse/special_scripts/theozyme_and_ligand_handling/\
motif_graft_onto_DFTpdb__STEP1__remark666_pregrafted_pdb.py \
      --ungrafted_dft_theozyme_pdb_ref \
/home/woodbuse/projects/zn_amidase/amcPA_substrate/theozymes/combined_theozymes/\
group1_HEXXH_Yoxyhole_amcPA_dftTS_lig_HAT.pdb \
      --pregrafted_motif_pdb_file \
/home/woodbuse/projects/zn_amidase/amcPA_substrate/theozymes/combined_theozymes/\
grafted_motif/group1_HEXXH_Yoxyhole_amcPA_dftTS_lig_HAT_preGRAFTED_MOTIF.pdb

Inputs:
    --ungrafted_dft_theozyme_pdb_ref
        Reference PDB containing the original HEADER and REMARK 666 lines.
    --pregrafted_motif_pdb_file
        The pregrafted motif PDB to clean.

Output:
    A new PDB at the same location/name as the input motif file,
    but with “__STEP1_CLEANED.pdb” appended. This file contains:
      • All reference HEADER and REMARK 666 lines
      • Newly generated REMARK 666 lines for any residues not already present
      • Only ATOM, HETATM, and TER records thereafter
"""

import argparse
import os
import shutil
import re
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean pregrafted motif PDB file"
    )
    parser.add_argument(
        '--ungrafted_dft_theozyme_pdb_ref', required=True,
        help="Path to ungrafted DFT theozyme PDB reference file"
    )
    parser.add_argument(
        '--pregrafted_motif_pdb_file', required=True,
        help="Path to pregrafted motif PDB file"
    )
    return parser.parse_args()


def get_output_path(pregrafted_path):
    dirpath, filename = os.path.split(pregrafted_path)
    name, ext = os.path.splitext(filename)
    return os.path.join(dirpath, f"{name}__STEP1_CLEANED{ext}")


def extract_header_and_remarks(path):
    """Extract HEADER and all REMARK 666 lines from reference PDB"""
    header_lines = []
    remark_lines = []
    with open(path) as f:
        for line in f:
            if line.startswith('HEADER'):
                header_lines.append(line)
            elif line.startswith('REMARK 666'):
                remark_lines.append(line)
    return header_lines, remark_lines


def parse_existing_motifs(remark_lines):
    """
    Parse existing REMARK 666 motif entries.
    Returns a set of (chain, res_name, res_seq) and max remark index.
    """
    existing = set()
    max_idx = 0
    # Match lines like:
    # REMARK 666 MATCH TEMPLATE X HAT    0 MATCH MOTIF A HIS  1  1  1
    pattern = re.compile(
        r'^REMARK\s+666\s+MATCH\s+TEMPLATE\s+\S+\s+\S+\s+\d+\s+MATCH\s+MOTIF'
        r'\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)'
    )
    for line in remark_lines:
        m = pattern.match(line)
        if m:
            chain, res_name, seq_str, idx_str = m.groups()
            existing.add((chain, res_name, seq_str))
            idx = int(idx_str)
            if idx > max_idx:
                max_idx = idx
        else:
            print(f"[WARN] Remark line did not match expected format: {line.strip()}", file=sys.stderr)
    return existing, max_idx


def extract_ligand_code(lines):
    ligands = set()
    for line in lines:
        if line.startswith('HETATM'):
            code = line[17:20].strip()
            if code:
                ligands.add(code)
    if len(ligands) == 1:
        return ligands.pop()
    elif ligands:
        return list(ligands)[0]
    else:
        return 'LIG'


def extract_residues(lines):
    residues = set()
    for line in lines:
        if line.startswith('ATOM'):
            res_name = line[17:20].strip()
            chain    = line[21].strip()
            res_seq  = line[22:26].strip()
            if res_name and chain and res_seq:
                residues.add((chain, res_name, res_seq))
    return residues


def main():
    args = parse_args()
    out_path = get_output_path(args.pregrafted_motif_pdb_file)
    print(f"[INFO] Copying {args.pregrafted_motif_pdb_file} -> {out_path}")
    shutil.copy(args.pregrafted_motif_pdb_file, out_path)

    header_lines, remark_lines = extract_header_and_remarks(
        args.ungrafted_dft_theozyme_pdb_ref
    )
    print(f"[INFO] Found {len(header_lines)} HEADER lines and {len(remark_lines)} REMARK lines in reference")

    existing, max_idx = parse_existing_motifs(remark_lines)
    print(f"[DEBUG] Existing motif entries: {existing}")
    print(f"[DEBUG] Max existing remark index: {max_idx}")

    with open(args.pregrafted_motif_pdb_file) as f:
        pregrafted_lines = f.readlines()
    residues = extract_residues(pregrafted_lines)
    print(f"[INFO] Identified {len(residues)} unique ATOM residues: {sorted(residues, key=lambda x: (x[0], int(x[2])))}")

    new_residues = [r for r in sorted(residues, key=lambda x: (x[0], int(x[2]))) if r not in existing]
    print(f"[INFO] New residues for remarks: {new_residues}")

    ligand = extract_ligand_code(pregrafted_lines)
    print(f"[INFO] Detected ligand code: {ligand}")

    new_remarks = []
    for i, (chain, res_name, res_seq) in enumerate(new_residues, start=1):
        idx = max_idx + i
        remark_line = (
            f"REMARK 666 MATCH TEMPLATE X {ligand:<3}    0 MATCH MOTIF "
            f"{chain} {res_name:<3} {int(res_seq):>3}  {idx:<1}  {1}\n"
            #f"{chain} {res_name:<3} {int(res_seq):>3}  {idx:<3}  {1:<3}\n"
            #f"{chain} {res_name:<3} {int(res_seq):>3}  {idx:<2}{1:<2}\n"
        )
        print(f"[DEBUG] Adding remark: {remark_line.strip()}")
        new_remarks.append(remark_line)

    # assemble final lines: keep original order of header + remarks from reference
    filtered = []
    filtered.extend(header_lines)
    filtered.extend(remark_lines)
    # then append only new remarks
    filtered.extend(new_remarks)
    # then all ATOM/HETATM/TER lines
    for line in pregrafted_lines:
        if line.startswith(('ATOM', 'HETATM', 'TER')):
            filtered.append(line)

    with open(out_path, 'w') as f:
        f.writelines(filtered)
    print(f"[INFO] Output written to {out_path}")

if __name__ == '__main__':
    main()
