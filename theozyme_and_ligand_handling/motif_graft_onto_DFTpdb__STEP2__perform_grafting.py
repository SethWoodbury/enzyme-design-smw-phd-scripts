#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: Merge Selected Atoms Between Residues in a PDB
-----------------------------------------------------

Purpose:
    For each specified merge operation:
      - Copy selected atoms from a source residue into a target residue.
      - Remove all other atoms from the source residue.
      - Remove the same set of atoms from the target residue.
      - Update the PDB chain ID and residue number of the moved atoms.
      - Update and prune corresponding REMARK 666 entries:
          • Change the source-residue remark to point at the target.
          • Delete the original target-residue remark.

Inputs:
    --ungrafted_dft_theozyme_pdb_ref    PDB file used for REMARK 666 reference (e.g., DFT-theozyme).
    --pregrafted_STEP1_motif_pdb_file    PDB from STEP1_CLEANED (to merge into).
    --merge                              One or more merge specs of the form:
                                           TGT_CHAIN:TGT_RES<-SRC_CHAIN:SRC_RES:ATOM1,ATOM2,...
                                         e.g.: --merge A:92<-A:1:NE2,CE1,ND1,CD2,CG,CB
    --output_pdb                         (optional) Explicit output path;
                                         otherwise appends __STEP2_GRAFTED to the STEP1 filename.

Output:
    A new PDB file containing:
      - All original HEADER lines
      - Updated REMARK 666 lines
      - ATOM/HETATM/TER records with merged atoms repositioned per spec

Example:
    python merge_pdb_residues_step2.py \
      --ungrafted_dft_theozyme_pdb_ref ref_TS.pdb \
      --pregrafted_STEP1_motif_pdb_file example__STEP1_CLEANED.pdb \
      --merge A:92<-A:1:NE2,CE1,ND1,CD2,CG,CB \
      --merge B:5<-B:3:N,CA,C,O

Author:
    Seth M. Woodbury, David Baker Lab, University of Washington
Date:
    2025-04-29
"""

import argparse
import os
import shutil
import re
import sys
from dataclasses import dataclass
from typing import List

@dataclass
class MergeOp:
    tgt_chain: str
    tgt_res:   int
    src_chain: str
    src_res:   int
    atoms:     List[str]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 2: merge selected atoms between residues in a PDB"
    )
    parser.add_argument(
        '--ungrafted_dft_theozyme_pdb_ref', required=True,
        help="Reference DFT-theozyme PDB (for original REMARK 666)"
    )
    parser.add_argument(
        '--pregrafted_STEP1_motif_pdb_file', required=True,
        help="STEP1_CLEANED motif PDB to graft into"
    )
    parser.add_argument(
        '--merge', required=True, action='append',
        help=(
            "Merge spec, format: TGT_CHAIN:TGT_RES<-SRC_CHAIN:SRC_RES:ATOM1,ATOM2,...\n"
            "e.g. A:92<-A:1:NE2,CE1,ND1,CD2,CG,CB"
        )
    )
    parser.add_argument(
        '--output_pdb', required=False,
        help="Optional explicit output path; default appends __STEP2_GRAFTED"
    )
    return parser.parse_args()


def parse_merge_specs(specs: List[str]) -> List[MergeOp]:
    ops: List[MergeOp] = []
    for raw in specs:
        spec = raw.strip().strip('"').strip("'")
        spec = spec.replace('\\<-', '<-')
        if '<-' not in spec:
            print(f"[ERROR] Merge spec missing '<-': {raw!r}", file=sys.stderr)
            sys.exit(1)
        tgt_part, rest = spec.split('<-', 1)
        # parse target
        try:
            tgt_chain, tgt_res = tgt_part.split(':')
        except ValueError:
            print(f"[ERROR] Target spec invalid: {tgt_part!r}", file=sys.stderr)
            sys.exit(1)
        # parse source and atoms by splitting rest into exactly 3 parts
        parts = rest.split(':', 2)
        if len(parts) != 3:
            print(f"[ERROR] Source/atoms spec invalid: {rest!r}", file=sys.stderr)
            sys.exit(1)
        src_chain, src_res, atom_str = parts
        atoms = [a.strip() for a in atom_str.split(',') if a.strip()]
        try:
            ops.append(MergeOp(
                tgt_chain=tgt_chain,
                tgt_res=int(tgt_res),
                src_chain=src_chain,
                src_res=int(src_res),
                atoms=atoms
            ))
        except ValueError:
            print(f"[ERROR] Invalid residue number in spec: {raw!r}", file=sys.stderr)
            sys.exit(1)
    return ops


def get_output_path(input_path: str) -> str:
    dirpath, filename = os.path.split(input_path)
    name, ext = os.path.splitext(filename)
    return os.path.join(dirpath, f"{name}__STEP2_GRAFTED{ext}")


def extract_lines(path: str):
    header_lines, remark_lines, body_lines = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith('HEADER'):
                header_lines.append(line)
            elif line.startswith('REMARK 666'):
                remark_lines.append(line)
            elif line.startswith(('ATOM', 'HETATM', 'TER')):
                body_lines.append(line)
    return header_lines, remark_lines, body_lines


def parse_remark_entries(remark_lines: List[str]):
    entries = []
    pattern = re.compile(
        r'^(REMARK\s+666\s+MATCH\s+TEMPLATE\s+\S+\s+\S+\s+\d+\s+MATCH\s+MOTIF)'  # prefix
        r'\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)'  # chain, res_name, seq, idx, last
    )
    for line in remark_lines:
        m = pattern.match(line)
        if m:
            prefix, chain, res_name, seq_str, idx_str, last_str = m.groups()
            entries.append({'prefix': prefix,
                             'chain': chain,
                             'res_name': res_name,
                             'seq': int(seq_str),
                             'idx': int(idx_str),
                             'last': int(last_str)})
        else:
            entries.append({'raw': line})
    return entries


def process_remark_entries(entries, merges: List[MergeOp]):
    updated = []
    for entry in entries:
        if 'raw' in entry:
            updated.append(entry['raw']); continue
        # delete target remark
        if any(entry['chain'] == m.tgt_chain and entry['seq'] == m.tgt_res for m in merges):
            print(f"[INFO] Deleting REMARK for target {entry['chain']}{entry['seq']}")
            continue
        # update source remark to target
        for m in merges:
            if entry['chain'] == m.src_chain and entry['seq'] == m.src_res:
                print(f"[INFO] Updating REMARK from {m.src_chain}{m.src_res} -> {m.tgt_chain}{m.tgt_res}")
                entry['chain'], entry['seq'] = m.tgt_chain, m.tgt_res
                break
        # reconstruct line
        updated.append(
            f"{entry['prefix']} {entry['chain']} {entry['res_name']} "
            f"{entry['seq']:>3}  {entry['idx']}  {entry['last']}\n"
        )
    return updated


def process_body_lines(body_lines: List[str], merges: List[MergeOp]):
    processed = []
    for line in body_lines:
        if not line.startswith('ATOM'):
            processed.append(line); continue
        rec_chain, rec_res_str = line[21], line[22:26].strip()
        atom_name = line[12:16].strip()
        try:
            rec_res = int(rec_res_str)
        except ValueError:
            processed.append(line); continue
        drop, new_line = False, line
        for m in merges:
            if rec_chain == m.src_chain and rec_res == m.src_res:
                if atom_name in m.atoms:
                    new_chain, new_resseq = m.tgt_chain, f"{m.tgt_res:>4d}"
                    new_line = line[:21] + new_chain + new_resseq + line[26:]
                else:
                    drop=True
                break
            if rec_chain == m.tgt_chain and rec_res == m.tgt_res and atom_name in m.atoms:
                drop=True; break
        if not drop: processed.append(new_line)
    return processed


def main():
    args = parse_args(); merges = parse_merge_specs(args.merge)
    in_path, out_path = args.pregrafted_STEP1_motif_pdb_file, args.output_pdb or get_output_path(args.pregrafted_STEP1_motif_pdb_file)
    print(f"[INFO] Copying STEP1 file {in_path} -> {out_path}")
    shutil.copy(in_path, out_path)
    header_lines, remark_lines, body_lines = extract_lines(in_path)
    remark_entries = parse_remark_entries(remark_lines)
    new_remarks = process_remark_entries(remark_entries, merges)
    new_body = process_body_lines(body_lines, merges)
    with open(out_path, 'w') as f:
        f.writelines(header_lines)
        f.writelines(new_remarks)
        f.writelines(new_body)
    print(f"[INFO] Merge complete. Output written to {out_path}")

if __name__ == '__main__':
    main()
