#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:
    Seth M. Woodbury, David Baker Lab, University of Washington

Email:
    woodbuse@uw.edu

Date:
    2025-05-14

Script: fragment_maximal_AA_subset_theozyme_into_manyPDBs__MAIN.py

Purpose:
    - Take an input PDB and produce a filtered PDB containing only a subset of protein residues
    - User may specify either --residues_to_keep or --residues_to_throw_away (mutually exclusive)
    - Always keep all HETATM (ligand) entries
    - Verbosely report which residues are kept and which are removed
    - Remove associated REMARK 666 lines for deleted residues; preserve and reindex kept lines

Usage:
    python fragment_maximal_AA_subset_theozyme_into_manyPDBs__MAIN.py \
        --input_pdb /path/to/input.pdb \
        --output_pdb_path /path/to/output.pdb \
        --residues_to_keep A96 A34 A119

    Or:
    python fragment_maximal_AA_subset_theozyme_into_manyPDBs__MAIN.py \
        --input_pdb /path/to/input.pdb \
        --output_pdb_path /path/to/output.pdb \
        --residues_to_throw_away A7 A92
"""
import argparse
import sys
import os
import re

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter a PDB to keep or remove specified residues, update REMARK 666"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--residues_to_keep', nargs='+',
        help='List of ChainResidue entries (e.g. A96 A34) to retain'
    )
    group.add_argument(
        '--residues_to_throw_away', nargs='+',
        help='List of ChainResidue entries (e.g. A96 A34) to remove'
    )
    parser.add_argument(
        '--input_pdb', required=True,
        help='Path to the input PDB file'
    )
    parser.add_argument(
        '--output_pdb_path', required=True,
        help='Path for the filtered PDB output'
    )
    return parser.parse_args()


def parse_chain_res(spec):
    """Parse 'A96' → ('A',96)"""
    chain = spec[0]
    num = int(spec[1:])
    return chain, num


def main():
    args = parse_args()

    # Make sure output directory exists
    os.makedirs(os.path.dirname(args.output_pdb_path), exist_ok=True)

    # Build sets of keys
    if args.residues_to_keep:
        keep_keys = [parse_chain_res(r) for r in args.residues_to_keep]
        keep_keys = set(keep_keys)
        mode = 'keep'
    else:
        throw_keys = set(parse_chain_res(r) for r in args.residues_to_throw_away)
        mode = 'throw'

    # Read all lines
    with open(args.input_pdb) as f:
        lines = f.readlines()

    # Segment into prefix, remark666_lines, suffix
    prefix = []
    remark666 = []
    suffix = []
    state = 'prefix'
    for L in lines:
        if state == 'prefix':
            if L.startswith('REMARK 666'):
                state='remark'; remark666.append(L)
            else:
                prefix.append(L)
        elif state == 'remark':
            if L.startswith('REMARK 666'):
                remark666.append(L)
            else:
                state='suffix'; suffix.append(L)
        else:
            suffix.append(L)

    # Parse remark666 entries
    remarks = []  # list of dicts
    for L in remark666:
        tok = L.split()
        # e.g. ['REMARK','666','MATCH','TEMPLATE','X',lig,'0','MATCH','MOTIF','A','HIS',' 37','  1','  1']
        ligand = tok[5]
        chain  = tok[9]
        res    = tok[10]
        num    = int(tok[11])
        remarks.append({'chain':chain,'res':res,'num':num,'ligand':ligand, 'orig':L})

    # Determine final keep_keys if throwing mode
    if mode=='throw':
        all_keys = {(r['chain'], r['num']) for r in remarks}
        keep_keys = all_keys - throw_keys

    # Verbose report
    sys.stdout.write(f"[INFO] Mode: {mode}.\n")
    for r in remarks:
        key = (r['chain'],r['num'])
        if key in keep_keys:
            sys.stdout.write(f"[INFO] Keeping residue {r['chain']}{r['num']}\n")
        else:
            sys.stdout.write(f"[INFO] Removing residue {r['chain']}{r['num']}\n")

    # Filter remark lines and reindex
    new_remarks = []
    idx = 1
    for r in remarks:
        key = (r['chain'],r['num'])
        if key in keep_keys:
            line = (
                f"REMARK 666 MATCH TEMPLATE X {r['ligand']:<3}    0 MATCH MOTIF "
                f"{r['chain']} {r['res']:<3} {r['num']:>4}{idx:>4}{1:>3}\n"
            )
            new_remarks.append(line)
            idx += 1

    # Filter suffix (ATOM/HETATM) lines
    new_suffix = []
    atom_re = re.compile(r'^ATOM')
    het_re  = re.compile(r'^HETATM')
    for L in suffix:
        if atom_re.match(L):
            chain = L[21]
            num   = int(L[22:26])
            if (chain,num) in keep_keys:
                new_suffix.append(L)
        elif het_re.match(L):
            new_suffix.append(L)
        else:
            new_suffix.append(L)

    # Write output
    with open(args.output_pdb_path, 'w') as out:
        for L in prefix:
            out.write(L)
        for L in new_remarks:
            out.write(L)
        for L in new_suffix:
            out.write(L)

    sys.stdout.write(f"[INFO] Written filtered PDB to {args.output_pdb_path}\n")

if __name__=='__main__':
    main()
