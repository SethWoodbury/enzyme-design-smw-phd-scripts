#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Map atoms from a PDB file to an unmodified XYZ file, and from the unmodified XYZ to a modified XYZ.
Records a line-based mapping for reverse lookup.

Usage:
    python map_pdb_to_xyz_mapping.py \
        --pdb <path/to/input.pdb> \
        --unmod_xyz <path/to/unmodified.xyz> \
        --mod_xyz <path/to/modified.xyz> \
        --output_map <path/to/output_mapping.tsv> \
        [--tol 0.05]

Arguments:
    --pdb         PDB file with ATOM/HETATM entries
    --unmod_xyz   "Unmodified" XYZ file (one-to-one with PDB atoms)
    --mod_xyz     "Modified" XYZ file (may differ atom count)
    --output_map  Path to write TSV mapping: pdb_idx,chain,res_name,res_seq,atom_name,element,unmod_idx,mod_idx
    --tol         Distance tolerance in Å for matching coordinates (default: 0.05)

The script will:
 1. Parse PDB ATOM/HETATM lines into a list of atoms (chain, res_name, res_seq, atom_name, x,y,z, element).
 2. Parse unmodified and modified XYZ files into lists of (element, x,y,z).
 3. Ensure PDB and unmodified XYZ have the same atom count, else exit with error.
 4. Map each PDB atom to exactly one unmod XYZ line by matching element and coordinate within tol.
     • First pass uses exact element match.
     • Second pass (if unmatched) uses case-insensitive match.
 5. Map each unmod XYZ line to a modified XYZ line by the same criterion (may skip unmatched).
 6. Report counts of successful mappings and any missing.
 7. Write a TSV mapping file for each PDB atom:
       pdb_idx, chain, res_name, res_seq, atom_name, element, unmod_idx, mod_idx (or blank if missing).
"""
import argparse
import sys
import csv
import os
import math


def parse_args():
    p = argparse.ArgumentParser(description="Map PDB->unmod XYZ->mod XYZ")
    p.add_argument('--pdb',       required=True, help="Input PDB file")
    p.add_argument('--unmod_xyz', required=True, help="Unmodified XYZ file")
    p.add_argument('--mod_xyz',   required=True, help="Modified XYZ file")
    p.add_argument('--output_map',required=True, help="Output TSV mapping file")
    p.add_argument('--tol', type=float, default=0.05,
                   help="Coordinate tolerance in Å for matching")
    return p.parse_args()


def read_pdb_atoms(pdb_path):
    atoms = []
    with open(pdb_path) as f:
        for idx,line in enumerate(f, start=1):
            if line.startswith(('ATOM','HETATM')):
                chain    = line[21].strip()
                res_name = line[17:20].strip()
                res_seq  = line[22:26].strip()
                atom_name= line[12:16].strip()
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                except ValueError:
                    sys.exit(f"[ERROR] Invalid coordinates at PDB line {idx}")
                element = (line[76:78].strip() or atom_name[0]).upper()
                atoms.append({
                    'pdb_idx': len(atoms)+1,
                    'chain':chain,
                    'res_name':res_name,
                    'res_seq':res_seq,
                    'atom_name':atom_name,
                    'element':element,
                    'x':x,'y':y,'z':z
                })
    return atoms


def read_xyz(xyz_path):
    coords = []
    with open(xyz_path) as f:
        lines = f.readlines()
    if len(lines) < 3:
        sys.exit(f"[ERROR] XYZ file {xyz_path} too short")
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4: continue
        elem = parts[0].upper()
        try:
            x,y,z = map(float, parts[1:4])
        except ValueError:
            sys.exit(f"[ERROR] Invalid XYZ coordinates: {line}")
        coords.append({'element':elem,'x':x,'y':y,'z':z})
    return coords


def distance(a,b):
    return math.sqrt((a['x']-b['x'])**2 + (a['y']-b['y'])**2 + (a['z']-b['z'])**2)


def main():
    args = parse_args()
    pdb_atoms = read_pdb_atoms(args.pdb)
    unmod_xyz = read_xyz(args.unmod_xyz)
    mod_xyz   = read_xyz(args.mod_xyz)

    n_pdb = len(pdb_atoms); n_un = len(unmod_xyz)
    print(f"[INFO] Read {n_pdb} PDB atoms; {n_un} unmod XYZ atoms")
    if n_pdb != n_un:
        sys.exit(f"[ERROR] Atom count mismatch: PDB({n_pdb}) vs unmod XYZ({n_un})")

    tol = args.tol
    unmap = {}
    unmatched = []
    # First pass: exact element match
    for atom in pdb_atoms:
        pidx=atom['pdb_idx']; best=None
        for i,u in enumerate(unmod_xyz, start=1):
            if u['element']!=atom['element']: continue
            if distance(atom,u)<=tol:
                best=i; break
        if best is None:
            unmatched.append(atom)
        else:
            unmap[pidx]=best
    # Second pass: case-insensitive (attempt for unmatched)
    for atom in list(unmatched):
        pidx=atom['pdb_idx']; best=None
        for i,u in enumerate(unmod_xyz, start=1):
            if u['element'].lower()!=atom['element'].lower(): continue
            if distance(atom,u)<=tol:
                best=i; break
        if best:
            unmap[pidx]=best; unmatched.remove(atom)
    for atom in unmatched:
        print(f"[WARN] No unmod mapping for PDB atom {atom['pdb_idx']} {atom['atom_name']} {atom['chain']}{atom['res_seq']} element={atom['element']}")
    print(f"[INFO] Mapped {len(unmap)}/{n_pdb} atoms PDB->unmod within tol={tol} Å")

    mmap = {}
    # unmod->mod (exact uppercase match only)
    for i,u in enumerate(unmod_xyz, start=1):
        for j,m in enumerate(mod_xyz, start=1):
            if u['element']==m['element'] and distance(u,m)<=tol:
                mmap[i]=j; break
    print(f"[INFO] Mapped {len(mmap)}/{n_un} atoms unmod->mod within tol={tol} Å")

    os.makedirs(os.path.dirname(args.output_map), exist_ok=True)
    with open(args.output_map,'w',newline='') as csvf:
        w=csv.writer(csvf,delimiter='\t')
        w.writerow(['pdb_idx','chain','res_name','res_seq','atom_name','element','unmod_idx','mod_idx'])
        for atom in pdb_atoms:
            p=atom['pdb_idx']; um=unmap.get(p,''); mm=mmap.get(um,'')
            w.writerow([p,atom['chain'],atom['res_name'],atom['res_seq'],atom['atom_name'],atom['element'],um,mm])
    print(f"[INFO] Wrote mapping to {args.output_map}")

if __name__=='__main__':
    main()
