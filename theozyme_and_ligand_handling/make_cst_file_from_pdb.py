#!/usr/bin/env python3
"""
Generate a Rosetta .cst file from a PDB and user-specified residue pairs.
Supports automatic 3‑letter code detection, optional custom tolerances,
reordering based on REMARK 666 mapping, enhanced block formatting,
and verbose logging.
"""

import sys
import json
import re
import math
import argparse
from textwrap import dedent
import numpy as np

# --- PDB parsing classes ---
class AtomRecord:
    def __init__(self, name, resName, resSeq, coord):
        self.name = name.strip()
        self.resName = resName.strip()
        self.resSeq = resSeq
        self.coord = coord

    @classmethod
    def from_str(cls, line):
        name = line[12:16]
        resName = line[17:20]
        resSeq = int(line[22:26])
        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        return cls(name, resName, resSeq, np.array([x, y, z]))

class Residue:
    def __init__(self, atom_records):
        self.atom_records = atom_records
        self.coords = [a.coord for a in atom_records]
        self.resName = atom_records[0].resName if atom_records else ''
        self.resSeq = atom_records[0].resSeq if atom_records else None

    @classmethod
    def from_records(cls, records):
        return cls(records)

# --- Geometry measurement ---
def measure_distance(x, y):
    return math.sqrt(((x - y)**2).sum())

def measure_angle(a, b, c):
    ba = a - b
    bc = c - b
    ca = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return math.degrees(math.acos(max(min(ca, 1.0), -1.0)))

def measure_dihedral(p0, p1, p2, p3):
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1; b2 = p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w); y = np.dot(np.cross(b1, v), w)
    return math.degrees(math.atan2(y, x))

# combine into 6D geometry
def measure_geometry(r1_xyz, r2_xyz):
    r1a1, r1a2, r1a3 = r1_xyz
    r2a1, r2a2, r2a3 = r2_xyz
    d   = measure_distance(r1a1, r2a1)
    aA  = measure_angle(r1a2, r1a1, r2a1)
    aB  = measure_angle(r1a1, r2a1, r2a2)
    dA  = measure_dihedral(r1a3, r1a2, r1a1, r2a1)
    dAB = measure_dihedral(r1a2, r1a1, r2a1, r2a2)
    dB  = measure_dihedral(r1a1, r2a1, r2a2, r2a3)
    return d, aA, aB, dA, dAB, dB

# --- Read PDB and optional REMARK mapping ---
def read_models(pdb_path):
    models=[]; curr_model=[]; curr_atoms=[]; last_seq=None; multi=False
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('REMARK 666'): continue
            if line.startswith('MODEL'):
                multi=True
                if curr_atoms:
                    curr_model.append(Residue.from_records(curr_atoms)); curr_atoms=[]
                if curr_model:
                    models.append(curr_model)
                curr_model=[]; last_seq=None
                continue
            if line.startswith('ENDMDL'):
                if curr_atoms:
                    curr_model.append(Residue.from_records(curr_atoms)); curr_atoms=[]
                models.append(curr_model)
                curr_model=[]; last_seq=None
                continue
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom=AtomRecord.from_str(line)
                if last_seq is None or atom.resSeq!=last_seq:
                    if curr_atoms:
                        curr_model.append(Residue.from_records(curr_atoms))
                    curr_atoms=[atom]; last_seq=atom.resSeq
                else:
                    curr_atoms.append(atom)
    if curr_atoms:
        curr_model.append(Residue.from_records(curr_atoms))
    if not multi:
        models.append(curr_model)
    return models

def parse_remark_666(pdb_path):
    pat=re.compile(r'MOTIF\s+([A-Za-z0-9])\s+([A-Z]{3})\s+(\d+)\s+(\d+)')
    mapp={}
    with open(pdb_path) as f:
        for L in f:
            if not L.startswith('REMARK 666'): continue
            m=pat.search(L)
            if m:
                chain, _, seq, idx=m.group(1),m.group(2),int(m.group(3)),int(m.group(4))
                mapp[f"{chain}{seq}"]=idx
    return mapp

# --- Extract atom coord ---
def get_xyz(model, chain_seq, atom_name):
    seq=int(chain_seq[1:])
    for res in model:
        if res.resSeq==seq:
            for atom in res.atom_records:
                if atom.name==atom_name:
                    return atom.coord
    sys.exit(f"Atom {atom_name} in {chain_seq} not found")

# --- Block builder ---
def build_block(idx, entry, verbose=False):
    # unpack and optionally log
    name, r1, r2 = entry['name'], entry['r1_atoms'], entry['r2_atoms']
    d_tol, a_tol = entry.get('dist_tol', 0.02), entry.get('ang_tol', 5.0)
    if verbose:
        print(f"[DEBUG] Block {idx}: {name}, dist_tol={d_tol:.4f}, ang_tol={a_tol:.2f}", file=sys.stderr)
    # format geometry
    g6=entry['g6']
    dist_str = f"{g6[0]:.4f}".rjust(9)
    ang_strs = [f"{v:.2f}".rjust(7) for v in g6[1:]]
    dist_tol_str = f"{d_tol:.4f}".rjust(9)
    ang_tol_str  = f"{a_tol:.2f}".rjust(7)
    cov_flag = 1 if entry['covalent'] else 0

    lines=[f"### --- BLOCK {idx} START --- ###",
           "VARIABLE_CST::BEGIN",
           f"# {name}",
           "CST::BEGIN","",  
           f"  TEMPLATE::   ATOM_MAP: 1 atom_name: {' '.join(r1)}",
           f"  TEMPLATE::   ATOM_MAP: 1 residue3:  {entry['r1_code']}","",
           f"  TEMPLATE::   ATOM_MAP: 2 atom_name: {' '.join(r2)}",
           f"  TEMPLATE::   ATOM_MAP: 2 residue3:  {entry['r2_code']}","",
           f"  CONSTRAINT:: distanceAB: {dist_str} {dist_tol_str}   150     {cov_flag}   1",
           f"  CONSTRAINT::    angle_A: {ang_strs[0]}   {ang_tol_str}      50   360.  1",
           f"  CONSTRAINT::    angle_B: {ang_strs[1]}   {ang_tol_str}      50   360.  1",
           f"  CONSTRAINT::  torsion_A: {ang_strs[2]}   {ang_tol_str}      50   360.  1",
           f"  CONSTRAINT:: torsion_AB: {ang_strs[3]}   {ang_tol_str}      50   360.  1",
           f"  CONSTRAINT::  torsion_B: {ang_strs[4]}   {ang_tol_str}      50   360.  1","",
           "  ALGORITHM_INFO:: match",
           "   MAX_DUNBRACK_ENERGY 50.0"]
    if not entry['primary']:
        lines.append("   SECONDARY_MATCH: DOWNSTREAM")
    lines.extend(["  ALGORITHM_INFO::END","CST::END","\n"])
    return "\n".join(lines)

# --- Main ---
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input_pdb', required=True)
    parser.add_argument('--output_cst', required=True)
    parser.add_argument('--specs', required=True, help='JSON string of constraint specs')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    args=parser.parse_args()

    specs=json.loads(args.specs)
    model=read_models(args.input_pdb)[0]
    remark_map=parse_remark_666(args.input_pdb)
    if args.verbose:
        print(f"[INFO] Read {len(model)} residues from model", file=sys.stderr)
        print(f"[INFO] Found remark mapping entries: {len(remark_map)}", file=sys.stderr)

    entries=[]
    for spec in specs:
        id1,id2=spec['residue_1_identifier__ChainResidue'], spec['residue_2_identifier__ChainResidue']
        code1=spec.get('residue_1_3letter_code') or next((r.resName for r in model if r.resSeq==int(id1[1:])), 'UNK')
        code2=spec.get('residue_2_3letter_code') or next((r.resName for r in model if r.resSeq==int(id2[1:])), 'UNK')
        xyz1=[get_xyz(model,id1,a) for a in spec['residue_1_atoms']]
        xyz2=[get_xyz(model,id2,a) for a in spec['residue_2_atoms']]
        g6=measure_geometry(xyz1,xyz2)
        entries.append({
            'name': spec.get('name') or f"{code1}{id1}_{code2}{id2}",
            'r1_code': code1, 'r2_code': code2,
            'r1_atoms': spec['residue_1_atoms'], 'r2_atoms': spec['residue_2_atoms'],
            'primary': spec.get('primary',True), 'covalent': spec.get('covalent',False),
            'order': remark_map.get(id2,999), 'g6': g6,
            'dist_tol': spec.get('dist_tol',0.02), 'ang_tol': spec.get('ang_tol',5.0)
        })
    entries.sort(key=lambda x:x['order'])

    with open(args.output_cst,'w') as outf:
        for i,e in enumerate(entries,1):
            block=build_block(i,e,verbose=args.verbose)
            outf.write(block)
    print(f"Wrote {len(entries)} constraint blocks to {args.output_cst}")
