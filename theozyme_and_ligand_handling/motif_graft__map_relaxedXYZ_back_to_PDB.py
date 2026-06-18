#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Seth M. Woodbury
Date: 05-05-2025

Reverse‑map energy‑minimized XYZ back onto original PDB topology via per‑residue,
centroid‑based Kabsch superposition.

This script grafts “missing” atoms into a reference PDB by using the rigid‐body
transform that best aligns the original atom scaffold of each residue
(“anchors”) to their new, relaxed XYZ positions.  All atoms present in the
XYZ file remain fixed; any atoms not present are transformed by the same
R+T per‑residue so that local geometry is preserved.

Inputs:
    --pdb        Path to the reference PDB file (ATOM/HETATM entries, plus
                 all REMARK/TITLE/TER lines are preserved verbatim).
    --map_tsv    TSV mapping file with columns:
                   pdb_idx   (1‑based index into the PDB’s ATOM list)
                   mod_idx   (1‑based index into the XYZ file)
                 Only entries where both indices are valid will be used.
    --min_xyz    Relaxed/energy‑minimized XYZ file.  The XYZ atom order must
                 correspond exactly to the mod_idx values in the TSV.
    --output_pdb Path to write the grafted PDB.  The output will have the same
                 headers, formatting, and atom ordering as the input, but with
                 updated X/Y/Z columns.

Outputs:
    A new PDB file where:
      • All “anchor” atoms (those mapped to XYZ) are locked exactly to their
        relaxed XYZ coordinates.
      • All other atoms in the same residue are transformed by the same
        rotation+translation that best superimposes the original anchors onto
        the relaxed anchors.
      • Residues with fewer than 3 anchors are skipped (warned) and left at
        their original coordinates.
      • Extensive debug printout of centroids, singular values, RMSDs, and
        grafted coordinates is provided.

Example command:
    python graft_missing_atoms.py \
        --pdb        input.pdb \
        --map_tsv    mapping.tsv \
        --min_xyz    relaxed.xyz \
        --output_pdb grafted.pdb

Logic flow:
 1. Parse arguments and read:
       – PDB lines + ATOM list (chain, resSeq, atom name, original coords)
       – TSV → mapping from PDB index → XYZ index
       – relaxed XYZ coordinates into a list
 2. Initialize an array `newC` of length N_atoms:
       – For each mapped PDB index p → XYZ index v, set newC[p-1] = xyz_pts[v-1]
         (these anchors will never move again)
 3. Group atoms by residue key (chain, resSeq).
 4. For each residue:
       a. Identify “anchors” = indices i in this residue with newC[i] not None.
       b. Identify “missing” = indices j with newC[j] is None.
       c. If len(anchors) < 3: warn and skip (leave missing at original coords).
       d. Build two m×3 matrices:
            A = original_coords[anchors]
            B = newC         [anchors]
       e. Compute centroids C_A = mean(A), C_B = mean(B).
       f. Center A_c = A − C_A; B_c = B − C_B.
       g. Compute cross‑covariance H = A_cᵀ·B_c.
       h. SVD: H = U·Σ·Vᵀ; form R = V·Uᵀ; if det(R)<0, flip V’s last row then
          recompute R to guard against reflection.
       i. Translation T = C_B − R·C_A.
       j. Print: C_A, C_B, singular values Σ, translation‑only RMSD,
          full Kabsch RMSD, anchor‑by‑anchor P→Q mappings.
       k. For each missing atom index j in this residue:
            newC[j] = R @ original_coord[j] + T
          and print its grafted coordinate.
 5. Write out the output PDB:
       – Replay every line of the input file.
       – For ATOM/HETATM lines, replace the X/Y/Z fields with either newC[i]
         (if not None) or the original coordinate.
 6. Exit.

Dependencies:
    Python 3, numpy, and the standard library modules argparse, csv, os, math,
    and collections.

"""
import argparse, csv, os, math
import numpy as np
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pdb',        required=True)
    p.add_argument('--map_tsv',    required=True)
    p.add_argument('--min_xyz',    required=True)
    p.add_argument('--output_pdb', required=True)
    return p.parse_args()

def read_pdb(path):
    atoms, lines = [], []
    with open(path) as f:
        for i,l in enumerate(f):
            lines.append(l)
            if l.startswith(('ATOM','HETATM')):
                x,y,z = map(float,(l[30:38],l[38:46],l[46:54]))
                chain = l[21]
                resSeq= int(l[22:26])
                name  = l[12:16].strip()
                atoms.append({'idx':len(atoms),
                              'chain':chain,'resSeq':resSeq,'name':name,
                              'coord':np.array([x,y,z])})
    print(f"[INFO] Read {len(atoms)} ATOM/HETATM from {path}")
    return atoms, lines

def read_xyz(path):
    pts=[]
    with open(path) as f:
        for ln in f.readlines()[2:]:
            p = ln.split()
            if len(p)>=4:
                pts.append(np.array(list(map(float,p[1:4]))))
    print(f"[INFO] Read {len(pts)} relaxed‑XYZ points from {path}")
    return pts

def read_map(path):
    m={}
    with open(path) as f:
        for r in csv.DictReader(f,delimiter='\t'):
            p = int(r['pdb_idx'])
            mo = r['mod_idx'].strip()
            if mo.isdigit():
                m[p] = int(mo)
                print(f"[MAP] pdb_idx={p:3d} → xyz_idx={m[p]:3d}")
    print(f"[INFO] Loaded {len(m)} anchors (pdb_idx→mod_idx)\n")
    return m

def fit_kabsch_centroid(A, B):
    """
    A, B: (m×3) arrays of corresponding anchor points.
    Returns R (3×3), T (3,), Sigma (singular values).
    """
    # 1) centroids
    C_A = A.mean(axis=0)
    C_B = B.mean(axis=0)
    # 2) center
    A_c = A - C_A
    B_c = B - C_B
    # 3) H
    H = A_c.T @ B_c
    # 4) SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # fix reflection
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    # 5) translation
    T = C_B - R @ C_A
    return R, T, S, C_A, C_B

def compute_rmsd(P, Q):
    return np.sqrt(((P-Q)**2).sum(axis=1).mean())

def main():
    args      = parse_args()
    atoms, lines = read_pdb(args.pdb)
    xyz_pts      = read_xyz(args.min_xyz)
    mapping      = read_map(args.map_tsv)

    N    = len(atoms)
    orig = np.vstack([a['coord'] for a in atoms])
    newC = [None]*N

    # 1) lock anchors
    for p,v in mapping.items():
        if 1<=p<=N and 1<=v<=len(xyz_pts):
            newC[p-1] = xyz_pts[v-1]
    locked = sum(1 for c in newC if c is not None)
    print(f"[INFO] Anchors locked: {locked}/{N}\n")

    # 2) group by residue
    by_res = defaultdict(list)
    for i,a in enumerate(atoms):
        by_res[(a['chain'],a['resSeq'])].append(i)

    # 3) per‑residue fit missing
    for res, idxs in sorted(by_res.items()):
        anchors = [i for i in idxs if newC[i] is not None]
        missing = [i for i in idxs if newC[i] is None]
        if not missing: continue

        print(f"[RES] {res}: anchors={anchors}, missing={missing}")
        if len(anchors) < 3:
            print(f"  [WARN] only {len(anchors)} anchors; skipping residue\n")
            continue

        # build A and B
        A = np.vstack([orig[i]   for i in anchors])
        B = np.vstack([newC[i]   for i in anchors])

        # print anchor mapping
        print("  Anchor → row:")
        for r,i in enumerate(anchors):
            a=atoms[i]
            print(f"    row{r:2d}: idx{i+1:3d} {a['chain']}{a['resSeq']} '{a['name']}'")

        # centroid shift only
        C_A = A.mean(axis=0)
        C_B = B.mean(axis=0)
        A_t = A - C_A + C_B
        tr_rmsd = compute_rmsd(A_t, B)
        print(f"  Centroids: C_A={tuple(C_A)}, C_B={tuple(C_B)}")
        print(f"  Translation‑only RMSD = {tr_rmsd:.4f} Å")

        # full Kabsch
        R, T, S, _, _ = fit_kabsch_centroid(A, B)
        print(f"  Singular values Σ = {S.round(5)}")

        A_rt = (R @ A.T).T + T
        full_rmsd = compute_rmsd(A_rt, B)
        print("  A → after R+T (first 5 rows):\n", A_rt[:5])
        print(f"  Full Kabsch RMSD    = {full_rmsd:.4f} Å")

        # graft missing
        print("  Grafting missing atoms:")
        for i in missing:
            g = R @ orig[i] + T
            newC[i] = g
            print(f"    idx{i+1:3d} '{atoms[i]['name']}': → {tuple(g)}")
        print()

    # 4) write new PDB
    os.makedirs(os.path.dirname(args.output_pdb), exist_ok=True)
    ai = 0
    with open(args.output_pdb,'w') as out:
        for l in lines:
            if l.startswith(('ATOM','HETATM')):
                x,y,z = newC[ai] if newC[ai] is not None else orig[ai]
                out.write(f"{l[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{l[54:]}")
                ai+=1
            else:
                out.write(l)

    print(f"[INFO] Wrote {args.output_pdb}")

if __name__ == '__main__':
    main()
