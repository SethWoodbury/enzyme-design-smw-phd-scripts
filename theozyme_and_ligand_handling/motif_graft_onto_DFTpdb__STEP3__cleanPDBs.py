#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean a PDB file by filtering, sorting, renumbering, and verifying entries.

Steps:
 1. Copy the input dirty PDB to a new file suffixed with __CLEANEDpdb
 2. Remove any lines not starting with HEADER, REMARK, ATOM, or HETATM
 3. From preserved lines:
    - Collect HEADER lines (keep original order)
    - Collect REMARK lines:
        • Parse and extract REMARK 666 entries with their original index.
        • Sort these REMARK 666 entries by old index.
        • Renumber their index field sequentially starting at 1, preserving spacing.
        • Keep all other REMARK lines in original order after REMARK 666s.
    - Collect ATOM and HETATM lines.
    - Verify that HETATM atom names (columns 13-16) are unique.
        • Print result; if duplicates, rename HETATM atom names by element + counter.
    - Verify that HETATM chain IDs are disjoint from ATOM chain IDs.
        • Print result; if overlapping, pick a new unused chain and assign to all HETATM.
    - Sort ATOM lines by chain then residue number; HETATM by original atom serial.
    - Fix any stray chain-ID characters in column 72 for all ATOM/HETATM lines.
    - Renumber all HETATM serials 1..N, then all ATOM serials N+1..M (columns 7-11).
    - Place ATOM block first, then TER, then HETATM block, then TER.
 4. Write out: HEADERs, sorted/renumbered REMARKs, ATOMs+TER, HETATMs+TER

Usage:
    python clean_pdb.py --input_dirty_PDB <path/to/dirty.pdb>

Author: Seth M. Woodbury, David Baker Lab
Date:   2025-04-29
"""
import argparse
import shutil
import os
import re
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Clean a PDB file.")
    p.add_argument(
        '--input_dirty_PDB', required=True,
        help="Path to dirty PDB file"
    )
    return p.parse_args()


def get_output_path(path):
    d, f = os.path.split(path)
    name, ext = os.path.splitext(f)
    return os.path.join(d, name + '__CLEANEDpdb' + ext)


def main():
    args = parse_args()
    inp = args.input_dirty_PDB
    outp = get_output_path(inp)
    print(f"[INFO] Copying {inp} -> {outp}")
    shutil.copy(inp, outp)

    headers = []
    remark666 = []  # list of tuples (old_idx, raw_line, match)
    other_remarks = []
    atoms = []
    hetatms = []

    # regex to parse REMARK 666 with preserved spacing
    r66 = re.compile(
        r'^(?P<prefix>REMARK\s+666\s+MATCH TEMPLATE X\s+\S+\s+\d+\s+MATCH MOTIF\s+\S+\s+\S+\s+)'  # prefix
        r'(?P<seq>\d+)(?P<sp1>\s+)'  # residue
        r'(?P<idx>\d+)(?P<sp2>\s+)'  # old index
        r'(?P<last>\d+)(?P<suffix>.*)$'  # final number + suffix
    )

    print("[INFO] Parsing lines from intermediate CLEANED file...")
    with open(outp) as f:
        for line in f:
            if line.startswith('HEADER'):
                headers.append(line)
            elif line.startswith('REMARK'):
                m = r66.match(line.rstrip('\n'))
                if m:
                    old_idx = int(m.group('idx'))
                    remark666.append((old_idx, line.rstrip('\n'), m))
                else:
                    other_remarks.append(line)
            elif line.startswith('ATOM'):
                atoms.append(line)
            elif line.startswith('HETATM'):
                hetatms.append(line)
            else:
                # dropped line
                pass

    # process REMARK 666: sort and renumber
    print(f"[INFO] Found {len(remark666)} REMARK 666 entries; renumbering...")
    remark666.sort(key=lambda x: x[0])
    new_remark_lines = []
    for new_idx, (old_idx, raw, match) in enumerate(remark666, start=1):
        prefix = match.group('prefix')
        seq    = match.group('seq')
        sp1    = match.group('sp1')
        sp2    = match.group('sp2')
        last   = match.group('last')
        suffix = match.group('suffix')
        line = f"{prefix}{seq}{sp1}{new_idx}{sp2}{last}{suffix}\n"
        new_remark_lines.append(line)
        print(f"[DEBUG] REMARK old={old_idx} -> new={new_idx}")

    # combine all remarks
    remarks = new_remark_lines + other_remarks

    # verify HETATM atom-name uniqueness
    het_names = [l[12:16].strip() for l in hetatms]
    unique_names = len(het_names) == len(set(het_names))
    print(f"[INFO] HETATM names unique? {unique_names}")
    if not unique_names:
        print("[INFO] Renaming duplicate HETATM atom names...")
        elem_counts = {}
        updated_het = []
        for line in hetatms:
            element = line[76:78].strip() or line[12:13].strip()
            count = elem_counts.get(element, 0) + 1
            elem_counts[element] = count
            new_name = f"{element}{count}"
            line = line[:12] + f"{new_name:>4}" + line[16:]
            updated_het.append(line)
        hetatms = updated_het

    # verify HETATM chain uniqueness
    atom_chains = {l[21] for l in atoms}
    het_chains  = {l[21] for l in hetatms}
    disjoint = atom_chains.isdisjoint(het_chains)
    print(f"[INFO] HETATM chains disjoint from ATOM chains? {disjoint}")
    if not disjoint:
        candidates = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        for c in candidates:
            if c not in atom_chains:
                new_chain = c
                break
        print(f"[INFO] Reassigning HETATM chain to '{new_chain}'")
        hetatms = [l[:21] + new_chain + l[22:] for l in hetatms]

    # sort ATOM and HETATM blocks
    print("[INFO] Sorting ATOM and HETATM blocks...")
    atoms_sorted = sorted(atoms, key=lambda l: (l[21], int(l[22:26])))
    het_sorted   = sorted(hetatms, key=lambda l: int(l[6:11]))

    # renumber serials and fix chain-ID at col72
    het_lines = []
    atom_lines = []
    count = 1
    print("[INFO] Renumbering HETATM serials...")
    for l in het_sorted:
        l = l[:72] + ' ' + l[73:]
        het_lines.append(f"HETATM{count:5d}" + l[11:])
        print(f"[DEBUG] HETATM new serial {count}")
        count += 1
    het_lines.append('TER   \n')
    print("[INFO] Prepared HETATM block + TER")

    print("[INFO] Renumbering ATOM serials...")
    for l in atoms_sorted:
        l = l[:72] + ' ' + l[73:]
        atom_lines.append(f"ATOM  {count:5d}" + l[11:])
        print(f"[DEBUG] ATOM  new serial {count}")
        count += 1
    atom_lines.append('TER   \n')
    print("[INFO] Prepared ATOM block + TER")

    # write out final cleaned PDB (ATOM block first)
    print(f"[INFO] Writing cleaned PDB to {outp}...")
    with open(outp, 'w') as out:
        out.writelines(headers)
        out.writelines(remarks)
        out.writelines(atom_lines)
        out.writelines(het_lines)

    print(f"[INFO] Cleaned PDB written to {outp}")

if __name__ == '__main__':
    main()
