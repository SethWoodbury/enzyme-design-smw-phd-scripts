#!/usr/bin/env python3
"""
Author: Seth M. Woodbury
Description:
-------------
Scan all PDB files in a given directory, extract protein sequences as fast as possible
(by parsing SEQRES first, then falling back to full parsing), identify duplicates,
optionally move redundant files to a `sequence_duplicates/` subdirectory, and print each
duplicate cluster before a final summary.

Usage Example:
--------------
# to identify & move:
python filter_sequence_duplicates__MAIN.py --pdb_directory /path/to/pdbs/

# to identify only, without moving:
python filter_sequence_duplicates__MAIN.py \
    --pdb_directory /path/to/pdbs/ \
    --identify_duplicateSEQs_but_NO_filter
"""

import os
import argparse
import shutil
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# mapping three-letter → one-letter codes
_three_to_one = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}

def extract_sequence_fast(pdb_directory_pdb_file):
    """
    Parse the actual ATOM records to get the sequence:
    - Walk every ATOM line in file order
    - For each new (chain, residue number), append the one‐letter code
    Returns (pdb_filename, sequence_string).
    """
    pdb_directory, pdb_file = pdb_directory_pdb_file
    pdb_path = os.path.join(pdb_directory, pdb_file)

    seen = set()     # track (chain, resnum) pairs
    seq = []

    try:
        with open(pdb_path) as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                resname = line[17:20].strip().upper()
                chain   = line[21]
                resnum  = line[22:26].strip()
                key = (chain, resnum)
                if key in seen:
                    continue
                seen.add(key)

                aa = _three_to_one.get(resname)
                if aa:
                    seq.append(aa)
        return pdb_file, ''.join(seq)
    except Exception as e:
        # if anything goes wrong reading the file
        print(f"[WARNING] Could not parse ATOMs in {pdb_path}: {e}", flush=True)
        return pdb_file, ''

def main(pdb_directory, dry_run):
    start = time.time()
    dup_dir = os.path.join(pdb_directory, "sequence_duplicates")
    # we still create the dir so user can inspect, even if dry_run
    os.makedirs(dup_dir, exist_ok=True)

    pdb_files = sorted(
        f for f in os.listdir(pdb_directory)
        if f.lower().endswith('.pdb') and os.path.isfile(os.path.join(pdb_directory, f))
    )
    total = len(pdb_files)
    print(f"[INFO] Found {total} PDB files to scan.")

    sequence_to_files = defaultdict(list)
    tasks = [(pdb_directory, f) for f in pdb_files]
    completed = 0

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for pdb_file, seq in executor.map(extract_sequence_fast, tasks):
            completed += 1
            if seq:
                sequence_to_files[seq].append(pdb_file)
            if completed % 1000 == 0:
                print(f"[STATUS] Processed {completed}/{total} files...")

    elapsed = time.time() - start
    print(f"[INFO] Sequence extraction done in {elapsed:.1f}s.\n")

    duplicate_clusters = []
    for seq, files in sequence_to_files.items():
        if len(files) > 1:
            files_sorted = sorted(files, reverse=True)
            keep = files_sorted[0]
            to_move = files_sorted[1:]
            duplicate_clusters.append([keep] + to_move)
            if not dry_run:
                for f in to_move:
                    shutil.move(
                        os.path.join(pdb_directory, f),
                        os.path.join(dup_dir, f)
                    )

    # print all clusters
    for idx, cluster in enumerate(duplicate_clusters, 1):
        print(f"### DUPLICATE CLUSTER {idx} ###")
        for fn in cluster:
            print(fn)
        print()

    # final summary
    total_groups = len(duplicate_clusters)
    total_files_duped = sum(len(c) for c in duplicate_clusters)
    total_moved = 0 if dry_run else (total_files_duped - total_groups)
    retained = total - total_moved

    print("[SUMMARY]")
    print(f"Total duplicate groups found:       {total_groups}")
    print(f"Total files in duplicate clusters:  {total_files_duped}")
    print(f"Files moved to sequence_duplicates/:{total_moved}")
    print(f"Files retained:                     {retained}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pdb_directory', required=True,
        help='Directory containing .pdb files to deduplicate by sequence'
    )
    parser.add_argument(
        '--identify_duplicateSEQs_but_NO_filter',
        action='store_true',
        dest='dry_run',
        help="If set, only identify and report duplicates; do NOT move files."
    )
    args = parser.parse_args()
    main(args.pdb_directory, args.dry_run)
