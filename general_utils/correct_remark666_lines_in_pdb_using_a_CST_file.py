#!/usr/bin/env python3
"""
Author:      Seth M. Woodbury
Date:        2025-05-23

Script to correct REMARK 666 lines in a PDB file based on UPSTREAM_CST references in a CST file.

Usage:
  python correct_remark666_lines_in_pdb_using_a_CST_file.py \
    --pdb /path/to/input.pdb \
    --cst_file /path/to/constraints.cst \
    [--output_pdb /path/to/edited.pdb]

If --output_pdb is omitted, the input PDB is overwritten.
Verbosity is printed to stderr.
"""
import argparse
import re
import sys
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Correct REMARK 666 lines based on CST UPSTREAM_CST markers.")
    p.add_argument('--pdb',       required=True, help="Input PDB file to modify")
    p.add_argument('--cst_file',  required=True, help="CST file with UPSTREAM_CST directives")
    p.add_argument('--output_pdb',help="Path to write the modified PDB (in-place if omitted)")
    return p.parse_args()

# Regex to match REMARK 666 lines of interest
RE_MARK = re.compile(
    r'^(REMARK 666 MATCH TEMPLATE)\s+(\w)\s+(\w{3})\s+(\d+)'
    r'\s+MATCH MOTIF\s+(\w)\s+(\w{3})\s+(\d+)\s+(\d+)(.*)$'
)

def load_cst_upstreams(cst_path):
    """Parse CST file and return mapping: upstream_block -> [downstream_blocks,...]"""
    cst_lines = Path(cst_path).read_text().splitlines()
    # find block boundaries
    block_bounds = {}
    curr = None
    for idx, L in enumerate(cst_lines):
        m = re.match(r'^### --- BLOCK (\d+) START', L)
        if m:
            curr = int(m.group(1))
            block_bounds[curr] = [idx, None]
        elif curr is not None and L.startswith('VARIABLE_CST::END'):
            block_bounds[curr][1] = idx
            curr = None
    # collect template-provided UPSTREAM_CST
    ups = {}
    for blk, (s,e) in block_bounds.items():
        for L in cst_lines[s:e]:
            m = re.search(r'SECONDARY_MATCH:\s+UPSTREAM_CST\s+(\d+)', L)
            if m:
                parent = int(m.group(1))
                ups.setdefault(parent, []).append(blk)
    return ups

def main():
    args = parse_args()
    pdb_path = Path(args.pdb)
    out_path = Path(args.output_pdb) if args.output_pdb else pdb_path

    # 1) Parse CST to get upstream->downstreams
    ups = load_cst_upstreams(args.cst_file)
    if not ups:
        print("[INFO] No UPSTREAM_CST markers found in CST file", file=sys.stderr)
        return

    # 2) Read PDB and parse REMARK 666 entries
    lines = pdb_path.read_text().splitlines(keepends=True)
    remark_entries = {}  # block_idx -> (line_idx, groups)
    parsed = []
    for i, L in enumerate(lines):
        m = RE_MARK.match(L)
        if not m:
            continue
        # groups: 1=prefix,2=tmpl_chain,3=tmpl_code,4=tmpl_pos,5=motif_chain,6=motif_code,7=motif_pos,8=block_idx,9=rest
        blk = int(m.group(8))
        remark_entries[blk] = (i, m)
    
    # 3) For each upstream parent, update its downstream remark
    for parent, downs in ups.items():
        # need parent's motif info
        if parent not in remark_entries:
            print(f"[WARN] No REMARK for upstream block {parent}", file=sys.stderr)
            continue
        _, pm = remark_entries[parent]
        # pm.group(5,6,7) => motif_chain, motif_code, motif_pos
        new_chain, new_code, new_pos = pm.group(5), pm.group(6), pm.group(7)
        for d in downs:
            if d not in remark_entries:
                print(f"[WARN] No REMARK for downstream block {d}", file=sys.stderr)
                continue
            li, dm = remark_entries[d]
            # preserve original motif2 (downstream) values
            old_motif = dm.group(5,6,7)
            rest = dm.group(9)
            # reconstruct with proper spacing and keep the downstream block idx + rest
            # dm.group(1)  = 'REMARK 666 MATCH TEMPLATE'
            # new_chain    = parent motif_chain
            # new_code     = parent motif_code
            # new_pos      = parent motif_pos
            # dm.group(5)  = downstream motif_chain
            # dm.group(6)  = downstream motif_code
            # dm.group(7)  = downstream motif_pos
            # dm.group(8)  = downstream block idx
            # dm.group(9)  = everything after that (leading space, weight, trailing spaces)
            new_line = (
                f"{dm.group(1)} {new_chain} {new_code} {new_pos:>4} MATCH MOTIF "
                f"{dm.group(5)} {dm.group(6)} {dm.group(7):>4}  {dm.group(8)}{dm.group(9)}\n"
            )

            print(f"[INFO] Updating REMARK block {d}: '{lines[li].rstrip()}' → '{new_line.rstrip()}'", file=sys.stderr)
            lines[li] = new_line

    # 4) Write output
    out_path.write_text(''.join(lines))
    print(f"[INFO] Wrote corrected PDB to {out_path}", file=sys.stderr)

if __name__ == '__main__':
    main()
