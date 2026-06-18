#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main driver script to run the grafting and cleaning pipeline in three steps:

1) STEP1: Add/clean REMARK 666 lines
2) STEP2: Merge specified atoms between residues
3) STEP3: Clean and renumber PDB entries
4) Copy final cleaned PDB to a "_GRAFTED_unrelaxed.pdb" in the original motif directory
5) Remove intermediate PDB files, leaving only inputs and final output

Usage:
    python run_full_graft_pipeline.py \
      --ungrafted_dft_theozyme_pdb_ref <path/to/ref.pdb> \
      --pregrafted_motif_pdb_file <path/to/preGRAFTED_MOTIF.pdb> \
      --merge <spec1> --merge <spec2> [...]

Example:
    python run_full_graft_pipeline.py \
      --ungrafted_dft_theozyme_pdb_ref /home/woodbuse/projects/.../group1_DFT.pdb \
      --pregrafted_motif_pdb_file /home/woodbuse/.../group1_preGRAFTED_MOTIF.pdb \
      --merge 'A:92<-C:3:NE2,CE1,ND1,CD2,CG,CB' \
      --merge 'A:93<-D:4:OE2,OE1,CD,CG' \
      --merge 'A:96<-A:1:NE2,CE1,ND1,CD2,CG,CB'
"""
import argparse
import os
import subprocess
import shutil
import sys

def parse_args():
    p = argparse.ArgumentParser(
        description="Run the full graft+clean pipeline (STEP1 -> STEP2 -> STEP3)"
    )
    p.add_argument(
        '--ungrafted_dft_theozyme_pdb_ref', required=True,
        help="Reference DFT theozyme PDB file"
    )
    p.add_argument(
        '--pregrafted_motif_pdb_file', required=True,
        help="Input pregrafted motif PDB file"
    )
    p.add_argument(
        '--merge', required=True, action='append',
        help="Merge spec: TGT_CHAIN:TGT_RES<-SRC_CHAIN:SRC_RES:ATOM1,ATOM2,..."
    )
    return p.parse_args()


def main():
    args = parse_args()
    ref_pdb = args.ungrafted_dft_theozyme_pdb_ref
    motif_pdb = args.pregrafted_motif_pdb_file
    merges = args.merge

    base_dir = '/home/woodbuse/special_scripts/theozyme_and_ligand_handling'
    step1_script = os.path.join(base_dir, 'motif_graft_onto_DFTpdb__STEP1__remark666_pregrafted_pdb.py')
    step2_script = os.path.join(base_dir, 'motif_graft_onto_DFTpdb__STEP2__perform_grafting.py')
    step3_script = os.path.join(base_dir, 'motif_graft_onto_DFTpdb__STEP3__cleanPDBs.py')

    # Derive intermediate filenames
    stem1, ext1 = os.path.splitext(motif_pdb)
    step1_pdb = f"{stem1}__STEP1_CLEANED{ext1}"

    stem2 = os.path.splitext(step1_pdb)[0]
    step2_pdb = f"{stem2}__STEP2_GRAFTED{ext1}"

    stem3 = os.path.splitext(step2_pdb)[0]
    step3_pdb = f"{stem3}__CLEANEDpdb{ext1}"

    # STEP1: remark666 clean
    print(f"[INFO] STEP1: adding/cleaning REMARK 666 entries -> {step1_pdb}")
    cmd1 = [
        sys.executable, step1_script,
        '--ungrafted_dft_theozyme_pdb_ref', ref_pdb,
        '--pregrafted_motif_pdb_file', motif_pdb
    ]
    subprocess.run(cmd1, check=True)
    print("[INFO] STEP1 complete")

    # STEP2: atom merging
    print(f"[INFO] STEP2: merging residues -> {step2_pdb}")
    cmd2 = [
        sys.executable, step2_script,
        '--ungrafted_dft_theozyme_pdb_ref', ref_pdb,
        '--pregrafted_STEP1_motif_pdb_file', step1_pdb
    ]
    for m in merges:
        cmd2 += ['--merge', m]
    subprocess.run(cmd2, check=True)
    print("[INFO] STEP2 complete")

    # STEP3: clean and renumber
    print(f"[INFO] STEP3: final cleaning -> {step3_pdb}")
    cmd3 = [
        sys.executable, step3_script,
        '--input_dirty_PDB', step2_pdb
    ]
    subprocess.run(cmd3, check=True)
    print("[INFO] STEP3 complete")

    # Create final renamed copy
    out_dir = os.path.dirname(motif_pdb)
    ung_base = os.path.splitext(os.path.basename(ref_pdb))[0]
    final_name = f"{ung_base}_GRAFTED_unrelaxed{ext1}"
    final_pdb = os.path.join(out_dir, final_name)
    print(f"[INFO] Copying final cleaned PDB to {final_pdb}")
    shutil.copy(step3_pdb, final_pdb)

    # Remove intermediate files
    for f in (step1_pdb, step2_pdb, step3_pdb):
        if os.path.exists(f):
            os.remove(f)
            print(f"[INFO] Deleted intermediate file {f}")

    print(f"[SUCCESS] Pipeline complete. Final PDB: {final_pdb}")

if __name__ == '__main__':
    main()
