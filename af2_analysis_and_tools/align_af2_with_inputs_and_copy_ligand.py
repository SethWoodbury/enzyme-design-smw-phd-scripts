#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 20:11:21 2022

@author: indrek & Seth

Description:
    This script aligns an input PDB (or multiple) to a reference PDB (or multiple)
    and copies REMARK 666 lines, ligand info, etc.

NEW ARGUMENTS:
    --exact_single_pdb_path: If specified, only this single PDB file is used as input.
    --exact_ref_pdb_path: If specified, only this single PDB file is used as the reference.

Everything else retains the existing functionality, including:
    - Checking --pdb, --pdb_path, or --pdblist if exact_single_pdb_path is not set
    - Checking --ref, --ref_path if exact_ref_pdb_path is not set
    - Multi-process alignment
    - Marking catalytic residues, etc.
"""

import pyrosetta as pyr
import pyrosetta.rosetta
import pyrosetta.distributed.io
import glob
import os
import sys
import pandas as pd
import numpy as np
import queue
import threading
import multiprocessing
import argparse

# --- locate repo root + shared external paths ---
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "repo_paths.py").is_file():
        _sys.path.insert(0, str(_anc)); break
import repo_paths

if os.path.exists("/home/ikalvet"):
    HOME = ""
else:
    HOME = "/home/indrek/UW_Digs"
sys.path.append(repo_paths.ENZYME_DESIGN_UTILS)
import design_utils


def add_matcher_line_to_pose(pose, ref_pose, tgt_residues, ref_residues):
    """
    Takes REMARK 666 lines from ref pose and appends them to the new pose,
    skipping any existing REMARK 666 lines in the new pose.
    """
    if len(tgt_residues) == 0:
        return pose

    _str_ref = open(ref_pose.pdb_info().name(), "r").readlines()
    _ref_remarks = [l for l in _str_ref if "REMARK 666" in l]

    _str = pyrosetta.distributed.io.to_pdbstring(pose)
    pdbff = _str.split("\n")

    new_pdb = []
    # If the first line starts with "ATOM", there's no header to inject remarks after
    if "ATOM" in pdbff[0]:
        for lr in _ref_remarks:
            new_pdb.append(lr)
        for l in pdbff:
            new_pdb.append(l)
    else:
        for l in pdbff:
            if "HEADER" in l:
                new_pdb.append(l)
                for lr in _ref_remarks:
                    new_pdb.append(lr)
            elif "REMARK 666" in l:  # skip
                continue
            else:
                new_pdb.append(l)

    pose2 = pyrosetta.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose2, "\n".join(new_pdb))
    return pose2


parser = argparse.ArgumentParser()

# Existing arguments
parser.add_argument("--pdb", nargs="+", help="One or more input PDBs. If not given, we try other sources.")
parser.add_argument("--pdb_path", help="Path containing one or more PDBs.")
parser.add_argument("--pdblist", type=str, help="File containing list of PDB filenames.")
parser.add_argument("--params", nargs="+", default=[
    f"{HOME}/home/ikalvet/projects/Heme/theozyme/HMM/HMM_high.params",
    f"{HOME}/home/ikalvet/projects/Heme/theozyme/HPM/biphenyl/HPM.params",
    f"{HOME}/home/ikalvet/Rosetta/2019_06_14/main/database/chemical/residue_type_sets/fa_standard/residue_types/sidechain_conjugation/CYX.params"
], help="List of params files for extra residues.")

parser.add_argument("--align_start", type=int, help="Start position of the alignment region in the reference PDB")
parser.add_argument("--align_end", type=int, help="End position of the alignment region in the reference PDB")

parser.add_argument("--ref", nargs="+", type=str, help="Reference PDB(s).")
parser.add_argument("--ref_path", type=str, help="Reference PDB path.")

parser.add_argument("--outdir", type=str, default="./", help="Output directory.")
parser.add_argument("--clobber", action="store_true", default=False, help="Overwrite existing files?")
parser.add_argument("--fix_catres", action="store_true", default=False, help="Fix the rotamer of catalytic residue?")

# NEW arguments
parser.add_argument("--exact_single_pdb_path", type=str,
                    help="If provided, use exactly this single PDB file (ignores --pdb, --pdb_path, --pdblist).")
parser.add_argument("--exact_ref_pdb_path", type=str,
                    help="If provided, use exactly this single reference PDB (ignores --ref, --ref_path).")

args = parser.parse_args()

########################
### DETERMINE PDBS   ###
########################
if args.exact_single_pdb_path:
    # If user provided a single exact PDB, just use that
    pdbfiles = [args.exact_single_pdb_path]

elif args.pdb is not None:
    pdbfiles = args.pdb

elif args.pdblist is not None:
    # read lines from the file
    pdbfiles = open(args.pdblist, 'r').readlines()
    extension = ".pdb"
    # If the first line already ends with ".pdb", we assume it's direct
    if pdbfiles and pdbfiles[0].rstrip().endswith(".pdb"):
        extension = ""
    pdbfiles = [f.rstrip() + extension for f in pdbfiles]

elif args.pdb_path is not None:
    pdbfiles = glob.glob(os.path.join(args.pdb_path, "*.pdb"))

else:
    # fallback to all in current dir
    pdbfiles = glob.glob("*.pdb")


########################
### DETERMINE REFS   ###
########################
if args.exact_ref_pdb_path:
    # If user provided a single exact reference PDB
    args.ref = [args.exact_ref_pdb_path]

elif args.ref is None:
    # if not provided, we rely on ref_path
    assert args.ref_path is not None, "Either --ref or --ref_path or --exact_ref_pdb_path is required."
    args.ref = glob.glob(os.path.join(args.ref_path, "*.pdb"))


########################
### INIT PYROSETTA   ###
########################
extra_res = " ".join(args.params)
pyr.init(f"-extra_res_fa {extra_res} -mute all -run:preserve_header")

if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)

ref_poses = {}
ref_path = os.path.dirname(args.ref[0]) if args.ref else "."

ref_suffixes = ["_native", "_T", "_model_", "_packed_temp", "_enhMPNN"]

######################################################
### SHARED FUNCTION: ALIGN A SINGLE PDB WITH A REF ###
######################################################
def align_single_pose_with_ref(pdbfile, ref_list, ref_poses_dict, fix_catres=False, outdir="./", clobber=False):
    """
    Performs the alignment, remark insertion, catalytic residue fix, etc.
    on a SINGLE PDB (pdbfile) given a list of references (ref_list).
    Uses the suffix logic to find which reference to use.
    Stores loaded reference poses in ref_poses_dict to avoid re-loading.
    """
    import pyrosetta
    pose = pyr.pose_from_file(pdbfile)

    # The script tries to find which ref is correct by name
    ref_names = []
    for r in ref_list:
        if any([os.path.basename(r).replace(".pdb", suffix) in pdbfile for suffix in ref_suffixes]):
            ref_names.append(r)

    if len(ref_names) != 1:
        print(f"Can't find ref structure for {pdbfile}: {ref_names}")
        return

    ref_name = ref_names[0]

    # If we haven't loaded this reference pose before, do so now
    if ref_name not in ref_poses_dict:
        ref_poses_dict[ref_name] = pyr.pose_from_file(ref_name)

    overlay_pos = pyrosetta.rosetta.utility.vector1_unsigned_long()
    for n in range(1, pose.size() + 1):
        overlay_pos.append(n)

    matched_residues = design_utils.get_matcher_residues(ref_name)
    pose2 = pose.clone()

    # Superimpose on CA
    rmsd = pyrosetta.rosetta.protocols.toolbox.pose_manipulation.superimpose_pose_on_subset_CA(
        pose2, ref_poses_dict[ref_name], overlay_pos, 0
    )
    print(f"{pdbfile}: alignment RMSD = {rmsd:.3f}")

    # Append the reference pose (the last residue) to the pose2
    pyrosetta.rosetta.core.pose.append_subpose_to_pose(
        pose2, ref_poses_dict[ref_name], ref_poses_dict[ref_name].size(), ref_poses_dict[ref_name].size(), True
    )

    # Fix catalytic residues
    for catres_seqpos in matched_residues:
        catres_AA = ref_poses_dict[ref_name].residue(catres_seqpos).name()
        catres_AA3 = ref_poses_dict[ref_name].residue(catres_seqpos).name3()
        print(f"{pdbfile}: fixing {catres_AA3}{catres_seqpos} with reference {catres_AA}")

        mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
        mutres.set_res_name(catres_AA)  # e.g. fixes HIS_D, etc.
        mutres.set_target(catres_seqpos)
        mutres.apply(pose2)

        if fix_catres:
            # Fix catalytic residue rotamers
            for chi_idx in range(ref_poses_dict[ref_name].residue(catres_seqpos).nchi()):
                pose2.residue(catres_seqpos).set_chi(
                    chi_idx + 1,
                    ref_poses_dict[ref_name].residue(catres_seqpos).chi(chi_idx + 1)
                )

    pose2 = add_matcher_line_to_pose(pose2, ref_poses_dict[ref_name], matched_residues, matched_residues)

    # Add motif labels
    motif_label_sel = pyrosetta.rosetta.core.select.residue_selector.ResiduePDBInfoHasLabelSelector(label_str="motif")
    for rn in pyrosetta.rosetta.core.select.get_residue_set_from_subset(motif_label_sel.apply(ref_poses_dict[ref_name])):
        pose2.pdb_info().add_reslabel(rn, "motif")

    ligand_name = pose2.residue(pose2.size()).name3()
    save_name = os.path.join(
        outdir,
        os.path.basename(pdbfile).replace(".pdb", f"_{ligand_name}.pdb")
    )
    if os.path.exists(save_name) and not clobber:
        print(f"Warning! file exists! Use --clobber to overwrite it: {save_name}")
        return

    pose2.dump_pdb(save_name)


#########################
### POSSIBLE 1:1 MODE ###
#########################
single_mode = (len(pdbfiles) == 1 and len(args.ref) == 1)

if single_mode:
    # We skip multiprocessing entirely and just do the single alignment
    print("Detected a single PDB file and a single reference file. Skipping multiprocessing.")
    align_single_pose_with_ref(
        pdbfile=pdbfiles[0],
        ref_list=args.ref,
        ref_poses_dict=ref_poses,  # empty dict, will fill
        fix_catres=args.fix_catres,
        outdir=args.outdir,
        clobber=args.clobber
    )

else:
    #########################
    ### MULTIPROCESS MODE ###
    #########################
    print("Multiple input PDBs or multiple references detected. Using multiprocessing...")

    the_queue = multiprocessing.Queue()
    manager = multiprocessing.Manager()
    ref_poses = manager.dict()

    # Fill queue
    for i, pdbfile in enumerate(pdbfiles):
        the_queue.put((i, pdbfile))

    def process(q):
        while True:
            p = q.get(block=True)
            if p is None:
                return
            i, pdbfile = p
            align_single_pose_with_ref(
                pdbfile,
                args.ref,
                ref_poses,
                fix_catres=args.fix_catres,
                outdir=args.outdir,
                clobber=args.clobber
            )

    # Start processes
    pool = multiprocessing.Pool(os.cpu_count(), process, (the_queue,))

    # Insert sentinel values
    for _i in range(os.cpu_count()):
        the_queue.put(None)

    the_queue.close()
    the_queue.join_thread()
    pool.close()
    pool.join()
