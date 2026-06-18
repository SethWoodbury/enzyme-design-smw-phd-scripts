#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Indrek Kalvet
ikalvet@uw.edu
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import time
import queue
import threading
import multiprocessing
import argparse
from shutil import copy2
import pyrosetta as pyr
import pyrosetta.rosetta
import pyrosetta.distributed.io
from pyrosetta.rosetta.core.scoring import score_type_from_name
from pyrosetta.rosetta.core.import_pose import ImportPoseOptions, FileType
from pyrosetta import Pose
from pyrosetta.rosetta.core.import_pose import pose_from_pdbstring
import itertools
import json
import copy

# --- locate repo root + shared external paths ---
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "repo_paths.py").is_file():
        _sys.path.insert(0, str(_anc)); break
import repo_paths

sys.path.insert(0, repo_paths.RFDIFFUSION_AA)


aa3to1 = {
    "ALA":'A', "ARG":'R', "ASN":'N', "ASP":'D', "CYS":'C',
    "GLN":'Q', "GLU":'E', "GLY":'G', "HIS":'H', "ILE":'I',
    "LEU":'L', "LYS":'K', "MET":'M', "PHE":'F', "PRO":'P',
    "SER":'S', "THR":'T', "TRP":'W', "TYR":'Y', "VAL":'V' }

aa1to3 = {val: k for k, val in aa3to1.items()}

from pyrosetta import Pose
from pyrosetta.rosetta.core.import_pose import pose_from_pdbstring

def load_plain_pdb(fn, rescode):
    """
    Load only ATOM/HETATM for rescode from fn, 
    but *temporarily* swap rescode->'LIG' so Rosetta never splits it.
    """
    newcode = "LIG"      # a code Rosetta doesn't treat as a sugar
    lines = open(fn).read().splitlines()
    filtered = []
    for l in lines:
        if l.startswith("ATOM"):
            filtered.append(l + "\n")
        elif l.startswith("HETATM") and l[17:20].strip() == rescode:
            # build a line with 'LIG' in cols 17–20
            filtered.append(l[:17] + f"{newcode:3}" + l[20:] + "\n")
    p = Pose()
    pose_from_pdbstring(p, "".join(filtered))

    # optional: rename back from LIG->rescode in the Pose
    for r in range(1, p.size()+1):
        if p.residue(r).name3() == newcode:
            p.conformation().set_icode(r, rescode)  # or use MutateResidue

    return p


def read_pose_from_str_and_fix_issues(pdbfile, trb):
    """
    align_atoms = [n1, n2, n3]
    """
    # _str = pyrosetta.distributed.io.to_pdbstring(pose2)
    if os.path.exists(pdbfile):
        pdbff = open(pdbfile, "r").readlines()
    else:
        pdbff = pdbfile.split("\n")  # In case contents of a PDB file are provided

    # Replacing all hallucinated longer residues with GLY
    new_pdb = []
    for i, l in enumerate(pdbff):
        if "ATOM" in l:
            if int(l.split()[5])-1 not in trb["con_hal_idx0"]:
                if l.split()[3] in ["VAL", "GLN", "ARG", "LYS", "GLU", "ASN", "ASP", "MET", "PRO"]:
                    new_pdb.append(l[:17]+"GLY"+l[20:])
                else:
                    new_pdb.append(l)
            else:
                new_pdb.append(l)
        elif "HETATM" in l:
            # Fixing the column formatting of ligand atom names
            atom_name = l[11:16].strip()
            ltrs = "".join([x for x in atom_name if not x.isnumeric()])
            nmbrs = "".join([x for x in atom_name if x.isnumeric()])
            atom_name_fix = f"{ltrs:>3}{nmbrs:<2}"
            new_pdb.append(l[:11]+atom_name_fix+l[16:])
        else:
            new_pdb.append(l)

    pose3 = pyrosetta.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose3, "\n".join(new_pdb))
    return pose3


def add_matcher_line_to_pose(pose, resno, ligand_name):
    _str = pyrosetta.distributed.io.to_pdbstring(pose)
    pdbff = _str.split("\n")

    new_pdb = []
    if "ATOM" in pdbff[0]:
        new_pdb.append(f"REMARK 666 MATCH TEMPLATE X {ligand_name}    0 MATCH MOTIF A {pose.residue(resno).name3()}  {resno}  1  1               \n")
        for l in pdbff:
            new_pdb.append(l)
    else:
        for l in pdbff:
            new_pdb.append(l)
            if "HEADER" in l:
                new_pdb.append(f"REMARK 666 MATCH TEMPLATE X {ligand_name}    0 MATCH MOTIF A {pose.residue(resno).name3()}  {resno}  1  1               \n")

    pose3 = pyrosetta.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose3, "\n".join(new_pdb))
    return pose3


def add_matcher_line_to_pose(pose, ref_pose, tgt_residues, ref_residues):
    """
    Takes REMARK 666 lines from ref pose and adjusts them based on the new positions in tgt_residues
    """
    if len(tgt_residues) == 0:
        return pose

    ligand_name = pose.residue(pose.size()).name3()
    # _str_ref = pyrosetta.distributed.io.to_pdbstring(ref_pose).split("\n")
    # _ref_remarks = [l for l in _str_ref if "REMARK 666" in l]

    _new_remarks = []

    for i, r in enumerate(tgt_residues):
        _new_remarks.append(f"REMARK 666 MATCH TEMPLATE {tgt_residues[r]['target_chain']} {tgt_residues[r]['target_name']}"
                            f"  {tgt_residues[r]['target_resno']:>3} MATCH MOTIF {tgt_residues[r]['chain']} "
                            f"{tgt_residues[r]['name3']}  {r:>3}  {tgt_residues[r]['cst_no']}  "
                            f"{tgt_residues[r]['cst_no_var']}               \n")

    _str = pyrosetta.distributed.io.to_pdbstring(pose)
    pdbff = _str.split("\n")

    new_pdb = []
    if "ATOM" in pdbff[0]:
        for lr in _new_remarks:
            new_pdb.append(lr)
        for l in pdbff:
            new_pdb.append(l)
    else:
        for l in pdbff:
            if "HEADER" in l:
                new_pdb.append(l)
                for lr in _new_remarks:
                    new_pdb.append(lr)
            elif "REMARK 666" in l:  # Skipping existing REMARK 666 lines
                continue
            else:
                new_pdb.append(l)
    pose2 = pyrosetta.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose2, "\n".join(new_pdb))
    return pose2


def get_matcher_residues(filename):
    pdbfile = open(filename, 'r').readlines()

    matches = {}
    for line in pdbfile:
        if "ATOM" in line:
            break
        if "REMARK 666" in line:
            lspl = line.split()
            resno = int(lspl[11])

            matches[resno] = {'target_name': lspl[5],
                              'target_chain': lspl[4],
                              'target_resno': int(lspl[6]),
                              'chain': lspl[9],
                              'name3': lspl[10],
                              'cst_no': int(lspl[12]),
                              'cst_no_var': int(lspl[13])}
    return matches


def getSASA(pose, resno=None, SASA_atoms=None, ignore_sc=False):
    """
    Takes in a pose and calculates its SASA.
    Or calculates SASA of a given residue.
    Or calculates SASA of specified atoms in a given residue.

    Procedure by Brian Coventry
    """

    atoms = pyr.rosetta.core.id.AtomID_Map_bool_t()
    atoms.resize(pose.size())

    n_ligands = 0
    for res in pose.residues:
        if res.is_ligand():
            n_ligands += 1

    for i, res in enumerate(pose.residues):
        if res.is_ligand():
            atoms.resize(i+1, res.natoms(), True)
        else:
            atoms.resize(i+1, res.natoms(), not(ignore_sc))
            if ignore_sc is True:
                for n in range(1, res.natoms()+1):
                    if res.atom_is_backbone(n) and not res.atom_is_hydrogen(n):
                        atoms[i+1][n] = True

    surf_vol = pyr.rosetta.core.scoring.packing.get_surf_vol(pose, atoms, 1.4)

    if resno is not None:
        if isinstance(resno, int):
            res_surf = 0.0
            for i in range(1, pose.residue(resno).natoms()+1):
                if SASA_atoms is not None and i not in SASA_atoms:
                    continue
                res_surf += surf_vol.surf(resno, i)
            return res_surf
        elif isinstance(resno, list):
            res_surf = 0.0
            for rn in resno:
                for i in range(1, pose.residue(rn).natoms()+1):
                    if SASA_atoms is not None and i not in SASA_atoms:
                        continue
                    res_surf += surf_vol.surf(rn, i)
            return res_surf
    else:
        return surf_vol


def get_ROG(pose):
    centroid = np.array([np.average([res.xyz("CA").__getattribute__(c) for res in pose.residues if res.is_protein()]) for c in "xyz"])
    ROG = max([np.linalg.norm(centroid - res.xyz("CA")) for res in pose.residues if res.is_protein()])
    return ROG


def sidechain_connectivity(res):
    """
    Evaluates the physical correctness of the sidechain of a residue
    """
    ref_res_pose = pyr.pose_from_sequence("A"+res.name1()+"A")
    ref_res = ref_res_pose.residue(2)
    bondlen_deviations = []
    for an in range(1, res.natoms()+1):
        if res.atom_type(an).element() == "H":
            continue
        for nn in res.bonded_neighbor(an):
            if res.atom_type(nn).element() == "H":
                continue
            bondlen_deviations.append(abs((res.xyz(an)-res.xyz(nn)).norm() - (ref_res.xyz(an)-ref_res.xyz(nn)).norm()))
    return max(bondlen_deviations)


def thread_seq_to_pose(pose, sequence, skip_resnos=None):
    if skip_resnos is None:
        skip_resnos = []
    pose2 = pose.clone()
    for i, r in enumerate(sequence):
        if i+1 in skip_resnos:
            continue
        if pose.residue(i+1).is_ligand():
            continue
        mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
        mutres.set_target(i+1)
        mutres.set_res_name(aa1to3[r])
        mutres.apply(pose2)
    return pose2


def load_multimodel_PDB_to_poses(pdbfile, trb, count=None, num_random=None):
    pdbstr = open(pdbfile, "r").read()
    models = pdbstr.split("ENDMDL")
    poses = []

    if count == num_random:
        traj_ids_to_save = list(range(0, count+1))
    else:
        traj_ids_to_save = []
        # Figuring out <traj_N_save> random unique step id's
        while len(traj_ids_to_save) < num_random:
            random_id = np.random.randint(1, count)
            if random_id not in traj_ids_to_save:
                traj_ids_to_save.append(random_id)

    for i, mdl in enumerate(models):
        if len(mdl) < 10:
            continue
        if i > count:
            break
        if i not in traj_ids_to_save:
            continue
        poses.append(read_pose_from_str_and_fix_issues(mdl, trb))
    return poses


def get_rosetta_scores(pose, sfx, sfx_cart, catres_list):
    """
    Getting sidechain quality scores for a given pose.
    Inspired by Florence Hardy and Aiko Muraishi
    """
    sfx(pose)
    sfx_cart(pose)
    
    scoredict = {}
    for k in ["cart_bonded", "fa_dun"]:
        scoredict[k] = {
        res.seqpos(): pose.energies().residue_total_energies(res.seqpos()).get(score_type_from_name(k))
        for res in pose.residues if res.seqpos() in catres_list}

    averages = {k: np.average(list(scores.values())) for k,scores in scoredict.items()}

    return averages, scoredict


def dump_scorefile(df, filename):
    widths = {}
    for k in df.keys():
        if k in ["SCORE:", "description", "name"]:
            widths[k] = 0
        if len(k) >= 12:
            widths[k] = len(k) + 1
        else:
            widths[k] = 12

    with open(filename, "w") as file:
        title = ""
        for k in df.keys():
            if k == "SCORE:":
                title += k
            elif k in ["description", "name"]:
                title += f" {k}"
            else:
                title += f"{k:>{widths[k]}}"
        if all([t not in df.keys() for t in ["description", "name"]]):
            title += f" {'description'}"
        file.write(title + "\n")
        
        for index, row in df.iterrows():
            line = ""
            for k in df.keys():
                if isinstance(row[k], (float, np.float16)):
                    val = f"{row[k]:.3f}"
                else:
                    val = row[k]
                if k == "SCORE:":
                    line += val
                elif k in ["description", "name"]:
                    line += f" {val}"
                else:
                    line += f"{val:>{widths[k]}}"
            if all([t not in df.keys() for t in ["description", "name"]]):
                line += f" {index}"
            file.write(line + "\n")

# if __name__ == "__main__":
def main(args):
    assert any([x is not None for x in [args.pdb, args.pdbpath]]), "Need to provide either --pdb or --pdbpath"

    if args.pdb is not None:
        pdbfiles = args.pdb
    elif args.pdbpath is not None:
        pdbfiles = []
        for pth in args.pdbpath:
            pdbfiles += glob.glob(pth+"/*.pdb")
    params = args.params
    NPROC = args.nproc
    # SASA_limit = args.limit

    ref_pdbs = None
    if args.ref is not None:
        ref_pdbs = args.ref
    elif args.ref_path is not None:
        ref_pdbs = sorted(glob.glob(args.ref_path+"/*.pdb"))


    if args.traj is not None:
        if "/" in args.traj:
            traj_N_save = int(args.traj.split("/")[0])
            traj_N_steps = int(args.traj.split("/")[1])
            assert traj_N_save <= traj_N_steps
        else:
            traj_N_save = int(args.traj)
            traj_N_steps = traj_N_save


    ref_catres = []
    if args.ref_catres is not None:
        for r in args.ref_catres:
            if "-" in r:
                _start_pos, _end_pos = r.split("-")[0], r.split("-")[1]
                _ch = r[0]
                for n in range(int(_start_pos[1:]), int(_end_pos)+1):
                    ref_catres.append(f"{_ch}{n}")
            else:
                ref_catres.append(r)

    if args.ligand_exposed_atoms is not None and args.exposed_atom_SASA is None:
        sys.exit("Defined --ligand_exposed_atoms but not --exposed_atom_SASA")
    if args.exposed_atom_SASA is not None and args.ligand_exposed_atoms is None:
        sys.exit("Defined --exposed_atom_SASA but not --ligand_exposed_atoms")

    filtered_dir = args.outdir
    try:
        if not os.path.exists(filtered_dir):
            os.mkdir(filtered_dir)
    except PermissionError:
        pass


    ### Getting PyRosetta started
    extra_res_fa = ""
    if len(params) > 0:
        extra_res_fa = "-extra_res_fa"
        for p in params:
            extra_res_fa += " " + p
        #extra_res_fa = extra_res_fa + " " + "-load_PDB_components false"
    print("### EXTRA RES ###")
    print(extra_res_fa)

    DAB = repo_paths.DALPHABALL
    if not os.path.exists(DAB):
        DAB = None

    assert DAB is not None, "Please compile DAlphaBall.gcc and manually provide a path to it in this script under the variable `DAB`\n"\
                            "For more info on DAlphaBall, visit: https://www.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Filters/HolesFilter"

    #pyr.init(f"{extra_res_fa} -mute all -dalphaball {DAB} -run:preserve_header -in:fast_restyping true")
    pyr.init(f"{extra_res_fa} -dalphaball {DAB} -mute all -run:preserve_header -in:fast_restyping false")

    # Setting ImportPoseOptions for loading reference poses properly
    opts = pyrosetta.rosetta.core.import_pose.ImportPoseOptions()
    opts.set_fast_restyping(False)

    sfx = pyr.get_fa_scorefxn()
    sfx_cart = sfx.clone()
    sfx_cart.set_weight(score_type_from_name("cart_bonded"), 0.5)
    sfx_cart.set_weight(score_type_from_name("pro_close"), 0.0)


    ### Starting processing
    start = time.time()

    ligand_found = False
    for p in params:
        par = open(p, 'r').readlines()
        for l in par:
            if "TYPE LIGAND" in l:
                ligand_pdb = p.replace(".params", ".pdb")
                print("")
                print("### LOOKING FOR LIGAND PDB ###")
                print(ligand_pdb)
                print("")
                print("### LOADING LIGAND POSE ###")
                #ligand = load_plain_pdb(ligand_pdb, "LAT")
                ligand = pyr.pose_from_file(ligand_pdb)
                print("### SUCCESSFUL POSE FROM FILE ###")
                print("Residues in the loaded ligand pose:")
                print(ligand)
                #for i in range(1, ligand.size()+1):
                 #   print("-- RUN --")
                 #   print(f"  {i:>2}  {ligand.residue(i).name3()}")
                print("")
                print("LOOKING FOR LIGAND RESIDUE")
                #res = ligand.residue(1)
                #print("Is Rosetta treating it as a sugar? ", ligand.is_carbohydrate())
                ligand_SASA = getSASA(ligand).tot_surf
                ligand_found = True
                break
        if ligand_found:
            break

    if not ligand_found:
        sys.exit("Was not able to find a ligand from the params files specified.")

    the_queue = multiprocessing.Queue()  # Queue stores the iterables

    manager = multiprocessing.Manager() 
    ref_poses = manager.dict()  # Need a special dictionary to store outputs from multiple processes
    scores = manager.dict()
    dssps = manager.dict()




    print(len(pdbfiles), "designs to analyze.")

    print("Building multiprocessing queue.")
    count = 0
    for i, pdbfile in enumerate(pdbfiles):
        scores[count] = manager.dict()
        the_queue.put((count, pdbfile))
        count += 1

    # reserving additional entries in the scores dictionary for the trajectory models
    if args.traj is not None:
        for i, pdbfile in enumerate(pdbfiles):
            traj_file = os.path.dirname(os.path.realpath(pdbfile)) + "/traj/" + os.path.basename(pdbfile).replace(".pdb", "_pX0_traj.pdb")
            if os.path.exists(traj_file):
                for n in range(1, traj_N_save):
                    scores[i+len(pdbfiles)*n] = manager.dict()
                    # scores[len(scores)+1] = manager.dict()
            else:
                print(f"No trajectory file found for {pdbfile}, {traj_file}")

    datacolumns = manager.list()
    datacolumns += ["chainbreak", "rCA_nonadj", "lig_dist", "bondlen_dev"]
    if args.cart_bonded is not None:
            datacolumns.append("cart_bonded_avg")
    if args.fa_dun is not None:
            datacolumns.append("fa_dun_avg")
    datacolumns += ["loop_frac", "longest_helix", "rog"]
    if args.loop_catres is True:
            datacolums.append("loop_at_motif")
    datacolumns += ["term_mindist", "SASA", "SASA_rel"]
    if args.ligand_exposed_atoms is not None:
        datacolumns.append("SASA_exposed_atoms")



    def process(q, ref_pdbs, ref_poses):
        while True:
            p = q.get(block=True)
            if p is None:
                return
            i = p[0]
            pdbfile = p[1]
            pdbfile_orig = pdbfile
            scores[i]["description"] = pdbfile
            
            if not args.analyze:
                scores[i]["passed"] = False

            if args.trb is None:
                trbfile = pdbfile.replace(".pdb", ".trb")
            else:
                __trbfs = [x for x in args.trb if os.path.basename(pdbfile).replace(".pdb", "") in x]
                assert len(__trbfs) == 1, f"Bad number of trbs for {pdbfile}: {__trbfs}"
                trbfile = __trbfs[0]

            ### Loading trb file and figuring out fixed positions between ref and hal
            try:
                trb = np.load(trbfile, allow_pickle=True)
            except FileNotFoundError:
                print(trbfile, "not found!!!!!!!!!!!")
                continue

            if ref_pdbs is not None:
                __refs = [r for r in ref_pdbs if os.path.basename(r).replace(".pdb", "_") in os.path.basename(pdbfile)]
                assert len(__refs) == 1, f"Bad number of reference PDBS found for {pdbfile}: {__refs}"
                ref_pdb = __refs[0]
            else:
                ref_pdb = trb["config"]["inference"]["input_pdb"]

            if ref_pdb not in ref_poses.keys():
                # Loading reference pose with proper restyping
                ref_poses[ref_pdb] = pyr.pose_from_file(filename=ref_pdb, options=opts, read_fold_tree=True, type=pyrosetta.rosetta.core.import_pose.FileType(1))
                # TODO: clean up pose from this dict once it's no longer needed?
            ref_pose = ref_poses[ref_pdb].clone()

            matched_residues = get_matcher_residues(ref_pdb)  # REMARK 666 contents of reference

            # numbering_offset = ref_pose.pdb_info().number(1) -1

            if args.partial is True:
                fixed_pos_in_hal0 = trb["con_hal_idx0"]
                fixed_pos_in_hal = [x+1 for x in fixed_pos_in_hal0]
                fixed_pos_in_ref = trb["con_ref_pdb_idx"]
                _ref_catres = [f"{x[0]}{x[1]}" for x in fixed_pos_in_ref]
            else:
                fixed_pos_in_hal0 = trb["con_hal_idx0"]
                fixed_pos_in_hal = [x+1 for x in fixed_pos_in_hal0]
                fixed_pos_in_ref = trb["con_ref_pdb_idx"]
                _ref_catres = ref_catres
                
                if args.ref_catres is None and len(matched_residues) > 0:
                    ## Trying to parse catalytic residues from the reference PDB file
                    _ref_catres = [f"{d['chain']}{rn}" for rn,d in matched_residues.items() if ref_pose.residue(rn).is_protein()]


            if args.fix is True:
                # This fixes ligand atom name issues and stuff like that
                # Only for very old AA-diffusion outputs
                # pose = read_pose_from_str_and_fix_ligand(pdbfile, ref_pdb, args.align_atoms)
                pose = read_pose_from_str_and_fix_issues(pdbfile, trb)
            else:
                pose = pyr.pose_from_file(pdbfile)

            poses_to_parse = [pose]

            if args.traj is not None:
                if "atomize_indices2atomname" not in trb.keys() or len(trb["atomize_indices2atomname"]) == 0:
                    traj_file = os.path.dirname(os.path.realpath(pdbfile)) + "/traj/" + os.path.basename(pdbfile).replace(".pdb", "_pX0_traj.pdb")
                    if os.path.exists(traj_file):
                        poses_to_parse += load_multimodel_PDB_to_poses(traj_file, trb, count=traj_N_steps, num_random=traj_N_save)[1:traj_N_save+1]
                else:
                    print("Trajectory is useless in case of atomized motifs.")

            for _j, pose in enumerate(poses_to_parse):
                if _j != 0:
                    # Setting correct output PDB name and scorefile line for each trajectory pose
                    pdbfile = pdbfile_orig.replace(".pdb", f"_traj{_j}.pdb")
                    idx = p[0] + len(pdbfiles)*_j
                    if "description" in scores[idx].keys():
                        print(f"{pdbfile} {idx} = {i} {_j}: Can't figure out where to store scores !!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    i = idx
                    scores[i]["description"] = pdbfile

                if scores[i]["description"] != pdbfile:
                    print(pdbfile, "Filling out wrong line in scores !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    sys.exit(1)

                pose2 = pose.clone()

                ## A legacy function for cases when partial diffusion generated backbone doesn't contain a ligand, but reference structure does
                ## This was from the era when allatom partial diffusion did not exist
                if args.partial is True and any([res.is_ligand() for res in ref_pose.residues]) and not any([res.is_ligand() for res in pose2.residues]):
                    # Add ligand to pose if it exists in the reference structure
                    # But not in the diffused structure
                    align_map = pyrosetta.rosetta.std.map_core_id_AtomID_core_id_AtomID()
                    aln_atoms = ['N', 'CA', 'C', 'O']
                    for template_i, target_i in zip(trb["con_ref_idx0"], trb["con_hal_idx0"]):
                        res_template_i = ref_pose.residue(template_i+1)
                        res_target_i = pose2.residue(target_i+1)
                        for n in aln_atoms:
                            template_atom_idx = res_template_i.atom_index(n)
                            atom_id_template = pyrosetta.rosetta.core.id.AtomID(template_atom_idx, template_i+1)
                            target_atom_idx = res_target_i.atom_index(n)
                            atom_id_target = pyrosetta.rosetta.core.id.AtomID(target_atom_idx, target_i+1)
                            align_map[atom_id_target] = atom_id_template

                    rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(pose2, ref_pose, align_map)
                    print(f"{pdbfile}: alignment RMSD = {rmsd:.3f}")

                    ### Adding ligand to pose ###
                    ligands = [res for res in ref_pose.residues if res.is_ligand()]
                    for lig in ligands:
                        pyrosetta.rosetta.core.pose.append_subpose_to_pose(pose2, ref_pose, lig.seqpos(), lig.seqpos(), True)


                """
                First some scaffold quality analysis
                """
                ### Checking for chainbreaks
                dists = []
                for n in range(1, pose.size()):
                    if pose.residue(n).is_ligand():
                        continue
                    if pose.residue(n+1).is_ligand():
                        continue
                    if pose.chain(n) != pose.chain(n+1):
                        continue
                    dists.append((pose.residue(n).xyz("CA") - pose.residue(n+1).xyz("CA")).norm())
    
                scores[i]["chainbreak"] = max(dists)
                if args.analyze is False and max(dists) > 4.5:
                    print(f"{pdbfile}: chainbreak found! {max(dists):.2f}")
                    continue

                ### Checking if there are non-adjacent CA-CA contacts that are too short
                ### It seems there are veeeery few cases in nature where non-adj CA atoms are closer than 3.6A from each other
                nonadjacentCAs = []
                for (r1, r2) in itertools.combinations(pose.residues, 2):
                    if r1.is_ligand() or r2.is_ligand():
                        continue
                    if r1.is_virtual_residue() or r2.is_virtual_residue():
                        continue
                    if not r1.is_protein() or not r2.is_protein():
                        continue
                    if abs(r1.seqpos() - r2.seqpos()) == 1:
                        continue
                    nonadjacentCAs.append((r1.xyz("CA") - r2.xyz("CA")).norm())
    
                scores[i]["rCA_nonadj"] = min(nonadjacentCAs)
                if args.analyze is False and min(nonadjacentCAs) < 3.0:
                    print(f"{pdbfile}: some residues are too close to each other: {min(nonadjacentCAs):.2f}")
                    continue


                ### Checking if there are clashes with any ligands
                ligands = [res for res in pose2.residues if res.is_ligand()]
                if len(ligands) > 0:
                    lig_dists = []
                    for lig in ligands:
                        ligand_HAs = [n for n in range(1, lig.natoms()+1) if not lig.atom_is_hydrogen(n)]
                        for res in pose2.residues:
                            # excluding motif residues
                            # if res.seqpos() in fixed_pos_in_hal:
                            #     continue
                            if (res.nbr_atom_xyz() - lig.nbr_atom_xyz()).norm() > 15.0:
                                continue
                            if res.is_ligand():
                                continue
                            for lha in ligand_HAs:
                                if args.exclude_clash_atoms is not None:
                                    if lig.atom_name(lha).strip() in args.exclude_clash_atoms:
                                        continue
                                for n in range(1, 5):
                                    lig_dists.append((res.xyz(n) - lig.xyz(lha)).norm())
                    if len(lig_dists) == 0:
                        lig_dists = [9.9]
                        print(f"!!!!!!!!!!  {pdbfile} no clashcheck-valid residues around ligand found !!!!!!!!!!!")
                    scores[i]["lig_dist"] = min(lig_dists)
                    if args.analyze is False and min(lig_dists) < args.lig_dist:
                        print(f"{pdbfile}: ligand is too close to the backbone {min(lig_dists):.2f}")
                        continue

                ## If atomized residues were used (i.e. tip-atom diffusion), then check if the sidechain is physically real or has too long bonds
                ## Currently doing it for all atomized residues, and not just catalytic ones
                if "atomize_indices2atomname" in trb.keys() and len(trb["atomize_indices2atomname"]) > 0:
                    motif_res_bond_deviations = []
                    for resno in trb["atomize_indices2atomname"].keys():
                        motif_res_bond_deviations.append(sidechain_connectivity(pose2.residue(resno+1)))

                    scores[i]["bondlen_dev"] = max(motif_res_bond_deviations)
                    if args.analyze is False and scores[i]["bondlen_dev"] > args.bondlen_dev:
                        print(f"{pdbfile}: some motif residue geometry is too distorted based on 'bondlen_dev': {scores[i]['bondlen_dev']:.2f}")
                        continue


                    ## Performing sidechain quality analysis using Rosetta cart_bonded scoreterm
                    if args.cart_bonded is not None or args.fa_dun is not None:
                        averages, scoredict = get_rosetta_scores(pose, sfx, sfx_cart, [r+1 for r in trb["atomize_indices2atomname"].keys()])
                        for k in averages:
                            scores[i][f"{k}_avg"] = averages[k]
                        
                        if args.analyze is False and args.cart_bonded is not None and scores[i]["cart_bonded_avg"] > args.cart_bonded:
                            print(f"{pdbfile}: some motif residue geometry is too distorted based on 'cart_bonded': {scores[i]['cart_bonded_avg']:.2f}")
                            continue
                        if args.analyze is False and args.fa_dun is not None and scores[i]["fa_dun_avg"] > args.fa_dun:
                            print(f"{pdbfile}: some motif residue rotamer is suboptimal based on 'fa_dun': {scores[i]['fa_dun_avg']:.2f}")
                            continue


                """
                Then let's do some more subjective scaffold quality analysis:
                1) loop fraction
                2) longest helix
                3) radius of gyration
                4) whether motif residues are on loops
                5) how far are the termini from the ligands
                """
                ### Finding how much loop content the structure has
                dssp = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose2)
                secstruct = dssp.get_dssp_secstruct()
                loop_frac = secstruct.count("L") / pose2.size()
                scores[i]["loop_frac"] = loop_frac

                dssps[pdbfile] = secstruct

                if args.analyze is False and loop_frac > args.loop_limit:
                    if loop_frac > 0.9 and _j > 0:
                        print(f"{pdbfile}: something wrong with trajectory loopyness? loop_frac = {loop_frac:.3f}")
                    else:
                        print(f"{pdbfile}: protein too loopy: {loop_frac:.3f}")
                        continue


                ###  Analyzing how long is the longest helix
                if "H" in secstruct:
                    longest_helix = max([len(x.replace("E", "")) for x in secstruct.split("L") if "H" in x])
                else:
                    longest_helix = 0
                scores[i]["longest_helix"] = longest_helix

                if args.analyze is False and longest_helix > args.longest_helix:
                    print(f"{pdbfile}: longest helix too long: {longest_helix}")
                    continue


                ### Calculating radius of gyration
                scores[i]["rog"] = get_ROG(pose)
                if args.analyze is False and scores[i]["rog"] > args.rog:
                    print(f"{pdbfile}: radius of gyration too high: {scores[i]['rog']:.1f}")
                    continue


                ### Finding out if catalytic residues are between loops
                if args.loop_catres is True:
                    loops_next_to_catres = False
                    for r, _or in zip(fixed_pos_in_hal0, fixed_pos_in_ref):
                        # Only calculating it for true catalytic residues that the user provides
                        # Not calculated for partial diffusion outputs
                        if f"{_or[0]}{_or[1]}" not in ref_catres:
                            continue
                        if secstruct[r-2:r] == "LL" and secstruct[r+1:r+3] == "LL":
                            loops_next_to_catres = True
                            break
                    scores[i]["loop_at_motif"] = int(loops_next_to_catres)
        
                    if args.analyze is False and loops_next_to_catres is True:
                        print(f"{pdbfile}: catalytic residue between loops")
                        continue


                ### Checking how far C and N termini are from the ligands, based on CA distance to any ligand heavyatom
                if len(ligands) != 0:

                    ligands = [res for res in pose2.residues if res.is_ligand()]
                    term_mindists = []
                    for lig in ligands:
                        _lig_HAs = [n+1 for n in range(lig.natoms()) if lig.atom_type(n+1).element() != "H"]
                        d_Nt_lig = min( [(pose2.residue(1).xyz("CA") - lig.xyz(a)).norm() for a in _lig_HAs] )
                        d_Ct_lig = min( [(pose2.residue(pose2.size()-len(ligands)).xyz("CA") - lig.xyz(a)).norm() for a in _lig_HAs] )

                        term_mindist = min([d_Nt_lig, d_Ct_lig])
                        term_mindists.append(term_mindist)
                    scores[i]["term_mindist"] = min(term_mindists)
    
                    if args.analyze is False and scores[i]["term_mindist"] < args.term_limit:
                        print(f"{pdbfile}: terminus too close to ligand: {scores[i]['term_mindist']:.2f}")
                        continue
                else:
                    if "term_mindist" in datacolumns:
                        datacolumns.remove("term_mindist")


                #############################################################
                ### DOING ADJUSTMENTS ON STRUCTURES THAT PASS ALL FILTERS ###
                ############################################################# 


                ref_catres_nos = []
                hal_catres_nos = []
                for r in _ref_catres:
                    _ch = r[0]
                    _ref_resno = int(r[1:])
                    assert (_ch, _ref_resno) in fixed_pos_in_ref, f"Can't find residue {r} in trb con_ref_pdb_idx: {fixed_pos_in_ref}"
                    # re-calculating numbering offset because gaps in reference structure might throw it off otherwise
                    ref_resno_in_pose = None
                    for res in ref_pose.residues:
                        if ref_pose.pdb_info().chain(res.seqpos())+str(ref_pose.pdb_info().number(res.seqpos())) == r:
                            ref_resno_in_pose = res.seqpos()
                            break
                    if ref_resno_in_pose is None:
                        print(f"Could not find what is the ref_pose residue number of reference residue {r}")
                        sys.exit(1)
                    ref_catres_nos.append(ref_resno_in_pose)
                    hal_catres_nos.append(fixed_pos_in_hal[fixed_pos_in_ref.index((_ch, _ref_resno))])

                for j, ref_catres_no in enumerate(ref_catres_nos):
                    catres_seqpos = hal_catres_nos[j]
                    catres_AA = ref_pose.residue(ref_catres_no).name().split(":")[0]
                    catres_AA3 = ref_pose.residue(ref_catres_no).name3()
                    if "ProteinFull" in pose2.residue(catres_seqpos).name():
                        catres_AA = catres_AA+":"+pose2.residue(catres_seqpos).name().split(":")[1]

                    # Fixing catalytic residue identity to be the same as in the reference
                    print(f"{pdbfile}: fixing {pose2.residue(catres_seqpos).name()}-{catres_seqpos} with reference {catres_AA}-{ref_catres_no}")
                    mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
                    mutres.set_res_name(catres_AA)  # fixes HIS_D as well
                    mutres.set_target(catres_seqpos)
                    mutres.apply(pose2)

                    # Fixing catalytic residue rotamer
                    # Skipping residues that were atomized (tip-atom diffusion)
                    if "atomize_indices2atomname" not in trb.keys() or ("atomize_indices2atomname" in trb.keys() and catres_seqpos-1 not in trb["atomize_indices2atomname"].keys()):
                        for n in range(ref_pose.residue(ref_catres_no).nchi()):
                            pose2.residue(catres_seqpos).set_chi(n+1, ref_pose.residue(ref_catres_no).chi(n+1))

                if args.rethread is True:
                    pose2 = thread_seq_to_pose(pose2, pose2.sequence(), skip_resnos=hal_catres_nos)

                if len(ligands) != 0:
                    free_ligands = {}
                    for lig in ligands:
                        _tmp_pose = pyrosetta.rosetta.core.pose.Pose()
                        _tmp_pose.append_residue_by_jump(lig, 0)
                        free_ligands[lig.name3()] = _tmp_pose.clone()

                    free_ligand_SASA = sum([getSASA(p, resno=1) for p in free_ligands.values()])  # it's a bit wasteful to recalculate it, but oh well...

                    scores[i]["SASA"] = getSASA(pose2, resno=[lig.seqpos() for lig in ligands])
                    scores[i]["SASA_rel"] = scores[i]["SASA"] / free_ligand_SASA
    
                    if args.analyze is False and scores[i]["SASA_rel"] > args.SASA_limit:
                        print(f"{pdbfile}: ligand too exposed, L_SASA = {scores[i]['SASA_rel']:.3f}")
                        continue
    
                    if args.analyze is False and scores[i]["SASA_rel"] < 0.01:
                        print(f"{pdbfile}: ligand too buried, L_SASA = {scores[i]['SASA_rel']:.3f}")
                        continue
    
                    if args.ligand_exposed_atoms is not None and args.exposed_atom_SASA is not None:
                        target_ligand = [res for res in pose2.residues if all([res.has(a) for a in args.ligand_exposed_atoms])]
                        if not target_ligand:
                            print(f"Cannot find a ligand that contains atoms: {args.ligand_exposed_atoms}")
                            continue
                        indexes = [target_ligand[0].atom_index(x) for x in args.ligand_exposed_atoms]
                        surf_vol_nosc = getSASA(pose2, ignore_sc=True)
                        scores[i]["SASA_exposed_atoms"] = sum([surf_vol_nosc.surf(pose2.size(), i) for i in indexes])
    
                        if args.analyze is False and scores[i]["SASA_exposed_atoms"] < args.exposed_atom_SASA:
                            print(f"{pdbfile}: ligand atoms {args.ligand_exposed_atoms} too buried: {scores[i]['SASA_exposed_atoms']:.3f}")
                            continue
                else:
                    if "SASA" in datacolumns:
                        datacolumns.remove("SASA")
                    if "SASA_rel" in datacolumns:
                        datacolumns.remove("SASA_rel")

                if not args.analyze:
                    scores[i]["passed"] = True

                ### Trying to add matcher catalytic residue info to fixed PDB's, if available
                if args.analyze is False:
                    print(f"{pdbfile}: GOOD design")
                    # matched_residues = get_matcher_residues(ref_pdb)
                    if len(matched_residues) != 0:
                        matched_residues_in_design = {}
                        for r in matched_residues:
                            if ref_pose.pdb_rsd((matched_residues[r]["chain"], r)).is_protein():
                                for i, res in enumerate(trb["con_ref_pdb_idx"]):
                                    if res == (matched_residues[r]["chain"], np.int64(r)):
                                        resno_in_design = trb["con_hal_pdb_idx"][i][1]
                                        matched_residues_in_design[resno_in_design] = copy.deepcopy(matched_residues[r])
                                        matched_residues_in_design[resno_in_design]["chain"] = trb["con_hal_pdb_idx"][i][0]
                                        # Adjusting target residue number if it's not ligand. In case of an upstream match
                                        tgt_resno_orig = matched_residues_in_design[resno_in_design]["target_resno"]
                                        if tgt_resno_orig != 0 and ref_pose.residue(tgt_resno_orig).is_protein()\
                                            and (matched_residues_in_design[resno_in_design]["target_chain"], np.int64(tgt_resno_orig)) not in trb["con_hal_pdb_idx"]:
                                                (_ch, _rn) = trb["con_hal_pdb_idx"][trb["con_ref_pdb_idx"].index((matched_residues[r]["target_chain"],
                                                                                                                  np.int64(matched_residues[r]["target_resno"])))]
                                                matched_residues_in_design[resno_in_design]["target_chain"] = _ch
                                                matched_residues_in_design[resno_in_design]["target_resno"] = _rn
                                        elif tgt_resno_orig != 0 and ref_pose.residue(tgt_resno_orig).is_ligand():
                                            ligands2_in_design = [res for res in pose2.residues if res.name3() == matched_residues[r]["target_name"]]
                                            if len(ligands2_in_design) == 1:
                                                matched_residues_in_design[resno_in_design]["target_chain"] = pose2.pdb_info().chain(ligands2_in_design[0].seqpos())
                                                matched_residues_in_design[resno_in_design]["target_resno"] = pose2.pdb_info().number(ligands2_in_design[0].seqpos())
                                            else:
                                                print("    WARNING!!! multiple ligands with the same name in the system. REMARK 666 adjustment will not work properly!!")
                                                continue
                                        break
                            else:
                                if ref_pose.pdb_rsd((matched_residues[r]["target_chain"], matched_residues[r]["target_resno"])).is_ligand():
                                    # Figuring out which ligand in the diffusion output corresponds to which ligand in input
                                    ligands_in_design = [res for res in pose2.residues if res.name3() == matched_residues[r]["name3"]]
                                    ligands2_in_design = [res for res in pose2.residues if res.name3() == matched_residues[r]["target_name"]]
                                    if len(ligands_in_design) == 1 and len(ligands2_in_design) == 1:
                                        # easy case
                                        resno_in_design = pose2.pdb_info().number(ligands_in_design[0].seqpos())
                                        chain_in_design = pose2.pdb_info().chain(ligands_in_design[0].seqpos())
                                        matched_residues_in_design[resno_in_design] = copy.deepcopy(matched_residues[r])
                                        matched_residues_in_design[resno_in_design]["chain"] = chain_in_design
                                        matched_residues_in_design[resno_in_design]["target_chain"] = pose2.pdb_info().chain(ligands2_in_design[0].seqpos())
                                        matched_residues_in_design[resno_in_design]["target_resno"] = pose2.pdb_info().number(ligands2_in_design[0].seqpos())
                                    else:
                                        print("    WARNING!!! multiple ligands with the same name in the system. REMARK 666 adjustment will not work properly!!")
                                        continue
                                else:
                                    print("    WARNING!!! Ligand matched to a secondary residue. REMARK 666 adjustment not implemented yet!!")


                        # Adjusting matched residue info in case some of the reference REMARK 666 residues were not included in diffusion contigs
                        missing_matched_resnos = []
                        for k in matched_residues:
                            if (matched_residues[k]["chain"], k) not in trb["con_ref_pdb_idx"] and not ref_pose.pdb_rsd((matched_residues[r]["target_chain"], matched_residues[r]["target_resno"])).is_ligand():
                                print(f"  {pdbfile}, deleting {k} from REMARK 666")
                                missing_matched_resnos.append(k)
                        for k in missing_matched_resnos:
                                matched_residues.__delitem__(k)

                        pose2 = add_matcher_line_to_pose(pose2, ref_pose, matched_residues_in_design, matched_residues)

                    # Adding motif residues also a REMARK PDBinfo-LABEL:
                    if any(["literal" in x for x in trb.keys()]):  # guideposted diffusion has broken inpaint_seq TRB
                        motif_residues_fixed = [x for x in trb["con_hal_idx0"]]
                    else:
                        motif_residues_fixed = [x for x in trb["con_hal_idx0"] if trb["inpaint_seq"][x] == True]
                    for rn in motif_residues_fixed:
                        pose2.pdb_info().add_reslabel(rn+1, "motif")


                    pose2.dump_pdb(f"{filtered_dir}/{os.path.basename(pdbfile)}")
                    if _j == 0:
                        copy2(trbfile, f"{filtered_dir}/{os.path.basename(trbfile)}")
                    else:
                        copy2(trbfile, f"{filtered_dir}/{os.path.basename(pdbfile).replace('.pdb', '.trb')}")



    if args.nproc is not None:
        N_PROCESSES = args.nproc
    elif "SLURM_CPUS_ON_NODE" in os.environ:
        N_PROCESSES = int(os.environ["SLURM_CPUS_ON_NODE"])
    elif "OMP_NUM_THREADS" in os.environ:
        N_PROCESSES = int(os.environ["OMP_NUM_THREADS"])
    elif args.nproc is None:
        N_PROCESSES = os.cpu_count()

    print(f"Using {N_PROCESSES} processes")
    pool = multiprocessing.Pool(processes=N_PROCESSES,
                                initializer=process,
                                initargs=(the_queue, ref_pdbs, ref_poses, ))

    # None to end each process
    for _i in range(N_PROCESSES):
        the_queue.put(None)

    # Closing the queue and the pool
    the_queue.close()
    the_queue.join_thread()
    pool.close()
    pool.join()

    end = time.time()

    print("Analyzing diffusion outputs took {:.3f} seconds.".format(end - start))

    df = pd.DataFrame()
    # Finding a scoredict entry with all of the keys and prebuilding the DataFrame
    keylens = {i: len(scores[i]) for i in scores.keys()}
    all_keys_idx = max(keylens, key=keylens.get)

    for k in scores[all_keys_idx].keys():
        if k == "description":
            continue
        df[k] = float
    df["description"] = str

    # populating the dataframe with values
    for i in scores.keys():
        for k in scores[i].keys():
            if k == "description":
                continue
            else:
                df.at[i, k] = scores[i][k]
        df.at[i, "description"] = scores[i]["description"]

    # Adding empty columns of scores that were not calculated
    for k in datacolumns:
        if k not in df.keys():
            df[k] = np.nan


    for k in df.keys():
        if k == "description":
            continue
        df = df.sort_values(k)

    if len(df) < 200:
        print(df)
    else:
        print("Too many structures, only printing passing scores...")
        print(df.dropna())

    if args.analyze is False:
        print(f"##### {len(df.loc[df.passed == 1.0])}/{len(df)} backbones passed all filters  #######")
        print(f"Backbones and TRBs that pass filters have been copied to `{args.outdir}`")
    print(f"Saving the analysis scores of each backbone into `{args.scorefile_out}`")
    dump_scorefile(df, args.scorefile_out)

    print("Saving the secondary structure DSSP strings of each backbone into `dssps.fasta`")
    with open("dssps.fasta", "w") as file:
        for k in dssps.keys():
            file.write(f">{k}\n")
            file.write(dssps[k]+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pdb", nargs="+", type=str, help="Input PDBs from aa_diffusion")
    parser.add_argument("--pdbpath", nargs="+", type=str, help="Directories where input PDBs can be found")
    parser.add_argument("--trb", nargs="+", type=str, help="(optional) TRB files from aa_diffusion")
    parser.add_argument("--ref", nargs="+", type=str, help="Reference PDBs used as input for diffusion. If not provided, then the reference structure will be taken from TRB files.")
    parser.add_argument("--ref_path", type=str, help="Path where reference PDBs can be found.")
    parser.add_argument("--analyze", action="store_true", default=False, help="Analyze only. Will not move any files. Will calculate all metrics and take a bit longer.")

    parser.add_argument("--fix", action="store_true", default=False, help="Fixes issues with broken residue sidechains. Need to use it if the script fails because diffusion has produced broken sidechains.")
    parser.add_argument("--traj", help="How many steps of the trajectory of a design should be parsed.\nUse <N> to pick last N structures from the trajcetory. Use <N>/<M> to pick random <N> structures from the last <M> steps."
                        "Disabled by default.\nAssumes that for a design PDB 'path/pdbfile.pdb' a trajectory file can be found at 'path/traj/pdbfile_pX0_traj.pdb'")
    parser.add_argument("--rethread", action="store_true", default=False, help="Rethread the existing sequence to the backbone. This will fix sidechain weirdnesses coming from diffusion.")
    parser.add_argument("--params", nargs="+", type=str, help="Params files of ligands and noncanonicals")

    parser.add_argument("--lig_dist", default=2.5, type=float, help="(default 2.5) Cutoff for smallest allowed backbone to ligand heavyatom distance.")
    parser.add_argument("--SASA_limit", default=0.20, type=float, help="(default 0.2) Cutoff for ligand relative SASA")
    parser.add_argument("--loop_limit", default=0.30, type=float, help="(default 0.3) Cutoff for maximum allowed loop content")
    parser.add_argument("--longest_helix", default=30, type=int, help="(default 30) Longest allowed heix length")
    parser.add_argument("--rog", default=30.0, type=float, help="(default 30.0) Largest allowed radius of gyration")
    parser.add_argument("--term_limit", default=15.0, type=float, help="(default 15.0) Cutoff for how close the termini of the protein can be to any ligand heavyatom.")
    parser.add_argument("--bondlen_dev", default=0.1, type=float, help="(default 0.1) Maximum allowed sidechain bondlength deviation from normal in case of tip-atom diffusion.")
    parser.add_argument("--exclude_clash_atoms", type=str, nargs="+", help="Ligand atom names that will be excluded from ligand clashchecking")
    parser.add_argument("--ligand_exposed_atoms", type=str, nargs="+", help="Ligand atoms with --ligand_exposed_atoms should have SASA above this cutoff")
    parser.add_argument("--exposed_atom_SASA", type=float, help="Relative SASA cutoff for ligand atoms defined with --exposed_atom_SASA")
    parser.add_argument("--ref_catres", type=str, nargs="+", help="(optional) Catalytic residue positions in reference structure. Ranges can be represented with a dash.")
    parser.add_argument("--loop_catres", action="store_false", default=True, help="(default = True) If enabled, structures where any catalytic residue has 2 loopy residues on both side will be filtered out.")
    parser.add_argument("--cart_bonded", type=float, help="(default None) Cutoff for analyzing motif residues for sidechain quality based on Rosetta cart_bonded scoreterm.")
    parser.add_argument("--fa_dun", type=float, help="(default None) Cutoff for analyzing motif residue rotamers based on Rosetta fa_dun scoreterm.")

    parser.add_argument("--scorefile_out", type=str, default="diffusion_analysis.sc", help="Filename of the output scorefile.")
    parser.add_argument("--outdir", type=str, default="filtered_structures", help="Where are the fixed and filtered output PDBs copied?")
    parser.add_argument("--partial", action="store_true", default=False, help="Are you running this on partial diffusion output?")
    parser.add_argument("--nproc", type=int, help="# of CPU cores used")

    args = parser.parse_args()
    main(args)
