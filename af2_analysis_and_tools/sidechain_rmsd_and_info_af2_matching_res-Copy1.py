#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 2022
Updated on Tue Apr 30 2024

@author: ikalvet & updated by Seth Woodbury + Donghyo Kim
"""
import sys, os
sys.path = [x for x in sys.path if sys.base_exec_prefix in x] + [x for x in sys.path if sys.base_exec_prefix not in x]
import pyrosetta as pyr
import pyrosetta.rosetta
import re
import pandas as pd
import numpy as np
import argparse
import scipy.spatial
import queue
import threading
import multiprocessing
import json


def reorder_df_columns(df):
    namekeys = ["Name", "Output_PDB"]
    # Reordering hte columns to make sure that the name column is last
    cols = df.columns.tolist()
    cols = [x for x in cols if x not in namekeys] + [x for x in cols if x in namekeys]
    df = df[cols]
    return df


def dump_scorefile(df, filename):
    widths = {}
    namekeys = ["Output_PDB", "Name"]

    for k in df.keys():
        if k in ["SCORE:"] + namekeys:
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
            elif k in namekeys:
                title += f" {k}"
            else:
                title += f"{k:>{widths[k]}}"
        if all([t not in df.keys() for t in namekeys]):
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
                elif k in namekeys:
                    line += f" {val}"
                else:
                    line += f"{val:>{widths[k]}}"
            if all([t not in df.keys() for t in namekeys]):
                line += f" {index}"
            file.write(line + "\n")


def rmsd(geom, target):
    return np.sqrt(((geom - target) ** 2).mean())



def get_per_res_rmsd(design: pyrosetta.rosetta.core.pose.Pose, prediction: pyrosetta.rosetta.core.pose.Pose, rmsd_type: str, N_term_offset=0) -> list:
    """calculate per residue rmsd (Ca or sc) of prediction to design"""    
    result, result_bb, result_bbcb, result_imidazol = {}, {}, {}, {}
    bb_atoms, bbcb_atoms, imidazol_atoms = ["N", "CA", "C", "O"], ["N", "CA", "C", "O", "CB"], ["CG", "CE1", "ND1", "NE2", "CD2"]
    for resno in range(1, prediction.size()+1):
        resp = prediction.residue(resno)
        resd = design.residue(resno+N_term_offset)
        if resp.name3() != resd.name3():
            result[resno-1] = 0.0
        else:
            atoms = [resp.atom_name(n).strip() for n in range(1, resp.natoms()+1) if not resp.atom_is_hydrogen(n) and not resp.is_virtual(n)]
            ref_coords = [resd.xyz(a) for a in atoms]
            mdl_coords = [resp.xyz(a) for a in atoms]
            
            ref_bb_coords = [resd.xyz(a) for a in bb_atoms]
            mdl_bb_coords = [resp.xyz(a) for a in bb_atoms]
            
            res_bbcb_atom = list(set(atoms).intersection(bbcb_atoms))
            ref_bbcb_coords = [resd.xyz(a) for a in res_bbcb_atom]
            mdl_bbcb_coords = [resp.xyz(a) for a in res_bbcb_atom]
             
            result[resno-1] = np.sqrt(sum([(np.linalg.norm(c1-c2))**2 for c1, c2 in zip(ref_coords, mdl_coords)])/len(atoms))
            result_bb[resno-1] = np.sqrt(sum([(np.linalg.norm(c1-c2))**2 for c1, c2 in zip(ref_bb_coords, mdl_bb_coords)])/len(bb_atoms))
            result_bbcb[resno-1] = np.sqrt(sum([(np.linalg.norm(c1-c2))**2 for c1, c2 in zip(ref_bbcb_coords, mdl_bbcb_coords)])/len(res_bbcb_atom))

            if resd.name3() == "HIS":
                ref_imidazol_coords = [resd.xyz(a) for a in imidazol_atoms]
                mdl_imidazol_coords = [resp.xyz(a) for a in imidazol_atoms]

                result_imidazol[resno-1] = np.sqrt(sum([(np.linalg.norm(c1-c2))**2 for c1, c2 in zip(ref_imidazol_coords, mdl_imidazol_coords)])/len(imidazol_atoms))
            else:
                result_imidazol[resno-1] = np.isnan
    return result, result_bb, result_bbcb, result_imidazol



def get_matcher_residues(filename):
    pdbfile = open(filename, 'r').readlines()

    matches = {}
    for l in pdbfile:
        if "ATOM" in l:
            break
        if "REMARK 666" in l:
            lspl = l.split()
            chain = lspl[9]
            res3 = lspl[10]
            resno = int(lspl[11])
            
            matches[resno] = {'name3': res3,
                              'chain': chain}
    return matches


def get_residues_with_close_sc_or_bb(pose, ref_resno, residues=None, exclude_residues=None):
    """
    """
    if residues is None:
        residues = [x for x in range(1, pose.size()+1)]
    if exclude_residues is None:
        exclude_residues = []

    ref_residue = pose.residue(ref_resno)
    heavyatoms = [ref_residue.atom_name(n).strip() for n in range(1, ref_residue.natoms()+1) if ref_residue.atom_type(n).is_heavyatom()]

    close_ones = []
    for resno in residues:
        if resno in exclude_residues:
            continue
        if pose.residue(resno).is_ligand():
            continue
        if (pose.residue(resno).nbr_atom_xyz() - ref_residue.nbr_atom_xyz()).norm() > 14.0:
            continue
        res = pose.residue(resno)
        close_enough = False

        for ha in heavyatoms:
            # If the CA of residue is really close then keep it, no questions asked
            if (res.xyz("CA") - ref_residue.xyz(ha)).norm() < 5.5:
                close_enough = True
                close_ones.append(resno)
                break
            else:
                # If the CA is further then check if any sidechain atoms is really close
                for atomno in range(1, res.natoms()):
                    if res.atom_type(atomno).is_heavyatom():
                        if (res.xyz(atomno) - ref_residue.xyz(ha)).norm() < 4.5:
                            close_enough = True
                            close_ones.append(resno)
                            break
                if close_enough is True:
                    break
            if close_enough is True:
                break
    return close_ones


# def main():
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # parser.add_argument('--params', metavar='FILENAME', nargs='+', help='params files needed to load any PDBs')
    # parser.add_argument('--ref_path', metavar='FILENAME', nargs='+', help='params files needed to load any PDBs')
    
    params = None
    scorefile = None
    ref_path = None
    pred_pdb = None
    align = False
    suffix = ""
    pdblist = None
    
    if "--help" in sys.argv:
        sys.exit(0)
    
    if '--params' in sys.argv:
        params = sys.argv[sys.argv.index('--params')+1:]
        params = [p for p in params if ".params" in p]
    
    if '--ref' in sys.argv:
        ref_pdb = sys.argv[sys.argv.index('--ref')+1]
    elif "--ref_path" in sys.argv:
        pass
    else:
        sys.exit("Need --ref <pdbfile>")

    if "--ref_path" in sys.argv:
        ref_path = sys.argv[sys.argv.index('--ref_path')+1]

    if '--pred' in sys.argv:
        pred_pdb = sys.argv[sys.argv.index('--pred')+1:]
        pred_pdb = [f for f in pred_pdb if ".pdb" in f]
    # else:
    #     sys.exit("Need --pred <pdbfile(s)>")

    if "--scorefile" in sys.argv:
        scorefile = sys.argv[sys.argv.index('--scorefile')+1]

    if "--align" in sys.argv:
        align = True

    mpnn_naming = False
    if "--mpnn" in sys.argv:
        mpnn_naming = True
    
    if "--suffix" in sys.argv:
        suffix = "_" + sys.argv[sys.argv.index('--suffix')+1]
    else:
        suffix = "_matchres"

    if "--pdblist" in sys.argv:
        pdblist_file = sys.argv[sys.argv.index('--pdblist')+1]
        pdblist = open(pdblist_file, "r").readlines()
        pdblist = [x.rstrip() for x in pdblist]


    if os.path.exists(f"scores{suffix}.sc"):
        print(f"Skipping {scorefile}")
        sys.exit(0)

    scores = pd.read_csv(scorefile)

    if params is None:
        pyr.init('-mute all -beta_nov16')
    else:
        pyr.init('-extra_res_fa {} -mute all'.format(" ".join(params)))


    the_queue = multiprocessing.Queue()  # Queue stores the iterables

    manager = multiprocessing.Manager() 
    ref_poses = manager.dict()  # Need a special dictionary to store outputs from multiple processes
    results = manager.dict()

    for idx, row in scores.iterrows():
        results[idx] = manager.dict()
        the_queue.put(idx)


    def process(q):
        while True:
            idx = q.get(block=True)
            if idx is None:
                return
            row = scores.iloc[idx]

            results[idx]["Output_PDB"] = row["Output_PDB"]
            results[idx]['rmsd'] = np.nan

            if "Output_PDB" in scores.keys():
                model_name = f"{row['Output_PDB']}"
                if ".pdb" not in model_name:
                    model_name += ".pdb"
            else:
                model_name = f"{row.ID}_{row['Model/Tag']}"

            if mpnn_naming is True:
                if "_native" in row['Name'][-7:]:
                    # replacing last instance of 'native'
                    ref_name = "".join(row['Name'].rsplit("_native", 1))+".pdb"
                elif bool(re.search("_T[0-9].[0-9]_s[0-9]_", row['Name'])) or bool(re.search("_T[0-9].[0-9]_sample[0-9]_", row['Name'])):
                    ref_name = "_".join(row['Name'].split("_")[:-3])+".pdb"
                elif "model_4" in row["Name"]:
                    ref_name = row['Name'].replace("_model_4_ptm_seed_0_unrelaxed.pdb", ".pdb")
                else:
                    ref_name = f"{row['Name']}"+".pdb"
            else:
                ref_name = f"{row['Name']}"
            results[idx]["ref_path"] = os.path.join(ref_path, ref_name)
                
            if ref_name not in ref_poses.keys() or ref_poses[ref_name] is None:
                ref_poses[ref_name] = pyr.pose_from_file(os.path.join(ref_path, ref_name))
                
            ref_pose = ref_poses[ref_name]

            if "Output_PDB" in scores.keys():
                model_pdb = f"{row['Output_PDB']}.pdb"
                if ".pdb" in row['Output_PDB']:
                    model_pdb = f"{row['Output_PDB']}"
                model_pose = pyr.pose_from_file(model_pdb)
            else:
                model_pose = pyr.pose_from_file(f"{row.ID}_{row['Model/Tag']}.pdb")

            reslist = pyrosetta.rosetta.std.list_unsigned_long_t()
            for n in range(1, model_pose.size()+1):
                reslist.append(n)

            matches = get_matcher_residues(os.path.join(ref_path, ref_name))

            # Finding how the sequence should be aligned
            # It doesn't do non-continuous sequences yet - TODO!
            if len(ref_pose.sequence()) < len(model_pose.sequence()):
                shorter_seq = ref_pose.sequence()
                longer_seq = model_pose.sequence()
                _sp = ref_pose
                _lp = model_pose
            else:
                shorter_seq = model_pose.sequence()
                longer_seq = ref_pose.sequence()
                _lp = ref_pose
                _sp = model_pose

            N_term_offset = longer_seq.find(shorter_seq)
            if mpnn_naming is True:
                N_term_offset = 0
            C_term_offset = len(longer_seq[N_term_offset:]) - len(shorter_seq)

            if N_term_offset == -1:
                print(f"{row.Output_PDB}: Sequences not alignable?\n{shorter_seq}\n{longer_seq}")
                continue
            # elif any([x != 0 for x in (N_term_offset, C_term_offset)]):
            #     print(f"{row.Output_PDB}: Using offsets {N_term_offset}, {C_term_offset}")


            if align is False:
                
                overlay_pos = pyrosetta.rosetta.utility.vector1_unsigned_long()
                for n in range(1, _sp.size()+1):
                    overlay_pos.append(n)
                rmse = pyrosetta.rosetta.protocols.toolbox.pose_manipulation.superimpose_pose_on_subset_CA(_lp, _sp, overlay_pos, N_term_offset)
                results[idx]['rmsd'] = rmse

            elif align is True:


                # Finding the distance matrices of CA atoms in both poses
                shorter_distmat = scipy.spatial.distance.pdist([np.array(_sp.residue(n+1).xyz("CA")) for n in range(len(shorter_seq))], 'euclidean')
                longer_distmat = scipy.spatial.distance.pdist([np.array(_lp.residue(n+1).xyz("CA")) for n in range(N_term_offset, len(longer_seq)-C_term_offset)], 'euclidean')

                assert len(shorter_distmat) == len(longer_distmat), f"{row.Output_PDB}: distmats not equal length"

                # Calculating CA RMSD
                rmse = rmsd(shorter_distmat, longer_distmat)
                results[idx]["rmsd"] = rmse
                # scores.at[idx, 'rmsd'] = rmse

            print(f"{row.Output_PDB}: length = {min([ref_pose.size(), model_pose.size()])}, "
                  f"plDDT = {row.plDDT:.2f}, rmsd = {rmse:.3f}, ")

            # Calculating sidechain RMSD of each residue
            res_rmsd, res_bbrmsd, res_bbcbrmsd, res_imidazolrmsd = get_per_res_rmsd(ref_pose, model_pose, 'sc', N_term_offset=N_term_offset)

            # Finding RMSD's of matched residues
            mr_rmsds, mr_bbrmsds, mr_bbcbrmsds, mr_imidazolrmsds = [], [], [], []
            for i, resno in enumerate(matches):
                # scores.at[idx, f'rmsd_SR{i+1}'] = res_rmsd[resno-1]
                results[idx][f'rmsd_SR{i+1}'] = res_rmsd[resno-1]
                results[idx][f'bbrmsd_SR{i+1}'] = res_bbrmsd[resno-1]
                results[idx][f'bbcbrmsd_SR{i+1}'] = res_bbcbrmsd[resno-1]
                results[idx][f'imidazolrmsd_SR{i+1}'] = res_imidazolrmsd[resno-1]
                
                if np.isnan(res_rmsd[resno-1]):
                    results[idx][f'rmsd_SR{i+1}'] = 0.0
                if np.isnan(res_bbrmsd[resno-1]):
                    results[idx][f'bbrmsd_SR{i+1}'] = 0.0
                if np.isnan(res_bbcbrmsd[resno-1]):
                    results[idx][f'bbcbrmsd_SR{i+1}'] = 0.0
                if np.isnan(res_imidazolrmsd[resno-1]):
                    results[idx][f'imidazolrmsd_SR{i+1}'] = 0.0
                    
                mr_rmsds.append(results[idx][f'rmsd_SR{i+1}'])
                mr_bbrmsds.append(results[idx][f'bbrmsd_SR{i+1}'])
                mr_bbcbrmsds.append(results[idx][f'bbcbrmsd_SR{i+1}'])
                mr_imidazolrmsds.append(results[idx][f'imidazolrmsd_SR{i+1}'])
            results[idx]['avr_rmsd_SR'] = np.mean(mr_rmsds)
            results[idx]['avr_bbrmsd_SR'] = np.mean(mr_bbrmsds)
            results[idx]['avr_bbcbrmsd_SR'] = np.mean(mr_bbcbrmsds)
            results[idx]['avr_imidazolrmsd_SR'] = np.mean(mr_imidazolrmsds)
            
            # Freeing up memory if a particular reference pose is no longer used
            if all([ref_name not in row['Name'] for i,row in scores.iloc[idx:].iterrows()]):
                ref_poses[ref_name] = None

    if "OMP_NUM_THREADS" in os.environ:
        N_PROCESSES = int(os.environ["OMP_NUM_THREADS"])
        print(f"Using {N_PROCESSES} processes")
    else:
        N_PROCESSES = os.cpu_count() - 1


    pool = multiprocessing.Pool(processes=N_PROCESSES,
                                initializer=process,
                                initargs=(the_queue, ))

    # None to end each process
    for _i in range(N_PROCESSES):
        the_queue.put(None)

    # Closing the queue and the pool
    the_queue.close()
    the_queue.join_thread()
    pool.close()
    pool.join()


    for i in results.keys():
        assert scores.at[i, "Output_PDB"] == results[i]["Output_PDB"], f"Bad match between scores dataframe and results dict? {results[i].items(), scores.iloc[i]}"
        for _k in results[i].keys():
            scores.at[i, _k] = results[i][_k]


    # print(scores.keys())
    #scores = reorder_df_columns(scores.drop(columns=["Sequence"]))

    _title = f"{('plDDT'):>8} {('rmsd'):>8}"
    for k in scores.keys():
        if "SR" in k:
            _title += f" {k:>8}"
        #if "rmsd_pocket" in k:
        #    _title += f"{('rmsd_pocket'):>12}"
    _title += f" {('Name'):<}"
    print(_title)

    for idx, row in scores.iterrows():
        if row.hasnans:
            continue
        _line = f"{row['plDDT']:>8.3f} {row['rmsd']:>8.3f}"
        for k in scores.keys():
            if "SR" in k:
                _line += f" {row[k]:>8.3f}"
        _line += f" {row['Name']:<}"
        print(_line)
        
    #dump_scorefile(scores, f"scores{suffix}.sc")
    scores.to_csv(f"{os.path.dirname(scorefile)}/scores_matres.sc", sep="\t")

# if __name__ == "__main__":
#     main()
