#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 13:11:08 2023

@author: ikalvet
"""

import pandas as pd
import matplotlib.pyplot as plt
import pyrosetta as pyr
import pyrosetta.rosetta
import pyrosetta.distributed.io
import numpy as np
import glob, os, sys
from shutil import copy2
import multiprocessing
import argparse
import math
from textwrap import wrap

# --- locate repo root + shared external paths ---
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "repo_paths.py").is_file():
        _sys.path.insert(0, str(_anc)); break
import repo_paths

HOME = ""
sys.path.append(repo_paths.ENZYME_DESIGN_UTILS)
import design_utils
import scoring_utils
sys.path.append(repo_paths.INVROTZYME_UTILS)
import align_pdbs

import warnings
warnings.filterwarnings('ignore')

comparisons = {'<=': '__le__',
               '<': '__lt__',
               '>': '__gt__',
               '>=': '__ge__',
               '=': '__eq__'}

def get_pocket_residues(pose):
    ligands = [res for res in pose.residues if res.is_ligand()]
    heavyatoms = {ligand.seqpos(): [ligand.atom_name(n+1).strip() for n in range(ligand.natoms()) if ligand.atom_type(n+1).element() != "H"] for ligand in ligands}

    pocket_residues = []
    for res in pose.residues:
        if res.is_ligand():
            continue
        if min([(res.nbr_atom_xyz() - lig.nbr_atom_xyz()).norm() for lig in ligands]) > 15.0:
            continue
        for ligand in ligands:
            if res.seqpos() in pocket_residues:
                continue
            for ha in heavyatoms[ligand.seqpos()]:
                if res.seqpos() in pocket_residues:
                    break
                if (res.xyz("CA") - ligand.xyz(ha)).norm() < 6.0:
                    pocket_residues.append(res.seqpos())
                    break
                for an in range(res.natoms()):
                    if (res.xyz(an) - ligand.xyz(ha)).norm() < 4.0:
                        pocket_residues.append(res.seqpos())
                        break
    pocket_residues_pdb = [pose.pdb_info().number(rn) for rn in pocket_residues]
    return pocket_residues, pocket_residues_pdb

def get_residue_rmsd(residue1, residue2, specified_atoms=None):
    if residue1.name3() != residue2.name3():
        return None
    else:
        atoms = [residue1.atom_name(n).strip() for n in range(1, residue1.natoms()+1) if not residue1.atom_is_hydrogen(n)]
        if specified_atoms:
            atoms = specified_atoms
        ref_coords = [residue2.xyz(a) for a in atoms]
        mdl_coords = [residue1.xyz(a) for a in atoms]
        rmsd = np.sqrt(sum([(np.linalg.norm(c1-c2))**2 for c1, c2 in zip(ref_coords, mdl_coords)])/len(atoms))
        return rmsd


# PNear calculation function by Vikram from tools/analysis/compute_pnear.py
# Given a vector of scores, a matching vector of rmsds, and values for lambda and kbt,
# compute the PNear value.
def calculate_pnear(scores, rmsds, lambda_val=1.5, kbt=0.62):
    nscores = len(scores)
    assert nscores == len(rmsds), "Error in calculate_pnear(): The scores and rmsds lists must be of the same length."
    assert nscores > 0, "Error in calculate_pnear(): At least one score/rmsd pair must be provided."
    assert kbt > 1e-15, "Error in calculate_pnear(): kbt must be greater than zero!"
    assert lambda_val > 1e-15, "Error in calculate_pnear(): lambda must be greater than zero!"
    minscore = min(scores)
    weighted_sum = 0.0
    Z = 0.0
    lambdasq = lambda_val * lambda_val
    for i in range(nscores):
        val1 = math.exp(-(rmsds[i] * rmsds[i]) / lambdasq)
        val2 = math.exp(-(scores[i] - minscore) / kbt)
        weighted_sum += val1 * val2
        Z += val2
    assert Z > 1e-15, "Math error in calculate_pnear()! This shouldn't happen."
    return weighted_sum / Z


def load_poses(models):
    poses = {}
    for i, mdl in enumerate(models):
        poses[i] = pyrosetta.distributed.io.pose_from_pdbstring(mdl).pose.clone()
    return [poses[n].clone() for n in range(len(poses))]


def filter_scores(scores, filters):
    """
    Filters are defined in this importable module
    """
    filtered_scores = scores.copy()

    for s in filters.keys():
        if filters[s] is not None and s in scores.keys():
            val = filters[s][0]
            sign = comparisons[filters[s][1]]
            filtered_scores = filtered_scores.loc[(filtered_scores[s].__getattribute__(sign)(val))]
            n_passed = len(scores.loc[(scores[s].__getattribute__(sign)(val))])
            print(f"{s:<24} {filters[s][1]:<2} {val:>7.3f}: {len(filtered_scores)} "
                  f"designs left. {n_passed} pass ({(n_passed / len(scores)) * 100:.0f}%).")
    return filtered_scores


def plot_scores(scores):
    """
    Plotting of ChemNet scores vs lddt
    """
    scoreterms = ["kabsch", "plddt", "rmsd", "prmsd"]
    designs = sorted(list(set(scores.label.values)))
    rows = len(designs)
    plt.figure(figsize=(6 * len(scoreterms), rows * 4))

    n = 1
    for i, d in enumerate(designs):
        _df = scores.loc[scores.label.str.contains(d)]
        for sc in scoreterms:
            plt.subplot(rows, len(scoreterms), n)
            if sc == scoreterms[0]:
                plt.title("\n".join(wrap(d, 40)), weight="bold", fontsize=14)
            plt.xlabel(sc, weight="bold", fontsize=14)
            plt.ylabel("lddt", weight="bold", fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.scatter(x=_df[sc], y=_df["lddt"], alpha=0.5)
            plt.tight_layout()
            n += 1

    plt.savefig(f"{args.outdir}/chemnet_scores_vs_lddt.png", dpi=300)

def find_metal_contacts(pose, metal_atoms, distances=[3.0, 2.75, 2.5, 2.3]):
    contacts = {f"{metal}_{str(d).replace('.', 'point')}A": {'residues': [], 'atoms': []} for metal in metal_atoms for d in distances}
    for metal in metal_atoms:
        metal_res_idx = None
        metal_atom_idx = None
        
        # Find the metal atom's residue and atom index
        for res in pose.residues:
            for atom in range(1, res.natoms() + 1):
                if res.atom_name(atom).strip() == metal:
                    metal_res_idx = res.seqpos()
                    metal_atom_idx = atom
                    break
            if metal_res_idx is not None:
                break
        
        if metal_res_idx is None:
            print(f"Metal atom {metal} not found in the pose.")
            continue

        metal_xyz = pose.residue(metal_res_idx).xyz(metal_atom_idx)
        
        for res in pose.residues:
            if res.seqpos() == metal_res_idx:
                continue
            for atom in range(1, res.natoms() + 1):
                if res.atom_is_hydrogen(atom):
                    continue
                distance = metal_xyz.distance(res.xyz(atom))
                for d in distances:
                    if distance < d:
                        contacts[f"{metal}_{str(d).replace('.', 'point')}A"]['residues'].append(res.name1() + str(pose.pdb_info().number(res.seqpos())))
                        contacts[f"{metal}_{str(d).replace('.', 'point')}A"]['atoms'].append(res.atom_name(atom).strip())
                        break  # Assuming one atom contact is enough to count the residue
    return contacts

def find_hbonds(pose, target_atoms, distance_cutoff=3.75):
    hbond_data = {atom: {'residues': [], 'atoms': []} for atom in target_atoms}
    for atom in target_atoms:
        target_res = pose.residue(atom)
        target_xyz = target_res.xyz(1)  # Assuming target atom is the first atom in the residue
        for res in pose.residues:
            if res.seqpos() == target_res.seqpos():
                continue
            for atom_idx in range(1, res.natoms() + 1):
                if res.atom_is_hydrogen(atom_idx):
                    continue
                distance = target_xyz.distance(res.xyz(atom_idx))
                if distance < distance_cutoff:
                    hbond_data[atom]['residues'].append(res.name1() + str(pose.pdb_info().number(res.seqpos())))
                    hbond_data[atom]['atoms'].append(res.atom_name(atom_idx).strip())
                    break  # Assuming one atom contact is enough to count the residue
    return hbond_data

def find_charge_interactions(pose, target_atoms, distances=[5.0, 10.0]):
    charged_residues = ['ARG', 'LYS', 'ASP', 'GLU']
    charge_data = {f"{atom}_{str(dist).replace('.', 'point')}A": [] for atom in target_atoms for dist in distances}
    for atom in target_atoms:
        target_res_idx = None
        target_atom_idx = None

        # Find the target atom's residue and atom index
        for res in pose.residues:
            for atom_idx in range(1, res.natoms() + 1):
                if res.atom_name(atom_idx).strip() == atom:
                    target_res_idx = res.seqpos()
                    target_atom_idx = atom_idx
                    break
            if target_res_idx is not None:
                break

        if target_res_idx is None:
            print(f"Target atom {atom} not found in the pose.")
            continue

        target_xyz = pose.residue(target_res_idx).xyz(target_atom_idx)

        for res in pose.residues:
            if res.name3() not in charged_residues or res.seqpos() == target_res_idx:
                continue
            for atom_idx in range(1, res.natoms() + 1):
                if res.atom_is_hydrogen(atom_idx):
                    continue
                distance = target_xyz.distance(res.xyz(atom_idx))
                for dist in distances:
                    if distance < dist:
                        charge_data[f"{atom}_{str(dist).replace('.', 'point')}A"].append(res.name1() + str(pose.pdb_info().number(res.seqpos())))
    return charge_data


arguments = sys.argv.copy()

parser = argparse.ArgumentParser()
parser.add_argument("--pdb", type=str, nargs="+", help="PDB files used as input(s) for ChemNet")
parser.add_argument("--pdb_path", type=str, help="Path to PDB files used as input(s) for ChemNet")
parser.add_argument("--chemnet_pdb_path", type=str, help="Path to PDB files used as output(s) for ChemNet")
parser.add_argument("--lig_name", type=str, help="Name of target ligand")
parser.add_argument("--lig_atom", type=str, nargs="+", help="Name of ions of target ligand")
parser.add_argument("--scorefile", type=str, nargs="+", help="ChemNet output scorefiles")
parser.add_argument("--scorefile_list", type=str, help="File with a list of chemnet output scorefiles")
parser.add_argument("--params", nargs="+", required=False, help="params files")
parser.add_argument("--nproc", type=int, default=os.cpu_count(), help="How many CPU cores")
parser.add_argument("--outdir", type=str, default="./", help="Output directory")
parser.add_argument("--scorefile_out", type=str, default="scorefile.txt", help="Output scorefile name")
parser.add_argument("--dump", action="store_true", default=False, help="Dump top 5 models as full protein PDB files")

# New arguments
parser.add_argument("--metals_for_contacts", type=str, nargs="+", help="Specify metal atoms for contact analysis")
parser.add_argument("--atoms_for_hbonds", type=str, nargs="+", help="Specify atoms for hydrogen bond analysis")
parser.add_argument("--charge_charge_atoms", type=str, nargs="+", help="Specify atoms for charge-charge interaction analysis")

args = parser.parse_args()

extra_res_fa = ""
if args.params is not None:
    extra_res_fa = "-extra_res_fa "
    for p in args.params:
        extra_res_fa += f"{p} "
else:
    extra_res_fa = f"-extra_res_fa {HOME}/home/ikalvet/projects/Heme/theozyme/HEM/no_h2o/HEM.params"

pyr.init(f"{extra_res_fa} -mute all -beta_nov16 -run:preserve_header")
sfx = pyr.get_fa_scorefxn()

if args.dump:
    dump_dir = os.path.join(args.chemnet_pdb_path, "top5_models")
    os.makedirs(dump_dir, exist_ok=True)

if args.scorefile_list is not None and args.scorefile is None:
    args.scorefile = open(args.scorefile_list, "r").readlines()
    args.scorefile = [x.strip() + ".csv" if ".csv" not in x else x.strip() for x in args.scorefile]

if args.pdb_path is not None and args.pdb is None:
    args.pdb = glob.glob(args.pdb_path + "/*.pdb")

## Reading the scorefiles
scores = pd.DataFrame()
for scf in args.scorefile:
    scores = pd.concat([scores, pd.read_csv(scf)], ignore_index=True)

designs = sorted(list(set(scores.label.values)))

# Assigning input PDBs to design names
ref_pdbs = {}
for d in designs:
    for p in args.pdb:
        if os.path.basename(p).replace(".pdb", "") == d:
            ref_pdbs[d] = p
    if d not in ref_pdbs.keys():
        print(f"Can't find reference PDB for design: {d}")

if len(scores) < 10:
    plot_scores(scores)

chi_pairs = [(1, 4), (2, 5), (3, 6), (4, 1), (5, 2), (6, 3)]

############################
### MAIN FILTERING BLOCK ###
############################

the_queue = multiprocessing.Queue()
manager = multiprocessing.Manager()

results = manager.dict()
frame_results = manager.dict()

scoreterms = ["kabsch", "plddt", "rmsd_new", "prmsd"]

for i, d in enumerate(designs):
    the_queue.put((i, d))

print(f"{('frac_good_kabsch'):>17}{('plddt_top5'):>15}{('rmsd_top5'):>11}{('plddt_pnear_good'):>18}{('Design'):>8}")

def process(q):
    while True:
        asd = q.get(block=True)
        if asd is None:
            return
        i = asd[0]
        d = asd[1]

        pocket_residues = {}
        pocket_residues_pdb = {}
        pocket_residues_crop = {}
        pocket_residue_rmsds = {}
        catalytic_residues = {}
        catalytic_residue_rmsds = {}
        pocket_residue_rmsds_good = {}
        model_res_scores = {}

        DF = pd.DataFrame()
        _df = scores.loc[scores.label.str.contains(d)]
        DF.at[i, "description"] = d
        DF.at[i, "frac_good_kabsch"] = len(_df.loc[_df.kabsch < 1.5]) / len(_df)
        DF.at[i, "kabsch"] = _df.kabsch.mean()

        pdbstr = open(os.path.join(args.chemnet_pdb_path, d + "_model.pdb"), "r").read()
        models = [mdl for mdl in pdbstr.split("ENDMDL") if len(mdl) > 10]
        model_poses = load_poses(models)

        pocket_residues_crop[d] = []
        best_plddt_idx = _df.sort_values("plddt", ascending=False).iloc[0]['model_idx']
        p = model_poses[best_plddt_idx - 1].clone()

        pr, prpdb = get_pocket_residues(p)
        pocket_residues[d] = pr
        pocket_residues_pdb[d] = prpdb

        catalytic_residues[d] = design_utils.get_matcher_residues(ref_pdbs[d])

        for res in p.residues:
            if p.pdb_info().number(res.seqpos()) in prpdb:
                pocket_residues_crop[d].append(p.pdb_info().number(res.seqpos()))

        model_res_scores[d] = pd.DataFrame()
        for j, mdl in enumerate(models):
            _mdl = mdl.split("\n")
            model_res_scores[d].at[len(model_res_scores[d]) + 1, "iter"] = i
            for resno in pocket_residues_pdb[d]:
                res_lines = [l for l in _mdl if f"A{resno:>4}" in l]
                res_scores = [float(l[61:67]) for l in res_lines]
                model_res_scores[d].at[len(model_res_scores[d]), resno] = np.average(res_scores)

        catalytic_residue_rmsds[d] = pd.DataFrame()
        ref_pose = pyr.pose_from_file(ref_pdbs[d])

        ref_lig_seqpos = None
        ref_ligands = [res for res in ref_pose.residues if res.is_ligand()]
        if len(ref_ligands) > 1:
            for res in ref_ligands:
                if res.name3() == args.lig_name:
                    ref_lig_seqpos = res.seqpos()
        else:
            assert ref_ligands[0].name3() == args.lig_name
            ref_lig_seqpos = ref_ligands[0].seqpos()

        _df_good = scores.loc[(scores.label.str.contains(d)) & (scores.kabsch <= 1.5)]

        mdl_ligands = [res for res in model_poses[0].residues if res.is_ligand()]
        mdl_lig_seqpos = None
        if len(mdl_ligands) > 1:
            for res in mdl_ligands:
                if res.name3() == args.lig_name:
                    mdl_lig_seqpos = res.seqpos()
        else:
            assert mdl_ligands[0].name3() == args.lig_name
            mdl_lig_seqpos = mdl_ligands[0].seqpos()

        for idx, row in _df.iterrows():
            p = model_poses[row.model_idx - 1]
            rmsd = pyrosetta.rosetta.core.scoring.automorphic_rmsd(ref_pose.residue(ref_lig_seqpos), p.residue(mdl_lig_seqpos), False)
            _df.at[idx, "rmsd_new"] = rmsd

        DF.at[i, "rmsd_top5"] = _df.sort_values("plddt", ascending=False).rmsd_new.iloc[:5].mean()
        DF.at[i, "plddt_top5"] = _df.sort_values("plddt", ascending=False).plddt.iloc[:5].mean()
        DF.at[i, "lddt_pnear"] = calculate_pnear(np.array(_df["lddt"]), np.array(_df["rmsd_new"]), lambda_val=1.5, kbt=0.62)
        DF.at[i, "plddt_pnear"] = calculate_pnear(np.array(_df["plddt"]), np.array(_df["rmsd_new"]), lambda_val=1.5, kbt=0.62)
        DF.at[i, "plddt_pde_pnear"] = calculate_pnear(np.array(_df["plddt_pde"]), np.array(_df["rmsd_new"]), lambda_val=1.5, kbt=0.62)

        DF.at[i, "rmsd_pde_top5"] = _df.sort_values("plddt_pde", ascending=False).rmsd_new.iloc[:5].mean()
        DF.at[i, "plddt_pde_top5"] = _df.sort_values("plddt_pde", ascending=False).plddt_pde.iloc[:5].mean()

        for jj, cr in enumerate(catalytic_residues[d]):
            good_rmsds = []
            for j, p in enumerate(model_poses):
                _cr_in_crop = pocket_residues[d][pocket_residues_pdb[d].index(cr)]
                catalytic_residue_rmsds[d].at[j, cr] = get_residue_rmsd(ref_pose.residue(cr), p.residue(_cr_in_crop))
                if j + 1 in _df_good.model_idx.values:
                    good_rmsds.append(catalytic_residue_rmsds[d].at[j, cr])
            DF.at[i, f"rmsd_catres{jj}"] = np.median(catalytic_residue_rmsds[d][cr])
            DF.at[i, f"rmsd_std_catres{jj}"] = np.std(catalytic_residue_rmsds[d][cr])

        lig_rmsds = []
        lig_ref_pos = ref_pose.size()
        for j, p in enumerate(model_poses):
            lig_mdl_pos = p.size()
            lig_rmsds.append(get_residue_rmsd(ref_pose.residue(lig_ref_pos), p.residue(lig_mdl_pos), args.lig_atom))
        DF.at[i, "rmsd_ligand"] = np.average(lig_rmsds)
        DF.at[i, "rmsd_std_ligand"] = np.std(lig_rmsds)

        pocket_residue_rmsds[d] = pd.DataFrame()
        pocket_residue_rmsds_good[d] = pd.DataFrame()
        for pr_no, prpdb_no in zip(pocket_residues[d], pocket_residues_pdb[d]):
            good_rmsds = []
            for j, p in enumerate(model_poses):
                pocket_residue_rmsds[d].at[j, prpdb_no] = get_residue_rmsd(ref_pose.residue(prpdb_no), p.residue(pr_no))
                if j + 1 in _df_good.model_idx.values:
                    pocket_residue_rmsds_good[d].at[j, prpdb_no] = pocket_residue_rmsds[d].at[j, prpdb_no]

        DF.at[i, "rmsd_pocket"] = pocket_residue_rmsds[d][pocket_residues_pdb[d]].median().median()
        DF.at[i, "rmsd_pocket_worst"] = pocket_residue_rmsds[d][pocket_residues_pdb[d]].median().max()
        DF.at[i, "rmsd_pocket_good_worst"] = pocket_residue_rmsds_good[d][pocket_residues_pdb[d]].median().max()

        DF.at[i, "u_pocket"] = model_res_scores[d][pocket_residues_pdb[d]].median().median()
        DF.at[i, "u_pocket_std"] = model_res_scores[d][pocket_residues_pdb[d]].std().median()
        DF.at[i, "u_pocket_worst"] = model_res_scores[d][pocket_residues_pdb[d]].median().max()

        print(f"{DF.at[i, 'frac_good_kabsch']:>17.2f}{DF.at[i, 'plddt_top5']:>15.2f}{DF.at[i, 'rmsd_top5']:>11.2f}{DF.at[i, 'plddt_pnear']:>18.2f}  {d}")

        if args.metals_for_contacts:
            metal_contacts = find_metal_contacts(p, args.metals_for_contacts)
            for key, data in metal_contacts.items():
                DF.at[i, f"{key}_res_contacts"] = ','.join(data['residues'])
                DF.at[i, f"{key}_atom_contacts"] = ','.join(data['atoms'])

        if args.atoms_for_hbonds:
            hbond_data = find_hbonds(p, args.atoms_for_hbonds)
            for atom, data in hbond_data.items():
                DF.at[i, f"hbonds_to_{atom}"] = len(data['residues'])
                DF.at[i, f"hbonds_{atom}_residues"] = ','.join(data['residues'])

        if args.charge_charge_atoms:
            charge_data = find_charge_interactions(p, args.charge_charge_atoms)
            for key, data in charge_data.items():
                DF.at[i, f"{key}"] = ','.join(data)

        results[i] = DF.copy()

        if args.dump is True:
            for j, (idx, row) in enumerate(_df.sort_values("plddt", ascending=False).iloc[:5].iterrows()):
                _p = model_poses[row.model_idx - 1]
                pose2 = ref_pose.clone()
                for rn in range(1, _p.size() + 1):
                    res = _p.residue(rn)
                    orig_resno = _p.pdb_info().number(res.seqpos())
                    if res.is_ligand() and not pose2.residue(orig_resno).is_ligand():
                        orig_resno = pose2.size()
                        assert pose2.residue(orig_resno).name3() == res.name3(), "Something is wrong with ligand residue numbering?"

                    if res.name3() == "HIS" and res.name() != ref_pose.residue(orig_resno).name():
                        mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
                        mutres.set_res_name(ref_pose.residue(orig_resno).name())
                        mutres.set_target(rn)
                        mutres.apply(_p)
                        res = _p.residue(rn)
                    if res.natoms() != pose2.residue(orig_resno).natoms():
                        continue

                    for n in range(1, res.natoms() + 1):
                        if res.atom_name(n).strip() in ["1H", "2H", "3H", "OXT", "CAV"] and res.is_protein():
                            continue
                        pose2.residue(orig_resno).set_xyz(res.atom_name(n), res.xyz(n))
                sfx(pose2)
                for k in _df.keys():
                    pose2.scores[k] = row[k]
                for jj, cr in enumerate(catalytic_residues[d]):
                    pose2.scores[f"rmsd_catres{jj}"] = catalytic_residue_rmsds[d].at[row.model_idx - 1, cr]
                    pose2.scores[f"u_catres{jj}"] = model_res_scores[d].at[row.model_idx, cr]
                out_path = os.path.join(dump_dir, f"{d}_chemnet_{j}.pdb")
                pose2.dump_pdb(out_path)
                print(f"Dumped {out_path} with pLDDT: {row['plddt']} and RMSD: {row['rmsd_new']}")

        frame_results[d] = _df.copy()

pool = multiprocessing.Pool(processes=args.nproc,
                            initializer=process,
                            initargs=(the_queue,))
# None to end each process
for _i in range(args.nproc):
    the_queue.put(None)

# Closing the queue and the pool
the_queue.close()
the_queue.join_thread()
pool.close()
pool.join()

DF = pd.DataFrame()
for i in results.keys():
    for k in results[i].keys():
        DF.at[i, k] = results[i].at[i, k]

DF.to_csv(args.scorefile_out)

for d, df in frame_results.items():
    frame_csv_path = os.path.join(args.outdir, f"{d}_scores.csv")
    df.to_csv(frame_csv_path, index=False)
    print(f"Frame-by-frame scores saved to {frame_csv_path}")

#scoring_utils.dump_scorefile(DF, args.scorefile_out)
