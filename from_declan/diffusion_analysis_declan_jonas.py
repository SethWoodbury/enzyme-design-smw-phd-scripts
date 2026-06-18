#!/home/jonaswil/miniforge3/envs/jw_pyrosetta/bin/python -u

import os
import sys
import re
import time
import argparse
import numpy as np
import pandas as pd
import pyrosetta as pyr
import pyrosetta.rosetta
from pyrosetta.rosetta.core.select import residue_selector
from pyrosetta.rosetta.core.pack.task import operation

def list_pdb_files(path):
    '''List all files that end with ".pdb" in a directory'''
    files = [i for i in os.listdir(path) if i.endswith('.pdb')]
    files.sort()
    return files


def getSASA(pose, resno=None, SASA_atoms=None, ignore_sc=False):
    '''
    Takes in a pose and calculates its SASA.
    Or calculates SASA of a given residue.
    Or calculates SASA of specified atoms in a given residue.

    Procedure by Brian Coventry
    '''

    atoms = pyr.rosetta.core.id.AtomID_Map_bool_t()
    atoms.resize(pose.size())

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
        res_surf = 0.0
        for i in range(1, pose.residue(resno).natoms()+1):
            if SASA_atoms is not None and i not in SASA_atoms:
                continue
            res_surf += surf_vol.surf(resno, i)
        return res_surf
    else:
        return surf_vol


def get_atom_wise_SASA(pose, resno, probe_radius=1.4):
    '''
    Takes in a pose and a residue number and computes
    SASA for each individual atom in the residue

    Procedure adapted from Brian Coventry
    '''

    atoms = pyr.rosetta.core.id.AtomID_Map_bool_t()
    atoms.resize(pose.size())

    for i, res in enumerate(pose.residues):
        atoms.resize(i+1, res.natoms(), True)

    surf_vol = pyr.rosetta.core.scoring.packing.get_surf_vol(pose, atoms, probe_radius)

    residue = pose.residue(resno)
    res_surf = {}
    for i in range(1, residue.natoms()+1):
        SASA = surf_vol.surf(resno, i)
        SASA = round(SASA, 6)
        res_surf[residue.atom_name(i).strip()] = SASA

    return res_surf


def diffusion_pdb_to_pose(pdb_file):
    pdb_lines = open(pdb_file, 'r').readlines()

    # Remove CB atoms and change residue definition to GLY for all ALA residues
    pdb_lines_gly = []
    for line in pdb_lines:
        if line.startswith('ATOM'):
            if 'ALA' in line:
                if 'CB' not in line:
                    line_gly = line
                    line_gly = line_gly.replace('ALA', 'GLY')
                    pdb_lines_gly.append(line_gly)
            else:
                pdb_lines_gly.append(line)
    
    # Get residue numbers of non ALA residues
    non_ala_resno = set()
    for line in pdb_lines:
        if line.startswith('ATOM'):
            if 'ALA' not in line:
                resno = int(line[23:26])
                non_ala_resno.add(resno)
    
    # Get ligand PBD lines
    pdb_lines_ligand = []
    for line in pdb_lines:
        if line.startswith('HETATM'):
            pdb_lines_ligand.append(line)
    
    pdb_string = pdb_lines_gly + pdb_lines_ligand
    pose = pyrosetta.Pose()
    pyr.rosetta.core.import_pose.pose_from_pdbstring(pose, ''.join(pdb_string))
    non_ala_resno = list(non_ala_resno)
    return pose, non_ala_resno
    

def pack_around_ligand(pose, ligand_seqpos, cat_seqpos, distance=10):
    # Selectors for ligand and catalytic residues
    ligand_selector = residue_selector.ResidueIndexSelector(str(ligand_seqpos))
    catres_selector = residue_selector.ResidueIndexSelector(','.join([str(i) for i in cat_seqpos]))
    not_catres_selector = residue_selector.NotResidueSelector(catres_selector)

    # Select neighborhood of ligand
    nbr_selector = residue_selector.NeighborhoodResidueSelector()
    nbr_selector.set_distance(distance)
    nbr_selector.set_focus_selector(ligand_selector)
    nbr_selector.set_include_focus_in_subset(False)

    # Exclude catatlytic residues from neighborhood
    not_catres_and_nbr_selector = residue_selector.AndResidueSelector()
    not_catres_and_nbr_selector.add_residue_selector(nbr_selector)
    not_catres_and_nbr_selector.add_residue_selector(not_catres_selector)

    # Set up task factory
    task_factory = pyr.rosetta.core.pack.task.TaskFactory()
    task_factory.push_back(operation.InitializeFromCommandline())
    task_factory.push_back(operation.IncludeCurrent())
    prevent_repacking_RLT = operation.PreventRepackingRLT()
    prevent_repacking = operation.OperateOnResidueSubset(prevent_repacking_RLT, not_catres_and_nbr_selector, True)
    task_factory.push_back(prevent_repacking)
    # extra_rotamers = operation.ExtraRotamersGeneric()
    # extra_rotamers.ex1(True)
    # extra_rotamers.ex2(True)
    # task_factory.push_back(extra_rotamers)

    # Initialize and run the packer
    packer = pyr.rosetta.protocols.minimization_packing.PackRotamersMover()
    packer.task_factory(task_factory)
    packer.apply(pose)

    return pose


def get_ROG(pose):
    centroid = np.array([np.average([res.xyz("CA").__getattribute__(c) for res in pose.residues if res.is_protein()]) for c in "xyz"])
    ROG = max([np.linalg.norm(centroid - res.xyz("CA")) for res in pose.residues if res.is_protein()])
    return ROG


def get_max_CA_dist(pose):
    dists = []
    for n in range(1, pose.size()):
        if pose.residue(n).is_ligand():
            continue
        if pose.residue(n+1).is_ligand():
            continue
        if pose.chain(n) != pose.chain(n+1):
            continue
        dists.append((pose.residue(n).xyz("CA") - pose.residue(n+1).xyz("CA")).norm())
    
    return max(dists)


def analyse_sec_struct(pose):
    dssp = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose)
    sec_struct = dssp.get_dssp_secstruct()
    loop_frac = sec_struct.count("L") / pose.size()
    longest_helix = len(max(re.findall(r'H+', sec_struct), key=len, default=''))
    return loop_frac, longest_helix, sec_struct


def get_ligand_fa_rep(pose, ligand_seqpos, cat_seqpos):
    # Score the structure
    scorefxn = pyrosetta.get_fa_scorefxn()
    scorefxn(pose)

    pairwise_fa_rep = pyr.toolbox.atom_pair_energy._reisude_pair_energies(
        res=ligand_seqpos, 
        pose=pose, 
        sfxn=scorefxn, 
        score_type=pyrosetta.rosetta.core.scoring.ScoreType.fa_rep,
        threshold=0.0)
    
    # Generate dict and omit interactions with catalytic residues
    fa_rep_dict = {}
    ligand_fa_rep = 0
    for tuple in pairwise_fa_rep:
        resn = tuple[0]
        energy = tuple[1]
        fa_rep_dict[resn] = energy
        if resn is not ligand_seqpos and resn not in cat_seqpos:
            ligand_fa_rep += energy
    ligand_fa_rep_df = pd.DataFrame.from_dict(fa_rep_dict, orient='index', columns = ['fa_rep'])

    return ligand_fa_rep, ligand_fa_rep_df


def min_cart_cst(pose, atom_names_unfreeze=['CB', 'CG', 'CD'], residue_number=1):
    temp_pose = pose.clone()
    movemap = pyr.MoveMap()
    movemap.set_bb(False)
    movemap.set_chi(False)
    movemap.set_jump(False)
    for atom_name in atom_names_unfreeze:
        atom_id = pose.residue(residue_number).atom_index(atom_name)
        movemap.set_atom(pyr.rosetta.core.id.AtomID(atom_id, residue_number), True)

        ## Iterate over all atoms in the residue to find hydrogens bonded to the selected atom
        #for atom in range(1, temp_pose.residue(residue_number).natoms() + 1):
        #    if temp_pose.residue(residue_number).atom_is_hydrogen(atom):
        #        bonded_neighbors = temp_pose.residue(residue_number).bonded_neighbor(atom)
        #        for bonded_atom in bonded_neighbors:
        #            if bonded_atom == atom_id:
        #                movemap.set_atom(pyr.rosetta.core.id.AtomID(atom, residue_number), True)
        #                print(f"Atom {temp_pose.residue(residue_number).atom_name(atom).strip()} bonded to {temp_pose.residue(residue_number).atom_name(atom_id).strip()} is set as movable.")

    # Iterate over all atoms in the residue to find hydrogens and make them movable
    for atom in range(1, temp_pose.residue(residue_number).natoms() + 1):
        if temp_pose.residue(residue_number).atom_is_hydrogen(atom):
            movemap.set_atom(pyr.rosetta.core.id.AtomID(atom, residue_number), True)
            print(f"Atom {temp_pose.residue(residue_number).atom_name(atom).strip()} is set as movable.")

    cart_scorefxn = pyr.create_score_function('ref2015_cart.wts')
    min_mover = pyr.rosetta.protocols.minimization_packing.MinMover()
    min_mover.score_function(cart_scorefxn)
    min_mover.movemap(movemap)
    min_mover.min_type('lbfgs_armijo')
    min_mover.cartesian(True)
    min_mover.max_iter(1000)
    min_mover.tolerance(0.0001)
    min_mover.apply(temp_pose)

    return temp_pose


def get_pyr_score(pose, score_type = 'fa_dun'):
    scorefxn = pyr.get_fa_scorefxn()
    scorefxn.set_weight(pyr.rosetta.core.scoring.cart_bonded, 1.0)
    scorefxn(pose)
    ScoreTypes = scorefxn.get_nonzero_weighted_scoretypes()
    score_dict = {}
    total_score = 0
    for ScoreType in ScoreTypes:
        Score = scorefxn.score_by_scoretype(pose, ScoreType)
        ScoreType_str = pyr.rosetta.core.scoring.name_from_score_type(ScoreType)
        score_dict.update({
            ScoreType_str: Score
        })
        total_score += Score
    
    return score_dict[score_type]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diff_dir", type=str, required=True, 
                        help="Diffusion output directory with pdb files to analyze")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Output directory for this analysis script")
    parser.add_argument("--buried_atoms", type=str, nargs="+", required=True,
                        help="List of atom names in the ligand that are supposed to be buried")
    parser.add_argument("--exposed_atoms", type=str, nargs="+", required=True,
                        help="List of atom names in the ligand that are supposed to be exposed")
    parser.add_argument("--packing_radius", type=float, default=16,
                        help="Neighborhood radius around ligand that should be packed")
    parser.add_argument("--params_path", type=str, required=True,
                        help="Path to search for params files")
    parser.add_argument("--params_ext", type=str, required=True,
                        help="File extension of the params files")
    parser.add_argument("--slurm_array_job", action="store_true", default=False, 
                        help="(default = False) Set flag if the script is running as a slurm array job. \
                            Files to analyse will be automatically distributed between the array tasks.")
    parser.add_argument("--output_atom_wise_SASA", action="store_true", default=False, 
                        help="(default = False) Set flag to output SASA information for every ligand atom in the scorefile.")
    parser.add_argument("--extra_probe_radii_atom_wise_SASA", type=float, nargs="+", 
                        help="(default = None) Set additional probe radii for atoms wise SASA calculation. Use max one decimal.")
    parser.add_argument("--min_sidechain", action="store_true", default=False, 
                        help="Perform pyrosetta minimization of diffused side chains (from tip atoms). \
                            If this option is used --atoms_to_unfreeze has to be set.")
    parser.add_argument("--atoms_to_unfreeze", type=str, nargs="+", 
                        help="Atom names of atoms that should be movable during sidechain minimization. \
                            All Hydrogens are automatically unfreezed.")
    parser.add_argument("--dump_packed_pdb", action="store_true", default=False, 
                        help="Dumps the structure with packed sidechains around the ligand as pdb file. \
                            Requires --dump_path_packed_pdb to be set.")
    parser.add_argument("--dump_path_packed_pdb", type=str, nargs="?", 
                        help="Specifies path to dump packed pdbs to.")
    parser.add_argument("--dump_minimized_pdb", action="store_true", default=False, 
                        help="Dumps the structure with minimized sidechains (from tip atoms). \
                            Requires --dump_path to be set.")
    parser.add_argument("--dump_path_minimized_pdb", type=str, nargs="?", 
                        help="Specifies path to dump minimized pdbs to.")

    args = parser.parse_args()        

    diffusion_output_dir = args.diff_dir
    analysis_output_dir = args.output_dir
    params_path = args.params_path
    os.makedirs(analysis_output_dir, exist_ok=True)

    buried_atoms = args.buried_atoms
    exposed_atoms = args.exposed_atoms

    pdb_files = list_pdb_files(diffusion_output_dir)
    params = [i for i in os.listdir(params_path) if i.endswith(args.params_ext)]

    if args.slurm_array_job:
        num_tasks = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        num_files = len(pdb_files)

        files_per_task = num_files // num_tasks
        extra_files = num_files % num_tasks

        if task_id <= extra_files:
            start_file = (task_id - 1) * (files_per_task + 1) + 1
            end_file = start_file + files_per_task
        else:
            start_file = extra_files * (files_per_task + 1) + 1 + (task_id - 1 - extra_files) * files_per_task
            end_file = start_file + files_per_task - 1

        pdb_files = pdb_files[start_file-1:end_file]

        print(f'########## Task {task_id} of {num_tasks} total tasks')
        print(f'########## Analyzing pdb files {start_file} to {end_file}')

    DAB = '/net/software/lab/scripts/enzyme_design/DAlphaBall.gcc'
    if not os.path.exists(DAB):
       DAB = None
    assert DAB is not None, 'Please compile DAlphaBall.gcc and manually provide a path to it in this script under the variable `DAB`\n'\
                            'For more info on DAlphaBall, visit: https://www.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Filters/HolesFilter'

    # pdb_files = pdb_files[:1]
    current_conformer = ''

    for pdb_file in pdb_files:
        start_time = time.time()
        conformer_name = pdb_file.split('_')[0]

        if conformer_name != current_conformer:
            pyr.init(f'-extra_res_fa {params_path}/{conformer_name}.{args.params_ext} \
               -dalphaball {DAB} \
               -run:preserve_header \
               -use_input_sc \
               -mute all')
            current_conformer = conformer_name

        pose, cat_seqpos = diffusion_pdb_to_pose(f'{diffusion_output_dir}/{pdb_file}')
        original_pose = pose.clone()

        # Find ligand residue number
        for res in pose.residues:
            if res.is_ligand():
                ligand_seqpos = res.seqpos()

        ligand_pose = pyr.Pose()
        ligand_pose.append_residue_by_jump(pose.residue(ligand_seqpos), 1)

        bb_pose = pose.clone()
        for i in range(1, bb_pose.total_residue() + 1):
            if bb_pose.residue(i).name1() != 'G':
                pyr.toolbox.mutants.mutate_residue(bb_pose, i, 'G')
       
        ROG = get_ROG(original_pose)

        # Calculate cart_bonded for diffused side chains (from tip atoms)
        catres_cart_bonded = 0
        catres_cart_bonded_min = 0
        for catres in cat_seqpos:
            catres_pose = pyr.Pose()
            catres_pose.append_residue_by_bond(original_pose.residue(catres))
            cart_bonded = get_pyr_score(pose=catres_pose, score_type='cart_bonded')
            catres_cart_bonded += cart_bonded

            if args.min_sidechain:
                min_pose = min_cart_cst(pose=original_pose, atom_names_unfreeze=args.atoms_to_unfreeze, residue_number=catres)
                catres_pose_min = pyr.Pose()
                catres_pose_min.append_residue_by_bond(min_pose.residue(catres))
                cart_bonded_min = get_pyr_score(pose=catres_pose_min, score_type='cart_bonded')
                if cart_bonded_min < cart_bonded:
                    catres_cart_bonded_min += cart_bonded_min
                    if args.dump_minimized_pdb:
                        min_pose.dump_pdb(os.path.join(args.dump_path_minimized_pdb, pdb_file))
                else:
                    catres_cart_bonded_min += cart_bonded
                    if args.dump_minimized_pdb:
                        original_pose.dump_pdb(os.path.join(args.dump_path_minimized_pdb, pdb_file))
            else:
                catres_cart_bonded_min = "NA"

        ligand_fa_rep = get_ligand_fa_rep(original_pose, ligand_seqpos, cat_seqpos)[0]
        pose = pack_around_ligand(pose, ligand_seqpos, cat_seqpos, distance=args.packing_radius)

        if args.dump_packed_pdb:
            pose.dump_pdb(os.path.join(args.dump_path_packed_pdb, pdb_file))

        ligand_SASA = getSASA(ligand_pose).tot_surf
        ligand_SASA_unpacked = getSASA(bb_pose, resno=ligand_seqpos)
        ligand_SASA_packed = getSASA(pose, resno=ligand_seqpos)

        probe_radii = [1.4]
        if args.extra_probe_radii_atom_wise_SASA is not None:
            probe_radii = probe_radii + args.extra_probe_radii_atom_wise_SASA

        atom_wise_SASA = {}
        for probe_radius in  probe_radii:
            SASA_dict = {
                'free': get_atom_wise_SASA(ligand_pose, resno=1, probe_radius=probe_radius),
                'unpacked': get_atom_wise_SASA(bb_pose, resno=ligand_seqpos, probe_radius=probe_radius),
                'packed': get_atom_wise_SASA(pose, resno=ligand_seqpos, probe_radius=probe_radius)
            }
            atom_wise_SASA[probe_radius] = SASA_dict
        
        atom_wise_SASA_std_radius = atom_wise_SASA[1.4]

        max_CA_dist = get_max_CA_dist(original_pose)
        loop_frac, longest_helix, sec_struct = analyse_sec_struct(original_pose)

        catres_in_loop = False
        for res in cat_seqpos:
            if sec_struct[res-3:res-1] == "LL" and sec_struct[res:res+2] == "LL":
                catres_in_loop = True

        scores = {
            'SASA_ligand_free': ligand_SASA,
            'SASA_ligand_unpacked': ligand_SASA_unpacked,
            'SASA_ligand_packed': ligand_SASA_packed,
            'SASA_ligand_rel': ligand_SASA_packed / ligand_SASA,
            'ROG': ROG,
            'ligand_fa_rep_clashes': ligand_fa_rep,
            'max_CA_dist': max_CA_dist,
            'loop_frac': loop_frac,
            'longest_helix': longest_helix,
            'catres_in_loop': catres_in_loop,
            'catres_cart_bonded': catres_cart_bonded,
            'catres_cart_bonded_min': catres_cart_bonded_min
        }

        for key in atom_wise_SASA_std_radius:
            buried_atoms_SASA = 0
            for atom_id in buried_atoms:
                buried_atoms_SASA += atom_wise_SASA_std_radius[key][atom_id]

            exposed_atoms_SASA = 0
            for atom_id in exposed_atoms:
                exposed_atoms_SASA += atom_wise_SASA_std_radius[key][atom_id]

            scores.update(
                {'SASA_burried_atoms_'+key: buried_atoms_SASA,
                 'SASA_exposed_atoms_'+key: exposed_atoms_SASA}
            )

        scores.update(
            {'sec_struct': sec_struct,
             'path': f'{diffusion_output_dir}/{pdb_file}'}
        )

        if args.output_atom_wise_SASA:
            for radius in atom_wise_SASA:
                if len(probe_radii) == 1:
                    radius_str = ''
                else:
                    radius_str = f'{radius:.1f}_'
                SASA_dict = atom_wise_SASA[radius]
                for atom_name in SASA_dict['free']:
                    scores.update(
                        {'SASA_free_'+radius_str+atom_name: SASA_dict['free'][atom_name],
                         'SASA_unpacked_'+radius_str+atom_name: SASA_dict['unpacked'][atom_name],
                         'SASA_packed_'+radius_str+atom_name: SASA_dict['packed'][atom_name]}
                    )

        for key in scores:
            if isinstance(scores[key], float):
                scores[key] = round(scores[key], 6)

        scores = {pdb_file: scores}
        # all_scores.update(scores)

        scoresDF = pd.DataFrame.from_dict(scores, orient='index')

        if args.slurm_array_job:
            os.makedirs(f'{analysis_output_dir}/scores', exist_ok=True)
            scorefile_path = f'{analysis_output_dir}/scores/scores_{task_id}.csv'
        else:
            scorefile_path = f'{analysis_output_dir}/scores.csv'

        write_header = not os.path.exists(scorefile_path)

        with open(scorefile_path, 'a') as file:
            # portalocker.lock(file, portalocker.LOCK_EX)
            scoresDF.to_csv(file, index_label='pdb_file', header=write_header)
            # portalocker.unlock(file)
        
        print(f'--- analysis of {pdb_file} took %s seconds ---' % round(time.time() - start_time, 2))
