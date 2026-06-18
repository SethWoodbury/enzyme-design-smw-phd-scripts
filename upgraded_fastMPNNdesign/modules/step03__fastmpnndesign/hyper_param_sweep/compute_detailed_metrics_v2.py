#!/usr/bin/env python3
"""
Compute detailed metrics for all sweep outputs using PyRosetta.
Version 2 - More robust with better error handling and diagnostics.
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

SCRIPT_DIR = Path(__file__).parent
MODULE_DIR = SCRIPT_DIR.parent

def init_pyrosetta():
    """Initialize PyRosetta."""
    import pyrosetta
    pyrosetta.init("-mute all -ignore_unrecognized_res true -ignore_zero_occupancy false")
    return pyrosetta

def load_pose_with_params(pdb_path, params_files):
    """Load PDB with params files."""
    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose
    from pyrosetta.rosetta.core.import_pose import pose_from_file

    existing_params = [pf for pf in params_files if os.path.exists(pf)]
    if existing_params:
        rts = pyrosetta.generate_nonstandard_residue_set(Pose(), existing_params)
        pose = Pose()
        pose_from_file(pose, rts, str(pdb_path))
        return pose
    return pyrosetta.pose_from_pdb(str(pdb_path))

def get_atom_coords_by_name(pose, chain, resno, atom_names):
    """Get coordinates for specific atoms."""
    coords = []
    pdb_info = pose.pdb_info()

    for i in range(1, pose.total_residue() + 1):
        if pdb_info.chain(i) == chain and pdb_info.number(i) == resno:
            res = pose.residue(i)
            for atom_name in atom_names:
                if res.has(atom_name):
                    xyz = res.xyz(atom_name)
                    coords.append([xyz.x, xyz.y, xyz.z])
            break
    return np.array(coords) if coords else None

def get_ligand_heavy_atoms(pose):
    """Get all ligand heavy atom coordinates."""
    coords = []
    atom_names = []

    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if res.is_ligand():
            for j in range(1, res.natoms() + 1):
                if not res.atom_is_hydrogen(j):
                    xyz = res.xyz(j)
                    coords.append([xyz.x, xyz.y, xyz.z])
                    atom_names.append(res.atom_name(j).strip())
    return np.array(coords) if coords else None, atom_names

def get_all_ca_coords(pose):
    """Get all CA coordinates."""
    coords = []
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if res.is_protein() and res.has("CA"):
            xyz = res.xyz("CA")
            coords.append([xyz.x, xyz.y, xyz.z])
    return np.array(coords) if coords else None

def get_all_backbone_coords(pose):
    """Get all backbone (N, CA, C, O) coordinates."""
    coords = []
    bb_atoms = ["N", "CA", "C", "O"]
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if res.is_protein():
            for atom in bb_atoms:
                if res.has(atom):
                    xyz = res.xyz(atom)
                    coords.append([xyz.x, xyz.y, xyz.z])
    return np.array(coords) if coords else None

def get_catres_heavy_atoms(pose, catres_list):
    """Get all heavy atom coords for catres."""
    coords = []
    pdb_info = pose.pdb_info()

    for chain_resno in catres_list:
        chain = chain_resno[0]
        resno = int(chain_resno[1:])

        for i in range(1, pose.total_residue() + 1):
            if pdb_info.chain(i) == chain and pdb_info.number(i) == resno:
                res = pose.residue(i)
                for j in range(1, res.natoms() + 1):
                    if not res.atom_is_hydrogen(j):
                        xyz = res.xyz(j)
                        coords.append([xyz.x, xyz.y, xyz.z])
                break
    return np.array(coords) if coords else None

def compute_rmsd(coords1, coords2):
    """Compute RMSD."""
    if coords1 is None or coords2 is None:
        return None
    if len(coords1) != len(coords2):
        return None
    if len(coords1) == 0:
        return None
    diff = coords1 - coords2
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

def get_bond_geometry_deviations(pose, catres_set=None):
    """
    Get bond length and angle deviations using Rosetta's internal checks.
    """
    from pyrosetta.rosetta.core.scoring import ScoreFunction, ScoreType
    from pyrosetta.rosetta.core.scoring.methods import EnergyMethodOptions

    pdb_info = pose.pdb_info()

    # Collect per-residue bond info
    bond_len_devs_all = []
    bond_len_devs_unconstrained = []
    bond_len_devs_catres = []

    bond_ang_devs_all = []
    bond_ang_devs_unconstrained = []
    bond_ang_devs_catres = []

    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if not res.is_protein():
            continue

        chain_resno = f"{pdb_info.chain(i)}{pdb_info.number(i)}"
        is_catres = catres_set and chain_resno in catres_set

        res_type = res.type()

        # Bond lengths
        for j in range(1, res.natoms() + 1):
            for k in res.bonded_neighbor(j):
                if k > j:
                    # Actual length
                    xyz_j = res.xyz(j)
                    xyz_k = res.xyz(k)
                    actual = np.sqrt((xyz_j.x - xyz_k.x)**2 +
                                    (xyz_j.y - xyz_k.y)**2 +
                                    (xyz_j.z - xyz_k.z)**2)

                    # Ideal from residue type
                    try:
                        ideal = res_type.bond_length(j, k)
                        if ideal > 0.5 and ideal < 3.0:  # Sanity check
                            dev = abs(actual - ideal)
                            bond_len_devs_all.append(dev)
                            if is_catres:
                                bond_len_devs_catres.append(dev)
                            else:
                                bond_len_devs_unconstrained.append(dev)
                    except:
                        pass

        # Bond angles
        for j in range(1, res.natoms() + 1):
            neighbors = list(res.bonded_neighbor(j))
            for idx1 in range(len(neighbors)):
                for idx2 in range(idx1 + 1, len(neighbors)):
                    k1, k2 = neighbors[idx1], neighbors[idx2]

                    xyz_1 = res.xyz(k1)
                    xyz_j = res.xyz(j)
                    xyz_2 = res.xyz(k2)

                    v1 = np.array([xyz_1.x - xyz_j.x, xyz_1.y - xyz_j.y, xyz_1.z - xyz_j.z])
                    v2 = np.array([xyz_2.x - xyz_j.x, xyz_2.y - xyz_j.y, xyz_2.z - xyz_j.z])

                    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if norm1 > 0.01 and norm2 > 0.01:
                        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                        cos_angle = np.clip(cos_angle, -1, 1)
                        actual_deg = np.degrees(np.arccos(cos_angle))

                        try:
                            ideal_rad = res_type.bond_angle(k1, j, k2)
                            if ideal_rad > 0:
                                ideal_deg = np.degrees(ideal_rad)
                                dev = abs(actual_deg - ideal_deg)
                                bond_ang_devs_all.append(dev)
                                if is_catres:
                                    bond_ang_devs_catres.append(dev)
                                else:
                                    bond_ang_devs_unconstrained.append(dev)
                        except:
                            pass

    return {
        'bond_length': {
            'all': {'mean': np.mean(bond_len_devs_all) if bond_len_devs_all else 0,
                   'max': np.max(bond_len_devs_all) if bond_len_devs_all else 0,
                   'count': len(bond_len_devs_all)},
            'unconstrained': {'mean': np.mean(bond_len_devs_unconstrained) if bond_len_devs_unconstrained else 0,
                             'max': np.max(bond_len_devs_unconstrained) if bond_len_devs_unconstrained else 0,
                             'count': len(bond_len_devs_unconstrained)},
            'catres': {'mean': np.mean(bond_len_devs_catres) if bond_len_devs_catres else 0,
                      'max': np.max(bond_len_devs_catres) if bond_len_devs_catres else 0,
                      'count': len(bond_len_devs_catres)},
        },
        'bond_angle': {
            'all': {'mean': np.mean(bond_ang_devs_all) if bond_ang_devs_all else 0,
                   'max': np.max(bond_ang_devs_all) if bond_ang_devs_all else 0,
                   'count': len(bond_ang_devs_all)},
            'unconstrained': {'mean': np.mean(bond_ang_devs_unconstrained) if bond_ang_devs_unconstrained else 0,
                             'max': np.max(bond_ang_devs_unconstrained) if bond_ang_devs_unconstrained else 0,
                             'count': len(bond_ang_devs_unconstrained)},
            'catres': {'mean': np.mean(bond_ang_devs_catres) if bond_ang_devs_catres else 0,
                      'max': np.max(bond_ang_devs_catres) if bond_ang_devs_catres else 0,
                      'count': len(bond_ang_devs_catres)},
        }
    }

def compute_design_metrics(design_pdb, step02_pdb, step01_pdb, catres_list, params_files):
    """Compute all metrics for a single design."""
    metrics = {'diagnostics': {}}

    try:
        # Load poses
        design_pose = load_pose_with_params(design_pdb, params_files)
        step02_pose = load_pose_with_params(step02_pdb, params_files)

        metrics['diagnostics']['design_residues'] = design_pose.total_residue()
        metrics['diagnostics']['step02_residues'] = step02_pose.total_residue()

        step01_pose = None
        if step01_pdb and os.path.exists(step01_pdb):
            step01_pose = load_pose_with_params(step01_pdb, params_files)
            metrics['diagnostics']['step01_residues'] = step01_pose.total_residue()

        # Ligand RMSD
        design_lig, design_lig_names = get_ligand_heavy_atoms(design_pose)
        step02_lig, step02_lig_names = get_ligand_heavy_atoms(step02_pose)

        metrics['diagnostics']['design_ligand_atoms'] = len(design_lig) if design_lig is not None else 0
        metrics['diagnostics']['step02_ligand_atoms'] = len(step02_lig) if step02_lig is not None else 0

        if design_lig is not None and step02_lig is not None and len(design_lig) == len(step02_lig):
            metrics['ligand_rmsd'] = compute_rmsd(design_lig, step02_lig)
        else:
            metrics['ligand_rmsd'] = None
            metrics['diagnostics']['ligand_mismatch'] = True

        # Catres heavy atom RMSD
        catres_set = set(catres_list)
        design_catres = get_catres_heavy_atoms(design_pose, catres_list)
        step02_catres = get_catres_heavy_atoms(step02_pose, catres_list)

        metrics['diagnostics']['design_catres_atoms'] = len(design_catres) if design_catres is not None else 0
        metrics['diagnostics']['step02_catres_atoms'] = len(step02_catres) if step02_catres is not None else 0

        metrics['catres_heavy_rmsd'] = compute_rmsd(design_catres, step02_catres)

        # CA RMSD vs step02
        design_ca = get_all_ca_coords(design_pose)
        step02_ca = get_all_ca_coords(step02_pose)
        metrics['ca_rmsd_vs_step02'] = compute_rmsd(design_ca, step02_ca)

        # Backbone RMSD vs step02
        design_bb = get_all_backbone_coords(design_pose)
        step02_bb = get_all_backbone_coords(step02_pose)
        metrics['backbone_rmsd_vs_step02'] = compute_rmsd(design_bb, step02_bb)

        # RMSD vs step01
        if step01_pose:
            step01_ca = get_all_ca_coords(step01_pose)
            step01_bb = get_all_backbone_coords(step01_pose)
            metrics['ca_rmsd_vs_step01'] = compute_rmsd(design_ca, step01_ca)
            metrics['backbone_rmsd_vs_step01'] = compute_rmsd(design_bb, step01_bb)
        else:
            metrics['ca_rmsd_vs_step01'] = None
            metrics['backbone_rmsd_vs_step01'] = None

        # Bond geometry
        geom = get_bond_geometry_deviations(design_pose, catres_set)

        metrics['bond_length_all_mean'] = geom['bond_length']['all']['mean']
        metrics['bond_length_all_max'] = geom['bond_length']['all']['max']
        metrics['bond_length_unconstrained_mean'] = geom['bond_length']['unconstrained']['mean']
        metrics['bond_length_unconstrained_max'] = geom['bond_length']['unconstrained']['max']
        metrics['bond_length_catres_mean'] = geom['bond_length']['catres']['mean']
        metrics['bond_length_catres_max'] = geom['bond_length']['catres']['max']

        metrics['bond_angle_all_mean'] = geom['bond_angle']['all']['mean']
        metrics['bond_angle_all_max'] = geom['bond_angle']['all']['max']
        metrics['bond_angle_unconstrained_mean'] = geom['bond_angle']['unconstrained']['mean']
        metrics['bond_angle_unconstrained_max'] = geom['bond_angle']['unconstrained']['max']
        metrics['bond_angle_catres_mean'] = geom['bond_angle']['catres']['mean']
        metrics['bond_angle_catres_max'] = geom['bond_angle']['catres']['max']

        metrics['diagnostics']['bond_count'] = geom['bond_length']['all']['count']
        metrics['diagnostics']['angle_count'] = geom['bond_angle']['all']['count']

    except Exception as e:
        metrics['error'] = str(e)
        import traceback
        metrics['traceback'] = traceback.format_exc()

    return metrics

def process_job(job_dir, params_files):
    """Process a single job directory."""
    results_file = job_dir / "fastmpnn_design_results.json"
    if not results_file.exists():
        return None

    with open(results_file) as f:
        results = json.load(f)

    if not results.get('output_designs'):
        return None

    step02_pdb = results['metadata'].get('step02_pdb')
    step01_pdb = results['metadata'].get('step01_pdb')
    catres_list = results['residue_classification'].get('fixed_residues', [])

    job_result = {
        'job_name': job_dir.name,
        'protocol': results['metadata'].get('protocol'),
        'runtime': results['metadata'].get('runtime_seconds'),
        'num_designs': len(results['output_designs']),
        'catres_count': len(catres_list),
        'designs': []
    }

    for design in results['output_designs']:
        design_pdb = design.get('pdb_path')
        if design_pdb and os.path.exists(design_pdb):
            metrics = compute_design_metrics(design_pdb, step02_pdb, step01_pdb, catres_list, params_files)

            # Add sequence info
            seq_metrics = design.get('metrics', {}).get('sequence_metrics', {})
            metrics['sequence'] = design.get('sequence')
            metrics['num_mutations'] = seq_metrics.get('num_mutations')
            metrics['mutations'] = seq_metrics.get('mutations', [])
            metrics['sequence_identity'] = seq_metrics.get('sequence_identity_vs_step02')

            job_result['designs'].append(metrics)

    return job_result

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='detailed_metrics_v2.json')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    print("Initializing PyRosetta...")
    init_pyrosetta()

    outputs_dir = SCRIPT_DIR / "outputs"
    job_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith('job')])

    if args.limit > 0:
        job_dirs = job_dirs[:args.limit]

    print(f"Found {len(job_dirs)} job directories")

    params_files = [str(MODULE_DIR / "test/params/XDW.params")]

    all_results = []
    for i, job_dir in enumerate(job_dirs):
        print(f"Processing {i+1}/{len(job_dirs)}: {job_dir.name}")
        try:
            result = process_job(job_dir, params_files)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  Error: {e}")

    output_path = SCRIPT_DIR / args.output
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"Processed {len(all_results)} jobs")

if __name__ == '__main__':
    main()
