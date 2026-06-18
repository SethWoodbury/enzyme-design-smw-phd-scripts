#!/usr/bin/env python3
"""
Compute detailed metrics for all sweep outputs using PyRosetta.

This script should be run inside the universal container:
    apptainer exec /net/software/containers/universal.sif python compute_detailed_metrics.py

Computes:
- Ligand RMSD (should be ~0.00)
- Constrained catres atom RMSD (should be ~0.00)
- CA RMSD vs step01 and step02
- Backbone RMSD vs step01 and step02
- Bond length deviations (mean, max) - overall and unconstrained
- Bond angle deviations (mean, max) - overall and unconstrained
- Per-catres metrics
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

# Add parent modules to path
SCRIPT_DIR = Path(__file__).parent
MODULE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MODULE_DIR.parent.parent))

def init_pyrosetta():
    """Initialize PyRosetta with standard options."""
    import pyrosetta
    pyrosetta.init("-mute all -ignore_unrecognized_res true")
    return pyrosetta

def load_pose(pdb_path, params_files=None):
    """Load a PDB file into a pose."""
    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose
    from pyrosetta.rosetta.core.import_pose import pose_from_file

    if params_files:
        # Load with params files
        existing_params = [pf for pf in params_files if os.path.exists(pf)]
        if existing_params:
            # Generate residue type set with params
            rts = pyrosetta.generate_nonstandard_residue_set(Pose(), existing_params)
            pose = Pose()
            pose_from_file(pose, rts, str(pdb_path))
            return pose

    return pyrosetta.pose_from_pdb(str(pdb_path))

def get_ligand_atoms(pose):
    """Get ligand heavy atom coordinates."""
    from pyrosetta.rosetta.core.chemical import ResidueProperty

    coords = []
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if res.is_ligand():
            for j in range(1, res.natoms() + 1):
                if not res.atom_is_hydrogen(j):
                    xyz = res.xyz(j)
                    coords.append(np.array([xyz.x, xyz.y, xyz.z]))
    return np.array(coords) if coords else None

def get_constrained_atom_coords(pose, catres_positions):
    """Get coordinates of constrained atoms (catres heavy atoms)."""
    coords = []
    for chain_resno in catres_positions:
        # Parse chain and resno (e.g., "A13" -> chain A, resno 13)
        chain = chain_resno[0]
        resno = int(chain_resno[1:])

        # Find pose index
        pdb_info = pose.pdb_info()
        for i in range(1, pose.total_residue() + 1):
            if pdb_info.chain(i) == chain and pdb_info.number(i) == resno:
                res = pose.residue(i)
                for j in range(1, res.natoms() + 1):
                    if not res.atom_is_hydrogen(j):
                        xyz = res.xyz(j)
                        coords.append(np.array([xyz.x, xyz.y, xyz.z]))
                break
    return np.array(coords) if coords else None

def get_ca_coords(pose, residue_subset=None):
    """Get CA coordinates for all protein residues."""
    coords = []
    indices = []
    pdb_info = pose.pdb_info()

    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if res.is_protein() and res.has("CA"):
            if residue_subset is None:
                coords.append(np.array([res.xyz("CA").x, res.xyz("CA").y, res.xyz("CA").z]))
                indices.append(i)
            else:
                chain_resno = f"{pdb_info.chain(i)}{pdb_info.number(i)}"
                if chain_resno in residue_subset:
                    coords.append(np.array([res.xyz("CA").x, res.xyz("CA").y, res.xyz("CA").z]))
                    indices.append(i)

    return np.array(coords), indices

def get_backbone_coords(pose, residue_subset=None):
    """Get backbone (N, CA, C, O) coordinates for all protein residues."""
    coords = []
    pdb_info = pose.pdb_info()
    bb_atoms = ["N", "CA", "C", "O"]

    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if res.is_protein():
            include = True
            if residue_subset is not None:
                chain_resno = f"{pdb_info.chain(i)}{pdb_info.number(i)}"
                include = chain_resno in residue_subset

            if include:
                for atom in bb_atoms:
                    if res.has(atom):
                        xyz = res.xyz(atom)
                        coords.append(np.array([xyz.x, xyz.y, xyz.z]))

    return np.array(coords) if coords else None

def compute_rmsd(coords1, coords2):
    """Compute RMSD between two sets of coordinates."""
    if coords1 is None or coords2 is None:
        return None
    if len(coords1) != len(coords2):
        return None
    if len(coords1) == 0:
        return None

    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

def get_bond_length_deviations(pose, unconstrained_residues=None):
    """
    Compute bond length deviations from ideal.
    Returns deviations for all bonds and for unconstrained atoms only.
    """
    from pyrosetta.rosetta.core.scoring import ScoreFunction
    from pyrosetta.rosetta.core.scoring import cart_bonded

    pdb_info = pose.pdb_info()
    all_devs = []
    unconstrained_devs = []
    catres_devs = []

    # Get ideal bond lengths from residue types
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if not res.is_protein():
            continue

        chain_resno = f"{pdb_info.chain(i)}{pdb_info.number(i)}"
        is_unconstrained = unconstrained_residues is None or chain_resno in unconstrained_residues
        is_catres = unconstrained_residues is not None and chain_resno not in unconstrained_residues

        res_type = res.type()

        # Check all bonds in residue
        for j in range(1, res.natoms() + 1):
            for k in res.bonded_neighbor(j):
                if k > j:  # Avoid double counting
                    # Get actual bond length
                    xyz_j = res.xyz(j)
                    xyz_k = res.xyz(k)
                    actual = np.sqrt((xyz_j.x - xyz_k.x)**2 +
                                    (xyz_j.y - xyz_k.y)**2 +
                                    (xyz_j.z - xyz_k.z)**2)

                    # Get ideal bond length from icoor
                    try:
                        ideal = res_type.bond_length(j, k)
                        if ideal > 0:
                            dev = abs(actual - ideal)
                            all_devs.append(dev)
                            if is_unconstrained:
                                unconstrained_devs.append(dev)
                            if is_catres:
                                catres_devs.append(dev)
                    except:
                        pass

    return {
        'all': {'mean': np.mean(all_devs) if all_devs else 0,
                'max': np.max(all_devs) if all_devs else 0,
                'count': len(all_devs)},
        'unconstrained': {'mean': np.mean(unconstrained_devs) if unconstrained_devs else 0,
                         'max': np.max(unconstrained_devs) if unconstrained_devs else 0,
                         'count': len(unconstrained_devs)},
        'catres': {'mean': np.mean(catres_devs) if catres_devs else 0,
                  'max': np.max(catres_devs) if catres_devs else 0,
                  'count': len(catres_devs)}
    }

def get_bond_angle_deviations(pose, unconstrained_residues=None):
    """
    Compute bond angle deviations from ideal.
    """
    pdb_info = pose.pdb_info()
    all_devs = []
    unconstrained_devs = []
    catres_devs = []

    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if not res.is_protein():
            continue

        chain_resno = f"{pdb_info.chain(i)}{pdb_info.number(i)}"
        is_unconstrained = unconstrained_residues is None or chain_resno in unconstrained_residues
        is_catres = unconstrained_residues is not None and chain_resno not in unconstrained_residues

        res_type = res.type()

        # Check angles (atom triplets where middle atom is bonded to both ends)
        for j in range(1, res.natoms() + 1):
            neighbors = list(res.bonded_neighbor(j))
            for idx1 in range(len(neighbors)):
                for idx2 in range(idx1 + 1, len(neighbors)):
                    k1, k2 = neighbors[idx1], neighbors[idx2]

                    # Compute actual angle
                    xyz_1 = res.xyz(k1)
                    xyz_j = res.xyz(j)
                    xyz_2 = res.xyz(k2)

                    v1 = np.array([xyz_1.x - xyz_j.x, xyz_1.y - xyz_j.y, xyz_1.z - xyz_j.z])
                    v2 = np.array([xyz_2.x - xyz_j.x, xyz_2.y - xyz_j.y, xyz_2.z - xyz_j.z])

                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    actual = np.degrees(np.arccos(cos_angle))

                    # Get ideal angle
                    try:
                        ideal = res_type.bond_angle(k1, j, k2)
                        if ideal > 0:
                            ideal_deg = np.degrees(ideal)
                            dev = abs(actual - ideal_deg)
                            all_devs.append(dev)
                            if is_unconstrained:
                                unconstrained_devs.append(dev)
                            if is_catres:
                                catres_devs.append(dev)
                    except:
                        pass

    return {
        'all': {'mean': np.mean(all_devs) if all_devs else 0,
                'max': np.max(all_devs) if all_devs else 0,
                'count': len(all_devs)},
        'unconstrained': {'mean': np.mean(unconstrained_devs) if unconstrained_devs else 0,
                         'max': np.max(unconstrained_devs) if unconstrained_devs else 0,
                         'count': len(unconstrained_devs)},
        'catres': {'mean': np.mean(catres_devs) if catres_devs else 0,
                  'max': np.max(catres_devs) if catres_devs else 0,
                  'count': len(catres_devs)}
    }

def compute_metrics_for_design(design_pdb, step02_pdb, step01_pdb, catres_positions, params_files):
    """Compute all metrics for a single design."""
    metrics = {}

    try:
        # Load poses
        design_pose = load_pose(design_pdb, params_files)
        step02_pose = load_pose(step02_pdb, params_files)
        step01_pose = load_pose(step01_pdb, params_files) if step01_pdb and os.path.exists(step01_pdb) else None

        # Ligand RMSD
        design_lig = get_ligand_atoms(design_pose)
        step02_lig = get_ligand_atoms(step02_pose)
        metrics['ligand_rmsd'] = compute_rmsd(design_lig, step02_lig)

        # Constrained catres RMSD
        design_catres = get_constrained_atom_coords(design_pose, catres_positions)
        step02_catres = get_constrained_atom_coords(step02_pose, catres_positions)
        metrics['catres_rmsd'] = compute_rmsd(design_catres, step02_catres)

        # CA RMSD vs step02
        design_ca, _ = get_ca_coords(design_pose)
        step02_ca, _ = get_ca_coords(step02_pose)
        metrics['ca_rmsd_vs_step02'] = compute_rmsd(design_ca, step02_ca)

        # Backbone RMSD vs step02
        design_bb = get_backbone_coords(design_pose)
        step02_bb = get_backbone_coords(step02_pose)
        metrics['backbone_rmsd_vs_step02'] = compute_rmsd(design_bb, step02_bb)

        # CA and backbone RMSD vs step01 (if available)
        if step01_pose:
            step01_ca, _ = get_ca_coords(step01_pose)
            step01_bb = get_backbone_coords(step01_pose)
            metrics['ca_rmsd_vs_step01'] = compute_rmsd(design_ca, step01_ca)
            metrics['backbone_rmsd_vs_step01'] = compute_rmsd(design_bb, step01_bb)
        else:
            metrics['ca_rmsd_vs_step01'] = None
            metrics['backbone_rmsd_vs_step01'] = None

        # Catres-only RMSD
        design_catres_ca, _ = get_ca_coords(design_pose, set(catres_positions))
        step02_catres_ca, _ = get_ca_coords(step02_pose, set(catres_positions))
        metrics['catres_ca_rmsd'] = compute_rmsd(design_catres_ca, step02_catres_ca)

        # Unconstrained residues (all except catres)
        all_residues = set()
        pdb_info = design_pose.pdb_info()
        for i in range(1, design_pose.total_residue() + 1):
            if design_pose.residue(i).is_protein():
                all_residues.add(f"{pdb_info.chain(i)}{pdb_info.number(i)}")
        unconstrained = all_residues - set(catres_positions)

        # Bond length deviations
        bond_len = get_bond_length_deviations(design_pose, unconstrained)
        metrics['bond_length_all_mean'] = bond_len['all']['mean']
        metrics['bond_length_all_max'] = bond_len['all']['max']
        metrics['bond_length_unconstrained_mean'] = bond_len['unconstrained']['mean']
        metrics['bond_length_unconstrained_max'] = bond_len['unconstrained']['max']
        metrics['bond_length_catres_mean'] = bond_len['catres']['mean']
        metrics['bond_length_catres_max'] = bond_len['catres']['max']

        # Bond angle deviations
        bond_ang = get_bond_angle_deviations(design_pose, unconstrained)
        metrics['bond_angle_all_mean'] = bond_ang['all']['mean']
        metrics['bond_angle_all_max'] = bond_ang['all']['max']
        metrics['bond_angle_unconstrained_mean'] = bond_ang['unconstrained']['mean']
        metrics['bond_angle_unconstrained_max'] = bond_ang['unconstrained']['max']
        metrics['bond_angle_catres_mean'] = bond_ang['catres']['mean']
        metrics['bond_angle_catres_max'] = bond_ang['catres']['max']

    except Exception as e:
        metrics['error'] = str(e)

    return metrics

def analyze_job(job_dir, params_files):
    """Analyze a single job directory."""
    results_file = job_dir / "fastmpnn_design_results.json"
    if not results_file.exists():
        return None

    with open(results_file) as f:
        results = json.load(f)

    if not results.get('output_designs'):
        return None

    step02_pdb = results['metadata'].get('step02_pdb')
    step01_pdb = results['metadata'].get('step01_pdb')
    catres_positions = results['residue_classification'].get('fixed_residues', [])

    job_metrics = {
        'job_name': job_dir.name,
        'protocol': results['metadata'].get('protocol'),
        'runtime': results['metadata'].get('runtime_seconds'),
        'num_designs': len(results['output_designs']),
        'designs': []
    }

    for design in results['output_designs']:
        design_pdb = design.get('pdb_path')
        if design_pdb and os.path.exists(design_pdb):
            metrics = compute_metrics_for_design(
                design_pdb, step02_pdb, step01_pdb, catres_positions, params_files
            )
            metrics['sequence'] = design.get('sequence')
            metrics['num_mutations'] = design.get('metrics', {}).get('sequence_metrics', {}).get('num_mutations')
            metrics['mutations'] = design.get('metrics', {}).get('sequence_metrics', {}).get('mutations', [])
            job_metrics['designs'].append(metrics)

    return job_metrics

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='detailed_metrics.json')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of jobs to process (0=all)')
    args = parser.parse_args()

    # Initialize PyRosetta
    print("Initializing PyRosetta...")
    pyrosetta = init_pyrosetta()

    # Find all job directories
    outputs_dir = SCRIPT_DIR / "outputs"
    job_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith('job')])

    if args.limit > 0:
        job_dirs = job_dirs[:args.limit]

    print(f"Found {len(job_dirs)} job directories")

    # Params file
    params_files = [str(MODULE_DIR / "test/params/XDW.params")]

    # Process each job
    all_results = []
    for i, job_dir in enumerate(job_dirs):
        print(f"Processing {i+1}/{len(job_dirs)}: {job_dir.name}")
        try:
            metrics = analyze_job(job_dir, params_files)
            if metrics:
                all_results.append(metrics)
        except Exception as e:
            print(f"  Error: {e}")

    # Save results
    output_path = SCRIPT_DIR / args.output
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"Processed {len(all_results)} jobs with designs")

if __name__ == '__main__':
    main()
