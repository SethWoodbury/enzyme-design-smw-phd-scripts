#!/usr/bin/env python3
"""
Quick single-structure relaxation test for validating the new protocol.

Runs a single relaxation and reports detailed geometry metrics.
Can be run directly in the PyRosetta container.

Usage:
    apptainer exec /net/software/containers/pyrosetta.sif python quick_relax_test.py \\
        --pdb input.pdb --params ligand.params --output_dir test_out --config baseline
"""

import json
import sys
import os
import time
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test configurations
CONFIGS = {
    "baseline_new": {
        "use_multistage_relax": True,
        "allow_catres_bb": True,
        "initial_coord_cst_weight": 1000.0,
        "final_coord_cst_weight": 100.0,
        "initial_fa_rep_scale": 0.15,
        "n_relax_stages": 3,
        "fastrelax_cycles": 2,
        "cart_bonded_weight": 0.5,
    },
    "old_protocol": {
        "use_multistage_relax": False,
        "allow_catres_bb": False,
        "coord_cst_weight": 200.0,
        "fastrelax_cycles": 2,
        "cart_bonded_weight": 0.5,
    },
    "high_initial_cst": {
        "use_multistage_relax": True,
        "allow_catres_bb": True,
        "initial_coord_cst_weight": 2000.0,
        "final_coord_cst_weight": 100.0,
        "initial_fa_rep_scale": 0.15,
        "n_relax_stages": 3,
        "fastrelax_cycles": 2,
        "cart_bonded_weight": 0.5,
    },
    "high_cart_bonded": {
        "use_multistage_relax": True,
        "allow_catres_bb": True,
        "initial_coord_cst_weight": 1000.0,
        "final_coord_cst_weight": 100.0,
        "initial_fa_rep_scale": 0.15,
        "n_relax_stages": 3,
        "fastrelax_cycles": 2,
        "cart_bonded_weight": 1.0,
    },
    "low_fa_rep": {
        "use_multistage_relax": True,
        "allow_catres_bb": True,
        "initial_coord_cst_weight": 1000.0,
        "final_coord_cst_weight": 100.0,
        "initial_fa_rep_scale": 0.05,
        "n_relax_stages": 3,
        "fastrelax_cycles": 2,
        "cart_bonded_weight": 0.5,
    },
    "four_stages": {
        "use_multistage_relax": True,
        "allow_catres_bb": True,
        "initial_coord_cst_weight": 1000.0,
        "final_coord_cst_weight": 100.0,
        "initial_fa_rep_scale": 0.15,
        "n_relax_stages": 4,
        "fastrelax_cycles": 2,
        "cart_bonded_weight": 0.5,
    },
    "frozen_bb_failure": {
        "use_multistage_relax": True,
        "allow_catres_bb": False,  # Should fail
        "initial_coord_cst_weight": 1000.0,
        "final_coord_cst_weight": 100.0,
        "initial_fa_rep_scale": 0.15,
        "n_relax_stages": 3,
        "fastrelax_cycles": 2,
        "cart_bonded_weight": 0.5,
    },
}


def compute_bond_deviations_pyrosetta(pose, residue_indices):
    """Compute bond length deviations for specified residues using PyRosetta directly."""
    from pyrosetta.rosetta.core.scoring import ScoreType

    deviations = []

    for res_i in residue_indices:
        if res_i < 1 or res_i > pose.total_residue():
            continue

        residue = pose.residue(res_i)
        resname = residue.name3()

        for atom_i in range(1, residue.natoms() + 1):
            atom_name_i = residue.atom_name(atom_i).strip()

            # Skip hydrogens
            if atom_name_i.startswith('H') or (len(atom_name_i) > 1 and atom_name_i[0].isdigit()):
                continue

            xyz_i = residue.xyz(atom_i)
            bonded = residue.bonded_neighbor(atom_i)

            for atom_j in bonded:
                if atom_j <= atom_i:  # Avoid double counting
                    continue

                atom_name_j = residue.atom_name(atom_j).strip()
                if atom_name_j.startswith('H') or (len(atom_name_j) > 1 and atom_name_j[0].isdigit()):
                    continue

                xyz_j = residue.xyz(atom_j)

                actual = math.sqrt(
                    (xyz_i.x - xyz_j.x)**2 +
                    (xyz_i.y - xyz_j.y)**2 +
                    (xyz_i.z - xyz_j.z)**2
                )

                try:
                    ideal = residue.type().bond_length(atom_i, atom_j)
                    deviation = abs(actual - ideal)
                    deviations.append({
                        'resnum': res_i,
                        'resname': resname,
                        'atom1': atom_name_i,
                        'atom2': atom_name_j,
                        'ideal': ideal,
                        'actual': actual,
                        'deviation': deviation,
                    })
                except Exception:
                    pass

    return deviations


def run_test(pdb_path: Path, params_files: List[Path], output_dir: Path,
             config_name: str, config: Dict) -> Dict:
    """Run a single relaxation test."""
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import ScoreType
    from pyrosetta.rosetta.core.scoring.constraints import CoordinateConstraint as RosettaCoordCst
    from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.numeric import xyzVector_double_t
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover

    start_time = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdb = output_dir / f"{config_name}_relaxed.pdb"

    result = {
        'config_name': config_name,
        'config': config,
        'success': False,
        'scores': {},
        'displacements': {},
        'bond_deviations': {},
        'metrics': {},
    }

    # Initialize PyRosetta
    options = ["-mute all", "-corrections:beta_jan25"]
    params_str = " ".join(str(p) for p in params_files if p.exists())
    if params_str:
        options.append(f"-extra_res_fa {params_str}")

    pyrosetta.init(" ".join(options))

    # Load pose
    pose = pyrosetta.pose_from_pdb(str(pdb_path))
    pdb_info = pose.pdb_info()

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Test: {config_name}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Config: {config}", file=sys.stderr)

    # Parse REMARK 666 to find catres
    catres_residues = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('REMARK 666'):
                parts = line.split()
                try:
                    # Find MATCH MOTIF positions
                    motif_idx = parts.index('MOTIF')
                    chain = parts[motif_idx + 1]
                    resnum = int(parts[motif_idx + 3])
                    catres_residues.append((chain, resnum))
                except (ValueError, IndexError):
                    pass

    print(f"Found {len(catres_residues)} catres from REMARK 666", file=sys.stderr)

    # Find catres in pose
    catres_pose_indices = []
    for chain, resnum in catres_residues:
        for res_i in range(1, pose.total_residue() + 1):
            if pdb_info.chain(res_i) == chain and pdb_info.number(res_i) == resnum:
                catres_pose_indices.append(res_i)
                break

    # Find ligand residues (HETATM with >5 atoms)
    ligand_resnums = set()
    for res_i in range(1, pose.total_residue() + 1):
        if not pose.residue(res_i).is_protein():
            if pose.residue(res_i).natoms() > 5:
                ligand_resnums.add(res_i)

    print(f"Identified {len(ligand_resnums)} ligand residues", file=sys.stderr)

    # Store original catres coordinates
    original_coords = {}
    for res_i in catres_pose_indices:
        residue = pose.residue(res_i)
        chain = pdb_info.chain(res_i)
        resnum = pdb_info.number(res_i)
        for atom_i in range(1, residue.natoms() + 1):
            atom_name = residue.atom_name(atom_i).strip()
            if not atom_name.startswith('H'):
                xyz = residue.xyz(atom_i)
                key = f"{chain}_{resnum}_{atom_name}"
                original_coords[key] = (xyz.x, xyz.y, xyz.z)

    # Store original ligand coordinates
    ligand_coords = {}
    for res_i in ligand_resnums:
        residue = pose.residue(res_i)
        chain = pdb_info.chain(res_i)
        resnum = pdb_info.number(res_i)
        for atom_i in range(1, residue.natoms() + 1):
            atom_name = residue.atom_name(atom_i).strip()
            xyz = residue.xyz(atom_i)
            key = f"{chain}_{resnum}_{atom_name}"
            ligand_coords[key] = (xyz.x, xyz.y, xyz.z)

    # Find anchor atom
    anchor_atom_id = AtomID(1, 1)
    for res_i in range(1, pose.total_residue() + 1):
        if pose.residue(res_i).is_protein() and pose.residue(res_i).has("CA"):
            anchor_atom_id = AtomID(pose.residue(res_i).atom_index("CA"), res_i)
            break

    # Apply coordinate constraints to catres sidechains
    n_cst = 0
    for key, (x, y, z) in original_coords.items():
        parts = key.split('_')
        chain, resnum, atom_name = parts[0], int(parts[1]), '_'.join(parts[2:])

        for res_i in range(1, pose.total_residue() + 1):
            if pdb_info.chain(res_i) == chain and pdb_info.number(res_i) == resnum:
                residue = pose.residue(res_i)
                if residue.has(atom_name):
                    atom_idx = residue.atom_index(atom_name)
                    target_id = AtomID(atom_idx, res_i)
                    xyz_vec = xyzVector_double_t(x, y, z)
                    func = HarmonicFunc(0.0, 0.2)  # stdev 0.2
                    cst = RosettaCoordCst(target_id, anchor_atom_id, xyz_vec, func)
                    pose.add_constraint(cst)
                    n_cst += 1
                break

    # Apply tight constraints to ligand atoms
    n_lig_cst = 0
    for key, (x, y, z) in ligand_coords.items():
        parts = key.split('_')
        chain, resnum, atom_name = parts[0], int(parts[1]), '_'.join(parts[2:])

        for res_i in range(1, pose.total_residue() + 1):
            if pdb_info.chain(res_i) == chain and pdb_info.number(res_i) == resnum:
                residue = pose.residue(res_i)
                if residue.has(atom_name):
                    atom_idx = residue.atom_index(atom_name)
                    target_id = AtomID(atom_idx, res_i)
                    xyz_vec = xyzVector_double_t(x, y, z)
                    func = HarmonicFunc(0.0, 0.001)  # Very tight
                    cst = RosettaCoordCst(target_id, anchor_atom_id, xyz_vec, func)
                    pose.add_constraint(cst)
                    n_lig_cst += 1
                break

    print(f"Applied {n_cst} catres constraints, {n_lig_cst} ligand constraints", file=sys.stderr)

    # Determine mobile center
    if ligand_coords:
        xs = [c[0] for c in ligand_coords.values()]
        ys = [c[1] for c in ligand_coords.values()]
        zs = [c[2] for c in ligand_coords.values()]
        mobile_center = (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))
    else:
        mobile_center = (0, 0, 0)

    # Create movemap
    mm = MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)

    mobile_radius = 10.0
    allow_catres_bb = config.get('allow_catres_bb', True)

    n_mobile_bb = 0
    n_mobile_chi = 0

    for res_i in range(1, pose.total_residue() + 1):
        if res_i in ligand_resnums:
            mm.set_bb(res_i, False)
            mm.set_chi(res_i, False)
            continue

        try:
            if pose.residue(res_i).has("CA"):
                xyz = pose.residue(res_i).xyz("CA")
            else:
                xyz = pose.residue(res_i).xyz(1)

            dist = math.sqrt(
                (mobile_center[0] - xyz.x)**2 +
                (mobile_center[1] - xyz.y)**2 +
                (mobile_center[2] - xyz.z)**2
            )

            if dist <= mobile_radius:
                if res_i in catres_pose_indices:
                    mm.set_chi(res_i, True)
                    n_mobile_chi += 1
                    if allow_catres_bb:
                        mm.set_bb(res_i, True)
                        n_mobile_bb += 1
                else:
                    mm.set_bb(res_i, True)
                    mm.set_chi(res_i, True)
                    n_mobile_bb += 1
                    n_mobile_chi += 1
        except Exception:
            pass

    print(f"MoveMap: {n_mobile_bb} BB mobile, {n_mobile_chi} chi mobile", file=sys.stderr)

    # Score initial
    sfxn = pyrosetta.create_score_function("beta_jan25")
    sfxn.set_weight(ScoreType.cart_bonded, config.get('cart_bonded_weight', 0.5))
    sfxn.set_weight(ScoreType.coordinate_constraint, config.get('coord_cst_weight', 200.0))
    sfxn.set_weight(ScoreType.pro_close, 0.0)

    initial_score = sfxn(pose)
    print(f"Initial score: {initial_score:.2f}", file=sys.stderr)

    # Run relaxation
    use_multistage = config.get('use_multistage_relax', True)

    if use_multistage:
        print("Running multi-stage relaxation...", file=sys.stderr)

        initial_cst = config.get('initial_coord_cst_weight', 1000.0)
        final_cst = config.get('final_coord_cst_weight', 100.0)
        initial_fa_rep = config.get('initial_fa_rep_scale', 0.15)
        n_stages = config.get('n_relax_stages', 3)
        fastrelax_cycles = config.get('fastrelax_cycles', 2)

        for stage in range(n_stages):
            progress = stage / max(n_stages - 1, 1)
            cst_weight = initial_cst + (final_cst - initial_cst) * progress
            fa_rep_scale = initial_fa_rep + (1.0 - initial_fa_rep) * progress

            sfxn_stage = pyrosetta.create_score_function("beta_jan25")
            sfxn_stage.set_weight(ScoreType.cart_bonded, config.get('cart_bonded_weight', 0.5))
            sfxn_stage.set_weight(ScoreType.coordinate_constraint, cst_weight)
            sfxn_stage.set_weight(ScoreType.fa_rep, sfxn_stage.get_weight(ScoreType.fa_rep) * fa_rep_scale)
            sfxn_stage.set_weight(ScoreType.pro_close, 0.0)

            # Initial minimization in stage 1
            if stage == 0:
                min_mover = MinMover()
                min_mover.score_function(sfxn_stage)
                min_mover.movemap(mm)
                min_mover.cartesian(True)
                min_mover.tolerance(0.001)
                min_mover.apply(pose)
                print(f"  Stage {stage+1} minimization done", file=sys.stderr)

            # FastRelax
            relax = FastRelax()
            relax.set_scorefxn(sfxn_stage)
            relax.set_movemap(mm)
            relax.cartesian(True)
            relax.min_type("lbfgs_armijo_nonmonotone")
            relax.max_iter(200)

            for cycle in range(fastrelax_cycles):
                relax.apply(pose)

            score = sfxn_stage(pose)
            cart_bonded = pose.energies().total_energies()[ScoreType.cart_bonded]
            print(f"  Stage {stage+1}: cst_weight={cst_weight:.0f}, fa_rep_scale={fa_rep_scale:.2f}, "
                  f"score={score:.1f}, cart_bonded={cart_bonded:.1f}", file=sys.stderr)

        # Final polish
        final_sfxn = pyrosetta.create_score_function("beta_jan25")
        final_sfxn.set_weight(ScoreType.cart_bonded, config.get('cart_bonded_weight', 0.5))
        final_sfxn.set_weight(ScoreType.coordinate_constraint, final_cst)
        final_sfxn.set_weight(ScoreType.pro_close, 0.0)

        min_mover = MinMover()
        min_mover.score_function(final_sfxn)
        min_mover.movemap(mm)
        min_mover.cartesian(True)
        min_mover.tolerance(0.00001)
        min_mover.apply(pose)
        print("  Final polish done", file=sys.stderr)

    else:
        print("Running old-style relaxation...", file=sys.stderr)
        relax = FastRelax()
        relax.set_scorefxn(sfxn)
        relax.set_movemap(mm)
        relax.cartesian(True)
        relax.min_type("lbfgs_armijo_nonmonotone")

        for cycle in range(config.get('fastrelax_cycles', 2)):
            relax.apply(pose)
            score = sfxn(pose)
            print(f"  Cycle {cycle+1}: score={score:.1f}", file=sys.stderr)

    # Final scores
    final_score = sfxn(pose)
    cart_bonded = pose.energies().total_energies()[ScoreType.cart_bonded]
    coord_cst = pose.energies().total_energies()[ScoreType.coordinate_constraint]

    print(f"\nFinal scores:", file=sys.stderr)
    print(f"  Total: {final_score:.2f}", file=sys.stderr)
    print(f"  cart_bonded: {cart_bonded:.2f}", file=sys.stderr)
    print(f"  coordinate_constraint: {coord_cst:.2f}", file=sys.stderr)

    result['scores'] = {
        'total': final_score,
        'cart_bonded': cart_bonded,
        'coordinate_constraint': coord_cst,
    }

    # Compute displacements
    catres_displacements = {}
    for key, (orig_x, orig_y, orig_z) in original_coords.items():
        parts = key.split('_')
        chain, resnum, atom_name = parts[0], int(parts[1]), '_'.join(parts[2:])

        for res_i in range(1, pose.total_residue() + 1):
            if pdb_info.chain(res_i) == chain and pdb_info.number(res_i) == resnum:
                residue = pose.residue(res_i)
                if residue.has(atom_name):
                    xyz = residue.xyz(atom_name)
                    dist = math.sqrt(
                        (orig_x - xyz.x)**2 + (orig_y - xyz.y)**2 + (orig_z - xyz.z)**2
                    )
                    catres_displacements[key] = dist
                break

    if catres_displacements:
        mean_disp = sum(catres_displacements.values()) / len(catres_displacements)
        max_disp = max(catres_displacements.values())
        print(f"  Catres displacement: mean={mean_disp:.3f}, max={max_disp:.3f}", file=sys.stderr)
        result['displacements']['catres_mean'] = mean_disp
        result['displacements']['catres_max'] = max_disp

    # Compute ligand displacements
    ligand_displacements = {}
    for key, (orig_x, orig_y, orig_z) in ligand_coords.items():
        parts = key.split('_')
        chain, resnum, atom_name = parts[0], int(parts[1]), '_'.join(parts[2:])

        for res_i in range(1, pose.total_residue() + 1):
            if pdb_info.chain(res_i) == chain and pdb_info.number(res_i) == resnum:
                residue = pose.residue(res_i)
                if residue.has(atom_name):
                    xyz = residue.xyz(atom_name)
                    dist = math.sqrt(
                        (orig_x - xyz.x)**2 + (orig_y - xyz.y)**2 + (orig_z - xyz.z)**2
                    )
                    ligand_displacements[key] = dist
                break

    if ligand_displacements:
        max_lig_disp = max(ligand_displacements.values())
        print(f"  Ligand displacement: max={max_lig_disp:.6f}", file=sys.stderr)
        result['displacements']['ligand_max'] = max_lig_disp

    # Compute bond deviations
    bond_devs = compute_bond_deviations_pyrosetta(pose, catres_pose_indices)
    if bond_devs:
        deviations = [b['deviation'] for b in bond_devs]
        mean_dev = sum(deviations) / len(deviations)
        max_dev = max(deviations)
        n_critical = sum(1 for d in deviations if d > 0.1)

        print(f"  Bond deviations: mean={mean_dev:.4f}, max={max_dev:.4f}, n_critical={n_critical}", file=sys.stderr)
        result['bond_deviations']['mean'] = mean_dev
        result['bond_deviations']['max'] = max_dev
        result['bond_deviations']['n_critical'] = n_critical

        # Store worst bonds
        worst = sorted(bond_devs, key=lambda x: x['deviation'], reverse=True)[:5]
        result['bond_deviations']['worst'] = worst

    # Ring flip sampling
    ring_flip_residues = {'HIS', 'PHE', 'TYR'}
    n_flipped = 0
    n_tried = 0

    for res_i in catres_pose_indices:
        residue = pose.residue(res_i)
        resname = residue.name3()[:3]

        if resname not in ring_flip_residues:
            continue
        if residue.nchi() < 2:
            continue

        n_tried += 1
        original_chi2 = pose.chi(2, res_i)
        score_before = sfxn(pose)

        pose.set_chi(2, res_i, original_chi2 + 180.0)
        score_after = sfxn(pose)

        if score_after < score_before:
            n_flipped += 1
            print(f"  Flipped {pdb_info.chain(res_i)}{pdb_info.number(res_i)}: "
                  f"{score_before:.1f} -> {score_after:.1f}", file=sys.stderr)
        else:
            pose.set_chi(2, res_i, original_chi2)

    print(f"  Ring flips: tried={n_tried}, accepted={n_flipped}", file=sys.stderr)
    result['metrics']['ring_flips_tried'] = n_tried
    result['metrics']['ring_flips_accepted'] = n_flipped

    # Save output
    pose.dump_pdb(str(output_pdb))
    result['success'] = True
    result['output_pdb'] = str(output_pdb)
    result['duration_seconds'] = time.time() - start_time

    # Quality assessment
    passes = True
    failures = []

    if cart_bonded > 50:
        failures.append(f"cart_bonded {cart_bonded:.1f} > 50")
        passes = False

    max_bond_dev = result['bond_deviations'].get('max', 0)
    if max_bond_dev > 0.04:
        failures.append(f"max_bond_dev {max_bond_dev:.3f} > 0.04")
        passes = False

    n_critical = result['bond_deviations'].get('n_critical', 0)
    if n_critical > 0:
        failures.append(f"{n_critical} critical bonds")
        passes = False

    max_lig_disp = result['displacements'].get('ligand_max', 0)
    if max_lig_disp > 0.01:
        failures.append(f"ligand moved {max_lig_disp:.4f}")
        passes = False

    result['metrics']['passes_criteria'] = passes
    result['metrics']['failures'] = failures

    print(f"\n{'PASS' if passes else 'FAIL'}: {', '.join(failures) if failures else 'All criteria met'}", file=sys.stderr)

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--params", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default="baseline_new", choices=list(CONFIGS.keys()))

    args = parser.parse_args()

    pdb_path = Path(args.pdb)
    params_files = [Path(p) for p in args.params]
    output_dir = Path(args.output_dir)

    config = CONFIGS[args.config]

    result = run_test(pdb_path, params_files, output_dir, args.config, config)

    # Save result
    result_file = output_dir / f"{args.config}_result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nResult saved to: {result_file}", file=sys.stderr)

    # Print summary to stdout
    print(json.dumps({
        'config': args.config,
        'cart_bonded': result['scores'].get('cart_bonded'),
        'max_bond_dev': result['bond_deviations'].get('max'),
        'passes': result['metrics'].get('passes_criteria'),
        'failures': result['metrics'].get('failures'),
    }, indent=2))


if __name__ == "__main__":
    main()
