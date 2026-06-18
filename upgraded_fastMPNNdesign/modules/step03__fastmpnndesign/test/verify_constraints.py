#!/usr/bin/env python3
"""
Quick verification test for constraint loading and application.

This script:
1. Loads the step02 JSON and verifies constraint parsing
2. Creates a PyRosetta pose and applies constraints
3. Reports on ligand RMSD before/after a brief relax
"""

import sys
import os
import json
import tempfile

# Add project root to path
PROJECT_ROOT = "/home/woodbuse/special_scripts/upgraded_fastMPNNdesign"
sys.path.insert(0, PROJECT_ROOT)

from modules.step03__fastmpnndesign.fastmpnn_design import FastMPNNDesigner
from modules.step03__fastmpnndesign.rosetta_relax import add_coordinate_constraints

# Paths
STEP02_JSON = "/home/woodbuse/special_scripts/upgraded_fastMPNNdesign/modules/step02__constrained_cart_relax/test/output_dir/input_pdb_aligned_relaxed_metrics.json"
STEP02_PDB = "/home/woodbuse/special_scripts/upgraded_fastMPNNdesign/modules/step02__constrained_cart_relax/test/output_dir/input_pdb_aligned_relaxed.pdb"
PARAMS = "/home/woodbuse/special_scripts/upgraded_fastMPNNdesign/modules/step03__fastmpnndesign/test/params/XDW.params"


def test_constraint_loading():
    """Test that constraints are loaded correctly from step02 JSON."""
    print("=" * 70)
    print("TEST 1: CONSTRAINT LOADING")
    print("=" * 70)

    with open(STEP02_JSON) as f:
        data = json.load(f)

    # Check for residue_constraints (both top-level and in metadata)
    rc = data.get("residue_constraints") or data.get("metadata", {}).get("residue_constraints")

    if not rc:
        print("ERROR: No residue_constraints found in JSON!")
        return False

    print(f"Found {len(rc)} constraint entries")

    # Check ligand
    ligand = rc.get("ligand")
    if not ligand:
        print("ERROR: No ligand entry in constraints!")
        return False

    print(f"\nLigand constraint:")
    print(f"  Chain: {ligand.get('chain')}")
    print(f"  Resno: {ligand.get('resno')}")
    print(f"  Resname: {ligand.get('resname')}")
    print(f"  is_ligand: {ligand.get('is_ligand')}")
    print(f"  Atoms in JSON: {len(ligand.get('constrain_atoms', []))}")

    # Count catres
    catres_count = 0
    for key, val in rc.items():
        if key != "ligand" and not val.get("is_ligand"):
            catres_count += 1

    print(f"\nCatalytic residues: {catres_count}")

    return True


def test_fastmpnn_constraint_parsing():
    """Test FastMPNNDesigner constraint parsing."""
    print("\n" + "=" * 70)
    print("TEST 2: FASTMPNN DESIGNER CONSTRAINT PARSING")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        designer = FastMPNNDesigner(
            step02_json_path=STEP02_JSON,
            params_files=[PARAMS],
            output_dir=tmpdir,
            protocol="fast",
            debug=True,
        )

        # Initialize loads constraints
        designer.initialize()

        print(f"\nLoaded constraints for {len(designer.constrained_atoms)} residues")

        # Check ligand constraint
        ligand_key = None
        for key, atoms in designer.constrained_atoms.items():
            if atoms == "ALL_ATOMS" or atoms == "ALL_HEAVY":
                ligand_key = key
                print(f"\n[LIGAND] {key}: {atoms}")
            elif isinstance(atoms, list):
                print(f"[CATRES] {key}: {len(atoms)} atoms")

        if not ligand_key:
            print("WARNING: No ligand with ALL_ATOMS/ALL_HEAVY found!")

        return len(designer.constrained_atoms) > 0


def test_pyrosetta_constraints():
    """Test that constraints are actually applied to PyRosetta pose."""
    print("\n" + "=" * 70)
    print("TEST 3: PYROSETTA CONSTRAINT APPLICATION")
    print("=" * 70)

    # Initialize PyRosetta in container
    import subprocess

    # Create test script
    test_script = f'''
import pyrosetta
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.core.scoring import ScoreFunction
import numpy as np
import json

# Initialize PyRosetta
pyrosetta.init("-mute all -extra_res_fa {PARAMS}")

# Load pose
pose = pose_from_pdb("{STEP02_PDB}")
print(f"Loaded pose with {{pose.total_residue()}} residues")

# Load constraints from JSON
with open("{STEP02_JSON}") as f:
    data = json.load(f)

rc = data.get("residue_constraints") or data.get("metadata", {{}}).get("residue_constraints") or {{}}
print(f"Loaded {{len(rc)}} constraint entries from JSON")

# Build constraints dict
constraints = {{}}
for key, val in rc.items():
    chain = val.get("chain", "A")
    resno = val.get("resno")
    is_ligand = val.get("is_ligand", False) or key == "ligand"

    if resno is None:
        continue

    cst_key = f"{{chain}}:{{resno}}"

    if is_ligand:
        # ALL_ATOMS for ligand (including H)
        constraints[cst_key] = "ALL_ATOMS"
    else:
        atoms = val.get("constrain_atoms", [])
        if atoms:
            constraints[cst_key] = atoms

print(f"Built constraints for {{len(constraints)}} residues")

# Find ligand
ligand_key = None
for key, atoms in constraints.items():
    if atoms == "ALL_ATOMS":
        ligand_key = key
        break

if ligand_key:
    chain, resno = ligand_key.split(":")
    print(f"Ligand: {{ligand_key}}")

    # Get ligand residue
    from pyrosetta.rosetta.core.pose import PDBInfo
    pdb_info = pose.pdb_info()
    pose_idx = pdb_info.pdb2pose(chain, int(resno))
    if pose_idx > 0:
        residue = pose.residue(pose_idx)
        print(f"  Pose index: {{pose_idx}}")
        print(f"  Residue name: {{residue.name()}}")
        print(f"  Total atoms: {{residue.natoms()}}")

        # Count heavy vs H
        n_heavy = sum(1 for i in range(1, residue.natoms()+1) if not residue.atom_is_hydrogen(i))
        n_h = residue.natoms() - n_heavy
        print(f"  Heavy atoms: {{n_heavy}}")
        print(f"  Hydrogen atoms: {{n_h}}")

        # Get original coordinates for RMSD later
        orig_coords = {{}}
        for i in range(1, residue.natoms()+1):
            name = residue.atom_name(i).strip()
            xyz = residue.xyz(name)
            orig_coords[name] = (xyz.x, xyz.y, xyz.z)

# Add constraints
from pyrosetta.rosetta.core.scoring.constraints import CoordinateConstraint
from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
from pyrosetta.rosetta.core.id import AtomID

n_cst = 0
for cst_key, atom_spec in constraints.items():
    chain, resno = cst_key.split(":")
    pose_idx = pdb_info.pdb2pose(chain, int(resno))

    if pose_idx <= 0:
        continue

    residue = pose.residue(pose_idx)

    # Determine atoms
    if atom_spec == "ALL_ATOMS":
        atom_names = [residue.atom_name(i).strip() for i in range(1, residue.natoms()+1)]
    elif atom_spec == "ALL_HEAVY":
        atom_names = [residue.atom_name(i).strip() for i in range(1, residue.natoms()+1)
                      if not residue.atom_is_hydrogen(i)]
    else:
        atom_names = atom_spec if isinstance(atom_spec, list) else [atom_spec]

    for atom_name in atom_names:
        if not residue.has(atom_name):
            continue

        atom_idx = residue.atom_index(atom_name)
        xyz = residue.xyz(atom_name)

        func = HarmonicFunc(0.0, 0.01)  # Very tight: 0.01A stdev
        cst = CoordinateConstraint(
            AtomID(atom_idx, pose_idx),
            AtomID(1, 1),
            xyz,
            func
        )
        pose.add_constraint(cst)
        n_cst += 1

print(f"\\nAdded {{n_cst}} coordinate constraints")

# Check constraint count
from pyrosetta.rosetta.core.scoring.constraints import ConstraintSet
cst_set = pose.constraint_set()
print(f"Pose constraint count: {{len(cst_set.get_all_constraints())}}")

# Score pose with constraints
sfxn = ScoreFunction()
sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.coordinate_constraint, 750.0)
sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.fa_atr, 1.0)
sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.fa_rep, 0.55)

score_before = sfxn(pose)
cst_score_before = pose.energies().total_energies()[pyrosetta.rosetta.core.scoring.ScoreType.coordinate_constraint]
print(f"\\nScore before relax: {{score_before:.2f}}")
print(f"Constraint score before: {{cst_score_before:.4f}}")

# Do a brief minimization to test constraint effectiveness
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.core.kinematics import MoveMap

# Full movemap - allow everything to move
mm = MoveMap()
mm.set_bb(True)
mm.set_chi(True)
mm.set_jump(True)

# Create full scorefunction
sfxn_full = pyrosetta.create_score_function("ref2015_cart")
sfxn_full.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.coordinate_constraint, 750.0)

min_mover = MinMover()
min_mover.movemap(mm)
min_mover.score_function(sfxn_full)
min_mover.min_type("lbfgs_armijo_nonmonotone")
min_mover.tolerance(0.001)
min_mover.cartesian(True)
min_mover.max_iter(100)

print("\\nRunning brief cartesian minimization (100 iterations)...")
min_mover.apply(pose)

score_after = sfxn(pose)
cst_score_after = pose.energies().total_energies()[pyrosetta.rosetta.core.scoring.ScoreType.coordinate_constraint]
print(f"Score after minimize: {{score_after:.2f}}")
print(f"Constraint score after: {{cst_score_after:.4f}}")

# Calculate ligand RMSD
if ligand_key:
    chain, resno = ligand_key.split(":")
    pose_idx = pdb_info.pdb2pose(chain, int(resno))
    if pose_idx > 0:
        residue = pose.residue(pose_idx)

        rmsd_sum = 0.0
        n_atoms = 0
        max_dev = 0.0
        max_dev_atom = ""

        for atom_name, (ox, oy, oz) in orig_coords.items():
            if not residue.has(atom_name):
                continue
            xyz = residue.xyz(atom_name)
            dx = xyz.x - ox
            dy = xyz.y - oy
            dz = xyz.z - oz
            dev = (dx*dx + dy*dy + dz*dz) ** 0.5

            rmsd_sum += dev * dev
            n_atoms += 1

            if dev > max_dev:
                max_dev = dev
                max_dev_atom = atom_name

        rmsd = (rmsd_sum / n_atoms) ** 0.5 if n_atoms > 0 else 0
        print(f"\\nLigand RMSD after minimize: {{rmsd:.4f}} A")
        print(f"Max atom deviation: {{max_dev:.4f}} A ({{max_dev_atom}})")

        if rmsd < 0.1:
            print("\\n[PASS] Ligand RMSD < 0.1 A - constraints are working!")
        else:
            print("\\n[FAIL] Ligand RMSD > 0.1 A - constraints may not be effective!")
'''

    # Write script to temp file
    script_path = "/tmp/test_constraints.py"
    with open(script_path, "w") as f:
        f.write(test_script)

    # Run in universal container (has PyRosetta 2026.03)
    cmd = [
        "apptainer", "exec",
        "/net/software/containers/universal.sif",
        "python", script_path
    ]

    print("Running PyRosetta constraint test in container...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    print("\n--- PyRosetta Output ---")
    print(result.stdout)

    if result.stderr:
        print("\n--- Errors ---")
        print(result.stderr)

    return result.returncode == 0


def main():
    print("=" * 70)
    print("CONSTRAINT VERIFICATION TEST")
    print("=" * 70)
    print(f"Step02 JSON: {STEP02_JSON}")
    print(f"Step02 PDB:  {STEP02_PDB}")
    print(f"Params:      {PARAMS}")

    # Test 1: Constraint loading from JSON
    test1_ok = test_constraint_loading()

    # Test 2: FastMPNNDesigner parsing
    test2_ok = test_fastmpnn_constraint_parsing()

    # Test 3: PyRosetta constraint application
    test3_ok = test_pyrosetta_constraints()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Test 1 (JSON loading):      {'PASS' if test1_ok else 'FAIL'}")
    print(f"Test 2 (Designer parsing):  {'PASS' if test2_ok else 'FAIL'}")
    print(f"Test 3 (PyRosetta apply):   {'PASS' if test3_ok else 'FAIL'}")

    if test1_ok and test2_ok and test3_ok:
        print("\n[SUCCESS] All constraint tests passed!")
        return 0
    else:
        print("\n[FAILURE] Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
