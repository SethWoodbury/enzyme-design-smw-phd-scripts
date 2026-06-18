import sys
import argparse
import os
import pyrosetta as pyr
import pyrosetta.rosetta

# --- locate repo root + shared external paths ---
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "repo_paths.py").is_file():
        _sys.path.insert(0, str(_anc)); break
import repo_paths

# Append the path to access design_utils
sys.path.append(repo_paths.ENZYME_DESIGN_UTILS)
import design_utils

# Set up argument parsing
parser = argparse.ArgumentParser(description="Relax structure with constraints using PyRosetta")
parser.add_argument("--input_pdb", required=True, help="Path to the input PDB file")
parser.add_argument("--params", required=True, help="Extra residue parameters file (e.g. LIG.params)")
parser.add_argument("--cst_file", required=True, help="Constraint file (e.g. file.cst)")
parser.add_argument("--output_directory", required=True, help="Output directory to save the relaxed structure")
args = parser.parse_args()

# Initialize PyRosetta with the extra residue parameter file and other options
pyr.init(f"-extra_res_fa {args.params} -run:preserve_header")

# Get the full-atom scoring function and set constraint weights
sfx = pyr.get_fa_scorefxn()
sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("atom_pair_constraint"), 1.0)
sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("angle_constraint"), 1.0)
sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("dihedral_constraint"), 1.0)

# Create the constraint mover using the provided constraint file
cst_mover = design_utils.CSTs(args.cst_file, sfx)

#sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("atom_pair_constraint"), 100.0)
#sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("angle_constraint"), 100.0)
#sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("dihedral_constraint"), 100.0)

# Set up the FastRelax protocol with constraints ramping
fastrelax = pyrosetta.rosetta.protocols.relax.FastRelax(sfx, 2)
fastrelax.constrain_relax_to_start_coords(True)
fastrelax.ramp_down_constraints(False)

# Define a task factory for repacking (using several operations)
tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())
tf.push_back(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())
fastrelax.set_task_factory(tf)

# Set up the MoveMap to allow backbone, side-chain, and jump movement
mm = pyrosetta.rosetta.core.kinematics.MoveMap()
mm.set_chi(True)
mm.set_bb(True)
mm.set_jump(True)
fastrelax.set_movemap(mm)

# Load the input PDB structure
pose = pyr.pose_from_file(args.input_pdb)

cst_mover.add_cst(pose)

# Apply the fast relax protocol
fastrelax.apply(pose)

# (Optional) score the relaxed pose
sfx(pose)

# Ensure the output directory exists and build the output filename
if not os.path.exists(args.output_directory):
    os.makedirs(args.output_directory)
output_file = os.path.join(
    args.output_directory, os.path.basename(args.input_pdb).replace(".pdb", "_relaxed.pdb")
)

# Save the relaxed structure
pose.dump_pdb(output_file)
