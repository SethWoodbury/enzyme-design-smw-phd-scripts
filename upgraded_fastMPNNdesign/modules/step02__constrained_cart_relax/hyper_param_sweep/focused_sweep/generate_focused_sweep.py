#!/usr/bin/env python3
"""
Generate focused parameter sweep with replicates.

Based on initial sweep findings:
- ref2015_cart is the best scorefunction (fixed)
- cart_bonded_weight around 2.0 is optimal (test 1.5, 2.0, 2.5, 3.0)
- 1x3, 2x3, 3x3, 3x5 all performed similarly (need replicates to distinguish)
- bond_geometry_min had mixed results (test both)
- 5 replicates per condition to assess Monte Carlo variance
"""

import os

# Paths
SCRIPT_DIR = "/home/woodbuse/special_scripts/upgraded_fastMPNNdesign/modules/step02__constrained_cart_relax"
SWEEP_DIR = f"{SCRIPT_DIR}/hyper_param_sweep/focused_sweep"
STEP01_JSON = f"{SCRIPT_DIR}/test/step01_outputs/input_pdb_recommended_atom_cst.json"
PARAMS = f"{SCRIPT_DIR}/test/params/XDW.params"
OUTPUT_BASE = f"{SWEEP_DIR}/outputs"

# Fixed parameters (clear winners from initial sweep)
SCOREFUNCTION = "ref2015_cart"
AUTO_EXPAND = True
CART_BONDED_MAX = 4.0  # Higher ceiling for scaling

# Variable parameters to sweep
CART_BONDED_WEIGHTS = [1.5, 2.0, 2.5, 3.0]
REPEATS_STAGES = [
    (1, 3),   # 3 total
    (2, 3),   # 6 total
    (3, 3),   # 9 total
    (3, 5),   # 15 total
]
BOND_GEOMETRY_MIN = [True, False]
NUM_REPLICATES = 5

commands = []
job_id = 0

for cart_weight in CART_BONDED_WEIGHTS:
    for (repeats, stages) in REPEATS_STAGES:
        for bond_geom in BOND_GEOMETRY_MIN:
            for rep in range(1, NUM_REPLICATES + 1):
                job_id += 1
                out_name = f"job{job_id:03d}__r{repeats}s{stages}__cb{cart_weight}__bg{'on' if bond_geom else 'off'}__rep{rep}"
                out_dir = f"{OUTPUT_BASE}/{out_name}"

                cmd_parts = [
                    f"apptainer exec /net/software/containers/universal.sif python {SCRIPT_DIR}/constrained_cart_relax.py",
                    f"--step01_json {STEP01_JSON}",
                    f"--params {PARAMS}",
                    f"--scorefunction {SCOREFUNCTION}",
                    f"--fastrelax_repeats {repeats}",
                    f"--fastrelax_ramp_stages {stages}",
                    f"--cart_bonded_weight {cart_weight}",
                    f"--cart_bonded_max {CART_BONDED_MAX}",
                    "--enable_bond_geometry_min" if bond_geom else "--disable_bond_geometry_min",
                    "--auto_expand_mobile",
                    f"--output_dir {out_dir}",
                    "--max_runtime 3600",
                ]
                commands.append(" ".join(cmd_parts))

# Create directories
os.makedirs(f"{SWEEP_DIR}/cmds", exist_ok=True)
os.makedirs(f"{SWEEP_DIR}/logs", exist_ok=True)
os.makedirs(OUTPUT_BASE, exist_ok=True)

# Write commands file
cmds_file = f"{SWEEP_DIR}/cmds/all_sweep_commands.txt"
with open(cmds_file, "w") as f:
    for cmd in commands:
        f.write(cmd + "\n")

print(f"Generated {len(commands)} commands")
print(f"Commands written to: {cmds_file}")

# Print summary
print("\nFocused sweep summary:")
print(f"  Scorefunction: {SCOREFUNCTION} (fixed)")
print(f"  Cart_bonded weights: {CART_BONDED_WEIGHTS}")
print(f"  Repeats x Stages: {REPEATS_STAGES}")
print(f"  Bond geometry min: {BOND_GEOMETRY_MIN}")
print(f"  Replicates per condition: {NUM_REPLICATES}")
print(f"  Cart_bonded_max: {CART_BONDED_MAX}")
print(f"  Total jobs: {len(commands)}")
print(f"\nConditions: {len(CART_BONDED_WEIGHTS)} x {len(REPEATS_STAGES)} x {len(BOND_GEOMETRY_MIN)} = {len(CART_BONDED_WEIGHTS) * len(REPEATS_STAGES) * len(BOND_GEOMETRY_MIN)}")
print(f"With {NUM_REPLICATES} replicates each = {len(commands)} total jobs")
