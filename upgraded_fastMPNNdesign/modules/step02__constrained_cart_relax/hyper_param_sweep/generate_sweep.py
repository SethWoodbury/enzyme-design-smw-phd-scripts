#!/usr/bin/env python3
"""
Generate parameter sweep commands for constrained_cart_relax.py

Sweeps over scorefunctions, repeats/stages, cart_bonded weights, and bond geometry min.
"""

import os

# Paths
SCRIPT_DIR = "/home/woodbuse/special_scripts/upgraded_fastMPNNdesign/modules/step02__constrained_cart_relax"
SWEEP_DIR = f"{SCRIPT_DIR}/hyper_param_sweep"
STEP01_JSON = f"{SCRIPT_DIR}/test/step01_outputs/input_pdb_recommended_atom_cst.json"
PARAMS = f"{SCRIPT_DIR}/test/params/XDW.params"
OUTPUT_BASE = f"{SWEEP_DIR}/outputs"

# Parameter sweep options
SCOREFUNCTIONS = ["ref2015_cart", "beta_jan25_cart"]
REPEATS_STAGES = [
    (1, 3),   # 3 total (fast)
    (2, 3),   # 6 total (light)
    (3, 4),   # 12 total (balanced)
    (3, 5),   # 15 total (default)
    (5, 5),   # 25 total (thorough)
]
CART_BONDED_WEIGHTS = [0.5, 0.8, 1.0, 1.5, 2.0]
BOND_GEOMETRY_MIN = [True, False]

commands = []
job_id = 0

# Core sweep: scorefunction x repeats_stages x cart_bonded x bond_geom_min
for sfxn in SCOREFUNCTIONS:
    for (repeats, stages) in REPEATS_STAGES:
        for cart_weight in CART_BONDED_WEIGHTS:
            for bond_geom in BOND_GEOMETRY_MIN:
                job_id += 1
                out_name = f"job{job_id:03d}__{sfxn.replace('_cart', '')}__r{repeats}s{stages}__cb{cart_weight}__bg{'on' if bond_geom else 'off'}"
                out_dir = f"{OUTPUT_BASE}/{out_name}"

                cmd_parts = [
                    f"apptainer exec /net/software/containers/universal.sif python {SCRIPT_DIR}/constrained_cart_relax.py",
                    f"--step01_json {STEP01_JSON}",
                    f"--params {PARAMS}",
                    f"--scorefunction {sfxn}",
                    f"--fastrelax_repeats {repeats}",
                    f"--fastrelax_ramp_stages {stages}",
                    f"--cart_bonded_weight {cart_weight}",
                    "--enable_bond_geometry_min" if bond_geom else "--disable_bond_geometry_min",
                    "--auto_expand_mobile",
                    f"--output_dir {out_dir}",
                    "--max_runtime 3600",
                ]
                commands.append(" ".join(cmd_parts))

# Add preset tests
for preset in ["fast", "balanced", "thorough", "aggressive"]:
    for sfxn in SCOREFUNCTIONS:
        job_id += 1
        out_name = f"job{job_id:03d}__preset_{preset}__{sfxn.replace('_cart', '')}"
        out_dir = f"{OUTPUT_BASE}/{out_name}"

        cmd_parts = [
            f"apptainer exec /net/software/containers/universal.sif python {SCRIPT_DIR}/constrained_cart_relax.py",
            f"--step01_json {STEP01_JSON}",
            f"--params {PARAMS}",
            f"--scorefunction {sfxn}",
            f"--preset {preset}",
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
print("\nSweep summary:")
print(f"  Scorefunctions: {SCOREFUNCTIONS}")
print(f"  Repeats x Stages: {REPEATS_STAGES}")
print(f"  Cart_bonded weights: {CART_BONDED_WEIGHTS}")
print(f"  Bond geometry min: {BOND_GEOMETRY_MIN}")
print(f"  Plus preset tests (fast, balanced, thorough, aggressive)")
print(f"  Total jobs: {len(commands)}")
