#!/usr/bin/env python3
"""
Generate parameter sweep commands for fastmpnn_design.py

Sweeps over:
- JSON protocol files (fast, default, thorough, aggressive)
- MPNN temperatures
- Number of designs per round
- Design scope (primary only vs primary+secondary)
- Scorefunction combinations
- Conservation settings
"""

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# Paths
PROJECT_ROOT = str(_HERE.parents[2])
SCRIPT_DIR = f"{PROJECT_ROOT}/modules/step03__fastmpnndesign"
SWEEP_DIR = f"{SCRIPT_DIR}/hyper_param_sweep"
STEP02_JSON = f"{SCRIPT_DIR}/test/step02_outputs/input_pdb_aligned_relaxed_metrics.json"
PARAMS = f"{SCRIPT_DIR}/test/params/XDW.params"
OUTPUT_BASE = f"{SWEEP_DIR}/outputs_v2"  # Fresh outputs with working constraints

# Module execution command (needed for relative imports to work)
MODULE_CMD = f"cd {PROJECT_ROOT} && python -m modules.step03__fastmpnndesign.fastmpnn_design"

# Parameter sweep options (JSON protocols)
PROTOCOLS = ["fast", "default", "thorough", "aggressive", "design_only"]
MPNN_TEMPERATURES = [0.1, 0.2, 0.3]
MPNN_NUM_DESIGNS = [4, 8, 16]
DESIGN_SECONDARY = [False, True]
CONSERVATION = [True, False]

# Scorefunction combinations
SCOREFUNCTION_COMBOS = [
    ("ref2015_cart", "beta_jan25"),      # Default
    ("ref2015_cart", "ref2015"),         # All ref2015
]

commands = []
job_id = 0

# 1. Protocol sweep
for protocol in PROTOCOLS:
    job_id += 1
    out_name = f"job{job_id:03d}__protocol_{protocol}"
    out_dir = f"{OUTPUT_BASE}/{out_name}"

    cmd_parts = [
        MODULE_CMD,
        f"--step02_json {STEP02_JSON}",
        f"--params {PARAMS}",
        f"--protocol {protocol}",
        f"--output_dir {out_dir}",
        "--num_final_designs 5",
        "--max_runtime 3600",
    ]
    commands.append(" ".join(cmd_parts))

# 2. Temperature sweep (using default protocol)
for temp in MPNN_TEMPERATURES:
    job_id += 1
    out_name = f"job{job_id:03d}__temp_{temp}"
    out_dir = f"{OUTPUT_BASE}/{out_name}"

    cmd_parts = [
        MODULE_CMD,
        f"--step02_json {STEP02_JSON}",
        f"--params {PARAMS}",
        "--protocol default",
        f"--mpnn_temperature {temp}",
        f"--output_dir {out_dir}",
        "--num_final_designs 5",
        "--max_runtime 3600",
    ]
    commands.append(" ".join(cmd_parts))

# 3. Number of designs sweep
for num_designs in MPNN_NUM_DESIGNS:
    job_id += 1
    out_name = f"job{job_id:03d}__ndesigns_{num_designs}"
    out_dir = f"{OUTPUT_BASE}/{out_name}"

    cmd_parts = [
        MODULE_CMD,
        f"--step02_json {STEP02_JSON}",
        f"--params {PARAMS}",
        "--protocol default",
        f"--mpnn_num_designs {num_designs}",
        f"--output_dir {out_dir}",
        "--num_final_designs 5",
        "--max_runtime 3600",
    ]
    commands.append(" ".join(cmd_parts))

# 4. Design scope sweep
for design_secondary in DESIGN_SECONDARY:
    job_id += 1
    scope_name = "pri_sec" if design_secondary else "primary"
    out_name = f"job{job_id:03d}__scope_{scope_name}"
    out_dir = f"{OUTPUT_BASE}/{out_name}"

    cmd_parts = [
        MODULE_CMD,
        f"--step02_json {STEP02_JSON}",
        f"--params {PARAMS}",
        "--protocol default",
    ]
    if design_secondary:
        cmd_parts.append("--design_secondary_sphere")
    cmd_parts.extend([
        f"--output_dir {out_dir}",
        "--num_final_designs 5",
        "--max_runtime 3600",
    ])
    commands.append(" ".join(cmd_parts))

# 5. Conservation sweep
for conserve in CONSERVATION:
    job_id += 1
    cons_name = "conserve" if conserve else "no_conserve"
    out_name = f"job{job_id:03d}__{cons_name}"
    out_dir = f"{OUTPUT_BASE}/{out_name}"

    cmd_parts = [
        MODULE_CMD,
        f"--step02_json {STEP02_JSON}",
        f"--params {PARAMS}",
        "--protocol default",
    ]
    if not conserve:
        cmd_parts.append("--no_conserve_interactions")
    cmd_parts.extend([
        f"--output_dir {out_dir}",
        "--num_final_designs 5",
        "--max_runtime 3600",
    ])
    commands.append(" ".join(cmd_parts))

# 6. Scorefunction combination sweep
for sfxn_cart, sfxn_tors in SCOREFUNCTION_COMBOS:
    job_id += 1
    out_name = f"job{job_id:03d}__sfxn_{sfxn_cart.replace('_cart', '')}_{sfxn_tors}"
    out_dir = f"{OUTPUT_BASE}/{out_name}"

    cmd_parts = [
        MODULE_CMD,
        f"--step02_json {STEP02_JSON}",
        f"--params {PARAMS}",
        "--protocol default",
        f"--scorefunction_cart {sfxn_cart}",
        f"--scorefunction_torsional {sfxn_tors}",
        f"--output_dir {out_dir}",
        "--num_final_designs 5",
        "--max_runtime 3600",
    ]
    commands.append(" ".join(cmd_parts))

# 7. Combined sweeps (temperature x protocol)
for protocol in ["fast", "default"]:
    for temp in [0.1, 0.2]:
        job_id += 1
        out_name = f"job{job_id:03d}__{protocol}__temp{temp}"
        out_dir = f"{OUTPUT_BASE}/{out_name}"

        cmd_parts = [
            MODULE_CMD,
            f"--step02_json {STEP02_JSON}",
            f"--params {PARAMS}",
            f"--protocol {protocol}",
            f"--mpnn_temperature {temp}",
            f"--output_dir {out_dir}",
            "--num_final_designs 5",
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
print(f"  Protocols: {PROTOCOLS}")
print(f"  MPNN temperatures: {MPNN_TEMPERATURES}")
print(f"  MPNN num designs: {MPNN_NUM_DESIGNS}")
print(f"  Design secondary: {DESIGN_SECONDARY}")
print(f"  Conservation: {CONSERVATION}")
print(f"  Scorefunction combos: {SCOREFUNCTION_COMBOS}")
print(f"  Total jobs: {len(commands)}")
