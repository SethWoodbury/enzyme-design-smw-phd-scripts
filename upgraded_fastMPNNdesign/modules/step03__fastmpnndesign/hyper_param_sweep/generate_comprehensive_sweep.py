#!/usr/bin/env python3
"""
Comprehensive hyperparameter sweep generator for step03 FastMPNN design.

Generates hundreds of sweep commands covering:
- JSON protocol files (fast/default/thorough/etc.)
- MPNN temperature and design count variations
- Rosetta scorefunction combinations
- Relaxation parameters (cart_bonded, fa_rep, rounds, cycles)
- Global coordinate constraint options
- Design scope (primary, secondary)
- Conservation settings
- Custom multi-stage protocols

Each condition has 3 replicates for statistical analysis.

Key metrics being optimized:
1. Constrained atom RMSD (should be ~0)
2. Bond geometry for unconstrained catres (< 0.05A bond, < 7.5 deg angle)
3. CA RMSD vs step01 (minimize to stay close to AlphaFold3 prediction)
4. No clashes involving catalytic residues
5. No mutations to catalytic residues
6. HIS tautomer preservation
7. Good secondary structure and Dunbrack rotamers

Usage:
    python generate_comprehensive_sweep.py [--dry-run] [--subset SUBSET]

Options:
    --dry-run       Don't write files, just print summary
    --subset        Generate only a subset: "quick" (10 jobs), "medium" (50), "full" (all)
"""

import os
import sys
import json
import argparse
from itertools import product
from typing import List, Dict, Tuple

# Paths
PROJECT_ROOT = "/home/woodbuse/special_scripts/upgraded_fastMPNNdesign"
SCRIPT_DIR = f"{PROJECT_ROOT}/modules/step03__fastmpnndesign"
SWEEP_DIR = f"{SCRIPT_DIR}/hyper_param_sweep"
STEP02_JSON = f"{SCRIPT_DIR}/test/step02_outputs/input_pdb_aligned_relaxed_metrics.json"
PARAMS = f"{SCRIPT_DIR}/test/params/XDW.params"
STEP01_PDB = f"{SCRIPT_DIR}/test/step01_outputs/input_pdb_aligned.pdb"
OUTPUT_BASE = f"{SWEEP_DIR}/outputs"

# Module execution command
MODULE_CMD = f"cd {PROJECT_ROOT} && python -m modules.step03__fastmpnndesign.fastmpnn_design"
PROTOCOL_DIR = f"{SWEEP_DIR}/protocols"

# Control whether protocol files are written (disabled on --dry-run)
WRITE_PROTOCOL_FILES = True
_PROTOCOL_FILE_CACHE: Dict[str, str] = {}

# Number of replicates per condition
NUM_REPLICATES = 3

# =============================================================================
# PARAMETER SPACES
# =============================================================================

# Protocols (JSON basenames)
PROTOCOLS = [
    "fast",
    "balanced",
    "default",
    "thorough",
    "aggressive",
    "design_only",
    "geometry_only",
    "breadth",
    "depth",
    "iterative_refine",
    "progressive",
    "geometry_first",
    "design_secondary_shell",
]

# MPNN parameters
MPNN_TEMPERATURES = [0.05, 0.1, 0.15, 0.2, 0.3]
MPNN_NUM_DESIGNS = [2, 4, 8, 16, 32]
MPNN_BATCH_SIZES = [1, 4]  # Lower = more diversity

# Rosetta scorefunctions
CART_SCOREFUNCTIONS = ["ref2015_cart"]  # beta_nov16_cart available but less tested
TORSIONAL_SCOREFUNCTIONS = ["beta_jan25", "ref2015"]

# Relaxation parameters
CART_BONDED_WEIGHTS = [1.0, 2.0, 4.0]  # Default 2.0
FA_REP_WEIGHTS = [0.3, 0.55, 0.8]  # Default 0.55, lower = more permissive
RELAX_ROUNDS = [1, 3, 5]  # Number of FastRelax outer cycles
COORD_CST_WEIGHTS = [500.0, 750.0, 1000.0]  # For constrained atoms
COORD_CST_STDEVS = [0.01, 0.025, 0.05]  # Tighter = less movement allowed

# Global constraint options (for non-catres atoms)
GLOBAL_COORD_CST_WEIGHTS = [0.0, 0.1, 1.0]  # 0 = no global constraints
GLOBAL_COORD_CST_STDEVS = [0.3, 0.5, 1.0]

# Design scope
DESIGN_SECONDARY_OPTIONS = [False, True]

# Conservation
CONSERVATION_OPTIONS = [True, False]

# Custom protocol strings (for specific multi-stage designs)
CUSTOM_PROTOCOLS = [
    # Iterative refinement patterns
    ("custom_iterative_refine_short",
     "mpnn:T0.2:N1 -> torsional_relax:R1S2 -> mpnn:T0.15:N1 -> torsional_relax:R1S2 -> mpnn:T0.1:N10 -> torsional_relax:R2S3"),

    # Geometry-first with conservative design
    ("custom_geometry_first",
     "cart_relax:R2S3 -> mpnn:T0.1:N20 -> torsional_relax:R2S3"),

    # Breadth-focused (high diversity)
    ("custom_breadth_highT",
     "mpnn:T0.3:N32 -> torsional_relax:R1S2"),

    # Depth-focused (conservative)
    ("custom_depth_lowT",
     "mpnn:T0.05:N8 -> torsional_relax:R2S3 -> mpnn:T0.05:N8 -> torsional_relax:R2S3"),

    # Multi-round with selection
    ("custom_multiround",
     "mpnn:T0.2:N8 -> torsional_relax:R1S2 -> mpnn:T0.1:N8 -> torsional_relax:R2S3"),

    # Secondary sphere focus
    ("custom_secondary_focus",
     "mpnn:T0.1:N4 -> torsional_relax:R1S2 -> mpnn:T0.15:N16 -> torsional_relax:R1S3"),
]

# =============================================================================
# COMMAND GENERATION
# =============================================================================

def make_base_cmd(output_dir: str, extra_args: List[str] = None) -> str:
    """Create base command with required arguments."""
    cmd_parts = [
        MODULE_CMD,
        f'--step02_json "{STEP02_JSON}"',
        f'--params "{PARAMS}"',
        f'--step01_pdb "{STEP01_PDB}"',
        f'--output_dir "{output_dir}"',
        "--num_final_designs 5",
        "--max_runtime 7200",  # 2 hours max
    ]
    if extra_args:
        cmd_parts.extend(extra_args)
    return " ".join(cmd_parts)


def protocol_file_for_string(name_hint: str, protocol_str: str) -> str:
    """Write a protocol .txt file and return its path."""
    import hashlib
    import re

    key = protocol_str.strip()
    if key in _PROTOCOL_FILE_CACHE:
        return _PROTOCOL_FILE_CACHE[key]

    os.makedirs(PROTOCOL_DIR, exist_ok=True)
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name_hint or "protocol").strip("_")
    filename = f"{safe_name}_{digest}.txt"
    path = os.path.join(PROTOCOL_DIR, filename)

    if WRITE_PROTOCOL_FILES:
        with open(path, "w") as f:
            f.write(key + "\n")

    _PROTOCOL_FILE_CACHE[key] = path
    return path


def generate_protocol_sweep(job_counter: int) -> Tuple[List[str], int]:
    """Generate sweep over JSON protocols."""
    commands = []

    for protocol in PROTOCOLS:
        for rep in range(1, NUM_REPLICATES + 1):
            job_counter += 1
            out_name = f"job{job_counter:04d}__protocol_{protocol}__rep{rep}"
            out_dir = f"{OUTPUT_BASE}/{out_name}"

            cmd = make_base_cmd(out_dir, [f"--protocol {protocol}"])
            commands.append(cmd)

    return commands, job_counter


def generate_mpnn_temperature_sweep(job_counter: int) -> Tuple[List[str], int]:
    """Generate sweep over MPNN temperatures."""
    commands = []

    for temp in MPNN_TEMPERATURES:
        for rep in range(1, NUM_REPLICATES + 1):
            job_counter += 1
            out_name = f"job{job_counter:04d}__temp_{temp}__rep{rep}"
            out_dir = f"{OUTPUT_BASE}/{out_name}"

            cmd = make_base_cmd(out_dir, [
                "--protocol balanced",
                f"--mpnn_temperature {temp}",
            ])
            commands.append(cmd)

    return commands, job_counter


def generate_mpnn_designs_sweep(job_counter: int) -> Tuple[List[str], int]:
    """Generate sweep over number of designs per MPNN round."""
    commands = []

    for n_designs in MPNN_NUM_DESIGNS:
        for rep in range(1, NUM_REPLICATES + 1):
            job_counter += 1
            out_name = f"job{job_counter:04d}__ndesigns_{n_designs}__rep{rep}"
            out_dir = f"{OUTPUT_BASE}/{out_name}"

            cmd = make_base_cmd(out_dir, [
                "--protocol balanced",
                f"--mpnn_num_designs {n_designs}",
            ])
            commands.append(cmd)

    return commands, job_counter


def generate_scorefunction_sweep(job_counter: int) -> Tuple[List[str], int]:
    """Generate sweep over scorefunction combinations."""
    commands = []

    for cart_sfxn in CART_SCOREFUNCTIONS:
        for tors_sfxn in TORSIONAL_SCOREFUNCTIONS:
            for rep in range(1, NUM_REPLICATES + 1):
                job_counter += 1
                out_name = f"job{job_counter:04d}__sfxn_{cart_sfxn.replace('_cart','')}_{tors_sfxn}__rep{rep}"
                out_dir = f"{OUTPUT_BASE}/{out_name}"

                cmd = make_base_cmd(out_dir, [
                    "--protocol balanced",
                    f"--scorefunction_cart {cart_sfxn}",
                    f"--scorefunction_torsional {tors_sfxn}",
                ])
                commands.append(cmd)

    return commands, job_counter


def generate_cart_bonded_sweep(job_counter: int) -> Tuple[List[str], int]:
    """Generate sweep over cart_bonded weights."""
    commands = []

    for weight in CART_BONDED_WEIGHTS:
        for rep in range(1, NUM_REPLICATES + 1):
            job_counter += 1
            out_name = f"job{job_counter:04d}__cart_bonded_{weight}__rep{rep}"
            out_dir = f"{OUTPUT_BASE}/{out_name}"

            cmd = make_base_cmd(out_dir, [
                "--protocol balanced",
                f"--cart_bonded_weight {weight}",
            ])
            commands.append(cmd)

    return commands, job_counter


def generate_fa_rep_sweep(job_counter: int) -> Tuple[List[str], int]:
    """Generate sweep over fa_rep weights (clash tolerance)."""
    commands = []

    for weight in FA_REP_WEIGHTS:
        for rep in range(1, NUM_REPLICATES + 1):
            job_counter += 1
            out_name = f"job{job_counter:04d}__fa_rep_{weight}__rep{rep}"
            out_dir = f"{OUTPUT_BASE}/{out_name}"

            cmd = make_base_cmd(out_dir, [
                "--protocol balanced",
                f"--fa_rep_weight {weight}",
            ])
            commands.append(cmd)

    return commands, job_counter


def generate_relax_rounds_sweep(job_counter: int) -> Tuple[List[str], int]:
    """Generate sweep over relaxation rounds."""
    commands = []

    for rounds in RELAX_ROUNDS:
        for rep in range(1, NUM_REPLICATES + 1):
            job_counter += 1
            out_name = f"job{job_counter:04d}__relax_rounds_{rounds}__rep{rep}"
            out_dir = f"{OUTPUT_BASE}/{out_name}"

            cmd = make_base_cmd(out_dir, [
                "--protocol balanced",
                f"--relax_rounds {rounds}",
            ])
            commands.append(cmd)

    return commands, job_counter


def generate_coord_cst_sweep(job_counter: int) -> Tuple[List[str], int]:
    """Generate sweep over coordinate constraint parameters."""
    commands = []

    for weight, stdev in product(COORD_CST_WEIGHTS, COORD_CST_STDEVS):
        for rep in range(1, NUM_REPLICATES + 1):
            job_counter += 1
            out_name = f"job{job_counter:04d}__cst_w{weight}_sd{stdev}__rep{rep}"
            out_dir = f"{OUTPUT_BASE}/{out_name}"

            cmd = make_base_cmd(out_dir, [
                "--protocol balanced",
                f"--coord_cst_weight {weight}",
                f"--coord_cst_stdev {stdev}",
            ])
            commands.append(cmd)

    return commands, job_counter


def generate_global_cst_sweep(job_counter: int) -> Tuple[List[str], int]:
    """Generate sweep over global coordinate constraints."""
    commands = []

    for weight in GLOBAL_COORD_CST_WEIGHTS:
        if weight == 0.0:
            # No global constraints
            for rep in range(1, NUM_REPLICATES + 1):
                job_counter += 1
                out_name = f"job{job_counter:04d}__global_cst_off__rep{rep}"
                out_dir = f"{OUTPUT_BASE}/{out_name}"

                cmd = make_base_cmd(out_dir, ["--protocol balanced"])
                commands.append(cmd)
        else:
            for stdev in GLOBAL_COORD_CST_STDEVS:
                for rep in range(1, NUM_REPLICATES + 1):
                    job_counter += 1
                    out_name = f"job{job_counter:04d}__global_cst_w{weight}_sd{stdev}__rep{rep}"
                    out_dir = f"{OUTPUT_BASE}/{out_name}"

                    cmd = make_base_cmd(out_dir, [
                        "--protocol balanced",
                        f"--global_coord_cst_weight {weight}",
                        f"--global_coord_cst_stdev {stdev}",
                    ])
                    commands.append(cmd)

    return commands, job_counter


def generate_design_scope_sweep(job_counter: int) -> Tuple[List[str], int]:
    """Generate sweep over design scope (primary vs secondary sphere)."""
    commands = []

    for design_secondary in DESIGN_SECONDARY_OPTIONS:
        for rep in range(1, NUM_REPLICATES + 1):
            job_counter += 1
            scope = "pri_sec" if design_secondary else "primary"
            out_name = f"job{job_counter:04d}__scope_{scope}__rep{rep}"
            out_dir = f"{OUTPUT_BASE}/{out_name}"

            extra_args = ["--protocol balanced"]
            if design_secondary:
                extra_args.append("--design_secondary_sphere")

            cmd = make_base_cmd(out_dir, extra_args)
            commands.append(cmd)

    return commands, job_counter


def generate_conservation_sweep(job_counter: int) -> Tuple[List[str], int]:
    """Generate sweep over conservation settings."""
    commands = []

    for conserve in CONSERVATION_OPTIONS:
        for rep in range(1, NUM_REPLICATES + 1):
            job_counter += 1
            cons = "conserve" if conserve else "no_conserve"
            out_name = f"job{job_counter:04d}__{cons}__rep{rep}"
            out_dir = f"{OUTPUT_BASE}/{out_name}"

            extra_args = ["--protocol balanced"]
            if not conserve:
                extra_args.append("--no_conserve_interactions")

            cmd = make_base_cmd(out_dir, extra_args)
            commands.append(cmd)

    return commands, job_counter


def generate_custom_protocol_sweep(job_counter: int) -> Tuple[List[str], int]:
    """Generate sweep over custom protocols."""
    commands = []

    for name, protocol in CUSTOM_PROTOCOLS:
        for rep in range(1, NUM_REPLICATES + 1):
            job_counter += 1
            out_name = f"job{job_counter:04d}__custom_{name}__rep{rep}"
            out_dir = f"{OUTPUT_BASE}/{out_name}"

            proto_file = protocol_file_for_string(name, protocol)
            cmd = make_base_cmd(out_dir, [f'--protocol_file "{proto_file}"'])
            commands.append(cmd)

    return commands, job_counter


def generate_combined_sweep(job_counter: int) -> Tuple[List[str], int]:
    """Generate combined sweeps (key parameter combinations)."""
    commands = []

    # Temperature x Design scope
    for temp in [0.1, 0.2]:
        for design_secondary in [False, True]:
            for rep in range(1, NUM_REPLICATES + 1):
                job_counter += 1
                scope = "pri_sec" if design_secondary else "primary"
                out_name = f"job{job_counter:04d}__temp{temp}_scope_{scope}__rep{rep}"
                out_dir = f"{OUTPUT_BASE}/{out_name}"

                extra_args = [
                    "--protocol balanced",
                    f"--mpnn_temperature {temp}",
                ]
                if design_secondary:
                    extra_args.append("--design_secondary_sphere")

                cmd = make_base_cmd(out_dir, extra_args)
                commands.append(cmd)

    # Breadth vs Depth configurations
    breadth_configs = [
        ("breadth_high_temp", {"extra_args": ["--mpnn_temperature 0.3", "--mpnn_num_designs 32"]}),
        ("breadth_multi_round", {"protocol_str": "mpnn:T0.3:N16 -> torsional_relax:R1S2 -> mpnn:T0.2:N16"}),
    ]

    depth_configs = [
        ("depth_low_temp", {"extra_args": ["--mpnn_temperature 0.05", "--mpnn_num_designs 8"]}),
        ("depth_multi_relax", {"protocol_str": "mpnn:T0.1:N4 -> torsional_relax:R3S3 -> mpnn:T0.1:N4 -> torsional_relax:R3S3"}),
    ]

    for config_name, config in breadth_configs + depth_configs:
        for rep in range(1, NUM_REPLICATES + 1):
            job_counter += 1
            out_name = f"job{job_counter:04d}__{config_name}__rep{rep}"
            out_dir = f"{OUTPUT_BASE}/{out_name}"

            if "protocol_str" in config:
                proto_file = protocol_file_for_string(config_name, config["protocol_str"])
                extra_args = [f'--protocol_file "{proto_file}"']
            else:
                extra_args = ["--protocol balanced"] + config.get("extra_args", [])

            cmd = make_base_cmd(out_dir, extra_args)
            commands.append(cmd)

    # Relaxation stringency combinations
    stringency_configs = [
        ("relax_light", ["--relax_rounds 1", "--cart_bonded_weight 1.0"]),
        ("relax_moderate", ["--relax_rounds 3", "--cart_bonded_weight 2.0"]),
        ("relax_thorough", ["--relax_rounds 5", "--cart_bonded_weight 4.0"]),
    ]

    for config_name, extra_args in stringency_configs:
        for rep in range(1, NUM_REPLICATES + 1):
            job_counter += 1
            out_name = f"job{job_counter:04d}__{config_name}__rep{rep}"
            out_dir = f"{OUTPUT_BASE}/{out_name}"

            cmd = make_base_cmd(out_dir, ["--protocol balanced"] + extra_args)
            commands.append(cmd)

    return commands, job_counter


def generate_quick_subset() -> List[str]:
    """Generate quick test subset (10 jobs)."""
    commands = []
    job_counter = 0

    # Just basic protocols, 1 replicate
    for protocol in ["fast", "balanced", "design_only"]:
        job_counter += 1
        out_name = f"job{job_counter:04d}__protocol_{protocol}__rep1"
        out_dir = f"{OUTPUT_BASE}/{out_name}"
        cmd = make_base_cmd(out_dir, [f"--protocol {protocol}"])
        commands.append(cmd)

    # Key temperatures
    for temp in [0.1, 0.2]:
        job_counter += 1
        out_name = f"job{job_counter:04d}__temp_{temp}__rep1"
        out_dir = f"{OUTPUT_BASE}/{out_name}"
        cmd = make_base_cmd(out_dir, ["--protocol balanced", f"--mpnn_temperature {temp}"])
        commands.append(cmd)

    # Design scope
    for scope in ["primary", "secondary"]:
        job_counter += 1
        extra = ["--protocol balanced"]
        if scope == "secondary":
            extra.append("--design_secondary_sphere")
        out_name = f"job{job_counter:04d}__scope_{scope}__rep1"
        out_dir = f"{OUTPUT_BASE}/{out_name}"
        cmd = make_base_cmd(out_dir, extra)
        commands.append(cmd)

    # Custom protocols
    for name, protocol in CUSTOM_PROTOCOLS[:3]:
        job_counter += 1
        out_name = f"job{job_counter:04d}__custom_{name}__rep1"
        out_dir = f"{OUTPUT_BASE}/{out_name}"
        proto_file = protocol_file_for_string(name, protocol)
        cmd = make_base_cmd(out_dir, [f'--protocol_file "{proto_file}"'])
        commands.append(cmd)

    return commands


def generate_medium_subset() -> List[str]:
    """Generate medium test subset (~50 jobs)."""
    commands = []
    job_counter = 0

    # Protocols with 2 replicates
    cmds, job_counter = generate_protocol_sweep(job_counter)
    commands.extend(cmds[:len(PROTOCOLS) * 2])  # 2 reps for basic protocols

    # Key parameter sweeps with 1 replicate
    for temp in [0.1, 0.15, 0.2]:
        job_counter += 1
        out_name = f"job{job_counter:04d}__temp_{temp}__rep1"
        out_dir = f"{OUTPUT_BASE}/{out_name}"
        cmd = make_base_cmd(out_dir, ["--protocol balanced", f"--mpnn_temperature {temp}"])
        commands.append(cmd)

    for n_des in [4, 8, 16]:
        job_counter += 1
        out_name = f"job{job_counter:04d}__ndes_{n_des}__rep1"
        out_dir = f"{OUTPUT_BASE}/{out_name}"
        cmd = make_base_cmd(out_dir, ["--protocol balanced", f"--mpnn_num_designs {n_des}"])
        commands.append(cmd)

    # Relax parameters
    for rounds in [1, 3, 5]:
        job_counter += 1
        out_name = f"job{job_counter:04d}__rounds_{rounds}__rep1"
        out_dir = f"{OUTPUT_BASE}/{out_name}"
        cmd = make_base_cmd(out_dir, ["--protocol balanced", f"--relax_rounds {rounds}"])
        commands.append(cmd)

    # Cart bonded
    for weight in [1.0, 2.0, 4.0]:
        job_counter += 1
        out_name = f"job{job_counter:04d}__cart_bonded_{weight}__rep1"
        out_dir = f"{OUTPUT_BASE}/{out_name}"
        cmd = make_base_cmd(out_dir, ["--protocol balanced", f"--cart_bonded_weight {weight}"])
        commands.append(cmd)

    # Custom protocols with 2 replicates
    for name, protocol in CUSTOM_PROTOCOLS:
        for rep in [1, 2]:
            job_counter += 1
            out_name = f"job{job_counter:04d}__custom_{name}__rep{rep}"
            out_dir = f"{OUTPUT_BASE}/{out_name}"
            proto_file = protocol_file_for_string(name, protocol)
            cmd = make_base_cmd(out_dir, [f'--protocol_file "{proto_file}"'])
            commands.append(cmd)

    # Combined configs
    combined_cmds, job_counter = generate_combined_sweep(job_counter)
    commands.extend(combined_cmds[:20])  # First 20 combined

    return commands


def generate_full_sweep() -> List[str]:
    """Generate full sweep with all parameters and replicates."""
    commands = []
    job_counter = 0

    # All individual parameter sweeps
    sweeps = [
        generate_protocol_sweep,
        generate_mpnn_temperature_sweep,
        generate_mpnn_designs_sweep,
        generate_scorefunction_sweep,
        generate_cart_bonded_sweep,
        generate_fa_rep_sweep,
        generate_relax_rounds_sweep,
        generate_coord_cst_sweep,
        generate_global_cst_sweep,
        generate_design_scope_sweep,
        generate_conservation_sweep,
        generate_custom_protocol_sweep,
        generate_combined_sweep,
    ]

    for sweep_func in sweeps:
        cmds, job_counter = sweep_func(job_counter)
        commands.extend(cmds)

    return commands


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate hyperparameter sweep commands")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files")
    parser.add_argument("--subset", choices=["quick", "medium", "full"], default="full",
                       help="Subset size: quick (~10), medium (~50), full (all)")
    args = parser.parse_args()

    global WRITE_PROTOCOL_FILES
    WRITE_PROTOCOL_FILES = not args.dry_run

    print("=" * 60)
    print("Step03 FastMPNN Comprehensive Hyperparameter Sweep Generator")
    print("=" * 60)

    # Generate commands based on subset
    if args.subset == "quick":
        commands = generate_quick_subset()
    elif args.subset == "medium":
        commands = generate_medium_subset()
    else:
        commands = generate_full_sweep()

    print(f"\nGenerated {len(commands)} commands ({args.subset} subset)")
    print(f"Replicates per condition: {NUM_REPLICATES if args.subset == 'full' else 'varies'}")

    # Summary
    print("\nSweep parameters:")
    print(f"  Protocols: {PROTOCOLS}")
    print(f"  MPNN temperatures: {MPNN_TEMPERATURES}")
    print(f"  MPNN num designs: {MPNN_NUM_DESIGNS}")
    print(f"  Cart scorefunctions: {CART_SCOREFUNCTIONS}")
    print(f"  Torsional scorefunctions: {TORSIONAL_SCOREFUNCTIONS}")
    print(f"  Cart bonded weights: {CART_BONDED_WEIGHTS}")
    print(f"  FA rep weights: {FA_REP_WEIGHTS}")
    print(f"  Relax rounds: {RELAX_ROUNDS}")
    print(f"  Custom protocols: {len(CUSTOM_PROTOCOLS)}")

    if args.dry_run:
        print("\n[DRY RUN] Not writing files")
        print("\nFirst 5 commands:")
        for cmd in commands[:5]:
            print(f"  {cmd[:100]}...")
        return

    # Create directories
    os.makedirs(f"{SWEEP_DIR}/cmds", exist_ok=True)
    os.makedirs(f"{SWEEP_DIR}/logs", exist_ok=True)
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # Write commands file
    cmds_file = f"{SWEEP_DIR}/cmds/sweep_{args.subset}_commands.txt"
    with open(cmds_file, "w") as f:
        for cmd in commands:
            f.write(cmd + "\n")

    print(f"\nCommands written to: {cmds_file}")

    # Also write as "active" commands file for submission
    active_file = f"{SWEEP_DIR}/cmds/all_sweep_commands.txt"
    with open(active_file, "w") as f:
        for cmd in commands:
            f.write(cmd + "\n")

    print(f"Active commands: {active_file}")

    # Write metadata
    metadata = {
        "subset": args.subset,
        "num_commands": len(commands),
        "replicates": NUM_REPLICATES if args.subset == "full" else "varies",
        "protocols": PROTOCOLS,
        "mpnn_temperatures": MPNN_TEMPERATURES,
        "mpnn_num_designs": MPNN_NUM_DESIGNS,
        "custom_protocols": CUSTOM_PROTOCOLS,
        "output_base": OUTPUT_BASE,
    }

    metadata_file = f"{SWEEP_DIR}/cmds/sweep_{args.subset}_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata: {metadata_file}")

    print(f"\nTo submit jobs:")
    print(f"  cd {SWEEP_DIR}")
    print(f"  ./submit_array.sh  # For SLURM array job")
    print(f"  # OR")
    print(f"  ./submit_sweep.sh 4  # Run 4 parallel local jobs")


if __name__ == "__main__":
    main()
