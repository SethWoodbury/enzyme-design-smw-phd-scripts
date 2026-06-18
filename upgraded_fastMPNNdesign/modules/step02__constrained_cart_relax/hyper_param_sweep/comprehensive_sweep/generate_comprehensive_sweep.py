#!/usr/bin/env python3
"""
Generate comprehensive hyperparameter sweep for constrained_cart_relax.py

This sweep tests:
- fa_rep scaling and ramping (new feature)
- cart_bonded weight and ceiling
- FastRelax configurations (repeats × stages)
- Bond geometry minimization

Optimization priorities:
1. Minimize max bond length/angle deviations (unconstrained atoms)
2. Maximize catalytic residues passing thresholds
3. Ensure ligand/constrained RMSD ≈ 0.00
4. Minimize CA RMSD (target < 0.75-1.00 Å)
5. Minimize runtime and clashes

N=3 replicates per condition for Monte Carlo variance estimation.
"""

import os
import itertools
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
MODULE_DIR = SCRIPT_DIR.parent.parent
RELAX_SCRIPT = MODULE_DIR / "constrained_cart_relax.py"
STEP01_JSON = MODULE_DIR / "test" / "step01_outputs" / "input_pdb_recommended_atom_cst.json"
OUTPUT_BASE = SCRIPT_DIR / "outputs"
CMD_DIR = SCRIPT_DIR / "cmds"
LOG_DIR = SCRIPT_DIR / "logs"

# Create directories
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
CMD_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SWEEP PARAMETERS
# =============================================================================

# Score term scaling (new feature - test fa_rep reduction)
FA_REP_SCALES = [0.3, 0.5, 0.7, 1.0]
RAMP_FA_REP = [True, False]
FA_REP_MIN_SCALES = [0.1, 0.2]  # Only matters when ramp=True, but include anyway

# Cart_bonded settings
CART_BONDED_WEIGHTS = [2.0, 2.5, 3.0, 3.5]
CART_BONDED_MAX = [4.0, 5.0]

# FastRelax configurations (repeats, stages)
FASTRELAX_CONFIGS = [
    (1, 3),   # Minimal: 3 internal rounds
    (2, 3),   # Light: 6 internal rounds
    (3, 3),   # Medium: 9 internal rounds
    (3, 5),   # Standard: 15 internal rounds
]

# Bond geometry minimization
BOND_GEOMETRY_MIN = [True, False]

# Fixed parameters (based on previous sweeps)
FIXED_PARAMS = {
    "scorefunction": "ref2015_cart",
    "coord_cst_weight": 750.0,
    "coord_cst_stdev": 0.01,
    "mobile_radius": 10.0,
    "max_adaptive_rounds": 5,  # Reduced for sweep efficiency
    "auto_expand_mobile": True,
    "max_runtime": 3600,
}

# Replicates
N_REPLICATES = 3

# =============================================================================
# GENERATE SWEEP
# =============================================================================

def generate_job_id(params: dict, replicate: int) -> str:
    """Generate unique job ID from parameters."""
    parts = [
        f"fr{params['fa_rep_scale']:.1f}",
        f"ramp{'Y' if params['ramp_fa_rep'] else 'N'}",
        f"cb{params['cart_bonded_weight']:.1f}",
        f"cbm{params['cart_bonded_max']:.1f}",
        f"rx{params['repeats']}x{params['stages']}",
        f"bg{'Y' if params['bond_geometry_min'] else 'N'}",
        f"r{replicate}",
    ]
    return "_".join(parts)


# Apptainer container for PyRosetta (universal.sif has newer PyRosetta 2026.03 with threading support)
APPTAINER_CMD = "apptainer exec /net/software/containers/universal.sif python"

# Params file for ligand
PARAMS_FILE = MODULE_DIR / "test" / "params" / "XDW.params"


def generate_command(params: dict, job_id: str) -> str:
    """Generate command line for a single job."""
    output_dir = OUTPUT_BASE / job_id

    cmd_parts = [
        f"{APPTAINER_CMD} {RELAX_SCRIPT}",
        f"--step01_json {STEP01_JSON}",
        f"--params {PARAMS_FILE}",
        f"--output_dir {output_dir}",
        # Score term scaling
        f"--fa_rep_scale {params['fa_rep_scale']}",
        f"--fa_rep_min_scale {params['fa_rep_min_scale']}",
    ]

    if params['ramp_fa_rep']:
        cmd_parts.append("--ramp_fa_rep")

    cmd_parts.extend([
        # Cart_bonded
        f"--cart_bonded_weight {params['cart_bonded_weight']}",
        f"--cart_bonded_max {params['cart_bonded_max']}",
        # FastRelax
        f"--fastrelax_repeats {params['repeats']}",
        f"--fastrelax_ramp_stages {params['stages']}",
    ])

    # Bond geometry minimization
    if params['bond_geometry_min']:
        cmd_parts.append("--enable_bond_geometry_min")
    else:
        cmd_parts.append("--disable_bond_geometry_min")

    # Fixed parameters
    for key, value in FIXED_PARAMS.items():
        if isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{key}")
        else:
            cmd_parts.append(f"--{key} {value}")

    return " ".join(cmd_parts)


def main():
    """Generate all sweep commands."""

    # Calculate total combinations
    # When ramp=False, fa_rep_min_scale doesn't matter, so we group those

    all_commands = []
    job_ids = []

    for fa_rep_scale in FA_REP_SCALES:
        for ramp_fa_rep in RAMP_FA_REP:
            # When ramp=False, only use one fa_rep_min_scale value
            min_scales = FA_REP_MIN_SCALES if ramp_fa_rep else [0.2]

            for fa_rep_min_scale in min_scales:
                for cart_bonded_weight in CART_BONDED_WEIGHTS:
                    for cart_bonded_max in CART_BONDED_MAX:
                        for repeats, stages in FASTRELAX_CONFIGS:
                            for bond_geometry_min in BOND_GEOMETRY_MIN:
                                for replicate in range(1, N_REPLICATES + 1):
                                    params = {
                                        'fa_rep_scale': fa_rep_scale,
                                        'ramp_fa_rep': ramp_fa_rep,
                                        'fa_rep_min_scale': fa_rep_min_scale,
                                        'cart_bonded_weight': cart_bonded_weight,
                                        'cart_bonded_max': cart_bonded_max,
                                        'repeats': repeats,
                                        'stages': stages,
                                        'bond_geometry_min': bond_geometry_min,
                                    }

                                    job_id = generate_job_id(params, replicate)
                                    cmd = generate_command(params, job_id)

                                    all_commands.append(cmd)
                                    job_ids.append(job_id)

    # Write all commands to file
    cmd_file = CMD_DIR / "all_sweep_commands.txt"
    with open(cmd_file, 'w') as f:
        for cmd in all_commands:
            f.write(cmd + "\n")

    # Write job IDs for reference
    id_file = CMD_DIR / "job_ids.txt"
    with open(id_file, 'w') as f:
        for i, job_id in enumerate(job_ids, 1):
            f.write(f"{i}\t{job_id}\n")

    # Calculate statistics
    n_conditions = len(all_commands) // N_REPLICATES

    print("=" * 70)
    print("COMPREHENSIVE HYPERPARAMETER SWEEP GENERATED")
    print("=" * 70)
    print()
    print("Parameter space:")
    print(f"  fa_rep_scale:        {FA_REP_SCALES}")
    print(f"  ramp_fa_rep:         {RAMP_FA_REP}")
    print(f"  fa_rep_min_scale:    {FA_REP_MIN_SCALES} (only when ramp=True)")
    print(f"  cart_bonded_weight:  {CART_BONDED_WEIGHTS}")
    print(f"  cart_bonded_max:     {CART_BONDED_MAX}")
    print(f"  FastRelax configs:   {FASTRELAX_CONFIGS}")
    print(f"  bond_geometry_min:   {BOND_GEOMETRY_MIN}")
    print()
    print(f"Fixed parameters:")
    for key, value in FIXED_PARAMS.items():
        print(f"  {key}: {value}")
    print()
    print(f"Total unique conditions: {n_conditions}")
    print(f"Replicates per condition: {N_REPLICATES}")
    print(f"Total jobs: {len(all_commands)}")
    print()
    print(f"Commands written to: {cmd_file}")
    print(f"Job IDs written to: {id_file}")
    print()

    # Estimate runtime
    # Based on previous sweeps, average ~700-1500s per job
    avg_runtime_min = 15  # minutes per job (conservative)
    total_cpu_hours = (len(all_commands) * avg_runtime_min) / 60
    print(f"Estimated total CPU time: {total_cpu_hours:.0f} hours")
    print(f"  (at ~{avg_runtime_min} min/job average)")
    print()
    print("Next steps:")
    print("  1. Review submit_sweep.sh")
    print("  2. sbatch submit_sweep.sh")
    print("  3. After completion: python analyze_comprehensive_sweep.py")


if __name__ == "__main__":
    main()
