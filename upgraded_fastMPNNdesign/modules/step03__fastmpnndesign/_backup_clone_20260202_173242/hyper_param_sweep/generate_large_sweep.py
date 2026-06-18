#!/usr/bin/env python3
"""
Generate a large-scale hyperparameter sweep for FastMPNN design.

This generates ~400-600 jobs systematically exploring:
- Temperature ranges (0.05-0.4)
- Number of designs (1-64)
- Relaxation configurations
- Protocol architectures (single-shot, multi-round, annealing)
- Constraint variations
- Scorefunction choices

Usage:
    python generate_large_sweep.py --output cmds/large_sweep_commands.txt
    python generate_large_sweep.py --dry-run  # Show counts only
"""

import argparse
import itertools
from pathlib import Path
from typing import List, Tuple

# Base paths
SCRIPT_DIR = Path(__file__).parent
MODULE_DIR = SCRIPT_DIR.parent
BASE_DIR = MODULE_DIR.parent.parent

# Test inputs (from step02)
STEP02_JSON = MODULE_DIR / "test/step02_outputs/input_pdb_aligned_relaxed_metrics.json"
PARAMS_FILE = MODULE_DIR / "test/params/XDW.params"
STEP01_PDB = MODULE_DIR / "test/step01_outputs/input_pdb_aligned.pdb"
OUTPUT_BASE = SCRIPT_DIR / "outputs_v2"  # Fresh outputs with working constraints


def build_command(
    job_name: str,
    protocol: str = None,
    preset: str = None,
    extra_args: dict = None,
    num_designs: int = 5,
) -> str:
    """Build a single sweep command."""
    output_dir = OUTPUT_BASE / job_name

    cmd = [
        f'cd {BASE_DIR}',
        '&&',
        'python -m modules.step03__fastmpnndesign.fastmpnn_design',
        f'--step02_json "{STEP02_JSON}"',
        f'--params "{PARAMS_FILE}"',
        f'--step01_pdb "{STEP01_PDB}"',
        f'--output_dir "{output_dir}"',
        f'--num_final_designs {num_designs}',
        '--max_runtime 7200',
    ]

    if preset:
        cmd.append(f'--preset {preset}')
    elif protocol:
        cmd.append(f'--protocol "{protocol}"')

    if extra_args:
        for key, value in extra_args.items():
            if value is not None:
                cmd.append(f'--{key} {value}')

    return ' '.join(cmd)


def generate_single_shot_protocols() -> List[Tuple[str, str]]:
    """Generate single-shot MPNN -> relax protocols."""
    jobs = []

    # Temperature sweep with fixed N
    temperatures = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    for temp in temperatures:
        protocol = f"mpnn:T{temp}:N16 -> torsional_relax:R1S3"
        jobs.append((f"single_T{temp}_N16", protocol))

    # N sweep with fixed temperature
    n_values = [1, 2, 4, 8, 16, 32, 64]
    for n in n_values:
        protocol = f"mpnn:T0.2:N{n} -> torsional_relax:R1S3"
        jobs.append((f"single_T0.2_N{n}", protocol))

    # Temperature × N grid (subset)
    temp_subset = [0.1, 0.2, 0.3]
    n_subset = [4, 16, 32]
    for temp in temp_subset:
        for n in n_subset:
            protocol = f"mpnn:T{temp}:N{n} -> torsional_relax:R1S3"
            jobs.append((f"single_T{temp}_N{n}", protocol))

    # Relax configuration sweep
    relax_configs = ["R1S2", "R1S3", "R2S2", "R2S3", "R3S3"]
    for config in relax_configs:
        protocol = f"mpnn:T0.2:N16 -> torsional_relax:{config}"
        jobs.append((f"single_relax_{config}", protocol))

    return jobs


def generate_two_round_protocols() -> List[Tuple[str, str]]:
    """Generate two-round annealing protocols."""
    jobs = []

    # Temperature annealing pairs
    temp_pairs = [
        (0.3, 0.1), (0.3, 0.15), (0.3, 0.2),
        (0.25, 0.1), (0.25, 0.15),
        (0.2, 0.1), (0.2, 0.05),
        (0.15, 0.05), (0.15, 0.1),
    ]

    for t1, t2 in temp_pairs:
        # Standard annealing
        protocol = f"mpnn:T{t1}:N8 -> torsional_relax:R1S2 -> mpnn:T{t2}:N16 -> torsional_relax:R2S3"
        jobs.append((f"anneal_T{t1}_to_T{t2}", protocol))

        # High N version
        protocol = f"mpnn:T{t1}:N16 -> torsional_relax:R1S2 -> mpnn:T{t2}:N32 -> torsional_relax:R2S3"
        jobs.append((f"anneal_T{t1}_to_T{t2}_highN", protocol))

    # N expansion patterns
    n_pairs = [(4, 16), (8, 32), (2, 8), (16, 64)]
    for n1, n2 in n_pairs:
        protocol = f"mpnn:T0.2:N{n1} -> torsional_relax:R1S2 -> mpnn:T0.1:N{n2} -> torsional_relax:R2S3"
        jobs.append((f"expand_N{n1}_to_N{n2}", protocol))

    return jobs


def generate_multi_round_protocols() -> List[Tuple[str, str]]:
    """Generate multi-round (3+) protocols."""
    jobs = []

    # Three-round progressive
    three_round_temps = [
        (0.3, 0.2, 0.1),
        (0.25, 0.15, 0.1),
        (0.3, 0.15, 0.05),
        (0.4, 0.2, 0.1),
    ]
    for t1, t2, t3 in three_round_temps:
        protocol = f"mpnn:T{t1}:N8 -> torsional_relax:R1S2 -> mpnn:T{t2}:N8 -> torsional_relax:R1S2 -> mpnn:T{t3}:N16 -> torsional_relax:R2S3"
        jobs.append((f"3round_T{t1}_{t2}_{t3}", protocol))

    # Four-round progressive
    protocol = "mpnn:T0.3:N4 -> torsional_relax:R1S2 -> mpnn:T0.2:N8 -> torsional_relax:R1S2 -> mpnn:T0.15:N8 -> torsional_relax:R1S2 -> mpnn:T0.1:N16 -> torsional_relax:R2S3"
    jobs.append(("4round_progressive", protocol))

    # Six-round fine-grained
    protocol = "mpnn:T0.3:N2 -> torsional_relax:R1S2 -> mpnn:T0.25:N4 -> torsional_relax:R1S2 -> mpnn:T0.2:N4 -> torsional_relax:R1S2 -> mpnn:T0.15:N8 -> torsional_relax:R1S2 -> mpnn:T0.1:N8 -> torsional_relax:R1S3 -> mpnn:T0.05:N16 -> torsional_relax:R2S3"
    jobs.append(("6round_finegrained", protocol))

    # High-N multi-round
    protocol = "mpnn:T0.3:N32 -> torsional_relax:R1S2 -> mpnn:T0.2:N32 -> torsional_relax:R1S2 -> mpnn:T0.1:N32 -> torsional_relax:R2S3"
    jobs.append(("3round_highN", protocol))

    # Low-N multi-round
    protocol = "mpnn:T0.3:N2 -> torsional_relax:R1S2 -> mpnn:T0.2:N2 -> torsional_relax:R1S2 -> mpnn:T0.15:N2 -> torsional_relax:R1S2 -> mpnn:T0.1:N4 -> torsional_relax:R2S3"
    jobs.append(("4round_lowN", protocol))

    return jobs


def generate_geometry_first_protocols() -> List[Tuple[str, str]]:
    """Generate protocols with cartesian relaxation first."""
    jobs = []

    # Basic geometry-first
    cart_configs = ["R1S2", "R2S3", "R3S4"]
    for cart in cart_configs:
        protocol = f"cart_relax:{cart} -> mpnn:T0.2:N16 -> torsional_relax:R2S3"
        jobs.append((f"geom_cart{cart}_mpnn", protocol))

    # Geometry-first with annealing
    for cart in ["R2S3", "R3S4"]:
        protocol = f"cart_relax:{cart} -> mpnn:T0.2:N8 -> torsional_relax:R1S2 -> mpnn:T0.1:N16 -> torsional_relax:R2S3"
        jobs.append((f"geom_cart{cart}_anneal", protocol))

    # Heavy geometry optimization
    protocol = "cart_relax:R3S4 -> mpnn:T0.15:N32 -> torsional_relax:R2S3 -> mpnn:T0.1:N16 -> torsional_relax:R3S4"
    jobs.append(("geom_heavy", protocol))

    return jobs


def generate_design_first_protocols() -> List[Tuple[str, str]]:
    """Generate protocols with design first, then geometry refinement."""
    jobs = []

    # Design then heavy cart
    protocol = "mpnn:T0.2:N16 -> cart_relax:R2S3 -> mpnn:T0.1:N8 -> torsional_relax:R2S3"
    jobs.append(("design_then_cart", protocol))

    # Multiple design rounds then cart
    protocol = "mpnn:T0.3:N8 -> torsional_relax:R1S2 -> mpnn:T0.2:N8 -> cart_relax:R2S3 -> mpnn:T0.1:N16 -> torsional_relax:R2S3"
    jobs.append(("multidesign_then_cart", protocol))

    # Light design, heavy refinement
    protocol = "mpnn:T0.3:N4 -> torsional_relax:R1S2 -> cart_relax:R3S4 -> mpnn:T0.1:N16 -> torsional_relax:R3S4"
    jobs.append(("light_design_heavy_refine", protocol))

    return jobs


def generate_breadth_protocols() -> List[Tuple[str, str]]:
    """Generate high-diversity breadth protocols."""
    jobs = []

    # Very high temperature
    for temp in [0.35, 0.4, 0.5]:
        protocol = f"mpnn:T{temp}:N32 -> torsional_relax:R1S2"
        jobs.append((f"breadth_T{temp}", protocol))

    # Very high N
    for n in [48, 64, 96, 128]:
        protocol = f"mpnn:T0.3:N{n} -> torsional_relax:R1S2"
        jobs.append((f"breadth_N{n}", protocol))

    # High T + high N combinations
    for temp, n in [(0.35, 64), (0.4, 48), (0.3, 96)]:
        protocol = f"mpnn:T{temp}:N{n} -> torsional_relax:R1S2"
        jobs.append((f"breadth_T{temp}_N{n}", protocol))

    # Multi-round breadth
    protocol = "mpnn:T0.4:N32 -> torsional_relax:R1S2 -> mpnn:T0.3:N32 -> torsional_relax:R1S2 -> mpnn:T0.2:N32 -> torsional_relax:R2S3"
    jobs.append(("breadth_3round_highT", protocol))

    return jobs


def generate_depth_protocols() -> List[Tuple[str, str]]:
    """Generate conservative depth/refinement protocols."""
    jobs = []

    # Very low temperature
    for temp in [0.03, 0.05, 0.07]:
        protocol = f"mpnn:T{temp}:N16 -> torsional_relax:R2S3"
        jobs.append((f"depth_T{temp}", protocol))

    # Low T with heavy relaxation
    relax_configs = ["R2S3", "R2S4", "R3S3", "R3S4"]
    for config in relax_configs:
        protocol = f"mpnn:T0.05:N16 -> torsional_relax:{config}"
        jobs.append((f"depth_T0.05_{config}", protocol))

    # Multi-round depth
    protocol = "mpnn:T0.1:N8 -> torsional_relax:R2S3 -> mpnn:T0.07:N8 -> torsional_relax:R2S3 -> mpnn:T0.05:N16 -> torsional_relax:R3S4"
    jobs.append(("depth_3round_refine", protocol))

    # Conservative with geometry
    protocol = "cart_relax:R2S3 -> mpnn:T0.05:N16 -> torsional_relax:R3S4"
    jobs.append(("depth_geom_conservative", protocol))

    return jobs


def generate_constraint_sweep() -> List[Tuple[str, str, dict]]:
    """Generate constraint parameter sweep."""
    jobs = []
    base_protocol = "mpnn:T0.2:N16 -> torsional_relax:R2S3"

    # Coordinate constraint weight
    for weight in [250, 500, 750, 1000, 1500]:
        jobs.append((
            f"cst_weight_{weight}",
            base_protocol,
            {"coord_cst_weight": weight}
        ))

    # Coordinate constraint stdev
    for stdev in [0.005, 0.01, 0.02, 0.05, 0.1]:
        jobs.append((
            f"cst_stdev_{stdev}",
            base_protocol,
            {"coord_cst_stdev": stdev}
        ))

    # Global constraint weight
    for gweight in [0, 10, 25, 50, 100, 200]:
        jobs.append((
            f"global_cst_{gweight}",
            base_protocol,
            {"global_coord_cst_weight": gweight, "global_coord_cst_stdev": 0.5}
        ))

    # Global constraint stdev
    for gstdev in [0.1, 0.25, 0.5, 1.0, 2.0]:
        jobs.append((
            f"global_stdev_{gstdev}",
            base_protocol,
            {"global_coord_cst_weight": 50, "global_coord_cst_stdev": gstdev}
        ))

    # Cart bonded weight
    for cbw in [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]:
        cart_protocol = "cart_relax:R2S3 -> mpnn:T0.2:N16 -> torsional_relax:R2S3"
        jobs.append((
            f"cart_bonded_{cbw}",
            cart_protocol,
            {"cart_bonded_weight": cbw}
        ))

    return jobs


def generate_scorefunction_sweep() -> List[Tuple[str, str, dict]]:
    """Generate scorefunction comparison sweep."""
    jobs = []
    base_protocol = "mpnn:T0.2:N16 -> torsional_relax:R2S3"

    # Torsional scorefunctions
    for sf in ["beta_jan25", "ref2015", "beta_nov16"]:
        jobs.append((
            f"sf_tors_{sf}",
            base_protocol,
            {"scorefunction_torsional": sf}
        ))

    # Cartesian scorefunctions
    cart_protocol = "cart_relax:R2S3 -> mpnn:T0.2:N16 -> torsional_relax:R2S3"
    for sf in ["ref2015_cart", "beta_nov16_cart"]:
        jobs.append((
            f"sf_cart_{sf}",
            cart_protocol,
            {"scorefunction_cart": sf}
        ))

    # fa_rep weight
    for fa_rep in [0.3, 0.4, 0.55, 0.7, 1.0]:
        jobs.append((
            f"fa_rep_{fa_rep}",
            base_protocol,
            {"fa_rep_weight": fa_rep}
        ))

    return jobs


def generate_relax_rounds_sweep() -> List[Tuple[str, str, dict]]:
    """Generate relaxation rounds/cycles sweep."""
    jobs = []
    base_protocol = "mpnn:T0.2:N16 -> torsional_relax:R2S3"

    # Relax rounds
    for rounds in [1, 3, 5, 7, 10]:
        jobs.append((
            f"rounds_{rounds}",
            base_protocol,
            {"relax_rounds": rounds}
        ))

    return jobs


def generate_hybrid_protocols() -> List[Tuple[str, str]]:
    """Generate hybrid strategy protocols."""
    jobs = []

    # Breadth-to-depth transition
    protocol = "mpnn:T0.4:N32 -> torsional_relax:R1S2 -> mpnn:T0.2:N16 -> torsional_relax:R1S2 -> mpnn:T0.05:N8 -> torsional_relax:R3S4"
    jobs.append(("hybrid_breadth_to_depth", protocol))

    # Interleaved cart/tors
    protocol = "cart_relax:R1S2 -> mpnn:T0.2:N8 -> torsional_relax:R1S2 -> mpnn:T0.15:N8 -> cart_relax:R1S2 -> mpnn:T0.1:N16 -> torsional_relax:R2S3"
    jobs.append(("hybrid_interleaved_cart", protocol))

    # N expansion with T reduction
    protocol = "mpnn:T0.3:N4 -> torsional_relax:R1S2 -> mpnn:T0.2:N8 -> torsional_relax:R1S2 -> mpnn:T0.15:N16 -> torsional_relax:R1S3 -> mpnn:T0.1:N32 -> torsional_relax:R2S3"
    jobs.append(("hybrid_expand_N_reduce_T", protocol))

    # Intensive refinement
    protocol = "mpnn:T0.2:N8 -> torsional_relax:R2S3 -> mpnn:T0.1:N8 -> torsional_relax:R2S3 -> mpnn:T0.05:N8 -> torsional_relax:R3S4"
    jobs.append(("hybrid_intensive_refine", protocol))

    # Quick exploration + careful refinement
    protocol = "mpnn:T0.35:N48 -> torsional_relax:R1S2 -> mpnn:T0.05:N4 -> torsional_relax:R3S4"
    jobs.append(("hybrid_explore_then_refine", protocol))

    return jobs


def generate_preset_retest() -> List[Tuple[str, str]]:
    """Re-test presets with fixed select_best."""
    jobs = []

    # These use the presets directly (which now have fixed select_best)
    presets = ["fast", "balanced", "thorough"]
    for preset in presets:
        jobs.append((f"preset_{preset}_fixed", None, {"preset": preset}))

    return jobs


def main():
    parser = argparse.ArgumentParser(description="Generate large-scale hyperparameter sweep")
    parser.add_argument("--output", default="cmds/large_sweep_commands.txt",
                       help="Output file for commands")
    parser.add_argument("--dry-run", action="store_true",
                       help="Just show job counts, don't write file")
    parser.add_argument("--replicates", type=int, default=2,
                       help="Number of replicates per job (default: 2)")
    parser.add_argument("--start-job", type=int, default=100,
                       help="Starting job number (default: 100, to avoid conflicts with initial sweep)")
    args = parser.parse_args()

    # Generate all protocol families
    all_jobs = []

    # Protocol-based jobs (name, protocol)
    protocol_jobs = []
    protocol_jobs.extend(generate_single_shot_protocols())
    protocol_jobs.extend(generate_two_round_protocols())
    protocol_jobs.extend(generate_multi_round_protocols())
    protocol_jobs.extend(generate_geometry_first_protocols())
    protocol_jobs.extend(generate_design_first_protocols())
    protocol_jobs.extend(generate_breadth_protocols())
    protocol_jobs.extend(generate_depth_protocols())
    protocol_jobs.extend(generate_hybrid_protocols())

    # Convert to full jobs with replicates
    job_num = args.start_job
    for name, protocol in protocol_jobs:
        for rep in range(1, args.replicates + 1):
            job_name = f"job{job_num:04d}__{name}__rep{rep}"
            cmd = build_command(job_name, protocol=protocol)
            all_jobs.append(cmd)
            job_num += 1

    # Parameter sweep jobs (name, protocol, extra_args)
    param_jobs = []
    param_jobs.extend(generate_constraint_sweep())
    param_jobs.extend(generate_scorefunction_sweep())
    param_jobs.extend(generate_relax_rounds_sweep())

    for job_data in param_jobs:
        if len(job_data) == 3:
            name, protocol, extra_args = job_data
        else:
            name, protocol = job_data
            extra_args = {}

        for rep in range(1, args.replicates + 1):
            job_name = f"job{job_num:04d}__{name}__rep{rep}"
            cmd = build_command(job_name, protocol=protocol, extra_args=extra_args)
            all_jobs.append(cmd)
            job_num += 1

    # Preset re-test
    for name, _, kwargs in generate_preset_retest():
        for rep in range(1, args.replicates + 1):
            job_name = f"job{job_num:04d}__{name}__rep{rep}"
            preset = kwargs.get("preset")
            cmd = build_command(job_name, preset=preset)
            all_jobs.append(cmd)
            job_num += 1

    # Print summary
    print("=" * 80)
    print("LARGE-SCALE HYPERPARAMETER SWEEP GENERATION")
    print("=" * 80)
    print(f"\nTotal jobs: {len(all_jobs)}")
    print(f"Replicates per condition: {args.replicates}")
    print(f"\nJob breakdown:")
    print(f"  Single-shot protocols: {len(generate_single_shot_protocols()) * args.replicates}")
    print(f"  Two-round protocols: {len(generate_two_round_protocols()) * args.replicates}")
    print(f"  Multi-round protocols: {len(generate_multi_round_protocols()) * args.replicates}")
    print(f"  Geometry-first protocols: {len(generate_geometry_first_protocols()) * args.replicates}")
    print(f"  Design-first protocols: {len(generate_design_first_protocols()) * args.replicates}")
    print(f"  Breadth protocols: {len(generate_breadth_protocols()) * args.replicates}")
    print(f"  Depth protocols: {len(generate_depth_protocols()) * args.replicates}")
    print(f"  Hybrid protocols: {len(generate_hybrid_protocols()) * args.replicates}")
    print(f"  Constraint sweep: {len(generate_constraint_sweep()) * args.replicates}")
    print(f"  Scorefunction sweep: {len(generate_scorefunction_sweep()) * args.replicates}")
    print(f"  Relax rounds sweep: {len(generate_relax_rounds_sweep()) * args.replicates}")
    print(f"  Preset re-test: {len(generate_preset_retest()) * args.replicates}")

    if args.dry_run:
        print("\n[DRY RUN - no file written]")
        return

    # Write commands file
    output_path = SCRIPT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for cmd in all_jobs:
            f.write(cmd + "\n")

    print(f"\nCommands written to: {output_path}")
    print(f"\nTo submit:")
    print(f"  ./submit_array.sh --commands {args.output}")


if __name__ == "__main__":
    main()
