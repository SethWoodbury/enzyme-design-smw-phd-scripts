#!/usr/bin/env python3
"""
Analyze comprehensive hyperparameter sweep results.

Optimization priorities (in order):
1. Minimize max bond length deviations (unconstrained atoms)
2. Minimize max bond angle deviations (unconstrained atoms)
3. Maximize catalytic residues passing thresholds
4. Ensure ligand/constrained RMSD ≈ 0.00
5. Minimize CA RMSD (target < 0.75-1.00 Å)
6. Minimize runtime
7. Minimize clashes

Aggregates N=3 replicates per condition with mean ± std.
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
import statistics

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_BASE = SCRIPT_DIR / "outputs"
ANALYSIS_FILE = SCRIPT_DIR / "COMPREHENSIVE_ANALYSIS.md"

# Thresholds for "good" results
BOND_THRESHOLD = 0.05  # Å
ANGLE_THRESHOLD = 10.0  # degrees
CA_RMSD_GOOD = 1.0  # Å
CA_RMSD_EXCELLENT = 0.75  # Å


def parse_job_id(job_id: str) -> dict:
    """Parse job ID back into parameters."""
    # Format: fr0.5_rampY_cb2.5_cbm4.0_rx2x3_bgY_r1
    params = {}

    match = re.match(
        r'fr([\d.]+)_ramp([YN])_cb([\d.]+)_cbm([\d.]+)_rx(\d+)x(\d+)_bg([YN])_r(\d+)',
        job_id
    )

    if match:
        params['fa_rep_scale'] = float(match.group(1))
        params['ramp_fa_rep'] = match.group(2) == 'Y'
        params['cart_bonded_weight'] = float(match.group(3))
        params['cart_bonded_max'] = float(match.group(4))
        params['repeats'] = int(match.group(5))
        params['stages'] = int(match.group(6))
        params['bond_geometry_min'] = match.group(7) == 'Y'
        params['replicate'] = int(match.group(8))

    return params


def get_condition_key(params: dict) -> str:
    """Get condition key (without replicate)."""
    return (
        f"fr{params['fa_rep_scale']:.1f}_"
        f"ramp{'Y' if params['ramp_fa_rep'] else 'N'}_"
        f"cb{params['cart_bonded_weight']:.1f}_"
        f"cbm{params['cart_bonded_max']:.1f}_"
        f"rx{params['repeats']}x{params['stages']}_"
        f"bg{'Y' if params['bond_geometry_min'] else 'N'}"
    )


def load_all_results() -> dict:
    """Load all results from output directories."""
    results = defaultdict(list)

    if not OUTPUT_BASE.exists():
        print(f"No output directory found: {OUTPUT_BASE}")
        return results

    for job_dir in OUTPUT_BASE.iterdir():
        if not job_dir.is_dir():
            continue

        job_id = job_dir.name
        params = parse_job_id(job_id)
        if not params:
            continue

        # Find metrics JSON
        metrics_files = list(job_dir.glob("*_metrics.json"))
        if not metrics_files:
            continue

        metrics_file = metrics_files[0]
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {metrics_file}: {e}")
            continue

        # Extract key metrics from actual JSON structure
        conv = metrics.get('convergence', {})
        history = conv.get('history', [])
        final_round = history[-1] if history else {}

        rmsd = metrics.get('rmsd', {})
        quality = metrics.get('quality', {})
        catres = metrics.get('catres_geometry_status', {})

        # Handle nested constrained_atoms structure
        const_rmsd = rmsd.get('constrained_atoms', {})
        if isinstance(const_rmsd, dict):
            const_rmsd_val = const_rmsd.get('aggregate_rmsd', 0.0)
        else:
            const_rmsd_val = const_rmsd

        result = {
            'job_id': job_id,
            'params': params,
            'metrics': {
                # Bond geometry from convergence history (unconstrained)
                'max_bond_dev': final_round.get('max_bond_dev', None),
                'mean_bond_dev': final_round.get('mean_bond_dev', None),
                'max_angle_dev': final_round.get('max_angle_dev', None),
                'mean_angle_dev': final_round.get('mean_angle_dev', None),

                # Catres status
                'catres_failing': catres.get('num_failing', None),
                'catres_total': catres.get('num_total', 17),

                # RMSD values
                'ligand_rmsd': rmsd.get('ligand', None),
                'constrained_rmsd': const_rmsd_val,
                'ca_rmsd': rmsd.get('global_ca', None),

                # Quality indicators
                'clashes': quality.get('clashes_after', None),
                'chain_breaks': quality.get('chain_breaks_after', None),

                # Runtime
                'total_runtime': conv.get('total_time', None),
                'fastrelax_rounds': conv.get('num_rounds', None),
            }
        }

        condition_key = get_condition_key(params)
        results[condition_key].append(result)

    return results


def aggregate_results(results: dict) -> list:
    """Aggregate replicates into mean ± std."""
    aggregated = []

    for condition_key, replicates in results.items():
        if not replicates:
            continue

        params = replicates[0]['params'].copy()
        del params['replicate']

        n_replicates = len(replicates)

        # Aggregate each metric
        agg_metrics = {}
        metric_keys = replicates[0]['metrics'].keys()

        for key in metric_keys:
            values = [r['metrics'][key] for r in replicates if r['metrics'][key] is not None]
            if values:
                agg_metrics[f'{key}_mean'] = statistics.mean(values)
                agg_metrics[f'{key}_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
                agg_metrics[f'{key}_min'] = min(values)
                agg_metrics[f'{key}_max'] = max(values)
            else:
                agg_metrics[f'{key}_mean'] = None
                agg_metrics[f'{key}_std'] = None
                agg_metrics[f'{key}_min'] = None
                agg_metrics[f'{key}_max'] = None

        aggregated.append({
            'condition': condition_key,
            'params': params,
            'n_replicates': n_replicates,
            'metrics': agg_metrics,
        })

    return aggregated


def calculate_composite_score(agg: dict) -> float:
    """
    Calculate composite score for ranking.

    Priorities (weights):
    1. Max bond deviation (unconstrained) - HIGHEST priority
    2. Max angle deviation (unconstrained) - HIGH priority
    3. Catres failing count - HIGH priority
    4. Constrained/ligand RMSD should be ~0 - CRITICAL (penalty if not)
    5. CA RMSD - MEDIUM priority
    6. Runtime - LOW priority
    7. Clashes - LOW priority
    """
    m = agg['metrics']

    # Start with base score of 0 (lower is better)
    score = 0.0

    # 1. Bond deviation (weight: 100)
    # Target: < 0.05 Å, penalize heavily if > 0.1 Å
    max_bond = m.get('max_bond_dev_mean')
    if max_bond is not None:
        if max_bond < BOND_THRESHOLD:
            score += max_bond * 100  # Small bonus for being under threshold
        else:
            score += max_bond * 200  # Higher penalty for exceeding
    else:
        score += 100  # Penalty for missing data

    # 2. Angle deviation (weight: 50)
    # Target: < 10 deg
    max_angle = m.get('max_angle_dev_mean')
    if max_angle is not None:
        if max_angle < ANGLE_THRESHOLD:
            score += max_angle * 5  # Normalize: 10 deg = 50 points
        else:
            score += max_angle * 10  # Higher penalty for exceeding
    else:
        score += 100

    # 3. Catres failing (weight: 30 per failing residue)
    catres_fail = m.get('catres_failing_mean')
    if catres_fail is not None:
        score += catres_fail * 30
    else:
        score += 100

    # 4. Constrained RMSD (should be ~0, heavy penalty if not)
    constrained_rmsd = m.get('constrained_rmsd_mean')
    if constrained_rmsd is not None:
        if constrained_rmsd > 0.01:
            score += 1000  # Major penalty - constraints not working!
    ligand_rmsd = m.get('ligand_rmsd_mean')
    if ligand_rmsd is not None:
        if ligand_rmsd > 0.01:
            score += 1000  # Major penalty

    # 5. CA RMSD (weight: 20)
    # Target: < 1.0 Å, bonus if < 0.75 Å
    ca_rmsd = m.get('ca_rmsd_mean')
    if ca_rmsd is not None:
        if ca_rmsd <= CA_RMSD_EXCELLENT:
            score += ca_rmsd * 10  # Bonus for excellent
        elif ca_rmsd <= CA_RMSD_GOOD:
            score += ca_rmsd * 20
        else:
            score += ca_rmsd * 30  # Penalty for > 1.0 Å

    # 6. Runtime (weight: 0.01 per second)
    # ~1000s = 10 points
    runtime = m.get('total_runtime_mean')
    if runtime is not None:
        score += runtime * 0.01

    # 7. Clashes (weight: 2 per clash)
    clashes = m.get('clashes_mean')
    if clashes is not None:
        score += clashes * 2

    return score


def format_table_row(agg: dict, rank: int) -> str:
    """Format a single row for markdown table."""
    p = agg['params']
    m = agg['metrics']

    # Format FastRelax config
    fr_config = f"{p['repeats']}×{p['stages']}"

    # Format metrics with uncertainty
    def fmt_val(key, decimals=3, suffix=''):
        mean = m.get(f'{key}_mean')
        std = m.get(f'{key}_std')
        if mean is None:
            return 'N/A'
        if std and std > 0:
            return f"{mean:.{decimals}f}±{std:.{decimals}f}{suffix}"
        return f"{mean:.{decimals}f}{suffix}"

    row = f"| {rank} | {p['fa_rep_scale']:.1f} | {'Y' if p['ramp_fa_rep'] else 'N'} | "
    row += f"{p['cart_bonded_weight']:.1f} | {p['cart_bonded_max']:.1f} | {fr_config} | "
    row += f"{'Y' if p['bond_geometry_min'] else 'N'} | "
    row += f"{fmt_val('max_bond_dev', 4, 'Å')} | {fmt_val('max_angle_dev', 1, '°')} | "
    row += f"{fmt_val('catres_failing', 1)} | "
    row += f"{fmt_val('ca_rmsd', 3, 'Å')} | "
    row += f"{fmt_val('clashes', 1)} | "
    row += f"{fmt_val('total_runtime', 0, 's')} |"

    return row


def analyze_by_parameter(aggregated: list) -> dict:
    """Analyze effect of each parameter."""
    analysis = {}

    # Group by each parameter
    param_groups = {
        'fa_rep_scale': defaultdict(list),
        'ramp_fa_rep': defaultdict(list),
        'cart_bonded_weight': defaultdict(list),
        'cart_bonded_max': defaultdict(list),
        'fastrelax_config': defaultdict(list),
        'bond_geometry_min': defaultdict(list),
    }

    for agg in aggregated:
        p = agg['params']
        m = agg['metrics']

        param_groups['fa_rep_scale'][p['fa_rep_scale']].append(m)
        param_groups['ramp_fa_rep'][p['ramp_fa_rep']].append(m)
        param_groups['cart_bonded_weight'][p['cart_bonded_weight']].append(m)
        param_groups['cart_bonded_max'][p['cart_bonded_max']].append(m)
        param_groups['fastrelax_config'][f"{p['repeats']}x{p['stages']}"].append(m)
        param_groups['bond_geometry_min'][p['bond_geometry_min']].append(m)

    # Calculate averages for each parameter value
    for param_name, value_groups in param_groups.items():
        analysis[param_name] = {}
        for value, metrics_list in value_groups.items():
            avg_metrics = {}
            for key in ['max_bond_dev_mean', 'max_angle_dev_mean', 'catres_failing_mean',
                        'ca_rmsd_mean', 'clashes_mean', 'total_runtime_mean']:
                values = [m[key] for m in metrics_list if m.get(key) is not None]
                if values:
                    avg_metrics[key] = statistics.mean(values)
            analysis[param_name][value] = avg_metrics

    return analysis


def main():
    print("=" * 70)
    print("COMPREHENSIVE HYPERPARAMETER SWEEP ANALYSIS")
    print("=" * 70)
    print()

    # Load all results
    print("Loading results...")
    results = load_all_results()

    if not results:
        print("No results found!")
        return

    total_jobs = sum(len(v) for v in results.values())
    print(f"Found {len(results)} conditions with {total_jobs} total jobs")
    print()

    # Aggregate replicates
    print("Aggregating replicates...")
    aggregated = aggregate_results(results)

    # Calculate composite scores
    for agg in aggregated:
        agg['score'] = calculate_composite_score(agg)

    # Sort by composite score (lower is better)
    aggregated.sort(key=lambda x: x['score'])

    # Verify constraints are working
    print("\nVerifying constraint satisfaction...")
    constraint_issues = []
    for agg in aggregated:
        m = agg['metrics']
        ligand = m.get('ligand_rmsd_mean', 0)
        constrained = m.get('constrained_rmsd_mean', 0)
        if ligand and ligand > 0.01:
            constraint_issues.append(f"{agg['condition']}: ligand RMSD = {ligand:.4f}")
        if constrained and constrained > 0.01:
            constraint_issues.append(f"{agg['condition']}: constrained RMSD = {constrained:.4f}")

    if constraint_issues:
        print(f"  WARNING: {len(constraint_issues)} conditions have constraint issues!")
        for issue in constraint_issues[:5]:
            print(f"    {issue}")
    else:
        print("  All conditions have ligand/constrained RMSD ≈ 0.00 ✓")

    # Analyze by parameter
    print("\nAnalyzing parameter effects...")
    param_analysis = analyze_by_parameter(aggregated)

    # Generate report
    print(f"\nWriting analysis to {ANALYSIS_FILE}...")

    with open(ANALYSIS_FILE, 'w') as f:
        f.write("# Comprehensive Hyperparameter Sweep Analysis\n\n")
        f.write("## Overview\n\n")
        f.write(f"- **Total conditions tested:** {len(aggregated)}\n")
        f.write(f"- **Total jobs (with replicates):** {total_jobs}\n")
        f.write(f"- **Replicates per condition:** {aggregated[0]['n_replicates'] if aggregated else 'N/A'}\n\n")

        # Constraint verification
        f.write("## Constraint Verification\n\n")
        if constraint_issues:
            f.write(f"**WARNING:** {len(constraint_issues)} conditions have constraint issues!\n\n")
            for issue in constraint_issues:
                f.write(f"- {issue}\n")
        else:
            f.write("✅ All conditions maintain ligand RMSD ≈ 0.00 and constrained atom RMSD ≈ 0.00\n\n")

        # Top 30 results
        f.write("## Top 30 Configurations\n\n")
        f.write("Ranked by composite score (lower is better). Priorities: bond geometry > angle geometry > catres > CA RMSD > runtime > clashes\n\n")
        f.write("| Rank | fa_rep | Ramp | CB | CBmax | FR | BG | MaxBond | MaxAngle | CatFail | CA_RMSD | Clash | Time |\n")
        f.write("|------|--------|------|-----|-------|-----|-----|---------|----------|---------|---------|-------|------|\n")

        for i, agg in enumerate(aggregated[:30], 1):
            f.write(format_table_row(agg, i) + "\n")

        f.write("\n**Legend:**\n")
        f.write("- fa_rep: fa_rep_scale factor\n")
        f.write("- Ramp: ramp_fa_rep enabled\n")
        f.write("- CB: cart_bonded_weight\n")
        f.write("- CBmax: cart_bonded_max\n")
        f.write("- FR: FastRelax config (repeats×stages)\n")
        f.write("- BG: bond_geometry_min enabled\n")
        f.write("- Values shown as mean±std from N=3 replicates\n\n")

        # Parameter analysis
        f.write("## Parameter Effects\n\n")

        for param_name, values in param_analysis.items():
            f.write(f"### {param_name}\n\n")
            f.write("| Value | Avg MaxBond | Avg MaxAngle | Avg CatFail | Avg CA_RMSD | Avg Time |\n")
            f.write("|-------|-------------|--------------|-------------|-------------|----------|\n")

            for value, metrics in sorted(values.items(), key=lambda x: str(x[0])):
                f.write(f"| {value} | ")
                f.write(f"{metrics.get('max_bond_dev_mean', 0):.4f}Å | ")
                f.write(f"{metrics.get('max_angle_dev_mean', 0):.1f}° | ")
                f.write(f"{metrics.get('catres_failing_mean', 0):.1f} | ")
                f.write(f"{metrics.get('ca_rmsd_mean', 0):.3f}Å | ")
                f.write(f"{metrics.get('total_runtime_mean', 0):.0f}s |\n")

            f.write("\n")

        # Best configuration summary
        f.write("## Recommended Configuration\n\n")
        if aggregated:
            best = aggregated[0]
            p = best['params']
            m = best['metrics']

            f.write(f"**Best performing configuration:**\n\n")
            f.write("```\n")
            f.write(f"--fa_rep_scale {p['fa_rep_scale']}\n")
            if p['ramp_fa_rep']:
                f.write(f"--ramp_fa_rep\n")
            f.write(f"--cart_bonded_weight {p['cart_bonded_weight']}\n")
            f.write(f"--cart_bonded_max {p['cart_bonded_max']}\n")
            f.write(f"--fastrelax_repeats {p['repeats']}\n")
            f.write(f"--fastrelax_ramp_stages {p['stages']}\n")
            if p['bond_geometry_min']:
                f.write(f"--enable_bond_geometry_min\n")
            else:
                f.write(f"--disable_bond_geometry_min\n")
            f.write("```\n\n")

            f.write("**Results:**\n")
            f.write(f"- Max bond deviation: {m.get('max_bond_dev_mean', 'N/A'):.4f} ± {m.get('max_bond_dev_std', 0):.4f} Å\n")
            f.write(f"- Max angle deviation: {m.get('max_angle_dev_mean', 'N/A'):.1f} ± {m.get('max_angle_dev_std', 0):.1f}°\n")
            f.write(f"- Catres failing: {m.get('catres_failing_mean', 'N/A'):.1f} ± {m.get('catres_failing_std', 0):.1f}\n")
            f.write(f"- CA RMSD: {m.get('ca_rmsd_mean', 'N/A'):.3f} ± {m.get('ca_rmsd_std', 0):.3f} Å\n")
            f.write(f"- Clashes: {m.get('clashes_mean', 'N/A'):.1f} ± {m.get('clashes_std', 0):.1f}\n")
            f.write(f"- Runtime: {m.get('total_runtime_mean', 'N/A'):.0f} ± {m.get('total_runtime_std', 0):.0f} s\n")

        # Summary statistics
        f.write("\n## Summary Statistics\n\n")

        # Count how many meet various thresholds
        bond_passing = sum(1 for a in aggregated if a['metrics'].get('max_bond_dev_mean', 999) < BOND_THRESHOLD)
        angle_passing = sum(1 for a in aggregated if a['metrics'].get('max_angle_dev_mean', 999) < ANGLE_THRESHOLD)
        ca_excellent = sum(1 for a in aggregated if a['metrics'].get('ca_rmsd_mean', 999) < CA_RMSD_EXCELLENT)
        ca_good = sum(1 for a in aggregated if a['metrics'].get('ca_rmsd_mean', 999) < CA_RMSD_GOOD)

        f.write(f"| Metric | Threshold | Passing | Percentage |\n")
        f.write(f"|--------|-----------|---------|------------|\n")
        f.write(f"| Max bond deviation | < {BOND_THRESHOLD} Å | {bond_passing} | {100*bond_passing/len(aggregated):.1f}% |\n")
        f.write(f"| Max angle deviation | < {ANGLE_THRESHOLD}° | {angle_passing} | {100*angle_passing/len(aggregated):.1f}% |\n")
        f.write(f"| CA RMSD | < {CA_RMSD_EXCELLENT} Å | {ca_excellent} | {100*ca_excellent/len(aggregated):.1f}% |\n")
        f.write(f"| CA RMSD | < {CA_RMSD_GOOD} Å | {ca_good} | {100*ca_good/len(aggregated):.1f}% |\n")

    print("\nAnalysis complete!")
    print(f"Report written to: {ANALYSIS_FILE}")

    # Print top 5 to console
    print("\nTop 5 configurations:")
    print("-" * 100)
    for i, agg in enumerate(aggregated[:5], 1):
        p = agg['params']
        m = agg['metrics']
        print(f"{i}. fa_rep={p['fa_rep_scale']}, ramp={'Y' if p['ramp_fa_rep'] else 'N'}, "
              f"cb={p['cart_bonded_weight']}, {p['repeats']}×{p['stages']}, "
              f"bg={'Y' if p['bond_geometry_min'] else 'N'}")
        print(f"   Bond: {m.get('max_bond_dev_mean', 0):.4f}Å, Angle: {m.get('max_angle_dev_mean', 0):.1f}°, "
              f"CatFail: {m.get('catres_failing_mean', 0):.1f}, CA: {m.get('ca_rmsd_mean', 0):.3f}Å, "
              f"Time: {m.get('total_runtime_mean', 0):.0f}s")


if __name__ == "__main__":
    main()
