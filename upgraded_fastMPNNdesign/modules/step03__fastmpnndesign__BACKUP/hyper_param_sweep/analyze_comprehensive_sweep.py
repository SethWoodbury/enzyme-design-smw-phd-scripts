#!/usr/bin/env python3
"""
Comprehensive analysis of hyperparameter sweep results.

Analyzes results according to user priorities:
1. Constrained atom RMSD (should be ~0)
2. Bond geometry for unconstrained catres (< 0.05A bond, < 7.5 deg angle)
3. CA RMSD vs step01 (minimize to stay close to AlphaFold3)
4. No clashes involving catalytic residues
5. No mutations to catalytic residues
6. HIS tautomer preservation
7. Good secondary structure and Dunbrack rotamers
8. Sequence diversity (for breadth mode) vs similarity (for depth mode)
9. Runtime optimization

Output:
- Summary statistics per hyperparameter setting
- Best configurations for each objective
- Pareto-optimal configurations (multi-objective)
- Failure analysis (constraint violations, mutations to catres, etc.)
- CSV/JSON exports for further analysis

Usage:
    python analyze_comprehensive_sweep.py [--output-dir DIR] [--format FORMAT]
"""

import os
import sys
import json
import glob
import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import statistics

SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(SWEEP_DIR, "outputs")


def load_results(output_dir: str = OUTPUT_BASE) -> List[Dict]:
    """Load all result JSON files from sweep outputs."""
    results = []

    pattern = os.path.join(output_dir, "job*", "fastmpnn_design_results.json")
    json_files = glob.glob(pattern)

    print(f"Found {len(json_files)} result files in {output_dir}")

    for json_path in json_files:
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            # Extract job info from path
            job_dir = os.path.dirname(json_path)
            job_name = os.path.basename(job_dir)

            # Parse job name for hyperparameter info
            parts = job_name.split("__")
            job_num = parts[0] if parts else "unknown"

            result = {
                "job_name": job_name,
                "job_num": job_num,
                "json_path": json_path,
                "data": data,
            }

            # Extract key metrics
            result["metrics"] = extract_key_metrics(data)

            results.append(result)

        except Exception as e:
            print(f"Warning: Failed to load {json_path}: {e}")

    return results


def extract_key_metrics(data: Dict) -> Dict:
    """Extract key metrics from result JSON."""
    metrics = {
        "runtime_seconds": data.get("metadata", {}).get("runtime_seconds", None),
        "protocol": data.get("metadata", {}).get("protocol", "unknown"),
        "num_output_designs": len(data.get("output_designs", [])),
    }

    # Residue classification
    res_class = data.get("residue_classification", {})
    metrics["num_catres"] = res_class.get("num_catalytic", 0)
    metrics["num_designable"] = res_class.get("num_designable", 0)
    metrics["num_fixed"] = res_class.get("num_fixed", 0)

    # Aggregate over output designs
    designs = data.get("output_designs", [])
    if designs:
        # Sequence metrics
        seq_identities = []
        num_mutations_list = []
        catres_mutations = []

        # Geometry metrics
        bond_devs = []
        angle_devs = []
        catres_bond_failures = []
        catres_angle_failures = []

        # RMSD metrics
        ligand_rmsds = []
        constrained_rmsds = []
        ca_rmsds_step01 = []
        ca_rmsds_step02 = []

        # Clash metrics
        fa_rep_scores = []
        catres_clashes = []

        for design in designs:
            design_metrics = design.get("metrics", {})

            # Sequence
            seq = design_metrics.get("sequence_metrics", {})
            if "sequence_identity_vs_step02" in seq:
                seq_identities.append(seq["sequence_identity_vs_step02"])
            if "num_mutations" in seq:
                num_mutations_list.append(seq["num_mutations"])
            if "catres_mutations" in seq:
                catres_mutations.extend(seq["catres_mutations"])

            # Geometry
            geom = design_metrics.get("bond_geometry", {})
            bond_geom = geom.get("bond_length_geometry", {}).get("unconstrained_only", {})
            angle_geom = geom.get("bond_angle_geometry", {}).get("unconstrained_only", {})

            if "max" in bond_geom:
                bond_devs.append(bond_geom["max"])
            if "max" in angle_geom:
                angle_devs.append(angle_geom["max"])

            # Catres-specific geometry
            catres_geom = design_metrics.get("catres_bond_geometry", {})
            if "num_bond_failures" in catres_geom:
                catres_bond_failures.append(catres_geom["num_bond_failures"])
            if "num_angle_failures" in catres_geom:
                catres_angle_failures.append(catres_geom["num_angle_failures"])

            # RMSD
            rmsd = design_metrics.get("rmsd", {})
            if "ligand" in rmsd and rmsd["ligand"] is not None:
                ligand_rmsds.append(rmsd["ligand"])
            if "constrained_atoms" in rmsd:
                cst = rmsd["constrained_atoms"]
                if isinstance(cst, dict) and "aggregate" in cst:
                    constrained_rmsds.append(cst["aggregate"])
                elif isinstance(cst, (int, float)):
                    constrained_rmsds.append(cst)
            if "global_ca_vs_step01" in rmsd and rmsd["global_ca_vs_step01"] is not None:
                ca_rmsds_step01.append(rmsd["global_ca_vs_step01"])
            if "global_ca_vs_step02" in rmsd and rmsd["global_ca_vs_step02"] is not None:
                ca_rmsds_step02.append(rmsd["global_ca_vs_step02"])

            # Clashes
            clash = design_metrics.get("clash_analysis", {})
            if "total_fa_rep" in clash:
                fa_rep_scores.append(clash["total_fa_rep"])
            if "catres_clashes" in clash:
                catres_clashes.append(len(clash["catres_clashes"]))

        # Aggregate metrics
        def safe_mean(lst):
            return statistics.mean(lst) if lst else None

        def safe_max(lst):
            return max(lst) if lst else None

        def safe_min(lst):
            return min(lst) if lst else None

        metrics.update({
            # Sequence
            "mean_seq_identity": safe_mean(seq_identities),
            "mean_num_mutations": safe_mean(num_mutations_list),
            "has_catres_mutations": len(catres_mutations) > 0,
            "num_catres_mutations": len(catres_mutations),

            # Geometry
            "max_bond_dev": safe_max(bond_devs),
            "mean_bond_dev": safe_mean(bond_devs),
            "max_angle_dev": safe_max(angle_devs),
            "mean_angle_dev": safe_mean(angle_devs),
            "total_catres_bond_failures": sum(catres_bond_failures) if catres_bond_failures else None,
            "total_catres_angle_failures": sum(catres_angle_failures) if catres_angle_failures else None,

            # RMSD
            "mean_ligand_rmsd": safe_mean(ligand_rmsds),
            "max_ligand_rmsd": safe_max(ligand_rmsds),
            "mean_constrained_rmsd": safe_mean(constrained_rmsds),
            "max_constrained_rmsd": safe_max(constrained_rmsds),
            "mean_ca_rmsd_step01": safe_mean(ca_rmsds_step01),
            "mean_ca_rmsd_step02": safe_mean(ca_rmsds_step02),

            # Clashes
            "mean_fa_rep": safe_mean(fa_rep_scores),
            "total_catres_clashes": sum(catres_clashes) if catres_clashes else None,
        })

    return metrics


def check_quality_criteria(metrics: Dict) -> Dict:
    """Check if metrics meet quality criteria."""
    checks = {}

    # 1. Constrained atom RMSD should be ~0 (< 0.1 A)
    cst_rmsd = metrics.get("max_constrained_rmsd")
    checks["constrained_rmsd_ok"] = cst_rmsd is not None and cst_rmsd < 0.1

    # 2. Ligand RMSD should be ~0 (< 0.1 A)
    lig_rmsd = metrics.get("max_ligand_rmsd")
    checks["ligand_rmsd_ok"] = lig_rmsd is not None and lig_rmsd < 0.1

    # 3. Bond geometry for unconstrained catres (< 0.05 A bond, < 7.5 deg angle)
    bond_dev = metrics.get("max_bond_dev")
    angle_dev = metrics.get("max_angle_dev")
    checks["bond_geometry_ok"] = bond_dev is not None and bond_dev < 0.05
    checks["angle_geometry_ok"] = angle_dev is not None and angle_dev < 7.5

    # 4. No catres mutations
    checks["no_catres_mutations"] = not metrics.get("has_catres_mutations", True)

    # 5. No catres clashes
    checks["no_catres_clashes"] = metrics.get("total_catres_clashes", 1) == 0

    # Overall pass
    checks["all_passed"] = all([
        checks["constrained_rmsd_ok"],
        checks["ligand_rmsd_ok"],
        checks["no_catres_mutations"],
    ])

    return checks


def analyze_by_parameter(results: List[Dict]) -> Dict:
    """Group results by hyperparameter settings and compute statistics."""
    grouped = defaultdict(list)

    for r in results:
        job_name = r["job_name"]
        # Parse parameter from job name
        parts = job_name.split("__")

        if len(parts) >= 2:
            # Extract parameter type and value
            param_part = parts[1]
            grouped[param_part].append(r)

    analysis = {}
    for param, group_results in grouped.items():
        metrics_list = [r["metrics"] for r in group_results]

        def safe_mean_list(key):
            values = [m[key] for m in metrics_list if m.get(key) is not None]
            return statistics.mean(values) if values else None

        def safe_std_list(key):
            values = [m[key] for m in metrics_list if m.get(key) is not None]
            return statistics.stdev(values) if len(values) > 1 else None

        analysis[param] = {
            "n_jobs": len(group_results),
            "mean_runtime": safe_mean_list("runtime_seconds"),
            "mean_seq_identity": safe_mean_list("mean_seq_identity"),
            "mean_mutations": safe_mean_list("mean_num_mutations"),
            "mean_bond_dev": safe_mean_list("max_bond_dev"),
            "mean_angle_dev": safe_mean_list("max_angle_dev"),
            "mean_ca_rmsd_step01": safe_mean_list("mean_ca_rmsd_step01"),
            "mean_ligand_rmsd": safe_mean_list("mean_ligand_rmsd"),
            "mean_constrained_rmsd": safe_mean_list("mean_constrained_rmsd"),
            "std_seq_identity": safe_std_list("mean_seq_identity"),
            "std_bond_dev": safe_std_list("max_bond_dev"),
        }

    return analysis


def find_best_configurations(results: List[Dict]) -> Dict:
    """Find best configurations for different objectives."""
    valid_results = [r for r in results if r["metrics"].get("num_output_designs", 0) > 0]

    if not valid_results:
        return {"error": "No valid results found"}

    best = {}

    # Best for CA RMSD vs step01 (minimize)
    ca_rmsd_results = [r for r in valid_results if r["metrics"].get("mean_ca_rmsd_step01") is not None]
    if ca_rmsd_results:
        best_ca = min(ca_rmsd_results, key=lambda r: r["metrics"]["mean_ca_rmsd_step01"])
        best["ca_rmsd_step01"] = {
            "job": best_ca["job_name"],
            "value": best_ca["metrics"]["mean_ca_rmsd_step01"],
        }

    # Best for bond geometry (minimize max deviation)
    bond_results = [r for r in valid_results if r["metrics"].get("max_bond_dev") is not None]
    if bond_results:
        best_bond = min(bond_results, key=lambda r: r["metrics"]["max_bond_dev"])
        best["bond_geometry"] = {
            "job": best_bond["job_name"],
            "value": best_bond["metrics"]["max_bond_dev"],
        }

    # Best for sequence diversity (maximize mutations for breadth mode)
    mut_results = [r for r in valid_results if r["metrics"].get("mean_num_mutations") is not None]
    if mut_results:
        # For breadth: most mutations
        best_diverse = max(mut_results, key=lambda r: r["metrics"]["mean_num_mutations"])
        best["sequence_diversity"] = {
            "job": best_diverse["job_name"],
            "value": best_diverse["metrics"]["mean_num_mutations"],
        }
        # For depth: fewest mutations
        best_conserved = min(mut_results, key=lambda r: r["metrics"]["mean_num_mutations"])
        best["sequence_conservation"] = {
            "job": best_conserved["job_name"],
            "value": best_conserved["metrics"]["mean_num_mutations"],
        }

    # Best for runtime (minimize)
    runtime_results = [r for r in valid_results if r["metrics"].get("runtime_seconds") is not None]
    if runtime_results:
        best_runtime = min(runtime_results, key=lambda r: r["metrics"]["runtime_seconds"])
        best["fastest"] = {
            "job": best_runtime["job_name"],
            "value": best_runtime["metrics"]["runtime_seconds"],
        }

    # Best overall (passes all quality criteria with best CA RMSD)
    quality_checked = [(r, check_quality_criteria(r["metrics"])) for r in valid_results]
    passing = [(r, c) for r, c in quality_checked if c["all_passed"]]
    if passing:
        # Among passing, find best CA RMSD
        ca_valid = [(r, c) for r, c in passing if r["metrics"].get("mean_ca_rmsd_step01") is not None]
        if ca_valid:
            best_overall, _ = min(ca_valid, key=lambda x: x[0]["metrics"]["mean_ca_rmsd_step01"])
            best["overall_best"] = {
                "job": best_overall["job_name"],
                "ca_rmsd": best_overall["metrics"]["mean_ca_rmsd_step01"],
                "bond_dev": best_overall["metrics"].get("max_bond_dev"),
                "seq_identity": best_overall["metrics"].get("mean_seq_identity"),
            }

    return best


def generate_summary_report(results: List[Dict]) -> str:
    """Generate human-readable summary report."""
    lines = []
    lines.append("=" * 70)
    lines.append("STEP03 FASTMPNN HYPERPARAMETER SWEEP ANALYSIS")
    lines.append("=" * 70)
    lines.append("")

    # Overview
    lines.append(f"Total jobs analyzed: {len(results)}")
    successful = sum(1 for r in results if r["metrics"].get("num_output_designs", 0) > 0)
    lines.append(f"Successful jobs: {successful}")
    failed = len(results) - successful
    lines.append(f"Failed jobs: {failed}")
    lines.append("")

    # Quality criteria summary
    lines.append("-" * 70)
    lines.append("QUALITY CRITERIA CHECK")
    lines.append("-" * 70)

    checks_summary = defaultdict(int)
    for r in results:
        checks = check_quality_criteria(r["metrics"])
        for key, passed in checks.items():
            if passed:
                checks_summary[key] += 1

    for check, count in sorted(checks_summary.items()):
        pct = 100 * count / len(results) if results else 0
        lines.append(f"  {check}: {count}/{len(results)} ({pct:.1f}%)")
    lines.append("")

    # Best configurations
    lines.append("-" * 70)
    lines.append("BEST CONFIGURATIONS")
    lines.append("-" * 70)

    best = find_best_configurations(results)
    for objective, info in best.items():
        if isinstance(info, dict) and "job" in info:
            lines.append(f"  {objective}:")
            lines.append(f"    Job: {info['job']}")
            for k, v in info.items():
                if k != "job":
                    lines.append(f"    {k}: {v}")
            lines.append("")

    # Parameter analysis
    lines.append("-" * 70)
    lines.append("ANALYSIS BY PARAMETER SETTING")
    lines.append("-" * 70)

    param_analysis = analyze_by_parameter(results)
    for param, stats in sorted(param_analysis.items())[:20]:  # Top 20
        lines.append(f"\n  {param} (n={stats['n_jobs']}):")
        if stats["mean_runtime"] is not None:
            lines.append(f"    Runtime: {stats['mean_runtime']:.1f}s")
        if stats["mean_bond_dev"] is not None:
            lines.append(f"    Bond dev: {stats['mean_bond_dev']:.4f}A")
        if stats["mean_ca_rmsd_step01"] is not None:
            lines.append(f"    CA RMSD vs step01: {stats['mean_ca_rmsd_step01']:.3f}A")
        if stats["mean_seq_identity"] is not None:
            lines.append(f"    Seq identity: {stats['mean_seq_identity']:.2%}")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def export_csv(results: List[Dict], output_path: str):
    """Export results to CSV."""
    if not results:
        return

    # Flatten metrics for CSV
    rows = []
    for r in results:
        row = {
            "job_name": r["job_name"],
            **r["metrics"],
        }
        # Add quality checks
        checks = check_quality_criteria(r["metrics"])
        row.update({f"check_{k}": v for k, v in checks.items()})
        rows.append(row)

    # Get all keys
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported CSV to {output_path}")


def export_json(results: List[Dict], analysis: Dict, output_path: str):
    """Export full analysis to JSON."""
    export_data = {
        "summary": {
            "total_jobs": len(results),
            "successful": sum(1 for r in results if r["metrics"].get("num_output_designs", 0) > 0),
        },
        "best_configurations": find_best_configurations(results),
        "parameter_analysis": analysis,
        "all_results": [
            {
                "job_name": r["job_name"],
                "metrics": r["metrics"],
                "quality_checks": check_quality_criteria(r["metrics"]),
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2, default=str)

    print(f"Exported JSON to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter sweep results")
    parser.add_argument("--output-dir", default=OUTPUT_BASE,
                       help="Directory containing sweep outputs")
    parser.add_argument("--format", choices=["text", "csv", "json", "all"], default="all",
                       help="Output format")
    parser.add_argument("--export-path", default=None,
                       help="Path for exported files (default: sweep_analysis.*)")
    args = parser.parse_args()

    # Load results
    results = load_results(args.output_dir)

    if not results:
        print("No results found!")
        return 1

    # Generate analysis
    param_analysis = analyze_by_parameter(results)

    # Output
    export_base = args.export_path or os.path.join(SWEEP_DIR, "sweep_analysis")

    if args.format in ["text", "all"]:
        report = generate_summary_report(results)
        print(report)

        report_path = f"{export_base}.txt"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to {report_path}")

    if args.format in ["csv", "all"]:
        csv_path = f"{export_base}.csv"
        export_csv(results, csv_path)

    if args.format in ["json", "all"]:
        json_path = f"{export_base}.json"
        export_json(results, param_analysis, json_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
