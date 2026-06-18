#!/usr/bin/env python3
"""
Analyze hyperparameter sweep results for step03 FastMPNN design.

Reads output JSON files from sweep runs and aggregates metrics for comparison.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import argparse


def load_results(output_dir: str) -> List[Dict]:
    """Load all result JSON files from sweep outputs."""
    results = []
    output_path = Path(output_dir)

    for job_dir in sorted(output_path.iterdir()):
        if not job_dir.is_dir():
            continue

        json_file = job_dir / "fastmpnn_design_results.json"
        if not json_file.exists():
            continue

        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            result = {
                "job_name": job_dir.name,
                "protocol": data.get("metadata", {}).get("protocol", "unknown"),
                "runtime": data.get("metadata", {}).get("runtime_seconds", 0),
                "num_designs": len(data.get("output_designs", [])),
            }

            # Aggregate design metrics
            designs = data.get("output_designs", [])
            if designs:
                # Average metrics across designs
                num_mutations = []
                seq_identity = []
                max_bond_dev = []
                ca_rmsd_step02 = []

                for d in designs:
                    seq_metrics = d.get("metrics", {}).get("sequence_metrics", {})
                    geom = d.get("metrics", {}).get("bond_geometry", {})
                    rmsd = d.get("metrics", {}).get("rmsd", {})

                    if "num_mutations" in seq_metrics:
                        num_mutations.append(seq_metrics["num_mutations"])
                    if "sequence_identity_vs_step02" in seq_metrics:
                        seq_identity.append(seq_metrics["sequence_identity_vs_step02"])

                    unconstrained = geom.get("bond_length_geometry", {}).get("unconstrained_only", {})
                    if "max" in unconstrained:
                        max_bond_dev.append(unconstrained["max"])

                    if "global_ca_vs_step02" in rmsd:
                        ca_rmsd_step02.append(rmsd["global_ca_vs_step02"])

                result["avg_mutations"] = sum(num_mutations) / len(num_mutations) if num_mutations else 0
                result["avg_seq_identity"] = sum(seq_identity) / len(seq_identity) if seq_identity else 0
                result["avg_max_bond_dev"] = sum(max_bond_dev) / len(max_bond_dev) if max_bond_dev else 0
                result["avg_ca_rmsd"] = sum(ca_rmsd_step02) / len(ca_rmsd_step02) if ca_rmsd_step02 else 0

                # Best design (lowest bond deviation)
                best_idx = max_bond_dev.index(min(max_bond_dev)) if max_bond_dev else 0
                result["best_bond_dev"] = min(max_bond_dev) if max_bond_dev else 0
                result["best_mutations"] = num_mutations[best_idx] if num_mutations else 0

            results.append(result)

        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return results


def print_summary_table(results: List[Dict]) -> None:
    """Print summary table of results."""
    if not results:
        print("No results found")
        return

    print("\n" + "=" * 120)
    print("HYPERPARAMETER SWEEP RESULTS")
    print("=" * 120)

    # Header
    print(f"{'Job Name':<40} {'Protocol':<15} {'Designs':<8} {'Avg Mut':<8} "
          f"{'SeqID':<8} {'BondDev':<10} {'CA RMSD':<10} {'Runtime':<8}")
    print("-" * 120)

    # Sort by best bond deviation
    results.sort(key=lambda x: x.get("best_bond_dev", float("inf")))

    for r in results:
        job_name = r["job_name"][:38]
        protocol = r.get("protocol", "?")[:13]
        num_designs = r.get("num_designs", 0)
        avg_mut = r.get("avg_mutations", 0)
        seq_id = r.get("avg_seq_identity", 0) * 100
        bond_dev = r.get("avg_max_bond_dev", 0)
        ca_rmsd = r.get("avg_ca_rmsd", 0)
        runtime = r.get("runtime", 0)

        print(f"{job_name:<40} {protocol:<15} {num_designs:<8} {avg_mut:<8.1f} "
              f"{seq_id:<7.1f}% {bond_dev:<10.4f} {ca_rmsd:<10.4f} {runtime:<8.0f}s")

    print("-" * 120)

    # Best configurations
    print("\nTop 5 by bond geometry:")
    for i, r in enumerate(results[:5]):
        print(f"  {i+1}. {r['job_name']}: bond_dev={r.get('best_bond_dev', 0):.4f}, "
              f"mutations={r.get('best_mutations', 0)}")


def export_csv(results: List[Dict], output_file: str) -> None:
    """Export results to CSV."""
    import csv

    fieldnames = [
        "job_name", "protocol", "num_designs", "avg_mutations",
        "avg_seq_identity", "avg_max_bond_dev", "avg_ca_rmsd",
        "best_bond_dev", "best_mutations", "runtime"
    ]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\nResults exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze step03 hyperparameter sweep results"
    )
    parser.add_argument(
        "--output_dir",
        default="/home/woodbuse/special_scripts/upgraded_fastMPNNdesign/modules/step03__fastmpnndesign/hyper_param_sweep/outputs",
        help="Directory containing sweep outputs"
    )
    parser.add_argument(
        "--export_csv",
        default=None,
        help="Export results to CSV file"
    )

    args = parser.parse_args()

    results = load_results(args.output_dir)
    print_summary_table(results)

    if args.export_csv:
        export_csv(results, args.export_csv)


if __name__ == "__main__":
    main()
