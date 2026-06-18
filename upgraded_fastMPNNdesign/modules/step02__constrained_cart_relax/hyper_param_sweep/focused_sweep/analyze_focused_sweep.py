#!/usr/bin/env python3
"""
Analyze focused parameter sweep results with replicate statistics.

Computes mean, std, min, max for each condition across replicates.
"""

import os
import json
import glob
import argparse
from collections import defaultdict
from pathlib import Path
from typing import List, Dict
import statistics

_HERE = Path(__file__).resolve().parent

def parse_job_name(job_name: str) -> Dict:
    """Extract parameters from job name."""
    params = {
        "repeats": 0,
        "stages": 0,
        "cart_bonded_weight": 0.0,
        "bond_geometry_min": True,
        "replicate": 0,
    }

    parts = job_name.split("__")
    for part in parts:
        if part.startswith("r") and "s" in part:
            try:
                r, s = part[1:].split("s")
                params["repeats"] = int(r)
                params["stages"] = int(s)
            except:
                pass
        elif part.startswith("cb"):
            try:
                params["cart_bonded_weight"] = float(part[2:])
            except:
                pass
        elif part.startswith("bg"):
            params["bond_geometry_min"] = "on" in part
        elif part.startswith("rep"):
            try:
                params["replicate"] = int(part[3:])
            except:
                pass

    return params


def load_result(json_path: str) -> Dict:
    """Load a single result from JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        job_name = os.path.basename(os.path.dirname(json_path))
        params = parse_job_name(job_name)

        # Extract key metrics
        bond_unc = data.get("bond_length_geometry", {}).get("after", {}).get("unconstrained_only", {})
        angle_unc = data.get("bond_angle_geometry", {}).get("after", {}).get("unconstrained_only", {})
        rmsd = data.get("rmsd", {})
        cg = data.get("catres_geometry_status", {})
        conv = data.get("convergence", {})
        quality = data.get("quality", {})

        return {
            "job_name": job_name,
            "repeats": params["repeats"],
            "stages": params["stages"],
            "cart_bonded_weight": params["cart_bonded_weight"],
            "bond_geometry_min": params["bond_geometry_min"],
            "replicate": params["replicate"],
            "max_bond_dev": bond_unc.get("max_deviation", float('inf')),
            "mean_bond_dev": bond_unc.get("mean_deviation", float('inf')),
            "max_angle_dev": angle_unc.get("max_deviation", float('inf')),
            "mean_angle_dev": angle_unc.get("mean_deviation", float('inf')),
            "ca_rmsd": rmsd.get("global_ca", float('inf')),
            "cst_rmsd": rmsd.get("constrained_atoms", {}).get("aggregate_rmsd", float('inf')),
            "lig_rmsd": rmsd.get("ligand", float('inf')),
            "catres_failing": cg.get("num_failing", 999),
            "catres_total": cg.get("num_passing", 0) + cg.get("num_failing", 0),
            "clashes": quality.get("clashes_after", 999),
            "time": conv.get("total_time", float('inf')),
            "num_rounds": conv.get("num_rounds", 0),
        }
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None


def compute_stats(values: List[float]) -> Dict:
    """Compute statistics for a list of values."""
    if not values:
        return {"mean": float('inf'), "std": 0, "min": float('inf'), "max": float('inf'), "n": 0}

    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0
    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def analyze_sweep(output_dir: str):
    """Analyze focused sweep results with replicate statistics."""
    pattern = os.path.join(output_dir, "*/*_metrics.json")
    json_files = glob.glob(pattern)

    if not json_files:
        print(f"No metrics files found in {output_dir}")
        return

    print(f"Found {len(json_files)} result files")
    print("=" * 120)

    # Load all results
    results = []
    for json_path in json_files:
        result = load_result(json_path)
        if result:
            results.append(result)

    if not results:
        print("No valid results to analyze")
        return

    print(f"Successfully loaded {len(results)} results")
    print()

    # Verify constrained atom and ligand RMSD
    print("=" * 120)
    print("VERIFICATION: Constrained Atom and Ligand RMSD")
    print("=" * 120)
    max_cst = max(r["cst_rmsd"] for r in results)
    max_lig = max(r["lig_rmsd"] for r in results)
    print(f"Max constrained atom RMSD: {max_cst:.4f}A (should be ~0)")
    print(f"Max ligand RMSD: {max_lig:.4f}A (should be ~0)")
    print()

    # Group by condition (excluding replicate)
    conditions = defaultdict(list)
    for r in results:
        key = (r["repeats"], r["stages"], r["cart_bonded_weight"], r["bond_geometry_min"])
        conditions[key].append(r)

    print("=" * 120)
    print("RESULTS BY CONDITION (with replicate statistics)")
    print("=" * 120)
    print(f"{'Config':<20} {'CB':>5} {'BG':>4} {'N':>3} | {'MaxBond':>18} | {'MaxAngle':>18} | {'CatFail':>12} | {'Time':>12}")
    print(f"{'':20} {'':>5} {'':>4} {'':>3} | {'mean±std (range)':>18} | {'mean±std (range)':>18} | {'mean±std':>12} | {'mean±std':>12}")
    print("-" * 120)

    # Compute stats for each condition
    condition_stats = []
    for key, reps in sorted(conditions.items()):
        repeats, stages, cb, bg = key
        config = f"{repeats}x{stages}"

        bond_stats = compute_stats([r["max_bond_dev"] for r in reps])
        angle_stats = compute_stats([r["max_angle_dev"] for r in reps])
        fail_stats = compute_stats([r["catres_failing"] for r in reps])
        time_stats = compute_stats([r["time"] for r in reps])

        condition_stats.append({
            "config": config,
            "repeats": repeats,
            "stages": stages,
            "cb": cb,
            "bg": bg,
            "bond_stats": bond_stats,
            "angle_stats": angle_stats,
            "fail_stats": fail_stats,
            "time_stats": time_stats,
            "n": len(reps),
        })

        bg_str = "ON" if bg else "OFF"
        bond_str = f"{bond_stats['mean']:.4f}±{bond_stats['std']:.4f}"
        angle_str = f"{angle_stats['mean']:.1f}±{angle_stats['std']:.1f}"
        fail_str = f"{fail_stats['mean']:.1f}±{fail_stats['std']:.1f}"
        time_str = f"{time_stats['mean']:.0f}±{time_stats['std']:.0f}"

        print(f"{config:<20} {cb:>5.1f} {bg_str:>4} {len(reps):>3} | {bond_str:>18} | {angle_str:>18} | {fail_str:>12} | {time_str:>12}s")

    print()

    # Rank conditions by combined score
    print("=" * 120)
    print("RANKING BY COMBINED SCORE (lower is better)")
    print("=" * 120)

    def combined_score(cs):
        # Weight: bond geometry most important, then angle, then failures
        bond_score = cs["bond_stats"]["mean"] / 0.05  # Normalize by tolerance
        angle_score = cs["angle_stats"]["mean"] / 15.0  # Normalize by practical min (~15)
        fail_score = cs["fail_stats"]["mean"] / 5.0  # Normalize
        return 2 * bond_score + angle_score + fail_score

    condition_stats.sort(key=combined_score)

    print(f"{'Rank':<5} {'Config':<10} {'CB':>5} {'BG':>4} | {'Bond (mean±std)':>18} | {'Angle (mean±std)':>16} | {'Fail':>8} | {'Time':>10}")
    print("-" * 100)

    for i, cs in enumerate(condition_stats[:15], 1):
        bg_str = "ON" if cs["bg"] else "OFF"
        bond_str = f"{cs['bond_stats']['mean']:.4f}±{cs['bond_stats']['std']:.4f}"
        angle_str = f"{cs['angle_stats']['mean']:.1f}±{cs['angle_stats']['std']:.1f}"
        print(f"{i:<5} {cs['config']:<10} {cs['cb']:>5.1f} {bg_str:>4} | {bond_str:>18} | {angle_str:>16} | {cs['fail_stats']['mean']:>8.1f} | {cs['time_stats']['mean']:>10.0f}s")

    print()

    # Analysis by parameter (averaging across conditions)
    print("=" * 120)
    print("ANALYSIS BY PARAMETER (mean across all conditions)")
    print("=" * 120)

    # By cart_bonded_weight
    print("\nBy CART_BONDED_WEIGHT:")
    by_cb = defaultdict(list)
    for cs in condition_stats:
        by_cb[cs["cb"]].append(cs)

    for cb in sorted(by_cb.keys()):
        css = by_cb[cb]
        avg_bond = statistics.mean([cs["bond_stats"]["mean"] for cs in css])
        avg_angle = statistics.mean([cs["angle_stats"]["mean"] for cs in css])
        avg_fail = statistics.mean([cs["fail_stats"]["mean"] for cs in css])
        avg_time = statistics.mean([cs["time_stats"]["mean"] for cs in css])
        print(f"  cb={cb}: avg_bond={avg_bond:.4f}A, avg_angle={avg_angle:.1f}°, avg_fail={avg_fail:.1f}, avg_time={avg_time:.0f}s")

    # By repeats x stages
    print("\nBy REPEATS x STAGES:")
    by_config = defaultdict(list)
    for cs in condition_stats:
        by_config[cs["config"]].append(cs)

    for config in ["1x3", "2x3", "3x3", "3x5"]:
        if config not in by_config:
            continue
        css = by_config[config]
        avg_bond = statistics.mean([cs["bond_stats"]["mean"] for cs in css])
        avg_angle = statistics.mean([cs["angle_stats"]["mean"] for cs in css])
        avg_fail = statistics.mean([cs["fail_stats"]["mean"] for cs in css])
        avg_time = statistics.mean([cs["time_stats"]["mean"] for cs in css])
        print(f"  {config}: avg_bond={avg_bond:.4f}A, avg_angle={avg_angle:.1f}°, avg_fail={avg_fail:.1f}, avg_time={avg_time:.0f}s")

    # By bond_geometry_min
    print("\nBy BOND_GEOMETRY_MIN:")
    by_bg = defaultdict(list)
    for cs in condition_stats:
        by_bg[cs["bg"]].append(cs)

    for bg in [True, False]:
        css = by_bg[bg]
        avg_bond = statistics.mean([cs["bond_stats"]["mean"] for cs in css])
        avg_angle = statistics.mean([cs["angle_stats"]["mean"] for cs in css])
        print(f"  {'ON' if bg else 'OFF'}: avg_bond={avg_bond:.4f}A, avg_angle={avg_angle:.1f}°")

    print()

    # Recommendations
    print("=" * 120)
    print("RECOMMENDATIONS")
    print("=" * 120)

    best = condition_stats[0]
    print(f"\nBest configuration: {best['config']} with cart_bonded={best['cb']}, bond_geom_min={'ON' if best['bg'] else 'OFF'}")
    print(f"  Bond deviation: {best['bond_stats']['mean']:.4f} ± {best['bond_stats']['std']:.4f} A")
    print(f"  Angle deviation: {best['angle_stats']['mean']:.1f} ± {best['angle_stats']['std']:.1f} deg")
    print(f"  Catres failing: {best['fail_stats']['mean']:.1f} ± {best['fail_stats']['std']:.1f}")
    print(f"  Runtime: {best['time_stats']['mean']:.0f} ± {best['time_stats']['std']:.0f} s")
    print(f"  (Based on {best['n']} replicates)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze focused parameter sweep results")
    parser.add_argument("--output_dir", type=str,
                        default=str(_HERE / "outputs"),
                        help="Directory containing sweep output directories")
    args = parser.parse_args()

    analyze_sweep(args.output_dir)
