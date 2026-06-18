#!/usr/bin/env python
"""
Analyze and compare results from multiple fastMPNNdesign test runs.
Usage: python analyze_results.py <output_dir1> [output_dir2] [output_dir3] ...
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import statistics

def load_metrics(output_dir: Path) -> Dict[str, Any]:
    """Load metrics JSON from output directory."""
    # Find metrics file (pattern: *_metrics.json)
    metrics_files = list(output_dir.glob("*_metrics.json"))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics file found in {output_dir}")
    with open(metrics_files[0]) as f:
        return json.load(f)

def analyze_run(metrics: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Extract key metrics from a run."""
    results = {"name": name}

    # Basic info
    results["n_cycles"] = metrics.get("n_cycles", 0)
    results["n_final_candidates"] = metrics.get("n_final_candidates", 0)

    # Extract from final candidates
    final = metrics.get("final_candidates", [])
    if final:
        # Geometry metrics
        mean_disps = [c.get("geometry", {}).get("mean_displacement", 999) for c in final]
        max_disps = [c.get("geometry", {}).get("max_displacement", 999) for c in final]
        results["mean_displacement_avg"] = statistics.mean(mean_disps) if mean_disps else None
        results["max_displacement_min"] = min(max_disps) if max_disps else None

        # Catres RMSD
        sc_rmsds = [c.get("catres_rmsd", {}).get("catres_sidechain_rmsd", 999) for c in final if c.get("catres_rmsd")]
        results["catres_sc_rmsd_avg"] = statistics.mean(sc_rmsds) if sc_rmsds else None
        results["catres_sc_rmsd_min"] = min(sc_rmsds) if sc_rmsds else None

        # Rotamer quality
        rotamer_probs = []
        for c in final:
            rq = c.get("rotamer_quality", {}).get("per_residue_rotamer", {})
            for res, data in rq.items():
                prob = data.get("rotamer_prob", 0)
                rotamer_probs.append(prob)
        if rotamer_probs:
            results["rotamer_prob_avg"] = statistics.mean(rotamer_probs)
            results["rotamer_prob_min"] = min(rotamer_probs)
            results["n_unfavorable_rotamers"] = sum(1 for p in rotamer_probs if p < 0.01)

        # Sequence distance
        n_muts = [c.get("sequence_distance", {}).get("n_mutations", 0) for c in final]
        results["mutations_avg"] = statistics.mean(n_muts) if n_muts else None

    return results

def print_comparison(runs: List[Dict[str, Any]]):
    """Print comparison table."""
    print("\n" + "="*80)
    print("FASTMPNDESIGN TEST COMPARISON")
    print("="*80)

    # Headers
    headers = ["Metric"] + [r["name"] for r in runs]
    col_width = max(20, max(len(h) for h in headers) + 2)

    print(f"{'Metric':<30}", end="")
    for r in runs:
        print(f"{r['name']:<{col_width}}", end="")
    print()
    print("-"*80)

    # Metrics to compare
    metrics_to_show = [
        ("n_cycles", "Cycles"),
        ("n_final_candidates", "Final Candidates"),
        ("mean_displacement_avg", "Mean Displacement (A)"),
        ("max_displacement_min", "Best Max Disp (A)"),
        ("catres_sc_rmsd_avg", "Catres SC RMSD Avg (A)"),
        ("catres_sc_rmsd_min", "Catres SC RMSD Min (A)"),
        ("rotamer_prob_avg", "Rotamer Prob Avg"),
        ("rotamer_prob_min", "Rotamer Prob Min"),
        ("n_unfavorable_rotamers", "Unfavorable Rotamers"),
        ("mutations_avg", "Mutations Avg"),
    ]

    for key, label in metrics_to_show:
        print(f"{label:<30}", end="")
        for r in runs:
            val = r.get(key)
            if val is None:
                print(f"{'N/A':<{col_width}}", end="")
            elif isinstance(val, float):
                print(f"{val:<{col_width}.4f}", end="")
            else:
                print(f"{val:<{col_width}}", end="")
        print()

    print("="*80)

    # Recommendations
    print("\nRECOMMENDATIONS:")
    best_sc_rmsd = min(runs, key=lambda r: r.get("catres_sc_rmsd_min", 999))
    print(f"- Best catres sidechain RMSD: {best_sc_rmsd['name']} ({best_sc_rmsd.get('catres_sc_rmsd_min', 'N/A'):.4f} A)")

    best_disp = min(runs, key=lambda r: r.get("mean_displacement_avg", 999))
    print(f"- Best mean displacement: {best_disp['name']} ({best_disp.get('mean_displacement_avg', 'N/A'):.4f} A)")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <output_dir1> [output_dir2] ...")
        sys.exit(1)

    runs = []
    for path in sys.argv[1:]:
        output_dir = Path(path)
        if not output_dir.exists():
            print(f"Warning: {path} does not exist, skipping")
            continue
        try:
            metrics = load_metrics(output_dir)
            name = output_dir.name
            analysis = analyze_run(metrics, name)
            runs.append(analysis)
            print(f"Loaded: {name}")
        except Exception as e:
            print(f"Error loading {path}: {e}")

    if runs:
        print_comparison(runs)

if __name__ == "__main__":
    main()
