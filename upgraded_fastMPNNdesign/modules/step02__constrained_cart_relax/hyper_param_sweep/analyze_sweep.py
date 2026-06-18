#!/usr/bin/env python3
"""
Analyze parameter sweep v2 results from constrained_cart_relax.py

Reports:
- Constrained atom RMSD (should be ~0.00)
- Ligand RMSD (should be ~0.00)
- CA RMSD to input
- Catres failures (unconstrained metrics)
- Max bond/angle violations
- Runtime
- Clashes
"""

import os
import json
import glob
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import argparse


@dataclass
class SweepResult:
    """Container for sweep result metrics."""
    job_name: str
    json_path: str

    # Settings
    scorefunction: str = ""
    repeats: int = 0
    stages: int = 0
    cart_bonded_weight: float = 0.0
    enable_bond_geometry_min: bool = True
    preset: str = ""

    # Bond/angle metrics (unconstrained only - what we optimize)
    max_bond_dev: float = float('inf')
    mean_bond_dev: float = float('inf')
    max_angle_dev: float = float('inf')
    mean_angle_dev: float = float('inf')

    # Bond/angle metrics (all - for reference)
    max_bond_dev_all: float = float('inf')
    max_angle_dev_all: float = float('inf')

    # RMSD
    ca_rmsd: float = float('inf')
    constrained_rmsd: float = float('inf')
    ligand_rmsd: float = float('inf')

    # Quality
    clashes_after: int = 999
    clashes_before: int = 999

    # Catres-specific (unconstrained)
    catres_failing: int = 999
    catres_total: int = 0
    catres_worst_bond: float = 0.0
    catres_worst_angle: float = 0.0

    # Runtime
    total_time: float = float('inf')
    num_rounds: int = 0


def parse_job_name(job_name: str) -> Dict:
    """Extract parameters from job name."""
    params = {
        "scorefunction": "",
        "repeats": 0,
        "stages": 0,
        "cart_bonded_weight": 0.0,
        "enable_bond_geometry_min": True,
        "preset": "",
    }

    parts = job_name.split("__")
    for part in parts:
        if part in ["ref2015", "beta_jan25"]:
            params["scorefunction"] = part + "_cart"
        elif part.startswith("r") and "s" in part and not part.startswith("ref"):
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
            params["enable_bond_geometry_min"] = "on" in part
        elif part.startswith("preset_"):
            params["preset"] = part.replace("preset_", "")

    return params


def load_result(json_path: str) -> Optional[SweepResult]:
    """Load a single result from JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        job_name = os.path.basename(os.path.dirname(json_path))

        result = SweepResult(
            job_name=job_name,
            json_path=json_path,
        )

        # Parse job name for parameters
        params = parse_job_name(job_name)
        result.scorefunction = params.get("scorefunction", "")
        result.repeats = params.get("repeats", 0)
        result.stages = params.get("stages", 0)
        result.cart_bonded_weight = params.get("cart_bonded_weight", 0.0)
        result.enable_bond_geometry_min = params.get("enable_bond_geometry_min", True)
        result.preset = params.get("preset", "")

        # Extract from convergence info
        conv = data.get("convergence", {})
        if conv:
            result.repeats = conv.get("fastrelax_repeats", result.repeats)
            result.stages = conv.get("fastrelax_ramp_stages", result.stages)
            result.enable_bond_geometry_min = conv.get("enable_bond_geometry_min", result.enable_bond_geometry_min)
            result.total_time = conv.get("total_time", float('inf'))
            result.num_rounds = conv.get("num_rounds", 0)

        # Bond geometry (after) - use unconstrained_only metrics
        bond_geom_after = data.get("bond_length_geometry", {}).get("after", {})
        if "unconstrained_only" in bond_geom_after:
            bond_geom = bond_geom_after.get("unconstrained_only", {})
            bond_geom_all = bond_geom_after.get("all", {})
        else:
            bond_geom = bond_geom_after
            bond_geom_all = bond_geom_after
        result.max_bond_dev = bond_geom.get("max_deviation", float('inf'))
        result.mean_bond_dev = bond_geom.get("mean_deviation", float('inf'))
        result.max_bond_dev_all = bond_geom_all.get("max_deviation", float('inf'))

        angle_geom_after = data.get("bond_angle_geometry", {}).get("after", {})
        if "unconstrained_only" in angle_geom_after:
            angle_geom = angle_geom_after.get("unconstrained_only", {})
            angle_geom_all = angle_geom_after.get("all", {})
        else:
            angle_geom = angle_geom_after
            angle_geom_all = angle_geom_after
        result.max_angle_dev = angle_geom.get("max_deviation", float('inf'))
        result.mean_angle_dev = angle_geom.get("mean_deviation", float('inf'))
        result.max_angle_dev_all = angle_geom_all.get("max_deviation", float('inf'))

        # RMSD
        rmsd = data.get("rmsd", {})
        result.ca_rmsd = rmsd.get("global_ca", float('inf'))
        result.ligand_rmsd = rmsd.get("ligand", float('inf'))
        constrained = rmsd.get("constrained_atoms", {})
        if isinstance(constrained, dict):
            result.constrained_rmsd = constrained.get("aggregate_rmsd", float('inf'))
        else:
            result.constrained_rmsd = float('inf')

        # Quality
        quality = data.get("quality", {})
        result.clashes_after = quality.get("clashes_after", 999)
        result.clashes_before = quality.get("clashes_before", 999)

        # Catres status
        catres_status = data.get("catres_geometry_status", {})
        result.catres_failing = catres_status.get("num_failing", 999)
        result.catres_total = catres_status.get("num_passing", 0) + catres_status.get("num_failing", 0)

        # Get worst catres violations
        failing_catres = catres_status.get("failing_catres", [])
        if failing_catres:
            result.catres_worst_bond = max(c.get("max_bond_dev", 0) for c in failing_catres)
            result.catres_worst_angle = max(c.get("max_angle_dev", 0) for c in failing_catres)

        return result

    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None


def analyze_sweep(output_dir: str):
    """Analyze all sweep results."""
    # Find all metrics files
    pattern = os.path.join(output_dir, "*/*_metrics.json")
    json_files = glob.glob(pattern)

    if not json_files:
        print(f"No metrics files found in {output_dir}")
        return

    print(f"Found {len(json_files)} result files")
    print("=" * 140)

    # Load all results
    results: List[SweepResult] = []
    for json_path in json_files:
        result = load_result(json_path)
        if result:
            results.append(result)

    if not results:
        print("No valid results to analyze")
        return

    print(f"Successfully loaded {len(results)} results")
    print()

    # Verify constrained atom and ligand RMSD are ~0
    print("=" * 140)
    print("VERIFICATION: Constrained Atom and Ligand RMSD")
    print("=" * 140)
    max_constrained_rmsd = max(r.constrained_rmsd for r in results)
    max_ligand_rmsd = max(r.ligand_rmsd for r in results)
    print(f"Max constrained atom RMSD across all jobs: {max_constrained_rmsd:.4f}A (should be ~0)")
    print(f"Max ligand RMSD across all jobs: {max_ligand_rmsd:.4f}A (should be ~0)")
    if max_constrained_rmsd > 0.01:
        print("WARNING: Some jobs have constrained atom RMSD > 0.01A!")
    if max_ligand_rmsd > 0.01:
        print("WARNING: Some jobs have ligand RMSD > 0.01A!")
    print()

    # Define scoring function (lower is better)
    def score_overall(r: SweepResult) -> float:
        # Prioritize unconstrained bond/angle geometry
        bond_score = r.max_bond_dev / 0.05  # 1.0 = at tolerance
        angle_score = r.max_angle_dev / 10.0  # 1.0 = at tolerance
        rmsd_penalty = r.ca_rmsd / 1.0
        catres_penalty = r.catres_failing * 0.5
        clash_penalty = r.clashes_after * 0.1
        return 2 * bond_score + angle_score + rmsd_penalty + catres_penalty + clash_penalty

    # Sort by overall score
    results.sort(key=score_overall)

    # Print comprehensive table
    print("=" * 140)
    print("TOP 25 RESULTS (by combined score - unconstrained bond/angle + RMSD + catres)")
    print("=" * 140)
    header = f"{'Rank':<4} {'Scorefunction':<14} {'R×S':<5} {'CB':<4} {'BG':<3} {'MaxBond':<8} {'MaxAngle':<9} {'CA_RMSD':<8} {'CstRMSD':<8} {'LigRMSD':<8} {'CatFail':<8} {'Clash':<6} {'Time':<8}"
    print(header)
    print("-" * 140)

    for i, r in enumerate(results[:25], 1):
        sfxn_short = r.scorefunction.replace("_cart", "")
        rs = f"{r.repeats}x{r.stages}" if r.repeats else r.preset[:6]
        bg = "on" if r.enable_bond_geometry_min else "off"
        cb = f"{r.cart_bonded_weight:.1f}" if r.cart_bonded_weight else "-"
        cat_fail = f"{r.catres_failing}/{r.catres_total}"
        print(f"{i:<4} {sfxn_short:<14} {rs:<5} {cb:<4} {bg:<3} {r.max_bond_dev:<8.4f} {r.max_angle_dev:<9.2f} {r.ca_rmsd:<8.4f} {r.constrained_rmsd:<8.4f} {r.ligand_rmsd:<8.4f} {cat_fail:<8} {r.clashes_after:<6} {r.total_time:<8.1f}")

    print()

    # Best by individual metrics
    print("=" * 140)
    print("BEST BY INDIVIDUAL METRICS (UNCONSTRAINED)")
    print("=" * 140)

    results.sort(key=lambda r: r.max_bond_dev)
    print("\nBest by MAX BOND DEVIATION (unconstrained, lower is better):")
    for i, r in enumerate(results[:5], 1):
        sfxn_short = r.scorefunction.replace("_cart", "")
        rs = f"{r.repeats}x{r.stages}" if r.repeats else r.preset
        print(f"  {i}. {sfxn_short} {rs} cb={r.cart_bonded_weight}: max_bond={r.max_bond_dev:.4f}A (all={r.max_bond_dev_all:.4f}A)")

    results.sort(key=lambda r: r.max_angle_dev)
    print("\nBest by MAX ANGLE DEVIATION (unconstrained, lower is better):")
    for i, r in enumerate(results[:5], 1):
        sfxn_short = r.scorefunction.replace("_cart", "")
        rs = f"{r.repeats}x{r.stages}" if r.repeats else r.preset
        print(f"  {i}. {sfxn_short} {rs} cb={r.cart_bonded_weight}: max_angle={r.max_angle_dev:.2f}deg (all={r.max_angle_dev_all:.2f}deg)")

    results.sort(key=lambda r: r.ca_rmsd)
    print("\nBest by CA RMSD (lower is better):")
    for i, r in enumerate(results[:5], 1):
        sfxn_short = r.scorefunction.replace("_cart", "")
        rs = f"{r.repeats}x{r.stages}" if r.repeats else r.preset
        print(f"  {i}. {sfxn_short} {rs} cb={r.cart_bonded_weight}: ca_rmsd={r.ca_rmsd:.4f}A")

    results.sort(key=lambda r: r.catres_failing)
    print("\nBest by CATRES FAILURES (unconstrained, lower is better):")
    for i, r in enumerate(results[:5], 1):
        sfxn_short = r.scorefunction.replace("_cart", "")
        rs = f"{r.repeats}x{r.stages}" if r.repeats else r.preset
        print(f"  {i}. {sfxn_short} {rs} cb={r.cart_bonded_weight}: failing={r.catres_failing}/{r.catres_total} (worst bond={r.catres_worst_bond:.4f}A, angle={r.catres_worst_angle:.2f}deg)")

    results.sort(key=lambda r: r.total_time)
    print("\nFastest (lower is better):")
    for i, r in enumerate(results[:5], 1):
        sfxn_short = r.scorefunction.replace("_cart", "")
        rs = f"{r.repeats}x{r.stages}" if r.repeats else r.preset
        print(f"  {i}. {sfxn_short} {rs} cb={r.cart_bonded_weight}: time={r.total_time:.1f}s, max_bond={r.max_bond_dev:.4f}A")

    print()

    # Analysis by parameter
    print("=" * 140)
    print("ANALYSIS BY PARAMETER")
    print("=" * 140)

    # By scorefunction
    by_sfxn = {}
    for r in results:
        sfxn = r.scorefunction or "unknown"
        if sfxn not in by_sfxn:
            by_sfxn[sfxn] = []
        by_sfxn[sfxn].append(r)

    print("\nBy SCOREFUNCTION:")
    for sfxn, sfxn_results in sorted(by_sfxn.items()):
        avg_bond = sum(r.max_bond_dev for r in sfxn_results) / len(sfxn_results)
        avg_angle = sum(r.max_angle_dev for r in sfxn_results) / len(sfxn_results)
        avg_rmsd = sum(r.ca_rmsd for r in sfxn_results) / len(sfxn_results)
        avg_time = sum(r.total_time for r in sfxn_results) / len(sfxn_results)
        print(f"  {sfxn}: n={len(sfxn_results)}, avg_max_bond={avg_bond:.4f}A, avg_max_angle={avg_angle:.2f}deg, avg_ca_rmsd={avg_rmsd:.4f}A, avg_time={avg_time:.1f}s")

    # By bond_geometry_min
    by_bgm = {True: [], False: []}
    for r in results:
        by_bgm[r.enable_bond_geometry_min].append(r)

    print("\nBy BOND_GEOMETRY_MIN:")
    for bgm, bgm_results in by_bgm.items():
        if bgm_results:
            avg_bond = sum(r.max_bond_dev for r in bgm_results) / len(bgm_results)
            avg_angle = sum(r.max_angle_dev for r in bgm_results) / len(bgm_results)
            print(f"  {'ON' if bgm else 'OFF'}: n={len(bgm_results)}, avg_max_bond={avg_bond:.4f}A, avg_max_angle={avg_angle:.2f}deg")

    # By cart_bonded_weight
    by_cb = {}
    for r in results:
        cb = r.cart_bonded_weight
        if cb > 0:
            if cb not in by_cb:
                by_cb[cb] = []
            by_cb[cb].append(r)

    print("\nBy CART_BONDED_WEIGHT:")
    for cb in sorted(by_cb.keys()):
        cb_results = by_cb[cb]
        avg_bond = sum(r.max_bond_dev for r in cb_results) / len(cb_results)
        avg_angle = sum(r.max_angle_dev for r in cb_results) / len(cb_results)
        avg_time = sum(r.total_time for r in cb_results) / len(cb_results)
        print(f"  {cb}: n={len(cb_results)}, avg_max_bond={avg_bond:.4f}A, avg_max_angle={avg_angle:.2f}deg, avg_time={avg_time:.1f}s")

    # By repeats x stages
    by_rounds = {}
    for r in results:
        if r.repeats > 0 and r.stages > 0:
            total = r.repeats * r.stages
            if total not in by_rounds:
                by_rounds[total] = []
            by_rounds[total].append(r)

    print("\nBy TOTAL INTERNAL ROUNDS (repeats x stages):")
    for total in sorted(by_rounds.keys()):
        r_list = by_rounds[total]
        avg_bond = sum(r.max_bond_dev for r in r_list) / len(r_list)
        avg_angle = sum(r.max_angle_dev for r in r_list) / len(r_list)
        avg_time = sum(r.total_time for r in r_list) / len(r_list)
        print(f"  {total} rounds: n={len(r_list)}, avg_max_bond={avg_bond:.4f}A, avg_max_angle={avg_angle:.2f}deg, avg_time={avg_time:.1f}s")

    print()
    print("=" * 140)
    print("RECOMMENDATIONS")
    print("=" * 140)

    # Get best overall
    results.sort(key=score_overall)
    best = results[0]
    print(f"\nBest overall configuration: {best.job_name}")
    print(f"  Scorefunction: {best.scorefunction}")
    print(f"  Repeats x Stages: {best.repeats} x {best.stages} = {best.repeats * best.stages}")
    print(f"  Cart_bonded weight: {best.cart_bonded_weight}")
    print(f"  Bond geometry min: {'ON' if best.enable_bond_geometry_min else 'OFF'}")
    print(f"  Results (unconstrained metrics):")
    print(f"    Max bond deviation: {best.max_bond_dev:.4f}A (tolerance: 0.05A)")
    print(f"    Max angle deviation: {best.max_angle_dev:.2f}deg (tolerance: 10deg)")
    print(f"    CA RMSD: {best.ca_rmsd:.4f}A")
    print(f"    Constrained atom RMSD: {best.constrained_rmsd:.4f}A")
    print(f"    Ligand RMSD: {best.ligand_rmsd:.4f}A")
    print(f"    Catres failing: {best.catres_failing}/{best.catres_total}")
    print(f"    Clashes: {best.clashes_after}")
    print(f"    Runtime: {best.total_time:.1f}s")

    # Find best that meets tolerances
    print("\n--- Configurations meeting tolerances (bond < 0.05A, angle < 10deg): ---")
    meeting_tol = [r for r in results if r.max_bond_dev < 0.05 and r.max_angle_dev < 10.0]
    if meeting_tol:
        meeting_tol.sort(key=lambda r: r.ca_rmsd)  # Sort by CA RMSD if meeting tolerance
        best_tol = meeting_tol[0]
        print(f"  Best (by CA RMSD): {best_tol.job_name}")
        print(f"    max_bond={best_tol.max_bond_dev:.4f}A, max_angle={best_tol.max_angle_dev:.2f}deg, ca_rmsd={best_tol.ca_rmsd:.4f}A, time={best_tol.total_time:.1f}s")
    else:
        print("  No configurations meet both tolerance thresholds.")
        # Find closest
        results.sort(key=lambda r: max(r.max_bond_dev / 0.05, r.max_angle_dev / 10.0))
        closest = results[0]
        print(f"  Closest: {closest.job_name}")
        print(f"    max_bond={closest.max_bond_dev:.4f}A ({closest.max_bond_dev/0.05:.1f}x tol), max_angle={closest.max_angle_dev:.2f}deg ({closest.max_angle_dev/10.0:.1f}x tol)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze parameter sweep results")
    parser.add_argument("--output_dir", type=str,
                        default="/home/woodbuse/special_scripts/upgraded_fastMPNNdesign/modules/step02__constrained_cart_relax/hyper_param_sweep/outputs",
                        help="Directory containing sweep output directories")
    args = parser.parse_args()

    analyze_sweep(args.output_dir)
