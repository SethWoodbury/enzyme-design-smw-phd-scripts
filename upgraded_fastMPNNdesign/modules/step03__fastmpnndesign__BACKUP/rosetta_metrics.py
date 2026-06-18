#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Container-safe metrics runner for FastMPNN design.

This script is intended to be executed inside the PyRosetta apptainer
and writes metrics to JSON for use by the host process.
"""
import argparse
import json
import logging
import os
import sys

# Ensure repo root is on sys.path so package-relative imports work.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(MODULES_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.step03__fastmpnndesign.metrics import MetricsCalculator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def _parse_ligand_info(value: str):
    """Parse ligand_info string (chain,resname,resno)."""
    if not value:
        return None
    token = value.replace(":", ",")
    parts = [p.strip() for p in token.split(",") if p.strip()]
    if len(parts) < 3:
        return None
    chain = parts[0]
    resname = parts[1]
    try:
        resno = int(parts[2])
    except ValueError:
        return None
    return (chain, resname, resno)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute metrics inside PyRosetta container")
    parser.add_argument("--designed_pdb", required=True, help="Designed PDB path")
    parser.add_argument("--step02_pdb", required=True, help="Step02 PDB path")
    parser.add_argument("--step01_pdb", default=None, help="Step01 PDB path (optional)")
    parser.add_argument("--params", nargs="+", default=[], help="Ligand .params files")
    parser.add_argument("--constraints_json", default=None, help="Constraints JSON file")
    parser.add_argument("--catres_positions_json", default=None, help="Catres positions JSON file")
    parser.add_argument("--motif_positions_json", default=None, help="Motif positions JSON file")
    parser.add_argument("--ligand_info", default=None, help="Ligand info: chain,resname,resno")
    parser.add_argument("--bond_length_tolerance", type=float, default=0.05)
    parser.add_argument("--bond_angle_tolerance", type=float, default=10.0)
    parser.add_argument("--catres_bond_tolerance", type=float, default=0.05)
    parser.add_argument("--catres_angle_tolerance", type=float, default=7.5)
    parser.add_argument("--mode", choices=["all", "comprehensive", "score"], default="all")
    parser.add_argument("--output_json", required=True, help="Output metrics JSON")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load constraints
    constraints = {}
    if args.constraints_json:
        with open(args.constraints_json, "r") as f:
            constraints = json.load(f)

    # Load catres positions
    catres_positions = []
    if args.catres_positions_json:
        with open(args.catres_positions_json, "r") as f:
            raw_positions = json.load(f)
        catres_positions = [tuple(p) for p in raw_positions if p]

    motif_positions = []
    if args.motif_positions_json:
        with open(args.motif_positions_json, "r") as f:
            raw_positions = json.load(f)
        motif_positions = [tuple(p) for p in raw_positions if p]

    ligand_info = _parse_ligand_info(args.ligand_info)

    calc = MetricsCalculator(
        designed_pdb=args.designed_pdb,
        step02_pdb=args.step02_pdb,
        step01_pdb=args.step01_pdb,
        params_files=args.params,
        constrained_atoms=constraints,
        catres_positions=catres_positions,
        motif_positions=motif_positions,
        ligand_info=ligand_info,
        bond_length_tolerance=args.bond_length_tolerance,
        bond_angle_tolerance=args.bond_angle_tolerance,
        catres_bond_tolerance=args.catres_bond_tolerance,
        catres_angle_tolerance=args.catres_angle_tolerance,
        use_container_fallback=False,
    )

    if args.mode == "comprehensive":
        metrics = calc.calculate_comprehensive_metrics()
    elif args.mode == "score":
        metrics = {"rosetta_score": calc.calculate_rosetta_score()}
    else:
        metrics = calc.calculate_all_metrics()

    with open(args.output_json, "w") as f:
        json.dump(metrics, f, indent=2)

    LOGGER.info(f"Metrics written to {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
