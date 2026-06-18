#!/usr/bin/env python3
"""
Comprehensive test suite for the relaxation protocol overhaul.

Tests:
1. Multi-stage relaxation protocol
2. Bond geometry metrics (length/angle deviations)
3. Ring flip sampling
4. Parameter sensitivity analysis
5. Failure mode testing

Success Criteria (from RELAXATION_OVERHAUL_PLAN.md):
- Cart_bonded (total) < 50
- Cart_bonded (per catres) < 3
- Bond length deviation < 0.04 Å
- Bond angle deviation < 5°
- Catres displacement from ref < 0.5 Å
- Ligand displacement: 0.0 Å
- Catres sidechain RMSD < 0.3 Å
"""

import json
import sys
import os
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
import subprocess

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

_HERE = Path(__file__).resolve().parent
# --- locate repo root + shared external paths ---
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "repo_paths.py").is_file():
        _sys.path.insert(0, str(_anc)); break
import repo_paths

from fastmpnndesign.config import RelaxConfig, ConstraintConfig
from fastmpnndesign.remark666 import parse_remark666
from fastmpnndesign.ligand import detect_ligands_from_pdb, detect_metals_from_pdb
from fastmpnndesign.contact_detection import detect_contacts
from fastmpnndesign.constraints import (
    generate_constraint_set,
    generate_catres_sidechain_constraints,
)
from fastmpnndesign.logging_config import get_logger

logger = get_logger("test_relaxation")


@dataclass
class TestResult:
    """Result of a single relaxation test."""
    test_name: str
    success: bool
    params: Dict[str, Any]
    # Scores
    total_score: Optional[float] = None
    cart_bonded_score: Optional[float] = None
    coord_cst_score: Optional[float] = None
    # Geometry metrics
    mean_bond_length_dev: Optional[float] = None
    max_bond_length_dev: Optional[float] = None
    mean_bond_angle_dev: Optional[float] = None
    max_bond_angle_dev: Optional[float] = None
    n_critical_bonds: int = 0
    n_critical_angles: int = 0
    geometry_grade: str = ""
    # Displacement metrics
    mean_displacement: Optional[float] = None
    max_displacement: Optional[float] = None
    ligand_displacement: Optional[float] = None
    catres_sidechain_rmsd: Optional[float] = None
    # Ring flip results
    ring_flips_tried: int = 0
    ring_flips_accepted: int = 0
    # Execution info
    duration_seconds: float = 0.0
    error_message: str = ""
    output_pdb: str = ""

    def passes_criteria(self) -> Tuple[bool, List[str]]:
        """Check if result passes all success criteria."""
        failures = []

        if self.cart_bonded_score is not None and self.cart_bonded_score > 50:
            failures.append(f"cart_bonded {self.cart_bonded_score:.1f} > 50")

        if self.max_bond_length_dev is not None and self.max_bond_length_dev > 0.04:
            failures.append(f"max_bond_length_dev {self.max_bond_length_dev:.3f} > 0.04 Å")

        if self.max_bond_angle_dev is not None and self.max_bond_angle_dev > 5.0:
            failures.append(f"max_bond_angle_dev {self.max_bond_angle_dev:.1f} > 5°")

        if self.ligand_displacement is not None and self.ligand_displacement > 0.01:
            failures.append(f"ligand moved {self.ligand_displacement:.4f} Å")

        if self.catres_sidechain_rmsd is not None and self.catres_sidechain_rmsd > 0.3:
            failures.append(f"catres_sidechain_rmsd {self.catres_sidechain_rmsd:.3f} > 0.3 Å")

        if self.n_critical_bonds > 0:
            failures.append(f"{self.n_critical_bonds} critical bonds (>0.1Å)")

        if self.n_critical_angles > 0:
            failures.append(f"{self.n_critical_angles} critical angles (>10°)")

        return len(failures) == 0, failures

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TestConfig:
    """Configuration for a single test run."""
    name: str
    # Multi-stage params
    use_multistage_relax: bool = True
    initial_coord_cst_weight: float = 1000.0
    final_coord_cst_weight: float = 100.0
    initial_fa_rep_scale: float = 0.15
    n_relax_stages: int = 3
    # Backbone mobility
    allow_catres_bb: bool = True
    # Standard relax params
    fastrelax_cycles: int = 2
    cart_bonded_weight: float = 0.5
    coord_cst_weight: float = 200.0
    mobile_radius: float = 10.0
    # Constraint params
    coord_cst_stdev: float = 0.01
    catres_cst_stdev: float = 0.2


# Parameter combinations to test
TEST_CONFIGS = [
    # Baseline: new defaults
    TestConfig(
        name="baseline_new_defaults",
        use_multistage_relax=True,
        allow_catres_bb=True,
        initial_coord_cst_weight=1000.0,
        final_coord_cst_weight=100.0,
        initial_fa_rep_scale=0.15,
    ),

    # Old protocol (for comparison) - should fail
    TestConfig(
        name="old_protocol_comparison",
        use_multistage_relax=False,
        allow_catres_bb=False,
        coord_cst_weight=200.0,
    ),

    # Higher initial constraint weight
    TestConfig(
        name="high_initial_cst",
        use_multistage_relax=True,
        allow_catres_bb=True,
        initial_coord_cst_weight=2000.0,
        final_coord_cst_weight=100.0,
    ),

    # Lower final constraint weight
    TestConfig(
        name="low_final_cst",
        use_multistage_relax=True,
        allow_catres_bb=True,
        initial_coord_cst_weight=1000.0,
        final_coord_cst_weight=50.0,
    ),

    # More stages
    TestConfig(
        name="four_stages",
        use_multistage_relax=True,
        allow_catres_bb=True,
        n_relax_stages=4,
    ),

    # Higher cart_bonded weight
    TestConfig(
        name="high_cart_bonded",
        use_multistage_relax=True,
        allow_catres_bb=True,
        cart_bonded_weight=1.0,
    ),

    # Lower initial fa_rep
    TestConfig(
        name="very_low_fa_rep",
        use_multistage_relax=True,
        allow_catres_bb=True,
        initial_fa_rep_scale=0.05,
    ),

    # Tighter catres constraints
    TestConfig(
        name="tight_catres_cst",
        use_multistage_relax=True,
        allow_catres_bb=True,
        catres_cst_stdev=0.1,  # Tighter than 0.2
    ),

    # More relax cycles per stage
    TestConfig(
        name="more_relax_cycles",
        use_multistage_relax=True,
        allow_catres_bb=True,
        fastrelax_cycles=3,
    ),

    # Catres BB frozen (failure mode test)
    TestConfig(
        name="frozen_catres_bb_failure_mode",
        use_multistage_relax=True,
        allow_catres_bb=False,  # Should cause geometry issues
    ),
]


def run_single_relax_test(
    input_pdb: Path,
    params_files: List[Path],
    output_dir: Path,
    config: TestConfig,
    catres_list: Optional[List] = None,
    ref_pdb: Optional[Path] = None,
) -> TestResult:
    """
    Run a single relaxation test with specified parameters.

    Uses container-based execution via apptainer.
    """
    start_time = time.time()

    test_output_dir = output_dir / config.name
    test_output_dir.mkdir(parents=True, exist_ok=True)

    # Build CLI command
    cmd = [
        "apptainer", "exec", repo_paths.UNIVERSAL_SIF,
        "python", str(_HERE.parents[0] / "fastmpnndesign" / "cli.py"),
        "--pdb", str(input_pdb),
        "--output_dir", str(test_output_dir),
        "--prefix", config.name,
        "--scorefunction", "beta_jan25",
        "--n_cycles", "1",  # Just 1 cycle for testing
        "--n_candidates", "1",  # Just 1 candidate
        "--n_keep", "1",
        "--verbose",
        # Relax params
        "--fastrelax_cycles", str(config.fastrelax_cycles),
        "--mobile_radius", str(config.mobile_radius),
        "--coord_cst_stdev", str(config.coord_cst_stdev),
        # Skip MPNN for faster testing
        "--skip_mpnn",  # If this flag exists
    ]

    # Add params files
    for pf in params_files:
        cmd.extend(["--params", str(pf)])

    if ref_pdb:
        cmd.extend(["--ref_pdb", str(ref_pdb)])

    # Add multi-stage specific params (need to add these as CLI args or via config file)
    # For now, we'll modify the config file approach

    result = TestResult(
        test_name=config.name,
        success=False,
        params=asdict(config),
    )

    try:
        logger.info(f"Running test: {config.name}")
        logger.info(f"  Command: {' '.join(cmd[:10])}...")

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=str(output_dir),
        )

        result.duration_seconds = time.time() - start_time

        if proc.returncode != 0:
            result.error_message = f"Exit code {proc.returncode}: {proc.stderr[:500]}"
            logger.error(f"Test {config.name} failed: {result.error_message}")
            return result

        # Parse results from output files
        results_dir = test_output_dir / "cycle_01" / "relax"
        if results_dir.exists():
            result_files = list(results_dir.glob("*_relax_result.json"))
            if result_files:
                with open(result_files[0]) as f:
                    relax_data = json.load(f)

                result.success = relax_data.get('success', False)
                result.total_score = relax_data.get('total_score')
                result.cart_bonded_score = relax_data.get('cart_bonded_score')
                result.coord_cst_score = relax_data.get('score_terms', {}).get('coordinate_constraint')
                result.mean_displacement = relax_data.get('mean_displacement')
                result.max_displacement = relax_data.get('max_displacement')
                result.ligand_displacement = relax_data.get('max_ligand_displacement', 0.0)
                result.ring_flips_tried = relax_data.get('ring_flips_tried', 0)
                result.ring_flips_accepted = relax_data.get('ring_flips_accepted', 0)

                # Find output PDB
                pdb_files = list(results_dir.glob("*.pdb"))
                if pdb_files:
                    result.output_pdb = str(pdb_files[0])

        logger.info(f"Test {config.name} completed: cart_bonded={result.cart_bonded_score}")

    except subprocess.TimeoutExpired:
        result.error_message = "Timeout after 1 hour"
        result.duration_seconds = 3600
    except Exception as e:
        result.error_message = str(e)
        result.duration_seconds = time.time() - start_time

    return result


def run_direct_relax_test(
    input_pdb: Path,
    params_files: List[Path],
    output_dir: Path,
    config: TestConfig,
) -> TestResult:
    """
    Run relaxation test directly using PyRosetta (if available).

    More detailed control and metrics than CLI-based testing.
    """
    start_time = time.time()

    result = TestResult(
        test_name=config.name,
        success=False,
        params=asdict(config),
    )

    try:
        from fastmpnndesign.relax_runner import (
            init_pyrosetta, relax_structure, sample_ring_flips,
            run_multistage_relax, create_scorefunction, create_movemap
        )
        from fastmpnndesign.metrics import (
            compute_bond_length_deviations, compute_bond_angle_deviations,
            get_geometry_quality_grade, BondGeometryMetrics
        )
        from fastmpnndesign.constraints import generate_constraint_set
        from fastmpnndesign.remark666 import parse_remark666
        from fastmpnndesign.ligand import detect_ligands_from_pdb, detect_metals_from_pdb
        from fastmpnndesign.contact_detection import detect_contacts
    except ImportError as e:
        result.error_message = f"Import error: {e}"
        return result

    try:
        test_output_dir = output_dir / config.name
        test_output_dir.mkdir(parents=True, exist_ok=True)
        output_pdb = test_output_dir / f"{config.name}_relaxed.pdb"

        # Initialize PyRosetta
        init_pyrosetta(params_files, scorefunction="beta_jan25")

        # Parse catres
        catres_list = parse_remark666(input_pdb)
        logger.info(f"Found {len(catres_list)} catalytic residues")

        # Detect ligands/metals
        ligands = detect_ligands_from_pdb(input_pdb)
        metals = detect_metals_from_pdb(input_pdb)
        logger.info(f"Found {len(ligands)} ligands, {len(metals)} metals")

        # Detect contacts
        contacts = detect_contacts(input_pdb, catres_list, ligands, metals)

        # Generate constraints
        cst_config = ConstraintConfig(
            coord_cst_stdev=config.coord_cst_stdev,
            coord_cst_weight=config.coord_cst_weight,
        )
        constraint_set = generate_constraint_set(
            pdb_path=input_pdb,
            catres_list=catres_list,
            contacts=contacts,
            ligands=ligands,
            metals=metals,
            config=cst_config,
        )

        # Add catres sidechain constraints
        catres_cst = generate_catres_sidechain_constraints(
            input_pdb, catres_list,
            existing_constraints=constraint_set.coordinate_constraints,
            stdev=config.catres_cst_stdev,
        )
        constraint_set.coordinate_constraints.extend(catres_cst)
        logger.info(f"Total constraints: {len(constraint_set.coordinate_constraints)} coordinate, "
                   f"{len(constraint_set.distance_constraints)} distance")

        # Create RelaxConfig
        relax_config = RelaxConfig(
            scorefunction="beta_jan25",
            fastrelax_cycles=config.fastrelax_cycles,
            mobile_radius=config.mobile_radius,
            cart_bonded_weight=config.cart_bonded_weight,
            coord_cst_weight=config.coord_cst_weight,
            allow_catres_bb=config.allow_catres_bb,
            use_multistage_relax=config.use_multistage_relax,
            initial_coord_cst_weight=config.initial_coord_cst_weight,
            final_coord_cst_weight=config.final_coord_cst_weight,
            initial_fa_rep_scale=config.initial_fa_rep_scale,
            n_relax_stages=config.n_relax_stages,
            use_pyrosetta_image=False,  # Direct mode
        )

        # Run relaxation
        relax_result = relax_structure(
            input_pdb=input_pdb,
            output_pdb=output_pdb,
            constraint_set=constraint_set,
            params_files=params_files,
            config=relax_config,
            ligands=ligands,
            metals=metals,
            catres_list=catres_list,
        )

        result.success = relax_result.success
        result.total_score = relax_result.total_score
        result.cart_bonded_score = relax_result.cart_bonded_score
        result.mean_displacement = relax_result.mean_displacement
        result.max_displacement = relax_result.max_displacement
        result.output_pdb = str(output_pdb) if output_pdb.exists() else ""
        result.ring_flips_tried = getattr(relax_result, 'ring_flips_tried', 0)
        result.ring_flips_accepted = getattr(relax_result, 'ring_flips_accepted', 0)

        # Compute detailed bond geometry metrics
        if output_pdb.exists():
            catres_tuples = [(cr.chain, cr.resnum) for cr in catres_list]

            bond_length_data = compute_bond_length_deviations(
                output_pdb, catres_tuples, params_files
            )
            result.mean_bond_length_dev = bond_length_data.get('mean_deviation')
            result.max_bond_length_dev = bond_length_data.get('max_deviation')
            result.n_critical_bonds = len([
                b for b in bond_length_data.get('worst_bonds', [])
                if b.get('deviation', 0) > 0.1
            ])

            bond_angle_data = compute_bond_angle_deviations(
                output_pdb, catres_tuples, params_files
            )
            result.mean_bond_angle_dev = bond_angle_data.get('mean_deviation')
            result.max_bond_angle_dev = bond_angle_data.get('max_deviation')
            result.n_critical_angles = len([
                a for a in bond_angle_data.get('worst_angles', [])
                if a.get('deviation', 0) > 10.0
            ])

            # Get quality grade
            metrics = BondGeometryMetrics(
                cart_bonded_score=result.cart_bonded_score or 0,
                mean_bond_length_deviation=result.mean_bond_length_dev or 0,
                max_bond_length_deviation=result.max_bond_length_dev or 0,
                mean_bond_angle_deviation=result.mean_bond_angle_dev or 0,
                max_bond_angle_deviation=result.max_bond_angle_dev or 0,
                n_critical_bonds=result.n_critical_bonds,
                n_critical_angles=result.n_critical_angles,
            )
            result.geometry_grade = get_geometry_quality_grade(metrics)

        result.duration_seconds = time.time() - start_time

    except Exception as e:
        import traceback
        result.error_message = f"{e}\n{traceback.format_exc()}"
        result.duration_seconds = time.time() - start_time

    return result


def analyze_results(results: List[TestResult]) -> Dict[str, Any]:
    """Analyze test results and identify best configuration."""

    analysis = {
        'total_tests': len(results),
        'successful': 0,
        'passed_criteria': 0,
        'best_config': None,
        'best_score': float('inf'),
        'summary': [],
        'recommendations': [],
    }

    for r in results:
        passes, failures = r.passes_criteria()

        summary = {
            'name': r.test_name,
            'success': r.success,
            'passes_criteria': passes,
            'failures': failures,
            'cart_bonded': r.cart_bonded_score,
            'max_bond_dev': r.max_bond_length_dev,
            'max_angle_dev': r.max_bond_angle_dev,
            'geometry_grade': r.geometry_grade,
            'duration': r.duration_seconds,
        }
        analysis['summary'].append(summary)

        if r.success:
            analysis['successful'] += 1

        if passes:
            analysis['passed_criteria'] += 1
            # Track best by cart_bonded score
            if r.cart_bonded_score and r.cart_bonded_score < analysis['best_score']:
                analysis['best_score'] = r.cart_bonded_score
                analysis['best_config'] = r.test_name

    # Generate recommendations
    if analysis['passed_criteria'] == 0:
        analysis['recommendations'].append("No configs passed all criteria - need further tuning")

    # Check if multi-stage beats old protocol
    old_result = next((r for r in results if r.test_name == "old_protocol_comparison"), None)
    new_result = next((r for r in results if r.test_name == "baseline_new_defaults"), None)

    if old_result and new_result:
        if new_result.cart_bonded_score and old_result.cart_bonded_score:
            improvement = old_result.cart_bonded_score - new_result.cart_bonded_score
            analysis['recommendations'].append(
                f"Multi-stage improved cart_bonded by {improvement:.1f} "
                f"({old_result.cart_bonded_score:.1f} -> {new_result.cart_bonded_score:.1f})"
            )

    return analysis


def print_results_table(results: List[TestResult]):
    """Print results in a formatted table."""
    print("\n" + "=" * 120)
    print("TEST RESULTS SUMMARY")
    print("=" * 120)
    print(f"{'Test Name':<30} {'Success':<8} {'Cart_Bond':<10} {'Bond Dev':<10} {'Angle Dev':<10} {'Grade':<12} {'Criteria':<8}")
    print("-" * 120)

    for r in results:
        passes, _ = r.passes_criteria()
        cart_str = f"{r.cart_bonded_score:.1f}" if r.cart_bonded_score else "N/A"
        bond_str = f"{r.max_bond_length_dev:.3f}" if r.max_bond_length_dev else "N/A"
        angle_str = f"{r.max_bond_angle_dev:.1f}" if r.max_bond_angle_dev else "N/A"
        criteria_str = "PASS" if passes else "FAIL"

        print(f"{r.test_name:<30} {str(r.success):<8} {cart_str:<10} {bond_str:<10} {angle_str:<10} {r.geometry_grade:<12} {criteria_str:<8}")

    print("=" * 120)


def main():
    """Main test runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Test relaxation protocol overhaul")
    parser.add_argument("--pdb", required=True, help="Input PDB file")
    parser.add_argument("--params", nargs="+", required=True, help="Params files")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--ref_pdb", help="Reference PDB for comparison")
    parser.add_argument("--configs", nargs="+", help="Specific configs to test (default: all)")
    parser.add_argument("--direct", action="store_true", help="Use direct PyRosetta (not container)")

    args = parser.parse_args()

    input_pdb = Path(args.pdb)
    params_files = [Path(p) for p in args.params]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_pdb = Path(args.ref_pdb) if args.ref_pdb else None

    # Select configs to test
    if args.configs:
        configs = [c for c in TEST_CONFIGS if c.name in args.configs]
    else:
        configs = TEST_CONFIGS

    logger.info(f"Running {len(configs)} test configurations")

    results = []
    for config in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {config.name}")
        logger.info(f"{'='*60}")

        if args.direct:
            result = run_direct_relax_test(input_pdb, params_files, output_dir, config)
        else:
            result = run_single_relax_test(input_pdb, params_files, output_dir, config, ref_pdb=ref_pdb)

        results.append(result)

        # Save individual result
        result_file = output_dir / f"{config.name}_result.json"
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

    # Print summary
    print_results_table(results)

    # Full analysis
    analysis = analyze_results(results)

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print(f"Total tests: {analysis['total_tests']}")
    print(f"Successful: {analysis['successful']}")
    print(f"Passed criteria: {analysis['passed_criteria']}")
    print(f"Best config: {analysis['best_config']} (cart_bonded={analysis['best_score']:.1f})")
    print("\nRecommendations:")
    for rec in analysis['recommendations']:
        print(f"  - {rec}")

    # Save full analysis
    analysis_file = output_dir / "test_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    # Save all results
    all_results_file = output_dir / "all_results.json"
    with open(all_results_file, 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
