#!/usr/bin/env python3
"""
Analyze and compare comprehensive test results for fastMPNNdesign.

Compares catres geometry, displacement metrics, and identifies the best parameters.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import sys


def load_relax_results(test_dir: Path) -> List[Dict]:
    """Load all relax results from a test directory."""
    results = []
    relax_dir = test_dir / "cycle_01" / "relax"
    if not relax_dir.exists():
        return results

    for json_file in sorted(relax_dir.glob("*_relax_result.json")):
        with open(json_file) as f:
            data = json.load(f)
            data['file'] = json_file.name
            results.append(data)

    return results


def analyze_catres_displacement(results: List[Dict], catres_names: List[str]) -> Dict:
    """Analyze displacement for specific catres residues."""
    if not results:
        return {}

    catres_displacements = {name: [] for name in catres_names}

    for result in results:
        for atom, disp in result.get('displacements', {}).items():
            for catres in catres_names:
                if atom.startswith(catres + '_'):
                    catres_displacements[catres].append(disp)

    summary = {}
    for catres, disps in catres_displacements.items():
        if disps:
            summary[catres] = {
                'mean': sum(disps) / len(disps),
                'max': max(disps),
                'n_atoms': len(disps)
            }

    return summary


def analyze_test(test_dir: Path) -> Dict:
    """Analyze a single test directory."""
    test_dir = Path(test_dir)

    # Load config
    config_file = test_dir / "run_config.json"
    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

    # Load results
    results = load_relax_results(test_dir)

    if not results:
        return {
            'test_name': test_dir.name,
            'status': 'no_results',
            'config': config
        }

    # Key metrics
    mean_displacements = [r['mean_displacement'] for r in results]
    max_displacements = [r['max_displacement'] for r in results]
    cart_bonded_scores = [r.get('cart_bonded_score', 0) for r in results]
    ligand_displacements = [r.get('mean_ligand_displacement', 0) for r in results]

    # Analyze secondary catres
    secondary_catres = ['A_19', 'A_21', 'A_30']
    catres_analysis = analyze_catres_displacement(results, secondary_catres)

    return {
        'test_name': test_dir.name,
        'status': 'complete' if len(results) >= 5 else f'{len(results)}/5 structures',
        'n_structures': len(results),
        'metrics': {
            'mean_displacement': {
                'avg': sum(mean_displacements) / len(mean_displacements),
                'min': min(mean_displacements),
                'max': max(mean_displacements)
            },
            'max_displacement': {
                'avg': sum(max_displacements) / len(max_displacements),
                'min': min(max_displacements),
                'max': max(max_displacements)
            },
            'cart_bonded': {
                'avg': sum(cart_bonded_scores) / len(cart_bonded_scores),
                'min': min(cart_bonded_scores),
                'max': max(cart_bonded_scores)
            },
            'ligand_displacement': {
                'avg': sum(ligand_displacements) / len(ligand_displacements),
                'max': max(ligand_displacements)
            }
        },
        'secondary_catres': catres_analysis,
        'config_summary': {
            'coord_cst_stdev': config.get('constraints', {}).get('coord_cst_stdev'),
            'coord_cst_weight': config.get('constraints', {}).get('coord_cst_weight'),
            'cart_bonded_weight': config.get('relax', {}).get('cart_bonded_weight'),
            'allow_catres_bb': config.get('relax', {}).get('allow_catres_bb'),
            'mobile_radius': config.get('relax', {}).get('mobile_radius'),
            'fastrelax_cycles': config.get('relax', {}).get('fastrelax_cycles')
        }
    }


def print_analysis(analysis: Dict):
    """Print analysis results in a readable format."""
    print(f"\n{'='*60}")
    print(f"  {analysis['test_name']}")
    print(f"{'='*60}")

    print(f"  Status: {analysis['status']}")
    print(f"  Structures: {analysis['n_structures']}")

    if analysis['status'] == 'no_results':
        print("  (No results yet)")
        return

    print(f"\n  Configuration:")
    for key, val in analysis['config_summary'].items():
        if val is not None:
            print(f"    {key}: {val}")

    print(f"\n  Metrics:")
    metrics = analysis['metrics']
    print(f"    Mean displacement:  avg={metrics['mean_displacement']['avg']:.3f} A")
    print(f"    Max displacement:   avg={metrics['max_displacement']['avg']:.3f} A")
    print(f"    Cart bonded score:  avg={metrics['cart_bonded']['avg']:.1f}")
    print(f"    Ligand displacement: avg={metrics['ligand_displacement']['avg']:.6f} A")

    print(f"\n  Secondary Catres (A19, A21, A30):")
    for catres, data in analysis['secondary_catres'].items():
        print(f"    {catres}: mean={data['mean']:.3f} A, max={data['max']:.3f} A")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_test_results.py <test_dir1> [test_dir2] ...")
        print("\nExample:")
        print("  python analyze_test_results.py comprehensive_test1 comprehensive_test2")
        sys.exit(1)

    base_path = Path("/net/scratch/woodbuse/organophosphatase/round2/fastMPNNdesign_out/i1")

    analyses = []
    for test_name in sys.argv[1:]:
        test_dir = base_path / test_name if not Path(test_name).is_absolute() else Path(test_name)
        if test_dir.exists():
            analysis = analyze_test(test_dir)
            analyses.append(analysis)
            print_analysis(analysis)
        else:
            print(f"\nTest directory not found: {test_dir}")

    # Summary comparison
    if len(analyses) > 1:
        print(f"\n{'='*60}")
        print("  COMPARISON SUMMARY")
        print(f"{'='*60}")

        # Find best results
        valid = [a for a in analyses if a['status'] != 'no_results']
        if valid:
            best_mean_disp = min(valid, key=lambda x: x['metrics']['mean_displacement']['avg'])
            best_max_disp = min(valid, key=lambda x: x['metrics']['max_displacement']['avg'])
            best_cart_bonded = min(valid, key=lambda x: x['metrics']['cart_bonded']['avg'])

            print(f"\n  Best mean displacement: {best_mean_disp['test_name']}")
            print(f"    value: {best_mean_disp['metrics']['mean_displacement']['avg']:.3f} A")

            print(f"\n  Best max displacement: {best_max_disp['test_name']}")
            print(f"    value: {best_max_disp['metrics']['max_displacement']['avg']:.3f} A")

            print(f"\n  Best cart_bonded: {best_cart_bonded['test_name']}")
            print(f"    value: {best_cart_bonded['metrics']['cart_bonded']['avg']:.1f}")


if __name__ == "__main__":
    main()
