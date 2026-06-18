#!/usr/bin/env python3
"""
Analyze detailed metrics from hyperparameter sweep.

Reports on:
1. Ligand RMSD (should be ~0.0 if constraints working)
2. Catres heavy atom RMSD (should be ~0.0 if constraints working)
3. CA/backbone RMSD vs step01 and step02
4. Bond length deviations (mean/max) for unconstrained atoms
5. Bond angle deviations (mean/max) for unconstrained atoms
6. Sequence diversity and mutation statistics
7. Per-protocol and per-hyperparameter analysis
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np


def load_metrics(json_path: str) -> list:
    """Load detailed metrics JSON."""
    with open(json_path) as f:
        return json.load(f)


def analyze_metrics(data: list) -> dict:
    """Comprehensive analysis of metrics."""

    # Collect all values for statistical analysis
    all_ligand_rmsd = []
    all_catres_rmsd = []
    all_ca_rmsd_step02 = []
    all_ca_rmsd_step01 = []
    all_bb_rmsd_step02 = []
    all_bb_rmsd_step01 = []
    all_bond_len_mean = []
    all_bond_len_max = []
    all_bond_ang_mean = []
    all_bond_ang_max = []
    all_bond_len_catres_mean = []
    all_bond_len_catres_max = []
    all_mutations = []
    all_seq_identity = []
    all_sequences = set()

    # Per-protocol metrics
    by_protocol = defaultdict(lambda: {
        'ligand_rmsd': [],
        'catres_rmsd': [],
        'ca_rmsd_step02': [],
        'ca_rmsd_step01': [],
        'bond_len_max': [],
        'bond_ang_max': [],
        'mutations': [],
        'sequences': set(),
        'runtimes': [],
        'num_designs': [],
    })

    # Per-hyperparameter metrics
    by_param = defaultdict(lambda: defaultdict(list))

    job_count = 0
    design_count = 0

    for job in data:
        job_name = job.get('job_name', '')
        protocol = job.get('protocol', 'unknown')
        runtime = job.get('runtime', 0)
        num_designs = job.get('num_designs', 0)

        job_count += 1
        by_protocol[protocol]['runtimes'].append(runtime)
        by_protocol[protocol]['num_designs'].append(num_designs)

        # Extract hyperparameter from job name (e.g., job0123__temp_0.2__rep1)
        parts = job_name.split('__')
        if len(parts) >= 2:
            param_part = parts[1]
            # Parse parameter name and value
            if '_' in param_part and not param_part.startswith('rep'):
                param_name = '_'.join(param_part.split('_')[:-1])
                param_value = param_part.split('_')[-1]
            else:
                param_name = param_part
                param_value = 'default'
        else:
            param_name = 'protocol'
            param_value = protocol

        designs = job.get('designs', [])
        for design in designs:
            design_count += 1

            # Ligand RMSD
            lig_rmsd = design.get('ligand_rmsd')
            if lig_rmsd is not None and lig_rmsd > 0:
                all_ligand_rmsd.append(lig_rmsd)
                by_protocol[protocol]['ligand_rmsd'].append(lig_rmsd)
                by_param[param_name][param_value].append(('ligand_rmsd', lig_rmsd))

            # Catres RMSD
            cat_rmsd = design.get('catres_heavy_rmsd')
            if cat_rmsd is not None:
                all_catres_rmsd.append(cat_rmsd)
                by_protocol[protocol]['catres_rmsd'].append(cat_rmsd)

            # CA RMSD
            ca_step02 = design.get('ca_rmsd_vs_step02')
            if ca_step02 is not None:
                all_ca_rmsd_step02.append(ca_step02)
                by_protocol[protocol]['ca_rmsd_step02'].append(ca_step02)

            ca_step01 = design.get('ca_rmsd_vs_step01')
            if ca_step01 is not None:
                all_ca_rmsd_step01.append(ca_step01)
                by_protocol[protocol]['ca_rmsd_step01'].append(ca_step01)

            # Backbone RMSD
            bb_step02 = design.get('backbone_rmsd_vs_step02')
            if bb_step02 is not None:
                all_bb_rmsd_step02.append(bb_step02)

            bb_step01 = design.get('backbone_rmsd_vs_step01')
            if bb_step01 is not None:
                all_bb_rmsd_step01.append(bb_step01)

            # Bond length deviations (unconstrained)
            bl_mean = design.get('bond_length_unconstrained_mean')
            if bl_mean is not None and bl_mean > 0:
                all_bond_len_mean.append(bl_mean)

            bl_max = design.get('bond_length_unconstrained_max')
            if bl_max is not None and bl_max > 0:
                all_bond_len_max.append(bl_max)
                by_protocol[protocol]['bond_len_max'].append(bl_max)

            # Bond angle deviations
            ba_mean = design.get('bond_angle_unconstrained_mean')
            if ba_mean is not None and ba_mean > 0:
                all_bond_ang_mean.append(ba_mean)

            ba_max = design.get('bond_angle_unconstrained_max')
            if ba_max is not None and ba_max > 0:
                all_bond_ang_max.append(ba_max)
                by_protocol[protocol]['bond_ang_max'].append(ba_max)

            # Catres bond geometry
            bl_cat_mean = design.get('bond_length_catres_mean')
            if bl_cat_mean is not None and bl_cat_mean > 0:
                all_bond_len_catres_mean.append(bl_cat_mean)

            bl_cat_max = design.get('bond_length_catres_max')
            if bl_cat_max is not None and bl_cat_max > 0:
                all_bond_len_catres_max.append(bl_cat_max)

            # Sequence metrics
            n_mut = design.get('num_mutations', 0)
            all_mutations.append(n_mut)
            by_protocol[protocol]['mutations'].append(n_mut)

            seq_id = design.get('sequence_identity')
            if seq_id is not None:
                all_seq_identity.append(seq_id)

            seq = design.get('sequence', '')
            if seq:
                all_sequences.add(seq)
                by_protocol[protocol]['sequences'].add(seq)

    # Compute statistics
    def stats(values):
        if not values:
            return {'n': 0, 'mean': None, 'std': None, 'min': None, 'max': None, 'median': None}
        arr = np.array(values)
        return {
            'n': len(values),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'median': float(np.median(arr)),
        }

    results = {
        'summary': {
            'total_jobs': job_count,
            'total_designs': design_count,
            'unique_sequences': len(all_sequences),
            'sequence_diversity': len(all_sequences) / design_count if design_count > 0 else 0,
        },
        'ligand_rmsd': stats(all_ligand_rmsd),
        'catres_heavy_rmsd': stats(all_catres_rmsd),
        'ca_rmsd_vs_step02': stats(all_ca_rmsd_step02),
        'ca_rmsd_vs_step01': stats(all_ca_rmsd_step01),
        'backbone_rmsd_vs_step02': stats(all_bb_rmsd_step02),
        'backbone_rmsd_vs_step01': stats(all_bb_rmsd_step01),
        'bond_length_unconstrained_mean': stats(all_bond_len_mean),
        'bond_length_unconstrained_max': stats(all_bond_len_max),
        'bond_angle_unconstrained_mean': stats(all_bond_ang_mean),
        'bond_angle_unconstrained_max': stats(all_bond_ang_max),
        'bond_length_catres_mean': stats(all_bond_len_catres_mean),
        'bond_length_catres_max': stats(all_bond_len_catres_max),
        'mutations': stats(all_mutations),
        'sequence_identity': stats(all_seq_identity),
        'by_protocol': {},
    }

    # Per-protocol summary
    for protocol, pdata in by_protocol.items():
        results['by_protocol'][protocol] = {
            'n_jobs': len(pdata['runtimes']),
            'n_designs': sum(pdata['num_designs']),
            'unique_sequences': len(pdata['sequences']),
            'mean_runtime': float(np.mean(pdata['runtimes'])) if pdata['runtimes'] else None,
            'ligand_rmsd': stats(pdata['ligand_rmsd']),
            'catres_rmsd': stats(pdata['catres_rmsd']),
            'ca_rmsd_step02': stats(pdata['ca_rmsd_step02']),
            'ca_rmsd_step01': stats(pdata['ca_rmsd_step01']),
            'bond_len_max': stats(pdata['bond_len_max']),
            'bond_ang_max': stats(pdata['bond_ang_max']),
            'mutations': stats(pdata['mutations']),
        }

    return results


def print_report(results: dict):
    """Print formatted analysis report."""

    print("=" * 80)
    print("HYPERPARAMETER SWEEP DETAILED METRICS ANALYSIS")
    print("=" * 80)
    print()

    # Summary
    s = results['summary']
    print(f"SUMMARY")
    print(f"  Total jobs: {s['total_jobs']}")
    print(f"  Total designs: {s['total_designs']}")
    print(f"  Unique sequences: {s['unique_sequences']}")
    print(f"  Sequence diversity: {s['sequence_diversity']:.1%}")
    print()

    # Critical metrics - Ligand and Catres RMSD
    print("-" * 80)
    print("CRITICAL: CONSTRAINT VALIDATION (should be ~0.0 if constraints working)")
    print("-" * 80)

    lig = results['ligand_rmsd']
    print(f"\nLIGAND RMSD (all heavy atoms):")
    if lig['n'] > 0:
        print(f"  Mean: {lig['mean']:.4f} A")
        print(f"  Std:  {lig['std']:.4f} A")
        print(f"  Min:  {lig['min']:.4f} A")
        print(f"  Max:  {lig['max']:.4f} A")
        print(f"  N:    {lig['n']}")
        if lig['mean'] > 0.5:
            print(f"  *** WARNING: Ligand RMSD >> 0 indicates CONSTRAINTS NOT WORKING! ***")
    else:
        print(f"  No data available")

    cat = results['catres_heavy_rmsd']
    print(f"\nCATRES HEAVY ATOM RMSD (constrained atoms):")
    if cat['n'] > 0:
        print(f"  Mean: {cat['mean']:.4f} A")
        print(f"  Std:  {cat['std']:.4f} A")
        print(f"  Min:  {cat['min']:.4f} A")
        print(f"  Max:  {cat['max']:.4f} A")
        print(f"  N:    {cat['n']}")
        if cat['mean'] > 0.5:
            print(f"  *** WARNING: Catres RMSD >> 0 indicates CONSTRAINTS NOT WORKING! ***")
    else:
        print(f"  No data available")

    # Global backbone RMSD
    print("-" * 80)
    print("BACKBONE RMSD (should be reasonably low)")
    print("-" * 80)

    ca2 = results['ca_rmsd_vs_step02']
    print(f"\nCA RMSD vs Step02 (relaxed input):")
    if ca2['n'] > 0:
        print(f"  Mean: {ca2['mean']:.4f} A, Max: {ca2['max']:.4f} A, N: {ca2['n']}")
    else:
        print(f"  No data available")

    ca1 = results['ca_rmsd_vs_step01']
    print(f"\nCA RMSD vs Step01 (original AF3):")
    if ca1['n'] > 0:
        print(f"  Mean: {ca1['mean']:.4f} A, Max: {ca1['max']:.4f} A, N: {ca1['n']}")
    else:
        print(f"  No data available")

    bb2 = results['backbone_rmsd_vs_step02']
    print(f"\nBackbone RMSD vs Step02:")
    if bb2['n'] > 0:
        print(f"  Mean: {bb2['mean']:.4f} A, Max: {bb2['max']:.4f} A, N: {bb2['n']}")
    else:
        print(f"  No data available")

    bb1 = results['backbone_rmsd_vs_step01']
    print(f"\nBackbone RMSD vs Step01:")
    if bb1['n'] > 0:
        print(f"  Mean: {bb1['mean']:.4f} A, Max: {bb1['max']:.4f} A, N: {bb1['n']}")
    else:
        print(f"  No data available")

    # Bond geometry
    print("-" * 80)
    print("BOND GEOMETRY DEVIATIONS")
    print("-" * 80)

    bl_mean = results['bond_length_unconstrained_mean']
    bl_max = results['bond_length_unconstrained_max']
    print(f"\nBond Length Deviations (unconstrained atoms):")
    if bl_mean['n'] > 0:
        print(f"  Mean of means: {bl_mean['mean']:.4f} A")
        print(f"  Mean of maxes: {bl_max['mean']:.4f} A")
        print(f"  Max deviation: {bl_max['max']:.4f} A")
        print(f"  N: {bl_mean['n']}")
    else:
        print(f"  No data available")

    ba_mean = results['bond_angle_unconstrained_mean']
    ba_max = results['bond_angle_unconstrained_max']
    print(f"\nBond Angle Deviations (unconstrained atoms):")
    if ba_mean['n'] > 0:
        print(f"  Mean of means: {ba_mean['mean']:.2f} deg")
        print(f"  Mean of maxes: {ba_max['mean']:.2f} deg")
        print(f"  Max deviation: {ba_max['max']:.2f} deg")
        print(f"  N: {ba_mean['n']}")
    else:
        print(f"  No data available")

    bl_cat = results['bond_length_catres_max']
    print(f"\nCatres Bond Length Deviations:")
    if bl_cat['n'] > 0:
        print(f"  Mean of maxes: {bl_cat['mean']:.4f} A")
        print(f"  Max deviation: {bl_cat['max']:.4f} A")
        print(f"  N: {bl_cat['n']}")
    else:
        print(f"  No data available")

    # Sequence diversity
    print("-" * 80)
    print("SEQUENCE METRICS")
    print("-" * 80)

    mut = results['mutations']
    print(f"\nMutations per design:")
    if mut['n'] > 0:
        print(f"  Mean: {mut['mean']:.2f}")
        print(f"  Std:  {mut['std']:.2f}")
        print(f"  Min:  {int(mut['min'])}")
        print(f"  Max:  {int(mut['max'])}")
        print(f"  N:    {mut['n']}")

    sid = results['sequence_identity']
    print(f"\nSequence identity vs wildtype:")
    if sid['n'] > 0:
        print(f"  Mean: {sid['mean']:.2%}")
        print(f"  Min:  {sid['min']:.2%}")
        print(f"  Max:  {sid['max']:.2%}")

    # Per-protocol breakdown
    print("-" * 80)
    print("PER-PROTOCOL BREAKDOWN")
    print("-" * 80)

    for protocol, pdata in sorted(results['by_protocol'].items()):
        print(f"\n{protocol.upper()}:")
        print(f"  Jobs: {pdata['n_jobs']}, Designs: {pdata['n_designs']}, Unique seqs: {pdata['unique_sequences']}")
        if pdata['mean_runtime']:
            print(f"  Mean runtime: {pdata['mean_runtime']:.1f}s")

        lig_p = pdata['ligand_rmsd']
        if lig_p['n'] > 0:
            print(f"  Ligand RMSD: mean={lig_p['mean']:.3f}A, max={lig_p['max']:.3f}A")

        mut_p = pdata['mutations']
        if mut_p['n'] > 0:
            print(f"  Mutations: mean={mut_p['mean']:.1f}, max={int(mut_p['max'])}")

    print()
    print("=" * 80)
    print("END OF REPORT")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze detailed metrics from sweep")
    parser.add_argument("json_file", help="Path to detailed_metrics_all.json")
    parser.add_argument("--output", "-o", help="Output JSON file for results")
    args = parser.parse_args()

    data = load_metrics(args.json_file)
    results = analyze_metrics(data)

    print_report(results)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
