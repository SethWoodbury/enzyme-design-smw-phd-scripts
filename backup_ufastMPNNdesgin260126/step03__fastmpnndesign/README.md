# Step03: FastMPNN Design with Rosetta Refinement

Iterative MPNN sequence design with Rosetta relaxation for enzyme active site optimization.

## Overview

This module takes step02 output (relaxed PDB + metrics JSON) and performs:
1. Residue classification (catalytic, conserved motif, primary/secondary sphere)
2. Iterative MPNN sequence design with configurable protocols
3. Rosetta relaxation (cartesian and torsional) with coordinate constraints
4. Favorable interaction detection and conservation ("H-bond keeper")
5. Comprehensive metrics tracking (vs both step01 and step02)

## Quick Start

### Recommended Default Protocol

Based on hyperparameter sweep analysis, the recommended default protocol is:

```bash
python fastmpnn_design.py \
    --step02_json step02_outputs/relaxed_metrics.json \
    --params params/LIG.params \
    --output_dir output/ \
    --protocol "mpnn:T0.2:N8 -> torsional_relax:R1S2 -> mpnn:T0.1:N16 -> torsional_relax:R2S3"
```

This produces 4-6 unique designs with 4-6 mutations each in approximately 5-8 minutes.

### Basic Usage with Presets

```bash
python fastmpnn_design.py \
    --step02_json step02_outputs/relaxed_metrics.json \
    --params params/LIG.params \
    --output_dir output/ \
    --preset balanced
```

## Protocol Selection Guide

| Goal | Recommended Protocol | Expected Runtime |
|------|---------------------|------------------|
| Quick prototyping | `mpnn:T0.3:N8 -> torsional_relax:R1S2` | 2-3 min |
| Standard design | `mpnn:T0.2:N8 -> torsional_relax:R1S2 -> mpnn:T0.1:N16 -> torsional_relax:R2S3` | 5-8 min |
| Maximum diversity | `mpnn:T0.3:N32 -> torsional_relax:R1S2 -> mpnn:T0.2:N32 -> torsional_relax:R2S3` | 10-15 min |
| Geometry critical | `cart_relax:R2S3 -> mpnn:T0.1:N16 -> torsional_relax:R2S3` | 15-25 min |
| Publication quality | `cart_relax:R3S4 -> mpnn:T0.15:N32 -> torsional_relax:R2S3 -> mpnn:T0.1:N16 -> torsional_relax:R3S4` | 30-45 min |

## Key Findings from Hyperparameter Sweep

1. **Torsional relaxation is usually sufficient** - Cartesian adds runtime with marginal benefit for most designs
2. **Temperature annealing works well** - Start high (0.2-0.3), end low (0.1)
3. **More designs per round is efficient** - N=16-32 gives good diversity without excessive runtime
4. **Simple protocols often match complex ones** - Progressive temperature protocol (106s) produced similar quality to 20-minute protocols
5. **Optimal mutation range: 4-6 per design** across all successful protocols

## Full CLI Options

```bash
python fastmpnn_design.py \
    --step02_json <path>            # Step02 metrics JSON (required)
    --params <file> [file ...]      # Ligand .params files (required)
    --output_dir <path>             # Output directory (required)

    # Protocol Selection
    --preset {fast,balanced,thorough,aggressive,design_only,geometry_only}
    --protocol <string>             # Custom protocol (overrides preset)

    # Design Scope
    --catres_subset "1,2,5"         # Override catres subset
    --design_secondary_sphere       # Also design secondary sphere
    --design_gly_pro               # Allow GLY/PRO redesign
    --layer_cuts 6.0 8.0 10.0 12.0 # Distance cuts for spheres

    # MPNN Settings
    --mpnn_temperature 0.1          # Sampling temperature
    --mpnn_num_designs 8            # Designs per round
    --mpnn_batch_size 1             # Batch size (1 = max diversity)
    --mpnn_omit_aa CM               # Amino acids to omit

    # Rosetta Settings
    --coord_cst_weight 750.0        # Coordinate constraint weight
    --coord_cst_stdev 0.01          # Coordinate constraint stdev
    --cart_bonded_weight 2.0        # Cartesian bonded weight
    --global_cst_weight 0.0         # Global coordinate constraint weight
    --global_cst_stdev 0.5          # Global coordinate constraint stdev
    --scorefunction_cart ref2015_cart
    --scorefunction_torsional beta_jan25

    # Convergence
    --bond_length_tolerance 0.05    # Angstroms
    --bond_angle_tolerance 10.0     # Degrees

    # References
    --step01_pdb <path>             # Original step01 PDB for CA RMSD

    # Interaction Conservation
    --conserve_favorable_interactions
    --no_conserve_interactions      # Disable conservation
    --conservation_probability 0.5

    # Output
    --num_final_designs 10

    # Runtime
    --max_runtime 7200              # Seconds
    --debug
    --test
```

## Custom Protocol Syntax

Custom protocols are specified as a sequence of steps separated by `->`:

```
step1 -> step2 -> step3 -> ...
```

### Available Step Types

| Step Type | Syntax | Example |
|-----------|--------|---------|
| MPNN Design | `mpnn:T<temp>:N<num_designs>` | `mpnn:T0.2:N16` |
| Torsional Relax | `torsional_relax:R<repeats>S<stages>` | `torsional_relax:R2S3` |
| Cartesian Relax | `cart_relax:R<repeats>S<stages>` | `cart_relax:R2S3` |
| Select Best | `select_best:N<count>` | `select_best:N1` |

### Example Custom Protocols

**Progressive Temperature Reduction (FASTEST - ~2 min):**
```
mpnn:T0.2:N1 -> torsional_relax:R1S2 -> mpnn:T0.15:N1 -> torsional_relax:R1S2 -> mpnn:T0.1:N10 -> torsional_relax:R2S3
```

**Single-Shot High Diversity (SIMPLE - ~4 min):**
```
mpnn:T0.3:N32 -> torsional_relax:R1S2
```

**Geometry-First (THOROUGH - ~15 min):**
```
cart_relax:R2S3 -> mpnn:T0.1:N20 -> torsional_relax:R2S3
```

**Multi-Round Breadth (COMPREHENSIVE - ~20 min):**
```
mpnn:T0.3:N16 -> torsional_relax:R1S2 -> mpnn:T0.2:N16 -> torsional_relax:R1S2 -> mpnn:T0.1:N16 -> torsional_relax:R2S3
```

## Recommended Parameter Defaults

| Parameter | Recommended Default | Range to Explore |
|-----------|---------------------|------------------|
| Temperature (initial) | 0.2 | 0.15-0.3 |
| Temperature (final) | 0.1 | 0.05-0.15 |
| Designs per round | 8-16 | 4-64 |
| Torsional repeats | 1-2 | 1-3 |
| Torsional stages | 2-3 | 2-4 |
| Cartesian repeats | 2 | 1-3 |
| Cartesian stages | 3 | 2-4 |
| Cart bonded weight | 2.0 | 1.0-4.0 |
| Coord constraint weight | 750.0 | 500-1000 |
| Coord constraint stdev | 0.01 | 0.005-0.05 |

### Temperature Effects

| Temperature | Effect on Mutations | Effect on Diversity | Recommended Use |
|-------------|---------------------|---------------------|-----------------|
| 0.05 | 3-4 mutations | Very low | Conservative refinement |
| 0.1 | 4-5 mutations | Low | Standard design |
| 0.15 | 4-5 mutations | Medium | Balanced approach |
| 0.2 | 5-6 mutations | Medium-High | Exploration |
| 0.3 | 5-7 mutations | High | Maximum diversity |
| 0.4+ | 6-8 mutations | Very high | May destabilize |

### Relaxation Configuration

| Configuration | Runtime | Quality | When to Use |
|--------------|---------|---------|-------------|
| R1S2 (torsional) | ~30-60s | Good | Most cases |
| R1S3 (torsional) | ~45-90s | Better | Final refinement |
| R2S3 (torsional) | ~90-180s | Best | Critical designs |
| R2S3 (cartesian) | ~10-20min | Excellent | Geometry critical |
| R3S4 (cartesian) | ~20-40min | Premium | Publication quality |

## Protocol Presets

### fast
Light geometry fix + single design round. Good for quick testing.
- 2 MPNN designs + light cart relax
- 4 MPNN designs + torsional relax

### balanced (default)
Geometry optimization first, then iterative design.
- Cart relax until converged
- Select best 1 structure
- 2 rounds of MPNN + torsional relax

### thorough
Extensive geometry optimization + multi-round design + secondary sphere.
- Extensive cart relax with convergence check
- Multiple MPNN rounds at decreasing temperatures
- Secondary sphere design ("2nd shell")

### aggressive
More designs, less strict optimization.
- Designs both primary and secondary spheres
- Higher temperatures for diversity
- Lighter optimization

### design_only
Single round design without geometry optimization.

### geometry_only
Cartesian relaxation only, no sequence design.

## Sphere Classification

Residues are classified based on distance from ligand:

| Sphere | Distance | Design | Behavior |
|--------|----------|--------|----------|
| Primary | 0-6 A | Yes | Full redesign |
| Secondary | 6-8 A | Optional | CB orientation check |
| Repack Primary | 8-10 A | No | Sidechain repack only |
| Repack Secondary | 10-12 A | No | Light repack |
| Distant | >12 A | No | Fixed |

### CB Orientation Check
For secondary sphere residues, we check if the CB atom points toward the ligand (CB distance < CA distance). This identifies residues whose sidechains interact with the active site.

### Automatic Secondary Sphere Expansion
If all primary sphere residues are catalytic (fixed), the design scope automatically expands to include secondary sphere residues to ensure there are designable positions.

## Favorable Interaction Conservation

The module can conserve beneficial mutations that form H-bonds or pi-stacking interactions with catalytic residues (similar to "H-bond keeper" in traditional design):

1. After MPNN design, analyze mutations for interactions with catres
2. Mutations forming H-bonds/pi-stacks are probabilistically fixed
3. MPNN biases are applied to favor beneficial amino acids

Enable with `--conserve_favorable_interactions` (default on).

## Output Structure

```
output_dir/
├── fastmpnn_design_results.json    # Complete results
├── design_00.pdb                   # Best design
├── design_01.pdb                   # Second best
├── ...
├── mpnn_*/                         # MPNN intermediate outputs
│   ├── seqs/
│   ├── backbones/
│   └── packed/
└── *.cst.json                      # Constraint files
```

## Output JSON Structure

```json
{
  "metadata": {
    "step02_json": "/path/to/step02_metrics.json",
    "step02_pdb": "/path/to/step02_relaxed.pdb",
    "step01_pdb": "/path/to/step01_aligned.pdb",
    "protocol": "balanced",
    "runtime_seconds": 1234.5
  },
  "residue_classification": {
    "num_catalytic": 17,
    "num_conserved_motif": 2,
    "num_primary": 45,
    "num_secondary": 33,
    "fixed_residues": ["A13", "A15", ...],
    "design_residues": ["A10", "A11", ...]
  },
  "output_designs": [
    {
      "rank": 0,
      "pdb_path": "/path/to/design_00.pdb",
      "sequence": "MVKLTI...",
      "metrics": {
        "sequence_metrics": {
          "sequence_identity_vs_step02": 0.85,
          "num_mutations": 12,
          "mutations": ["A10V", "L15F", ...]
        },
        "rmsd": {
          "ligand": 0.001,
          "constrained_atoms": {"aggregate": 0.002},
          "global_ca_vs_step01": 0.89,
          "global_ca_vs_step02": 0.45
        },
        "bond_geometry": {
          "bond_length_geometry": {
            "all": {...},
            "unconstrained_only": {"max": 0.04, "mean": 0.02}
          }
        },
        "convergence": {
          "bond_length_converged": true,
          "bond_angle_converged": true
        }
      }
    }
  ]
}
```

## Module Architecture

```
step03__fastmpnndesign/
├── __init__.py
├── fastmpnn_design.py      # Main orchestrator/CLI
├── mpnn_runner.py          # MPNN subprocess execution
├── rosetta_relax.py        # Rosetta relaxation (standalone)
├── residue_classifier.py   # ResidueInfo dataclass + sphere classification
├── pdb_restoration.py      # Restore H atoms, REMARK 666, tautomers
├── protocol_parser.py      # Parse protocol strings into steps
├── interaction_analyzer.py # Detect favorable interactions
├── metrics.py              # Metrics calculation
├── README.md
├── test/
│   ├── run_test.sh
│   ├── step02_outputs/     # Symlink to step02 outputs
│   ├── params/             # Symlink to params
│   └── output_dir/
└── hyper_param_sweep/
    ├── generate_sweep.py         # Initial sweep generator
    ├── generate_large_sweep.py   # Large-scale sweep (~400+ jobs)
    ├── analyze_sweep.py          # Analyze results
    ├── submit_array.sh           # SLURM array job submission
    ├── sbatch_template.sh        # SLURM template
    ├── monitor_jobs.sh           # Job monitoring
    ├── ANALYSIS_REPORT.md        # Detailed analysis of sweep results
    ├── cmds/
    ├── logs/
    └── outputs/
```

## Key Design Decisions

1. **Phased Protocol**: Geometry first (cart relax until converged) -> Design -> Optimize (torsional)
2. **Scorefunction Selection**: ref2015_cart for cartesian, beta_jan25 for torsional
3. **Sphere Selection**: Distance-based with CB orientation check
4. **MPNN Diversity**: Lower batch_size (1) + higher number_of_batches = more diversity
5. **Favorable Interaction Conservation**: H-bond keeper style probabilistic fixing
6. **Duplicate Handling**: Remove by sequence, keep structure with best geometry
7. **RMSD Tracking**: vs both step01 (original) AND step02 (relaxed reference)
8. **Constraint Verification**: Ligand and catres atoms should have ~0.0 RMSD
9. **Automatic Scope Expansion**: If primary sphere has no designable residues, automatically include secondary

## Dependencies

- PyRosetta (for Rosetta relaxation, run via apptainer container)
- MPNN (ligand_mpnn via fused_mpnn)
- numpy
- module_utils (constants, pdb_utils, sequence_utils, interaction_utils)

## Testing

```bash
cd test/
./run_test.sh          # Run with fast preset
./run_test.sh balanced # Run with balanced preset
```

## Hyperparameter Sweep

### Quick Sweep (52 jobs)
```bash
cd hyper_param_sweep/
python generate_sweep.py                         # Generate commands
./submit_array.sh --max-parallel 20              # Submit to SLURM
python analyze_sweep.py                          # Analyze results
```

### Large-Scale Sweep (~400+ jobs)
```bash
cd hyper_param_sweep/
python generate_large_sweep.py --output cmds/large_sweep_commands.txt
./submit_array.sh --commands cmds/large_sweep_commands.txt --max-parallel 50
```

### Sweep Analysis

See `hyper_param_sweep/ANALYSIS_REPORT.md` for detailed analysis including:
- Protocol performance comparison
- Parameter effects tables
- Recommended defaults
- Hypotheses for optimization

## Related Modules

- **step01__catres_alignment**: Aligns catalytic residue coordinates
- **step02__constrained_cart_relax**: Optimizes bond geometry with constraints
- **module_utils**: Shared utilities (constants, pdb_utils, sequence_utils, interaction_utils)
