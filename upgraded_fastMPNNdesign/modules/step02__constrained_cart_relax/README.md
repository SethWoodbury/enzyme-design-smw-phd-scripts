# Step 02: Constrained Cartesian Relaxation

Idealize bond geometry while preserving catalytic residue and ligand positions using
constrained Cartesian FastRelax with adaptive protocols.

## Overview

This script takes the output from step01 (catres_alignment) and performs adaptive
Cartesian relaxation to fix distorted bond lengths and angles while keeping
catalytic residue atoms fixed in absolute space using tight coordinate constraints.

## Key Features

- **Adaptive Protocol**: Automatically runs additional rounds until geometry converges or plateaus
- **Coordinate Constraints**: Tight harmonic constraints (default: 0.01A stdev, weight 750) on catres and ligand atoms
- **Bond Geometry Minimization**: Optional `minimize_bond_angles`/`minimize_bond_lengths` for better idealization (disabled by default)
- **Smart Mobile Region**: Includes SS elements, sequence neighbors, and spatial neighborhood
- **Automatic Region Expansion**: Expands mobile region around stuck residues when geometry not improving
- **Comprehensive Metrics**: Detailed geometry quality reporting with severity breakdown and catres-specific status
- **Full CLI Control**: All parameters controllable via command line for easy parameter sweeping

## Constrained vs Unconstrained Metrics

This script distinguishes between constrained atoms (from the quantum chemistry theozyme) and unconstrained atoms when calculating geometry metrics. This distinction is critical for understanding which geometry issues can actually be fixed by relaxation.

### Background

- **Constrained atoms** come from the quantum chemistry theozyme and represent the ground truth catalytic geometry. These atoms are held fixed in space with tight coordinate constraints.
- **Bonds between two constrained atoms cannot change** - they are fixed in space and represent the theozyme geometry exactly as designed.
- Only bonds/angles involving at least one unconstrained atom can be optimized during relaxation.

### Metric Categories

The script separates geometry metrics into two categories:

| Category | Description | Use Case |
|----------|-------------|----------|
| `all` | All bonds/angles in the structure | Complete geometry picture |
| `unconstrained_only` | Only bonds/angles where at least one atom is NOT constrained | What relaxation can actually optimize |

### How It Works

1. **Bond classification**: A bond is "unconstrained" if at least one of its two atoms is not in the constrained atom set
2. **Angle classification**: An angle is "unconstrained" if at least one of its three atoms is not in the constrained atom set
3. **Convergence checking**: Uses `unconstrained_only` metrics since those are what we can actually optimize
4. **Catres pass/fail**: Now evaluated using unconstrained metrics only

### Why This Matters

When the theozyme is placed into a protein scaffold, the bonds between constrained atoms are fixed by design. If these bonds have deviations from ideal geometry, that's a property of the theozyme itself - not something relaxation can fix. By separating metrics, you can:

- See the true geometry quality of atoms that relaxation can influence
- Avoid false "failures" from theozyme geometry that cannot be changed
- Focus optimization efforts on the parts of the structure that can actually improve

## Installation

Requires PyRosetta. Run via Apptainer:

```bash
/net/software/containers/universal.sif python constrained_cart_relax.py --step01_json ... --params ...
```

## Quick Start

### Basic Usage

```bash
python constrained_cart_relax.py \
    --step01_json step01_output.json \
    --params ligand.params \
    --output relaxed.pdb
```

### Using Presets

```bash
# Fast cleanup (quick, lower quality)
python constrained_cart_relax.py --step01_json ... --params ... --preset fast

# Balanced (default settings)
python constrained_cart_relax.py --step01_json ... --params ... --preset balanced

# Thorough idealization (slower, higher quality)
python constrained_cart_relax.py --step01_json ... --params ... --preset thorough

# Aggressive (for difficult cases with persistent geometry issues)
python constrained_cart_relax.py --step01_json ... --params ... --preset aggressive
```

## Rosetta Concepts Explained

### Cartesian vs Torsional Minimization

**Torsional Minimization (Internal Coordinates):**
- Moves atoms by changing phi/psi/chi dihedral angles
- Bond lengths and angles are FIXED - cannot change
- Fast but cannot fix distorted geometry

**Cartesian Minimization:**
- Moves atoms directly in 3D Cartesian space (x, y, z)
- Bond lengths and angles CAN change
- Required for fixing geometry distortions from step01 alignment
- Uses `cart_bonded` score term to penalize deviations from ideal

### The cart_bonded Score Term

The `cart_bonded` term penalizes deviations from ideal bond geometry:

```
E_cart_bonded = Σ k_bond * (d - d_ideal)² + Σ k_angle * (θ - θ_ideal)²
```

Where:
- `d` = actual bond length, `d_ideal` = ideal bond length
- `θ` = actual bond angle, `θ_ideal` = ideal bond angle
- `k_bond`, `k_angle` = spring constants

**Higher weight** = stronger drive toward ideal geometry, but can conflict with
keeping catres atoms in place. The adaptive protocol balances these by:

1. Starting with moderate weight (default: 3.0)
2. Increasing weight by 1.5x per round if geometry not converging
3. Capping at maximum weight (default: 3.0)
4. Using very tight coordinate constraints (750.0 weight, 0.01A stdev) on catres

### Bond Geometry Minimization Options

Standard Cartesian minimization uses the cart_bonded term. The FastRelax
`minimize_bond_angles(True)` and `minimize_bond_lengths(True)` options enable additional explicit minimization:

- **minimize_bond_angles**: Explicit 3-body angle minimization (A-B-C angles)
- **minimize_bond_lengths**: Explicit 2-body bond minimization (A-B distances)

These provide finer control over geometry idealization and are **disabled by default**.
To enable: `--enable_bond_geometry_min`

### Coordinate Constraints

Coordinate constraints fix specific atoms in absolute space using a harmonic potential:

```
E_coord = weight * 0.5 * ((distance - 0) / stdev)²
```

With default settings (weight=750, stdev=0.01A):
- 0.01A displacement: penalty = 750 * 0.5 * 1² = 375
- 0.1A displacement: penalty = 750 * 0.5 * 100² = 3,750,000

This makes it extremely unfavorable for constrained atoms to move.

### FastRelax Protocol

FastRelax performs cycles of alternating repacking and minimization with fa_rep ramping:

```
For each ramping stage (1 to N):
  1. scale:fa_rep {0.02...1.0}  # Ramp from low to full
  2. repack                      # Optimize rotamers
  3. min {tolerance}             # Minimize coordinates
Accept best structure
```

The fa_rep ramping allows initial clash resolution before full van der Waals repulsion.

**Internal Structure:**
- **Repeats (M)**: Number of times to repeat the full ramping script
- **Stages (N)**: Number of fa_rep ramping steps per repeat
- **Total internal rounds** = M × N (default: 3 × 3 = 9)

### Adaptive Protocol

The adaptive protocol runs FastRelax rounds until convergence:

```
Round 1 → Calculate geometry metrics → Check convergence
           ↓
         If not converged AND bonds > tolerance:
           → Increase cart_bonded by scale_factor (1.5x)
           → If stuck: expand mobile region

Round 2+ → Repeat

Stop when:
  • Both bonds AND angles < tolerance
  • Plateau detected (improvements < threshold)
  • Time limit exceeded (80% of max_runtime)
  • Max rounds reached (default: 10)
```

## Parameters Reference

### Core Protocol Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--fastrelax_repeats` | 3 | Number of script repeats (M) |
| `--fastrelax_ramp_stages` | 3 | Number of ramping stages (N). Total = M × N |
| `--cart_bonded_weight` | 3.0 | Initial cart_bonded weight |
| `--coord_cst_weight` | 750.0 | Coordinate constraint weight |
| `--coord_cst_stdev` | 0.01 | Coordinate constraint stdev (Angstroms) |
| `--mobile_radius` | 10.0 | Radius for mobile region around ligand/catres (Angstroms) |
| `--sequence_neighbor_buffer` | 5 | Include +/- N residues in sequence around catres |

### Scorefunction Selection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--scorefunction` | ref2015_cart | Scorefunction to use. Options: `ref2015_cart`, `beta_nov16_cart`. |

### Convergence Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--bond_length_tolerance` | 0.05 | Stop when max bond deviation < this (Angstroms) |
| `--bond_angle_tolerance` | 10.0 | Stop when max angle deviation < this (degrees) |
| `--max_adaptive_rounds` | 10 | Maximum adaptive rounds before stopping |
| `--max_runtime` | 3600 | Maximum runtime in seconds |

### Cart_bonded Scaling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cart_bonded_scale_factor` | 1.5 | Factor to multiply cart_bonded by when not converging |
| `--cart_bonded_max` | 3.0 | Maximum cart_bonded weight cap |

### Bond Geometry Minimization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enable_bond_geometry_min` | False | Enable `minimize_bond_angles` + `minimize_bond_lengths` in FastRelax |
| `--disable_bond_geometry_min` | - | Disable explicit bond geometry minimization |

### Repacking Scope

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--global_repack` | False | Enable global repacking (all protein residues) |
| `--repack_shell` | None | Repack within N Angstroms of mobile region |

### Mobile Region Expansion

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--auto_expand_mobile` | True | Automatically expand mobile region if stuck |
| `--no_auto_expand_mobile` | - | Disable automatic mobile region expansion |
| `--expansion_radius` | 5.0 | Radius for mobile region expansion (Angstroms) |
| `--max_expansions` | 3 | Maximum number of mobile region expansions |

### Catres-Specific Convergence

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--catres_bond_tolerance` | 0.05 | Catres-specific bond tolerance (Angstroms) |
| `--catres_angle_tolerance` | 10.0 | Catres-specific angle tolerance (degrees) |
| `--require_catres_converged` | True | Warn if catres geometry bad |
| `--no_require_catres_converged` | - | Disable catres-specific convergence check |

### Offender Thresholds (Severity Categories)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--severe_bond_threshold` | 0.2 | Bond deviation above which is 'severe' (Angstroms) |
| `--moderate_bond_threshold` | 0.1 | Bond deviation for 'moderate' category (Angstroms) |
| `--severe_angle_threshold` | 15.0 | Angle deviation above which is 'severe' (degrees) |
| `--moderate_angle_threshold` | 10.0 | Angle deviation for 'moderate' category (degrees) |

### Loop Modeling (Experimental)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enable_loop_rebuild` | False | Enable experimental loop rebuilding capability |
| `--loop_rebuild_threshold` | None | Bond deviation (A) triggering loop rebuild |

### Execution Flags

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--fast` | False | Shorthand for 1×3=3 rounds (similar to `--preset fast` but doesn't modify other settings) |
| `--skip_torsional_relax` | False | Skip torsional FastRelax at end |
| `--skip_minimize` | False | Skip final MinMover |
| `--debug` | False | Enable verbose debug output |

### Presets

| Preset | Repeats×Stages | Key Settings |
|--------|----------------|--------------|
| `fast` | 1×3=3 | `enable_bond_geometry_min=False`, `auto_expand_mobile=False`, `max_adaptive_rounds=3` |
| `balanced` | 3×3=9 | `enable_bond_geometry_min=False`, `max_adaptive_rounds=5`, tuned weights |
| `thorough` | 5×5=25 | `bond_length_tolerance=0.03`, `bond_angle_tolerance=5.0`, `max_adaptive_rounds=15` |
| `aggressive` | 3×5=15 | `cart_bonded_weight=3.5`, `cart_bonded_max=5.0`, `cart_bonded_scale_factor=2.0`, `max_expansions=5` |

## Output Files

### 1. Relaxed PDB (`*_relaxed.pdb`)

Idealized structure with original REMARK 666 lines preserved.

### 2. Metrics JSON (`*_relaxed_metrics.json`)

Comprehensive validation metrics including:

```json
{
  "metadata": {
    "input_pdb": "...",
    "step01_json": "...",
    "num_residues": 287,
    "num_catres": 17,
    "coord_cst_weight": 750.0,
    "cart_bonded_weight": 3.0
  },
  "scores": {
    "total_before": -450.23,
    "total_after": -512.45,
    "delta": -62.22,
    "cart_bonded": 12.34,
    "fa_rep": 45.67,
    "coordinate_constraint": 0.0012
  },
  "bond_length_geometry": {
    "before": {"mean_deviation": 0.034, "max_deviation": 0.156, ...},
    "after": {
      "all": {"mean_deviation": 0.012, "max_deviation": 0.048, ...},
      "unconstrained_only": {"mean_deviation": 0.010, "max_deviation": 0.035, ...}
    }
  },
  "bond_angle_geometry": {
    "before": {"mean_deviation": 4.5, "max_deviation": 28.3, ...},
    "after": {
      "all": {"mean_deviation": 2.1, "max_deviation": 9.8, ...},
      "unconstrained_only": {"mean_deviation": 1.8, "max_deviation": 7.2, ...}
    }
  },
  "geometry_offenders": {
    "all": {
      "bond_length": {
        "total_offenders": 5,
        "severe": {"count": 1, "residues": ["LIG1"]},
        "moderate": {"count": 2, "residues": ["A176", "LIG1"]},
        "minor": {"count": 2, "residues": ["A30", "A45"]},
        "per_residue_summary": {...}
      },
      "bond_angle": {...}
    },
    "unconstrained_only": {
      "bond_length": {
        "total_offenders": 3,
        "severe": {"count": 0, "residues": []},
        "moderate": {"count": 1, "residues": ["A176"]},
        "minor": {"count": 2, "residues": ["A30", "A45"]},
        "per_residue_summary": {...}
      },
      "bond_angle": {...}
    },
    "catres_offender_summary": {
      "num_catres_with_bad_geometry": 0,
      "total_catres": 17,
      "catres_offenders": []
    }
  },
  "catres_geometry_status": {
    "all_converged": true,
    "num_passing": 17,
    "num_failing": 0,
    "catres_bond_tolerance": 0.05,
    "catres_angle_tolerance": 10.0,
    "passing_catres": [...],
    "failing_catres": [],
    "note": "Evaluated using unconstrained metrics only"
  },
  "rmsd": {
    "ligand": 0.0001,
    "constrained_atoms": {"aggregate_rmsd": 0.0002, ...},
    "global_ca": 0.85
  },
  "quality": {
    "chain_breaks_before": 0,
    "chain_breaks_after": 0,
    "clashes_before": 36,
    "clashes_after": 5
  },
  "convergence": {
    "num_rounds": 4,
    "fastrelax_repeats": 3,
    "fastrelax_ramp_stages": 3,
    "total_internal_rounds_per_cycle": 9,
    "enable_bond_geometry_min": false,
    "num_mobile_expansions": 0,
    "convergence_metric": "unconstrained_only",
    "history": [...]
  }
}
```

## Troubleshooting

### "Geometry not converging"

1. Try `--preset aggressive`
2. Increase `--cart_bonded_max` to 4.0 or 5.0
3. Enable `--enable_bond_geometry_min` if needed (default is off)
4. Increase `--max_adaptive_rounds` to 15
5. Check if catres placement from step01 is reasonable (large distortions may indicate input issues)

### "Catres atoms moving too much"

1. Increase `--coord_cst_weight` (try 1000.0 or 1500.0)
2. Decrease `--coord_cst_stdev` (try 0.005)
3. Check RMSD values in metrics JSON - constrained_atoms should be < 0.01A

### "Runtime too long"

1. Use `--preset fast`
2. Reduce `--max_adaptive_rounds`
3. Enable `--skip_torsional_relax` and `--skip_minimize`
4. Reduce `--fastrelax_repeats` or `--fastrelax_ramp_stages`

### "Persistent geometry issues on specific residues"

1. Enable auto-expansion: ensure `--auto_expand_mobile` is on
2. Increase `--max_expansions` to 5
3. Increase `--expansion_radius` to 8.0
4. Try `--global_repack` for better sidechain packing

### "catres_geometry_status shows failing catres"

This means catalytic residues have geometry outside the catres-specific tolerances.

1. Check which catres are failing in the JSON
2. If it's constrained atoms: they may be in physically impossible positions
3. If it's non-constrained atoms: try more aggressive settings
4. Consider relaxing `--catres_bond_tolerance` or `--catres_angle_tolerance`

### "Geometry offenders include constrained bonds/angles (theozyme geometry)"

If you see geometry issues on bonds or angles between constrained atoms (e.g., within the ligand or between ligand and catres atoms that are both constrained), these are from the **theozyme** and **cannot be fixed by relaxation**.

**Why this happens:**
- Constrained atoms are held fixed in space with tight coordinate constraints
- Bonds between two constrained atoms cannot change - their geometry is determined by the quantum chemistry theozyme design
- If the theozyme has non-ideal bond lengths or angles, those will persist

**What to do:**
1. Check the `geometry_offenders.all` vs `geometry_offenders.unconstrained_only` sections in the JSON
2. If offenders appear in `all` but not in `unconstrained_only`, they are theozyme geometry issues
3. Theozyme geometry issues must be addressed upstream (in the theozyme design itself, not in relaxation)
4. Focus on the `unconstrained_only` metrics - those are what relaxation can actually optimize
5. The catres pass/fail status uses unconstrained metrics, so theozyme geometry won't cause false failures

## Algorithm Flow

```
1. Load step01 JSON and input PDB
2. Identify ligand from REMARK 666
3. Build atoms_to_constrain (ligand ALL_HEAVY, catres specific atoms)
4. Load pose and add coordinate constraints
5. Define mobile region:
   - Ligand neighborhood (mobile_radius)
   - Catres neighborhood (mobile_radius)
   - Sequence neighbors (+/- 5 residues)
   - Complete SS elements containing catres
6. Build MoveMap (mobile residues can move, ligands frozen)
7. Build TaskFactory (repacking scope based on settings)

8. ADAPTIVE LOOP:
   a. Run FastRelax round (M repeats × N stages)
   b. Calculate bond length/angle metrics
   c. Check convergence (both < tolerance?)
   d. Check plateau (improvements < threshold?)
   e. If stuck and auto_expand: expand mobile region around offenders
   f. If not converging: increase cart_bonded weight
   g. Repeat until converged, plateaued, or max rounds

9. Optional: Loop rebuild for severely distorted regions (experimental)
10. Optional: Torsional FastRelax polish
11. Optional: Final Cartesian minimize
12. Calculate comprehensive metrics (offenders, catres status)
13. Write outputs (PDB + JSON)
```

## Examples

### Parameter Sweep - Stages

```bash
for stages in 3 4 5 6; do
    python constrained_cart_relax.py \
        --step01_json input.json \
        --params ligand.params \
        --fastrelax_ramp_stages $stages \
        --output_dir sweep_stages_${stages}/
done
```

### Parameter Sweep - Cart_bonded Weight

```bash
for weight in 0.5 0.8 1.0 1.5; do
    python constrained_cart_relax.py \
        --step01_json input.json \
        --params ligand.params \
        --cart_bonded_weight $weight \
        --output_dir sweep_cart_${weight}/
done
```

### Parameter Sweep - Presets

```bash
for preset in fast balanced thorough aggressive; do
    python constrained_cart_relax.py \
        --step01_json input.json \
        --params ligand.params \
        --preset $preset \
        --output_dir sweep_${preset}/
done
```

### High-Quality Refinement

```bash
python constrained_cart_relax.py \
    --step01_json input.json \
    --params ligand.params \
    --fastrelax_repeats 5 \
    --fastrelax_ramp_stages 5 \
    --enable_bond_geometry_min \
    --bond_length_tolerance 0.03 \
    --bond_angle_tolerance 5.0 \
    --cart_bonded_max 4.0 \
    --auto_expand_mobile \
    --max_expansions 5 \
    --output refined.pdb
```

### Testing with Different Repacking Scopes

```bash
# Local only (default)
python constrained_cart_relax.py --step01_json input.json --params ligand.params \
    --output_dir test_local/

# With repack shell
python constrained_cart_relax.py --step01_json input.json --params ligand.params \
    --repack_shell 15.0 --output_dir test_shell/

# Global repacking
python constrained_cart_relax.py --step01_json input.json --params ligand.params \
    --global_repack --output_dir test_global/
```

## Glossary

| Term | Definition |
|------|------------|
| **cart_bonded** | Score term penalizing deviations from ideal bond geometry in Cartesian space |
| **fa_rep** | van der Waals repulsive term - penalizes atomic clashes |
| **FastRelax** | Rosetta protocol combining repacking and minimization with fa_rep ramping |
| **MoveMap** | Defines which degrees of freedom (backbone, sidechain) can move |
| **TaskFactory** | Controls which residues can repack (change rotamers) |
| **Coordinate Constraint** | Harmonic restraint fixing an atom at a specific 3D position |
| **Ramping** | Gradually increasing a score term weight across protocol stages |
| **Adaptive** | Automatically adjusting parameters based on progress/metrics |
| **Mobile Region** | Residues allowed to move during relaxation |
| **Catres** | Catalytic residues - identified from REMARK 666 |

## Hyperparameter Sweep Results

Extensive parameter sweeps were conducted to optimize default settings. Key findings:

- **cart_bonded_weight=3.0** provides optimal bond geometry (default updated from 0.8)
- **ref2015_cart** outperforms beta_nov16_cart for bond geometry
- **1×3 to 3×5 cycles** perform similarly when cart_bonded is high (default is 3×3)
- **bond_geometry_min=OFF** slightly outperforms ON

See `hyper_param_sweep/ANALYSIS.md` for detailed results, statistical analysis, and methodology.

## References

- [Rosetta FastRelax](https://docs.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Movers/movers_pages/FastRelaxMover)
- [Cartesian Minimization](https://www.rosettacommons.org/docs/latest/application_documentation/analysis/cartesian-analysis)
- [Loop Modeling Tutorial](https://docs.rosettacommons.org/demos/latest/tutorials/loop_modeling/loop_modeling)
