# Enzyme Design Pipeline

A unified CLI runner for the three-step enzyme design workflow: catalytic residue alignment, constrained geometry relaxation, and FastMPNN sequence design with Rosetta refinement.

## Overview

This pipeline orchestrates three sequential steps to design enzyme active sites:

1. **Step 1: Catalytic Residue Alignment** (`align_catres.py`)
   - Aligns catalytic residue coordinates from a reference theozyme to an input structure
   - Analyzes geometric interactions (H-bonds, metal coordination, pi-stacking)
   - Generates constraint recommendations for downstream steps

2. **Step 2: Constrained Cartesian Relaxation** (`constrained_cart_relax.py`)
   - Idealizes bond lengths and angles while keeping catalytic atoms fixed
   - Uses adaptive relaxation with coordinate constraints
   - Produces geometry-optimized structures

3. **Step 3: FastMPNN Design** (`fastmpnn_design.py`)
   - Iterative sequence design using ProteinMPNN
   - Integrates Rosetta relaxation for structure refinement
   - Conserves favorable interactions with catalytic residues

## Installation

### Prerequisites

- Python 3.8+
- Apptainer or Singularity container runtime
- Container image:
  - `universal.sif` - For all steps (PyRosetta 2026.03, ProteinMPNN, numpy)

### Setup

The pipeline is ready to use. All steps use the unified container by default:

```bash
# Default container location (can be overridden via CLI)
/net/software/containers/universal.sif
```

The `universal.sif` container includes PyRosetta 2026.03 with multi-threading support, serialization, and the `beta_jan25` scorefunction (improved LJ parameters).

## Quick Start

### Basic Usage

```bash
python run_pipeline.py \
    --input_pdb my_structure.pdb \
    --ref_pdb theozyme.pdb \
    --params ligand.params \
    --output_dir results/
```

### With Step 2 Preset

```bash
python run_pipeline.py \
    --input_pdb my_structure.pdb \
    --ref_pdb theozyme.pdb \
    --params ligand.params \
    --output_dir results/ \
    --step2_preset fast
```

### Dry Run (Preview Commands)

```bash
python run_pipeline.py \
    --input_pdb my_structure.pdb \
    --ref_pdb theozyme.pdb \
    --params ligand.params \
    --output_dir results/ \
    --dry_run
```

## CLI Reference

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--input_pdb` | Input PDB file (structure prediction with ligand aligned) |
| `--ref_pdb` | Reference PDB file (theozyme) |
| `--params` | Ligand .params file(s) (space-separated for multiple) |
| `--output_dir` | Output directory for final results |

### Global Options

| Argument | Description |
|----------|-------------|
| `--catres_subset` | Comma-separated REMARK 666 block indices for catalytic motif |
| `--verbose` | Enable verbose logging |
| `--debug` | Enable debug logging |
| `--quiet` | Minimal output |
| `--dry_run` | Print commands without executing |
| `--keep_intermediates` | Keep work directory after completion |
| `--output_tag` | Suffix appended to final output filenames (auto-uses SLURM job/task IDs if present) |
| `--short_internal_basename` | Use short hash-based basenames for intermediates (final outputs unchanged) |
| `--metrics_subdir` | Subdirectory for final metrics JSON (default: `scores_and_metrics`) |
| `--no_metrics_subdir` | Place final metrics JSON in output_dir root instead of subdir |
| `--no_metrics_history_scrub` | Keep intermediate `metrics_history` PDB paths in the final metrics JSON |

### Execution Control

| Argument | Description |
|----------|-------------|
| `--skip_step1` | Skip step 1, use existing outputs |
| `--skip_step2` | Skip step 2, use existing outputs |
| `--skip_step3` | Skip step 3 |
| `--step1_output_dir` | Path to existing step1 outputs (required if `--skip_step1`) |
| `--step2_output_dir` | Path to existing step2 outputs (required if `--skip_step2`) |

### Container Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--container_runtime` | `apptainer` | Container runtime (`apptainer` or `singularity`) |
| `--universal_container` | `/net/software/containers/universal.sif` | Path to universal.sif |
| `--pyrosetta_container` | `/net/software/containers/universal.sif` | Path to pyrosetta container (defaults to universal.sif) |
| `--container_nv` / `--nv` | off | Enable GPU passthrough for containers |

**Note**: Both container arguments default to `universal.sif`, which contains everything needed for all pipeline steps. When both containers are the same, the pipeline automatically uses in-process PyRosetta for step 3, avoiding nested container overhead.

### Step 1 Arguments

| Argument | Description |
|----------|-------------|
| `--step1_strict_backbone_importance` | BB-to-BB H-bonds alone don't make backbone important |
| `--step1_exclude_bb_only_hbond_constraints` | Exclude BB atoms from constraints for BB-only H-bonds |
| `--step1_flex_res_move_all_sc` | For ARG/LYS: move entire sidechain |
| `--step1_flex_res_constrain_all_sc` | For ARG/LYS: constrain entire sidechain |

### Step 2 Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--step2_preset` | - | Step 2 preset: `fast`, `balanced`, `thorough`, `aggressive` |
| `--step2_coord_cst_weight` | 750.0 | Coordinate constraint weight |
| `--step2_coord_cst_stdev` | 0.01 | Constraint stdev (Angstroms) |
| `--step2_cart_bonded_weight` | 3.0 | Cart_bonded term weight |
| `--step2_mobile_radius` | 10.0 | Mobile region radius (Angstroms) |
| `--step2_fastrelax_repeats` | 3 | FastRelax repeats |
| `--step2_fastrelax_ramp_stages` | 3 | FastRelax ramp stages |
| `--step2_bond_length_tolerance` | 0.05 | Bond length tolerance (Angstroms) |
| `--step2_bond_angle_tolerance` | 10.0 | Bond angle tolerance (degrees) |
| `--step2_sequence_neighbor_buffer` | 5 | Include residues +/- N from catres |
| `--step2_max_adaptive_rounds` | 10 | Max adaptive rounds |
| `--step2_scorefunction` | `ref2015_cart` | Scorefunction |
| `--step2_max_runtime` | 3600 | Max runtime (seconds) |
| `--step2_cart_bonded_scale_factor` | 1.5 | Cart_bonded scale factor when not converging |
| `--step2_cart_bonded_max` | 4.0 | Max cart_bonded weight cap |
| `--step2_fa_rep_scale` | 0.5 | Scale factor for fa_rep term |
| `--step2_fa_atr_scale` | 1.0 | Scale factor for fa_atr term |
| `--step2_fa_elec_scale` | 1.0 | Scale factor for fa_elec term |
| `--step2_ramp_fa_rep` | false | Ramp fa_rep across adaptive rounds |
| `--step2_fa_rep_min_scale` | 0.2 | Starting fa_rep scale when ramping |
| `--step2_auto_expand_mobile` | (unset) | Enable/disable auto mobile expansion (true/false, hard override) |
| `--step2_expansion_radius` | 5.0 | Mobile expansion radius (Angstroms) |
| `--step2_max_expansions` | 3 | Maximum expansions |
| `--step2_catres_bond_tolerance` | 0.05 | Catres-specific bond tolerance |
| `--step2_catres_angle_tolerance` | 10.0 | Catres-specific angle tolerance |
| `--step2_require_catres_converged` | (unset) | Require catres geometry convergence (true/false, hard override) |
| `--step2_enable_bond_geometry_min` | (unset) | Enable/disable bond geometry minimization (true/false, hard override) |

### Step 3 Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--step3_protocol` | - | Protocol name (JSON basename in step03 protocols/) |
| `--step3_protocol_file` | `modules/step03__fastmpnndesign/protocols/default.json` | Path to protocol file (.json or .txt); default unless protocol is provided |
| `--step3_design_secondary_sphere` | false | Include secondary sphere in design |
| `--step3_design_gly_pro` | false | Allow GLY/PRO redesign |
| `--step3_mpnn_spheres` | - | Override design spheres |
| `--step3_mpnn_temperature` | 0.1 | MPNN sampling temperature |
| `--step3_mpnn_num_designs` | 8 | Designs per MPNN round |
| `--step3_mpnn_batch_size` | 1 | MPNN batch size |
| `--step3_mpnn_omit_aa` | `CM` | Amino acids to exclude |
| `--step3_coord_cst_weight` | 750.0 | Coordinate constraint weight |
| `--step3_coord_cst_stdev` | 0.01 | Constraint stdev |
| `--step3_scorefunction_cart` | `ref2015_cart` | Cartesian scorefunction |
| `--step3_scorefunction_torsional` | `beta_jan25` | Torsional scorefunction |
| `--step3_num_final_designs` | - | Number of final designs (overrides protocol target_count) |
| `--step3_max_runtime` | 7200 | Max runtime (seconds) |
| `--step3_skip_initial_cart_relax` | false | Skip initial cart relaxation |
| `--step3_no_mpnn_server` | false | Disable MPNN server mode |
| `--step3_mpnn_server_port` | 5000 | MPNN server port (auto-assigned for SLURM arrays if unset) |
| `--step3_mpnn_use_gpu` | false | Force MPNN to use GPU |
| `--step3_mpnn_container_image` | (universal) | Apptainer image for MPNN |

## Step 2 Presets

Use `--step2_preset` to apply curated relaxation settings for step 2 only.
Explicit `--step2_*` flags override preset values.

### fast
Quick run for testing. Minimal relaxation cycles.
- 1 repeat × 3 stages, no bond geometry min, no auto expansion

### balanced (default)
Good balance of quality and speed. Recommended for most use cases.
- 3 repeats × 3 stages, tuned weights (from sweep)

### thorough
High-quality run with stricter tolerances and more cycles.
- 5 repeats × 5 stages, tighter tolerances (0.03Å bonds, 5° angles)

### aggressive
Aggressive optimization with higher weights and more region expansions.
- Higher cart_bonded weights, more expansion attempts

## Output Files

### Final Outputs (in `output_dir/`)

```
{basename}_design_00.pdb      # Best design (or {basename}_{output_tag}_design_00.pdb if output_tag set)
{basename}_design_01.pdb      # Second best design
...
{basename}_design_09.pdb      # Tenth design
scores_and_metrics/{basename}_design_metrics.json # Comprehensive metrics (tagged if output_tag set)
```

Note: The final metrics JSON is post-processed to reduce repetition. Shared constants
(e.g., bond/angle tolerances, catres counts, lDDT thresholds/cutoff) are lifted into
`metadata.metrics_constants` and removed from each per-design metrics block when uniform.

### Intermediate Files (with `--keep_intermediates`)

```
output_dir/
├── {basename}_design_*.pdb
├── scores_and_metrics/
│   └── {basename}_design_metrics.json
└── .pipeline_work_{basename}_{timestamp}/
    ├── step01/
    │   ├── {basename}_aligned.pdb
    │   ├── {basename}_interactions.json
    │   └── {basename}_recommended_atom_cst.json
    ├── step02/
    │   ├── {basename}_aligned_relaxed.pdb
    │   └── {basename}_aligned_relaxed_metrics.json
    └── step03/
        ├── design_00.pdb ... design_09.pdb
        └── fastmpnn_design_results.json
```

## Advanced Usage

### Custom Protocol (Step 3)

Specify a protocol file for step 3 (.json or .txt):

```bash
python run_pipeline.py \
    --input_pdb input.pdb \
    --ref_pdb ref.pdb \
    --params ligand.params \
    --output_dir results/ \
    --step3_protocol_file /path/to/protocol.txt
```

Or select a built-in JSON protocol by name:

```bash
python run_pipeline.py \
    --input_pdb input.pdb \
    --ref_pdb ref.pdb \
    --params ligand.params \
    --output_dir results/ \
    --step3_protocol balanced
```

### Resume from Failed Run

If the pipeline fails, intermediate files are preserved. Resume from where you left off:

```bash
# Resume from step 2 (step 1 completed)
python run_pipeline.py \
    --input_pdb input.pdb \
    --ref_pdb ref.pdb \
    --params ligand.params \
    --output_dir results/ \
    --skip_step1 \
    --step1_output_dir results/.pipeline_work_{basename}_{timestamp}/step01/
```

### Using Multiple Params Files

```bash
python run_pipeline.py \
    --input_pdb input.pdb \
    --ref_pdb ref.pdb \
    --params ligand1.params ligand2.params cofactor.params \
    --output_dir results/
```

### Specifying Catalytic Residues

Only treat specific REMARK 666 blocks as catalytic (others become conserved motif):

```bash
python run_pipeline.py \
    --input_pdb input.pdb \
    --ref_pdb ref.pdb \
    --params ligand.params \
    --output_dir results/ \
    --catres_subset 1,2,5
```

## Extending the Pipeline

### Adapting to Step Script Changes

If you modify the underlying step scripts, here's what needs updating in the pipeline runner:

#### If Step Script Arguments Change

1. **`pipeline_constants.py`**: Update the corresponding `STEP*_DEFAULTS` dict and `STEP*_ARG_MAPPING`
2. **`run_pipeline.py`**: Update argparse definitions in `build_parser()` for the relevant step group
3. **`README.md`**: Update the CLI reference section

#### If Step Output Files Change

1. **`pipeline_constants.py`**: Update `STEP*_OUTPUT_PATTERNS`
2. **`run_pipeline.py`**: Update `_copy_final_outputs()` if step3 outputs changed

#### If Adding a New Step

1. **`pipeline_constants.py`**:
   - Add `STEP*_SCRIPT` path
   - Add `STEP*_DEFAULTS` dict
   - Add `STEP*_ARG_MAPPING`
   - Add `STEP*_OUTPUT_PATTERNS`
   - Update `STEP_CONTAINERS` mapping

2. **`run_pipeline.py`**:
   - Add `run_step*()` method following the pattern of existing steps
   - Add `_build_step*_args()` method
   - Add `--step*_*` arguments in `build_parser()`
   - Update `run()` to call new step in sequence
   - Update `--skip_step*` logic if needed

3. **`README.md`**: Document new step and its arguments

### Code Organization

```
upgraded_fastMPNNdesign/
├── run_pipeline.py          # Main CLI entry point
├── pipeline_constants.py    # All configuration constants
├── README.md                # This documentation
└── modules/
    ├── module_utils/        # Shared utilities
    ├── step01__catres_alignment/
    ├── step02__constrained_cart_relax/
    └── step03__fastmpnndesign/
```

### Key Classes and Functions

**`run_pipeline.py`**:
- `PipelineRunner`: Main orchestration class
  - `run()`: Execute full pipeline
  - `run_step1()`, `run_step2()`, `run_step3()`: Individual step execution
  - `_build_container_command()`: Build apptainer/singularity commands
  - `_copy_final_outputs()`: Copy and rename final designs
  - `_cleanup()`: Remove intermediate files
- `build_parser()`: Construct argparse argument parser
- `validate_args()`: Validate CLI arguments before execution

**`pipeline_constants.py`**:
- `STEP*_DEFAULTS`: Default values for each step
- `STEP*_ARG_MAPPING`: Maps pipeline args to step script args
- `format_duration()`: Human-readable time formatting

## Troubleshooting

### Container Not Found

```
Error: Universal container not found: /net/software/containers/universal.sif
```

**Solution**: Specify the correct container path:
```bash
python run_pipeline.py ... \
    --universal_container /path/to/universal.sif
```

### Step 1 Constraints JSON Not Found

```
Step 1 constraints JSON not found
```

**Solution**: Check that step 1 completed successfully. If using `--skip_step1`, ensure `--step1_output_dir` contains the `*_recommended_atom_cst.json` file.

### Permission Denied

```
PermissionError: [Errno 13] Permission denied
```

**Solution**: Ensure you have write permissions to the output directory and that the container can access input file paths.

### Out of Memory

For large structures, step 2 or step 3 may run out of memory.

**Solution**: Use the `fast` preset or reduce parameters:
```bash
python run_pipeline.py ... \
    --step2_preset fast \
    --step3_mpnn_num_designs 4 \
    --step3_num_final_designs 5
```

### Debugging

Enable verbose logging to see detailed output:
```bash
python run_pipeline.py ... --verbose
```

For even more detail:
```bash
python run_pipeline.py ... --debug
```

Keep intermediate files for inspection:
```bash
python run_pipeline.py ... --keep_intermediates
```

### Output Tag Behavior

By default, SLURM job/task IDs (or a user-supplied `--output_tag`) are stored in
the **metrics metadata only**. Final output filenames are **not** modified unless
you explicitly pass:

```bash
--append_output_tag
```

### Low Output Count (Unexpectedly Few Designs)

If you requested many final designs (e.g., `--step3_num_final_designs 50`) but only
see a handful of outputs, this was previously caused by **filename collisions**
in step03. The fix adds deterministic suffixes to MPNN and Rosetta outputs, which
prevents silent overwrites. If you modify step03 or add custom steps, ensure all
outputs use the collision-safe naming helpers or a unique suffix.

## Architecture

### Pipeline Flow

```
Input PDB + Reference Theozyme + Ligand Params
                    │
                    ▼
    ┌───────────────────────────────────┐
    │  Step 1: Catalytic Alignment      │
    │  - Ligand superposition           │
    │  - Interaction analysis           │
    │  - Constraint generation          │
    └───────────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────┐
    │  Step 2: Geometry Relaxation      │
    │  - Coordinate constraints         │
    │  - Cartesian FastRelax            │
    │  - Bond/angle optimization        │
    └───────────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────┐
    │  Step 3: Sequence Design          │
    │  - Residue sphere classification  │
    │  - MPNN sequence design           │
    │  - Rosetta refinement             │
    │  - Interaction conservation       │
    └───────────────────────────────────┘
                    │
                    ▼
           Final Designs (PDBs + Metrics)
```

### Container Usage

| Step | Container | Purpose |
|------|-----------|---------|
| 1 | universal.sif | NumPy, standard scientific Python |
| 2 | pyrosetta.sif | PyRosetta for Rosetta protocols |
| 3 | pyrosetta.sif | PyRosetta + ProteinMPNN |

### File Dependencies

```
step1 outputs:
  *_recommended_atom_cst.json  →  step2 input

step2 outputs:
  *_relaxed_metrics.json       →  step3 input

step3 outputs:
  design_*.pdb                 →  final outputs
  fastmpnn_design_results.json →  final metrics
```
