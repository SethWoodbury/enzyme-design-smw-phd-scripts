# Step03: FastMPNN Design with Rosetta Refinement

Iterative MPNN sequence design with Rosetta relaxation for enzyme active site optimization.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation & Dependencies](#installation--dependencies)
- [CLI Reference](#cli-reference)
- [Protocol System](#protocol-system)
- [Sphere Classification](#sphere-classification)
- [Interaction Conservation](#interaction-conservation)
- [Metrics & Output](#metrics--output)
- [Module Architecture](#module-architecture)
- [Testing](#testing)
- [Hyperparameter Sweep](#hyperparameter-sweep)
- [MPNN Server Mode](#mpnn-server-mode)
- [Troubleshooting](#troubleshooting)

---

## Overview

This module takes step02 output (relaxed PDB + metrics JSON) and performs:

1. **Residue classification** - Catalytic, conserved motif, design spheres (core/shell/flex/frozen)
2. **Iterative MPNN sequence design** - Configurable protocols with temperature annealing
3. **MPNN PDB restoration** - H atoms, REMARK 666, HIS tautomers restored after **every** MPNN step
4. **Rosetta relaxation** - Cartesian and torsional FastRelax with coordinate constraints
5. **Favorable interaction detection** - "H-bond keeper" style conservation of beneficial mutations
6. **Comprehensive metrics** - Bond geometry, RMSD, TM-score, lDDT, sequence identity vs step01/step02

All Rosetta and MPNN operations are executed inside the `universal.sif` apptainer container, which contains both PyRosetta (2026.03) and ProteinMPNN.

### Key Features

- **Modular protocol system** - JSON/text protocol files (built-in + custom)
- **Dynamic sphere control** - Adjust design scope per-step or mid-protocol
- **7 interaction types** - H-bond, pi-stacking, metal, hydrophobic, charged, cation-pi, halogen
- **Dual reference tracking** - Metrics vs both step01 (original) and step02 (relaxed)
- **Automatic scope expansion** - If all core residues are fixed, shell is included automatically

---

## Quick Start

### Recommended Default Protocol (default.json)

```bash
python -m modules.step03__fastmpnndesign.fastmpnn_design \
    --step02_json step02_outputs/relaxed_metrics.json \
    --params params/LIG.params \
    --output_dir output/ \
    --protocol default
```

### Using Built-in Protocols

```bash
# Fast prototyping
--protocol fast

# Moderate optimization
--protocol balanced

# Extensive optimization
--protocol thorough
```

### Using Protocol Files

```bash
python -m modules.step03__fastmpnndesign.fastmpnn_design \
    --step02_json step02_outputs/relaxed_metrics.json \
    --params params/LIG.params \
    --protocol_file protocols/default.json \
    --output_dir output/
```

---

## Installation & Dependencies

### Required

| Dependency | Purpose | Notes |
|------------|---------|-------|
| PyRosetta | Relaxation, hydrogenation, metrics | Via `universal.sif` (PyRosetta 2026.03) |
| MPNN | Sequence design | Via `universal.sif` (ligand_mpnn/fused_mpnn) |
| numpy | Array operations | Standard Python package |
| apptainer/singularity | Container execution | Single `universal.sif` for everything |

### Internal Dependencies

- `module_utils.constants` - Shared constants (temperatures, layer cuts, etc.)
- `module_utils.pdb_utils` - PDB parsing (REMARK 666, atoms, sequences)
- `module_utils.sequence_utils` - Sequence identity, mutation tracking
- `module_utils.interaction_utils` - H-bond, pi-stacking, metal coordination detection
- `module_utils.pyrosetta_utils` - PyRosetta initialization fallback

### Container Configuration

The default container is `universal.sif` which contains both PyRosetta (2026.03) and MPNN:

```bash
# Default container (recommended)
/net/software/containers/universal.sif

# Override container path if needed
--pyrosetta_image /path/to/custom_container.sif

# Run PyRosetta in-process (if PyRosetta installed on host, or inside container)
--rosetta_in_process
```

**Note on beta_jan25 scorefunction**: The `beta_jan25` scorefunction (improved LJ parameters, merged Dec 2025) is enabled by default using the `-beta_jan25` PyRosetta flag. This provides better steric clash handling for relaxed/designed proteins.

### Unified Container Benefits

The `universal.sif` container (default: `/net/software/containers/universal.sif`) provides:

| Feature | universal.sif | pyrosetta.sif |
|---------|---------------|---------------|
| PyRosetta version | 2026.03 (newer) | 2025.51 (older) |
| beta_jan25 scorefunction | ✓ | ✓ |
| Multi-threading (cxx11thread) | ✓ | ✗ |
| Serialization | ✓ | ✗ |
| ProteinMPNN | ✓ | ✗ |

Using a single container for both MPNN and Rosetta operations simplifies execution and avoids nested container overhead. When running via the pipeline (`run_pipeline.py`), the `--rosetta_in_process` flag is automatically added to avoid nested container calls.

### Execution Modes

Three execution modes are available:

| Mode | Flag | Use Case |
|------|------|----------|
| **Container** (default) | none | Running from host, uses `universal.sif` for all operations |
| **In-process** | `--rosetta_in_process` | PyRosetta installed on host or already inside container |
| **No-container** | `--no-container` | Already inside container, skip container wrapping |

**When to use each:**
- **Container mode** - Default. Use when running from a host system without PyRosetta installed.
- **In-process mode** - Use when PyRosetta is available in the current environment (installed locally or running inside a container). Faster startup, no subprocess overhead.
- **No-container mode** - Use when already inside the `universal.sif` container to avoid nested container calls. The script auto-detects container environments and suggests this flag.

---

## CLI Reference

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--step02_json` | Path to step02 metrics JSON (required) |
| `--params` | Ligand .params file(s) (required, can specify multiple) |
| `--output_dir` | Output directory (required) |

### Protocol Selection

| Argument | Default | Description |
|----------|---------|-------------|
| `--protocol` | `default` | Protocol name (basename of JSON file in `protocols/`) |
| `--protocol_file` | - | Path to protocol `.json` or `.txt` file (overrides `--protocol`) |

### Design Scope

| Argument | Default | Description |
|----------|---------|-------------|
| `--catres_subset` | - | Override catalytic residue subset (e.g., "1,2,5") |
| `--design_secondary_sphere` | false | Include shell sphere in design |
| `--design_gly_pro` | false | Allow GLY/PRO redesign |
| `--layer_cuts` | 6.0 8.0 12.0 | Distance cutoffs for spheres (3 values) |
| `--mpnn_spheres` | - | Override MPNN spheres globally (e.g., "core,shell") |

### MPNN Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--mpnn_temperature` | 0.1 | Sampling temperature |
| `--mpnn_num_designs` | 8 | Designs per MPNN round |
| `--mpnn_num_designs_after_first` | - | Reduce designs after first round |
| `--mpnn_batch_size` | 1 | Batch size (1 = max diversity) |
| `--mpnn_omit_aa` | CM | Amino acids to omit from design |

### Rosetta Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--coord_cst_weight` | 750.0 | Coordinate constraint weight |
| `--coord_cst_stdev` | 0.01 | Coordinate constraint standard deviation |
| `--global_coord_cst_weight` | 0.0 | Global backbone constraint weight |
| `--global_coord_cst_stdev` | 0.5 | Global backbone constraint stdev |
| `--cart_bonded_weight` | 2.0 | Cartesian bonded term weight |
| `--scorefunction_cart` | ref2015_cart | Scorefunction for cartesian relax |
| `--scorefunction_torsional` | beta_jan25 | Scorefunction for torsional relax |
| `--fa_rep_weight` | - | Override fa_rep weight |
| `--relax_rounds` | 5 | FastRelax rounds per repeat |
| `--relax_inner_cycles` | - | Override inner cycles |

**Important:** Global coordinate constraints are **DISABLED by default** (`--global_coord_cst_weight 0.0`). This means only catalytic residues and ligand atoms are constrained. To enable backbone constraints on the entire protein, set `--global_coord_cst_weight` to a positive value (recommend 50.0-250.0). The global constraints use a much looser stdev (0.5Å vs 0.01Å for catres) to allow overall flexibility while maintaining fold.

**Constraint Energy Formula:** `penalty = weight × 0.5 × (displacement / stdev)²`
- At 0.01Å displacement with weight=750, stdev=0.01: penalty = 375
- At 0.1Å displacement: penalty = 37,500 (exponential increase)

### Convergence Criteria

| Argument | Default | Description |
|----------|---------|-------------|
| `--bond_length_tolerance` | 0.05 | Bond length tolerance (Angstroms) |
| `--bond_angle_tolerance` | 10.0 | Bond angle tolerance (degrees) |

### Reference Structures

| Argument | Default | Description |
|----------|---------|-------------|
| `--step01_pdb` | - | Original step01 PDB for CA RMSD comparison |

### Protocol Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--skip_initial_cart_relax` | false | Skip leading cart_relax steps |
| `--cart_relax_max_rounds` | 5 | Max rounds for until_converged cart_relax |

### Interaction Conservation

Interaction conservation is **protocol-driven** via `keep_interactions` steps.
Use those steps to define which interactions to preserve and with what probability.

| Argument | Default | Description |
|----------|---------|-------------|
| `--include_bb_hbond_constraints` | false | Include backbone H-bond constraints |

### Output Control

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_final_designs` | - | Number of final designs to output (overrides protocol `target_count`) |
| `--keep_intermediates` | false | Keep intermediate MPNN/relax outputs |

### Runtime & Execution

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_runtime` | 7200 | Maximum total runtime (seconds) |
| `--rosetta_timeout` | 7200 | Timeout per Rosetta subprocess (seconds) |
| `--pyrosetta_image` | (auto) | Path to PyRosetta apptainer image |
| `--pyrosetta_dir` | - | PyRosetta installation directory |
| `--rosetta_in_process` | false | Run PyRosetta in-process (no container) |
| `--no-container` | false | Disable container execution |

### MPNN Server Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--no-mpnn-server` | false | Disable MPNN server (use subprocess fallback; server is default) |
| `--mpnn-server-host` | localhost | MPNN server hostname |
| `--mpnn-server-port` | 5000 | MPNN server port |
| `--no-auto-start-mpnn-server` | false | Don't auto-start server on first MPNN call |

### MPNN Execution Controls

| Argument | Default | Description |
|----------|---------|-------------|
| `--mpnn_use_container` | false | Force MPNN to use container runtime (apptainer) |
| `--mpnn_no_container` | false | Disable container for MPNN (run run.py directly) |
| `--mpnn_use_gpu` | false | Force GPU usage for MPNN (`--nv`) |
| `--mpnn_no_gpu` | false | Force CPU-only MPNN (no `--nv`) |
| `--mpnn_container_image` | (universal.sif) | Apptainer image for MPNN subprocess/server |

**Defaults:** If you do not set any of the flags above, MPNN auto-detects GPU availability and uses the container runtime unless disabled.

### Verbosity

| Argument | Description |
|----------|-------------|
| `--quiet` | Minimal output (errors and final summary only) |
| `--verbose` | Detailed output (timing, all metrics, debug info) |
| `--debug` | Debug mode with extra logging |
| `--test` | Test mode (minimal operations) |

---

## Protocol System

### Available Protocols (JSON)

Built-in protocol files live in `protocols/` and are selected via `--protocol <name>`.

| Protocol | Description | Use Case |
|--------|-------------|----------|
| `fast` | Light geometry + single design round | Quick testing |
| `balanced` | Geometry optimization + iterative design | Default, recommended |
| `thorough` | Extensive cart_relax + multi-round design | Publication quality |
| `aggressive` | High diversity, lighter optimization | Exploration |
| `design_only` | Single MPNN round + torsional relax | Fast prototyping |
| `geometry_only` | Cart relax only, no sequence design | Geometry optimization |
| `breadth` | High diversity multi-round sampling | Sequence space exploration |
| `depth` | Low temperature, thorough geometry | Conservative refinement |
| `iterative_refine` | Progressive temperature reduction | Efficient refinement |
| `progressive` | Core → shell expansion | Staged design |
| `geometry_first` | Cart relax → design on optimized geometry | Geometry-critical |
| `design_secondary_shell` | Include shell sphere design | Extended active site |
| `progressive_expansion` | Dynamic scope expansion example | Advanced protocols |

See `protocols/` for full JSON files and descriptions.

### Text Protocol Syntax (.txt)

Custom protocol strings are used in `.txt` protocol files (or inline within a `.txt` line) and are specified as steps separated by `->`. Run them with `--protocol_file path/to/protocol.txt`:

```
step1 -> step2 -> step3 -> ...
```

### Step Types Reference

#### MPNN Design Steps

| Step | Syntax | Description |
|------|--------|-------------|
| `mpnn` / `design_core` | `mpnn:T0.2:N8` | Design core sphere (0-6Å) |
| `design_core_shell` | `design_core_shell:T0.2:N8` | Design core + shell (0-8Å) |
| `design_shell_only` | `design_shell_only:T0.1:N4` | Design shell only (6-8Å) |
| `design_shell_flex` | `design_shell_flex:T0.1:N4` | Design shell + flex (6-12Å) |
| `design_flex_only` | `design_flex_only:T0.1:N4` | Design flex zone only (8-12Å) |
| `design_core_shell_flex` | `design_core_shell_flex:T0.1:N8` | Design all three zones (0-12Å) |
| `design_distant_only` | `design_distant_only:T0.1:N4` | Design distant/frozen only (>12Å) |
| `design_global` | `design_global:T0.3:N32` | Design all non-fixed residues |

**MPNN Parameters:**
- `T<float>` - Temperature (e.g., `T0.2`). Use 0.1 for conservative, 0.2-0.3 for diversity
- `N<int>` - Number of designs (e.g., `N8`)
- `B<int>` - Batch size (e.g., `B1`). Lower = more diversity (default: 1)
- `spheres=<list>` - Override spheres (e.g., `spheres=core,shell`)
- `enhance=<model>` - Enhancement model (e.g., `plddt_3_20240930-f9c9ea0f`)
- `use_sc_context=<bool>` - Include sidechain context (default: true)
- `pack_side_chains=<bool>` - Pack sidechains during design (default: true)
- `sc_denoising_steps=<int>` - Sidechain denoising iterations (default: 3)
- `omit_aa=<str>` - Amino acids to exclude (e.g., `omit_aa=CM` excludes Cys/Met)
- `repack_everything=<bool>` - Repack all residues vs selective (default: false)

**Temperature Guidance:**
- `T=0.1` - Very conservative, locally similar sequences
- `T=0.2` - Moderate sampling (recommended starting point)
- `T=0.3+` - High diversity, more exploration

**Legacy aliases:** `mpnn_primary`, `mpnn_secondary`, `mpnn_2nd_shell`, `mpnn_all`

#### Multi-strategy MPNN (JSON only)

You can pool multiple MPNN strategies in a single step and continue with the merged, de-duplicated outputs:

```json
{
  "type": "mpnn_multi",
  "defaults": {"batch_size": 1, "omit_aa": "CM"},
  "parallel": false,
  "max_workers": 2,
  "strategies": [
    {"type": "design_core_shell", "temperature": 0.2, "num_designs": 2},
    {"type": "design_core_shell_flex", "temperature": 0.2, "num_designs": 2}
  ]
}
```

Each strategy runs against the same input structures. Outputs are restored (H atoms/REMARK 666) as usual, then pooled and de-duplicated before the next step. Use `defaults` to share parameters and override per-strategy as needed.

Parallel notes:
- `parallel: true` enables concurrent strategy execution when resources allow.
- `max_workers` caps workers (auto-detected from `SLURM_CPUS_PER_TASK` / CPU count).
- Parallel is automatically disabled when `--rosetta_in_process` is active.
- `use_mpnn_server` can override server usage for this step (parallel forces server off).

#### Relaxation Steps

| Step | Syntax | Description |
|------|--------|-------------|
| `cart_relax` | `cart_relax:R2S3` | Cartesian FastRelax |
| `torsional_relax` | `torsional_relax:R1S2` | Torsional FastRelax |
| `minimize` | `minimize:T0.01:I200` | Minimization only |
| `repack` | `repack:shell=8.0` or `repack:scope=core_shell_flex` | Sidechain repacking |

**Relax Parameters:**
- `R<int>` - Repeats (e.g., `R2`)
- `S<int>` - Stages (e.g., `S3`)
- `until_converged` - Run until geometry converges (cart_relax only)
- `sf=<name>` / `scorefunction=<name>` - Override scorefunction (e.g., `beta_jan25`, `ref2015`, `ref2015_cart`)
- `minimize_scope` / `scope` - Limit minimize to core/core_shell/core_shell_flex/global (minimize only)

**Repack notes:**
- `repack_shell` (legacy) overrides `repack_scope` if both are provided.
- Repack never includes catalytic residues or other fixed residues.

#### Selection Steps

| Step | Syntax | Description |
|------|--------|-------------|
| `select_best` | `select_best:N1:metric=geometry` | Select top N structures |
| `cluster` | `cluster:sequence:N2` | Cluster by sequence/structure |
| `keep_cluster_best` | `keep_cluster_best:N1:metric=smart` | Best per cluster |

**Selection Metrics:**
- `geometry` - Bond length deviation
- `score` - Rosetta total score
- `smart` - Multi-criteria (geometry → CA RMSD → score)
- `ca_rmsd` - CA RMSD vs reference
- `sequence_similarity_high` / `sequence_similarity_low` - Sequence identity

#### Configuration Steps

| Step | Syntax | Description |
|------|--------|-------------|
| `set_layer_cuts` | `set_layer_cuts:6.0:8.0:12.0` | Update sphere boundaries mid-protocol |
| `scale` | `scale:fa_rep=0.15` | Scale scorefunction terms |
| `set` | `set:mpnn_temperature=0.15` | Update runtime defaults |
| `time_check` | JSON only | Conditional time-based branch |

**Scale step JSON format** (uses `terms` dict):
```json
{"type": "scale", "terms": {"fa_rep": 0.15, "global_coord_constraint": 0.0}, "scope": "global"}
```
You can also reset a term to its scorefunction default by using `"reset"`:
```json
{"type": "scale", "terms": {"fa_rep": "reset"}}
```

**Set step JSON format** (uses `options` dict):
```json
{"type": "set", "options": {"mpnn_temperature": 0.15, "mpnn_num_designs": 8}}
```

**Set layer cuts** - Allows dynamic mid-protocol adjustment of design sphere boundaries:
```json
{"type": "set_layer_cuts", "core_cutoff": 5.0, "shell_cutoff": 7.0, "flex_cutoff": 11.0}
```

**Score term note:** `global_coord_constraint` uses the same Rosetta
`coordinate_constraint` term. The global weight is applied by scaling the
global constraint stdev so that catres and global constraints can have
different effective weights.

#### Interaction Steps

| Step | Syntax | Description |
|------|--------|-------------|
| `keep_interactions` | `keep_interactions:target=ligand:types=hbond,pi:prob=0.75` | Fix interacting residues |

#### Advanced Steps

| Step | Syntax | Description |
|------|--------|-------------|
| `final_diversify` | JSON only | Multi-temperature MPNN with target count |

**Final Diversify** (used in `default.json`):
Runs MPNN at multiple temperatures, deduplicates, clusters, and expands to reach a target design count.

```json
{
  "type": "final_diversify",
  "temperatures": [0.1, 0.15, 0.2],
  "target_count": 30,
  "designs_per_temp": 3,
  "max_iterations": 3,
  "design_scope": "shell_only",
  "overshoot_threshold": 0.75,
  "fallback_include_flex": true
}
```

Parameters:
- `temperatures` - List of temperatures to sample at (default: [0.1, 0.2, 0.3])
- `target_count` - Target number of final designs (used when `--num_final_designs` is not set)
- `designs_per_temp` - Designs generated per temperature per input (default: 3)
- `max_iterations` - Max expansion iterations if target not met (default: 3)
- `design_scope` - Region to design: "shell_only", "core", "global", etc. (default: "shell_only")
- `overshoot_threshold` - If max_possible < threshold × target, double batch_size (default: 0.75)
- `fallback_include_flex` - Try shell+flex if iterations exhausted (default: true)

### Protocol File Formats

#### JSON Format

```json
{
  "name": "my_protocol",
  "description": "Custom protocol description",
  "version": "2.2",
  "layer_cuts": [6.0, 8.0, 12.0],
  "steps": [
    {"type": "design_core", "temperature": 0.2, "num_designs": 8},
    {"type": "cart_relax", "repeats": 2, "stages": 3, "until_converged": true},
    {"type": "select_best", "n": 1, "metric": "geometry"},
    {"type": "design_core", "temperature": 0.1, "num_designs": 16},
    {"type": "torsional_relax", "repeats": 2, "stages": 3}
  ]
}
```

#### Text Format

```
# Comment lines start with #
mpnn:T0.2:N8:spheres=core
cart_relax:R2S3
select_best:N1:metric=smart
mpnn:T0.1:N16
torsional_relax:R2S3
```

See `protocols/` for built-in JSON protocols (e.g., `default.json`, `fast.json`, `balanced.json`, `thorough.json`, `progressive_expansion.json`).

**Time-based fallback (JSON only):**
```json
{
  "type": "time_check",
  "max_elapsed": 3600,
  "target_total_designs": 30,
  "then": [
    {"type": "cluster", "method": "sequence", "n_clusters": 10},
    {"type": "keep_cluster_best", "n": 1, "metric": "geometry"},
    {"type": "design_core_shell", "temperature": 0.1}
  ]
}
```

### Expected Output Counts (High-Level)

Expected output count depends on:
- MPNN branching (`num_designs` per MPNN step)
- Selection steps (`select_best`, `keep_cluster_best`)
- Dedupe (duplicate sequences are removed after each MPNN step)

There is no pre‑run estimator; counts are logged per step as the protocol executes.

### Logs

- **Rosetta subprocess logs**: `output_dir/logs/rosetta_*.out` and `output_dir/logs/rosetta_*.err`
- **MPNN logs**: `mpnn.stdout` / `mpnn.stderr` inside each `mpnn_*` output folder

### Example Protocols

**Progressive Temperature Reduction (Fast, ~2 min):**
```
mpnn:T0.2:N1 -> torsional_relax:R1S2 -> mpnn:T0.15:N1 -> torsional_relax:R1S2 -> mpnn:T0.1:N10 -> torsional_relax:R2S3
```

**Single-Shot High Diversity (~4 min):**
```
mpnn:T0.3:N32 -> torsional_relax:R1S2
```

**Geometry-First (~15 min):**
```
cart_relax:R2S3 -> mpnn:T0.1:N20 -> torsional_relax:R2S3
```

**Multi-Round Breadth (~20 min):**
```
mpnn:T0.3:N16 -> torsional_relax:R1S2 -> mpnn:T0.2:N16 -> torsional_relax:R1S2 -> mpnn:T0.1:N16 -> torsional_relax:R2S3
```

---

## Sphere Classification

Residues are classified based on distance from ligand heavy atoms:

| Sphere | Default Range | MPNN Behavior | Rosetta Behavior |
|--------|---------------|---------------|------------------|
| `DESIGN_CORE` | 0-6Å | Full redesign | Full flexibility |
| `DESIGN_SHELL` | 6-8Å | Redesign (with CB check) | Full flexibility |
| `FLEX` | 8-12Å | Sequence fixed | Sidechain repack only |
| `FROZEN` | >12Å | Sequence fixed | Completely fixed |

### CB Orientation Check

For shell residues, we check if the CB atom points toward the ligand (CB distance < CA distance). This identifies residues whose sidechains interact with the active site.

### Automatic Scope Expansion

If all core sphere residues are catalytic (fixed), the design scope automatically expands to include shell residues.

### Fixed Residues

These are never redesigned:
- **Catalytic residues** - In the catres_subset (constrained)
- **Conserved motif** - In REMARK 666 but not catres_subset
- **Protected residues** - GLY/PRO (unless `--design_gly_pro`)

Repack steps also exclude these fixed residues (catres are never repacked).

### Adjusting Layer Cuts

```bash
# CLI override
--layer_cuts 5.0 7.0 11.0

# Mid-protocol adjustment (in protocol file)
{"type": "set_layer_cuts", "core_cutoff": 5.0, "shell_cutoff": 7.0, "flex_cutoff": 11.0}
```

---

## Interaction Conservation

The module conserves beneficial mutations that form favorable interactions ("H-bond keeper"):

### Supported Interaction Types

| Type | Description | Default Bias |
|------|-------------|--------------|
| `hbond` | Hydrogen bonds | 2.0 |
| `pi_stack` | π-stacking interactions | 1.5 |
| `metal` | Metal coordination | 2.5 |
| `hydrophobic` | Hydrophobic contacts | 0.5 |
| `charged` | Salt bridges, ionic | 2.0 |
| `cation_pi` | Cation-π interactions | 1.5 |
| `halogen` | Halogen bonds | 1.0 |

### Conservation Workflow

1. After MPNN design, analyze mutations for interactions with catalytic residues or ligand
2. Mutations forming favorable interactions are probabilistically fixed
3. MPNN biases are applied to favor beneficial amino acids in subsequent rounds

### Protocol Step (recommended)

```
keep_interactions:target=ligand:types=hbond,pi:prob=0.75
keep_interactions:target=catres:types=hbond:prob=0.5
```

---

## Metrics & Output

### Output Directory Structure

```
output_dir/
├── fastmpnn_design_results.json    # Complete results with all metrics
├── design_00.pdb                   # Best design (rank 0)
├── design_01.pdb                   # Second best
├── ...
├── design_N.pdb                    # Nth design
├── mpnn_0/                         # MPNN step 0 outputs
│   ├── seqs/
│   ├── backbones/
│   └── packed/
├── mpnn_1/                         # MPNN step 1 outputs
├── *.cst.json                      # Constraint files
├── *.restored_h.pdb                # Hydrated, REMARK-restored intermediates
└── *.ligfixed.pdb                  # Ligand forced to reference coordinates
```

### Filename Collision Prevention (Important)

We previously encountered a failure mode where MPNN outputs from different branches
shared the same basename (e.g., `input_for_mpnn_packed_2_1...`). When these were
restored into a common directory, later steps silently **overwrote** files, collapsing
50 designs down to 4–5. This was fixed and the pipeline now enforces **collision-safe**
names at every stage:

- **MPNN outputs** include a deterministic packed suffix (e.g., `_m001_s02`) via
  `--packed_suffix`, so parallel strategies never overwrite each other.
- **Rosetta outputs** now include a short path hash (e.g., `.repack.ab12cd34.pdb`)
  so repeated basenames from different branches remain unique.
- **Final restoration** also adds a short hash to prevent flattening collisions.

If you add new steps or new file-writing code, **always** use the collision-safe
helpers (e.g., `_make_tagged_output_path`) or append a unique suffix. This avoids
silent data loss and undercounts in final outputs.

**Name length safety**: `_make_tagged_output_path` includes a safeguard that
shortens very long filenames (default max length = 200 chars) using deterministic
hashes. This keeps paths stable and collision‑safe while avoiding filesystem or
MPNN name limits.

### Safety Checks & Logging

Step03 emits warnings when:
- MPNN output counts are far below expected (dedupe or constraints often cause this).
- Cluster counts are lower than requested.
- Potential basename collisions are detected at any step.
- Final unique designs are below target.

These checks are designed to surface silent failures early. If you adjust protocols,
watch for these warnings and validate the `output_designs` count in the results JSON.

### Results JSON Structure

```json
{
  "metadata": {
    "step02_json": "/path/to/step02_metrics.json",
    "step02_pdb": "/path/to/step02_relaxed.pdb",
    "step01_pdb": "/path/to/step01_aligned.pdb",
    "protocol": "protocol:balanced",
    "protocol_steps": ["mpnn:T0.2:N2", "cart_relax:R2S3", ...],
    "runtime_seconds": 1234.5,
    "timestamp": "2026-01-28T12:00:00"
  },
  "residue_classification": {
    "num_catalytic": 17,
    "num_conserved_motif": 2,
    "num_design_core": 45,
    "num_design_shell": 33,
    "num_flex": 48,
    "num_frozen": 112,
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
        "tm_score": {
          "vs_step02": {"tm_score": 0.98, "num_matched": 256},
          "vs_step01": {"tm_score": 0.96, "num_matched": 256}
        },
        "lddt": {
          "vs_step02": {"lddt_ca": 0.99, "num_matched": 256},
          "vs_step01": {"lddt_ca": 0.97, "num_matched": 256}
        },
        "bond_geometry": {
          "bond_length_geometry": {
            "all": {"max": 0.08, "mean": 0.02, "std": 0.01},
            "unconstrained_only": {"max": 0.04, "mean": 0.02}
          },
          "bond_angle_geometry": {
            "all": {"max": 8.5, "mean": 2.1},
            "unconstrained_only": {"max": 6.2, "mean": 1.8}
          }
        },
        "convergence": {
          "bond_length_converged": true,
          "bond_angle_converged": true
        },
        "motif_mutation_check": {
          "motif_preserved": true,
          "num_motif_checked": 19
        },
        "rosetta_score": -245.3,
        "dunbrack_rotamer_quality": 0.92,
        "dssp_secondary_structure": "HHHHHCCCEEEE...",
        "sasa": {"total": 12500.0, "polar": 4200.0, "nonpolar": 8300.0}
      }
    }
  ],
  "metrics_history": [...]
}
```

### Metrics Descriptions

| Metric | Description |
|--------|-------------|
| `sequence_identity_vs_step02` | Fraction of identical residues vs step02 |
| `num_mutations` | Total mutations from step02 |
| `ligand_rmsd` | RMSD of ligand heavy atoms (should be ~0.0) |
| `constrained_atoms_rmsd` | RMSD of constrained atoms (should be ~0.0) |
| `global_ca_vs_step01` | CA RMSD vs original step01 structure |
| `global_ca_vs_step02` | CA RMSD vs relaxed step02 structure |
| `tm_score` | TM-score (CA-only, Kabsch alignment) |
| `lddt` | lDDT score (CA-only, 15Å cutoff, 0.5/1/2/4Å thresholds) |
| `bond_length_geometry` | Bond length deviations from ideal |
| `bond_angle_geometry` | Bond angle deviations from ideal |
| `unconstrained_only` | Geometry excluding fully constrained atoms |
| `dunbrack_rotamer_quality` | Fraction of residues in favorable rotamers |
| `dssp_secondary_structure` | DSSP secondary structure assignment |
| `sasa` | Solvent accessible surface area |

### Hydrogen / REMARK 666 / HIS Tautomer Handling

MPNN outputs do not include hydrogens and may lose REMARK 666 lines. This module **forces restoration after every MPNN step**:

1. **Normalize input for MPNN** - Remove H atoms, convert HIS variants → HIS
2. **Restore MPNN outputs** - Re-add REMARK 666, restore HIS tautomers, restore HETATM
3. **Hydrogenate via PyRosetta** - Add H atoms without minimization
4. **Re-insert REMARK 666** - After PyRosetta dump
5. **Force ligand atoms** - Match reference coordinates (including hydrogens)

---

## Module Architecture

```
step03__fastmpnndesign/
├── __init__.py                     (0.6 KB)   Module initialization
├── fastmpnn_design.py              (108 KB)   Main orchestrator/CLI
├── protocol_parser.py              (91 KB)    Protocol parsing, step types
├── metrics.py                      (91 KB)    Comprehensive metrics calculation
├── interaction_analyzer.py         (57 KB)    Interaction detection, conservation
├── rosetta_relax.py                (34 KB)    Rosetta relaxation (cartesian/torsional)
├── pdb_restoration.py              (32 KB)    H atoms, REMARK 666, tautomers
├── residue_classifier.py           (28 KB)    Sphere classification, ResidueInfo
├── mpnn_runner.py                  (21 KB)    MPNN subprocess/server execution
├── mpnn_server.py                  (22 KB)    Persistent MPNN server (GPU memory)
├── rosetta_metrics.py              (4.5 KB)   Rosetta-based metrics helper
├── README.md                                   This file
├── PROTOCOL_ADVANCED.md                        Advanced protocol guide
├── protocols/                                  Protocol file examples
│   ├── README.md
│   ├── default.json
│   ├── fast.json
│   ├── thorough.json
│   └── progressive_expansion.json
├── test/                                       Test suite
│   ├── test_all.py                            Comprehensive unit tests
│   ├── test_protocol_file_parser.py           Protocol parsing tests
│   ├── verify_constraints.py                  Constraint loading tests
│   ├── run_test.sh                            Test runner
│   ├── run_integration_tests.sh               Integration tests
│   ├── example_protocol.json                  Example protocol
│   ├── example_protocol.txt                   Example text protocol
│   ├── testdata/                              Example outputs from prior runs
│   ├── params -> step02 params                Symlink
│   ├── step01_outputs -> step02 step01        Symlink
│   └── step02_outputs -> step02 output        Symlink
└── hyper_param_sweep/                          Hyperparameter optimization
    ├── ANALYSIS_REPORT.md                     Sweep analysis findings
    ├── generate_sweep.py                      Generate sweep commands
    ├── generate_large_sweep.py                Large-scale sweep (~400+ jobs)
    ├── analyze_sweep.py                       Analyze sweep results
    ├── analyze_comprehensive_sweep.py         Detailed analysis
    ├── analyze_detailed_metrics.py            Metrics analysis
    ├── compute_detailed_metrics.py            Metrics computation
    ├── compute_detailed_metrics_v2.py         V2 metrics computation
    ├── detailed_metrics_analysis.json         Analysis results
    ├── submit_array.sh                        SLURM array submission
    ├── sbatch_template.sh                     SLURM template
    ├── monitor_jobs.sh                        Job monitoring
    ├── collect_results.sh                     Results collection
    ├── FEATURES.md                            Feature documentation
    ├── INSTALLATION_SUMMARY.md                Installation guide
    ├── QUICKSTART.md                          Quick start guide
    ├── README_SLURM.md                        SLURM documentation
    ├── cmds/                                  Command files
    ├── logs/                                  Job logs
    └── outputs/                               Sweep outputs
```

### Key Design Decisions

1. **Phased Protocol** - Geometry first (cart relax) → Design → Optimize (torsional)
2. **Scorefunction Selection** - ref2015_cart for cartesian, beta_jan25 for torsional
3. **Sphere Selection** - Distance-based with CB orientation check for shell residues
4. **MPNN Diversity** - Lower batch_size (1) + higher num_designs = more diversity
5. **Favorable Interaction Conservation** - Probabilistic fixing of beneficial mutations (enabled by default)
6. **Duplicate Handling** - Remove by sequence, keep structure with best geometry
7. **Dual Reference Tracking** - Metrics vs both step01 and step02
8. **Constraint Verification** - Ligand and catres atoms should have ~0.0 RMSD
9. **Automatic Scope Expansion** - If core has no designable residues, include shell
10. **Hydrogen Guarantee** - Restored after every MPNN step
11. **MPNN Server Mode** - Persistent model in memory eliminates ~4-6s startup overhead per call
12. **Global Constraints Disabled** - Only catres/ligand constrained by default; enable with `--global_coord_cst_weight`
13. **Temperature Annealing** - Start high (T=0.2-0.3) for diversity, end low (T=0.1) for refinement
14. **Final Ranking by Geometry** - Output designs ranked by bond length deviation (best geometry = rank 0)

---

## Testing

### Run Unit Tests

```bash
cd /path/to/upgraded_fastMPNNdesign
python -m modules.step03__fastmpnndesign.test.test_all
```

### Run with Test Scripts

```bash
cd modules/step03__fastmpnndesign/test/
./run_test.sh          # Run with fast protocol
./run_test.sh balanced # Run with balanced protocol
```

### Run Protocol Parser Tests

```bash
python -m modules.step03__fastmpnndesign.test.test_protocol_file_parser
```

### Test Coverage

| Test Module | Coverage |
|-------------|----------|
| `test_all.py` | InteractionAnalyzer, ProtocolParser, PDB restoration, ResidueClassifier, FastMPNNDesigner init |
| `test_protocol_file_parser.py` | JSON/text parsing, validation, all step types |
| `verify_constraints.py` | Step02 constraint loading, PyRosetta pose creation |

---

## Hyperparameter Sweep

### Key Findings (from ANALYSIS_REPORT.md)

1. **Torsional relaxation is usually sufficient** - Cartesian adds runtime with marginal benefit
2. **Temperature annealing works well** - Start high (0.2-0.3), end low (0.1)
3. **More designs per round is efficient** - N=16-32 gives good diversity
4. **Simple protocols often match complex ones** - Progressive temperature (106s) ≈ 20-min protocols
5. **Optimal mutation range: 4-6 per design** across all successful protocols

### Recommended Defaults

| Parameter | Recommended | Range to Explore |
|-----------|-------------|------------------|
| Temperature (initial) | 0.2 | 0.15-0.3 |
| Temperature (final) | 0.1 | 0.05-0.15 |
| Designs per round | 8-16 | 4-64 |
| Torsional repeats | 1-2 | 1-3 |
| Torsional stages | 2-3 | 2-4 |
| Cartesian repeats | 2 | 1-3 |
| Cartesian stages | 3 | 2-4 |
| Cart bonded weight | 2.0 | 1.0-4.0 |
| Coord constraint weight | 750.0 | 500-1000 |

### Running a Sweep

```bash
cd hyper_param_sweep/

# Generate sweep commands
python generate_sweep.py

# Submit to SLURM
./submit_array.sh --max-parallel 20

# Analyze results
python analyze_sweep.py
```

### Large-Scale Sweep

```bash
python generate_large_sweep.py --output cmds/large_sweep_commands.txt
./submit_array.sh --commands cmds/large_sweep_commands.txt --max-parallel 50
```

---

## MPNN Server Mode

### Overview

The MPNN server keeps model weights loaded in memory between calls, eliminating the Python/PyTorch startup overhead (~4-6 seconds on CPU, ~15-20 seconds on GPU) per MPNN invocation.

**Expected speedup:**
- **GPU:** ~30s → ~4s per call (5-10x improvement)
- **CPU:** ~10s → ~4s per call (2-3x improvement)

For a protocol with 10 MPNN calls, this saves 40-200 seconds of pure startup overhead.

### Architecture

```
┌──────────────────────────┐         ┌─────────────────────────────────┐
│ FastMPNN Pipeline        │         │ MPNN Server (inside container)  │
│ (fastmpnn_design.py)     │         │                                 │
│                          │         │ - ProteinMPNN model in GPU mem  │
│ MPNNRunner               │         │ - Packer model in GPU mem       │
│   └─ MPNNServerClient ───┼── TCP ──┼─▶ handle_request()             │
│      (auto-start server) │  :5000  │   └─ model.sample()             │
│      (fallback to subpr) │         │   └─ pack_side_chains()         │
└──────────────────────────┘         │   └─ write output files         │
                                     └─────────────────────────────────┘
```

### Features

- **Enabled by default** - Server is used automatically when GPU is available
- **Auto-start** - Server starts on first MPNN call (no manual setup needed)
- **Graceful fallback** - Falls back to subprocess if server fails to start
- **Persistent models** - ProteinMPNN + Packer models stay loaded between calls
- **TCP communication** - Length-prefixed JSON protocol on localhost:5000

### Usage

```bash
# Default: server mode enabled with auto-start
python -m modules.step03__fastmpnndesign.fastmpnn_design \
    --step02_json input.json \
    --params LIG.params \
    --output_dir output/

# Disable server (use subprocess for each call)
python -m modules.step03__fastmpnndesign.fastmpnn_design \
    --step02_json input.json \
    --params LIG.params \
    --output_dir output/ \
    --no-mpnn-server

# Custom server settings
python -m modules.step03__fastmpnndesign.fastmpnn_design \
    --step02_json input.json \
    --params LIG.params \
    --output_dir output/ \
    --mpnn-server-host localhost \
    --mpnn-server-port 5001
```

### Manual Server Control

For advanced use cases, the server can be started manually:

```bash
# Start server manually (inside container)
apptainer exec --nv /net/software/containers/universal.sif \
    python /path/to/mpnn_server.py --host localhost --port 5000

# Health check
python -c "
import socket, struct, json
s = socket.socket()
s.connect(('localhost', 5000))
msg = json.dumps({'type': 'health'}).encode()
s.sendall(struct.pack('>I', len(msg)) + msg)
length = struct.unpack('>I', s.recv(4))[0]
print(json.loads(s.recv(length)))
"

# Graceful shutdown
python -c "
import socket, struct, json
s = socket.socket()
s.connect(('localhost', 5000))
msg = json.dumps({'type': 'shutdown'}).encode()
s.sendall(struct.pack('>I', len(msg)) + msg)
"
```

### Memory Requirements

| Component | Memory (GPU or CPU) |
|-----------|---------------------|
| PyTorch + imports | ~500 MB |
| ProteinMPNN model | ~200-400 MB |
| Packer model | ~100-200 MB |
| **Total persistent** | ~600-800 MB |

Server mode works on both GPU and CPU systems. The memory footprint is the same regardless of device.

### When to Disable Server Mode

Use `--no-mpnn-server` when:
- Running inside a container that already has models loaded
- Debugging MPNN subprocess execution
- Memory constraints require minimal footprint

### Fallback Behavior

If server fails to start or becomes unavailable:
1. Warning is logged
2. Execution continues via subprocess fallback
3. Each MPNN call loads models fresh (~4-6s on CPU, ~15-20s on GPU)

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Container not found | Ensure `/net/software/containers/universal.sif` exists |
| PyRosetta not found | Container should have PyRosetta 2026.03, or use `--pyrosetta_image` |
| No designs produced | Check catres_subset, increase temperature or num_designs |
| High ligand RMSD | Verify constraint files are loading correctly |
| Bond geometry not improving | Use cart_relax with higher cart_bonded_weight |
| MPNN server won't start | Check GPU availability (`nvidia-smi`), try `--no-mpnn-server` |
| Server connection refused | Check port availability, try different `--mpnn-server-port` |
| Slow MPNN on CPU | Expected; server mode only helps on GPU systems |

### Debugging

```bash
# Verbose output
--verbose

# Debug mode
--debug

# Keep intermediate files
--keep_intermediates

# Test mode (minimal operations)
--test
```

### Container Issues

```bash
# Run without spawning container subprocesses (use when already inside container)
--rosetta_in_process

# Override default container path
--pyrosetta_image /path/to/custom_container.sif

# Default container location
/net/software/containers/universal.sif
```

---

## Related Modules

- **step01__catres_alignment** - Aligns catalytic residue coordinates
- **step02__constrained_cart_relax** - Optimizes bond geometry with constraints
- **module_utils** - Shared utilities (constants, pdb_utils, sequence_utils, interaction_utils)

---

## References

- LigandMPNN: [GitHub](https://github.com/dauparas/LigandMPNN)
- PyRosetta: [Documentation](https://www.pyrosetta.org/)
- FastRelax: [RosettaCommons](https://www.rosettacommons.org/)
