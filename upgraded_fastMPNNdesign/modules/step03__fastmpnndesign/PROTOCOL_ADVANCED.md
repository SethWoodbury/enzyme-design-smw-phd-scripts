# Advanced Protocol Guide (Step03 FastMPNN Design)

This guide documents the **protocol system** for step03, including all supported
step types, parameters, and recommended use cases. It is intended to be the
canonical reference for JSON/text protocols.

The protocol system is designed to be:
- **Modular** (step-by-step control)
- **Explicit** (every step is declared)
- **Dynamic** (mid-protocol changes are allowed)
- **Compatible** with legacy text syntax

---

## 1) Protocol File Formats

### 1.1 JSON (.json)
Top-level fields:
- `name` (string, optional)
- `description` (string, optional)
- `version` (string/number, optional)
- `layer_cuts` (list of 3 floats) for core/shell/flex boundaries
- `steps` (list) of protocol steps

Example:
```json
{
  "name": "my_protocol",
  "description": "Custom protocol",
  "version": "2.3",
  "layer_cuts": [6.0, 8.0, 12.0],
  "steps": [
    {"type": "design_core", "temperature": 0.2, "num_designs": 4},
    {"type": "torsional_relax", "repeats": 1, "stages": 2},
    {"type": "select_best", "n": 1, "metric": "smart"}
  ]
}
```

### 1.2 Text (.txt)
Supports **compact syntax** and **legacy whitespace** syntax.

Compact (current syntax):
```
mpnn:T0.2:N8:spheres=primary,secondary
cart_relax:R2S3
select_best:N1:metric=smart
```

Legacy (whitespace):
```
scale:fa_rep 0.15
mpnn 0.2 8
repack
min 0.01 cartesian
keep_best 2 metric=smart
```

Inline chain in a single line is also supported:
```
mpnn:T0.2:N8 -> torsional_relax:R1S2 -> mpnn:T0.1:N8
```

---

## 2) Step Types (Quick Reference)

### MPNN design steps (JSON + text)
These define **which spheres** are designed (based on ligand distance):
- `mpnn` (default = core)
- `design_core`
- `design_core_shell`
- `design_shell_only`
- `design_shell_flex`
- `design_flex_only`
- `design_core_shell_flex`
- `design_distant_only` ("distant" = frozen region > flex cutoff)
- `design_shell_flex_distant`
- `design_global`

Legacy aliases (text or JSON):
- `mpnn_primary`, `mpnn_secondary`, `mpnn_2nd_shell`, `mpnn_all`

### Multi-strategy MPNN (JSON only)
- `mpnn_multi` (run multiple MPNN strategies, pool and dedupe results)

### Rosetta refinement
- `cart_relax` (FastRelax cartesian)
- `torsional_relax` (FastRelax torsional)
- `minimize` (MinMover only)
- `repack` (sidechain repack only)

### Selection / clustering
- `select_best`
- `cluster`
- `keep_cluster_best`

### Configuration / control
- `scale` / `scale_scoreterm`
- `set` / `set_options`
- `set_layer_cuts`
- `time_check`

### Interaction-based fixing
- `keep_interactions`

### Final diversification
- `final_diversify` (multi-temperature MPNN to reach target count)

---

## 3) Detailed Step Reference

### 3.1 MPNN design steps (`mpnn`, `design_*`)
Design sequences in specified **spheres**. All MPNN outputs are restored
(H atoms, REMARK 666, ligand fix, HIS tautomers) before continuing.

**Sphere definitions (default layer cuts = [6, 8, 12]):**
- **core**: CA distance <= core cutoff (default 6.0A) to nearest ligand heavy atom
- **shell**: CA distance <= shell cutoff (default 8.0A) with CB orientation check
- **flex**: CA distance <= flex cutoff (default 12.0A)
- **distant**: beyond flex cutoff (> 12.0A). This is normally frozen, but can be designed with
  `design_distant_only` or `design_shell_flex_distant`.

**JSON fields:**
- `temperature` (float) - sampling temperature
- `num_designs` (int) - number of designs to generate
- `batch_size` (int)
- `design_scope` (string) - core, core_shell, shell_only, shell_flex, flex_only,
  core_shell_flex, distant_only, shell_flex_distant, global
- `spheres` / `design_spheres` (list) - explicit sphere list (overrides scope)
- `use_sc_context` (bool)
- `pack_side_chains` (bool)
- `sc_denoising_steps` (int)
- `omit_aa` (string)
- `enhance` (string or null)
- `repack_everything` (bool)
- `bias_aa` (dict of AA -> float)
- `bias_aa_per_residue` (dict of residue_id -> {AA: bias})

**Notes:**
- If `num_designs` is omitted, runtime defaults are used.
- If `--mpnn_num_designs_after_first` is set, later MPNN steps may auto-reduce
  branching unless the step explicitly sets `num_designs`.

**Use cases:**
- Early exploration: higher `temperature`, broader scope
- Later refinement: lower `temperature`, narrower scope
- Distant or shell+flex design to increase peripheral diversity without touching the core

---

### 3.2 Multi-strategy MPNN (`mpnn_multi`, JSON only)
Run multiple MPNN strategies in a single step, then pool and de-duplicate
outputs before moving on.

Example:
```json
{
  "type": "mpnn_multi",
  "defaults": {"batch_size": 1, "omit_aa": "CM"},
  "parallel": false,
  "max_workers": 2,
  "min_workers": 1,
  "strategies": [
    {"type": "design_core_shell", "temperature": 0.2, "num_designs": 2},
    {"type": "design_shell_flex", "temperature": 0.2, "num_designs": 2}
  ]
}
```

**Fields:**
- `strategies` (list of MPNN steps)
- `defaults` (dict): merged into each strategy (strategy values override defaults)
- `dedupe_pool` (bool, default true)
- `parallel` (bool)
- `max_workers`, `min_workers` (ints)
- `use_mpnn_server` (bool or null)

**Notes:**
- Each strategy is restored (REMARK 666, HIS, ligand fix) before pooling.
- Parallel execution is disabled automatically with `--rosetta_in_process`.
- When parallel is enabled, MPNN server mode is forced off for safety.

**Use cases:**
- Run core-only and shell-only designs in parallel, then pool
- Mix exploration and refinement settings in the same step

---

### 3.3 Cartesian FastRelax (`cart_relax`)
Full cartesian FastRelax to improve bond geometry.

**Text syntax:**
```
cart_relax:R2S3
```

**JSON fields (subset):**
- `repeats` (int)
- `stages` (int)
- `scorefunction` (string)
- `coord_cst_weight`, `coord_cst_stdev`
- `global_coord_cst_weight`, `global_coord_cst_stdev`
- `cart_bonded_weight`
- `fa_rep_weight` (optional)
- `until_converged` (bool)

**Use cases:**
- Early geometry cleanup after big sequence changes
- Fix bond/angle outliers

---

### 3.4 Torsional FastRelax (`torsional_relax`)
Faster relaxation using torsional moves only (no explicit bond geometry optim).

**Text syntax:**
```
torsional_relax:R1S2
```

**Use cases:**
- Quick cleanup between MPNN steps
- Lower-cost refinement

---

### 3.5 Minimize (`minimize`)
Continuous minimization only (no repack). Default torsional; cartesian optional.

**Text syntax:**
```
minimize:T0.01:I200:cartesian:sf=ref2015_cart
```

**JSON fields:**
- `tolerance` (float)
- `max_iter` (int)
- `cartesian` (bool)
- `scorefunction` (string)
- `coord_cst_weight`, `coord_cst_stdev`
- `global_coord_cst_weight`, `global_coord_cst_stdev`
- `cart_bonded_weight`
- `fa_rep_weight`
- `minimize_scope` / `scope` (core/core_shell/core_shell_flex/global)
- `min_backbone_rmsd_cutoff` (float) - revert if backbone RMSD exceeds cutoff

**Use cases:**
- Lightweight geometry refinement
- Cartesian minimize to directly improve bond/angle deviations

---

### 3.6 Repack (`repack`)
Sidechain repacking only (rotamer optimization). Backbone fixed.

**Text syntax:**
```
repack:shell=8.0
repack:scope=core_shell_flex
```

**JSON fields:**
- `repack_scope` (core/core_shell/core_shell_flex/global)
- `repack_shell` (float) - legacy cutoff in Angstroms
- `scorefunction` (string)
- `fa_rep_weight` (optional)
- `coord_cst_weight`, `coord_cst_stdev`
- `global_coord_cst_weight`, `global_coord_cst_stdev`
- `score_term_weights` (dict)

**Notes:**
- `repack_shell` overrides `repack_scope` if both are set.
- Catalytic residues and fixed residues are never repacked.

**Use cases:**
- Quick sidechain cleanup before minimization
- Fix packing without backbone motion

---

### 3.7 Selection (`select_best`)
Select top N structures by a metric.

**Metrics:**
- `geometry` (bond length max)
- `score` / `rosetta_score`
- `sequence_similarity_high` / `sequence_similarity_low`
- `ca_rmsd` / `ca_rmsd_step01`
- `smart` (geometry -> RMSD -> score)

**JSON fields:**
- `n` (int)
- `metric` (string)
- `bond_length_tolerance`, `bond_angle_tolerance`
- `sequence_ref` (parent/step02/step01)
- `scorefunction` (optional)

**Use cases:**
- Collapse branching after a high-diversity step

---

### 3.8 Clustering (`cluster`, `keep_cluster_best`)
Cluster by sequence or structure, then optionally keep best per cluster.

**Cluster fields:**
- `method`: `sequence` or `structure`
- `n_clusters` (int)
- `threshold` (identity for sequence; RMSD for structure)
- `sequence_ref` (parent/step02/step01)

**Keep cluster best fields:**
- `n` (int) per cluster
- `metric` (same as `select_best`)
- `scorefunction` (optional)

**Use cases:**
- Preserve diversity while pruning

---

### 3.9 Scale score terms (`scale` / `scale_scoreterm`)
Adjust scorefunction weights persistently, or for next step only.

**Text syntax:**
```
scale:fa_rep=0.15:coordinate_constraint=1.0
scale:fa_rep reset
```

**JSON fields:**
- `terms` (dict of score term -> value or "reset")
- `scope`: `global` (default) or `next`

**Special terms:**
- `coordinate_constraint` = catres coordinate constraints
- `global_coord_constraint` = global backbone constraints

**Note:** `global_coord_constraint` uses the Rosetta `coordinate_constraint` term.
The global weight is applied by scaling stdevs to keep catres constraints distinct.

---

### 3.10 Set runtime defaults (`set` / `set_options`)
Update default settings for later steps.

Example:
```
set:mpnn_temperature=0.2:mpnn_num_designs=8
```

**JSON fields:**
- `options` (dict of runtime defaults)

Common options:
- `mpnn_temperature`, `mpnn_num_designs`, `mpnn_batch_size`, `mpnn_omit_aa`
- `mpnn_spheres`, `mpnn_use_sc_context`, `mpnn_pack_side_chains`

---

### 3.11 Set layer cuts (`set_layer_cuts`)
Adjust sphere boundaries mid-protocol.

**Text syntax:**
```
set_layer_cuts:6.0:8.0:12.0
```

**JSON fields:**
- `core_cutoff`, `shell_cutoff`, `flex_cutoff`

**Use cases:**
- Expand or contract design regions as protocol progresses

---

### 3.12 Keep interactions (`keep_interactions`)
Add residues to the fixed list based on detected interactions.

**Text syntax:**
```
keep_interactions:target=ligand:types=hbond,pi:prob=0.75
```

**JSON fields:**
- `target`: `catres`, `catres_subset`, `motif`, `catres_or_motif`, `ligand`
- `interaction_types`: list of types
- `probability`: float (0-1)
- `mutator_atoms`: sidechain|backbone|either
- `target_atoms`: sidechain|backbone|either
- `include_ligand_interactions` (bool)
- `include_catres_interactions` (bool)
- `strong_interaction_types` (list)
- `hbond_accept_probability` (float)

**Supported interaction types:**
- `hbond`, `pi_stack` (or `pi`), `hydrophobic`, `metal`, `charged`, `cation_pi`, `halogen`

**Use cases:**
- Preserve ligand or catres interactions probabilistically

---

### 3.13 Task operation (`task_operation`)
Call a custom Python function to add fixed residues dynamically.

**Text syntax:**
```
task_operation /path/to/my_taskop.py compute param1=foo
```

**JSON fields:**
- `module` (path to .py)
- `function` (callable name)
- `args` (dict)

The callable receives `(pdb_path, context, **kwargs)` where `context` includes
step02/step01 paths, catres/motif positions, ligand info, etc.

---

### 3.14 Time check (`time_check`)
Conditionally replace or extend the remaining protocol if runtime is too long.

**JSON example:**
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

**Fields:**
- `max_elapsed` (seconds)
- `min_remaining` (seconds)
- `max_runtime_fraction` (0-1)
- `then` (list of steps)
- `mode`: `replace_remaining` (default) or `continue`
- `target_total_designs` (optional quota helper)

---

### 3.15 Final diversify (`final_diversify`)
Multi-temperature MPNN sampling to reach a target design count.

Example:
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

**Fields:**
- `temperatures` (list)
- `target_count` (int; overridden by `--num_final_designs`)
- `designs_per_temp` (int)
- `max_iterations` (int)
- `design_scope` / `design_spheres`
- `batch_size` (int)
- `omit_aa`, `enhance`, `use_sc_context`, `pack_side_chains`, `sc_denoising_steps`
- `overshoot_threshold` (float)
- `fallback_include_flex` (bool)

**Use cases:**
- Final diversification to reach a fixed output count

---

## 4) Repack vs Minimize (Practical Difference)

- **Repack**: discrete sidechain rotamer swaps only. Backbone fixed.
  Fast, conservative, good for packing cleanup.
- **Minimize**: continuous coordinate optimization (torsional or cartesian).
  Can change bond angles/lengths and (optionally) backbone.
  Slower but improves geometry and constraint satisfaction.

A common pattern is: **MPNN -> repack -> minimize**.

---

## 5) Logs and Outputs

- Rosetta logs: `output_dir/logs/rosetta_*.out` and `output_dir/logs/rosetta_*.err`
- MPNN logs: `mpnn.stdout` / `mpnn.stderr` inside each `mpnn_*` folder

---

## 6) Common Pitfalls

- If no constraints are loaded, coordinate constraints are not applied.
- If a protocol ends with MPNN, geometry may be less optimized.
- Overly aggressive branching (high `num_designs` across many steps) can explode
  runtime and disk usage.

---

For built-in examples, see `protocols/` and the main `README.md`.
