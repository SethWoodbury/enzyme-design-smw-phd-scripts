# Advanced Protocol Guide (Step03 FastMPNN Design)

This document describes the **advanced protocol system** for step03, including
new step types, legacy‑style syntax support, and per‑step hyperparameters.

The goal is to be **modular, explicit, and dynamic** while preserving the
existing fastrelax/MPNN pipeline and the MPNN output restoration flow.

---

## 1) Protocol File Formats

### 1.1 Text (.txt)
The parser supports both the **new compact syntax** and **legacy whitespace commands**.
You can mix styles in the same file.

**Compact style (current syntax):**
```
mpnn:T0.2:N8:spheres=primary,secondary
cart_relax:R2S3
select_best:N1:metric=smart
```

**Legacy style (whitespace):**
```
scale:coordinate_constraint 1.0
scale:fa_rep 0.15
mpnn 0.4 8
repack
min 0.01 cartesian
keep_best 2 metric=smart
```

**Inline protocol string inside a line:**
```
mpnn:T0.2:N8 -> torsional_relax:R1S2 -> mpnn:T0.1:N16
```

### 1.2 JSON (.json)
```
{
  "steps": [
    {"type": "scale", "terms": {"coordinate_constraint": 1.0, "fa_rep": 0.15}},
    {"type": "mpnn", "temperature": 0.2, "num_designs": 8, "spheres": ["primary"]},
    {"type": "torsional_relax", "repeats": 1, "stages": 2},
    {"type": "select_best", "n": 1, "metric": "smart"}
  ]
}
```

---

## 2) Step Types (Overview)

### 2.1 MPNN
Design sequences in selected spheres. All outputs are **restored** (REMARK 666,
HIS tautomers, ligand fix, H‑addition) after each MPNN step.

**Compact syntax:**
```
mpnn:T0.2:N8:spheres=primary,secondary
```

**Legacy style:**
```
mpnn 0.2 8 spheres=primary,secondary
```

**JSON fields:**
- `temperature` (float)
- `num_designs` (int)
- `batch_size` (int)
- `spheres` (list)
- `use_sc_context` (bool)
- `pack_side_chains` (bool)
- `sc_denoising_steps` (int)
- `omit_aa` (string)
- `enhance` (string, optional)
- `repack_everything` (bool)

**Notes:**
- If `num_designs` is omitted in a step, the runtime default is used.
- If `--mpnn_num_designs_after_first` is set, later MPNN steps can auto‑reduce
  branching **unless the step explicitly sets `num_designs`**.

### 2.2 Cartesian Relax (FastRelax)
Uses FastRelax (cartesian) with fa_rep ramp and optional bond geometry minimization.

```
cart_relax:R2S3
```

JSON fields (subset):
- `repeats`, `stages`, `scorefunction`, `fa_rep_weight`
- `coord_cst_weight`, `coord_cst_stdev`
- `global_coord_cst_weight`, `global_coord_cst_stdev`
- `cart_bonded_weight`
- `relax_rounds`, `relax_inner_cycles`

### 2.3 Torsional Relax (FastRelax)
Uses FastRelax (torsional). Faster but doesn’t explicitly optimize bond geometry.

```
torsional_relax:R1S2
```

**Scorefunction override (all Rosetta steps):**
You can pass `sf=<name>` or `scorefunction=<name>` to `cart_relax`,
`torsional_relax`, `minimize`, or `repack` (e.g., `beta_jan25`, `ref2015`,
`ref2015_cart`, `beta_nov16_cart`).

### 2.4 Minimize (MinMover)
A **minimization‑only** step (no repack). Default torsional; can be cartesian.

**Compact syntax:**
```
minimize:T0.01:I200:cartesian:sf=ref2015_cart
```

**Legacy style:**
```
min 0.01 cartesian
```

JSON fields:
- `tolerance`, `max_iter`
- `cartesian` (bool)
- `scorefunction`
- `coord_cst_weight`, `coord_cst_stdev`
- `global_coord_cst_weight`, `global_coord_cst_stdev`
- `fa_rep_weight`
- `cart_bonded_weight`
- `min_backbone_rmsd_cutoff`
- `minimize_scope` / `scope` (optional): core, core_shell, core_shell_flex, global

**RMSD cutoff behavior:** If `min_backbone_rmsd_cutoff` is set, the minimizer
rejects moves that exceed that CA RMSD and reverts to the pre‑min pose.

### 2.5 Rosetta Repack Only
Sidechain repacking without design. Equivalent to legacy `repack`.

```
repack:shell=8.0
```

or legacy:
```
repack 8.0
```

**Scope-based repack (new):**
```
repack:scope=core_shell_flex
```

Notes:
- `repack_shell` (legacy) **overrides** `repack_scope` if both are provided.
- Repacking **never** includes catalytic residues (catres_subset) and other fixed residues.

### 2.6 Select / Keep Best
Select top N designs by a metric.

**Metrics supported:**
- `geometry` (bond_length max)
- `score` / `rosetta_score`
- `sequence_similarity_high` / `sequence_similarity_low`
- `ca_rmsd` / `ca_rmsd_step01`
- `smart` (multi‑criteria; see below)

**Smart metric behavior:**
1) Enforce bond/angle cutoff (if any). If none pass, choose closest to cutoffs.
2) Break ties by CA RMSD vs step01 (lower is better).
3) If still tied, break ties by Rosetta total score.

Example:
```
select_best:N1:metric=smart
```

### 2.7 Scale Score Terms
Adjust scorefunction weights **persistently** (or next‑step only) like legacy
protocols.

```
scale:coordinate_constraint 1.0
scale:fa_rep 0.15
```

or:
```
scale:fa_rep=0.15:coordinate_constraint=1.0
```

Special recognized terms:
- `coordinate_constraint` → catres coordinate constraints
- `global_coord_constraint` → global backbone coordinate constraints
- `fa_rep`
- `cart_bonded`

**Important:** `global_coord_constraint` is implemented using the same Rosetta
`coordinate_constraint` score term. The global weight is applied by scaling the
global constraint stdev to match the requested effective weight while keeping
catres constraints at their own effective weight.

All other terms are passed directly by name to Rosetta’s scorefunction.

### 2.8 Set Options
Update runtime defaults during the protocol (e.g. MPNN defaults).

```
set:mpnn_temperature=0.2:mpnn_num_designs=8
```

### 2.9 Keep Interactions
Add residues to the fixed list if they form interactions.

```
keep_interactions:target=ligand:types=hbond,pi:prob=0.75
keep_interactions:target=catres:types=hbond:prob=0.5
```

Parameters:
- `target`: `catres`, `motif`, `catres_or_motif`, `ligand`
- `types`: interaction types (`hbond`, `pi`, `metal`, `charged`, `hydrophobic`, `cation_pi`, `halogen`)
- `prob`: probability to keep
- `mutator_atoms`: `sidechain|backbone|either`
- `target_atoms`: `sidechain|backbone|either`

### 2.10 Clustering + Keep Cluster Best
Cluster by sequence or structure, then select best per cluster.

```
cluster:sequence:N2
keep_cluster_best:N1:metric=smart
```

**Note:** structure clustering falls back to sequence clustering if PyRosetta
is unavailable in‑process.

### 2.11 Task Operation (Plugin)
Load a user plugin to add fixed residues dynamically.

**Text example:**
```
task_operation /path/to/my_taskop.py compute param1=foo
```

Your function should return a list of residue IDs (e.g. `A10`) or a dict mapping
PDB path → list. It receives `(pdb_path, context, **kwargs)` where `context`
contains step02/step01 paths, catres/motif positions, ligand info, etc.

### 2.12 Time Check (Conditional Branch)
Conditionally replace the remaining protocol if runtime is too long.

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

Notes:
- If `target_total_designs` is set, the **first** MPNN step in `then` without
  `num_designs` is filled to hit the target quota.
- `mode` can be `"replace_remaining"` (default) or `"continue"`.

### 2.13 Logs
- **Rosetta subprocess logs:** `output_dir/logs/rosetta_*.out` and `output_dir/logs/rosetta_*.err`
- **MPNN logs:** `mpnn.stdout` / `mpnn.stderr` inside each `mpnn_*` output folder

---

## 3) Examples

**Legacy‑style fast protocol:**
```
scale:coordinate_constraint 1.0
scale:fa_rep 0.15
mpnn 0.4 8
repack
min 0.01
scale:coordinate_constraint 0.5
scale:fa_rep 0.36
mpnn 0.2 4
repack
min 0.01
```

**Modern protocol with interaction keepers:**
```
mpnn:T0.2:N8:spheres=primary
keep_interactions:target=ligand:types=hbond,pi:prob=0.75
torsional_relax:R1S2
mpnn:T0.1:N8:spheres=primary,secondary
keep_interactions:target=catres:types=hbond:prob=0.5
torsional_relax:R2S3
select_best:N1:metric=smart
```

---

## 4) Notes on Performance

- MPNN uses `universal.sif`; Rosetta uses `pyrosetta.sif` by default.
- `--rosetta_in_process` can skip container overhead **if PyRosetta is installed**
  on the host (optional).
- Minimization‑only steps are faster than FastRelax and avoid repacking unless
  you explicitly add a repack step.

---

## 5) Common Pitfalls

- If no constraint file is loaded, coordinate constraints will not be applied.
- If a protocol ends with MPNN (no relax), the pipeline still restores REMARKs
  and hydrogens, but geometry may be less optimized.
