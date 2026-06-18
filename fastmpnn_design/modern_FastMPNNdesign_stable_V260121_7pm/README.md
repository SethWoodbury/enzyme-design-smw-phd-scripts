# FastMPNN Enzyme Design Pipeline (Modern)

This repository contains a production-oriented enzyme design pipeline that integrates
LigandMPNN/ProteinMPNN sequence design with PyRosetta-based structure optimization and
constraint-driven scoring. The pipeline is optimized for metal/cofactor systems using
matcher-style CST files and REMARK 666 annotations.

This README is intentionally detailed and opinionated. It is meant to be the single
reference for:
- What the pipeline does and why.
- Which files control each stage.
- Which arguments are required vs optional.
- Where outputs go and how to interpret them.
- Where the pipeline still needs improvement.

---

## Quick Start

Minimal (required input only):
```bash
python enzyme_design.py \
  --pdb /path/to/input_with_ligand_and_REMARK_666.pdb
```

Typical enzyme run (recommended inputs):
```bash
python enzyme_design.py \
  --pdb /path/to/input.pdb \
  --params /path/to/ligand.params \
  --cstfile /path/to/constraints.cst \
  --ref_pdb /path/to/reference.pdb \
  --catres_cst_subset 1,2,3,4 \
  --nstruct 5 \
  --debug
```

---

## Directory Structure (Key Files)

```
modern_FastMPNNdesign/
├── enzyme_design.py        # Main entry point + CLI
├── design_protocol.py      # FastMPNNdesign class, protocol engine
├── rosetta_utils.py        # PyRosetta utilities, flips, constraints, RMSD
├── ref_pdb_utils.py        # Ref PDB alignment + metric derivation
├── default_scoring.py      # Default scoring and metrics
├── constants.py            # Default protocol + defaults
├── hbond_selectors.py      # H-bond selection logic
├── mpnn_runner.py          # Local MPNN runner (in-process)
├── mpnn_server.py          # Optional MPNN server mode
└── common_bugs_and_cautions.md
```

---

## Required Inputs

### Required
- `--pdb`: Input PDB with ligand and REMARK 666 lines.
  - The pipeline expects the ligand to be the last residue.
  - REMARK 666 lines describe catalytic residues and constraint blocks.

### Strongly Recommended
- `--params`: Rosetta params file(s) for ligand and any non-canonical residues.
- `--cstfile`: Enzdes constraints file (CST) for catalytic geometry.
- `--ref_pdb`: Reference PDB with ideal ligand + catalytic residue geometry.
  - Used to align and compute RMSD metrics.
  - Can also be used to derive constraints if no CST is provided.

---

## Pipeline Flow (High-Level)

The pipeline runs per-design iteration and follows this flow:

1) **Argument parsing + environment checks**
   - `enzyme_design.py` checks if a container is required for the chosen scorefunction.
   - If so, it re-executes itself in the PyRosetta container.

2) **Input setup**
   - Load PDB, ligand params, REMARK 666 entries.
   - Identify catalytic residues and prepare design positions.
   - Optionally align to `--ref_pdb`.

3) **Constraint setup**
   - If `--cstfile` is provided, initialize EnzConstraintIO and apply constraints.
   - If no `--cstfile`, constraints can be derived from `--ref_pdb`.

4) **Pre-relaxation (FastRelax)**
   - Cartesian relax by default (better for ligand systems).
   - Optionally torsion-space if `--no_cartesian_relax` is passed.

5) **FastMPNN protocol execution**
   - Controlled by `--protocol` or `constants.DEFAULT_PROTOCOL`.
   - MPNN design steps interleaved with repack/min and keep-best stages.

6) **Automatic sidechain flip gating**
   - Flips are allowed only after a stage gate.
   - Requires chi convergence + persistent CST violation.
   - Requires torsion_AB deviation above a threshold (default 120 degrees).

7) **Scoring + output**
   - Writes PDBs for final designs.
   - Writes scorefile in `scores/`.
   - If `--debug`, writes intermediate structures and JSON pipeline report.

---

## Protocol System

The protocol is a simple line-based script (see `constants.DEFAULT_PROTOCOL` or custom
`--protocol`). Each command modifies the pose or filters the pool.

Common commands:
- `mpnn [temperature] [num_sequences]`: Run LigandMPNN/ProteinMPNN.
- `repack`: Rosetta PackRotamers on selected residues.
- `min`: Rosetta minimization (Cartesian or torsion).
- `keep_best [N]`: Keep best N designs by score (mode configurable).
- `2nd_shell_mpnn`: Run MPNN on second shell residues.
- `flip_catres [threshold] [max_iter]`: Force flip attempts for catres.

The protocol determines how many sequences are generated per iteration; the pipeline
estimates total outputs before starting (printed in the log).

---

## Rosetta Details (Core Logic)

### Scorefunction
- Default: `ref2015` (beta_nov16).
- Supports `ref2015_cart`, `talaris2014`, `beta_jan25`, `beta_july15`, etc.
- Constraint weights are set from `--constraint_weight` (default 1.0).

### Constraints
- Enzyme constraints are added via EnzConstraintIO using the CST file.
- Constraint score terms:
  - `atom_pair_constraint`
  - `angle_constraint`
  - `dihedral_constraint`

### Coordinate Constraints (optional)
- `--ref_coord_cst` and `--adaptive_coord_cst` allow 3D positional constraints
  on functional groups and adaptive weakening when geometry is off.

### Relaxation
- Cartesian relax is default (better stability with ligands).
- Use `--no_cartesian_relax` for torsion-space.

---

## Sidechain Flip Logic (Current Design)

Flips are intended to escape local minima for symmetric sidechains. The current
logic enforces strict gating:

**Gates (all must pass):**
1) **Stage gate**: no flips before protocol step N (`flip_stage_gate`, default 5).
2) **Chi convergence**: chi angles stable across recent steps.
3) **Persistent CST violation**: CST score above threshold for several steps.
4) **torsion_AB deviation**: the dihedral involving ligand + sidechain is far off
   (default >= 120 degrees).

**Flip decision:**
- Only accepted if CST improves meaningfully, or similar CST but total score improves.
- Rejected flips are reverted with local minimization.

All residues in `FLIPPABLE_SIDECHAINS` remain eligible. Chemical sensitivity
logic has been removed.

---

## Command-Line Arguments (Required vs Optional)

### Required
- `--pdb` (str): Input PDB with ligand + REMARK 666.

### Output / Control
- `--nstruct` (int): Number of iterations.
- `--suffix` (str): Output file suffix.
- `--outdir` (str): Output directory root (default: current working directory).

### Structure + Constraints
- `--params` (str, list): Ligand/NCAA params.
- `--cstfile` (str): CST file.
- `--ref_pdb` (str): Reference PDB for alignment and metrics.
- `--catres_cst_subset` (str): Comma-separated CST indices to apply.
- `--ref_coord_cst` (flag): Add 3D coord constraints from ref.

### Design Positions
- `--design_pos` (int list): Positions to redesign.
- `--keep_pos` (int list): Positions to keep fixed (repack only).
- `--global_seq_redesign` (flag): Redesign full sequence.

### Layer Distance Cutoffs
- `--layer_design_inner` / `--layer_design_outer`
- `--layer_repack_inner` / `--layer_repack_outer`

### Protocol + Scoring
- `--protocol` (str): Custom protocol file.
- `--scoring` (str): Custom scoring module.
- `--filter` (flag): Only output filtered designs.
- `--keep_best_mode` (str): `cst_priority`, `total_score`, `cst_only`.
- `--cst_comparable_threshold` (float): Threshold for `cst_priority`.

### Biasing
- `--position_bias` (float)
- `--bias_atoms` (str list)
- `--bias_AAs` (str)

### Ligand Rigidity
- `--ligand_rigidity` (str): `fixed`, `rigid_body`, `flexible`.

### Scorefunction / Relax
- `--scorefunction` (str)
- `--constraint_weight` (float)
- `--cart_bonded_weight` (float)
- `--no_cartesian_relax` (flag)

### Adaptive Coordinate Constraints
- `--adaptive_coord_cst` (flag)
- `--cst_deviation_threshold` (float)
- `--coord_cst_neighbor_window` (int)
- `--reduced_coord_cst_weight` (float)

### MPNN Configuration
- `--mpnn_runner` (str)
- `--mpnn_model_type` (str)
- `--mpnn_omit_aa` (str)
- `--apptainer_image` (str)
- `--mpnn_use_sc_context` (int)
- `--mpnn_pack_side_chains` (int)
- `--mpnn_repack_everything` (int)
- `--no_mpnn_cache` (flag)

### MPNN Server
- `--mpnn_server` (flag)
- `--mpnn_server_host` (str)
- `--mpnn_server_port` (int)

### Debug / Test
- `--debug` (flag)
- `--quick_test` (flag)

---

## Output Layout

### Standard output
- `scores/` directory with `.sc` file per run.
- Final design PDBs in the working directory or under `--outdir` if specified.

### Debug mode (`--debug`)
Creates a directory:
```
<outdir>/debug_output_<pdbname>/
  full_log.txt
  pose_0000_pipeline_report.json
  structures/
    stepXX_*.pdb
    mpnn_raw_packed_*.pdb
```

The tracker JSON is written to the parent debug dir alongside `full_log.txt`.

---

## Metrics and Scorefile Columns

The default scoring module writes a wide set of columns. Key groups:

### Rosetta Energy Terms
- `total_score`, `fa_atr`, `fa_rep`, `fa_sol`, `fa_elec`, `hbond_*`, etc.

### Constraint Metrics
- `all_cst`: total constraint score (sum of constraint terms).

### Ligand / Interface Metrics
- `corrected_ddg`: interaction ddG.
- `L_SASA`: relative ligand SASA.
- `L_SASA_abs`: absolute ligand SASA.
- `substrate_SASA`: ligand SASA excluding metals.
- `total_acceptor_hbonds`, `total_donor_hbonds`, `total_hbonds`.
- `sc`: shape complementarity.
- `cms`: contact molecular surface.
- `cms_per_atom`.

### No-Ligand-Repack (NLR)
- `nlr_totrms`: RMSD after removing ligand and repacking.
- Additional NLR terms are included from `rosetta_utils.no_ligand_repack()`.

### RMSD / Reference Metrics
- `CA_rmsd`, `CA_rmsd_converge`, `sequence_identity`.
- `lig_rmsd_to_refpdb`.
- `catres_subset_allatom_sc_rmsd_to_refpdb`.
- `catres_subset_allatom_bb_rmsd_to_refpdb`.
- `catres_subset_interact_rmsd_to_refpdb`.
- `ca_rmsd_to_refpdb`, `ca_rmsd_converge_to_refpdb`.

---

## Where to Configure Defaults

Defaults live in `constants.py`:
- `DEFAULT_PROTOCOL`
- `DEFAULT_SCOREFUNCTION`
- `DEFAULT_LAYER_CUTS`
- `DEFAULT_LIGAND_RIGIDITY`
- `DEFAULT_CART_BONDED_WEIGHT`
- MPNN defaults and bias defaults

---

## Common Failure Modes and Debugging

See `common_bugs_and_cautions.md` for known issues.

Recent fixes include:
- Constraint scoring must use the configured scorefunction (not a new default).
- Debug structure output directories are now correctly separated from tracker output.

---

## Future Improvements (Known Gaps)

1) **SASA calculation compatibility**
   - Current `SasaCalc.get_atom_sasa` can fail with some PyRosetta builds.
   - Should be hardened using a version-safe SASA API.

2) **More explicit torsion_AB mapping**
   - Current torsion_AB gate detects dihedral constraints in the pose.
   - If CST parsing exposes atom maps, this can be further refined.

3) **Output filtering and ranking**
   - More explicit multi-objective ranking combining CST, ddG, RMSD, etc.

4) **Protocol validation**
   - Validate custom protocol files for unknown commands or malformed inputs.

---

## Where to Look for Specific Logic

- **Argument parsing**: `enzyme_design.py:843+`
- **Protocol execution**: `design_protocol.py`
- **Flip gating and torsion_AB**: `design_protocol.py`, `rosetta_utils.py`
- **Constraint setup**: `enzyme_design.py`, `rosetta_utils.py`
- **Ref PDB metrics**: `ref_pdb_utils.py`
- **Scoring**: `default_scoring.py`

---

## Example: Full Command with Explanation

```bash
python enzyme_design.py \
  --pdb input.pdb \
  --params lig.params \
  --cstfile enzyme.cst \
  --ref_pdb reference.pdb \
  --catres_cst_subset 1,2,3 \
  --nstruct 10 \
  --keep_best_mode cst_priority \
  --constraint_weight 1.0 \
  --ligand_rigidity fixed \
  --debug
```

- `--pdb` provides ligand + REMARK 666.
- `--params` loads ligand.
- `--cstfile` applies enzyme constraints.
- `--ref_pdb` allows alignment and RMSD metrics.
- `--keep_best_mode cst_priority` prioritizes constraint satisfaction.
- `--debug` writes full log + debug structures + tracker report.

---

## Contact and Maintenance

This pipeline is designed for iterative enzyme design work. Keep the `README.md`
updated as new logic or constraints are added so collaborators can understand
the full context.
