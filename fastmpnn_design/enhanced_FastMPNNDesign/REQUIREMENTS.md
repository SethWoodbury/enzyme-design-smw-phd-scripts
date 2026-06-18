# Enhanced FastMPNN Design - Requirements & Configuration Reference

This document contains all dependencies, paths, environment requirements, and configuration options needed to run or transfer this pipeline to another system.

---

## 1. Python Dependencies

### Required Packages
```
pyrosetta          # PyRosetta (requires license from RosettaCommons)
pandas             # Data manipulation
numpy              # Numerical operations
```

### Standard Library (no install needed)
```
sys, os, pathlib, typing, logging, subprocess, shlex, json
dataclasses, enum, copy, importlib.util
```

### PyRosetta Installation
PyRosetta requires a license from RosettaCommons. Once licensed:
```bash
# Typical conda install
conda install -c rosettacommons pyrosetta

# Or pip wheel (if provided)
pip install pyrosetta-<version>.whl
```

---

## 2. External Software Paths

These paths are configured in `constants.py` and can be overridden via CLI.

| Software | Default Path | Purpose | CLI Override |
|----------|--------------|---------|--------------|
| **MPNN Runner** | `/net/software/lab/fused_mpnn/seth_temp/run.py` | LigandMPNN/EnhancedMPNN | `--mpnn_runner` |
| **DAlphaBall** | `/net/software/lab/scripts/enzyme_design/DAlphaBall.gcc` | Packing hole detection | N/A (init option) |
| **Apptainer Image** | `/software/containers/universal.sif` | Container for MPNN | `--apptainer_image` |
| **design_utils** | `/software/scripts/enzyme_design/utils` | Design helper functions | Hardcoded |
| **FastMPNNDesign** | `/home/woodbuse/special_scripts/fastmpnn_design/FastMPNNDesign` | MPNN Python API | Hardcoded |

### To Transfer to Another System
Edit `constants.py` lines 16-24 and 177-181:
```python
DEFAULT_MPNN_RUNNER = "/your/path/to/mpnn/run.py"
DEFAULT_DALPHABALL = "/your/path/to/DAlphaBall.gcc"
DEFAULT_APPTAINER_IMAGE = "/your/path/to/container.sif"
FASTMPNN_DESIGN_PATH = "/your/path/to/FastMPNNDesign"
DESIGN_UTILS_PATH = "/your/path/to/design_utils"
```

---

## 3. PyRosetta Scorefunction Configuration

### Available Scorefunctions

| Name | Description | Init Flag Required |
|------|-------------|-------------------|
| `beta_jan25` | **Default** - January 2025 beta | `-beta_jan25` |
| `beta_nov16` | November 2016 beta | `-beta_nov16` |
| `beta_july15` | July 2015 beta | `-beta_july15` |
| `beta_nov15` | November 2015 beta | `-beta_nov15` |
| `beta` | Generic beta | `-beta` |
| `ref2015` | Standard Rosetta 2015 | None |

### How Beta Scorefunctions Work

Beta scorefunctions require a **two-step process**:

```python
# Step 1: Initialize PyRosetta with the beta flag
pyrosetta.init("-beta_jan25 -extra_res_fa ligand.params ...")

# Step 2: Get the scorefunction (uses weights from init)
sfx = pyrosetta.get_fa_scorefxn()  # Returns beta_jan25 scorefunction
```

**Important:** You cannot create a beta scorefunction directly with `create_score_function("beta_jan25")`. The flag must be passed at initialization time.

### PyRosetta Initialization Options

Full options string built by `rosetta_init.py`:
```
-beta_jan25                              # Scorefunction (if beta)
-extra_res_fa /path/to/ligand.params     # Ligand parameters
-dalphaball /path/to/DAlphaBall.gcc      # Hole detection
-run:preserve_header                      # Keep REMARK lines in output
-multithreading true                      # Enable threading
-multithreading:total_threads N           # Thread count
-multithreading:interaction_graph_threads N
-mute all                                 # Quiet mode (optional)
```

---

## 4. Energy Terms Reference

### Constraint Weights (applied when cstfile provided)

| Term | Default Weight | Purpose |
|------|---------------|---------|
| `atom_pair_constraint` | 1.0 | Distance constraints between atoms |
| `angle_constraint` | 1.0 | Angle constraints |
| `dihedral_constraint` | 1.0 | Torsion angle constraints |
| `coordinate_constraint` | Variable | Backbone position restraints |

### Cartesian Relaxation Weights

| Term | Default Weight | Purpose |
|------|---------------|---------|
| `cart_bonded` | 0.5 | Bond geometry in Cartesian space |
| `pro_close` | 0.0 | Proline ring closure (disabled) |

### Key Scoring Terms in Protocol

| Term | Role in Protocol |
|------|-----------------|
| `fa_rep` | Scaled from 0.15 → 1.0 during design (clash tolerance) |
| `fa_atr` | Attractive van der Waals |
| `fa_sol` | Solvation energy |
| `fa_elec` | Electrostatics |
| `hbond_*` | Hydrogen bonding terms |

### Protocol Weight Scaling

The protocol scales weights progressively:
```
Round 1: fa_rep=0.150, coord_cst=1.0  (explore broadly)
Round 2: fa_rep=0.365, coord_cst=0.5  (medium)
Round 3: fa_rep=0.659, coord_cst=0.0  (strict)
Round 4: fa_rep=1.000, coord_cst=0.0  (final)
```

---

## 5. Environment Variables

| Variable | Purpose | Detection |
|----------|---------|-----------|
| `OMP_NUM_THREADS` | Thread count for PyRosetta | Primary |
| `SLURM_CPUS_ON_NODE` | SLURM job CPU allocation | Fallback |

If neither is set, defaults to `os.cpu_count()`.

---

## 6. Input File Requirements

### PDB File
Must contain **REMARK 666** lines from Rosetta Matcher defining catalytic residues:
```
REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A PHE  150  10  1
           ↑              ↑ ↑    ↑             ↑ ↑    ↑    ↑   ↑
           keyword        │ │    │             │ │    │    │   └─ constraint variant
                          │ │    │             │ │    │    └──── constraint block (catres index)
                          │ │    │             │ │    └───────── residue number
                          │ │    │             │ └────────────── residue name
                          │ │    │             └──────────────── chain
                          │ │    └────────────────────────────── ligand resnum
                          │ └─────────────────────────────────── ligand name
                          └───────────────────────────────────── ligand chain
```

### Params Files (`.params`)
Rosetta parameter files for:
- Ligand molecules (e.g., `ZRE.params`)
- Non-canonical amino acids

### Constraint Files (`.cst`)
Rosetta enzyme design constraint files defining:
- Distance constraints
- Angle constraints
- Torsion constraints

---

## 7. CLI Options Summary

### Required
```
--pdb PATH                     Input PDB file with REMARK 666 lines
```

### Structure Options
```
--nstruct N                    Number of design iterations (default: 1)
--prefix STR                   Prefix for output filenames
--suffix STR                   Suffix for output filenames
--params PATH [PATH ...]       Ligand/NCAA params files
--cstfile PATH                 Matcher constraint file
--ref_pdb PATH                 Reference PDB for comparison
```

### Design Position Options
```
--design_pos N [N ...]         Specific positions to redesign
--keep_pos N [N ...]           Positions to keep fixed
--detect_pocket                Auto-detect designable pocket
```

### Catalytic Residue Options
```
--catres_subset 1,2,6          Which catres get tight constraints
--redesign_non_subset_catres   Allow redesigning non-subset catres
```

### MPNN Options
```
--mpnn_runner PATH             MPNN runner script path
--model_type STR               Model type (default: ligand_mpnn)
--enhance MODEL                EnhancedMPNN model name
--no_enhance                   Disable EnhancedMPNN
--temperature FLOAT            Sampling temperature (default: 0.3)
--number_of_batches N          Batch count (default: 10)
--batch_size N                 Samples per batch (default: 1)
--pack_side_chains {0,1}       Enable SC packing (default: 1)
--sc_num_denoising_steps N     Denoising steps (default: 3)
--omit_AA STR                  Omit amino acids (default: CM)
--repack_everything {0,1}      Repack all positions
--use_apptainer                Run MPNN in container
--apptainer_image PATH         Container image path
--ligand_mpnn_use_side_chain_context {0,1}  Use SC context (default: 1)
```

### Scoring & Protocol
```
--scorefunction NAME           Rosetta scorefunction (default: beta_jan25)
--protocol PATH                Custom protocol file
--scoring PATH                 Custom scoring module
--filter                       Enable filtering
--output_dir PATH              Output directory
```

### Debug Options
```
--verbose                      Verbose logging
--quiet                        Suppress output
--dry_run                      Print commands without executing
```

---

## 8. Default Protocol

The default protocol (from `constants.py`) progressively tightens constraints:

```
# Round 1: Explore broadly
scale:coordinate_constraint 1.0
scale:fa_rep 0.150
mpnn 0.3 10              # T=0.3, 10 sequences
repack
scale:fa_rep 0.200
min 0.01
keep_best 5
task_operation keep_hbonds_to_ligand_and_catres

# Round 2: Medium stringency
scale:coordinate_constraint 0.5
scale:fa_rep 0.365
mpnn 0.2 2               # T=0.2, 2 sequences
repack
keep_best 5
scale:fa_rep 0.480
min 0.01
task_operation keep_hbonds_to_ligand_and_catres

# Round 3: Strict
scale:coordinate_constraint 0.0
scale:fa_rep 0.659
mpnn 0.1 2               # T=0.1, 2 sequences
repack
keep_best 5
scale:fa_rep 0.750
min 0.01
task_operation keep_hbonds_to_ligand_and_catres

# Round 4: Final refinement
scale:coordinate_constraint 0.0
scale:fa_rep 1
mpnn 0.1 2
repack
min 0.00001
keep_best 8
```

---

## 9. Output Structure

```
<output_dir>/
├── scores/
│   └── <pdbname><suffix>.sc     # Tab-separated scorefile
├── seqs/                         # 2nd layer MPNN outputs (if --mpnn)
└── <pdbname><suffix>_<iter>_<i>.pdb  # Designed structures
```

---

## 10. Quick Start Checklist

To transfer this pipeline to another system:

1. **Install PyRosetta** (requires license)
2. **Install Python packages**: `pip install pandas numpy`
3. **Obtain/install LigandMPNN** and note the path
4. **Obtain DAlphaBall** executable
5. **Update paths in `constants.py`**:
   - `DEFAULT_MPNN_RUNNER`
   - `DEFAULT_DALPHABALL`
   - `DESIGN_UTILS_PATH`
   - `FASTMPNN_DESIGN_PATH`
6. **Prepare input files**:
   - PDB with REMARK 666 lines
   - Ligand params file
   - Constraint file (optional)
7. **Run**:
   ```bash
   python cli.py --pdb input.pdb --params ligand.params --cstfile constraints.cst
   ```

---

## 11. Container Usage

If using Apptainer/Singularity:
```bash
apptainer exec /path/to/container.sif python /path/to/cli.py \
    --pdb input.pdb \
    --params ligand.params \
    --use_apptainer \
    --apptainer_image /path/to/container.sif
```

The `--use_apptainer` flag wraps MPNN calls in the container, while the outer Python runs in the same container.
