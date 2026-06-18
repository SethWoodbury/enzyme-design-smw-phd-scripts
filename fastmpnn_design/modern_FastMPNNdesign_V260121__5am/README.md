# Enhanced FastMPNN Design

A modular protein design pipeline using FastMPNN (ProteinMPNN/LigandMPNN) integrated with PyRosetta for enzyme design. This package enables automated sequence design with constraint handling, position biasing, and multi-layer design protocols.

---

## Directory Structure

```
modern_FastMPNNdesign/
├── enzyme_design.py        # Main executable script
├── design_protocol.py      # Core MPNN+Rosetta design protocol class
├── mpnn_runner.py          # MPNN execution wrapper
├── rosetta_utils.py        # Consolidated PyRosetta utilities
├── ref_pdb_utils.py        # Reference PDB alignment, constraint derivation, metrics
├── default_scoring.py      # Default scoring with auto H-bond detection
├── hbond_selectors.py      # H-bond residue selectors
├── constants.py            # Centralized defaults and configuration (incl. DEFAULT_PROTOCOL)
└── README.md
```

---

## Overview

This pipeline combines:
- **ProteinMPNN/LigandMPNN**: Neural network-based sequence design
- **PyRosetta**: Structure refinement, energy minimization, and scoring
- **Enzyme constraints**: Maintain catalytic geometry during design
- **Automated scoring**: Default scoring with auto-detection of polar ligand atoms

---

## PyRosetta Integration

### Initialization

PyRosetta is initialized with correction flags appropriate for the chosen scorefunction:

```python
# Default (ref2015):
pyr.init("-extra_res_fa ligand.params -dalphaball DAlphaBall.gcc -beta_nov16 ...")

# With beta_jan25 scorefunction:
pyr.init("-extra_res_fa ligand.params -dalphaball DAlphaBall.gcc -corrections:beta_jan25 ...")
```

**Key flags:**
| Flag | Purpose |
|------|---------|
| `-beta_nov16` | Use REF2015 beta November 2016 energy weights (default) |
| `-corrections:beta_jan25` | Use January 2025 beta weights (for `--scorefunction beta_jan25`) |
| `-corrections:beta_july15` | Use July 2015 beta weights (for `--scorefunction beta_july15`) |
| `-extra_res_fa` | Load ligand/NCAA parameter files |
| `-dalphaball` | Path to DAlphaBall for SASA calculations |
| `-run:preserve_header` | Preserve PDB REMARK lines (matcher constraints) |
| `-multithreading` | Enable parallel packing/minimization |

The appropriate correction flag is automatically selected based on the `--scorefunction` argument.

### Scorefunction

The pipeline uses PyRosetta's **REF2015 (beta_nov16)** full-atom scorefunction by default, which is the standard Rosetta energy function for protein design.

```python
sfx = pyr.get_fa_scorefxn()  # Returns REF2015 with beta_nov16 weights
```

**Available Scorefunctions:**
| Name | Description |
|------|-------------|
| `ref2015` | REF2015 scorefunction (DEFAULT) |
| `ref2015_cart` | REF2015 optimized for Cartesian minimization |
| `beta_nov16` | Explicit beta_nov16 weights |
| `talaris2014` | Older Talaris scorefunction |
| `score12` | Legacy score12 |
| Custom path | Any Rosetta weights file |

**Beta Scorefunctions (Newer):**
| Name | Init Flag | Description |
|------|-----------|-------------|
| `beta_jan25` | `-corrections:beta_jan25` | January 2025 beta weights (newest) |
| `beta_nov16` | `-beta_nov16` | November 2016 beta weights |
| `beta_july15` | `-corrections:beta_july15` | July 2015 beta weights |

Beta scorefunctions require special `-corrections:` flags during PyRosetta initialization. The pipeline automatically handles this when you specify a beta scorefunction via `--scorefunction`.

**Container Requirements:**
Some newer beta scorefunctions (like `beta_jan25`) require a newer PyRosetta version only available in a container. The pipeline handles this automatically:

1. When you specify `--scorefunction beta_jan25`, the script detects this requires a container
2. If not already in a container, it automatically re-executes itself inside `pyrosetta.sif`
3. MPNN calls are handled with fallback logic for nested containers

**Usage:**
```bash
# Default REF2015
python enzyme_design.py --pdb input.pdb

# Use different scorefunction
python enzyme_design.py --pdb input.pdb --scorefunction talaris2014

# Use newer beta scorefunction (container auto-detected and used)
python enzyme_design.py --pdb input.pdb --scorefunction beta_jan25

# Use custom weights file
python enzyme_design.py --pdb input.pdb --scorefunction /path/to/my_weights.wts
```

**Note on Nested Containers:**
When using `beta_jan25` with `--2nd-shell-mpnn-seqs` (secondary MPNN design to seqs/ folder), the script attempts nested container execution. If this fails, it will:
1. Try the primary MPNN container (universal.sif)
2. Fall back to the PyRosetta container (pyrosetta.sif)
3. Print detailed error messages with solutions if both fail

**REF2015 Energy Terms:**
| Term | Description |
|------|-------------|
| `fa_atr` | Lennard-Jones attractive |
| `fa_rep` | Lennard-Jones repulsive |
| `fa_sol` | Lazaridis-Karplus solvation |
| `fa_elec` | Coulombic electrostatics |
| `hbond_*` | Hydrogen bonding (bb-bb, bb-sc, sc-sc) |
| `rama_prepro` | Ramachandran preferences |
| `omega` | Peptide bond planarity |
| `fa_dun` | Dunbrack rotamer probability |
| `p_aa_pp` | Probability of amino acid given phi/psi |
| `ref` | Reference energies per amino acid |

**Constraint Terms (added for enzyme design):**
```python
sfx.set_weight(atom_pair_constraint, 1.0)  # Distance constraints
sfx.set_weight(angle_constraint, 1.0)       # Angle constraints
sfx.set_weight(dihedral_constraint, 1.0)    # Dihedral constraints
```

These constraint weights enforce catalytic geometry defined in the `.cst` file. The weight can be adjusted:
```bash
python enzyme_design.py --pdb input.pdb --constraint_weight 0.5  # Softer constraints
```

### Pre-Relaxation (FastRelax)

Before design, the structure undergoes **Cartesian FastRelax** by default to resolve clashes and optimize geometry:

```python
# Cartesian-specific weights
sfx.set_weight(cart_bonded, 0.5)  # Bonded geometry in Cartesian space
sfx.set_weight(pro_close, 0.0)    # Disable (incompatible with Cartesian)

fastRelax = setup_fastrelax(sfx, crude=True)
fastRelax.cartesian(True)
fastRelax.apply(pose)
```

**Cartesian vs Torsion-Space Relaxation:**
| Mode | Command | Description |
|------|---------|-------------|
| Cartesian (default) | `--cart_bonded_weight 0.5` | Moves atoms in x,y,z space |
| Torsion | `--no_cartesian_relax` | Moves through torsion angles |

**Why Cartesian by default?** Cartesian minimization is more stable for structures with ligands and constraints because it doesn't propagate errors through bond geometry.

**Customizing Cartesian weights:**
```bash
# Adjust cart_bonded weight (default: 0.5)
python enzyme_design.py --pdb input.pdb --cart_bonded_weight 0.3

# Use torsion-space instead
python enzyme_design.py --pdb input.pdb --no_cartesian_relax
```

**Clash handling:** Residues clashing with the ligand are mutated to ALA before relaxation, then restored during MPNN design.

### Packer (Side-Chain Repacking)

The **PackRotamersMover** optimizes side-chain rotamers using Rosetta's rotamer library:

```python
packer = PackRotamersMover()
packer.task_factory(taskfactory)  # Controls which residues can repack
packer.apply(pose)
```

**TaskFactory Operations:**
| Operation | Purpose |
|-----------|---------|
| `InitializeFromCommandline` | Respect command-line packing options |
| `IncludeCurrent` | Include current rotamer as option |
| `NoRepackDisulfides` | Don't break disulfide bonds |
| `RestrictToRepacking` | Residues can repack but not mutate |
| `PreventRepackingRLT` | Residues are completely fixed |

### Minimizer

The **MinMover** performs gradient-based energy minimization:

```python
min_mover = MinMover()
min_mover.set_type("lbfgs_armijo_nonmonotone")  # L-BFGS algorithm
min_mover.cartesian(False)  # Torsion-space minimization
min_mover.set_movemap(movemap)
min_mover.apply(pose)
```

**MoveMap Settings:**
```python
movemap.set_chi(True)   # Allow chi angle changes (side-chains)
movemap.set_bb(True)    # Allow backbone phi/psi changes
movemap.set_jump(True)  # Allow rigid-body movements (ligand)
```

**RMSD Cutoff:** If backbone moves >3.0Å during minimization, the structure is rejected (prevents unrealistic conformational changes).

### Design Protocol Flow

The protocol alternates between MPNN sequence design and Rosetta refinement:

```
┌─────────────────────────────────────────────────────────────┐
│  1. scale:fa_rep 0.15         ← Reduce repulsion (soft)     │
│  2. mpnn 0.3 10               ← MPNN at T=0.3, 10 seqs      │
│  3. repack                    ← Rosetta side-chain packing  │
│  4. scale:fa_rep 0.2          ← Slightly increase repulsion │
│  5. min 0.01                  ← Minimize with tolerance 0.01│
│  6. keep_best 5               ← Keep top 5 by Rosetta score │
│  7. task_operation hbonds     ← Fix H-bonding residues      │
│  ...                          ← Repeat with harder fa_rep   │
│  N. scale:fa_rep 1.0          ← Full repulsion (final)      │
│  N+1. min 0.00001             ← Final tight minimization    │
│  N+2. keep_best 8             ← Return top 8 designs        │
│  N+3. 2nd_shell_mpnn 0.1 2    ← Design outer shell (T=0.1)  │
│  N+4. 2nd_shell_mpnn 0.2 2    ← Design outer shell (T=0.2)  │
└─────────────────────────────────────────────────────────────┘
```

**fa_rep Ramping:** Starting with low repulsion allows MPNN to explore sequences that might initially clash, then gradually increasing repulsion resolves clashes while preserving good sequences.

### Expected Sequence Output

At startup, the pipeline estimates and displays the expected number of output sequences based on the protocol:

```
  === Expected Output Summary ===
  MPNN steps: 4 (T0.3 x10, T0.2 x2, T0.1 x2, T0.1 x2)
  Keep best steps: 4 (5, 5, 5, 8)
  2nd shell MPNN: 2 (T0.1 x2, T0.2 x2)
  Per iteration: ~32 sequences
  Total expected: ~160 sequences (from 5 iterations)
```

This helps estimate computational requirements before running.

### Backbone/CB Clash Detection

Before pre-relaxation, the pipeline automatically detects residues whose backbone atoms (N, CA, C, O) or CB clash with ligand atoms (< 2.5Å). These residues and their ±3 neighbors receive reduced coordinate constraint weights to allow backbone movement during relaxation.

```
  [BB/CB Clash] Found 2 residues with backbone/CB clashing with ligand:
    Clashing: 45+78
    Freeing (incl. neighbors): 42+43+44+45+46+47+48+75+76+77+78+79+80+81
```

### Duplicate Sequence Removal

Output sequences are automatically deduplicated. If MPNN generates identical sequences across different poses, only unique sequences are kept:

```
  [DUPLICATE] Removed 3 duplicate sequences, 5 unique remain
```

### Keep Best Scoring

The `keep_best` command selects the best N poses at each step. You can control the scoring mode:

**Default mode (cst_priority):** Prioritizes constraint score, but falls back to total score if constraint scores are comparable (all low or small range).

**Protocol-level syntax:**
```
keep_best 5              # Use default mode (--keep_best_mode)
keep_best 5 cst          # Keep best by constraint score only
keep_best 5 total        # Keep best by total Rosetta score
keep_best 5 cst_priority # Explicitly use cst_priority mode
```

**Example protocol with mixed modes:**
```
scale:fa_rep 0.150
mpnn 0.3 10
repack
min 0.01
keep_best 5 cst          # Early: prioritize constraint satisfaction
scale:fa_rep 0.365
mpnn 0.2 2
repack
min 0.01
keep_best 5 cst_priority # Middle: balanced
scale:fa_rep 1.0
mpnn 0.1 2
repack
min 0.00001
keep_best 8 total        # Final: prioritize overall energy
```

**Output at each keep_best step:**
```
[keep_best] Scoring 10 poses (mode=cst_priority):
  Pose   Total       CST   AtomPair      Angle   Dihedral      Other
  --------------------------------------------------------------------
  0      -245.32     1.23      0.45       0.38       0.40    -246.55
  1      -242.18     8.76      5.21       2.15       1.40    -250.94
  ...
  CST   - Min: 0.82, Max: 12.45, Range: 11.63
  Total - Min: -248.12, Max: -235.67
  [keep_best] CST scores differ significantly - prioritizing constraint score
  [keep_best] Keeping poses: [0, 3, 5, 7, 9]
```

### 2nd Shell MPNN Design

The `2nd_shell_mpnn` protocol command designs the outer shell residues while keeping the inner pocket (close residues, H-bonders, catalytic) fixed. This is useful for optimizing the second sphere of the active site.

**Protocol syntax:**
```
2nd_shell_mpnn <temperature> [num_sequences]
```

**Example usage in protocol:**
```
# Final stages of design
scale:fa_rep 1.0
mpnn 0.1 2
repack
min 0.00001
keep_best 8
2nd_shell_mpnn 0.1 2    # T=0.1, 2 sequences per pose
2nd_shell_mpnn 0.2 2    # T=0.2, 2 sequences per pose
```

**What gets fixed:**
- Residues within 6Å of ligand (backbone distance)
- Residues with side-chains within 5Å of ligand
- Residues that form H-bonds to ligand (from H-bond keeper task operation)
- Catalytic residues and motif positions

**What gets designed:**
- Residues in outer shell layers (6-12Å from ligand)
- Alanine residues not in the pocket (often from clash removal)

**Default behavior:** The default protocol includes `2nd_shell_mpnn 0.1 2` and `2nd_shell_mpnn 0.2 2` at the end to generate additional sequence diversity in the outer shell.

**Note:** For saving 2nd shell designs to a separate `seqs/` folder (legacy behavior), use the `--2nd-shell-mpnn-seqs` command line flag instead.

### Constraint System

Enzyme design constraints (from `.cst` files) define:

1. **Atom pair constraints**: Distance between catalytic atoms
2. **Angle constraints**: Bond angles at catalytic center
3. **Dihedral constraints**: Torsion angles for proper orientation

Example constraint (simplified):
```
# Catalytic Ser-His-Asp triad
CST::BEGIN
  TEMPLATE:: ATOM A OG ATOM B NE2
  CONSTRAINT:: distanceAB: 2.8 0.2 100 0  # 2.8Å ± 0.2Å
CST::END
```

The `ConstraintManager` class handles adding/removing constraints from poses.

### Adaptive Coordinate Constraints

By default, all residues have equal coordinate constraint weights during pre-relaxation, which can prevent catalytic residues from moving to satisfy enzyme constraints if they start far from ideal geometry.

**The Problem:**
```
Coordinate constraint: "Stay near starting position" (all residues, weight=1.0)
Enzyme constraint:     "Maintain catalytic geometry" (catalytic residues)

If starting geometry is poor → these constraints fight each other!
```

**The Solution:** Adaptive coordinate constraints evaluate constraint satisfaction at the start and reduce coordinate constraint weights for catalytic residues (and their neighbors) that are far from their target geometry.

**Usage:**
```bash
# Enable adaptive coordinate constraints
python enzyme_design.py --pdb input.pdb --cstfile constraints.cst --adaptive_coord_cst

# Customize parameters
python enzyme_design.py --pdb input.pdb --cstfile constraints.cst \
    --adaptive_coord_cst \
    --cst_deviation_threshold 5.0 \    # Score threshold (default: 5.0 REU)
    --coord_cst_neighbor_window 3 \    # ±N neighbors to also free (default: 3)
    --reduced_coord_cst_weight 0.2     # Reduced weight (default: 0.2)
```

**How it works:**
1. Before pre-relaxation, evaluate enzyme constraint scores for each catalytic residue
2. Identify residues with scores above threshold (poorly satisfied constraints)
3. Mark those residues + neighboring ±N residues for reduced coordinate constraints
4. Apply per-residue coordinate constraint weights during FastRelax

**Output:** When enabled, you'll see detailed constraint score information:
```
  [Constraint Scores] Catalytic Residues:
  Res    Name  Total   AtomPair      Angle   Dihedral
  -------------------------------------------------------
  45     HIS    12.30       8.50       2.10       1.70
  78     ASP     0.80       0.30       0.25       0.25
  102    SER     1.20       0.70       0.30       0.20
  -------------------------------------------------------
  Mean          4.77
  Std           6.42
  Min           0.80
  Max          12.30

  [Adaptive Coord Constraints] Far-off catalytic residues (1):
    - HIS 45: score = 12.30 (threshold = 5.0)
  [Adaptive Coord Constraints] Residues with reduced coord constraints (7):
    42+43+44+45+46+47+48
```

### Ligand Rigidity

By default, all ligands are kept completely rigid during the design pipeline. This is controlled by the `--ligand_rigidity` flag:

```bash
# Default: ligands completely fixed
python enzyme_design.py --pdb input.pdb

# Allow rigid-body movement (internally rigid)
python enzyme_design.py --pdb input.pdb --ligand_rigidity rigid_body

# Full flexibility
python enzyme_design.py --pdb input.pdb --ligand_rigidity flexible
```

**How it works:**
- **Internal rigidity** is controlled by disabling chi and backbone angles for ligand residues in the MoveMap
- **Rigid-body movement** is controlled by disabling/enabling jumps for ligand residues

**MoveMap Configuration:**
```python
# For "fixed" mode:
movemap.set_chi(ligand_resno, False)   # No internal torsion changes
movemap.set_bb(ligand_resno, False)    # No backbone changes
movemap.set_jump(ligand_jump, False)   # No rigid-body movement

# For "rigid_body" mode:
movemap.set_chi(ligand_resno, False)   # No internal torsion changes
movemap.set_jump(ligand_jump, True)    # Allow rigid-body movement

# For "flexible" mode:
movemap.set_chi(ligand_resno, True)    # Allow internal changes
movemap.set_jump(ligand_jump, True)    # Allow rigid-body movement
```

**Multi-ligand systems:** When multiple ligands are present, each has its own jump in the fold tree. In `rigid_body` mode, each ligand can move independently. To make all ligands move as a single rigid unit, you would need to modify the fold tree or add inter-ligand constraints (not currently implemented).

---

## Scoring System

### Default Scoring (default_scoring.py)

The default scoring module automatically detects polar atoms on the ligand for H-bond analysis:

**Auto-detected atoms:**
- **Acceptors**: O, N, S atoms (excluding metals like ZN, MG, etc.)
- **Donors**: H atoms bonded to O, N, or S

**Calculated Metrics:**
| Metric | Description |
|--------|-------------|
| `total_score` | Rosetta total energy |
| `score_per_res` | Energy per residue |
| `corrected_ddg` | Interaction energy (corrected for covalent bonds) |
| `L_SASA` | Relative ligand SASA (bound/free) |
| `L_SASA_abs` | Absolute ligand SASA |
| `substrate_SASA` | SASA of substrate atoms |
| `hbond_X` | H-bonds to each polar atom X |
| `total_acceptor_hbonds` | Total H-bonds received by ligand |
| `total_donor_hbonds` | Total H-bonds donated by ligand |
| `sc` | Shape complementarity |
| `cms` | Contact molecular surface |
| `nlr_dE` | No-ligand-repack energy change |
| `nlr_totrms` | No-ligand-repack RMSD |

**Default Filters:**
```python
filters = {
    "L_SASA": [0.25, "<="],           # Low burial = good
    "corrected_ddg": [-25.0, "<="],   # Strong interaction
    "sc": [0.55, ">="],               # Good shape match
    "nlr_totrms": [1.0, "<="],        # Pre-organized pocket
}
```

**Additional Metrics (added automatically):**
| Metric | Description |
|--------|-------------|
| `CA_rmsd` | CA RMSD to input structure (after superposition) |
| `CA_rmsd_converge` | CA RMSD with outlier residues removed iteratively until convergence |
| `CA_rmsd_n_outliers` | Number of outlier residues removed during convergence |
| `sequence_identity` | Sequence identity to input structure (fraction 0.0-1.0) |
| `design_time_sec` | Time taken for this design iteration (seconds) |

**Reference PDB Metrics (when `--ref_pdb` provided):**
| Metric | Description |
|--------|-------------|
| `lig_rmsd_to_refpdb` | Ligand heavy-atom RMSD to reference (after alignment) |
| `catres_subset_allatom_sc_rmsd_to_refpdb` | All-atom sidechain RMSD of catalytic residues |
| `catres_subset_allatom_bb_rmsd_to_refpdb` | All-atom backbone RMSD of catalytic residues |
| `catres_subset_interact_rmsd_to_refpdb` | RMSD of functional groups interacting with ligand |
| `ca_rmsd_to_refpdb` | CA RMSD of entire structure to reference |
| `ca_rmsd_converge_to_refpdb` | Converged CA RMSD to reference (outliers removed) |

### Custom Scoring

You can provide your own scoring module via `--scoring`:

```bash
python enzyme_design.py --pdb input.pdb --scoring my_scoring.py
```

Custom scoring modules must implement:
- `score_design(pose, sfx, catres)` → DataFrame
- `filter_scores(scores)` → DataFrame
- `filters` → dict

---

## Reference PDB Feature

The `--ref_pdb` option allows you to provide a reference PDB with ideal catalytic residue and ligand positioning. This enables:

1. **Automatic Constraint Derivation**: If no `.cst` file is provided, constraints are derived from the reference structure
2. **Metrics Calculation**: Compare designs to the ideal reference geometry

### Ligand-Based Alignment

The reference PDB is aligned to the input PDB by superimposing ligand heavy atoms:

```
  [Ligand Alignment] Mobile ligand atoms: 24
  [Ligand Alignment] Target ligand atoms: 24
  [Ligand Alignment] Ligand RMSD after alignment: 0.0023 Å
```

The ligands must have matching atoms (by name). If the ligand RMSD > 0.5 Å, a warning is printed.

### Catalytic Residue Subset Selection

Use `--catres_cst_subset` to select specific REMARK 666 lines:

```bash
# Use only REMARK 666 lines 1, 3, and 5
python enzyme_design.py --pdb input.pdb --ref_pdb ideal.pdb --catres_cst_subset "1,3,5"
```

**Validation:**
- Residue types must match between input and reference (e.g., both HIS at line 1)
- If a line number exceeds available lines, a warning is printed but others are used

```
  [CatRes Subset] Input PDB REMARK 666 lines: 5
  [CatRes Subset] Ref PDB REMARK 666 lines: 5
  [CatRes Subset] Using subset indices: [1, 3, 5]
  [CatRes Subset] Valid catalytic residues: 3
    Line 1: HIS 45 (chain A)
    Line 3: GLU 123 (chain A)
    Line 5: TRP 201 (chain A)
```

### Automatic Constraint Derivation

When no `--cstfile` is provided and `--ref_pdb` is specified, constraints are derived:

**Distance Constraints:**
- Find closest sidechain/backbone atom to closest ligand atom
- Prefer non-carbon atoms (O, N, S) over carbon atoms
- Exclude hydrogen atoms

**Angle Constraints:**
- Angle1: 2 sidechain atoms + 1 ligand atom
- Angle2: 1 sidechain atom + 2 ligand atoms

**Dihedral Constraints:**
- Dihedral1: 3 sidechain atoms + 1 ligand atom
- Dihedral2: 1 sidechain atom + 3 ligand atoms
- Dihedral3: 2 sidechain atoms + 2 ligand atoms

```
  [Constraint Derivation] Ligand: ZRE 500 (chain X)
  [Constraint Derivation] Deriving constraints for 3 catalytic residues:

  Residue HIS 45 (chain A):
    Distance: HIS45:NE2 - LIG:ZN1 = 2.10 Å
    Angle1 (2SC+1L): CE1-NE2-ZN1 = 125.3°
    Angle2 (1SC+2L): NE2-ZN1-O1 = 108.5°
    Dihedral3 (2SC+2L): -45.2°
```

### Coordinate Constraints (Optional)

Add `--ref_coord_cst` to include 3D coordinate constraints for functional groups:

```bash
python enzyme_design.py --pdb input.pdb --ref_pdb ideal.pdb --ref_coord_cst
```

### Example Usage

```bash
# Basic: derive constraints from ref_pdb
python enzyme_design.py --pdb input.pdb --ref_pdb ideal_pose.pdb --params LIG.params

# With subset of catalytic residues
python enzyme_design.py --pdb input.pdb --ref_pdb ideal_pose.pdb \
    --catres_cst_subset "1,3,5" --params LIG.params

# With explicit CST file (ref_pdb used only for metrics)
python enzyme_design.py --pdb input.pdb --ref_pdb ideal_pose.pdb \
    --cstfile enzyme.cst --params LIG.params

# With coordinate constraints for functional groups
python enzyme_design.py --pdb input.pdb --ref_pdb ideal_pose.pdb \
    --ref_coord_cst --params LIG.params
```

---

## MPNN Runner (mpnn_runner.py)

A standalone module for executing MPNN via external `run.py`.

### Default Settings
```python
DEFAULT_LIGAND_MPNN_USE_SC_CONTEXT = 1  # Use side-chain context
DEFAULT_REPACK_EVERYTHING = 0           # Only repack designed residues
DEFAULT_PACK_SIDE_CHAINS = 1            # Enable SC packing
```

### Programmatic Usage
```python
from mpnn_runner import MPNNConfig, run_mpnn_from_pose

config = MPNNConfig(model_type="ligand_mpnn")
result = run_mpnn_from_pose(pose, config, "output/", "my_design",
                            temperature=0.1, fixed_residues=["A10", "A20"])
```

### Command Line
```bash
python mpnn_runner.py --pdb input.pdb --output_dir output/ --name my_design \
    --temperature 0.1 --fixed_residues A10,A20
```

---

## Rosetta Utilities (rosetta_utils.py)

Consolidated PyRosetta utilities including:

**Pose Manipulation:**
- `get_matcher_residues()` - Parse REMARK 666 lines
- `get_ligand_heavyatoms()` - Get ligand heavy atom names
- `mutate_residues()` - Mutate residues
- `thread_seq_to_pose()` - Thread sequence onto structure
- `fix_catalytic_residue_rotamers()` - Restore catalytic rotamers
- `repack()` - Repack side-chains
- `separate_protein_and_ligand()` - Translate ligand away

**Layer Selection:**
- `get_layer_selections()` - Get residue selectors by distance
- `get_residues_with_close_sc()` - Find residues with close side-chains
- `find_clashes_between_target_and_sidechains()` - Detect clashes

**Scoring:**
- `calculate_ddg()` - Calculate interaction ddG
- `getSASA()` - Calculate SASA
- `find_hbonds_to_residue_atom()` - Count H-bonds
- `no_ligand_repack()` - Run NLR analysis
- `dump_scorefile()` - Write Rosetta scorefile
- `filter_scores()` - Filter by criteria

---

## Usage

### Basic
```bash
python enzyme_design.py --pdb input.pdb --params ligand.params --cstfile constraints.cst
```

### Full Example
```bash
python enzyme_design.py \
    --pdb my_enzyme.pdb \
    --params ZRE.params \
    --cstfile esterase.cst \
    --nstruct 5 \
    --suffix v1 \
    --bias_atoms H1 \
    --filter \
    --2nd-shell-mpnn-seqs
```

### With Custom Scoring
```bash
python enzyme_design.py \
    --pdb my_enzyme.pdb \
    --params ZRE.params \
    --scoring my_custom_scoring.py \
    --filter
```

---

## Command Line Arguments

### Required
| Argument | Description |
|----------|-------------|
| `--pdb` | Input PDB with ligand and REMARK 666 matcher lines |

### Output Control
| Argument | Default | Description |
|----------|---------|-------------|
| `--nstruct` | 1 | Number of design iterations |
| `--suffix` | "" | Output filename suffix |

### Structure Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--params` | ZRE.params | Ligand/NCAA params file(s) |
| `--cstfile` | None | Enzyme constraint file (.cst) |

### Design Positions
| Argument | Default | Description |
|----------|---------|-------------|
| `--design_pos` | None | Specific positions to redesign (if not specified, uses pocket detection) |
| `--keep_pos` | None | Positions to keep (repack only) |
| `--global_seq_redesign` | False | Redesign entire sequence globally instead of just pocket residues |

### Layer Selection Cutoffs
| Argument | Default | Description |
|----------|---------|-------------|
| `--layer_design_inner` | 6.0 | Inner design layer cutoff (Å from CA to ligand) |
| `--layer_design_outer` | 8.0 | Outer design layer cutoff (Å, with CB check) |
| `--layer_repack_inner` | 10.0 | Inner repack layer cutoff (Å) |
| `--layer_repack_outer` | 12.0 | Outer repack layer cutoff (Å, with CB check) |

Distance is measured from protein CA/CB atoms to the **closest ligand heavy atom**.

### Scoring
| Argument | Default | Description |
|----------|---------|-------------|
| `--scoring` | None | Custom scoring script (uses default_scoring.py if not provided) |
| `--filter` | False | Only save passing designs |

### Biasing
| Argument | Default | Description |
|----------|---------|-------------|
| `--bias_atoms` | None | Ligand atoms for AA biasing |
| `--bias_AAs` | KREDYQWSTH | AAs to bias |
| `--position_bias` | -1.0 | Bias value (negative=disfavor) |

### Processing
| Argument | Default | Description |
|----------|---------|-------------|
| `--filter` | False | Only output designs passing filter criteria |
| `--2nd-shell-mpnn-seqs` | False | Run 2nd shell MPNN and save to seqs/ folder |

### H-Bond Keeper Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--hbond_accept_prob` | 0.75 | Probability to fix each H-bonding residue (0.0-1.0) |
| `--disable_hbond_keeper` | False | Disable automatic fixing of H-bonding residues |

### Keep Best Scoring Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--keep_best_mode` | cst_priority | Scoring mode: cst_priority, total_score, or cst_only |
| `--cst_comparable_threshold` | 2.0 | Threshold (REU) for "comparable" CST scores |

**Keep Best Modes:**
| Mode | Description |
|------|-------------|
| `cst_priority` | Prioritize constraint score; fall back to total if scores comparable (NEW DEFAULT) |
| `total_score` | Use total Rosetta score only (original behavior) |
| `cst_only` | Use constraint score only |

### Ligand Rigidity
| Argument | Default | Description |
|----------|---------|-------------|
| `--ligand_rigidity` | fixed | Ligand rigidity mode (see below) |

**Ligand Rigidity Modes:**
| Mode | Internal Flexibility | Rigid-Body Movement | Description |
|------|---------------------|---------------------|-------------|
| `fixed` | No | No | Ligands completely fixed (DEFAULT) |
| `rigid_body` | No | Yes | Ligands internally rigid, can move independently |
| `flexible` | Yes | Yes | Full ligand flexibility |

### Scorefunction Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--scorefunction` | ref2015 | Scorefunction name or weights file |
| `--constraint_weight` | 1.0 | Weight for all constraint terms |
| `--cart_bonded_weight` | 0.5 | Weight for cart_bonded in Cartesian relax |
| `--no_cartesian_relax` | False | Use torsion-space relaxation instead |

### Adaptive Coordinate Constraints
| Argument | Default | Description |
|----------|---------|-------------|
| `--adaptive_coord_cst` | False | Enable adaptive coordinate constraints |
| `--cst_deviation_threshold` | 5.0 | Constraint score threshold (REU) above which a residue is "far off" |
| `--coord_cst_neighbor_window` | 3 | Number of neighboring residues (±N) to also reduce constraints for |
| `--reduced_coord_cst_weight` | 0.2 | Coordinate constraint weight for far-off residues (0.0=free, 1.0=full) |

### Reference PDB Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--ref_pdb` | None | Reference PDB with ideal catalytic geometry; aligned by ligand |
| `--catres_cst_subset` | None | Comma-separated REMARK 666 line numbers to consider (e.g., "1,3,5") |
| `--ref_coord_cst` | False | Include 3D coordinate constraints for functional groups from ref_pdb |

**When `--ref_pdb` is provided:**
1. The reference structure is aligned to the input PDB by superimposing ligand atoms
2. If no `--cstfile` is provided, constraints are automatically derived from the reference
3. Metrics comparing designs to the reference are calculated and saved

See the "Reference PDB Feature" section below for detailed documentation.

### MPNN Configuration
| Argument | Default | Description |
|----------|---------|-------------|
| `--mpnn_runner` | /net/.../run.py | MPNN script path |
| `--mpnn_model_type` | ligand_mpnn | Model type |
| `--mpnn_omit_aa` | CM | AAs to omit |
| `--mpnn_use_sc_context` | 1 | Use SC context |
| `--mpnn_pack_side_chains` | 1 | Pack designed SCs |
| `--mpnn_repack_everything` | 0 | Repack all SCs |

---

## External Dependencies

**Fused MPNN API:**
```
/net/software/lab/fused_mpnn/seth_temp/
├── run.py             # MPNN runner
└── fusedmpnn.py       # MPNN API
```

**Apptainer Containers:**
```
/software/containers/universal.sif   # MPNN execution (default)
/software/containers/pyrosetta.sif   # PyRosetta with beta_jan25 support
```

---

## Container Support

### Automatic Container Detection

The pipeline automatically detects when a container is required and handles execution:

| Scorefunction | Requires Container | Auto-handled |
|---------------|-------------------|--------------|
| `ref2015` | No | N/A |
| `beta_nov16` | No | N/A |
| `beta_jan25` | Yes (`pyrosetta.sif`) | Yes |
| `beta_july15` | Depends | Yes |

### How It Works

1. **Detection**: When you specify `--scorefunction beta_jan25`, the script checks if you're already in a container
2. **Re-execution**: If not in a container, it automatically re-runs itself inside `pyrosetta.sif`
3. **Environment**: Sets `FASTMPNN_IN_CONTAINER=1` to track container state
4. **Bind Mounts**: Automatically binds necessary paths (input files, output directory, etc.)

### Nested Container Handling (MPNN)

When running inside a container and using `--2nd-shell-mpnn-seqs`:

```
┌─────────────────────────────────────────────────────────────────┐
│  enzyme_design.py (inside pyrosetta.sif)                        │
│                                                                 │
│  → Calls MPNN                                                   │
│    ├─ Try 1: apptainer exec universal.sif python run.py ...    │
│    │         (may fail due to nested container restrictions)    │
│    │                                                            │
│    └─ Try 2: apptainer exec pyrosetta.sif python run.py ...    │
│              (fallback - pyrosetta.sif may have MPNN deps)      │
│                                                                 │
│  → If both fail: Print detailed error with solutions            │
└─────────────────────────────────────────────────────────────────┘
```

### Troubleshooting Container Issues

**Error: Nested container execution failed**

Solutions:
1. **Use unified container**: Create a container with both PyRosetta and MPNN
2. **Skip seqs/ output**: Remove `--2nd-shell-mpnn-seqs` flag (main design with 2nd_shell_mpnn in protocol still works)
3. **Use standard scorefunction**: `--scorefunction ref2015` doesn't need a container
4. **Run MPNN separately**: Save intermediate PDBs and run MPNN manually

**Error: Apptainer not found**

Solutions:
1. Load the module: `module load apptainer` or `module load singularity`
2. Install Apptainer: https://apptainer.org/docs/admin/main/installation.html
3. Run manually: `/path/to/apptainer exec pyrosetta.sif python enzyme_design.py ...`

---

## Output Files

| Location | Description |
|----------|-------------|
| `./{name}_{suffix}_{iter}_{i}.pdb` | Designed structures (includes 2nd_shell_mpnn designs) |
| `./scores/{name}_{suffix}.sc` | Rosetta score file |
| `./seqs/` | 2nd shell MPNN outputs (only with `--2nd-shell-mpnn-seqs` flag) |

**Temperature naming convention:**
Files use underscores instead of periods for temperatures (e.g., `T0_1` for T=0.1, `T0_2` for T=0.2).

**Timing Output:**
The pipeline tracks and reports timing information:
- Per-iteration time: `[ITER 1] Completed in 2m 45s`
- Total pipeline time: `Total pipeline time: 15m 30s`
- Per-design time: Saved in scorefile as `design_time_sec`

---

## Authors

- Original: ikalvet
- Refactored: 2025
