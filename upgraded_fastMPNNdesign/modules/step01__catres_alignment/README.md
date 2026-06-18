# Step 01: Catalytic Residue Alignment

Align catalytic residue coordinates from a reference PDB to an input PDB, preserving crucial geometric interactions for enzyme design.

## Overview

This script is part of the `upgraded_fastMPNNdesign` pipeline. It takes a reference PDB (e.g., a theozyme or experimental structure) and an input PDB (e.g., a structure prediction with the ligand already aligned), then:

1. **Aligns** the ref_pdb to input_pdb coordinate frame via ligand superposition
2. **Validates** REMARK 666 consistency between both files
3. **Classifies** residues as `catalytic_motif` (catres_subset) or `conserved_motif`
4. **Detects** crucial interactions for each catalytic residue (H-bonds, metal coordination, pi-stacking, etc.)
5. **Transforms** coordinates selectively based on backbone/sidechain importance
6. **Outputs** three files:
   - Aligned PDB with transformed catalytic residue coordinates
   - Interaction analysis JSON with detailed interaction data
   - Constraint recommendations JSON for downstream Rosetta coordinate constraints

## Installation

The script requires NumPy. Run via Apptainer:

```bash
/net/software/containers/universal.sif python /path/to/align_catres.py --input_pdb ... --ref_pdb ... --outdir ...
```

Or with a local Python environment:

```bash
pip install numpy
python align_catres.py --input_pdb ... --ref_pdb ... --outdir ...
```

## Usage

### Basic Usage

```bash
python align_catres.py \
    --input_pdb input_structure.pdb \
    --ref_pdb reference_theozyme.pdb \
    --outdir output_directory/
```

### With Catalytic Residue Subset

```bash
python align_catres.py \
    --input_pdb input_structure.pdb \
    --ref_pdb reference_theozyme.pdb \
    --outdir output_directory/ \
    --catres_subset 1,3,5,7
```

### Test Mode

```bash
/net/software/containers/universal.sif /home/woodbuse/special_scripts/upgraded_fastMPNNdesign/modules/step01__catres_alignment/align_catres.py --test
```

### Full Options

```
usage: align_catres.py [-h] [--input_pdb INPUT_PDB] [--ref_pdb REF_PDB]
                       [--outdir OUTDIR] [--catres_subset CATRES_SUBSET]
                       [--outfile_bn OUTFILE_BN] [--strict_backbone_importance]
                       [--exclude_bb_only_hbond_constraints]
                       [--flex_res_move_all_sc] [--flex_res_constrain_all_sc]
                       [--test] [-v]

Arguments:
  --input_pdb      Input PDB file (structure prediction with ligand aligned)
  --ref_pdb        Reference PDB file (original theozyme/template)
  --outdir         Output directory
  --catres_subset  Comma-separated REMARK 666 block indices for catalytic_motif
                   (default: all REMARK 666 residues are catres_subset)
  --outfile_bn     Custom basename for output files (default: input_pdb basename)
  --strict_backbone_importance
                   Backbone-to-backbone H-bonds alone don't make backbone_important=True
                   (requires interaction to sidechain or ligand for backbone to be important)
  --exclude_bb_only_hbond_constraints
                   Don't include backbone atoms in constraint recommendations when
                   backbone is only important due to backbone_hbond_* interactions
  --flex_res_move_all_sc
                   For ARG/LYS with sidechain-only importance: move entire sidechain
                   instead of just tip atoms (default: only move tip atoms)
  --flex_res_constrain_all_sc
                   For ARG/LYS with sidechain-only importance: constrain entire sidechain
                   instead of just tip atoms (implies --flex_res_move_all_sc)
  --test           Run with hardcoded test data
  -v, --verbose    Enable debug logging
```

## Input Requirements

### PDB Files

Both `input_pdb` and `ref_pdb` must contain:

1. **REMARK 666 lines** defining catalytic residue / ligand constraints
2. **HETATM records** for the ligand (identical atom names and order)
3. **ATOM records** for all catalytic residues

### REMARK 666 Format

```
REMARK 666 MATCH TEMPLATE <chain> <resname> <resno> MATCH MOTIF <chain> <resname> <resno> <block_idx> <variant>
```

Example:
```
REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A HIS   13  1  1
REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A ASP   53  5  1
REMARK 666 MATCH TEMPLATE A ASP   53 MATCH MOTIF A HIS  226  8  1
```

- **TEMPLATE**: The entity being matched against (ligand or another residue)
- **MOTIF**: The catalytic residue position in the designed structure
- **block_idx**: 1-indexed constraint block number
- **variant**: Constraint variant (typically 1)

### Consistency Requirements

- REMARK 666 entries must be identical between ref_pdb and input_pdb
- Same number of entries, same block indices, same residue types and numbers
- Ligand atoms must have identical atom names in both files

## Output Files

### 1. Aligned PDB (`<basename>_aligned.pdb`)

Modified input_pdb with catres_subset residue coordinates transformed to match ref_pdb geometry:

- **If importance = "backbone" or "both"**: All atom coordinates (backbone + sidechain) replaced
- **If importance = "sidechain"**: Only sidechain coordinates replaced, backbone kept in place

### 2. Interaction Analysis JSON (`<basename>_interactions.json`)

```json
{
  "input_pdb": "/path/to/input.pdb",
  "ref_pdb": "/path/to/ref.pdb",
  "ligand": {"chain": "B", "resname": "XDW", "resno": 257},
  "alignment_rmsd": 0.0,
  "catres_subset_blocks": [1, 3, 5],
  "conserved_motif_blocks": [2, 4, 6],
  "residue_analysis": {
    "1": {
      "chain": "A",
      "resno": 13,
      "resname": "HIS",
      "interactions": [
        {"type": "metal_coord", "catres_atom": "NE2", "target_atom": "ZN1", "distance": 2.1, "is_backbone": false, "target_type": "ligand"}
      ],
      "backbone_important": false,
      "sidechain_important": true,
      "importance": "sidechain"
    }
  }
}
```

### 3. Constraint Recommendations JSON (`<basename>_recommended_atom_cst.json`)

Specifies which atoms to constrain for downstream Rosetta coordinate constraints:

```json
{
  "input_pdb": "/path/to/input.pdb",
  "ref_pdb": "/path/to/ref.pdb",
  "output_pdb": "/path/to/output_aligned.pdb",
  "exclude_bb_only_hbond_constraints": false,
  "flex_res_move_all_sc": false,
  "flex_res_constrain_all_sc": false,
  "residue_constraints": {
    "1": {
      "chain": "A",
      "resno": 13,
      "resname": "HIS",
      "constrain_atoms": ["C", "CA", "CB", "CD2", "CE1", "CG", "H", "HD1", "N", "ND1", "NE2", "O"],
      "backbone_important": true,
      "sidechain_important": true,
      "importance": "both",
      "backbone_important_only_for_BB_BB_hbond": false
    }
  }
}
```

**Per-residue fields:**
- `backbone_important_only_for_BB_BB_hbond`: `true` if the backbone is only making H-bonds to other backbone atoms (backbone_hbond_bb_donor/acceptor). `false` if the backbone has no interactions OR if it has interactions with sidechains or ligand atoms. This helps identify residues where backbone geometry may be less critical for catalysis.

**Atom selection logic:**
- **Sidechain important (not backbone)**: All sidechain heavy atoms (C, N, O, S) + sidechain polar hydrogens
- **Backbone important (not sidechain)**: Backbone heavy atoms (N, CA, C, O) + backbone polar H
- **Both important**: All heavy atoms + all polar hydrogens

**`--exclude_bb_only_hbond_constraints` flag:**
When enabled, if backbone was only marked important due to `backbone_hbond_*` interactions (H-bonds to another backbone), the backbone atoms are excluded from the constraint list. This is useful when backbone-to-backbone H-bonds moved the backbone but those interactions aren't catalytically important enough to warrant coordinate constraints.

### Extremely Flexible Residues (ARG, LYS)

ARG and LYS have long, flexible sidechains where the functional tip (guanidinium for ARG, amine for LYS) is often the only part that participates in important interactions. By default, when these residues have **sidechain-only importance**, only the tip atoms are moved and constrained:

| Residue | Tip Heavy Atoms | Tip Polar Hydrogens |
|---------|-----------------|---------------------|
| LYS | NZ, CE, CD | HZ1, HZ2, HZ3 (or 1HZ, 2HZ, 3HZ) |
| ARG | NH1, CZ, NH2, NE, CD | HE, HH11, HH12, HH21, HH22 (or 1HH1, 2HH1, 1HH2, 2HH2) |

**Flags to override default behavior:**

- `--flex_res_move_all_sc`: Move the entire sidechain (all atoms from CB onward) for ARG/LYS, but still only constrain the tip atoms
- `--flex_res_constrain_all_sc`: Constrain the entire sidechain (all heavy atoms + polar H) for ARG/LYS. This flag implies `--flex_res_move_all_sc`.

**Use cases:**
- **Default (no flags)**: Best for cases where the CB-CG portion of the sidechain might adopt a different rotamer in the designed structure, but the tip position is critical
- **`--flex_res_move_all_sc`**: Move the full sidechain from ref_pdb but only constrain the tip, allowing some flexibility in the middle of the sidechain
- **`--flex_res_constrain_all_sc`**: Strictly maintain the entire sidechain geometry from ref_pdb

**Example constraint atoms for LYS:**
- Default (tip-only): `[CD, CE, NZ, 1HZ, 2HZ, 3HZ]` (6 atoms)
- `--flex_res_constrain_all_sc`: `[CB, CG, CD, CE, NZ, 1HZ, 2HZ, 3HZ]` (8 atoms)

**Note:** When backbone is also important (importance="both"), the entire residue is always moved and constrained normally, regardless of these flags.

## Interaction Types Detected

### Hydrogen Bond Detection (Geometry-Based)

H-bonds are detected using both distance AND angle criteria:
- **Distance**: H...Acceptor ≤ 3.5 Å
- **D-H...A angle**: ≥ 120° (donor atom - hydrogen - acceptor linearity)
- **H...A-B angle**: ≥ 100° (hydrogen - acceptor - base atom, lone pair orientation)

Polar hydrogens are identified by connectivity (bonded to N, O, or S within 1.4 Å), not by atom name.

### Backbone Interactions (trigger `importance: "backbone"` or `"both"`)

| Type | Description | Criteria |
|------|-------------|----------|
| `hbond_bb_donor` | Backbone H donating to sidechain/ligand acceptor | H bonded to N, geometry validated |
| `hbond_bb_acceptor` | Backbone O accepting from sidechain/ligand H | Donor H bonded to N/O/S, geometry validated |
| `backbone_hbond_bb_donor` | Backbone H donating to another backbone O | Both atoms are backbone, geometry validated |
| `backbone_hbond_bb_acceptor` | Backbone O accepting from another backbone H | Both atoms are backbone, geometry validated |

**Note:** `backbone_hbond_*` types represent backbone-to-backbone H-bonds between different residues. By default, these count toward `backbone_important=True`. With `--strict_backbone_importance`, these are excluded from the importance calculation (only backbone interactions to sidechains or ligands count).

### Sidechain Interactions (trigger `importance: "sidechain"`)

| Type | Description | Criteria |
|------|-------------|----------|
| `hbond_sc_donor` | Sidechain polar H donating | H bonded to N/O/S, geometry validated |
| `hbond_sc_acceptor` | Sidechain N/O/S accepting | Geometry validated |
| `metal_coord` | Coordination to metal ion (Zn, Mg, Fe, etc.) | ≤ 2.8 Å |
| `charged` | Ionic interaction from charged atom to heteroatom (consolidated per target) | ≤ 4.5 Å, specific atoms only (see below) |
| `acid_base_mod` | Catalytic proton transfer (H within bonding distance) | ≤ 1.5 Å, acid/base atoms only (see below) |
| `hydrophobic` | C-C contact for non-aromatic hydrophobic residues | ≤ 4.5 Å, consolidated per target |
| `pi_lig_interaction` | Aromatic ring contact with ligand carbons | Ring centroid to C ≤ 4.5 Å |
| `pi_pi_stacking` | Face-to-face aromatic stacking | Geometry-based (see below) |
| `pi_pi_stacking_displaced` | Parallel-displaced aromatic stacking | Geometry-based (see below) |
| `pi_pi_Tshape` | T-shaped / edge-to-face aromatic interaction | Geometry-based (see below) |
| `post_translational_mod` | Covalent-distance heavy atom contact | ≤ 2.2 Å, excludes H and metal_coord pairs |

### Pi-Stacking Geometry Detection

Pi interactions between aromatic residues (PHE, TYR, TRP, HIS) use ring-geometry analysis instead of simple atom distances:

**Ring atoms used:**
- PHE/TYR: CG, CD1, CD2, CE1, CE2, CZ (6-member)
- HIS: CG, ND1, CD2, CE1, NE2 (5-member imidazole)
- TRP: Both 5-member pyrrole and 6-member benzene rings

**Geometry criteria:**
- **Centroid distance**: 3.3–6.0 Å between ring centroids
- **Interplanar angle (θ)**: Angle between ring plane normals
- **Perpendicular separation (h)**: Projection of centroid vector onto plane normal
- **Lateral offset**: In-plane displacement between centroids

**Classification:**
| Type | Angle (θ) | Perpendicular (h) | Offset |
|------|-----------|-------------------|--------|
| `pi_pi_stacking` | ≤ 30° | ≤ 4.0 Å | ≤ 2.0 Å |
| `pi_pi_stacking_displaced` | ≤ 30° | ≤ 4.0 Å | 2.0–5.0 Å |
| `pi_pi_Tshape` | 60–90° | — | Min atom distance ≤ 4.5 Å |

### Charged Atom Restrictions

Only actual ionizable atoms can participate in `charged` interactions:

| Residue | Charged Atoms | Charge |
|---------|---------------|--------|
| ASP | OD1, OD2 | Negative |
| GLU | OE1, OE2 | Negative |
| LYS | NZ | Positive |
| ARG | NE, NH1, NH2 | Positive |
| HIS | ND1, NE2 | Positive (if protonated) |

**Note:** If a residue has a `metal_coord` interaction, all `charged` interactions are removed for that residue. Metal-coordinating residues (e.g., HIS coordinating Zn) should not also be considered as having ionic interactions.

### Acid/Base Atoms for Proton Transfer

The `acid_base_mod` interaction is detected when a hydrogen atom is within bonding distance (≤ 1.5 Å) of an acid/base atom on a catalytic residue, indicating potential catalytic proton transfer:

| Residue | Acid/Base Atoms | Role |
|---------|-----------------|------|
| ASP | OD1, OD2 | Proton acceptor (general base) |
| GLU | OE1, OE2 | Proton acceptor (general base) |
| HIS | ND1, NE2 | Proton shuttle |
| LYS | NZ | Proton donor (general acid) |
| ARG | NE, NH1, NH2 | Proton donor |
| CYS | SG | Nucleophile / proton donor |
| TYR | OH | Proton donor |
| SER | OG | Nucleophile / proton donor |

**Note:** `acid_base_mod` interactions are not double-counted as hydrogen bonds.

## Residue Classification

### catalytic_motif (catres_subset)

Residues specified in `--catres_subset` (or all REMARK 666 residues if not specified). These residues:

- Have their geometry constrained to match the reference
- Are analyzed for crucial interactions
- Have coordinates transformed (backbone, sidechain, or both) based on interaction analysis

### conserved_motif

REMARK 666 residues NOT in catres_subset. These residues:

- Will be conserved in sequence during downstream design
- Are NOT geometrically constrained
- Keep their coordinates from input_pdb

## Algorithm Details

### Step 1: Ligand Alignment

The reference PDB is aligned to the input PDB coordinate frame using the Kabsch algorithm on ligand heavy atoms. Since the ligand geometry should be identical in both files, RMSD should be ~0.

### Step 2: Interaction Analysis

For each catres_subset residue, the script checks:

1. **Ligand interactions**: All atom pairs between residue and ligand
2. **Inter-catres interactions**: Interactions with other catres_subset residues
3. **Intra-residue interactions**: Backbone-sidechain contacts within the same residue

### Step 3: Importance Classification

```
IF backbone_interactions exist AND sidechain_interactions exist:
    importance = "both"
ELSE IF backbone_interactions exist:
    importance = "backbone"
ELSE:
    importance = "sidechain"  # default
```

### Step 4: Coordinate Transformation

```
IF importance == "both" OR importance == "backbone":
    Replace ALL atom coordinates with ref_pdb
ELSE (importance == "sidechain"):
    Replace ONLY sidechain coordinates with ref_pdb
    Keep backbone coordinates from input_pdb
```

## Example Workflow

```bash
# 1. Run alignment
/net/software/containers/universal.sif python align_catres.py \
    --input_pdb prediction.pdb \
    --ref_pdb theozyme.pdb \
    --outdir results/ \
    --catres_subset 1,2,3,4,5

# 2. Check interaction analysis
cat results/prediction_interactions.json | python -m json.tool

# 3. Use aligned PDB for downstream design
# ... continue with step02__constrained_cart_relax
```

## Test Data

Test files are located in:
```
test/
├── input_pdb.pdb   # Structure prediction with 19 REMARK 666 entries
├── ref_pdb.pdb     # Reference theozyme with same 19 entries
└── output_dir/     # Output directory for test runs
```

Run test mode:
```bash
/net/software/containers/universal.sif /home/woodbuse/special_scripts/upgraded_fastMPNNdesign/modules/step01__catres_alignment/align_catres.py --test
```

This uses catres_subset `1,2,3,4,5,6,7,8,9,10,11,13,15,16,17,18,19` (blocks 12 and 14 are conserved_motif).

## Troubleshooting

### "REMARK 666 count mismatch"

Ensure both PDB files have the same REMARK 666 lines. Check for:
- Missing or extra constraint blocks
- Different block indices

### "Ligand atom count mismatch"

The ligand (HETATM records) must have:
- Same number of atoms
- Same atom names
- Same ordering (or the script will sort by atom name)

### "Ligand RMSD higher than expected"

If RMSD > 0.1 Å, the ligands may not be identical. Check:
- Ligand protonation states
- Different conformers
- Atom naming differences

### "No atoms found for residue"

The residue specified in REMARK 666 doesn't exist in the ATOM records. Check:
- Chain ID matches
- Residue number matches
- Residue is present in the PDB file

## File Structure

```
step01__catres_alignment/
├── align_catres.py      # Main script
├── README.md            # This documentation
└── test/                # Test data directory
    ├── input_pdb.pdb
    ├── ref_pdb.pdb
    └── output_dir/

module_utils/
├── __init__.py
└── pdb_utils.py         # Shared PDB parsing utilities
```

## Constants

The following thresholds are defined at the top of `align_catres.py`:

```python
# Distance cutoffs (Angstroms)
HBOND_DIST_MAX = 3.5           # H-bond H...Acceptor distance
METAL_COORD_DIST_MAX = 2.8     # Metal coordination
CHARGED_DIST_MAX = 4.5         # Charged interactions
HYDROPHOBIC_DIST_MAX = 4.5     # Hydrophobic contacts
COVALENT_DIST_MAX = 2.2        # Post-translational modification
ACID_BASE_DIST_MAX = 1.5       # Acid/base proton transfer (within bonding distance)

# H-bond geometry thresholds (degrees)
HBOND_DONOR_ANGLE_MIN = 120.0  # D-H...A angle minimum (linearity)
HBOND_ACCEPTOR_ANGLE_MIN = 100.0  # H...A-B angle minimum (lone pair)

# Bond detection thresholds (Angstroms)
H_BOND_DIST_MAX = 1.4          # H bonded to heavy atom
HEAVY_BOND_DIST_MAX = 1.8      # Heavy atom bonds

# Pi-stacking geometry thresholds
PI_CENTROID_DIST_MIN = 3.3     # Min centroid-centroid distance (A)
PI_CENTROID_DIST_MAX = 6.0     # Max centroid-centroid distance (A)
PI_PARALLEL_ANGLE_MAX = 30.0   # Max angle for parallel stacking (degrees)
PI_TSHAPE_ANGLE_MIN = 60.0     # Min angle for T-shaped (degrees)
PI_TSHAPE_ANGLE_MAX = 90.0     # Max angle for T-shaped (degrees)
PI_PERP_SEPARATION_MAX = 4.0   # Max perpendicular separation (A)
PI_OFFSET_FACE_TO_FACE = 2.0   # Max offset for face-to-face (A)
PI_OFFSET_DISPLACED_MAX = 5.0  # Max offset for displaced stacking (A)
PI_TSHAPE_CONTACT_MAX = 4.5    # Max atom-atom distance for T-shape contact (A)

# Extremely flexible residue tip atoms
FLEXIBLE_RESIDUE_TIP_HEAVY = {
    "LYS": {"NZ", "CE", "CD"},
    "ARG": {"NH1", "CZ", "NH2", "NE", "CD"},
}
```

## Dependencies

- Python 3.7+
- NumPy

## Author

Generated for the upgraded_fastMPNNdesign enzyme design pipeline.
