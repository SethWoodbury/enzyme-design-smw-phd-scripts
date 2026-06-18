## Bug 1: HIS Tautomer Mutation with Metal Pseudobonds

**Error:**
```
[HIS Tautomer] Mutating A13 from HIS:MP-NE2-connect to HIS_D
core.conformation.Residue: [ ERROR ] Unable to handle change in the number of residue connections in the presence of pseudobonds!
```

**Root Cause:**
- PyRosetta cannot mutate residues (change residue type) when pseudobonds exist
- Metal-coordinating HIS residues have pseudobonds to zinc atoms
- **CRITICAL**: Pseudobonds are created from GEOMETRY detection, not just REMARK 666 lines!
- When zinc atoms are present near HIS residues, PyRosetta auto-creates pseudobonds during `pose_from_file()`

**Attempted Fixes (FAILED):**
1. First attempt: Skip metal-coordinating HIS - REJECTED because this leaves wrong tautomers
2. Second attempt: Strip REMARK 666 lines before loading - FAILED because PyRosetta detects metals from geometry
3. Third attempt: Remove duplicate tautomer call - FAILED because pseudobonds still created from geometry

**What We Learned:**
- Stripping REMARK 666 is NOT enough - PyRosetta detects metal coordination from atomic coordinates
- The only way to prevent pseudobond creation is to load the PDB WITHOUT metal atoms (HETATM)
- Tautomers must be applied to a protein-only pose, then metals can be added back

**WORKING FIX (in `setup_pose_from_mpnn_output()`):**

The solution is to load the protein separately from HETATM (metals/ligands):

```python
# STEP 1: Separate protein from HETATM
protein_lines = [line for line in pdb if line.startswith('ATOM') or line.startswith('TER')]
hetatm_lines = [line for line in pdb if line.startswith('HETATM')]

# STEP 2: Load PROTEIN-ONLY (no metals = no pseudobonds!)
protein_only_pdb = '\n'.join(protein_lines) + '\nEND\n'
pose = pyr.pose_from_file(protein_only_temp_path)  # PyRosetta adds H here

# STEP 3: Apply HIS tautomers (SAFE - no metals, no pseudobonds)
apply_his_tautomers_to_pose(pose, his_tautomer_map)

# STEP 4: Export protein with correct tautomers, merge back HETATM
protein_with_H = pyr.distributed.io.to_pdbstring(pose)
full_pdb = headers + remark666 + protein_with_H + hetatm_lines + END

# STEP 5: Load full PDB (now pseudobonds are created with CORRECT tautomers)
pose = pyr.pose_from_file(full_temp_path)
```

**Debug output files (when --debug is used):**
- `pose{n}_step1_protein_only.pdb` - Protein without HETATM (no H yet)
- `pose{n}_step2_after_tautomers.pdb` - After tautomer correction
- `pose{n}_step3_full_pdb_fixed_tautomers.pdb` - Full PDB before loading
- `pose{n}_step4_pose_with_hetatm.pdb` - After loading with metals
- `pose{n}_step5_final_pose.pdb` - Final pose after all setup

## Bug 2: LYS 174 SidechainConjugation Warning

**Warning:**
```
core.conformation.Residue: [ WARNING ] Residue::inter_residue_connection_partner: Invalid residue connection, returning BOGUS ID: this_rsd= LYS:SidechainConjugation 174 connid= 3 partner_seqpos= 0
```

**What This Means:**
- LYS 174 has a sidechain conjugation patch but no partner residue
- This suggests a covalent bond that wasn't properly established

## General Cautions

1. **Metal detection from geometry**: PyRosetta detects metal coordination from atomic coordinates, not just REMARK 666 lines. If zinc atoms are within coordination distance of HIS/CYS/ASP/GLU, pseudobonds will be created.

2. **HETATM must be separated**: To avoid premature pseudobond creation, load protein-only first, then add HETATM back.

3. **Order of operations for HIS tautomers**:
   1. Load protein-only (no metals)
   2. Apply tautomer corrections (safe - no pseudobonds)
   3. Export protein with correct H atoms
   4. Merge back HETATM + headers
   5. Load full structure (pseudobonds now created with correct tautomers)

4. **MPNN output**: Raw MPNN output has no hydrogens on protein (only HETATM). Hydrogens are added by PyRosetta during `pose_from_file()`.

5. **MutateResidue limitations**: Cannot change residue types when pseudobonds exist. This includes HIS↔HIS_D tautomer changes.

6. **Inspired by**: The working approach is based on `/home/woodbuse/special_scripts/general_utils/add_remark666_lines_AND_rosetta_hydrogens_to_pdb_OPTIMIZED.py` which uses the same strategy of loading protein-only, applying mutations, then merging back.

---

## Bug 3: CST Scores Always Zero (FIXED)

**Symptom:**
- `.sc` files show `all_cst=0.0` and all individual constraint scores as zero
- RMSD to reference is high despite "good" constraint scores
- Constraint violations not being captured during scoring

**Root Cause:**
- `enzyme_design.py:778` was using `pyr.get_fa_scorefxn()` instead of the configured `sfx`
- The default scorefunction has zero weights for constraint terms
- The configured `sfx` has constraint weights set from `--constraint_weight` argument

**Fix:**
```python
# OLD (BROKEN):
scores_df = scoring_module.score_design(p, pyr.get_fa_scorefxn(), list(catres.keys()))

# NEW (FIXED):
scores_df = scoring_module.score_design(p, sfx, list(catres.keys()))
```

**Verification:**
```bash
grep all_cst scores/*.sc  # Should now show non-zero values when constraints violated
```

---

## Bug 4: Debug Structures Not Saved to Correct Directory (FIXED)

**Symptom:**
- `structures/` subdirectory empty despite `--debug` flag
- Debug structures appearing in unexpected locations

**Root Cause:**
- `enzyme_design.py:1310-1318` created a different `debug_output_dir` and overwrote the correct setting
- Tracker outputs and structure dumps were going to different directories
- The per-iteration debug directory was created near the input PDB, not in the working directory

**Fix:**
- Tracker outputs now go to the parent debug directory (`debug_output_{pdbname}/`)
- Structure dumps go to `debug_output_{pdbname}/structures/`
- Removed redundant `fmd.set_debug_output_dir()` call that was overwriting correct setting

**Verification:**
```bash
ls debug_output_*/structures/        # Should contain stepXX_min_poseY.pdb files
ls debug_output_*/                   # Should contain *_pipeline_report.json
```

---

## Sidechain Flip Logic Changes

### Chemical Sensitivity ABOLISHED

**Previous behavior:**
- HIS flips were disabled by default due to "chemical sensitivity"
- `CHEMICALLY_SENSITIVE_FLIPS` dict gated certain residue types

**New behavior:**
- All residues in `FLIPPABLE_SIDECHAINS` are eligible for flips
- No residue-type-specific exclusions
- Flips are instead gated by evidence-based criteria (see below)

### New Flip Gating System

Flips are now gated by three criteria that must all be met:

1. **Stage Gate** (`flip_stage_gate=5`):
   - No flips before protocol step N
   - Allows structure to settle before attempting flips

2. **Chi Convergence** (`flip_chi_delta_thresh=5.0°`, `flip_convergence_window=3`):
   - Chi angles must be stable for N consecutive steps
   - Prevents flipping residues that are still moving significantly

3. **Persistent CST** (`flip_cst_threshold=2.0`, `flip_cst_persistence_window=3`):
   - Constraint score must be above threshold for N consecutive steps
   - Only flips residues with sustained poor constraint satisfaction

### Configuration

```python
fmd.set_catres_flip_config(
    enable_auto=True,           # Enable automatic flips after min/repack
    cst_threshold=2.0,          # Threshold for "poor" CST (raised from 1.0)
    max_min_iter=10,            # Minimization iterations after flip
    stage_gate=5,               # No flips before step 5
    chi_delta_thresh=5.0,       # Max chi change for convergence (degrees)
    convergence_window=3,       # Steps required for chi convergence
    cst_persistence_window=3    # Steps required for persistent poor CST
)
```

### Expected Behavior

- TRP and other aromatic residues no longer flip early in the protocol
- Flips only occur after the structure has converged
- Flips only attempted for residues with sustained constraint violations
- Flip acceptance still requires CST improvement (>0.1 REU)

---

## Debug Output Summary

When `--debug` is enabled:

| Location | Contents |
|----------|----------|
| `debug_output_{pdb}/` | Pipeline tracker reports (JSON), logs |
| `debug_output_{pdb}/structures/` | PDB files at each protocol step |
| `debug_output_{pdb}/full_log.txt` | Complete stdout/stderr capture |

### Flip Gate Logging

Each flip attempt logs:
- `[Flip Gate] Step X < stage gate Y, skipping flips` - Stage gate not passed
- `[Flip Gate] RES N: not converged, skipping` - Chi angles still changing
- `[Flip Gate] RES N: CST not persistently poor, skipping` - CST improved recently
- `[Flip Gate] RES N: ELIGIBLE (converged + persistent CST)` - All gates passed

Flip outcomes log:
- `[Flip] RES N: ACCEPTED - cst X -> Y (delta=Z)` - Flip improved CST
- `[Flip] RES N: REJECTED - cst X -> Y (reverted)` - No improvement, reverted
