# Relaxation Protocol Overhaul Plan

**Date:** 2026-01-16
**Status:** Critical issues identified, major overhaul required

---

## 1. Critical Problems Identified

### 1.1 Backbone Frozen During Relaxation
**Symptom:** Sidechains detaching from backbone to fulfill constraints. Cart_bonded scores of 1700-3100 (should be < 50).

**Root Cause:** Current implementation uses `freeze_catres=False` but the mobile region is too restricted. The MoveMap allows chi movement but backbone movement is constrained by coordinate constraints that target specific atom positions, forcing sidechains to stretch/bend unnaturally.

**Evidence:**
- Test 1: cart_bonded = 1942 avg
- Test 5: cart_bonded = 3126
- Bond length/angle deviations causing strain

### 1.2 Histidine Ring Flip Issue
**Symptom:** Some histidine imidazole rings appear 180° rotated from optimal orientation, but can't flip due to steric bulk.

**Root Cause:**
- HIS/HIS_D assignment is based on HD1/HE2 atoms in input PDB
- FastRelax doesn't try alternative rotamers when stuck in local minima
- No mechanism to sample symmetric ring flips

**Proposed Solution:**
1. After initial relaxation, detect histidines with high constraint scores
2. Apply 180° chi2 rotation (imidazole flip)
3. Re-score and keep better conformation
4. Do this for any residue with planar symmetry (PHE, TYR, HIS, TRP indole)

### 1.3 HD2 Hydrogen Flipped Into Ring
**Symptom:** HD2 hydrogens on histidines occasionally positioned inside the imidazole ring.

**Root Cause:** PyRosetta hydrogen placement during pose loading or relaxation.

**Solution:** Post-relaxation validation and re-optimization of hydrogen positions.

### 1.4 Cartesian Relax May Not Be Optimal
**Insight from original script:** `/home/woodbuse/special_scripts/fastmpnn_design/fastmpnn_ZnEsterase_SETH_LINKED.py`

The original script uses a different approach:
1. Quick cartesian pre-relax with ALA mutations for clashes
2. Protocol-based gradual optimization:
   - Start: high coordinate_constraint (1.0), low fa_rep (0.15)
   - Multiple MPNN → repack → minimize cycles
   - Gradually increase fa_rep and decrease coordinate_constraint
3. Uses minimization (`min`) more than FastRelax
4. Uses enzyme design constraints (.cst file with atom_pair, angle, dihedral)

**Key difference:** Original uses enzyme design constraints (distance + angle + dihedral), not just coordinate constraints. This allows backbone to move while maintaining relative geometry.

---

## 2. Bond Length/Angle Metrics to Implement

### 2.1 Bond Length Deviation Metric
**Acceptable threshold:** < 0.03-0.04 Å deviation from ideal

```python
def compute_bond_length_deviations(pose, residue_list):
    """
    Compare actual bond lengths to ideal values from residue parameters.

    Returns dict with:
    - per_residue_max_deviation: max deviation per residue
    - per_residue_mean_deviation: mean deviation per residue
    - worst_bonds: list of (residue, bond, deviation) tuples
    """
    # Use pose.residue(i).type().bond_length() for ideal values
    # Compare to actual distances from xyz coordinates
```

### 2.2 Bond Angle Deviation Metric
**Acceptable threshold:** < 3-5° deviation from ideal

```python
def compute_bond_angle_deviations(pose, residue_list):
    """
    Compare actual bond angles to ideal values.

    Returns similar structure to bond length metric.
    """
    # Use pose.residue(i).type().bond_angle() for ideal values
```

### 2.3 Combined Geometry Quality Score
```python
def compute_geometry_quality(pose, catres_list):
    """
    Overall geometry quality metric combining:
    - Bond length deviations
    - Bond angle deviations
    - Cart_bonded per-residue breakdown
    - Ramachandran outliers
    """
```

---

## 3. Proposed Relaxation Protocol Overhaul

### 3.1 Switch from Pure Coordinate Constraints to Enzyme Design Constraints

**Current approach:**
- CoordinateConstraint on ligand + catres atoms
- Very restrictive, forces atoms to absolute positions

**New approach:**
- Use distance constraints (AtomPair) for key interactions
- Use angle constraints for coordination geometry
- Allow backbone movement while maintaining relative geometry
- Reference: EnzConstraintIO in original script

### 3.2 Multi-Stage Relaxation Protocol

Inspired by original script lines 310-341:

```
Stage 1: Initial Constraint Satisfaction
- High coordinate_constraint weight (500-1000)
- Low fa_rep (0.15)
- Allow full backbone movement
- 1-2 FastRelax cycles

Stage 2: Geometry Optimization
- Medium coordinate_constraint (200-500)
- Medium fa_rep (0.5)
- Minimize with tighter tolerances
- Check bond length/angle deviations

Stage 3: Final Refinement
- Low/no coordinate_constraint (0-100)
- Full fa_rep (1.0)
- Final minimization
- Validate geometry
```

### 3.3 Ring Flip Sampling

```python
def sample_ring_flips(pose, residue_list, sfxn):
    """
    For residues with planar symmetry, try 180° flip and keep better.

    Applies to: HIS (chi2), PHE (chi2), TYR (chi2), TRP (indole)
    """
    for resnum in residue_list:
        res = pose.residue(resnum)
        if res.name3() in ['HIS', 'PHE', 'TYR']:
            # Get current chi2
            chi2 = pose.chi(2, resnum)
            score_before = sfxn(pose)

            # Flip 180°
            pose.set_chi(2, resnum, chi2 + 180)
            score_after = sfxn(pose)

            # Keep better
            if score_after > score_before:
                pose.set_chi(2, resnum, chi2)  # Revert
```

### 3.4 Hydrogen Optimization

```python
def fix_hydrogen_positions(pose):
    """
    Re-optimize hydrogen positions after heavy atom relaxation.

    Specifically addresses HD2 flipped into HIS ring.
    """
    # Use Rosetta's idealize_hydrogens or repack with H-only movemap
```

---

## 4. Implementation Plan

### Phase 1: Metrics Implementation (Agent 1 - haiku)
1. Implement `compute_bond_length_deviations()`
2. Implement `compute_bond_angle_deviations()`
3. Add to RelaxResult dataclass
4. Add filtering based on geometry quality

### Phase 2: Relaxation Protocol Overhaul (Agent 2 - sonnet)
1. Implement multi-stage relaxation
2. Add enzyme design constraint support (distance + angle)
3. Allow configurable backbone flexibility
4. Add ring flip sampling post-relaxation

### Phase 3: Hydrogen Fixes (Agent 3 - haiku)
1. Implement HD2/HE2 validation for histidines
2. Add hydrogen re-optimization step
3. Validate imidazole geometry

### Phase 4: Testing & Validation (Agent 4 - sonnet)
1. Run comprehensive tests with new protocol
2. Compare cart_bonded, bond deviations, constraint satisfaction
3. Iterate on parameters

---

## 5. Key Files to Reference

### Original Script (patterns to follow)
- `/home/woodbuse/special_scripts/fastmpnn_design/fastmpnn_ZnEsterase_SETH_LINKED.py`
  - Lines 234-261: Pre-relax with cartesian
  - Lines 310-341: Protocol with gradual constraint ramping
  - Uses EnzConstraintIO for proper enzyme constraints

### Current Implementation (to modify)
- `fastmpnndesign/relax_runner.py` - Main relaxation logic
- `fastmpnndesign/constraints.py` - Constraint generation
- `fastmpnndesign/metrics.py` - Quality metrics

### Design Utils (helpful functions)
- `/net/software/lab/scripts/enzyme_design/utils/design_utils.py`
- `setup_fastrelax()`, `get_matcher_residues()`, etc.

---

## 6. Success Criteria

After overhaul, relaxed structures should meet:

| Metric | Threshold |
|--------|-----------|
| Cart_bonded (total) | < 50 |
| Cart_bonded (per catres) | < 3 |
| Bond length deviation | < 0.04 Å |
| Bond angle deviation | < 5° |
| Catres displacement from ref | < 0.5 Å |
| Ligand displacement | 0.0 Å |
| Catres sidechain RMSD | < 0.3 Å |

---

## 7. Immediate Actions

1. **STOP** current tests - they're using broken relaxation
2. Write geometry deviation metrics
3. Overhaul relax_runner.py with multi-stage protocol
4. Add ring flip sampling
5. Fix hydrogen placement
6. Test with single structure before scaling up

---

## 8. Notes on Constraint Philosophy

**Current (broken):** "Fix atoms in absolute space"
- Forces sidechains to stretch unnaturally
- Backbone can't adjust to accommodate sidechains
- Results in high cart_bonded scores

**Correct approach:** "Maintain relative geometry while allowing global movement"
- Use distance constraints to maintain key interactions
- Use angle/dihedral constraints for coordination geometry
- Allow backbone to flex to accommodate sidechain positions
- This is what enzyme design .cst files do

The original script uses:
```python
sfx.set_weight("atom_pair_constraint", 1.0)
sfx.set_weight("angle_constraint", 1.0)
sfx.set_weight("dihedral_constraint", 1.0)
cst_io.add_constraints_to_pose(pose, sfx, True)
```

Not just coordinate constraints on fixed positions.
