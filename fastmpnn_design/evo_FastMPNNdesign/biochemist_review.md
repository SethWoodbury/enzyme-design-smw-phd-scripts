# Biochemist Review: fastMPNNdesign Pipeline

**Reviewer:** PyRosetta/Rosetta Expert
**Date:** 2026-01-16
**Scope:** Review of constraint strategy, relaxation protocol, histidine protonation, and metrics

---

## Executive Summary

The fastMPNNdesign pipeline is well-designed for iterative sequence design with ligand/metal coordination preservation. However, several areas require attention for optimal performance in metalloenzyme design, particularly around histidine protonation states, constraint weighting, and relaxation parameters.

**Critical Issues:**
1. No explicit handling of histidine protonation states for metal coordination
2. Coordinate constraint standard deviations are extremely tight (potentially too restrictive)
3. Missing Dunbrack rotamer probability metrics
4. Some scorefunction weights may need adjustment

---

## 1. Histidine Protonation Analysis

### Current State
The codebase defines histidine tautomer names in `constants.py` (lines 186-202):
- `HIS` (standard, proton on NE2)
- `HIS_D` (delta-protonated, proton on ND1)
- Recognition of AMBER/CHARMM naming conventions (HID, HIE, HIP, HSE, HSD, HSP)

The `METAL_COORDINATION_CUTOFF` is set to 2.5 A, which is appropriate.

### Issues Identified

**Critical:** The pipeline does not actively check or correct histidine protonation states when coordinating metals. When a histidine coordinates Zn2+ through NE2, that nitrogen should NOT have a proton - the proton should be on ND1 (making it HIS_D in Rosetta nomenclature).

**Current behavior:** The code detects metal contacts but does not verify or enforce correct protonation:
```python
# In contact_detection.py - only distance check, no protonation awareness
if dist <= metal_cutoff:
    contacts.append(Contact(...))
```

### Recommendations

1. **Add protonation state validation:**
   - After detecting a HIS-metal contact, check which nitrogen is coordinating
   - If NE2 coordinates metal: residue should be HIS_D (proton on ND1)
   - If ND1 coordinates metal: residue should be HIS (standard, proton on NE2)

2. **Expected coordination distances for Zn-His:**
   - Zn-NE2 or Zn-ND1: 2.0-2.2 A (optimal ~2.05 A)
   - Current `METAL_CONTACT_CUTOFF` of 2.6 A is appropriate for detection
   - However, constraint stdev of 0.005 A is extremely tight (see below)

3. **Add a function to suggest/enforce correct protonation:**
   ```
   detect_his_metal_coordination(pdb_path) -> List[Tuple[residue, coordinating_nitrogen, suggested_protonation]]
   ```

---

## 2. Constraint Strategy Analysis

### Current Default Values (from `constants.py`)

| Parameter | Current Value | Recommended | Notes |
|-----------|--------------|-------------|-------|
| `COORD_CST_STDEV` | 0.01 A | 0.05-0.1 A | Too tight, may cause strain |
| `METAL_CST_STDEV` | 0.005 A | 0.02-0.05 A | Way too tight for realistic sampling |
| `PRIMARY_CST_STDEV` | 0.01 A | 0.05-0.1 A | Reasonable for primary contacts |
| `SECONDARY_CST_STDEV` | 0.05 A | 0.1-0.2 A | Acceptable |
| `COORD_CST_WEIGHT` | 100.0 | 100-500 | Reasonable, depends on stdev |

### Issues Identified

**Issue 1: Constraint Standard Deviations Too Tight**

The current stdev values (especially `METAL_CST_STDEV = 0.005 A`) are extremely restrictive:
- 0.005 A = 0.05 nm = 50 pm
- This is tighter than typical crystallographic coordinate uncertainty
- Will create very high constraint scores and may prevent any meaningful relaxation
- The HarmonicFunc penalty = (distance/stdev)^2, so at 0.1 A deviation with stdev=0.005, penalty = 400!

**Issue 2: Scorefunction Weight Application (relax_runner.py lines 140-145)**

```python
sfxn.set_weight(ScoreType.cart_bonded, config.cart_bonded_weight)  # 0.5
sfxn.set_weight(ScoreType.coordinate_constraint, 100.0)  # Hardcoded!
sfxn.set_weight(ScoreType.atom_pair_constraint, 10.0)
```

The coordinate_constraint weight is hardcoded at 100.0, ignoring `config.coord_cst_weight`. This should use the config value.

**Issue 3: Distance Constraints Not Applied**

The `ConstraintSet` generates both coordinate and distance constraints, but in `relax_runner.py`, only coordinate constraints are applied:
```python
n_constraints = apply_coordinate_constraints(pose, constraint_set.coordinate_constraints)
```

Distance constraints from contacts are generated but never used during relaxation.

### Recommendations

1. **Adjust standard deviations:**
   ```python
   COORD_CST_STDEV = 0.1       # Allow 0.1 A flexibility (was 0.01)
   METAL_CST_STDEV = 0.05      # Metals need some flexibility too (was 0.005)
   PRIMARY_CST_STDEV = 0.1     # Primary contacts (was 0.01)
   SECONDARY_CST_STDEV = 0.2   # Secondary contacts (was 0.05)
   ```

2. **Use tiered constraint weights:**
   - Metal coordination: weight 500-1000, stdev 0.05 A
   - Primary contacts: weight 100-200, stdev 0.1 A
   - Secondary contacts: weight 50-100, stdev 0.2 A

3. **Fix the hardcoded weight:**
   ```python
   sfxn.set_weight(ScoreType.coordinate_constraint, config.coord_cst_weight)
   ```

4. **Apply distance constraints:**
   Add a function to apply `DistanceConstraint` objects as `AtomPairConstraint` in PyRosetta.

---

## 3. Relaxation Protocol Analysis

### Current Implementation (relax_runner.py)

**Scorefunction setup:**
- Base: `beta_jan25` (correct, modern scorefunction)
- `cart_bonded` weight: 0.5 (appropriate)
- `pro_close` disabled (correct for Cartesian)
- Corrections flag applied for beta scorefunctions (correct)

**FastRelax configuration:**
- Cartesian mode: enabled (correct)
- Minimizer: `lbfgs_armijo_nonmonotone` (correct)
- max_iter: 200 (reasonable)
- Cycles: 2 (default)

**MoveMap setup (lines 183-237):**
- Correctly freezes ligands and metals
- Uses mobile_radius to restrict movement
- Optional catres freezing

### Issues Identified

**Issue 1: Running FastRelax Multiple Times Instead of Cycles**

```python
relax.max_iter(200)
for i in range(n_cycles):
    relax.apply(pose)  # This runs the FULL FastRelax protocol each time
```

This actually runs the complete FastRelax protocol n_cycles times (each with internal cycles). This is:
- Computationally expensive
- Potentially good for convergence but may be overkill

The standard FastRelax already has internal cycles. Setting `relax.max_iter(200)` limits minimization iterations, not FastRelax cycles.

**Issue 2: No max_iter for FastRelax Cycles**

For proper control, use:
```python
relax.set_max_cycles(n_cycles)  # Controls FastRelax internal cycles
```
instead of a Python loop.

**Issue 3: Catres Backbone Movement**

Currently, catres backbone movement depends on `freeze_catres` parameter:
```python
movemap = create_movemap(
    ...
    freeze_catres=False,  # Line 515 - always False
    catres_residues=catres_set
)
```

This is set to `False`, meaning catres can move. For catalytic residue preservation, typically:
- **Chi angles:** Should be flexible (allow rotamer optimization)
- **Backbone:** Should be constrained or frozen

### Recommendations

1. **FastRelax cycles:**
   - 2-3 cycles is typically sufficient
   - Use `relax.set_max_cycles(n_cycles)` instead of external loop
   - Or if you want repeated relaxation, 2 external iterations with 3 internal cycles each

2. **Catres handling:**
   - Allow chi movement for all catres
   - Consider freezing backbone (BB) for catres, especially metal-coordinating ones
   - Or apply tight coordinate constraints to catres backbone atoms (N, CA, C)

3. **Recommended FastRelax setup:**
   ```python
   relax = FastRelax()
   relax.set_scorefxn(sfxn)
   relax.set_movemap(movemap)
   relax.cartesian(True)
   relax.min_type("lbfgs_armijo_nonmonotone")
   relax.max_iter(200)
   # Don't loop externally - let FastRelax handle cycles internally
   relax.apply(pose)
   ```

4. **Consider ramp_down_constraints:**
   FastRelax can ramp constraint weights. For tight constraint preservation, you may want to keep constraints constant or only slightly ramp.

---

## 4. Metrics Analysis

### Current Metrics (metrics.py)

**Geometry metrics computed:**
- Mean displacement from constraints
- Max displacement
- Percentage within 0.1 A tolerance
- Percentage within 0.5 A tolerance
- Ligand RMSD (optional)

**Sequence metrics computed:**
- Full sequence
- Identity to native
- Mutations
- Active site sequence

**Scoring metrics computed:**
- Total score
- fa_atr, fa_rep, fa_elec

### Missing Critical Metrics

**1. Dunbrack Rotamer Probability**

Not computed anywhere in the codebase. This is critical for evaluating whether designed residues adopt reasonable conformations.

Interpretation:
- P(rotamer) > 0.1: Good
- P(rotamer) > 0.01: Acceptable
- P(rotamer) < 0.01: Concerning, likely strained

**2. Cart_bonded Score Per Residue**

The total cart_bonded score is tracked in `RelaxResult`, but per-residue breakdown is not computed or reported.

Interpretation:
- Per-residue cart_bonded < 2: Good
- Per-residue cart_bonded 2-5: Acceptable
- Per-residue cart_bonded > 5: Concerning geometry

**3. Bond Length/Angle Deviations**

Not explicitly computed. Would be valuable for:
- Ligand geometry validation
- Metal coordination geometry
- Detecting strained residues

Interpretation:
- Bond length deviation < 0.02 A: Excellent
- Bond length deviation < 0.05 A: Good
- Bond length deviation > 0.1 A: Bad

- Bond angle deviation < 3 degrees: Excellent
- Bond angle deviation < 5 degrees: Good
- Bond angle deviation > 10 degrees: Bad

### Recommendations

1. **Add Dunbrack rotamer probability:**
   ```python
   from pyrosetta.rosetta.core.scoring import ScoreType
   from pyrosetta.rosetta.core.pack.dunbrack import DunbrackRotamer

   def compute_dunbrack_probability(pose, resnum):
       # Implementation to get rotamer probability
   ```

2. **Add per-residue cart_bonded scoring:**
   ```python
   def get_per_residue_cart_bonded(pose, sfxn):
       scores = {}
       for i in range(1, pose.total_residue() + 1):
           score = pose.energies().residue_total_energies(i)[ScoreType.cart_bonded]
           scores[i] = score
       return scores
   ```

3. **Track catres-specific metrics:**
   - Rotamer probability for each catres
   - Cart_bonded for each catres
   - Metal coordination distances and angles

---

## 5. Candidate Selection Criteria (filtering.py)

### Current Defaults

```python
FilterCriteria:
    max_mean_displacement = 0.5 A
    max_max_displacement = 1.0 A  # Note: 1.5 A used in orchestrator
    min_pct_within_tolerance = 50%
    max_cart_bonded_score = 5.0
```

### Analysis

**Mean displacement threshold (0.5 A):**
- Appropriate for active site preservation
- Consider tightening to 0.3 A for high-precision applications

**Max displacement threshold:**
- Inconsistent: FilterCriteria uses 1.0 A, orchestrator uses 1.5 A
- Recommend standardizing to 1.0 A (tighter is better)

**Percentage within tolerance (50%):**
- Reasonable minimum
- Consider requiring 80%+ for high-quality designs

**Cart_bonded threshold (5.0):**
- Per-structure total, not per-residue
- May be too permissive for larger structures
- Consider per-residue thresholds instead

### Recommendations

1. **Standardize thresholds:**
   ```python
   # Recommended defaults
   max_mean_displacement = 0.3  # Tight for catalytic sites
   max_max_displacement = 1.0   # No atom should move >1A
   min_pct_within_tolerance = 80.0  # Most atoms within 0.5A
   ```

2. **Add Dunbrack-based filtering:**
   ```python
   min_catres_rotamer_prob = 0.01  # All catres must have P > 0.01
   ```

3. **Add per-residue cart_bonded filter:**
   ```python
   max_per_residue_cart_bonded = 3.0  # No residue should exceed this
   ```

---

## 6. Recommended Default Parameters

### Constraint Configuration

```python
# constants.py updates
COORD_CST_STDEV = 0.1        # Was 0.01 - too tight
METAL_CST_STDEV = 0.05       # Was 0.005 - way too tight
PRIMARY_CST_STDEV = 0.1      # Was 0.01
SECONDARY_CST_STDEV = 0.2    # Was 0.05
COORD_CST_WEIGHT = 200.0     # Was 100.0 - increase for looser stdev
```

### Relaxation Configuration

```python
# config.py RelaxConfig updates
CART_BONDED_WEIGHT = 0.5     # Current value is good
FASTRELAX_CYCLES = 3         # Was 2, recommend 3
MOBILE_RADIUS = 12.0         # Was 10.0, allow slightly more
```

### Scorefunction Weights

```python
# For relax_runner.py create_scorefunction()
sfxn.set_weight(ScoreType.cart_bonded, 0.5)
sfxn.set_weight(ScoreType.coordinate_constraint, 200.0)  # Use config value!
sfxn.set_weight(ScoreType.atom_pair_constraint, 50.0)    # Was 10.0
sfxn.set_weight(ScoreType.pro_close, 0.0)  # Keep disabled for Cartesian
```

### Candidate Selection

```python
# For FilterCriteria
max_mean_displacement = 0.3
max_max_displacement = 1.0
min_pct_within_tolerance = 80.0
max_cart_bonded_score = 5.0
# Add:
max_catres_displacement = 0.2
min_catres_rotamer_prob = 0.01
```

---

## 7. Summary of Recommended Code Changes

### High Priority

1. **Fix hardcoded constraint weight** in `relax_runner.py` line 141
2. **Increase constraint stdev values** in `constants.py`
3. **Add distance constraint application** during relaxation
4. **Add Dunbrack rotamer probability metric**
5. **Add histidine protonation validation** for metal coordination

### Medium Priority

1. **Standardize displacement thresholds** between FilterCriteria and orchestrator
2. **Add per-residue cart_bonded scoring**
3. **Consider catres backbone constraints** for metal-coordinating residues
4. **Fix FastRelax cycle handling** (use internal cycles vs external loop)

### Low Priority (Enhancements)

1. Add bond length/angle deviation metrics
2. Add metal coordination angle analysis (tetrahedral, octahedral geometry)
3. Add visual reporting of constraint satisfaction
4. Add catres-specific scoring summary in final output

---

## 8. Testing Recommendations

After implementing changes, validate with:

1. **Metal coordination test case:**
   - Zn2+ with 4 coordinating residues (typical Zn-finger or enzyme active site)
   - Verify His protonation states
   - Check Zn-N distances after relaxation (~2.0-2.2 A)

2. **Geometry preservation test:**
   - Relax known structure, measure RMSD to crystal structure
   - Should achieve < 0.3 A backbone RMSD for active site
   - All catres rotamers should have P > 0.01

3. **Constraint satisfaction test:**
   - Apply constraints, measure post-relax displacements
   - Mean displacement < 0.2 A expected with recommended settings

---

*End of Review*

---

## 9. Initial Test Results (Added 2026-01-16)

### First Relaxation Analysis

After running the first comprehensive test with:
- ref_pdb constraints (ligand-aligned)
- 3 histidines correctly identified as HIS_D (A13, A15, A203)
- 29 catres-catres contacts (13 H-bonds, 10 salt bridges, 6 pi-stacking)
- Ligand perfectly frozen (0.0 Å displacement)

**Catres RMSD Results (from first relaxed structure):**
- All-atom RMSD: 0.904 Å
- Sidechain RMSD: 0.448 Å (good!)
- Mean displacement: 0.197 Å

**Issues observed:**
- Some secondary sphere catres (A19, A21, A30) moved significantly (2-3 Å)
- High constraint score due to tight stdev (0.01 Å in test 1)

**Tests in progress:**
- Test 1: Old stdev (0.01), coord_weight 100 - baseline
- Test 2: New stdev (0.1), coord_weight 200, 3 cycles, 8 candidates
- Test 3: New stdev (0.1), allow_catres_bb, larger mobile_radius (15 Å)
- Test 4: New stdev (0.1), coord_weight 500, cart_bonded 2.0, smaller catres subset (1-11)

**Expectations:**
- Tests 2-4 should show better constraint satisfaction with relaxed stdev
- Test 3 should show lower catres RMSD with BB movement allowed
- Test 4 should show best primary catres geometry with focused constraints

---

## 10. Bug Fix: Secondary Catres Movement (2026-01-16 02:10)

### Issue Identified

Test 1 results showed secondary catres (A19 PHE, A21 TYR, A30 LEU) moving 2-3 Å during relaxation, despite being in the catres_subset.

**Root Cause:** These residues don't have direct ligand contacts (no atoms within 4.2 Å of ligand) and don't form hydrogen bonds or salt bridges with other catres. Therefore, the constraint system generated NO constraints for them.

### Fix Implemented

Added `generate_catres_sidechain_constraints()` function in `constraints.py` that:
1. Generates coordinate constraints for ALL catres sidechain heavy atoms
2. Uses the existing constraint list to avoid duplicates
3. Applies a moderate stdev (SECONDARY_CST_STDEV = 0.2 Å) for non-contact atoms

**Before fix:** 62 coordinate constraints (24 ligand + 38 contacts)
**After fix:** 136 coordinate constraints (24 ligand + 38 contacts + 74 catres sidechains)

### Test 5: Validation with Catres Sidechain Fix

Running test 5 with the fix to validate:
- coord_cst_weight: 200
- coord_cst_stdev: 0.1 Å
- All catres sidechains now constrained

**Expected improvement:**
- A19, A21, A30 should have displacement < 0.5 Å instead of 2-3 Å
- Overall mean displacement should decrease
- Cart_bonded score should be reasonable (< 100) instead of 1700+

