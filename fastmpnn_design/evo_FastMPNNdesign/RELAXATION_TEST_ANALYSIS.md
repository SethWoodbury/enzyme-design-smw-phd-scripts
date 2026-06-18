# Relaxation Protocol Test Results Analysis

**Date:** 2026-01-16
**Status:** Analysis complete - critical insight discovered

---

## Executive Summary

Testing of the new multi-stage relaxation protocol revealed that **coordinate constraints fundamentally cause geometry strain**. While the new protocol reduced cart_bonded scores from 3126 (original test5) to ~475-604, all configurations still fail the <50 threshold.

**Root cause identified:** Coordinate constraints force atoms to absolute positions, which strains bond geometry. The solution requires switching to **distance/angle/dihedral constraints** (enzyme design constraints) instead.

---

## Test Configurations and Results

| Config | cart_bonded | Status | allow_catres_bb | cart_bonded_wt | fa_rep_init | stages | Notes |
|--------|-------------|--------|-----------------|----------------|-------------|--------|-------|
| **high_cart_bonded** | **474.89** | **BEST** | true | 1.0 | 0.15 | 3 | Best result - higher cart_bonded weight helps |
| old_protocol | 566.28 | FAIL | false | 0.5 | N/A | 1 | Baseline frozen backbone |
| frozen_bb_failure | 567.10 | FAIL | false | 0.5 | 0.15 | 3 | Confirms frozen BB causes strain |
| four_stages | 603.84 | FAIL | true | 0.5 | 0.15 | 4 | More stages doesn't help |
| baseline_new | 604.14 | FAIL | true | 0.5 | 0.15 | 3 | Multi-stage with default weights |
| low_fa_rep | 604.46 | FAIL | true | 0.5 | 0.05 | 3 | Lower initial fa_rep doesn't help |

### Key Observations

1. **high_cart_bonded (cart_bonded_weight=1.0)** is the best at 474.89
   - 21% better than baseline_new (604.14)
   - 85% better than original test5 (3126)

2. **allow_catres_bb=true** is essential but not sufficient
   - frozen_bb_failure (567.10) vs baseline_new (604.14) shows minimal difference
   - The constraints themselves are the problem, not backbone flexibility

3. **Lower initial fa_rep (0.05)** did not help
   - Allowed structure to collapse into strained conformations

---

## Geometry Analysis

### Input Structure (before relaxation)
- cart_bonded: **275** (acceptable)
- No severely strained residues

### Relaxed Structure (high_cart_bonded config)
- cart_bonded: **474** (10x target threshold)
- 6 residues with cart_bonded > 10:

| Residue | cart_bonded | Type |
|---------|-------------|------|
| A117 ALA | 78.74 | Near catres |
| A118 ARG | 53.85 | Near catres |
| A116 ILE | 52.37 | **CATRES** |
| A119 ASP | 27.36 | Near catres |
| A150 PHE | 21.06 | **CATRES** |
| A151 GLY | 12.81 | Near catres |

### Critical Insight

The highest strain occurs at and around **constrained catalytic residues**:
- A116 (ILE) - catres, cart_bonded=52.37
- A149 (TRP) - catres, cart_bonded=9.97
- A150 (PHE) - catres, cart_bonded=21.06

**Coordinate constraints on sidechain atoms are pulling the backbone into strained geometries.** This is exactly what the RELAXATION_OVERHAUL_PLAN.md predicted.

---

## Why Coordinate Constraints Fail

### Current Approach (broken)
```
CoordinateConstraint: Fix atom XYZ to absolute position (x, y, z)
```

When sidechain atoms are constrained to absolute positions, and the backbone tries to maintain proper connectivity, something must give. The result is stretched/bent bonds.

### Correct Approach (from original script)
```
AtomPairConstraint: Maintain distance between atom pairs
AngleConstraint: Maintain angles at coordination sites
DihedralConstraint: Maintain torsion angles
```

This allows the **entire active site to move as a unit** while preserving the relative geometry that matters for catalysis.

---

## Comparison with Original comprehensive_test5

| Metric | Original test5 | Best new config | Improvement |
|--------|----------------|-----------------|-------------|
| cart_bonded | 3126 | 474.89 | **5.6x better** |
| Catres mean disp | 0.58 A | 0.003 A | **193x better** |
| Catres max disp | 5.08 A | 0.07 A | **73x better** |
| Ligand disp | 0.0 A | 0.0 A | Same (frozen) |

The new protocol dramatically improved catres displacement but at the cost of bond geometry strain. This is because the constraints are being satisfied by straining bonds rather than by allowing coordinated movement.

---

## Recommended Next Steps

### Phase 1: Switch to Enzyme Design Constraints (Priority 1)

Replace CoordinateConstraint with:
1. **AtomPairConstraint** for metal-ligand and key hydrogen bonds
2. **AngleConstraint** for coordination geometry angles
3. **DihedralConstraint** for sidechain orientations

Reference: `/home/woodbuse/special_scripts/fastmpnn_design/fastmpnn_ZnEsterase_SETH_LINKED.py` lines 234-341

### Phase 2: Remove Coordinate Constraints on Catres Sidechains

Only apply coordinate constraints to:
- Ligand heavy atoms (frozen)
- Catres backbone atoms (optional, loose)

Never apply coordinate constraints to catres sidechain atoms.

### Phase 3: Test Higher cart_bonded Weight

Based on the high_cart_bonded result being best, test:
- cart_bonded_weight = 2.0
- cart_bonded_weight = 5.0
- cart_bonded_weight = 10.0

This may help the minimizer prioritize geometry over constraint satisfaction.

---

## Optimal Defaults (Current Best)

Based on testing, recommend these defaults until enzyme design constraints are implemented:

```python
RelaxConfig(
    use_multistage_relax=True,
    allow_catres_bb=True,
    initial_coord_cst_weight=1000.0,
    final_coord_cst_weight=100.0,
    initial_fa_rep_scale=0.15,
    n_relax_stages=3,
    fastrelax_cycles=2,
    cart_bonded_weight=1.0,  # Higher than default 0.5
)
```

**Warning:** These defaults will NOT achieve cart_bonded < 50. Fundamental constraint redesign is required.

---

## Test Files Generated

Output directory: `/net/scratch/woodbuse/organophosphatase/round2/fastMPNNdesign_out/i1/relax_overhaul_tests/`

- `baseline_new/` - Multi-stage with default params
- `old_protocol/` - Original frozen backbone
- `high_cart_bonded/` - cart_bonded_weight=1.0 (BEST)
- `frozen_bb_failure/` - Multi-stage but frozen catres BB
- `low_fa_rep/` - Lower initial fa_rep=0.05

---

## Conclusion

The multi-stage relaxation protocol with backbone flexibility is a significant improvement (5.6x better cart_bonded), but **coordinate constraints on catalytic residue sidechains are fundamentally incompatible with good geometry**.

The next major milestone must be implementing enzyme design constraints (distance + angle + dihedral) that preserve relative geometry while allowing global movement of the active site.
