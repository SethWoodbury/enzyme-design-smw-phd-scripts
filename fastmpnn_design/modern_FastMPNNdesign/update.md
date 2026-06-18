# FastMPNN Enzyme Design Pipeline Update Plan

Location: `/home/woodbuse/special_scripts/fastmpnn_design/modern_FastMPNNdesign`

This document is a detailed, agent-readable execution plan for fixing critical bugs and implementing improved sidechain flip logic. It incorporates the latest findings and user constraints, removes the concept of “chemical sensitivity,” and adds explicit gating based on protocol stage, chi convergence, and persistent constraint violations. It also specifies where to change code, what to log, and how to verify.

---

## 0) Operating Constraints (Must Follow)

- Do **not** remove any residues from `FLIPPABLE_SIDECHAINS`. All listed residue types remain eligible for flips.
- **Abolish** “chemical sensitivity” logic entirely. Remove `CHEMICALLY_SENSITIVE_FLIPS` usage and all gating that disables flips based on residue type.
- Do **not** flip early in the protocol. Sidechain flips must be gated by:
  - A protocol stage gate (no flips before step `N`).
  - Convergence of chi angles (stable over recent iterations).
  - Persistent poor CST scores (sustained over multiple iterations).
- If a 2D torsion constraint (e.g., `torsion_AB` from CST files) is ~180 degrees off, that **flags** the residue for flip **after** gates pass.
- Accept flips **only** if constraint scores improve significantly; otherwise revert.
- Debug output layout: **tracker outputs stay in the parent debug directory**, not in `structures/`.

---

## 1) Critical Bugs (Must Fix)

### BUG 1: Debug structures saved to wrong directory

**Symptom**: `structures/` subdirectory empty despite `--debug`.

**Root Cause**: `FastMPNNdesign` is constructed with `debug_structures_dir` but later overwritten with a different `debug_output_dir` based on `os.path.dirname(args.pdb)`.

**Exact Location**:
- `enzyme_design.py:1310-1318`

**Fix**:
- Keep tracker outputs in the parent debug dir (e.g., `debug_output_{pdbname}/`) and keep `structures/` for structure dumps only.
- Do **not** change `fmd.set_debug_output_dir` to the parent dir; it should remain `debug_structures_dir`.
- The tracker should write to the parent debug dir. That means adjust only the tracker output path.

**Implementation detail**:
- `debug_structures_dir` is created at `enzyme_design.py:989-993` as `<debug_output_dir>/structures`.
- Add/verify a separate parent debug directory variable (already `debug_dir`) and ensure:
  - `PipelineTracker(output_dir=debug_dir, ...)`
  - `fmd.set_debug_output_dir(debug_structures_dir)`
  - `print` lines clearly distinguish tracker dir vs structure dir

**Logging**:
- Log both paths:
  - `Structures directory: <debug_dir>/structures`
  - `Tracker output directory: <debug_dir>`

---

### BUG 2: CST scores always zero

**Symptom**: `all_cst=0.0`, all CSTs zero, yet RMSD to ref is high.

**Root Cause**: `score_design` uses `pyr.get_fa_scorefxn()` rather than the configured `sfx` with constraints.

**Exact Location**:
- `enzyme_design.py:778`

**Fix**:
- Replace `pyr.get_fa_scorefxn()` with the passed-in `sfx`.

**Logging**:
- None required beyond existing logging, but confirm that CST terms are non-zero in `.sc` file after fix.

---

## 2) Flip Logic Overhaul (No Chemical Sensitivity)

### Goal
Stop early, uninformed flips. Only flip when:
1. Protocol stage gate is passed.
2. Residue chi angles are converged.
3. Residue shows persistently poor CST scores.
4. Additional dihedral (2D `torsion_AB`) is ~180 degrees off from target.

### Core Design Principles
- **No chemical sensitivity**: remove `CHEMICALLY_SENSITIVE_FLIPS` and any gating logic tied to it.
- **Always eligible**: residues in `FLIPPABLE_SIDECHAINS` are eligible (no exclusions).
- **Evidence-based flip**: require convergence + persistent CST deviation + torsion_AB “flip signal”.

---

## 3) New Flip Gate Definitions

### 3.1 Protocol Stage Gate
- Add parameter: `flip_stage_gate` (default: 5).
- No automatic flip attempts before step `i >= flip_stage_gate`.

### 3.2 Chi Convergence Gate
- Track chi angles across recent iterations for each catalytic residue.
- Define convergence as:
  - `abs(chi_t - chi_{t-1}) <= chi_delta_thresh` for `k` consecutive steps.
- Recommended defaults:
  - `chi_delta_thresh = 5.0 degrees`
  - `convergence_window = 3 steps`

### 3.3 Persistent CST Gate
- Track per-residue CST totals across recent iterations.
- Define persistent poor CST if:
  - `cst_total > cst_threshold` for `m` consecutive steps.
- Recommended defaults:
  - `cst_threshold = 2.0 REU`
  - `cst_persistence_window = 3 steps`

### 3.4 Torsion_AB “Flip Signal” (2D Constraint Gate)
- If a residue has a `torsion_AB` constraint involving sidechain and ligand atoms, calculate deviation:
  - `delta = angular_difference(current_torsion_AB, target_torsion_AB)`
  - If `delta >= 120°` (or a tunable threshold near 180°), this indicates the residue is oriented “flipped”.
- Only use this signal if the constraint is a **2-body ligand–sidechain** torsion of type `torsion_AB`.
- If present and large, it **enables** flip attempt (not required for all residues).

---

## 4) File-by-File Implementation Plan

### 4.1 `enzyme_design.py`

**A) Fix CST scoring**
- **Line**: `~778`
- **Change**: `scoring_module.score_design(p, sfx, list(catres.keys()))`
- **Rationale**: Use the configured `sfx` with constraint weights.

**B) Fix debug paths**
- **Line**: `~1310-1318`
- **Target behavior**:
  - `tracker.output_dir` = parent debug dir
  - `fmd.debug_output_dir` = `debug_structures_dir`

**Implementation**:
- Compute parent debug dir in the same function that creates `debug_structures_dir`:
  - `debug_dir = f"debug_output_{pdbname}{suffix}"`
  - `debug_structures_dir = os.path.join(debug_dir, "structures")`
- Ensure tracker uses `debug_dir` and `fmd.set_debug_output_dir(debug_structures_dir)`.

**Logging**:
- Print both paths distinctly.

---

### 4.2 `design_protocol.py`

**A) Add flip gating parameters**

Add new instance variables with defaults (in `__init__` around lines `406-416`):
- `self.__flip_stage_gate = 5`
- `self.__flip_chi_delta_thresh = 5.0`
- `self.__flip_convergence_window = 3`
- `self.__flip_cst_persistence_window = 3`
- `self.__flip_cst_threshold = 2.0` (raise default)

**B) Add chi/CST history tracking**

Add a `FlipHistory` structure (new class or dict) to track per-residue:
- `chi_history[seqpos] -> list of last N chi angles (by chi index)`
- `cst_history[seqpos] -> list of last N CST totals`

Store at class level:
- `self.__flip_history = FlipHistory()`

**C) Update setter for flip config**

Update `set_catres_flip_config()` to include new parameters:
- `flip_stage_gate`
- `chi_delta_thresh`
- `convergence_window`
- `cst_persistence_window`

**D) Insert flip gating checks**

In the automatic flip block (around `design_protocol.py:1449-1467`):

Before calling `attempt_catres_sidechain_flips`, do:
1. Check protocol step `i` vs `self.__flip_stage_gate`.
2. Update history:
   - For each catalytic residue: record chi angles and CST total.
3. Evaluate gating per residue:
   - `converged = chi_converged(seqpos)`
   - `persistent_poor_cst = cst_persistent(seqpos)`
   - `torsion_flip_signal = torsion_ab_is_flipped(seqpos)`

Pass only the subset of residues that pass all gates to `attempt_catres_sidechain_flips`.

**Logging**:
- Per step, log summary:
  - `flip_gate: step i, eligible_residues=[...]`
  - For each residue filtered out, log reason: `not_converged`, `cst_not_persistent`, `stage_gate`, `no_torsion_flip_signal`.

---

### 4.3 `rosetta_utils.py`

**A) Remove chemical sensitivity**
- Remove or ignore `CHEMICALLY_SENSITIVE_FLIPS` and all logic in `attempt_catres_sidechain_flips` related to it.
- Ensure no residue-specific flip disablement.

**B) Support torsion_AB evaluation**

Add helper functions to compute torsion_AB deviation:
- `get_torsion_ab_target(cst_io, seqpos)`
- `compute_current_torsion_ab(pose, seqpos, ligand_seqpos, atom_map)`
- `torsion_ab_delta(current, target)`

Use `cst_io` from `FastMPNNdesign` to resolve relevant constraints for each catalytic residue. The CST file provides atom mapping; use this to compute torsion in the pose.

**C) Modify `attempt_catres_sidechain_flips` to accept residue subset**
- Accept a list of residues already filtered by gates (no chemical sensitivity checks inside).
- Keep flip accept/reject based on CST improvement and total score improvement (as currently done).

**Logging**:
- Log per residue:
  - `current_cst`, `new_cst`, `delta_cst`, `accept/reject`, and if reject then reason.

---

## 5) Torsion_AB Gate Logic (Based on CST Examples)

The CST file defines `torsion_AB` constraints involving ligand and sidechain atoms.

Use these steps:
1. Parse CST blocks via `cst_io` (already available in the protocol).
2. For each catalytic residue, identify any `torsion_AB` constraint that includes:
   - **2 atoms from ligand** and **2 atoms from sidechain**.
3. Compute target torsion from CST definition (`torsion_AB` value).
4. Compute current torsion using the mapped atoms in the pose.
5. If `abs(angular_difference(current, target)) >= 120°` (tunable), mark `torsion_ab_flip_signal=True`.

**Interpretation**:
- This large delta implies a 180-degree flip might align the sidechain with the ligand constraint.
- Only attempt flip after convergence + persistent CST is met.

---

## 6) Logging Requirements

### Global logging (stdout)
- On startup (debug mode):
  - `DEBUG: Structures directory = <debug_dir>/structures`
  - `DEBUG: Tracker directory = <debug_dir>`

### Per iteration, per step
- `Flip Gate Summary`:
  - step index
  - residues considered
  - residues eligible
  - reasons for ineligibility

### Flip execution
- For each attempted flip:
  - `seqpos`, `resname`, `current_cst`, `new_cst`, `delta_cst`
  - `accept/reject`
  - `torsion_ab_delta` if computed

---

## 7) Verification Checklist

1) Debug structures present:
```
ls debug_output_*/structures/
```
Expect: `stepXX_min_poseY.pdb`, etc.

2) Tracker outputs in parent directory:
```
ls debug_output_*/
```
Expect: `*_pipeline_report.json` and checkpoint files alongside `structures/`.

3) CST scores non-zero:
```
grep all_cst scores/*.sc
```
Expect: non-zero values when constraints violated.

4) Flip gating works:
- TRP does not flip early.
- Flips occur only after stage gate and convergence.
- Flip accepted only if CST improves.

5) RMSD to ref improves:
- `catres_subset_allatom_sc_rmsd_to_refpdb < 0.5 Å` for good designs.

---

## 8) Known Findings to Preserve in Documentation

Include these in `common_bugs_and_cautions.md` / `README.md`:

**Critical**
- `enzyme_design.py:778`: CST scoring bug from using `pyr.get_fa_scorefxn()`.
- `enzyme_design.py:1310-1318`: debug directory mismatch.

**Medium**
- Early flipping without convergence or CST persistence leads to incorrect packing.
- Add gating with stage gate + chi/CST history.

**Removed Concept**
- `CHEMICALLY_SENSITIVE_FLIPS` is abolished. No residue is gated by chemical sensitivity.

---

## 9) Agent Execution Plan (Subagents)

### Subagent 1: Debug Directory Fix

**File**: `enzyme_design.py`
- Fix tracker dir to parent debug dir
- Ensure `fmd.set_debug_output_dir(debug_structures_dir)` remains for structures
- Log both paths

**Verify**:
- Debug structures appear in `debug_output_*/structures/`
- Tracker outputs in parent `debug_output_*/`

### Subagent 2: CST Scoring Fix

**File**: `enzyme_design.py:778`
- Replace `pyr.get_fa_scorefxn()` with `sfx`

**Verify**:
- Non-zero CST scores in `.sc` outputs

### Subagent 3: Flip Logic Overhaul

**Files**:
- `design_protocol.py` (gating, history, stage gate)
- `rosetta_utils.py` (remove chemical sensitivity, torsion_AB gating helpers)

**Tasks**:
1) Add flip history tracking of chi + CST
2) Add gating logic in `design_protocol.py`
3) Remove chemical sensitivity logic in `rosetta_utils.py`
4) Implement torsion_AB gate

**Verify**:
- Flip attempts only after stage gate and convergence
- Flip attempts only for residues with persistent poor CST and torsion_AB deviation

### Subagent 4: Testing Setup

**Location**: `test_runs/`
- Create `test_quick.sh` with the user command
- Create `test_full.sh` for extended tests

### Subagent 5: Documentation

**Files**:
- `common_bugs_and_cautions.md`
- `README.md`

**Tasks**:
- Add bug entries for debug dir and CST scoring
- Document new flip gating design (no chemical sensitivity)
- Update configuration options and logging description

---

## 10) Open Questions (Resolved)

- Tracker outputs should stay in the parent debug dir, not `structures/`. (Confirmed)
- Chemical sensitivity should be abolished entirely. (Confirmed)
- Stage gate should incorporate torsion_AB deviations for CST-based gating. (Confirmed)

---

## 11) Exact Next Steps for Agents

1) Implement Bug 1 and Bug 2 fixes in `enzyme_design.py`.
2) Implement flip gating logic in `design_protocol.py`:
   - Add history tracking
   - Stage gate
   - Chi convergence + CST persistence
   - Build filtered residue list
3) Implement torsion_AB evaluation and remove chemical sensitivity in `rosetta_utils.py`.
4) Add logging for all gates and flip outcomes.
5) Update docs and add test scripts.

---

## 12) Expected Outcome

After implementing this plan:
- Debug structures appear in the correct directory.
- Constraint scores are non-zero and meaningful.
- Flips are rare, delayed, and justified by convergence + CST persistence + torsion_AB deviation.
- TRP no longer flips early; flips only occur when CST evidence supports it.

