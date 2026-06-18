# Design Spec — `prepare_PDB_structure_into_theozyme.py`

Author: Seth M. Woodbury (woodbuse@uw.edu), David Baker Lab, UW
Date: 2026-05-16
Status: APPROVED design, pending spec review → implementation plan

## 1. Purpose

A single one-stop script that consolidates and generalizes the theozyme/ligand
PDB-preparation tooling. Replaces the multi-script wrapper
(`prepare_PDB_structure_into_theozyme__MAIN/__OLD/__STEP1/__STEP2`) and folds in
the *logic* of `fragment_maximal_AA_subset_theozyme_into_manyPDBs__MAIN.py`,
`renumber_pdb.py`, `find_nonCCD_lig_codes.py`, and
`check_if_ligand_3string_code_exists_in_rosetta.py` (those four remain as
standalone scripts; only the `prepare_*` family is archived to `backup/`).

No subprocess calls — STEP1/STEP2 logic is inlined as pure functions.

## 2. Core principle: preserve-by-default

Run with only `--input_pdb` and `--output_pdb_path` → output is functionally
identical to input. Specifically, by default the script does NOT:

- strip protein/ligand hydrogens
- alter partial charges (B-factor cols 61–66) or formal charges (cols 77–80, e.g. `N1-`)
- merge HETATM ligands into one residue
- blank segID cols 72–75
- strip residue insertion codes
- drop or reorder records (TER/CONECT/MASTER/END/CRYST1/HEADER/LINK preserved)
- regenerate or reorder existing REMARK 666
- remove any existing REMARK lines

Every legacy/destructive behavior is an explicit opt-in flag. Acceptance test:
`pte_kcx_hbond_TS1.pdb` (already a theozyme with REMARK 665+666, partial+formal
charges, H atoms) run with no flags → semantically unchanged round-trip.

## 3. Architecture (single file, library-structured)

Pure functions; I/O isolated to `read_pdb`, `write_pdb`, and the two ligand
checks. Components:

1. `parse_cli()` / `validate_options()` — mutual-exclusion & ambiguity checks.
2. `read_pdb_lines()` / `write_pdb_lines()` — preserve raw order & unknown records.
3. Column helpers: `pad_line()`, `get_field()`, `replace_field()` (exact-length),
   `atom_name_raw()` (never `.strip()`-compare where ` CA `≠`CA`).
   Never reconstruct full ATOM/HETATM lines except for newly created atoms.
4. `scan_structure(lines)` → indexes: atoms, residues (chain,resname,resseq,icode),
   ligands (HETATM), ncAAs, hydrogens, partial-charge presence, formal-charge
   presence, CONECT, MASTER, existing REMARK 665/666 (possibly non-contiguous),
   multi-MODEL detection, serial/resSeq overflow detection, altLoc/blank-chain.
5. `WarningLog` — prints `[WARN]` immediately AND stores for the final SUMMARY.
6. ncAA transforms (ported from MAIN, behavior unchanged):
   `frag_ncaa()`, `convert_ncaa_hetatm_to_atom()` / revert,
   `add_CA_to_labeled_frag(lines, ca_cb_bond_length=1.53, wl=None)` (wl MUST be
   passed as keyword arg from `main`), `protect_sidechain_polarH()` / revert.
   `NONCANONICAL_AA_MAP`, `PROTEIN_RESNAMES`, `PROTECTED_POLAR_H_MAP` retained as
   editable top-of-file dicts.
7. `filter_structure()` — residue & ligand keep/throw (generalized fragment).
8. `apply_legacy_cleaning()` — opt-in: strip protein H, blank segID,
   strip partial charges, strip formal charges, merge ligands.
9. `remark666_manager()` — detect/parse/preserve/generate/filter/reorder/reindex
   + ensure the required REMARK 665 header pair.
10. `connectivity_repair()` — prune dangling CONECT, update MASTER on atom-count change.
11. `renumber_atoms()` — opt-in, last structural mutation; old→new serial map
    drives CONECT/MASTER updates.
12. `check_ligand_codes()` — always-on, non-fatal CCD + Rosetta checks.
13. `emit_summary()` — single delimited end-of-run warning block.

## 4. Canonical pipeline order

1. `read_pdb_lines` + `scan_structure` (before any destructive transform).
2. `validate_options`.
3. (opt) insertion-code strip — early; residue keys/filters/REMARK matching
   depend on final residue IDs; detect resseq collisions and warn.
4. ncAA transforms — fragment/protect/leave before H-strip & merge;
   `add_CA_to_labeled_frag` and intelligent H-strip need H present.
5. (opt) residue/ligand filtering — before REMARK 666 so stale anchors drop.
6. (opt) legacy cleaning — capture H/charge counts BEFORE stripping; then strip
   H / blank segID / strip partial / strip formal / merge ligands.
7. REMARK 666 manager — after final residue & ligand identity known.
8. (opt) `renumber_atoms` — last structural mutation (remaps serials + CONECT).
9. `connectivity_repair` — always; runs AFTER renumber so it prunes dangling
   CONECT and recounts MASTER (numCoord cols 51-55, numConect 41-45) on the
   FINAL atom set, satisfying §9 even without `--renumber_atoms`.
10. `check_ligand_codes` — after final ligand resnames known.
11. `write_pdb_lines` + `emit_summary`.

## 5. REMARK 666 / 665 rules (user convention — mandatory)

- Required header pair, emitted verbatim whenever 666 lines are (re)written and
  ensured present (added if missing, never duplicated):

```
REMARK 665 REMARK 666 = Rosetta enzyme-matcher catalytic-motif anchors
REMARK 665 fmt: REMARK 666 MATCH TEMPLATE <tCH tNAME tRESI> MATCH MOTIF <mCH mRESN mRESI IDX VAR>
```

- Existing valid REMARK 666 are preserved & reused; rewritten ONLY when:
  filtering removed an anchored residue, reorder requested, ligand merge changes
  the template token, exclusion list applies, or `--force_regenerate_remark666`.
- **Residue removal → 666 adjustment (mandatory):** when residue/ligand
  filtering (or ncAA transforms) removes a residue that has a REMARK 666 line,
  that line is dropped, AND all remaining 666 lines are re-emitted with
  **contiguous IDX renumbering** (1..N in their final order) — including the
  case where the removed residue sat in the *middle* of the 666 block, so no IDX
  gaps remain. Each dropped anchor is logged and listed in SUMMARY.
- **Exclusion list:** `--remark666_exclude_residues A131 A254 ...` suppresses
  REMARK 666 lines for the listed residues even though the residues themselves
  remain in the structure (atoms untouched). Applies to BOTH fresh generation
  and preserve/reindex of existing lines; triggers contiguous IDX renumbering of
  the survivors. Excluded residues are logged + summarized.
- **Partial-coverage edge case:** input has REMARK 666 for some protein
  motif residues but not all. Behavior: preserve all existing valid lines, then
  `[WARN]` listing exactly which protein residues currently lack a 666 line
  (partial coverage). Default = warn only (preserve-by-default, do not silently
  fabricate anchors). With `--complete_remark666`, generate the missing lines
  for the uncovered protein residues (honoring `--remark666_exclude_residues`),
  merge with the preserved set, then reindex IDX contiguously. Mixed/duplicate
  or malformed existing 666 entries are warned and de-duplicated by
  (chain,resseq).
- Handles non-contiguous 666 blocks (legacy STEP2 assumed a single block).
- Fresh generation only when none exist (or forced/completed). Template ligand
  token: a single ligand is auto-inferred; ambiguous multi-ligand requires
  `--remark666_template_ligand CODE` (else hard error with clear message);
  zero ligands → default token `"LIG"` with a `[WARN]` (non-fatal, no die).
- All non-666 REMARK lines untouched unless `--clean_remarks` (which drops all
  REMARK lines not part of the 665/666 canonical pair while preserving 665+666).
- **Exact-pair 665 validation:** `remark665_header_present` (and `trigger_g` in
  `remark666_manager`) requires BOTH canonical `R665_HEADER` strings, each present
  exactly once, with no other `REMARK 665 ... REMARK 666` lines.  Two copies of
  header line 0 with line 1 missing = invalid → trigger fires → header normalized.
- `--remark666_residue_front_order`: listed order, placed first.
- `--remark666_residue_back_order`: **listed order** (NOT reversed — deliberate
  change from legacy STEP2; one-line log notes it), placed last.
- Front/back residue not found → `[WARN]` + skip (non-fatal; legacy hard-exited).

## 6. Default layout

Default preserves the input's original record layout/order. The theozyme layout
(REMARK → REMARK 666 → ATOM → TER → HETATM → TER) is applied only with
`--theozyme_layout`, which is auto-enabled when `--merge_ligands_as` is used.

## 7. Charge / column safety

- Cols 61–66 (B-factor = partial charge) and 77–80 (element+formal charge) are
  read-only unless `--strip_partial_charges` / `--strip_formal_charges`.
- Element detection must not blindly read `line[76:78]` (would misparse `N1-`);
  parse element vs charge sub-fields separately, fall back to atom-name element
  guess only when col 77–78 absent.
- All lines padded to ≥80 chars before any field replacement.
- Atom-name field compared positionally, never via `.strip()` where the 4-char
  justification is semantically meaningful (Cα ` CA ` vs calcium `CA  `).
- Any operation that changes H count, charges, atom count, or atom names is
  logged and surfaced in SUMMARY (incl. merge-rename = potential params break).

## 8. Generalized features

- **Filtering** (keyed off real ATOM/HETATM records, not REMARK):
  `--residues_to_keep` XOR `--residues_to_throw_away`;
  `--ligands_to_keep` XOR `--ligands_to_throw_away` (cut ligands too).
  Selectors accept `A55` (chain+resseq) and `CHAIN:RESNAME:RESSEQ` forms.
- **Waters:** `HOH` is a NORMAL HETATM ligand by default — subject to the
  ligand keep/throw selectors and to merge, exactly like any other ligand.
  `--preserve_waters` opts back into special handling: HOH is always kept
  (bypasses ligand filters) and is never merged. (No hard-coded `HOH`
  exceptions anywhere; the exception is flag-gated.)
- **Ligand merge** (opt-in): `--merge_ligands_as CODE`
  `[--merged_ligand_chain Z] [--merged_ligand_resseq 999]`
  `[--merge_only <selectors>]`. Reuses STEP1 unify logic; preserves cols 27+
  (coords/occupancy/bfactor/element/charge) intact. Merges all non-`HOH`
  HETATM by default; merges `HOH` too unless `--preserve_waters`.
- **Renumber** (opt-in): `--renumber_atoms` — group by (chain,resseq,icode) in
  file order, rewrite ONLY serial cols 7–11, update CONECT via old→new map,
  update MASTER. Drops `renumber_pdb.py`'s unsafe col-17 UNL hack,
  chain-ignoring, and REMARK discarding.
- **Ligand-code checks** (always-on, non-fatal): for each distinct final ligand
  resname → CCD (`https://files.rcsb.org/ligands/view/{code}.cif`, 404=missing,
  200=exists, error/offline=unknown) + Rosetta `residue_types.txt` scan
  (filename / standalone / loose-`.params` match). Status-returning helpers,
  never `sys.exit`. Path via `--rosetta_residue_types` (default the standard
  `/net/software/rosetta/.../fa_standard/residue_types.txt`; missing → unknown).
  Results → SUMMARY (`exists` / `missing` / `unknown`).
  Three match tiers mirror the original
  `check_if_ligand_3string_code_exists_in_rosetta.py`: (1) filename `CODE.params`
  → `present`; (2) standalone `\bCODE\b` → `present`; (3) loose `.params`-line
  containing CODE anywhere without a stronger match → `absent` but a distinct
  non-fatal `[WARN]` with SUMMARY category `rosetta_loose`. Never `sys.exit`.

## 9. Edge cases / guards

Serial >99999, resSeq >9999 → warn; multi-MODEL → warn; blank chainID, altLoc,
insertion-code collisions on strip → warn; CONECT/MASTER staleness after any
add/remove → repaired (CONECT pruned, MASTER recounted) even without
`--renumber_atoms`; short lines padded; ligand code not 3 chars → warn;
mutually exclusive ncAA flags rejected (`protect` vs `leave` vs `frag`); merge +
fresh-666 multi-ligand ambiguity resolved per §5; partial REMARK 666 coverage,
mid-block anchor removal with IDX reindex, and per-residue 666 exclusion all
handled per §5; `--remark666_exclude_residues` referencing an absent residue →
`[WARN]` + skip.

## 10. CLI surface

Required: `--input_pdb`, `--output_pdb_path`.

REMARK 666: `--force_regenerate_remark666`, `--complete_remark666`,
`--remark666_exclude_residues [A131 ..]`, `--remark666_template_ligand`,
`--remark666_template_chain` (default `X`), `--remark666_template_resi`
(default `0`), `--remark666_residue_front_order [..]`,
`--remark666_residue_back_order [..]`, `--clean_remarks`.

Filtering: `--residues_to_keep`, `--residues_to_throw_away`,
`--ligands_to_keep`, `--ligands_to_throw_away`, `--preserve_waters`.

ncAA: `--frag_ncAA_into_cAA_plus_lig [..]`, `--protect_ncAA_from_ligandization`,
`--leave_ncAA_as_ATOM`, `--add_CA_to_labeled_frag`,
`--protect_sidechain_polarH [..]`, `--disable_intelligent_hstrip`.

Legacy opt-ins: `--strip_insertion_codes`, `--strip_protein_hydrogens`,
`--blank_segid`, `--strip_partial_charges`, `--strip_formal_charges`,
`--merge_ligands_as`, `--merged_ligand_chain`, `--merged_ligand_resseq`,
`--merge_only`, `--theozyme_layout`.

Renumber/checks/debug: `--renumber_atoms`, `--rosetta_residue_types`,
`--ccd_timeout` (default 4.0), `--no_ligand_code_checks` (escape hatch),
`--verbose`.

(No `--keep_precleaned_pdb`: the inlined single-pass design has no subprocess
intermediate file, unlike the old wrapper.)

## 11. Deliverables

1. `prepare_PDB_structure_into_theozyme.py` (this script).
2. Updated notebook driver cell in Seth's existing style (vars → build `cmd`
   list → print), supporting writing the assembled command to a `/tmp/*.txt`
   for paste-in.
3. `backup/` move of `prepare_*__MAIN/__STEP1/__STEP2.py` plus the already
   deprecated `prepare_*__OLD.py` (archived only; no logic mined from `__OLD`).
4. Acceptance run on `pte_kcx_hbond_TS1.pdb` (no-flag round-trip + a flagged run).
5. Post-implementation Codex review of the finished script.
