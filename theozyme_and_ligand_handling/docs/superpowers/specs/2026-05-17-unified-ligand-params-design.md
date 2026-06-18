# Design: `ligands_to_params__UNIFIED.py`

**Date:** 2026-05-17
**Author:** Seth M. Woodbury (design assisted by Claude + Codex)
**Status:** Approved design — pending spec review

## 1. Purpose

A single, centralized script that extracts ligands from one or more PDB files and
produces Rosetta `.params` (+ PDB) files, with full control over **how ligands are
grouped**: kept separate, merged into one residue via intra-group pseudobonds, or
treated as conformers of each other.

It supersedes (by absorbing their logic; originals left untouched) and generalizes:

- `extract_ligands_from_SINGLE_pdb_and_create_PARAMS__MODERN.py` (current pipeline)
- `mol2_with_confs_to_params.py` (Rosetta wrapper + PDB merge)
- `make_FullyBonded_mol2_file_from_singleXYZ_ThatCanHave_multipleXYZinside.py` (bond-fix)
- `combine_atomlist_from_multipleXYZ_into_singleXYZ_lig.py` (conformer concatenation)
- `give_unique_atom_names_to_pdb_lig_v2.py` (atom-name uniquification)
- `check_if_ligand_3string_code_exists_in_rosetta.py` (Rosetta code collision check)

The MODERN script and all listed scripts remain on disk, unmodified, as fallbacks.

## 2. Core problem with the current script

The current bond-fix stitches **every** unbonded atom and **all** disconnected
fragments into one connected graph across the entire selection. Distinct ligands
(substrate, waters, metals) get fused into one fake "molecule" and one params file.
The new default must keep ligands **separate**, with bond-fixing scoped so pseudobonds
are only ever drawn *within* an intended group, never across distinct ligands.

## 3. Architecture: instance-first planner

Compute the complete plan of output residues and validate it **before** writing any
file. Stages, each an independently testable function:

1. **Parse / normalize args** — legacy flags + new flags; legacy translation.
2. **Read primary PDB** — all HETATM records with full metadata.
3. **Select** — filter by `--ligands_to_extract_via_3letter_code` (default: all HETATM).
4. **Exclude** — drop atoms matching `--ignore_codes` / `--ignore_residues`.
5. **Build instances** — group atoms into `LigandInstance` by `(resname, chain, resSeq, source_pdb)`.
6. **Apply grouping** — `GroupSpec`s → `OutputUnit`s; ungrouped instances become their own units.
7. **Assign + validate codes** — 3-char alnum, unique within run, Rosetta collision check.
8. **Per unit: write XYZ or ligand-only PDB** (per `--preserve_pdb_ligand_atom_order`).
9. **Per unit: Open Babel → MOL2.**
10. **Per unit: bond-fix scoped to that unit only** (intra-unit pseudobonds; standardize labels).
11. **Conformer assembly** — for conformer-mode units, validate identical composition across
    all collected instances, concatenate MOL2 blocks.
12. **Per code: run Rosetta `molfile_to_params.py`**, merge `{CODE}.pdb` onto `{CODE}_conformers.pdb`.

### Data model

| Object | Meaning |
|---|---|
| `PdbAtom` | One HETATM line + parsed metadata (resname, chain, resSeq, atom name, element, coords, original order, source PDB). |
| `LigandInstance` | Atoms sharing `(resname, chainID, resSeq, source_pdb)`. Default separation unit. |
| `GroupSpec` | User intent: code, resname set (or wildcard `*`), mode (`network`/`conformer`). |
| `OutputUnit` | One Rosetta residue to produce: code, source instances, ordered atoms, unique-name map, mode, artifact paths. |
| `ConformerSet` | A conformer-mode `OutputUnit`'s ordered list of per-instance MOL2 blocks. |

## 4. Grouping modes

Two explicit modes, selected by a token in the group spec (default `network`):

- **`network`** — all instances of the listed resnames (from the **primary PDB only**)
  are merged into ONE Rosetta residue. Atom names are made unique within the residue.
  Bond-fix draws pseudobonds (unbonded→nearest non-H, then fragment stitching) so the
  whole merged set is a single connected graph. Example: the entire water box → one
  `WAT` residue.
- **`conformer`** — each instance of the listed resnames is a **conformer** of the same
  residue. Instances are collected across the **primary PDB and all `--extra_conformer_pdbs`**.
  NO pseudobonds are drawn between instances. Each conformer is individually bond-fixed
  to be internally connected. All instances must share identical composition (see §6).
  Example: `SUB` from TS1.pdb + TS2.pdb → `SUB.params` with 2 conformers.
- **Ungrouped instance** — its own separate `OutputUnit` / params file.

## 5. CLI

### Preserved (identical legacy behavior)

`--input_single_pdb` (now = the primary PDB), `--ligands_to_extract_via_3letter_code`,
`--output_dir_for_params_stuff`, `--desired_ligand_3letter_code`,
`--preserve_pdb_ligand_atom_order`, `--stop_after_XYZ_is_made`,
`--stop_after_MOL2_is_made`, `--skip_bond_fix`, `--verbose`.

**Legacy translation:** `--desired_ligand_3letter_code CODE` with no `--group`/`--grouping_json`
is internally translated to a single wildcard group `CODE=*:network` — reproducing
today's lump-all-selected-HETATM-into-one-params behavior. Existing driver cells are
unaffected.

### New flags

| Flag | Meaning |
|---|---|
| `--group "CODE=RES[,RES...][:mode]"` (repeatable) | Define a group. `mode` ∈ `network` (default) / `conformer`. Resname list may be `*` (all selected). |
| `--grouping_json FILE` | JSON: `{ "CODE": {"resnames": [...], "mode": "network\|conformer"} }`. Combined with `--group` (must not conflict). |
| `--extra_conformer_pdbs PDB [PDB ...]` | Additional PDBs scanned **only** for resnames belonging to conformer-mode groups. |
| `--ignore_codes CODE [CODE ...]` | Drop all HETATM with these 3-letter codes before processing. |
| `--ignore_residues TOKEN [TOKEN ...]` | Drop specific residues. Token = `<ChainLetter><resSeq>`, e.g. `B12`, `Z999`. |
| `--auto_codes` | Override the multi-instance hard-stop: auto-generate unique 3-char codes for ungrouped multi-instance resnames. |
| `--allow_rosetta_code_collision` | Downgrade a Rosetta `residue_types.txt` code collision from fatal to warning. |

## 6. Default policy & validation

- **Ungrouped multi-instance resname → hard stop.** If a selected, non-ignored resname
  has >1 ungrouped instance, abort with: the offending resname(s) + instance keys, and
  concrete suggested fixes (`--group "X=RES:network"`, `--group "X=RES:conformer"`,
  `--ignore_codes RES`, or `--auto_codes`). Single ungrouped instance → its own params,
  no stop.
- **Conformer composition validation.** All instances in a conformer set must match on
  atom count, element sequence, atom names, and order. On mismatch → **hard fail** with
  a printed diff (e.g. `TS1/Z/999: 32 atoms` vs `TS2/Z/999: 31 atoms`). No silent reorder.
- **3-letter codes.** Exactly 3 alphanumeric chars; unique across the run; checked
  against Rosetta `residue_types.txt`. Collision is fatal unless `--allow_rosetta_code_collision`.
- **Atom-name uniqueness.** When a `network` group merges multiple instances, colliding
  names (`O`, `H1`, `H2` ×N) are deterministically renamed (element + counter) preserving
  atom order; a name-mapping file is emitted next to the params.

## 7. Edge cases

- **Single-atom metals (Zn).** Bond-fix is a no-op when a unit has no second heavy atom;
  emit a warning that `molfile_to_params.py` may be unsuitable for native Rosetta metals.
- **Waters never cross-bond by default.** Per-unit-only bond-fix prevents O–O / H bridges
  across separate waters. Cross-water pseudobonds happen *only* if the user explicitly
  puts waters in a `network` group; warn when fragment-stitching adds such bonds.
- **`--preserve_pdb_ligand_atom_order`.** Order is preserved; literal duplicate names are
  not — uniqueness wins when a network merge would collide names.
- **`--ignore_residues` ambiguity.** Resolved by using a dedicated flag with
  `<ChainLetter><resSeq>` tokens, separate from `--ignore_codes`.
- **Missing conformer instance.** If a conformer-mode resname is absent from an
  `--extra_conformer_pdbs` file, that's allowed (fewer conformers); a conformer set with
  <1 instance after collection is an error.
- **Empty selection.** Fail early reporting selected codes, input PDB, grouping summary.
- **Overlapping groups.** A resname assigned to >1 output code → reject. Wildcard `*`
  consumes all otherwise-ungrouped selected resnames; mixing `*` with specific groups
  means `*` takes the remainder.
- **`--stop_after_XYZ_is_made`.** Multi-output: write all planned XYZ, print per-code
  downstream commands, exit. In `--preserve_pdb_ligand_atom_order` mode (no XYZ stage),
  warn and treat as stop-after-ligand-PDB.
- **`--stop_after_MOL2_is_made`.** Generate all per-unit and conformer MOL2s, exit before
  Rosetta.
- **`chdir`.** Use absolute paths throughout; `chdir` into the output dir only around each
  Rosetta call; always restore cwd (try/finally).
- **Intermediate filename collisions.** Per-unit intermediates (XYZ/PDB/MOL2) are named
  with a stable slug = code + instance key, so same-resname units never overwrite.

## 8. Worked examples (target use cases)

**Keep everything separate, waters merged, cofactor merged:**
```
python ligands_to_params__UNIFIED.py \
  --input_single_pdb pte_kcx_hbond_TS1.pdb \
  --output_dir_for_params_stuff ./params \
  --group "WAT=HOH:network" \
  --group "COF=ZN2,OHX:network"
# => SUB.params (separate), CO2.params (separate),
#    WAT.params (all waters, one connected residue),
#    COF.params (zincs+hydroxide, one residue)
```

**Only SUB, as conformers from TS1 + TS2, ignore everything else:**
```
python ligands_to_params__UNIFIED.py \
  --input_single_pdb pte_kcx_hbond_TS1.pdb \
  --extra_conformer_pdbs pte_kcx_hbond_TS2.pdb \
  --ligands_to_extract_via_3letter_code SUB \
  --group "SUB=SUB:conformer" \
  --output_dir_for_params_stuff ./params
# => SUB.params with 2 validated conformers; nothing else
```

**Legacy (unchanged behavior):**
```
python ligands_to_params__UNIFIED.py \
  --input_single_pdb ..._lig_ZDW.pdb \
  --output_dir_for_params_stuff ./params \
  --desired_ligand_3letter_code ZDW \
  --preserve_pdb_ligand_atom_order --verbose
# => internally ZDW=*:network ; identical to MODERN.py output
```

## 9. Out of scope

- Conformer atom reordering / atom-name remapping across PDBs (hard-fail only for now;
  possible future `--reorder_conformers_by_name`).
- Replacing native Rosetta metal handling.
- Modifying or deleting any existing script.

## 10. Notes

- Working directory is not a git repository; the design doc is saved but not committed.
- `OPENBABEL_BIN` and `MOLFILE_TO_PARAMS_SCRIPT` path constants carried over from MODERN.

## 11. Implementation status (2026-05-17)

Implemented in `ligands_to_params__UNIFIED.py`; 40 tests in
`tests/test_ligands_to_params__UNIFIED.py` (full repo suite: 73 passed).
Two Codex review rounds applied. Post-review hardening beyond the original
spec:

- `molfile_to_params.py` is run with `--keep-names`, and planned atom names
  are written into the MOL2 *after* Open Babel (`apply_atom_names_to_mol2`),
  so uniquified/canonical names actually reach the final params (was cosmetic).
- Conformer compatibility is validated on the FINAL MOL2 (post Open Babel +
  bond-fix): atom count, element order, names, and bond topology must match
  conformer 1, since Rosetta builds the residue from conformer 1's topology.
- `assert_single_block` guards every Open-Babel output (verified: Open Babel
  emits one `@<TRIPOS>MOLECULE` block even for disconnected fragments, so a
  network water-box still becomes one residue via `connect_fragments`).
- Single-atom units (lone metals) warn instead of silently making bad params.
- Insertion codes (`iCode`) are part of the instance key; `--ignore_residues`
  grammar covers blank/numeric chain, signed resSeq, and insertion codes.
- `--stop_after_XYZ_is_made` also stops after the ligand-PDB in
  `--preserve_pdb_ligand_atom_order` mode.
- `--extra_conformer_pdbs` is de-duplicated by full instance key and never
  re-reads the primary PDB.
- Uniquified atom names are guaranteed ≤4 chars (PDB/Rosetta limit).
- Two-letter element inference from atom names when the element column is blank.
