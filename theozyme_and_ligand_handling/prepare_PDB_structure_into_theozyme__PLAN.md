# prepare_PDB_structure_into_theozyme.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one preserve-by-default PDB→theozyme preparation script that
inlines the old STEP1/STEP2 logic and folds in residue/ligand filtering,
renumbering, and CCD/Rosetta ligand-code checks, per the approved DESIGN spec.

**Architecture:** Single library-structured file
`prepare_PDB_structure_into_theozyme.py`. Pure functions operate on a list of
fixed-width PDB line strings; I/O isolated to read/write + two ligand checks.
Pipeline runs the order fixed in DESIGN §4. Tests are pytest, data-driven from
small synthetic PDBs plus a no-flag round-trip of the real `pte_kcx_hbond_TS1.pdb`.

**Tech Stack:** Python 3, stdlib (`argparse`, `re`, `math`, `string`,
`urllib`/`requests`), `pytest` 9.

**Environment note:** This directory is NOT a git repo. Replace every "Commit"
TDD step with a **Checkpoint**: run the full suite
(`python3 -m pytest tests/ -q`) and confirm green before moving on. Do not run
`git init` here (shared scripts dir).

**Spec:** `/home/woodbuse/special_scripts/theozyme_and_ligand_handling/prepare_PDB_structure_into_theozyme__DESIGN.md`

---

## File Structure

- Create: `/home/woodbuse/special_scripts/theozyme_and_ligand_handling/prepare_PDB_structure_into_theozyme.py`
  Single one-stop script (user requirement: ONE script). Internally sectioned:
  CONSTANTS/DICTS · column helpers · `scan_structure` · `WarningLog` · ncAA
  transforms · `filter_structure` · `apply_legacy_cleaning` · `remark666_manager`
  · `connectivity_repair` · `renumber_atoms` · `check_ligand_codes` ·
  `build_pipeline`/`main`.
- Create: `/home/woodbuse/special_scripts/theozyme_and_ligand_handling/tests/test_prepare_PDB_structure_into_theozyme.py`
  All pytest tests + inline synthetic-PDB fixtures.
- Create: `/home/woodbuse/special_scripts/theozyme_and_ligand_handling/tests/conftest.py`
  `sys.path` shim so the hyphen-free module imports.
- Modify (final task only): move `prepare_PDB_structure_into_theozyme__MAIN.py`,
  `__OLD.py`, `__STEP1__cleanPDB_and_addREMARK666.py`,
  `__STEP2__reorder_REMARK666_lines.py` into `backup/`.

The script must be importable without side effects (all work under
`def main()` + `if __name__ == "__main__":`).

---

## Conventions used by every task

- PDB columns are 0-based Python slices: record `[0:6]`, serial `[6:11]`,
  atom-name `[12:16]`, altLoc `[16]`, resName `[17:20]`, chainID `[21]`,
  resSeq `[22:26]`, iCode `[26]`, x `[30:38]`, y `[38:46]`, z `[46:54]`,
  occupancy `[54:60]`, tempFactor/partial-charge `[60:66]`, element `[76:78]`,
  charge `[78:80]`.
- Lines are stored WITHOUT trailing newline internally; writer re-adds `\n`.
- `replace_field` requires exact length and pads the line to ≥80 first.

---

### Task 1: Test scaffold + importable skeleton

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_prepare_PDB_structure_into_theozyme.py`
- Create: `prepare_PDB_structure_into_theozyme.py`

- [ ] **Step 1: Write the failing test**

`tests/conftest.py`:
```python
import os, sys, importlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_mod():
    return importlib.import_module("prepare_PDB_structure_into_theozyme")
```

In `tests/test_prepare_PDB_structure_into_theozyme.py`:
```python
from conftest import load_mod

def test_module_imports_without_running():
    mod = load_mod()
    assert hasattr(mod, "main")
    assert hasattr(mod, "PROTEIN_RESNAMES")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/woodbuse/special_scripts/theozyme_and_ligand_handling && python3 -m pytest tests/ -q`
Expected: FAIL (ModuleNotFoundError: prepare_PDB_structure_into_theozyme).

- [ ] **Step 3: Write minimal implementation**

Create `prepare_PDB_structure_into_theozyme.py`:
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prepare_PDB_structure_into_theozyme.py — one-stop theozyme/ligand PDB prep.
Author: Seth M. Woodbury, David Baker Lab, UW (woodbuse@uw.edu)
See prepare_PDB_structure_into_theozyme__DESIGN.md for the full spec.
Preserve-by-default: no destructive change unless an explicit flag is given.
"""
import argparse, math, os, re, string, sys

PROTEIN_RESNAMES = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS",
    "MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
}

def main():
    raise SystemExit("not implemented yet")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/ -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Checkpoint** — `python3 -m pytest tests/ -q` green.

---

### Task 2: Column helpers (`pad_line`, `get_field`, `replace_field`, `atom_name_field`, `parse_charge_fields`)

**Files:**
- Modify: `prepare_PDB_structure_into_theozyme.py` (add helpers section)
- Modify: `tests/test_prepare_PDB_structure_into_theozyme.py`

- [ ] **Step 1: Write the failing test**

```python
def test_column_helpers():
    mod = load_mod()
    short = "ATOM      1  N   HIS A  55      48.399  26.664  28.044  1.00 -0.39"
    padded = mod.pad_line(short)
    assert len(padded) >= 80
    line = ("ATOM     85  NZ  LYS A 169      43.616  33.449  24.887"
            "  1.00 -0.19           N1-")
    assert mod.get_field(line, 17, 20).strip() == "LYS"
    assert mod.atom_name_field(line) == " NZ "
    elem, chg = mod.parse_charge_fields(line)
    assert elem == "N" and chg == "1-"
    out = mod.replace_field(line, 0, 6, "HETATM")
    assert out.startswith("HETATM") and len(out) == len(mod.pad_line(line))
    # Cα vs calcium must stay distinguishable
    ca = "ATOM      4  CA  HIS A  55      48.319  27.977  27.421  1.00  0.03           C"
    assert mod.atom_name_field(ca) == " CA "
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/ -q -k column_helpers`
Expected: FAIL (AttributeError: module has no attribute 'pad_line').

- [ ] **Step 3: Write minimal implementation**

Add to script:
```python
def pad_line(line):
    line = line.rstrip("\n")
    return line if len(line) >= 80 else line.ljust(80)

def get_field(line, start, end):
    return pad_line(line)[start:end]

def replace_field(line, start, end, new):
    if len(new) != (end - start):
        raise ValueError(f"replace_field length mismatch: {end-start} vs {len(new)}")
    line = pad_line(line)
    return line[:start] + new + line[end:]

def is_atom_record(line):
    return line[:6] in ("ATOM  ", "HETATM")

def atom_name_field(line):
    """Return the raw 4-char atom-name field (cols 13-16), spaces intact."""
    return get_field(line, 12, 16)

def parse_charge_fields(line):
    """Return (element, formal_charge_str). Element from cols 77-78; charge
    from 79-80 (e.g. '1-'). Falls back to atom-name first alpha char for
    element only when col 77-78 is blank. Never misreads 'N1-' as element."""
    p = pad_line(line)
    elem = p[76:78].strip()
    chg = p[78:80].strip()
    if not elem:
        nm = atom_name_field(line).strip()
        elem = "".join(c for c in nm if c.isalpha())[:2].upper()[:1]
    return elem, chg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/ -q -k column_helpers`
Expected: PASS.

- [ ] **Step 5: Checkpoint** — full suite green.

---

### Task 3: `WarningLog`

**Files:** Modify script + tests.

- [ ] **Step 1: Write the failing test**

```python
def test_warning_log(capsys):
    mod = load_mod()
    w = mod.WarningLog()
    w.info("doing thing")
    w.warn("ligand ABC not in CCD", category="ccd")
    w.warn("residue A99 removed", category="filter")
    out = capsys.readouterr().out
    assert "[INFO] doing thing" in out
    assert "[WARN] ligand ABC not in CCD" in out
    summary = w.render_summary()
    assert "SUMMARY" in summary
    assert "ligand ABC not in CCD" in summary
    assert "residue A99 removed" in summary
    assert w.count == 2
```

- [ ] **Step 2: Run** `python3 -m pytest tests/ -q -k warning_log` → FAIL.

- [ ] **Step 3: Implement**

```python
class WarningLog:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.warnings = []  # list[(category, msg)]
    @property
    def count(self):
        return len(self.warnings)
    def info(self, msg):
        if self.verbose:
            print(f"[INFO] {msg}")
    def warn(self, msg, category="general"):
        self.warnings.append((category, msg))
        print(f"[WARN] {msg}", file=sys.stderr)
    def render_summary(self):
        bar = "=" * 79
        lines = [bar, "  RUN SUMMARY", bar]
        if not self.warnings:
            lines.append("  No warnings. All operations completed cleanly.")
        else:
            lines.append(f"  {len(self.warnings)} warning(s):")
            for cat, msg in self.warnings:
                lines.append(f"   [{cat}] {msg}")
        lines.append(bar)
        return "\n".join(lines)
```

- [ ] **Step 4: Run** `-k warning_log` → PASS.
- [ ] **Step 5: Checkpoint** — full suite green.

---

### Task 4: `scan_structure`

**Files:** Modify script + tests.

- [ ] **Step 1: Write the failing test**

```python
SYNTH = """REMARK 665 REMARK 666 = Rosetta enzyme-matcher catalytic-motif anchors
REMARK 665 fmt: REMARK 666 MATCH TEMPLATE <tCH tNAME tRESI> MATCH MOTIF <mCH mRESN mRESI IDX VAR>
REMARK 666 MATCH TEMPLATE X  SUB    0 MATCH MOTIF A HIS   55   1   1
REMARK 666 MATCH TEMPLATE X  SUB    0 MATCH MOTIF A ASP  233   2   1
ATOM      1  N   HIS A  55      48.399  26.664  28.044  1.00 -0.39           N
ATOM      2  H   HIS A  55      48.863  26.702  28.943  1.00  0.17           H
ATOM      3  CG  ASP A 233      52.664  37.571  32.340  1.00  0.37           C
HETATM    4  P1  SUB Z 999      48.416  36.273  23.466  1.00  0.78           P
CONECT    4    3
END
""".splitlines()

def test_scan_structure():
    mod = load_mod()
    s = mod.scan_structure(SYNTH)
    assert ("A", "55", " ") in s.residues
    assert s.residues[("A","55"," ")]["resname"] == "HIS"
    assert s.has_partial_charges is True          # B-factor col non-zero/non-blank
    assert s.has_formal_charges is False          # none in this sample
    assert {r["resseq"] for r in s.ligands.values()} == {"999"}
    assert s.remark666 and s.remark666[0]["chain"] == "A" and s.remark666[0]["resseq"] == "55"
    assert s.remark665_header_present is True
    assert s.conect and s.multi_model is False
    assert s.hydrogen_atom_indices  # index of the H line captured
```

- [ ] **Step 2: Run** `-k scan_structure` → FAIL.

- [ ] **Step 3: Implement** a `Structure` dataclass-like object and `scan_structure`:

```python
class Structure:
    def __init__(self):
        self.lines = []
        self.residues = {}      # (chain,resseq,icode) -> {resname, indices:[...], is_het}
        self.ligands = {}       # same key, HETATM & non-water non-protein
        self.ncaa = {}          # key -> resname (in NONCANONICAL_AA_MAP)
        self.remark666 = []     # [{chain,res,resseq,idx,var,ligand,raw}]
        self.remark665_header_present = False
        self.conect = []        # raw CONECT lines (index positions)
        self.master_index = None
        self.has_partial_charges = False
        self.has_formal_charges = False
        self.hydrogen_atom_indices = set()
        self.multi_model = False
        self.overflow_warn = []

R666_RE = re.compile(
    r"^REMARK 666 MATCH TEMPLATE\s+(\S+)\s+(\S+)\s+(\S+)\s+MATCH MOTIF\s+"
    r"(\S)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)")

def scan_structure(lines):
    s = Structure()
    s.lines = [l.rstrip("\n") for l in lines]
    for i, line in enumerate(s.lines):
        rec = line[:6]
        if line.startswith("MODEL "):
            s.multi_model = s.multi_model or False
        if rec == "MODEL " and any(x.startswith("MODEL ") for x in s.lines[:i]):
            s.multi_model = True
        if _is_r665_666_header(line):  # shared predicate; also used by trigger_g
            s.remark665_header_present = True
        if line.startswith("REMARK 666 MATCH TEMPLATE"):
            m = R666_RE.match(line)
            if m:
                s.remark666.append({
                    "tch": m.group(1), "ligand": m.group(2), "tresi": m.group(3),
                    "chain": m.group(4), "res": m.group(5), "resseq": m.group(6),
                    "idx": int(m.group(7)), "var": int(m.group(8)), "raw": line})
            else:
                s._malformed_666 = getattr(s, "_malformed_666", [])
                s._malformed_666.append(line)
        if rec == "CONECT":
            s.conect.append(i)
        if rec == "MASTER":
            s.master_index = i
        if is_atom_record(line):
            chain = line[21]
            resseq = get_field(line, 22, 26).strip()
            icode = get_field(line, 26, 27)
            icode = icode if icode != " " else " "
            resname = get_field(line, 17, 20).strip()
            key = (chain, resseq, icode)
            bucket = s.residues.setdefault(
                key, {"resname": resname, "indices": [], "is_het": rec == "HETATM"})
            bucket["indices"].append(i)
            elem, chg = parse_charge_fields(line)
            if elem == "H" or atom_name_field(line).strip().startswith("H"):
                s.hydrogen_atom_indices.add(i)
            bcol = get_field(line, 60, 66).strip()
            if bcol not in ("", "0.00", "0.0", "0"):
                try:
                    if abs(float(bcol)) > 0:
                        s.has_partial_charges = True
                except ValueError:
                    pass
            if chg:
                s.has_formal_charges = True
            try:
                if int(get_field(line, 6, 11)) > 99999:
                    s.overflow_warn.append(f"serial overflow near line {i}")
            except ValueError:
                pass
            if rec == "HETATM" and resname != "HOH":
                s.ligands[key] = bucket
            if resname in NONCANONICAL_AA_MAP:
                s.ncaa[key] = resname
    return s
```

(`NONCANONICAL_AA_MAP` is added in Task 6; for now add a temporary
`NONCANONICAL_AA_MAP = {}` near `PROTEIN_RESNAMES` so this task runs, replaced in
Task 6.)

- [ ] **Step 4: Run** `-k scan_structure` → PASS.
- [ ] **Step 5: Checkpoint** — full suite green.

---

### Task 5: CLI parser + `validate_options`

**Files:** Modify script + tests.

- [ ] **Step 1: Write the failing test**

```python
def test_cli_and_validation():
    mod = load_mod()
    a = mod.parse_cli(["--input_pdb","i.pdb","--output_pdb_path","o.pdb"])
    assert a.input_pdb == "i.pdb" and a.theozyme_layout is False
    # mutually exclusive residue selectors
    import pytest
    bad = mod.parse_cli(["--input_pdb","i","--output_pdb_path","o",
                         "--residues_to_keep","A1","--residues_to_throw_away","A2"])
    with pytest.raises(SystemExit):
        mod.validate_options(bad)
    bad2 = mod.parse_cli(["--input_pdb","i","--output_pdb_path","o",
                          "--protect_ncAA_from_ligandization","--leave_ncAA_as_ATOM"])
    with pytest.raises(SystemExit):
        mod.validate_options(bad2)
    # merge auto-enables theozyme layout
    m = mod.parse_cli(["--input_pdb","i","--output_pdb_path","o",
                       "--merge_ligands_as","LIG"])
    mod.validate_options(m)
    assert m.theozyme_layout is True
```

- [ ] **Step 2: Run** `-k cli_and_validation` → FAIL.

- [ ] **Step 3: Implement** `parse_cli(argv)` with the full DESIGN §10 flag
surface and `validate_options(args)`:

```python
def parse_cli(argv=None):
    p = argparse.ArgumentParser(description="One-stop PDB→theozyme prep (preserve-by-default).")
    p.add_argument("--input_pdb", required=True)
    p.add_argument("--output_pdb_path", required=True)
    # REMARK 666
    p.add_argument("--force_regenerate_remark666", action="store_true")
    p.add_argument("--complete_remark666", action="store_true")
    p.add_argument("--remark666_exclude_residues", nargs="*", default=[])
    p.add_argument("--remark666_template_ligand", default=None)
    p.add_argument("--remark666_template_chain", default="X")
    p.add_argument("--remark666_template_resi", default="0")
    p.add_argument("--remark666_residue_front_order", nargs="*", default=[])
    p.add_argument("--remark666_residue_back_order", nargs="*", default=[])
    p.add_argument("--clean_remarks", action="store_true")
    # filtering
    p.add_argument("--residues_to_keep", nargs="*", default=None)
    p.add_argument("--residues_to_throw_away", nargs="*", default=None)
    p.add_argument("--ligands_to_keep", nargs="*", default=None)
    p.add_argument("--ligands_to_throw_away", nargs="*", default=None)
    # ncAA
    p.add_argument("--frag_ncAA_into_cAA_plus_lig", nargs="*", default=None)
    p.add_argument("--protect_ncAA_from_ligandization", action="store_true")
    p.add_argument("--leave_ncAA_as_ATOM", action="store_true")
    p.add_argument("--add_CA_to_labeled_frag", action="store_true")
    p.add_argument("--protect_sidechain_polarH", nargs="*", default=None)
    p.add_argument("--disable_intelligent_hstrip", action="store_true")
    # legacy opt-ins
    p.add_argument("--strip_insertion_codes", action="store_true")
    p.add_argument("--strip_protein_hydrogens", action="store_true")
    p.add_argument("--blank_segid", action="store_true")
    p.add_argument("--strip_partial_charges", action="store_true")
    p.add_argument("--strip_formal_charges", action="store_true")
    p.add_argument("--merge_ligands_as", default=None)
    p.add_argument("--merged_ligand_chain", default="Z")
    p.add_argument("--merged_ligand_resseq", default="999")
    p.add_argument("--merge_only", nargs="*", default=None)
    p.add_argument("--theozyme_layout", action="store_true")
    # renumber / checks / debug
    p.add_argument("--renumber_atoms", action="store_true")
    p.add_argument("--rosetta_residue_types",
        default="/net/software/rosetta/main/database/chemical/residue_type_sets/fa_standard/residue_types.txt")
    p.add_argument("--ccd_timeout", type=float, default=4.0)
    p.add_argument("--no_ligand_code_checks", action="store_true")
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args(argv)

def _die(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(2)

def validate_options(args):
    if args.residues_to_keep is not None and args.residues_to_throw_away is not None:
        _die("--residues_to_keep and --residues_to_throw_away are mutually exclusive.")
    if args.ligands_to_keep is not None and args.ligands_to_throw_away is not None:
        _die("--ligands_to_keep and --ligands_to_throw_away are mutually exclusive.")
    if args.protect_ncAA_from_ligandization and args.leave_ncAA_as_ATOM:
        _die("--protect_ncAA_from_ligandization and --leave_ncAA_as_ATOM are mutually exclusive.")
    if args.frag_ncAA_into_cAA_plus_lig is not None and (
        args.protect_ncAA_from_ligandization or args.leave_ncAA_as_ATOM):
        _die("--frag_ncAA_into_cAA_plus_lig cannot combine with ncAA protect/leave.")
    if args.merge_ligands_as is not None:
        args.theozyme_layout = True
    return args
```

- [ ] **Step 4: Run** `-k cli_and_validation` → PASS.
- [ ] **Step 5: Checkpoint** — full suite green.

---

### Task 6: Port ncAA dictionaries + transforms verbatim (with safe element parsing)

**Files:** Modify script + tests.

Port these from `prepare_PDB_structure_into_theozyme__MAIN.py` **verbatim**,
then apply the listed patches:

- `NONCANONICAL_AA_MAP` (MAIN lines 68–80) — replaces the temporary `{}` from Task 4.
- `PROTECTED_POLAR_H_MAP` (MAIN lines 95–112).
- `strip_insertion_code` (MAIN 151–157).
- `detect_ncaa` (MAIN 160–173).
- `gather_used_positions` (MAIN 176–185).
- `new_lig_position_generator` (MAIN 188–196).
- `build_frag_filters` (MAIN 206–226).
- `frag_ncaa` (MAIN 229–374).
- `convert_ncaa_hetatm_to_atom` / `revert_protected_ncaa_atom_to_hetatm` (MAIN 376–414).
- `protect_sidechain_polarH` / `revert_protected_sidechain_polarH` (MAIN 483–640).
- `add_CA_to_labeled_frag` (MAIN 642–783).

**Patches to apply during port (exact):**
1. Every occurrence of `line[76:78].strip()` for element detection → replace
   with `parse_charge_fields(line)[0]` (defined Task 2) so `N1-` is not
   misparsed.
2. Every `replace_field` used here must be the Task 2 version (already
   length-checked + padding). Remove MAIN's local `replace_field` (MAIN 199–203).
3. These functions operate on lists of newline-terminated strings in MAIN;
   standardize on **no-newline** strings (our internal convention). Where MAIN
   does `line.endswith("\n")` / `rstrip("\n")` (add_CA, MAIN 764–768), keep the
   pad-to-80 behavior but drop newline handling (writer adds `\n`).
4. Replace `print(...)`/`print(...,file=sys.stderr)` calls inside these
   functions with calls on a passed-in `wl: WarningLog` (`wl.info` / `wl.warn`).
   Add `wl` as the last parameter to each ported function.

- [ ] **Step 1: Write the failing test**

```python
KCX_PDB = """ATOM      1  N   KCX A 169      38.260  29.949  26.887  1.00  0.00           N
ATOM      2  CA  KCX A 169      38.986  31.192  26.721  1.00  0.00           C
ATOM      3  CB  KCX A 169      39.868  31.085  25.465  1.00  0.00           C
ATOM      4  NZ  KCX A 169      43.616  33.449  24.887  1.00  0.00           N
ATOM      5  CX  KCX A 169      44.860  33.980  24.812  1.00  0.00           C
HETATM    6  O   HOH W 1       10.000  10.000  10.000  1.00  0.00           O
END""".splitlines()

def test_frag_ncaa_splits_kcx():
    mod = load_mod()
    wl = mod.WarningLog(verbose=False)
    filt = mod.build_frag_filters([])          # all ncAAs
    out = mod.frag_ncaa([l for l in KCX_PDB], filt, True, wl)
    text = "\n".join(out)
    assert " LYS A 169" in text                # canonical fragment
    assert any(l.startswith("HETATM") and "LIG" in l for l in out)  # CX ligandized
    assert mod.NONCANONICAL_AA_MAP["KCX"]["canonical_resname"] == "LYS"
```

- [ ] **Step 2: Run** `-k frag_ncaa_splits_kcx` → FAIL.
- [ ] **Step 3:** Perform the verbatim port + the 4 patches above.
- [ ] **Step 4: Run** `-k frag_ncaa_splits_kcx` → PASS.
- [ ] **Step 5: Checkpoint** — full suite green (Task 4 scan test still passes
  with the now-real `NONCANONICAL_AA_MAP`).

---

### Task 7: `filter_structure` (residue + ligand keep/throw, selector parsing)

**Files:** Modify script + tests.

**Single-responsibility rule (Phase B code-review correction):**
`filter_structure` is ONLY responsible for ATOM/HETATM records.  It must NOT
touch REMARK 666 lines — those belong exclusively to `remark666_manager`.  A
stale post-pass that stripped REMARK 666 entries from `filter_structure` was
removed; `remark666_manager` receives `dropped` and uses `trigger_a` to detect
the dropped anchor, emit per-anchor `[WARN]`/SUMMARY lines, and contiguously
reindex survivors.  The key: `remark666_manager` compares on `(chain, resseq)`
(ignoring icode) so it correctly matches 3-tuple dropped keys against 2-tuple
666 entries.

- [ ] **Step 1: Write the failing test** (PRECISE assertions — NOT weakened)

```python
def test_filter_structure():
    mod = load_mod()
    wl = mod.WarningLog(verbose=False)
    lines = [l for l in SYNTH]                  # from Task 4
    # keep only A55 (drops A233 ATOM records); ligand kept (no ligand filter)
    kept, dropped = mod.filter_structure(
        lines, residues_to_keep=["A55"], residues_to_throw_away=None,
        ligands_to_keep=None, ligands_to_throw_away=None, wl=wl)
    txt = "\n".join(kept)
    assert " HIS A  55" in txt
    # PRECISE assertion: no ATOM/HETATM record for the dropped residue remains.
    # filter_structure is ONLY responsible for ATOM/HETATM records —
    # it must NOT strip REMARK 666 lines (that is remark666_manager's job).
    atom_lines = [l for l in kept if mod.is_atom_record(l)]
    assert not any("ASP" in l and "233" in l for l in atom_lines), \
        "filter_structure must remove all ATOM/HETATM for A233"
    # The REMARK 666 line for A233 must STILL be present (filter left it to manager).
    r666_for_233 = [l for l in kept
                    if l.startswith("REMARK 666") and "233" in l]
    assert r666_for_233, \
        "filter_structure must leave REMARK 666 for A233 intact (manager's responsibility)"
    assert ("A","233"," ") in dropped["residues"]
    assert "P1  SUB Z 999" in txt               # ligand untouched
    # selector form CHAIN:RESNAME:RESSEQ for ligand cut
    kept2, dropped2 = mod.filter_structure(
        lines, None, None, None, ["Z:SUB:999"], wl)
    assert "P1  SUB Z 999" not in "\n".join(kept2)
    assert ("Z","999"," ") in dropped2["ligands"]
```

- [ ] **Step 2: Run** `-k filter_structure` → FAIL.

- [ ] **Step 3: Implement**

```python
def parse_selector(tok):
    """'A55' -> ('A','55',None) ; 'Z:SUB:999' -> ('Z','999','SUB').
    Returns (chain, resseq, resname_or_None)."""
    tok = tok.strip()
    if ":" in tok:
        parts = tok.split(":")
        if len(parts) == 3:
            return parts[0], parts[2], parts[1]
        raise ValueError(f"bad selector '{tok}', expected CHAIN:RESNAME:RESSEQ")
    chain = tok[0]
    resseq = "".join(c for c in tok[1:] if c.isdigit())
    if not resseq:
        raise ValueError(f"bad selector '{tok}', expected like A169")
    return chain, resseq, None

def _match(line, sels):
    chain = line[21]
    resseq = get_field(line, 22, 26).strip()
    resname = get_field(line, 17, 20).strip()
    for c, rs, rn in sels:
        if c == chain and rs == resseq and (rn is None or rn == resname):
            return True
    return False

def filter_structure(lines, residues_to_keep, residues_to_throw_away,
                     ligands_to_keep, ligands_to_throw_away, wl):
    dropped = {"residues": set(), "ligands": set()}
    keep_sel = [parse_selector(t) for t in (residues_to_keep or [])]
    throw_sel = [parse_selector(t) for t in (residues_to_throw_away or [])]
    lkeep_sel = [parse_selector(t) for t in (ligands_to_keep or [])]
    lthrow_sel = [parse_selector(t) for t in (ligands_to_throw_away or [])]
    out = []
    for line in lines:
        if not is_atom_record(line):
            out.append(line); continue
        rec = line[:6]
        is_het = rec == "HETATM"
        resname = get_field(line, 17, 20).strip()
        chain = line[21]; resseq = get_field(line, 22, 26).strip()
        icode = get_field(line, 26, 27)
        key = (chain, resseq, icode if icode != " " else " ")
        # ligand record handling
        if is_het and resname != "HOH":
            if lkeep_sel and not _match(line, lkeep_sel):
                dropped["ligands"].add(key); continue
            if lthrow_sel and _match(line, lthrow_sel):
                dropped["ligands"].add(key); continue
            out.append(line); continue
        if is_het:                       # water / other HETATM: keep unless thrown
            out.append(line); continue
        # protein ATOM record
        if keep_sel:
            if _match(line, keep_sel):
                out.append(line)
            else:
                dropped["residues"].add(key)
            continue
        if throw_sel and _match(line, throw_sel):
            dropped["residues"].add(key); continue
        out.append(line)
    for k in sorted(dropped["residues"]):
        wl.warn(f"residue {k[0]}{k[1]} removed by filter", category="filter")
    for k in sorted(dropped["ligands"]):
        wl.warn(f"ligand {k[0]}{k[1]} removed by filter", category="filter")
    # NOTE: REMARK 666 lines are intentionally left untouched here.
    # remark666_manager is the SOLE owner of REMARK 666 lines; it receives
    # `dropped` and uses trigger_a to detect + drop stale anchors, emit
    # per-anchor warnings, and contiguously reindex survivors.
    return out, dropped
```

- [ ] **Step 3b: Add integration test** `test_filter_then_remark666_midblock_reindex_chain`
  (add directly after `test_filter_structure`):

```python
def test_filter_then_remark666_midblock_reindex_chain():
    """Integration test: build a 3-anchor block (HIS55 idx1, LYS169 idx2,
    ASP233 idx3) with matching ATOM lines; call filter_structure to drop A169;
    then scan_structure; then remark666_manager.  Asserts:
      - exactly 2 surviving REMARK 666 lines
      - IDX values are contiguous [1, 2] (NOT [1, 3] — the pre-fix bug)
      - '169' is absent from all surviving REMARK 666 lines
      - a dropped-anchor warning was recorded by wl

    This test MUST FAIL before the fix (filter_structure was stripping the
    REMARK 666 for A169 before remark666_manager could see it, so trigger_a
    never fired and the stale IDX gap [1,3] persisted) and PASS after.
    """
    mod = load_mod()
    wl = mod.WarningLog(verbose=False)
    R665_local = (
        "REMARK 665 REMARK 666 = Rosetta enzyme-matcher catalytic-motif anchors",
        "REMARK 665 fmt: REMARK 666 MATCH TEMPLATE <tCH tNAME tRESI>"
        " MATCH MOTIF <mCH mRESN mRESI IDX VAR>",
    )
    def _mk_local(chain, res, resseq, idx):
        return (f"REMARK 666 MATCH TEMPLATE X  SUB    0 MATCH MOTIF "
                f"{chain} {res:<3} {int(resseq):>4}{idx:>4}{1:>4}")

    lines = (
        list(R665_local) +
        [_mk_local("A", "HIS",  55, 1),
         _mk_local("A", "LYS", 169, 2),
         _mk_local("A", "ASP", 233, 3)] +
        ["ATOM      1  N   HIS A  55       0.000   0.000   0.000  1.00  0.00           N  ",
         "ATOM      2  N   LYS A 169       0.000   0.000   0.000  1.00  0.00           N  ",
         "ATOM      3  N   ASP A 233       0.000   0.000   0.000  1.00  0.00           N  "]
    )

    # Step 1: filter_structure drops LYS A169 ATOM records but MUST leave REMARK 666
    out1, dropped = mod.filter_structure(
        lines, residues_to_keep=None, residues_to_throw_away=["A169"],
        ligands_to_keep=None, ligands_to_throw_away=None, wl=wl)

    # Verify filter_structure left the REMARK 666 for 169 intact
    assert any("169" in l and l.startswith("REMARK 666") for l in out1), \
        "filter_structure must NOT strip REMARK 666 (that is remark666_manager's job)"

    # Step 2: re-scan the filtered output
    s = mod.scan_structure(out1)

    # Step 3: remark666_manager must detect the dropped anchor and reindex
    wl2 = mod.WarningLog(verbose=False)
    args = mod.parse_cli(["--input_pdb", "i", "--output_pdb_path", "o"])
    out2 = mod.remark666_manager(out1, s, dropped, args, wl2)

    # Assert: exactly 2 surviving REMARK 666 lines
    r666 = [l for l in out2 if l.startswith("REMARK 666")]
    assert len(r666) == 2, f"Expected 2 REMARK 666 lines, got {len(r666)}: {r666}"

    # Assert: IDX values are contiguous [1, 2] — NOT [1, 3]
    idxs = [int(l.split()[-2]) for l in r666]
    assert idxs == [1, 2], f"Expected contiguous IDX [1,2], got {idxs} — mid-block drop not reindexed"

    # Assert: '169' absent from all surviving REMARK 666 lines
    assert not any("169" in l for l in r666), \
        f"169 must not appear in surviving REMARK 666 lines: {r666}"

    # Assert: a dropped-anchor warning was recorded
    assert any("169" in m and ("dropped" in m.lower() or "reindexing" in m.lower())
               for _, m in wl2.warnings), \
        f"Expected dropped-anchor warning for A169 in wl.warnings: {wl2.warnings}"
```

- [ ] **Step 4: Run** `-k filter_structure` → PASS.
- [ ] **Step 5: Checkpoint** — full suite green.

---

### Task 8: `remark666_manager` — preserve / drop+reindex / exclude / complete / reorder / 665 header

**Files:** Modify script + tests. This is the spec's highest-risk unit (DESIGN §5).

- [ ] **Step 1: Write the failing tests** (multiple behaviors)

```python
R665 = ("REMARK 665 REMARK 666 = Rosetta enzyme-matcher catalytic-motif anchors",
        "REMARK 665 fmt: REMARK 666 MATCH TEMPLATE <tCH tNAME tRESI> MATCH MOTIF <mCH mRESN mRESI IDX VAR>")

def _mk(chain,res,resseq,idx):
    return (f"REMARK 666 MATCH TEMPLATE X  SUB    0 MATCH MOTIF "
            f"{chain} {res:<3} {int(resseq):>4}{idx:>4}{1:>4}")

def test_r666_preserve_default():
    mod = load_mod()
    wl = mod.WarningLog(verbose=False)
    lines = list(R665) + [_mk("A","HIS",55,1), _mk("A","ASP",233,2),
                          "ATOM      1  N   HIS A  55      0.0 0.0 0.0  1.00 0.00           N",
                          "ATOM      2  N   ASP A 233      0.0 0.0 0.0  1.00 0.00           N"]
    s = mod.scan_structure(lines)
    out = mod.remark666_manager(lines, s, dropped=None, args=mod.parse_cli(
        ["--input_pdb","i","--output_pdb_path","o"]), wl=wl)
    # unchanged, header still single
    assert sum(1 for l in out if l.startswith("REMARK 665 REMARK 666 =")) == 1
    assert _mk("A","HIS",55,1) in out and _mk("A","ASP",233,2) in out

def test_r666_drop_midblock_reindexes():
    mod = load_mod()
    wl = mod.WarningLog(verbose=False)
    lines = list(R665) + [_mk("A","HIS",55,1), _mk("A","LYS",169,2), _mk("A","ASP",233,3)]
    s = mod.scan_structure(lines)
    dropped = {"residues": {("A","169"," ")}, "ligands": set()}
    out = mod.remark666_manager(lines, s, dropped=dropped,
        args=mod.parse_cli(["--input_pdb","i","--output_pdb_path","o"]), wl=wl)
    r = [l for l in out if l.startswith("REMARK 666")]
    assert len(r) == 2
    assert r[0].split()[-2:] == ["1","1"]      # HIS55 -> idx 1
    assert r[1].split()[-2:] == ["2","1"]      # ASP233 -> idx 2 (gap removed)
    assert "169" not in " ".join(r)

def test_r666_exclude_residues():
    mod = load_mod()
    wl = mod.WarningLog(verbose=False)
    lines = list(R665) + [_mk("A","HIS",55,1), _mk("A","ASP",233,2)]
    s = mod.scan_structure(lines)
    args = mod.parse_cli(["--input_pdb","i","--output_pdb_path","o",
                          "--remark666_exclude_residues","A55"])
    out = mod.remark666_manager(lines, s, None, args, wl)
    r = [l for l in out if l.startswith("REMARK 666")]
    assert len(r) == 1 and "233" in r[0] and r[0].split()[-2:] == ["1","1"]

def test_r666_partial_coverage_warns_then_completes():
    mod = load_mod()
    wl = mod.WarningLog(verbose=False)
    # ASP233 protein atom present but has NO 666 line
    lines = list(R665) + [_mk("A","HIS",55,1),
        "ATOM 1 N HIS A  55 0 0 0 1.00 0.00 N".ljust(80),
        "ATOM 2 N ASP A 233 0 0 0 1.00 0.00 N".ljust(80)]
    s = mod.scan_structure(lines)
    a = mod.parse_cli(["--input_pdb","i","--output_pdb_path","o"])
    mod.remark666_manager(lines, s, None, a, wl)
    assert any("partial" in m.lower() or "lack" in m.lower()
               for _, m in wl.warnings)
    a2 = mod.parse_cli(["--input_pdb","i","--output_pdb_path","o","--complete_remark666"])
    out2 = mod.remark666_manager(lines, s, None, a2, mod.WarningLog(verbose=False))
    r = [l for l in out2 if l.startswith("REMARK 666")]
    assert len(r) == 2

def test_r666_front_and_back_listed_order():
    mod = load_mod()
    wl = mod.WarningLog(verbose=False)
    lines = list(R665) + [_mk("A","HIS",55,1), _mk("A","ASP",233,2),
                          _mk("A","LYS",169,3), _mk("A","TRP",131,4)]
    s = mod.scan_structure(lines)
    a = mod.parse_cli(["--input_pdb","i","--output_pdb_path","o",
        "--remark666_residue_front_order","A169","A55",
        "--remark666_residue_back_order","A131","A233"])
    out = mod.remark666_manager(lines, s, None, a, wl)
    r = [l for l in out if l.startswith("REMARK 666")]
    order = [(l.split()[10], l.split()[11]) for l in r]
    assert order == [("LYS","169"),("HIS","55"),("TRP","131"),("ASP","233")]
```

- [ ] **Step 2: Run** `-k r666` → FAIL.

- [ ] **Step 3: Implement**

**Phase B code-review corrections (applied on top of original implementation):**
- `_is_r665_666_header` helper unifies the 665-header predicate used by
  `scan_structure` (sets `remark665_header_present`) and `remark666_manager`
  trigger_g (counts header lines). Prevents silent predicate drift.
- `_partial_coverage_warning` helper de-duplicates the partial-coverage
  computation that was identical in preserve-path and rebuild-path, preventing
  future drift between the two.
- When `final` is non-empty but `template is None`: emit `wl.warn` instead of
  silently dropping all anchors.
- trigger_a: drop_keys are `(chain,resseq)` 2-tuples compared against 666
  entries' `(e["chain"], e["resseq"])`; dropped["residues"] keys are
  `(chain,resseq,icode)` 3-tuples → correctly normalized by the comprehension
  `{(c, rs) for (c, rs, _ic) in ...}`.

```python
R665_HEADER = (
    "REMARK 665 REMARK 666 = Rosetta enzyme-matcher catalytic-motif anchors",
    "REMARK 665 fmt: REMARK 666 MATCH TEMPLATE <tCH tNAME tRESI> MATCH MOTIF <mCH mRESN mRESI IDX VAR>",
)

def _is_r665_666_header(line):
    """Return True if this line is one of the two canonical REMARK 665 header
    lines that introduce the REMARK 666 block.  Used by scan_structure (to set
    remark665_header_present) and by remark666_manager trigger_g (to count
    how many such lines are present and detect missing or duplicated headers).
    Canonical predicate: starts with 'REMARK 665' AND contains 'REMARK 666'.
    """
    return line.startswith("REMARK 665") and "REMARK 666" in line

def _partial_coverage_warning(s, entries, excl_set, wl):
    """Emit a [WARN] if any protein residue in the structure lacks a REMARK 666
    entry.  Shared by both the preserve-path and rebuild-path of
    remark666_manager so the logic cannot drift between the two callers.
    Returns the set of missing (chain, resseq) pairs (may be empty)."""
    prot = {}
    for (chain, resseq, icode), info in s.residues.items():
        if (not info["is_het"]) and info["resname"] in PROTEIN_RESNAMES:
            prot[(chain, resseq)] = info["resname"]
    covered = {(e["chain"], e["resseq"]) for e in entries}
    missing = [(c, rs) for (c, rs) in prot
               if (c, rs) not in covered and (c, rs) not in excl_set]
    if missing:
        wl.warn("partial REMARK 666 coverage: protein residues lacking a 666 "
                "line: " + ", ".join(f"{c}{rs}" for c, rs in sorted(missing)),
                category="remark666")
    return set(missing)

def _fmt_666(chain, res, resseq, idx, ligand, tch, tresi, var=1):
    return (f"REMARK 666 MATCH TEMPLATE {tch:<3}{ligand:<3}{int(tresi):>5} "
            f"MATCH MOTIF {chain} {res:<3} {int(resseq):>4}{idx:>4}{var:>4}")

def _norm_sel(tok):
    chain = tok[0]; rs = "".join(c for c in tok[1:] if c.isdigit())
    return (chain, rs)

def remark666_manager(lines, s, dropped, args, wl):
    # 1. Collect existing parsed entries (dedupe by chain,resseq).
    seen = {}
    for e in s.remark666:
        seen.setdefault((e["chain"], e["resseq"]), e)
    has_duplicates = (len(seen) != len(s.remark666))
    has_malformed = bool(getattr(s, "_malformed_666", None))
    if has_duplicates:
        wl.warn("duplicate REMARK 666 entries de-duplicated by (chain,resseq)",
                category="remark666")
    if has_malformed:
        wl.warn(f"{len(s._malformed_666)} malformed REMARK 666 line(s) ignored",
                category="remark666")
    entries = list(seen.values())
    template = (args.remark666_template_ligand or
                (entries[0]["ligand"] if entries else None))
    tch = args.remark666_template_chain
    tresi = args.remark666_template_resi

    # --- Compute trigger conditions (a)-(g) for preserve-by-default ---
    # (a) filtering removed an anchored residue.
    # drop_keys are (chain,resseq) 2-tuples; dropped["residues"] keys are
    # (chain,resseq,icode) 3-tuples — normalize with the comprehension below.
    # trigger_a fires if ANY of those 2-tuples matches an existing 666 entry.
    # remark666_manager MUST receive the REMARK 666 lines untouched (filter_structure
    # must not strip them) so s.remark666 contains all original anchors here.
    drop_keys = set()
    if dropped:
        drop_keys = {(c, rs) for (c, rs, _ic) in dropped.get("residues", set())}
    trigger_a = bool(drop_keys and any(
        (e["chain"], e["resseq"]) in drop_keys for e in entries))
    # (b) ordering flags given (non-empty)
    trigger_b = bool(args.remark666_residue_front_order or
                     args.remark666_residue_back_order)
    # (c) --remark666_exclude_residues matches an existing entry
    excl_set = {_norm_sel(t) for t in args.remark666_exclude_residues}
    trigger_c = bool(excl_set and any(
        (e["chain"], e["resseq"]) in excl_set for e in entries))
    # (d) --force_regenerate_remark666
    trigger_d = bool(args.force_regenerate_remark666)
    # (e) --complete_remark666
    trigger_e = bool(args.complete_remark666)
    # (f) duplicate or malformed existing 666 entries
    trigger_f = has_duplicates or has_malformed
    # (g) REMARK 665 header missing or duplicated.
    # Uses _is_r665_666_header (shared with scan_structure) to prevent predicate drift.
    header_665_count = sum(1 for l in lines if _is_r665_666_header(l))
    trigger_g = (header_665_count != 2)  # expect exactly 2 (the two R665_HEADER lines)

    any_trigger = (trigger_a or trigger_b or trigger_c or trigger_d or
                   trigger_e or trigger_f or trigger_g)

    # --- PRESERVE PATH: no trigger fired → return verbatim (but still warn) ---
    if not any_trigger:
        # Compute partial-coverage warning even in preserve path (informational).
        # Uses shared helper to prevent logic drift vs rebuild path.
        _partial_coverage_warning(s, entries, excl_set, wl)
        # Return lines as-is; 665 header is already correct (trigger_g is False)
        return list(lines)  # verbatim: authored order, original IDX/VAR untouched

    # --- REBUILD PATH: at least one trigger fired ---

    # 2. Drop entries whose residue was removed by filtering.
    if drop_keys:
        kept = []
        for e in entries:
            if (e["chain"], e["resseq"]) in drop_keys:
                wl.warn(f"REMARK 666 for {e['chain']}{e['resseq']} dropped "
                        f"(residue removed); reindexing", category="remark666")
            else:
                kept.append(e)
        entries = kept

    # 3. Exclusion list.
    excl = excl_set
    present_keys = {(e["chain"], e["resseq"]) for e in entries}
    for c, rs in excl:
        if (c, rs) not in present_keys and not args.complete_remark666:
            wl.warn(f"--remark666_exclude_residues {c}{rs}: no current 666 line",
                    category="remark666")
    entries = [e for e in entries if (e["chain"], e["resseq"]) not in excl]

    # 4. Determine protein residues present in structure (for coverage/complete).
    prot = {}
    for (chain, resseq, icode), info in s.residues.items():
        if (not info["is_het"]) and info["resname"] in PROTEIN_RESNAMES:
            prot[(chain, resseq)] = info["resname"]

    force = args.force_regenerate_remark666
    if force:
        entries = []
        missing = [(c, rs) for (c, rs) in prot if (c, rs) not in excl]
    else:
        # Use shared helper to compute (and emit) partial-coverage warning.
        # This avoids duplicating the predicate logic vs the preserve path.
        missing_set = _partial_coverage_warning(s, entries, excl, wl) \
                      if not args.complete_remark666 else set()
        # Build the list form needed by the complete path below.
        covered = {(e["chain"], e["resseq"]) for e in entries}
        missing = [(c, rs) for (c, rs) in prot if (c, rs) not in covered
                   and (c, rs) not in excl]

    if args.complete_remark666 or force:
        if template is None and len(prot) and not entries:
            # need a ligand token; infer from single ligand else error
            lig_names = {info["resname"] for k, info in s.ligands.items()}
            if len(lig_names) == 1:
                template = next(iter(lig_names))
            elif len(lig_names) == 0:
                template = "LIG"
            else:
                _die("ambiguous ligand for fresh REMARK 666; pass "
                     "--remark666_template_ligand CODE")
        for (c, rs) in missing:
            entries.append({"chain": c, "res": prot[(c, rs)], "resseq": rs,
                            "ligand": template or "LIG", "idx": 0, "var": 1})

    # 5. Ordering: front (listed), middle (sorted), back (listed, NOT reversed).
    by_key = {(e["chain"], e["resseq"]): e for e in entries}
    front = [k for k in (_norm_sel(t) for t in args.remark666_residue_front_order)]
    back = [k for k in (_norm_sel(t) for t in args.remark666_residue_back_order)]
    for k in front + back:
        if k not in by_key:
            wl.warn(f"REMARK 666 order: {k[0]}{k[1]} not among entries; skipped",
                    category="remark666")
    front = [k for k in front if k in by_key]
    back = [k for k in back if k in by_key]
    if args.remark666_residue_back_order:
        wl.info("back_order applied in LISTED order (not reversed; differs "
                "from legacy STEP2).")
    mid = sorted([k for k in by_key if k not in front and k not in back],
                 key=lambda x: (x[0], int(x[1])))
    final = front + mid + back

    # 6. Re-emit with contiguous IDX, plus ensured single 665 header.
    new_666 = []
    if final and (template is not None):
        for i, k in enumerate(final, start=1):
            e = by_key[k]
            new_666.append(_fmt_666(e["chain"], e["res"], e["resseq"], i,
                                    e.get("ligand", template), tch, tresi,
                                    e.get("var", 1)))
    elif final and (template is None):
        # Anchors exist but no ligand template could be determined; warn instead
        # of silently dropping all anchors.
        anchor_list = ", ".join(f"{by_key[k]['chain']}{by_key[k]['resseq']}" for k in final)
        wl.warn(
            f"remark666_manager: {len(final)} anchor(s) ({anchor_list}) dropped "
            f"because no ligand template could be determined (no REMARK 666 entries "
            f"and --remark666_template_ligand not given); pass --remark666_template_ligand.",
            category="remark666")
    # Strip every existing REMARK 665(=666 header) + REMARK 666 line, reinsert.
    # Use _is_r665_666_header (shared predicate) for the 665-header strip.
    body = [l for l in lines
            if not l.startswith("REMARK 666")
            and not _is_r665_666_header(l)]
    if not new_666:
        return body
    # Insert header+666 immediately before first ATOM/HETATM, else after last REMARK.
    insert_at = next((i for i, l in enumerate(body) if is_atom_record(l)), len(body))
    return body[:insert_at] + list(R665_HEADER) + new_666 + body[insert_at:]
```

- [ ] **Step 4: Run** `-k r666` → all PASS.
- [ ] **Step 5: Checkpoint** — full suite green.

---

### Task 9: `apply_legacy_cleaning` (opt-in strip H / segID / charges / merge ligands)

**Files:** Modify script + tests.

- [ ] **Step 1: Write the failing test**

```python
def test_legacy_cleaning_optins():
    mod = load_mod()
    wl = mod.WarningLog(verbose=False)
    lines = [
      "ATOM      1  N   HIS A  55      0.0 0.0 0.0  1.00 -0.39           N",
      "ATOM      2  H   HIS A  55      0.0 0.0 0.0  1.00  0.17           H",
      "HETATM    3  P1  SUB Z 999      0.0 0.0 0.0  1.00  0.78           P",
      "HETATM    4  O1  CO2 C 997      0.0 0.0 0.0  1.00 -0.46           O1-",
    ]
    a = mod.parse_cli(["--input_pdb","i","--output_pdb_path","o",
        "--strip_protein_hydrogens","--strip_partial_charges",
        "--strip_formal_charges","--blank_segid",
        "--merge_ligands_as","LIG"])
    mod.validate_options(a)
    out = mod.apply_legacy_cleaning(lines, a, wl)
    txt = "\n".join(out)
    assert "  H   HIS" not in txt                       # protein H stripped
    assert "  1.00 -0.39" not in txt                    # partial charge zeroed
    assert all(not l[78:80].strip() for l in out if mod.is_atom_record(l))  # formal cleared
    assert " LIG " in txt                               # ligands merged
    # default (no flags) must be a no-op
    out2 = mod.apply_legacy_cleaning(lines,
        mod.validate_options(mod.parse_cli(["--input_pdb","i","--output_pdb_path","o"])), wl)
    assert out2 == lines
```

- [ ] **Step 2: Run** `-k legacy_cleaning_optins` → FAIL.

- [ ] **Step 3: Implement**

```python
def apply_legacy_cleaning(lines, args, wl):
    out = list(lines)
    if args.strip_protein_hydrogens:
        n0 = len(out)
        kept = []
        for l in out:
            if l.startswith("ATOM  ") and atom_name_field(l).strip().startswith("H"):
                continue
            kept.append(l)
        out = kept
        wl.warn(f"stripped {n0-len(out)} protein hydrogen atom(s)", category="clean")
    if args.strip_partial_charges:
        out = [replace_field(l, 60, 66, "  0.00") if is_atom_record(l) else l
               for l in out]
        wl.warn("partial charges (B-factor col) zeroed", category="clean")
    if args.strip_formal_charges:
        out = [replace_field(l, 78, 80, "  ") if is_atom_record(l) else l
               for l in out]
        wl.warn("formal charge column (79-80) cleared", category="clean")
    if args.blank_segid:
        out = [replace_field(l, 72, 76, "    ") if is_atom_record(l) else l
               for l in out]
    if args.merge_ligands_as:
        out = _merge_ligands(out, args, wl)
    return out

def _merge_ligands(lines, args, wl):
    code = args.merge_ligands_as
    if len(code) != 3:
        wl.warn(f"merge ligand code '{code}' is not 3 chars", category="ligand")
    ch = args.merged_ligand_chain
    rs = f"{int(args.merged_ligand_resseq):>4}"
    sel = None
    if args.merge_only:
        sel = [parse_selector(t) for t in args.merge_only]
    out = []
    elem_counts = {}
    n = 0
    for l in lines:
        if not l.startswith("HETATM"):
            out.append(l); continue
        rn = get_field(l, 17, 20).strip()
        if rn == "HOH":
            out.append(l); continue
        if sel is not None and not _match(l, sel):
            out.append(l); continue
        elem = parse_charge_fields(l)[0] or "X"
        elem_counts[elem] = elem_counts.get(elem, 0) + 1
        name = f"{elem}{elem_counts[elem]}"
        l = replace_field(l, 12, 16, f"{name:<4}"[:4])
        l = replace_field(l, 17, 20, f"{code:>3}")
        l = replace_field(l, 21, 22, ch[0])
        l = replace_field(l, 22, 26, rs)
        l = replace_field(l, 26, 27, " ")
        out.append(l); n += 1
    if n:
        wl.warn(f"merged {n} HETATM atom(s) into ligand '{code}' "
                f"({ch} {rs.strip()}); atom names regenerated (may break "
                f"pre-existing Rosetta params)", category="ligand")
    return out
```

- [ ] **Step 4: Run** `-k legacy_cleaning_optins` → PASS.
- [ ] **Step 5: Checkpoint** — full suite green.

---

### Task 10: `connectivity_repair` + `renumber_atoms`

**Files:** Modify script + tests.

- [ ] **Step 1: Write the failing test**

```python
def test_renumber_and_conect_repair():
    mod = load_mod()
    wl = mod.WarningLog(verbose=False)
    lines = [
      "ATOM      5  N   HIS A  55      0.0 0.0 0.0  1.00 0.00           N",
      "ATOM      9  CA  HIS A  55      0.0 0.0 0.0  1.00 0.00           C",
      "HETATM   12  P1  SUB Z 999      0.0 0.0 0.0  1.00 0.00           P",
      "CONECT    9   12",
      "CONECT    9   99",            # 99 dangling -> pruned
    ]
    out = mod.renumber_atoms(lines, wl)
    serials = [int(l[6:11]) for l in out if mod.is_atom_record(l)]
    assert serials == [1, 2, 3]
    con = [l for l in out if l.startswith("CONECT")]
    assert "CONECT    2    3" in "\n".join(con)        # 9->2, 12->3
    assert "99" not in "\n".join(con)                   # dangling pruned
    # repair-only (no renumber): dangling CONECT pruned, serials unchanged
    out2, changed = mod.connectivity_repair(lines, wl)
    assert [int(l[6:11]) for l in out2 if mod.is_atom_record(l)] == [5,9,12]
    assert "99" not in "\n".join(l for l in out2 if l.startswith("CONECT"))
```

- [ ] **Step 2: Run** `-k renumber_and_conect_repair` → FAIL.

- [ ] **Step 3: Implement** (rewrites ONLY serial cols 7–11; no UNL col hack):

> **Phase C corrections applied** (CRITICAL + IMPORTANT 1 fixes):
> - `_recount_master` shared helper added — writes numCoord to the correct slice
>   `[50:55]` (PDB cols 51–55) and numConect to `[40:45]` (PDB cols 41–45).
>   The old ad-hoc rewrite in `renumber_atoms` used `[30:35]` (numTurn field) —
>   that was wrong and corrupted the MASTER record.
> - `connectivity_repair` now calls `_recount_master` unconditionally at its end,
>   so MASTER is always recounted even without `--renumber_atoms`.
> - `renumber_atoms` no longer touches MASTER directly — MASTER recount is
>   delegated entirely to `_recount_master` via `connectivity_repair`.
> - `main` pipeline order swapped: `renumber_atoms` (step 9) runs BEFORE
>   `connectivity_repair` (step 8/10) so `_recount_master` sees the final state.

```python
def _atom_serials(lines):
    s = {}
    for l in lines:
        if is_atom_record(l):
            try: s[int(l[6:11])] = True
            except ValueError: pass
    return set(s)

def _recount_master(lines):
    """If a MASTER record exists, rewrite numCoord (cols 51-55, slice [50:55])
    to the current ATOM+HETATM count and numConect (cols 41-45, slice [40:45])
    to the current CONECT count. Leave other MASTER fields untouched. No-op if
    no MASTER record. Returns possibly-updated lines.

    PDB MASTER field layout (0-based Python slices, PDB v3.3 spec):
      [0:6]   record name "MASTER"
      [6:10]  numRemark
      [10:15] 0 (always zero)
      [15:20] numHet
      [20:25] numHelix
      [25:30] numSheet
      [30:35] numTurn (deprecated)
      [35:40] numSite
      [40:45] numXform / numConect  ← CONECT count
      [45:50] (unused in this scheme)
      [50:55] numCoord (ATOM+HETATM count)  ← atom count
      [55:60] numTer
      [60:65] numSeqres
    """
    natom = sum(1 for l in lines if is_atom_record(l))
    nconect = sum(1 for l in lines if l.startswith("CONECT"))
    new_lines = []
    for l in lines:
        if l.startswith("MASTER"):
            l = pad_line(l)
            l = replace_field(l, 40, 45, f"{nconect:>5}")
            l = replace_field(l, 50, 55, f"{natom:>5}")
        new_lines.append(l)
    return new_lines

def connectivity_repair(lines, wl):
    """Prune dangling CONECT references (atoms not in current serial set).
    Then unconditionally recounts MASTER via _recount_master.
    Returns (new_lines, changed_bool)."""
    valid = _atom_serials(lines)
    out = []
    changed = False
    for l in lines:
        if l.startswith("CONECT"):
            toks = l.split()
            nums = [t for t in toks[1:] if t.isdigit()]
            if not nums or int(nums[0]) not in valid:
                changed = True; continue
            kept = [nums[0]] + [t for t in nums[1:] if int(t) in valid]
            if len(kept) != len(nums):
                changed = True
            out.append("CONECT" + "".join(f"{int(t):>5}" for t in kept))
        else:
            out.append(l)
    if changed:
        wl.warn("CONECT records repaired (dangling refs pruned)",
                category="connect")
    # Always recount MASTER (numCoord + numConect) so it is correct even on
    # filter/merge runs that do not pass --renumber_atoms.
    out = _recount_master(out)
    return out, changed

def renumber_atoms(lines, wl):
    # group by (chain,resseq,icode) in first-seen order, keep within-group order
    order = []
    groups = {}
    for i, l in enumerate(lines):
        if is_atom_record(l):
            key = (l[21], get_field(l,22,26), get_field(l,26,27))
            if key not in groups:
                groups[key] = []; order.append(key)
            groups[key].append(i)
    mapping = {}
    new = 1
    new_lines = list(lines)
    for key in order:
        for idx in groups[key]:
            old = None
            try: old = int(lines[idx][6:11])
            except ValueError: pass
            if old is not None:
                mapping[old] = new
            new_lines[idx] = replace_field(lines[idx], 6, 11, f"{new:>5}")
            new += 1
    # rewrite CONECT via mapping, prune unmapped.
    # NOTE: MASTER recount is NOT done here — delegated to _recount_master
    # which connectivity_repair (always called after renumber in main) invokes.
    out = []
    for l in new_lines:
        if l.startswith("CONECT"):
            toks = l.split()
            nums = [int(t) for t in toks[1:] if t.lstrip("-").isdigit()]
            mapped = [mapping[n] for n in nums if n in mapping]
            if len(mapped) >= 2:
                out.append("CONECT" + "".join(f"{m:>5}" for m in mapped))
            else:
                wl.warn("CONECT dropped during renumber (dangling)",
                        category="connect")
        else:
            out.append(l)
    wl.info(f"renumbered {new-1} atom serial(s) grouped by residue")
    return out
```

- [ ] **Step 4: Run** `-k renumber_and_conect_repair` → PASS.
- [ ] **Step 5: Checkpoint** — full suite green.

---

### Task 11: `check_ligand_codes` (always-on, non-fatal CCD + Rosetta)

**Files:** Modify script + tests.

- [ ] **Step 1: Write the failing test** (network mocked; Rosetta uses a temp file)

```python
def test_check_ligand_codes(tmp_path, monkeypatch):
    mod = load_mod()
    wl = mod.WarningLog(verbose=False)
    rt = tmp_path / "residue_types.txt"
    rt.write_text("residue_types/l-caa/SUB.params\nother line\n")
    def fake_ccd(code, timeout):
        return {"LIG": "missing", "SUB": "exists"}.get(code, "unknown")
    monkeypatch.setattr(mod, "_ccd_status", fake_ccd)
    res = mod.check_ligand_codes(
        ligand_codes=["SUB","LIG"], rosetta_path=str(rt),
        ccd_timeout=4.0, wl=wl)
    assert res["SUB"]["ccd"] == "exists"
    assert res["SUB"]["rosetta"] == "present"
    assert res["LIG"]["ccd"] == "missing"
    assert res["LIG"]["rosetta"] == "absent"
    assert any("SUB" in m and "Rosetta" in m for _, m in wl.warnings)

def test_check_ligand_codes_offline(tmp_path, monkeypatch):
    mod = load_mod()
    wl = mod.WarningLog(verbose=False)
    monkeypatch.setattr(mod, "_ccd_status", lambda c, t: "unknown")
    res = mod.check_ligand_codes(["ZZZ"], "/nonexistent/path.txt", 4.0, wl)
    assert res["ZZZ"]["ccd"] == "unknown"
    assert res["ZZZ"]["rosetta"] == "unknown"   # path missing -> non-fatal
```

- [ ] **Step 2: Run** `-k check_ligand_codes` → FAIL.

- [ ] **Step 3: Implement**

```python
def _ccd_status(code, timeout):
    """exists / missing / unknown via RCSB ligand cif endpoint."""
    url = f"https://files.rcsb.org/ligands/view/{code}.cif"
    try:
        import requests
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200: return "exists"
        if r.status_code == 404: return "missing"
        return "unknown"
    except Exception:
        return "unknown"

def _rosetta_status(code, path):
    """present / absent / unknown by scanning residue_types.txt."""
    if not path or not os.path.isfile(path):
        return "unknown"
    try:
        fn_re = re.compile(rf"{re.escape(code)}\.params\b")
        sa_re = re.compile(rf"\b{re.escape(code)}\b")
        with open(path) as fh:
            for raw in fh:
                if fn_re.search(raw) or sa_re.search(raw):
                    return "present"
        return "absent"
    except Exception:
        return "unknown"

def check_ligand_codes(ligand_codes, rosetta_path, ccd_timeout, wl):
    results = {}
    for code in sorted(set(ligand_codes)):
        if len(code) != 3:
            wl.warn(f"ligand code '{code}' is not 3 characters", category="ligand")
        ccd = _ccd_status(code, ccd_timeout)
        ros = _rosetta_status(code, rosetta_path)
        results[code] = {"ccd": ccd, "rosetta": "present" if ros == "present"
                         else ("absent" if ros == "absent" else "unknown")}
        if ccd == "missing":
            wl.warn(f"ligand '{code}' NOT found in PDB CCD", category="ligand")
        elif ccd == "unknown":
            wl.warn(f"ligand '{code}' CCD status unknown (offline?)", category="ligand")
        if results[code]["rosetta"] == "present":
            wl.warn(f"ligand '{code}' ALREADY exists in Rosetta DB "
                    f"(name collision risk)", category="ligand")
        elif results[code]["rosetta"] == "unknown":
            wl.warn(f"ligand '{code}' Rosetta DB status unknown "
                    f"(residue_types.txt unreachable)", category="ligand")
    return results
```

- [ ] **Step 4: Run** `-k check_ligand_codes` → PASS.
- [ ] **Step 5: Checkpoint** — full suite green.

---

### Task 12: `main` pipeline wiring (DESIGN §4 order) + layout

**Files:** Modify script + tests.

- [ ] **Step 1: Write the failing test**

```python
def test_main_no_flags_roundtrips_ts1(tmp_path):
    mod = load_mod()
    src = "/home/woodbuse/projects/organophosphatase/pxn/design_campaign_i4__pte_hbond_260515/theozymes/kcx_set1__pte_hbond/pte_kcx_hbond_TS1.pdb"
    out = tmp_path / "out.pdb"
    mod.main(["--input_pdb", src, "--output_pdb_path", str(out),
              "--no_ligand_code_checks"])
    a = [l.rstrip() for l in open(src)]
    b = [l.rstrip() for l in open(out)]
    # preserve-by-default: same REMARK 665/666 header, same atom count,
    # charges intact, single 665 header, no duplicate 666
    assert b.count("REMARK 665 REMARK 666 = Rosetta enzyme-matcher catalytic-motif anchors") == 1
    assert sum(l.startswith(("ATOM  ","HETATM")) for l in a) == \
           sum(l.startswith(("ATOM  ","HETATM")) for l in b)
    assert any(l[78:80].strip() in ("1-","1+","2+") for l in b if l.startswith(("ATOM  ","HETATM")))
    assert any(l[60:66].strip() not in ("","0.00") for l in b if l.startswith(("ATOM  ","HETATM")))
    assert sum(l.startswith("REMARK 666") for l in a) == sum(l.startswith("REMARK 666") for l in b)

def test_main_writes_summary(tmp_path, capsys):
    mod = load_mod()
    src = "/home/woodbuse/projects/organophosphatase/pxn/design_campaign_i4__pte_hbond_260515/theozymes/kcx_set1__pte_hbond/pte_kcx_hbond_TS1.pdb"
    out = tmp_path / "o.pdb"
    mod.main(["--input_pdb", src, "--output_pdb_path", str(out)])
    assert "RUN SUMMARY" in capsys.readouterr().out
```

- [ ] **Step 2: Run** `-k "main_no_flags or main_writes_summary"` → FAIL.

- [ ] **Step 3: Implement** `main` replacing the Task 1 stub:

> **Phase C corrections applied** (IMPORTANT 1 + IMPORTANT 2 fixes):
> - Pipeline order swapped: `renumber_atoms` now runs BEFORE `connectivity_repair`
>   so `_recount_master` (called inside `connectivity_repair`) sees the final atom
>   serials and CONECT list.
> - `_apply_theozyme_layout` rewritten to preserve ALL record classes (HEADER,
>   CRYST1, LINK, etc.) and to only emit a second TER when HETATM block is
>   non-empty.

```python
def _read(path):
    with open(path) as fh:
        return [l.rstrip("\n") for l in fh]

def _write(path, lines):
    d = os.path.dirname(path)
    if d: os.makedirs(d, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

def main(argv=None):
    args = parse_cli(argv)
    validate_options(args)
    wl = WarningLog(verbose=args.verbose)
    lines = _read(args.input_pdb)
    s0 = scan_structure(lines)
    if s0.multi_model:
        wl.warn("multi-MODEL PDB; all models processed as a flat atom list",
                category="structure")
    for w in s0.overflow_warn:
        wl.warn(w, category="structure")

    if args.strip_insertion_codes:
        lines = [strip_insertion_code(l) for l in lines]
        wl.info("stripped residue insertion codes (opt-in).")

    # ncAA transforms
    if args.frag_ncAA_into_cAA_plus_lig is not None:
        lines = frag_ncaa(lines, build_frag_filters(args.frag_ncAA_into_cAA_plus_lig),
                          not args.disable_intelligent_hstrip, wl)
    protected = set()
    if args.protect_ncAA_from_ligandization or args.leave_ncAA_as_ATOM:
        lines, protected = convert_ncaa_hetatm_to_atom(lines, wl)
    if args.add_CA_to_labeled_frag:
        lines = add_CA_to_labeled_frag(lines, wl)
    if args.protect_sidechain_polarH is not None:
        lines = protect_sidechain_polarH(lines, args.protect_sidechain_polarH, wl)

    # filtering
    dropped = None
    if any(x is not None for x in (args.residues_to_keep,
            args.residues_to_throw_away, args.ligands_to_keep,
            args.ligands_to_throw_away)):
        lines, dropped = filter_structure(
            lines, args.residues_to_keep, args.residues_to_throw_away,
            args.ligands_to_keep, args.ligands_to_throw_away, wl)

    # legacy cleaning (opt-in)
    lines = apply_legacy_cleaning(lines, args, wl)

    # REMARK 666 (re-scan post-mutation for accurate residue universe)
    s = scan_structure(lines)
    lines = remark666_manager(lines, s, dropped, args, wl)

    # renumber first (so connectivity_repair/_recount_master sees final serials)
    if args.renumber_atoms:
        lines = renumber_atoms(lines, wl)

    # connectivity repair always; _recount_master runs unconditionally inside it,
    # keeping MASTER correct on filter/merge runs without --renumber_atoms too.
    lines, _ = connectivity_repair(lines, wl)

    if args.theozyme_layout:
        lines = _apply_theozyme_layout(lines)

    if args.protect_ncAA_from_ligandization and protected:
        lines = revert_protected_ncaa_atom_to_hetatm(lines, protected)
    if args.protect_sidechain_polarH is not None:
        lines = revert_protected_sidechain_polarH(lines)

    if not args.no_ligand_code_checks:
        s2 = scan_structure(lines)
        codes = sorted({info["resname"] for info in s2.ligands.values()})
        if args.merge_ligands_as:
            codes = sorted(set(codes) | {args.merge_ligands_as})
        check_ligand_codes(codes, args.rosetta_residue_types,
                           args.ccd_timeout, wl)

    _write(args.output_pdb_path, lines)
    wl.info(f"wrote {args.output_pdb_path}")
    print(wl.render_summary())

def _apply_theozyme_layout(lines):
    """Reorder records into canonical theozyme layout while preserving ALL records.

    Output order:
      1. All records NOT in (REMARK / ATOM / HETATM / TER / CONECT / MASTER /
         END), in original relative order (HEADER, CRYST1, LINK, MODEL, …).
      2. Non-665/666 REMARK lines (original order).
      3. REMARK 665/666 header + REMARK 666 anchor lines (original order).
      4. ATOM block.
      5. TER (one, always).
      6. HETATM block (if non-empty).
      7. TER (one, ONLY if ≥1 HETATM — no spurious TER when HETATM is empty).
      8. CONECT, MASTER, END — original order — at the tail.

    Original TER lines are replaced by the structured TERs; nothing else dropped.
    """
    top = []   # HEADER, CRYST1, LINK, MODEL/ENDMDL, SEQRES, etc.
    rem = []   # plain REMARK (non-665/666)
    hdr = []   # REMARK 665 header lines + REMARK 666 lines
    atoms = [] # ATOM
    hets = []  # HETATM
    tail = []  # CONECT, MASTER, END

    for l in lines:
        rec6 = l[:6]
        if rec6 == "ATOM  ":
            atoms.append(l)
        elif rec6 == "HETATM":
            hets.append(l)
        elif rec6 in ("CONECT", "MASTER") or l.startswith("END"):
            tail.append(l)
        elif rec6 == "TER   " or l.startswith("TER"):
            pass  # original TERs dropped; structured TERs inserted below
        elif l.startswith("REMARK"):
            if _is_r665_666_header(l) or l.startswith("REMARK 666"):
                hdr.append(l)
            else:
                rem.append(l)
        else:
            top.append(l)

    out = top + rem + hdr + atoms + ["TER"]
    if hets:
        out += hets + ["TER"]
    out += tail
    return out
```

- [ ] **Step 4: Run** `-k "main_no_flags or main_writes_summary"` → PASS.
- [ ] **Step 5: Checkpoint** — full suite green.

---

### Task 13: Acceptance — flagged end-to-end run on a copy of TS1

**Files:** Modify tests only.

- [ ] **Step 1: Write the failing test**

```python
def test_acceptance_flagged_run(tmp_path):
    mod = load_mod()
    src = "/home/woodbuse/projects/organophosphatase/pxn/design_campaign_i4__pte_hbond_260515/theozymes/kcx_set1__pte_hbond/pte_kcx_hbond_TS1.pdb"
    out = tmp_path / "flagged.pdb"
    mod.main(["--input_pdb", src, "--output_pdb_path", str(out),
              "--remark666_exclude_residues", "A131",
              "--remark666_residue_front_order", "A55", "A57",
              "--no_ligand_code_checks"])
    body = [l.rstrip() for l in open(out)]
    r = [l for l in body if l.startswith("REMARK 666")]
    # A131 excluded
    assert not any(l.split()[11] == "131" for l in r)
    # contiguous IDX 1..N, front order honored
    idxs = [int(l.split()[12]) for l in r]
    assert idxs == list(range(1, len(r) + 1))
    assert r[0].split()[11] == "55" and r[1].split()[11] == "57"
    # charges still intact (exclusion must not touch atoms)
    assert any(l[78:80].strip() for l in body if l.startswith(("ATOM  ","HETATM")))
```

- [ ] **Step 2: Run** `-k acceptance_flagged_run` → FAIL or PASS; if FAIL, fix the
  implicated unit (do NOT weaken the test).
- [ ] **Step 3:** Make minimal fixes in the implicated function only.
- [ ] **Step 4: Run** full suite → all PASS.
- [ ] **Step 5: Checkpoint** — full suite green.

---

### Task 14: Codex review of finished script, then archive `prepare_*` family

**Files:** Move into `backup/`.

- [ ] **Step 1: Codex review**

Run:
```bash
cd /home/woodbuse/special_scripts/theozyme_and_ligand_handling && \
codex exec --skip-git-repo-check -s read-only \
  -C /home/woodbuse/special_scripts/theozyme_and_ligand_handling/ \
  -o /tmp/codex_theozyme_final_review.md - <<'EOF'
Review prepare_PDB_structure_into_theozyme.py against
prepare_PDB_structure_into_theozyme__DESIGN.md. Confirm: preserve-by-default
holds (no destructive change without a flag); REMARK 665/666 rules incl.
mid-block drop+reindex, exclude list, partial-coverage, listed-order back;
charge/column safety (cols 61-66 & 77-80 untouched by default); no subprocess;
CONECT/MASTER repair; ligand checks non-fatal. List any spec deviations or bugs.
EOF
```
Read `/tmp/codex_theozyme_final_review.md`; fix any real spec deviations it
finds (re-run full suite after each fix).

- [ ] **Step 2: Verify no-flag round-trip once more**

Run: `python3 -m pytest tests/ -q` → all green.

- [ ] **Step 3: Archive only the superseded prepare_* family**

```bash
cd /home/woodbuse/special_scripts/theozyme_and_ligand_handling && \
mv prepare_PDB_structure_into_theozyme__MAIN.py \
   prepare_PDB_structure_into_theozyme__OLD.py \
   prepare_PDB_structure_into_theozyme__STEP1__cleanPDB_and_addREMARK666.py \
   prepare_PDB_structure_into_theozyme__STEP2__reorder_REMARK666_lines.py \
   backup/
```
Expected: 4 files moved; `prepare_PDB_structure_into_theozyme.py`,
`fragment_*`, `renumber_pdb.py`, `find_nonCCD_lig_codes.py`,
`check_if_ligand_3string_code_exists_in_rosetta.py` remain in place.

- [ ] **Step 4: Sanity** — `python3 prepare_PDB_structure_into_theozyme.py --help`
  prints the full flag surface with exit 0.

- [ ] **Step 5: Final checkpoint** — full suite green; DESIGN + this PLAN remain
  in the dir as living docs.

---

### Task 15: Regenerate the notebook driver cell + /tmp command writer

**Status: COMPLETE (Phase D hardening applied)**

**Files:**
- `prepare_PDB_structure_into_theozyme__DRIVER_CELL.py` — importable Python module, `build_command()` exposes ALL 34 optional parse_cli flags
- `prepare_PDB_structure_into_theozyme__DRIVER_CELL_TEMPLATE.txt` — paste-into-notebook cell, ALL 34 optional parse_cli flags present, `shlex.quote`-based join for shell safety

**Full-surface coverage requirement:** Both deliverables must expose EVERY flag
from `parse_cli`'s argparse parser (34 optional flags, excluding `--input_pdb`
and `--output_pdb_path` which are always in the base cmd). No functionality may
be silently inaccessible from the primary (paste-and-use) interface.

**Strengthened test:** `test_driver_surface_is_complete` (in
`tests/test_prepare_PDB_structure_into_theozyme.py`) introspects `parse_cli`
at runtime, asserts every `--flag` literal appears in `DRIVER_CELL_TEMPLATE.txt`,
and calls `build_command` with kwargs sets that exercise each flag (split across
safe, non-mutually-exclusive combinations) to assert every token is emittable.
This test fails before Phase D FIX 1/2 and passes after.

**Phase D fixes applied:**
- FIX 1: Added `--clean_remarks` and `--verbose` to `build_command` signature and append logic.
- FIX 2: Added `--ccd_timeout`, `--clean_remarks`, `--merge_only`, `--merged_ligand_chain`, `--merged_ligand_resseq`, `--remark666_template_chain`, `--remark666_template_ligand`, `--remark666_template_resi`, `--verbose`, `--rosetta_residue_types` to `DRIVER_CELL_TEMPLATE.txt`; replaced `" ".join(cmd)` with `shlex.quote`-based join.
- FIX 3: `.txt` uses `import shlex` and `shlex.quote` join; variable names aligned with `.py` (`frag_ncAA` consistently, ordering vars annotated with their `--flag`).
- FIX 4: Added `test_driver_surface_is_complete` with full introspection + build_command emission coverage.

**Checkpoint:** `python3 -m pytest tests/test_prepare_PDB_structure_into_theozyme.py -q` → 33 passed.

- [x] **Step 1: Write the failing test** (original basic test)
- [x] **Step 2: Run** `-k driver_cell_builds_command` → PASS.
- [x] **Step 3: Implement** `DRIVER_CELL.py` with full flag surface.
- [x] **Step 4: Implement** `DRIVER_CELL_TEMPLATE.txt` with full flag surface + shlex safety.
- [x] **Step 5: Strengthen test** `test_driver_surface_is_complete` — introspects parse_cli, asserts 100% template + build_command coverage.
- [x] **Step 6: Final checkpoint** — `python3 -m pytest tests/test_prepare_PDB_structure_into_theozyme.py -q` → 33 passed.

---

## Self-Review (performed)

**Spec coverage:** §2 preserve-by-default → Task 9 no-op test + Task 12
round-trip. §3 architecture → Tasks 2–12 map 1:1 to components. §4 pipeline
order → Task 12 `main`. §5 REMARK rules (drop+reindex, mid-block, exclude,
complete, partial-coverage, listed back, 665 header, non-contiguous,
multi-ligand template) → Task 8 (5 tests) + Task 13. §6 default layout vs
`--theozyme_layout` → Task 12 `_apply_theozyme_layout` + auto-enable in Task 5.
§7 charge/column safety → Task 2 + Task 9 + Task 12 round-trip assertions. §8
filtering/merge/renumber/checks → Tasks 7,9,10,11. §9 edge cases → Task 4
(overflow/multi-model), Task 8 (dedupe/malformed), Task 10 (CONECT/MASTER). §10
CLI → Task 5. §11 deliverables → Tasks 14 (backup, Codex review) + 15 (driver).

**Placeholder scan:** none — every code/test step is concrete; ncAA ports give
exact source line ranges + 4 explicit patches.

**Type consistency:** `WarningLog.warn(msg, category=)`, `scan_structure ->
Structure`, `filter_structure -> (lines, dropped{'residues','ligands'})`,
`remark666_manager(lines, s, dropped, args, wl)`, `parse_charge_fields ->
(elem,chg)`, `replace_field` signature consistent across Tasks 2–15.

**Environment:** git-free → "Commit" replaced by full-suite "Checkpoint"
throughout; confirmed pytest 9.0.2 + requests 2.32.5 + accessible Rosetta path.

---

## Post-Codex Review Fixes (6 defects, 2026-05-17)

A final independent review (Codex) identified 6 defects. All fixed with regression
tests. Final pytest count: **46 passed** (33 original + 13 new). TS1 no-flag
round-trip remains byte-identical.

### Defect 1 (CRITICAL — DESIGN §5): fresh REMARK 666 never generated when none exist
**Root cause:** `remark666_manager` had no trigger for the case "zero existing 666
entries AND protein residues present" → preserve path returned verbatim → raw PDB
emitted zero anchors.
**Fix:** Added `trigger_h` in `remark666_manager`: fires when `len(entries)==0` and
protein residues present and not `force_regenerate`. Fresh generation path
(`fresh_gen = complete_remark666 or force or trigger_h`) handles template inference
(single ligand → auto-infer; multiple ligands → `_die()`; zero ligands → default
"LIG" + `wl.warn`). Partial-coverage warning suppressed when `trigger_h` active.
**Tests:** `test_defect1_fresh_666_generated_single_ligand`,
`test_defect1_fresh_666_multi_ligand_dies`, `test_defect1_ts1_unaffected`.

### Defect 2 (Important — DESIGN §5): 665 header validation too weak
**Root cause:** `_is_r665_666_header` counted any 2 lines matching
`REMARK 665.*REMARK 666`; two copies of line 0 with line 1 missing still = count
of 2 → `trigger_g=False` → malformed header preserved.
**Fix:** Added `_has_exact_r665_header_pair(lines)` which checks each canonical
`R665_HEADER` string appears exactly once and no other matching lines exist.
`scan_structure.remark665_header_present` and `trigger_g` both use this function.
**Tests:** `test_defect2_duplicate_header_line1_normalized`,
`test_defect2_ts1_exact_pair_recognized`.

### Defect 3 (Important — DESIGN §5/§10): `--clean_remarks` parsed but unimplemented
**Root cause:** `args.clean_remarks` was accepted by `parse_cli` but never acted on.
**Fix:** `remark666_manager` strips non-665/666 `REMARK` lines from `body` when
`args.clean_remarks` is set, in both the preserve path and the rebuild path.
665+666 lines are always preserved/reinserted normally regardless.
**Tests:** `test_defect3_clean_remarks_strips_other_remarks`,
`test_defect3_ts1_no_clean_remarks_byte_identical`.

### Defect 4 (CRITICAL, crash — DESIGN §3/§4): add_CA_to_labeled_frag called wrong
**Root cause:** `main()` called `add_CA_to_labeled_frag(lines, wl)` — positional arg
`wl` bound to `ca_cb_bond_length` parameter → `TypeError: float expected` on any
missing-CA residue.
**Fix:** `main()` now calls `add_CA_to_labeled_frag(lines, wl=wl)` (keyword arg).
**Tests:** `test_defect4_add_CA_to_labeled_frag_no_type_error`,
`test_defect4_add_CA_noop_when_ca_present`.

### Defect 5 (Important — DESIGN §8): loose `.params` match dropped from Rosetta check
**Root cause:** `_rosetta_status` only checked filename (`CODE.params`) and
standalone (`\bCODE\b`) matches; the original script's third "loose" check
(`.params` line containing the code anywhere, non-adjacent) was absent.
**Fix:** `_rosetta_status` now returns `(status_str, has_loose_match_bool)`.
`check_ligand_codes` unpacks the 2-tuple and emits a distinct non-fatal `[WARN]`
with category `"rosetta_loose"` when a loose occurrence is found without a stronger
match. Never `sys.exit`.
**Tests:** `test_defect5_loose_params_warning_emitted`,
`test_defect5_clean_file_no_loose_warning`.

### Defect 6 (Footgun — DESIGN §8/§10): driver template advertised invalid --merge_only tokens
**Root cause:** `DRIVER_CELL_TEMPLATE.txt` showed `merge_only = ["LIG","CO"]` (bare
3-letter codes). `parse_selector` rejects bare codes (needs `A55` or
`CHAIN:RESNAME:RESSEQ`). Also `DRIVER_CELL.py` `merge_only` docstring gave no
selector format hint.
**Fix:** Template example updated to `["Z:LIG:901","Z:CO:902"]`. `DRIVER_CELL.py`
`merge_only` comment updated with valid-selector note and `residues_to_keep/throw`
and `ligands_to_keep/throw` comments updated with example selectors.
**Tests:** `test_defect6_driver_template_no_bare_code_in_merge_only`,
`test_defect6_driver_cell_merge_only_docstring_valid`.

---

## Post-Codex Addendum 2 — `--preserve_waters` (2026-05-18)

User correction: `HOH` must be a normal HETATM ligand by default (subject to
`--ligands_to_keep/throw` and merge), with special always-keep/never-merge
handling only behind the new opt-in `--preserve_waters` flag. Implemented TDD:
removed hard-coded `resname != "HOH"` / `rn == "HOH"` exceptions in
`filter_structure` (now takes `preserve_waters=False`) and `_merge_ligands`
(checks `args.preserve_waters`); added `--preserve_waters` to `parse_cli`,
both driver deliverables, DESIGN §8/§10. 4 new tests
(`test_waters_default_thrown_like_any_ligand`,
`test_waters_default_dropped_when_not_in_ligands_to_keep`,
`test_preserve_waters_flag_keeps_and_excludes_from_merge`,
`test_waters_merged_by_default`). Suite 50 passed; TS1 no-flag round-trip still
byte-identical (preserve-by-default unaffected — filter/merge remain opt-in).
