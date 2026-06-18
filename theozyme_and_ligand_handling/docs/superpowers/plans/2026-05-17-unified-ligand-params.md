# Unified Ligand→Params Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `ligands_to_params__UNIFIED.py` — one script that extracts ligands from one or more PDBs and produces Rosetta params, supporting separate / network-merged / conformer grouping, ignore filters, and full legacy backward-compat.

**Architecture:** Instance-first planner. Pure functions parse → select → exclude → build instances → plan output units → validate, then side-effecting stages write files / call Open Babel / bond-fix / call Rosetta. External tools (obabel, molfile_to_params) are isolated behind thin wrappers so the planner/validators are unit-testable without them.

**Tech Stack:** Python 3 stdlib (argparse, os, re, math, json, subprocess, dataclasses), Open Babel CLI, Rosetta `molfile_to_params.py`, pytest.

---

## File Structure

- Create: `ligands_to_params__UNIFIED.py` — the whole tool, organized in sections:
  PATH CONSTANTS / DATA MODEL / PDB PARSING / SELECTION+IGNORE / INSTANCE BUILDER /
  GROUP PARSING / OUTPUT-UNIT PLANNER / CODE VALIDATION / ATOM-NAME UNIQUIFY /
  INPUT WRITERS / OPENBABEL / BOND-FIX / CONFORMER ASSEMBLY / ROSETTA WRAPPER /
  MAIN ORCHESTRATION / ARGPARSE.
- Create: `tests/test_ligands_to_params__UNIFIED.py` — unit + integration tests.
- Test infra reuses existing `tests/conftest.py` pattern (sys.path insert of repo root).

Pure (unit-tested, no external tools): parsing, selection, ignore, instance build,
group-spec parse, planner, code validation, atom-name uniquify, conformer composition
validation, bond-fix graph logic. Side-effecting (mocked in tests): obabel, rosetta.

---

### Task 1: Scaffold + data model + argparse + legacy translation

**Files:**
- Create: `ligands_to_params__UNIFIED.py`
- Test: `tests/test_ligands_to_params__UNIFIED.py`

- [ ] **Step 1: Write failing tests**

```python
import os, sys, importlib, pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
M = importlib.import_module("ligands_to_params__UNIFIED")

def test_parse_group_spec_network_default():
    g = M.parse_group_spec("WAT=HOH")
    assert g.code == "WAT" and g.resnames == ["HOH"] and g.mode == "network"

def test_parse_group_spec_conformer_and_multi():
    g = M.parse_group_spec("COF=ZN2,OHX:conformer")
    assert g.code == "COF" and g.resnames == ["ZN2", "OHX"] and g.mode == "conformer"

def test_parse_group_spec_wildcard():
    g = M.parse_group_spec("ZDW=*:network")
    assert g.resnames == ["*"] and g.is_wildcard

def test_parse_group_spec_bad_mode():
    with pytest.raises(ValueError):
        M.parse_group_spec("X=A:banana")

def test_legacy_desired_code_becomes_wildcard_network():
    args = M.parse_args([
        "--input_single_pdb", "x.pdb",
        "--output_dir_for_params_stuff", "o",
        "--desired_ligand_3letter_code", "ZDW",
    ])
    groups = M.resolve_groups(args)
    assert len(groups) == 1
    assert groups[0].code == "ZDW" and groups[0].is_wildcard and groups[0].mode == "network"

def test_ignore_residue_token_parsing():
    chain, resseq = M.parse_residue_token("B12")
    assert chain == "B" and resseq == 12
    chain, resseq = M.parse_residue_token("Z999")
    assert chain == "Z" and resseq == 999
```

- [ ] **Step 2: Run, expect fail** — `python -m pytest tests/test_ligands_to_params__UNIFIED.py -x -q` → ModuleNotFound / attribute errors.

- [ ] **Step 3: Implement** scaffold: shebang/docstring; PATH CONSTANTS (`OPENBABEL_BIN`, `MOLFILE_TO_PARAMS_SCRIPT` copied from MODERN); dataclasses `PdbAtom`, `LigandInstance`, `GroupSpec` (with `is_wildcard` property), `OutputUnit`; `parse_group_spec(s)` (split on first `=`, optional `:mode` suffix, validate mode in {network,conformer}, resnames split on `,`, `*` → wildcard); `parse_residue_token(tok)` (regex `^([A-Za-z])(\d+)$`, else ValueError); `parse_args(argv=None)` with all preserved + new flags; `resolve_groups(args)` (merge `--group` + `--grouping_json`; if none and `--desired_ligand_3letter_code` set → single `GroupSpec(code, ["*"], "network")`; conflict detection raises).

- [ ] **Step 4: Run, expect pass.**

- [ ] **Step 5: Commit-equivalent** (no git repo): note progress in plan checkboxes.

### Task 2: PDB parsing + selection + ignore

- [ ] **Step 1: Failing tests**

```python
def _write(tmp_path, name, lines):
    p = tmp_path / name; p.write_text("".join(lines)); return str(p)

HET = ("HETATM  173   C1 CO2 C 997      44.860  33.980  24.812  1.00  0.41           C1+\n"
       "HETATM  213  H1  HOH W1000      44.421  36.655  20.622  1.00  0.35           H  \n"
       "HETATM  178   P1 SUB Z 999      48.416  36.273  23.466  1.00  0.78           P\n"
       "ATOM      4  CA  HIS A  55      48.319  27.977  27.421  1.00  0.03           C\n")

def test_read_hetatm_metadata(tmp_path):
    f = _write(tmp_path, "a.pdb", [HET])
    atoms = M.read_pdb_hetatms(f)
    assert len(atoms) == 3
    a = atoms[0]
    assert a.resname == "CO2" and a.chain == "C" and a.resseq == 997
    assert a.element == "C" and a.name == "C1"

def test_select_filter(tmp_path):
    atoms = M.read_pdb_hetatms(_write(tmp_path, "a.pdb", [HET]))
    sel = M.select_atoms(atoms, codes=["SUB", "CO2"])
    assert {a.resname for a in sel} == {"SUB", "CO2"}
    assert M.select_atoms(atoms, codes=None) == atoms

def test_ignore_codes_and_residues(tmp_path):
    atoms = M.read_pdb_hetatms(_write(tmp_path, "a.pdb", [HET]))
    kept = M.apply_ignore(atoms, ignore_codes=["HOH"], ignore_residues=[("C", 997)])
    assert {a.resname for a in kept} == {"SUB"}
```

- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** `read_pdb_hetatms(path, source=None)` (parse cols: name 12:16 strip, resname 17:20, chain 21, resseq 22:26 int, x/y/z, element 76:78 or fallback, keep raw line + order index + source); `select_atoms(atoms, codes)`; `apply_ignore(atoms, ignore_codes, ignore_residues)` where ignore_residues is list of (chain,resseq).
- [ ] **Step 4: Run, expect pass.**

### Task 3: Instance builder + planner + multi-instance hard-stop

- [ ] **Step 1: Failing tests**

```python
def _atoms_multi(tmp_path):
    lines = [
      "HETATM  213  O   HOH W1000      44.5  36.0  20.0  1.00  0.0           O\n",
      "HETATM  215  O   HOH W1001      50.1  39.1  30.0  1.00  0.0           O\n",
      "HETATM  178   P1 SUB Z 999      48.4  36.3  23.5  1.00  0.0           P\n",
    ]
    return M.read_pdb_hetatms(_write(tmp_path, "m.pdb", lines))

def test_build_instances_keyed(tmp_path):
    insts = M.build_instances(_atoms_multi(tmp_path))
    keys = sorted(i.key for i in insts)
    assert keys == [("HOH","W",1000,"m.pdb"), ("HOH","W",1001,"m.pdb"), ("SUB","Z",999,"m.pdb")]

def test_plan_hardstop_on_ungrouped_multi(tmp_path):
    insts = M.build_instances(_atoms_multi(tmp_path))
    with pytest.raises(M.PlanError) as e:
        M.plan_output_units(insts, groups=[], auto_codes=False)
    assert "HOH" in str(e.value) and "2" in str(e.value)

def test_plan_single_instance_ok(tmp_path):
    insts = [i for i in M.build_instances(_atoms_multi(tmp_path)) if i.resname == "SUB"]
    units = M.plan_output_units(insts, groups=[], auto_codes=False)
    assert len(units) == 1 and units[0].code == "SUB" and units[0].mode == "network"

def test_plan_network_group_merges(tmp_path):
    insts = M.build_instances(_atoms_multi(tmp_path))
    g = [M.parse_group_spec("WAT=HOH:network")]
    units = M.plan_output_units([i for i in insts if i.resname=="HOH"], groups=g, auto_codes=False)
    assert len(units) == 1 and units[0].code == "WAT" and len(units[0].instances) == 2

def test_plan_wildcard_lumps_all(tmp_path):
    insts = M.build_instances(_atoms_multi(tmp_path))
    g = [M.parse_group_spec("ALL=*:network")]
    units = M.plan_output_units(insts, groups=g, auto_codes=False)
    assert len(units) == 1 and len(units[0].instances) == 3

def test_plan_auto_codes(tmp_path):
    insts = M.build_instances(_atoms_multi(tmp_path))
    units = M.plan_output_units([i for i in insts if i.resname=="HOH"], groups=[], auto_codes=True)
    assert len(units) == 2 and len({u.code for u in units}) == 2
    assert all(len(u.code) == 3 for u in units)
```

- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** `build_instances(atoms)` → list of `LigandInstance` (key = (resname,chain,resseq,source), atoms ordered by original index); define `PlanError(Exception)`; `plan_output_units(instances, groups, auto_codes)`:
  - Expand wildcard group to all resnames among ungrouped instances.
  - For each non-wildcard group: gather instances whose resname ∈ group.resnames → one OutputUnit (network: all instances merged; conformer: instances kept as conformer list). Reject resname assigned to >1 group.
  - Remaining ungrouped instances: group by resname; if a resname has >1 instance and not auto_codes → `PlanError` with resname, count, and suggested fixes text; if ==1 → its own unit (code = resname, mode network); if >1 and auto_codes → generate unique 3-char codes (`_auto_code(resname, idx, used)`).
  - `_auto_code`: take resname[:2] + base36 idx, uppercased, ensure 3 chars + uniqueness.
- [ ] **Step 4: Run, expect pass.**

### Task 4: Code validation + atom-name uniquify

- [ ] **Step 1: Failing tests**

```python
def test_validate_codes_format_and_uniqueness():
    with pytest.raises(M.PlanError):
        M.validate_codes(["AB", "XYZ"], rosetta_txt=None, allow_collision=False)  # AB not 3 chars
    with pytest.raises(M.PlanError):
        M.validate_codes(["ABC", "ABC"], rosetta_txt=None, allow_collision=False)  # dup
    M.validate_codes(["ABC", "XY1"], rosetta_txt=None, allow_collision=False)  # ok

def test_rosetta_collision(tmp_path):
    rt = _write(tmp_path, "residue_types.txt", ["chemical/.../ABC.params\n"])
    with pytest.raises(M.PlanError):
        M.validate_codes(["ABC"], rosetta_txt=rt, allow_collision=False)
    M.validate_codes(["ABC"], rosetta_txt=rt, allow_collision=True)  # warn only

def test_uniquify_atom_names_within_unit():
    names = ["O","H1","H2","O","H1","H2"]
    elems = ["O","H","H","O","H","H"]
    out, mapping = M.uniquify_atom_names(names, elems)
    assert len(set(out)) == 6
    assert len(mapping) == 6
```

- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** `validate_codes(codes, rosetta_txt, allow_collision)` (regex `^[A-Za-z0-9]{3}$`; dup check; if rosetta_txt given grep `CODE.params`/word-boundary → PlanError unless allow_collision then print warning); `uniquify_atom_names(names, elements)` → only rename when duplicates exist; deterministic element+counter; returns (new_names, old→new mapping list). Wire `uniquify` to run only for network units with >1 instance OR any dup.
- [ ] **Step 4: Run, expect pass.**

### Task 5: Input writers + bond-fix (scoped) + conformer validation

- [ ] **Step 1: Failing tests**

```python
def test_write_xyz_and_ligpdb(tmp_path):
    atoms = M.read_pdb_hetatms(_write(tmp_path,"a.pdb",[
        "HETATM  178   P1 SUB Z 999      48.4  36.3  23.5  1.00  0.0           P\n"]))
    x = str(tmp_path/"o.xyz"); M.write_xyz(atoms, x)
    assert open(x).readline().strip() == "1"
    p = str(tmp_path/"o.pdb"); M.write_ligand_pdb(atoms, p)
    assert "HETATM" in open(p).read()

def test_bondfix_scoped_no_cross(tmp_path):
    # two far-apart single atoms = two molecules; scoped fix must NOT bond across
    mol2 = tmp_path/"m.mol2"; mol2.write_text(
        "@<TRIPOS>MOLECULE\nx\n2 0 0 0 0\n@<TRIPOS>ATOM\n"
        "1 O 0.0 0.0 0.0 O 1 UNL1 0.0\n2 O 50.0 0.0 0.0 O 1 UNL1 0.0\n"
        "@<TRIPOS>BOND\n")
    M.bond_fix_mol2(str(mol2))  # connects the 2 atoms WITHIN this single unit
    txt = open(mol2).read()
    assert "@<TRIPOS>BOND" in txt and txt.count("\n") > 6

def test_conformer_composition_validation():
    a = M._mol_signature(["P","C","C"], ["P1","C2","C3"])
    b = M._mol_signature(["P","C","C"], ["P1","C2","C3"])
    c = M._mol_signature(["P","C"], ["P1","C2"])
    assert M.assert_conformers_compatible([("u1",a),("u2",b)]) is None
    with pytest.raises(M.PlanError) as e:
        M.assert_conformers_compatible([("u1",a),("u3",c)])
    assert "atoms" in str(e.value)
```

- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** `write_xyz(atoms,path)`, `write_ligand_pdb(atoms,path)` (preserve raw lines), `parse_mol2/gather_new_bonds/find_nearest_heteroatom/connect_fragments/partial_update_mol2/standardize_residue_labels/bond_fix_mol2` (ported verbatim from MODERN, but `bond_fix_mol2` operates on a single-unit MOL2 so scoping is structural — never called on a multi-unit file); `_mol_signature(elements,names)` → tuple; `assert_conformers_compatible(list_of (label,sig))` → PlanError with diff on first mismatch.
- [ ] **Step 4: Run, expect pass.**

### Task 6: Open Babel + Rosetta wrappers + main orchestration

- [ ] **Step 1: Failing test (mocked externals)**

```python
def test_end_to_end_plan_dry(tmp_path, monkeypatch):
    pdb = _write(tmp_path,"TS.pdb",[
      "HETATM  178   P1 SUB Z 999      48.4 36.3 23.5  1.00 0.0           P\n",
      "HETATM  179   C2 SUB Z 999      46.9 41.3 23.3  1.00 0.0           C\n",
      "HETATM  213   O  HOH W1000      44.5 36.0 20.0  1.00 0.0           O\n",
      "HETATM  214   H1 HOH W1000      44.4 36.6 20.6  1.00 0.0           H\n",
      "HETATM  215   H2 HOH W1000      44.0 36.1 19.2  1.00 0.0           H\n"])
    calls = []
    monkeypatch.setattr(M, "run_obabel", lambda *a, **k: calls.append(("ob",)+a) or _stub_mol2(a))
    monkeypatch.setattr(M, "run_molfile_to_params", lambda *a, **k: calls.append(("rosetta",)+a))
    rc = M.run_pipeline(M.parse_args([
        "--input_single_pdb", pdb, "--output_dir_for_params_stuff", str(tmp_path/"out"),
        "--group", "WAT=HOH:network", "--group", "SUB=SUB:network"]))
    assert rc == 0
    assert any(c[0]=="rosetta" for c in calls)
```
(`_stub_mol2` writes a minimal valid mol2 at the expected path.)

- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** `run_obabel(input_path, input_format, out_mol2)` (subprocess to OPENBABEL_BIN), `run_molfile_to_params(code, mol2, out_dir)` (chdir try/finally, subprocess, merge `{code}.pdb` onto `{code}_conformers.pdb`), `run_pipeline(args)` orchestrating all stages incl. `--stop_after_XYZ_is_made` / `--stop_after_MOL2_is_made` / `--skip_bond_fix` / `--preserve_pdb_ligand_atom_order`; conformer units: per-instance mol2 → validate compat → concatenate blocks → single rosetta call; network units: merged atoms (uniquified) → one mol2 → bond-fix → rosetta. `main()` calls `sys.exit(run_pipeline(parse_args()))`.
- [ ] **Step 4: Run, expect pass.**

### Task 7: Integration test on real example PDBs (rosetta+obabel mocked) + self-review

- [ ] **Step 1** Add test using the two real TS PDBs verifying: (a) `--ligands_to_extract_via_3letter_code SUB --extra_conformer_pdbs TS2 --group SUB=SUB:conformer` plans one conformer unit with 2 conformers; (b) full TS1 with `WAT=HOH:network`, `COF=ZN2,OHX:network`, ignoring nothing, no hard-stop. obabel/rosetta monkeypatched.
- [ ] **Step 2** Run full suite `python -m pytest tests/test_ligands_to_params__UNIFIED.py -q` → all pass.
- [ ] **Step 3** Codex review pass (architecture, edge cases, unification); fold fixes.

---

## Self-Review

- **Spec coverage:** network/conformer/separate modes (T3,T5,T6); legacy translation (T1); ignore flags (T2); multi-instance hard-stop + auto_codes (T3); code validation + rosetta collision (T4); atom-name uniquify (T4); conformer composition hard-fail (T5); per-unit-only bond-fix (T5); stop stages + chdir (T6); real-PDB use cases (T7). All §-sections mapped.
- **Placeholder scan:** none — every step has concrete code/commands.
- **Type consistency:** `OutputUnit.code/.mode/.instances`, `LigandInstance.key/.resname/.atoms`, `PlanError`, `parse_group_spec`, `plan_output_units`, `validate_codes`, `bond_fix_mol2`, `run_pipeline` used consistently across tasks.
