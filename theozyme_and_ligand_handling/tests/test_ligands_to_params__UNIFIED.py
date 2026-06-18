import os
import sys
import importlib

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
M = importlib.import_module("ligands_to_params__UNIFIED")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _write(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text("".join(lines))
    return str(p)


HET = (
    "HETATM  173   C1 CO2 C 997      44.860  33.980  24.812  1.00  0.41           C1+\n"
    "HETATM  213  H1  HOH W1000      44.421  36.655  20.622  1.00  0.35           H  \n"
    "HETATM  178   P1 SUB Z 999      48.416  36.273  23.466  1.00  0.78           P\n"
    "ATOM      4  CA  HIS A  55      48.319  27.977  27.421  1.00  0.03           C\n"
)

MULTI = [
    "HETATM  213  O   HOH W1000      44.500  36.000  20.000  1.00  0.00           O\n",
    "HETATM  214  H1  HOH W1000      44.400  36.600  20.600  1.00  0.00           H\n",
    "HETATM  215  H2  HOH W1000      44.000  36.100  19.200  1.00  0.00           H\n",
    "HETATM  216  O   HOH W1001      50.100  39.100  30.000  1.00  0.00           O\n",
    "HETATM  217  H1  HOH W1001      49.900  40.000  30.300  1.00  0.00           H\n",
    "HETATM  218  H2  HOH W1001      50.900  38.900  30.500  1.00  0.00           H\n",
    "HETATM  178   P1 SUB Z 999      48.416  36.273  23.466  1.00  0.00           P\n",
    "HETATM  179   C2 SUB Z 999      46.982  41.377  23.344  1.00  0.00           C\n",
]


def _stub_mol2_at(path, n_atoms=2):
    body = ["@<TRIPOS>MOLECULE\n", "stub\n", f" {n_atoms} 0 0 0 0\n",
            "@<TRIPOS>ATOM\n"]
    for i in range(1, n_atoms + 1):
        body.append(f"{i} A{i} {float(i):.4f} 0.0000 0.0000 C 1 UNL1 0.0000\n")
    body.append("@<TRIPOS>BOND\n")
    with open(path, "w") as fh:
        fh.writelines(body)
    return path


def _het(serial, name, res, chain, resseq, x, y, z, elem="", icode=""):
    """Build a column-correct PDB HETATM line."""
    line = list(" " * 80)
    line[0:6] = "HETATM"
    s = f"{serial:>5d}"
    line[6:11] = s
    nm = f"{name:<4s}"
    line[12:16] = nm
    rn = f"{res:>3s}"
    line[17:20] = rn
    line[21] = chain if chain else " "
    rs = f"{resseq:>4d}"
    line[22:26] = rs
    line[26] = icode if icode else " "
    xf = f"{x:>8.3f}"
    line[30:38] = xf
    yf = f"{y:>8.3f}"
    line[38:46] = yf
    zf = f"{z:>8.3f}"
    line[46:54] = zf
    line[54:60] = f"{1.0:>6.2f}"
    line[60:66] = f"{0.0:>6.2f}"
    if elem:
        ev = f"{elem:>2s}"
        line[76:78] = ev
    return "".join(line) + "\n"


def _read_input_atoms(input_path):
    """Return [(element, x, y, z), ...] from a stub XYZ or PDB input."""
    out = []
    if input_path.endswith(".xyz"):
        with open(input_path) as fh:
            lines = fh.readlines()
        n = int(lines[0].strip())
        for ln in lines[2:2 + n]:
            p = ln.split()
            out.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    else:
        with open(input_path) as fh:
            for ln in fh:
                if ln.startswith(("HETATM", "ATOM")):
                    pad = ln.rstrip("\n").ljust(80)
                    el = pad[76:78].strip() or pad[12:16].strip()[:1]
                    out.append((el, float(pad[30:38]),
                                float(pad[38:46]), float(pad[46:54])))
    return out


def _fake_obabel(input_path, input_format, out_mol2):
    """Stub Open Babel: emit a single-block MOL2 preserving the input atom
    COORDINATES (as real Open Babel does) so coordinate-based name mapping
    in apply_atom_names_to_mol2 works."""
    atoms = _read_input_atoms(input_path)
    n = len(atoms)
    nb = max(n - 1, 0)
    body = ["@<TRIPOS>MOLECULE\n", "stub\n",
            f" {n} {nb} 0 0 0\n", "@<TRIPOS>ATOM\n"]
    for i, (el, x, y, z) in enumerate(atoms, start=1):
        body.append(
            f"{i} A{i} {x:.4f} {y:.4f} {z:.4f} {el} 1 UNL1 0.0000\n")
    body.append("@<TRIPOS>BOND\n")
    # Deterministic linear-chain connectivity: mimics real Open Babel giving
    # the SAME bond graph for every conformer of one molecule (independent of
    # 3D geometry), so per-conformer bond-fix stays topology-consistent.
    for b in range(1, n):
        body.append(f" {b:>5} {b:>5} {b + 1:>5} {1:>4}\n")
    with open(out_mol2, "w") as fh:
        fh.writelines(body)
    return out_mol2


# --------------------------------------------------------------------------- #
# Task 1: group/token/arg parsing + legacy translation
# --------------------------------------------------------------------------- #
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
    assert groups[0].code == "ZDW" and groups[0].is_wildcard
    assert groups[0].mode == "network"


def test_resolve_groups_duplicate_code():
    args = M.parse_args([
        "--input_single_pdb", "x.pdb", "--output_dir_for_params_stuff", "o",
        "--group", "ABC=HOH", "--group", "ABC=SUB",
    ])
    with pytest.raises(M.PlanError):
        M.resolve_groups(args)


def test_ignore_residue_token_parsing():
    assert M.parse_residue_token("B12") == ("B", 12, "")
    assert M.parse_residue_token("Z999") == ("Z", 999, "")
    assert M.parse_residue_token("_15") == ("", 15, "")   # blank chain
    assert M.parse_residue_token("B12A") == ("B", 12, "A")  # insertion code
    with pytest.raises(ValueError):
        M.parse_residue_token("ABC")  # no resSeq digits


# --------------------------------------------------------------------------- #
# Task 2: PDB parse / select / ignore
# --------------------------------------------------------------------------- #
def test_read_hetatm_metadata(tmp_path):
    atoms = M.read_pdb_hetatms(_write(tmp_path, "a.pdb", [HET]))
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
    kept = M.apply_ignore(atoms, ignore_codes=["HOH"],
                          ignore_residues=[("C", 997)])
    assert {a.resname for a in kept} == {"SUB"}


# --------------------------------------------------------------------------- #
# Task 3: instances + planner
# --------------------------------------------------------------------------- #
def _multi_instances(tmp_path):
    return M.build_instances(
        M.read_pdb_hetatms(_write(tmp_path, "m.pdb", MULTI)))


def test_build_instances_keyed(tmp_path):
    insts = _multi_instances(tmp_path)
    keys = sorted(i.key for i in insts)
    assert keys == [
        ("HOH", "W", 1000, "", "m.pdb"),
        ("HOH", "W", 1001, "", "m.pdb"),
        ("SUB", "Z", 999, "", "m.pdb"),
    ]


def test_plan_hardstop_on_ungrouped_multi(tmp_path):
    insts = _multi_instances(tmp_path)
    with pytest.raises(M.PlanError) as e:
        M.plan_output_units(insts, groups=[], auto_codes=False)
    assert "HOH" in str(e.value) and "2" in str(e.value)


def test_plan_single_instance_ok(tmp_path):
    insts = [i for i in _multi_instances(tmp_path) if i.resname == "SUB"]
    units = M.plan_output_units(insts, groups=[], auto_codes=False)
    assert len(units) == 1
    assert units[0].code == "SUB" and units[0].mode == "network"


def test_plan_network_group_merges(tmp_path):
    insts = [i for i in _multi_instances(tmp_path) if i.resname == "HOH"]
    g = [M.parse_group_spec("WAT=HOH:network")]
    units = M.plan_output_units(insts, groups=g, auto_codes=False)
    assert len(units) == 1
    assert units[0].code == "WAT" and len(units[0].instances) == 2


def test_plan_conformer_group(tmp_path):
    insts = [i for i in _multi_instances(tmp_path) if i.resname == "HOH"]
    g = [M.parse_group_spec("WAT=HOH:conformer")]
    units = M.plan_output_units(insts, groups=g, auto_codes=False)
    assert len(units) == 1 and units[0].mode == "conformer"
    assert len(units[0].instances) == 2


def test_plan_wildcard_lumps_all(tmp_path):
    insts = _multi_instances(tmp_path)
    g = [M.parse_group_spec("ALL=*:network")]
    units = M.plan_output_units(insts, groups=g, auto_codes=False)
    assert len(units) == 1 and len(units[0].instances) == 3


def test_plan_auto_codes(tmp_path):
    insts = [i for i in _multi_instances(tmp_path) if i.resname == "HOH"]
    units = M.plan_output_units(insts, groups=[], auto_codes=True)
    assert len(units) == 2
    assert len({u.code for u in units}) == 2
    assert all(len(u.code) == 3 for u in units)


def test_plan_overlapping_groups_rejected(tmp_path):
    insts = _multi_instances(tmp_path)
    g = [M.parse_group_spec("AAA=HOH"), M.parse_group_spec("BBB=HOH")]
    with pytest.raises(M.PlanError):
        M.plan_output_units(insts, groups=g, auto_codes=False)


# --------------------------------------------------------------------------- #
# Task 4: code validation + atom-name uniquify
# --------------------------------------------------------------------------- #
def test_validate_codes_format_and_uniqueness():
    with pytest.raises(M.PlanError):
        M.validate_codes(["AB", "XYZ"], rosetta_txt=None, allow_collision=False)
    with pytest.raises(M.PlanError):
        M.validate_codes(["ABC", "ABC"], rosetta_txt=None, allow_collision=False)
    M.validate_codes(["ABC", "XY1"], rosetta_txt=None, allow_collision=False)


def test_rosetta_collision(tmp_path):
    rt = _write(tmp_path, "residue_types.txt", ["chemical/foo/ABC.params\n"])
    with pytest.raises(M.PlanError):
        M.validate_codes(["ABC"], rosetta_txt=rt, allow_collision=False)
    M.validate_codes(["ABC"], rosetta_txt=rt, allow_collision=True)


def test_uniquify_atom_names_within_unit():
    names = ["O", "H1", "H2", "O", "H1", "H2"]
    elems = ["O", "H", "H", "O", "H", "H"]
    out, mapping = M.uniquify_atom_names(names, elems)
    assert len(set(out)) == 6
    assert len(mapping) == 6


def test_uniquify_noop_when_unique():
    names = ["P1", "C2", "O3"]
    out, mapping = M.uniquify_atom_names(names, ["P", "C", "O"])
    assert out == names
    assert all(o == n for o, n in mapping)


# --------------------------------------------------------------------------- #
# Task 5: writers / bond-fix scoped / conformer composition
# --------------------------------------------------------------------------- #
def test_write_xyz_and_ligpdb(tmp_path):
    atoms = M.read_pdb_hetatms(_write(tmp_path, "a.pdb", [
        "HETATM  178   P1 SUB Z 999      48.416  36.273  23.466  1.00  0.0           P\n"]))
    x = str(tmp_path / "o.xyz")
    M.write_xyz(atoms, x)
    assert open(x).readline().strip() == "1"
    p = str(tmp_path / "o.pdb")
    M.write_ligand_pdb(atoms, p)
    assert "HETATM" in open(p).read()


def test_write_ligpdb_renames(tmp_path):
    atoms = M.read_pdb_hetatms(_write(tmp_path, "a.pdb", [
        "HETATM  178   P1 SUB Z 999      48.416  36.273  23.466  1.00  0.0           P\n"]))
    p = str(tmp_path / "o.pdb")
    M.write_ligand_pdb(atoms, p, names=["XX9"])
    assert "XX9" in open(p).read()


def test_bondfix_scoped_no_cross(tmp_path):
    mol2 = tmp_path / "m.mol2"
    mol2.write_text(
        "@<TRIPOS>MOLECULE\nx\n2 0 0 0 0\n@<TRIPOS>ATOM\n"
        "1 O 0.0 0.0 0.0 O 1 UNL1 0.0\n2 O 50.0 0.0 0.0 O 1 UNL1 0.0\n"
        "@<TRIPOS>BOND\n")
    M.bond_fix_mol2(str(mol2))
    txt = open(mol2).read()
    assert "@<TRIPOS>BOND" in txt
    assert txt.count("\n") > 6


def test_conformer_composition_validation():
    a = M._mol_signature(["P", "C", "C"], ["P1", "C2", "C3"])
    b = M._mol_signature(["P", "C", "C"], ["P1", "C2", "C3"])
    c = M._mol_signature(["P", "C"], ["P1", "C2"])
    assert M.assert_conformers_compatible([("u1", a), ("u2", b)]) is None
    with pytest.raises(M.PlanError) as e:
        M.assert_conformers_compatible([("u1", a), ("u3", c)])
    assert "atoms" in str(e.value)


# --------------------------------------------------------------------------- #
# Task 6: end-to-end with mocked obabel + rosetta
# --------------------------------------------------------------------------- #
def test_end_to_end_plan_dry(tmp_path, monkeypatch):
    pdb = _write(tmp_path, "TS.pdb", MULTI)
    calls = []

    def fake_obabel(input_path, input_format, out_mol2):
        calls.append(("ob", out_mol2))
        return _fake_obabel(input_path, input_format, out_mol2)

    def fake_rosetta(code, mol2_basename, out_dir, **kw):
        calls.append(("rosetta", code))

    monkeypatch.setattr(M, "run_obabel", fake_obabel)
    monkeypatch.setattr(M, "run_molfile_to_params", fake_rosetta)

    rc = M.run_pipeline(M.parse_args([
        "--input_single_pdb", pdb,
        "--output_dir_for_params_stuff", str(tmp_path / "out"),
        "--rosetta_residue_types", str(tmp_path / "nonexistent.txt"),
        "--group", "WAT=HOH:network",
        "--group", "SUB=SUB:network",
    ]))
    assert rc == 0
    codes = {c[1] for c in calls if c[0] == "rosetta"}
    assert codes == {"WAT", "SUB"}


def test_end_to_end_conformers_two_pdbs(tmp_path, monkeypatch):
    sub_t1 = [
        "HETATM  178   P1 SUB Z 999      48.416  36.273  23.466  1.00  0.0           P\n",
        "HETATM  179   C2 SUB Z 999      46.982  41.377  23.344  1.00  0.0           C\n",
    ]
    sub_t2 = [
        "HETATM  178   P1 SUB Z 999      48.500  36.300  23.500  1.00  0.0           P\n",
        "HETATM  179   C2 SUB Z 999      47.000  41.400  23.300  1.00  0.0           C\n",
    ]
    waters = [
        "HETATM  213  O   HOH W1000      44.500  36.000  20.000  1.00  0.0           O\n",
    ]
    t1 = _write(tmp_path, "TS1.pdb", sub_t1 + waters)
    t2 = _write(tmp_path, "TS2.pdb", sub_t2 + waters)
    confs = []

    def fake_rosetta(code, mol2_basename, out_dir, **kw):
        confs.append((code, mol2_basename))

    monkeypatch.setattr(M, "run_obabel", _fake_obabel)
    monkeypatch.setattr(M, "run_molfile_to_params", fake_rosetta)

    rc = M.run_pipeline(M.parse_args([
        "--input_single_pdb", t1,
        "--extra_conformer_pdbs", t2,
        "--ligands_to_extract_via_3letter_code", "SUB",
        "--group", "SUB=SUB:conformer",
        "--output_dir_for_params_stuff", str(tmp_path / "out"),
        "--rosetta_residue_types", str(tmp_path / "none.txt"),
    ]))
    assert rc == 0
    assert confs == [("SUB", "SUB.mol2")]
    # combined mol2 should contain 2 MOLECULE blocks (2 conformers)
    combined = open(str(tmp_path / "out" / "SUB.mol2")).read()
    assert combined.count("@<TRIPOS>MOLECULE") == 2


def test_end_to_end_conformer_mismatch_fails(tmp_path, monkeypatch):
    t1 = _write(tmp_path, "TS1.pdb", [
        "HETATM  178   P1 SUB Z 999      48.416  36.273  23.466  1.00 0.0           P\n",
        "HETATM  179   C2 SUB Z 999      46.982  41.377  23.344  1.00 0.0           C\n",
    ])
    t2 = _write(tmp_path, "TS2.pdb", [
        "HETATM  178   P1 SUB Z 999      48.500  36.300  23.500  1.00 0.0           P\n",
    ])
    monkeypatch.setattr(M, "run_obabel", _fake_obabel)
    monkeypatch.setattr(M, "run_molfile_to_params", lambda *a, **k: None)
    rc = M.run_pipeline(M.parse_args([
        "--input_single_pdb", t1, "--extra_conformer_pdbs", t2,
        "--ligands_to_extract_via_3letter_code", "SUB",
        "--group", "SUB=SUB:conformer",
        "--output_dir_for_params_stuff", str(tmp_path / "out"),
        "--rosetta_residue_types", str(tmp_path / "none.txt"),
    ]))
    assert rc == 1  # PlanError -> nonzero, handled cleanly


def test_empty_selection_errors(tmp_path):
    pdb = _write(tmp_path, "a.pdb", [HET])
    rc = M.run_pipeline(M.parse_args([
        "--input_single_pdb", pdb,
        "--ligands_to_extract_via_3letter_code", "ZZZ",
        "--output_dir_for_params_stuff", str(tmp_path / "out"),
        "--rosetta_residue_types", str(tmp_path / "none.txt"),
    ]))
    assert rc == 1


# --------------------------------------------------------------------------- #
# Task 7: integration on the real example TS PDBs (obabel/rosetta mocked)
# --------------------------------------------------------------------------- #
REAL_TS1 = ("/home/woodbuse/projects/organophosphatase/pxn/"
            "design_campaign_i4__pte_hbond_260515/theozymes/"
            "kcx_set1__pte_hbond/pte_kcx_hbond_TS1.pdb")
REAL_TS2 = ("/home/woodbuse/projects/organophosphatase/pxn/"
            "design_campaign_i4__pte_hbond_260515/theozymes/"
            "kcx_set1__pte_hbond/pte_kcx_hbond_TS2.pdb")


@pytest.mark.skipif(not os.path.isfile(REAL_TS1),
                    reason="real example PDBs not present")
def test_real_sub_conformers_only(tmp_path, monkeypatch):
    confs = []
    monkeypatch.setattr(M, "run_obabel", _fake_obabel)
    monkeypatch.setattr(M, "run_molfile_to_params",
                        lambda c, m, d, **kw: confs.append(c))
    rc = M.run_pipeline(M.parse_args([
        "--input_single_pdb", REAL_TS1,
        "--extra_conformer_pdbs", REAL_TS2,
        "--ligands_to_extract_via_3letter_code", "SUB",
        "--group", "SUB=SUB:conformer",
        "--output_dir_for_params_stuff", str(tmp_path / "out"),
        "--rosetta_residue_types", str(tmp_path / "none.txt"),
    ]))
    assert rc == 0 and confs == ["SUB"]
    assert open(str(tmp_path / "out" / "SUB.mol2")).read().count(
        "@<TRIPOS>MOLECULE") == 2


@pytest.mark.skipif(not os.path.isfile(REAL_TS1),
                    reason="real example PDBs not present")
def test_real_full_separate_with_groups(tmp_path, monkeypatch):
    rosetta_codes = []
    monkeypatch.setattr(M, "run_obabel", _fake_obabel)
    monkeypatch.setattr(M, "run_molfile_to_params",
                        lambda c, m, d, **kw: rosetta_codes.append(c))
    rc = M.run_pipeline(M.parse_args([
        "--input_single_pdb", REAL_TS1,
        "--output_dir_for_params_stuff", str(tmp_path / "out"),
        "--group", "WAT=HOH:network",
        "--group", "COF=ZN2,OHX:network",
        "--rosetta_residue_types", str(tmp_path / "none.txt"),
    ]))
    assert rc == 0
    # SUB and CO2 stay separate; WAT + COF are merged groups
    assert "WAT" in rosetta_codes and "COF" in rosetta_codes
    assert "SUB" in rosetta_codes and "CO2" in rosetta_codes


# --------------------------------------------------------------------------- #
# Post-review fixes (Codex): keep-names, name application, single-atom,
# insertion code separation, stop-after in preserve mode, extra-pdb dedupe
# --------------------------------------------------------------------------- #
def test_rosetta_command_uses_keep_names(tmp_path, monkeypatch):
    captured = {}
    _stub_mol2_at(str(tmp_path / "x.mol2"), 1)

    def fake_run(cmd, check):
        captured["cmd"] = cmd

        class R:
            pass
        return R()

    monkeypatch.setattr(M.subprocess, "run", fake_run)
    # main/conf pdb missing -> merge skipped, but command still captured
    M.run_molfile_to_params("ABC", "x.mol2", str(tmp_path))
    assert "--keep-names" in captured["cmd"]


def test_apply_atom_names_reaches_mol2(tmp_path):
    m = str(tmp_path / "n.mol2")
    _stub_mol2_at(m, 3)
    M.apply_atom_names_to_mol2(m, ["P1", "QQ2", "R3"])
    txt = open(m).read()
    assert "P1" in txt and "QQ2" in txt and "R3" in txt
    assert "A1 " not in txt  # stub default names overwritten


def test_apply_atom_names_count_mismatch(tmp_path):
    m = str(tmp_path / "n.mol2")
    _stub_mol2_at(m, 3)
    with pytest.raises(M.PlanError):
        M.apply_atom_names_to_mol2(m, ["ONLY1"])


def test_single_atom_unit_warns_not_fatal(tmp_path, monkeypatch, capsys):
    pdb = _write(tmp_path, "zn.pdb", [
        _het(176, "ZN1", "ZN2", "M", 901, 47.606, 32.961, 24.217, elem="ZN")])
    monkeypatch.setattr(M, "run_obabel", _fake_obabel)
    monkeypatch.setattr(M, "run_molfile_to_params", lambda *a, **k: None)
    rc = M.run_pipeline(M.parse_args([
        "--input_single_pdb", pdb,
        "--output_dir_for_params_stuff", str(tmp_path / "o"),
        "--rosetta_residue_types", str(tmp_path / "none.txt")]))
    assert rc == 0
    assert "SINGLE atom" in capsys.readouterr().out


def test_two_letter_element_inference(tmp_path):
    atoms = M.read_pdb_hetatms(_write(tmp_path, "zn.pdb", [
        _het(176, "ZN1", "ZN2", "M", 901, 47.6, 32.9, 24.2, elem="")]))
    assert atoms[0].element == "Zn"


def test_insertion_code_separates_instances(tmp_path):
    pdb = _write(tmp_path, "ic.pdb", [
        _het(1, "C1", "LIG", "A", 10, 0.0, 0.0, 0.0, elem="C"),
        _het(2, "C1", "LIG", "A", 10, 1.0, 0.0, 0.0, elem="C", icode="A")])
    insts = M.build_instances(M.read_pdb_hetatms(pdb))
    assert len(insts) == 2
    assert {i.icode for i in insts} == {"", "A"}


def test_stop_after_xyz_in_preserve_mode(tmp_path, monkeypatch):
    pdb = _write(tmp_path, "a.pdb", [
        _het(178, "P1", "SUB", "Z", 999, 48.4, 36.3, 23.5, elem="P")])
    called = []
    monkeypatch.setattr(M, "run_obabel",
                        lambda *a, **k: called.append("ob"))
    monkeypatch.setattr(M, "run_molfile_to_params",
                        lambda *a, **k: called.append("rosetta"))
    rc = M.run_pipeline(M.parse_args([
        "--input_single_pdb", pdb,
        "--output_dir_for_params_stuff", str(tmp_path / "o"),
        "--preserve_pdb_ligand_atom_order",
        "--stop_after_XYZ_is_made",
        "--rosetta_residue_types", str(tmp_path / "none.txt")]))
    assert rc == 0 and called == []  # stopped after ligand-PDB, no obabel/rosetta
    assert os.path.isdir(str(tmp_path / "o"))


def test_extra_conformer_pdb_same_as_primary_skipped(tmp_path, monkeypatch):
    pdb = _write(tmp_path, "TS.pdb", [
        _het(178, "P1", "SUB", "Z", 999, 48.4, 36.3, 23.5, elem="P"),
        _het(179, "C2", "SUB", "Z", 999, 46.9, 41.3, 23.3, elem="C")])
    confs = []
    monkeypatch.setattr(M, "run_obabel", _fake_obabel)
    monkeypatch.setattr(M, "run_molfile_to_params",
                        lambda c, m, d, **kw: confs.append(c))
    rc = M.run_pipeline(M.parse_args([
        "--input_single_pdb", pdb,
        "--extra_conformer_pdbs", pdb,            # same file as primary
        "--ligands_to_extract_via_3letter_code", "SUB",
        "--group", "SUB=SUB:conformer",
        "--output_dir_for_params_stuff", str(tmp_path / "o"),
        "--rosetta_residue_types", str(tmp_path / "none.txt")]))
    assert rc == 0
    # only ONE conformer (primary), extra-same-as-primary skipped
    combined = open(str(tmp_path / "o" / "SUB.mol2")).read()
    assert combined.count("@<TRIPOS>MOLECULE") == 1


# --------------------------------------------------------------------------- #
# Conformer topology: reference (default) vs strict
# --------------------------------------------------------------------------- #
def _obabel_factory(bondsets):
    state = {"n": 0}

    def f(input_path, fmt, out_mol2):
        atoms = _read_input_atoms(input_path)
        n = len(atoms)
        bonds = bondsets[min(state["n"], len(bondsets) - 1)]
        state["n"] += 1
        body = ["@<TRIPOS>MOLECULE\n", "stub\n",
                f" {n} {len(bonds)} 0 0 0\n", "@<TRIPOS>ATOM\n"]
        for i, (el, x, y, z) in enumerate(atoms, start=1):
            body.append(
                f"{i} A{i} {x:.4f} {y:.4f} {z:.4f} {el} 1 UNL1 0.0000\n")
        body.append("@<TRIPOS>BOND\n")
        for bi, (a, b) in enumerate(bonds, start=1):
            body.append(f" {bi:>5} {a:>5} {b:>5} {1:>4}\n")
        with open(out_mol2, "w") as fh:
            fh.writelines(body)
        return out_mol2

    return f, state


def test_build_conformer_from_reference_unit(tmp_path):
    ref = str(tmp_path / "ref.mol2")
    _stub_mol2_at(ref, 2)  # names A1, A2 at (1,0,0),(2,0,0)
    M.apply_atom_names_to_mol2(ref, ["P1", "C2"])  # index mode
    outp = str(tmp_path / "c1.mol2")
    M.build_conformer_from_reference(
        ref, {"P1": (9.0, 8.0, 7.0), "C2": (6.0, 5.0, 4.0)}, outp)
    txt = open(outp).read()
    assert txt.count("@<TRIPOS>MOLECULE") == 1
    assert "9.0000" in txt and "6.0000" in txt          # new coords
    assert "P1" in txt and "C2" in txt                   # names preserved


def test_conformer_reference_mode_succeeds_despite_obabel_divergence(
        tmp_path, monkeypatch):
    t1 = _write(tmp_path, "TS1.pdb", [
        _het(178, "P1", "SUB", "Z", 999, 48.4, 36.3, 23.5, elem="P"),
        _het(179, "C2", "SUB", "Z", 999, 46.9, 41.3, 23.3, elem="C"),
        _het(180, "C3", "SUB", "Z", 999, 45.0, 40.0, 22.0, elem="C")])
    t2 = _write(tmp_path, "TS2.pdb", [
        _het(178, "P1", "SUB", "Z", 999, 49.9, 35.0, 24.0, elem="P"),
        _het(179, "C2", "SUB", "Z", 999, 47.0, 41.0, 23.0, elem="C"),
        _het(180, "C3", "SUB", "Z", 999, 44.0, 39.0, 21.0, elem="C")])
    # obabel WOULD perceive different bonds per geometry (divergent)
    ob, state = _obabel_factory([[(1, 2), (2, 3)], [(1, 2)]])
    monkeypatch.setattr(M, "run_obabel", ob)
    monkeypatch.setattr(M, "run_molfile_to_params", lambda *a, **k: None)
    rc = M.run_pipeline(M.parse_args([
        "--input_single_pdb", t1, "--extra_conformer_pdbs", t2,
        "--ligands_to_extract_via_3letter_code", "SUB",
        "--group", "SUB=SUB:conformer", "--skip_bond_fix",
        "--output_dir_for_params_stuff", str(tmp_path / "o"),
        "--rosetta_residue_types", str(tmp_path / "none.txt")]))
    assert rc == 0
    assert state["n"] == 1            # obabel called ONCE (conformer 1 only)
    combined = open(str(tmp_path / "o" / "SUB.mol2")).read()
    assert combined.count("@<TRIPOS>MOLECULE") == 2
    # both blocks carry conformer-1 topology (2 bonds), coords differ
    assert combined.count("@<TRIPOS>BOND") == 2
    assert "49.9000" in combined and "48.4000" in combined


def test_conformer_strict_mode_hard_fails_on_divergence(
        tmp_path, monkeypatch):
    t1 = _write(tmp_path, "TS1.pdb", [
        _het(178, "P1", "SUB", "Z", 999, 48.4, 36.3, 23.5, elem="P"),
        _het(179, "C2", "SUB", "Z", 999, 46.9, 41.3, 23.3, elem="C"),
        _het(180, "C3", "SUB", "Z", 999, 45.0, 40.0, 22.0, elem="C")])
    t2 = _write(tmp_path, "TS2.pdb", [
        _het(178, "P1", "SUB", "Z", 999, 49.9, 35.0, 24.0, elem="P"),
        _het(179, "C2", "SUB", "Z", 999, 47.0, 41.0, 23.0, elem="C"),
        _het(180, "C3", "SUB", "Z", 999, 44.0, 39.0, 21.0, elem="C")])
    ob, _ = _obabel_factory([[(1, 2), (2, 3)], [(1, 2)]])
    monkeypatch.setattr(M, "run_obabel", ob)
    monkeypatch.setattr(M, "run_molfile_to_params", lambda *a, **k: None)
    rc = M.run_pipeline(M.parse_args([
        "--input_single_pdb", t1, "--extra_conformer_pdbs", t2,
        "--ligands_to_extract_via_3letter_code", "SUB",
        "--group", "SUB=SUB:conformer", "--skip_bond_fix",
        "--conformer_topology", "strict",
        "--output_dir_for_params_stuff", str(tmp_path / "o"),
        "--rosetta_residue_types", str(tmp_path / "none.txt")]))
    assert rc == 1  # divergent perceived topology -> hard fail in strict mode
