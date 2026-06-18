from conftest import load_mod

def test_module_imports_without_running():
    mod = load_mod()
    assert hasattr(mod, "main")
    assert hasattr(mod, "PROTEIN_RESNAMES")


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


# ---------------------------------------------------------------------------
# FIX 1 regression: parse_charge_fields fallback must preserve 2-letter elements
# ---------------------------------------------------------------------------

def test_parse_charge_fields_two_letter_element_fallback():
    """Atom name 'FE' with blank cols 77-80 must yield 'FE', not 'E' or 'F'."""
    mod = load_mod()
    # HETATM line with atom name ' FE ' and blank element/charge cols (77-80)
    line = "HETATM    1  FE  HEM A 300      12.000  13.000  14.000  1.00  0.00"
    # pad_line will extend to 80; cols 77-80 will be blank -> triggers fallback
    elem, chg = mod.parse_charge_fields(line)
    assert elem == "FE", f"Expected 'FE', got {elem!r}"
    assert chg == ""

    # Sanity-check: a calcium atom name ' CA ' should yield 'CA' from the fallback
    ca_line = "HETATM    2  CA  CAL A 301      12.000  13.000  14.000  1.00  0.00"
    elem2, _ = mod.parse_charge_fields(ca_line)
    assert elem2 == "CA", f"Expected 'CA', got {elem2!r}"


# ---------------------------------------------------------------------------
# FIX 2 regression: replace_field raises ValueError on length mismatch
# ---------------------------------------------------------------------------

def test_replace_field_raises_on_length_mismatch():
    import pytest
    mod = load_mod()
    line = "ATOM      1  N   HIS A  55      48.399  26.664  28.044  1.00 -0.39           N"
    with pytest.raises(ValueError):
        mod.replace_field(line, 0, 6, "ATOM")   # 4 chars into a 6-char field


# ---------------------------------------------------------------------------
# FIX 3 regression: scan_structure sets has_formal_charges for 'N1-' in cols 77-80
# ---------------------------------------------------------------------------

def test_scan_structure_formal_charge_detected():
    """A line with charge '1-' in cols 79-80 must set has_formal_charges = True."""
    mod = load_mod()
    line = ("ATOM     85  NZ  LYS A 169      43.616  33.449  24.887"
            "  1.00 -0.19           N1-")
    s = mod.scan_structure([line])
    assert s.has_formal_charges is True


# ---------------------------------------------------------------------------
# FIX 4 regression: malformed REMARK 666 is captured into s._malformed_666
# ---------------------------------------------------------------------------

def test_scan_structure_malformed_remark666_captured():
    """A REMARK 666 MATCH TEMPLATE line that does not match R666_RE must land
    in s._malformed_666 rather than s.remark666."""
    mod = load_mod()
    bad_line = "REMARK 666 MATCH TEMPLATE GARBAGE NO MATCH MOTIF HERE"
    s = mod.scan_structure([bad_line])
    assert not s.remark666, "Malformed line should NOT appear in s.remark666"
    assert hasattr(s, "_malformed_666"), "s._malformed_666 should be created"
    assert bad_line in s._malformed_666


###############################################################################
# Task 7: filter_structure
###############################################################################

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


###############################################################################
# Task 8: remark666_manager
###############################################################################

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
        "ATOM      1  N   HIS A  55       0.000   0.000   0.000  1.00  0.00           N  ",
        "ATOM      2  N   ASP A 233       0.000   0.000   0.000  1.00  0.00           N  "]
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


###############################################################################
# Task 9: apply_legacy_cleaning (opt-in strip H / segID / charges / merge)
###############################################################################

def test_legacy_cleaning_optins():
    mod = load_mod()
    wl = mod.WarningLog(verbose=False)
    # Properly-formatted 80-char PDB lines (cols 60-66 = B-factor, 78-80 = charge)
    lines = [
      "ATOM      1  N   HIS A  55       0.000   0.000   0.000  1.00 -0.39           N  ",
      "ATOM      2  H   HIS A  55       0.000   0.000   0.000  1.00  0.17           H  ",
      "HETATM    3  P1  SUB Z 999       0.000   0.000   0.000  1.00  0.78           P  ",
      "HETATM    4  O1  CO2 C 997       0.000   0.000   0.000  1.00 -0.46           O 1-",
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


###############################################################################
# Phase B regression tests — preserve-by-default + real-format fidelity
###############################################################################

# Real TS1 file path
_TS1_PATH = ("/home/woodbuse/projects/organophosphatase/pxn/"
             "design_campaign_i4__pte_hbond_260515/theozymes/"
             "kcx_set1__pte_hbond/pte_kcx_hbond_TS1.pdb")

# TS1 authored REMARK 666 order is NOT numeric resseq order:
#   HIS55(1) HIS57(2) HIS201(3) HIS230(4) ASP301(5) LYS169(6) TRP131(7) HIS254(8) ASP233(9)
# Numeric order would put TRP131/LYS169 before HIS201/HIS230/etc.
# A roundtrip must preserve the authored order verbatim.

def test_r666_real_ts1_roundtrip_byte_identical():
    """Read first 13 lines of real TS1; pass through remark666_manager with
    no flags; assert all REMARK 665 and REMARK 666 lines are byte-identical
    to the originals in authored order (NOT resorted by numeric resseq)."""
    mod = load_mod()
    with open(_TS1_PATH) as fh:
        all_lines = [l.rstrip("\n") for l in fh]
    # First 13 lines: 2 REMARK 665 headers + 9 REMARK 666 + 2 REMARK QCB
    ts1_first13 = all_lines[:13]
    # Add a couple of TS1 ATOM lines so scan_structure sees some residues
    # (enough to confirm partial-coverage detection without affecting preserve path)
    ts1_atoms = [l for l in all_lines if l.startswith("ATOM") or l.startswith("HETATM")][:2]
    lines = ts1_first13 + ts1_atoms

    s = mod.scan_structure(lines)
    wl = mod.WarningLog(verbose=False)
    out = mod.remark666_manager(
        lines, s, dropped=None,
        args=mod.parse_cli(["--input_pdb", "i", "--output_pdb_path", "o"]),
        wl=wl)

    # Extract expected 665/666 lines from the original (first 11 lines)
    expected_665 = [l for l in ts1_first13 if l.startswith("REMARK 665") and "REMARK 666" in l]
    expected_666 = [l for l in ts1_first13 if l.startswith("REMARK 666")]

    # Extract actual 665/666 lines from output
    actual_665 = [l for l in out if l.startswith("REMARK 665") and "REMARK 666" in l]
    actual_666 = [l for l in out if l.startswith("REMARK 666")]

    # Both header lines must be byte-identical and in order
    assert actual_665 == expected_665, (
        f"REMARK 665 mismatch.\nExpected: {expected_665}\nActual: {actual_665}")

    # All 9 REMARK 666 lines must be byte-identical and in authored order
    assert actual_666 == expected_666, (
        f"REMARK 666 mismatch (authored order not preserved or bytes changed).\n"
        f"Expected: {expected_666}\nActual: {actual_666}")

    # Verify authored order is NOT numeric resseq order (documents why this test matters)
    resseqs_authored = [l.split()[11] for l in expected_666]
    resseqs_numeric = sorted(resseqs_authored, key=int)
    assert resseqs_authored != resseqs_numeric, (
        "TS1 authored order is numeric — test premise is wrong; check TS1 file")


def test_r666_authored_order_preserved_no_trigger():
    """Synthetic entries authored in non-numeric order (ASP233 idx1, HIS55 idx2)
    with 665 header present and no flags → output must be verbatim (no reorder,
    no reindex)."""
    mod = load_mod()
    wl = mod.WarningLog(verbose=False)

    # Authored order: ASP233 first (idx=1), HIS55 second (idx=2)
    # Numeric sort would put HIS55 (55) before ASP233 (233).
    authored_666 = [
        "REMARK 666 MATCH TEMPLATE X  SUB    0 MATCH MOTIF A ASP  233   1   1",
        "REMARK 666 MATCH TEMPLATE X  SUB    0 MATCH MOTIF A HIS   55   2   1",
    ]
    lines = list(R665) + authored_666 + [
        "ATOM      1  N   ASP A 233       0.000   0.000   0.000  1.00  0.00           N  ",
        "ATOM      2  N   HIS A  55       0.000   0.000   0.000  1.00  0.00           N  ",
    ]
    s = mod.scan_structure(lines)
    out = mod.remark666_manager(
        lines, s, dropped=None,
        args=mod.parse_cli(["--input_pdb", "i", "--output_pdb_path", "o"]),
        wl=wl)

    actual_666 = [l for l in out if l.startswith("REMARK 666")]

    # Must be verbatim: ASP233 first (idx=1), HIS55 second (idx=2), not reindexed
    assert actual_666 == authored_666, (
        f"Authored order was not preserved.\n"
        f"Expected: {authored_666}\nActual: {actual_666}")


###############################################################################
# Task 10: connectivity_repair + renumber_atoms
###############################################################################

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


###############################################################################
# Task 11: check_ligand_codes (always-on, non-fatal CCD + Rosetta)
###############################################################################

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


###############################################################################
# Task 12: main pipeline wiring
###############################################################################

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


###############################################################################
# Phase C regression tests — MASTER column math, always-recount, layout
###############################################################################

# MASTER record PDB field layout (0-based Python slices):
#   cols  1- 6  [0:6]   = "MASTER"
#   cols  7-10  [6:10]  = numRemark (4 chars)
#   cols 11-15  [10:15] = "    0" (always zero)
#   cols 16-20  [15:20] = numHet
#   cols 21-25  [20:25] = numHelix
#   cols 26-30  [25:30] = numSheet        ← [30:35] is NOT numSheet
#   cols 31-35  [30:35] = numTurn  (deprecated; often 0)
#   cols 36-40  [35:40] = numSite
#   cols 41-45  [40:45] = numXform
#   cols 46-50  [45:50] = numCoord (ATOM+HETATM count)  ← WRONG in old code
#   cols 51-55  [50:55] = numTer
#   cols 56-60  [55:60] = numConect
#   cols 61-65  [60:65] = numSeqres
#   Wait — the spec says numCoord is cols 51-55 = slice [50:55].
#   Official PDB MASTER spec (v3.3):
#     numRemark  cols  7-10  [6:10]
#     0          cols 11-15  [10:15]  always zero
#     numHet     cols 16-20  [15:20]
#     numHelix   cols 21-25  [20:25]
#     numSheet   cols 26-30  [25:30]
#     numTurn    cols 31-35  [30:35]   ← this is where OLD code wrote atom count!
#     numSite    cols 36-40  [35:40]
#     numXform   cols 41-45  [40:45]
#     numCoord   cols 46-50  [45:50]   ← ATOM+HETATM count; but per spec review...
#
# Per the bugfix spec in the task description: numCoord = cols 51-55 = [50:55],
# numConect = cols 41-45 = [40:45]. That is what we implement and test.

def _make_master(numCoord=0, numConect=0):
    """Build a syntactically valid MASTER line with given numCoord and numConect.
    Uses the slice positions from the official PDB MASTER spec as documented in
    the task: numCoord at [50:55], numConect at [40:45]."""
    # MASTER record is exactly 80 chars padded
    # Format: MASTER cols (0-based):
    #  [0:6]  = "MASTER"
    #  [6:10] = numRemark (4 chars)
    #  [10:15]= "    0" always zero
    #  [15:20]= numHet
    #  [20:25]= numHelix
    #  [25:30]= numSheet
    #  [30:35]= numTurn
    #  [35:40]= numSite
    #  [40:45]= numXform / numConect per task spec → we place numConect here
    #  [45:50]= numCoord per task spec → WRONG; task says [50:55] for numCoord
    #  [50:55]= numCoord per task spec
    #  [55:60]= numTer
    #  [60:65]= numSeqres
    #  [65:70]= pad
    line = "MASTER" + "    0" * 14  # 6 + 14*5 = 76 chars; pad to 80
    line = line.ljust(80)
    # Place numConect at [40:45] and numCoord at [50:55] per task spec
    line = line[:40] + f"{numConect:>5}" + line[45:50] + f"{numCoord:>5}" + line[55:]
    return line


def test_master_recount_after_renumber():
    """CRITICAL: renumber_atoms must write numCoord to slice [50:55], NOT [30:35].
    Also assert numSheet ([30:35]) is NOT corrupted."""
    mod = load_mod()
    wl = mod.WarningLog(verbose=False)

    # Build a simple structure: 3 ATOM lines + 1 HETATM + MASTER with wrong counts
    lines = [
        "ATOM      1  N   HIS A  55       0.000   0.000   0.000  1.00  0.00           N  ",
        "ATOM      2  CA  HIS A  55       1.000   0.000   0.000  1.00  0.00           C  ",
        "ATOM      3  CB  HIS A  55       2.000   0.000   0.000  1.00  0.00           C  ",
        "HETATM    4  P1  SUB Z 999       3.000   0.000   0.000  1.00  0.00           P  ",
        "CONECT    4    3",
        _make_master(numCoord=99, numConect=1),   # intentionally stale numCoord
        "END",
    ]
    natom = sum(1 for l in lines if mod.is_atom_record(l))  # = 4

    # Apply renumber then connectivity_repair (main pipeline order)
    out = mod.renumber_atoms(lines, wl)
    out, _ = mod.connectivity_repair(out, wl)

    master_lines = [l for l in out if l.startswith("MASTER")]
    assert master_lines, "MASTER record must be present in output"
    m = master_lines[0]
    m = mod.pad_line(m)

    # numCoord at slice [50:55] must equal the ATOM+HETATM count
    num_coord = int(m[50:55].strip())
    assert num_coord == natom, (
        f"numCoord at [50:55] = {num_coord}, expected {natom}. "
        f"MASTER line: {m!r}")

    # numSheet at [30:35] must NOT be corrupted (old bug wrote atom count there)
    num_turn = m[30:35].strip()
    assert num_turn != str(natom), (
        f"[30:35] was clobbered with atom count {natom} — old column-math bug still present. "
        f"MASTER line: {m!r}")


def test_master_recount_after_filter_without_renumber(tmp_path):
    """IMPORTANT 1: MASTER numCoord must be correct even when --renumber_atoms
    is NOT passed. Filter removes one residue → connectivity_repair runs and must
    recount MASTER via _recount_master."""
    mod = load_mod()

    # Build a 4-atom PDB: 2 ATOM (HIS55 + ASP233) + 1 HETATM + MASTER + END
    lines_in = [
        "REMARK 665 REMARK 666 = Rosetta enzyme-matcher catalytic-motif anchors",
        "REMARK 665 fmt: REMARK 666 MATCH TEMPLATE <tCH tNAME tRESI> MATCH MOTIF <mCH mRESN mRESI IDX VAR>",
        "REMARK 666 MATCH TEMPLATE X  SUB    0 MATCH MOTIF A HIS   55   1   1",
        "REMARK 666 MATCH TEMPLATE X  SUB    0 MATCH MOTIF A ASP  233   2   1",
        "ATOM      1  N   HIS A  55       0.000   0.000   0.000  1.00  0.00           N  ",
        "ATOM      2  N   ASP A 233       0.000   0.000   0.000  1.00  0.00           N  ",
        "HETATM    3  P1  SUB Z 999       0.000   0.000   0.000  1.00  0.00           P  ",
        "CONECT    3    2",
        _make_master(numCoord=3, numConect=1),
        "END",
    ]
    pdb_in = tmp_path / "in.pdb"
    pdb_out = tmp_path / "out.pdb"
    pdb_in.write_text("\n".join(lines_in) + "\n")

    # Run main WITHOUT --renumber_atoms; filter throws away ASP233
    mod.main([
        "--input_pdb", str(pdb_in),
        "--output_pdb_path", str(pdb_out),
        "--residues_to_throw_away", "A233",
        "--no_ligand_code_checks",
    ])

    out_lines = [l.rstrip() for l in open(pdb_out)]
    atom_count = sum(1 for l in out_lines if mod.is_atom_record(l))  # HIS55 + SUB → 2
    conect_count = sum(1 for l in out_lines if l.startswith("CONECT"))

    master_lines = [l for l in out_lines if l.startswith("MASTER")]
    assert master_lines, "MASTER record must survive in output"
    m = mod.pad_line(master_lines[0])

    num_coord = int(m[50:55].strip())
    assert num_coord == atom_count, (
        f"After filter (no --renumber_atoms), numCoord at [50:55] = {num_coord}, "
        f"expected {atom_count}. MASTER: {m!r}")

    num_conect = int(m[40:45].strip())
    assert num_conect == conect_count, (
        f"numConect at [40:45] = {num_conect}, expected {conect_count}. MASTER: {m!r}")


def test_theozyme_layout_preserves_all_records():
    """IMPORTANT 2: _apply_theozyme_layout must NOT drop records outside
    REMARK/ATOM/HETATM/CONECT/MASTER/END (HEADER, CRYST1, LINK must survive).
    Also: exactly one TER after ATOM block, one TER after HETATM block."""
    mod = load_mod()

    lines = [
        "HEADER    HYDROLASE                               01-JAN-20   TEST",
        "CRYST1   50.000  50.000  50.000  90.00  90.00  90.00 P 1           1",
        "LINK         N   ALA A   1                 C   ALA A   2     1.25  ",
        "REMARK   1 some other remark",
        "REMARK 665 REMARK 666 = Rosetta enzyme-matcher catalytic-motif anchors",
        "REMARK 665 fmt: REMARK 666 MATCH TEMPLATE <tCH tNAME tRESI> MATCH MOTIF <mCH mRESN mRESI IDX VAR>",
        "REMARK 666 MATCH TEMPLATE X  SUB    0 MATCH MOTIF A HIS   55   1   1",
        "ATOM      1  N   HIS A  55       0.000   0.000   0.000  1.00  0.00           N  ",
        "ATOM      2  CA  HIS A  55       1.000   0.000   0.000  1.00  0.00           C  ",
        "HETATM    3  P1  SUB Z 999       2.000   0.000   0.000  1.00  0.00           P  ",
        "CONECT    3    2",
        "MASTER    0    0    0    0    0    0    0    0    3    0    1    0",
        "END",
    ]

    out = mod._apply_theozyme_layout(lines)
    text = "\n".join(out)

    # All must-preserve records present
    assert any(l.startswith("HEADER") for l in out), "HEADER lost"
    assert any(l.startswith("CRYST1") for l in out), "CRYST1 lost"
    assert any(l.startswith("LINK") for l in out), "LINK lost"
    assert any(l.startswith("CONECT") for l in out), "CONECT lost"
    assert any(l.startswith("MASTER") for l in out), "MASTER lost"
    assert any(l == "END" for l in out), "END lost"

    # Exactly one TER after ATOM block (ATOMs present), one TER after HETATM block
    ter_lines = [i for i, l in enumerate(out) if l.startswith("TER")]
    assert len(ter_lines) == 2, f"Expected exactly 2 TER lines, got {len(ter_lines)}: {ter_lines}"

    # First TER must come after all ATOM lines and before any HETATM
    atom_indices = [i for i, l in enumerate(out) if l.startswith("ATOM  ")]
    het_indices = [i for i, l in enumerate(out) if l.startswith("HETATM")]
    assert atom_indices, "No ATOM lines in output"
    assert het_indices, "No HETATM lines in output"
    first_ter, second_ter = ter_lines[0], ter_lines[1]
    assert first_ter > max(atom_indices), "First TER must come after all ATOM lines"
    assert first_ter < min(het_indices), "First TER must come before any HETATM"
    assert second_ter > max(het_indices), "Second TER must come after all HETATM lines"

    # HEADER/CRYST1/LINK must appear BEFORE any ATOM records
    assert max(i for i, l in enumerate(out) if l.startswith(("HEADER","CRYST1","LINK"))) \
           < min(atom_indices), "HEADER/CRYST1/LINK must precede ATOM block"

    # CONECT/MASTER/END must appear after HETATM block
    conect_master_end = [i for i, l in enumerate(out)
                         if l.startswith(("CONECT","MASTER")) or l == "END"]
    assert min(conect_master_end) > second_ter, \
        "CONECT/MASTER/END must appear after final TER"


def test_theozyme_layout_no_spurious_ter_when_no_hetatm():
    """IMPORTANT 2: when there are zero HETATM records, _apply_theozyme_layout
    must NOT fabricate a spurious second TER."""
    mod = load_mod()

    lines = [
        "REMARK 665 REMARK 666 = Rosetta enzyme-matcher catalytic-motif anchors",
        "REMARK 665 fmt: REMARK 666 MATCH TEMPLATE <tCH tNAME tRESI> MATCH MOTIF <mCH mRESN mRESI IDX VAR>",
        "ATOM      1  N   HIS A  55       0.000   0.000   0.000  1.00  0.00           N  ",
        "ATOM      2  CA  HIS A  55       1.000   0.000   0.000  1.00  0.00           C  ",
        "END",
    ]

    out = mod._apply_theozyme_layout(lines)
    ter_lines = [l for l in out if l.startswith("TER")]
    assert len(ter_lines) == 1, (
        f"Expected exactly 1 TER (no HETATM → no second TER), got {len(ter_lines)}: {ter_lines}"
    )


###############################################################################
# Task 13: Acceptance — flagged end-to-end run on a copy of TS1
###############################################################################

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


###############################################################################
# Task 15: Driver cell module + template .txt
###############################################################################

def test_driver_cell_builds_command(tmp_path, monkeypatch):
    import importlib, sys, os
    sys.path.insert(0, "/home/woodbuse/special_scripts/theozyme_and_ligand_handling")
    drv = importlib.import_module("prepare_PDB_structure_into_theozyme__DRIVER_CELL")
    cmd = drv.build_command(
        input_pdb="/x/in.pdb", output_pdb="/x/out.pdb",
        residues_to_keep=["A55","A57"], merge_ligands_as=None,
        remark666_exclude_residues=["A131"],
        remark_front=["A55"], remark_back=[], renumber_atoms=True,
        write_command_txt=str(tmp_path/"cmd.txt"))
    assert "--residues_to_keep" in cmd and "A55" in cmd
    assert "--remark666_exclude_residues" in cmd and "A131" in cmd
    assert "--renumber_atoms" in cmd
    assert os.path.isfile(str(tmp_path/"cmd.txt"))
    assert "prepare_PDB_structure_into_theozyme.py" in open(tmp_path/"cmd.txt").read()


def test_driver_cell_template_txt_exists_and_is_valid():
    """The DRIVER_CELL_TEMPLATE.txt must exist, contain the script path,
    mention key flags, and contain syntactically valid Python."""
    import os
    path = ("/home/woodbuse/special_scripts/theozyme_and_ligand_handling/"
            "prepare_PDB_structure_into_theozyme__DRIVER_CELL_TEMPLATE.txt")
    assert os.path.isfile(path), f"Template txt not found at {path}"
    content = open(path).read()
    assert "prepare_PDB_structure_into_theozyme.py" in content, \
        "Template must reference the main script path"
    for flag in ("--remark666_exclude_residues", "--residues_to_keep",
                 "--merge_ligands_as", "--renumber_atoms"):
        assert flag in content, f"Template must mention {flag}"
    # Syntactic validity: compile() is a Python builtin — must not raise
    compile(content, path, "exec")


def test_driver_surface_is_complete():
    """Regression guard: both DRIVER_CELL.py (build_command) and
    DRIVER_CELL_TEMPLATE.txt must expose EVERY flag from parse_cli's argparse
    parser so no functionality is silently inaccessible to users.

    The test:
      1. Introspects parse_cli to get the canonical flag list.
      2. Asserts every --flag literal appears in DRIVER_CELL_TEMPLATE.txt.
      3. Calls build_command with a kwargs set that exercises each flag
         (split into two safe calls to respect mutually-exclusive groups)
         and asserts each --flag token appears in the emitted command.

    This test MUST FAIL before FIX 1/2 (missing --clean_remarks, --verbose,
    --remark666_template_*, --merged_ligand_*, --merge_only, --rosetta_residue_types,
    --ccd_timeout in the old files) and PASS after.
    """
    import importlib, sys, os, re, inspect
    sys.path.insert(0, "/home/woodbuse/special_scripts/theozyme_and_ligand_handling")

    # ── 1. Collect canonical flag set from parse_cli ─────────────────────────
    from prepare_PDB_structure_into_theozyme import parse_cli
    src_parse = inspect.getsource(parse_cli)
    all_flags = re.findall(r'add_argument\("(--[^"]+)"', src_parse)
    # --input_pdb and --output_pdb_path are always in the base cmd — skip them
    optional_flags = [f for f in all_flags
                      if f not in ("--input_pdb", "--output_pdb_path")]
    assert len(optional_flags) >= 34, (
        f"Expected >=34 optional flags from parse_cli, got {len(optional_flags)}")

    # ── 2. Assert every flag appears literally in DRIVER_CELL_TEMPLATE.txt ───
    template_path = ("/home/woodbuse/special_scripts/theozyme_and_ligand_handling/"
                     "prepare_PDB_structure_into_theozyme__DRIVER_CELL_TEMPLATE.txt")
    assert os.path.isfile(template_path), f"Template not found: {template_path}"
    template_content = open(template_path).read()

    # Syntactic validity (belt-and-suspenders)
    compile(template_content, template_path, "exec")

    missing_from_template = [f for f in optional_flags if f not in template_content]
    assert not missing_from_template, (
        f"Flags missing from DRIVER_CELL_TEMPLATE.txt: {missing_from_template}")

    # ── 3. Assert build_command can emit every flag ───────────────────────────
    drv = importlib.import_module("prepare_PDB_structure_into_theozyme__DRIVER_CELL")

    # Call A: flags compatible with frag_ncAA=None, residues_to_keep group,
    #         ligands_to_keep group, protect_ncAA_from_ligandization (no frag).
    #         Also exercises merge, clean, verbose, renumber, ccd, rosetta, etc.
    cmd_a = drv.build_command(
        input_pdb="/x/in.pdb",
        output_pdb="/x/out.pdb",
        # REMARK 666
        remark666_exclude_residues=["A131"],
        remark_front=["A55", "A57"],
        remark_back=["A233"],
        complete_remark666=True,
        force_regenerate_remark666=False,   # mutually-exclusive with complete in practice; keep False
        remark666_template_ligand="SUB",
        remark666_template_chain="X",
        remark666_template_resi="0",
        clean_remarks=True,
        # filtering — keep group
        residues_to_keep=["A55"],
        residues_to_throw_away=None,
        ligands_to_keep=["Z:SUB:999"],
        ligands_to_throw_away=None,
        # ncAA — no frag; protect
        frag_ncAA=None,
        protect_ncAA_from_ligandization=True,
        leave_ncAA_as_ATOM=False,
        add_CA_to_labeled_frag=False,
        protect_sidechain_polarH=["A32"],
        disable_intelligent_hstrip=False,
        # merge
        merge_ligands_as="LIG",
        merged_ligand_chain="Z",
        merged_ligand_resseq="999",
        merge_only=["LIG", "CO"],
        # legacy cleaning
        strip_insertion_codes=True,
        strip_protein_hydrogens=True,
        blank_segid=True,
        strip_partial_charges=True,
        strip_formal_charges=True,
        # layout
        theozyme_layout=True,
        # renumber / checks
        renumber_atoms=True,
        no_ligand_code_checks=True,
        rosetta_residue_types="/path/to/residue_types.txt",
        ccd_timeout=8.0,
        verbose=True,
    )

    # Call B: exercises the mutually-exclusive alternatives and nargs="*" zero-arg forms
    #   - residues_to_throw_away (not keep)
    #   - ligands_to_throw_away (not keep)
    #   - frag_ncAA=[] (all ncAAs, so protect/leave must be False)
    #   - protect_sidechain_polarH=[] (all protein residues)
    #   - leave_ncAA_as_ATOM (alternative to protect)
    #   (we do two separate calls to avoid mutual-exclusion raises)
    cmd_b1 = drv.build_command(
        input_pdb="/x/in.pdb",
        output_pdb="/x/out.pdb",
        residues_to_throw_away=["A169"],
        residues_to_keep=None,
        ligands_to_throw_away=["HOH"],
        ligands_to_keep=None,
        frag_ncAA=[],                       # [] = fragment all ncAAs
        protect_ncAA_from_ligandization=False,
        leave_ncAA_as_ATOM=False,
        protect_sidechain_polarH=[],        # [] = protect all protein residues
    )
    cmd_b2 = drv.build_command(
        input_pdb="/x/in.pdb",
        output_pdb="/x/out.pdb",
        frag_ncAA=None,
        protect_ncAA_from_ligandization=False,
        leave_ncAA_as_ATOM=True,
        disable_intelligent_hstrip=True,
    )

    # Map each optional flag to the call(s) that should emit it
    flag_to_cmd = {
        "--force_regenerate_remark666":       None,   # skip: False in A, handled below
        "--complete_remark666":               cmd_a,
        "--remark666_exclude_residues":       cmd_a,
        "--remark666_template_ligand":        cmd_a,
        "--remark666_template_chain":         cmd_a,
        "--remark666_template_resi":          cmd_a,
        "--remark666_residue_front_order":    cmd_a,
        "--remark666_residue_back_order":     cmd_a,
        "--clean_remarks":                    cmd_a,
        "--residues_to_keep":                 cmd_a,
        "--residues_to_throw_away":           cmd_b1,
        "--ligands_to_keep":                  cmd_a,
        "--ligands_to_throw_away":            cmd_b1,
        "--frag_ncAA_into_cAA_plus_lig":      cmd_b1,
        "--protect_ncAA_from_ligandization":  cmd_a,
        "--leave_ncAA_as_ATOM":               cmd_b2,
        "--add_CA_to_labeled_frag":           None,   # False in all calls; bool flag
        "--protect_sidechain_polarH":         cmd_a,
        "--disable_intelligent_hstrip":       cmd_b2,
        "--strip_insertion_codes":            cmd_a,
        "--strip_protein_hydrogens":          cmd_a,
        "--blank_segid":                      cmd_a,
        "--strip_partial_charges":            cmd_a,
        "--strip_formal_charges":             cmd_a,
        "--merge_ligands_as":                 cmd_a,
        "--merged_ligand_chain":              cmd_a,
        "--merged_ligand_resseq":             cmd_a,
        "--merge_only":                       cmd_a,
        "--theozyme_layout":                  cmd_a,
        "--renumber_atoms":                   cmd_a,
        "--no_ligand_code_checks":            cmd_a,
        "--rosetta_residue_types":            cmd_a,
        "--ccd_timeout":                      cmd_a,
        "--verbose":                          cmd_a,
    }

    # For bool flags not exercised above, make a dedicated call
    cmd_bool = drv.build_command(
        input_pdb="/x/in.pdb",
        output_pdb="/x/out.pdb",
        force_regenerate_remark666=True,
        add_CA_to_labeled_frag=True,
        frag_ncAA=None,
        protect_ncAA_from_ligandization=False,
        leave_ncAA_as_ATOM=False,
    )
    flag_to_cmd["--force_regenerate_remark666"] = cmd_bool
    flag_to_cmd["--add_CA_to_labeled_frag"]     = cmd_bool

    # Check every flag appears in its designated command
    missing_from_build = []
    for flag, cmd in flag_to_cmd.items():
        if cmd is None:
            missing_from_build.append(f"{flag} (no test call mapped)")
            continue
        if flag not in cmd:
            missing_from_build.append(flag)

    assert not missing_from_build, (
        f"build_command failed to emit these flags: {missing_from_build}")


###############################################################################
# DEFECT 1 regression: fresh REMARK 666 generated when none exist
###############################################################################

_RAW_PDB_PROT_LIG = [
    "ATOM      1  N   HIS A  55      48.399  26.664  28.044  1.00  0.00           N",
    "ATOM      2  CA  HIS A  55      48.000  27.000  28.000  1.00  0.00           C",
    "ATOM      3  N   ASP A 233      52.664  37.571  32.340  1.00  0.00           N",
    "HETATM    4  P1  SUB Z 999      48.416  36.273  23.466  1.00  0.78           P",
    "END",
]


def test_defect1_fresh_666_generated_single_ligand():
    """Raw PDB with protein residues + 1 ligand, zero REMARK 666 → fresh 666
    for each protein residue, exact 665 pair, contiguous IDX, 4-wide VAR.
    Pre-fix: remark666_manager returned verbatim (no 666 lines).
    Post-fix: trigger_h fires → full fresh generation."""
    mod = load_mod()
    lines = list(_RAW_PDB_PROT_LIG)
    s = mod.scan_structure(lines)
    assert not s.remark666, "Pre-condition: zero existing 666 entries"
    wl = mod.WarningLog(verbose=False)
    args = mod.parse_cli(["--input_pdb", "i", "--output_pdb_path", "o"])
    out = mod.remark666_manager(lines, s, None, args, wl)

    r666 = [l for l in out if l.startswith("REMARK 666")]
    r665 = [l for l in out if l.startswith("REMARK 665") and "REMARK 666" in l]

    # Must generate 666 for each protein residue (HIS55 + ASP233)
    assert len(r666) == 2, f"Expected 2 fresh 666 lines, got {len(r666)}: {r666}"

    # Exact 665 pair: both canonical R665_HEADER strings, each exactly once
    assert len(r665) == 2, f"Expected 2 R665 header lines, got {len(r665)}"
    assert r665[0] == mod.R665_HEADER[0], f"First R665 mismatch: {r665[0]!r}"
    assert r665[1] == mod.R665_HEADER[1], f"Second R665 mismatch: {r665[1]!r}"

    # Contiguous IDX 1..2
    idxs = [int(l.split()[-2]) for l in r666]
    assert idxs == [1, 2], f"Expected contiguous IDX [1,2], got {idxs}"

    # 4-wide VAR field (last token in each line is VAR)
    for l in r666:
        var_tok = l.split()[-1]
        assert var_tok == "1", f"VAR should be '1', got {var_tok!r}"

    # Template ligand auto-inferred from single ligand (SUB)
    for l in r666:
        assert "SUB" in l, f"Expected ligand token 'SUB' in: {l}"

    # No spurious warnings (no partial-coverage warn since trigger_h covers all)
    assert not wl.warnings, f"Expected no warnings, got: {wl.warnings}"


def test_defect1_fresh_666_multi_ligand_dies():
    """Raw PDB with protein + ≥2 distinct ligands, no --remark666_template_ligand
    → SystemExit from _die with a clear message.
    Pre-fix: returned verbatim (no 666 generated, no error).
    Post-fix: trigger_h fires → template inference → _die on ambiguous ligand."""
    import pytest
    mod = load_mod()
    lines = [
        "ATOM      1  N   HIS A  55      48.399  26.664  28.044  1.00  0.00           N",
        "HETATM    2  P1  SUB Z 999      48.416  36.273  23.466  1.00  0.78           P",
        "HETATM    3  C1  CO2 X 888      50.000  36.000  23.000  1.00  0.50           C",
        "END",
    ]
    s = mod.scan_structure(lines)
    assert not s.remark666
    wl = mod.WarningLog(verbose=False)
    args = mod.parse_cli(["--input_pdb", "i", "--output_pdb_path", "o"])
    with pytest.raises(SystemExit):
        mod.remark666_manager(lines, s, None, args, wl)


def test_defect1_ts1_unaffected():
    """TS1 already has 9 REMARK 666 lines → trigger_h must NOT fire → preserve
    path returns lines verbatim (no fresh generation)."""
    mod = load_mod()
    _TS1 = ("/home/woodbuse/projects/organophosphatase/pxn/"
            "design_campaign_i4__pte_hbond_260515/theozymes/"
            "kcx_set1__pte_hbond/pte_kcx_hbond_TS1.pdb")
    with open(_TS1) as fh:
        all_lines = [l.rstrip("\n") for l in fh]
    s = mod.scan_structure(all_lines)
    original_count = len(s.remark666)
    assert original_count == 9, f"TS1 should have 9 REMARK 666 entries, got {original_count}"
    wl = mod.WarningLog(verbose=False)
    args = mod.parse_cli(["--input_pdb", "i", "--output_pdb_path", "o"])
    out = mod.remark666_manager(all_lines, s, None, args, wl)
    out_count = sum(1 for l in out if l.startswith("REMARK 666"))
    assert out_count == 9, f"TS1 preserve path: 9 → {out_count} (trigger_h must not fire)"
    assert not wl.warnings, f"TS1 preserve path should have no warnings: {wl.warnings}"


###############################################################################
# DEFECT 2 regression: 665 header validation uses exact-pair check
###############################################################################

def test_defect2_duplicate_header_line1_normalized():
    """Input with line 0 duplicated AND line 1 (fmt) missing → trigger_g fires
    → output contains exactly the correct 2-line pair once.
    Pre-fix: count==2 was satisfied by two copies of line 0 → trigger_g=False
    → preserve path left the malformed header untouched.
    Post-fix: exact-pair check fails → trigger_g=True → manager normalizes."""
    mod = load_mod()
    lines = [
        mod.R665_HEADER[0],           # line 0 present
        mod.R665_HEADER[0],           # line 0 AGAIN (duplicate, line 1 missing)
        "REMARK 666 MATCH TEMPLATE X  SUB    0 MATCH MOTIF A HIS   55   1   1",
        "ATOM      1  N   HIS A  55      48.399  26.664  28.044  1.00  0.00           N",
        "HETATM    2  P1  SUB Z 999      48.416  36.273  23.466  1.00  0.78           P",
    ]
    s = mod.scan_structure(lines)
    # Pre-condition: not an exact pair (line 0 twice, line 1 absent)
    assert not s.remark665_header_present, \
        "scan_structure must report False for the malformed pair"
    wl = mod.WarningLog(verbose=False)
    args = mod.parse_cli(["--input_pdb", "i", "--output_pdb_path", "o"])
    out = mod.remark666_manager(lines, s, None, args, wl)

    # Output must contain exactly the correct 2-line pair
    c0 = sum(1 for l in out if l == mod.R665_HEADER[0])
    c1 = sum(1 for l in out if l == mod.R665_HEADER[1])
    assert c0 == 1, f"R665_HEADER[0] should appear exactly once, got {c0}"
    assert c1 == 1, f"R665_HEADER[1] should appear exactly once, got {c1}"

    # 666 body lines must be preserved
    r666 = [l for l in out if l.startswith("REMARK 666")]
    assert len(r666) == 1, f"Expected 1 REMARK 666 line, got {len(r666)}"


def test_defect2_ts1_exact_pair_recognized():
    """TS1 has the exact 665 pair → scan_structure must set
    remark665_header_present=True → trigger_g=False → preserve path."""
    mod = load_mod()
    _TS1 = ("/home/woodbuse/projects/organophosphatase/pxn/"
            "design_campaign_i4__pte_hbond_260515/theozymes/"
            "kcx_set1__pte_hbond/pte_kcx_hbond_TS1.pdb")
    with open(_TS1) as fh:
        all_lines = [l.rstrip("\n") for l in fh]
    s = mod.scan_structure(all_lines)
    assert s.remark665_header_present, \
        "TS1 has the exact R665 pair; remark665_header_present must be True"


###############################################################################
# DEFECT 3 regression: --clean_remarks implemented
###############################################################################

def test_defect3_clean_remarks_strips_other_remarks():
    """With --clean_remarks, non-665/666 REMARK lines are dropped.
    Without the flag, they are preserved (DESIGN §5).
    Pre-fix: flag was parsed but never acted on.
    Post-fix: remark666_manager filters non-665/666 REMARKs from the body."""
    mod = load_mod()
    lines = [
        "REMARK 350 BIOMOLECULE: 1",
        "REMARK   2 RESOLUTION.    1.80 ANGSTROMS.",
        mod.R665_HEADER[0],
        mod.R665_HEADER[1],
        "REMARK 666 MATCH TEMPLATE X  SUB    0 MATCH MOTIF A HIS   55   1   1",
        "ATOM      1  N   HIS A  55      48.399  26.664  28.044  1.00  0.00           N",
        "HETATM    2  P1  SUB Z 999      48.416  36.273  23.466  1.00  0.78           P",
    ]
    s = mod.scan_structure(lines)

    # WITH --clean_remarks
    wl = mod.WarningLog(verbose=False)
    args_clean = mod.parse_cli(["--input_pdb", "i", "--output_pdb_path", "o",
                                "--clean_remarks"])
    out_clean = mod.remark666_manager(lines, s, None, args_clean, wl)
    assert not any(l.startswith("REMARK 350") for l in out_clean), \
        "REMARK 350 must be dropped with --clean_remarks"
    assert not any(l.startswith("REMARK   2") for l in out_clean), \
        "REMARK 2 must be dropped with --clean_remarks"
    # 665 header and 666 must be retained
    assert sum(1 for l in out_clean if l == mod.R665_HEADER[0]) == 1
    assert sum(1 for l in out_clean if l.startswith("REMARK 666")) == 1

    # WITHOUT --clean_remarks — other REMARKs preserved
    args_no = mod.parse_cli(["--input_pdb", "i", "--output_pdb_path", "o"])
    out_no = mod.remark666_manager(lines, s, None, args_no, mod.WarningLog(verbose=False))
    assert any(l.startswith("REMARK 350") for l in out_no), \
        "REMARK 350 must be preserved without --clean_remarks"
    assert any(l.startswith("REMARK   2") for l in out_no), \
        "REMARK 2 must be preserved without --clean_remarks"


def test_defect3_ts1_no_clean_remarks_byte_identical():
    """TS1 + no --clean_remarks → preserve path must still be byte-identical.
    This confirms the DEFECT-3 fix does not perturb the TS1 invariant."""
    import tempfile, os
    mod = load_mod()
    src = ("/home/woodbuse/projects/organophosphatase/pxn/"
           "design_campaign_i4__pte_hbond_260515/theozymes/"
           "kcx_set1__pte_hbond/pte_kcx_hbond_TS1.pdb")
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        out_path = f.name
    try:
        mod.main(["--input_pdb", src, "--output_pdb_path", out_path,
                  "--no_ligand_code_checks"])
        src_lines = [l.rstrip() for l in open(src)]
        out_lines = [l.rstrip() for l in open(out_path)]
        assert src_lines == out_lines, "TS1 must be byte-identical with no flags"
    finally:
        os.unlink(out_path)


###############################################################################
# DEFECT 4 regression: add_CA_to_labeled_frag called wrong
###############################################################################

def test_defect4_add_CA_to_labeled_frag_no_type_error():
    """Calling add_CA_to_labeled_frag with wl as kwarg must not raise TypeError.
    Pre-fix: main() called add_CA_to_labeled_frag(lines, wl) → wl bound to
    ca_cb_bond_length parameter → TypeError on any missing-CA residue.
    Post-fix: add_CA_to_labeled_frag(lines, wl=wl) binds wl correctly."""
    mod = load_mod()
    wl = mod.WarningLog(verbose=False)
    # A residue missing CA but with CB and a CB-bound H
    lines = [
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N",
        "ATOM      2  CB  ALA A   1       1.520   0.000   0.000  1.00  0.00           C",
        "ATOM      3  HB1 ALA A   1       2.540   0.000   0.000  1.00  0.00           H",
    ]
    # Must not raise TypeError; must insert a CA line
    try:
        out = mod.add_CA_to_labeled_frag(lines, wl=wl)
    except TypeError as e:
        raise AssertionError(
            f"add_CA_to_labeled_frag raised TypeError with wl kwarg: {e}") from e
    ca_lines = [l for l in out if l[12:16].strip() == "CA"]
    assert ca_lines, "Expected a CA atom to be inserted"


def test_defect4_add_CA_noop_when_ca_present():
    """Residue that already has CA → add_CA_to_labeled_frag is a strict no-op."""
    mod = load_mod()
    wl = mod.WarningLog(verbose=False)
    lines = [
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N",
        "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C",
        "ATOM      3  CB  ALA A   1       2.000   0.000   0.000  1.00  0.00           C",
    ]
    out = mod.add_CA_to_labeled_frag(lines, wl=wl)
    assert out == lines, "No-op when CA is already present"


###############################################################################
# DEFECT 5 regression: loose .params match warning
###############################################################################

def test_defect5_loose_params_warning_emitted(tmp_path):
    """A line containing '.params' AND the code (e.g. A2LATB.params for LAT)
    without a filename/standalone match → _rosetta_status returns (absent, True)
    → check_ligand_codes emits a 'rosetta_loose' category warning.
    Pre-fix: _rosetta_status only returned a string; loose occurrence ignored.
    Post-fix: returns 2-tuple; check_ligand_codes emits distinct non-fatal warn."""
    mod = load_mod()
    # A2LATB.params: .params in line, LAT in line, but not LAT.params (fn match)
    # and not \bLAT\b (sa match — L is preceded by non-word-boundary chars)
    rt = tmp_path / "residue_types.txt"
    rt.write_text("residue_types/l-caa/OTHER.params\n"
                  "residue_types/nucleic/A2LATB.params\n")

    status, loose = mod._rosetta_status("LAT", str(rt))
    assert status == "absent", f"Expected 'absent', got {status!r}"
    assert loose is True, f"Expected loose=True for A2LATB.params"

    wl = mod.WarningLog(verbose=False)
    result = mod.check_ligand_codes(["LAT"], str(rt), 4.0, wl)
    assert result["LAT"]["rosetta"] == "absent"
    loose_warns = [(cat, msg) for cat, msg in wl.warnings if cat == "rosetta_loose"]
    assert loose_warns, (
        f"Expected a 'rosetta_loose' warning for LAT; got: {wl.warnings}")
    # Must be non-fatal (no sys.exit, result is a normal dict)
    assert isinstance(result, dict)


def test_defect5_clean_file_no_loose_warning(tmp_path):
    """A file with no occurrence of the code → no loose warning."""
    mod = load_mod()
    rt = tmp_path / "residue_types.txt"
    rt.write_text("residue_types/l-caa/OTHER.params\nsome_other_line\n")
    wl = mod.WarningLog(verbose=False)
    mod.check_ligand_codes(["LAT"], str(rt), 4.0, wl)
    assert not any(cat == "rosetta_loose" for cat, _ in wl.warnings), \
        f"Clean file must not emit rosetta_loose warning: {wl.warnings}"


###############################################################################
# DEFECT 6 regression: driver template uses valid selector syntax
###############################################################################

def test_defect6_driver_template_no_bare_code_in_merge_only():
    """DRIVER_CELL_TEMPLATE.txt must NOT show bare codes like 'LIG','CO' as
    --merge_only examples; must show valid parse_selector syntax instead.
    Pre-fix: merge_only = ['LIG','CO'] appeared in the template → user would
    copy it, hit parse_selector ValueError at runtime.
    Post-fix: example updated to valid selector syntax like 'Z:LIG:901'."""
    import os
    path = ("/home/woodbuse/special_scripts/theozyme_and_ligand_handling/"
            "prepare_PDB_structure_into_theozyme__DRIVER_CELL_TEMPLATE.txt")
    assert os.path.isfile(path)
    content = open(path).read()

    # The old bad example (bare codes with no chain+resseq) must not appear
    import re as _re
    bad_pattern = _re.compile(r'merge_only\s*=\s*\["[A-Z]{2,3}"\s*,\s*"[A-Z]{2,3}"\]')
    assert not bad_pattern.search(content), (
        "merge_only example in template must not use bare 3-letter codes without selectors")


def test_defect6_driver_cell_merge_only_docstring_valid():
    """DRIVER_CELL.py merge_only parameter docstring must indicate valid selector
    syntax (chain+resseq or CHAIN:RESNAME:RESSEQ), not bare codes.
    Pre-fix: comment said 'list[str] → --merge_only' with no selector hint.
    Post-fix: comment explicitly notes selector form."""
    content = open("/home/woodbuse/special_scripts/theozyme_and_ligand_handling/"
                   "prepare_PDB_structure_into_theozyme__DRIVER_CELL.py").read()
    # Must mention selector syntax (at least one of these keywords)
    assert ("selector" in content or "A901" in content or "RESNAME" in content
            or "CHAIN:RESNAME:RESSEQ" in content), (
        "DRIVER_CELL.py merge_only must document valid selector syntax")


# ---------------------------------------------------------------------------
# Waters are normal HETATM ligands by default; --preserve_waters opts back
# into the old special handling (always-keep + never-merge).
# ---------------------------------------------------------------------------

_WAT_PDB = [
    "ATOM      1  N   HIS A  55      48.399  26.664  28.044  1.00  0.00           N",
    "HETATM    2  P1  SUB Z 999      48.416  36.273  23.466  1.00  0.00           P",
    "HETATM    3  O   HOH W   1      10.000  10.000  10.000  1.00  0.00           O",
]


def test_waters_default_thrown_like_any_ligand():
    mod = load_mod()
    wl = mod.WarningLog(verbose=False)
    # default (preserve_waters not passed) -> water is a normal ligand and
    # is removed by an explicit ligand throw selector.
    kept, dropped = mod.filter_structure(
        list(_WAT_PDB), None, None, None, ["W:HOH:1"], wl)
    txt = "\n".join(kept)
    assert "HOH W   1" not in txt
    assert ("W", "1", " ") in dropped["ligands"]
    assert "P1  SUB Z 999" in txt


def test_waters_default_dropped_when_not_in_ligands_to_keep():
    mod = load_mod()
    wl = mod.WarningLog(verbose=False)
    kept, dropped = mod.filter_structure(
        list(_WAT_PDB), None, None, ["Z:SUB:999"], None, wl)
    txt = "\n".join(kept)
    assert "P1  SUB Z 999" in txt
    assert "HOH W   1" not in txt          # not in keep list -> dropped
    assert ("W", "1", " ") in dropped["ligands"]


def test_preserve_waters_flag_keeps_and_excludes_from_merge():
    mod = load_mod()
    wl = mod.WarningLog(verbose=False)
    # preserve_waters=True -> water bypasses ligand filter (kept even though
    # not in ligands_to_keep) ...
    kept, dropped = mod.filter_structure(
        list(_WAT_PDB), None, None, ["Z:SUB:999"], None, wl,
        preserve_waters=True)
    assert "HOH W   1" in "\n".join(kept)
    assert ("W", "1", " ") not in dropped["ligands"]
    # ... and is NOT merged.
    a = mod.parse_cli(["--input_pdb", "i", "--output_pdb_path", "o",
                       "--merge_ligands_as", "LIG", "--preserve_waters"])
    merged = mod._merge_ligands(list(_WAT_PDB), a, wl)
    mtxt = "\n".join(merged)
    assert "HOH W   1" in mtxt              # water untouched
    assert " LIG " in mtxt                  # SUB still merged


def test_waters_merged_by_default():
    mod = load_mod()
    wl = mod.WarningLog(verbose=False)
    a = mod.parse_cli(["--input_pdb", "i", "--output_pdb_path", "o",
                       "--merge_ligands_as", "LIG"])
    merged = mod._merge_ligands(list(_WAT_PDB), a, wl)
    mtxt = "\n".join(merged)
    assert "HOH W   1" not in mtxt          # water merged like any HETATM
    assert mtxt.count(" LIG ") >= 2         # SUB + former HOH now both LIG
