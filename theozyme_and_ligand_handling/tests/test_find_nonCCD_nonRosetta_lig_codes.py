import os
import sys
import importlib

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
M = importlib.import_module("find_nonCCD_nonRosetta_lig_codes")

ROS = [
    "residue_types/l-caa/ALA.params",
    "residue_types/metal_ions/ZN.params",
    "residue_types/ligands/CO2.params",          # [filename] for CO2
    "# a standalone note about CO2 here",          # [standalone] for CO2
    "residue_types/foo/CO2X.params",               # [warning] for CO2
]


def test_parse_class_spec_ok_and_bad():
    name, patt = M.parse_class_spec("CLASS_0:S??")
    assert name == "CLASS_0" and patt == ["S", None, None]
    with pytest.raises(ValueError):
        M.parse_class_spec("NOPATTERN")
    with pytest.raises(ValueError):
        M.parse_class_spec("X:ABCD")  # not 3 chars


def test_codes_from_pattern():
    codes = list(M.codes_from_pattern(["S", None], "AB"))
    assert codes == ["SA", "SB"]


def test_code_in_rosetta_categories():
    found, matches = M.code_in_rosetta("CO2", ROS)
    assert found
    cats = " ".join(matches)
    assert "[filename]" in cats          # CO2.params
    assert "[standalone]" in cats        # "...note about CO2 here"
    assert "[warning]" in cats           # CO2X.params line
    found2, _ = M.code_in_rosetta("ZZ9", ROS)
    assert not found2


def test_check_mode_report(capsys, monkeypatch, tmp_path):
    rt = tmp_path / "rt.txt"
    rt.write_text("\n".join(ROS) + "\n")
    monkeypatch.setattr(
        M, "code_ccd_state",
        lambda c, t, v: ("YES", M.BASE_URL.format(code=c)) if c == "CO2"
        else ("no", M.BASE_URL.format(code=c)))
    with pytest.raises(SystemExit) as e:
        M.main(["--code", "CO2", "ZZ9", "--rosetta-txt", str(rt)])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "CO2" in out and "ROSETTA=YES" in out
    assert f"Rosetta txt: {rt}" in out and "lines)" in out
    assert "WARNING" in out and "NOT recommended" in out
    assert "CCD: https://files.rcsb.org/ligands/view/CO2.cif" in out
    assert "ZZ9" in out and "AVAILABLE" in out
    assert "=== SUMMARY ===" in out and "[RECOMMENDATION]" in out


def test_check_mode_ccd_error_is_uncertain(capsys, monkeypatch, tmp_path):
    rt = tmp_path / "rt.txt"
    rt.write_text("\n".join(ROS) + "\n")
    monkeypatch.setattr(
        M, "code_ccd_state",
        lambda c, t, v: ("ERR", M.BASE_URL.format(code=c)))
    with pytest.raises(SystemExit) as e:
        M.main(["--code", "ZZ9", "--rosetta-txt", str(rt)])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "UNCERTAIN" in out and "lookup" in out


def test_mutually_exclusive(monkeypatch, tmp_path):
    rt = tmp_path / "rt.txt"
    rt.write_text("\n".join(ROS) + "\n")
    with pytest.raises(SystemExit) as e:
        M.main(["--code", "CO2", "--class", "C0:S??",
                "--rosetta-txt", str(rt)])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e2:
        M.main(["--rosetta-txt", str(rt)])  # neither
    assert e2.value.code == 2


def test_search_rosetta_only_no_http(capsys, monkeypatch, tmp_path):
    rt = tmp_path / "rt.txt"
    rt.write_text("\n".join(ROS) + "\n")

    def boom(*a, **k):
        raise AssertionError("CCD HTTP must not be called in --check rosetta")

    monkeypatch.setattr(M, "code_in_ccd", boom)
    # Pattern CO? over letters '12' -> CO1, CO2 ; CO2 is in Rosetta, CO1 is not
    M.main(["--class", "T:CO?", "--letters", "12", "--check", "rosetta",
            "--target-per-class", "5", "--rosetta-txt", str(rt)])
    out = capsys.readouterr().out
    assert "found available code CO1" in out
    assert "CO2" not in out.split("SUGGESTED CODES")[-1]  # CO2 excluded


def test_rosetta_file_missing_errors():
    with pytest.raises(SystemExit) as e:
        M.main(["--code", "CO2", "--check", "rosetta",
                "--rosetta-txt", "/no/such/file.txt"])
    assert e.value.code == 2
