"""Tests for REMARK 666 parsing."""

import pytest
from pathlib import Path
import tempfile

from fastmpnndesign.remark666 import (
    parse_remark666_line,
    parse_remark666_from_pdb,
    parse_remark666_from_lines,
    get_catres_subset,
    catres_to_fixed_residues,
)
from fastmpnndesign.config import CatalyticResidue


class TestParseRemark666Line:
    """Tests for parse_remark666_line function."""

    def test_standard_format(self):
        """Test parsing standard REMARK 666 format."""
        line = "REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A PHE  150  10  1"
        result = parse_remark666_line(line)

        assert result is not None
        assert result.catres_index == 10
        assert result.chain == "A"
        assert result.resnum == 150
        assert result.resname == "PHE"
        assert result.cst_block == 10
        assert result.cst_var == 1

    def test_format_with_insertion_code(self):
        """Test parsing with insertion code."""
        line = "REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A HIS  152A  5  2"
        result = parse_remark666_line(line)

        assert result is not None
        assert result.resnum == 152
        assert result.icode == "A"

    def test_different_residue_names(self):
        """Test parsing various residue names."""
        test_cases = [
            ("REMARK 666 MATCH TEMPLATE B LIG  1 MATCH MOTIF A GLU  45  1  1", "GLU"),
            ("REMARK 666 MATCH TEMPLATE B LIG  1 MATCH MOTIF A HIS  78  2  1", "HIS"),
            ("REMARK 666 MATCH TEMPLATE B LIG  1 MATCH MOTIF A CYS  99  3  1", "CYS"),
            ("REMARK 666 MATCH TEMPLATE B LIG  1 MATCH MOTIF A KCX  120  4  1", "KCX"),
        ]

        for line, expected_resname in test_cases:
            result = parse_remark666_line(line)
            assert result is not None
            assert result.resname == expected_resname

    def test_non_remark666_line(self):
        """Test that non-REMARK 666 lines return None."""
        lines = [
            "ATOM      1  N   ALA A   1      0.000   0.000   0.000  1.00  0.00",
            "REMARK 100 Some other remark",
            "HETATM    1  C1  LIG B   1      0.000   0.000   0.000  1.00  0.00",
        ]

        for line in lines:
            result = parse_remark666_line(line)
            assert result is None

    def test_default_index(self):
        """Test default index when not in line."""
        line = "REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A PHE  150"
        result = parse_remark666_line(line, index=5)

        assert result is not None
        assert result.catres_index == 5


class TestParseRemark666FromLines:
    """Tests for parse_remark666_from_lines function."""

    def test_multiple_lines(self):
        """Test parsing multiple REMARK 666 lines."""
        lines = [
            "REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A PHE  150  1  1\n",
            "REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A GLU  45  2  1\n",
            "REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A HIS  78  3  1\n",
        ]

        result = parse_remark666_from_lines(lines)

        assert len(result) == 3
        assert result[0].catres_index == 1
        assert result[1].catres_index == 2
        assert result[2].catres_index == 3

    def test_mixed_lines(self):
        """Test parsing with non-REMARK 666 lines mixed in."""
        lines = [
            "HEADER    TEST\n",
            "REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A PHE  150  1  1\n",
            "ATOM      1  N   ALA A   1      0.000   0.000   0.000  1.00  0.00\n",
            "REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A GLU  45  2  1\n",
        ]

        result = parse_remark666_from_lines(lines)

        assert len(result) == 2

    def test_sorted_by_index(self):
        """Test that results are sorted by catres_index."""
        lines = [
            "REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A PHE  150  5  1\n",
            "REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A GLU  45  1  1\n",
            "REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A HIS  78  3  1\n",
        ]

        result = parse_remark666_from_lines(lines)

        assert result[0].catres_index == 1
        assert result[1].catres_index == 3
        assert result[2].catres_index == 5


class TestParseRemark666FromPdb:
    """Tests for parse_remark666_from_pdb function."""

    def test_parse_from_file(self):
        """Test parsing REMARK 666 from actual PDB file."""
        pdb_content = """HEADER    TEST
REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A PHE  150  1  1
REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A GLU  45  2  1
ATOM      1  N   ALA A   1      0.000   0.000   0.000  1.00  0.00
ATOM      2  CA  ALA A   1      1.458   0.000   0.000  1.00  0.00
END
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_content)
            tmp_path = Path(f.name)

        try:
            result = parse_remark666_from_pdb(tmp_path)
            assert len(result) == 2
            assert result[0].resname == "GLU"  # Sorted by index
            assert result[1].resname == "PHE"
        finally:
            tmp_path.unlink()

    def test_file_not_found(self):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            parse_remark666_from_pdb(Path("/nonexistent/path.pdb"))


class TestGetCatresSubset:
    """Tests for get_catres_subset function."""

    def test_no_subset(self):
        """Test with no subset specified (all in subset)."""
        catres = [
            CatalyticResidue(catres_index=1, chain="A", resnum=150, resname="PHE"),
            CatalyticResidue(catres_index=2, chain="A", resnum=45, resname="GLU"),
        ]

        subset, non_subset = get_catres_subset(catres, None)

        assert len(subset) == 2
        assert len(non_subset) == 0

    def test_partial_subset(self):
        """Test with partial subset."""
        catres = [
            CatalyticResidue(catres_index=1, chain="A", resnum=150, resname="PHE"),
            CatalyticResidue(catres_index=2, chain="A", resnum=45, resname="GLU"),
            CatalyticResidue(catres_index=3, chain="A", resnum=78, resname="HIS"),
        ]

        subset, non_subset = get_catres_subset(catres, [1, 3])

        assert len(subset) == 2
        assert len(non_subset) == 1
        assert subset[0].catres_index == 1
        assert subset[1].catres_index == 3
        assert non_subset[0].catres_index == 2


class TestCatresToFixedResidues:
    """Tests for catres_to_fixed_residues function."""

    def test_conversion(self):
        """Test conversion to MPNN fixed residues format."""
        catres = [
            CatalyticResidue(catres_index=1, chain="A", resnum=150, resname="PHE"),
            CatalyticResidue(catres_index=2, chain="B", resnum=45, resname="GLU"),
        ]

        result = catres_to_fixed_residues(catres)

        assert result == ["A150", "B45"]

    def test_with_insertion_code(self):
        """Test conversion with insertion codes."""
        catres = [
            CatalyticResidue(catres_index=1, chain="A", resnum=150, resname="PHE", icode="A"),
        ]

        result = catres_to_fixed_residues(catres)

        assert result == ["A150A"]
