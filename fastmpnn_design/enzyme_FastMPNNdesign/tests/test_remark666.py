"""
Tests for REMARK 666 parsing.
"""

import pytest

from remastered_fastmpnn.core.remark666 import (
    parse_remark666_line,
    parse_remark666_lines,
    validate_remark666_consistency,
    get_catres_from_remarks,
    Remark666Info,
)


class TestParseRemark666Line:
    """Tests for parse_remark666_line function."""

    def test_parse_valid_line(self, sample_remark666_line):
        """Test parsing a valid REMARK 666 line."""
        result = parse_remark666_line(sample_remark666_line, line_index=1)

        assert result is not None
        assert result.motif_chain == "A"
        assert result.motif_resname == "HIS"
        assert result.motif_resnum == 13
        assert result.block_index == 1
        assert result.block_variant == 1
        assert result.template_resname == "XDW"
        assert result.template_resnum == 257
        assert result.line_index == 1

    def test_parse_line_with_catres_template(self):
        """Test parsing REMARK 666 with catres as template."""
        line = "REMARK 666 MATCH TEMPLATE A ASP   53 MATCH MOTIF A HIS  226  8  1"
        result = parse_remark666_line(line)

        assert result is not None
        assert result.template_resname == "ASP"
        assert result.template_resnum == 53
        assert result.motif_resname == "HIS"
        assert result.motif_resnum == 226
        assert result.block_index == 8

    def test_parse_invalid_line_no_remark(self):
        """Test that non-REMARK lines return None."""
        line = "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00"
        result = parse_remark666_line(line)
        assert result is None

    def test_parse_invalid_line_no_motif(self):
        """Test that REMARK 666 without MOTIF returns None."""
        line = "REMARK 666 MATCH TEMPLATE B XDW  257"
        result = parse_remark666_line(line)
        assert result is None

    def test_motif_identifier(self):
        """Test that motif_identifier property works correctly."""
        line = "REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A HIS   13  1  1"
        result = parse_remark666_line(line)

        assert result is not None
        assert result.motif_identifier == "A13"

    def test_is_ligand_template(self):
        """Test ligand template detection."""
        # Ligand template
        line1 = "REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A HIS   13  1  1"
        result1 = parse_remark666_line(line1)
        assert result1.is_ligand_template() is True

        # Protein template
        line2 = "REMARK 666 MATCH TEMPLATE A ASP   53 MATCH MOTIF A HIS  226  8  1"
        result2 = parse_remark666_line(line2)
        assert result2.is_ligand_template() is False


class TestParseRemark666Lines:
    """Tests for parse_remark666_lines function."""

    def test_parse_multiple_lines(self, sample_remark666_lines):
        """Test parsing multiple REMARK 666 lines."""
        results = parse_remark666_lines(sample_remark666_lines)

        assert len(results) == 5
        assert results[0].motif_resnum == 13
        assert results[1].motif_resnum == 15
        assert results[2].motif_resnum == 176
        assert results[3].motif_resnum == 203
        assert results[4].motif_resnum == 53

    def test_line_indices_are_1_indexed(self, sample_remark666_lines):
        """Test that line indices start at 1."""
        results = parse_remark666_lines(sample_remark666_lines)

        assert results[0].line_index == 1
        assert results[1].line_index == 2
        assert results[4].line_index == 5


class TestValidateConsistency:
    """Tests for validate_remark666_consistency function."""

    def test_identical_remarks_pass(self, sample_remark666_lines):
        """Test that identical remarks pass validation."""
        remarks1 = parse_remark666_lines(sample_remark666_lines)
        remarks2 = parse_remark666_lines(sample_remark666_lines)

        is_valid, errors = validate_remark666_consistency(remarks1, remarks2)

        assert is_valid is True
        assert len(errors) == 0

    def test_different_counts_fail(self):
        """Test that different counts fail validation."""
        lines1 = ["REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A HIS   13  1  1"]
        lines2 = [
            "REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A HIS   13  1  1",
            "REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A HIS   15  2  1",
        ]

        remarks1 = parse_remark666_lines(lines1)
        remarks2 = parse_remark666_lines(lines2)

        is_valid, errors = validate_remark666_consistency(remarks1, remarks2)

        assert is_valid is False
        assert any("count mismatch" in e.lower() for e in errors)

    def test_different_resname_fails(self):
        """Test that different residue names fail validation."""
        lines1 = ["REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A HIS   13  1  1"]
        lines2 = ["REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A ASP   13  1  1"]

        remarks1 = parse_remark666_lines(lines1)
        remarks2 = parse_remark666_lines(lines2)

        is_valid, errors = validate_remark666_consistency(remarks1, remarks2)

        assert is_valid is False
        assert any("resname mismatch" in e.lower() for e in errors)


class TestGetCatresFromRemarks:
    """Tests for get_catres_from_remarks function."""

    def test_all_as_subset_when_none(self, sample_remark666_lines):
        """Test that all remarks are catres_subset when indices is None."""
        remarks = parse_remark666_lines(sample_remark666_lines)
        subset, motif = get_catres_from_remarks(remarks, None)

        assert len(subset) == 5
        assert len(motif) == 0

    def test_filtering_by_indices(self, sample_remark666_lines):
        """Test filtering by specific indices."""
        remarks = parse_remark666_lines(sample_remark666_lines)
        subset, motif = get_catres_from_remarks(remarks, [1, 3, 5])

        assert len(subset) == 3
        assert len(motif) == 2

        # Verify the right ones are in each group
        subset_indices = {r.line_index for r in subset}
        assert subset_indices == {1, 3, 5}

        motif_indices = {r.line_index for r in motif}
        assert motif_indices == {2, 4}
