"""
Pytest configuration and fixtures for remastered_fastmpnn tests.
"""

import pytest
from pathlib import Path

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def ref_pdb_path(fixtures_dir) -> Path:
    """Return path to reference PDB fixture."""
    return fixtures_dir / "ref_pdb.pdb"


@pytest.fixture
def input_pdb_path(fixtures_dir) -> Path:
    """Return path to input PDB fixture."""
    return fixtures_dir / "input_pdb.pdb"


@pytest.fixture
def catres_subset_indices() -> list:
    """Return the catres_subset indices for the test PDB."""
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19]


@pytest.fixture
def sample_remark666_line() -> str:
    """Return a sample REMARK 666 line for parsing tests."""
    return "REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A HIS   13  1  1"


@pytest.fixture
def sample_remark666_lines() -> list:
    """Return multiple sample REMARK 666 lines."""
    return [
        "REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A HIS   13  1  1",
        "REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A HIS   15  2  1",
        "REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A HIS  176  3  1",
        "REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A HIS  203  4  1",
        "REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A ASP   53  5  1",
    ]


@pytest.fixture
def tmp_output_dir(tmp_path) -> Path:
    """Return a temporary directory for test outputs."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
