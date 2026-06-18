"""Tests for MPNN runner."""

import pytest
from pathlib import Path
import tempfile
import json

from fastmpnndesign.mpnn_runner import (
    build_fixed_residues_json,
    build_mpnn_command,
    parse_mpnn_sequences,
    MPNNResult,
)
from fastmpnndesign.config import MPNNConfig, CatalyticResidue


class TestBuildFixedResiduesJson:
    """Tests for build_fixed_residues_json function."""

    def test_basic_creation(self):
        """Test basic JSON file creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "test.pdb"
            pdb_path.touch()
            output_path = Path(tmpdir) / "fixed.json"

            fixed_residues = ["A150", "A152", "B45"]

            result = build_fixed_residues_json(pdb_path, fixed_residues, output_path)

            assert result == output_path
            assert output_path.exists()

            with open(output_path, 'r') as f:
                data = json.load(f)

            assert "test" in data
            assert data["test"] == ["A150", "A152", "B45"]

    def test_empty_residues(self):
        """Test with empty fixed residues list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "test.pdb"
            pdb_path.touch()
            output_path = Path(tmpdir) / "fixed.json"

            result = build_fixed_residues_json(pdb_path, [], output_path)

            with open(output_path, 'r') as f:
                data = json.load(f)

            assert data["test"] == []


class TestBuildMpnnCommand:
    """Tests for build_mpnn_command function."""

    def test_basic_command(self):
        """Test basic command construction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "test.pdb"
            output_dir = Path(tmpdir) / "output"
            fixed_json = Path(tmpdir) / "fixed.json"

            config = MPNNConfig(
                mpnn_runner=Path("/path/to/run.py"),
                model_type="ligand_mpnn",
                temperature=0.1,
                number_of_batches=10,
                batch_size=1,
                enhance_model="plddt_3_20240930-f9c9ea0f",
                omit_AA="CM",
                use_apptainer=False
            )

            cmd = build_mpnn_command(
                pdb_path, output_dir, fixed_json, config
            )

            assert "python" in cmd
            assert "/path/to/run.py" in cmd
            assert "--model_type" in cmd
            assert "ligand_mpnn" in cmd
            assert "--temperature" in cmd
            assert "0.1" in cmd
            assert "--enhance" in cmd
            assert "plddt_3_20240930-f9c9ea0f" in cmd
            assert "--omit_AA" in cmd
            assert "CM" in cmd

    def test_apptainer_command(self):
        """Test command with Apptainer wrapper."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "test.pdb"
            output_dir = Path(tmpdir) / "output"

            config = MPNNConfig(
                mpnn_runner=Path("/path/to/run.py"),
                use_apptainer=True,
                apptainer_image=Path("/path/to/image.sif")
            )

            cmd = build_mpnn_command(pdb_path, output_dir, None, config)

            assert cmd[0] == "apptainer"
            assert cmd[1] == "exec"
            assert "/path/to/image.sif" in cmd
            assert "python" in cmd

    def test_no_enhance(self):
        """Test command without enhancement model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "test.pdb"
            output_dir = Path(tmpdir) / "output"

            config = MPNNConfig(
                mpnn_runner=Path("/path/to/run.py"),
                enhance_model=None
            )

            cmd = build_mpnn_command(pdb_path, output_dir, None, config)

            assert "--enhance" not in cmd


class TestParseMpnnSequences:
    """Tests for parse_mpnn_sequences function."""

    def test_parse_fasta(self):
        """Test parsing MPNN FASTA output."""
        fasta_content = """>design_0, score=-2.345, seq_recovery=0.85
MKTAYIAKQRQISFVKSHFSRQDILDLWIYHT
>design_1, score=-2.123, seq_recovery=0.82
MKTAYIAKQRQISFVKSHFSRQDILDLWIYHT
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            seqs_dir = Path(tmpdir)
            fasta_file = seqs_dir / "test.fa"

            with open(fasta_file, 'w') as f:
                f.write(fasta_content)

            sequences = parse_mpnn_sequences(seqs_dir)

            assert len(sequences) == 2
            assert sequences[0]['name'] == "design_0"
            assert sequences[0]['score'] == -2.345
            assert sequences[0]['sequence'].startswith("MKTAY")

    def test_empty_directory(self):
        """Test with no FASTA files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sequences = parse_mpnn_sequences(Path(tmpdir))
            assert sequences == []


class TestMPNNResult:
    """Tests for MPNNResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = MPNNResult(
            success=True,
            output_dir=Path("/tmp/output")
        )

        assert result.success
        assert result.n_sequences == 0
        assert result.sequences == []
        assert result.command == ""

    def test_with_sequences(self):
        """Test with populated sequences."""
        result = MPNNResult(
            success=True,
            output_dir=Path("/tmp/output"),
            n_sequences=5,
            sequences=[{"name": "seq1", "sequence": "MKTAY"}]
        )

        assert result.n_sequences == 5
        assert len(result.sequences) == 1
