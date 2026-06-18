"""
Integration tests for the full Step 1 pipeline.
"""

import pytest
from pathlib import Path
import json

from remastered_fastmpnn.core.pdb_io import PDBParser
from remastered_fastmpnn.core.residue_data import ResidueRegistry, ImportantComponent
from remastered_fastmpnn.step1_coordinate_transform.coordinate_transformer import (
    CoordinateTransformer,
)
from remastered_fastmpnn.step1_coordinate_transform.cli import main as cli_main


class TestFullPipeline:
    """Integration tests for the full Step 1 pipeline."""

    def test_end_to_end_transformation(
        self, ref_pdb_path, input_pdb_path, tmp_output_dir, catres_subset_indices
    ):
        """Test complete end-to-end transformation."""
        output_pdb = tmp_output_dir / "e2e_output.pdb"
        output_json = tmp_output_dir / "e2e_registry.json"

        # Run transformation
        transformer = CoordinateTransformer(
            input_pdb=input_pdb_path,
            ref_pdb=ref_pdb_path,
            catres_subset_indices=catres_subset_indices,
            verbose=False,
        )

        result_pdb, registry = transformer.run(output_pdb)

        # Save registry
        registry.save_json(output_json)

        # Verify outputs exist
        assert output_pdb.exists(), "Output PDB should exist"
        assert output_json.exists(), "Output JSON should exist"

        # Verify PDB is valid
        output_parser = PDBParser(output_pdb)
        assert len(output_parser.atoms) > 0, "Output PDB should have atoms"

        # Verify REMARK 666 lines preserved
        assert len(output_parser.remark666_lines) > 0, "REMARK 666 should be preserved"

        # Verify registry structure
        with open(output_json) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "residues" in data
        assert data["metadata"]["catres_subset_count"] > 0

    def test_catres_coords_match_reference(
        self, ref_pdb_path, input_pdb_path, tmp_output_dir, catres_subset_indices
    ):
        """Test that catres coordinates in output match reference."""
        import copy
        from remastered_fastmpnn.core.coordinate_utils import align_by_ligand

        output_pdb = tmp_output_dir / "coord_check.pdb"

        # Run transformation
        transformer = CoordinateTransformer(
            input_pdb=input_pdb_path,
            ref_pdb=ref_pdb_path,
            catres_subset_indices=catres_subset_indices,
            verbose=False,
        )

        result_pdb, registry = transformer.run(output_pdb)

        # Load output and aligned reference
        output_parser = PDBParser(output_pdb)
        ref_parser = PDBParser(ref_pdb_path)
        input_parser = PDBParser(input_pdb_path)

        # Align reference to input (same as transformer does)
        aligned_ref, rmsd, R, t = align_by_ligand(
            copy.deepcopy(ref_parser), input_parser
        )

        # Check that catres_subset sidechain coords match aligned reference
        for residue in registry.get_catres_subset():
            if residue.catres_subset_info and residue.catres_subset_info.sidechain_coords_copied:
                chain = residue.chain
                resnum = residue.residue_num

                output_atoms = output_parser.get_residue_atoms(chain, resnum)
                ref_atoms = aligned_ref.get_residue_atoms(chain, resnum)

                if not output_atoms or not ref_atoms:
                    continue

                # Compare sidechain atoms
                for out_atom in output_atoms:
                    if out_atom.is_sidechain():
                        name = out_atom.name.strip()
                        ref_match = [a for a in ref_atoms if a.name.strip() == name]
                        if ref_match:
                            ref_atom = ref_match[0]
                            # Coordinates should match
                            assert abs(out_atom.x - ref_atom.x) < 0.01
                            assert abs(out_atom.y - ref_atom.y) < 0.01
                            assert abs(out_atom.z - ref_atom.z) < 0.01

    def test_non_catres_unchanged(
        self, ref_pdb_path, input_pdb_path, tmp_output_dir
    ):
        """Test that non-catres residues are unchanged."""
        output_pdb = tmp_output_dir / "unchanged_check.pdb"

        # Only make first residue catres_subset
        transformer = CoordinateTransformer(
            input_pdb=input_pdb_path,
            ref_pdb=ref_pdb_path,
            catres_subset_indices=[1],
            verbose=False,
        )

        result_pdb, registry = transformer.run(output_pdb)

        # Load input and output
        input_parser = PDBParser(input_pdb_path)
        output_parser = PDBParser(output_pdb)

        # Find a residue that is NOT in the catres list
        catres_ids = {r.identifier for r in registry.get_catres_subset()}
        catres_ids.update({r.identifier for r in registry.get_conserved_motif()})

        # Check that some other residue (not in any catres list) is unchanged
        for chain, resnum, resname, atoms in input_parser.iter_residues():
            identifier = f"{chain}{resnum}"
            if identifier not in catres_ids and resname not in {"XDW"}:
                # This residue should be unchanged
                output_atoms = output_parser.get_residue_atoms(chain, resnum)

                for inp_atom in atoms[:3]:  # Check first 3 atoms
                    name = inp_atom.name.strip()
                    out_match = [a for a in output_atoms if a.name.strip() == name]
                    if out_match:
                        out_atom = out_match[0]
                        # Coordinates should be identical
                        assert inp_atom.x == out_atom.x
                        assert inp_atom.y == out_atom.y
                        assert inp_atom.z == out_atom.z
                break  # Only need to check one


class TestCLI:
    """Tests for CLI interface."""

    def test_cli_basic_run(
        self, ref_pdb_path, input_pdb_path, tmp_output_dir, catres_subset_indices
    ):
        """Test basic CLI invocation."""
        output_pdb = tmp_output_dir / "cli_output.pdb"

        catres_str = ",".join(str(i) for i in catres_subset_indices)

        exit_code = cli_main([
            "--input_pdb", str(input_pdb_path),
            "--ref_pdb", str(ref_pdb_path),
            "--catres_subset", catres_str,
            "--output_pdb", str(output_pdb),
        ])

        assert exit_code == 0
        assert output_pdb.exists()

    def test_cli_with_json_output(
        self, ref_pdb_path, input_pdb_path, tmp_output_dir, catres_subset_indices
    ):
        """Test CLI with JSON output."""
        output_pdb = tmp_output_dir / "cli_json_output.pdb"
        output_json = tmp_output_dir / "cli_registry.json"

        catres_str = ",".join(str(i) for i in catres_subset_indices)

        exit_code = cli_main([
            "--input_pdb", str(input_pdb_path),
            "--ref_pdb", str(ref_pdb_path),
            "--catres_subset", catres_str,
            "--output_pdb", str(output_pdb),
            "--output_json", str(output_json),
        ])

        assert exit_code == 0
        assert output_pdb.exists()
        assert output_json.exists()

    def test_cli_dry_run(
        self, ref_pdb_path, input_pdb_path, tmp_output_dir, catres_subset_indices
    ):
        """Test CLI dry run mode."""
        output_pdb = tmp_output_dir / "cli_dry_output.pdb"

        catres_str = ",".join(str(i) for i in catres_subset_indices)

        exit_code = cli_main([
            "--input_pdb", str(input_pdb_path),
            "--ref_pdb", str(ref_pdb_path),
            "--catres_subset", catres_str,
            "--output_pdb", str(output_pdb),
            "--dry_run",
        ])

        assert exit_code == 0
        # In dry run, file should NOT be written
        assert not output_pdb.exists()

    def test_cli_invalid_input(self, tmp_output_dir):
        """Test CLI with invalid input file."""
        exit_code = cli_main([
            "--input_pdb", "/nonexistent/path.pdb",
            "--ref_pdb", "/also/nonexistent.pdb",
            "--output_pdb", str(tmp_output_dir / "output.pdb"),
        ])

        assert exit_code != 0
