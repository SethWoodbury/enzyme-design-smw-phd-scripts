"""
Tests for coordinate transformation module.
"""

import pytest
from pathlib import Path
import numpy as np

from remastered_fastmpnn.core.pdb_io import PDBParser, PDBWriter
from remastered_fastmpnn.core.coordinate_utils import (
    kabsch_align,
    calculate_rmsd,
    align_by_ligand,
)
from remastered_fastmpnn.step1_coordinate_transform.coordinate_transformer import (
    CoordinateTransformer,
)


class TestKabschAlignment:
    """Tests for Kabsch alignment algorithm."""

    def test_identical_coords_zero_rmsd(self):
        """Test that identical coordinates give RMSD of 0."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        R, t, rmsd = kabsch_align(coords, coords)

        assert rmsd < 1e-10
        # Rotation should be identity
        assert np.allclose(R, np.eye(3), atol=1e-10)

    def test_translated_coords(self):
        """Test alignment of translated coordinates."""
        coords1 = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        coords2 = coords1 + np.array([5.0, 5.0, 5.0])

        R, t, rmsd = kabsch_align(coords1, coords2)

        assert rmsd < 1e-10

    def test_rotated_coords(self):
        """Test alignment of rotated coordinates."""
        coords1 = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])

        # 90-degree rotation around Z
        R_true = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=float)
        coords2 = (R_true @ coords1.T).T

        R, t, rmsd = kabsch_align(coords1, coords2)

        assert rmsd < 1e-10


class TestLigandAlignment:
    """Tests for ligand-based alignment."""

    def test_align_by_ligand(self, ref_pdb_path, input_pdb_path):
        """Test that ligand alignment works on real PDBs."""
        ref_parser = PDBParser(ref_pdb_path)
        input_parser = PDBParser(input_pdb_path)

        # Align reference to input
        aligned_ref, rmsd, R, t = align_by_ligand(ref_parser, input_parser)

        # Ligand RMSD should be very small (geometries should match)
        assert rmsd < 0.5, f"Ligand RMSD too high: {rmsd}"

    def test_alignment_preserves_internal_geometry(self, ref_pdb_path, input_pdb_path):
        """Test that alignment preserves internal distances."""
        import copy

        ref_parser = PDBParser(ref_pdb_path)
        input_parser = PDBParser(input_pdb_path)

        # Calculate internal distance before alignment
        ref_atoms = ref_parser.get_residue_atoms("A", 13)
        if len(ref_atoms) >= 2:
            dist_before = np.sqrt(
                (ref_atoms[0].x - ref_atoms[1].x)**2 +
                (ref_atoms[0].y - ref_atoms[1].y)**2 +
                (ref_atoms[0].z - ref_atoms[1].z)**2
            )

            # Align
            ref_copy = copy.deepcopy(ref_parser)
            aligned_ref, rmsd, R, t = align_by_ligand(ref_copy, input_parser)

            # Calculate internal distance after alignment
            aligned_atoms = aligned_ref.get_residue_atoms("A", 13)
            dist_after = np.sqrt(
                (aligned_atoms[0].x - aligned_atoms[1].x)**2 +
                (aligned_atoms[0].y - aligned_atoms[1].y)**2 +
                (aligned_atoms[0].z - aligned_atoms[1].z)**2
            )

            # Internal distances should be preserved
            assert abs(dist_before - dist_after) < 1e-6


class TestCoordinateTransformer:
    """Tests for CoordinateTransformer class."""

    def test_full_transformation(self, ref_pdb_path, input_pdb_path, tmp_output_dir, catres_subset_indices):
        """Test full transformation workflow."""
        output_pdb = tmp_output_dir / "transformed.pdb"

        transformer = CoordinateTransformer(
            input_pdb=input_pdb_path,
            ref_pdb=ref_pdb_path,
            catres_subset_indices=catres_subset_indices,
            verbose=False,
        )

        result_pdb, registry = transformer.run(output_pdb)

        # Output file should exist
        assert output_pdb.exists()

        # Registry should have entries
        assert len(registry) > 0

        # Should have catres_subset entries
        catres = registry.get_catres_subset()
        assert len(catres) > 0

    def test_catres_subset_filtering(self, ref_pdb_path, input_pdb_path, tmp_output_dir):
        """Test that catres_subset filtering works correctly."""
        output_pdb = tmp_output_dir / "filtered.pdb"

        # Only use first 5 indices
        transformer = CoordinateTransformer(
            input_pdb=input_pdb_path,
            ref_pdb=ref_pdb_path,
            catres_subset_indices=[1, 2, 3, 4, 5],
            verbose=False,
        )

        result_pdb, registry = transformer.run(output_pdb)

        catres = registry.get_catres_subset()
        conserved = registry.get_conserved_motif()

        # Should have exactly 5 catres_subset (may be fewer due to unique residues)
        assert len(catres) <= 5
        # Should have some conserved_motif
        assert len(conserved) > 0

    def test_json_output(self, ref_pdb_path, input_pdb_path, tmp_output_dir, catres_subset_indices):
        """Test JSON registry output."""
        output_pdb = tmp_output_dir / "output.pdb"
        output_json = tmp_output_dir / "registry.json"

        transformer = CoordinateTransformer(
            input_pdb=input_pdb_path,
            ref_pdb=ref_pdb_path,
            catres_subset_indices=catres_subset_indices,
            verbose=False,
        )

        result_pdb, registry = transformer.run(output_pdb)
        registry.save_json(output_json)

        # JSON should exist
        assert output_json.exists()

        # Should be loadable
        import json
        with open(output_json) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "residues" in data
        assert len(data["residues"]) > 0
