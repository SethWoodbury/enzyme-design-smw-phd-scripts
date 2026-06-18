"""
Tests for interaction detection module.
"""

import pytest
from pathlib import Path

from remastered_fastmpnn.core.pdb_io import PDBParser
from remastered_fastmpnn.core.residue_data import ImportantComponent
from remastered_fastmpnn.step1_coordinate_transform.interaction_detection import (
    InteractionDetector,
)


class TestInteractionDetector:
    """Tests for InteractionDetector class."""

    @pytest.fixture
    def detector(self):
        """Create an InteractionDetector instance."""
        return InteractionDetector(verbose=False)

    @pytest.fixture
    def ref_parser(self, ref_pdb_path):
        """Load reference PDB parser."""
        return PDBParser(ref_pdb_path)

    def test_his_metal_coordination_is_sidechain(self, detector, ref_parser):
        """Test that HIS coordinating metal is classified as sidechain."""
        # HIS 13 coordinates ZN in the reference PDB
        ligand_atoms = ref_parser.get_ligand_atoms()
        metal_atoms = ref_parser.get_metal_atoms()
        residue_atoms = ref_parser.get_residue_atoms("A", 13)

        assert len(residue_atoms) > 0, "HIS 13 should exist"
        assert len(metal_atoms) > 0, "Metal atoms should exist"

        component, interactions, bb_count, sc_count = detector.detect_all_interactions(
            residue_atoms=residue_atoms,
            ligand_atoms=ligand_atoms,
            metal_atoms=metal_atoms,
        )

        # HIS coordinating metal should have sidechain interactions
        assert sc_count > 0, "Should have sidechain interactions"

        # Check for metal coordination interaction
        metal_interactions = [
            i for i in interactions.values()
            if i.interaction_type == "metal_coordination"
        ]
        assert len(metal_interactions) > 0, "Should have metal coordination"

        # Metal coordination is always sidechain
        for interaction in metal_interactions:
            assert interaction.from_component == "sidechain"

    def test_aromatic_residue_is_sidechain(self, detector, ref_parser):
        """Test that aromatic residue interactions are classified as sidechain."""
        # TRP 149 or PHE 150 should have aromatic interactions
        ligand_atoms = ref_parser.get_ligand_atoms()
        metal_atoms = ref_parser.get_metal_atoms()

        # Try TRP 149
        residue_atoms = ref_parser.get_residue_atoms("A", 149)
        if residue_atoms:
            component, interactions, bb_count, sc_count = detector.detect_all_interactions(
                residue_atoms=residue_atoms,
                ligand_atoms=ligand_atoms,
                metal_atoms=metal_atoms,
            )

            # Aromatic residues should have sidechain interactions
            if interactions:
                assert component in (
                    ImportantComponent.SIDECHAIN,
                    ImportantComponent.BOTH
                )

    def test_default_to_sidechain_when_no_interactions(self, detector):
        """Test that default is sidechain when no interactions detected."""
        # Empty atoms should return sidechain
        component, interactions, bb_count, sc_count = detector.detect_all_interactions(
            residue_atoms=[],
            ligand_atoms=[],
            metal_atoms=[],
        )

        assert component == ImportantComponent.SIDECHAIN
        assert len(interactions) == 0

    def test_analyze_catres_subset(self, detector, ref_parser):
        """Test analyzing multiple catres_subset residues."""
        # Analyze first 5 catres
        catres_ids = [
            ("A", 13),   # HIS
            ("A", 15),   # HIS
            ("A", 176),  # HIS
            ("A", 203),  # HIS
            ("A", 53),   # ASP
        ]

        results = detector.analyze_catres_subset(ref_parser, catres_ids)

        assert len(results) == 5
        assert "A13" in results
        assert "A15" in results

        # All should have some categorization
        for identifier, info in results.items():
            assert info.important_component in ImportantComponent

    def test_charged_interaction_is_sidechain(self, detector, ref_parser):
        """Test that charged residue interactions are classified as sidechain."""
        # ASP 53 or GLU 14 should have charged interactions
        ligand_atoms = ref_parser.get_ligand_atoms()
        metal_atoms = ref_parser.get_metal_atoms()

        residue_atoms = ref_parser.get_residue_atoms("A", 53)  # ASP
        if residue_atoms:
            component, interactions, bb_count, sc_count = detector.detect_all_interactions(
                residue_atoms=residue_atoms,
                ligand_atoms=ligand_atoms,
                metal_atoms=metal_atoms,
            )

            # Check if any charged interactions were detected
            charged_interactions = [
                i for i in interactions.values()
                if i.interaction_type == "charged_interaction"
            ]

            # If charged interactions exist, they should be sidechain
            for interaction in charged_interactions:
                assert interaction.from_component == "sidechain"


class TestInteractionDetectorEdgeCases:
    """Edge case tests for InteractionDetector."""

    def test_handles_missing_residue(self):
        """Test that missing residues are handled gracefully."""
        detector = InteractionDetector(verbose=False)

        component, interactions, bb_count, sc_count = detector.detect_all_interactions(
            residue_atoms=[],
            ligand_atoms=[],
            metal_atoms=[],
        )

        # Should return default sidechain, no interactions
        assert component == ImportantComponent.SIDECHAIN
        assert len(interactions) == 0
        assert bb_count == 0
        assert sc_count == 0
