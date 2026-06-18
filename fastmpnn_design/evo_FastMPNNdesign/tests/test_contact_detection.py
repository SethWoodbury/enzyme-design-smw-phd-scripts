"""Tests for contact detection."""

import pytest
from pathlib import Path
import tempfile

from fastmpnndesign.contact_detection import (
    detect_contacts,
    get_protein_atoms,
    get_unique_residues_from_contacts,
    get_best_contact_per_residue,
    summarize_contacts,
    ProteinAtom,
)
from fastmpnndesign.ligand import Ligand, LigandAtom
from fastmpnndesign.config import Contact


class TestProteinAtom:
    """Tests for ProteinAtom class."""

    def test_is_heteroatom(self):
        """Test heteroatom detection."""
        atom_n = ProteinAtom(name="N", element="N", x=0, y=0, z=0,
                            chain="A", resnum=1, resname="ALA")
        atom_o = ProteinAtom(name="O", element="O", x=0, y=0, z=0,
                            chain="A", resnum=1, resname="ALA")
        atom_s = ProteinAtom(name="SG", element="S", x=0, y=0, z=0,
                            chain="A", resnum=1, resname="CYS")
        atom_c = ProteinAtom(name="CA", element="C", x=0, y=0, z=0,
                            chain="A", resnum=1, resname="ALA")

        assert atom_n.is_heteroatom
        assert atom_o.is_heteroatom
        assert atom_s.is_heteroatom
        assert not atom_c.is_heteroatom

    def test_is_carbon(self):
        """Test carbon detection."""
        atom_c = ProteinAtom(name="CA", element="C", x=0, y=0, z=0,
                            chain="A", resnum=1, resname="ALA")
        atom_n = ProteinAtom(name="N", element="N", x=0, y=0, z=0,
                            chain="A", resnum=1, resname="ALA")

        assert atom_c.is_carbon
        assert not atom_n.is_carbon


class TestDetectContacts:
    """Tests for detect_contacts function."""

    def test_metal_contacts(self):
        """Test detection of metal coordination contacts."""
        # Create a zinc ion
        metals = [Ligand(
            resname="ZN",
            chain="B",
            resnum=1,
            atoms=[LigandAtom(
                name="ZN", element="ZN", x=0.0, y=0.0, z=0.0,
                serial=1, resname="ZN", chain="B", resnum=1
            )]
        )]

        # Create protein atoms coordinating the zinc
        protein_atoms = [
            ProteinAtom(name="NE2", element="N", x=2.0, y=0.0, z=0.0,
                       chain="A", resnum=50, resname="HIS"),  # ~2.0 A
            ProteinAtom(name="OE1", element="O", x=0.0, y=2.3, z=0.0,
                       chain="A", resnum=60, resname="GLU"),  # ~2.3 A
            ProteinAtom(name="CA", element="C", x=5.0, y=0.0, z=0.0,
                       chain="A", resnum=70, resname="ALA"),  # 5.0 A - too far
        ]

        contacts = detect_contacts(
            ligands=[],
            metals=metals,
            protein_atoms=protein_atoms,
            metal_cutoff=2.6
        )

        # Should find 2 metal contacts
        metal_contacts = [c for c in contacts if c.contact_type == 'metal']
        assert len(metal_contacts) == 2

        # Check priorities
        for c in metal_contacts:
            assert c.priority == 100  # PRIORITY_METAL

    def test_primary_contacts(self):
        """Test detection of primary contacts (within 3.6 A)."""
        ligands = [Ligand(
            resname="LIG",
            chain="B",
            resnum=1,
            atoms=[
                LigandAtom(name="O1", element="O", x=0.0, y=0.0, z=0.0,
                          serial=1, resname="LIG", chain="B", resnum=1),
            ]
        )]

        protein_atoms = [
            ProteinAtom(name="N", element="N", x=3.0, y=0.0, z=0.0,
                       chain="A", resnum=50, resname="GLY"),  # ~3.0 A
        ]

        contacts = detect_contacts(
            ligands=ligands,
            metals=[],
            protein_atoms=protein_atoms,
            primary_cutoff=3.6,
            secondary_cutoff=4.2
        )

        primary_contacts = [c for c in contacts if c.contact_type == 'primary']
        assert len(primary_contacts) == 1
        assert primary_contacts[0].is_heteroatom_contact

    def test_secondary_contacts(self):
        """Test detection of secondary contacts (3.6-4.2 A)."""
        ligands = [Ligand(
            resname="LIG",
            chain="B",
            resnum=1,
            atoms=[
                LigandAtom(name="C1", element="C", x=0.0, y=0.0, z=0.0,
                          serial=1, resname="LIG", chain="B", resnum=1),
            ]
        )]

        protein_atoms = [
            ProteinAtom(name="CB", element="C", x=4.0, y=0.0, z=0.0,
                       chain="A", resnum=50, resname="ALA"),  # ~4.0 A
        ]

        contacts = detect_contacts(
            ligands=ligands,
            metals=[],
            protein_atoms=protein_atoms,
            primary_cutoff=3.6,
            secondary_cutoff=4.2
        )

        secondary_contacts = [c for c in contacts if c.contact_type == 'secondary']
        assert len(secondary_contacts) == 1

    def test_heteroatom_priority(self):
        """Test that heteroatom contacts have higher priority."""
        ligands = [Ligand(
            resname="LIG",
            chain="B",
            resnum=1,
            atoms=[
                LigandAtom(name="O1", element="O", x=0.0, y=0.0, z=0.0,
                          serial=1, resname="LIG", chain="B", resnum=1),
            ]
        )]

        protein_atoms = [
            ProteinAtom(name="N", element="N", x=3.0, y=0.0, z=0.0,
                       chain="A", resnum=50, resname="GLY"),
            ProteinAtom(name="CA", element="C", x=3.0, y=0.1, z=0.0,
                       chain="A", resnum=51, resname="ALA"),
        ]

        contacts = detect_contacts(
            ligands=ligands,
            metals=[],
            protein_atoms=protein_atoms
        )

        # Heteroatom contact should be first (higher priority)
        assert contacts[0].protein_atom == "N"
        assert contacts[0].priority > contacts[1].priority


class TestContactUtilities:
    """Tests for contact utility functions."""

    def test_get_unique_residues(self):
        """Test extraction of unique residues from contacts."""
        contacts = [
            Contact(
                ligand_chain="B", ligand_resnum=1, ligand_resname="LIG",
                ligand_atom="O1", protein_chain="A", protein_resnum=50,
                protein_resname="HIS", protein_atom="NE2", distance=2.5,
                contact_type="primary", priority=50, is_heteroatom_contact=True
            ),
            Contact(
                ligand_chain="B", ligand_resnum=1, ligand_resname="LIG",
                ligand_atom="O1", protein_chain="A", protein_resnum=50,
                protein_resname="HIS", protein_atom="ND1", distance=2.8,
                contact_type="primary", priority=50, is_heteroatom_contact=True
            ),
            Contact(
                ligand_chain="B", ligand_resnum=1, ligand_resname="LIG",
                ligand_atom="C1", protein_chain="A", protein_resnum=60,
                protein_resname="ALA", protein_atom="CB", distance=3.5,
                contact_type="primary", priority=30, is_heteroatom_contact=False
            ),
        ]

        unique = get_unique_residues_from_contacts(contacts)

        assert len(unique) == 2
        assert ("A", 50, "HIS") in unique
        assert ("A", 60, "ALA") in unique

    def test_get_best_contact_per_residue(self):
        """Test selection of best contact per residue."""
        contacts = [
            Contact(
                ligand_chain="B", ligand_resnum=1, ligand_resname="LIG",
                ligand_atom="O1", protein_chain="A", protein_resnum=50,
                protein_resname="HIS", protein_atom="NE2", distance=2.5,
                contact_type="primary", priority=50, is_heteroatom_contact=True
            ),
            Contact(
                ligand_chain="B", ligand_resnum=1, ligand_resname="LIG",
                ligand_atom="O1", protein_chain="A", protein_resnum=50,
                protein_resname="HIS", protein_atom="ND1", distance=2.8,
                contact_type="primary", priority=50, is_heteroatom_contact=True
            ),
        ]

        best = get_best_contact_per_residue(contacts)

        assert len(best) == 1
        assert best[("A", 50)].protein_atom == "NE2"  # Shorter distance

    def test_summarize_contacts(self):
        """Test contact summary generation."""
        contacts = [
            Contact(
                ligand_chain="B", ligand_resnum=1, ligand_resname="LIG",
                ligand_atom="O1", protein_chain="A", protein_resnum=50,
                protein_resname="HIS", protein_atom="NE2", distance=2.5,
                contact_type="metal", priority=100, is_heteroatom_contact=True
            ),
            Contact(
                ligand_chain="B", ligand_resnum=1, ligand_resname="LIG",
                ligand_atom="O1", protein_chain="A", protein_resnum=60,
                protein_resname="GLU", protein_atom="OE1", distance=3.0,
                contact_type="primary", priority=50, is_heteroatom_contact=True
            ),
        ]

        summary = summarize_contacts(contacts)

        assert summary['total'] == 2
        assert summary['by_type']['metal']['count'] == 1
        assert summary['by_type']['primary']['count'] == 1
        assert summary['unique_residues'] == 2
