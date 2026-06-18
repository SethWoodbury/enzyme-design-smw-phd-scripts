"""Tests for constraint generation."""

import pytest
from pathlib import Path
import tempfile
import json

from fastmpnndesign.constraints import (
    CoordinateConstraint,
    DistanceConstraint,
    ConstraintSet,
    generate_ligand_coordinate_constraints,
    generate_distance_constraints_from_contacts,
)
from fastmpnndesign.ligand import Ligand, LigandAtom
from fastmpnndesign.config import Contact


class TestCoordinateConstraint:
    """Tests for CoordinateConstraint class."""

    def test_to_rosetta_cst(self):
        """Test Rosetta constraint file format output."""
        cst = CoordinateConstraint(
            chain="A",
            resnum=150,
            resname="PHE",
            atom_name="CA",
            x=10.0,
            y=20.0,
            z=30.0,
            stdev=0.01,
            constraint_type="catres_primary",
            source="self"
        )

        line = cst.to_rosetta_cst(anchor_chain="A", anchor_resnum=1)

        assert "CoordinateConstraint" in line
        assert "CA" in line
        assert "150A" in line
        assert "10.000" in line
        assert "20.000" in line
        assert "30.000" in line
        assert "HARMONIC" in line
        assert "0.0100" in line

    def test_to_dict(self):
        """Test dictionary conversion."""
        cst = CoordinateConstraint(
            chain="A",
            resnum=150,
            resname="PHE",
            atom_name="CA",
            x=10.0,
            y=20.0,
            z=30.0,
            stdev=0.01,
            constraint_type="ligand",
            source="self"
        )

        d = cst.to_dict()

        assert d['chain'] == "A"
        assert d['resnum'] == 150
        assert d['atom_name'] == "CA"
        assert d['x'] == 10.0
        assert d['stdev'] == 0.01


class TestDistanceConstraint:
    """Tests for DistanceConstraint class."""

    def test_to_rosetta_cst(self):
        """Test Rosetta constraint file format output."""
        cst = DistanceConstraint(
            chain1="B",
            resnum1=1,
            resname1="LIG",
            atom1="O1",
            chain2="A",
            resnum2=50,
            resname2="HIS",
            atom2="NE2",
            distance=2.8,
            stdev=0.01,
            constraint_type="primary"
        )

        line = cst.to_rosetta_cst()

        assert "AtomPair" in line
        assert "O1" in line
        assert "1B" in line
        assert "NE2" in line
        assert "50A" in line
        assert "HARMONIC" in line
        assert "2.800" in line
        assert "0.0100" in line


class TestConstraintSet:
    """Tests for ConstraintSet class."""

    def test_to_rosetta_cst_file(self):
        """Test writing constraints to file."""
        coord_cst = CoordinateConstraint(
            chain="A", resnum=150, resname="PHE", atom_name="CA",
            x=10.0, y=20.0, z=30.0, stdev=0.01,
            constraint_type="catres_primary", source="self"
        )
        dist_cst = DistanceConstraint(
            chain1="B", resnum1=1, resname1="LIG", atom1="O1",
            chain2="A", resnum2=50, resname2="HIS", atom2="NE2",
            distance=2.8, stdev=0.01, constraint_type="primary"
        )

        cst_set = ConstraintSet(
            coordinate_constraints=[coord_cst],
            distance_constraints=[dist_cst],
            source_pdb="/path/to/input.pdb"
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.cst', delete=False) as f:
            tmp_path = Path(f.name)

        try:
            cst_set.to_rosetta_cst_file(tmp_path)

            with open(tmp_path, 'r') as f:
                content = f.read()

            assert "CoordinateConstraint" in content
            assert "AtomPair" in content
            assert "input.pdb" in content
        finally:
            tmp_path.unlink()

    def test_to_summary_json(self):
        """Test writing constraint summary to JSON."""
        coord_cst = CoordinateConstraint(
            chain="A", resnum=150, resname="PHE", atom_name="CA",
            x=10.0, y=20.0, z=30.0, stdev=0.01,
            constraint_type="ligand", source="self"
        )

        cst_set = ConstraintSet(
            coordinate_constraints=[coord_cst],
            distance_constraints=[],
            source_pdb="/path/to/input.pdb"
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            tmp_path = Path(f.name)

        try:
            cst_set.to_summary_json(tmp_path)

            with open(tmp_path, 'r') as f:
                data = json.load(f)

            assert data['source_pdb'] == "/path/to/input.pdb"
            assert data['n_coordinate_constraints'] == 1
            assert len(data['coordinate_constraints']) == 1
            assert data['summary']['coordinate_by_type']['ligand'] == 1
        finally:
            tmp_path.unlink()


class TestGenerateLigandConstraints:
    """Tests for generate_ligand_coordinate_constraints."""

    def test_generate_for_ligand(self):
        """Test constraint generation for ligand atoms."""
        ligands = [Ligand(
            resname="LIG",
            chain="B",
            resnum=1,
            atoms=[
                LigandAtom(name="C1", element="C", x=1.0, y=2.0, z=3.0,
                          serial=1, resname="LIG", chain="B", resnum=1),
                LigandAtom(name="O1", element="O", x=4.0, y=5.0, z=6.0,
                          serial=2, resname="LIG", chain="B", resnum=1),
            ]
        )]

        constraints = generate_ligand_coordinate_constraints(
            ligands=ligands, metals=[], stdev=0.01, source="self"
        )

        assert len(constraints) == 2
        assert constraints[0].atom_name == "C1"
        assert constraints[0].x == 1.0
        assert constraints[1].atom_name == "O1"
        assert constraints[1].x == 4.0

    def test_generate_for_metal(self):
        """Test constraint generation for metal ions."""
        metals = [Ligand(
            resname="ZN",
            chain="B",
            resnum=1,
            atoms=[
                LigandAtom(name="ZN", element="ZN", x=10.0, y=20.0, z=30.0,
                          serial=1, resname="ZN", chain="B", resnum=1),
            ]
        )]

        constraints = generate_ligand_coordinate_constraints(
            ligands=[], metals=metals, stdev=0.01, source="self"
        )

        assert len(constraints) == 1
        assert constraints[0].atom_name == "ZN"
        assert constraints[0].constraint_type == "metal"
        assert constraints[0].stdev == 0.005  # Metal stdev should be tighter


class TestGenerateDistanceConstraints:
    """Tests for generate_distance_constraints_from_contacts."""

    def test_generate_from_contacts(self):
        """Test distance constraint generation from contacts."""
        contacts = [
            Contact(
                ligand_chain="B", ligand_resnum=1, ligand_resname="LIG",
                ligand_atom="O1", protein_chain="A", protein_resnum=50,
                protein_resname="HIS", protein_atom="NE2", distance=2.5,
                contact_type="metal", priority=100, is_heteroatom_contact=True
            ),
            Contact(
                ligand_chain="B", ligand_resnum=1, ligand_resname="LIG",
                ligand_atom="C1", protein_chain="A", protein_resnum=60,
                protein_resname="ALA", protein_atom="CB", distance=3.5,
                contact_type="primary", priority=30, is_heteroatom_contact=False
            ),
        ]

        constraints = generate_distance_constraints_from_contacts(contacts)

        assert len(constraints) == 2

        # Check metal constraint has tight stdev
        metal_cst = [c for c in constraints if c.constraint_type == "metal"][0]
        assert metal_cst.stdev == 0.005

        # Check primary constraint
        primary_cst = [c for c in constraints if c.constraint_type == "primary"][0]
        assert primary_cst.stdev == 0.01
        assert primary_cst.distance == 3.5
