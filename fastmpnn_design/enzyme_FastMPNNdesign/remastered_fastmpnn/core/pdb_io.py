"""
Pure Python PDB file parsing and writing utilities.

Provides PDBParser for reading and PDBWriter for writing PDB files
without external dependencies (no PyRosetta required).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator, Set
from pathlib import Path
import re

from remastered_fastmpnn.constants import (
    BACKBONE_ATOMS,
    PROTEIN_RESIDUES,
    SOLVENTS,
    BUFFERS,
)
from remastered_fastmpnn.logging_config import get_logger

logger = get_logger("pdb_io")


@dataclass
class AtomRecord:
    """
    Represents a single ATOM or HETATM record from a PDB file.

    PDB format columns (1-indexed):
        1-6:   Record type (ATOM/HETATM)
        7-11:  Atom serial number
        13-16: Atom name
        17:    Alternate location indicator
        18-20: Residue name
        22:    Chain identifier
        23-26: Residue sequence number
        27:    Insertion code
        31-38: X coordinate
        39-46: Y coordinate
        47-54: Z coordinate
        55-60: Occupancy
        61-66: Temperature factor
        77-78: Element symbol
        79-80: Charge
    """
    record_type: str  # "ATOM" or "HETATM"
    serial: int
    name: str  # Atom name (e.g., "CA", "ND1")
    altloc: str
    resname: str  # Residue name (e.g., "HIS", "XDW")
    chain: str
    resnum: int
    icode: str  # Insertion code
    x: float
    y: float
    z: float
    occupancy: float
    tempfactor: float
    element: str
    charge: str

    # Original line (for preserving formatting)
    original_line: str = ""

    @property
    def coords(self) -> Tuple[float, float, float]:
        """Return coordinates as tuple."""
        return (self.x, self.y, self.z)

    @coords.setter
    def coords(self, value: Tuple[float, float, float]) -> None:
        """Set coordinates from tuple."""
        self.x, self.y, self.z = value

    @property
    def identifier(self) -> str:
        """Return residue identifier (chain + resnum)."""
        return f"{self.chain}{self.resnum}"

    def is_backbone(self) -> bool:
        """Check if this is a backbone atom."""
        return self.name.strip() in BACKBONE_ATOMS

    def is_sidechain(self) -> bool:
        """Check if this is a sidechain atom (not backbone, not H on backbone)."""
        name = self.name.strip()
        if name in BACKBONE_ATOMS:
            return False
        # Also check for backbone hydrogens
        if name in {'H', 'HA', 'HA2', 'HA3', 'HN'}:
            return False
        return True

    def is_hydrogen(self) -> bool:
        """Check if this is a hydrogen atom."""
        return self.element.strip().upper() == 'H'

    def is_protein(self) -> bool:
        """Check if this atom belongs to a protein residue."""
        return self.resname in PROTEIN_RESIDUES

    def is_ligand(self) -> bool:
        """Check if this atom belongs to a ligand."""
        return (
            self.record_type == "HETATM" and
            self.resname not in SOLVENTS and
            self.resname not in BUFFERS and
            self.resname not in PROTEIN_RESIDUES
        )

    def is_metal(self) -> bool:
        """Check if this is a metal ion."""
        elem = self.element.strip().upper()
        return elem in {'ZN', 'FE', 'MG', 'CA', 'MN', 'CO', 'NI', 'CU', 'MO', 'W'}

    def to_pdb_line(self) -> str:
        """Convert atom record back to PDB format line."""
        # Standard PDB format
        return (
            f"{self.record_type:<6}"
            f"{self.serial:>5} "
            f"{self.name:<4}"
            f"{self.altloc:1}"
            f"{self.resname:>3} "
            f"{self.chain:1}"
            f"{self.resnum:>4}"
            f"{self.icode:1}   "
            f"{self.x:>8.3f}"
            f"{self.y:>8.3f}"
            f"{self.z:>8.3f}"
            f"{self.occupancy:>6.2f}"
            f"{self.tempfactor:>6.2f}          "
            f"{self.element:>2}"
            f"{self.charge:>2}"
        )

    @classmethod
    def from_pdb_line(cls, line: str) -> Optional["AtomRecord"]:
        """
        Parse an ATOM or HETATM line from a PDB file.

        Args:
            line: Raw PDB line

        Returns:
            AtomRecord or None if parsing fails
        """
        if not line.startswith(("ATOM", "HETATM")):
            return None

        try:
            # Pad line to ensure we can access all columns
            line = line.ljust(80)

            record_type = line[0:6].strip()
            serial = int(line[6:11].strip() or 0)
            name = line[12:16]  # Keep spacing for atom names
            altloc = line[16:17]
            resname = line[17:20].strip()
            chain = line[21:22]
            resnum = int(line[22:26].strip() or 0)
            icode = line[26:27]
            x = float(line[30:38].strip() or 0.0)
            y = float(line[38:46].strip() or 0.0)
            z = float(line[46:54].strip() or 0.0)
            occupancy = float(line[54:60].strip() or 1.0)
            tempfactor = float(line[60:66].strip() or 0.0)
            element = line[76:78].strip() if len(line) > 76 else ""
            charge = line[78:80].strip() if len(line) > 78 else ""

            # Infer element from atom name if not provided
            if not element:
                element = cls._infer_element(name)

            return cls(
                record_type=record_type,
                serial=serial,
                name=name,
                altloc=altloc,
                resname=resname,
                chain=chain,
                resnum=resnum,
                icode=icode,
                x=x,
                y=y,
                z=z,
                occupancy=occupancy,
                tempfactor=tempfactor,
                element=element,
                charge=charge,
                original_line=line.rstrip(),
            )
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse PDB line: {line.rstrip()}: {e}")
            return None

    @staticmethod
    def _infer_element(name: str) -> str:
        """Infer element symbol from atom name."""
        name = name.strip()
        if not name:
            return ""

        # Standard inference rules
        if name.startswith(("CA", "CB", "CG", "CD", "CE", "CZ", "CH")):
            return "C"
        if name.startswith(("N", "ND", "NE", "NH", "NZ")):
            return "N"
        if name.startswith(("O", "OD", "OE", "OG", "OH", "OXT")):
            return "O"
        if name.startswith(("S", "SD", "SG")):
            return "S"
        if name.startswith("H"):
            return "H"
        if name.startswith("ZN"):
            return "ZN"
        if name.startswith("FE"):
            return "FE"
        if name.startswith("P"):
            return "P"

        # Default: first letter
        return name[0].upper()


@dataclass
class PDBParser:
    """
    Pure Python PDB file parser.

    Parses ATOM, HETATM, REMARK, and other records from PDB files.
    """
    path: Path
    atoms: List[AtomRecord] = field(default_factory=list)
    remarks: List[str] = field(default_factory=list)
    remark666_lines: List[str] = field(default_factory=list)
    other_lines: List[str] = field(default_factory=list)

    # Indexing structures (built on demand)
    _residue_index: Optional[Dict[str, List[AtomRecord]]] = None
    _chain_index: Optional[Dict[str, List[AtomRecord]]] = None

    def __post_init__(self):
        """Parse the PDB file after initialization."""
        self.path = Path(self.path)
        if self.path.exists():
            self._parse()
        else:
            logger.warning(f"PDB file not found: {self.path}")

    def _parse(self) -> None:
        """Parse the PDB file."""
        logger.debug(f"Parsing PDB: {self.path}")

        with open(self.path, 'r') as f:
            for line in f:
                line = line.rstrip('\n\r')

                if line.startswith(("ATOM", "HETATM")):
                    atom = AtomRecord.from_pdb_line(line)
                    if atom:
                        self.atoms.append(atom)
                elif line.startswith("REMARK 666"):
                    self.remark666_lines.append(line)
                    self.remarks.append(line)
                elif line.startswith("REMARK"):
                    self.remarks.append(line)
                else:
                    self.other_lines.append(line)

        logger.debug(
            f"Parsed {len(self.atoms)} atoms, "
            f"{len(self.remark666_lines)} REMARK 666 lines"
        )

    def _build_indices(self) -> None:
        """Build residue and chain indices for fast lookup."""
        if self._residue_index is not None:
            return

        self._residue_index = {}
        self._chain_index = {}

        for atom in self.atoms:
            # Residue index
            key = f"{atom.chain}{atom.resnum}"
            if key not in self._residue_index:
                self._residue_index[key] = []
            self._residue_index[key].append(atom)

            # Chain index
            if atom.chain not in self._chain_index:
                self._chain_index[atom.chain] = []
            self._chain_index[atom.chain].append(atom)

    def get_residue_atoms(self, chain: str, resnum: int) -> List[AtomRecord]:
        """
        Get all atoms for a specific residue.

        Args:
            chain: Chain identifier
            resnum: Residue number

        Returns:
            List of AtomRecord for the residue
        """
        self._build_indices()
        key = f"{chain}{resnum}"
        return self._residue_index.get(key, [])

    def get_chain_atoms(self, chain: str) -> List[AtomRecord]:
        """Get all atoms in a specific chain."""
        self._build_indices()
        return self._chain_index.get(chain, [])

    def get_ligand_atoms(self) -> List[AtomRecord]:
        """Get all ligand atoms (HETATM that are not solvent/buffer)."""
        return [a for a in self.atoms if a.is_ligand()]

    def get_metal_atoms(self) -> List[AtomRecord]:
        """Get all metal atoms."""
        return [a for a in self.atoms if a.is_metal()]

    def get_protein_atoms(self) -> List[AtomRecord]:
        """Get all protein atoms."""
        return [a for a in self.atoms if a.is_protein()]

    def get_backbone_atoms(self, chain: Optional[str] = None) -> List[AtomRecord]:
        """Get all backbone atoms, optionally filtered by chain."""
        atoms = self.atoms if chain is None else self.get_chain_atoms(chain)
        return [a for a in atoms if a.is_backbone()]

    def get_sidechain_atoms(
        self, chain: str, resnum: int
    ) -> List[AtomRecord]:
        """Get sidechain atoms for a specific residue."""
        return [a for a in self.get_residue_atoms(chain, resnum) if a.is_sidechain()]

    def get_chains(self) -> Set[str]:
        """Get all chain identifiers in the PDB."""
        return {a.chain for a in self.atoms}

    def get_residue_identifiers(self) -> Set[str]:
        """Get all residue identifiers (chain + resnum)."""
        return {a.identifier for a in self.atoms}

    def get_residue_info(
        self, chain: str, resnum: int
    ) -> Optional[Tuple[str, str]]:
        """
        Get residue name and identifier for a position.

        Returns:
            Tuple of (resname, identifier) or None
        """
        atoms = self.get_residue_atoms(chain, resnum)
        if atoms:
            return (atoms[0].resname, atoms[0].identifier)
        return None

    def get_ligand_name(self) -> Optional[str]:
        """Get the name of the first ligand found."""
        ligand_atoms = self.get_ligand_atoms()
        if ligand_atoms:
            # Return the most common resname among ligand atoms
            names = [a.resname for a in ligand_atoms]
            return max(set(names), key=names.count)
        return None

    def get_remark666_lines(self) -> List[str]:
        """Get all REMARK 666 lines."""
        return self.remark666_lines

    def iter_residues(self) -> Iterator[Tuple[str, int, str, List[AtomRecord]]]:
        """
        Iterate over residues in the PDB.

        Yields:
            Tuples of (chain, resnum, resname, atoms)
        """
        self._build_indices()
        seen = set()

        for atom in self.atoms:
            key = (atom.chain, atom.resnum)
            if key not in seen:
                seen.add(key)
                atoms = self.get_residue_atoms(atom.chain, atom.resnum)
                if atoms:
                    yield (atom.chain, atom.resnum, atoms[0].resname, atoms)

    def copy_coordinates_from(
        self,
        source: "PDBParser",
        chain: str,
        resnum: int,
        backbone_only: bool = False,
        sidechain_only: bool = False,
    ) -> int:
        """
        Copy coordinates for a residue from another PDB.

        Args:
            source: Source PDBParser to copy from
            chain: Chain identifier
            resnum: Residue number
            backbone_only: If True, only copy backbone atoms
            sidechain_only: If True, only copy sidechain atoms

        Returns:
            Number of atoms updated
        """
        source_atoms = source.get_residue_atoms(chain, resnum)
        if not source_atoms:
            logger.warning(f"No atoms found in source for {chain}{resnum}")
            return 0

        # Build lookup by atom name
        source_coords = {a.name.strip(): (a.x, a.y, a.z) for a in source_atoms}

        updated = 0
        for atom in self.get_residue_atoms(chain, resnum):
            name = atom.name.strip()

            # Filter by backbone/sidechain if requested
            if backbone_only and not atom.is_backbone():
                continue
            if sidechain_only and not atom.is_sidechain():
                continue

            if name in source_coords:
                atom.x, atom.y, atom.z = source_coords[name]
                updated += 1

        return updated


class PDBWriter:
    """
    PDB file writer.

    Writes PDBParser contents to a new PDB file with optional modifications.
    """

    @staticmethod
    def write(
        parser: PDBParser,
        output_path: Path,
        preserve_remarks: bool = True,
        preserve_other: bool = True,
        renumber_atoms: bool = False,
    ) -> None:
        """
        Write PDB contents to file.

        Args:
            parser: PDBParser with data to write
            output_path: Output file path
            preserve_remarks: Include REMARK lines
            preserve_other: Include other (non-ATOM/HETATM/REMARK) lines
            renumber_atoms: Renumber atoms sequentially starting from 1
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            # Write remarks first
            if preserve_remarks:
                for remark in parser.remarks:
                    f.write(remark + "\n")

            # Write other header lines
            if preserve_other:
                for line in parser.other_lines:
                    # Skip END/ENDMDL - we'll add them at the end
                    if not line.startswith(("END", "TER")):
                        f.write(line + "\n")

            # Write atoms
            current_chain = None
            serial = 0

            for atom in parser.atoms:
                # Add TER between chains
                if current_chain is not None and atom.chain != current_chain:
                    f.write("TER\n")
                current_chain = atom.chain

                serial += 1
                if renumber_atoms:
                    atom.serial = serial

                f.write(atom.to_pdb_line() + "\n")

            # Final TER and END
            f.write("TER\n")
            f.write("END\n")

        logger.info(f"Wrote PDB to {output_path}")

    @staticmethod
    def write_minimal(
        atoms: List[AtomRecord],
        output_path: Path,
        remarks: Optional[List[str]] = None,
    ) -> None:
        """
        Write a minimal PDB file from a list of atoms.

        Args:
            atoms: List of AtomRecord to write
            output_path: Output file path
            remarks: Optional REMARK lines to include
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            if remarks:
                for remark in remarks:
                    f.write(remark + "\n")

            for i, atom in enumerate(atoms, 1):
                atom.serial = i
                f.write(atom.to_pdb_line() + "\n")

            f.write("END\n")
