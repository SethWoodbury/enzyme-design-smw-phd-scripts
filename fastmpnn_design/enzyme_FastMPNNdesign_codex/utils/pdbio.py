"""Lightweight PDB IO utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)


def _infer_element(atom_name: str) -> str:
    stripped = atom_name.strip()
    if not stripped:
        return ""
    for char in stripped:
        if char.isalpha():
            return char.upper()
    return ""


@dataclass
class AtomRecord:
    record_name: str
    serial: int
    name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    occupancy: float
    tempfactor: float
    element: str
    charge: str
    line_index: int

    def coord(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def set_coord(self, xyz: Iterable[float]) -> None:
        self.x, self.y, self.z = xyz

    def to_pdb_line(self) -> str:
        return (
            f"{self.record_name:<6}{self.serial:>5d} {self.name:<4}{self.altloc:1}"
            f"{self.resname:>3} {self.chain:1}{self.resseq:>4d}{self.icode:1}   "
            f"{self.x:>8.3f}{self.y:>8.3f}{self.z:>8.3f}"
            f"{self.occupancy:>6.2f}{self.tempfactor:>6.2f}          "
            f"{self.element:>2}{self.charge:>2}"
        )


class PdbStructure:
    def __init__(self, lines: List[str], atoms: List[AtomRecord]) -> None:
        self.lines = lines
        self.atoms = atoms
        self._residue_index: Dict[Tuple[str, int, str], List[AtomRecord]] = {}
        for atom in atoms:
            key = (atom.chain, atom.resseq, atom.icode)
            self._residue_index.setdefault(key, []).append(atom)

    @classmethod
    def from_file(cls, path: str) -> "PdbStructure":
        """Load a PDB structure from a file.

        Args:
            path: Path to the PDB file

        Returns:
            PdbStructure instance
        """
        LOGGER.debug("Loading PDB from: %s", path)

        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()

        LOGGER.debug("Read %d lines from PDB file", len(lines))

        atoms: List[AtomRecord] = []
        hetatm_count = 0
        atom_count = 0

        for idx, line in enumerate(lines):
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            record_name = line[0:6].strip()
            serial = int(line[6:11])
            name = line[12:16].strip()
            altloc = line[16:17].strip()
            resname = line[17:20].strip()
            chain = line[21:22].strip()
            resseq = int(line[22:26])
            icode = line[26:27].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            occupancy = float(line[54:60] or 0.0)
            tempfactor = float(line[60:66] or 0.0)
            element = line[76:78].strip() or _infer_element(name)
            charge = line[78:80].strip()

            if record_name == "HETATM":
                hetatm_count += 1
            else:
                atom_count += 1

            atoms.append(
                AtomRecord(
                    record_name=record_name,
                    serial=serial,
                    name=name,
                    altloc=altloc,
                    resname=resname,
                    chain=chain,
                    resseq=resseq,
                    icode=icode,
                    x=x,
                    y=y,
                    z=z,
                    occupancy=occupancy,
                    tempfactor=tempfactor,
                    element=element,
                    charge=charge,
                    line_index=idx,
                )
            )

        LOGGER.debug("Parsed %d ATOM records and %d HETATM records", atom_count, hetatm_count)

        return cls(lines, atoms)

    def residue_atoms(self, chain: str, resseq: int, icode: str = "") -> List[AtomRecord]:
        return list(self._residue_index.get((chain, resseq, icode), []))

    def residues(self) -> Dict[Tuple[str, int, str], List[AtomRecord]]:
        return dict(self._residue_index)

    def write(self, path: str) -> None:
        """Write the PDB structure to a file.

        Args:
            path: Path to write the PDB file
        """
        LOGGER.debug("Writing PDB to: %s", path)

        lines = list(self.lines)
        modified_count = 0
        for atom in self.atoms:
            new_line = atom.to_pdb_line()
            if lines[atom.line_index] != new_line:
                modified_count += 1
            lines[atom.line_index] = new_line

        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

        LOGGER.debug("Wrote %d lines (%d atom lines modified)", len(lines), modified_count)


def parse_remark_666(lines: Iterable[str]) -> List[Dict[str, str]]:
    """Parse REMARK 666 lines from a PDB file.

    REMARK 666 lines contain motif information for enzyme design, including
    the template ligand and catalytic residue positions.

    Args:
        lines: Lines from a PDB file

    Returns:
        List of dictionaries containing parsed REMARK 666 data
    """
    entries: List[Dict[str, str]] = []
    remark_count = 0

    for line in lines:
        if not line.startswith("REMARK 666"):
            continue
        remark_count += 1
        tokens = line.split()
        if "MOTIF" not in tokens or "MATCH" not in tokens:
            LOGGER.debug("Skipping REMARK 666 line (missing MOTIF/MATCH): %s", line[:60])
            continue
        motif_idx = tokens.index("MOTIF")
        try:
            motif_chain = tokens[motif_idx + 1]
            motif_resname = tokens[motif_idx + 2]
            motif_resno = int(tokens[motif_idx + 3])
            block_index = int(tokens[-2])
            block_variant = int(tokens[-1])
        except (IndexError, ValueError) as e:
            LOGGER.debug("Failed to parse REMARK 666 line: %s (%s)", line[:60], e)
            continue

        entry = {
            "line": line.rstrip(),
            "motif_chain": motif_chain,
            "motif_resname": motif_resname,
            "motif_resno": motif_resno,
            "block_index": block_index,
            "block_variant": block_variant,
        }

        if "TEMPLATE" in tokens:
            template_idx = tokens.index("TEMPLATE")
            try:
                entry["template_chain"] = tokens[template_idx + 1]
                entry["template_resname"] = tokens[template_idx + 2]
                entry["template_resno"] = int(tokens[template_idx + 3])
                LOGGER.debug(
                    "Found template: %s %s %s",
                    entry["template_chain"],
                    entry["template_resname"],
                    entry["template_resno"],
                )
            except (IndexError, ValueError):
                pass

        entries.append(entry)

    LOGGER.debug("Parsed %d REMARK 666 entries from %d REMARK 666 lines", len(entries), remark_count)
    return entries


def find_template_id(entries: List[Dict[str, str]]) -> Optional[Tuple[str, str, int]]:
    for entry in entries:
        chain = entry.get("template_chain")
        resname = entry.get("template_resname")
        resno = entry.get("template_resno")
        if chain and resname and resno is not None:
            return (chain, resname, resno)
    return None
