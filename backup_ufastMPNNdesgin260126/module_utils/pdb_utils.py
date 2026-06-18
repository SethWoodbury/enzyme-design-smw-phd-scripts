"""PDB parsing and manipulation utilities for the upgraded_fastMPNNdesign pipeline.

This module provides functions for:
- Parsing REMARK 666 lines (catalytic residue / ligand constraint definitions)
- Reading and writing PDB files with atom-level access
- Parsing catres_subset arguments from CLI
"""
import logging
from typing import List, Dict, Iterable, Optional, Tuple
import numpy as np

LOGGER = logging.getLogger(__name__)

# Backbone atom names for standard amino acids
BACKBONE_ATOMS = {"N", "CA", "C", "O", "H", "HA", "1H", "2H", "3H", "OXT"}

def parse_remark_666(lines: Iterable[str]) -> List[Dict]:
    """Parse REMARK 666 lines from a PDB file.

    REMARK 666 lines contain motif information for enzyme design, including the template
    (ligand or another residue) and catalytic residue positions. Format:
        REMARK 666 MATCH TEMPLATE <chain> <resname> <resno> MATCH MOTIF <chain> <resname> <resno> <block_idx> <variant>

    Args:
        lines: Iterable of lines from a PDB file

    Returns:
        List of dicts with keys: line, motif_chain, motif_resname, motif_resno,
                                 template_chain, template_resname, template_resno, block_index, block_variant
    """
    entries: List[Dict] = []
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
            motif_chain, motif_resname, motif_resno = tokens[motif_idx + 1], tokens[motif_idx + 2], int(tokens[motif_idx + 3])
            block_index, block_variant = int(tokens[-2]), int(tokens[-1])
        except (IndexError, ValueError) as e:
            LOGGER.debug("Failed to parse REMARK 666 line: %s (%s)", line[:60], e)
            continue
        entry = {"line": line.rstrip(), "motif_chain": motif_chain, "motif_resname": motif_resname,
                 "motif_resno": motif_resno, "block_index": block_index, "block_variant": block_variant}
        if "TEMPLATE" in tokens:
            template_idx = tokens.index("TEMPLATE")
            try:
                entry["template_chain"], entry["template_resname"] = tokens[template_idx + 1], tokens[template_idx + 2]
                entry["template_resno"] = int(tokens[template_idx + 3])
            except (IndexError, ValueError):
                pass
        entries.append(entry)
    LOGGER.debug("Parsed %d REMARK 666 entries from %d REMARK 666 lines", len(entries), remark_count)
    return entries


def parse_catres_subset(subset_str: Optional[str], max_block: int) -> List[int]:
    """Parse catres_subset argument string into list of 1-indexed block indices.

    Args:
        subset_str: Comma-separated string like "1,3,5" or None for all
        max_block: Maximum block index (inclusive) from REMARK 666 entries

    Returns:
        Sorted list of 1-indexed block indices
    """
    if subset_str is None:
        return list(range(1, max_block + 1))
    indices = sorted(set(int(x.strip()) for x in subset_str.split(",") if x.strip()))
    invalid = [i for i in indices if i < 1 or i > max_block]
    if invalid:
        raise ValueError(f"catres_subset indices {invalid} out of range [1, {max_block}]")
    return indices


# Known 5-character residue names (e.g., HIS tautomers in Rosetta/PyRosetta)
FIVE_CHAR_RESNAMES = {"HIS_D", "HIS_E", "CYS_D"}


def read_pdb_atoms(filepath: str) -> Tuple[List[str], List[Dict]]:
    """Read a PDB file and extract all lines and parsed atom records.

    Handles both standard 3-character residue names (positions 17-19) and
    5-character residue names like HIS_D (positions 16-20, used by PyRosetta).

    Args:
        filepath: Path to PDB file

    Returns:
        Tuple of (all_lines, atoms) where atoms is a list of dicts with keys:
            record_type, atom_serial, atom_name, alt_loc, resname, chain, resno, icode, x, y, z,
            occupancy, bfactor, element, charge, line_idx, raw_line
    """
    all_lines = []
    atoms = []
    with open(filepath, "r") as f:
        for line_idx, line in enumerate(f):
            all_lines.append(line)
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                # Check for 5-character residue name (positions 16-20)
                # These names use the alt_loc position (16) as part of the residue name
                resname_5char = line[16:21].strip()
                if resname_5char in FIVE_CHAR_RESNAMES:
                    resname = resname_5char
                    alt_loc = ""  # alt_loc position is used for residue name
                else:
                    resname = line[17:20].strip()
                    alt_loc = line[16:17].strip()

                atom = {"record_type": line[:6].strip(), "atom_serial": int(line[6:11]), "atom_name": line[12:16].strip(),
                        "alt_loc": alt_loc, "resname": resname, "chain": line[21:22],
                        "resno": int(line[22:26]), "icode": line[26:27].strip(),
                        "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54]),
                        "occupancy": float(line[54:60]) if line[54:60].strip() else 1.0,
                        "bfactor": float(line[60:66]) if line[60:66].strip() else 0.0,
                        "element": line[76:78].strip() if len(line) > 76 else "",
                        "charge": line[78:80].strip() if len(line) > 78 else "",
                        "line_idx": line_idx, "raw_line": line}
                atoms.append(atom)
            except (ValueError, IndexError) as e:
                LOGGER.warning("Failed to parse atom line %d: %s", line_idx, e)
    return all_lines, atoms


def get_residue_atoms(atoms: List[Dict], chain: str, resno: int) -> List[Dict]:
    """Get all atoms for a specific residue."""
    return [a for a in atoms if a["chain"] == chain and a["resno"] == resno]


def get_ligand_atoms(atoms: List[Dict], lig_chain: str, lig_resno: int) -> List[Dict]:
    """Get all HETATM atoms for the ligand."""
    return [a for a in atoms if a["record_type"] == "HETATM" and a["chain"] == lig_chain and a["resno"] == lig_resno]


def atoms_to_coords(atoms: List[Dict]) -> np.ndarray:
    """Convert list of atom dicts to Nx3 coordinate array."""
    return np.array([[a["x"], a["y"], a["z"]] for a in atoms])


def is_backbone_atom(atom_name: str) -> bool:
    """Check if atom name is a backbone atom."""
    return atom_name in BACKBONE_ATOMS or atom_name.startswith("H") and len(atom_name) == 2 and atom_name[1].isdigit()


def format_atom_line(atom: Dict) -> str:
    """Format an atom dict back to PDB ATOM/HETATM line format.

    Handles both standard 3-character residue names and 5-character residue names
    like HIS_D (used by PyRosetta for HIS tautomers).
    """
    record = atom["record_type"].ljust(6)
    serial = str(atom["atom_serial"]).rjust(5)
    name = atom["atom_name"].center(4) if len(atom["atom_name"]) < 4 else atom["atom_name"]
    resname = atom["resname"]
    chain = atom["chain"]
    resno = str(atom["resno"]).rjust(4)
    icode = atom.get("icode", " ") or " "
    x, y, z = atom["x"], atom["y"], atom["z"]
    occ = atom.get("occupancy", 1.0)
    bf = atom.get("bfactor", 0.0)
    elem = atom.get("element", "").rjust(2)
    charge = atom.get("charge", "").rjust(2)

    # Handle 5-character residue names (e.g., HIS_D) by using alt_loc position
    if len(resname) > 3:
        # 5-char resname uses positions 16-20, no alt_loc space
        resname_field = resname.ljust(5)  # Pad to 5 chars
        return f"{record}{serial} {name}{resname_field}{chain}{resno}{icode}   {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bf:6.2f}          {elem}{charge}\n"
    else:
        # Standard 3-char resname with alt_loc space
        alt = atom.get("alt_loc", " ") or " "
        resname_field = resname.rjust(3)
        return f"{record}{serial} {name}{alt}{resname_field} {chain}{resno}{icode}   {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bf:6.2f}          {elem}{charge}\n"
