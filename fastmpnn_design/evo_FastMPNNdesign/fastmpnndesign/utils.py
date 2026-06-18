"""
Utility functions for fastmpnndesign.

Path handling, I/O utilities, and helper functions.
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterator
import re

from fastmpnndesign.logging_config import get_logger

logger = get_logger("utils")


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist, return the path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_pdb_path(path: Path) -> Path:
    """Validate that a PDB file exists and is readable."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDB file not found: {path}")
    if not path.suffix.lower() == '.pdb':
        logger.warning(f"File does not have .pdb extension: {path}")
    return path


def validate_params_paths(paths: List[Path]) -> List[Path]:
    """Validate that all params files exist."""
    validated = []
    for p in paths:
        p = Path(p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Params file not found: {p}")
        validated.append(p)
    return validated


def read_pdb_lines(pdb_path: Path) -> List[str]:
    """Read all lines from a PDB file."""
    with open(pdb_path, 'r') as f:
        return f.readlines()


def write_pdb_lines(pdb_path: Path, lines: List[str]) -> None:
    """Write lines to a PDB file."""
    with open(pdb_path, 'w') as f:
        f.writelines(lines)


def read_json(path: Path) -> Dict[str, Any]:
    """Read JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any], indent: int = 2) -> None:
    """Write JSON file with pretty printing."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def parse_resid(resid: str) -> Tuple[str, int, str]:
    """
    Parse residue ID string into chain, resnum, insertion code.

    Handles formats: 'A150', '150A', 'A150B' (with insertion code)

    Returns:
        Tuple of (chain, resnum, icode)
    """
    # Pattern: optional leading chain, digits, optional trailing chain/icode
    # Examples: A150, 150A, A150B, 150
    match = re.match(r'^([A-Za-z]?)(\d+)([A-Za-z]?)$', resid.strip())
    if not match:
        raise ValueError(f"Cannot parse residue ID: {resid}")

    prefix, resnum, suffix = match.groups()

    if prefix and suffix:
        # Format like A150B - prefix is chain, suffix is icode
        return prefix.upper(), int(resnum), suffix.upper()
    elif prefix:
        # Format like A150 - prefix is chain
        return prefix.upper(), int(resnum), ""
    elif suffix:
        # Format like 150A - suffix is chain
        return suffix.upper(), int(resnum), ""
    else:
        # Just number like 150
        return "", int(resnum), ""


def format_resid_rosetta(chain: str, resnum: int, icode: str = "") -> str:
    """Format residue ID in Rosetta style: 150A or 150IA (with insertion code)."""
    if icode:
        return f"{resnum}{icode}{chain}"
    return f"{resnum}{chain}"


def format_resid_pdb(chain: str, resnum: int, icode: str = "") -> str:
    """Format residue ID in PDB style: A150 or A150B (with insertion code)."""
    if icode:
        return f"{chain}{resnum}{icode}"
    return f"{chain}{resnum}"


def copy_file(src: Path, dst: Path) -> Path:
    """Copy file and return destination path."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def iter_pdb_atoms(pdb_path: Path) -> Iterator[Dict[str, Any]]:
    """
    Iterate over ATOM/HETATM records in a PDB file.

    Yields dictionaries with parsed atom information.
    """
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                yield parse_atom_line(line)


def parse_atom_line(line: str) -> Dict[str, Any]:
    """
    Parse a PDB ATOM/HETATM line into a dictionary.

    Returns dict with keys: record_type, serial, name, altloc, resname, chain,
                           resnum, icode, x, y, z, occupancy, bfactor, element
    """
    return {
        'record_type': line[0:6].strip(),
        'serial': int(line[6:11].strip()) if line[6:11].strip() else 0,
        'name': line[12:16].strip(),
        'altloc': line[16:17].strip(),
        'resname': line[17:20].strip(),
        'chain': line[21:22].strip(),
        'resnum': int(line[22:26].strip()) if line[22:26].strip() else 0,
        'icode': line[26:27].strip(),
        'x': float(line[30:38].strip()) if line[30:38].strip() else 0.0,
        'y': float(line[38:46].strip()) if line[38:46].strip() else 0.0,
        'z': float(line[46:54].strip()) if line[46:54].strip() else 0.0,
        'occupancy': float(line[54:60].strip()) if line[54:60].strip() else 1.0,
        'bfactor': float(line[60:66].strip()) if line[60:66].strip() else 0.0,
        'element': line[76:78].strip() if len(line) > 76 else "",
        'raw_line': line
    }


def get_element_from_atom_name(atom_name: str) -> str:
    """
    Infer element from PDB atom name.

    Uses common naming conventions when element column is missing.
    """
    name = atom_name.strip()
    if not name:
        return ""

    # Common patterns
    if name[0].isdigit():
        # Names like 1HB, 2HG - hydrogen
        return 'H'

    # First character is usually the element
    first = name[0].upper()

    # Two-letter elements that start with common letters
    if len(name) >= 2:
        two = name[:2].upper()
        if two in ('CL', 'BR', 'FE', 'ZN', 'MG', 'CA', 'MN', 'CO', 'NI', 'CU', 'MO'):
            return two

    return first


def calculate_distance(
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float
) -> float:
    """Calculate Euclidean distance between two 3D points."""
    return ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5


def get_pdb_sequence(pdb_path: Path, chain: Optional[str] = None) -> Dict[str, str]:
    """
    Extract amino acid sequence from PDB file.

    Args:
        pdb_path: Path to PDB file.
        chain: Specific chain to extract. If None, returns all chains.

    Returns:
        Dictionary mapping chain IDs to sequences.
    """
    from fastmpnndesign.constants import AA_3TO1, STANDARD_AA, NONSTANDARD_AA

    sequences: Dict[str, List[Tuple[int, str]]] = {}
    seen_residues: Dict[str, set] = {}

    for atom in iter_pdb_atoms(pdb_path):
        if atom['record_type'] != 'ATOM':
            continue
        if chain and atom['chain'] != chain:
            continue

        resname = atom['resname']
        if resname not in STANDARD_AA and resname not in NONSTANDARD_AA:
            continue

        chain_id = atom['chain'] or '_'
        resnum = atom['resnum']
        key = (chain_id, resnum, atom['icode'])

        if chain_id not in seen_residues:
            seen_residues[chain_id] = set()
            sequences[chain_id] = []

        if key not in seen_residues[chain_id]:
            seen_residues[chain_id].add(key)
            aa_1letter = AA_3TO1.get(resname, 'X')
            sequences[chain_id].append((resnum, aa_1letter))

    # Sort by residue number and convert to string
    result = {}
    for chain_id, residues in sequences.items():
        residues.sort(key=lambda x: x[0])
        result[chain_id] = ''.join(aa for _, aa in residues)

    return result


def extract_ligand_resnames(pdb_path: Path) -> List[str]:
    """
    Extract unique HETATM residue names that could be ligands.

    Filters out common solvents/buffers.
    """
    from fastmpnndesign.constants import SOLVENTS, BUFFERS, METALS

    exclude = SOLVENTS | BUFFERS | METALS
    resnames = set()

    for atom in iter_pdb_atoms(pdb_path):
        if atom['record_type'] == 'HETATM':
            resname = atom['resname']
            if resname not in exclude:
                resnames.add(resname)

    return sorted(resnames)


def get_residues_outside_shell(
    pdb_path: Path,
    ligand_coords: List[Tuple[float, float, float]],
    shell_radius: float,
    exclude_resnames: Optional[List[str]] = None
) -> List[str]:
    """
    Get protein residues outside a shell around ligand atoms.

    A residue is considered INSIDE the shell if ANY of its atoms is within
    shell_radius of ANY ligand atom. This ensures even coverage around the
    ligand contour, which is important for elongated or complex ligands.

    Args:
        pdb_path: Path to PDB file.
        ligand_coords: List of all ligand atom coordinates [(x, y, z), ...].
        shell_radius: Radius of the design shell.
        exclude_resnames: Residue names to exclude (e.g., ligands).

    Returns:
        List of residue IDs outside the shell in format ['A150', 'A152', ...].
    """
    from fastmpnndesign.constants import STANDARD_AA, NONSTANDARD_AA

    if exclude_resnames is None:
        exclude_resnames = []

    # Track which residues have at least one atom inside the shell
    residues_in_shell = set()
    all_residues = {}  # (chain, resnum) -> True

    for atom in iter_pdb_atoms(pdb_path):
        # Only consider protein residues
        if atom['resname'] not in STANDARD_AA and atom['resname'] not in NONSTANDARD_AA:
            continue
        if atom['resname'] in exclude_resnames:
            continue

        chain = atom['chain'] or 'A'
        resnum = atom['resnum']
        key = (chain, resnum)

        all_residues[key] = True

        # Check if any atom of this residue is within shell_radius of ANY ligand atom
        if key not in residues_in_shell:
            atom_x, atom_y, atom_z = atom['x'], atom['y'], atom['z']
            for lig_x, lig_y, lig_z in ligand_coords:
                dist = calculate_distance(lig_x, lig_y, lig_z, atom_x, atom_y, atom_z)
                if dist <= shell_radius:
                    residues_in_shell.add(key)
                    break  # No need to check other ligand atoms

    # Residues outside shell = all residues - residues in shell
    residues_outside = []
    for (chain, resnum) in sorted(all_residues.keys(), key=lambda x: (x[0], x[1])):
        if (chain, resnum) not in residues_in_shell:
            residues_outside.append(f"{chain}{resnum}")

    return residues_outside


def get_ligand_atom_coords(pdb_path: Path, ligand_resname: Optional[str] = None) -> List[Tuple[float, float, float]]:
    """
    Get all heavy atom coordinates of ligand(s) in a PDB file.

    Args:
        pdb_path: Path to PDB file.
        ligand_resname: Specific ligand resname to use (optional).

    Returns:
        List of heavy atom coordinates [(x, y, z), ...], empty list if no ligand found.
    """
    from fastmpnndesign.constants import SOLVENTS, BUFFERS, STANDARD_AA, NONSTANDARD_AA

    exclude = SOLVENTS | BUFFERS | STANDARD_AA | NONSTANDARD_AA

    coords = []
    for atom in iter_pdb_atoms(pdb_path):
        resname = atom['resname']

        # Skip if we have a specific ligand and this isn't it
        if ligand_resname and resname != ligand_resname:
            continue

        # Include HETATM records that aren't excluded
        if atom['record_type'] == 'HETATM' and resname not in exclude:
            # Skip hydrogen atoms (only keep heavy atoms)
            element = atom['element'] or get_element_from_atom_name(atom['name'])
            if element.upper() != 'H':
                coords.append((atom['x'], atom['y'], atom['z']))

    return coords
