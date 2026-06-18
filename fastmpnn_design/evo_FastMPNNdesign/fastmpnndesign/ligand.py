"""
Ligand detection and handling utilities.

Provides functions to identify ligands in PDB files and extract their atoms.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple

from fastmpnndesign.constants import (
    SOLVENTS, BUFFERS, METALS, STANDARD_AA, NONSTANDARD_AA, HYDROGEN_NAMES
)
from fastmpnndesign.utils import iter_pdb_atoms, get_element_from_atom_name
from fastmpnndesign.logging_config import get_logger

logger = get_logger("ligand")


@dataclass
class LigandAtom:
    """Represents a single ligand atom."""
    name: str
    element: str
    x: float
    y: float
    z: float
    serial: int
    resname: str
    chain: str
    resnum: int
    is_heavy: bool = True

    def coords(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class Ligand:
    """Represents a ligand molecule with its atoms."""
    resname: str
    chain: str
    resnum: int
    atoms: List[LigandAtom] = field(default_factory=list)

    @property
    def heavy_atoms(self) -> List[LigandAtom]:
        """Return only heavy (non-hydrogen) atoms."""
        return [a for a in self.atoms if a.is_heavy]

    @property
    def center_of_mass(self) -> Tuple[float, float, float]:
        """Calculate center of mass of heavy atoms."""
        heavy = self.heavy_atoms
        if not heavy:
            return (0.0, 0.0, 0.0)
        x = sum(a.x for a in heavy) / len(heavy)
        y = sum(a.y for a in heavy) / len(heavy)
        z = sum(a.z for a in heavy) / len(heavy)
        return (x, y, z)

    @property
    def resid(self) -> str:
        """Return residue ID in format 'chain_resname_resnum'."""
        return f"{self.chain}_{self.resname}_{self.resnum}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'resname': self.resname,
            'chain': self.chain,
            'resnum': self.resnum,
            'n_atoms': len(self.atoms),
            'n_heavy_atoms': len(self.heavy_atoms),
            'center_of_mass': self.center_of_mass
        }


def is_ligand_residue(resname: str) -> bool:
    """
    Determine if a residue name represents a ligand.

    A residue is considered a ligand if:
    - It is not a standard amino acid
    - It is not a nonstandard amino acid
    - It is not a solvent
    - It is not a common buffer/crystallization additive
    - It is not a metal ion

    Args:
        resname: 3-letter residue name.

    Returns:
        True if the residue is likely a ligand.
    """
    resname = resname.upper().strip()

    # Not a standard or nonstandard amino acid
    if resname in STANDARD_AA or resname in NONSTANDARD_AA:
        return False

    # Not solvent
    if resname in SOLVENTS:
        return False

    # Not common buffer/additive
    if resname in BUFFERS:
        return False

    # Not a metal ion (single atom residues)
    if resname in METALS:
        return False

    return True


def is_metal_residue(resname: str) -> bool:
    """Check if residue is a metal ion."""
    return resname.upper().strip() in METALS


def is_hydrogen_atom(atom_name: str, element: str = "") -> bool:
    """Check if an atom is a hydrogen."""
    name = atom_name.strip().upper()

    # Check element if provided
    if element and element.strip().upper() == 'H':
        return True

    # Check common hydrogen naming patterns
    if name in HYDROGEN_NAMES:
        return True

    # Check if starts with H followed by digit or nothing
    if name.startswith('H') and (len(name) == 1 or name[1].isdigit()):
        return True

    # Numeric prefix like 1H, 2H
    if name[0].isdigit() and len(name) > 1 and name[1] == 'H':
        return True

    return False


def detect_ligands_from_pdb(pdb_path: Path) -> List[Ligand]:
    """
    Detect all ligands in a PDB file.

    Scans HETATM records and groups atoms by residue, filtering out
    solvents, buffers, and other non-ligand entities.

    Args:
        pdb_path: Path to PDB file.

    Returns:
        List of Ligand objects.
    """
    pdb_path = Path(pdb_path)

    # Group atoms by (chain, resname, resnum)
    residue_atoms: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}

    for atom in iter_pdb_atoms(pdb_path):
        if atom['record_type'] != 'HETATM':
            continue

        resname = atom['resname']
        if not is_ligand_residue(resname):
            continue

        key = (atom['chain'], resname, atom['resnum'])
        if key not in residue_atoms:
            residue_atoms[key] = []
        residue_atoms[key].append(atom)

    # Convert to Ligand objects
    ligands = []
    for (chain, resname, resnum), atoms in residue_atoms.items():
        ligand = Ligand(
            resname=resname,
            chain=chain,
            resnum=resnum,
            atoms=[]
        )

        for atom in atoms:
            element = atom.get('element', '')
            if not element:
                element = get_element_from_atom_name(atom['name'])

            is_heavy = not is_hydrogen_atom(atom['name'], element)

            ligand.atoms.append(LigandAtom(
                name=atom['name'],
                element=element.upper(),
                x=atom['x'],
                y=atom['y'],
                z=atom['z'],
                serial=atom['serial'],
                resname=resname,
                chain=chain,
                resnum=resnum,
                is_heavy=is_heavy
            ))

        ligands.append(ligand)

    logger.info(f"Detected {len(ligands)} ligand(s) in PDB")
    for lig in ligands:
        logger.debug(
            f"  Ligand: {lig.resname} chain {lig.chain} res {lig.resnum} "
            f"({len(lig.heavy_atoms)} heavy atoms)"
        )

    return ligands


def detect_metals_from_pdb(pdb_path: Path) -> List[Ligand]:
    """
    Detect metal ions in a PDB file.

    Args:
        pdb_path: Path to PDB file.

    Returns:
        List of Ligand objects representing metal ions.
    """
    pdb_path = Path(pdb_path)
    metals = []

    for atom in iter_pdb_atoms(pdb_path):
        if atom['record_type'] != 'HETATM':
            continue

        resname = atom['resname'].upper()
        if not is_metal_residue(resname):
            continue

        element = atom.get('element', '')
        if not element:
            element = resname  # For metals, resname is often the element

        metal = Ligand(
            resname=resname,
            chain=atom['chain'],
            resnum=atom['resnum'],
            atoms=[LigandAtom(
                name=atom['name'],
                element=element.upper(),
                x=atom['x'],
                y=atom['y'],
                z=atom['z'],
                serial=atom['serial'],
                resname=resname,
                chain=atom['chain'],
                resnum=atom['resnum'],
                is_heavy=True
            )]
        )
        metals.append(metal)

    logger.info(f"Detected {len(metals)} metal ion(s) in PDB")
    for m in metals:
        logger.debug(f"  Metal: {m.resname} chain {m.chain} res {m.resnum}")

    return metals


def validate_params_for_ligands(
    ligands: List[Ligand],
    params_paths: List[Path]
) -> Dict[str, Optional[Path]]:
    """
    Check that params files exist for detected ligands.

    Attempts to match ligand residue names to params file names.

    Args:
        ligands: List of detected ligands.
        params_paths: List of params file paths.

    Returns:
        Dictionary mapping ligand resname to params path (or None if not found).
    """
    # Build lookup of params by filename stem
    params_lookup = {}
    for p in params_paths:
        p = Path(p)
        # Try both with and without .params extension
        params_lookup[p.stem.upper()] = p
        params_lookup[p.name.upper()] = p

    result = {}
    for lig in ligands:
        resname = lig.resname.upper()
        params_path = params_lookup.get(resname)

        if params_path:
            logger.debug(f"Found params for {resname}: {params_path}")
        else:
            logger.warning(f"No params file found for ligand {resname}")

        result[resname] = params_path

    return result


def get_ligand_atom_coords(ligands: List[Ligand]) -> Dict[str, Tuple[float, float, float]]:
    """
    Get coordinates of all heavy atoms from ligands.

    Returns dictionary mapping atom identifier to coordinates.
    Identifier format: 'chain_resname_resnum_atomname'
    """
    coords = {}
    for lig in ligands:
        for atom in lig.heavy_atoms:
            key = f"{lig.chain}_{lig.resname}_{lig.resnum}_{atom.name}"
            coords[key] = atom.coords()
    return coords


def get_all_ligand_entities(pdb_path: Path) -> Tuple[List[Ligand], List[Ligand]]:
    """
    Get all ligands and metals from a PDB file.

    Args:
        pdb_path: Path to PDB file.

    Returns:
        Tuple of (ligands, metals).
    """
    ligands = detect_ligands_from_pdb(pdb_path)
    metals = detect_metals_from_pdb(pdb_path)
    return ligands, metals
