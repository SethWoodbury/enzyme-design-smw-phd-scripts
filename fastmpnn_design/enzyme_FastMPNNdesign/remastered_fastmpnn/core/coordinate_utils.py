"""
Coordinate transformation utilities.

Provides Kabsch alignment algorithm, RMSD calculation, and
coordinate transformation functions using numpy.
"""

from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from remastered_fastmpnn.core.pdb_io import PDBParser, AtomRecord
from remastered_fastmpnn.logging_config import get_logger

logger = get_logger("coordinate_utils")


def calculate_rmsd(
    coords1: NDArray[np.float64],
    coords2: NDArray[np.float64],
) -> float:
    """
    Calculate RMSD between two coordinate sets.

    Args:
        coords1: First coordinate array (N x 3)
        coords2: Second coordinate array (N x 3)

    Returns:
        RMSD value in Angstroms
    """
    if coords1.shape != coords2.shape:
        raise ValueError(
            f"Coordinate shape mismatch: {coords1.shape} vs {coords2.shape}"
        )

    n = coords1.shape[0]
    if n == 0:
        return 0.0

    diff = coords1 - coords2
    return np.sqrt(np.sum(diff * diff) / n)


def centroid(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculate centroid of coordinates.

    Args:
        coords: Coordinate array (N x 3)

    Returns:
        Centroid as 1D array of length 3
    """
    return np.mean(coords, axis=0)


def kabsch_rotation(
    P: NDArray[np.float64],
    Q: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate optimal rotation matrix using Kabsch algorithm.

    Finds the optimal rotation matrix R that minimizes RMSD
    when transforming P to match Q: Q ≈ R @ P

    Both P and Q should be centered (centroid at origin).

    Args:
        P: Source coordinates (N x 3), centered
        Q: Target coordinates (N x 3), centered

    Returns:
        3x3 rotation matrix R
    """
    # Compute covariance matrix
    H = P.T @ Q

    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    if d < 0:
        # Reflection detected, correct it
        Vt[-1, :] *= -1

    # Calculate rotation matrix
    R = Vt.T @ U.T

    return R


def kabsch_align(
    source_coords: NDArray[np.float64],
    target_coords: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """
    Align source coordinates to target using Kabsch algorithm.

    Computes optimal rotation R and translation t such that:
        aligned = R @ source + t
    minimizes RMSD to target.

    Args:
        source_coords: Source coordinates (N x 3)
        target_coords: Target coordinates (N x 3)

    Returns:
        Tuple of (rotation_matrix, translation_vector, rmsd)
    """
    if source_coords.shape != target_coords.shape:
        raise ValueError(
            f"Coordinate shape mismatch: "
            f"{source_coords.shape} vs {target_coords.shape}"
        )

    # Calculate centroids
    source_centroid = centroid(source_coords)
    target_centroid = centroid(target_coords)

    # Center coordinates
    P = source_coords - source_centroid
    Q = target_coords - target_centroid

    # Get rotation matrix
    R = kabsch_rotation(P, Q)

    # Calculate translation: t = target_centroid - R @ source_centroid
    t = target_centroid - R @ source_centroid

    # Calculate RMSD after alignment
    aligned = transform_coordinates(source_coords, R, t)
    rmsd = calculate_rmsd(aligned, target_coords)

    return R, t, rmsd


def transform_coordinates(
    coords: NDArray[np.float64],
    R: NDArray[np.float64],
    t: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Apply rotation and translation to coordinates.

    Args:
        coords: Coordinates to transform (N x 3)
        R: 3x3 rotation matrix
        t: Translation vector (length 3)

    Returns:
        Transformed coordinates (N x 3)
    """
    return (R @ coords.T).T + t


def atoms_to_coords(atoms: List[AtomRecord]) -> NDArray[np.float64]:
    """
    Extract coordinates from list of AtomRecord.

    Args:
        atoms: List of AtomRecord objects

    Returns:
        Numpy array of shape (N, 3)
    """
    return np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)


def update_atom_coords(
    atoms: List[AtomRecord],
    coords: NDArray[np.float64],
) -> None:
    """
    Update AtomRecord coordinates from numpy array (in-place).

    Args:
        atoms: List of AtomRecord to update
        coords: New coordinates (N x 3)
    """
    if len(atoms) != coords.shape[0]:
        raise ValueError(
            f"Atom count mismatch: {len(atoms)} atoms vs {coords.shape[0]} coords"
        )

    for atom, (x, y, z) in zip(atoms, coords):
        atom.x = float(x)
        atom.y = float(y)
        atom.z = float(z)


def align_by_ligand(
    ref_parser: PDBParser,
    input_parser: PDBParser,
    ligand_name: Optional[str] = None,
) -> Tuple[PDBParser, float, NDArray[np.float64], NDArray[np.float64]]:
    """
    Align reference PDB to input PDB by ligand coordinates.

    Finds the optimal transformation to align the reference ligand
    to the input ligand, then applies this transformation to ALL
    atoms in the reference PDB.

    Args:
        ref_parser: Reference PDB parser (will be modified in-place)
        input_parser: Input PDB parser (target)
        ligand_name: Ligand residue name (auto-detected if None)

    Returns:
        Tuple of (modified_ref_parser, rmsd, rotation_matrix, translation_vector)

    Raises:
        ValueError: If ligands cannot be matched
    """
    # Get ligand atoms
    ref_ligand = ref_parser.get_ligand_atoms()
    input_ligand = input_parser.get_ligand_atoms()

    if not ref_ligand:
        raise ValueError("No ligand atoms found in reference PDB")
    if not input_ligand:
        raise ValueError("No ligand atoms found in input PDB")

    # Filter by ligand name if specified
    if ligand_name:
        ref_ligand = [a for a in ref_ligand if a.resname == ligand_name]
        input_ligand = [a for a in input_ligand if a.resname == ligand_name]

    logger.info(f"Reference ligand: {len(ref_ligand)} atoms")
    logger.info(f"Input ligand: {len(input_ligand)} atoms")

    # Match atoms by name
    ref_by_name = {a.name.strip(): a for a in ref_ligand}
    input_by_name = {a.name.strip(): a for a in input_ligand}

    # Find common atoms
    common_names = set(ref_by_name.keys()) & set(input_by_name.keys())
    if not common_names:
        raise ValueError("No matching ligand atoms found between ref and input")

    logger.info(f"Matching ligand atoms: {len(common_names)}")

    # Sort names for consistent ordering
    common_names = sorted(common_names)

    # Extract matched coordinates
    ref_coords = np.array([
        [ref_by_name[n].x, ref_by_name[n].y, ref_by_name[n].z]
        for n in common_names
    ])
    input_coords = np.array([
        [input_by_name[n].x, input_by_name[n].y, input_by_name[n].z]
        for n in common_names
    ])

    # Compute alignment (align ref to input)
    R, t, rmsd = kabsch_align(ref_coords, input_coords)

    logger.info(f"Ligand alignment RMSD: {rmsd:.6f} A")

    # Apply transformation to ALL atoms in reference PDB
    all_ref_coords = atoms_to_coords(ref_parser.atoms)
    transformed_coords = transform_coordinates(all_ref_coords, R, t)
    update_atom_coords(ref_parser.atoms, transformed_coords)

    # Clear cached indices since coordinates changed
    ref_parser._residue_index = None
    ref_parser._chain_index = None

    return ref_parser, rmsd, R, t


def calculate_residue_rmsd(
    parser1: PDBParser,
    parser2: PDBParser,
    chain: str,
    resnum: int,
    backbone_only: bool = False,
    sidechain_only: bool = False,
) -> Optional[float]:
    """
    Calculate RMSD between a residue in two PDB structures.

    Args:
        parser1: First PDB parser
        parser2: Second PDB parser
        chain: Chain identifier
        resnum: Residue number
        backbone_only: Only consider backbone atoms
        sidechain_only: Only consider sidechain atoms

    Returns:
        RMSD in Angstroms, or None if residue not found
    """
    atoms1 = parser1.get_residue_atoms(chain, resnum)
    atoms2 = parser2.get_residue_atoms(chain, resnum)

    if not atoms1 or not atoms2:
        return None

    # Filter by backbone/sidechain if requested
    if backbone_only:
        atoms1 = [a for a in atoms1 if a.is_backbone()]
        atoms2 = [a for a in atoms2 if a.is_backbone()]
    elif sidechain_only:
        atoms1 = [a for a in atoms1 if a.is_sidechain()]
        atoms2 = [a for a in atoms2 if a.is_sidechain()]

    # Match by atom name
    names1 = {a.name.strip(): a for a in atoms1}
    names2 = {a.name.strip(): a for a in atoms2}
    common = set(names1.keys()) & set(names2.keys())

    if not common:
        return None

    coords1 = np.array([[names1[n].x, names1[n].y, names1[n].z] for n in common])
    coords2 = np.array([[names2[n].x, names2[n].y, names2[n].z] for n in common])

    return calculate_rmsd(coords1, coords2)


def calculate_distance(
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float,
) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        x1, y1, z1: First point coordinates
        x2, y2, z2: Second point coordinates

    Returns:
        Distance in Angstroms
    """
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)


def atom_distance(atom1: AtomRecord, atom2: AtomRecord) -> float:
    """
    Calculate distance between two atoms.

    Args:
        atom1: First atom
        atom2: Second atom

    Returns:
        Distance in Angstroms
    """
    return calculate_distance(
        atom1.x, atom1.y, atom1.z,
        atom2.x, atom2.y, atom2.z
    )
