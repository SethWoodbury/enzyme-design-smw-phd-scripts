"""
Ligand-based alignment utilities for fastmpnndesign.

Provides functions to align reference PDB to input PDB using ligand HETATM atoms.
This ensures coordinate constraints from ref_pdb are in the input PDB's coordinate frame.

Uses the Kabsch algorithm to compute optimal rotation and translation.
"""

from pathlib import Path
from typing import Tuple, Dict, List, Optional
import numpy as np

from fastmpnndesign.utils import iter_pdb_atoms, get_element_from_atom_name
from fastmpnndesign.constants import SOLVENTS, BUFFERS, STANDARD_AA, NONSTANDARD_AA
from fastmpnndesign.logging_config import get_logger

logger = get_logger("alignment")


def extract_ligand_coords(pdb_path: Path, ligand_resname: str) -> np.ndarray:
    """
    Extract heavy atom coordinates of ligand.

    Args:
        pdb_path: Path to PDB file.
        ligand_resname: 3-letter residue name of the ligand.

    Returns:
        np.ndarray of shape (N, 3) containing heavy atom coordinates,
        sorted by atom name for consistent ordering.

    Raises:
        ValueError: If no ligand atoms found with the given resname.
    """
    pdb_path = Path(pdb_path)

    # Collect atoms with their names for sorting
    atoms_data: List[Tuple[str, float, float, float]] = []

    for atom in iter_pdb_atoms(pdb_path):
        if atom['record_type'] != 'HETATM':
            continue

        if atom['resname'].upper() != ligand_resname.upper():
            continue

        # Skip hydrogens (only keep heavy atoms)
        element = atom.get('element', '')
        if not element:
            element = get_element_from_atom_name(atom['name'])
        if element.upper() == 'H':
            continue

        atoms_data.append((
            atom['name'],
            atom['x'],
            atom['y'],
            atom['z']
        ))

    if not atoms_data:
        raise ValueError(
            f"No heavy atoms found for ligand '{ligand_resname}' in {pdb_path}"
        )

    # Sort by atom name for consistent ordering between PDBs
    atoms_data.sort(key=lambda x: x[0])

    # Extract coordinates
    coords = np.array([[x, y, z] for (_, x, y, z) in atoms_data])

    logger.debug(
        f"Extracted {len(coords)} heavy atoms for ligand {ligand_resname} from {pdb_path.name}"
    )

    return coords


def extract_ligand_coords_with_names(
    pdb_path: Path,
    ligand_resname: str
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract heavy atom coordinates and names of ligand.

    Args:
        pdb_path: Path to PDB file.
        ligand_resname: 3-letter residue name of the ligand.

    Returns:
        Tuple of (coords array of shape (N, 3), list of atom names).

    Raises:
        ValueError: If no ligand atoms found with the given resname.
    """
    pdb_path = Path(pdb_path)

    atoms_data: List[Tuple[str, float, float, float]] = []

    for atom in iter_pdb_atoms(pdb_path):
        if atom['record_type'] != 'HETATM':
            continue

        if atom['resname'].upper() != ligand_resname.upper():
            continue

        element = atom.get('element', '')
        if not element:
            element = get_element_from_atom_name(atom['name'])
        if element.upper() == 'H':
            continue

        atoms_data.append((
            atom['name'],
            atom['x'],
            atom['y'],
            atom['z']
        ))

    if not atoms_data:
        raise ValueError(
            f"No heavy atoms found for ligand '{ligand_resname}' in {pdb_path}"
        )

    # Sort by atom name for consistent ordering
    atoms_data.sort(key=lambda x: x[0])

    names = [name for (name, _, _, _) in atoms_data]
    coords = np.array([[x, y, z] for (_, x, y, z) in atoms_data])

    return coords, names


def compute_alignment_transform(
    ref_coords: np.ndarray,
    target_coords: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rotation matrix R and translation t to align ref to target using Kabsch algorithm.

    The transformation aligns ref_coords to target_coords such that:
        aligned_ref = (ref_coords - ref_centroid) @ R + target_centroid

    Or equivalently:
        aligned_point = R @ (point - ref_centroid) + target_centroid
        aligned_point = R @ point + t
        where t = target_centroid - R @ ref_centroid

    Args:
        ref_coords: Reference coordinates to be aligned, shape (N, 3).
        target_coords: Target coordinates to align to, shape (N, 3).

    Returns:
        Tuple of (R, t) where:
            R: Rotation matrix (3, 3)
            t: Translation vector (3,)

        To transform a point p from ref frame to target frame:
            p_aligned = R @ p + t

    Raises:
        ValueError: If coordinate arrays have different shapes or too few points.
    """
    if ref_coords.shape != target_coords.shape:
        raise ValueError(
            f"Coordinate arrays must have same shape. "
            f"Got ref: {ref_coords.shape}, target: {target_coords.shape}"
        )

    if len(ref_coords) < 3:
        raise ValueError(
            f"Need at least 3 points for alignment, got {len(ref_coords)}"
        )

    # Center the coordinates
    ref_centroid = np.mean(ref_coords, axis=0)
    target_centroid = np.mean(target_coords, axis=0)

    ref_centered = ref_coords - ref_centroid
    target_centered = target_coords - target_centroid

    # Compute covariance matrix H = ref^T @ target
    H = ref_centered.T @ target_centered

    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    # R = V @ U^T
    R = Vt.T @ U.T

    # Handle reflection case (ensure proper rotation, det(R) = +1)
    if np.linalg.det(R) < 0:
        # Flip the sign of the last column of V
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    # t = target_centroid - R @ ref_centroid
    t = target_centroid - R @ ref_centroid

    return R, t


def transform_coordinates(
    coords: Tuple[float, float, float],
    R: np.ndarray,
    t: np.ndarray
) -> Tuple[float, float, float]:
    """
    Apply transformation to a coordinate.

    Args:
        coords: Input coordinate (x, y, z).
        R: Rotation matrix (3, 3).
        t: Translation vector (3,).

    Returns:
        Transformed coordinate (x', y', z').
    """
    point = np.array(coords)
    transformed = R @ point + t
    return (float(transformed[0]), float(transformed[1]), float(transformed[2]))


def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Compute RMSD between two coordinate sets.

    Args:
        coords1: First coordinate set, shape (N, 3).
        coords2: Second coordinate set, shape (N, 3).

    Returns:
        RMSD value in Angstroms.
    """
    diff = coords1 - coords2
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def align_ref_to_input_by_ligand(
    ref_pdb: Path,
    input_pdb: Path,
    ligand_resname: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align ref_pdb to input_pdb by ligand atoms.

    Extracts ligand heavy atoms from both PDBs, computes the optimal
    rotation and translation to align ref_pdb ligand to input_pdb ligand.

    The ligand should be identical between ref_pdb and input_pdb, so the
    alignment should have near-zero RMSD (typically < 0.01 A for identical
    ligands in different conformations).

    Args:
        ref_pdb: Path to reference PDB file.
        input_pdb: Path to input PDB file (target for alignment).
        ligand_resname: 3-letter residue name of the ligand.

    Returns:
        Tuple of (R, t) transformation matrices.
        To transform a point p from ref_pdb frame to input_pdb frame:
            p_aligned = R @ p + t

    Raises:
        ValueError: If ligand atoms don't match between PDBs.
    """
    ref_pdb = Path(ref_pdb)
    input_pdb = Path(input_pdb)

    # Extract ligand coordinates with atom names
    ref_coords, ref_names = extract_ligand_coords_with_names(ref_pdb, ligand_resname)
    input_coords, input_names = extract_ligand_coords_with_names(input_pdb, ligand_resname)

    # Verify atom names match
    if ref_names != input_names:
        # Try to find common atoms
        common_names = set(ref_names) & set(input_names)
        if len(common_names) < 3:
            raise ValueError(
                f"Ligand atoms don't match between PDBs. "
                f"ref has {ref_names}, input has {input_names}. "
                f"Only {len(common_names)} common atoms."
            )

        # Filter to common atoms
        logger.warning(
            f"Ligand atom names differ between PDBs. "
            f"Using {len(common_names)} common atoms for alignment."
        )

        ref_mask = [name in common_names for name in ref_names]
        input_mask = [name in common_names for name in input_names]

        # Need to reorder to match
        ref_common = {name: ref_coords[i] for i, name in enumerate(ref_names) if ref_mask[i]}
        input_common = {name: input_coords[i] for i, name in enumerate(input_names) if input_mask[i]}

        common_sorted = sorted(common_names)
        ref_coords = np.array([ref_common[name] for name in common_sorted])
        input_coords = np.array([input_common[name] for name in common_sorted])

    # Compute alignment transformation
    R, t = compute_alignment_transform(ref_coords, input_coords)

    # Compute RMSD to verify alignment quality
    ref_aligned = (ref_coords @ R.T) + t
    rmsd = compute_rmsd(ref_aligned, input_coords)

    logger.info(
        f"Aligned ref_pdb to input_pdb by ligand {ligand_resname}: "
        f"RMSD = {rmsd:.4f} A ({len(ref_coords)} atoms)"
    )

    if rmsd > 0.5:
        logger.warning(
            f"High alignment RMSD ({rmsd:.4f} A) - ligands may not be identical!"
        )
    elif rmsd > 0.1:
        logger.warning(
            f"Moderate alignment RMSD ({rmsd:.4f} A) - check ligand conformations"
        )

    return R, t


def get_alignment_rmsd(
    ref_pdb: Path,
    input_pdb: Path,
    ligand_resname: str
) -> float:
    """
    Compute the RMSD after aligning ref_pdb to input_pdb by ligand.

    Useful for verifying that ligands are indeed identical.

    Args:
        ref_pdb: Path to reference PDB file.
        input_pdb: Path to input PDB file.
        ligand_resname: 3-letter residue name of the ligand.

    Returns:
        RMSD value in Angstroms after optimal alignment.
    """
    ref_coords, ref_names = extract_ligand_coords_with_names(ref_pdb, ligand_resname)
    input_coords, input_names = extract_ligand_coords_with_names(input_pdb, ligand_resname)

    # Handle mismatched atoms
    if ref_names != input_names:
        common_names = sorted(set(ref_names) & set(input_names))
        if len(common_names) < 3:
            raise ValueError(f"Too few common atoms ({len(common_names)}) for RMSD calculation")

        ref_common = {name: ref_coords[i] for i, name in enumerate(ref_names)}
        input_common = {name: input_coords[i] for i, name in enumerate(input_names)}

        ref_coords = np.array([ref_common[name] for name in common_names])
        input_coords = np.array([input_common[name] for name in common_names])

    R, t = compute_alignment_transform(ref_coords, input_coords)
    ref_aligned = (ref_coords @ R.T) + t

    return compute_rmsd(ref_aligned, input_coords)


def detect_ligand_resname(pdb_path: Path) -> Optional[str]:
    """
    Detect the ligand residue name from a PDB file.

    Returns the first HETATM residue that isn't a solvent, buffer, or amino acid.

    Args:
        pdb_path: Path to PDB file.

    Returns:
        Ligand resname if found, None otherwise.
    """
    exclude = SOLVENTS | BUFFERS | STANDARD_AA | NONSTANDARD_AA

    for atom in iter_pdb_atoms(pdb_path):
        if atom['record_type'] == 'HETATM':
            resname = atom['resname'].upper()
            if resname not in exclude:
                return resname

    return None
