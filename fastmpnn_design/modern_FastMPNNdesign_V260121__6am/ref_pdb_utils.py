#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference PDB Utilities for Enzyme Design

This module provides functionality for:
- Aligning structures by ligand atoms
- Deriving enzyme design constraints from a reference PDB
- Calculating metrics comparing designs to a reference structure

Used when a reference PDB with ideal catalytic geometry is provided via --ref_pdb.
"""

import os
import sys
import numpy as np
import pyrosetta as pyr
import pyrosetta.rosetta

# Local imports
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
import rosetta_utils

from constants import (
    DEFAULT_CONSTRAINT_WEIGHTS,
)


# =============================================================================
# Constants for Constraint Derivation
# =============================================================================

# Atom preference for distance constraints (prefer O, N, S over C)
CONSTRAINT_ATOM_PREFERENCE = ["O", "N", "S", "C"]

# Default tolerances for derived constraints
DEFAULT_DISTANCE_TOLERANCE = 0.5   # Angstroms
DEFAULT_ANGLE_TOLERANCE = 15.0     # Degrees
DEFAULT_DIHEDRAL_TOLERANCE = 20.0  # Degrees
DEFAULT_DISTANCE_FORCE = 100.0     # Force constant
DEFAULT_ANGLE_FORCE = 50.0         # Force constant
DEFAULT_DIHEDRAL_FORCE = 50.0      # Force constant

# Backbone atom names
BACKBONE_ATOMS = {"N", "CA", "C", "O"}


# =============================================================================
# Ligand Alignment Functions
# =============================================================================

def get_ligand_heavy_atom_coords(pose, ligand_resno=None):
    """
    Get coordinates of ligand heavy atoms.

    Arguments:
        pose: PyRosetta Pose
        ligand_resno: Ligand residue number (default: auto-detect last ligand)

    Returns:
        tuple: (coords_array, atom_names) where coords is Nx3 numpy array
    """
    if ligand_resno is None:
        # Find last ligand residue
        for i in range(pose.size(), 0, -1):
            if pose.residue(i).is_ligand():
                ligand_resno = i
                break

    if ligand_resno is None or not pose.residue(ligand_resno).is_ligand():
        raise ValueError(f"No ligand found in pose")

    lig = pose.residue(ligand_resno)
    coords = []
    names = []

    for i in range(1, lig.natoms() + 1):
        if not lig.atom_is_hydrogen(i):
            coords.append(np.array(lig.xyz(i)))
            names.append(lig.atom_name(i).strip())

    return np.array(coords), names


def align_by_ligand(mobile_pose, target_pose, mobile_ligand_resno=None, target_ligand_resno=None, verbose=True):
    """
    Align mobile_pose to target_pose by superimposing their ligands.

    Arguments:
        mobile_pose: Pose to be transformed (typically ref_pdb)
        target_pose: Reference pose to align to (typically input_pdb)
        mobile_ligand_resno: Ligand residue in mobile_pose (default: auto-detect)
        target_ligand_resno: Ligand residue in target_pose (default: auto-detect)
        verbose: Print alignment information

    Returns:
        tuple: (aligned_pose, ligand_rmsd, rotation_matrix, translation_vector)

    Notes:
        - Uses Kabsch algorithm on ligand heavy atoms
        - Applies transformation to entire mobile_pose
        - Ligand RMSD should be ~0 for identical ligands
    """
    # Get ligand coordinates
    mobile_coords, mobile_names = get_ligand_heavy_atom_coords(mobile_pose, mobile_ligand_resno)
    target_coords, target_names = get_ligand_heavy_atom_coords(target_pose, target_ligand_resno)

    if verbose:
        print(f"  [Ligand Alignment] Mobile ligand atoms: {len(mobile_names)}")
        print(f"  [Ligand Alignment] Target ligand atoms: {len(target_names)}")

    # Check atom counts match
    if len(mobile_coords) != len(target_coords):
        raise ValueError(f"Ligand atom count mismatch: mobile={len(mobile_coords)}, target={len(target_coords)}")

    # Match atoms by name (in case ordering differs)
    if mobile_names != target_names:
        if verbose:
            print(f"  [Ligand Alignment] Atom names differ, attempting to match by name...")
        # Reorder mobile coords to match target names
        mobile_name_to_coord = {name: coord for name, coord in zip(mobile_names, mobile_coords)}
        reordered_coords = []
        for name in target_names:
            if name not in mobile_name_to_coord:
                raise ValueError(f"Atom {name} not found in mobile ligand")
            reordered_coords.append(mobile_name_to_coord[name])
        mobile_coords = np.array(reordered_coords)

    # Kabsch algorithm
    # Center both coordinate sets
    mobile_centroid = np.mean(mobile_coords, axis=0)
    target_centroid = np.mean(target_coords, axis=0)
    mobile_centered = mobile_coords - mobile_centroid
    target_centered = target_coords - target_centroid

    # Compute optimal rotation using SVD
    H = mobile_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Calculate translation
    t = target_centroid - R @ mobile_centroid

    # Calculate RMSD
    mobile_transformed = (R @ mobile_coords.T).T + t
    rmsd = np.sqrt(np.mean(np.sum((mobile_transformed - target_coords) ** 2, axis=1)))

    if verbose:
        print(f"  [Ligand Alignment] Ligand RMSD after alignment: {rmsd:.4f} Å")
        if rmsd > 0.5:
            print(f"  [WARNING] Ligand RMSD > 0.5 Å - ligands may have different conformations!")

    # Apply transformation to entire mobile pose
    aligned_pose = mobile_pose.clone()

    for resno in range(1, aligned_pose.size() + 1):
        res = aligned_pose.residue(resno)
        for atomno in range(1, res.natoms() + 1):
            old_xyz = np.array(res.xyz(atomno))
            new_xyz = R @ old_xyz + t
            aligned_pose.set_xyz(
                pyrosetta.rosetta.core.id.AtomID(atomno, resno),
                pyrosetta.rosetta.numeric.xyzVector_double_t(new_xyz[0], new_xyz[1], new_xyz[2])
            )

    return aligned_pose, rmsd, R, t


# =============================================================================
# REMARK 666 Parsing and Subset Selection
# =============================================================================

def get_remark666_lines(filename):
    """
    Get raw REMARK 666 lines from a PDB file.

    Returns:
        list: List of (line_number, line_text, parsed_dict) tuples (1-indexed)
    """
    if isinstance(filename, str):
        with open(filename, 'r') as f:
            pdbfile = f.readlines()
    elif isinstance(filename, pyr.rosetta.core.pose.Pose):
        pdbfile = pyr.distributed.io.to_pdbstring(filename).split("\n")
    else:
        raise TypeError(f"Expected str or Pose, got {type(filename)}")

    remark_lines = []
    line_num = 0

    for line in pdbfile:
        if "ATOM" in line:
            break
        if "REMARK 666" in line:
            line_num += 1
            lspl = line.split()
            parsed = {
                'target_name': lspl[5],
                'target_chain': lspl[4],
                'target_resno': int(lspl[6]),
                'chain': lspl[9],
                'name3': lspl[10],
                'resno': int(lspl[11]),
                'cst_no': int(lspl[12]),
                'cst_no_var': int(lspl[13])
            }
            remark_lines.append((line_num, line.strip(), parsed))

    return remark_lines


def get_catres_subset(input_pose, ref_pose, subset_indices=None, verbose=True):
    """
    Get catalytic residues from both poses, optionally filtering by REMARK 666 line indices.

    Arguments:
        input_pose: Input PDB pose (or path)
        ref_pose: Reference PDB pose (or path)
        subset_indices: List of 1-based REMARK 666 line indices (e.g., [1, 3, 5])
                       None means use all REMARK 666 lines
        verbose: Print information about selection

    Returns:
        tuple: (input_catres_dict, ref_catres_dict, warnings)

    Raises:
        ValueError: If residue types don't match between input and ref for selected indices
    """
    input_remarks = get_remark666_lines(input_pose)
    ref_remarks = get_remark666_lines(ref_pose)

    if verbose:
        print(f"  [CatRes Subset] Input PDB REMARK 666 lines: {len(input_remarks)}")
        print(f"  [CatRes Subset] Ref PDB REMARK 666 lines: {len(ref_remarks)}")

    warnings = []

    # Determine which indices to use
    if subset_indices is None:
        # Use all
        max_lines = min(len(input_remarks), len(ref_remarks))
        subset_indices = list(range(1, max_lines + 1))
        if verbose:
            print(f"  [CatRes Subset] Using all {max_lines} REMARK 666 lines")
    else:
        if verbose:
            print(f"  [CatRes Subset] Using subset indices: {subset_indices}")

    # Validate and collect
    input_catres = {}
    ref_catres = {}
    valid_indices = []

    for idx in subset_indices:
        # Check if index is valid for both files
        if idx > len(input_remarks):
            warnings.append(f"REMARK 666 line {idx} exceeds input PDB lines ({len(input_remarks)})")
            continue
        if idx > len(ref_remarks):
            warnings.append(f"REMARK 666 line {idx} exceeds ref PDB lines ({len(ref_remarks)})")
            continue

        input_parsed = input_remarks[idx - 1][2]
        ref_parsed = ref_remarks[idx - 1][2]

        # Validate residue types match
        if input_parsed['name3'] != ref_parsed['name3']:
            raise ValueError(
                f"Residue type mismatch at REMARK 666 line {idx}: "
                f"input={input_parsed['name3']} at {input_parsed['resno']}, "
                f"ref={ref_parsed['name3']} at {ref_parsed['resno']}"
            )

        input_catres[input_parsed['resno']] = input_parsed
        ref_catres[ref_parsed['resno']] = ref_parsed
        valid_indices.append(idx)

    if verbose:
        print(f"  [CatRes Subset] Valid catalytic residues: {len(valid_indices)}")
        for idx in valid_indices:
            inp = input_remarks[idx - 1][2]
            print(f"    Line {idx}: {inp['name3']} {inp['resno']} (chain {inp['chain']})")
        for w in warnings:
            print(f"  [WARNING] {w}")

    return input_catres, ref_catres, warnings


# =============================================================================
# Constraint Derivation Functions
# =============================================================================

def get_atom_element(residue, atom_index):
    """Get element symbol for an atom."""
    return residue.atom_type(atom_index).element()


def find_closest_atom_pair(res1, res2, prefer_non_carbon=True, exclude_hydrogen=True, res1_sc_only=False):
    """
    Find closest atom pair between two residues with element preferences.

    Arguments:
        res1: First residue (typically protein sidechain/backbone)
        res2: Second residue (typically ligand)
        prefer_non_carbon: Prefer O, N, S over C atoms
        exclude_hydrogen: Exclude hydrogen atoms
        res1_sc_only: Only consider sidechain atoms for res1

    Returns:
        tuple: ((atom1_idx, atom1_name, elem1), (atom2_idx, atom2_name, elem2), distance)
    """
    best_pair = None
    best_dist = float('inf')
    best_score = float('inf')  # Lower is better (non-C preferred)

    for i in range(1, res1.natoms() + 1):
        if exclude_hydrogen and res1.atom_is_hydrogen(i):
            continue
        if res1_sc_only and res1.atom_is_backbone(i):
            continue

        elem1 = get_atom_element(res1, i)
        xyz1 = np.array(res1.xyz(i))

        # Calculate element preference score (lower = better)
        if prefer_non_carbon:
            try:
                score1 = CONSTRAINT_ATOM_PREFERENCE.index(elem1)
            except ValueError:
                score1 = len(CONSTRAINT_ATOM_PREFERENCE)
        else:
            score1 = 0

        for j in range(1, res2.natoms() + 1):
            if exclude_hydrogen and res2.atom_is_hydrogen(j):
                continue

            elem2 = get_atom_element(res2, j)
            xyz2 = np.array(res2.xyz(j))

            dist = np.linalg.norm(xyz1 - xyz2)

            # Calculate element preference score for ligand atom
            if prefer_non_carbon:
                try:
                    score2 = CONSTRAINT_ATOM_PREFERENCE.index(elem2)
                except ValueError:
                    score2 = len(CONSTRAINT_ATOM_PREFERENCE)
            else:
                score2 = 0

            total_score = score1 + score2

            # Prefer by: 1) element preference, 2) distance
            # But only if distance is reasonably close (within 2x of best)
            if dist < best_dist:
                # Always update if closer
                if best_pair is None or dist < best_dist * 0.5 or total_score <= best_score:
                    best_dist = dist
                    best_score = total_score
                    best_pair = (
                        (i, res1.atom_name(i).strip(), elem1),
                        (j, res2.atom_name(j).strip(), elem2),
                        dist
                    )
            elif dist < best_dist * 1.5 and total_score < best_score:
                # Update if better element preference and not too much farther
                best_dist = dist
                best_score = total_score
                best_pair = (
                    (i, res1.atom_name(i).strip(), elem1),
                    (j, res2.atom_name(j).strip(), elem2),
                    dist
                )

    return best_pair


def get_bonded_heavy_atoms(residue, atom_index):
    """
    Get heavy atoms bonded to a given atom.

    Returns:
        list: List of (atom_index, atom_name, element) tuples
    """
    bonded = []
    for i in range(1, residue.natoms() + 1):
        if i == atom_index:
            continue
        if residue.atom_is_hydrogen(i):
            continue
        # Check if bonded (Rosetta provides this via atom connectivity)
        if residue.is_bonded(atom_index, i):
            bonded.append((i, residue.atom_name(i).strip(), get_atom_element(residue, i)))
    return bonded


def get_atom_chain_for_constraints(residue, anchor_atom_idx, depth=2):
    """
    Get a chain of bonded atoms from an anchor atom for constraint derivation.

    Arguments:
        residue: PyRosetta residue
        anchor_atom_idx: Starting atom index
        depth: How many bonded atoms to traverse

    Returns:
        list: List of atom indices forming a chain from anchor outward
    """
    chain = [anchor_atom_idx]
    current = anchor_atom_idx
    visited = {anchor_atom_idx}

    for _ in range(depth):
        bonded = get_bonded_heavy_atoms(residue, current)
        bonded = [b for b in bonded if b[0] not in visited]
        if not bonded:
            break
        # Pick the first bonded atom (could be improved with heuristics)
        next_atom = bonded[0][0]
        chain.append(next_atom)
        visited.add(next_atom)
        current = next_atom

    return chain


def derive_constraints_for_residue(catres, ligand, catres_resno, ligand_resno, verbose=True):
    """
    Derive distance, angle, and dihedral constraints for one catalytic residue.

    Arguments:
        catres: Catalytic residue (PyRosetta residue)
        ligand: Ligand residue (PyRosetta residue)
        catres_resno: Residue number of catalytic residue in pose
        ligand_resno: Residue number of ligand in pose
        verbose: Print constraint details

    Returns:
        dict: {
            'distance': (catres_atom, lig_atom, distance),
            'angle1': (catres_atom1, catres_atom2, lig_atom, angle),  # 2 SC + 1 lig
            'angle2': (catres_atom, lig_atom1, lig_atom2, angle),     # 1 SC + 2 lig
            'dihedral1': (cat1, cat2, cat3, lig1, dihedral),          # 3 SC + 1 lig
            'dihedral2': (cat1, lig1, lig2, lig3, dihedral),          # 1 SC + 3 lig
            'dihedral3': (cat1, cat2, lig1, lig2, dihedral),          # 2 SC + 2 lig
        }
    """
    result = {}

    # 1. Find closest atom pair (distance constraint)
    pair = find_closest_atom_pair(catres, ligand, prefer_non_carbon=True,
                                   exclude_hydrogen=True, res1_sc_only=False)

    if pair is None:
        if verbose:
            print(f"    [WARNING] No atom pair found for residue {catres_resno}")
        return None

    cat_atom, lig_atom, dist = pair
    result['distance'] = {
        'cat_atom_idx': cat_atom[0],
        'cat_atom_name': cat_atom[1],
        'lig_atom_idx': lig_atom[0],
        'lig_atom_name': lig_atom[1],
        'distance': dist
    }

    if verbose:
        print(f"    Distance: {catres.name3()}{catres_resno}:{cat_atom[1]} - LIG:{lig_atom[1]} = {dist:.2f} Å")

    # 2. Get atom chains for angle/dihedral constraints
    cat_chain = get_atom_chain_for_constraints(catres, cat_atom[0], depth=2)
    lig_chain = get_atom_chain_for_constraints(ligand, lig_atom[0], depth=2)

    # 3. Angle1: 2 SC atoms + 1 ligand atom
    if len(cat_chain) >= 2:
        # Atoms: cat_chain[1] - cat_chain[0] - lig_chain[0]
        xyz1 = np.array(catres.xyz(cat_chain[1]))
        xyz2 = np.array(catres.xyz(cat_chain[0]))
        xyz3 = np.array(ligand.xyz(lig_chain[0]))
        angle1 = rosetta_utils.get_angle(xyz1, xyz2, xyz3)
        result['angle1'] = {
            'cat_atom1_idx': cat_chain[1],
            'cat_atom1_name': catres.atom_name(cat_chain[1]).strip(),
            'cat_atom2_idx': cat_chain[0],
            'cat_atom2_name': catres.atom_name(cat_chain[0]).strip(),
            'lig_atom_idx': lig_chain[0],
            'lig_atom_name': ligand.atom_name(lig_chain[0]).strip(),
            'angle': angle1
        }
        if verbose:
            print(f"    Angle1 (2SC+1L): {catres.atom_name(cat_chain[1]).strip()}-{cat_atom[1]}-{lig_atom[1]} = {angle1:.1f}°")

    # 4. Angle2: 1 SC atom + 2 ligand atoms
    if len(lig_chain) >= 2:
        # Atoms: cat_chain[0] - lig_chain[0] - lig_chain[1]
        xyz1 = np.array(catres.xyz(cat_chain[0]))
        xyz2 = np.array(ligand.xyz(lig_chain[0]))
        xyz3 = np.array(ligand.xyz(lig_chain[1]))
        angle2 = rosetta_utils.get_angle(xyz1, xyz2, xyz3)
        result['angle2'] = {
            'cat_atom_idx': cat_chain[0],
            'cat_atom_name': catres.atom_name(cat_chain[0]).strip(),
            'lig_atom1_idx': lig_chain[0],
            'lig_atom1_name': ligand.atom_name(lig_chain[0]).strip(),
            'lig_atom2_idx': lig_chain[1],
            'lig_atom2_name': ligand.atom_name(lig_chain[1]).strip(),
            'angle': angle2
        }
        if verbose:
            print(f"    Angle2 (1SC+2L): {cat_atom[1]}-{lig_atom[1]}-{ligand.atom_name(lig_chain[1]).strip()} = {angle2:.1f}°")

    # 5. Dihedral1: 3 SC atoms + 1 ligand atom
    if len(cat_chain) >= 3:
        xyz1 = np.array(catres.xyz(cat_chain[2]))
        xyz2 = np.array(catres.xyz(cat_chain[1]))
        xyz3 = np.array(catres.xyz(cat_chain[0]))
        xyz4 = np.array(ligand.xyz(lig_chain[0]))
        dih1 = calculate_dihedral(xyz1, xyz2, xyz3, xyz4)
        result['dihedral1'] = {
            'cat_atoms': [cat_chain[2], cat_chain[1], cat_chain[0]],
            'cat_names': [catres.atom_name(i).strip() for i in [cat_chain[2], cat_chain[1], cat_chain[0]]],
            'lig_atoms': [lig_chain[0]],
            'lig_names': [ligand.atom_name(lig_chain[0]).strip()],
            'dihedral': dih1
        }
        if verbose:
            print(f"    Dihedral1 (3SC+1L): {dih1:.1f}°")

    # 6. Dihedral2: 1 SC atom + 3 ligand atoms
    if len(lig_chain) >= 3:
        xyz1 = np.array(catres.xyz(cat_chain[0]))
        xyz2 = np.array(ligand.xyz(lig_chain[0]))
        xyz3 = np.array(ligand.xyz(lig_chain[1]))
        xyz4 = np.array(ligand.xyz(lig_chain[2]))
        dih2 = calculate_dihedral(xyz1, xyz2, xyz3, xyz4)
        result['dihedral2'] = {
            'cat_atoms': [cat_chain[0]],
            'cat_names': [catres.atom_name(cat_chain[0]).strip()],
            'lig_atoms': [lig_chain[0], lig_chain[1], lig_chain[2]],
            'lig_names': [ligand.atom_name(i).strip() for i in [lig_chain[0], lig_chain[1], lig_chain[2]]],
            'dihedral': dih2
        }
        if verbose:
            print(f"    Dihedral2 (1SC+3L): {dih2:.1f}°")

    # 7. Dihedral3: 2 SC atoms + 2 ligand atoms
    if len(cat_chain) >= 2 and len(lig_chain) >= 2:
        xyz1 = np.array(catres.xyz(cat_chain[1]))
        xyz2 = np.array(catres.xyz(cat_chain[0]))
        xyz3 = np.array(ligand.xyz(lig_chain[0]))
        xyz4 = np.array(ligand.xyz(lig_chain[1]))
        dih3 = calculate_dihedral(xyz1, xyz2, xyz3, xyz4)
        result['dihedral3'] = {
            'cat_atoms': [cat_chain[1], cat_chain[0]],
            'cat_names': [catres.atom_name(i).strip() for i in [cat_chain[1], cat_chain[0]]],
            'lig_atoms': [lig_chain[0], lig_chain[1]],
            'lig_names': [ligand.atom_name(i).strip() for i in [lig_chain[0], lig_chain[1]]],
            'dihedral': dih3
        }
        if verbose:
            print(f"    Dihedral3 (2SC+2L): {dih3:.1f}°")

    return result


def calculate_dihedral(p1, p2, p3, p4):
    """Calculate dihedral angle between four points in degrees."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1 /= np.linalg.norm(n1) if np.linalg.norm(n1) > 0 else 1
    n2 /= np.linalg.norm(n2) if np.linalg.norm(n2) > 0 else 1

    m1 = np.cross(n1, b2 / np.linalg.norm(b2))

    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return np.degrees(np.arctan2(y, x))


def derive_constraints_from_ref_pdb(ref_pose, catres_dict, ligand_resno=None, include_coord_cst=False, verbose=True):
    """
    Derive enzyme design constraints from a reference PDB structure.

    Arguments:
        ref_pose: Reference pose with ideal catalytic geometry
        catres_dict: Catalytic residue dict from get_catres_subset()
        ligand_resno: Ligand residue number (default: auto-detect)
        include_coord_cst: Include coordinate constraints for functional groups
        verbose: Print constraint details

    Returns:
        dict: {
            'cst_file_content': str,  # Rosetta-format CST file content
            'constraints_by_residue': {...},  # Detailed constraints per residue
        }
    """
    if ligand_resno is None:
        for i in range(ref_pose.size(), 0, -1):
            if ref_pose.residue(i).is_ligand():
                ligand_resno = i
                break

    ligand = ref_pose.residue(ligand_resno)
    lig_name3 = ligand.name3()
    lig_chain = ref_pose.pdb_info().chain(ligand_resno)
    lig_pdb_resno = ref_pose.pdb_info().number(ligand_resno)

    if verbose:
        print(f"  [Constraint Derivation] Ligand: {lig_name3} {lig_pdb_resno} (chain {lig_chain})")
        print(f"  [Constraint Derivation] Deriving constraints for {len(catres_dict)} catalytic residues:")

    all_constraints = {}
    cst_blocks = []

    for resno, catres_info in catres_dict.items():
        catres = ref_pose.residue(resno)
        catres_name3 = catres.name3()
        catres_chain = ref_pose.pdb_info().chain(resno)
        catres_pdb_resno = ref_pose.pdb_info().number(resno)

        if verbose:
            print(f"\n  Residue {catres_name3} {catres_pdb_resno} (chain {catres_chain}):")

        cst = derive_constraints_for_residue(catres, ligand, resno, ligand_resno, verbose=verbose)

        if cst is None:
            continue

        all_constraints[resno] = cst

        # Build Rosetta CST block
        cst_block = format_rosetta_cst_block(
            catres_name3, catres_chain, catres_pdb_resno,
            lig_name3, lig_chain, lig_pdb_resno,
            cst, len(cst_blocks) + 1
        )
        cst_blocks.append(cst_block)

    cst_file_content = "\n\n".join(cst_blocks)

    if verbose:
        print(f"\n  [Constraint Derivation] Generated {len(cst_blocks)} constraint blocks")

    return {
        'cst_file_content': cst_file_content,
        'constraints_by_residue': all_constraints,
        'ligand_resno': ligand_resno
    }


def format_rosetta_cst_block(cat_name3, cat_chain, cat_resno, lig_name3, lig_chain, lig_resno, cst, cst_num):
    """
    Format a single constraint block in Rosetta .cst file format.
    """
    lines = []
    lines.append(f"CST::BEGIN")
    lines.append(f"  TEMPLATE::   ATOM_MAP: 1 atom_name: {cst['distance']['cat_atom_name']}")
    lines.append(f"  TEMPLATE::   ATOM_MAP: 1 residue3: {cat_name3}")
    lines.append(f"  TEMPLATE::   ATOM_MAP: 2 atom_name: {cst['distance']['lig_atom_name']}")
    lines.append(f"  TEMPLATE::   ATOM_MAP: 2 residue3: {lig_name3}")
    lines.append(f"")

    # Distance constraint
    dist = cst['distance']['distance']
    lines.append(f"  CONSTRAINT:: distanceAB:  {dist:.2f}  {DEFAULT_DISTANCE_TOLERANCE:.2f} {DEFAULT_DISTANCE_FORCE:.0f}  0  0")

    # Angle constraints
    if 'angle1' in cst:
        ang = cst['angle1']['angle']
        lines.append(f"  CONSTRAINT::    angle_A:  {ang:.1f}  {DEFAULT_ANGLE_TOLERANCE:.1f} {DEFAULT_ANGLE_FORCE:.0f}  360.0  1")

    if 'angle2' in cst:
        ang = cst['angle2']['angle']
        lines.append(f"  CONSTRAINT::    angle_B:  {ang:.1f}  {DEFAULT_ANGLE_TOLERANCE:.1f} {DEFAULT_ANGLE_FORCE:.0f}  360.0  1")

    # Dihedral constraints
    if 'dihedral3' in cst:  # Most commonly used dihedral
        dih = cst['dihedral3']['dihedral']
        lines.append(f"  CONSTRAINT::  torsion_A:  {dih:.1f}  {DEFAULT_DIHEDRAL_TOLERANCE:.1f} {DEFAULT_DIHEDRAL_FORCE:.0f}  360.0  1")

    # Algorithm hints
    lines.append(f"  ALGORITHM_INFO:: match")
    lines.append(f"     IGNORE_UPSTREAM_PROTON_CHI")
    lines.append(f"  ALGORITHM_INFO::END")

    lines.append(f"CST::END")

    return "\n".join(lines)


# =============================================================================
# Metric Calculation Functions
# =============================================================================

def calculate_ligand_rmsd(pose1, pose2, lig_resno1=None, lig_resno2=None):
    """
    Calculate ligand heavy-atom RMSD between two poses.
    Assumes poses are already aligned or will be aligned by this function.
    """
    coords1, names1 = get_ligand_heavy_atom_coords(pose1, lig_resno1)
    coords2, names2 = get_ligand_heavy_atom_coords(pose2, lig_resno2)

    if len(coords1) != len(coords2):
        raise ValueError(f"Ligand atom count mismatch: {len(coords1)} vs {len(coords2)}")

    # Match by name if needed
    if names1 != names2:
        name_to_coord2 = {name: coord for name, coord in zip(names2, coords2)}
        coords2_reordered = []
        for name in names1:
            if name in name_to_coord2:
                coords2_reordered.append(name_to_coord2[name])
            else:
                raise ValueError(f"Atom {name} not found in second ligand")
        coords2 = np.array(coords2_reordered)

    diff = coords1 - coords2
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    return rmsd


def calculate_sidechain_allatom_rmsd(pose1, pose2, residues, after_alignment=True):
    """
    Calculate all-atom sidechain RMSD for specified residues.

    Arguments:
        pose1, pose2: Poses to compare
        residues: List of residue numbers
        after_alignment: If True, assume poses are already aligned

    Returns:
        float: All-atom sidechain RMSD
    """
    coords1 = []
    coords2 = []

    for resno in residues:
        res1 = pose1.residue(resno)
        res2 = pose2.residue(resno)

        for i in range(1, min(res1.natoms(), res2.natoms()) + 1):
            # Skip backbone atoms
            if res1.atom_is_backbone(i):
                continue
            # Skip hydrogens
            if res1.atom_is_hydrogen(i):
                continue

            coords1.append(np.array(res1.xyz(i)))
            coords2.append(np.array(res2.xyz(i)))

    if len(coords1) == 0:
        return 0.0

    coords1 = np.array(coords1)
    coords2 = np.array(coords2)

    diff = coords1 - coords2
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    return rmsd


def calculate_backbone_allatom_rmsd(pose1, pose2, residues):
    """
    Calculate all-atom backbone RMSD (N, CA, C, O) for specified residues.
    """
    coords1 = []
    coords2 = []

    for resno in residues:
        res1 = pose1.residue(resno)
        res2 = pose2.residue(resno)

        for atom_name in ["N", "CA", "C", "O"]:
            if res1.has(atom_name) and res2.has(atom_name):
                coords1.append(np.array(res1.xyz(atom_name)))
                coords2.append(np.array(res2.xyz(atom_name)))

    if len(coords1) == 0:
        return 0.0

    coords1 = np.array(coords1)
    coords2 = np.array(coords2)

    diff = coords1 - coords2
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    return rmsd


def calculate_interacting_functional_group_rmsd(pose1, pose2, catres_dict, ligand_resno1=None, ligand_resno2=None):
    """
    Calculate RMSD of functional groups that interact with ligand.
    Uses the closest atoms (same logic as constraint derivation).
    """
    if ligand_resno1 is None:
        for i in range(pose1.size(), 0, -1):
            if pose1.residue(i).is_ligand():
                ligand_resno1 = i
                break

    if ligand_resno2 is None:
        for i in range(pose2.size(), 0, -1):
            if pose2.residue(i).is_ligand():
                ligand_resno2 = i
                break

    coords1 = []
    coords2 = []

    for resno in catres_dict.keys():
        catres1 = pose1.residue(resno)
        catres2 = pose2.residue(resno)
        ligand1 = pose1.residue(ligand_resno1)
        ligand2 = pose2.residue(ligand_resno2)

        # Find closest atom pair for pose1
        pair1 = find_closest_atom_pair(catres1, ligand1, prefer_non_carbon=True,
                                        exclude_hydrogen=True, res1_sc_only=False)
        if pair1 is None:
            continue

        cat_atom_idx = pair1[0][0]

        # Get corresponding atom in pose2
        if cat_atom_idx <= catres2.natoms():
            coords1.append(np.array(catres1.xyz(cat_atom_idx)))
            coords2.append(np.array(catres2.xyz(cat_atom_idx)))

    if len(coords1) == 0:
        return 0.0

    coords1 = np.array(coords1)
    coords2 = np.array(coords2)

    diff = coords1 - coords2
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    return rmsd


def calculate_ref_pdb_metrics(design_pose, ref_pose, catres_subset, input_pose=None, verbose=True):
    """
    Calculate all metrics comparing design to reference PDB.

    Arguments:
        design_pose: Designed pose to evaluate
        ref_pose: Reference PDB (already aligned to input by ligand)
        catres_subset: Dict of catalytic residues in the subset
        input_pose: Original input pose (optional, for additional metrics)
        verbose: Print metric details

    Returns:
        dict: All ref_pdb metrics
    """
    metrics = {}

    # First, align design to ref_pose by ligand
    try:
        aligned_design, lig_rmsd, _, _ = align_by_ligand(design_pose, ref_pose, verbose=False)
        metrics['lig_rmsd_to_refpdb'] = lig_rmsd
    except Exception as e:
        if verbose:
            print(f"  [WARNING] Failed to align design to ref_pdb: {e}")
        metrics['lig_rmsd_to_refpdb'] = np.nan
        return metrics

    catres_list = list(catres_subset.keys())

    # Sidechain all-atom RMSD
    try:
        sc_rmsd = calculate_sidechain_allatom_rmsd(aligned_design, ref_pose, catres_list)
        metrics['catres_subset_allatom_sc_rmsd_to_refpdb'] = sc_rmsd
    except Exception as e:
        if verbose:
            print(f"  [WARNING] SC RMSD calculation failed: {e}")
        metrics['catres_subset_allatom_sc_rmsd_to_refpdb'] = np.nan

    # Backbone all-atom RMSD
    try:
        bb_rmsd = calculate_backbone_allatom_rmsd(aligned_design, ref_pose, catres_list)
        metrics['catres_subset_allatom_bb_rmsd_to_refpdb'] = bb_rmsd
    except Exception as e:
        if verbose:
            print(f"  [WARNING] BB RMSD calculation failed: {e}")
        metrics['catres_subset_allatom_bb_rmsd_to_refpdb'] = np.nan

    # Interacting functional group RMSD
    try:
        interact_rmsd = calculate_interacting_functional_group_rmsd(aligned_design, ref_pose, catres_subset)
        metrics['catres_subset_interact_rmsd_to_refpdb'] = interact_rmsd
    except Exception as e:
        if verbose:
            print(f"  [WARNING] Interact RMSD calculation failed: {e}")
        metrics['catres_subset_interact_rmsd_to_refpdb'] = np.nan

    # CA RMSD to ref_pdb
    try:
        ca_rmsd = rosetta_utils.calculate_ca_rmsd(aligned_design, ref_pose)
        metrics['ca_rmsd_to_refpdb'] = ca_rmsd
    except Exception as e:
        if verbose:
            print(f"  [WARNING] CA RMSD calculation failed: {e}")
        metrics['ca_rmsd_to_refpdb'] = np.nan

    # CA RMSD converged to ref_pdb
    try:
        ca_rmsd_conv, _, _ = rosetta_utils.calculate_ca_rmsd_converged(aligned_design, ref_pose)
        metrics['ca_rmsd_converge_to_refpdb'] = ca_rmsd_conv
    except Exception as e:
        if verbose:
            print(f"  [WARNING] CA RMSD converged calculation failed: {e}")
        metrics['ca_rmsd_converge_to_refpdb'] = np.nan

    if verbose:
        print(f"  [Ref PDB Metrics]")
        print(f"    lig_rmsd_to_refpdb: {metrics.get('lig_rmsd_to_refpdb', np.nan):.3f} Å")
        print(f"    catres_sc_rmsd: {metrics.get('catres_subset_allatom_sc_rmsd_to_refpdb', np.nan):.3f} Å")
        print(f"    catres_bb_rmsd: {metrics.get('catres_subset_allatom_bb_rmsd_to_refpdb', np.nan):.3f} Å")
        print(f"    catres_interact_rmsd: {metrics.get('catres_subset_interact_rmsd_to_refpdb', np.nan):.3f} Å")
        print(f"    ca_rmsd_to_refpdb: {metrics.get('ca_rmsd_to_refpdb', np.nan):.3f} Å")

    return metrics


# =============================================================================
# Main Processing Function
# =============================================================================

def process_ref_pdb(input_pose, ref_pdb_path, catres_cst_subset=None, include_coord_cst=False,
                    derive_cst=True, verbose=True):
    """
    Main function to process a reference PDB for constraint derivation and metrics.

    Arguments:
        input_pose: Input pose (already loaded)
        ref_pdb_path: Path to reference PDB file
        catres_cst_subset: Comma-separated list of REMARK 666 line numbers (e.g., "1,3,5")
        include_coord_cst: Include coordinate constraints
        derive_cst: Whether to derive constraints (vs just setup for metrics)
        verbose: Print details

    Returns:
        dict: {
            'ref_pose': aligned reference pose,
            'input_catres': catalytic residues from input,
            'ref_catres': catalytic residues from ref,
            'cst_file_path': path to generated CST file (if derive_cst=True),
            'cst_result': full constraint derivation result,
            'ligand_rmsd': ligand RMSD after alignment,
        }
    """
    if verbose:
        print(f"\n  [Ref PDB] Loading: {ref_pdb_path}")

    # Load reference PDB
    ref_pose_raw = pyr.pose_from_file(ref_pdb_path)

    if verbose:
        print(f"  [Ref PDB] Loaded: {ref_pose_raw.size()} residues")

    # Parse subset indices
    subset_indices = None
    if catres_cst_subset:
        subset_indices = [int(x.strip()) for x in catres_cst_subset.split(",")]

    # Get and validate catalytic residue subsets
    input_catres, ref_catres, warnings = get_catres_subset(
        input_pose, ref_pose_raw, subset_indices, verbose=verbose
    )

    # Align ref_pdb to input_pdb by ligand
    ref_pose, lig_rmsd, R, t = align_by_ligand(ref_pose_raw, input_pose, verbose=verbose)

    result = {
        'ref_pose': ref_pose,
        'input_catres': input_catres,
        'ref_catres': ref_catres,
        'ligand_rmsd': lig_rmsd,
        'cst_file_path': None,
        'cst_result': None,
    }

    # Derive constraints if requested
    if derive_cst:
        if verbose:
            print(f"\n  [Ref PDB] Deriving constraints from reference structure...")

        cst_result = derive_constraints_from_ref_pdb(
            ref_pose, ref_catres, include_coord_cst=include_coord_cst, verbose=verbose
        )
        result['cst_result'] = cst_result

        # Write CST file
        import tempfile
        pdb_basename = os.path.basename(ref_pdb_path).replace(".pdb", "")
        cst_file_path = os.path.join(tempfile.gettempdir(), f"{pdb_basename}_derived.cst")

        with open(cst_file_path, 'w') as f:
            f.write(cst_result['cst_file_content'])

        result['cst_file_path'] = cst_file_path

        if verbose:
            print(f"  [Ref PDB] Wrote constraint file: {cst_file_path}")

    return result
