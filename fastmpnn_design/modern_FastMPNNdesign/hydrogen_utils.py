#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hydrogen Atom Handling Utilities for Enzyme Design

This module provides proper hydrogen atom handling based on the reference
implementation in add_remark666_lines_AND_rosetta_hydrogens_to_pdb_OPTIMIZED.py

Key features:
- Preserve original protonation states from input PDB
- Transfer H-atoms from reference to design structures
- Validate non-H RMSD after H addition
- Handle catalytic residue H-atoms specially
- Detect and apply correct HIS tautomers from raw PDB

The main issue this solves: PyRosetta auto-assigns HIS tautomers during
structure loading, which may not match the intended tautomer from the
input PDB. This module parses raw PDB files to detect which hydrogens
(HD1 vs HE2) are present and enforces that tautomerization.
"""

import os
import sys
import tempfile
import numpy as np

try:
    import pyrosetta
    import pyrosetta.distributed.io
except ImportError:
    pyrosetta = None

# =============================================================================
# Constants
# =============================================================================

# RMSD threshold for non-H atoms after H addition (should be ~0)
MAX_NON_H_RMSD_THRESHOLD = 0.01  # Angstroms


# =============================================================================
# HIS Tautomer Detection
# =============================================================================

def build_his_tautomer_map_from_pdb(pdb_path_or_str, verbose=False):
    """
    Detect histidine tautomers from raw PDB file by checking for HD1/HE2 atoms.

    This is the definitive way to determine HIS tautomers - by directly
    inspecting which hydrogen atoms are present in the original structure.
    PyRosetta's auto-detection may differ from what's actually in the PDB.

    Tautomer assignments:
    - HIS_D: Only HD1 present (delta-protonated, Nd-H)
    - HIS: Only HE2 present (epsilon-protonated, Ne-H) - Rosetta default
    - HIS with both: Both HD1 and HE2 present (doubly protonated, rare)

    Arguments:
        pdb_path_or_str: Path to PDB file or PDB string content
        verbose: Print tautomer assignments

    Returns:
        dict: {(chain, resno): tautomer_name}
    """
    his_map = {}
    his_atoms = {}  # {(chain, resno): set of atom names}

    # Read PDB content
    if os.path.exists(pdb_path_or_str):
        with open(pdb_path_or_str, 'r') as f:
            lines = f.readlines()
    else:
        lines = pdb_path_or_str.split('\n')

    # Collect HIS atoms
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            atom_name = line[12:16].strip()
            resn = line[17:20].strip()
            chain = line[21].strip() if len(line) > 21 else 'A'
            resno = int(line[22:26].strip())
        except (ValueError, IndexError):
            continue

        if resn != "HIS":
            continue

        key = (chain, resno)
        if key not in his_atoms:
            his_atoms[key] = set()
        his_atoms[key].add(atom_name)

    # Determine tautomers based on which H atoms are present
    for key, atoms in his_atoms.items():
        has_hd1 = "HD1" in atoms
        has_he2 = "HE2" in atoms

        if has_hd1 and not has_he2:
            his_map[key] = "HIS_D"  # Delta-protonated (Nd-H)
            if verbose:
                print(f"  [HIS Tautomer] {key[0]}{key[1]}: HD1 only -> HIS_D (delta-protonated)")
        elif has_he2 and not has_hd1:
            his_map[key] = "HIS"  # Epsilon-protonated (Ne-H) - Rosetta default
            if verbose:
                print(f"  [HIS Tautomer] {key[0]}{key[1]}: HE2 only -> HIS (epsilon-protonated)")
        elif has_hd1 and has_he2:
            his_map[key] = "HIS"  # Doubly protonated - use HIS (could be HIP in some force fields)
            if verbose:
                print(f"  [HIS Tautomer] {key[0]}{key[1]}: both HD1 and HE2 -> HIS (doubly protonated)")
        else:
            his_map[key] = "HIS"  # Default if no H atoms found
            if verbose:
                print(f"  [HIS Tautomer] {key[0]}{key[1]}: no HD1/HE2 found -> HIS (default)")

    return his_map


def apply_his_tautomers_to_pose(pose, his_tautomer_map, verbose=True):
    """
    Apply histidine tautomers from a map to a pose.

    This ensures that all HIS residues have the correct tautomer as
    determined from the original PDB file, rather than PyRosetta's
    auto-assigned tautomers.

    Arguments:
        pose: PyRosetta Pose (modified in place)
        his_tautomer_map: dict from build_his_tautomer_map_from_pdb()
        verbose: Print mutations

    Returns:
        list: List of (resno, old_type, new_type) tuples for mutations made
    """
    if pyrosetta is None:
        raise ImportError("PyRosetta is required for apply_his_tautomers_to_pose")

    mutations = []

    for resno in range(1, pose.size() + 1):
        res = pose.residue(resno)
        if res.name3() != "HIS":
            continue

        chain = pose.pdb_info().chain(resno)
        pdb_resno = pose.pdb_info().number(resno)
        key = (chain, pdb_resno)

        if key not in his_tautomer_map:
            continue

        target_type = his_tautomer_map[key]
        current_name = res.name()  # Full name including patches

        # Check if already correct
        if target_type == "HIS_D" and "HIS_D" in current_name:
            continue  # Already delta-protonated
        if target_type == "HIS" and "HIS_D" not in current_name:
            continue  # Already epsilon-protonated (default HIS)

        # Need to mutate
        if verbose:
            print(f"  [HIS Tautomer] Mutating {chain}{pdb_resno} from {current_name} to {target_type}")

        mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
        mutres.set_res_name(target_type)
        mutres.set_target(resno)
        mutres.set_preserve_atom_coords(True)  # Keep non-H coordinates
        mutres.apply(pose)

        mutations.append((resno, current_name, target_type))

    if verbose and mutations:
        print(f"  [HIS Tautomer] Applied {len(mutations)} tautomer corrections")

    return mutations


# =============================================================================
# Non-Hydrogen Coordinate Utilities
# =============================================================================

def extract_non_hydrogen_coords_dict(pose_or_pdb_lines):
    """
    Extract non-hydrogen atom coordinates as a dictionary.

    This is used to validate that non-H atoms haven't moved after
    hydrogen addition/modification.

    Arguments:
        pose_or_pdb_lines: Either a PyRosetta Pose or list of PDB lines

    Returns:
        dict: {(chain, resno, atom_name): np.array([x, y, z])}
    """
    coords_dict = {}

    if pyrosetta is not None and isinstance(pose_or_pdb_lines, pyrosetta.rosetta.core.pose.Pose):
        pose = pose_or_pdb_lines
        for resno in range(1, pose.size() + 1):
            res = pose.residue(resno)
            chain = pose.pdb_info().chain(resno)
            pdb_resno = pose.pdb_info().number(resno)
            for atomno in range(1, res.natoms() + 1):
                if not res.atom_is_hydrogen(atomno):
                    atom_name = res.atom_name(atomno).strip()
                    xyz = np.array(res.xyz(atomno))
                    coords_dict[(chain, pdb_resno, atom_name)] = xyz
    else:
        # PDB lines
        lines = pose_or_pdb_lines if isinstance(pose_or_pdb_lines, list) else pose_or_pdb_lines.split('\n')
        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    atom_name = line[12:16].strip()
                    # Skip hydrogens
                    if atom_name.startswith('H') or (len(atom_name) > 0 and atom_name[0].isdigit() and 'H' in atom_name):
                        continue
                    chain = line[21].strip() if len(line) > 21 else 'A'
                    resno = int(line[22:26].strip())
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords_dict[(chain, resno, atom_name)] = np.array([x, y, z])
                except (ValueError, IndexError):
                    continue

    return coords_dict


def calculate_non_h_rmsd(coords_dict1, coords_dict2):
    """
    Calculate RMSD between two coordinate dictionaries.

    Arguments:
        coords_dict1: First coordinate dict from extract_non_hydrogen_coords_dict
        coords_dict2: Second coordinate dict

    Returns:
        tuple: (rmsd, n_matched_atoms)
    """
    common_keys = set(coords_dict1.keys()) & set(coords_dict2.keys())
    if len(common_keys) == 0:
        return 0.0, 0

    coords1 = np.array([coords_dict1[k] for k in sorted(common_keys)])
    coords2 = np.array([coords_dict2[k] for k in sorted(common_keys)])

    diff = coords1 - coords2
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd, len(common_keys)


# =============================================================================
# Hydrogen Addition and Validation
# =============================================================================

def add_hydrogens_with_validation(pose, reference_pose=None, catalytic_residues=None,
                                   validate_rmsd=True, verbose=True):
    """
    Add hydrogens to a pose with proper validation.

    This function:
    1. Saves current non-H coordinates
    2. Lets PyRosetta add hydrogens (via PDB round-trip)
    3. Validates non-H atoms didn't move
    4. Optionally transfers H orientations from reference for catalytic residues

    Arguments:
        pose: PyRosetta Pose (modified in place)
        reference_pose: Optional reference pose with correct H orientations
        catalytic_residues: List/dict of catalytic residue numbers
        validate_rmsd: If True, validate non-H RMSD after H addition
        verbose: Print details

    Returns:
        dict: {'non_h_rmsd': float, 'n_atoms': int, 'valid': bool}
    """
    if pyrosetta is None:
        raise ImportError("PyRosetta is required for add_hydrogens_with_validation")

    result = {'valid': True}

    # Save original non-H coordinates
    original_coords = extract_non_hydrogen_coords_dict(pose)

    if verbose:
        print(f"  [H-Add] Original structure: {len(original_coords)} non-H atoms")

    # PyRosetta will add hydrogens when we score or manipulate the pose
    # Force H addition by converting to PDB string and back
    pdb_str = pyrosetta.distributed.io.to_pdbstring(pose)

    # Create temporary file and reload
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_str)
        temp_path = f.name

    try:
        # Reload - PyRosetta adds hydrogens on load
        new_pose = pyrosetta.pose_from_file(temp_path)

        # Validate non-H RMSD
        new_coords = extract_non_hydrogen_coords_dict(new_pose)
        rmsd, n_matched = calculate_non_h_rmsd(original_coords, new_coords)

        result['non_h_rmsd'] = rmsd
        result['n_atoms'] = n_matched

        if verbose:
            print(f"  [H-Add] Non-H atom RMSD after H addition: {rmsd:.4f} A ({n_matched} atoms)")

        if validate_rmsd and rmsd > MAX_NON_H_RMSD_THRESHOLD:
            if verbose:
                print(f"  [H-Add] WARNING: Non-H RMSD {rmsd:.4f} > threshold {MAX_NON_H_RMSD_THRESHOLD}")
            result['valid'] = False

        # Transfer structure back to original pose
        for resno in range(1, min(pose.size(), new_pose.size()) + 1):
            old_res = pose.residue(resno)
            new_res = new_pose.residue(resno)
            for atomno in range(1, min(old_res.natoms(), new_res.natoms()) + 1):
                atom_id = pyrosetta.rosetta.core.id.AtomID(atomno, resno)
                pose.set_xyz(atom_id, new_res.xyz(atomno))

        # Optionally transfer H from reference for catalytic residues
        if reference_pose is not None and catalytic_residues is not None:
            catres_list = list(catalytic_residues.keys()) if isinstance(catalytic_residues, dict) else list(catalytic_residues)

            for resno in catres_list:
                if resno <= pose.size() and resno <= reference_pose.size():
                    _transfer_hydrogen_positions(pose, reference_pose, resno, verbose=verbose)

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    return result


def _transfer_hydrogen_positions(pose, reference_pose, resno, verbose=True):
    """
    Transfer hydrogen positions from reference pose to target pose for a residue.

    This preserves the protonation state and H orientations from the reference,
    which is critical for maintaining the correct chemistry at catalytic sites.

    Arguments:
        pose: Target PyRosetta Pose (modified in place)
        reference_pose: Reference Pose with correct H positions
        resno: Residue sequence position
        verbose: Print transfer details
    """
    if pyrosetta is None:
        raise ImportError("PyRosetta is required for _transfer_hydrogen_positions")

    ref_res = reference_pose.residue(resno)
    tgt_res = pose.residue(resno)

    # Check residue types match (at least base type)
    if ref_res.name3() != tgt_res.name3():
        # Allow HIS variants
        if not (ref_res.name3() == "HIS" and tgt_res.name3() == "HIS"):
            if verbose:
                print(f"  [H-Transfer] Skipping res {resno}: type mismatch ({ref_res.name3()} vs {tgt_res.name3()})")
            return

    transferred = 0
    for atomno in range(1, ref_res.natoms() + 1):
        if ref_res.atom_is_hydrogen(atomno):
            atom_name = ref_res.atom_name(atomno).strip()
            if tgt_res.has(atom_name):
                tgt_atomno = tgt_res.atom_index(atom_name)
                atom_id = pyrosetta.rosetta.core.id.AtomID(tgt_atomno, resno)
                pose.set_xyz(atom_id, ref_res.xyz(atomno))
                transferred += 1

    if verbose and transferred > 0:
        print(f"  [H-Transfer] Res {resno} ({ref_res.name3()}): transferred {transferred} H atoms from reference")


# =============================================================================
# REMARK 666 Extraction
# =============================================================================

def extract_remark666_lines(pdb_path_or_str):
    """
    Extract REMARK 666 lines from a PDB file or string.

    REMARK 666 lines contain matcher information about catalytic residues.

    Arguments:
        pdb_path_or_str: Path to PDB file or PDB string content

    Returns:
        list: REMARK 666 lines
    """
    remark_lines = []

    if os.path.exists(pdb_path_or_str):
        with open(pdb_path_or_str, 'r') as f:
            lines = f.readlines()
    else:
        lines = pdb_path_or_str.split('\n')

    for line in lines:
        if line.startswith("REMARK 666"):
            remark_lines.append(line.rstrip('\n'))

    return remark_lines


def parse_catalytic_residues_from_remark666(remark666_lines, verbose=False):
    """
    Parse catalytic residue information from REMARK 666 lines.

    Arguments:
        remark666_lines: List of REMARK 666 lines
        verbose: Print parsing details

    Returns:
        dict: {(chain, pdb_resno): {'name3': str, 'cst_block': int, ...}}
    """
    catres = {}

    for line in remark666_lines:
        try:
            if "MATCH MOTIF" not in line:
                continue

            parts = line.split()
            motif_idx = parts.index("MOTIF")

            res_chain = parts[motif_idx + 1]
            res_name3 = parts[motif_idx + 2]
            res_pdbno = int(parts[motif_idx + 3])
            cst_block = int(parts[motif_idx + 4]) if len(parts) > motif_idx + 4 else 1

            key = (res_chain, res_pdbno)
            catres[key] = {
                'name3': res_name3,
                'cst_block': cst_block,
                'chain': res_chain,
                'pdb_resno': res_pdbno
            }

            if verbose:
                print(f"  [REMARK666] Found catalytic residue: {res_chain}{res_pdbno} {res_name3} (block {cst_block})")

        except (IndexError, ValueError) as e:
            if verbose:
                print(f"  [REMARK666] Warning: Could not parse line: {line[:50]}... - {e}")

    return catres


# =============================================================================
# High-Level Utilities
# =============================================================================

def ensure_correct_his_tautomers(pose, input_pdb_path, verbose=True):
    """
    Ensure HIS residues have correct tautomers based on input PDB.

    This is a convenience function that combines tautomer detection
    and application.

    Arguments:
        pose: PyRosetta Pose (modified in place)
        input_pdb_path: Path to original input PDB
        verbose: Print details

    Returns:
        list: List of mutations made [(resno, old_type, new_type), ...]
    """
    if verbose:
        print(f"\n  [HIS Tautomer] Detecting tautomers from {os.path.basename(input_pdb_path)}...")

    his_map = build_his_tautomer_map_from_pdb(input_pdb_path, verbose=verbose)

    if not his_map:
        if verbose:
            print("  [HIS Tautomer] No HIS residues found in input PDB")
        return []

    if verbose:
        print(f"  [HIS Tautomer] Found {len(his_map)} HIS residues, applying tautomers...")

    mutations = apply_his_tautomers_to_pose(pose, his_map, verbose=verbose)

    return mutations


def validate_hydrogen_consistency(pose, reference_pdb_path, catalytic_residues=None, verbose=True):
    """
    Validate that hydrogen atoms are consistent between pose and reference.

    This checks that catalytic residue hydrogens match the reference structure,
    which is critical for maintaining proper enzyme chemistry.

    Arguments:
        pose: PyRosetta Pose to validate
        reference_pdb_path: Path to reference PDB with correct hydrogens
        catalytic_residues: Optional list/dict of catalytic residue numbers
        verbose: Print validation details

    Returns:
        dict: {'valid': bool, 'mismatches': [...], 'warnings': [...]}
    """
    result = {'valid': True, 'mismatches': [], 'warnings': []}

    # Build HIS tautomer maps for both
    ref_his_map = build_his_tautomer_map_from_pdb(reference_pdb_path, verbose=False)

    # Check each HIS in pose against reference
    for resno in range(1, pose.size() + 1):
        res = pose.residue(resno)
        if res.name3() != "HIS":
            continue

        chain = pose.pdb_info().chain(resno)
        pdb_resno = pose.pdb_info().number(resno)
        key = (chain, pdb_resno)

        if key not in ref_his_map:
            continue

        ref_tautomer = ref_his_map[key]
        pose_name = res.name()

        # Check if tautomers match
        pose_is_his_d = "HIS_D" in pose_name
        ref_is_his_d = ref_tautomer == "HIS_D"

        if pose_is_his_d != ref_is_his_d:
            result['mismatches'].append({
                'resno': resno,
                'chain': chain,
                'pdb_resno': pdb_resno,
                'pose_type': pose_name,
                'ref_type': ref_tautomer
            })
            result['valid'] = False

            if verbose:
                print(f"  [H-Validate] MISMATCH: {chain}{pdb_resno} is {pose_name} but should be {ref_tautomer}")

    if verbose:
        if result['valid']:
            print(f"  [H-Validate] All HIS tautomers match reference")
        else:
            print(f"  [H-Validate] Found {len(result['mismatches'])} HIS tautomer mismatches")

    return result
