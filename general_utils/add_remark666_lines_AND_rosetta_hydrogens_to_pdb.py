#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2026-01-14 by Seth Woodbury (woodbuse@uw.edu)

This script combines functionality from:
1. add_remark666_lines_to_pdb.py - transfers REMARK 666 lines
2. align_af2_with_inputs_and_copy_ligand.py - adds hydrogens using Rosetta

Key features:
- Transfers REMARK 666 and header lines from reference PDB to output PDB
- Adds hydrogen atoms using PyRosetta (auto-populated when loading)
- Correctly handles HIS protonation states (HIS_D vs HIS) from reference PDB
- Uses MutateResidue with set_preserve_atom_coords(True) to fix catalytic residues WITHOUT changing coordinates
- Calculates non-H atom RMSD (matching atoms by ID) to verify coordinates preserved (warns if > 0.01 Å)
- Avoids needing params files by temporarily stripping ligands from pdbs
- Ligands are re-attached with exact original coordinates after hydrogen addition
- Renumbers all atoms (ATOM + HETATM) sequentially in final output
- Supports directory mode (auto-match by suffix) or single file mode
- Never modifies reference PDB files
- Auto-detects and skips reference files copied to output directory (but copies to final_output_dir if specified)

Usage examples:
  # Directory mode with auto-matching by suffix
  python add_remark666_lines_AND_rosetta_hydrogens_to_pdb.py \
    --ref_pdb_dir /path/to/references \
    --output_pdb_dir /path/to/outputs \
    --final_output_dir /path/to/final_outputs \
    --find_suffixes_from_random_sample 10 \
    --clobber --verbose

  # Single file mode
  python add_remark666_lines_AND_rosetta_hydrogens_to_pdb.py \
    --ref_pdb /path/to/ref.pdb \
    --output_pdb /path/to/output.pdb \
    --final_output_dir /path/to/final_outputs
"""

import os
import sys
import glob
import argparse
import time
import random
from pathlib import Path
import tempfile
import shutil
import numpy as np

try:
    import pyrosetta
    import pyrosetta.rosetta
    import pyrosetta.distributed.io
except ImportError:
    print("ERROR: PyRosetta is not installed or not in PYTHONPATH")
    print("Please install PyRosetta to use this script")
    sys.exit(1)

# Try to import design_utils for get_matcher_residues
try:
    sys.path.append("/net/software/scripts/enzyme_design/utils")
    import design_utils
    HAS_DESIGN_UTILS = True
except ImportError:
    HAS_DESIGN_UTILS = False
    print("WARNING: design_utils not found. Will extract matcher residues from REMARK 666 lines directly.")


def extract_remark666_and_headers(ref_pdb_path, verbose=False):
    """
    Extract REMARK 666 lines and other header lines from reference PDB.

    Args:
        ref_pdb_path: Path to reference PDB file
        verbose: Print verbose output

    Returns:
        List of header lines to add
    """
    headers_to_add = []
    seen_headers = set()

    with open(ref_pdb_path, 'r') as f:
        for line in f:
            if line.startswith(("HEADER", "REMARK", "HETNAM", "LINK")):
                if line not in seen_headers:
                    headers_to_add.append(line)
                    seen_headers.add(line)

    if verbose:
        print(f"  Extracted {len(headers_to_add)} header lines from {ref_pdb_path}")
        remark666_count = sum(1 for line in headers_to_add if "REMARK 666" in line)
        print(f"  Including {remark666_count} REMARK 666 lines")

    return headers_to_add


def get_matcher_residues_from_remark666(ref_pdb_path, verbose=False):
    """
    Extract matcher residue numbers from REMARK 666 lines.

    Args:
        ref_pdb_path: Path to reference PDB file
        verbose: Print verbose output

    Returns:
        List of residue sequence positions (1-indexed)
    """
    matcher_residues = []

    with open(ref_pdb_path, 'r') as f:
        for line in f:
            if "REMARK 666" in line and "MATCH TEMPLATE" in line:
                # Parse REMARK 666 format: typically contains residue info
                # Example: REMARK 666 MATCH TEMPLATE X HIS A  145
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        resnum = int(parts[6])
                        if resnum not in matcher_residues:
                            matcher_residues.append(resnum)
                    except (ValueError, IndexError):
                        continue

    if verbose and matcher_residues:
        print(f"  Found {len(matcher_residues)} matcher residues: {matcher_residues}")

    return matcher_residues


def check_pdb_has_remark666(pdb_path):
    """
    Check if PDB file already has REMARK 666 lines.

    Args:
        pdb_path: Path to PDB file

    Returns:
        bool: True if REMARK 666 lines are present
    """
    with open(pdb_path, 'r') as f:
        for line in f:
            if "REMARK 666" in line:
                return True
    return False


def check_pdb_has_hydrogens(pdb_path):
    """
    Check if PDB file already has hydrogen atoms.

    Args:
        pdb_path: Path to PDB file

    Returns:
        bool: True if hydrogen atoms are present
    """
    hydrogen_patterns = [' H ', ' 1H', ' 2H', ' 3H', 'HH', 'HG', 'HD', 'HE', 'HZ', 'HA', 'HB']

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                atom_name = line[12:16]  # Atom name is in columns 13-16
                # Check if this is a hydrogen atom
                if any(pattern in atom_name for pattern in hydrogen_patterns):
                    return True
    return False


def verify_pdb_is_ready(pdb_path, verbose=False):
    """
    Check if PDB already has both REMARK 666 lines and hydrogens.

    Args:
        pdb_path: Path to PDB file
        verbose: Print verbose output

    Returns:
        bool: True if PDB is ready (has both REMARK 666 and hydrogens)
    """
    has_remark666 = check_pdb_has_remark666(pdb_path)
    has_hydrogens = check_pdb_has_hydrogens(pdb_path)

    if verbose:
        print(f"  Verification of {os.path.basename(pdb_path)}:")
        print(f"    REMARK 666 present: {has_remark666}")
        print(f"    Hydrogens present: {has_hydrogens}")

    return has_remark666 and has_hydrogens


def separate_protein_and_hetatm(pdb_content):
    """
    Separate PDB content into protein atoms and HETATM/ligand lines.

    Args:
        pdb_content: List of lines from PDB file

    Returns:
        tuple: (protein_lines, hetatm_lines)
    """
    protein_lines = []
    hetatm_lines = []

    for line in pdb_content:
        if line.startswith("HETATM"):
            hetatm_lines.append(line)
        elif line.startswith(("ATOM", "TER")):
            protein_lines.append(line)
        # Skip other lines for now (headers, etc.)

    return protein_lines, hetatm_lines


def extract_non_hydrogen_coords_with_ids(pdb_lines):
    """
    Extract coordinates of non-hydrogen atoms from PDB lines with atom identifiers.

    Args:
        pdb_lines: List of PDB lines

    Returns:
        dict: (chain, resno, atom_name) -> (x, y, z)
    """
    coords_dict = {}
    for line in pdb_lines:
        if line.startswith(("ATOM", "HETATM")):
            try:
                atom_name = line[12:16].strip()
                chain = line[21].strip()
                resno = int(line[22:26].strip())

                # Skip hydrogen atoms
                if atom_name.startswith('H') or (len(atom_name) > 0 and atom_name[0].isdigit() and 'H' in atom_name):
                    continue

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                key = (chain, resno, atom_name)
                coords_dict[key] = np.array([x, y, z])
            except (ValueError, IndexError):
                continue
    return coords_dict


def extract_non_hydrogen_coords(pdb_lines):
    """
    Extract coordinates of non-hydrogen atoms from PDB lines.

    Args:
        pdb_lines: List of PDB lines

    Returns:
        numpy array of shape (N, 3) with xyz coordinates
    """
    coords = []
    for line in pdb_lines:
        if line.startswith(("ATOM", "HETATM")):
            atom_name = line[12:16].strip()
            # Skip hydrogen atoms - can start with H or digit+H (like 1H, 2H, 3H)
            # Element is in columns 77-78, but often missing, so rely on atom name
            if atom_name.startswith('H') or (len(atom_name) > 0 and atom_name[0].isdigit() and 'H' in atom_name):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
            except (ValueError, IndexError):
                continue
    return np.array(coords)


def calculate_rmsd(coords1, coords2):
    """
    Calculate RMSD between two sets of coordinates.

    Args:
        coords1: numpy array of shape (N, 3)
        coords2: numpy array of shape (N, 3)

    Returns:
        float: RMSD value
    """
    if len(coords1) != len(coords2):
        return -1.0  # Indicate error
    if len(coords1) == 0:
        return 0.0
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


def calculate_rmsd_from_dicts(coords_dict1, coords_dict2):
    """
    Calculate RMSD between two coordinate dictionaries, matching atoms by ID.

    Only atoms present in BOTH dictionaries are compared.

    Args:
        coords_dict1: dict (chain, resno, atom_name) -> xyz array
        coords_dict2: dict (chain, resno, atom_name) -> xyz array

    Returns:
        tuple: (rmsd, n_matched_atoms)
    """
    # Find common atoms
    common_keys = set(coords_dict1.keys()) & set(coords_dict2.keys())

    if len(common_keys) == 0:
        return 0.0, 0

    # Extract coordinates for common atoms
    coords1_list = []
    coords2_list = []
    for key in sorted(common_keys):
        coords1_list.append(coords_dict1[key])
        coords2_list.append(coords_dict2[key])

    coords1_arr = np.array(coords1_list)
    coords2_arr = np.array(coords2_list)

    diff = coords1_arr - coords2_arr
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    return rmsd, len(common_keys)


def build_his_tautomer_map_from_raw_pdb(pdb_path, verbose=False):
    """
    Determine HIS vs HIS_D for each histidine in a PDB file.

    Logic:
    - If HD1 present and HE2 absent -> HIS_D
    - If HE2 present and HD1 absent -> HIS
    - Otherwise -> default to HIS (epsilon protonation)

    Args:
        pdb_path: Path to PDB file
        verbose: Print verbose output

    Returns:
        dict: (chain, resno) -> "HIS" or "HIS_D"
    """
    his_map = {}
    his_atoms = {}  # (chain, resno) -> set of atom names

    with open(pdb_path, 'r') as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            try:
                atom_name = line[12:16].strip()
                resn = line[17:20].strip()
                chain = line[21].strip()
                resno = int(line[22:26].strip())
            except (ValueError, IndexError):
                continue

            if resn != "HIS":
                continue

            key = (chain, resno)
            if key not in his_atoms:
                his_atoms[key] = set()
            his_atoms[key].add(atom_name)

    # Determine tautomer for each HIS
    for key, atoms in his_atoms.items():
        has_hd1 = "HD1" in atoms
        has_he2 = "HE2" in atoms

        if has_hd1 and not has_he2:
            his_map[key] = "HIS_D"
            if verbose:
                print(f"  HIS {key[0]}{key[1]}: found HD1 only -> HIS_D")
        elif has_he2 and not has_hd1:
            his_map[key] = "HIS"
            if verbose:
                print(f"  HIS {key[0]}{key[1]}: found HE2 only -> HIS")
        else:
            # Default to HIS if ambiguous
            his_map[key] = "HIS"
            if verbose:
                status = "both HD1 and HE2" if (has_hd1 and has_he2) else "neither HD1 nor HE2"
                print(f"  HIS {key[0]}{key[1]}: {status} -> defaulting to HIS")

    return his_map


def renumber_pdb_atoms(pdb_lines, start_number=1):
    """
    Renumber all ATOM and HETATM records sequentially.

    Args:
        pdb_lines: List of PDB lines
        start_number: Starting atom number (default 1)

    Returns:
        List of PDB lines with renumbered atoms
    """
    renumbered = []
    atom_num = start_number

    for line in pdb_lines:
        if line.startswith(("ATOM", "HETATM")):
            # PDB format: atom number is in columns 7-11 (5 characters, right-justified)
            new_line = line[:6] + f"{atom_num:5d}" + line[11:]
            renumbered.append(new_line)
            atom_num += 1
        else:
            renumbered.append(line)

    return renumbered


def add_hydrogens_and_fix_catalytic_residues(
    output_pdb_path,
    ref_pdb_path,
    headers_to_add,
    final_output_path,
    clobber=False,
    verbose=False
):
    """
    Add hydrogens to output PDB using PyRosetta and transfer REMARK 666 lines.

    This function:
    1. Checks if output PDB already has REMARK 666 and hydrogens
       - If yes and it's a reference file copy: skips processing but copies to final_output_dir if needed
       - If no: processes as normal
    2. Extracts original non-H atom coordinates (with IDs) for RMSD calculation
    3. Temporarily strips ligands/HETATM from output PDB
    4. Loads protein-only PDB in PyRosetta (auto-adds hydrogens)
    5. Builds HIS tautomer map from reference PDB (HIS_D vs HIS)
    6. Uses MutateResidue with set_preserve_atom_coords(True) to fix catalytic residue protonation
    7. Calculates RMSD of matching non-H atoms (warns if > 0.01 Å)
    8. Re-attaches original HETATM/ligand lines with exact coordinates
    9. Adds REMARK 666 and header lines from reference
    10. Renumbers all atoms (ATOM + HETATM) sequentially
    11. Writes final output

    Args:
        output_pdb_path: Path to output PDB file (to add hydrogens to)
        ref_pdb_path: Path to reference PDB file
        headers_to_add: List of header lines from reference
        final_output_path: Path to write final output
        clobber: Overwrite existing files
        verbose: Print verbose output

    Returns:
        str: Status - "processed", "skipped_ready", or "failed"
    """
    # Check if output file already has REMARK 666 and hydrogens
    output_basename = os.path.basename(output_pdb_path)
    ref_basename = os.path.basename(ref_pdb_path)

    # Special case: if output basename == ref basename, this might be a copied reference file
    if output_basename == ref_basename:
        if verify_pdb_is_ready(output_pdb_path, verbose):
            # If final_output_path is different from output_pdb_path, copy the file
            if os.path.abspath(final_output_path) != os.path.abspath(output_pdb_path):
                print(f"  SKIPPING PROCESSING: {output_basename} (identical to reference, already has REMARK 666 and hydrogens)")
                print(f"  Copying to final output directory...")
                os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
                shutil.copy2(output_pdb_path, final_output_path)
                print(f"  Copied to: {final_output_path}")
            else:
                print(f"  SKIPPING: {output_basename} (identical to reference, already has REMARK 666 and hydrogens)")

            return "skipped_ready"  # File is already ready
        else:
            print(f"  NOTE: {output_basename} matches reference name but missing REMARK 666 or hydrogens - will process")

    if os.path.exists(final_output_path) and not clobber:
        print(f"  WARNING: Output file exists and --clobber not set: {final_output_path}")
        return "failed"

    if verbose:
        print(f"  Processing: {output_pdb_path}")
        print(f"    Reference: {ref_pdb_path}")
        print(f"    Output: {final_output_path}")

    # Read output PDB and separate protein from ligands
    with open(output_pdb_path, 'r') as f:
        output_content = f.readlines()

    protein_lines, hetatm_lines = separate_protein_and_hetatm(output_content)

    if verbose:
        print(f"  Separated {len(protein_lines)} protein lines and {len(hetatm_lines)} HETATM lines")

    # Extract original non-H coordinates for RMSD calculation (with atom IDs)
    original_coords_dict = extract_non_hydrogen_coords_with_ids(output_content)
    if verbose:
        print(f"  Extracted {len(original_coords_dict)} non-hydrogen atoms from original structure")

    # Create temporary protein-only PDB
    temp_protein_pdb = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False)
    temp_protein_pdb.writelines(protein_lines)
    temp_protein_pdb.close()

    try:
        # Load protein-only PDB in PyRosetta (auto-adds hydrogens)
        if verbose:
            print(f"  Loading protein in PyRosetta (hydrogens will be auto-added)...")

        pose = pyrosetta.pose_from_file(temp_protein_pdb.name)

        # Get matcher residues from reference
        if HAS_DESIGN_UTILS:
            try:
                matched_residues = design_utils.get_matcher_residues(ref_pdb_path)
            except:
                matched_residues = get_matcher_residues_from_remark666(ref_pdb_path, verbose)
        else:
            matched_residues = get_matcher_residues_from_remark666(ref_pdb_path, verbose)

        # Build HIS tautomer map from reference PDB
        if matched_residues and verbose:
            print(f"  Found {len(matched_residues)} catalytic residues in REMARK 666")
            print(f"  Building HIS tautomer map from reference PDB...")

        ref_his_map = build_his_tautomer_map_from_raw_pdb(ref_pdb_path, verbose)

        # Fix catalytic residue protonation states using MutateResidue with preserve_atom_coords
        if matched_residues:
            # Strip ligands from reference PDB to load it
            with open(ref_pdb_path, 'r') as f:
                ref_content = f.readlines()

            ref_protein_lines, _ = separate_protein_and_hetatm(ref_content)

            # Create temporary protein-only reference PDB
            temp_ref_protein_pdb = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False)
            temp_ref_protein_pdb.writelines(ref_protein_lines)
            temp_ref_protein_pdb.close()

            try:
                ref_pose = pyrosetta.pose_from_file(temp_ref_protein_pdb.name)

                # Fix catalytic residues using MutateResidue with set_preserve_atom_coords
                for catres_seqpos in matched_residues:
                    # Check if residue exists in both poses
                    if catres_seqpos > pose.size() or catres_seqpos > ref_pose.size():
                        if verbose:
                            print(f"  WARNING: Residue {catres_seqpos} out of range, skipping")
                        continue

                    # Get reference residue info
                    ref_rsd = ref_pose.residue(catres_seqpos)
                    catres_AA = ref_rsd.name()
                    catres_AA3 = ref_rsd.name3()

                    # For HIS, check if we need to override with raw PDB tautomer
                    if catres_AA3 == "HIS":
                        ref_chain = ref_pose.pdb_info().chain(catres_seqpos)
                        ref_pdbno = ref_pose.pdb_info().number(catres_seqpos)
                        key = (ref_chain, ref_pdbno)

                        if key in ref_his_map:
                            raw_his_type = ref_his_map[key]
                            if verbose:
                                print(f"  Catalytic HIS at {ref_chain}{ref_pdbno} (seqpos {catres_seqpos}): using {raw_his_type} from raw PDB")
                            catres_AA = raw_his_type
                        elif ":" in catres_AA:
                            # PyRosetta patched variant - need raw PDB info
                            if verbose:
                                print(f"  WARNING: Catalytic HIS at {ref_chain}{ref_pdbno} has patched type {catres_AA} but not in raw his_map - defaulting to HIS")
                            catres_AA = "HIS"

                    if verbose:
                        print(f"  Fixing catalytic residue {catres_AA3}{catres_seqpos} with reference type {catres_AA}")

                    mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
                    mutres.set_res_name(catres_AA)
                    mutres.set_target(catres_seqpos)
                    mutres.set_preserve_atom_coords(True)  # CRITICAL: preserve coordinates!
                    mutres.apply(pose)

            finally:
                # Clean up temporary reference file
                if os.path.exists(temp_ref_protein_pdb.name):
                    os.unlink(temp_ref_protein_pdb.name)

        # Convert pose to PDB string
        pdb_string = pyrosetta.distributed.io.to_pdbstring(pose)
        protein_with_h_lines = pdb_string.split('\n')

        # Calculate RMSD of non-H atoms to verify coordinates haven't changed
        rosetta_output_lines = [line + '\n' for line in protein_with_h_lines if line.startswith(("ATOM", "HETATM"))]
        rosetta_coords_dict = extract_non_hydrogen_coords_with_ids(rosetta_output_lines)

        if len(original_coords_dict) > 0 and len(rosetta_coords_dict) > 0:
            rmsd, n_matched = calculate_rmsd_from_dicts(original_coords_dict, rosetta_coords_dict)
            if verbose:
                print(f"  Non-hydrogen atom RMSD: {rmsd:.4f} Å ({n_matched} atoms matched)")
            if rmsd > 0.01:  # More than 0.01 Å is suspicious
                print(f"  WARNING: Non-hydrogen RMSD = {rmsd:.4f} Å (coordinates may have changed!)")
        else:
            if verbose:
                print(f"  WARNING: Could not calculate RMSD (original={len(original_coords_dict)}, rosetta={len(rosetta_coords_dict)} atoms)")

        # Build final output with headers + protein with H + ligands
        final_lines = []

        # Add headers from reference (including REMARK 666)
        final_lines.extend(headers_to_add)

        # Add protein with hydrogens (skip any headers that PyRosetta added)
        for line in protein_with_h_lines:
            if line.startswith(("ATOM", "TER")):
                final_lines.append(line + '\n')

        # Add back original HETATM/ligand lines (preserving exact coordinates)
        final_lines.extend(hetatm_lines)

        # Add END if not present
        if not any(line.startswith("END") for line in final_lines):
            final_lines.append("END\n")

        # Renumber all atoms sequentially (ATOM + HETATM)
        if verbose:
            print(f"  Renumbering all atoms sequentially...")
        final_lines = renumber_pdb_atoms(final_lines, start_number=1)

        # Write final output
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
        with open(final_output_path, 'w') as f:
            f.writelines(final_lines)

        if verbose:
            print(f"  SUCCESS: Wrote {final_output_path}")

        return "processed"

    except Exception as e:
        print(f"  ERROR: Failed to process {output_pdb_path}: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return "failed"

    finally:
        # Clean up temporary file
        if os.path.exists(temp_protein_pdb.name):
            os.unlink(temp_protein_pdb.name)


def identify_suffixes(ref_files, output_files, verbose=False, sample_size=None):
    """
    Identify suffixes added to output PDB files based on reference PDB file names.
    Optionally uses a random sample of reference files for suffix identification.

    Args:
        ref_files: List of reference PDB file paths
        output_files: List of output PDB file paths
        verbose: Print verbose output
        sample_size: Optional size of random sample of ref files to use for suffix identification

    Returns:
        tuple: (suffix_map, cumulative_suffixes)
            - suffix_map: Mapping of ref file base name to list of all unique suffixes found
            - cumulative_suffixes: Set of all unique suffixes identified across sampled files
    """
    if sample_size is not None and sample_size < len(ref_files):
        sampled_files = random.sample(ref_files, sample_size)
        if verbose:
            print(f"Using a random sample of {sample_size} reference files for suffix identification.")
    else:
        sampled_files = ref_files

    start_time = time.time()
    suffix_map = {}
    cumulative_suffixes = set()

    # First, discover suffixes from the chosen subset of ref files
    for ref_file in sampled_files:
        ref_base = Path(ref_file).stem
        suffixes_found = []

        for output_file in output_files:
            output_base = Path(output_file).stem
            if output_base.startswith(ref_base):
                suffix = output_base[len(ref_base):]
                suffixes_found.append(suffix)
                cumulative_suffixes.add(suffix)

        suffix_map[ref_base] = suffixes_found

        if verbose:
            print(f"Reference '{ref_base}' matched {len(suffixes_found)} output files with suffixes: {suffixes_found}")

    # Next, map every ref file base (not just the sample) to the set of all discovered suffixes
    for ref_file in ref_files:
        ref_base = Path(ref_file).stem
        suffix_map[ref_base] = list(cumulative_suffixes)

    print(f"Suffix identification completed in {time.time() - start_time:.2f} seconds.")
    print(f"Cumulative unique suffixes found: {sorted(cumulative_suffixes)}")
    print(f"Number of unique suffixes: {len(cumulative_suffixes)}")

    return suffix_map, cumulative_suffixes


def process_directory_mode(
    ref_pdb_dir,
    output_pdb_dir,
    final_output_dir,
    clobber=False,
    verbose=False,
    sample_size=None
):
    """
    Process all PDB files in directory mode with auto-matching by suffix.

    Args:
        ref_pdb_dir: Directory containing reference PDB files
        output_pdb_dir: Directory containing output PDB files to process
        final_output_dir: Directory to write final outputs (or None to overwrite output_pdb_dir)
        clobber: Overwrite existing files
        verbose: Print verbose output
        sample_size: Optional size of random sample for suffix identification
    """
    print("\n" + "="*80)
    print("DIRECTORY MODE: Auto-matching by suffix")
    print("="*80)

    # Use output_pdb_dir as final output if not specified
    if final_output_dir is None:
        final_output_dir = output_pdb_dir
        print("No separate output directory specified - will overwrite input files")

    # Get all PDB files
    ref_files = sorted(glob.glob(os.path.join(ref_pdb_dir, '*.pdb')))
    output_files = sorted(glob.glob(os.path.join(output_pdb_dir, '*.pdb')))

    print(f"\nFound {len(ref_files)} reference PDB files")
    print(f"Found {len(output_files)} output PDB files")

    if not ref_files:
        print("ERROR: No reference PDB files found!")
        return

    if not output_files:
        print("ERROR: No output PDB files found!")
        return

    # Identify suffix mappings
    print("\nIdentifying suffix mappings...")
    suffix_map, cumulative_suffixes = identify_suffixes(ref_files, output_files, verbose, sample_size)

    # Process each reference-output pair
    print("\n" + "-"*80)
    print("Processing PDB files...")
    print("-"*80)

    total_processed = 0
    total_skipped = 0
    total_already_ready = 0

    for ref_file in ref_files:
        ref_base = Path(ref_file).stem
        suffixes = suffix_map.get(ref_base, [])

        if not suffixes:
            print(f"\nWARNING: No output files found for reference '{ref_base}'")
            continue

        # Extract headers from reference (once per reference)
        headers = extract_remark666_and_headers(ref_file, verbose)

        # Process each output file matching this reference
        for suffix in suffixes:
            output_base = ref_base + suffix
            output_file = os.path.join(output_pdb_dir, output_base + '.pdb')
            final_output_file = os.path.join(final_output_dir, output_base + '.pdb')

            if not os.path.exists(output_file):
                print(f"\nWARNING: Expected output file not found: {output_file}")
                total_skipped += 1
                continue

            # Check if this is a ref file copied to output dir (no suffix = empty string)
            if suffix == "" and os.path.basename(output_file) == os.path.basename(ref_file):
                print(f"\n[Check] Verifying copied reference file: {ref_base}")
            else:
                print(f"\n[{total_processed + total_already_ready + 1}] Processing: {ref_base} + suffix '{suffix}'")

            status = add_hydrogens_and_fix_catalytic_residues(
                output_pdb_path=output_file,
                ref_pdb_path=ref_file,
                headers_to_add=headers,
                final_output_path=final_output_file,
                clobber=clobber,
                verbose=verbose
            )

            if status == "processed":
                total_processed += 1
            elif status == "skipped_ready":
                total_already_ready += 1
            else:  # "failed"
                total_skipped += 1

    print("\n" + "="*80)
    print(f"SUMMARY:")
    print(f"  Processed: {total_processed} files")
    print(f"  Already ready (skipped): {total_already_ready} files")
    print(f"  Skipped (errors): {total_skipped} files")
    print(f"  Total: {total_processed + total_already_ready + total_skipped} files")
    print(f"\n  Unique suffixes found: {sorted(cumulative_suffixes)}")
    print(f"  Number of unique suffixes: {len(cumulative_suffixes)}")
    print("="*80)


def process_single_file_mode(
    ref_pdb,
    output_pdb,
    final_output_dir,
    clobber=False,
    verbose=False
):
    """
    Process a single reference-output PDB pair.

    Args:
        ref_pdb: Path to reference PDB file
        output_pdb: Path to output PDB file to process
        final_output_dir: Directory to write final output (or None to overwrite output_pdb)
        clobber: Overwrite existing files
        verbose: Print verbose output
    """
    print("\n" + "="*80)
    print("SINGLE FILE MODE")
    print("="*80)

    if not os.path.exists(ref_pdb):
        print(f"ERROR: Reference PDB not found: {ref_pdb}")
        return

    if not os.path.exists(output_pdb):
        print(f"ERROR: Output PDB not found: {output_pdb}")
        return

    # Determine final output path
    if final_output_dir is None:
        final_output_path = output_pdb
        print("No separate output directory specified - will overwrite input file")
    else:
        os.makedirs(final_output_dir, exist_ok=True)
        final_output_path = os.path.join(final_output_dir, os.path.basename(output_pdb))

    # Check if files are identical (reference copied to output)
    if os.path.basename(ref_pdb) == os.path.basename(output_pdb):
        print("\nNOTE: Reference and output have same filename - checking if already processed...")

    # Extract headers from reference
    print("\nExtracting headers from reference PDB...")
    headers = extract_remark666_and_headers(ref_pdb, verbose)

    # Process the file
    print("\nProcessing PDB file...")
    status = add_hydrogens_and_fix_catalytic_residues(
        output_pdb_path=output_pdb,
        ref_pdb_path=ref_pdb,
        headers_to_add=headers,
        final_output_path=final_output_path,
        clobber=clobber,
        verbose=verbose
    )

    print("\n" + "="*80)
    if status == "processed":
        print("SUCCESS: File processed successfully")
    elif status == "skipped_ready":
        print("SKIPPED: File already has REMARK 666 and hydrogens")
    else:  # "failed"
        print("FAILED: File processing failed")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Add REMARK 666 lines and Rosetta hydrogens to PDB files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Directory mode with auto-matching
  %(prog)s --ref_pdb_dir /path/to/refs --output_pdb_dir /path/to/outputs --final_output_dir /path/to/final

  # Directory mode overwriting inputs
  %(prog)s --ref_pdb_dir /path/to/refs --output_pdb_dir /path/to/outputs --clobber

  # Single file mode
  %(prog)s --ref_pdb ref.pdb --output_pdb output.pdb --final_output_dir /path/to/final
        """
    )

    # Directory mode arguments
    parser.add_argument(
        '--ref_pdb_dir',
        type=str,
        help='Directory containing reference PDB files (for directory mode)'
    )
    parser.add_argument(
        '--output_pdb_dir',
        type=str,
        help='Directory containing output PDB files to process (for directory mode)'
    )

    # Single file mode arguments
    parser.add_argument(
        '--ref_pdb',
        type=str,
        help='Single reference PDB file (for single file mode)'
    )
    parser.add_argument(
        '--output_pdb',
        type=str,
        help='Single output PDB file to process (for single file mode)'
    )

    # Common arguments
    parser.add_argument(
        '--final_output_dir',
        type=str,
        default=None,
        help='Directory to write final output files. If not specified, overwrites output PDB files.'
    )
    parser.add_argument(
        '--clobber',
        action='store_true',
        default=False,
        help='Overwrite existing output files'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='Print verbose output'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='Alias for --verbose'
    )
    parser.add_argument(
        '--find_suffixes_from_random_sample',
        type=int,
        default=None,
        help='Optionally find suffixes from a random sample of reference PDB files. Provide an integer to specify sample size.'
    )

    args = parser.parse_args()

    # Set verbose from debug flag
    if args.debug:
        args.verbose = True

    # Determine mode
    directory_mode = args.ref_pdb_dir is not None and args.output_pdb_dir is not None
    single_file_mode = args.ref_pdb is not None and args.output_pdb is not None

    if not directory_mode and not single_file_mode:
        parser.error("Must specify either:\n"
                    "  - Directory mode: --ref_pdb_dir and --output_pdb_dir\n"
                    "  - Single file mode: --ref_pdb and --output_pdb")

    if directory_mode and single_file_mode:
        parser.error("Cannot specify both directory mode and single file mode arguments")

    # Initialize PyRosetta once at start (efficient for batch processing)
    print("\n" + "="*80)
    print("Initializing PyRosetta...")
    print("="*80)
    start_time = time.time()

    # Initialize with preserve_header to keep original headers
    pyrosetta.init("-mute all -run:preserve_header")

    init_time = time.time() - start_time
    print(f"PyRosetta initialized in {init_time:.2f} seconds")

    # Run appropriate mode
    overall_start = time.time()

    if directory_mode:
        process_directory_mode(
            ref_pdb_dir=args.ref_pdb_dir,
            output_pdb_dir=args.output_pdb_dir,
            final_output_dir=args.final_output_dir,
            clobber=args.clobber,
            verbose=args.verbose,
            sample_size=args.find_suffixes_from_random_sample
        )
    else:
        process_single_file_mode(
            ref_pdb=args.ref_pdb,
            output_pdb=args.output_pdb,
            final_output_dir=args.final_output_dir,
            clobber=args.clobber,
            verbose=args.verbose
        )

    total_time = time.time() - overall_start
    print(f"\nTotal execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
