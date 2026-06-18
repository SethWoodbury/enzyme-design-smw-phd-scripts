#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rosetta Utility Functions for Enzyme Design

Consolidated utilities for PyRosetta operations including:
- Pose manipulation (mutations, threading, ligand handling)
- Layer-based residue selection
- FastRelax setup
- Scoring and energy calculations
- No-ligand-repack analysis

Original authors: Indrek Kalvet, Chris Norn
Refactored and consolidated: 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import pyrosetta as pyr
import pyrosetta.rosetta
import pyrosetta.distributed.io
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector
from pyrosetta.rosetta.core.select.residue_selector import NotResidueSelector
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover


# =============================================================================
# Constants
# =============================================================================

AA_3_TO_1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"
}

AA_1_TO_3 = {v: k for k, v in AA_3_TO_1.items()}

COMPARISONS = {
    '<=': '__le__',
    '<': '__lt__',
    '>': '__gt__',
    '>=': '__ge__',
    '=': '__eq__'
}


# =============================================================================
# Scorefunction Utilities
# =============================================================================

def create_scorefunction(name="ref2015", verbose=True):
    """
    Create a PyRosetta scorefunction by name.

    Arguments:
        name: Scorefunction name. Options:
            - "ref2015" (default): REF2015 scorefunction
            - "ref2015_cart": REF2015 optimized for Cartesian minimization
            - "beta_nov16": Explicit beta_nov16 weights
            - "beta_jan25": Newer beta scorefunction (requires -corrections:beta_jan25 in init)
            - "beta_july15": Older beta scorefunction (requires -corrections:beta_july15 in init)
            - "talaris2014": Older Talaris scorefunction
            - Custom weights file path
        verbose: Print scorefunction info

    Note:
        Beta scorefunctions (beta_jan25, beta_july15, etc.) require the appropriate
        -corrections: flag to be set during pyrosetta.init() BEFORE calling this function.
        The enzyme_design.py script handles this automatically when --scorefunction is set.

    Returns:
        ScoreFunction: PyRosetta ScoreFunction object
    """
    # Standard scorefunctions that don't require special init flags
    standard_scorefunctions = ["ref2015", "ref2015_cart", "beta_nov16", "talaris2014", "score12"]

    # Beta scorefunctions that require -corrections: flags during init
    beta_scorefunctions = ["beta_jan25", "beta_july15"]

    known_scorefunctions = standard_scorefunctions + beta_scorefunctions

    if name in known_scorefunctions:
        sfx = pyrosetta.rosetta.core.scoring.ScoreFunctionFactory.create_score_function(name)
        if verbose:
            if name in beta_scorefunctions:
                print(f"  [Scorefunction] Created: {name} (beta scorefunction)")
            else:
                print(f"  [Scorefunction] Created: {name}")
    elif name == "default" or name is None:
        sfx = pyr.get_fa_scorefxn()
        if verbose:
            print(f"  [Scorefunction] Created: default (REF2015 with init flags)")
    else:
        # Assume it's a custom weights file path
        try:
            sfx = pyrosetta.rosetta.core.scoring.ScoreFunctionFactory.create_score_function(name)
            if verbose:
                print(f"  [Scorefunction] Created from weights file: {name}")
        except Exception as e:
            print(f"  [Scorefunction] Warning: Could not create scorefunction '{name}': {e}")
            print(f"  [Scorefunction] Falling back to default REF2015")
            sfx = pyr.get_fa_scorefxn()

    return sfx


def add_constraint_weights(sfx, atom_pair=1.0, angle=1.0, dihedral=1.0, verbose=True):
    """
    Add enzyme design constraint weights to a scorefunction.

    Arguments:
        sfx: ScoreFunction to modify (in place)
        atom_pair: Weight for atom_pair_constraint
        angle: Weight for angle_constraint
        dihedral: Weight for dihedral_constraint
        verbose: Print weight info

    Returns:
        ScoreFunction: The modified scorefunction (same object)
    """
    sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("atom_pair_constraint"), atom_pair)
    sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("angle_constraint"), angle)
    sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("dihedral_constraint"), dihedral)

    if verbose:
        print(f"  [Scorefunction] Constraint weights: atom_pair={atom_pair}, angle={angle}, dihedral={dihedral}")

    return sfx


def count_cst_blocks(cst_file: str) -> int:
    """
    Count the number of constraint blocks in a Rosetta CST file.

    A block is defined by VARIABLE_CST::BEGIN ... VARIABLE_CST::END

    Args:
        cst_file: Path to the CST file

    Returns:
        int: Number of constraint blocks found
    """
    if not os.path.exists(cst_file):
        return 0

    with open(cst_file, 'r') as f:
        content = f.read()

    # Count VARIABLE_CST::BEGIN occurrences (each marks start of a block)
    block_count = content.count("VARIABLE_CST::BEGIN")

    return block_count


def validate_catres_cst_subset(catres_subset: list, cst_file: str, verbose: bool = True) -> list:
    """
    Validate and filter catres_cst_subset to only include blocks that exist in the CST file.

    The REMARK 666 lines in a PDB reference constraint blocks by number (1-indexed).
    If a block number in catres_cst_subset exceeds the number of blocks in the CST file,
    Rosetta will fail with an error.

    Args:
        catres_subset: List of REMARK 666 line numbers (1-indexed) to consider
        cst_file: Path to the CST file
        verbose: Print warnings about filtered blocks

    Returns:
        list: Filtered list containing only valid block numbers
    """
    if not cst_file or not os.path.exists(cst_file):
        return catres_subset

    num_blocks = count_cst_blocks(cst_file)

    if num_blocks == 0:
        if verbose:
            print(f"  [CST] Warning: No constraint blocks found in {cst_file}")
        return catres_subset

    valid_subset = []
    invalid_blocks = []

    for block_num in catres_subset:
        if block_num <= num_blocks:
            valid_subset.append(block_num)
        else:
            invalid_blocks.append(block_num)

    if invalid_blocks and verbose:
        print(f"  [CST] Warning: CST file has {num_blocks} blocks, but catres_cst_subset includes blocks: {invalid_blocks}")
        print(f"  [CST] These blocks will be excluded from constraint application (no constraints for those REMARK 666 lines)")
        print(f"  [CST] Valid blocks: {valid_subset}")

    return valid_subset


def filter_pdb_remark666_by_cst_blocks(pdb_path: str, cst_file: str, output_path: str = None,
                                        verbose: bool = True) -> str:
    """
    Filter a PDB file to remove REMARK 666 lines that reference non-existent CST blocks.

    When Rosetta's EnzConstraintIO reads a PDB, it looks at ALL REMARK 666 lines and tries
    to find matching constraint blocks. If a REMARK 666 references block N but only M < N
    blocks exist, Rosetta fails. This function removes such invalid REMARK 666 lines.

    Args:
        pdb_path: Path to the input PDB file
        cst_file: Path to the CST file
        output_path: Path for filtered PDB (default: creates temp file)
        verbose: Print info about filtered lines

    Returns:
        str: Path to the filtered PDB file
    """
    import tempfile

    num_blocks = count_cst_blocks(cst_file)

    if num_blocks == 0:
        if verbose:
            print(f"  [CST] Warning: No constraint blocks found in {cst_file}")
        return pdb_path

    with open(pdb_path, 'r') as f:
        lines = f.readlines()

    filtered_lines = []
    removed_count = 0
    removed_blocks = set()

    for line in lines:
        if line.startswith("REMARK 666"):
            # Parse the constraint block number (field 12, 0-indexed = 12th field)
            # Format: REMARK 666 MATCH TEMPLATE X RES N MATCH MOTIF Y RES M cst_no cst_var
            parts = line.split()
            if len(parts) >= 13:
                try:
                    cst_block = int(parts[12])
                    if cst_block > num_blocks:
                        removed_count += 1
                        removed_blocks.add(cst_block)
                        continue  # Skip this line
                except ValueError:
                    pass  # Keep line if we can't parse it
        filtered_lines.append(line)

    if removed_count > 0:
        if verbose:
            print(f"  [CST] Filtered {removed_count} REMARK 666 lines referencing non-existent blocks: {sorted(removed_blocks)}")
            print(f"  [CST] CST file has {num_blocks} blocks; lines referencing blocks > {num_blocks} were removed")

        # Write filtered PDB
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix='.pdb', prefix='filtered_')
            os.close(fd)

        with open(output_path, 'w') as f:
            f.writelines(filtered_lines)

        return output_path
    else:
        # No filtering needed
        return pdb_path


def filter_pose_remark666_by_cst_blocks(pose, cst_file: str, verbose: bool = True):
    """
    Create a new pose from an existing pose with invalid REMARK 666 lines removed.

    Args:
        pose: PyRosetta Pose object
        cst_file: Path to the CST file
        verbose: Print info about filtered lines

    Returns:
        Pose: New pose with filtered REMARK 666 lines
    """
    import tempfile

    # Export pose to PDB string
    pdb_string = pyr.distributed.io.to_pdbstring(pose)

    num_blocks = count_cst_blocks(cst_file)

    if num_blocks == 0:
        return pose

    lines = pdb_string.split('\n')
    filtered_lines = []
    removed_count = 0
    removed_blocks = set()

    for line in lines:
        if line.startswith("REMARK 666"):
            parts = line.split()
            if len(parts) >= 13:
                try:
                    cst_block = int(parts[12])
                    if cst_block > num_blocks:
                        removed_count += 1
                        removed_blocks.add(cst_block)
                        continue
                except ValueError:
                    pass
        filtered_lines.append(line)

    if removed_count > 0:
        if verbose:
            print(f"  [CST] Filtered {removed_count} REMARK 666 lines from pose referencing blocks: {sorted(removed_blocks)}")

        # Create new pose from filtered PDB string
        filtered_pdb = '\n'.join(filtered_lines)

        # Write to temp file and load
        fd, tmp_path = tempfile.mkstemp(suffix='.pdb', prefix='filtered_pose_')
        os.close(fd)
        try:
            with open(tmp_path, 'w') as f:
                f.write(filtered_pdb)
            new_pose = pyr.pose_from_file(tmp_path)
            return new_pose
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        return pose


def configure_for_cartesian(sfx, cart_bonded=0.5, pro_close=0.0, verbose=True):
    """
    Configure a scorefunction for Cartesian minimization.

    Arguments:
        sfx: ScoreFunction to modify (in place)
        cart_bonded: Weight for cart_bonded term (required for Cartesian)
        pro_close: Weight for pro_close term (must be 0 for Cartesian)
        verbose: Print configuration info

    Returns:
        ScoreFunction: The modified scorefunction (same object)
    """
    sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("cart_bonded"), cart_bonded)
    sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("pro_close"), pro_close)

    if verbose:
        print(f"  [Scorefunction] Configured for Cartesian: cart_bonded={cart_bonded}, pro_close={pro_close}")

    return sfx


def get_scorefunction_info(sfx):
    """
    Get information about a scorefunction's non-zero weights.

    Arguments:
        sfx: ScoreFunction

    Returns:
        dict: {score_term: weight} for all non-zero weights
    """
    weights = {}
    for st in pyrosetta.rosetta.core.scoring.ScoreType.__members__.values():
        w = sfx.get_weight(st)
        if w != 0.0:
            weights[st.name] = w
    return weights


def print_scorefunction_info(sfx, title="Scorefunction Weights"):
    """Print non-zero scorefunction weights."""
    print(f"\n  [{title}]")
    weights = get_scorefunction_info(sfx)
    for term, weight in sorted(weights.items()):
        print(f"    {term}: {weight:.4f}")


# =============================================================================
# Amino Acid Utilities
# =============================================================================

def aa321(aa3):
    """Convert 3-letter amino acid code to 1-letter code."""
    return AA_3_TO_1.get(aa3, "X")


def aa123(aa1):
    """Convert 1-letter amino acid code to 3-letter code."""
    return AA_1_TO_3.get(aa1, "UNK")


# =============================================================================
# Pose Information Utilities
# =============================================================================

def get_matcher_residues(filename):
    """
    Returns a dictionary with REMARK 666 contents (matcher/catalytic residues).

    Arguments:
        filename: str (path to PDB) or pyrosetta.rosetta.core.pose.Pose

    Returns:
        dict: {resno: {'target_name', 'target_chain', 'target_resno', 'chain', 'name3', 'cst_no', 'cst_no_var'}}
    """
    if isinstance(filename, str):
        pdbfile = open(filename, 'r').readlines()
    elif isinstance(filename, pyrosetta.rosetta.core.pose.Pose):
        pdbfile = pyrosetta.distributed.io.to_pdbstring(filename).split("\n")
    else:
        raise TypeError(f"Expected str or Pose, got {type(filename)}")

    matches = {}
    for line in pdbfile:
        if "ATOM" in line:
            break
        if "REMARK 666" in line:
            lspl = line.split()
            resno = int(lspl[11])
            matches[resno] = {
                'target_name': lspl[5],
                'target_chain': lspl[4],
                'target_resno': int(lspl[6]),
                'chain': lspl[9],
                'name3': lspl[10],
                'cst_no': int(lspl[12]),
                'cst_no_var': int(lspl[13])
            }
    return matches


def get_ligand_heavyatoms(pose, resno=None):
    """
    Returns a list of non-hydrogen atom names for a ligand residue.

    Arguments:
        pose: PyRosetta Pose object
        resno: Residue number (default: last residue)

    Returns:
        list: Atom names of heavy atoms
    """
    if resno is None:
        resno = pose.size()
    if not pose.residue(resno).is_ligand():
        print(f"Warning: Residue {resno} is not a ligand!")
        return None

    heavyatoms = []
    res = pose.residue(resno)
    for n in range(1, res.natoms() + 1):
        element = res.atom_type(n).element()
        if element != 'H':
            heavyatoms.append(res.atom_name(n).strip())
    return heavyatoms


# =============================================================================
# Layer Selection Functions
# =============================================================================

def get_packer_layers(pose, ref_resno, cuts, target_atoms, do_not_design=None, allow_design=None, design_GP=False):
    """
    Finds residues within certain distances from target atoms based on CA distances.

    Arguments:
        pose: PyRosetta Pose object
        ref_resno: Residue number around which layers are centered
        cuts: List of 4 floats defining CA distance cutoffs
        target_atoms: List of atom names or dict {resno: [atom_names]}
        do_not_design: List of residue numbers to exclude from design
        allow_design: List of residue numbers to force into design
        design_GP: If True, allows GLY/PRO redesign

    Returns:
        list: Lists of residue numbers for each layer, last list contains remaining residues
    """
    assert len(cuts) >= 4, f"Need at least 4 layer cut distances, got {cuts}"

    if do_not_design is None:
        do_not_design = []
    if allow_design is None:
        allow_design = []

    KEEP_RES = [] if design_GP else ["GLY", "PRO"]

    # Get ligand atom coordinates
    ligand_atoms = []
    if isinstance(target_atoms, list):
        for a in target_atoms:
            ligand_atoms.append(pose.residue(ref_resno).xyz(a))
    elif isinstance(target_atoms, dict):
        for k in target_atoms:
            for a in target_atoms[k]:
                ligand_atoms.append(pose.residue(k).xyz(a))

    residues = [[] for _ in cuts] + [[]]

    for resno in range(1, pose.size()):
        res = pose.residue(resno)
        if res.is_ligand() or res.is_virtual_residue():
            continue

        if resno in do_not_design:
            residues[2].append(resno)
            continue
        if resno in allow_design:
            residues[0].append(resno)
            continue

        resname = res.name3()
        CA = res.xyz('CA')
        CA_distances = [((a - CA).norm()) for a in ligand_atoms]
        CA_mindist = min(CA_distances)

        CB_mindist = float('inf')
        if resname != "GLY":
            CB = res.xyz('CB')
            CB_distances = [((a - CB).norm()) for a in ligand_atoms]
            CB_mindist = min(CB_distances)

        # Assign to layer based on distance
        if CA_mindist <= cuts[0] and resname not in KEEP_RES:
            residues[0].append(resno)
        elif CA_mindist <= cuts[1] and resname not in KEEP_RES:
            if resname == "GLY" or CB_mindist < CA_mindist or (CA_mindist < cuts[1]-1.0 and CB_mindist < cuts[1]-1.0):
                residues[1].append(resno)
            else:
                residues[2].append(resno)
        elif CA_mindist <= cuts[2]:
            residues[2].append(resno)
        elif CA_mindist <= cuts[3] and resname not in KEEP_RES:
            if resname == "GLY" or CB_mindist < CA_mindist:
                residues[3].append(resno)
            else:
                residues[-1].append(resno)
        else:
            residues[-1].append(resno)

    return residues


def get_layer_selections(pose, repack_only_pos, design_pos, ref_resno, heavyatoms, cuts=[6.0, 8.0, 10.0, 12.0], design_GP=False):
    """
    Gets ResidueSelectors for design, repack, and do-not-touch layers.

    Arguments:
        pose: PyRosetta Pose object
        repack_only_pos: List of positions that should only be repacked
        design_pos: List of positions that must be redesigned
        ref_resno: Reference residue number (typically ligand)
        heavyatoms: List of heavy atom names
        cuts: Distance cutoffs for layers [6.0, 8.0, 10.0, 12.0]
        design_GP: If True, allows GLY/PRO redesign

    Returns:
        tuple: (SEL_mutate, SEL_repack, SEL_do_not_repack, residues_list)
    """
    residues = get_packer_layers(pose, ref_resno, cuts, heavyatoms, repack_only_pos, design_pos, design_GP)

    SEL_repack_residues = ResidueIndexSelector()
    for res in residues[2] + residues[3]:
        SEL_repack_residues.append_index(res)

    SEL_do_not_repack = ResidueIndexSelector()
    for res in residues[4]:
        SEL_do_not_repack.append_index(res)

    SEL_mutate_residues = ResidueIndexSelector()
    for res in residues[0] + residues[1]:
        SEL_mutate_residues.append_index(res)

    return SEL_mutate_residues, SEL_repack_residues, SEL_do_not_repack, residues


def get_residues_with_close_sc(pose, ref_atoms, residues=None, exclude_residues=None, cutoff=4.5, ref_seqpos=None):
    """
    Find residues with side-chain atoms close to reference atoms.

    Arguments:
        pose: PyRosetta Pose
        ref_atoms: List of reference atom names
        residues: List of residue numbers to check (default: all)
        exclude_residues: List of residue numbers to exclude
        cutoff: Distance cutoff in Angstroms
        ref_seqpos: Reference residue position (default: last residue)

    Returns:
        list: Residue numbers with close side-chains
    """
    if residues is None:
        residues = list(range(1, pose.size() + 1))
    if exclude_residues is None:
        exclude_residues = []
    if ref_seqpos is None:
        ref_seqpos = pose.size()

    close_ones = []
    for resno in residues:
        if resno in exclude_residues or pose.residue(resno).is_ligand():
            continue

        res = pose.residue(resno)
        for atomno in range(1, res.natoms()):
            if res.atom_type(atomno).is_heavyatom():
                for ha in ref_atoms:
                    if (res.xyz(atomno) - pose.residue(ref_seqpos).xyz(ha)).norm() < cutoff:
                        close_ones.append(resno)
                        break
                else:
                    continue
                break
    return close_ones


# =============================================================================
# Clash Detection and Mutation Functions
# =============================================================================

def find_clashes_between_target_and_sidechains(pose, target_resno, target_atoms=None, residues=None, clash_dist=2.5):
    """
    Find residues with atoms clashing with target residue.

    Arguments:
        pose: PyRosetta Pose
        target_resno: Target residue number
        target_atoms: List of atom numbers to check (default: all heavy atoms)
        residues: List of residue numbers to check
        clash_dist: Clash distance threshold

    Returns:
        list: Residue numbers that clash
    """
    if target_atoms is None:
        res = pose.residue(target_resno)
        target_atoms = [res.atom_name(n).strip() for n in range(1, res.natoms() + 1)]

    clashes = []
    for res in pose.residues:
        if residues is not None and res.seqpos() not in residues:
            continue
        if not res.is_protein():
            continue

        for ano in range(1, res.natoms() + 1):
            if res.is_virtual(ano) or res.atom_is_hydrogen(ano):
                continue
            min_dist = min([(pose.residue(target_resno).xyz(ha) - res.xyz(ano)).norm() for ha in target_atoms])
            if min_dist < clash_dist:
                clashes.append(res.seqpos())
                break

    return clashes


def find_backbone_clashes_with_ligand(pose, ligand_resno=None, clash_dist=2.5, neighbor_window=3):
    """
    Find residues whose backbone atoms (N, CA, C, O) or CB clash with ligand atoms.
    Also returns neighboring residues (±N) that should be freed.

    Arguments:
        pose: PyRosetta Pose
        ligand_resno: Ligand residue number (default: last residue)
        clash_dist: Clash distance threshold in Angstroms
        neighbor_window: Number of neighboring residues (±N) to also free

    Returns:
        tuple: (clashing_residues, all_residues_to_free)
               clashing_residues: List of residues with backbone/CB clashes
               all_residues_to_free: List including neighbors
    """
    if ligand_resno is None:
        ligand_resno = pose.size()

    ligand = pose.residue(ligand_resno)
    assert ligand.is_ligand(), f"Residue {ligand_resno} is not a ligand!"

    # Get ligand heavy atom coordinates
    ligand_atoms = []
    for n in range(1, ligand.natoms() + 1):
        if not ligand.atom_is_hydrogen(n):
            ligand_atoms.append(ligand.xyz(n))

    # Backbone atoms to check
    bb_atoms = ['N', 'CA', 'C', 'O']

    clashing_residues = []
    for resno in range(1, pose.size() + 1):
        res = pose.residue(resno)
        if not res.is_protein():
            continue

        # Check backbone atoms
        has_clash = False
        for atom_name in bb_atoms:
            if res.has(atom_name):
                res_xyz = res.xyz(atom_name)
                for lig_xyz in ligand_atoms:
                    if (res_xyz - lig_xyz).norm() < clash_dist:
                        has_clash = True
                        break
            if has_clash:
                break

        # Check CB if present and no clash yet
        if not has_clash and res.name3() != "GLY" and res.has("CB"):
            cb_xyz = res.xyz("CB")
            for lig_xyz in ligand_atoms:
                if (cb_xyz - lig_xyz).norm() < clash_dist:
                    has_clash = True
                    break

        if has_clash:
            clashing_residues.append(resno)

    # Add neighbors (respecting chain boundaries)
    all_residues_to_free = set(clashing_residues)
    for resno in clashing_residues:
        res_chain = pose.residue(resno).chain()
        for offset in range(-neighbor_window, neighbor_window + 1):
            neighbor_resno = resno + offset
            if 1 <= neighbor_resno <= pose.size():
                neighbor_res = pose.residue(neighbor_resno)
                if neighbor_res.chain() == res_chain and neighbor_res.is_protein():
                    all_residues_to_free.add(neighbor_resno)

    return clashing_residues, sorted(all_residues_to_free)


def mutate_residues(pose, resnos, resname3):
    """
    Mutate given residues to specified amino acid.

    Arguments:
        pose: PyRosetta Pose
        resnos: List of residue numbers
        resname3: 3-letter amino acid code

    Returns:
        Pose: New pose with mutations
    """
    assert isinstance(resnos, list), "resnos must be a list"

    pose2 = pose.clone()
    mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
    for res in resnos:
        mutres.set_target(res)
        mutres.set_res_name(resname3)
        mutres.apply(pose2)
    return pose2


def thread_seq_to_pose(pose, sequence):
    """
    Thread a sequence onto a pose, mutating residues as needed.

    Arguments:
        pose: PyRosetta Pose
        sequence: String of 1-letter amino acid codes

    Returns:
        Pose: New pose with threaded sequence
    """
    pose2 = pose.clone()
    mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()

    for i, r in enumerate(sequence):
        if pose.residue(i + 1).name1() == r:
            continue
        mutres.set_target(i + 1)
        mutres.set_res_name(aa123(r))
        mutres.apply(pose2)

    return pose2


def fix_catalytic_residue_rotamers(pose, ref_pose, catalytic_residues):
    """
    Fix catalytic residue rotamers by replacing with reference rotamers.

    Arguments:
        pose: Pose to fix
        ref_pose: Reference pose with correct rotamers
        catalytic_residues: List of catalytic residue numbers

    Returns:
        Pose: Fixed pose
    """
    _pose = pose.clone()
    mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()

    for resno in catalytic_residues:
        if ref_pose.residue(resno).name() != _pose.residue(resno).name():
            print(f"  Fixing catalytic residue {_pose.residue(resno).name()}-{resno} -> {ref_pose.residue(resno).name()}")
            mutres.set_target(resno)
            mutres.set_res_name(ref_pose.residue(resno).name())
            mutres.apply(_pose)

    return _pose


# =============================================================================
# Packing and Relaxation Functions
# =============================================================================

def repack(pose, scorefxn):
    """
    Repack side-chains of a pose.

    Arguments:
        pose: PyRosetta Pose
        scorefxn: PyRosetta ScoreFunction

    Returns:
        Pose: Repacked pose
    """
    tmp_pose = pose.clone()

    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())

    e = pyrosetta.rosetta.core.pack.task.operation.ExtraRotamersGeneric()
    e.ex1(False)
    e.ex2(False)
    e.ex1aro(False)
    tf.push_back(e)
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())

    task = tf.create_task_and_apply_taskoperations(tmp_pose)
    pack_mover = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn, task)
    pack_mover.apply(tmp_pose)

    return tmp_pose


def _get_crude_fastrelax_script(coord_cst_weight=0.9):
    """
    Returns crude FastRelax script lines.

    Arguments:
        coord_cst_weight: Initial coordinate constraint weight (default: 0.9)
    """
    return [
        f"coord_cst_weight {coord_cst_weight}",
        "scale:fa_rep 0.1",
        "repack",
        "scale:fa_rep 0.280",
        "min 0.1",
        f"coord_cst_weight {coord_cst_weight * 0.5}",
        "scale:fa_rep 0.3",
        "repack",
        "scale:fa_rep 0.6",
        "min 0.05",
        "coord_cst_weight 0.0",
        "scale:fa_rep 1",
        "repack",
        "min 0.01",
        "accept_to_best"
    ]


def setup_fastrelax(sfx, crude=False, disable_min_resons=None, pose=None, ligand_rigidity="fixed",
                    coord_cst_weight=0.9):
    """
    Set up FastRelax mover with appropriate settings.

    Arguments:
        sfx: ScoreFunction
        crude: If True, use crude/fast relax script
        disable_min_resons: List of residue numbers to exclude from minimization
        pose: PyRosetta Pose (required if ligand_rigidity != "flexible")
        ligand_rigidity: One of "fixed", "rigid_body", "flexible"
            - "fixed": No ligand movement at all (DEFAULT)
            - "rigid_body": Ligands internally rigid, can move as rigid bodies
            - "flexible": Full ligand flexibility
        coord_cst_weight: Initial coordinate constraint weight (default: 0.9)

    Returns:
        FastRelax: Configured FastRelax mover
    """
    fastRelax = pyrosetta.rosetta.protocols.relax.FastRelax(sfx, 1)

    if crude:
        script = _get_crude_fastrelax_script(coord_cst_weight=coord_cst_weight)
        filelines = pyrosetta.rosetta.std.vector_std_string()
        for l in script:
            filelines.append(l.rstrip())
        fastRelax.set_script_from_lines(filelines)

    fastRelax.constrain_relax_to_start_coords(True)

    # Set up TaskFactory
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())

    e = pyrosetta.rosetta.core.pack.task.operation.ExtraRotamersGeneric()
    if crude:
        e.ex1(False)
        e.ex1aro(False)
        e.ex2(False)
    else:
        e.ex1(True)
        e.ex1aro(True)
        e.ex2(True)
        e.ex1_sample_level(pyrosetta.rosetta.core.pack.task.ExtraRotSample(1))
    tf.push_back(e)
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())
    fastRelax.set_task_factory(tf)

    # Set up MoveMap
    mm = pyrosetta.rosetta.core.kinematics.MoveMap()
    mm.set_chi(True)
    mm.set_bb(True)
    mm.set_jump(True)

    if disable_min_resons is not None:
        for resno in disable_min_resons:
            mm.set_chi(resno, False)
            mm.set_bb(resno, False)

    # Configure ligand rigidity
    if pose is not None and ligand_rigidity != "flexible":
        configure_movemap_ligand_rigidity(mm, pose, ligand_rigidity, verbose=True)

    fastRelax.set_movemap(mm)
    return fastRelax


def create_minimizer_movemap(pose=None, allow_chi=True, allow_bb=True, allow_jump=True,
                             disable_residues=None, ligand_rigidity="fixed"):
    """
    Create a MoveMap with ligand rigidity configuration.

    Arguments:
        pose: PyRosetta Pose (required for ligand rigidity configuration)
        allow_chi: Allow chi angle changes (default: True)
        allow_bb: Allow backbone changes (default: True)
        allow_jump: Allow jump/rigid-body changes (default: True)
        disable_residues: List of residue numbers to disable
        ligand_rigidity: One of "fixed", "rigid_body", "flexible"

    Returns:
        MoveMap: Configured MoveMap
    """
    mm = pyrosetta.rosetta.core.kinematics.MoveMap()
    mm.set_chi(allow_chi)
    mm.set_bb(allow_bb)
    mm.set_jump(allow_jump)

    if disable_residues is not None:
        for resno in disable_residues:
            mm.set_chi(resno, False)
            mm.set_bb(resno, False)

    # Configure ligand rigidity
    if pose is not None and ligand_rigidity != "flexible":
        configure_movemap_ligand_rigidity(mm, pose, ligand_rigidity, verbose=True)

    return mm


# =============================================================================
# Ligand Rigidity Functions
# =============================================================================

def get_ligand_residues(pose):
    """
    Get all ligand residue sequence positions.

    Arguments:
        pose: PyRosetta Pose

    Returns:
        list: Ligand residue sequence positions
    """
    return [res.seqpos() for res in pose.residues if res.is_ligand()]


def get_ligand_jumps(pose):
    """
    Get jump numbers that build ligand residues.

    Arguments:
        pose: PyRosetta Pose

    Returns:
        list: Jump numbers for ligands
    """
    ft = pose.fold_tree()
    ligand_jumps = []
    for res in pose.residues:
        if res.is_ligand():
            jump_no = ft.get_jump_that_builds_residue(res.seqpos())
            if jump_no > 0 and jump_no not in ligand_jumps:
                ligand_jumps.append(jump_no)
    return ligand_jumps


def configure_movemap_ligand_rigidity(movemap, pose, ligand_rigidity="fixed", verbose=True):
    """
    Configure MoveMap for ligand rigidity.

    Arguments:
        movemap: PyRosetta MoveMap to modify (in place)
        pose: PyRosetta Pose
        ligand_rigidity: One of "fixed", "rigid_body", "flexible"
            - "fixed": No internal or rigid-body movement (DEFAULT)
            - "rigid_body": Internal rigid, rigid-body movement allowed
            - "flexible": Full flexibility
        verbose: Print configuration info

    Returns:
        list: Ligand residue positions (for reference)
    """
    valid_modes = ["fixed", "rigid_body", "flexible"]
    if ligand_rigidity not in valid_modes:
        raise ValueError(f"ligand_rigidity must be one of {valid_modes}, got '{ligand_rigidity}'")

    ligand_resnos = get_ligand_residues(pose)
    ligand_jumps = get_ligand_jumps(pose)

    if verbose:
        print(f"  [Ligand Rigidity] Mode: {ligand_rigidity}")
        print(f"  [Ligand Rigidity] Ligand residues: {ligand_resnos}")
        print(f"  [Ligand Rigidity] Ligand jumps: {ligand_jumps}")

    if ligand_rigidity == "fixed":
        # Disable all ligand internal DOFs and jumps
        for resno in ligand_resnos:
            movemap.set_chi(resno, False)
            movemap.set_bb(resno, False)
        for jump_no in ligand_jumps:
            movemap.set_jump(jump_no, False)
        if verbose:
            print(f"  [Ligand Rigidity] Disabled chi/bb for residues {ligand_resnos}")
            print(f"  [Ligand Rigidity] Disabled jumps {ligand_jumps}")

    elif ligand_rigidity == "rigid_body":
        # Disable internal DOFs but allow rigid-body movement
        for resno in ligand_resnos:
            movemap.set_chi(resno, False)
            movemap.set_bb(resno, False)
        # Jumps remain enabled (set_jump(True) by default)
        for jump_no in ligand_jumps:
            movemap.set_jump(jump_no, True)
        if verbose:
            print(f"  [Ligand Rigidity] Disabled chi/bb for residues {ligand_resnos}")
            print(f"  [Ligand Rigidity] Enabled jumps {ligand_jumps} (independent rigid-body movement)")

    elif ligand_rigidity == "flexible":
        # Full flexibility - enable everything
        for resno in ligand_resnos:
            movemap.set_chi(resno, True)
            movemap.set_bb(resno, True)
        for jump_no in ligand_jumps:
            movemap.set_jump(jump_no, True)
        if verbose:
            print(f"  [Ligand Rigidity] Enabled chi/bb for residues {ligand_resnos}")
            print(f"  [Ligand Rigidity] Enabled jumps {ligand_jumps}")

    return ligand_resnos


# =============================================================================
# Ligand Manipulation Functions
# =============================================================================

def separate_protein_and_ligand(pose, resno=None):
    """
    Separate ligand from protein by translating it far away (666 Angstroms).

    Arguments:
        pose: PyRosetta Pose
        resno: Ligand residue number (default: last residue)

    Returns:
        Pose: Pose with separated ligand
    """
    tmp_pose = pose.clone()
    lig_seqpos = resno if resno is not None else tmp_pose.size()

    assert tmp_pose.residue(lig_seqpos).is_ligand(), f"Residue {lig_seqpos} is not a ligand!"

    lig_jump_no = tmp_pose.fold_tree().get_jump_that_builds_residue(lig_seqpos)
    rbt = pyr.rosetta.protocols.rigid.RigidBodyTransMover(tmp_pose, lig_jump_no)
    rbt.step_size(666)
    rbt.apply(tmp_pose)

    return tmp_pose


# =============================================================================
# Scoring Functions
# =============================================================================

def fix_scorefxn(sfxn, allow_double_bb=False):
    """
    Fix scorefunction for proper H-bond decomposition.

    Arguments:
        sfxn: ScoreFunction to modify
        allow_double_bb: Allow backbone-backbone H-bonds
    """
    opts = sfxn.energy_method_options()
    opts.hbond_options().decompose_bb_hb_into_pair_energies(True)
    opts.hbond_options().bb_donor_acceptor_check(not allow_double_bb)
    sfxn.set_energy_method_options(opts)


def get_one_and_twobody_energies(pose, scorefxn):
    """
    Calculate one-body and two-body energy terms.

    Arguments:
        pose: PyRosetta Pose
        scorefxn: ScoreFunction

    Returns:
        tuple: (onebody_energies, twobody_energies) as numpy arrays
    """
    nres = pose.size()
    res_energy_no_two_body = np.zeros(nres)
    res_pair_energy = np.zeros((nres, nres))

    scorefxn(pose)
    energy_graph = pose.energies().energy_graph()

    twobody_terms = energy_graph.active_2b_score_types()
    onebody_weights = pyrosetta.rosetta.core.scoring.EMapVector()
    onebody_weights.assign(scorefxn.weights())

    for term in twobody_terms:
        if 'intra' not in pyrosetta.rosetta.core.scoring.name_from_score_type(term):
            onebody_weights.set(term, 0)

    for i in range(1, pose.size() + 1):
        res_energy_no_two_body[i - 1] = pose.energies().residue_total_energies(i).dot(onebody_weights)
        for j in range(1, pose.size() + 1):
            if i != j:
                edge = energy_graph.find_edge(i, j)
                if edge is not None:
                    res_pair_energy[i - 1][j - 1] = edge.fill_energy_map().dot(scorefxn.weights())

    return res_energy_no_two_body, res_pair_energy


def calculate_ddg(pose, scorefxn):
    """
    Calculate interaction ddG based on two-body energies.
    Corrected for repulsive terms at constraint-bonded residue pairs.

    Arguments:
        pose: PyRosetta Pose
        scorefxn: ScoreFunction with decompose_bb_hb_into_pair_energies=True

    Returns:
        float: Interaction ddG
    """
    ligands = [r for r in pose.residues if r.is_ligand()]
    twobody_energies = get_one_and_twobody_energies(pose, scorefxn)[1]

    # Get energies without fa_rep for covalent corrections
    sfx_no_rep = scorefxn.clone()
    sfx_no_rep.set_weight(pyrosetta.rosetta.core.scoring.fa_rep, 0.0)
    twobody_energies_no_fa_rep = get_one_and_twobody_energies(pose, sfx_no_rep)[1]

    # Find covalent constraint bonds
    cst_covalents = []
    for lig in ligands:
        for res2 in pose.residues:
            if lig.seqpos() != res2.seqpos() and lig.is_bonded(res2):
                cst_covalents.append((lig, res2))

    protein_pos = np.array([r.seqpos() - 1 for r in pose.residues if r.is_protein()])

    ddg = 0.0
    for lig in ligands:
        ddg += np.sum(twobody_energies[lig.seqpos() - 1][protein_pos])

    # Add repulsion-corrected two-body energies for covalent pairs
    for covpair in cst_covalents:
        ddg += (
            -twobody_energies[covpair[0].seqpos() - 1][covpair[1].seqpos() - 1]
            + twobody_energies_no_fa_rep[covpair[0].seqpos() - 1][covpair[1].seqpos() - 1]
        )

    return ddg


def getSASA(pose, resno=None, SASA_atoms=None, ignore_sc=False, method="auto"):
    """
    Calculate Solvent Accessible Surface Area.

    By default uses VarSolDistSasaCalculator (pure PyRosetta, no external deps).
    Can optionally use DAlphaBall method if specified.

    Arguments:
        pose: PyRosetta Pose
        resno: Specific residue number (default: whole pose)
        SASA_atoms: List of atom numbers to include
        ignore_sc: If True, ignore side-chain atoms
        method: "auto" (default, uses varsoldist), "varsoldist", or "dalphaball"

    Returns:
        float or SurfVol: SASA value or full surface/volume object
    """
    if method == "dalphaball":
        return _getSASA_dalphaball(pose, resno, SASA_atoms, ignore_sc)
    else:
        # Default to VarSolDistSasaCalculator - no external dependencies
        return _getSASA_varsoldist(pose, resno, SASA_atoms, ignore_sc)


def _getSASA_dalphaball(pose, resno=None, SASA_atoms=None, ignore_sc=False):
    """SASA calculation using DAlphaBall (requires libgfortran)."""
    atoms = pyrosetta.rosetta.core.id.AtomID_Map_bool_t()
    atoms.resize(pose.size())

    for i, res in enumerate(pose.residues):
        if res.is_ligand():
            atoms.resize(i + 1, res.natoms(), True)
        else:
            atoms.resize(i + 1, res.natoms(), not ignore_sc)
            if ignore_sc:
                for n in range(1, res.natoms() + 1):
                    if res.atom_is_backbone(n) and not res.atom_is_hydrogen(n):
                        atoms[i + 1][n] = True

    surf_vol = pyrosetta.rosetta.core.scoring.packing.get_surf_vol(pose, atoms, 1.4)

    if resno is not None:
        res_surf = 0.0
        for i in range(1, pose.residue(resno).natoms() + 1):
            if SASA_atoms is not None and i not in SASA_atoms:
                continue
            res_surf += surf_vol.surf(resno, i)
        return res_surf

    return surf_vol


def _getSASA_varsoldist(pose, resno=None, SASA_atoms=None, ignore_sc=False):
    """
    Fallback SASA calculation using VarSolDistSasaCalculator.
    Does not require DAlphaBall/libgfortran.
    """
    sasa_calc = pyrosetta.rosetta.protocols.vardist_solaccess.VarSolDistSasaCalculator()
    sasa_map = sasa_calc.calculate(pose)

    if resno is not None:
        res_surf = 0.0
        res = pose.residue(resno)
        for i in range(1, res.natoms() + 1):
            if SASA_atoms is not None and i not in SASA_atoms:
                continue
            if ignore_sc and not res.atom_is_backbone(i):
                continue
            atom_id = pyrosetta.rosetta.core.id.AtomID(i, resno)
            res_surf += sasa_map[atom_id]
        return res_surf

    # Return total SASA for pose
    total_sasa = 0.0
    for res_i in range(1, pose.size() + 1):
        res = pose.residue(res_i)
        for atom_i in range(1, res.natoms() + 1):
            if ignore_sc and not res.atom_is_backbone(atom_i):
                continue
            atom_id = pyrosetta.rosetta.core.id.AtomID(atom_i, res_i)
            total_sasa += sasa_map[atom_id]
    return total_sasa


def get_angle(a1, a2, a3):
    """
    Calculate angle between three points.

    Arguments:
        a1, a2, a3: Coordinate arrays or PyRosetta xyzVector

    Returns:
        float: Angle in degrees
    """
    a1 = np.array(a1)
    a2 = np.array(a2)
    a3 = np.array(a3)

    ba = a1 - a2
    bc = a3 - a2

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return round(np.degrees(angle), 1)


def find_hbonds_to_residue_atom(pose, target_seqpos, target_atom):
    """
    Count H-bond contacts to a specific atom.

    Arguments:
        pose: PyRosetta Pose
        target_seqpos: Target residue sequence position
        target_atom: Atom name or number

    Returns:
        int: Number of H-bond contacts
    """
    HBond_res = 0
    target = pose.residue(target_seqpos)

    if isinstance(target_atom, int):
        target_atomno = target_atom
        target_atom = target.atom_name(target_atomno)
    else:
        target_atomno = target.atom_index(target_atom)

    for res in pose.residues:
        if res.seqpos() == target_seqpos or res.is_ligand():
            continue

        if (target.xyz(target_atom) - res.xyz('CA')).norm() < 10.0:
            # Check if target is acceptor (receiving H from protein)
            if target.atom_type(target_atomno).element() != "H":
                for polar_H in res.Hpos_polar():
                    if (target.xyz(target_atom) - res.xyz(polar_H)).norm() < 2.5:
                        if res.atom_is_backbone(polar_H):
                            if get_angle(res.xyz(1), res.xyz(polar_H), target.xyz(target_atom)) < 140.0:
                                continue
                        HBond_res += 1
                        break
            # Check if target is donor (H donating to protein acceptor)
            else:
                for acceptor in res.accpt_pos():
                    if (target.xyz(target_atom) - res.xyz(acceptor)).norm() < 2.5:
                        adjacent_heavy = target.get_adjacent_heavy_atoms(target_atomno)
                        if len(adjacent_heavy) > 0:
                            if get_angle(res.xyz(acceptor), target.xyz(target_atom), target.xyz(adjacent_heavy[1])) < 140.0:
                                continue
                        HBond_res += 1
                        break

    return HBond_res


# =============================================================================
# No-Ligand-Repack Functions
# =============================================================================

def _perform_no_ligand_repack(pose, scrfxn, repack_residues_list):
    """
    Perform repacking on specific residues.

    Arguments:
        pose: PyRosetta Pose
        scrfxn: ScoreFunction
        repack_residues_list: List of residue numbers to repack

    Returns:
        Pose: Repacked pose
    """
    repack_residues = ResidueIndexSelector()
    for r in repack_residues_list:
        repack_residues.append_index(r)
    do_not_repack = NotResidueSelector(repack_residues)

    tmp_pose = pose.clone()

    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())

    erg_RLT = pyrosetta.rosetta.core.pack.task.operation.ExtraRotamersGenericRLT()
    erg_RLT.ex1(True)
    erg_RLT.ex2(True)
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(erg_RLT, repack_residues))
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT(), repack_residues))
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT(), do_not_repack, False))

    task = tf.create_task_and_apply_taskoperations(tmp_pose)
    pack_mover = PackRotamersMover(scrfxn, task)
    pack_mover.apply(tmp_pose)

    return tmp_pose


def _rmsd_no_super(pose1, pose2, residues):
    """Calculate RMSD without superposition."""
    sum2 = 0.0
    natoms = 0
    for res in residues:
        num_atoms = pose1.residue(res).natoms()
        for atomno in range(1, num_atoms + 1):
            diff = pose1.residue(res).xyz(atomno) - pose2.residue(res).xyz(atomno)
            sum2 += diff.length_squared()
            natoms += 1
    return np.sqrt(sum2 / natoms) if natoms > 0 else 0.0


def _get_target_residues_from_csts(pose):
    """Get target residues from pose constraints."""
    if not pose.constraint_set().has_constraints():
        return []

    obs = pyrosetta.rosetta.protocols.toolbox.match_enzdes_util.get_enzdes_observer(pose)
    cc = obs.cst_cache()

    targets = []
    if cc is not None:
        for cst_no in range(1, cc.ncsts() + 1):
            for cst in cc.param_cache(cst_no).active_pose_constraints():
                for resno in cst.residues():
                    if resno not in targets:
                        res = pose.residue(resno)
                        if not res.is_ligand() and not res.is_virtual_residue():
                            targets.append(resno)
    return targets


def no_ligand_repack(pose, scorefxn, target_residues=None, ligand_resno=None, verbose=True):
    """
    Perform no-ligand-repack analysis.

    This removes the ligand and repacks the pocket to see how much the structure
    changes, which indicates how well the pocket is pre-organized.

    Arguments:
        pose: PyRosetta Pose
        scorefxn: ScoreFunction
        target_residues: List of target residue numbers (default: from constraints)
        ligand_resno: Ligand residue number (default: last residue)
        verbose: Print progress information

    Returns:
        DataFrame: NLR scores including nlr_dE, nlr_totrms, and per-residue RMSD
    """
    if verbose:
        print("  [NLR] Running no-ligand-repack analysis...")

    nlr_df = pd.DataFrame()

    if target_residues is None:
        if pose.constraint_set().has_constraints():
            target_residues = _get_target_residues_from_csts(pose)
        else:
            target_residues = []

    if ligand_resno is None:
        ligand_resno = pose.size()

    # Get layer residues for repacking
    heavyatoms = get_ligand_heavyatoms(pose, ligand_resno)
    _, _, _, residues = get_layer_selections(pose, [], [], ref_resno=ligand_resno, heavyatoms=heavyatoms)
    nlr_repack_residues = residues[0] + residues[1] + residues[2] + residues[3]

    if verbose:
        print(f"  [NLR] Repacking {len(nlr_repack_residues)} pocket residues")

    # Separate ligand
    pose_separated = separate_protein_and_ligand(pose, ligand_resno)

    # Score both poses
    scorefxn(pose)
    scorefxn(pose_separated)

    # Remove constraints for repacking
    if pose_separated.constraint_set().has_constraints():
        pose_separated.constraint_set().clear()

    # Perform repacking
    nlr_pose = _perform_no_ligand_repack(pose_separated, scorefxn, nlr_repack_residues)

    # Calculate metrics
    nlr_rms = _rmsd_no_super(pose_separated, nlr_pose, nlr_repack_residues)

    nlr_df.at[0, 'nlr_dE'] = nlr_pose.scores['total_score'] - pose.scores['total_score']
    nlr_df.at[0, 'nlr_totrms'] = nlr_rms

    if verbose:
        print(f"  [NLR] Total RMSD: {nlr_rms:.3f} A | dE: {nlr_df.at[0, 'nlr_dE']:.1f}")

    # Per-residue RMSD for target residues
    for i, resno in enumerate(target_residues):
        res_rms = _rmsd_no_super(pose_separated, nlr_pose, [resno])
        nlr_df.at[0, f'nlr_SR{i + 1}_rms'] = res_rms
        if verbose:
            print(f"  [NLR] Residue {resno} RMSD: {res_rms:.3f} A")

    return nlr_df


# =============================================================================
# Scorefile I/O Functions
# =============================================================================

def dump_scorefile(df, filename, append=True):
    """
    Write scores DataFrame to Rosetta-style scorefile.

    Arguments:
        df: pandas DataFrame with scores
        filename: Output filename
        append: If True, append to existing file
    """
    widths = {}
    for k in df.keys():
        if k in ["SCORE:", "description", "name"]:
            widths[k] = 0
        elif isinstance(df.at[df.index.values[0], k], str):
            max_val_len = max([len(row[k]) for index, row in df.iterrows()])
            widths[k] = max(max_val_len, len(k)) + 1
        elif len(k) >= 12:
            widths[k] = len(k) + 1
        else:
            widths[k] = 12

    keys = df.keys()
    write_title = True

    if os.path.exists(filename):
        if append:
            write_title = False
            keys = open(filename, 'r').readlines()[0].split()
            keys = [x.rstrip() for x in keys]
            if len(keys) != len(df.keys()):
                print(f"Warning: Column count mismatch in {filename}: {len(keys)} != {len(df.keys())}")
        else:
            print(f"Warning: Overwriting existing scorefile {filename}")

    mode = "a" if append else "w"
    with open(filename, mode) as file:
        if write_title:
            title = ""
            for k in df.keys():
                if k == "SCORE:":
                    title += k
                elif k in ["description", "name"]:
                    continue
                else:
                    title += f"{k:>{widths[k]}}"
            if 'description' in df.keys():
                title += " description"
            file.write(title + "\n")

        for index, row in df.iterrows():
            line = ""
            for k in keys:
                if k not in df.keys():
                    val = f"{np.nan}"
                    widths[k] = 11
                elif isinstance(row[k], (float, np.float16, np.float64, np.float32)):
                    val = f"{row[k]:.3f}"
                else:
                    val = row[k]
                if k == "SCORE:":
                    line += val
                elif k in ["description", "name"]:
                    continue
                else:
                    line += f"{val:>{widths[k]}}"
            if 'description' in df.keys():
                line += f" {row['description']}"
            file.write(line + "\n")


def calculate_ca_rmsd(pose1, pose2, residues=None):
    """
    Calculate CA RMSD between two poses after superposition.

    Arguments:
        pose1: First PyRosetta Pose
        pose2: Second PyRosetta Pose
        residues: List of residue numbers to include (default: all protein residues)

    Returns:
        float: CA RMSD in Angstroms
    """
    if residues is None:
        residues = [r.seqpos() for r in pose1.residues if r.is_protein()]

    # Get CA coordinates
    coords1 = []
    coords2 = []
    for resno in residues:
        if pose1.residue(resno).has("CA") and pose2.residue(resno).has("CA"):
            coords1.append(np.array(pose1.residue(resno).xyz("CA")))
            coords2.append(np.array(pose2.residue(resno).xyz("CA")))

    if len(coords1) == 0:
        return 0.0

    coords1 = np.array(coords1)
    coords2 = np.array(coords2)

    # Center both coordinate sets
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)
    coords1_centered = coords1 - centroid1
    coords2_centered = coords2 - centroid2

    # Compute optimal rotation using SVD (Kabsch algorithm)
    H = coords1_centered.T @ coords2_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Apply rotation
    coords1_rotated = coords1_centered @ R

    # Calculate RMSD
    diff = coords1_rotated - coords2_centered
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

    return rmsd


def calculate_ca_rmsd_converged(pose1, pose2, residues=None, outlier_threshold=2.0, max_iterations=10):
    """
    Calculate CA RMSD iteratively removing outliers until convergence.

    Arguments:
        pose1: First PyRosetta Pose
        pose2: Second PyRosetta Pose
        residues: List of residue numbers to include (default: all protein residues)
        outlier_threshold: Remove residues with per-residue RMSD > threshold * median (default: 2.0)
        max_iterations: Maximum iterations for convergence (default: 10)

    Returns:
        tuple: (converged_rmsd, num_residues_used, num_outliers_removed)
    """
    if residues is None:
        residues = [r.seqpos() for r in pose1.residues if r.is_protein()]

    current_residues = list(residues)
    total_outliers = 0

    for iteration in range(max_iterations):
        # Get CA coordinates for current residues
        coords1 = []
        coords2 = []
        valid_residues = []
        for resno in current_residues:
            if pose1.residue(resno).has("CA") and pose2.residue(resno).has("CA"):
                coords1.append(np.array(pose1.residue(resno).xyz("CA")))
                coords2.append(np.array(pose2.residue(resno).xyz("CA")))
                valid_residues.append(resno)

        if len(coords1) < 3:
            break

        coords1 = np.array(coords1)
        coords2 = np.array(coords2)

        # Center and superimpose
        centroid1 = np.mean(coords1, axis=0)
        centroid2 = np.mean(coords2, axis=0)
        coords1_centered = coords1 - centroid1
        coords2_centered = coords2 - centroid2

        H = coords1_centered.T @ coords2_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        coords1_rotated = coords1_centered @ R

        # Calculate per-residue distances
        per_res_dist = np.sqrt(np.sum((coords1_rotated - coords2_centered) ** 2, axis=1))

        # Find outliers
        median_dist = np.median(per_res_dist)
        threshold = outlier_threshold * median_dist if median_dist > 0 else outlier_threshold

        non_outliers = per_res_dist <= threshold
        new_residues = [valid_residues[i] for i in range(len(valid_residues)) if non_outliers[i]]
        outliers_this_round = len(valid_residues) - len(new_residues)
        total_outliers += outliers_this_round

        # Check for convergence
        if outliers_this_round == 0 or len(new_residues) == len(current_residues):
            break

        current_residues = new_residues

    # Calculate final RMSD
    final_rmsd = calculate_ca_rmsd(pose1, pose2, current_residues)

    return final_rmsd, len(current_residues), total_outliers


def calculate_sequence_identity(pose1, pose2, residues=None):
    """
    Calculate sequence identity between two poses.

    Arguments:
        pose1: First PyRosetta Pose
        pose2: Second PyRosetta Pose
        residues: List of residue numbers to compare (default: all protein residues)

    Returns:
        float: Sequence identity as fraction (0.0 to 1.0)
    """
    if residues is None:
        residues = [r.seqpos() for r in pose1.residues if r.is_protein()]

    identical = 0
    total = 0

    for resno in residues:
        res1 = pose1.residue(resno)
        res2 = pose2.residue(resno)
        if res1.is_protein() and res2.is_protein():
            if res1.name1() == res2.name1():
                identical += 1
            total += 1

    return identical / total if total > 0 else 0.0


def filter_scores(scores, filters):
    """
    Filter scores DataFrame based on filter criteria.

    Arguments:
        scores: pandas DataFrame with scores
        filters: dict {score_name: [value, comparison_operator]}
                 e.g., {"total_score": [-200.0, "<="], "sc": [0.5, ">="]}

    Returns:
        DataFrame: Filtered scores
    """
    filtered_scores = scores.copy()

    for s in filters.keys():
        if filters[s] is not None and s in scores.keys():
            val = filters[s][0]
            sign = COMPARISONS[filters[s][1]]
            filtered_scores = filtered_scores.loc[(filtered_scores[s].__getattribute__(sign)(val))]
            n_passed = len(scores.loc[(scores[s].__getattribute__(sign)(val))])
            print(f"  {s:<24} {filters[s][1]:<2} {val:>7.3f}: {len(filtered_scores)} left ({(n_passed/len(scores))*100:.0f}% pass)")

    return filtered_scores


# =============================================================================
# Adaptive Coordinate Constraint Functions
# =============================================================================

def get_constraint_score_terms():
    """Get the constraint score type objects."""
    st = pyrosetta.rosetta.core.scoring
    return {
        'atom_pair_constraint': st.score_type_from_name('atom_pair_constraint'),
        'angle_constraint': st.score_type_from_name('angle_constraint'),
        'dihedral_constraint': st.score_type_from_name('dihedral_constraint'),
        'coordinate_constraint': st.score_type_from_name('coordinate_constraint'),
    }


def evaluate_per_residue_constraint_scores(pose, sfx):
    """
    Evaluate constraint scores for each residue.

    Arguments:
        pose: PyRosetta Pose with constraints applied
        sfx: ScoreFunction with constraint weights

    Returns:
        dict: {resno: {'total': score, 'atom_pair': score, 'angle': score, 'dihedral': score}}
    """
    # Score the pose to populate energies
    sfx(pose)

    cst_terms = get_constraint_score_terms()
    scores = {}

    for resno in range(1, pose.size() + 1):
        res_energies = pose.energies().residue_total_energies(resno)
        scores[resno] = {
            'atom_pair': res_energies[cst_terms['atom_pair_constraint']],
            'angle': res_energies[cst_terms['angle_constraint']],
            'dihedral': res_energies[cst_terms['dihedral_constraint']],
        }
        scores[resno]['total'] = (
            scores[resno]['atom_pair'] +
            scores[resno]['angle'] +
            scores[resno]['dihedral']
        )

    return scores


def evaluate_catalytic_constraint_scores(pose, sfx, catalytic_residues, verbose=True):
    """
    Evaluate constraint scores specifically for catalytic residues.

    Arguments:
        pose: PyRosetta Pose with constraints applied
        sfx: ScoreFunction with constraint weights
        catalytic_residues: List of catalytic residue numbers or dict from get_matcher_residues()
        verbose: Print detailed constraint score information

    Returns:
        dict: {resno: {'total': score, 'atom_pair': score, 'angle': score, 'dihedral': score,
                       'name3': residue_name}}
    """
    # Handle both list and dict input
    if isinstance(catalytic_residues, dict):
        cat_resnos = list(catalytic_residues.keys())
    else:
        cat_resnos = list(catalytic_residues)

    # Get all per-residue scores
    all_scores = evaluate_per_residue_constraint_scores(pose, sfx)

    # Extract catalytic residue scores
    cat_scores = {}
    for resno in cat_resnos:
        if resno in all_scores:
            cat_scores[resno] = all_scores[resno].copy()
            cat_scores[resno]['name3'] = pose.residue(resno).name3()

    if verbose and cat_scores:
        print(f"\n  [Constraint Scores] Catalytic Residues:")
        print(f"  {'Res':<6} {'Name':<5} {'Total':>8} {'AtomPair':>10} {'Angle':>10} {'Dihedral':>10}")
        print(f"  {'-'*55}")

        total_scores = []
        for resno in sorted(cat_scores.keys()):
            s = cat_scores[resno]
            total_scores.append(s['total'])
            print(f"  {resno:<6} {s['name3']:<5} {s['total']:>8.2f} {s['atom_pair']:>10.2f} {s['angle']:>10.2f} {s['dihedral']:>10.2f}")

        if total_scores:
            mean_score = np.mean(total_scores)
            std_score = np.std(total_scores)
            print(f"  {'-'*55}")
            print(f"  {'Mean':<12} {mean_score:>8.2f}")
            print(f"  {'Std':<12} {std_score:>8.2f}")
            print(f"  {'Min':<12} {min(total_scores):>8.2f}")
            print(f"  {'Max':<12} {max(total_scores):>8.2f}")

    return cat_scores


def identify_residues_needing_freedom(pose, catalytic_scores, threshold=5.0,
                                       neighbor_window=3, verbose=True):
    """
    Identify catalytic residues that are "far off" from their constraints,
    plus their neighboring residues.

    Arguments:
        pose: PyRosetta Pose
        catalytic_scores: dict from evaluate_catalytic_constraint_scores()
        threshold: Constraint score threshold above which a residue is "far off"
        neighbor_window: Number of neighboring residues (±N) to also free
        verbose: Print information about identified residues

    Returns:
        dict: {resno: weight} where weight is the coordinate constraint weight to use
              (0.0 = completely free, 1.0 = fully constrained)
        Also returns list of "far off" catalytic residues for reference
    """
    far_off_residues = []
    residues_to_reduce = set()

    # Identify "far off" catalytic residues
    for resno, scores in catalytic_scores.items():
        if scores['total'] > threshold:
            far_off_residues.append(resno)
            residues_to_reduce.add(resno)

            # Add neighbors (respecting chain boundaries)
            res_chain = pose.residue(resno).chain()
            for offset in range(-neighbor_window, neighbor_window + 1):
                neighbor_resno = resno + offset
                if 1 <= neighbor_resno <= pose.size():
                    neighbor_res = pose.residue(neighbor_resno)
                    # Only add if same chain and is protein
                    if neighbor_res.chain() == res_chain and neighbor_res.is_protein():
                        residues_to_reduce.add(neighbor_resno)

    if verbose:
        print(f"\n  [Adaptive Coord Constraints] Threshold: {threshold:.1f} REU")
        print(f"  [Adaptive Coord Constraints] Neighbor window: ±{neighbor_window} residues")

        if far_off_residues:
            print(f"  [Adaptive Coord Constraints] Far-off catalytic residues ({len(far_off_residues)}):")
            for resno in far_off_residues:
                s = catalytic_scores[resno]
                print(f"    - {s['name3']} {resno}: score = {s['total']:.2f} (threshold = {threshold:.1f})")

            print(f"  [Adaptive Coord Constraints] Residues with reduced coord constraints ({len(residues_to_reduce)}):")
            print(f"    {'+'.join(str(r) for r in sorted(residues_to_reduce))}")
        else:
            print(f"  [Adaptive Coord Constraints] All catalytic residues satisfy threshold - no adaptation needed")

    return sorted(residues_to_reduce), far_off_residues


def create_per_residue_coord_cst_weights(pose, reduced_residues, normal_weight=1.0,
                                          reduced_weight=0.2, verbose=True):
    """
    Create a dictionary of per-residue coordinate constraint weights.

    Arguments:
        pose: PyRosetta Pose
        reduced_residues: List of residue numbers to apply reduced weight
        normal_weight: Weight for normal residues (default: 1.0)
        reduced_weight: Weight for reduced residues (default: 0.2)
        verbose: Print summary

    Returns:
        dict: {resno: weight} for all residues in pose
    """
    weights = {}
    for resno in range(1, pose.size() + 1):
        if resno in reduced_residues:
            weights[resno] = reduced_weight
        else:
            weights[resno] = normal_weight

    if verbose:
        n_reduced = len([r for r in weights.values() if r == reduced_weight])
        n_normal = len([r for r in weights.values() if r == normal_weight])
        print(f"  [Coord Cst Weights] Normal ({normal_weight}): {n_normal} residues")
        print(f"  [Coord Cst Weights] Reduced ({reduced_weight}): {n_reduced} residues")

    return weights


def apply_coordinate_constraints_with_weights(pose, coord_cst_weights, reference_pose=None):
    """
    Apply coordinate constraints to a pose with per-residue weights.

    Arguments:
        pose: PyRosetta Pose to add constraints to
        coord_cst_weights: dict {resno: weight} from create_per_residue_coord_cst_weights()
        reference_pose: Reference pose for coordinates (default: use current pose coords)

    Returns:
        Pose: Pose with coordinate constraints added
    """
    if reference_pose is None:
        reference_pose = pose.clone()

    # Clear existing coordinate constraints
    cst_set = pose.constraint_set().clone()

    # Create coordinate constraints with per-residue weights
    for resno, weight in coord_cst_weights.items():
        if weight <= 0.0:
            continue  # Skip residues with zero weight

        res = pose.residue(resno)
        if res.is_ligand() or res.is_virtual_residue():
            continue

        # Constrain CA atom (backbone representative)
        if res.has("CA"):
            atom_id = pyrosetta.rosetta.core.id.AtomID(res.atom_index("CA"), resno)
            ref_xyz = reference_pose.residue(resno).xyz("CA")

            # Create harmonic function with weight
            # Standard deviation controls tightness; weight scales the penalty
            harmonic = pyrosetta.rosetta.core.scoring.func.HarmonicFunc(0.0, 1.0 / weight)

            coord_cst = pyrosetta.rosetta.core.scoring.constraints.CoordinateConstraint(
                atom_id,
                pyrosetta.rosetta.core.id.AtomID(1, 1),  # Fixed reference atom
                ref_xyz,
                harmonic
            )
            cst_set.add_constraint(coord_cst)

    pose.constraint_set(cst_set)
    return pose


def setup_fastrelax_adaptive(sfx, coord_cst_weights=None, crude=False,
                              disable_min_resons=None, pose=None, ligand_rigidity="fixed"):
    """
    Set up FastRelax with adaptive per-residue coordinate constraints.

    This is an alternative to setup_fastrelax() that supports per-residue
    coordinate constraint weights for catalytic residues that need more freedom.

    Arguments:
        sfx: ScoreFunction
        coord_cst_weights: dict {resno: weight} or None for uniform weights
        crude: If True, use crude/fast relax script
        disable_min_resons: List of residue numbers to exclude from minimization
        pose: PyRosetta Pose (required if ligand_rigidity != "flexible")
        ligand_rigidity: One of "fixed", "rigid_body", "flexible"

    Returns:
        FastRelax: Configured FastRelax mover
    """
    fastRelax = pyrosetta.rosetta.protocols.relax.FastRelax(sfx, 1)

    if crude:
        script = _get_crude_fastrelax_script()
        filelines = pyrosetta.rosetta.std.vector_std_string()
        for l in script:
            filelines.append(l.rstrip())
        fastRelax.set_script_from_lines(filelines)

    # Only use uniform coordinate constraints if no per-residue weights specified
    if coord_cst_weights is None:
        fastRelax.constrain_relax_to_start_coords(True)
    else:
        # Per-residue constraints will be applied separately
        fastRelax.constrain_relax_to_start_coords(False)

    # Set up TaskFactory
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())

    e = pyrosetta.rosetta.core.pack.task.operation.ExtraRotamersGeneric()
    if crude:
        e.ex1(False)
        e.ex1aro(False)
        e.ex2(False)
    else:
        e.ex1(True)
        e.ex1aro(True)
        e.ex2(True)
        e.ex1_sample_level(pyrosetta.rosetta.core.pack.task.ExtraRotSample(1))
    tf.push_back(e)
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())
    fastRelax.set_task_factory(tf)

    # Set up MoveMap
    mm = pyrosetta.rosetta.core.kinematics.MoveMap()
    mm.set_chi(True)
    mm.set_bb(True)
    mm.set_jump(True)

    if disable_min_resons is not None:
        for resno in disable_min_resons:
            mm.set_chi(resno, False)
            mm.set_bb(resno, False)

    # Configure ligand rigidity
    if pose is not None and ligand_rigidity != "flexible":
        configure_movemap_ligand_rigidity(mm, pose, ligand_rigidity, verbose=True)

    fastRelax.set_movemap(mm)
    return fastRelax


# =============================================================================
# Covalent Connection and Ligand Constraint Functions
# =============================================================================

def get_covalent_connections_from_pose(pose, allowed_residues=None, require_ligand=True, verbose=True):
    """
    Extract covalent connection information from a pose.

    This identifies inter-residue chemical bonds (not standard peptide bonds),
    including covalent protein-ligand bonds.

    Arguments:
        pose: PyRosetta Pose with established covalent connections
        allowed_residues: List of seqpos that can participate in covalent bonds.
                          If provided, BOTH residues in a bond must be in this list
                          (unless one is a ligand and require_ligand=True).
                          If None, no filtering is applied.
        require_ligand: If True (default), at least one residue must be a ligand.
                        This prevents spurious sidechain-sidechain bonds from clashes.
        verbose: Print connection information

    Returns:
        list: List of dicts with connection info:
              [{'res1': seqpos, 'atom1': atom_name, 'res2': seqpos, 'atom2': atom_name,
                'res1_type': full_residue_type_name, 'res2_type': full_residue_type_name}]
    """
    connections = []

    # Get ligand residues for filtering
    ligand_resnos = set(get_ligand_residues(pose))

    # Build allowed set including ligands
    if allowed_residues is not None:
        allowed_set = set(allowed_residues) | ligand_resnos
    else:
        allowed_set = None

    for resno in range(1, pose.size() + 1):
        res = pose.residue(resno)

        # Check all connection points for this residue
        for conn_id in range(1, res.n_possible_residue_connections() + 1):
            partner_resno = res.connected_residue_at_resconn(conn_id)

            # Skip if no partner or if it's a standard peptide bond (conn_id 1 or 2 for proteins)
            if partner_resno == 0:
                continue

            # Skip standard N-C peptide connections for proteins
            if res.is_protein() and conn_id <= 2:
                continue

            partner_res = pose.residue(partner_resno)

            # Apply filtering: require_ligand check
            if require_ligand:
                if resno not in ligand_resnos and partner_resno not in ligand_resnos:
                    # Neither is a ligand - skip unless both are in allowed_residues (catres-catres)
                    if allowed_set is not None:
                        if resno not in allowed_residues or partner_resno not in allowed_residues:
                            continue
                    else:
                        continue  # No allowed_residues and no ligand involved - skip

            # Apply filtering: allowed_residues check
            if allowed_set is not None:
                if resno not in allowed_set or partner_resno not in allowed_set:
                    if verbose:
                        print(f"  [Covalent] Skipping bond {res.name3()}{resno} -- {partner_res.name3()}{partner_resno}: not in allowed residues")
                    continue

            # Get atom names for this connection
            atom_idx = res.residue_connect_atom_index(conn_id)
            atom_name = res.atom_name(atom_idx).strip()

            # Find partner's connection ID and atom
            partner_conn_id = res.residue_connection_conn_id(conn_id)
            if partner_conn_id > 0 and partner_conn_id <= partner_res.n_possible_residue_connections():
                partner_atom_idx = partner_res.residue_connect_atom_index(partner_conn_id)
                partner_atom_name = partner_res.atom_name(partner_atom_idx).strip()
            else:
                partner_atom_name = "UNK"

            # Only record each bond once (lower resno first)
            if resno < partner_resno:
                conn_info = {
                    'res1': resno,
                    'atom1': atom_name,
                    'res2': partner_resno,
                    'atom2': partner_atom_name,
                    'res1_type': res.type().name(),
                    'res2_type': partner_res.type().name()
                }
                connections.append(conn_info)

                if verbose:
                    print(f"  [Covalent] Found bond: {res.name3()}{resno}:{atom_name} -- {partner_res.name3()}{partner_resno}:{partner_atom_name}")
                    print(f"             Residue types: {res.type().name()} -- {partner_res.type().name()}")

    return connections


def reestablish_covalent_connections(pose, ref_pose, cst_io=None, sfx=None,
                                      allowed_residues=None, require_ligand=True, verbose=True):
    """
    Re-establish covalent connections in a pose based on a reference pose.

    This is needed when loading MPNN-packed structures, which lose the covalent
    connection information (like LYS:SidechainConjugation bonds to ligand).

    The function:
    1. Identifies covalent connections in the reference pose (filtered by allowed_residues)
    2. Applies necessary residue type patches to the new pose
    3. Declares the chemical bonds

    Arguments:
        pose: PyRosetta Pose to fix (modified in place)
        ref_pose: Reference pose with correct covalent connections (typically from --ref_pdb)
        cst_io: Optional EnzConstraintIO for setting up enzyme constraints
        sfx: ScoreFunction (required if cst_io is provided)
        allowed_residues: List of seqpos (catalytic residues) allowed for covalent bonds.
                          If provided, only catres-ligand or catres-catres bonds are restored.
        require_ligand: If True (default), at least one residue must be ligand (prevents
                        spurious sidechain-sidechain bonds from clashes)
        verbose: Print progress information

    Returns:
        Pose: The modified pose (same object, modified in place)
    """
    if verbose:
        print("  [Covalent] Re-establishing covalent connections from reference PDB...")
        if allowed_residues:
            print(f"  [Covalent] Allowed catalytic residues for covalent bonds: {allowed_residues}")

    # Get connections from reference pose with filtering
    ref_connections = get_covalent_connections_from_pose(
        ref_pose,
        allowed_residues=allowed_residues,
        require_ligand=require_ligand,
        verbose=verbose
    )

    if not ref_connections:
        if verbose:
            print("  [Covalent] No non-peptide covalent connections found in reference pose")
        return pose

    # For each connection, ensure proper residue types and declare bonds
    for conn in ref_connections:
        res1_seqpos = conn['res1']
        res2_seqpos = conn['res2']
        atom1_name = conn['atom1']
        atom2_name = conn['atom2']
        res1_type_name = conn['res1_type']
        res2_type_name = conn['res2_type']

        # Check if residue types match
        current_res1_type = pose.residue(res1_seqpos).type().name()
        current_res2_type = pose.residue(res2_seqpos).type().name()

        # Apply patches if needed (for protein residues with special patches)
        if current_res1_type != res1_type_name:
            if verbose:
                print(f"  [Covalent] Patching residue {res1_seqpos}: {current_res1_type} -> {res1_type_name}")
            try:
                mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
                mutres.set_target(res1_seqpos)
                mutres.set_res_name(res1_type_name)
                mutres.apply(pose)
            except Exception as e:
                if verbose:
                    print(f"  [Covalent] Warning: Could not apply patch to residue {res1_seqpos}: {e}")

        if current_res2_type != res2_type_name:
            if verbose:
                print(f"  [Covalent] Patching residue {res2_seqpos}: {current_res2_type} -> {res2_type_name}")
            try:
                mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
                mutres.set_target(res2_seqpos)
                mutres.set_res_name(res2_type_name)
                mutres.apply(pose)
            except Exception as e:
                if verbose:
                    print(f"  [Covalent] Warning: Could not apply patch to residue {res2_seqpos}: {e}")

        # Declare the chemical bond
        if verbose:
            print(f"  [Covalent] Declaring bond: {pose.residue(res1_seqpos).name3()}{res1_seqpos}:{atom1_name} -- "
                  f"{pose.residue(res2_seqpos).name3()}{res2_seqpos}:{atom2_name}")

        try:
            pose.conformation().declare_chemical_bond(
                res1_seqpos, atom1_name,
                res2_seqpos, atom2_name
            )
        except Exception as e:
            if verbose:
                print(f"  [Covalent] Warning: Could not declare bond: {e}")

    # If EnzConstraintIO is provided, use it to set up constraints
    # (this also helps establish connections via REMARK 666)
    if cst_io is not None and sfx is not None:
        if verbose:
            print("  [Covalent] Applying enzyme constraints via EnzConstraintIO...")
        try:
            cst_io.add_constraints_to_pose(pose, sfx, True)
        except Exception as e:
            if verbose:
                print(f"  [Covalent] Warning: EnzConstraintIO failed: {e}")

    return pose


def add_ligand_coordinate_constraints(pose, sfx, sd=0.01, include_hydrogens=True, verbose=True):
    """
    Add tight coordinate constraints to all ligand atoms to keep them completely frozen.

    This prevents ANY movement of ligand atoms during minimization, including hydrogens.

    Arguments:
        pose: PyRosetta Pose (modified in place)
        sfx: ScoreFunction (coordinate_constraint weight will be set if zero)
        sd: Standard deviation for harmonic constraint (smaller = tighter, default 0.01 A)
        include_hydrogens: If True, constrain hydrogen atoms too (default: True)
        verbose: Print progress information

    Returns:
        int: Number of constraints added
    """
    # Ensure coordinate_constraint has a weight
    coord_cst_st = pyrosetta.rosetta.core.scoring.score_type_from_name("coordinate_constraint")
    if sfx.get_weight(coord_cst_st) == 0.0:
        sfx.set_weight(coord_cst_st, 1.0)
        if verbose:
            print("  [LigandCst] Set coordinate_constraint weight to 1.0")

    ligand_resnos = get_ligand_residues(pose)

    if not ligand_resnos:
        if verbose:
            print("  [LigandCst] No ligand residues found")
        return 0

    # Get or create a virtual root for coordinate constraints
    # Use atom 1 of residue 1 as the fixed reference point
    fixed_atom_id = pyrosetta.rosetta.core.id.AtomID(1, 1)

    n_constraints = 0
    cst_set = pose.constraint_set().clone()

    for lig_resno in ligand_resnos:
        lig_res = pose.residue(lig_resno)

        if verbose:
            print(f"  [LigandCst] Adding coordinate constraints to ligand {lig_res.name3()} (residue {lig_resno})")

        n_heavy = 0
        n_hydrogen = 0

        for atom_idx in range(1, lig_res.natoms() + 1):
            is_hydrogen = lig_res.atom_is_hydrogen(atom_idx)

            if is_hydrogen and not include_hydrogens:
                continue

            # Skip virtual atoms
            if lig_res.is_virtual(atom_idx):
                continue

            atom_id = pyrosetta.rosetta.core.id.AtomID(atom_idx, lig_resno)
            atom_xyz = lig_res.xyz(atom_idx)

            # Create very tight harmonic constraint (sd=0.01 means ~0 movement allowed)
            harmonic_func = pyrosetta.rosetta.core.scoring.func.HarmonicFunc(0.0, sd)

            coord_cst = pyrosetta.rosetta.core.scoring.constraints.CoordinateConstraint(
                atom_id,
                fixed_atom_id,
                atom_xyz,
                harmonic_func
            )

            cst_set.add_constraint(coord_cst)
            n_constraints += 1

            if is_hydrogen:
                n_hydrogen += 1
            else:
                n_heavy += 1

        if verbose:
            print(f"  [LigandCst]   Added {n_heavy} heavy atom + {n_hydrogen} hydrogen constraints")

    pose.constraint_set(cst_set)

    if verbose:
        print(f"  [LigandCst] Total ligand coordinate constraints added: {n_constraints}")

    return n_constraints


def get_acid_base_residues(pose):
    """
    Get residue numbers of acid/base residues (ASP, GLU, HIS, LYS, ARG).

    Arguments:
        pose: PyRosetta Pose

    Returns:
        dict: {resno: {'name3': str, 'polar_atoms': [atom_names]}}
    """
    acid_base_res = {}
    acid_base_names = {'ASP', 'GLU', 'HIS', 'LYS', 'ARG'}

    # Polar sidechain atoms for each residue type
    polar_sc_atoms = {
        'ASP': ['OD1', 'OD2'],
        'GLU': ['OE1', 'OE2'],
        'HIS': ['ND1', 'NE2'],  # Can be protonated/deprotonated
        'LYS': ['NZ'],
        'ARG': ['NE', 'NH1', 'NH2'],
    }

    for res in pose.residues:
        if res.name3() in acid_base_names:
            atoms = polar_sc_atoms.get(res.name3(), [])
            # Also include hydrogens bonded to these polar atoms
            polar_with_H = list(atoms)
            for atom_name in atoms:
                if res.has(atom_name):
                    atom_idx = res.atom_index(atom_name)
                    # Get bonded hydrogens
                    for h_idx in range(1, res.natoms() + 1):
                        if res.atom_is_hydrogen(h_idx):
                            # Check if this H is bonded to our polar atom
                            bonded_heavy = res.atom_base(h_idx)
                            if bonded_heavy == atom_idx:
                                polar_with_H.append(res.atom_name(h_idx).strip())

            acid_base_res[res.seqpos()] = {
                'name3': res.name3(),
                'polar_atoms': polar_with_H
            }

    return acid_base_res


def find_ligand_H_close_contacts(pose, distance_cutoff=2.5, verbose=True):
    """
    Find close contacts between ligand hydrogens (bonded to polar atoms) and
    acid/base residue polar sidechain atoms.

    These contacts often represent transition state interactions that should be
    preserved, not "fixed" by Rosetta.

    Arguments:
        pose: PyRosetta Pose
        distance_cutoff: Distance threshold in Angstroms (default: 2.5)
        verbose: Print found contacts

    Returns:
        list: List of dicts with contact info:
              [{'lig_resno': int, 'lig_atom': str, 'prot_resno': int, 'prot_atom': str, 'distance': float}]
    """
    contacts = []
    ligand_resnos = get_ligand_residues(pose)
    acid_base_res = get_acid_base_residues(pose)

    # Polar elements that ligand H might be bonded to
    polar_elements = {'O', 'N', 'S'}

    for lig_resno in ligand_resnos:
        lig_res = pose.residue(lig_resno)

        # Find ligand hydrogens bonded to polar atoms
        for h_idx in range(1, lig_res.natoms() + 1):
            if not lig_res.atom_is_hydrogen(h_idx):
                continue

            # Check if bonded to polar atom
            bonded_heavy_idx = lig_res.atom_base(h_idx)
            if bonded_heavy_idx == 0:
                continue

            bonded_element = lig_res.atom_type(bonded_heavy_idx).element()
            if bonded_element not in polar_elements:
                continue

            h_xyz = lig_res.xyz(h_idx)
            h_name = lig_res.atom_name(h_idx).strip()

            # Check distance to acid/base polar atoms
            for prot_resno, prot_info in acid_base_res.items():
                prot_res = pose.residue(prot_resno)

                for prot_atom_name in prot_info['polar_atoms']:
                    if not prot_res.has(prot_atom_name):
                        continue

                    prot_xyz = prot_res.xyz(prot_atom_name)
                    dist = (h_xyz - prot_xyz).norm()

                    if dist < distance_cutoff:
                        contact = {
                            'lig_resno': lig_resno,
                            'lig_atom': h_name,
                            'lig_bonded_to': lig_res.atom_name(bonded_heavy_idx).strip(),
                            'prot_resno': prot_resno,
                            'prot_atom': prot_atom_name,
                            'prot_name3': prot_info['name3'],
                            'distance': dist
                        }
                        contacts.append(contact)

    if verbose and contacts:
        print(f"  [CloseContacts] Found {len(contacts)} ligand H - acid/base contacts:")
        for c in contacts:
            print(f"    Ligand {c['lig_atom']}(bonded to {c['lig_bonded_to']}) - "
                  f"{c['prot_name3']}{c['prot_resno']}:{c['prot_atom']} = {c['distance']:.2f} A")

    return contacts


def reduce_fa_rep_for_ligand_contacts(sfx, pose, contacts, rep_weight_factor=0.0, verbose=True):
    """
    Reduce or eliminate fa_rep for specific ligand H - acid/base contacts.

    This is done by setting up a modified scorefunction that reduces repulsion
    for these specific atom pairs. Note: This is a workaround - ideally we'd
    use pair-specific weights, but that's complex. Instead, we reduce fa_rep
    globally for these residue pairs.

    A simpler approach: Just add attractive constraints between these atoms
    that counteract the repulsion.

    Arguments:
        sfx: ScoreFunction to modify
        pose: PyRosetta Pose
        contacts: List from find_ligand_H_close_contacts()
        rep_weight_factor: Factor to multiply fa_rep by for these contacts (0.0 = ignore repulsion)
        verbose: Print modification info

    Returns:
        list: AtomPair constraints added (can be removed later if needed)
    """
    if not contacts:
        return []

    if verbose:
        print(f"  [fa_rep] Adding compensating constraints for {len(contacts)} close contacts")

    # Instead of modifying fa_rep (which is global), add flat-bottom constraints
    # that allow these atoms to be close without penalty
    cst_set = pose.constraint_set().clone()
    added_constraints = []

    for c in contacts:
        lig_resno = c['lig_resno']
        lig_atom = c['lig_atom']
        prot_resno = c['prot_resno']
        prot_atom = c['prot_atom']

        lig_res = pose.residue(lig_resno)
        prot_res = pose.residue(prot_resno)

        if not lig_res.has(lig_atom) or not prot_res.has(prot_atom):
            continue

        lig_atom_id = pyrosetta.rosetta.core.id.AtomID(lig_res.atom_index(lig_atom), lig_resno)
        prot_atom_id = pyrosetta.rosetta.core.id.AtomID(prot_res.atom_index(prot_atom), prot_resno)

        # Use a flat-bottom function that allows close approach
        # FlatHarmonicFunc(x0, sd, tol) - flat within x0±tol, harmonic outside
        # We want to allow distances from 0 to current distance without penalty
        current_dist = c['distance']

        # Use bounded func: constant zero score from 0 to well_depth, then harmonic
        # Actually, use FLAT_HARMONIC which is flat within tolerance
        flat_func = pyrosetta.rosetta.core.scoring.func.FlatHarmonicFunc(
            current_dist,  # x0 (ideal distance)
            0.3,           # sd (width of harmonic region)
            current_dist   # tol (flat region extends from 0 to x0)
        )

        dist_cst = pyrosetta.rosetta.core.scoring.constraints.AtomPairConstraint(
            lig_atom_id, prot_atom_id, flat_func
        )

        cst_set.add_constraint(dist_cst)
        added_constraints.append(dist_cst)

        if verbose:
            print(f"    Added flat constraint: {lig_atom}@{lig_resno} - {prot_atom}@{prot_resno} (allow dist <= {current_dist:.2f} A)")

    pose.constraint_set(cst_set)

    # Ensure atom_pair_constraint has weight
    apc_st = pyrosetta.rosetta.core.scoring.score_type_from_name("atom_pair_constraint")
    if sfx.get_weight(apc_st) == 0.0:
        sfx.set_weight(apc_st, 1.0)

    return added_constraints


def setup_pose_from_mpnn_output(mpnn_pdb_str, ref_pose, cst_io=None, sfx=None,
                                  constrain_ligand=True, handle_close_contacts=True,
                                  ref_pdb_pose=None, catres_for_covalent=None,
                                  his_tautomer_map=None,
                                  debug_output_dir=None, debug_prefix="",
                                  verbose=True):
    """
    Properly set up a pose from MPNN packed output, preserving MPNN geometry
    while optionally re-establishing covalent connections from a trusted reference.

    This is the main function to use when loading MPNN output for enzyme design.

    CRITICAL: To avoid pseudobond issues with metal-coordinating HIS residues:
    1. Load PROTEIN-ONLY (no HETATM) - prevents PyRosetta from detecting metals
    2. Apply HIS tautomers (safe - no metals = no pseudobonds)
    3. Merge back HETATM and apply constraints (creates pseudobonds with CORRECT tautomers)

    Arguments:
        mpnn_pdb_str: PDB string from MPNN packed output (with headers + HETATM)
        ref_pose: Current iteration pose (for copying labels, residue numbering)
        cst_io: EnzConstraintIO for setting up enzyme constraints
        sfx: ScoreFunction
        constrain_ligand: Add tight coordinate constraints to all ligand atoms (default: True)
        handle_close_contacts: Add constraints to allow ligand H - acid/base close contacts (default: True)
        ref_pdb_pose: Reference PDB pose for covalent connections (from --ref_pdb).
                      If None, covalent connection restoration is SKIPPED entirely.
                      This ensures we only use trusted sources for covalent bonds.
        catres_for_covalent: List of catalytic residue seqpos allowed for covalent bonds.
                             Only catres-ligand or catres-catres bonds will be restored.
        his_tautomer_map: Dict mapping (chain, resno) -> tautomer name ('HIS' or 'HIS_D').
                          If provided, HIS tautomers are applied BEFORE constraints are added.
                          This is critical for metal-coordinating HIS residues, as constraints
                          create pseudobonds that prevent later tautomer changes.
        debug_output_dir: Directory for saving debug structures at each stage
        debug_prefix: Prefix for debug output filenames
        verbose: Print progress information

    Returns:
        Pose: Properly initialized pose ready for Rosetta operations
    """
    import tempfile
    import hydrogen_utils

    if verbose:
        print("  [SetupPose] Setting up pose from MPNN output...")

    # =========================================================================
    # STEP 1: Separate protein ATOM lines from HETATM lines
    # PyRosetta auto-detects metal coordination from geometry during pose_from_file().
    # Even without REMARK 666 lines, if zinc atoms are present, pseudobonds are created.
    # Solution: Load protein-only first, apply tautomers, then add HETATM back.
    # =========================================================================
    pdb_lines = mpnn_pdb_str.split('\n')

    # Separate lines into categories
    header_lines = []      # HEADER, REMARK (except 666), HETNAM, LINK, etc.
    remark666_lines = []   # REMARK 666 lines (needed for constraints later)
    protein_lines = []     # ATOM and TER lines (protein only)
    hetatm_lines = []      # HETATM lines (ligands, metals, waters)

    for line in pdb_lines:
        if line.startswith('REMARK 666'):
            remark666_lines.append(line)
        elif line.startswith(('HEADER', 'REMARK', 'HETNAM', 'FORMUL', 'LINK', 'CRYST', 'SCALE', 'ORIGX')):
            header_lines.append(line)
        elif line.startswith('ATOM') or line.startswith('TER'):
            protein_lines.append(line)
        elif line.startswith('HETATM'):
            hetatm_lines.append(line)
        elif line.startswith('END'):
            pass  # Skip END, we'll add it later
        # Skip empty lines and other records

    if verbose:
        print(f"  [SetupPose] Separated PDB: {len(protein_lines)} protein lines, "
              f"{len(hetatm_lines)} HETATM lines, {len(remark666_lines)} REMARK 666 lines")

    # =========================================================================
    # STEP 2: Load PROTEIN-ONLY into PyRosetta (no metals = no pseudobonds!)
    # This is the key insight: without HETATM, PyRosetta cannot create pseudobonds.
    # =========================================================================
    protein_only_pdb = '\n'.join(protein_lines) + '\nEND\n'

    fd, tmp_path = tempfile.mkstemp(suffix='.pdb', prefix='mpnn_protein_only_')
    os.close(fd)

    try:
        with open(tmp_path, 'w') as f:
            f.write(protein_only_pdb)

        # Save debug output: protein-only PDB before loading
        if debug_output_dir:
            debug_path = os.path.join(debug_output_dir, f"{debug_prefix}step1_protein_only.pdb")
            with open(debug_path, 'w') as f:
                f.write(protein_only_pdb)
            if verbose:
                print(f"  [DEBUG] Saved protein-only PDB: {debug_path}")

        pose = pyr.pose_from_file(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if verbose:
        print(f"  [SetupPose] Loaded protein-only pose: {pose.size()} residues (NO metals, NO pseudobonds)")

    # =========================================================================
    # STEP 3: Apply HIS tautomers (SAFE - no pseudobonds exist!)
    # PyRosetta added hydrogens during pose_from_file() but may have guessed
    # wrong tautomers. We fix them now while mutation is still possible.
    # =========================================================================
    if his_tautomer_map:
        if verbose:
            print("  [SetupPose] Applying HIS tautomers (safe - no metals loaded yet)...")
        mutations = hydrogen_utils.apply_his_tautomers_to_pose(pose, his_tautomer_map, verbose=verbose)
        if verbose:
            print(f"  [SetupPose] Applied {len(mutations)} HIS tautomer corrections")

    # Save debug output: after tautomer application
    if debug_output_dir:
        debug_path = os.path.join(debug_output_dir, f"{debug_prefix}step2_after_tautomers.pdb")
        pose.dump_pdb(debug_path)
        if verbose:
            print(f"  [DEBUG] Saved post-tautomer pose: {debug_path}")

    # =========================================================================
    # STEP 4: Merge HETATM back and create full PDB with correct tautomers
    # Now we have a protein pose with correct HIS tautomers. We need to add
    # back the HETATM (ligands, metals) and create constraints.
    # =========================================================================

    # Export protein pose to PDB string (now has correct H atoms and tautomers)
    protein_with_H = pyr.distributed.io.to_pdbstring(pose)

    # Extract just the ATOM/TER lines from the exported pose
    protein_with_H_lines = []
    for line in protein_with_H.split('\n'):
        if line.startswith('ATOM') or line.startswith('TER'):
            protein_with_H_lines.append(line)

    # Ensure TER between protein and HETATM
    if protein_with_H_lines and not protein_with_H_lines[-1].startswith('TER'):
        protein_with_H_lines.append('TER')

    # Build full PDB: headers + REMARK 666 + protein + HETATM + END
    full_pdb_lines = header_lines + remark666_lines + protein_with_H_lines + hetatm_lines + ['END']
    full_pdb_str = '\n'.join(full_pdb_lines) + '\n'

    # Save debug output: full PDB with correct tautomers before final loading
    if debug_output_dir:
        debug_path = os.path.join(debug_output_dir, f"{debug_prefix}step3_full_pdb_fixed_tautomers.pdb")
        with open(debug_path, 'w') as f:
            f.write(full_pdb_str)
        if verbose:
            print(f"  [DEBUG] Saved full PDB with fixed tautomers: {debug_path}")

    # =========================================================================
    # STEP 5: Load full PDB (now with correct tautomers, metals will create pseudobonds)
    # =========================================================================
    fd, tmp_path = tempfile.mkstemp(suffix='.pdb', prefix='mpnn_full_')
    os.close(fd)

    try:
        with open(tmp_path, 'w') as f:
            f.write(full_pdb_str)

        pose = pyr.pose_from_file(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if verbose:
        print(f"  [SetupPose] Loaded full pose: {pose.size()} residues (with HETATM, pseudobonds now created with correct tautomers)")

    # Save debug output: after loading full PDB
    if debug_output_dir:
        debug_path = os.path.join(debug_output_dir, f"{debug_prefix}step4_pose_with_hetatm.pdb")
        pose.dump_pdb(debug_path)
        if verbose:
            print(f"  [DEBUG] Saved pose with HETATM: {debug_path}")

    # =========================================================================
    # STEP 6: Apply enzyme constraints (if not already created by pseudobonds)
    # =========================================================================

    # Re-establish covalent connections ONLY if ref_pdb_pose is provided
    # This ensures we only use trusted reference structures, not potentially clashing input
    if ref_pdb_pose is not None:
        reestablish_covalent_connections(
            pose, ref_pdb_pose,
            cst_io=cst_io, sfx=sfx,
            allowed_residues=catres_for_covalent,
            require_ligand=True,  # Only catres-ligand bonds (or catres-catres if both in subset)
            verbose=verbose
        )
    else:
        if verbose:
            print("  [Covalent] No ref_pdb provided, skipping covalent connection restoration")
            print("  [Covalent] Relying on CST file + EnzConstraintIO for enzyme constraints")
        # Still apply EnzConstraintIO if provided (for CST file constraints)
        if cst_io is not None and sfx is not None:
            try:
                cst_io.add_constraints_to_pose(pose, sfx, True)
            except Exception as e:
                if verbose:
                    print(f"  [Covalent] Warning: EnzConstraintIO failed: {e}")

    # Add coordinate constraints to ligand atoms (including H)
    if constrain_ligand and sfx is not None:
        add_ligand_coordinate_constraints(pose, sfx, sd=0.01, include_hydrogens=True, verbose=verbose)

    # Handle close contacts between ligand H and acid/base residues
    if handle_close_contacts and sfx is not None:
        contacts = find_ligand_H_close_contacts(pose, distance_cutoff=2.5, verbose=verbose)
        if contacts:
            reduce_fa_rep_for_ligand_contacts(sfx, pose, contacts, verbose=verbose)

    # Copy residue labels from ref_pose (current iteration pose, not ref_pdb_pose)
    if ref_pose is not None:
        for resno in range(1, min(pose.size(), ref_pose.size()) + 1):
            for label in ref_pose.pdb_info().get_reslabels(resno):
                pose.pdb_info().add_reslabel(resno, label)

    # Save debug output: final pose after all setup
    if debug_output_dir:
        debug_path = os.path.join(debug_output_dir, f"{debug_prefix}step5_final_pose.pdb")
        pose.dump_pdb(debug_path)
        if verbose:
            print(f"  [DEBUG] Saved final setup pose: {debug_path}")

    if verbose:
        print("  [SetupPose] Pose setup complete")

    return pose


# =============================================================================
# Sidechain Flip Functions for Catalytic Residues
# =============================================================================

# Mapping of residue types to their flippable chi angles
# These are symmetric/pseudo-symmetric functional groups that can get trapped
# in local minima during minimization
FLIPPABLE_SIDECHAINS = {
    "HIS": 2,   # χ2 - imidazole ring
    "PHE": 2,   # χ2 - phenyl ring
    "TYR": 2,   # χ2 - phenol ring
    "TRP": 2,   # χ2 - indole ring
    "ASP": 2,   # χ2 - carboxylate
    "GLU": 3,   # χ3 - carboxylate
    "ASN": 2,   # χ2 - amide
    "GLN": 3,   # χ3 - amide
    "ARG": 4,   # χ4 - guanidinium (complex, may need χ5 too)
}

# NOTE: Chemical sensitivity logic has been ABOLISHED per the design plan.
# All residues in FLIPPABLE_SIDECHAINS are now eligible for flips without
# residue-type-specific gating. Flips are instead gated by:
# - Protocol stage gate (no flips before step N)
# - Chi angle convergence
# - Persistent poor CST scores
# - Torsion_AB deviation (if applicable constraint exists)


def _angular_difference(angle1, angle2):
    """
    Calculate the angular difference between two angles in degrees.
    Handles wrap-around at +/- 180.

    Arguments:
        angle1: First angle in degrees
        angle2: Second angle in degrees

    Returns:
        float: Absolute angular difference in degrees [0, 180]
    """
    diff = abs(angle1 - angle2)
    if diff > 180:
        diff = 360 - diff
    return diff


def _extract_dihedral_target(func):
    """
    Best-effort extraction of the target dihedral angle from a Rosetta Func.
    """
    for attr in ("x0", "center", "mu", "mean"):
        if hasattr(func, attr):
            try:
                return float(getattr(func, attr)())
            except TypeError:
                try:
                    return float(getattr(func, attr))
                except (TypeError, ValueError):
                    pass
    return None


def _compute_dihedral_degrees(pose, atom_ids):
    """
    Compute dihedral angle (degrees) for four AtomIDs in a pose.
    """
    try:
        return pyrosetta.rosetta.numeric.dihedral_degrees(
            pose.xyz(atom_ids[0]),
            pose.xyz(atom_ids[1]),
            pose.xyz(atom_ids[2]),
            pose.xyz(atom_ids[3])
        )
    except Exception:
        # Fallback to numpy-based dihedral calculation
        p0 = np.array(pose.xyz(atom_ids[0]))
        p1 = np.array(pose.xyz(atom_ids[1]))
        p2 = np.array(pose.xyz(atom_ids[2]))
        p3 = np.array(pose.xyz(atom_ids[3]))

        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2

        b1 /= np.linalg.norm(b1)
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1

        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        return np.degrees(np.arctan2(y, x))


def _find_ligand_seqpos(pose):
    for idx in range(pose.size(), 0, -1):
        if pose.residue(idx).is_ligand():
            return idx
    return None


def _get_torsion_ab_constraints(pose, seqpos, ligand_seqpos):
    """
    Get torsion_AB-like dihedral constraints for a catalytic residue from the pose.

    We identify dihedral constraints that involve exactly two atoms from the ligand
    and two atoms from the catalytic residue. This matches torsion_AB behavior.
    """
    if ligand_seqpos is None:
        return []

    constraints = []
    cst_set = pose.constraint_set()
    if cst_set is None:
        return constraints

    try:
        all_constraints = cst_set.get_all_constraints()
    except Exception:
        return constraints

    for cst in all_constraints:
        if cst.__class__.__name__ != "DihedralConstraint":
            continue
        try:
            atom_ids = [cst.atom(i) for i in range(1, 5)]
        except Exception:
            continue

        resnos = [atom_id.rsd() for atom_id in atom_ids]
        if resnos.count(seqpos) != 2 or resnos.count(ligand_seqpos) != 2:
            continue

        target = _extract_dihedral_target(cst.func())
        if target is None:
            continue

        constraints.append({
            'target_angle': target,
            'atom_ids': atom_ids
        })

    return constraints


def evaluate_torsion_ab_deviation(pose, seqpos, ligand_seqpos=None, deviation_threshold=120.0):
    """
    Evaluate torsion_AB deviation for a catalytic residue.

    Returns:
        dict: {'has_torsion_ab': bool, 'deviation': float or None, 'suggests_flip': bool,
               'current': float or None, 'target': float or None}
    """
    result = {
        'has_torsion_ab': False,
        'deviation': None,
        'suggests_flip': False,
        'current': None,
        'target': None
    }

    if ligand_seqpos is None:
        ligand_seqpos = _find_ligand_seqpos(pose)
    if ligand_seqpos is None:
        return result

    constraints = _get_torsion_ab_constraints(pose, seqpos, ligand_seqpos)
    if not constraints:
        return result

    result['has_torsion_ab'] = True

    max_delta = -1.0
    for entry in constraints:
        current = _compute_dihedral_degrees(pose, entry['atom_ids'])
        target = entry['target_angle']
        delta = _angular_difference(current, target)
        if delta > max_delta:
            max_delta = delta
            result['deviation'] = delta
            result['current'] = current
            result['target'] = target

    if result['deviation'] is not None and result['deviation'] >= deviation_threshold:
        result['suggests_flip'] = True

    return result


def flip_sidechain_180(pose, seqpos, verbose=True):
    """
    Flip a residue's sidechain by 180 degrees around its terminal chi angle.

    This is useful for symmetric/pseudo-symmetric functional groups that can
    get trapped in local minima during minimization.

    Arguments:
        pose: PyRosetta Pose (modified in place)
        seqpos: Residue sequence position to flip
        verbose: Print flip information

    Returns:
        bool: True if flip was performed, False if residue is not flippable
    """
    res = pose.residue(seqpos)
    res_name3 = res.name3()

    # Check if this residue type is flippable
    if res_name3 not in FLIPPABLE_SIDECHAINS:
        if verbose:
            print(f"  [Flip] {res_name3} {seqpos} is not a flippable residue type")
        return False

    chi_to_flip = FLIPPABLE_SIDECHAINS[res_name3]

    # Check if residue has enough chi angles
    if res.nchi() < chi_to_flip:
        if verbose:
            print(f"  [Flip] {res_name3} {seqpos} does not have chi{chi_to_flip} (only {res.nchi()} chi angles)")
        return False

    # Get current chi and flip by 180 degrees
    current_chi = pose.chi(chi_to_flip, seqpos)
    new_chi = current_chi + 180.0

    # Normalize to [-180, 180]
    while new_chi > 180.0:
        new_chi -= 360.0
    while new_chi < -180.0:
        new_chi += 360.0

    pose.set_chi(chi_to_flip, seqpos, new_chi)

    if verbose:
        print(f"  [Flip] {res_name3} {seqpos}: chi{chi_to_flip} {current_chi:.1f} -> {new_chi:.1f}")

    return True


def brief_local_minimize(pose, seqpos, sfx, max_iter=10, verbose=True):
    """
    Perform a very brief local minimization of a single residue's chi angles.

    This is used after a sidechain flip to allow minor adjustments without
    extensive computation.

    Arguments:
        pose: PyRosetta Pose (modified in place)
        seqpos: Residue sequence position to minimize
        sfx: ScoreFunction
        max_iter: Maximum minimization iterations (default 10 for speed)
        verbose: Print minimization info

    Returns:
        float: Final score
    """
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover

    # Create restrictive MoveMap - only chi angles for this residue
    mm = MoveMap()
    mm.set_bb(False)  # No backbone movement
    mm.set_chi(False)  # Default all chi to false
    mm.set_chi(seqpos, True)  # Only this residue's chi angles

    # Also allow immediate neighbors (+/- 1) to adjust slightly
    if seqpos > 1:
        mm.set_chi(seqpos - 1, True)
    if seqpos < pose.size():
        mm.set_chi(seqpos + 1, True)

    # Create MinMover with very loose tolerance for speed
    min_mover = MinMover()
    min_mover.movemap(mm)
    min_mover.score_function(sfx)
    min_mover.min_type("dfpmin_armijo_nonmonotone")
    min_mover.tolerance(0.1)  # Very loose tolerance for speed
    min_mover.max_iter(max_iter)

    # Minimize
    min_mover.apply(pose)

    return sfx(pose)


def attempt_catres_sidechain_flips(pose, sfx, catalytic_residues,
                                    cst_threshold=2.0,
                                    do_minimize=True,
                                    max_min_iter=10,
                                    ref_pose=None,
                                    ligand_seqpos=None,
                                    torsion_ab_threshold=120.0,
                                    verbose=True):
    """
    Attempt 180-degree sidechain flips for catalytic residues with poor constraint satisfaction.

    This addresses the common problem where symmetric/pseudo-symmetric sidechains
    get trapped in local minima during relaxation. For each catalytic residue with
    a constraint score above the threshold, this function:
    1. Tries flipping the sidechain 180 degrees
    2. Performs a brief local minimization
    3. Accepts the flip if it improves the constraint score

    NOTE: Chemical sensitivity logic has been ABOLISHED. All residues in FLIPPABLE_SIDECHAINS
    are eligible for flips. Flips are gated at the protocol level by:
    - Protocol stage gate (no flips before step N)
    - Chi angle convergence
    - Persistent poor CST scores
    - torsion_AB deviation (must be present and far off)

    Arguments:
        pose: PyRosetta Pose (modified in place)
        sfx: ScoreFunction with constraint weights
        catalytic_residues: List of catalytic residue seqpos, or dict with seqpos as keys
        cst_threshold: Constraint score threshold to trigger flip attempt (default 2.0 REU)
        do_minimize: Whether to do brief minimization after flip (default True)
        max_min_iter: Maximum minimization iterations (default 10)
        ref_pose: Reference pose (optional, for logging purposes)
        ligand_seqpos: Ligand residue position (optional; auto-detected if None)
        torsion_ab_threshold: Min torsion_AB deviation (degrees) to suggest flip (default 120)
        verbose: Print detailed progress

    Returns:
        dict: {seqpos: {'flipped': bool, 'improved': bool, 'old_cst': float, 'new_cst': float}}
    """
    # Handle both list and dict input
    if isinstance(catalytic_residues, dict):
        cat_resnos = list(catalytic_residues.keys())
    else:
        cat_resnos = list(catalytic_residues)

    if not cat_resnos:
        if verbose:
            print("  [Flip] No catalytic residues provided")
        return {}

    if verbose:
        print(f"\n  [Flip] Attempting sidechain flips for {len(cat_resnos)} residues (threshold: {cst_threshold:.2f} REU)")

    # Get current constraint scores for catalytic residues
    cat_scores = evaluate_catalytic_constraint_scores(pose, sfx, cat_resnos, verbose=False)

    results = {}
    flips_attempted = 0
    flips_accepted = 0

    for seqpos in cat_resnos:
        res = pose.residue(seqpos)
        res_name3 = res.name3()

        # Check if this residue type is flippable
        if res_name3 not in FLIPPABLE_SIDECHAINS:
            if verbose:
                print(f"  [Flip] {res_name3} {seqpos}: not a flippable residue type, skipping")
            continue

        # Get current constraint score
        current_cst = cat_scores.get(seqpos, {}).get('total', 0.0)

        # Only try flip if constraint score is above threshold
        if current_cst < cst_threshold:
            if verbose:
                print(f"  [Flip] {res_name3} {seqpos}: cst={current_cst:.2f} (below threshold, skipping)")
            continue

        # Log torsion_AB deviation (actual gating happens at protocol level)
        torsion_ab_info = evaluate_torsion_ab_deviation(
            pose,
            seqpos,
            ligand_seqpos=ligand_seqpos,
            deviation_threshold=torsion_ab_threshold
        )
        if verbose and torsion_ab_info['has_torsion_ab']:
            deviation = torsion_ab_info['deviation']
            if deviation is None:
                deviation_str = "n/a"
            else:
                deviation_str = f"{deviation:.1f}"
            print(
                f"  [Flip] {res_name3} {seqpos}: torsion_AB deviation {deviation_str} (threshold {torsion_ab_threshold:.1f})"
            )

        flips_attempted += 1

        # Store original chi angle for potential reversion
        chi_to_flip = FLIPPABLE_SIDECHAINS[res_name3]
        original_chi = pose.chi(chi_to_flip, seqpos)
        original_score = sfx(pose)

        # Perform the flip
        if verbose:
            print(f"  [Flip] {res_name3} {seqpos}: cst={current_cst:.2f} - attempting flip...")

        flip_success = flip_sidechain_180(pose, seqpos, verbose=False)

        if not flip_success:
            results[seqpos] = {'flipped': False, 'improved': False, 'reason': 'flip_failed'}
            continue

        # Brief local minimization
        if do_minimize:
            brief_local_minimize(pose, seqpos, sfx, max_iter=max_min_iter, verbose=False)

        # Re-evaluate constraint scores
        new_cat_scores = evaluate_catalytic_constraint_scores(pose, sfx, [seqpos], verbose=False)
        new_cst = new_cat_scores.get(seqpos, {}).get('total', float('inf'))
        new_score = sfx(pose)

        # Decision: accept if constraint score improved significantly
        # (or if similar cst but better total score)
        improved = False
        delta_cst = current_cst - new_cst
        delta_score = original_score - new_score

        if new_cst < current_cst - 0.1:  # Clear improvement in constraint
            improved = True
        elif abs(new_cst - current_cst) < 0.2 and new_score < original_score - 0.5:
            # Similar constraint, but better total score
            improved = True

        if improved:
            flips_accepted += 1
            if verbose:
                print(f"  [Flip] {res_name3} {seqpos}: ACCEPTED - cst {current_cst:.2f} -> {new_cst:.2f} (delta={delta_cst:.2f})")
                print(f"         score {original_score:.2f} -> {new_score:.2f} (delta={delta_score:.2f})")
            results[seqpos] = {
                'flipped': True,
                'improved': True,
                'old_cst': current_cst,
                'new_cst': new_cst,
                'delta_cst': delta_cst,
                'old_score': original_score,
                'new_score': new_score,
                'delta_score': delta_score
            }
        else:
            # Revert the flip
            pose.set_chi(chi_to_flip, seqpos, original_chi)
            if do_minimize:
                # Re-minimize to restore original state
                brief_local_minimize(pose, seqpos, sfx, max_iter=max_min_iter, verbose=False)

            if verbose:
                print(f"  [Flip] {res_name3} {seqpos}: REJECTED - cst {current_cst:.2f} -> {new_cst:.2f} (reverted)")
            results[seqpos] = {
                'flipped': True,
                'improved': False,
                'old_cst': current_cst,
                'new_cst': new_cst,
                'reverted': True
            }

    if verbose:
        if flips_attempted > 0:
            print(f"  [Flip] Summary: {flips_accepted}/{flips_attempted} flips accepted")
        else:
            print(f"  [Flip] No residues above constraint threshold")

    return results


# =============================================================================
# Catalytic Residue RMSD and Constraint Functions
# =============================================================================

# Functional atoms for each residue type (used for RMSD calculations)
FUNCTIONAL_ATOMS = {
    "HIS": ["ND1", "NE2", "CG", "CD2", "CE1"],
    "ASP": ["OD1", "OD2", "CG"],
    "GLU": ["OE1", "OE2", "CD"],
    "SER": ["OG"],
    "THR": ["OG1"],
    "CYS": ["SG"],
    "LYS": ["NZ", "CE"],
    "ARG": ["NH1", "NH2", "NE", "CZ"],
    "TYR": ["OH", "CZ"],
    "TRP": ["NE1", "CD1", "CE2"],
    "ASN": ["OD1", "ND2", "CG"],
    "GLN": ["OE1", "NE2", "CD"],
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "MET": ["SD", "CE"],
}


def calculate_catalytic_residue_rmsd(pose, reference_pose, catalytic_residues,
                                       atoms="functional", verbose=True):
    """
    Calculate RMSD of catalytic residue atoms between pose and reference.

    This provides a direct measure of how much catalytic residues have moved,
    independent of constraint scores. Use this to track catalytic geometry
    drift through the design pipeline.

    Arguments:
        pose: Current PyRosetta Pose
        reference_pose: Reference pose (original structure)
        catalytic_residues: List or dict of catalytic residue numbers (seqpos)
        atoms: Which atoms to include:
            - "functional": Key functional atoms only (default, most relevant)
            - "sidechain": All sidechain heavy atoms
            - "backbone": Backbone atoms only (N, CA, C, O)
            - "all": All heavy atoms
        verbose: Print per-residue RMSD table

    Returns:
        dict: {
            'overall_rmsd': float,
            'per_residue': {resno: {'rmsd': float, 'n_atoms': int, 'name3': str}},
            'n_total_atoms': int
        }
    """
    if isinstance(catalytic_residues, dict):
        catres_list = list(catalytic_residues.keys())
    else:
        catres_list = list(catalytic_residues)

    all_diffs = []
    per_residue = {}

    for resno in catres_list:
        if resno > pose.size() or resno > reference_pose.size():
            if verbose:
                print(f"  [CatRes RMSD] WARNING: residue {resno} out of range, skipping")
            continue

        res = pose.residue(resno)
        ref_res = reference_pose.residue(resno)
        res_name3 = res.name3()
        res_diffs = []

        # Determine which atoms to include
        for i in range(1, min(res.natoms(), ref_res.natoms()) + 1):
            include = False

            if atoms == "functional":
                # Use functional atoms if defined, otherwise fall back to sidechain
                if res_name3 in FUNCTIONAL_ATOMS:
                    atom_name = res.atom_name(i).strip()
                    if atom_name in FUNCTIONAL_ATOMS[res_name3]:
                        include = True
                else:
                    # Fall back to all sidechain heavy atoms
                    if not res.atom_is_backbone(i) and not res.atom_is_hydrogen(i):
                        include = True

            elif atoms == "sidechain":
                if not res.atom_is_backbone(i) and not res.atom_is_hydrogen(i):
                    include = True

            elif atoms == "backbone":
                if res.atom_is_backbone(i) and not res.atom_is_hydrogen(i):
                    include = True

            elif atoms == "all":
                if not res.atom_is_hydrogen(i):
                    include = True

            if include:
                try:
                    xyz1 = np.array(res.xyz(i))
                    # Match by atom name in case of different atom ordering
                    atom_name = res.atom_name(i).strip()
                    if ref_res.has(atom_name):
                        xyz2 = np.array(ref_res.xyz(atom_name))
                        diff = np.sum((xyz1 - xyz2) ** 2)
                        res_diffs.append(diff)
                        all_diffs.append(diff)
                except:
                    pass

        if res_diffs:
            res_rmsd = np.sqrt(np.mean(res_diffs))
            per_residue[resno] = {
                'rmsd': res_rmsd,
                'n_atoms': len(res_diffs),
                'name3': res_name3
            }

    overall_rmsd = np.sqrt(np.mean(all_diffs)) if all_diffs else 0.0

    if verbose:
        print(f"\n  [CatRes RMSD] Atoms: {atoms}")
        print(f"  {'Res':<6} {'Name':<5} {'RMSD':>8} {'Atoms':>6}")
        print(f"  {'-'*30}")
        for resno in sorted(per_residue.keys()):
            info = per_residue[resno]
            rmsd_val = info['rmsd']
            # Flag residues with high RMSD
            flag = " ***" if rmsd_val > 1.5 else (" ** " if rmsd_val > 1.0 else "")
            print(f"  {resno:<6} {info['name3']:<5} {rmsd_val:>8.3f} {info['n_atoms']:>6}{flag}")
        print(f"  {'-'*30}")
        print(f"  {'OVERALL':<12} {overall_rmsd:>8.3f} {len(all_diffs):>6}")

    return {
        'overall_rmsd': overall_rmsd,
        'per_residue': per_residue,
        'n_total_atoms': len(all_diffs)
    }


def add_catalytic_residue_coordinate_constraints(pose, reference_pose, catalytic_residues,
                                                   constraint_atoms="functional",
                                                   sd=0.5, verbose=True):
    """
    Add coordinate constraints specifically for catalytic residue atoms.

    This prevents drift of catalytic residue positions during minimization,
    which the standard enzyme constraints (distance/angle/dihedral) don't
    fully prevent. The CST file typically only constrains ligand-protein
    interactions, not the absolute positions of the catalytic residues.

    Arguments:
        pose: PyRosetta Pose (modified in place)
        reference_pose: Reference pose with ideal catalytic geometry
        catalytic_residues: List or dict of catalytic residue numbers
        constraint_atoms: Which atoms to constrain:
            - "functional": Key functional atoms only (recommended)
            - "sidechain": All sidechain heavy atoms
            - "all": All heavy atoms including backbone
        sd: Standard deviation for harmonic constraint in Angstroms (default 0.5)
            Smaller = tighter constraint. 0.5 A is moderately tight.

    Returns:
        int: Number of constraints added
    """
    from pyrosetta.rosetta.core.scoring.constraints import CoordinateConstraint
    from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
    from pyrosetta.rosetta.core.id import AtomID

    # Handle input types
    if isinstance(catalytic_residues, dict):
        catres_list = list(catalytic_residues.keys())
    else:
        catres_list = list(catalytic_residues)

    if verbose:
        print(f"\n  [CatRes Coord CST] Adding coordinate constraints for {len(catres_list)} catalytic residues")
        print(f"  [CatRes Coord CST] Constraint atoms: {constraint_atoms}, sd: {sd} A")

    # Get or create constraint set
    cst_set = pose.constraint_set().clone()

    # Find a fixed reference atom (first CA in pose)
    # This is needed for CoordinateConstraint
    ref_atom_id = None
    for i in range(1, pose.size() + 1):
        res = pose.residue(i)
        if res.has("CA"):
            ref_atom_id = AtomID(res.atom_index("CA"), i)
            break

    if ref_atom_id is None:
        if verbose:
            print("  [CatRes Coord CST] ERROR: No CA atom found for reference")
        return 0

    n_constraints = 0

    for resno in catres_list:
        if resno > pose.size() or resno > reference_pose.size():
            if verbose:
                print(f"  [CatRes Coord CST] Skipping residue {resno} (out of range)")
            continue

        res = pose.residue(resno)
        ref_res = reference_pose.residue(resno)
        res_name3 = res.name3()

        # Determine which atoms to constrain
        atoms_to_constrain = []

        if constraint_atoms == "functional":
            # Key functional atoms for each residue type
            if res_name3 in FUNCTIONAL_ATOMS:
                atoms_to_constrain = FUNCTIONAL_ATOMS[res_name3]
            else:
                # Default: constrain all sidechain heavy atoms
                for i in range(1, res.natoms() + 1):
                    if not res.atom_is_backbone(i) and not res.atom_is_hydrogen(i):
                        atoms_to_constrain.append(res.atom_name(i).strip())

        elif constraint_atoms == "sidechain":
            for i in range(1, res.natoms() + 1):
                if not res.atom_is_backbone(i) and not res.atom_is_hydrogen(i):
                    atoms_to_constrain.append(res.atom_name(i).strip())

        elif constraint_atoms == "all":
            for i in range(1, res.natoms() + 1):
                if not res.atom_is_hydrogen(i):
                    atoms_to_constrain.append(res.atom_name(i).strip())

        # Add constraints
        atoms_constrained = 0
        for atom_name in atoms_to_constrain:
            if res.has(atom_name) and ref_res.has(atom_name):
                try:
                    atom_idx = res.atom_index(atom_name)
                    atom_id = AtomID(atom_idx, resno)
                    ref_xyz = ref_res.xyz(atom_name)

                    # Create harmonic constraint centered at reference position
                    harmonic = HarmonicFunc(0.0, sd)
                    coord_cst = CoordinateConstraint(atom_id, ref_atom_id, ref_xyz, harmonic)
                    cst_set.add_constraint(coord_cst)
                    n_constraints += 1
                    atoms_constrained += 1
                except Exception as e:
                    if verbose:
                        print(f"  [CatRes Coord CST] WARNING: Could not constrain {res_name3}{resno} {atom_name}: {e}")

        if verbose and atoms_constrained > 0:
            print(f"    {res_name3} {resno}: constrained {atoms_constrained} atoms")

    pose.constraint_set(cst_set)

    if verbose:
        print(f"  [CatRes Coord CST] Added {n_constraints} coordinate constraints total")

    return n_constraints


def validate_remark666_consistency(pose, remark666_lines=None, pdb_path=None, verbose=True):
    """
    Validate that REMARK 666 lines are consistent with the actual pose.

    This catches cases where the catalytic residues in the output structure
    don't match what REMARK 666 claims they should be.

    Checks:
    1. Residue numbers in REMARK 666 exist in pose
    2. Residue types match (e.g., HIS in REMARK 666 = HIS in pose)
    3. Reports any mismatches as errors

    Arguments:
        pose: PyRosetta Pose
        remark666_lines: List of REMARK 666 lines (if not provided, reads from pdb_path)
        pdb_path: Path to PDB file with REMARK 666 headers
        verbose: Print validation details

    Returns:
        dict: {
            'valid': bool,
            'errors': list of error strings,
            'warnings': list of warning strings,
            'matched_residues': list of (resno, name3) that matched
        }
    """
    result = {'valid': True, 'errors': [], 'warnings': [], 'matched_residues': []}

    # Get REMARK 666 lines
    if remark666_lines is None and pdb_path:
        remark666_lines = []
        with open(pdb_path, 'r') as f:
            for line in f:
                if "REMARK 666" in line and "MATCH TEMPLATE" in line:
                    remark666_lines.append(line)

    if not remark666_lines:
        result['warnings'].append("No REMARK 666 lines found")
        return result

    if verbose:
        print(f"\n  [REMARK666] Validating {len(remark666_lines)} REMARK 666 lines...")

    for line in remark666_lines:
        try:
            parts = line.split()
            # Parse: REMARK 666 MATCH TEMPLATE X LIG 0 MATCH MOTIF X RES 123 1 1
            # Find the MOTIF section
            if "MATCH MOTIF" not in line:
                continue

            motif_idx = parts.index("MOTIF")
            res_chain = parts[motif_idx + 1]
            res_name3 = parts[motif_idx + 2]
            res_pdbno = int(parts[motif_idx + 3])

            # Find this residue in pose
            found = False
            for seqpos in range(1, pose.size() + 1):
                pdb_info = pose.pdb_info()
                pose_chain = pdb_info.chain(seqpos)
                pose_pdbno = pdb_info.number(seqpos)

                if pose_chain == res_chain and pose_pdbno == res_pdbno:
                    found = True
                    pose_res_name3 = pose.residue(seqpos).name3()

                    # Check type match
                    if pose_res_name3 != res_name3:
                        # Special case: HIS variants
                        if res_name3 == "HIS" and pose_res_name3 in ["HIS", "HIS_D"]:
                            result['warnings'].append(
                                f"REMARK 666 says {res_name3} at {res_chain}{res_pdbno}, "
                                f"pose has {pose_res_name3} (HIS variant - OK)"
                            )
                            result['matched_residues'].append((seqpos, pose_res_name3))
                        else:
                            result['errors'].append(
                                f"Type mismatch at {res_chain}{res_pdbno}: "
                                f"REMARK 666 says {res_name3}, pose has {pose_res_name3}"
                            )
                            result['valid'] = False
                    else:
                        result['matched_residues'].append((seqpos, res_name3))
                    break

            if not found:
                result['errors'].append(
                    f"Residue {res_chain}{res_pdbno} ({res_name3}) from REMARK 666 not found in pose"
                )
                result['valid'] = False

        except (IndexError, ValueError) as e:
            result['warnings'].append(f"Failed to parse REMARK 666 line: {line.strip()[:50]}... - {e}")

    if verbose:
        if result['valid']:
            print(f"  [REMARK666] Validation PASSED - {len(result['matched_residues'])} residues matched")
        else:
            print(f"  [REMARK666] Validation FAILED:")
            for err in result['errors']:
                print(f"    ERROR: {err}")
        for warn in result['warnings']:
            print(f"    WARNING: {warn}")

    return result
