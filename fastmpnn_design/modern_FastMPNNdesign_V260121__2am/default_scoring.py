#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Default Scoring Module for Enzyme Design

This module provides default scoring functionality that auto-detects polar
atoms on ligands for H-bond analysis. It calculates useful metrics including:
- Total score and per-residue score
- Interaction ddG (corrected for covalent bonds)
- Ligand SASA (absolute and relative)
- H-bonds to polar ligand atoms (acceptors and donors)
- Shape complementarity and contact molecular surface
- No-ligand-repack analysis

This scoring module can be used as-is or customized via parameters.
Custom scoring files can also be provided to enzyme_design.py instead.

Usage:
    # As default (auto-detect polar atoms)
    scores = score_design(pose, sfx, catres)

    # With custom atoms to check
    scores = score_design(pose, sfx, catres, hbond_acceptor_atoms=["O1", "O2"],
                          hbond_donor_atoms=["H9"])

Original authors: Indrek Kalvet, Chris Norn
Refactored: 2025
"""

import os
import sys
import pandas as pd
import pyrosetta
import pyrosetta.rosetta

# Import local utilities
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import rosetta_utils


# =============================================================================
# Constants
# =============================================================================

# Elements considered polar for H-bond analysis
POLAR_ELEMENTS = {"O", "N", "S"}

# Atoms to exclude from automatic polar detection (e.g., metal ions, special atoms)
EXCLUDE_ATOMS = {"ZN", "ZN1", "ZN2", "MG", "CA", "FE", "MN", "CU", "NI", "CO"}

# Default filter criteria (can be overridden)
filters = {
    "L_SASA": [0.25, "<="],           # Relative ligand SASA
    "corrected_ddg": [-25.0, "<="],   # Interaction energy
    "sc": [0.55, ">="],               # Shape complementarity
    "nlr_totrms": [1.0, "<="],        # No-ligand-repack RMSD
}


# =============================================================================
# Utility Functions
# =============================================================================

def _get_polar_acceptor_atoms(ligand):
    """
    Auto-detect polar acceptor atoms (O, N, S) on a ligand.

    Arguments:
        ligand: PyRosetta Residue object

    Returns:
        list: Atom names that can accept H-bonds
    """
    acceptors = []
    for n in range(1, ligand.natoms() + 1):
        atom_name = ligand.atom_name(n).strip()
        if atom_name in EXCLUDE_ATOMS:
            continue
        element = ligand.atom_type(n).element()
        if element in POLAR_ELEMENTS and not ligand.atom_is_hydrogen(n):
            acceptors.append(atom_name)
    return acceptors


def _get_polar_donor_hydrogens(ligand):
    """
    Auto-detect hydrogen atoms bonded to polar atoms (potential H-bond donors).

    Arguments:
        ligand: PyRosetta Residue object

    Returns:
        list: Hydrogen atom names that can donate H-bonds
    """
    donors = []
    for n in range(1, ligand.natoms() + 1):
        if ligand.atom_is_hydrogen(n):
            # Get the atom this H is bonded to
            bonded_atoms = ligand.bonded_neighbor(n)
            for bonded_idx in bonded_atoms:
                bonded_element = ligand.atom_type(bonded_idx).element()
                if bonded_element in POLAR_ELEMENTS:
                    atom_name = ligand.atom_name(n).strip()
                    if atom_name not in EXCLUDE_ATOMS:
                        donors.append(atom_name)
                    break
    return donors


def _print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# =============================================================================
# Main Scoring Function
# =============================================================================

def score_design(pose, sfx, catres, hbond_acceptor_atoms=None, hbond_donor_atoms=None,
                 exclude_sasa_atoms=None, run_nlr=True, verbose=True):
    """
    Score a designed pose with comprehensive metrics.

    Auto-detects polar atoms on the ligand for H-bond analysis if not specified.

    Arguments:
        pose: PyRosetta Pose object
        sfx: ScoreFunction (will be modified for proper H-bond decomposition)
        catres: List of catalytic residue numbers or dict from get_matcher_residues
        hbond_acceptor_atoms: List of ligand atom names to check as H-bond acceptors
                              (default: auto-detect O, N, S atoms)
        hbond_donor_atoms: List of ligand H atoms to check as H-bond donors
                           (default: auto-detect H atoms on polar atoms)
        exclude_sasa_atoms: List of atom names to exclude from SASA calculation
                           (default: ["ZN1", "ZN2"])
        run_nlr: Run no-ligand-repack analysis (default: True)
        verbose: Print detailed output (default: True)

    Returns:
        DataFrame: Single-row DataFrame with all scores
    """
    if verbose:
        _print_header("Scoring Design")

    df_scores = pd.DataFrame()
    df_scores.at[0, "SCORE:"] = "SCORE:"

    # Get ligand information
    ligand_seqpos = pose.size()
    if not pose.residue(ligand_seqpos).is_ligand():
        raise ValueError(f"Expected ligand at position {ligand_seqpos}, got {pose.residue(ligand_seqpos).name3()}")

    ligand = pose.residue(ligand_seqpos)

    if verbose:
        print(f"  Ligand: {ligand.name3()} at position {ligand_seqpos}")

    # Handle catres input
    if isinstance(catres, dict):
        catres_list = list(catres.keys())
    else:
        catres_list = list(catres) if catres else []

    # Auto-detect polar atoms if not specified
    if hbond_acceptor_atoms is None:
        hbond_acceptor_atoms = _get_polar_acceptor_atoms(ligand)
        if verbose:
            print(f"  Auto-detected acceptor atoms: {', '.join(hbond_acceptor_atoms)}")

    if hbond_donor_atoms is None:
        hbond_donor_atoms = _get_polar_donor_hydrogens(ligand)
        if verbose:
            print(f"  Auto-detected donor H atoms: {', '.join(hbond_donor_atoms)}")

    if exclude_sasa_atoms is None:
        exclude_sasa_atoms = list(EXCLUDE_ATOMS)

    # ==========================================================================
    # Basic Rosetta Scores
    # ==========================================================================
    if verbose:
        print("\n  [1/6] Calculating Rosetta energies...")

    sfx(pose)
    for k in pose.scores:
        df_scores.at[0, k] = pose.scores[k]

    # Fix scorefunction for H-bond decomposition
    rosetta_utils.fix_scorefxn(sfx)

    # Add constraint scores if present
    if pose.constraint_set().has_constraints():
        sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("atom_pair_constraint"), 1.0)
        sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("angle_constraint"), 1.0)
        sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("dihedral_constraint"), 1.0)
        sfx(pose)
        df_scores.at[0, 'all_cst'] = sum([pose.scores[s] for s in pose.scores if "constraint" in s])
        if verbose:
            print(f"    Constraint score: {df_scores.at[0, 'all_cst']:.2f}")

    df_scores.at[0, "score_per_res"] = df_scores.at[0, "total_score"] / pose.size()

    if verbose:
        print(f"    Total score: {df_scores.at[0, 'total_score']:.1f}")
        print(f"    Score per residue: {df_scores.at[0, 'score_per_res']:.2f}")

    # ==========================================================================
    # Interaction ddG
    # ==========================================================================
    if verbose:
        print("\n  [2/6] Calculating interaction ddG...")

    df_scores.at[0, 'corrected_ddg'] = rosetta_utils.calculate_ddg(pose, sfx)

    if verbose:
        print(f"    Corrected ddG: {df_scores.at[0, 'corrected_ddg']:.1f}")

    # ==========================================================================
    # SASA Calculations
    # ==========================================================================
    if verbose:
        print("\n  [3/6] Calculating SASA...")

    try:
        # Use a simple per-atom SASA calculation that avoids problematic methods
        # This calculates SASA using PyRosetta's built-in SasaCalc
        sasa_calc = pyrosetta.rosetta.core.scoring.sasa.SasaCalc()
        sasa_calc.calculate(pose)

        # Get ligand SASA
        ligand_sasa = 0.0
        for i in range(1, ligand.natoms() + 1):
            ligand_sasa += sasa_calc.get_atom_sasa(ligand_seqpos, i)

        # Calculate free ligand SASA (create isolated ligand pose)
        ligand_pose = pyrosetta.rosetta.core.pose.Pose()
        pyrosetta.rosetta.core.pose.append_subpose_to_pose(ligand_pose, pose, pose.size(), pose.size(), 1)
        sasa_calc_free = pyrosetta.rosetta.core.scoring.sasa.SasaCalc()
        sasa_calc_free.calculate(ligand_pose)
        free_ligand_sasa = 0.0
        for i in range(1, ligand_pose.residue(1).natoms() + 1):
            free_ligand_sasa += sasa_calc_free.get_atom_sasa(1, i)

        df_scores.at[0, 'L_SASA'] = ligand_sasa / free_ligand_sasa if free_ligand_sasa > 0 else 0
        df_scores.at[0, 'L_SASA_abs'] = ligand_sasa

        # Substrate SASA (excluding specified atoms like metals)
        substrate_sasa = 0.0
        for n in range(1, ligand.natoms() + 1):
            atom_name = ligand.atom_name(n).strip()
            element = ligand.atom_type(n).element()
            if atom_name not in exclude_sasa_atoms and element != "H":
                substrate_sasa += sasa_calc.get_atom_sasa(ligand_seqpos, n)

        df_scores.at[0, 'substrate_SASA'] = substrate_sasa

        if verbose:
            print(f"    Relative ligand SASA: {df_scores.at[0, 'L_SASA']:.3f}")
            print(f"    Absolute ligand SASA: {df_scores.at[0, 'L_SASA_abs']:.1f} A^2")
            print(f"    Substrate SASA: {df_scores.at[0, 'substrate_SASA']:.1f} A^2")

    except Exception as e:
        # SASA calculation failed - set to NaN and continue
        if verbose:
            print(f"    WARNING: SASA calculation failed: {str(e)[:100]}")
            print(f"    Setting SASA values to NaN")
        df_scores.at[0, 'L_SASA'] = float('nan')
        df_scores.at[0, 'L_SASA_abs'] = float('nan')
        df_scores.at[0, 'substrate_SASA'] = float('nan')

    # ==========================================================================
    # H-bond Analysis
    # ==========================================================================
    if verbose:
        print("\n  [4/6] Analyzing H-bonds to ligand...")

    total_acceptor_hbonds = 0
    total_donor_hbonds = 0

    # Check acceptor atoms (receiving H from protein)
    for atom in hbond_acceptor_atoms:
        if ligand.has(atom):
            hb_count = rosetta_utils.find_hbonds_to_residue_atom(pose, ligand_seqpos, atom)
            df_scores.at[0, f"hbond_{atom}"] = hb_count
            total_acceptor_hbonds += hb_count
            if verbose and hb_count > 0:
                print(f"    {atom}: {hb_count} H-bond(s)")
        else:
            if verbose:
                print(f"    Warning: Atom {atom} not found in ligand")

    # Check donor hydrogens (donating to protein acceptors)
    for atom in hbond_donor_atoms:
        if ligand.has(atom):
            hb_count = rosetta_utils.find_hbonds_to_residue_atom(pose, ligand_seqpos, atom)
            df_scores.at[0, f"hbond_{atom}"] = hb_count
            total_donor_hbonds += hb_count
            if verbose and hb_count > 0:
                print(f"    {atom}: {hb_count} H-bond(s)")
        else:
            if verbose:
                print(f"    Warning: Atom {atom} not found in ligand")

    df_scores.at[0, 'total_acceptor_hbonds'] = total_acceptor_hbonds
    df_scores.at[0, 'total_donor_hbonds'] = total_donor_hbonds
    df_scores.at[0, 'total_hbonds'] = total_acceptor_hbonds + total_donor_hbonds

    if verbose:
        print(f"    Total acceptor H-bonds: {total_acceptor_hbonds}")
        print(f"    Total donor H-bonds: {total_donor_hbonds}")

    # ==========================================================================
    # Shape Complementarity and Contact Molecular Surface
    # ==========================================================================
    if verbose:
        print("\n  [5/6] Calculating shape metrics...")

    lig_sel = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(ligand_seqpos)
    protein_sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector("A")

    # Contact Molecular Surface
    cms = pyrosetta.rosetta.protocols.simple_filters.ContactMolecularSurfaceFilter()
    cms.use_rosetta_radii(True)
    cms.distance_weight(0.5)
    cms.selector1(protein_sel)
    cms.selector2(lig_sel)
    df_scores.at[0, "cms"] = cms.compute(pose)
    df_scores.at[0, "cms_per_atom"] = df_scores.at[0, "cms"] / ligand.natoms()

    # Shape Complementarity
    sc = pyrosetta.rosetta.protocols.simple_filters.ShapeComplementarityFilter()
    sc.use_rosetta_radii(True)
    sc.selector1(protein_sel)
    sc.selector2(lig_sel)
    df_scores.at[0, "sc"] = sc.score(pose)

    if verbose:
        print(f"    Shape complementarity: {df_scores.at[0, 'sc']:.3f}")
        print(f"    Contact molecular surface: {df_scores.at[0, 'cms']:.1f}")
        print(f"    CMS per atom: {df_scores.at[0, 'cms_per_atom']:.2f}")

    # ==========================================================================
    # No-Ligand-Repack Analysis
    # ==========================================================================
    if run_nlr:
        if verbose:
            print("\n  [6/6] Running no-ligand-repack analysis...")

        nlr_sfx = pyrosetta.get_fa_scorefxn()
        nlr_scores = rosetta_utils.no_ligand_repack(pose, nlr_sfx, ligand_resno=ligand_seqpos, verbose=verbose)

        for k in nlr_scores.keys():
            df_scores.at[0, k] = nlr_scores.iloc[0][k]
    else:
        if verbose:
            print("\n  [6/6] Skipping no-ligand-repack analysis")

    if verbose:
        _print_header("Scoring Complete")
        print(f"  Key metrics:")
        print(f"    Total score: {df_scores.at[0, 'total_score']:.1f}")
        print(f"    ddG: {df_scores.at[0, 'corrected_ddg']:.1f}")
        print(f"    Shape complementarity: {df_scores.at[0, 'sc']:.3f}")
        print(f"    Total H-bonds: {df_scores.at[0, 'total_hbonds']:.0f}")
        if run_nlr:
            print(f"    NLR RMSD: {df_scores.at[0, 'nlr_totrms']:.3f}")

    return df_scores


# =============================================================================
# Filter Function
# =============================================================================

def filter_scores(scores, filter_dict=None):
    """
    Filter scores based on criteria.

    Arguments:
        scores: DataFrame with scores
        filter_dict: dict of filter criteria (default: module-level filters)
                    Format: {score_name: [value, comparison_operator]}
                    e.g., {"total_score": [-200.0, "<="]}

    Returns:
        DataFrame: Filtered scores
    """
    if filter_dict is None:
        filter_dict = filters

    return rosetta_utils.filter_scores(scores, filter_dict)


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score a protein-ligand complex")
    parser.add_argument("--pdb", required=True, help="Input PDB file")
    parser.add_argument("--params", nargs="+", help="Ligand params file(s)")
    parser.add_argument("--no_nlr", action="store_true", help="Skip no-ligand-repack analysis")
    args = parser.parse_args()

    # Initialize PyRosetta
    extra_res = ""
    if args.params:
        extra_res = "-extra_res_fa " + " ".join(args.params)
    pyrosetta.init(f"{extra_res} -beta_nov16")

    # Load and score
    pose = pyrosetta.pose_from_file(args.pdb)
    sfx = pyrosetta.get_fa_scorefxn()
    catres = rosetta_utils.get_matcher_residues(args.pdb)

    scores = score_design(pose, sfx, catres, run_nlr=not args.no_nlr)

    print("\n\nFull scores:")
    for k in scores.keys():
        if k != "SCORE:":
            print(f"  {k}: {scores.at[0, k]}")
