#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastMPNNdesign - Protein sequence design using ProteinMPNN with PyRosetta integration.

This module provides FastDesign-like protein sequence design using ProteinMPNN
(via mpnn_runner) for sequence generation combined with Rosetta minimization and
packing. It supports iterative design protocols with customizable schedules,
ligand context awareness, and various task operations for fine-grained control
over designable positions.

Original author: ikalvet (2023)
"""

# =============================================================================
# IMPORTS
# =============================================================================
import copy
import io
import os
import sys
import time

import pandas as pd
import pyrosetta
import pyrosetta.distributed.io
import pyrosetta.rosetta

# =============================================================================
# PATH SETUP
# =============================================================================
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(SCRIPT_DIR)

# MPNN Runner - unified interface for ProteinMPNN/LigandMPNN
from mpnn_runner import MPNNRunner, restype_1to3

# Import constants and utilities
from constants import (
    DEFAULT_MODEL_TYPE, DEFAULT_OMIT_AA, DEFAULT_LIGAND_RIGIDITY,
    DEFAULT_KEEP_BEST_MODE, DEFAULT_CST_COMPARABLE_THRESHOLD, KEEP_BEST_MODES,
    DEFAULT_2ND_LAYER_TEMPS, DEFAULT_2ND_LAYER_BATCH_SIZE, DEFAULT_PROTOCOL
)
import rosetta_utils


# =============================================================================
# FLIP TRACKING
# =============================================================================

class FlipTracker:
    """Aggregate flip statistics across poses and protocol steps."""

    def __init__(self):
        self.history = []

    def record(self, flip_results, step_name, pose_idx):
        """
        Record flip results from a single operation.

        Arguments:
            flip_results: Dict from attempt_catres_sidechain_flips()
                          {seqpos: {'flipped': bool, 'improved': bool, 'old_cst': float, 'new_cst': float}}
            step_name: Name of the protocol step (e.g., "min_step3", "flip_catres_step5")
            pose_idx: Index of the pose being processed
        """
        if flip_results:
            self.history.append({
                'step': step_name,
                'pose_idx': pose_idx,
                'results': flip_results
            })

    def get_stats(self):
        """
        Calculate aggregate statistics across all recorded flips.

        Returns:
            dict: Statistics including attempted, accepted, rejected counts,
                  average and total improvement in REU
        """
        attempted = 0
        accepted = 0
        rejected = 0
        improvements = []

        for entry in self.history:
            for seqpos, result in entry['results'].items():
                if result.get('flipped'):
                    attempted += 1
                    if result.get('improved'):
                        accepted += 1
                        old_cst = result.get('old_cst', 0)
                        new_cst = result.get('new_cst', 0)
                        improvement = old_cst - new_cst
                        if improvement > 0:
                            improvements.append(improvement)
                    else:
                        rejected += 1

        import numpy as np
        return {
            'attempted': attempted,
            'accepted': accepted,
            'rejected': rejected,
            'avg_improvement': float(np.mean(improvements)) if improvements else 0.0,
            'total_improvement': sum(improvements) if improvements else 0.0
        }

    def print_summary(self):
        """Print a formatted summary table of flip statistics."""
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print(f"  SIDECHAIN FLIP SUMMARY")
        print(f"{'='*60}")
        print(f"  Flips Attempted:     {stats['attempted']:>8}")
        print(f"  Flips Accepted:      {stats['accepted']:>8}")
        print(f"  Flips Rejected:      {stats['rejected']:>8}")
        if stats['attempted'] > 0:
            rate = stats['accepted'] / stats['attempted'] * 100
            print(f"  Acceptance Rate:     {rate:>7.1f}%")
        print(f"  Avg CST Improvement: {stats['avg_improvement']:>7.2f} REU")
        print(f"  Total CST Improvement:{stats['total_improvement']:>6.2f} REU")
        print(f"{'='*60}")


# =============================================================================
# SCORING UTILITIES
# =============================================================================

def get_constraint_score(pose, sfx):
    """
    Calculate total constraint score for a pose.

    Returns:
        float: Sum of atom_pair_constraint, angle_constraint, dihedral_constraint
    """
    sfx(pose)  # Score the pose
    energies = pose.energies()

    cst_terms = [
        'atom_pair_constraint',
        'angle_constraint',
        'dihedral_constraint'
    ]

    total_cst = 0.0
    for term in cst_terms:
        try:
            st = pyrosetta.rosetta.core.scoring.score_type_from_name(term)
            total_cst += energies.total_energies()[st]
        except:
            pass

    return total_cst


def get_detailed_scores(pose, sfx):
    """
    Get detailed score breakdown for a pose.

    Returns:
        dict: {'total': float, 'cst': float, 'cst_breakdown': dict, 'other': float}
    """
    sfx(pose)
    energies = pose.energies()

    cst_terms = ['atom_pair_constraint', 'angle_constraint', 'dihedral_constraint']
    cst_breakdown = {}
    total_cst = 0.0

    for term in cst_terms:
        try:
            st = pyrosetta.rosetta.core.scoring.score_type_from_name(term)
            val = energies.total_energies()[st]
            cst_breakdown[term] = val
            total_cst += val
        except:
            cst_breakdown[term] = 0.0

    total = sfx(pose)

    return {
        'total': total,
        'cst': total_cst,
        'cst_breakdown': cst_breakdown,
        'other': total - total_cst
    }


def select_best_poses(poses, sfx, n_keep, mode="cst_priority", cst_threshold=2.0, verbose=True):
    """
    Select best poses based on specified scoring mode.

    Arguments:
        poses: dict of {idx: {"pose": pose, "mpnn_input": input}}
        sfx: ScoreFunction
        n_keep: Number of poses to keep
        mode: Scoring mode - "cst_priority", "total_score", or "cst_only"
        cst_threshold: Threshold for "comparable" constraint scores
        verbose: Print detailed scoring info

    Returns:
        list: Indices of poses to keep (sorted by score)
    """
    if len(poses) <= n_keep:
        return list(poses.keys())

    # Get detailed scores for all poses
    scores = {}
    for idx, p in poses.items():
        scores[idx] = get_detailed_scores(p["pose"], sfx)

    if verbose:
        print(f"\n[keep_best] Scoring {len(poses)} poses (mode={mode}):")
        print(f"  {'Pose':<6} {'Total':>10} {'CST':>10} {'AtomPair':>10} {'Angle':>10} {'Dihedral':>10} {'Other':>10}")
        print(f"  {'-'*68}")
        for idx in sorted(scores.keys()):
            s = scores[idx]
            cb = s['cst_breakdown']
            print(f"  {idx:<6} {s['total']:>10.2f} {s['cst']:>10.2f} {cb.get('atom_pair_constraint', 0):>10.2f} {cb.get('angle_constraint', 0):>10.2f} {cb.get('dihedral_constraint', 0):>10.2f} {s['other']:>10.2f}")

        # Print statistics
        cst_scores = [s['cst'] for s in scores.values()]
        total_scores = [s['total'] for s in scores.values()]
        print(f"  {'-'*68}")
        print(f"  CST   - Min: {min(cst_scores):.2f}, Max: {max(cst_scores):.2f}, Range: {max(cst_scores)-min(cst_scores):.2f}")
        print(f"  Total - Min: {min(total_scores):.2f}, Max: {max(total_scores):.2f}")

    # Select based on mode
    if mode == "total_score":
        # Original behavior - sort by total score
        sorted_idx = sorted(scores.keys(), key=lambda x: scores[x]['total'])
        if verbose:
            print(f"  [keep_best] Using total_score mode")

    elif mode == "cst_only":
        # Sort by constraint score only
        sorted_idx = sorted(scores.keys(), key=lambda x: scores[x]['cst'])
        if verbose:
            print(f"  [keep_best] Using cst_only mode")

    elif mode == "cst_priority":
        # Prioritize constraint score, fall back to total if comparable
        cst_scores = [scores[idx]['cst'] for idx in scores.keys()]
        cst_range = max(cst_scores) - min(cst_scores)
        all_low = all(s < cst_threshold for s in cst_scores)

        if all_low or cst_range < cst_threshold:
            # Constraint scores are comparable or all low - use total score
            sorted_idx = sorted(scores.keys(), key=lambda x: scores[x]['total'])
            if verbose:
                if all_low:
                    print(f"  [keep_best] CST scores all low (<{cst_threshold}) - using total_score")
                else:
                    print(f"  [keep_best] CST range ({cst_range:.2f}) < threshold ({cst_threshold}) - using total_score")
        else:
            # Constraint scores differ significantly - prioritize CST
            # Use CST as primary, total as tiebreaker
            sorted_idx = sorted(scores.keys(), key=lambda x: (scores[x]['cst'], scores[x]['total']))
            if verbose:
                print(f"  [keep_best] CST scores differ significantly - prioritizing constraint score")

    else:
        raise ValueError(f"Unknown keep_best mode: {mode}. Valid modes: {KEEP_BEST_MODES}")

    selected = sorted_idx[:n_keep]
    if verbose:
        print(f"  [keep_best] Keeping poses: {selected}")

    return selected


def get_2nd_shell_fixed_positions(pose, ligand_seqpos, heavyatoms, keep_pos):
    """
    Determine fixed positions for 2nd shell MPNN design.

    Fixes inner shell residues (close to ligand, H-bonders, catalytic)
    and designs the outer shell.

    Arguments:
        pose: PyRosetta Pose
        ligand_seqpos: Ligand residue number
        heavyatoms: Ligand heavy atom names
        keep_pos: Positions to always keep fixed (catalytic, etc.)

    Returns:
        tuple: (fixed_residues_list, design_residues_list)
               Both as chain+resno strings like ["A45", "A46", ...]
    """
    dist_bb = 6.0
    dist_sc = 5.0

    # Get H-bond keeper labeled positions
    motif_sel = pyrosetta.rosetta.core.select.residue_selector.ResiduePDBInfoHasLabelSelector(
        label_str="keep_hbonds_to_ligand_and_catres"
    )
    pocket_positions = list(keep_pos) + list(pyrosetta.rosetta.core.select.get_residue_set_from_subset(motif_sel.apply(pose)))
    pocket_positions = list(set(pocket_positions))

    # Get layer selections
    _, _, _, residues = rosetta_utils.get_layer_selections(
        pose, repack_only_pos=pocket_positions, design_pos=[], ref_resno=ligand_seqpos,
        heavyatoms=heavyatoms, cuts=[dist_bb, dist_bb+2.0, dist_bb+4.0, dist_bb+6.0], design_GP=True
    )

    # Add residues with close side-chains
    close_ones = rosetta_utils.get_residues_with_close_sc(
        pose, heavyatoms, residues[1] + residues[2], exclude_residues=pocket_positions, cutoff=dist_sc
    )
    pocket_positions += residues[0] + close_ones
    pocket_positions = list(set(pocket_positions))

    # Design residues are outer shell (not in pocket)
    design_residues = [x for x in residues[0] + residues[1] + residues[2] + residues[3] if x not in pocket_positions]

    # Include alanines not in pocket
    ala_positions = [res.seqpos() for res in pose.residues
                     if res.seqpos() not in pocket_positions + design_residues and res.name3() == "ALA"]
    design_residues += ala_positions

    # Convert to chain+resno format
    def to_chain_resno(resno):
        chain = pose.pdb_info().chain(resno)
        pdb_resno = pose.pdb_info().number(resno)
        return f"{chain}{pdb_resno}"

    fixed_residues = [to_chain_resno(r.seqpos()) for r in pose.residues
                      if r.seqpos() not in design_residues and r.is_protein()]
    design_residues_str = [to_chain_resno(r) for r in design_residues]

    return fixed_residues, design_residues_str


# =============================================================================
# MAIN CLASS
# =============================================================================
class FastMPNNdesign():
    def __init__(self, model_type=None, N_seq=5, params=None, name=None,
                 scorefxn=None, min_type="lbfgs_armijo_nonmonotone", script_file=None, taskfactory=None,
                 cartesian=False,
                 design_positions=None, repack_positions=None, do_not_repack_positions=None, omit_AA=None, cst_io=None, debug=False,
                 mpnn_pack_sc=True, ligand_mpnn_use_side_chain_context=True,
                 ligand_rigidity=DEFAULT_LIGAND_RIGIDITY,
                 keep_best_mode=DEFAULT_KEEP_BEST_MODE,
                 cst_comparable_threshold=DEFAULT_CST_COMPARABLE_THRESHOLD,
                 enhance_model=None,
                 debug_output_dir=None,
                 cache_mpnn_models=True,
                 use_mpnn_server=False,
                 mpnn_server_host="localhost",
                 mpnn_server_port=5000):

        # Attributes user can set
        self.__mpnnrunner = MPNNRunner(model_type, verbose=True, pack_sc=mpnn_pack_sc,
                                       ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
                                       enhance_model=enhance_model,
                                       cache_models=cache_mpnn_models,
                                       use_server=use_mpnn_server,
                                       server_host=mpnn_server_host,
                                       server_port=mpnn_server_port)
        self.__num_sequences = N_seq
        self.__num_sequences_original = N_seq
        self.__params_files = params
        self.__min_type = min_type
        self.__script = self._setup_schedule(script_file)
        self.__cst_io = cst_io
        self.__minimizer_rmsd_cutoff = 3.0
        self.__ligand_rigidity = ligand_rigidity
        self.__keep_best_mode = keep_best_mode
        self.__cst_comparable_threshold = cst_comparable_threshold

        if scorefxn is None:
            self.__scorefxn = pyrosetta.get_fa_scorefxn()
        else:
            self.__scorefxn = scorefxn

        self.__tf = taskfactory
        self.__cartesian = cartesian
        self.__design_positions = design_positions
        self.__repack_positions = repack_positions
        self.__do_not_repack_positions = do_not_repack_positions
        self.__name = name
        if name is None:
            self.__name = "pose_0000"
        self.__MPNN_pack_sc = mpnn_pack_sc
        self.__debug = debug
        self.__debug_output_dir = debug_output_dir
        self.__bias_AAs = None
        self.__bias_AAs_per_residue = None
        self.__omit_AA = omit_AA
        self.__task_operations = {}

        # 2nd shell MPNN configuration
        self.__ligand_seqpos = None
        self.__ligand_heavyatoms = None
        self.__keep_positions = None

        # Reference PDB for covalent connection restoration (from --ref_pdb)
        # Only used if explicitly provided - ensures trusted source for covalent bonds
        self.__ref_pdb_pose = None
        self.__catres_for_covalent = None

        # Sidechain flip configuration for catalytic residues
        # Addresses symmetric/pseudo-symmetric sidechains trapped in local minima
        self.__enable_auto_flip = True  # Automatic flip attempts after min/repack
        self.__flip_cst_threshold = 1.0  # Constraint score threshold to trigger flip attempt
        self.__flip_max_min_iter = 10  # Max iterations for brief local minimization after flip

        # Flip tracking for aggregate statistics
        self.__flip_tracker = FlipTracker()

        # Information stored at runtime and not meant to be settable
        self.__input_pose = None
        self.__not_design_pos_list = None
        self.__design_pos_list = None
        self.__movers = {}
        self.__movemap = None
        self.__mpnn_input = None
        self.__mpnn_N_seq_after_first = 1

    # =========================================================================
    # GETTERS AND SETTERS
    # =========================================================================
    def mpnnrunner(self):
        return self.__mpnnrunner

    def set_mpnn_N_seq_after_first(self, N):
        """
        How many sequences will be designed with MPNN after the first application of MPNN
        """
        self.__mpnn_N_seq_after_first = N

    def minimizer_rmsd_cutoff(self, cutoff=None):
        """
        Sets or returns the minimizer rmsd cutoff
        """
        if cutoff is None:
            return self.__minimizer_rmsd_cutoff
        else:
            assert isinstance(cutoff, float)
            self.__minimizer_rmsd_cutoff = cutoff

    def set_minimizer_movemap(self, movemap):
        assert isinstance(movemap, pyrosetta.rosetta.core.kinematics.MoveMap)
        self.__movemap = movemap

    def MPNN_pack_sc(self, enable_pack=None):
        """
        if enable_pack is None then returns the stored value for MPNN_pack_sc
        if enable_pack is bool then sets MPNN_pack_sc to that value
        """
        assert isinstance(enable_pack, (bool, type(None)))
        if enable_pack is None:
            return self.__MPNN_pack_sc
        else:
            self.__MPNN_pack_sc = enable_pack

    def ligand_rigidity(self, mode=None):
        """
        Get or set ligand rigidity mode.

        Arguments:
            mode: One of "fixed", "rigid_body", "flexible" or None to get current value

        Returns:
            str: Current ligand rigidity mode (if mode is None)
        """
        if mode is None:
            return self.__ligand_rigidity
        else:
            valid_modes = ["fixed", "rigid_body", "flexible"]
            assert mode in valid_modes, f"ligand_rigidity must be one of {valid_modes}"
            self.__ligand_rigidity = mode

    def keep_best_mode(self, mode=None):
        """
        Get or set keep_best scoring mode.

        Arguments:
            mode: One of "cst_priority", "total_score", "cst_only" or None to get current value

        Returns:
            str: Current keep_best mode (if mode is None)
        """
        if mode is None:
            return self.__keep_best_mode
        else:
            assert mode in KEEP_BEST_MODES, f"keep_best_mode must be one of {KEEP_BEST_MODES}"
            self.__keep_best_mode = mode

    def cst_comparable_threshold(self, threshold=None):
        """
        Get or set the threshold for 'comparable' constraint scores in cst_priority mode.

        Arguments:
            threshold: Threshold in REU, or None to get current value

        Returns:
            float: Current threshold (if threshold is None)
        """
        if threshold is None:
            return self.__cst_comparable_threshold
        else:
            assert isinstance(threshold, (int, float))
            self.__cst_comparable_threshold = float(threshold)

    def add_task_operation(self, taskop):
        """
        Adds a taskoperation instance to the method.
        The TaskOperation can be any arbitrary Python object that has the
        following methods implemented:
            compute(pose) -> list
            target() -> list
            target(list) :: sets a new value as target
            allow_updating() -> bool
            name() -> str
            copy() -> obj
        The method 'compute' takes 'pose' as argument and returns a list of
        residue numbers that would then be used to update the list of residues for MPNN
        """
        assert hasattr(taskop, "target")
        assert hasattr(taskop, "compute")
        assert hasattr(taskop, "allow_updating")
        assert hasattr(taskop, "name")
        assert hasattr(taskop, "copy")
        self.__task_operations[taskop.name()] = taskop.copy()

    def set_2nd_shell_config(self, ligand_seqpos, heavyatoms, keep_positions):
        """
        Configure 2nd shell MPNN design parameters.

        Arguments:
            ligand_seqpos: Ligand residue number
            heavyatoms: List of ligand heavy atom names
            keep_positions: List of positions to always keep fixed (catalytic, etc.)
        """
        self.__ligand_seqpos = ligand_seqpos
        self.__ligand_heavyatoms = heavyatoms
        self.__keep_positions = list(keep_positions) if keep_positions else []

    def set_covalent_reference(self, ref_pdb_pose, catres_seqpos_list):
        """
        Set the reference pose and allowed catalytic residues for covalent bond restoration.

        This should only be called when --ref_pdb is provided, to ensure covalent
        connections are only restored from a trusted reference structure.

        Arguments:
            ref_pdb_pose: Pose from --ref_pdb (aligned to input by ligand atoms)
            catres_seqpos_list: List of catalytic residue seqpos allowed for covalent bonds
                                (from catres_cst_subset, or all REMARK 666 catres if no subset)
        """
        self.__ref_pdb_pose = ref_pdb_pose
        self.__catres_for_covalent = list(catres_seqpos_list) if catres_seqpos_list else None
        print(f"[FastMPNN] Covalent reference set: {len(catres_seqpos_list) if catres_seqpos_list else 0} catalytic residues")

    def set_catres_flip_config(self, enable_auto=True, cst_threshold=1.0, max_min_iter=10):
        """
        Configure sidechain flip attempts for catalytic residues.

        Symmetric/pseudo-symmetric sidechains (HIS, PHE, TYR, ASP, GLU, ASN, GLN, TRP, ARG)
        can get trapped in local minima during minimization. This feature attempts 180-degree
        flips for catalytic residues with poor constraint satisfaction.

        Arguments:
            enable_auto: Enable automatic flip attempts after min/repack operations (default True)
            cst_threshold: Constraint score threshold to trigger flip attempt (default 1.0 REU)
                           Only residues with constraint scores above this are flipped.
            max_min_iter: Maximum iterations for brief local minimization after flip (default 10)
        """
        self.__enable_auto_flip = enable_auto
        self.__flip_cst_threshold = cst_threshold
        self.__flip_max_min_iter = max_min_iter
        status = "enabled" if enable_auto else "disabled"
        print(f"[FastMPNN] Catres sidechain flip: {status}, threshold={cst_threshold:.2f} REU")

    def scorefxn(self):
        return self.__scorefxn

    def script(self):
        return self.__script

    def mpnn_input(self):
        return self.__mpnn_input

    def set_mpnn_bias(self, bias_dict):
        """
        dict of per-AA bias
        """
        self.__bias_AAs = bias_dict

    def set_mpnn_bias_per_residue(self, bias_dict):
        """
        dict of per-AA bias per position
        Keys must be in format {chain}{resno}
        """
        for k in bias_dict.keys():
            assert isinstance(k, str)
            assert not k[0].isnumeric()
        self.__bias_AAs_per_residue = bias_dict

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    def _fix_mpnn_packed_pdb(self, mpnn_pdb_str: str, ref_pdb_str: str) -> str:
        """
        Fix MPNN packed PDB by restoring missing header information from reference PDB.

        MPNN outputs include the ligand HETATM but are missing:
        - REMARK 666 lines (enzyme constraint info)
        - HEADER, HETNAM, LINK lines

        This function restores these headers from the reference PDB string.

        Args:
            mpnn_pdb_str: PDB string from MPNN packed output (has protein + ligand)
            ref_pdb_str: Original PDB string (with REMARK 666, headers, etc.)

        Returns:
            Fixed PDB string with headers prepended
        """
        # Extract header lines from reference (REMARK 666, HEADER, HETNAM, LINK)
        header_lines = []
        seen_headers = set()

        for line in ref_pdb_str.split('\n'):
            if line.startswith(("HEADER", "REMARK", "HETNAM", "LINK")):
                if line not in seen_headers:
                    header_lines.append(line)
                    seen_headers.add(line)

        # Extract protein ATOM lines and HETATM lines from MPNN output
        protein_lines = []
        hetatm_lines = []
        has_ter = False

        for line in mpnn_pdb_str.split('\n'):
            if line.startswith("ATOM"):
                protein_lines.append(line)
            elif line.startswith("TER"):
                has_ter = True
                protein_lines.append(line)
            elif line.startswith("HETATM"):
                hetatm_lines.append(line)

        # Ensure TER record exists between protein and HETATM
        if protein_lines and hetatm_lines and not has_ter:
            protein_lines.append("TER")

        # Combine: headers + protein + HETATM + END
        final_lines = header_lines + protein_lines + hetatm_lines
        if not any(line.startswith("END") for line in final_lines):
            final_lines.append("END")

        return '\n'.join(final_lines)

    def _extract_sequence_from_pdb_str(self, pdb_str: str) -> str:
        """Extract protein sequence from PDB string (CA atoms only, sorted by chain/resno)."""
        residues = {}  # (chain, resno) -> resname
        restype_3to1 = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }

        for line in pdb_str.split('\n'):
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                try:
                    chain = line[21]
                    resno = int(line[22:26].strip())
                    resname = line[17:20].strip()
                    residues[(chain, resno)] = resname
                except (ValueError, IndexError):
                    continue

        # Sort by chain then residue number and convert to 1-letter codes
        sorted_residues = sorted(residues.items(), key=lambda x: (x[0][0], x[0][1]))
        sequence = "".join(restype_3to1.get(resname, 'X') for (chain, resno), resname in sorted_residues)
        return sequence

    def _build_pose_from_str_and_append_stuff(self, pdb_str, append_pose=None, append_pose_resnos=None, prepend_lines=None, append_lines=None, ref_pose=None):
        if prepend_lines is None:
            prepend_lines = []
        if append_lines is None:
            append_lines = []

        _pdb = "\n".join(prepend_lines) + pdb_str + "\n".join(append_lines)
        _pose = pyrosetta.rosetta.core.pose.Pose()
        pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(_pose, _pdb)

        # Adjusting residue PDB numbering based on a reference pose
        if ref_pose is not None:
            for res in _pose.residues:
                _pose.pdb_info().number(res.seqpos(), ref_pose.pdb_info().number(res.seqpos()))

        if append_pose is not None and append_pose_resnos is not None:
            for append_pose_resno in append_pose_resnos:
                pyrosetta.rosetta.core.pose.append_subpose_to_pose(_pose, append_pose, append_pose_resno, append_pose_resno, 1)

        # Apply user-provided constraint mover
        if self.__cst_io is not None:
            if isinstance(self.__cst_io, pyrosetta.rosetta.protocols.toolbox.match_enzdes_util.EnzConstraintIO):
                self.__cst_io.add_constraints_to_pose(_pose, self.scorefxn(), True)
                constrained_residues = self.__cst_io.ordered_constrained_positions(_pose)
                self.__cst_io.remove_constraints_from_pose(_pose, True, True)
                _pose.constraint_set().clear()
                _pose.constraint_set().clear_sequence_constraints()
                # Re-adjusting the rotamers of constrained residues because MPNN-packer can mess them up
                for resno in constrained_residues:
                    if _pose.residue(resno).is_ligand():
                        continue
                    # Making sure it's the same HIS tautomer
                    if _pose.residue(resno).name3() == "HIS" and (_pose.residue(resno).name() != ref_pose.residue(resno).name()):
                        print(f"[FastMPNN] Mutating residue {resno} from {_pose.residue(resno).name()} to {ref_pose.residue(resno).name()}")
                        mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
                        mutres.set_res_name(ref_pose.residue(resno).name())
                        mutres.set_target(resno)
                        mutres.apply(_pose)
                        print(f"[FastMPNN] Mutated residue {resno} to {_pose.residue(resno).name()}")

                    for chino in range(1, _pose.residue(resno).nchi()+1):
                        print(f"[FastMPNN] Changing chi {chino} from {_pose.residue(resno).chi(chino)} to {ref_pose.residue(resno).chi(chino)}")
                        _pose.residue(resno).set_chi(chino, ref_pose.residue(resno).chi(chino))
                self.__cst_io.add_constraints_to_pose(_pose, self.scorefxn(), True)

        return _pose

    def _setup_schedule(self, script_file):
        script = None
        script_list = []

        # Use DEFAULT_PROTOCOL if no script provided
        if script_file is None:
            script_file = DEFAULT_PROTOCOL
            print("[FastMPNN] Using default protocol from constants.py")

        if isinstance(script_file, str):
            if os.path.exists(script_file):
                print(f"[FastMPNN] Reading design script from {script_file}")
                script = open(script_file, "r").readlines()
            else:
                script = script_file.split("\n")
            print("[FastMPNN] ===== Parsed protocol =====")
            print("\n".join(script))
            print("[FastMPNN] ===========================")
            for l in script:
                if len(l) == 0:
                    continue
                if len(l.split()) > 1:
                    script_list.append([l.split()[0].strip()])
                    for x in l.split()[1:]:
                        if x[0].isalpha():
                            script_list[-1].append(x)
                        else:
                            script_list[-1].append(float(x))
                else:
                    script_list.append([l.strip()])
        elif isinstance(script_file, list):
            script_list = script_file
        return script_list

    def _report_seqs(self, seq1, seq2):
        str0 = "  Resno: 1  "
        str1 = " Before: "
        str2 = "Mutated: "
        str3 = "  After: "
        n = 1
        print("[MPNN] Sequences before and after MPNN design")
        for i, (r1, r2) in enumerate(zip(seq1, seq2)):
            if n == 81 or i == len(seq2)-1:
                n_spaces = n-5
                print(str0 + " "* n_spaces + f"{i:<3}")
                print(str1)
                print(str2)
                print(str3 + "\n")
                n = 1
                str0 = f"  Resno: {(i+1):<3}"
                str1 = " Before: "
                str2 = "Mutated: "
                str3 = "  After: "

            str1 += r1
            str3 += r2
            if r1 == r2:
                str2 += " "
            else:
                str2 += "*"
            n += 1

    def _thread_seq_to_pose(self, pose, sequence):
        pose2 = pose.clone()
        for i, r in enumerate(sequence):
            if r not in "ACDEFKRYPGLIVMHNWQST":
                print(f"[FastMPNN] _thread_seq_to_pose: residue {i} aa1 {r} not a canonical amino acid")
                continue
            if pose2.residue(i+1).name1() == r:
                continue
            mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
            mutres.set_target(i+1)
            res_name = restype_1to3[r]
            if i+1 == pose.chain_begin(pose.chain(i+1)):
                res_name += ":NtermProteinFull"
            if i+1 == pose.chain_end(pose.chain(i+1)):
                res_name += ":CtermProteinFull"
            mutres.set_res_name(res_name)
            mutres.apply(pose2)
        return pose2

    # =========================================================================
    # MPNN METHODS
    # =========================================================================
    def do_minimize(self, pose, tolerance, movemap, min_type):
        min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover(movemap, self.scorefxn, min_type, tolerance, True)
        min_mover.apply(pose)

    def do_mpnn(self, pose, mpnn_input, temperature, num_sequences):
        pdbstr = pyrosetta.distributed.io.to_pdbstring(pose)
        remarks = [l for l in pdbstr.split("\n") if "REMARK 666" in l]
        ligands = [res.seqpos() for res in pose if res.is_ligand()]

        # Making a new instance of mpnn input for this run call
        # it inherits attributes that were globally set
        mpnn_input = mpnn_input.copy()

        mpnn_input.pdb = pdbstr
        mpnn_input.name = self.__name
        mpnn_input.temperature = temperature
        num_sequences = int(num_sequences)
        if num_sequences <= 15:
            mpnn_input.batch_size = num_sequences
            mpnn_input.number_of_batches = 1
        else:
            # Finding the largest batch size up to 15 that would allow generating
            # the number of sequences that was requested
            batch_size_num = []
            for n in range(15, 0, -1):
                if num_sequences % n == 0:
                    batch_size_num = [n, num_sequences // n]
                    break
            if max(batch_size_num) <= 15:
                batch_size = max(batch_size_num)
                num_batches = min(batch_size_num)
            else:
                batch_size = min(batch_size_num)
                num_batches = max(batch_size_num)
            mpnn_input.batch_size = batch_size
            mpnn_input.number_of_batches = num_batches

        # Log residue information
        print(f"[MPNN] Generating {num_sequences} sequences with ligandMPNN (T={temperature})")

        # Determine design vs fixed residues for logging
        all_protein_residues = [r.seqpos() for r in pose.residues if r.is_protein()]
        fixed_residues_set = set(mpnn_input.fixed_residues) if mpnn_input.fixed_residues else set()

        # Parse fixed residues to get seqpos (they're in "ChainResno" format like "A45")
        fixed_seqpos = []
        for fr in mpnn_input.fixed_residues:
            try:
                chain = fr[0]
                resno = int(fr[1:])
                # Find matching seqpos
                for seqpos in all_protein_residues:
                    if pose.pdb_info().chain(seqpos) == chain and pose.pdb_info().number(seqpos) == resno:
                        fixed_seqpos.append(seqpos)
                        break
            except (ValueError, IndexError):
                pass

        design_seqpos = [r for r in all_protein_residues if r not in fixed_seqpos]

        # Get catalytic residues from REMARK 666 lines
        catres_seqpos = []
        for line in remarks:
            parts = line.split()
            try:
                chain_idx = parts.index("MOTIF") + 2
                resno_idx = parts.index("MOTIF") + 3
                chain = parts[chain_idx]
                resno = int(parts[resno_idx])
                for seqpos in all_protein_residues:
                    if pose.pdb_info().chain(seqpos) == chain and pose.pdb_info().number(seqpos) == resno:
                        if seqpos not in catres_seqpos:
                            catres_seqpos.append(seqpos)
                        break
            except (ValueError, IndexError):
                pass

        print(f"[MPNN]   Design residues ({len(design_seqpos)}): {sorted(design_seqpos)[:20]}{'...' if len(design_seqpos) > 20 else ''}")
        print(f"[MPNN]   Fixed residues ({len(fixed_seqpos)}): {sorted(fixed_seqpos)[:30]}{'...' if len(fixed_seqpos) > 30 else ''}")
        if catres_seqpos:
            print(f"[MPNN]   Catalytic residues ({len(catres_seqpos)}): {sorted(catres_seqpos)}")

        mpnn_out = self.__mpnnrunner.run(mpnn_input, pack_sc=self.__MPNN_pack_sc)

        # Check for MPNN failure
        if not mpnn_out.get("success", True) or not mpnn_out.get("generated_sequences") or not mpnn_out.get("packed"):
            print("[MPNN] ERROR: MPNN failed to generate sequences or packed structures!")
            print(f"[MPNN]   generated_sequences: {len(mpnn_out.get('generated_sequences', []))}")
            print(f"[MPNN]   packed structures: {len(mpnn_out.get('packed', {}))}")
            raise RuntimeError("MPNN failed to generate output. Check MPNN logs above for details.")

        sequences = mpnn_out["generated_sequences"]
        n_total = len(sequences)
        n_unique = len(set(sequences))
        print(f"[MPNN] {n_unique} / {n_total} unique sequences")

        # Report sequence changes for unique sequences
        seen_seqs_for_report = set()
        for seq in sequences:
            if seq not in seen_seqs_for_report:
                self._report_seqs(pose.sequence(), seq)
                seen_seqs_for_report.add(seq)

        # Finding which of the MPNN-packed structures has the least clashes with the ligand
        if self.__MPNN_pack_sc is True:
            print("[MPNN] Creating poses from MPNN-packed structures")
            if self.__ref_pdb_pose is not None:
                print("[MPNN] Covalent connections will be restored from --ref_pdb")
            else:
                print("[MPNN] No --ref_pdb provided, skipping covalent connection restoration")
            poses = []
            seen_sequences = set()
            n_duplicates = 0

            for n in sorted(mpnn_out["packed"].keys()):
                # Get sequence for this structure
                if n < len(sequences):
                    seq = sequences[n]
                else:
                    # Extract sequence from PDB if index mismatch
                    seq = self._extract_sequence_from_pdb_str(mpnn_out["packed"][n][0])

                # Skip duplicates - only keep first occurrence of each sequence
                if seq in seen_sequences:
                    n_duplicates += 1
                    continue
                seen_sequences.add(seq)

                # Fix the MPNN packed PDB by restoring missing header information from original
                # (REMARK 666, HETNAM, LINK lines - HETATM already present in MPNN output)
                fixed_pdb_str = self._fix_mpnn_packed_pdb(mpnn_out["packed"][n][0], pdbstr)

                # Use the new setup function that:
                # 1. Loads MPNN structure (preserving its geometry)
                # 2. Re-establishes covalent connections ONLY if --ref_pdb was provided
                #    (filtered to catres-ligand bonds from catres_cst_subset)
                # 3. Adds tight coordinate constraints to ALL ligand atoms (including H) to keep ligand frozen
                # 4. Handles close contacts between ligand H and acid/base residues
                try:
                    new_pose = rosetta_utils.setup_pose_from_mpnn_output(
                        mpnn_pdb_str=fixed_pdb_str,
                        ref_pose=pose,  # Current iteration pose (for labels, numbering)
                        cst_io=self.__cst_io,
                        sfx=self.__scorefxn,
                        constrain_ligand=True,
                        handle_close_contacts=True,
                        ref_pdb_pose=self.__ref_pdb_pose,  # ref_pdb for covalent bonds (None if not provided)
                        catres_for_covalent=self.__catres_for_covalent,  # Allowed catres for covalent bonds
                        verbose=True
                    )
                    poses.append(new_pose)
                except Exception as e:
                    print(f"[MPNN] Warning: Failed to set up pose {n}: {e}")
                    # Fall back to old method if new method fails
                    print("[MPNN] Falling back to legacy pose setup...")
                    poses.append(self._build_pose_from_str_and_append_stuff(
                        pdb_str=fixed_pdb_str,
                        append_pose=None, append_pose_resnos=None,
                        prepend_lines=None, ref_pose=pose
                    ))

            if n_duplicates > 0:
                print(f"[MPNN] Skipped {n_duplicates} duplicate sequences")
            print(f"[MPNN] {len(poses)} unique poses created")
        else:
            # Thread sequences to pose
            print("[MPNN] Threading MPNN sequences to input pose")
            poses = [self._thread_seq_to_pose(pose, seq) for seq in _df.seq.unique()]

        # Adding any residue labels to the new generated poses
        # Note: setup_pose_from_mpnn_output already copies labels, but this ensures consistency for threading path
        for i, p in enumerate(poses):
            for r in pose.residues:
                for label in pose.pdb_info().get_reslabels(r.seqpos()):
                    if not poses[i].pdb_info().res_haslabel(r.seqpos(), label):
                        poses[i].pdb_info().add_reslabel(r.seqpos(), label)
        return poses

    # =========================================================================
    # SETUP METHODS
    # =========================================================================
    def setup_mpnn(self, pose, design_positions, repack_positions, do_not_repack_positions):
        def _figure_out_ch_resno(pose, resno):
            if isinstance(resno, int):
                chain_let = pose.pdb_info().chain(resno)
            if isinstance(resno, str):
                if resno[0].isalpha():
                    chain_let = resno[0]
                    resno = int(resno[1:])
                else:
                    resno = int(resno)
                    chain_let = pose.pdb_info().chain(resno)
            resno = pose.pdb_info().number(resno)  # converting from pose-numbering to PDB-numbering for MPNN
            return f"{chain_let}{resno}"

        design_pos_list = []
        not_design_pos_list = []

        if repack_positions is not None:
            for resno in repack_positions:
                if isinstance(resno, int) and pose.residue(resno).is_ligand():
                    continue
                not_design_pos_list.append(_figure_out_ch_resno(pose, resno))

        if do_not_repack_positions is not None:
            for resno in do_not_repack_positions:
                if isinstance(resno, int) and pose.residue(resno).is_ligand():
                    continue
                not_design_pos_list.append(_figure_out_ch_resno(pose, resno))

        if design_positions is not None:
            for resno in design_positions:
                design_pos_list.append(_figure_out_ch_resno(pose, resno))

        if design_positions is not None and len(not_design_pos_list) == 0:
            for res in pose.residues:
                if res.is_protein() is True:
                    chain_let = pose.pdb_info().chain(res.seqpos())
                    if f"{chain_let}{resno}" in design_pos_list:
                        continue
                    not_design_pos_list.append(f"{chain_let}{res.seqpos()}")

        mpnn_input = self.__mpnnrunner.MPNN_Input()
        mpnn_input.fixed_residues = copy.deepcopy(not_design_pos_list)

        if self.__omit_AA is not None:
            mpnn_input.omit_AA = [x for x in self.__omit_AA]
        else:
            mpnn_input.omit_AA = ["C"]

        if self.__bias_AAs is not None:
            mpnn_input.bias_AA = self.__bias_AAs

        if self.__bias_AAs_per_residue is not None:
            mpnn_input.bias_AA_per_residue = self.__bias_AAs_per_residue

        mpnn_input.number_of_batches = 1
        return mpnn_input

    def setup_minimizer(self):
        if self.__movemap is None:
            # Create MoveMap with ligand rigidity configuration
            mm = rosetta_utils.create_minimizer_movemap(
                pose=self.__input_pose,
                allow_chi=True,
                allow_bb=True,
                allow_jump=True,
                ligand_rigidity=self.__ligand_rigidity
            )
            self.__movemap = mm.clone()

        min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover()
        min_mover.set_type(self.__min_type)
        min_mover.cartesian(self.__cartesian)
        min_mover.set_movemap(self.__movemap)
        min_mover.score_function(self.scorefxn())
        min_mover.nb_list(True)
        return min_mover

    def setup_packer(self):
        packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover()
        if self.__tf is None:
            self.__tf = self.setup_taskfactory()
        packer.task_factory(self.__tf)
        return packer

    def setup_taskfactory(self):
        tf = pyrosetta.rosetta.core.pack.task.TaskFactory()

        taskops = [pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline(),
                   pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent(),
                   pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides(),
                   pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking()]

        for to in taskops:
            tf.push_back(to)
        return tf

    def setup_packer_positions(self, design_resnos=None, repack_only_resnos=None, do_not_repack_resnos=None):
        if self.__tf is None:
            self.__tf = self.setup_taskfactory()

        # Design positions applies only to MPNN
        if design_resnos is None:
            self.__design_positions = []
            for res in self.__input_pose.residues:
                if res.seqpos() not in repack_only_resnos+do_not_repack_resnos:
                    self.__design_positions.append(res.seqpos())

        if repack_only_resnos is not None and len(repack_only_resnos) > 0:
            repack_only_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
            for resno in repack_only_resnos:
                repack_only_selector.append_index(resno)
            print(f"[FastMPNN] Adding RestrictToRepackingRLT for positions {repack_only_resnos}")
            self.__tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
                pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT(), repack_only_selector))

        if do_not_repack_resnos is not None and len(do_not_repack_resnos) > 0:
            do_not_repack_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
            for resno in do_not_repack_resnos:
                do_not_repack_selector.append_index(resno)
            print(f"[FastMPNN] Adding PreventRepackingRLT for positions {do_not_repack_resnos}")
            self.__tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
                pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT(), do_not_repack_selector))

    def setup_movers(self):
        setup_methods = {"repack": self.setup_packer, "min": self.setup_minimizer, "mpnn": self.setup_mpnn}
        for k in ["min", "repack"]:
            if k in [l[0] for l in self.script()]:
                self.__movers[k] = setup_methods[k]()

    # =========================================================================
    # MAIN APPLY METHOD
    # =========================================================================
    def apply(self, pose):
        """
        Performs FastDesign-like protein sequence design, following
        instructions set in a design script. Uses proteinMPNN (as implemented in fused-mpnn)
        to do sequence design.

        Protocol/script keywords that are respected:
            min, repack, mpnn, scale:{scorefunction_name}, task_operation, keep_best, 2nd_shell_mpnn, flip_catres
        Some keywords accept a number of values/arguments:
            min <float, minimizer tolerance>
            mpnn <float, mpnn temperature> <int num_sequences>
            scale:{scorefunction} <float, weight>
            task_operation <str, taskop name>
            keep_best <int, N best to keep> [mode] :: best N poses kept; mode can be 'cst', 'total', or 'cst_priority'
            2nd_shell_mpnn <float, temperature> [int, num_sequences] :: designs outer shell with inner fixed
            flip_catres [float, cst_threshold] [int, max_iter] :: try 180° sidechain flips for catres with poor constraints

        Note: Automatic sidechain flip attempts occur after min/repack operations by default.
              Use set_catres_flip_config() to configure or disable.

        Example protocol/script:
        scale:coordinate_constraint 1.0
        scale:fa_rep 0.150
        mpnn 0.4
        repack
        scale:fa_rep 0.200
        min 0.01
        scale:coordinate_constraint 0.5
        scale:fa_rep 0.365
        mpnn 0.2
        repack
        scale:fa_rep 0.480
        min 0.01
        scale:coordinate_constraint 0.0
        scale:fa_rep 0.659
        mpnn 0.1
        repack
        scale:fa_rep 0.750
        min 0.01
        scale:coordinate_constraint 0.0
        scale:fa_rep 1
        mpnn 0.1
        repack
        min 0.00001
        """
        start_time = time.time()
        self.__input_pose = pose.clone()

        self.setup_packer_positions(self.__design_positions, self.__repack_positions, self.__do_not_repack_positions)
        self.setup_movers()
        self.__mpnn_input = self.setup_mpnn(pose, self.__design_positions, self.__repack_positions, self.__do_not_repack_positions)

        poses = {0: {"pose": pose.clone(), "mpnn_input": self.__mpnn_input.copy()}}
        mpnn_iterations = 0

        for i, cmd in enumerate(self.script()):
            command = cmd[0]
            val = None
            if len(cmd) > 1:
                val = cmd[1:]
            print(f"[FastMPNN] Step {i}: {command}, {val}")

            if command == "min":
                self.__movers[command].tolerance(*val)

            elif command == "mpnn":
                assert len(val) >= 1, "Need to provide mpnn temperature and optionally num_seq"
                if len(val) == 1:
                    val.append(self.__num_sequences)
                poses_designed = []
                mpnn_inputs = []
                for pi in poses:
                    poses_designed += self.do_mpnn(pose=poses[pi]["pose"], mpnn_input=poses[pi]["mpnn_input"], temperature=val[0], num_sequences=val[1])
                    mpnn_inputs += [poses[pi]["mpnn_input"]]*int(val[1])
                poses = {pi: {"pose": p.clone(), "mpnn_input": mpnn_inputs[pi]} for pi, p in enumerate(poses_designed)}

                # Setting the number of MPNN sequences to 1, if more than 1 were designed in current round
                # This is ignored if the design protocol specifies the number of sequences for a given step
                if len(poses) > 1:
                    self.__num_sequences = self.__mpnn_N_seq_after_first
                mpnn_iterations += 1

            elif command[:6] == "scale:":
                _scoreterm = command.split(":")[1]
                self.__scorefxn.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name(_scoreterm), *val)

            elif command == "keep_best":
                # Parse arguments: keep_best N [mode]
                # Examples: keep_best 5, keep_best 5 cst, keep_best 5 total, keep_best 5 cst_priority
                N_keep = 1
                step_mode = self.__keep_best_mode  # Default to class-level mode

                if val is not None:
                    N_keep = int(val[0])
                    # Check if a mode was specified in the protocol
                    if len(val) > 1:
                        mode_arg = str(val[1]).lower()
                        # Allow shorthand aliases
                        mode_aliases = {
                            "cst": "cst_only",
                            "total": "total_score",
                            "cst_priority": "cst_priority",
                            "cst_only": "cst_only",
                            "total_score": "total_score"
                        }
                        if mode_arg in mode_aliases:
                            step_mode = mode_aliases[mode_arg]
                        else:
                            print(f"[FastMPNN] Warning: Unknown keep_best mode '{mode_arg}', using default '{self.__keep_best_mode}'")

                if len(poses) < N_keep:
                    print(f"[FastMPNN] Step {i}: {command}, requested N_keep = {N_keep} > number of poses ({len(poses)}). Keeping all.")
                    N_keep = len(poses)

                # Use new scoring selection function
                selected_indices = select_best_poses(
                    poses, self.__scorefxn, N_keep,
                    mode=step_mode,
                    cst_threshold=self.__cst_comparable_threshold,
                    verbose=True
                )

                _tmp_dict = copy.deepcopy(poses)
                poses = {}
                for _n, idx in enumerate(selected_indices):
                    poses[_n] = {"pose": _tmp_dict[idx]["pose"], "mpnn_input": _tmp_dict[idx]["mpnn_input"]}
                _tmp_dict = None

                # Setting the number of MPNN sequences to original value if only 1 best design is kept
                # This is ignored if the design protocol specifies the number of sequences for a given step
                if N_keep == 1:
                    self.__num_sequences = self.__num_sequences_original

            # Applying packer or minimizer
            if command in ["repack", "min"]:
                self.__movers[command].score_function(self.scorefxn())
                poses_moved = []
                for pi in poses:
                    _p = poses[pi]["pose"].clone()
                    print(f"[FastMPNN] {command} - pose has constraints: {_p.constraint_set().has_constraints()}")
                    self.__movers[command].apply(_p)

                    # Checking how much the backbone has moved during minimization
                    if command == "min":
                        overlay_pos = pyrosetta.rosetta.utility.vector1_unsigned_long()
                        for n in range(1, poses[pi]["pose"].size()+1):
                            if poses[pi]["pose"].residue(n).is_protein():
                                overlay_pos.append(n)
                        rmse = pyrosetta.rosetta.protocols.toolbox.pose_manipulation.superimpose_pose_on_subset_CA(poses[pi]["pose"], _p, overlay_pos, 0)
                        if rmse > self.__minimizer_rmsd_cutoff:
                            print(f"[FastMPNN] Backbone moved too much during minimization (rmsd={rmse:.3f} > {self.__minimizer_rmsd_cutoff})")
                            continue
                    poses_moved.append(_p.clone())
                if len(poses_moved) == 0:
                    print(f"[FastMPNN] No poses with backbone rmsd < {self.__minimizer_rmsd_cutoff} remained after {command} {val}")
                    sys.exit(1)

                poses = {pi: {"pose": p.clone(), "mpnn_input": poses[pi]["mpnn_input"]} for pi, p in enumerate(poses_moved)}

                # Automatic sidechain flip attempts for catalytic residues after min/repack
                # This helps escape local minima for symmetric sidechains (HIS, PHE, ASP, etc.)
                if self.__enable_auto_flip and self.__catres_for_covalent:
                    for pi in poses:
                        flip_results = rosetta_utils.attempt_catres_sidechain_flips(
                            poses[pi]["pose"],
                            self.__scorefxn,
                            self.__catres_for_covalent,
                            cst_threshold=self.__flip_cst_threshold,
                            do_minimize=True,
                            max_min_iter=self.__flip_max_min_iter,
                            verbose=True
                        )
                        # Record flip results for aggregate statistics
                        self.__flip_tracker.record(flip_results, f"{command}_step{i}", pi)

            # Explicit flip_catres protocol operation
            elif command == "flip_catres":
                # Parse optional arguments: flip_catres [threshold] [max_iter]
                # Examples: flip_catres, flip_catres 0.5, flip_catres 1.0 15
                threshold = self.__flip_cst_threshold
                max_iter = self.__flip_max_min_iter
                if val is not None:
                    threshold = float(val[0])
                    if len(val) > 1:
                        max_iter = int(val[1])

                if not self.__catres_for_covalent:
                    print(f"[FastMPNN] flip_catres: No catalytic residues configured, skipping")
                else:
                    print(f"[FastMPNN] flip_catres: threshold={threshold:.2f}, max_iter={max_iter}")
                    for pi in poses:
                        flip_results = rosetta_utils.attempt_catres_sidechain_flips(
                            poses[pi]["pose"],
                            self.__scorefxn,
                            self.__catres_for_covalent,
                            cst_threshold=threshold,
                            do_minimize=True,
                            max_min_iter=max_iter,
                            verbose=True
                        )
                        # Record flip results for aggregate statistics
                        self.__flip_tracker.record(flip_results, f"flip_catres_step{i}", pi)

            # Task operations - enables fixing additional residues for MPNN
            if command in ["task_operation"]:
                taskop_name = val[0]
                assert taskop_name in self.__task_operations.keys()
                for j, p in poses.items():
                    _taskop = self.__task_operations[taskop_name].copy()
                    if _taskop.target() is None or len(_taskop.target()) == 0:
                        print(f"[FastMPNN] No target for {taskop_name}: skipping")
                        continue

                    # Fetching the original set of target residues, if the pose has none
                    if all([p["pose"].pdb_info().res_haslabel(r.seqpos(), f"{taskop_name}_target") == False for r in p["pose"].residues]):
                        for r in _taskop.target():
                            poses[j]["pose"].pdb_info().add_reslabel(res=r, label=f"{taskop_name}_target")

                    # Updating the task operation target list based on any target residues the pose has
                    old_targets = [r.seqpos() for r in p["pose"].residues if p["pose"].pdb_info().res_haslabel(r.seqpos(), f"{taskop_name}_target")]
                    if _taskop.allow_updating() is True:
                        _taskop.target(old_targets)

                    # Finding new residues based on the task operation logic
                    selection = _taskop.compute(p["pose"])

                    # Setting any found residues as fixed for MPNN
                    if len(selection) > 0 and len([x for x in selection if x not in old_targets]) > 0:
                        _old_fixed = [p["pose"].pdb_rsd((r[0], int(r[1:]))).seqpos() for r in p["mpnn_input"].fixed_residues]
                        new_fixed_res = sorted(list(set(selection + _old_fixed)))
                        print(f"[FastMPNN] TaskOperation {taskop_name}: Updated fixed residues for pose {j}: {[x for x in selection if x not in old_targets]}")
                        new_design = [x for x in self.__design_positions if x not in new_fixed_res]
                        poses[j]["mpnn_input"] = self.setup_mpnn(p["pose"], new_design, new_fixed_res, self.__do_not_repack_positions)

                        # Updating the set of target residues on the pose
                        if _taskop.allow_updating() is True:
                            new_targets = sorted(list(set(selection+old_targets)))
                            for r in new_targets:
                                poses[j]["pose"].pdb_info().add_reslabel(res=r, label=f"{taskop_name}_target")
                    else:
                        print(f"[FastMPNN] TaskOperation {taskop_name}: No additional residues found for pose {j}")

            # 2nd shell MPNN - designs outer shell while fixing inner pocket residues
            elif command == "2nd_shell_mpnn":
                if self.__ligand_seqpos is None or self.__ligand_heavyatoms is None:
                    print(f"[FastMPNN] WARNING: 2nd_shell_mpnn requires set_2nd_shell_config() to be called. Skipping.")
                    continue

                # Parse temperature and optionally num_sequences
                temperature = 0.1  # default
                num_seq_2nd = 2    # default
                if val is not None:
                    temperature = float(val[0])
                    if len(val) > 1:
                        num_seq_2nd = int(val[1])

                print(f"[FastMPNN] 2nd_shell_mpnn: T={temperature}, N_seq={num_seq_2nd}")

                new_poses = {}
                pose_idx = 0
                for pi in poses:
                    p = poses[pi]["pose"]

                    # Get fixed positions for 2nd shell (inner pocket fixed, outer shell designed)
                    fixed_residues, design_residues_str = get_2nd_shell_fixed_positions(
                        p, self.__ligand_seqpos, self.__ligand_heavyatoms,
                        self.__keep_positions if self.__keep_positions else []
                    )

                    print(f"[FastMPNN] 2nd_shell_mpnn: Pose {pi} - fixing {len(fixed_residues)} residues, designing {len(design_residues_str)} residues")

                    # Create new MPNN input with 2nd shell fixed positions
                    mpnn_input_2nd = poses[pi]["mpnn_input"].copy()
                    mpnn_input_2nd.fixed_residues = fixed_residues

                    # Run MPNN on 2nd shell
                    designed_poses = self.do_mpnn(
                        pose=p,
                        mpnn_input=mpnn_input_2nd,
                        temperature=temperature,
                        num_sequences=num_seq_2nd
                    )

                    # Add all designed poses to pool
                    for dp in designed_poses:
                        new_poses[pose_idx] = {"pose": dp.clone(), "mpnn_input": poses[pi]["mpnn_input"].copy()}
                        pose_idx += 1

                poses = new_poses
                print(f"[FastMPNN] 2nd_shell_mpnn: Generated {len(poses)} poses")

            # Dumping PDBs if in debug mode
            if self.__debug is True and command in ["min", "repack", "mpnn", "flip_catres"]:
                for j, p in poses.items():
                    if self.__debug_output_dir:
                        output_path = os.path.join(
                            self.__debug_output_dir,
                            f"step{i:02d}_{command}_pose{j}.pdb"
                        )
                    else:
                        output_path = f"{self.__name}_{command}_{i}.{j}.pdb"
                    p["pose"].dump_pdb(output_path)
                    print(f"  [DEBUG] Saved: {output_path}")

        for i, p in poses.items():
            print(f"[FastMPNN] ===== Scoring final pose {i} =====")
            self.__scorefxn(p["pose"])
            for k, val in p["pose"].scores.items():
                print(f"  {k:>40}: {val:>20.3f}")

        # Print flip statistics summary if any flips were attempted
        if self.__flip_tracker.history:
            self.__flip_tracker.print_summary()

        elapsed = time.time() - start_time
        print(f"[FastMPNN] Finished FastMPNNdesign protocol in {elapsed:.3f} seconds")
        return [p["pose"].clone() for i, p in poses.items()]
