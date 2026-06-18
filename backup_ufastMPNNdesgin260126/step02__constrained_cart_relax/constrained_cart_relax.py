#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step02: Constrained Cartesian Relaxation

Takes step01 output JSON and performs adaptive Cartesian FastRelax to idealize
geometry while keeping ligand and specified catalytic residue atoms fixed in
absolute space.

Key Features:
- Adaptive protocol: continues FastRelax cycles while improving
- Smart mobile region selection: includes SS elements and sequence neighbors
- Tight coordinate constraints for ligand and catres atoms
- Comprehensive metrics for validation

Author: Created for woodbuse
Date: 2026-01-23
"""

import pyrosetta as pyr
from pyrosetta.rosetta.core.select.residue_selector import (
    NeighborhoodResidueSelector,
    ResidueIndexSelector,
    OrResidueSelector,
)
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.core.scoring import ScoreType, ScoreFunctionFactory, CA_rmsd
from pyrosetta.rosetta.core.scoring.constraints import CoordinateConstraint
from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.core.scoring.dssp import Dssp
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import (
    OperateOnResidueSubset,
    PreventRepackingRLT,
    RestrictToRepackingRLT,
)
from pyrosetta.rosetta.core.select.residue_selector import NotResidueSelector

import os
import sys
import json
import argparse
import logging
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

# Add module_utils to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, MODULE_DIR)

from module_utils.pdb_utils import (
    parse_remark_666,
    read_pdb_atoms,
    is_backbone_atom,
    BACKBONE_ATOMS,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# Test mode constants
TEST_STEP01_JSON = os.path.join(SCRIPT_DIR, "test/step01_outputs/input_pdb_recommended_atom_cst.json")
TEST_PARAMS = [os.path.join(SCRIPT_DIR, "test/params/XDW.params")]
TEST_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "test/output_dir")

# Standard amino acids for ligand identification
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
}


class ConstrainedCartesianRelax:
    """
    Perform constrained Cartesian relaxation on step01 output.

    Uses adaptive FastRelax cycles with tight coordinate constraints to
    idealize bond geometry while preserving ligand and catalytic residue positions.
    """

    def __init__(
        self,
        step01_json_path: str,
        params_files: List[str],
        catres_subset: Optional[str] = None,
        ref_pdb_override: Optional[str] = None,
        input_pdb_override: Optional[str] = None,
        ignore_bb_only_hbond_filter: bool = False,
        coord_cst_weight: float = 750.0,
        coord_cst_stdev: float = 0.01,
        cart_bonded_weight: float = 2.0,
        mobile_radius: float = 10.0,
        fastrelax_repeats: int = 3,
        fastrelax_ramp_stages: int = 5,
        max_runtime: int = 3600,
        sequence_neighbor_buffer: int = 5,
        skip_torsional_relax: bool = False,
        skip_minimize: bool = False,
        bond_length_tolerance: float = 0.05,
        bond_angle_tolerance: float = 10.0,
        fast_mode: bool = False,
        debug: bool = False,
        # New parameters for enhanced protocol
        cart_bonded_scale_factor: float = 1.5,
        cart_bonded_max: float = 3.0,
        enable_bond_geometry_min: bool = True,
        max_adaptive_rounds: int = 10,
        # Catres-specific convergence
        catres_bond_tolerance: float = 0.05,
        catres_angle_tolerance: float = 10.0,
        require_catres_converged: bool = True,
        # Repacking options
        global_repack: bool = False,
        repack_shell: Optional[float] = None,
        # Mobile region expansion
        auto_expand_mobile: bool = True,
        expansion_radius: float = 5.0,
        max_expansions: int = 3,
        # Offender thresholds
        severe_bond_threshold: float = 0.2,
        moderate_bond_threshold: float = 0.1,
        severe_angle_threshold: float = 15.0,
        moderate_angle_threshold: float = 10.0,
        # Loop modeling (experimental)
        enable_loop_rebuild: bool = False,
        loop_rebuild_threshold: Optional[float] = None,
        # Scorefunction selection
        scorefunction: str = "ref2015_cart",
    ):
        """
        Initialize the constrained cartesian relaxer.

        Args:
            step01_json_path: Path to step01 output JSON (*_recommended_atom_cst.json)
            params_files: List of ligand .params files
            catres_subset: Override catres subset (comma-separated block indices)
            ref_pdb_override: Override ref_pdb from JSON (for validation)
            input_pdb_override: Override input PDB (use different aligned PDB)
            ignore_bb_only_hbond_filter: If True, constrain backbone even when
                                         backbone_important_only_for_BB_BB_hbond=True
            coord_cst_weight: Weight for coordinate_constraint score term
            coord_cst_stdev: HarmonicFunc standard deviation in Angstroms
            cart_bonded_weight: Weight for cart_bonded term
            mobile_radius: Radius for mobile region around ligand/catres
            fastrelax_repeats: Number of script repeats per FastRelax call (M)
            fastrelax_ramp_stages: Number of ramping stages in relax script (N)
                                   Total rounds per FastRelax = M × N
            max_runtime: Maximum runtime in seconds
            sequence_neighbor_buffer: Include residues +/- N from catres
            skip_torsional_relax: Skip torsional FastRelax at end
            skip_minimize: Skip final MinMover
            bond_length_tolerance: Stop when max bond length deviation < this (Angstroms)
            bond_angle_tolerance: Stop when max bond angle deviation < this (degrees)
            fast_mode: If True, use 1 repeat × 3 stages = 3 total (overrides repeats/stages)
            debug: Enable verbose debug output
            cart_bonded_scale_factor: Factor to multiply cart_bonded by when not converging
            cart_bonded_max: Maximum cart_bonded weight cap
            enable_bond_geometry_min: Enable bondangle+bondlength minimization in FastRelax
            max_adaptive_rounds: Maximum adaptive rounds before stopping
            catres_bond_tolerance: Catres-specific bond tolerance (stricter check)
            catres_angle_tolerance: Catres-specific angle tolerance (stricter check)
            require_catres_converged: Warn/fail if catres geometry bad
            global_repack: Enable global repacking (all protein residues)
            repack_shell: Repack within N Angstroms of mobile region
            auto_expand_mobile: Automatically expand mobile region if stuck
            expansion_radius: Radius for mobile region expansion
            max_expansions: Maximum number of mobile region expansions
            severe_bond_threshold: Bond deviation above which is 'severe'
            moderate_bond_threshold: Bond deviation for 'moderate' category
            severe_angle_threshold: Angle deviation above which is 'severe'
            moderate_angle_threshold: Angle deviation for 'moderate' category
            enable_loop_rebuild: Enable experimental loop rebuilding
            loop_rebuild_threshold: Bond deviation triggering loop rebuild
            scorefunction: Scorefunction to use (ref2015_cart, beta_jan25_cart, etc.)
        """
        self.step01_json_path = step01_json_path
        self.params_files = params_files
        self.catres_subset_override = catres_subset
        self.ref_pdb_override = ref_pdb_override
        self.input_pdb_override = input_pdb_override
        self.ignore_bb_only_hbond_filter = ignore_bb_only_hbond_filter
        self.coord_cst_weight = coord_cst_weight
        self.coord_cst_stdev = coord_cst_stdev
        self.cart_bonded_weight = cart_bonded_weight
        self.mobile_radius = mobile_radius
        self.fastrelax_repeats = fastrelax_repeats
        self.fastrelax_ramp_stages = fastrelax_ramp_stages
        self.max_runtime = max_runtime
        self.sequence_neighbor_buffer = sequence_neighbor_buffer
        self.skip_torsional_relax = skip_torsional_relax
        self.skip_minimize = skip_minimize
        self.bond_length_tolerance = bond_length_tolerance
        self.bond_angle_tolerance = bond_angle_tolerance
        self.fast_mode = fast_mode
        self.debug = debug

        # New parameters for enhanced protocol
        self.cart_bonded_scale_factor = cart_bonded_scale_factor
        self.cart_bonded_max = cart_bonded_max
        self.enable_bond_geometry_min = enable_bond_geometry_min
        self.max_adaptive_rounds = max_adaptive_rounds

        # Catres-specific convergence
        self.catres_bond_tolerance = catres_bond_tolerance
        self.catres_angle_tolerance = catres_angle_tolerance
        self.require_catres_converged = require_catres_converged

        # Repacking options
        self.global_repack = global_repack
        self.repack_shell = repack_shell

        # Mobile region expansion
        self.auto_expand_mobile = auto_expand_mobile
        self.expansion_radius = expansion_radius
        self.max_expansions = max_expansions

        # Offender thresholds
        self.severe_bond_threshold = severe_bond_threshold
        self.moderate_bond_threshold = moderate_bond_threshold
        self.severe_angle_threshold = severe_angle_threshold
        self.moderate_angle_threshold = moderate_angle_threshold

        # Loop modeling (experimental)
        self.enable_loop_rebuild = enable_loop_rebuild
        self.loop_rebuild_threshold = loop_rebuild_threshold

        # Scorefunction selection
        self.scorefunction = scorefunction

        if debug:
            LOGGER.setLevel(logging.DEBUG)

        # Will be populated during load
        self.step01_data = None
        self.input_pdb_path = None
        self.ref_pdb_path = None
        self.residue_constraints = None
        self.ligand_info = None  # (chain, resname, resno)
        self.atoms_to_constrain = {}  # {(chain, resno): [atom_names] or "ALL_HEAVY"}
        self.constrained_atoms_set = set()  # {(chain, resno, atom_name), ...} for quick lookup
        self.catres_pose_indices = []
        self.ligand_pose_indices = []

        # Pose objects
        self.pose = None
        self.original_pose = None
        self.sfxn = None

        # Timing
        self.timings = {}
        self.start_time = None

    def load_step01_json(self) -> Dict:
        """Load and validate step01 JSON."""
        LOGGER.info(f"Loading step01 JSON: {self.step01_json_path}")

        with open(self.step01_json_path, 'r') as f:
            data = json.load(f)

        self.step01_data = data

        # Extract paths
        self.input_pdb_path = self.input_pdb_override or data.get("output_pdb")
        self.ref_pdb_path = self.ref_pdb_override or data.get("ref_pdb")

        if not self.input_pdb_path:
            raise ValueError("No input PDB path found (output_pdb in JSON or --input_prepped_pdb)")

        # Extract residue constraints
        self.residue_constraints = data.get("residue_constraints", {})

        # Apply catres_subset override if provided
        if self.catres_subset_override:
            subset_indices = set(self.catres_subset_override.split(','))
            self.residue_constraints = {
                k: v for k, v in self.residue_constraints.items()
                if k in subset_indices
            }
            LOGGER.info(f"Using catres subset: {sorted(subset_indices)}")

        LOGGER.info(f"  Input PDB: {self.input_pdb_path}")
        LOGGER.info(f"  Ref PDB: {self.ref_pdb_path}")
        LOGGER.info(f"  Residue constraints: {len(self.residue_constraints)} residues")

        return data

    def identify_ligand_from_pdb(self, pdb_path: str) -> Tuple[str, str, int]:
        """
        Parse REMARK 666 to identify ligand (non-protein TEMPLATE entry).

        Returns:
            (chain, resname, resno) of ligand
        """
        all_lines, _ = read_pdb_atoms(pdb_path)
        entries = parse_remark_666(all_lines)

        for entry in entries:
            template_resname = entry.get("template_resname", "")
            if template_resname and template_resname not in STANDARD_AA:
                return (
                    entry.get("template_chain", "X"),
                    template_resname,
                    entry.get("template_resno", 1)
                )

        raise ValueError("Could not identify ligand from REMARK 666 entries")

    def validate_remark_666_consistency(self, pdb_path: str) -> None:
        """
        Validate that residue_constraints block indices match REMARK 666 entries.
        """
        LOGGER.info("Validating REMARK 666 consistency...")

        all_lines, _ = read_pdb_atoms(pdb_path)
        entries = parse_remark_666(all_lines)

        # Build lookup by block_index
        remark_lookup = {str(e["block_index"]): e for e in entries}

        mismatches = []
        for block_idx, info in self.residue_constraints.items():
            if block_idx not in remark_lookup:
                mismatches.append(f"Block {block_idx} not found in REMARK 666")
                continue

            remark = remark_lookup[block_idx]

            # Check chain
            if info["chain"] != remark["motif_chain"]:
                mismatches.append(
                    f"Block {block_idx} chain mismatch: JSON={info['chain']}, "
                    f"REMARK={remark['motif_chain']}"
                )

            # Check resno
            if info["resno"] != remark["motif_resno"]:
                mismatches.append(
                    f"Block {block_idx} resno mismatch: JSON={info['resno']}, "
                    f"REMARK={remark['motif_resno']}"
                )

            # Check resname
            if info["resname"] != remark["motif_resname"]:
                mismatches.append(
                    f"Block {block_idx} resname mismatch: JSON={info['resname']}, "
                    f"REMARK={remark['motif_resname']}"
                )

        if mismatches:
            for m in mismatches:
                LOGGER.warning(f"  {m}")
            LOGGER.warning("Proceeding despite mismatches - check input files")
        else:
            LOGGER.info("  All block indices validated successfully")

    def build_atoms_to_constrain(self) -> Dict[Tuple[str, int], List[str]]:
        """
        Build dict of (chain, resno) -> [atom_names] to constrain.

        Logic:
        - Include ALL ligand heavy atoms (HETATM)
        - For each catres in residue_constraints:
          - If backbone_important_only_for_BB_BB_hbond=True AND NOT ignore_filter:
            - Skip backbone atoms from constrain_atoms list
          - Otherwise, include all atoms in constrain_atoms list
        """
        LOGGER.info("Building atoms to constrain...")

        atoms_to_constrain = {}

        # 1. Add ALL ligand heavy atoms
        lig_chain, lig_resname, lig_resno = self.ligand_info
        atoms_to_constrain[(lig_chain, lig_resno)] = "ALL_HEAVY"
        LOGGER.info(f"  Ligand {lig_chain}{lig_resno} ({lig_resname}): ALL heavy atoms")

        # 2. Process each catres from residue_constraints
        for block_idx, info in self.residue_constraints.items():
            chain = info["chain"]
            resno = info["resno"]
            resname = info["resname"]
            constrain_atoms = list(info.get("constrain_atoms", []))
            bb_only_hbond = info.get("backbone_important_only_for_BB_BB_hbond", False)

            original_count = len(constrain_atoms)

            # Apply backbone skip filter unless override flag is set
            if bb_only_hbond and not self.ignore_bb_only_hbond_filter:
                filtered_atoms = [a for a in constrain_atoms if not is_backbone_atom(a)]
                skipped = original_count - len(filtered_atoms)
                if skipped > 0:
                    LOGGER.debug(
                        f"  Block {block_idx} ({chain}{resno} {resname}): "
                        f"Skipped {skipped} backbone atoms (bb_only_hbond=True)"
                    )
                constrain_atoms = filtered_atoms

            if constrain_atoms:
                atoms_to_constrain[(chain, resno)] = constrain_atoms
                LOGGER.info(
                    f"  Block {block_idx} ({chain}{resno} {resname}): "
                    f"{len(constrain_atoms)} atoms"
                )

        self.atoms_to_constrain = atoms_to_constrain
        return atoms_to_constrain

    def build_constrained_atoms_set(self, pose) -> set:
        """
        Build a set of (chain, resno, atom_name) tuples for all constrained atoms.

        This is used to quickly check if a specific atom is constrained when
        calculating bond/angle metrics. We need the pose to expand "ALL_HEAVY"
        for the ligand.

        Returns:
            Set of (chain, resno, atom_name) tuples
        """
        constrained_set = set()

        for (chain, resno), atom_spec in self.atoms_to_constrain.items():
            pose_idx = pose.pdb_info().pdb2pose(chain, resno)
            if pose_idx == 0:
                continue

            residue = pose.residue(pose_idx)

            if atom_spec == "ALL_HEAVY":
                # Add all heavy atoms for this residue (used for ligand)
                for atom_idx in range(1, residue.natoms() + 1):
                    if residue.atom_type(atom_idx).element() != "H":
                        atom_name = residue.atom_name(atom_idx).strip()
                        constrained_set.add((chain, resno, atom_name))
            else:
                # Add specified atoms
                for atom_name in atom_spec:
                    constrained_set.add((chain, resno, atom_name))

        self.constrained_atoms_set = constrained_set
        LOGGER.info(f"Built constrained atoms set: {len(constrained_set)} atoms")
        return constrained_set

    def is_atom_constrained(self, chain: str, resno: int, atom_name: str) -> bool:
        """Check if a specific atom is in the constrained set."""
        return (chain, resno, atom_name) in self.constrained_atoms_set

    def build_residue_constraints_output(self, pose) -> Dict:
        """
        Build the final residue_constraints dictionary for output metadata.

        This reflects what was actually constrained (after filtering), not the
        original step01 JSON. Includes ligand and all catres with their final
        constrained atom lists.

        Returns:
            Dict with same format as step01 residue_constraints but with actual
            constrained atoms:
            {
                "ligand": {
                    "chain": "X",
                    "resno": 1,
                    "resname": "LIG",
                    "constrain_atoms": ["C1", "C2", ...],  # actual heavy atoms
                    "is_ligand": true
                },
                "1": {
                    "chain": "A",
                    "resno": 13,
                    "resname": "HIS",
                    "constrain_atoms": ["CB", "CG", ...],  # actually constrained
                    "backbone_important": false,
                    "sidechain_important": true,
                    "importance": "sidechain",
                    "backbone_important_only_for_BB_BB_hbond": true
                },
                ...
            }
        """
        output = {}

        # Add ligand entry
        lig_chain, lig_resname, lig_resno = self.ligand_info
        lig_key = (lig_chain, lig_resno)
        if lig_key in self.atoms_to_constrain:
            # Get actual ligand heavy atoms from pose
            pose_idx = pose.pdb_info().pdb2pose(lig_chain, lig_resno)
            if pose_idx > 0:
                residue = pose.residue(pose_idx)
                ligand_atoms = []
                for atom_idx in range(1, residue.natoms() + 1):
                    if residue.atom_type(atom_idx).element() != "H":
                        ligand_atoms.append(residue.atom_name(atom_idx).strip())
                output["ligand"] = {
                    "chain": lig_chain,
                    "resno": lig_resno,
                    "resname": lig_resname,
                    "constrain_atoms": sorted(ligand_atoms),
                    "is_ligand": True
                }

        # Add catres entries
        for block_idx, info in self.residue_constraints.items():
            chain = info["chain"]
            resno = info["resno"]
            key = (chain, resno)

            # Get the actually constrained atoms (after filtering)
            if key in self.atoms_to_constrain:
                actual_atoms = self.atoms_to_constrain[key]
                if isinstance(actual_atoms, list):
                    constrained_list = sorted(actual_atoms)
                else:
                    # Shouldn't happen for catres, but handle gracefully
                    constrained_list = []
            else:
                constrained_list = []

            output[str(block_idx)] = {
                "chain": chain,
                "resno": resno,
                "resname": info.get("resname", "UNK"),
                "constrain_atoms": constrained_list,
                "backbone_important": info.get("backbone_important", False),
                "sidechain_important": info.get("sidechain_important", False),
                "importance": info.get("importance", "unknown"),
                "backbone_important_only_for_BB_BB_hbond": info.get("backbone_important_only_for_BB_BB_hbond", False)
            }

        return output

    def add_coordinate_constraints(self, pose) -> int:
        """
        Add CoordinateConstraint with HarmonicFunc for each atom to constrain.

        Returns:
            Number of constraints added
        """
        LOGGER.info("Adding coordinate constraints...")

        constraint_count = 0
        missing_atoms = []

        for (chain, resno), atom_spec in self.atoms_to_constrain.items():
            pose_idx = pose.pdb_info().pdb2pose(chain, resno)
            if pose_idx == 0:
                LOGGER.warning(f"  Could not find {chain}{resno} in pose")
                continue

            residue = pose.residue(pose_idx)
            resname = residue.name3()

            # Determine which atoms to constrain
            if atom_spec == "ALL_HEAVY":
                # All heavy atoms for ligand
                atom_names = [
                    residue.atom_name(i).strip()
                    for i in range(1, residue.natoms() + 1)
                    if not residue.atom_is_hydrogen(i)
                ]
            else:
                atom_names = atom_spec

            added = 0
            for atom_name in atom_names:
                if not residue.has(atom_name):
                    missing_atoms.append(f"{chain}{resno}:{atom_name}")
                    continue

                atom_idx = residue.atom_index(atom_name)
                xyz = residue.xyz(atom_name)

                # HarmonicFunc(x0, sd): penalty = 0.5 * ((x - x0) / sd)^2
                func = HarmonicFunc(0.0, self.coord_cst_stdev)

                cst = CoordinateConstraint(
                    AtomID(atom_idx, pose_idx),
                    AtomID(1, 1),  # Anchor to first atom (uses absolute coords)
                    xyz,
                    func
                )

                pose.add_constraint(cst)
                constraint_count += 1
                added += 1

            LOGGER.debug(f"  {chain}{resno} ({resname}): {added} constraints")

        if missing_atoms and len(missing_atoms) <= 10:
            LOGGER.warning(f"  Missing atoms: {', '.join(missing_atoms)}")
        elif missing_atoms:
            LOGGER.warning(f"  Missing {len(missing_atoms)} atoms (see debug for details)")

        LOGGER.info(f"  Total: {constraint_count} coordinate constraints added")
        return constraint_count

    def setup_scorefunction(self):
        """
        Create scorefunction with appropriate weights.

        Supports: ref2015_cart, beta_jan25_cart, beta_nov16_cart
        """
        LOGGER.info(f"Setting up scorefunction: {self.scorefunction}")

        try:
            sfxn = ScoreFunctionFactory.create_score_function(self.scorefunction)
        except Exception as e:
            LOGGER.warning(f"Failed to create {self.scorefunction}: {e}")
            LOGGER.warning("Falling back to ref2015_cart")
            sfxn = ScoreFunctionFactory.create_score_function("ref2015_cart")

        # Critical settings for Cartesian relaxation
        sfxn.set_weight(ScoreType.cart_bonded, self.cart_bonded_weight)
        sfxn.set_weight(ScoreType.pro_close, 0.0)  # Incompatible with cart_bonded
        sfxn.set_weight(ScoreType.coordinate_constraint, self.coord_cst_weight)

        LOGGER.info(f"  scorefunction = {self.scorefunction}")
        LOGGER.info(f"  cart_bonded = {self.cart_bonded_weight}")
        LOGGER.info(f"  coordinate_constraint = {self.coord_cst_weight}")
        LOGGER.info(f"  pro_close = 0.0 (disabled)")

        self.sfxn = sfxn
        return sfxn

    def get_contiguous_ss_element(self, pose, start_idx: int, dssp) -> Set[int]:
        """
        Get all residue indices in the same secondary structure element.
        """
        ss_type = dssp.get_dssp_secstruct(start_idx)
        if ss_type not in ['H', 'E']:
            return {start_idx}

        element = {start_idx}

        # Expand backwards
        idx = start_idx - 1
        while idx >= 1 and dssp.get_dssp_secstruct(idx) == ss_type:
            element.add(idx)
            idx -= 1

        # Expand forwards
        idx = start_idx + 1
        while idx <= pose.size() and dssp.get_dssp_secstruct(idx) == ss_type:
            element.add(idx)
            idx += 1

        return element

    def define_mobile_region(self, pose) -> List[int]:
        """
        Define mobile residues using smart selection:
        1. Neighborhood around ALL ligand atoms
        2. Neighborhood around catres atoms
        3. Sequence neighbors: +/- buffer residues from each catres
        4. Entire secondary structure elements containing catres
        """
        LOGGER.info(f"Defining mobile region (radius={self.mobile_radius}A, seq_buffer={self.sequence_neighbor_buffer})...")

        mobile = set()

        # 1. Neighborhood around ligand
        if self.ligand_pose_indices:
            lig_selector = ResidueIndexSelector(','.join(map(str, self.ligand_pose_indices)))
            lig_neighborhood = NeighborhoodResidueSelector(
                lig_selector, distance=self.mobile_radius, include_focus_in_subset=True
            )
            lig_selected = lig_neighborhood.apply(pose)
            lig_mobile = [i for i in range(1, pose.size() + 1) if lig_selected[i]]
            mobile.update(lig_mobile)
            LOGGER.debug(f"  Ligand neighborhood: {len(lig_mobile)} residues")

        # 2. Neighborhood around catres
        if self.catres_pose_indices:
            cat_selector = ResidueIndexSelector(','.join(map(str, self.catres_pose_indices)))
            cat_neighborhood = NeighborhoodResidueSelector(
                cat_selector, distance=self.mobile_radius, include_focus_in_subset=True
            )
            cat_selected = cat_neighborhood.apply(pose)
            cat_mobile = [i for i in range(1, pose.size() + 1) if cat_selected[i]]
            mobile.update(cat_mobile)
            LOGGER.debug(f"  Catres neighborhood: {len(cat_mobile)} residues")

        # 3. Sequence neighbors
        seq_neighbors = set()
        for catres_idx in self.catres_pose_indices:
            for offset in range(-self.sequence_neighbor_buffer, self.sequence_neighbor_buffer + 1):
                neighbor_idx = catres_idx + offset
                if 1 <= neighbor_idx <= pose.size():
                    seq_neighbors.add(neighbor_idx)
        mobile.update(seq_neighbors)
        LOGGER.debug(f"  Sequence neighbors: {len(seq_neighbors)} residues")

        # 4. Secondary structure elements containing catres
        try:
            dssp = Dssp(pose)
            dssp.dssp_reduced()
            ss_residues = set()
            for catres_idx in self.catres_pose_indices:
                ss_element = self.get_contiguous_ss_element(pose, catres_idx, dssp)
                ss_residues.update(ss_element)
            mobile.update(ss_residues)
            LOGGER.debug(f"  SS elements: {len(ss_residues)} residues")
        except Exception as e:
            LOGGER.warning(f"  Could not compute DSSP: {e}")

        mobile_list = sorted(mobile)
        LOGGER.info(f"  Total mobile region: {len(mobile_list)} residues")

        return mobile_list

    def build_movemap(self, pose, mobile_residues: List[int]):
        """
        Build MoveMap allowing movement in mobile region.
        Ligands are frozen (belt + suspenders with constraints).
        """
        LOGGER.info("Building MoveMap...")

        mm = pyr.rosetta.core.kinematics.MoveMap()

        # Default: freeze everything
        mm.set_bb(False)
        mm.set_chi(False)
        mm.set_jump(False)

        # Enable movement for mobile residues (except ligands)
        mobile_non_lig = [r for r in mobile_residues if r not in self.ligand_pose_indices]
        for res_idx in mobile_non_lig:
            mm.set_bb(res_idx, True)
            mm.set_chi(res_idx, True)

        # Explicitly freeze ligands
        for lig_idx in self.ligand_pose_indices:
            mm.set_bb(lig_idx, False)
            mm.set_chi(lig_idx, False)

        LOGGER.info(f"  Mobile (non-ligand): {len(mobile_non_lig)} residues")
        LOGGER.info(f"  Frozen ligand: {len(self.ligand_pose_indices)} residues")

        return mm

    def build_task_factory(self, pose, mobile_residues: List[int]):
        """
        Build TaskFactory to control repacking scope.

        Supports three modes:
        1. global_repack=True: Repack all protein residues (not ligands)
        2. repack_shell>0: Repack within N Angstroms of mobile region
        3. Default: Repack only mobile residues

        All modes prevent repacking on ligands and prevent design (sequence changes).
        """
        tf = TaskFactory()

        # Get all protein residues (excluding ligands)
        all_protein_indices = [
            i for i in range(1, pose.size() + 1)
            if i not in self.ligand_pose_indices and pose.residue(i).is_protein()
        ]

        if self.global_repack:
            # Mode 1: Global repacking - all protein residues
            LOGGER.info("Building TaskFactory for GLOBAL repacking...")

            if all_protein_indices:
                protein_selector = ResidueIndexSelector(','.join(map(str, all_protein_indices)))

                # Restrict all protein to repacking only (no design)
                restrict_repack = OperateOnResidueSubset(RestrictToRepackingRLT(), protein_selector)
                tf.push_back(restrict_repack)

            repack_count = len(all_protein_indices)
            LOGGER.info(f"  Repacking enabled for {repack_count} protein residues (GLOBAL)")

        elif self.repack_shell is not None and self.repack_shell > 0:
            # Mode 2: Repack shell - mobile region + shell around it
            LOGGER.info(f"Building TaskFactory with {self.repack_shell}A repack shell...")

            mobile_non_lig = [r for r in mobile_residues if r not in self.ligand_pose_indices]

            if mobile_non_lig:
                mobile_selector = ResidueIndexSelector(','.join(map(str, mobile_non_lig)))

                # Expand to include shell
                shell_selector = NeighborhoodResidueSelector(
                    mobile_selector, distance=self.repack_shell, include_focus_in_subset=True
                )

                # Get the expanded set
                shell_selected = shell_selector.apply(pose)
                expanded_residues = [
                    i for i in range(1, pose.size() + 1)
                    if shell_selected[i] and i not in self.ligand_pose_indices
                ]

                if expanded_residues:
                    expanded_selector = ResidueIndexSelector(','.join(map(str, expanded_residues)))

                    # Prevent repacking outside expanded region
                    non_expanded_selector = NotResidueSelector(expanded_selector)
                    prevent_repack = OperateOnResidueSubset(PreventRepackingRLT(), non_expanded_selector)
                    tf.push_back(prevent_repack)

                    # Restrict expanded region to repacking only
                    restrict_repack = OperateOnResidueSubset(RestrictToRepackingRLT(), expanded_selector)
                    tf.push_back(restrict_repack)

                LOGGER.info(f"  Repacking enabled for {len(expanded_residues)} residues "
                           f"(mobile + {self.repack_shell}A shell)")
            else:
                LOGGER.info("  No mobile residues for shell expansion")

        else:
            # Mode 3: Default - only mobile residues
            LOGGER.info("Building TaskFactory for LOCAL repacking (mobile region only)...")

            mobile_non_lig = [r for r in mobile_residues if r not in self.ligand_pose_indices]

            if mobile_non_lig:
                mobile_selector = ResidueIndexSelector(','.join(map(str, mobile_non_lig)))

                # Prevent repacking on non-mobile residues
                non_mobile_selector = NotResidueSelector(mobile_selector)
                prevent_repack = OperateOnResidueSubset(PreventRepackingRLT(), non_mobile_selector)
                tf.push_back(prevent_repack)

                # Restrict mobile residues to repacking only (no design)
                restrict_repack = OperateOnResidueSubset(RestrictToRepackingRLT(), mobile_selector)
                tf.push_back(restrict_repack)

            LOGGER.info(f"  Repacking enabled for {len(mobile_non_lig)} mobile residues")

        # Always prevent repacking on ligands
        if self.ligand_pose_indices:
            lig_selector = ResidueIndexSelector(','.join(map(str, self.ligand_pose_indices)))
            prevent_lig_repack = OperateOnResidueSubset(PreventRepackingRLT(), lig_selector)
            tf.push_back(prevent_lig_repack)

        return tf

    def identify_stuck_residues(self, pose, prev_bond_metrics: Dict, curr_bond_metrics: Dict,
                                improvement_threshold: float = 0.005) -> List[int]:
        """
        Identify residues with persistent geometry problems (not improving).

        Args:
            pose: Current pose
            prev_bond_metrics: Bond metrics from previous round
            curr_bond_metrics: Bond metrics from current round
            improvement_threshold: Minimum improvement to not be considered stuck

        Returns:
            List of pose indices for stuck residues
        """
        stuck = set()

        # Get residues from worst bonds in current round
        for bond_info in curr_bond_metrics.get('worst_bonds', [])[:20]:
            residue_pdb = bond_info['residue']
            deviation = bond_info['deviation']

            # Only consider residues above tolerance
            if deviation <= self.bond_length_tolerance:
                continue

            # Check if this residue was also bad in previous round
            prev_worst = {b['residue']: b['deviation'] for b in prev_bond_metrics.get('worst_bonds', [])}
            if residue_pdb in prev_worst:
                improvement = prev_worst[residue_pdb] - deviation
                if improvement < improvement_threshold:
                    # This residue is stuck - find its pose index
                    chain = residue_pdb[0]
                    resno = int(residue_pdb[1:])
                    pose_idx = pose.pdb_info().pdb2pose(chain, resno)
                    if pose_idx > 0:
                        stuck.add(pose_idx)

        if stuck:
            LOGGER.info(f"  Identified {len(stuck)} stuck residues with persistent geometry issues")

        return list(stuck)

    def expand_mobile_region(self, pose, current_mobile: List[int],
                            stuck_residues: List[int]) -> List[int]:
        """
        Expand mobile region to include neighborhood around stuck residues.

        Args:
            pose: Rosetta pose
            current_mobile: Current list of mobile residue indices
            stuck_residues: List of stuck residue indices to expand around

        Returns:
            New list of mobile residue indices
        """
        if not stuck_residues:
            return current_mobile

        LOGGER.info(f"Expanding mobile region around {len(stuck_residues)} stuck residues "
                   f"(radius={self.expansion_radius}A)...")

        # Create selector for stuck residues
        stuck_selector = ResidueIndexSelector(','.join(map(str, stuck_residues)))

        # Get expanded neighborhood
        expansion_selector = NeighborhoodResidueSelector(
            stuck_selector, distance=self.expansion_radius, include_focus_in_subset=True
        )

        expanded = expansion_selector.apply(pose)
        new_mobile = set(current_mobile)

        for i in range(1, pose.size() + 1):
            if expanded[i] and i not in self.ligand_pose_indices:
                new_mobile.add(i)

        added = len(new_mobile) - len(current_mobile)
        LOGGER.info(f"  Added {added} residues to mobile region (now {len(new_mobile)} total)")

        return sorted(new_mobile)

    def calculate_bond_metrics(self, pose) -> Dict:
        """
        Calculate bond length deviations from ideal.

        Returns metrics split into:
        - "all": All bonds (including between constrained atoms)
        - "unconstrained_only": Bonds with at least one unconstrained atom
          (these are the ones we can actually optimize)

        Constrained atoms come from the theozyme and are considered ground truth,
        so we only need to optimize bonds involving unconstrained atoms.
        """
        all_bond_deviations = []
        unconstrained_bond_deviations = []

        for i in range(1, pose.size() + 1):
            residue = pose.residue(i)
            if residue.is_ligand():
                continue

            # Create ideal reference (A-X-A tripeptide)
            try:
                ref_pose = pyr.pose_from_sequence("A" + residue.name1() + "A")
                ref_res = ref_pose.residue(2)
            except:
                continue

            chain = pose.pdb_info().chain(i)
            resno = pose.pdb_info().number(i)
            pdb_id = f"{chain}{resno}"

            # Check bonds (heavy atoms only)
            for atom_idx in range(1, residue.natoms() + 1):
                if residue.atom_type(atom_idx).element() == "H":
                    continue
                atom_name = residue.atom_name(atom_idx).strip()

                if not ref_res.has(atom_name):
                    continue

                # Check if this atom is constrained
                atom1_constrained = self.is_atom_constrained(chain, resno, atom_name)

                for bonded_idx in residue.bonded_neighbor(atom_idx):
                    if bonded_idx <= atom_idx:  # Avoid double counting
                        continue
                    if residue.atom_type(bonded_idx).element() == "H":
                        continue
                    bonded_name = residue.atom_name(bonded_idx).strip()

                    if not ref_res.has(bonded_name):
                        continue

                    # Check if bonded atom is constrained
                    atom2_constrained = self.is_atom_constrained(chain, resno, bonded_name)
                    both_constrained = atom1_constrained and atom2_constrained

                    actual = (residue.xyz(atom_name) - residue.xyz(bonded_name)).norm()
                    ideal = (ref_res.xyz(atom_name) - ref_res.xyz(bonded_name)).norm()
                    dev = abs(actual - ideal)

                    bond_info = {
                        "residue": pdb_id,
                        "resname": residue.name3(),
                        "bond": f"{atom_name}-{bonded_name}",
                        "actual": float(actual),
                        "ideal": float(ideal),
                        "deviation": float(dev),
                        "both_constrained": both_constrained,
                        "atom1_constrained": atom1_constrained,
                        "atom2_constrained": atom2_constrained,
                    }

                    all_bond_deviations.append(bond_info)

                    # Only add to unconstrained list if at least one atom is NOT constrained
                    if not both_constrained:
                        unconstrained_bond_deviations.append(bond_info)

        def aggregate_stats(deviations, label=""):
            if deviations:
                devs = [b["deviation"] for b in deviations]
                sorted_devs = sorted(deviations, key=lambda x: x["deviation"], reverse=True)
                return {
                    "mean_deviation": float(np.mean(devs)),
                    "max_deviation": float(np.max(devs)),
                    "num_bonds": len(devs),
                    "worst_bonds": sorted_devs[:10]
                }
            else:
                return {
                    "mean_deviation": 0.0,
                    "max_deviation": 0.0,
                    "num_bonds": 0,
                    "worst_bonds": []
                }

        return {
            "all": aggregate_stats(all_bond_deviations),
            "unconstrained_only": aggregate_stats(unconstrained_bond_deviations),
        }

    def calculate_angle_metrics(self, pose) -> Dict:
        """
        Calculate bond angle deviations from ideal.

        For each atom with 2+ bonded neighbors, calculate the angle between them
        and compare to the ideal angle from a reference structure.

        Returns metrics split into:
        - "all": All angles (including those where all 3 atoms are constrained)
        - "unconstrained_only": Angles with at least one unconstrained atom
          (these are the ones we can actually optimize)

        Constrained atoms come from the theozyme and are considered ground truth,
        so we only need to optimize angles involving unconstrained atoms.
        """
        import math
        all_angle_deviations = []
        unconstrained_angle_deviations = []

        def calc_angle(xyz1, xyz2, xyz3):
            """Calculate angle at xyz2 between xyz1-xyz2-xyz3 in degrees."""
            v1 = xyz1 - xyz2
            v2 = xyz3 - xyz2
            cos_angle = v1.dot(v2) / (v1.norm() * v2.norm())
            # Clamp to [-1, 1] to avoid numerical issues
            cos_angle = max(-1.0, min(1.0, cos_angle))
            return math.degrees(math.acos(cos_angle))

        for i in range(1, pose.size() + 1):
            residue = pose.residue(i)
            if residue.is_ligand():
                continue

            # Create ideal reference (A-X-A tripeptide)
            try:
                ref_pose = pyr.pose_from_sequence("A" + residue.name1() + "A")
                ref_res = ref_pose.residue(2)
            except:
                continue

            chain = pose.pdb_info().chain(i)
            resno = pose.pdb_info().number(i)
            pdb_id = f"{chain}{resno}"

            # Check angles (heavy atoms only)
            for center_idx in range(1, residue.natoms() + 1):
                if residue.atom_type(center_idx).element() == "H":
                    continue
                center_name = residue.atom_name(center_idx).strip()

                if not ref_res.has(center_name):
                    continue

                # Check if center atom is constrained
                center_constrained = self.is_atom_constrained(chain, resno, center_name)

                # Get bonded neighbors (heavy atoms only)
                neighbors = [
                    n for n in residue.bonded_neighbor(center_idx)
                    if residue.atom_type(n).element() != "H"
                ]

                # Need at least 2 neighbors to form an angle
                if len(neighbors) < 2:
                    continue

                # Check all pairs of neighbors
                for j in range(len(neighbors)):
                    for k in range(j + 1, len(neighbors)):
                        n1_idx, n2_idx = neighbors[j], neighbors[k]
                        n1_name = residue.atom_name(n1_idx).strip()
                        n2_name = residue.atom_name(n2_idx).strip()

                        if not ref_res.has(n1_name) or not ref_res.has(n2_name):
                            continue

                        # Check if neighbor atoms are constrained
                        n1_constrained = self.is_atom_constrained(chain, resno, n1_name)
                        n2_constrained = self.is_atom_constrained(chain, resno, n2_name)
                        all_constrained = center_constrained and n1_constrained and n2_constrained

                        # Calculate actual angle
                        actual_angle = calc_angle(
                            residue.xyz(n1_name),
                            residue.xyz(center_name),
                            residue.xyz(n2_name)
                        )

                        # Calculate ideal angle
                        ideal_angle = calc_angle(
                            ref_res.xyz(n1_name),
                            ref_res.xyz(center_name),
                            ref_res.xyz(n2_name)
                        )

                        dev = abs(actual_angle - ideal_angle)

                        angle_info = {
                            "residue": pdb_id,
                            "resname": residue.name3(),
                            "angle": f"{n1_name}-{center_name}-{n2_name}",
                            "actual": float(actual_angle),
                            "ideal": float(ideal_angle),
                            "deviation": float(dev),
                            "all_constrained": all_constrained,
                            "atoms_constrained": {
                                n1_name: n1_constrained,
                                center_name: center_constrained,
                                n2_name: n2_constrained,
                            }
                        }

                        all_angle_deviations.append(angle_info)

                        # Only add to unconstrained list if at least one atom is NOT constrained
                        if not all_constrained:
                            unconstrained_angle_deviations.append(angle_info)

        def aggregate_stats(deviations):
            if deviations:
                devs = [a["deviation"] for a in deviations]
                sorted_devs = sorted(deviations, key=lambda x: x["deviation"], reverse=True)
                return {
                    "mean_deviation": float(np.mean(devs)),
                    "max_deviation": float(np.max(devs)),
                    "num_angles": len(devs),
                    "worst_angles": sorted_devs[:10]
                }
            else:
                return {
                    "mean_deviation": 0.0,
                    "max_deviation": 0.0,
                    "num_angles": 0,
                    "worst_angles": []
                }

        return {
            "all": aggregate_stats(all_angle_deviations),
            "unconstrained_only": aggregate_stats(unconstrained_angle_deviations),
        }

    def calculate_offender_metrics(self, pose) -> Dict:
        """
        Calculate comprehensive geometry offender metrics with severity breakdown.

        Categorizes bond length and angle deviations into:
        - severe: > severe_threshold
        - moderate: between moderate_threshold and severe_threshold
        - minor: between tolerance and moderate_threshold

        Separates into "all" (including constrained) and "unconstrained_only" metrics.
        Constrained atoms come from the theozyme and are considered ground truth,
        so we focus on optimizing unconstrained bonds/angles.

        Returns:
            Dict with bond_length, bond_angle, and catres_offender_summary
        """
        bond_metrics = self.calculate_bond_metrics(pose)
        angle_metrics = self.calculate_angle_metrics(pose)

        # Categorize bond length offenders - separate all vs unconstrained
        bond_severe_all = []
        bond_moderate_all = []
        bond_minor_all = []
        bond_severe_unconstrained = []
        bond_moderate_unconstrained = []
        bond_minor_unconstrained = []
        per_residue_bonds = {}
        per_residue_bonds_unconstrained = {}

        for i in range(1, pose.size() + 1):
            residue = pose.residue(i)
            if residue.is_ligand():
                continue

            try:
                ref_pose = pyr.pose_from_sequence("A" + residue.name1() + "A")
                ref_res = ref_pose.residue(2)
            except:
                continue

            chain = pose.pdb_info().chain(i)
            resno = pose.pdb_info().number(i)
            pdb_id = f"{chain}{resno}"

            residue_max_bond_dev = 0.0
            residue_worst_bond = None
            residue_num_bad_bonds = 0
            # For unconstrained only
            residue_max_bond_dev_unconstrained = 0.0
            residue_worst_bond_unconstrained = None
            residue_num_bad_bonds_unconstrained = 0

            for atom_idx in range(1, residue.natoms() + 1):
                if residue.atom_type(atom_idx).element() == "H":
                    continue
                atom_name = residue.atom_name(atom_idx).strip()
                if not ref_res.has(atom_name):
                    continue

                atom1_constrained = self.is_atom_constrained(chain, resno, atom_name)

                for bonded_idx in residue.bonded_neighbor(atom_idx):
                    if bonded_idx <= atom_idx:
                        continue
                    if residue.atom_type(bonded_idx).element() == "H":
                        continue
                    bonded_name = residue.atom_name(bonded_idx).strip()
                    if not ref_res.has(bonded_name):
                        continue

                    atom2_constrained = self.is_atom_constrained(chain, resno, bonded_name)
                    both_constrained = atom1_constrained and atom2_constrained

                    actual = (residue.xyz(atom_name) - residue.xyz(bonded_name)).norm()
                    ideal = (ref_res.xyz(atom_name) - ref_res.xyz(bonded_name)).norm()
                    dev = abs(actual - ideal)

                    if dev > self.bond_length_tolerance:
                        # Track for "all" metrics
                        residue_num_bad_bonds += 1
                        if dev > residue_max_bond_dev:
                            residue_max_bond_dev = dev
                            residue_worst_bond = f"{atom_name}-{bonded_name}"

                        # Categorize for "all"
                        if dev > self.severe_bond_threshold:
                            if pdb_id not in bond_severe_all:
                                bond_severe_all.append(pdb_id)
                        elif dev > self.moderate_bond_threshold:
                            if pdb_id not in bond_moderate_all and pdb_id not in bond_severe_all:
                                bond_moderate_all.append(pdb_id)
                        else:
                            if pdb_id not in bond_minor_all and pdb_id not in bond_moderate_all and pdb_id not in bond_severe_all:
                                bond_minor_all.append(pdb_id)

                        # Track for "unconstrained_only" metrics
                        if not both_constrained:
                            residue_num_bad_bonds_unconstrained += 1
                            if dev > residue_max_bond_dev_unconstrained:
                                residue_max_bond_dev_unconstrained = dev
                                residue_worst_bond_unconstrained = f"{atom_name}-{bonded_name}"

                            # Categorize for "unconstrained_only"
                            if dev > self.severe_bond_threshold:
                                if pdb_id not in bond_severe_unconstrained:
                                    bond_severe_unconstrained.append(pdb_id)
                            elif dev > self.moderate_bond_threshold:
                                if pdb_id not in bond_moderate_unconstrained and pdb_id not in bond_severe_unconstrained:
                                    bond_moderate_unconstrained.append(pdb_id)
                            else:
                                if pdb_id not in bond_minor_unconstrained and pdb_id not in bond_moderate_unconstrained and pdb_id not in bond_severe_unconstrained:
                                    bond_minor_unconstrained.append(pdb_id)

            if residue_num_bad_bonds > 0:
                # Check if this is a catres
                is_catres = i in self.catres_pose_indices
                catres_block = None
                if is_catres:
                    for block_idx, info in self.residue_constraints.items():
                        if info["chain"] == chain and info["resno"] == resno:
                            catres_block = int(block_idx)
                            break

                per_residue_bonds[pdb_id] = {
                    "resname": residue.name3(),
                    "worst_bond": residue_worst_bond,
                    "max_deviation": float(residue_max_bond_dev),
                    "num_bad_bonds": residue_num_bad_bonds,
                    "is_catres": is_catres,
                    "catres_block": catres_block
                }

            if residue_num_bad_bonds_unconstrained > 0:
                is_catres = i in self.catres_pose_indices
                catres_block = None
                if is_catres:
                    for block_idx, info in self.residue_constraints.items():
                        if info["chain"] == chain and info["resno"] == resno:
                            catres_block = int(block_idx)
                            break

                per_residue_bonds_unconstrained[pdb_id] = {
                    "resname": residue.name3(),
                    "worst_bond": residue_worst_bond_unconstrained,
                    "max_deviation": float(residue_max_bond_dev_unconstrained),
                    "num_bad_bonds": residue_num_bad_bonds_unconstrained,
                    "is_catres": is_catres,
                    "catres_block": catres_block
                }

        # Similar processing for angles - separate all vs unconstrained
        import math
        def calc_angle(xyz1, xyz2, xyz3):
            v1 = xyz1 - xyz2
            v2 = xyz3 - xyz2
            cos_angle = v1.dot(v2) / (v1.norm() * v2.norm())
            cos_angle = max(-1.0, min(1.0, cos_angle))
            return math.degrees(math.acos(cos_angle))

        angle_severe_all = []
        angle_moderate_all = []
        angle_minor_all = []
        angle_severe_unconstrained = []
        angle_moderate_unconstrained = []
        angle_minor_unconstrained = []
        per_residue_angles = {}
        per_residue_angles_unconstrained = {}

        for i in range(1, pose.size() + 1):
            residue = pose.residue(i)
            if residue.is_ligand():
                continue

            try:
                ref_pose = pyr.pose_from_sequence("A" + residue.name1() + "A")
                ref_res = ref_pose.residue(2)
            except:
                continue

            chain = pose.pdb_info().chain(i)
            resno = pose.pdb_info().number(i)
            pdb_id = f"{chain}{resno}"

            residue_max_angle_dev = 0.0
            residue_worst_angle = None
            residue_num_bad_angles = 0
            # For unconstrained only
            residue_max_angle_dev_unconstrained = 0.0
            residue_worst_angle_unconstrained = None
            residue_num_bad_angles_unconstrained = 0

            for center_idx in range(1, residue.natoms() + 1):
                if residue.atom_type(center_idx).element() == "H":
                    continue
                center_name = residue.atom_name(center_idx).strip()
                if not ref_res.has(center_name):
                    continue

                center_constrained = self.is_atom_constrained(chain, resno, center_name)

                neighbors = [n for n in residue.bonded_neighbor(center_idx)
                            if residue.atom_type(n).element() != "H"]
                if len(neighbors) < 2:
                    continue

                for j in range(len(neighbors)):
                    for k in range(j + 1, len(neighbors)):
                        n1_idx, n2_idx = neighbors[j], neighbors[k]
                        n1_name = residue.atom_name(n1_idx).strip()
                        n2_name = residue.atom_name(n2_idx).strip()

                        if not ref_res.has(n1_name) or not ref_res.has(n2_name):
                            continue

                        n1_constrained = self.is_atom_constrained(chain, resno, n1_name)
                        n2_constrained = self.is_atom_constrained(chain, resno, n2_name)
                        all_constrained = center_constrained and n1_constrained and n2_constrained

                        actual_angle = calc_angle(
                            residue.xyz(n1_name), residue.xyz(center_name), residue.xyz(n2_name)
                        )
                        ideal_angle = calc_angle(
                            ref_res.xyz(n1_name), ref_res.xyz(center_name), ref_res.xyz(n2_name)
                        )
                        dev = abs(actual_angle - ideal_angle)

                        if dev > self.bond_angle_tolerance:
                            # Track for "all" metrics
                            residue_num_bad_angles += 1
                            if dev > residue_max_angle_dev:
                                residue_max_angle_dev = dev
                                residue_worst_angle = f"{n1_name}-{center_name}-{n2_name}"

                            # Categorize for "all"
                            if dev > self.severe_angle_threshold:
                                if pdb_id not in angle_severe_all:
                                    angle_severe_all.append(pdb_id)
                            elif dev > self.moderate_angle_threshold:
                                if pdb_id not in angle_moderate_all and pdb_id not in angle_severe_all:
                                    angle_moderate_all.append(pdb_id)
                            else:
                                if pdb_id not in angle_minor_all and pdb_id not in angle_moderate_all and pdb_id not in angle_severe_all:
                                    angle_minor_all.append(pdb_id)

                            # Track for "unconstrained_only" metrics
                            if not all_constrained:
                                residue_num_bad_angles_unconstrained += 1
                                if dev > residue_max_angle_dev_unconstrained:
                                    residue_max_angle_dev_unconstrained = dev
                                    residue_worst_angle_unconstrained = f"{n1_name}-{center_name}-{n2_name}"

                                # Categorize for "unconstrained_only"
                                if dev > self.severe_angle_threshold:
                                    if pdb_id not in angle_severe_unconstrained:
                                        angle_severe_unconstrained.append(pdb_id)
                                elif dev > self.moderate_angle_threshold:
                                    if pdb_id not in angle_moderate_unconstrained and pdb_id not in angle_severe_unconstrained:
                                        angle_moderate_unconstrained.append(pdb_id)
                                else:
                                    if pdb_id not in angle_minor_unconstrained and pdb_id not in angle_moderate_unconstrained and pdb_id not in angle_severe_unconstrained:
                                        angle_minor_unconstrained.append(pdb_id)

            if residue_num_bad_angles > 0:
                is_catres = i in self.catres_pose_indices
                catres_block = None
                if is_catres:
                    for block_idx, info in self.residue_constraints.items():
                        if info["chain"] == chain and info["resno"] == resno:
                            catres_block = int(block_idx)
                            break

                per_residue_angles[pdb_id] = {
                    "resname": residue.name3(),
                    "worst_angle": residue_worst_angle,
                    "max_deviation": float(residue_max_angle_dev),
                    "num_bad_angles": residue_num_bad_angles,
                    "is_catres": is_catres,
                    "catres_block": catres_block
                }

            if residue_num_bad_angles_unconstrained > 0:
                is_catres = i in self.catres_pose_indices
                catres_block = None
                if is_catres:
                    for block_idx, info in self.residue_constraints.items():
                        if info["chain"] == chain and info["resno"] == resno:
                            catres_block = int(block_idx)
                            break

                per_residue_angles_unconstrained[pdb_id] = {
                    "resname": residue.name3(),
                    "worst_angle": residue_worst_angle_unconstrained,
                    "max_deviation": float(residue_max_angle_dev_unconstrained),
                    "num_bad_angles": residue_num_bad_angles_unconstrained,
                    "is_catres": is_catres,
                    "catres_block": catres_block
                }

        # Build catres-specific offender summary (using UNCONSTRAINED metrics - what we can actually fix)
        catres_offenders = []
        catres_with_issues = set()

        for block_idx, info in self.residue_constraints.items():
            pdb_id = f"{info['chain']}{info['resno']}"
            issues = []
            issues_unconstrained = []

            # Check unconstrained bond issues (these are fixable)
            if pdb_id in per_residue_bonds_unconstrained:
                bond_info = per_residue_bonds_unconstrained[pdb_id]
                if bond_info['max_deviation'] > self.severe_bond_threshold:
                    issues_unconstrained.append(f"bond_length: severe ({bond_info['max_deviation']:.3f}A) [{bond_info['worst_bond']}]")
                elif bond_info['max_deviation'] > self.moderate_bond_threshold:
                    issues_unconstrained.append(f"bond_length: moderate ({bond_info['max_deviation']:.3f}A) [{bond_info['worst_bond']}]")
                else:
                    issues_unconstrained.append(f"bond_length: minor ({bond_info['max_deviation']:.3f}A) [{bond_info['worst_bond']}]")

            # Check unconstrained angle issues (these are fixable)
            if pdb_id in per_residue_angles_unconstrained:
                angle_info = per_residue_angles_unconstrained[pdb_id]
                if angle_info['max_deviation'] > self.severe_angle_threshold:
                    issues_unconstrained.append(f"bond_angle: severe ({angle_info['max_deviation']:.1f}deg) [{angle_info['worst_angle']}]")
                elif angle_info['max_deviation'] > self.moderate_angle_threshold:
                    issues_unconstrained.append(f"bond_angle: moderate ({angle_info['max_deviation']:.1f}deg) [{angle_info['worst_angle']}]")
                else:
                    issues_unconstrained.append(f"bond_angle: minor ({angle_info['max_deviation']:.1f}deg) [{angle_info['worst_angle']}]")

            if issues_unconstrained:
                catres_with_issues.add(block_idx)
                catres_offenders.append({
                    "block": int(block_idx),
                    "pdb_id": pdb_id,
                    "resname": info['resname'],
                    "issues_unconstrained": issues_unconstrained,
                    # Also include constrained bond/angle info for reference
                    "bond_info_all": per_residue_bonds.get(pdb_id),
                    "angle_info_all": per_residue_angles.get(pdb_id),
                })

        return {
            "bond_length": {
                "all": {
                    "total_offenders": len(per_residue_bonds),
                    "severe": {"count": len(bond_severe_all), "residues": bond_severe_all},
                    "moderate": {"count": len(bond_moderate_all), "residues": bond_moderate_all},
                    "minor": {"count": len(bond_minor_all), "residues": bond_minor_all},
                    "per_residue_summary": per_residue_bonds
                },
                "unconstrained_only": {
                    "total_offenders": len(per_residue_bonds_unconstrained),
                    "severe": {"count": len(bond_severe_unconstrained), "residues": bond_severe_unconstrained},
                    "moderate": {"count": len(bond_moderate_unconstrained), "residues": bond_moderate_unconstrained},
                    "minor": {"count": len(bond_minor_unconstrained), "residues": bond_minor_unconstrained},
                    "per_residue_summary": per_residue_bonds_unconstrained
                }
            },
            "bond_angle": {
                "all": {
                    "total_offenders": len(per_residue_angles),
                    "severe": {"count": len(angle_severe_all), "residues": angle_severe_all},
                    "moderate": {"count": len(angle_moderate_all), "residues": angle_moderate_all},
                    "minor": {"count": len(angle_minor_all), "residues": angle_minor_all},
                    "per_residue_summary": per_residue_angles
                },
                "unconstrained_only": {
                    "total_offenders": len(per_residue_angles_unconstrained),
                    "severe": {"count": len(angle_severe_unconstrained), "residues": angle_severe_unconstrained},
                    "moderate": {"count": len(angle_moderate_unconstrained), "residues": angle_moderate_unconstrained},
                    "minor": {"count": len(angle_minor_unconstrained), "residues": angle_minor_unconstrained},
                    "per_residue_summary": per_residue_angles_unconstrained
                }
            },
            "catres_offender_summary": {
                "num_catres_with_unconstrained_issues": len(catres_with_issues),
                "total_catres": len(self.residue_constraints),
                "catres_offenders": catres_offenders
            }
        }

    def calculate_catres_geometry_status(self, pose) -> Dict:
        """
        Calculate geometry status specifically for catalytic residues.

        Uses catres-specific tolerances which can be stricter than global tolerances.
        Provides pass/fail status for each catres.

        Separates metrics into:
        - "all": All bonds/angles (including between constrained atoms)
        - "unconstrained_only": Only bonds/angles with at least one unconstrained atom
          (these are the ones we can actually optimize)

        Pass/fail is determined by unconstrained_only metrics since constrained atoms
        come from the theozyme and are considered ground truth.

        Returns:
            Dict with all_converged, num_passing, num_failing, and per-catres details
        """
        import math

        def calc_angle(xyz1, xyz2, xyz3):
            v1 = xyz1 - xyz2
            v2 = xyz3 - xyz2
            cos_angle = v1.dot(v2) / (v1.norm() * v2.norm())
            cos_angle = max(-1.0, min(1.0, cos_angle))
            return math.degrees(math.acos(cos_angle))

        passing = []
        failing = []

        for block_idx, info in self.residue_constraints.items():
            chain, resno, resname = info["chain"], info["resno"], info["resname"]
            pdb_id = f"{chain}{resno}"

            pose_idx = pose.pdb_info().pdb2pose(chain, resno)
            if pose_idx == 0:
                continue

            residue = pose.residue(pose_idx)

            # Calculate max bond/angle deviation - separate all vs unconstrained
            max_bond_dev_all = 0.0
            max_angle_dev_all = 0.0
            max_bond_dev_unconstrained = 0.0
            max_angle_dev_unconstrained = 0.0
            worst_bond_unconstrained = None
            worst_angle_unconstrained = None

            try:
                ref_pose = pyr.pose_from_sequence("A" + residue.name1() + "A")
                ref_res = ref_pose.residue(2)

                # Check bonds
                for atom_idx in range(1, residue.natoms() + 1):
                    if residue.atom_type(atom_idx).element() == "H":
                        continue
                    atom_name = residue.atom_name(atom_idx).strip()
                    if not ref_res.has(atom_name):
                        continue

                    atom1_constrained = self.is_atom_constrained(chain, resno, atom_name)

                    for bonded_idx in residue.bonded_neighbor(atom_idx):
                        if bonded_idx <= atom_idx:
                            continue
                        if residue.atom_type(bonded_idx).element() == "H":
                            continue
                        bonded_name = residue.atom_name(bonded_idx).strip()
                        if not ref_res.has(bonded_name):
                            continue

                        atom2_constrained = self.is_atom_constrained(chain, resno, bonded_name)
                        both_constrained = atom1_constrained and atom2_constrained

                        actual = (residue.xyz(atom_name) - residue.xyz(bonded_name)).norm()
                        ideal = (ref_res.xyz(atom_name) - ref_res.xyz(bonded_name)).norm()
                        dev = abs(actual - ideal)

                        # Track for "all"
                        max_bond_dev_all = max(max_bond_dev_all, dev)

                        # Track for "unconstrained_only"
                        if not both_constrained:
                            if dev > max_bond_dev_unconstrained:
                                max_bond_dev_unconstrained = dev
                                worst_bond_unconstrained = f"{atom_name}-{bonded_name}"

                # Check angles
                for center_idx in range(1, residue.natoms() + 1):
                    if residue.atom_type(center_idx).element() == "H":
                        continue
                    center_name = residue.atom_name(center_idx).strip()
                    if not ref_res.has(center_name):
                        continue

                    center_constrained = self.is_atom_constrained(chain, resno, center_name)

                    neighbors = [n for n in residue.bonded_neighbor(center_idx)
                                if residue.atom_type(n).element() != "H"]
                    if len(neighbors) < 2:
                        continue

                    for j in range(len(neighbors)):
                        for k in range(j + 1, len(neighbors)):
                            n1_idx, n2_idx = neighbors[j], neighbors[k]
                            n1_name = residue.atom_name(n1_idx).strip()
                            n2_name = residue.atom_name(n2_idx).strip()

                            if not ref_res.has(n1_name) or not ref_res.has(n2_name):
                                continue

                            n1_constrained = self.is_atom_constrained(chain, resno, n1_name)
                            n2_constrained = self.is_atom_constrained(chain, resno, n2_name)
                            all_constrained = center_constrained and n1_constrained and n2_constrained

                            actual_angle = calc_angle(
                                residue.xyz(n1_name), residue.xyz(center_name), residue.xyz(n2_name)
                            )
                            ideal_angle = calc_angle(
                                ref_res.xyz(n1_name), ref_res.xyz(center_name), ref_res.xyz(n2_name)
                            )
                            dev = abs(actual_angle - ideal_angle)

                            # Track for "all"
                            max_angle_dev_all = max(max_angle_dev_all, dev)

                            # Track for "unconstrained_only"
                            if not all_constrained:
                                if dev > max_angle_dev_unconstrained:
                                    max_angle_dev_unconstrained = dev
                                    worst_angle_unconstrained = f"{n1_name}-{center_name}-{n2_name}"

            except Exception as e:
                LOGGER.warning(f"Could not calculate geometry for {pdb_id}: {e}")
                continue

            # Check against catres-specific tolerances using UNCONSTRAINED metrics
            # (constrained atoms are from theozyme = ground truth)
            bond_ok = max_bond_dev_unconstrained <= self.catres_bond_tolerance
            angle_ok = max_angle_dev_unconstrained <= self.catres_angle_tolerance

            catres_info = {
                "block": int(block_idx),
                "pdb_id": pdb_id,
                "resname": resname,
                # Unconstrained metrics (used for pass/fail)
                "max_bond_dev": float(max_bond_dev_unconstrained),
                "max_angle_dev": float(max_angle_dev_unconstrained),
                "worst_bond": worst_bond_unconstrained,
                "worst_angle": worst_angle_unconstrained,
                # All metrics (for reference)
                "max_bond_dev_all": float(max_bond_dev_all),
                "max_angle_dev_all": float(max_angle_dev_all),
                "bond_ok": bond_ok,
                "angle_ok": angle_ok,
                "passed": bond_ok and angle_ok
            }

            if bond_ok and angle_ok:
                passing.append(catres_info)
            else:
                failing.append(catres_info)

        all_converged = len(failing) == 0

        # Log warning if catres geometry not converged and require_catres_converged is True
        if not all_converged and self.require_catres_converged:
            LOGGER.warning(f"CATRES GEOMETRY WARNING: {len(failing)}/{len(self.residue_constraints)} "
                          f"catalytic residues have geometry outside tolerances!")
            for f in failing:
                LOGGER.warning(f"  Block {f['block']} ({f['pdb_id']} {f['resname']}): "
                              f"bond={f['max_bond_dev']:.4f}A (tol={self.catres_bond_tolerance}A), "
                              f"angle={f['max_angle_dev']:.2f}deg (tol={self.catres_angle_tolerance}deg)")

        return {
            "all_converged": all_converged,
            "num_passing": len(passing),
            "num_failing": len(failing),
            "catres_bond_tolerance": self.catres_bond_tolerance,
            "catres_angle_tolerance": self.catres_angle_tolerance,
            "passing_catres": passing,
            "failing_catres": failing
        }

    def attempt_loop_rebuild(self, pose, severely_distorted_residues: List[int]) -> bool:
        """
        EXPERIMENTAL: Attempt KIC loop rebuild for severely distorted regions.

        This is an emergency fallback for cases where normal relaxation cannot
        fix severe geometry distortions. Uses Kinematic Closure (KIC) to rebuild
        small loop segments.

        Args:
            pose: Rosetta pose
            severely_distorted_residues: List of pose indices with severe distortions

        Returns:
            True if rebuild was attempted, False otherwise

        Note:
            This requires PyRosetta built with loop modeling support.
            Most geometry issues should be fixable with Cartesian FastRelax +
            bondangle/bondlength minimization - use this only as last resort.
        """
        if not self.enable_loop_rebuild:
            return False

        if not severely_distorted_residues:
            return False

        LOGGER.warning("="*70)
        LOGGER.warning("EXPERIMENTAL: Attempting loop rebuild for severely distorted regions")
        LOGGER.warning("="*70)

        try:
            from pyrosetta.rosetta.protocols.loops import Loops, Loop
            from pyrosetta.rosetta.protocols.loops.loop_mover.refine import LoopMover_Refine_KIC

            # Group consecutive residues into loop segments
            severely_distorted_residues.sort()
            loops = Loops()

            # Find contiguous segments
            start = severely_distorted_residues[0]
            end = start
            for res_idx in severely_distorted_residues[1:]:
                if res_idx == end + 1:
                    end = res_idx
                else:
                    # Add the previous segment as a loop (with 1-residue buffer)
                    loop_start = max(1, start - 1)
                    loop_end = min(pose.size(), end + 1)
                    cutpoint = (loop_start + loop_end) // 2
                    loops.add_loop(Loop(loop_start, loop_end, cutpoint))
                    start = res_idx
                    end = res_idx

            # Add final segment
            loop_start = max(1, start - 1)
            loop_end = min(pose.size(), end + 1)
            cutpoint = (loop_start + loop_end) // 2
            loops.add_loop(Loop(loop_start, loop_end, cutpoint))

            LOGGER.info(f"  Attempting KIC on {loops.num_loop()} loop segment(s)")

            # Apply KIC refinement
            kic_mover = LoopMover_Refine_KIC(loops)
            kic_mover.set_scorefxn(self.sfxn)
            kic_mover.apply(pose)

            LOGGER.info("  KIC loop rebuild completed")
            return True

        except ImportError as e:
            LOGGER.warning(f"  Loop modeling not available in this PyRosetta build: {e}")
            LOGGER.warning("  Skipping loop rebuild - try using --preset aggressive instead")
            return False
        except Exception as e:
            LOGGER.warning(f"  Loop rebuild failed: {e}")
            return False

    def detect_chain_breaks(self, pose) -> List[int]:
        """Detect peptide bond breaks (C-N distance > 2.0 A)."""
        breaks = []

        for i in range(1, pose.size()):
            try:
                if not pose.residue(i).is_protein() or not pose.residue(i + 1).is_protein():
                    continue

                c_xyz = pose.residue(i).xyz("C")
                n_xyz = pose.residue(i + 1).xyz("N")
                dist = c_xyz.distance(n_xyz)

                if dist > 2.0:  # Typical peptide bond ~ 1.33 A
                    breaks.append(i)
            except:
                continue

        return breaks

    def count_clashes(self, pose, threshold: float = 10.0) -> int:
        """Count residues with fa_rep > threshold."""
        self.sfxn(pose)  # Score to populate energies
        count = 0
        for i in range(1, pose.size() + 1):
            try:
                fa_rep = pose.energies().residue_total_energies(i)[ScoreType.fa_rep]
                if fa_rep > threshold:
                    count += 1
            except:
                continue
        return count

    def calculate_ligand_rmsd(self, pose_before, pose_after) -> float:
        """
        Calculate ligand heavy atom RMSD (no superposition needed).
        """
        if not self.ligand_info:
            return 0.0

        lig_chain, _, lig_resno = self.ligand_info

        pose_idx_before = pose_before.pdb_info().pdb2pose(lig_chain, lig_resno)
        pose_idx_after = pose_after.pdb_info().pdb2pose(lig_chain, lig_resno)

        if pose_idx_before == 0 or pose_idx_after == 0:
            return 0.0

        res_before = pose_before.residue(pose_idx_before)
        res_after = pose_after.residue(pose_idx_after)

        sq_devs = []
        for i in range(1, res_before.natoms() + 1):
            if res_before.atom_is_hydrogen(i):
                continue
            atom_name = res_before.atom_name(i).strip()

            if not res_after.has(atom_name):
                continue

            xyz_before = res_before.xyz(atom_name)
            xyz_after = res_after.xyz(atom_name)

            sq_devs.append(xyz_before.distance_squared(xyz_after))

        if sq_devs:
            return float(np.sqrt(np.mean(sq_devs)))
        return 0.0

    def calculate_constrained_atom_rmsd(self, pose_before, pose_after) -> Dict:
        """
        Calculate RMSD of constrained atoms (per residue and aggregate).
        """
        per_residue = {}
        all_sq_devs = []

        for (chain, resno), atom_spec in self.atoms_to_constrain.items():
            pose_idx_before = pose_before.pdb_info().pdb2pose(chain, resno)
            pose_idx_after = pose_after.pdb_info().pdb2pose(chain, resno)

            if pose_idx_before == 0 or pose_idx_after == 0:
                continue

            res_before = pose_before.residue(pose_idx_before)
            res_after = pose_after.residue(pose_idx_after)

            # Determine atoms
            if atom_spec == "ALL_HEAVY":
                atom_names = [
                    res_before.atom_name(i).strip()
                    for i in range(1, res_before.natoms() + 1)
                    if not res_before.atom_is_hydrogen(i)
                ]
            else:
                atom_names = atom_spec

            sq_devs = []
            for atom_name in atom_names:
                if not res_before.has(atom_name) or not res_after.has(atom_name):
                    continue

                xyz_before = res_before.xyz(atom_name)
                xyz_after = res_after.xyz(atom_name)

                sq_dev = xyz_before.distance_squared(xyz_after)
                sq_devs.append(sq_dev)
                all_sq_devs.append(sq_dev)

            if sq_devs:
                pdb_id = f"{chain}{resno}"
                per_residue[pdb_id] = {
                    "rmsd": float(np.sqrt(np.mean(sq_devs))),
                    "max_displacement": float(np.sqrt(np.max(sq_devs))),
                    "num_atoms": len(sq_devs)
                }

        aggregate_rmsd = float(np.sqrt(np.mean(all_sq_devs))) if all_sq_devs else 0.0

        return {
            "aggregate_rmsd": aggregate_rmsd,
            "per_residue": per_residue
        }

    def calculate_per_catres_metrics(self, pose) -> Dict:
        """
        Calculate detailed metrics for each catalytic residue.
        """
        self.sfxn(pose)  # Score to populate energies

        per_catres = {}

        for block_idx, info in self.residue_constraints.items():
            chain, resno, resname = info["chain"], info["resno"], info["resname"]
            pdb_id = f"{chain}{resno}"

            pose_idx = pose.pdb_info().pdb2pose(chain, resno)
            if pose_idx == 0:
                continue

            try:
                # Cart_bonded energy for this residue
                res_cart_bonded = pose.energies().residue_total_energies(pose_idx)[ScoreType.cart_bonded]

                # Fa_rep (clashes)
                res_fa_rep = pose.energies().residue_total_energies(pose_idx)[ScoreType.fa_rep]

                # Coordinate constraint energy
                res_coord_cst = pose.energies().residue_total_energies(pose_idx)[ScoreType.coordinate_constraint]

                per_catres[block_idx] = {
                    "pdb_id": pdb_id,
                    "resname": resname,
                    "cart_bonded": float(res_cart_bonded),
                    "fa_rep": float(res_fa_rep),
                    "coordinate_constraint": float(res_coord_cst),
                    "is_clashing": res_fa_rep > 10.0
                }
            except Exception as e:
                LOGGER.warning(f"Could not get metrics for {pdb_id}: {e}")

        return per_catres

    def generate_relax_script(self, num_stages: int = 4) -> List[str]:
        """
        Generate a custom relax script with specified number of ramping stages.

        The script ramps fa_rep from low to 1.0 across num_stages stages.
        IMPORTANT: We do NOT ramp down coordinate_constraint because we want
        our catres/ligand constraints to stay tight throughout.

        Args:
            num_stages: Number of ramping stages (default 4, like MonomerRelax2019)

        Returns:
            List of script lines for FastRelax.set_script_from_lines()
        """
        if num_stages < 1:
            num_stages = 1

        # Generate fa_rep scale values from ~0.02 to 1.0
        # Using similar progression to MonomerRelax2019
        if num_stages == 1:
            fa_rep_scales = [1.0]
        else:
            fa_rep_scales = []
            for i in range(num_stages):
                # fa_rep ramps from ~0.02 to 1.0
                t = i / (num_stages - 1) if num_stages > 1 else 1.0
                # Use a curve that matches default behavior roughly
                fa_rep = 0.02 + (1.0 - 0.02) * (t ** 0.7)
                fa_rep_scales.append(fa_rep)

        # Build script lines
        # Note: We don't use "repeat N" here because FastRelax(sfxn, repeats)
        # already handles the outer repeat count
        lines = []

        for i, fa_rep in enumerate(fa_rep_scales):
            # Minimization tolerance: tighter on final stage
            is_final = (i == num_stages - 1)
            min_tol = 0.0001 if is_final else 0.01

            lines.append(f"scale:fa_rep {fa_rep:.4f}")
            # Do NOT scale coordinate_constraint - keep it at full weight
            lines.append("repack")
            lines.append(f"min {min_tol}")

        lines.append("accept_to_best")

        return lines

    def run_fastrelax_round(self, pose, sfxn, movemap, task_factory=None, repeats: int = 3, ramp_stages: int = 5) -> float:
        """
        Run one round of Cartesian FastRelax.

        Args:
            pose: Pose to relax
            sfxn: ScoreFunction to use
            movemap: MoveMap defining mobile DOFs
            task_factory: Optional TaskFactory to restrict repacking
            repeats: Number of script repeats (M)
            ramp_stages: Number of ramping stages per repeat (N)
                         Total internal rounds = M × N

        Returns:
            Final score after this round
        """
        relax = FastRelax(sfxn, repeats)
        relax.set_movemap(movemap)
        relax.cartesian(True)
        relax.min_type("lbfgs_armijo_nonmonotone")

        # Enable bondangle and bondlength minimization for better geometry idealization
        if self.enable_bond_geometry_min:
            relax.minimize_bond_angles(True)
            relax.minimize_bond_lengths(True)
            LOGGER.debug("  Bond geometry minimization enabled (minimize_bond_angles=True, minimize_bond_lengths=True)")

        # Set custom relax script with specified number of ramping stages
        script_lines = self.generate_relax_script(ramp_stages)
        script_vector = pyr.rosetta.std.vector_std_string()
        for line in script_lines:
            script_vector.append(line)
        relax.set_script_from_lines(script_vector)

        # Apply TaskFactory to restrict repacking to mobile region
        if task_factory is not None:
            relax.set_task_factory(task_factory)

        relax.apply(pose)

        return sfxn(pose)

    def run_torsional_fastrelax(self, pose, sfxn, movemap, cycles: int = 1) -> float:
        """
        Run torsional FastRelax for backbone polishing.

        Returns:
            Final score
        """
        LOGGER.info(f"Running torsional FastRelax ({cycles} cycles)...")

        # Create torsional scorefunction (without cart_bonded)
        torsion_sfxn = ScoreFunctionFactory.create_score_function("ref2015")
        torsion_sfxn.set_weight(ScoreType.coordinate_constraint, self.coord_cst_weight)

        relax = FastRelax(torsion_sfxn, cycles)
        relax.set_movemap(movemap)
        relax.cartesian(False)
        relax.min_type("lbfgs_armijo_nonmonotone")

        score_before = torsion_sfxn(pose)
        relax.apply(pose)
        score_after = torsion_sfxn(pose)

        LOGGER.info(f"  Score: {score_before:.2f} -> {score_after:.2f}")

        return score_after

    def run_cartesian_minimize(self, pose, sfxn, movemap, tolerance: float = 0.0001) -> float:
        """
        Run final Cartesian minimization.

        Returns:
            Final score
        """
        LOGGER.info(f"Running final Cartesian minimization (tol={tolerance})...")

        min_mover = MinMover(movemap, sfxn, "lbfgs_armijo_nonmonotone", tolerance, True)
        min_mover.cartesian(True)

        score_before = sfxn(pose)
        min_mover.apply(pose)
        score_after = sfxn(pose)

        LOGGER.info(f"  Score: {score_before:.2f} -> {score_after:.2f}")

        return score_after

    def run_adaptive_protocol(self, pose, sfxn, movemap, task_factory=None,
                               mobile_residues: Optional[List[int]] = None) -> Dict:
        """
        Adaptive Cartesian FastRelax protocol.

        Each adaptive round runs FastRelax with M repeats × N ramping stages.
        Uses bond length and angle tolerances for convergence checking.
        In fast_mode, uses 1 repeat × 3 stages = 3 total (vs default 3 × 5 = 15).

        Supports automatic mobile region expansion when geometry is stuck.

        Args:
            pose: Pose to relax
            sfxn: ScoreFunction to use
            movemap: MoveMap defining mobile DOFs
            task_factory: Optional TaskFactory to restrict repacking
            mobile_residues: List of mobile residue indices (for expansion tracking)

        Returns:
            Dict with convergence info including round history
        """
        LOGGER.info("="*70)
        LOGGER.info("ADAPTIVE CARTESIAN FASTRELAX PROTOCOL")
        LOGGER.info("="*70)

        # Determine repeats and stages per round
        if self.fast_mode:
            repeats = 1
            ramp_stages = 3
            LOGGER.info(f"FAST MODE: Using {repeats} repeat × {ramp_stages} stages = {repeats * ramp_stages} total rounds")
        else:
            repeats = self.fastrelax_repeats
            ramp_stages = self.fastrelax_ramp_stages
            LOGGER.info(f"Using {repeats} repeats × {ramp_stages} stages = {repeats * ramp_stages} total rounds")

        LOGGER.info(f"Convergence tolerances: bond_length < {self.bond_length_tolerance}A, "
                   f"bond_angle < {self.bond_angle_tolerance}deg")
        LOGGER.info(f"Bond geometry minimization: {'ENABLED' if self.enable_bond_geometry_min else 'DISABLED'}")
        LOGGER.info(f"Cart_bonded scaling: factor={self.cart_bonded_scale_factor}, max={self.cart_bonded_max}")
        if self.auto_expand_mobile:
            LOGGER.info(f"Auto-expansion: ENABLED (radius={self.expansion_radius}A, max_expansions={self.max_expansions})")
        else:
            LOGGER.info("Auto-expansion: DISABLED")

        start_time = time.time()
        round_num = 0
        prev_max_bond_dev = float('inf')
        prev_max_angle_dev = float('inf')
        convergence_history = []
        expansion_count = 0
        prev_bond_metrics = None

        while True:
            round_num += 1
            elapsed = time.time() - start_time

            # Check time limit (leave 20% for final steps)
            if elapsed > self.max_runtime * 0.8:
                LOGGER.info(f"Approaching time limit ({elapsed:.0f}s), finishing protocol")
                break

            LOGGER.info(f"\n--- Round {round_num} ---")

            # Run FastRelax
            fr_start = time.time()
            score = self.run_fastrelax_round(pose, sfxn, movemap, task_factory, repeats=repeats, ramp_stages=ramp_stages)
            fr_time = time.time() - fr_start

            # Check bond length geometry (use unconstrained_only for convergence)
            bond_metrics = self.calculate_bond_metrics(pose)
            # Use unconstrained_only metrics for convergence checking
            # (constrained atoms are from theozyme and are ground truth)
            bond_unconstrained = bond_metrics["unconstrained_only"]
            bond_all = bond_metrics["all"]
            max_bond_dev = bond_unconstrained["max_deviation"]
            mean_bond_dev = bond_unconstrained["mean_deviation"]

            # Check bond angle geometry
            angle_metrics = self.calculate_angle_metrics(pose)
            angle_unconstrained = angle_metrics["unconstrained_only"]
            angle_all = angle_metrics["all"]
            max_angle_dev = angle_unconstrained["max_deviation"]
            mean_angle_dev = angle_unconstrained["mean_deviation"]

            LOGGER.info(f"  Time: {fr_time:.1f}s")
            LOGGER.info(f"  Score: {score:.2f}")
            LOGGER.info(f"  Bond length deviations (unconstrained): mean={mean_bond_dev:.4f}A, max={max_bond_dev:.4f}A")
            LOGGER.info(f"  Bond angle deviations (unconstrained): mean={mean_angle_dev:.2f}deg, max={max_angle_dev:.2f}deg")
            if bond_all["max_deviation"] != max_bond_dev or angle_all["max_deviation"] != max_angle_dev:
                LOGGER.info(f"  (All bonds: max={bond_all['max_deviation']:.4f}A, all angles: max={angle_all['max_deviation']:.2f}deg)")

            convergence_history.append({
                "round": round_num,
                "score": float(score),
                "max_bond_dev": float(max_bond_dev),
                "mean_bond_dev": float(mean_bond_dev),
                "max_angle_dev": float(max_angle_dev),
                "mean_angle_dev": float(mean_angle_dev),
                "time": float(fr_time)
            })

            # Check convergence: both bond length AND angle must be below tolerance
            bonds_converged = max_bond_dev < self.bond_length_tolerance
            angles_converged = max_angle_dev < self.bond_angle_tolerance

            if bonds_converged and angles_converged:
                LOGGER.info(f"Geometry converged (bonds < {self.bond_length_tolerance}A, "
                           f"angles < {self.bond_angle_tolerance}deg), stopping")
                break

            # Check if improving
            bond_improvement = prev_max_bond_dev - max_bond_dev
            angle_improvement = prev_max_angle_dev - max_angle_dev

            if round_num > 1 and bond_improvement < 0.001 and angle_improvement < 0.1:
                LOGGER.info("Convergence plateaued (bonds and angles not improving), stopping")
                break

            # Adaptive cart_bonded: increase if bonds still bad
            if max_bond_dev > self.bond_length_tolerance:
                current_weight = sfxn.get_weight(ScoreType.cart_bonded)
                new_weight = min(current_weight * self.cart_bonded_scale_factor, self.cart_bonded_max)
                if new_weight > current_weight:
                    sfxn.set_weight(ScoreType.cart_bonded, new_weight)
                    LOGGER.info(f"  Increased cart_bonded: {current_weight:.2f} -> {new_weight:.2f}")

            # Auto-expand mobile region if stuck
            if (self.auto_expand_mobile and
                mobile_residues is not None and
                expansion_count < self.max_expansions and
                prev_bond_metrics is not None and
                round_num > 2 and
                bond_improvement < 0.005 and
                max_bond_dev > self.bond_length_tolerance):

                stuck_residues = self.identify_stuck_residues(pose, prev_bond_metrics, bond_metrics)
                if stuck_residues:
                    mobile_residues = self.expand_mobile_region(pose, mobile_residues, stuck_residues)
                    movemap = self.build_movemap(pose, mobile_residues)
                    task_factory = self.build_task_factory(pose, mobile_residues)
                    expansion_count += 1
                    LOGGER.info(f"  Mobile region expansion {expansion_count}/{self.max_expansions}")

            prev_bond_metrics = bond_metrics
            prev_max_bond_dev = max_bond_dev
            prev_max_angle_dev = max_angle_dev

            # Safety: max rounds
            if round_num >= self.max_adaptive_rounds:
                LOGGER.info(f"Maximum rounds reached ({self.max_adaptive_rounds}), stopping")
                break

        # Phase 1.5: EXPERIMENTAL - Loop rebuild for severely distorted regions
        loop_rebuild_attempted = False
        if (self.enable_loop_rebuild and
            self.loop_rebuild_threshold is not None and
            prev_bond_metrics is not None):
            # Find residues exceeding the loop rebuild threshold
            severely_distorted = []
            for bond_info in prev_bond_metrics.get('worst_bonds', []):
                if bond_info['deviation'] > self.loop_rebuild_threshold:
                    residue_pdb = bond_info['residue']
                    chain = residue_pdb[0]
                    resno = int(residue_pdb[1:])
                    pose_idx = pose.pdb_info().pdb2pose(chain, resno)
                    if pose_idx > 0:
                        severely_distorted.append(pose_idx)

            if severely_distorted:
                LOGGER.warning(f"Found {len(severely_distorted)} residues exceeding loop rebuild threshold "
                              f"({self.loop_rebuild_threshold}A)")
                loop_rebuild_attempted = self.attempt_loop_rebuild(pose, severely_distorted)

        # Phase 2: Optional torsional FastRelax
        if not self.skip_torsional_relax:
            self.run_torsional_fastrelax(pose, sfxn, movemap, cycles=1)

        # Phase 3: Final MinMover
        if not self.skip_minimize:
            self.run_cartesian_minimize(pose, sfxn, movemap)

        total_time = time.time() - start_time
        LOGGER.info(f"\nProtocol complete: {round_num} rounds in {total_time:.1f}s")

        return {
            "num_rounds": round_num,
            "fastrelax_repeats": repeats,
            "fastrelax_ramp_stages": ramp_stages,
            "total_internal_rounds_per_cycle": repeats * ramp_stages,
            "fast_mode": self.fast_mode,
            "enable_bond_geometry_min": self.enable_bond_geometry_min,
            "total_time": float(total_time),
            "num_mobile_expansions": expansion_count,
            "final_mobile_region_size": len(mobile_residues) if mobile_residues else 0,
            "loop_rebuild_attempted": loop_rebuild_attempted,
            "history": convergence_history
        }

    def get_mobile_non_catalytic_residues(self, pose, mobile_residues: List[int]) -> List[Dict]:
        """
        Get list of mobile residues that are NOT catalytic (not in REMARK 666).

        Returns list of dicts with pdb_id, resname, pose_idx for each mobile non-catres.
        """
        # Get catres pose indices
        catres_indices = set(self.catres_pose_indices)
        ligand_indices = set(self.ligand_pose_indices)

        mobile_non_cat = []
        for pose_idx in mobile_residues:
            if pose_idx in catres_indices or pose_idx in ligand_indices:
                continue

            chain = pose.pdb_info().chain(pose_idx)
            resno = pose.pdb_info().number(pose_idx)
            resname = pose.residue(pose_idx).name3()

            mobile_non_cat.append({
                "pdb_id": f"{chain}{resno}",
                "resname": resname,
                "pose_idx": int(pose_idx)
            })

        return mobile_non_cat

    def calculate_all_metrics(self, pose_before, pose_after, mobile_residues: List[int]) -> Dict:
        """
        Calculate all validation metrics.
        """
        LOGGER.info("\nCalculating validation metrics...")

        metrics = {
            "metadata": {},
            "scores": {},
            "bond_length_geometry": {},
            "bond_angle_geometry": {},
            "geometry_offenders": {},
            "catres_geometry_status": {},
            "rmsd": {},
            "quality": {},
            "per_catres": {},
            "mobile_non_catalytic_residues": [],
            "convergence": {}
        }

        # Metadata (all paths are absolute)
        metrics["metadata"]["input_pdb"] = os.path.abspath(self.input_pdb_path) if self.input_pdb_path else None
        metrics["metadata"]["step01_json"] = os.path.abspath(self.step01_json_path) if self.step01_json_path else None
        metrics["metadata"]["num_residues"] = int(pose_after.size())
        metrics["metadata"]["num_catres"] = len(self.residue_constraints)
        metrics["metadata"]["coord_cst_weight"] = self.coord_cst_weight
        metrics["metadata"]["cart_bonded_weight"] = self.cart_bonded_weight

        # Scores
        score_before = self.sfxn(pose_before)
        score_after = self.sfxn(pose_after)
        metrics["scores"]["total_before"] = float(score_before)
        metrics["scores"]["total_after"] = float(score_after)
        metrics["scores"]["delta"] = float(score_after - score_before)

        # Get score breakdown
        metrics["scores"]["cart_bonded"] = float(
            pose_after.energies().total_energies()[ScoreType.cart_bonded]
        )
        metrics["scores"]["fa_rep"] = float(
            pose_after.energies().total_energies()[ScoreType.fa_rep]
        )
        metrics["scores"]["coordinate_constraint"] = float(
            pose_after.energies().total_energies()[ScoreType.coordinate_constraint]
        )

        # Bond length geometry
        bond_metrics_before = self.calculate_bond_metrics(pose_before)
        bond_metrics_after = self.calculate_bond_metrics(pose_after)
        metrics["bond_length_geometry"]["before"] = bond_metrics_before
        metrics["bond_length_geometry"]["after"] = bond_metrics_after

        # Bond angle geometry
        angle_metrics_before = self.calculate_angle_metrics(pose_before)
        angle_metrics_after = self.calculate_angle_metrics(pose_after)
        metrics["bond_angle_geometry"]["before"] = angle_metrics_before
        metrics["bond_angle_geometry"]["after"] = angle_metrics_after

        # Comprehensive geometry offender metrics
        offender_metrics = self.calculate_offender_metrics(pose_after)
        metrics["geometry_offenders"] = offender_metrics

        # Catres-specific geometry status (for pass/fail checking)
        catres_geometry_status = self.calculate_catres_geometry_status(pose_after)
        metrics["catres_geometry_status"] = catres_geometry_status

        # RMSD metrics
        metrics["rmsd"]["ligand"] = self.calculate_ligand_rmsd(pose_before, pose_after)
        metrics["rmsd"]["constrained_atoms"] = self.calculate_constrained_atom_rmsd(pose_before, pose_after)
        metrics["rmsd"]["global_ca"] = float(CA_rmsd(pose_before, pose_after))

        # Quality metrics
        chain_breaks_before = self.detect_chain_breaks(pose_before)
        chain_breaks_after = self.detect_chain_breaks(pose_after)
        metrics["quality"]["chain_breaks_before"] = len(chain_breaks_before)
        metrics["quality"]["chain_breaks_after"] = len(chain_breaks_after)
        metrics["quality"]["clashes_before"] = self.count_clashes(pose_before)
        metrics["quality"]["clashes_after"] = self.count_clashes(pose_after)

        # Per-catres metrics
        metrics["per_catres"] = self.calculate_per_catres_metrics(pose_after)

        # Mobile non-catalytic residues
        metrics["mobile_non_catalytic_residues"] = self.get_mobile_non_catalytic_residues(
            pose_after, mobile_residues
        )
        metrics["metadata"]["num_mobile_non_catalytic"] = len(metrics["mobile_non_catalytic_residues"])

        return metrics

    def strip_energy_table_from_pdb(self, pdb_path: str):
        """Remove per-residue energy table from PDB to save space."""
        with open(pdb_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        in_energy_table = False

        for line in lines:
            if line.startswith('# All scores below are weighted scores'):
                in_energy_table = True
                continue
            elif line.startswith('#END_POSE_ENERGIES_TABLE'):
                in_energy_table = False
                continue

            if not in_energy_table:
                new_lines.append(line)

        with open(pdb_path, 'w') as f:
            f.writelines(new_lines)

    def round_metrics(self, obj, decimals: int = 4):
        """Recursively round all float values."""
        if isinstance(obj, dict):
            return {k: self.round_metrics(v, decimals) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.round_metrics(item, decimals) for item in obj]
        elif isinstance(obj, float):
            return round(obj, decimals)
        else:
            return obj

    def run(
        self,
        output_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Main execution pipeline.

        Returns:
            Tuple of (output_pdb_path, output_metrics_json_path)
        """
        self.start_time = time.time()

        LOGGER.info("="*70)
        LOGGER.info("STEP02: CONSTRAINED CARTESIAN RELAXATION")
        LOGGER.info("="*70)

        # Load JSON
        self.load_step01_json()

        # Identify ligand
        self.ligand_info = self.identify_ligand_from_pdb(self.input_pdb_path)
        LOGGER.info(f"Ligand identified: {self.ligand_info[0]}{self.ligand_info[2]} ({self.ligand_info[1]})")

        # Validate REMARK 666
        self.validate_remark_666_consistency(self.input_pdb_path)

        # Build atoms to constrain
        self.build_atoms_to_constrain()

        # Load pose
        LOGGER.info(f"\nLoading pose: {self.input_pdb_path}")
        self.pose = pyr.pose_from_file(self.input_pdb_path)
        self.original_pose = self.pose.clone()
        LOGGER.info(f"  Pose loaded: {self.pose.size()} residues")

        # Map residues to pose indices
        for block_idx, info in self.residue_constraints.items():
            pose_idx = self.pose.pdb_info().pdb2pose(info["chain"], info["resno"])
            if pose_idx > 0:
                self.catres_pose_indices.append(pose_idx)

        lig_pose_idx = self.pose.pdb_info().pdb2pose(self.ligand_info[0], self.ligand_info[2])
        if lig_pose_idx > 0:
            self.ligand_pose_indices.append(lig_pose_idx)

        # Setup scorefunction
        self.setup_scorefunction()

        # Add coordinate constraints
        self.add_coordinate_constraints(self.pose)

        # Build constrained atoms set for filtering bond/angle metrics
        self.build_constrained_atoms_set(self.pose)

        # Define mobile region
        mobile_residues = self.define_mobile_region(self.pose)

        # Build MoveMap
        movemap = self.build_movemap(self.pose, mobile_residues)

        # Build TaskFactory (restricts repacking to mobile region)
        task_factory = self.build_task_factory(self.pose, mobile_residues)

        # Check initial state
        LOGGER.info("\nInitial state:")
        initial_breaks = self.detect_chain_breaks(self.pose)
        initial_clashes = self.count_clashes(self.pose)
        initial_bond_metrics = self.calculate_bond_metrics(self.pose)
        # Use unconstrained_only for primary reporting
        bond_unconstrained = initial_bond_metrics["unconstrained_only"]
        bond_all = initial_bond_metrics["all"]
        LOGGER.info(f"  Chain breaks: {len(initial_breaks)}")
        LOGGER.info(f"  Clashing residues: {initial_clashes}")
        LOGGER.info(f"  Bond deviations (unconstrained): mean={bond_unconstrained['mean_deviation']:.4f}A, "
                   f"max={bond_unconstrained['max_deviation']:.4f}A")
        if bond_all['max_deviation'] != bond_unconstrained['max_deviation']:
            LOGGER.info(f"  Bond deviations (all): mean={bond_all['mean_deviation']:.4f}A, "
                       f"max={bond_all['max_deviation']:.4f}A")

        # Run adaptive protocol (pass mobile_residues for potential expansion)
        convergence_info = self.run_adaptive_protocol(
            self.pose, self.sfxn, movemap, task_factory, mobile_residues=mobile_residues
        )

        # Calculate all metrics
        metrics = self.calculate_all_metrics(self.original_pose, self.pose, mobile_residues)
        metrics["convergence"] = convergence_info

        # Determine output paths
        input_basename = os.path.splitext(os.path.basename(self.input_pdb_path))[0]

        if output_path:
            out_pdb = output_path
        elif output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_pdb = os.path.join(output_dir, f"{input_basename}_relaxed.pdb")
        else:
            out_pdb = f"{input_basename}_relaxed.pdb"

        out_json = out_pdb.replace(".pdb", "_metrics.json")

        # Save outputs
        LOGGER.info(f"\nSaving outputs...")
        self.pose.dump_pdb(out_pdb)
        self.strip_energy_table_from_pdb(out_pdb)
        LOGGER.info(f"  PDB: {out_pdb}")

        # Add additional metadata before saving JSON (all paths absolute)
        metrics["metadata"]["output_pdb"] = os.path.abspath(out_pdb)
        metrics["metadata"]["ref_pdb"] = os.path.abspath(self.ref_pdb_path) if self.ref_pdb_path else None
        metrics["metadata"]["params_files"] = [os.path.abspath(p) for p in self.params_files] if self.params_files else []
        metrics["metadata"]["catres_subset"] = self.catres_subset_override
        metrics["metadata"]["scorefunction"] = self.scorefunction
        metrics["metadata"]["residue_constraints"] = self.build_residue_constraints_output(self.pose)

        metrics = self.round_metrics(metrics)
        with open(out_json, 'w') as f:
            json.dump(metrics, f, indent=2)
        LOGGER.info(f"  Metrics: {out_json}")

        # Print summary
        total_time = time.time() - self.start_time
        LOGGER.info("\n" + "="*70)
        LOGGER.info("SUMMARY")
        LOGGER.info("="*70)
        LOGGER.info(f"  Total runtime: {total_time:.1f}s")
        LOGGER.info(f"  FastRelax rounds: {convergence_info['num_rounds']}")

        # Use unconstrained_only metrics for primary reporting
        bond_before = metrics['bond_length_geometry']['before']['unconstrained_only']
        bond_after = metrics['bond_length_geometry']['after']['unconstrained_only']
        angle_before = metrics['bond_angle_geometry']['before']['unconstrained_only']
        angle_after = metrics['bond_angle_geometry']['after']['unconstrained_only']

        LOGGER.info(f"  Bond lengths (unconstrained): max {bond_before['max_deviation']:.4f}A -> "
                   f"{bond_after['max_deviation']:.4f}A")
        LOGGER.info(f"  Bond angles (unconstrained): max {angle_before['max_deviation']:.2f}deg -> "
                   f"{angle_after['max_deviation']:.2f}deg")

        # Show all metrics if different
        bond_all_after = metrics['bond_length_geometry']['after']['all']
        angle_all_after = metrics['bond_angle_geometry']['after']['all']
        if bond_all_after['max_deviation'] != bond_after['max_deviation']:
            LOGGER.info(f"  Bond lengths (all incl. constrained): max {bond_all_after['max_deviation']:.4f}A")
        if angle_all_after['max_deviation'] != angle_after['max_deviation']:
            LOGGER.info(f"  Bond angles (all incl. constrained): max {angle_all_after['max_deviation']:.2f}deg")

        LOGGER.info(f"  Ligand RMSD: {metrics['rmsd']['ligand']:.4f}A")
        LOGGER.info(f"  Constrained atom RMSD: {metrics['rmsd']['constrained_atoms']['aggregate_rmsd']:.4f}A")
        LOGGER.info(f"  Global CA RMSD: {metrics['rmsd']['global_ca']:.4f}A")
        LOGGER.info(f"  Chain breaks: {metrics['quality']['chain_breaks_before']} -> "
                   f"{metrics['quality']['chain_breaks_after']}")
        LOGGER.info(f"  Clashes: {metrics['quality']['clashes_before']} -> "
                   f"{metrics['quality']['clashes_after']}")
        LOGGER.info(f"  Mobile non-catalytic residues: {len(metrics['mobile_non_catalytic_residues'])}")
        LOGGER.info("="*70)

        return out_pdb, out_json


def init_pyrosetta(params_files: List[str], nproc: int = 1, scorefunction: str = "ref2015_cart") -> None:
    """
    Initialize PyRosetta with optimal settings.

    Note: Most apptainer/docker PyRosetta builds are single-threaded.
    Set nproc=1 for compatibility.

    Args:
        params_files: List of ligand .params files
        nproc: Number of threads (usually 1 for apptainer)
        scorefunction: Scorefunction name (affects which correction flags to use)
    """
    extra_res_fa = ""
    if params_files:
        extra_res_fa = "-extra_res_fa " + " ".join(params_files)

    # Use 1 thread by default for apptainer compatibility
    # Multi-threading requires special PyRosetta builds with extras=cxx11thread
    if nproc > 1:
        LOGGER.warning("Multi-threading requested but may not be available in this PyRosetta build")

    init_flags = [
        extra_res_fa,
        "-run:preserve_header",
        "-mute all",
        "-unmute core.scoring.ScoreFunction",
    ]

    # Add correction flags based on scorefunction
    if "beta_jan25" in scorefunction:
        init_flags.append("-corrections::beta_jan25 true")
        LOGGER.info("  Enabling beta_jan25 corrections")
    elif "beta_nov16" in scorefunction:
        init_flags.append("-corrections::beta_nov16 true")
        LOGGER.info("  Enabling beta_nov16 corrections")

    # Only add threading flags if nproc > 1
    if nproc > 1:
        init_flags.extend([
            f"-multithreading:total_threads {nproc}",
            f"-multithreading:interaction_graph_threads {nproc}",
        ])

    init_cmd = " ".join([f for f in init_flags if f])

    LOGGER.info("Initializing PyRosetta...")
    LOGGER.info(f"  Using {nproc} thread(s)")
    LOGGER.info(f"  Scorefunction: {scorefunction}")
    pyr.init(init_cmd)
    LOGGER.info("PyRosetta initialized")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Constrained Cartesian relaxation for step01 output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard usage with step01 JSON
  %(prog)s --step01_json step01_output.json --params ligand.params --output relaxed.pdb

  # Override catres_subset (use only blocks 1,2,5)
  %(prog)s --step01_json step01_output.json --params ligand.params --catres_subset 1,2,5

  # Force backbone constraints even for bb-only-hbond residues
  %(prog)s --step01_json step01_output.json --params ligand.params --ignore_bb_only_hbond_filter

  # Test mode
  %(prog)s --test
        """
    )

    # Required arguments (unless --test)
    parser.add_argument("--step01_json", type=str,
                        help="Path to step01 output JSON (*_recommended_atom_cst.json)")
    parser.add_argument("--params", type=str, nargs="+",
                        help="Ligand parameter file(s)")

    # Optional override arguments
    parser.add_argument("--catres_subset", type=str, default=None,
                        help="Override catres subset (comma-separated block indices)")
    parser.add_argument("--ref_pdb", type=str, default=None,
                        help="Override ref_pdb from JSON (for validation)")
    parser.add_argument("--input_prepped_pdb", type=str, default=None,
                        help="Override input PDB (use different aligned PDB)")
    parser.add_argument("--ignore_bb_only_hbond_filter", action="store_true",
                        help="Constrain backbone even when backbone_important_only_for_BB_BB_hbond=True")

    # Output arguments
    parser.add_argument("--output", type=str, default=None,
                        help="Output PDB path (default: <input>_relaxed.pdb)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory")

    # Protocol parameters
    parser.add_argument("--coord_cst_weight", type=float, default=750.0,
                        help="Weight for coordinate constraints (default: 750)")
    parser.add_argument("--coord_cst_stdev", type=float, default=0.01,
                        help="HarmonicFunc stdev in Angstroms (default: 0.01)")
    parser.add_argument("--cart_bonded_weight", type=float, default=2.0,
                        help="Weight for cart_bonded term (default: 2.0)")
    parser.add_argument("--mobile_radius", type=float, default=10.0,
                        help="Radius for mobile region (default: 10.0 A)")
    parser.add_argument("--fastrelax_repeats", type=int, default=3,
                        help="Number of script repeats per FastRelax call (M, default: 3)")
    parser.add_argument("--fastrelax_ramp_stages", type=int, default=5,
                        help="Number of ramping stages per repeat (N, default: 5). "
                             "Total rounds = M × N (default: 3 × 5 = 15)")
    parser.add_argument("--bond_length_tolerance", type=float, default=0.05,
                        help="Stop when max bond length deviation < this (default: 0.05 A)")
    parser.add_argument("--bond_angle_tolerance", type=float, default=10.0,
                        help="Stop when max bond angle deviation < this (default: 10.0 degrees)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: 1 repeat × 3 stages = 3 total rounds per adaptive cycle "
                             "(vs default 3 × 5 = 15)")
    parser.add_argument("--max_runtime", type=int, default=3600,
                        help="Maximum runtime in seconds (default: 3600)")
    parser.add_argument("--sequence_neighbor_buffer", type=int, default=5,
                        help="Include residues +/- N from catres (default: 5)")
    parser.add_argument("--max_adaptive_rounds", type=int, default=10,
                        help="Maximum adaptive rounds before stopping (default: 10)")

    # Cart_bonded scaling
    parser.add_argument("--cart_bonded_scale_factor", type=float, default=1.5,
                        help="Factor to multiply cart_bonded by when not converging (default: 1.5)")
    parser.add_argument("--cart_bonded_max", type=float, default=3.0,
                        help="Maximum cart_bonded weight cap (default: 3.0)")

    # Bond geometry minimization
    parser.add_argument("--enable_bond_geometry_min", action="store_true", default=True,
                        help="Enable bondangle+bondlength minimization (default: True)")
    parser.add_argument("--disable_bond_geometry_min", action="store_true",
                        help="Disable bondangle+bondlength minimization")

    # Repacking scope
    parser.add_argument("--global_repack", action="store_true",
                        help="Enable global repacking (all protein residues)")
    parser.add_argument("--repack_shell", type=float, default=None,
                        help="Repack within N Angstroms of mobile region")

    # Mobile region expansion
    parser.add_argument("--auto_expand_mobile", action="store_true", default=True,
                        help="Auto-expand mobile region if stuck (default: True)")
    parser.add_argument("--no_auto_expand_mobile", action="store_true",
                        help="Disable automatic mobile region expansion")
    parser.add_argument("--expansion_radius", type=float, default=5.0,
                        help="Radius for mobile region expansion (default: 5.0 A)")
    parser.add_argument("--max_expansions", type=int, default=3,
                        help="Maximum number of mobile region expansions (default: 3)")

    # Catres-specific convergence
    parser.add_argument("--catres_bond_tolerance", type=float, default=0.05,
                        help="Catres-specific bond tolerance (default: 0.05 A)")
    parser.add_argument("--catres_angle_tolerance", type=float, default=10.0,
                        help="Catres-specific angle tolerance (default: 10.0 deg)")
    parser.add_argument("--require_catres_converged", action="store_true", default=True,
                        help="Warn if catres geometry bad (default: True)")
    parser.add_argument("--no_require_catres_converged", action="store_true",
                        help="Disable catres-specific convergence check")

    # Offender thresholds
    parser.add_argument("--severe_bond_threshold", type=float, default=0.2,
                        help="Bond deviation above which is 'severe' (default: 0.2 A)")
    parser.add_argument("--moderate_bond_threshold", type=float, default=0.1,
                        help="Bond deviation for 'moderate' category (default: 0.1 A)")
    parser.add_argument("--severe_angle_threshold", type=float, default=15.0,
                        help="Angle deviation above which is 'severe' (default: 15.0 deg)")
    parser.add_argument("--moderate_angle_threshold", type=float, default=10.0,
                        help="Angle deviation for 'moderate' category (default: 10.0 deg)")

    # Loop modeling (experimental)
    parser.add_argument("--enable_loop_rebuild", action="store_true",
                        help="Enable experimental loop rebuilding capability")
    parser.add_argument("--loop_rebuild_threshold", type=float, default=None,
                        help="Bond deviation triggering loop rebuild (A)")

    # Presets
    parser.add_argument("--preset", type=str, choices=["fast", "balanced", "thorough", "aggressive"],
                        help="Preset parameter combinations")

    # Scorefunction selection
    parser.add_argument("--scorefunction", type=str, default="ref2015_cart",
                        choices=["ref2015_cart", "beta_jan25_cart", "beta_nov16_cart"],
                        help="Scorefunction to use (default: ref2015_cart). "
                             "beta_jan25_cart is the latest and may give better results.")

    # Execution flags
    parser.add_argument("--skip_torsional_relax", action="store_true",
                        help="Skip torsional FastRelax at end")
    parser.add_argument("--skip_minimize", action="store_true",
                        help="Skip final MinMover")

    # Output options
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debug output")

    # Test mode
    parser.add_argument("--test", action="store_true",
                        help="Run with hardcoded test data")

    return parser.parse_args()


def apply_preset(args):
    """Apply preset parameter combinations."""
    PRESETS = {
        "fast": {
            "fastrelax_repeats": 1,
            "fastrelax_ramp_stages": 3,
            "max_adaptive_rounds": 3,
            "enable_bond_geometry_min": False,
            "auto_expand_mobile": False,
        },
        "balanced": {
            "fastrelax_repeats": 3,
            "fastrelax_ramp_stages": 5,
            "max_adaptive_rounds": 10,
            "enable_bond_geometry_min": True,
        },
        "thorough": {
            "fastrelax_repeats": 5,
            "fastrelax_ramp_stages": 5,
            "max_adaptive_rounds": 15,
            "bond_length_tolerance": 0.03,
            "bond_angle_tolerance": 5.0,
            "enable_bond_geometry_min": True,
        },
        "aggressive": {
            "fastrelax_repeats": 3,
            "fastrelax_ramp_stages": 5,
            "cart_bonded_weight": 1.2,
            "cart_bonded_max": 4.0,
            "cart_bonded_scale_factor": 2.0,
            "auto_expand_mobile": True,
            "max_expansions": 5,
            "enable_bond_geometry_min": True,
        },
    }

    if args.preset and args.preset in PRESETS:
        preset = PRESETS[args.preset]
        LOGGER.info(f"Applying preset: {args.preset}")
        for key, value in preset.items():
            # Only apply if user hasn't explicitly set a different value
            if hasattr(args, key):
                setattr(args, key, value)
                LOGGER.info(f"  {key} = {value}")

    return args


def main():
    """Main entry point."""
    args = parse_args()

    # Apply preset if specified
    args = apply_preset(args)

    # Handle --no_* flags
    if args.disable_bond_geometry_min:
        args.enable_bond_geometry_min = False
    if args.no_auto_expand_mobile:
        args.auto_expand_mobile = False
    if args.no_require_catres_converged:
        args.require_catres_converged = False

    if args.test:
        LOGGER.info("Running in TEST mode with hardcoded paths")
        step01_json = TEST_STEP01_JSON
        params = TEST_PARAMS
        output_dir = TEST_OUTPUT_DIR
    else:
        if not args.step01_json or not args.params:
            LOGGER.error("Must provide --step01_json and --params (or use --test)")
            sys.exit(1)
        step01_json = args.step01_json
        params = args.params
        output_dir = args.output_dir

    # Initialize PyRosetta (single-threaded for apptainer compatibility)
    init_pyrosetta(params, nproc=1, scorefunction=args.scorefunction)

    # Create relaxer
    relaxer = ConstrainedCartesianRelax(
        step01_json_path=step01_json,
        params_files=params,
        catres_subset=args.catres_subset,
        ref_pdb_override=args.ref_pdb,
        input_pdb_override=args.input_prepped_pdb,
        ignore_bb_only_hbond_filter=args.ignore_bb_only_hbond_filter,
        coord_cst_weight=args.coord_cst_weight,
        coord_cst_stdev=args.coord_cst_stdev,
        cart_bonded_weight=args.cart_bonded_weight,
        mobile_radius=args.mobile_radius,
        fastrelax_repeats=args.fastrelax_repeats,
        fastrelax_ramp_stages=args.fastrelax_ramp_stages,
        max_runtime=args.max_runtime,
        sequence_neighbor_buffer=args.sequence_neighbor_buffer,
        skip_torsional_relax=args.skip_torsional_relax,
        skip_minimize=args.skip_minimize,
        bond_length_tolerance=args.bond_length_tolerance,
        bond_angle_tolerance=args.bond_angle_tolerance,
        fast_mode=args.fast,
        debug=args.debug,
        # New parameters
        cart_bonded_scale_factor=args.cart_bonded_scale_factor,
        cart_bonded_max=args.cart_bonded_max,
        enable_bond_geometry_min=args.enable_bond_geometry_min,
        max_adaptive_rounds=args.max_adaptive_rounds,
        catres_bond_tolerance=args.catres_bond_tolerance,
        catres_angle_tolerance=args.catres_angle_tolerance,
        require_catres_converged=args.require_catres_converged,
        global_repack=args.global_repack,
        repack_shell=args.repack_shell,
        auto_expand_mobile=args.auto_expand_mobile,
        expansion_radius=args.expansion_radius,
        max_expansions=args.max_expansions,
        severe_bond_threshold=args.severe_bond_threshold,
        moderate_bond_threshold=args.moderate_bond_threshold,
        severe_angle_threshold=args.severe_angle_threshold,
        moderate_angle_threshold=args.moderate_angle_threshold,
        enable_loop_rebuild=args.enable_loop_rebuild,
        loop_rebuild_threshold=args.loop_rebuild_threshold,
        scorefunction=args.scorefunction,
    )

    # Run
    out_pdb, out_json = relaxer.run(
        output_path=args.output,
        output_dir=output_dir,
    )

    LOGGER.info("\n" + "="*70)
    LOGGER.info("CONSTRAINED CARTESIAN RELAXATION COMPLETE!")
    LOGGER.info(f"Output PDB: {out_pdb}")
    LOGGER.info(f"Metrics JSON: {out_json}")
    LOGGER.info("="*70)


if __name__ == "__main__":
    main()
