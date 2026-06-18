"""Residue classification for FastMPNN design.

This module provides:
- ResidueInfo dataclass for comprehensive residue classification
- DesignSphere enum for distance-based layer classification
- ResidueClassifier for analyzing and classifying all residues
"""
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Add module_utils to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module_utils.pdb_utils import parse_remark_666, read_pdb_atoms, get_residue_atoms, atoms_to_coords
from module_utils.constants import (
    DEFAULT_LAYER_CUTS, PROTECTED_RESIDUES, STANDARD_AA_3, AA_3_TO_1,
)

LOGGER = logging.getLogger(__name__)


class DesignSphere(Enum):
    """Sphere classification based on distance from ligand."""
    PRIMARY = "primary"                    # 0 - cuts[0] Angstroms (design)
    SECONDARY = "secondary"                # cuts[0] - cuts[1] Angstroms (design with CB check)
    REPACK_PRIMARY = "repack_primary"      # cuts[1] - cuts[2] Angstroms (repack only)
    REPACK_SECONDARY = "repack_secondary"  # cuts[2] - cuts[3] Angstroms (repack only)
    DISTANT = "distant"                    # > cuts[3] Angstroms (fixed)


@dataclass
class ResidueInfo:
    """Comprehensive residue classification for design decisions.

    Populated from REMARK 666, step02 JSON, and spatial analysis.
    """
    # Identity
    chain: str
    resno: int
    resname: str
    pose_index: int = 0  # Rosetta 1-indexed pose position

    # Catalytic/Motif classification (from REMARK 666)
    is_catalytic_residue: bool = False      # In catres_subset (constrained)
    is_conserved_motif: bool = False        # In REMARK 666 but NOT catres_subset
    block_index: Optional[int] = None       # REMARK 666 block index

    # Constraint information (from step01/step02 JSON)
    constrain_atoms: List[str] = field(default_factory=list)
    backbone_important: bool = False
    sidechain_important: bool = False

    # Sphere classification (computed from ligand distance)
    sphere: DesignSphere = DesignSphere.DISTANT
    ca_distance_to_ligand: float = float('inf')
    cb_distance_to_ligand: float = float('inf')
    cb_points_toward_ligand: bool = False  # CB dist < CA dist

    # Design control
    redesign_probability: float = 1.0  # 0.0 = never redesign, 1.0 = always redesign
    is_protected: bool = False  # GLY/PRO or user-specified protection

    # Favorable interaction tracking (for conservation)
    makes_hbond_to_catres: bool = False
    makes_pi_stack_to_catres: bool = False
    interaction_score: float = 0.0  # For biasing MPNN

    @property
    def is_fixed(self) -> bool:
        """Residue should not be redesigned by MPNN."""
        return self.is_catalytic_residue or self.is_conserved_motif

    @property
    def is_designable(self) -> bool:
        """Residue can be redesigned by MPNN."""
        if self.is_fixed:
            return False
        if self.is_protected:
            return False
        return self.sphere in (DesignSphere.PRIMARY, DesignSphere.SECONDARY)

    @property
    def is_repackable(self) -> bool:
        """Residue can be repacked (but not designed)."""
        if self.is_fixed:
            return False
        return self.sphere in (DesignSphere.PRIMARY, DesignSphere.SECONDARY,
                               DesignSphere.REPACK_PRIMARY, DesignSphere.REPACK_SECONDARY)

    def to_mpnn_residue_id(self) -> str:
        """Format as MPNN-style residue identifier (e.g., 'A45')."""
        return f"{self.chain}{self.resno}"

    @classmethod
    def from_mpnn_residue_id(cls, res_id: str) -> Tuple[str, int]:
        """Parse MPNN-style residue ID to (chain, resno)."""
        chain = res_id[0]
        resno = int(res_id[1:])
        return chain, resno


@dataclass
class LigandInfo:
    """Ligand information extracted from REMARK 666."""
    chain: str
    resname: str
    resno: int
    heavy_atom_coords: List[np.ndarray] = field(default_factory=list)


class ResidueClassifier:
    """Classify all residues based on REMARK 666, step02 JSON, and spatial analysis.

    Uses distance-based layer classification with CB orientation check,
    following the logic from modern_FastMPNNdesign/rosetta_utils.py:get_packer_layers()
    """

    def __init__(
        self,
        step02_pdb_path: str,
        step02_json_path: str,
        catres_subset: Optional[str] = None,
        layer_cuts: Optional[List[float]] = None,
        design_gly_pro: bool = False,
    ):
        """Initialize the classifier.

        Args:
            step02_pdb_path: Path to step02 relaxed PDB file
            step02_json_path: Path to step02 metrics JSON file
            catres_subset: Optional comma-separated block indices to use
            layer_cuts: Distance cutoffs for layers [primary, secondary, repack1, repack2]
            design_gly_pro: If True, allow GLY/PRO to be designed
        """
        self.step02_pdb_path = step02_pdb_path
        self.step02_json_path = step02_json_path
        self.catres_subset_str = catres_subset
        self.layer_cuts = layer_cuts or DEFAULT_LAYER_CUTS
        self.design_gly_pro = design_gly_pro

        # Populated during classification
        self.residues: Dict[Tuple[str, int], ResidueInfo] = {}  # (chain, resno) -> ResidueInfo
        self.ligand: Optional[LigandInfo] = None
        self.catres_subset_blocks: Set[int] = set()
        self.conserved_motif_blocks: Set[int] = set()
        self.all_remark666_blocks: Set[int] = set()

        # Data storage
        self._pdb_lines: List[str] = []
        self._atoms: List[Dict] = []
        self._step02_data: Dict = {}

    def classify(self) -> Dict[Tuple[str, int], ResidueInfo]:
        """Main classification method.

        1. Parse REMARK 666 to identify ligand and motif residues
        2. Load step02 JSON for constraint information
        3. Compute distances to ligand for sphere assignment
        4. Set redesign probabilities based on classification

        Returns:
            Dict mapping (chain, resno) -> ResidueInfo
        """
        LOGGER.info(f"Classifying residues from {self.step02_pdb_path}")

        # Load PDB
        self._pdb_lines, self._atoms = read_pdb_atoms(self.step02_pdb_path)

        # Load step02 JSON
        with open(self.step02_json_path, 'r') as f:
            self._step02_data = json.load(f)

        # Parse REMARK 666 and identify ligand/catres
        self._parse_remark666()

        # Build residue info for all protein residues
        self._build_residue_info()

        # Classify spheres based on ligand distance
        self._classify_spheres()

        # Apply constraint info from step02 JSON
        self._apply_constraint_info()

        # Mark protected residues
        self._mark_protected_residues()

        # Log summary
        self._log_summary()

        return self.residues

    def _parse_remark666(self) -> None:
        """Parse REMARK 666 lines to identify ligand and catalytic residues."""
        remark_entries = parse_remark_666(self._pdb_lines)

        if not remark_entries:
            raise ValueError(f"No REMARK 666 entries found in {self.step02_pdb_path}")

        # Identify ligand (template that's not a standard AA)
        ligand_info = None
        for entry in remark_entries:
            if "template_resname" in entry:
                if entry["template_resname"] not in STANDARD_AA_3:
                    ligand_info = entry
                    break

        if ligand_info is None:
            # Fall back to first template entry
            for entry in remark_entries:
                if "template_chain" in entry:
                    ligand_info = entry
                    break

        if ligand_info is None:
            raise ValueError("Could not identify ligand from REMARK 666")

        # Create LigandInfo
        self.ligand = LigandInfo(
            chain=ligand_info["template_chain"],
            resname=ligand_info["template_resname"],
            resno=ligand_info["template_resno"],
        )

        # Get ligand atom coordinates
        lig_atoms = [a for a in self._atoms
                    if a["chain"] == self.ligand.chain and a["resno"] == self.ligand.resno]
        for atom in lig_atoms:
            elem = atom.get("element", atom["atom_name"][0])
            if elem.upper() not in {"H", ""}:  # Heavy atoms only
                self.ligand.heavy_atom_coords.append(
                    np.array([atom["x"], atom["y"], atom["z"]])
                )

        LOGGER.info(f"Ligand: {self.ligand.chain} {self.ligand.resname} {self.ligand.resno} "
                   f"({len(self.ligand.heavy_atom_coords)} heavy atoms)")

        # Collect all block indices
        self.all_remark666_blocks = {entry["block_index"] for entry in remark_entries}
        max_block = max(self.all_remark666_blocks)

        # Parse catres_subset
        if self.catres_subset_str:
            subset_indices = [int(x.strip()) for x in self.catres_subset_str.split(",") if x.strip()]
            self.catres_subset_blocks = set(subset_indices)
        else:
            self.catres_subset_blocks = self.all_remark666_blocks.copy()

        self.conserved_motif_blocks = self.all_remark666_blocks - self.catres_subset_blocks

        LOGGER.info(f"Catres subset blocks: {sorted(self.catres_subset_blocks)}")
        LOGGER.info(f"Conserved motif blocks: {sorted(self.conserved_motif_blocks)}")

        # Store block -> residue mapping
        self._block_to_residue: Dict[int, Tuple[str, int, str]] = {}
        for entry in remark_entries:
            self._block_to_residue[entry["block_index"]] = (
                entry["motif_chain"],
                entry["motif_resno"],
                entry["motif_resname"],
            )

    def _build_residue_info(self) -> None:
        """Build ResidueInfo for all protein residues."""
        # Group atoms by residue
        residue_atoms: Dict[Tuple[str, int], List[Dict]] = {}
        for atom in self._atoms:
            if atom["record_type"] != "ATOM":
                continue
            key = (atom["chain"], atom["resno"])
            if key not in residue_atoms:
                residue_atoms[key] = []
            residue_atoms[key].append(atom)

        # Create ResidueInfo for each residue
        pose_idx = 0
        for (chain, resno), atoms in sorted(residue_atoms.items()):
            pose_idx += 1
            resname = atoms[0]["resname"]

            # Check if this is a motif residue
            block_idx = None
            is_catres = False
            is_conserved = False

            for bidx, (bc, br, bn) in self._block_to_residue.items():
                if bc == chain and br == resno:
                    block_idx = bidx
                    if bidx in self.catres_subset_blocks:
                        is_catres = True
                    elif bidx in self.conserved_motif_blocks:
                        is_conserved = True
                    break

            res_info = ResidueInfo(
                chain=chain,
                resno=resno,
                resname=resname,
                pose_index=pose_idx,
                is_catalytic_residue=is_catres,
                is_conserved_motif=is_conserved,
                block_index=block_idx,
            )

            self.residues[(chain, resno)] = res_info

        LOGGER.info(f"Built info for {len(self.residues)} protein residues")

    def _classify_spheres(self) -> None:
        """Classify residues into spheres based on distance to ligand.

        Uses CB orientation check from modern_FastMPNNdesign:
        - Primary: CA within cuts[0]
        - Secondary: CA within cuts[1] AND (GLY or CB closer than CA or both close)
        - Repack primary: CA within cuts[2]
        - Repack secondary: CA within cuts[3]
        - Distant: beyond cuts[3]
        """
        if not self.ligand or not self.ligand.heavy_atom_coords:
            LOGGER.warning("No ligand coordinates, cannot classify spheres")
            return

        ligand_coords = self.ligand.heavy_atom_coords
        cuts = self.layer_cuts

        # Get atom lookup
        atom_lookup: Dict[Tuple[str, int, str], Dict] = {}
        for atom in self._atoms:
            key = (atom["chain"], atom["resno"], atom["atom_name"])
            atom_lookup[key] = atom

        for (chain, resno), res in self.residues.items():
            # Get CA coordinates
            ca_key = (chain, resno, "CA")
            if ca_key not in atom_lookup:
                continue
            ca_atom = atom_lookup[ca_key]
            ca_xyz = np.array([ca_atom["x"], ca_atom["y"], ca_atom["z"]])

            # Calculate CA distance to nearest ligand atom
            ca_dist = min(np.linalg.norm(ca_xyz - lig) for lig in ligand_coords)
            res.ca_distance_to_ligand = ca_dist

            # Get CB coordinates (if not GLY)
            cb_dist = float('inf')
            if res.resname != "GLY":
                cb_key = (chain, resno, "CB")
                if cb_key in atom_lookup:
                    cb_atom = atom_lookup[cb_key]
                    cb_xyz = np.array([cb_atom["x"], cb_atom["y"], cb_atom["z"]])
                    cb_dist = min(np.linalg.norm(cb_xyz - lig) for lig in ligand_coords)
                    res.cb_distance_to_ligand = cb_dist
                    res.cb_points_toward_ligand = cb_dist < ca_dist

            # Assign sphere based on distance + CB orientation
            if ca_dist <= cuts[0]:
                # Primary sphere (0 - cuts[0])
                res.sphere = DesignSphere.PRIMARY

            elif ca_dist <= cuts[1]:
                # Secondary check (cuts[0] - cuts[1])
                # Include if: GLY, or CB points toward ligand, or both CA and CB are close
                if (res.resname == "GLY" or
                    res.cb_points_toward_ligand or
                    (ca_dist < cuts[1] - 1.0 and cb_dist < cuts[1] - 1.0)):
                    res.sphere = DesignSphere.SECONDARY
                else:
                    res.sphere = DesignSphere.REPACK_PRIMARY

            elif ca_dist <= cuts[2]:
                # Repack primary (cuts[1] - cuts[2])
                res.sphere = DesignSphere.REPACK_PRIMARY

            elif ca_dist <= cuts[3]:
                # Repack secondary (cuts[2] - cuts[3])
                res.sphere = DesignSphere.REPACK_SECONDARY

            else:
                # Distant (> cuts[3])
                res.sphere = DesignSphere.DISTANT

    def _apply_constraint_info(self) -> None:
        """Apply constraint information from step02 JSON."""
        # Try to get residue_constraints from step02 JSON
        # This may be nested in the metadata or at top level

        # First try to load the original step01 JSON if referenced
        step01_json_path = self._step02_data.get("metadata", {}).get("step01_json")
        if step01_json_path and os.path.exists(step01_json_path):
            with open(step01_json_path, 'r') as f:
                step01_data = json.load(f)
            residue_constraints = step01_data.get("residue_constraints", {})
        else:
            # Fall back to step02 data
            residue_constraints = self._step02_data.get("residue_constraints", {})

        if not residue_constraints:
            LOGGER.warning("No residue_constraints found in step02 JSON")
            return

        for block_str, info in residue_constraints.items():
            block_idx = int(block_str)
            chain = info["chain"]
            resno = info["resno"]

            key = (chain, resno)
            if key not in self.residues:
                continue

            res = self.residues[key]
            res.constrain_atoms = info.get("constrain_atoms", [])
            res.backbone_important = info.get("backbone_important", False)
            res.sidechain_important = info.get("sidechain_important", False)

    def _mark_protected_residues(self) -> None:
        """Mark GLY/PRO and other protected residues."""
        for res in self.residues.values():
            if res.resname in PROTECTED_RESIDUES and not self.design_gly_pro:
                res.is_protected = True

    def _log_summary(self) -> None:
        """Log classification summary."""
        counts = {sphere: 0 for sphere in DesignSphere}
        catres_count = 0
        conserved_count = 0
        designable_count = 0

        for res in self.residues.values():
            counts[res.sphere] += 1
            if res.is_catalytic_residue:
                catres_count += 1
            if res.is_conserved_motif:
                conserved_count += 1
            if res.is_designable:
                designable_count += 1

        LOGGER.info(f"Classification summary:")
        LOGGER.info(f"  Total residues: {len(self.residues)}")
        LOGGER.info(f"  Catalytic residues: {catres_count}")
        LOGGER.info(f"  Conserved motif: {conserved_count}")
        LOGGER.info(f"  Designable: {designable_count}")
        for sphere, count in counts.items():
            LOGGER.info(f"  {sphere.value}: {count}")

    def get_summary(self) -> Dict:
        """Get classification summary as dictionary.

        Returns:
            Dict with classification counts and lists
        """
        counts = {sphere.value: 0 for sphere in DesignSphere}
        catres_count = 0
        conserved_count = 0
        designable_count = 0
        fixed_residues = []
        design_residues = []

        for res in self.residues.values():
            counts[res.sphere.value] += 1
            if res.is_catalytic_residue:
                catres_count += 1
            if res.is_conserved_motif:
                conserved_count += 1
            if res.is_designable:
                designable_count += 1
                design_residues.append(f"{res.chain}{res.resno}")
            if res.is_fixed:
                fixed_residues.append(f"{res.chain}{res.resno}")

        return {
            "num_total": len(self.residues),
            "num_catalytic": catres_count,
            "num_conserved_motif": conserved_count,
            "num_primary": counts["primary"],
            "num_secondary": counts["secondary"],
            "num_fixed": len(fixed_residues),
            "num_designable": designable_count,
            "fixed_residues": sorted(fixed_residues),
            "design_residues": sorted(design_residues),
            "sphere_counts": counts,
        }

    # =========================================================================
    # Query methods
    # =========================================================================

    def get_fixed_residues(self) -> List[str]:
        """Get MPNN-format list of fixed residue IDs."""
        return [res.to_mpnn_residue_id() for res in self.residues.values() if res.is_fixed]

    def get_design_residues(self, sphere: Optional[DesignSphere] = None) -> List[str]:
        """Get MPNN-format list of designable residue IDs."""
        residues = [res for res in self.residues.values() if res.is_designable]
        if sphere:
            residues = [res for res in residues if res.sphere == sphere]
        return [res.to_mpnn_residue_id() for res in residues]

    def get_sphere_residues(self, sphere: DesignSphere) -> List[ResidueInfo]:
        """Get all residues in a specific sphere."""
        return [res for res in self.residues.values() if res.sphere == sphere]

    def get_catres_positions(self) -> List[Tuple[str, int]]:
        """Get (chain, resno) for all catalytic residues."""
        return [(res.chain, res.resno) for res in self.residues.values() if res.is_catalytic_residue]

    def get_constrained_atoms(self) -> Dict[Tuple[str, int], List[str]]:
        """Get atoms to constrain for each catalytic residue."""
        result = {}
        for res in self.residues.values():
            if res.is_catalytic_residue and res.constrain_atoms:
                result[(res.chain, res.resno)] = res.constrain_atoms
        return result

    def get_residue_by_id(self, res_id: str) -> Optional[ResidueInfo]:
        """Get ResidueInfo by MPNN-format ID (e.g., 'A45')."""
        chain, resno = ResidueInfo.from_mpnn_residue_id(res_id)
        return self.residues.get((chain, resno))

    def to_dict(self) -> Dict:
        """Export classification to dictionary for JSON serialization."""
        return {
            "ligand": {
                "chain": self.ligand.chain,
                "resname": self.ligand.resname,
                "resno": self.ligand.resno,
            } if self.ligand else None,
            "layer_cuts": self.layer_cuts,
            "catres_subset_blocks": sorted(self.catres_subset_blocks),
            "conserved_motif_blocks": sorted(self.conserved_motif_blocks),
            "residues": {
                f"{res.chain}{res.resno}": {
                    "chain": res.chain,
                    "resno": res.resno,
                    "resname": res.resname,
                    "sphere": res.sphere.value,
                    "is_catalytic_residue": res.is_catalytic_residue,
                    "is_conserved_motif": res.is_conserved_motif,
                    "is_designable": res.is_designable,
                    "ca_distance_to_ligand": round(res.ca_distance_to_ligand, 2),
                    "block_index": res.block_index,
                }
                for res in self.residues.values()
            },
            "summary": {
                "total_residues": len(self.residues),
                "catres_count": len([r for r in self.residues.values() if r.is_catalytic_residue]),
                "conserved_count": len([r for r in self.residues.values() if r.is_conserved_motif]),
                "designable_count": len([r for r in self.residues.values() if r.is_designable]),
                "primary_sphere": len(self.get_sphere_residues(DesignSphere.PRIMARY)),
                "secondary_sphere": len(self.get_sphere_residues(DesignSphere.SECONDARY)),
            },
        }
