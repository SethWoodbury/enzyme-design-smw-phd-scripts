"""
Task operations for enhanced_fastmpnndesign.

Provides task operations like SelectHBondsToResidue for the design protocol.
"""

import sys
import copy
from pathlib import Path
from typing import List, Optional, Any
import numpy as np

# Add package directory to path for standalone execution
_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from logging_config import get_logger

logger = get_logger("task_operations")


class SelectHBondsToResidue:
    """
    Task operation that identifies H-bond contacts to target residues.

    Ported from FastMPNNDesign/Selectors.py for the 'keep_hbonds_to_ligand_and_catres'
    functionality used in the protocol.

    This is used to dynamically fix residues that form hydrogen bonds
    to the ligand and catalytic residues during design.
    """

    def __init__(self, name: str):
        """
        Initialize the HBond selector.

        Args:
            name: Name for this task operation (used in protocol)
        """
        self._name = name
        self._target_resnos: List[int] = []
        self._target_atoms: Optional[List[str]] = None
        self._accept_prob: float = 1.0
        self._accept_energy: float = 0.0
        self._include_bb_hbonds: bool = False
        self._enable_updating: bool = False

    def name(self) -> str:
        """Get the name of this task operation."""
        return self._name

    def target(self, resno: Optional[List[int]] = None) -> Optional[List[int]]:
        """
        Set or get target residue numbers for H-bond analysis.

        Args:
            resno: List of residue numbers to target, or None to get current

        Returns:
            Current target list if resno is None
        """
        if resno is None:
            return self._target_resnos
        else:
            if isinstance(resno, int):
                self._target_resnos = [resno]
            else:
                self._target_resnos = list(resno)
            return None

    def target_atoms(self, target_atoms: Optional[List[str]] = None) -> Optional[List[str]]:
        """
        Set or get target atom names (not currently used).

        Args:
            target_atoms: List of atom names, or None to get current

        Returns:
            Current target atoms if target_atoms is None
        """
        if target_atoms is None:
            return self._target_atoms
        else:
            self._target_atoms = target_atoms
            return None

    def accept_probability(self, probability: Optional[float] = None) -> Optional[float]:
        """
        Set or get the probability of accepting an identified H-bond contact.

        Args:
            probability: Probability value (0.0 to 1.0), or None to get current

        Returns:
            Current probability if probability is None
        """
        if probability is None:
            return self._accept_prob
        else:
            assert 0.0 <= probability <= 1.0, "Probability must be between 0 and 1"
            self._accept_prob = probability
            return None

    def accept_energy(self, energy: Optional[float] = None) -> Optional[float]:
        """
        Set or get energy threshold for H-bond acceptance (not currently used).

        Args:
            energy: Energy threshold, or None to get current

        Returns:
            Current energy threshold if energy is None
        """
        if energy is None:
            return self._accept_energy
        else:
            self._accept_energy = energy
            return None

    def include_backbone_hbonds(self, include: Optional[bool] = None) -> Optional[bool]:
        """
        Set or get whether to include backbone H-bonds.

        By default, only sidechain H-bonds are considered.

        Args:
            include: Whether to include backbone H-bonds, or None to get current

        Returns:
            Current setting if include is None
        """
        if include is None:
            return self._include_bb_hbonds
        else:
            self._include_bb_hbonds = include
            return None

    def allow_updating(self, allow: Optional[bool] = None) -> Optional[bool]:
        """
        Set or get whether target set can be updated during execution.

        Args:
            allow: Whether to allow updating, or None to get current

        Returns:
            Current setting if allow is None
        """
        if allow is None:
            return self._enable_updating
        else:
            self._enable_updating = allow
            return None

    def copy(self) -> 'SelectHBondsToResidue':
        """Create a deep copy of this task operation."""
        return copy.deepcopy(self)

    def compute(self, pose: Any) -> List[int]:
        """
        Compute residues with H-bonds to target residues.

        Uses PyRosetta HBondSet analysis to find all residues
        that form hydrogen bonds with the target residues.

        Args:
            pose: PyRosetta Pose object

        Returns:
            List of residue sequence positions to fix
        """
        import pandas as pd

        # Get H-bonds from pose
        hbonds = pose.get_hbonds()

        # Build DataFrame of H-bonds
        data = {
            'donor_residue': [],
            'acceptor_residue': [],
            'donor_atom': [],
            'acceptor_atom': [],
            'energy': []
        }

        for hbond in hbonds.hbonds():
            data['donor_residue'].append(hbond.don_res())
            data['acceptor_residue'].append(hbond.acc_res())
            data['donor_atom'].append(
                pose.residue(hbond.don_res()).atom_name(hbond.don_hatm())
            )
            data['acceptor_atom'].append(
                pose.residue(hbond.acc_res()).atom_name(hbond.acc_atm())
            )
            data['energy'].append(hbond.energy())

        hbond_df = pd.DataFrame(data)

        if len(hbond_df) == 0:
            return []

        # Filter for H-bonds involving targets
        targets = self.target()
        mask = (
            hbond_df['donor_residue'].isin(targets) |
            hbond_df['acceptor_residue'].isin(targets)
        )
        target_hbond_df = hbond_df[mask]

        if len(target_hbond_df) == 0:
            return []

        # Filter backbone H-bonds if not included
        if not self.include_backbone_hbonds():
            target_hbond_df = self._filter_backbone_hbonds(target_hbond_df, pose)

        if len(target_hbond_df) == 0:
            return []

        # Get unique residues involved in H-bonds (excluding targets)
        all_residues = set(target_hbond_df['donor_residue'].tolist() +
                          target_hbond_df['acceptor_residue'].tolist())
        selection = [x for x in all_residues if x not in targets]

        # Apply acceptance probability
        if self.accept_probability() < 1.0:
            selection = [
                x for x in selection
                if np.random.rand() <= self.accept_probability()
            ]

        logger.debug(f"HBond keeper selected {len(selection)} residues")
        return selection

    def _filter_backbone_hbonds(self, df: Any, pose: Any) -> Any:
        """
        Filter out backbone-only H-bonds.

        Logic from original Selectors.py lines 144-174.

        Args:
            df: DataFrame of H-bonds
            pose: PyRosetta Pose

        Returns:
            Filtered DataFrame
        """
        targets = self.target()
        mask = []

        for _, row in df.iterrows():
            keep = True

            # Both in targets - skip
            if row.donor_residue in targets and row.acceptor_residue in targets:
                keep = False

            # Ligand involved
            elif (pose.residue(row.donor_residue).is_ligand() or
                  pose.residue(row.acceptor_residue).is_ligand()):
                res1 = pose.residue(row.donor_residue)
                res2 = pose.residue(row.acceptor_residue)
                # Skip if backbone atoms involved
                try:
                    if (res1.atom_is_backbone(res1.atom_index(row.donor_atom)) or
                        res2.atom_is_backbone(res2.atom_index(row.acceptor_atom))):
                        keep = False
                except:
                    pass  # Ligand may not have atom_is_backbone method

            # Protein-protein H-bond
            else:
                # Determine which is target and which is other
                if row.donor_residue in targets:
                    tgt_pair = (row.donor_residue, row.donor_atom)
                    other_pair = (row.acceptor_residue, row.acceptor_atom)
                else:
                    tgt_pair = (row.acceptor_residue, row.acceptor_atom)
                    other_pair = (row.donor_residue, row.donor_atom)

                res_tgt = pose.residue(tgt_pair[0])
                res_other = pose.residue(other_pair[0])

                try:
                    tgt_is_bb = res_tgt.atom_is_backbone(res_tgt.atom_index(tgt_pair[1]))
                    other_is_bb = res_other.atom_is_backbone(res_other.atom_index(other_pair[1]))

                    # Both backbone - skip
                    if tgt_is_bb and other_is_bb:
                        keep = False
                    # Other is backbone - skip
                    elif other_is_bb:
                        keep = False
                except:
                    pass

            mask.append(keep)

        return df[mask]


def create_hbond_keeper(
    name: str,
    ligand_seqpos: int,
    catres_seqpos: List[int],
    accept_probability: float = 0.75
) -> SelectHBondsToResidue:
    """
    Create a configured HBond keeper task operation.

    Args:
        name: Name for the task operation
        ligand_seqpos: Sequence position of ligand
        catres_seqpos: List of catalytic residue sequence positions
        accept_probability: Probability of keeping identified H-bond contacts

    Returns:
        Configured SelectHBondsToResidue instance
    """
    hbond_keeper = SelectHBondsToResidue(name=name)
    hbond_keeper.target([ligand_seqpos] + catres_seqpos)
    hbond_keeper.allow_updating(True)
    hbond_keeper.accept_probability(accept_probability)

    logger.info(f"Created HBond keeper '{name}' targeting {1 + len(catres_seqpos)} residues")
    return hbond_keeper
