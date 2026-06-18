"""
Histidine protonation state detection and handling.

Provides functions to detect and preserve histidine protonation states in PDB files,
particularly important for metal-coordinating histidines.

Protonation States:
- HIS (standard): Proton on NE2 (tautomer with H on epsilon nitrogen)
- HIS_D: Proton on ND1 (tautomer with H on delta nitrogen)
  - Used when NE2 coordinates a metal ion

When NE2 coordinates a metal (e.g., ZN), the residue should be HIS_D so the
proton is on ND1 and NE2 can donate its lone pair to the metal.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import math

from fastmpnndesign.utils import iter_pdb_atoms, calculate_distance
from fastmpnndesign.ligand import detect_metals_from_pdb, Ligand
from fastmpnndesign.logging_config import get_logger

logger = get_logger("protonation")

# Metal coordination distance cutoff (typical for Zn-N coordination)
METAL_COORDINATION_CUTOFF: float = 2.5

# Histidine residue names
HIS_STANDARD: str = "HIS"
HIS_DELTA: str = "HIS_D"

# Histidine atoms involved in metal coordination
HIS_NE2: str = "NE2"  # Epsilon nitrogen
HIS_ND1: str = "ND1"  # Delta nitrogen
HIS_HE2: str = "HE2"  # Hydrogen on epsilon nitrogen (present in HIS)
HIS_HD1: str = "HD1"  # Hydrogen on delta nitrogen (present in HIS_D)


@dataclass
class HistidineProtonation:
    """Represents the protonation state of a histidine residue."""
    chain: str
    resnum: int
    icode: str
    original_resname: str
    protonation_state: str  # 'HIS' or 'HIS_D'
    reason: str  # Why this state was assigned
    coordinating_metal: Optional[str] = None  # Metal being coordinated, if any
    coordination_distance: Optional[float] = None  # Distance to coordinating metal

    def to_dict(self) -> Dict:
        return {
            'chain': self.chain,
            'resnum': self.resnum,
            'icode': self.icode,
            'original_resname': self.original_resname,
            'protonation_state': self.protonation_state,
            'reason': self.reason,
            'coordinating_metal': self.coordinating_metal,
            'coordination_distance': self.coordination_distance
        }

    @property
    def resid(self) -> str:
        """Return residue ID in format 'chain_resnum'."""
        icode_str = self.icode if self.icode else ""
        return f"{self.chain}{self.resnum}{icode_str}"


def detect_histidine_protonation_states(
    pdb_path: Path,
    metals: Optional[List[Ligand]] = None,
    coordination_cutoff: float = METAL_COORDINATION_CUTOFF
) -> Dict[Tuple[str, int, str], HistidineProtonation]:
    """
    Detect histidine protonation states from a PDB file.

    This function examines each histidine and determines its protonation state based on:
    1. Presence of HE2 atom -> HIS (standard, proton on NE2)
    2. Presence of HD1 atom -> HIS_D (proton on ND1)
    3. NE2 coordinating a metal within cutoff -> HIS_D (NE2 donates to metal)
    4. ND1 coordinating a metal within cutoff -> HIS (ND1 donates to metal)

    Args:
        pdb_path: Path to input PDB file.
        metals: Pre-detected metal ions (optional, will detect if not provided).
        coordination_cutoff: Distance cutoff for metal coordination (default 2.5 A).

    Returns:
        Dictionary mapping (chain, resnum, icode) to HistidineProtonation objects.
    """
    pdb_path = Path(pdb_path)

    # Detect metals if not provided
    if metals is None:
        metals = detect_metals_from_pdb(pdb_path)

    # Build metal coordinates lookup
    metal_coords: List[Tuple[str, float, float, float]] = []
    for metal in metals:
        for atom in metal.atoms:
            metal_id = f"{metal.resname}_{metal.chain}_{metal.resnum}"
            metal_coords.append((metal_id, atom.x, atom.y, atom.z))

    # Collect histidine atoms grouped by residue
    his_atoms: Dict[Tuple[str, int, str], Dict[str, Tuple[float, float, float]]] = {}
    his_resnames: Dict[Tuple[str, int, str], str] = {}

    for atom in iter_pdb_atoms(pdb_path):
        resname = atom['resname'].upper()

        # Check for histidine variants (HIS, HID, HIE, HIP, HIS_D, etc.)
        if not resname.startswith('HIS') and resname not in ('HID', 'HIE', 'HIP'):
            continue

        chain = atom['chain'] or 'A'
        resnum = atom['resnum']
        icode = atom['icode'] or ''
        key = (chain, resnum, icode)

        if key not in his_atoms:
            his_atoms[key] = {}
            his_resnames[key] = resname

        atom_name = atom['name'].strip()
        his_atoms[key][atom_name] = (atom['x'], atom['y'], atom['z'])

    # Determine protonation state for each histidine
    protonation_states: Dict[Tuple[str, int, str], HistidineProtonation] = {}

    for key, atoms in his_atoms.items():
        chain, resnum, icode = key
        original_resname = his_resnames[key]

        # Initialize with default
        state = HIS_STANDARD
        reason = "default"
        coord_metal = None
        coord_dist = None

        # Check for explicit hydrogen atoms indicating protonation
        has_HE2 = HIS_HE2 in atoms
        has_HD1 = HIS_HD1 in atoms

        if has_HE2 and not has_HD1:
            state = HIS_STANDARD
            reason = "HE2 atom present (proton on NE2)"
        elif has_HD1 and not has_HE2:
            state = HIS_DELTA
            reason = "HD1 atom present (proton on ND1)"
        elif has_HE2 and has_HD1:
            # Both hydrogens present - this is HIP (doubly protonated)
            # For metal coordination purposes, treat as HIS
            state = HIS_STANDARD
            reason = "Both HE2 and HD1 present (HIP doubly protonated)"

        # Check for metal coordination if no explicit hydrogens found
        # or to override based on coordination
        if HIS_NE2 in atoms and metal_coords:
            ne2_coords = atoms[HIS_NE2]
            for metal_id, mx, my, mz in metal_coords:
                dist = calculate_distance(
                    ne2_coords[0], ne2_coords[1], ne2_coords[2],
                    mx, my, mz
                )
                if dist <= coordination_cutoff:
                    state = HIS_DELTA
                    reason = f"NE2 coordinates metal {metal_id} at {dist:.2f} A"
                    coord_metal = metal_id
                    coord_dist = dist
                    logger.info(
                        f"  HIS {chain}{resnum}{icode}: NE2 coordinates {metal_id} "
                        f"at {dist:.2f} A -> HIS_D"
                    )
                    break

        if HIS_ND1 in atoms and metal_coords and state == HIS_STANDARD:
            nd1_coords = atoms[HIS_ND1]
            for metal_id, mx, my, mz in metal_coords:
                dist = calculate_distance(
                    nd1_coords[0], nd1_coords[1], nd1_coords[2],
                    mx, my, mz
                )
                if dist <= coordination_cutoff:
                    state = HIS_STANDARD
                    reason = f"ND1 coordinates metal {metal_id} at {dist:.2f} A"
                    coord_metal = metal_id
                    coord_dist = dist
                    logger.info(
                        f"  HIS {chain}{resnum}{icode}: ND1 coordinates {metal_id} "
                        f"at {dist:.2f} A -> HIS (standard)"
                    )
                    break

        # Check original residue name for explicit state
        if original_resname == 'HIS_D' or original_resname == 'HID':
            state = HIS_DELTA
            if 'metal' not in reason.lower():
                reason = f"Original resname {original_resname} indicates delta protonation"
        elif original_resname == 'HIE':
            state = HIS_STANDARD
            if 'metal' not in reason.lower():
                reason = f"Original resname {original_resname} indicates epsilon protonation"

        protonation_states[key] = HistidineProtonation(
            chain=chain,
            resnum=resnum,
            icode=icode,
            original_resname=original_resname,
            protonation_state=state,
            reason=reason,
            coordinating_metal=coord_metal,
            coordination_distance=coord_dist
        )

    # Log summary
    n_his_d = sum(1 for p in protonation_states.values() if p.protonation_state == HIS_DELTA)
    n_his = sum(1 for p in protonation_states.values() if p.protonation_state == HIS_STANDARD)
    logger.info(f"Detected {len(protonation_states)} histidines: {n_his} HIS, {n_his_d} HIS_D")

    return protonation_states


def get_protonation_state_dict(
    protonation_states: Dict[Tuple[str, int, str], HistidineProtonation]
) -> Dict[Tuple[str, int], str]:
    """
    Convert protonation states to a simple lookup dictionary.

    Args:
        protonation_states: Full protonation state information.

    Returns:
        Dictionary mapping (chain, resnum) to protonation state string ('HIS' or 'HIS_D').
    """
    return {
        (p.chain, p.resnum): p.protonation_state
        for p in protonation_states.values()
    }


def protonation_states_to_json(
    protonation_states: Dict[Tuple[str, int, str], HistidineProtonation]
) -> List[Dict]:
    """
    Convert protonation states to JSON-serializable format.

    Args:
        protonation_states: Protonation state information.

    Returns:
        List of dictionaries suitable for JSON serialization.
    """
    return [p.to_dict() for p in protonation_states.values()]


def protonation_states_from_json(
    data: List[Dict]
) -> Dict[Tuple[str, int, str], HistidineProtonation]:
    """
    Reconstruct protonation states from JSON data.

    Args:
        data: List of dictionaries from JSON.

    Returns:
        Dictionary mapping (chain, resnum, icode) to HistidineProtonation objects.
    """
    result = {}
    for d in data:
        key = (d['chain'], d['resnum'], d.get('icode', ''))
        result[key] = HistidineProtonation(
            chain=d['chain'],
            resnum=d['resnum'],
            icode=d.get('icode', ''),
            original_resname=d['original_resname'],
            protonation_state=d['protonation_state'],
            reason=d['reason'],
            coordinating_metal=d.get('coordinating_metal'),
            coordination_distance=d.get('coordination_distance')
        )
    return result


def generate_mutate_residue_commands(
    protonation_states: Dict[Tuple[str, int, str], HistidineProtonation]
) -> List[Dict[str, str]]:
    """
    Generate PyRosetta MutateResidue commands to enforce protonation states.

    This generates the data needed to apply MutateResidue movers in PyRosetta
    to change HIS to HIS_D or vice versa.

    Args:
        protonation_states: Protonation state information.

    Returns:
        List of dictionaries with 'chain', 'resnum', 'target_resname' keys.
    """
    commands = []
    for (chain, resnum, icode), prot in protonation_states.items():
        if prot.protonation_state == HIS_DELTA:
            commands.append({
                'chain': chain,
                'resnum': resnum,
                'icode': icode,
                'target_resname': 'HIS_D'
            })
    return commands


def verify_protonation_states(
    pdb_path: Path,
    expected_states: Dict[Tuple[str, int, str], HistidineProtonation]
) -> Tuple[bool, List[str]]:
    """
    Verify that a PDB file has the expected histidine protonation states.

    Args:
        pdb_path: Path to PDB file to verify.
        expected_states: Expected protonation states.

    Returns:
        Tuple of (all_match, list of mismatch messages).
    """
    actual_states = detect_histidine_protonation_states(pdb_path)

    mismatches = []
    for key, expected in expected_states.items():
        if key not in actual_states:
            mismatches.append(
                f"Missing histidine at {expected.chain}{expected.resnum}"
            )
            continue

        actual = actual_states[key]
        if actual.protonation_state != expected.protonation_state:
            mismatches.append(
                f"Protonation mismatch at {expected.chain}{expected.resnum}: "
                f"expected {expected.protonation_state}, got {actual.protonation_state}"
            )

    return len(mismatches) == 0, mismatches
