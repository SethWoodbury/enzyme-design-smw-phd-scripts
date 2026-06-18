"""Interaction detection utilities for the upgraded_fastMPNNdesign pipeline.

This module provides reusable functions for detecting molecular interactions:
- H-bond detection with geometry validation
- Pi-stacking classification (parallel, displaced, T-shaped)
- Hydrophobic contacts
- Metal coordination
- Charged/ionic interactions

These functions are extracted from step01/align_catres.py for reuse in step03.
"""
import logging
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from .constants import (
    HBOND_DIST_MAX, HBOND_DONOR_ANGLE_MIN, HBOND_ACCEPTOR_ANGLE_MIN,
    H_BOND_DIST_MAX, HEAVY_BOND_DIST_MAX,
    PI_CENTROID_DIST_MIN, PI_CENTROID_DIST_MAX, PI_PARALLEL_ANGLE_MAX,
    PI_TSHAPE_ANGLE_MIN, PI_TSHAPE_ANGLE_MAX, PI_PERP_SEPARATION_MAX,
    PI_OFFSET_FACE_TO_FACE, PI_OFFSET_DISPLACED_MAX, PI_TSHAPE_CONTACT_MAX,
    HYDROPHOBIC_DIST_MAX, CHARGED_DIST_MAX, METAL_COORD_DIST_MAX,
    AROMATIC_RESIDUES, AROMATIC_RING_ATOMS, HYDROPHOBIC_RESIDUES,
    CHARGED_RESIDUES, CHARGED_ATOMS_BY_RESIDUE, POLAR_ELEMENTS, METAL_ATOMS,
)
from .pdb_utils import is_backbone_atom

LOGGER = logging.getLogger(__name__)

# Polar sidechain atoms that can participate in H-bonds
POLAR_SIDECHAIN_ATOMS: Set[str] = {
    "OG", "OG1", "OD1", "OD2", "OE1", "OE2", "ND1", "ND2", "NE", "NE1", "NE2",
    "NH1", "NH2", "NZ", "OH", "SG", "SD", "OXT"
}

# Backbone atoms for H-bonding
BACKBONE_HBOND_DONORS: Set[str] = {"N", "H"}
BACKBONE_HBOND_ACCEPTORS: Set[str] = {"O", "OXT"}

# Non-polar sidechain residues (no polar SC atoms)
NONPOLAR_SIDECHAIN_RESIDUES: Set[str] = {"ALA", "VAL", "LEU", "ILE", "PHE", "PRO", "GLY"}


# =============================================================================
# Basic Geometry Functions
# =============================================================================

def distance(a1: Dict, a2: Dict) -> float:
    """Euclidean distance between two atoms."""
    return np.sqrt((a1["x"] - a2["x"])**2 + (a1["y"] - a2["y"])**2 + (a1["z"] - a2["z"])**2)


def distance_coords(c1: np.ndarray, c2: np.ndarray) -> float:
    """Euclidean distance between two coordinate arrays."""
    return np.linalg.norm(c1 - c2)


def get_coords(atom: Dict) -> np.ndarray:
    """Extract xyz coordinates from atom dict."""
    return np.array([atom["x"], atom["y"], atom["z"]])


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculate angle (in degrees) at p2 between vectors p2->p1 and p2->p3."""
    v1, v2 = p1 - p2, p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def is_polar_atom(elem: str) -> bool:
    """Check if element is a polar heavy atom (N, O, S) that can participate in H-bonds."""
    return elem.upper() in POLAR_ELEMENTS


def is_heteroatom(elem: str) -> bool:
    """Check if element is a non-carbon heteroatom (N, O, S, P, metals, halogens)."""
    return elem.upper() not in {"C", "H", ""}


# =============================================================================
# H-Bond Detection
# =============================================================================

def find_bonded_heavy_atom(h_atom: Dict, atoms: List[Dict]) -> Optional[Dict]:
    """Find the heavy atom that a hydrogen is bonded to (nearest heavy atom within bond distance)."""
    h_elem = h_atom.get("element", h_atom["atom_name"][0])
    if h_elem.upper() != "H":
        return None
    best_atom, best_dist = None, H_BOND_DIST_MAX
    for a in atoms:
        elem = a.get("element", a["atom_name"][0])
        if elem.upper() == "H" or a["atom_name"] == h_atom["atom_name"]:
            continue
        d = distance(h_atom, a)
        if d < best_dist:
            best_dist, best_atom = d, a
    return best_atom


def find_bonded_heavy_to_acceptor(acceptor: Dict, atoms: List[Dict]) -> Optional[Dict]:
    """Find a heavy atom bonded to the acceptor (for H...A-B angle calculation)."""
    best_atom, best_dist = None, HEAVY_BOND_DIST_MAX
    for a in atoms:
        elem = a.get("element", a["atom_name"][0])
        if elem.upper() == "H" or a["atom_name"] == acceptor["atom_name"]:
            continue
        d = distance(acceptor, a)
        if d < best_dist:
            best_dist, best_atom = d, a
    return best_atom


def check_hbond_geometry(
    donor_atom: Dict,
    h_atom: Dict,
    acceptor_atom: Dict,
    acceptor_base: Dict,
    donor_angle_min: float = HBOND_DONOR_ANGLE_MIN,
    acceptor_angle_min: float = HBOND_ACCEPTOR_ANGLE_MIN,
) -> bool:
    """Check if H-bond geometry is valid (angles within thresholds).

    Args:
        donor_atom: Heavy atom the H is bonded to (D in D-H...A)
        h_atom: The hydrogen atom
        acceptor_atom: The acceptor atom (A in D-H...A)
        acceptor_base: Heavy atom bonded to acceptor (B in H...A-B)
        donor_angle_min: Minimum D-H...A angle (default 120°)
        acceptor_angle_min: Minimum H...A-B angle (default 100°)

    Returns:
        True if both D-H...A and H...A-B angles are within acceptable ranges
    """
    if donor_atom is None or acceptor_base is None:
        return False
    d_coords, h_coords = get_coords(donor_atom), get_coords(h_atom)
    a_coords, b_coords = get_coords(acceptor_atom), get_coords(acceptor_base)

    # D-H...A angle (at H): should be >= 120° for good linearity
    angle_dha = calculate_angle(d_coords, h_coords, a_coords)
    if angle_dha < donor_angle_min:
        return False

    # H...A-B angle (at A): should be >= 100° for lone pair access
    angle_hab = calculate_angle(h_coords, a_coords, b_coords)
    if angle_hab < acceptor_angle_min:
        return False

    return True


def is_valid_hbond_donor(h_atom: Dict, donor_atoms: List[Dict]) -> Tuple[bool, Optional[Dict]]:
    """Check if hydrogen is bonded to a polar atom (N/O/S) making it a valid H-bond donor.

    Returns:
        Tuple of (is_valid, donor_heavy_atom)
    """
    donor = find_bonded_heavy_atom(h_atom, donor_atoms)
    if donor is None:
        return False, None
    donor_elem = donor.get("element", donor["atom_name"][0])
    return is_polar_atom(donor_elem), donor


def detect_hbonds(
    donor_atoms: List[Dict],
    acceptor_atoms: List[Dict],
    dist_max: float = HBOND_DIST_MAX,
) -> List[Dict]:
    """Detect H-bonds between donor and acceptor atom sets.

    Looks for H atoms in donor_atoms that are:
    1. Bonded to polar heavy atoms (N, O, S)
    2. Within dist_max of polar atoms in acceptor_atoms
    3. Have valid H-bond geometry

    Args:
        donor_atoms: List of atoms that may donate H-bonds (includes H atoms)
        acceptor_atoms: List of atoms that may accept H-bonds
        dist_max: Maximum H...A distance

    Returns:
        List of H-bond dicts with keys: donor_atom, h_atom, acceptor_atom, distance, angle_dha
    """
    hbonds = []

    # Find all H atoms in donor set
    h_atoms = [a for a in donor_atoms if a.get("element", a["atom_name"][0]).upper() == "H"]

    for h_atom in h_atoms:
        # Check if H is bonded to polar atom
        is_valid, donor_heavy = is_valid_hbond_donor(h_atom, donor_atoms)
        if not is_valid or donor_heavy is None:
            continue

        # Find potential acceptors
        for acc in acceptor_atoms:
            acc_elem = acc.get("element", acc["atom_name"][0])
            if not is_polar_atom(acc_elem):
                continue

            d = distance(h_atom, acc)
            if d > dist_max:
                continue

            # Get acceptor base for geometry check
            acc_base = find_bonded_heavy_to_acceptor(acc, acceptor_atoms)
            if acc_base is None:
                continue

            if check_hbond_geometry(donor_heavy, h_atom, acc, acc_base):
                # Calculate angles for reporting
                d_coords = get_coords(donor_heavy)
                h_coords = get_coords(h_atom)
                a_coords = get_coords(acc)

                angle_dha = calculate_angle(d_coords, h_coords, a_coords)

                hbonds.append({
                    "donor_atom": donor_heavy["atom_name"],
                    "h_atom": h_atom["atom_name"],
                    "acceptor_atom": acc["atom_name"],
                    "distance": round(d, 2),
                    "angle_dha": round(angle_dha, 1),
                })

    return hbonds


# =============================================================================
# Pi-Stacking Detection
# =============================================================================

def get_ring_atoms(atoms: List[Dict], resname: str) -> List[List[Dict]]:
    """Get aromatic ring atom lists for a residue.

    Args:
        atoms: All atoms of the residue
        resname: Three-letter residue code

    Returns:
        List of ring atom lists (each ring is a list of atom dicts)
    """
    if resname not in AROMATIC_RING_ATOMS:
        return []

    rings = []
    atom_lookup = {a["atom_name"]: a for a in atoms}

    for ring_atom_names in AROMATIC_RING_ATOMS[resname]:
        ring_atoms = []
        for name in ring_atom_names:
            if name in atom_lookup:
                ring_atoms.append(atom_lookup[name])
        # Only include ring if we found all atoms
        if len(ring_atoms) == len(ring_atom_names):
            rings.append(ring_atoms)

    return rings


def compute_ring_geometry(ring_atoms: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ring centroid and plane normal.

    Args:
        ring_atoms: List of atom dicts for the ring

    Returns:
        Tuple of (centroid, unit_normal)
    """
    coords = np.array([[a["x"], a["y"], a["z"]] for a in ring_atoms])
    centroid = coords.mean(axis=0)

    # Compute plane normal via SVD (best-fit plane)
    centered = coords - centroid
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]  # Last row of Vt is the normal to the best-fit plane

    # Ensure unit normal
    normal = normal / (np.linalg.norm(normal) + 1e-10)
    return centroid, normal


def classify_pi_pi_interaction(ring_a: List[Dict], ring_b: List[Dict]) -> Optional[str]:
    """Classify pi-pi interaction between two aromatic rings.

    Args:
        ring_a: Atoms of first ring
        ring_b: Atoms of second ring

    Returns:
        Interaction type: "pi_pi_stacking", "pi_pi_stacking_displaced", "pi_pi_Tshape", or None
    """
    if len(ring_a) < 3 or len(ring_b) < 3:
        return None

    centroid_a, normal_a = compute_ring_geometry(ring_a)
    centroid_b, normal_b = compute_ring_geometry(ring_b)

    # (a) Centroid distance
    centroid_vec = centroid_b - centroid_a
    d_cent = np.linalg.norm(centroid_vec)

    if d_cent < PI_CENTROID_DIST_MIN or d_cent > PI_CENTROID_DIST_MAX:
        return None

    # (b) Interplanar angle (use abs so 0° and 180° both mean "parallel")
    cos_theta = abs(np.dot(normal_a, normal_b))
    theta = np.degrees(np.arccos(np.clip(cos_theta, 0.0, 1.0)))

    # (c) Perpendicular separation (projection onto each plane's normal)
    h_a = abs(np.dot(centroid_vec, normal_a))
    h_b = abs(np.dot(centroid_vec, normal_b))

    # (d) Lateral offset (in-plane component)
    in_plane_vec = centroid_vec - np.dot(centroid_vec, normal_a) * normal_a
    offset = np.linalg.norm(in_plane_vec)

    # Classification
    if theta <= PI_PARALLEL_ANGLE_MAX:
        # Parallel stacking
        if h_a > PI_PERP_SEPARATION_MAX or h_b > PI_PERP_SEPARATION_MAX:
            return None
        if offset > PI_OFFSET_DISPLACED_MAX:
            return None
        if offset <= PI_OFFSET_FACE_TO_FACE:
            return "pi_pi_stacking"
        else:
            return "pi_pi_stacking_displaced"

    elif PI_TSHAPE_ANGLE_MIN <= theta <= PI_TSHAPE_ANGLE_MAX:
        # T-shaped: need to verify actual contact (min atom-atom distance)
        min_dist = float("inf")
        for a in ring_a:
            for b in ring_b:
                d = distance(a, b)
                if d < min_dist:
                    min_dist = d
        if min_dist <= PI_TSHAPE_CONTACT_MAX:
            return "pi_pi_Tshape"

    return None


def detect_pi_stacking(
    residue1_atoms: List[Dict],
    resname1: str,
    residue2_atoms: List[Dict],
    resname2: str,
) -> List[Dict]:
    """Detect pi-stacking interactions between two aromatic residues.

    Args:
        residue1_atoms: Atoms of first residue
        resname1: Three-letter code of first residue
        residue2_atoms: Atoms of second residue
        resname2: Three-letter code of second residue

    Returns:
        List of pi interaction dicts with keys: type, centroid_distance
    """
    if resname1 not in AROMATIC_RESIDUES or resname2 not in AROMATIC_RESIDUES:
        return []

    rings1 = get_ring_atoms(residue1_atoms, resname1)
    rings2 = get_ring_atoms(residue2_atoms, resname2)

    if not rings1 or not rings2:
        return []

    interactions = []
    best_interaction = None
    best_dist = float("inf")

    for ring_a in rings1:
        for ring_b in rings2:
            pi_type = classify_pi_pi_interaction(ring_a, ring_b)
            if pi_type:
                c1, _ = compute_ring_geometry(ring_a)
                c2, _ = compute_ring_geometry(ring_b)
                d = np.linalg.norm(c2 - c1)
                if d < best_dist:
                    best_dist = d
                    best_interaction = {
                        "type": pi_type,
                        "centroid_distance": round(d, 2),
                    }

    if best_interaction:
        interactions.append(best_interaction)

    return interactions


def detect_pi_ligand_interaction(
    ring_atoms: List[Dict],
    ligand_atoms: List[Dict],
    dist_max: float = HYDROPHOBIC_DIST_MAX,
) -> Optional[Dict]:
    """Detect pi-ligand interaction based on carbon contacts to ring.

    Args:
        ring_atoms: Atoms of the aromatic ring
        ligand_atoms: All ligand atoms
        dist_max: Maximum distance for interaction

    Returns:
        Interaction dict if detected, None otherwise
    """
    if len(ring_atoms) < 3:
        return None

    ring_centroid, _ = compute_ring_geometry(ring_atoms)
    min_dist = float("inf")

    # Check for carbon atoms in ligand within range of ring centroid
    for lig_atom in ligand_atoms:
        elem = lig_atom.get("element", lig_atom["atom_name"][0])
        if elem.upper() != "C":
            continue
        lig_coords = get_coords(lig_atom)
        d = np.linalg.norm(lig_coords - ring_centroid)
        if d <= dist_max and d < min_dist:
            min_dist = d

    if min_dist < float("inf"):
        return {
            "type": "pi_lig_interaction",
            "min_distance": round(min_dist, 2),
        }

    return None


# =============================================================================
# Hydrophobic Contact Detection
# =============================================================================

def detect_hydrophobic_contacts(
    atoms1: List[Dict],
    atoms2: List[Dict],
    dist_max: float = HYDROPHOBIC_DIST_MAX,
) -> List[Dict]:
    """Detect hydrophobic contacts between two atom sets (carbon-carbon).

    Args:
        atoms1: First set of atoms
        atoms2: Second set of atoms
        dist_max: Maximum C-C distance

    Returns:
        List of contact dicts with keys: atom1, atom2, distance
    """
    contacts = []
    min_dist = float("inf")
    best_pair = None

    for a1 in atoms1:
        elem1 = a1.get("element", a1["atom_name"][0])
        if elem1.upper() != "C":
            continue
        if is_backbone_atom(a1["atom_name"]):
            continue

        for a2 in atoms2:
            elem2 = a2.get("element", a2["atom_name"][0])
            if elem2.upper() != "C":
                continue

            d = distance(a1, a2)
            if d <= dist_max and d < min_dist:
                min_dist = d
                best_pair = (a1["atom_name"], a2["atom_name"])

    if best_pair:
        contacts.append({
            "type": "hydrophobic",
            "atom1": best_pair[0],
            "atom2": best_pair[1],
            "distance": round(min_dist, 2),
        })

    return contacts


# =============================================================================
# Metal Coordination Detection
# =============================================================================

def detect_metal_coordination(
    residue_atoms: List[Dict],
    target_atoms: List[Dict],
    dist_max: float = METAL_COORD_DIST_MAX,
) -> List[Dict]:
    """Detect metal coordination between residue polar atoms and metal atoms.

    Args:
        residue_atoms: Atoms of the residue
        target_atoms: Atoms of the target (may contain metals)
        dist_max: Maximum coordination distance

    Returns:
        List of coordination dicts
    """
    coordinations = []

    for res_atom in residue_atoms:
        res_name = res_atom["atom_name"]
        res_elem = res_atom.get("element", res_name[0])

        # Residue atom must be polar (can coordinate metal)
        is_polar = (res_name in POLAR_SIDECHAIN_ATOMS or
                   res_name in BACKBONE_HBOND_ACCEPTORS or
                   res_name in BACKBONE_HBOND_DONORS)

        if not is_polar and res_elem.upper() not in METAL_ATOMS:
            continue

        for tgt_atom in target_atoms:
            tgt_name = tgt_atom["atom_name"]
            tgt_elem = tgt_atom.get("element", tgt_name[0])

            d = distance(res_atom, tgt_atom)
            if d > dist_max:
                continue

            # Check for metal coordination
            if tgt_elem.upper() in METAL_ATOMS and is_polar:
                coordinations.append({
                    "type": "metal_coord",
                    "residue_atom": res_name,
                    "metal_atom": tgt_name,
                    "distance": round(d, 2),
                })
            elif res_elem.upper() in METAL_ATOMS:
                coordinations.append({
                    "type": "metal_coord",
                    "residue_atom": res_name,
                    "metal_atom": tgt_name,
                    "distance": round(d, 2),
                })

    return coordinations


# =============================================================================
# Charged/Ionic Interaction Detection
# =============================================================================

def detect_charged_interactions(
    residue_atoms: List[Dict],
    resname: str,
    target_atoms: List[Dict],
    dist_max: float = CHARGED_DIST_MAX,
) -> List[Dict]:
    """Detect charged/ionic interactions for charged residues.

    Only considers actual charged atoms (NZ for LYS, OD1/OD2 for ASP, etc.)

    Args:
        residue_atoms: Atoms of the charged residue
        resname: Three-letter residue code
        target_atoms: Target atoms (may have opposite charge)
        dist_max: Maximum interaction distance

    Returns:
        List of charged interaction dicts
    """
    if resname not in CHARGED_RESIDUES:
        return []

    charged_atom_names = CHARGED_ATOMS_BY_RESIDUE.get(resname, set())
    if not charged_atom_names:
        return []

    interactions = []
    min_dist = float("inf")

    for res_atom in residue_atoms:
        res_name = res_atom["atom_name"]
        if res_name not in charged_atom_names:
            continue
        if is_backbone_atom(res_name):
            continue

        for tgt_atom in target_atoms:
            tgt_elem = tgt_atom.get("element", tgt_atom["atom_name"][0])
            if not is_heteroatom(tgt_elem):
                continue

            d = distance(res_atom, tgt_atom)
            if d <= dist_max and d < min_dist:
                min_dist = d

    if min_dist < float("inf"):
        interactions.append({
            "type": "charged",
            "min_distance": round(min_dist, 2),
        })

    return interactions


# =============================================================================
# Comprehensive Interaction Analysis
# =============================================================================

def analyze_residue_interactions(
    residue_atoms: List[Dict],
    resname: str,
    target_atoms: List[Dict],
    target_resname: Optional[str] = None,
    target_type: str = "ligand",
) -> Dict:
    """Comprehensive interaction analysis between a residue and target.

    Detects H-bonds, pi-stacking, hydrophobic contacts, metal coordination,
    and charged interactions.

    Args:
        residue_atoms: All atoms of the residue
        resname: Three-letter residue code
        target_atoms: Atoms of target (ligand or another residue)
        target_resname: Three-letter code of target residue (None for ligand)
        target_type: "ligand" or "residue"

    Returns:
        Dict with keys: hbonds, pi_interactions, hydrophobic, metal_coord, charged
    """
    results = {
        "hbonds_as_donor": [],
        "hbonds_as_acceptor": [],
        "pi_interactions": [],
        "hydrophobic": [],
        "metal_coord": [],
        "charged": [],
    }

    # H-bonds: residue as donor
    hbonds_donor = detect_hbonds(residue_atoms, target_atoms)
    results["hbonds_as_donor"] = hbonds_donor

    # H-bonds: residue as acceptor
    hbonds_acceptor = detect_hbonds(target_atoms, residue_atoms)
    # Swap names to indicate residue is acceptor
    for hb in hbonds_acceptor:
        hb["role"] = "acceptor"
    results["hbonds_as_acceptor"] = hbonds_acceptor

    # Pi interactions
    if resname in AROMATIC_RESIDUES:
        rings = get_ring_atoms(residue_atoms, resname)
        for ring in rings:
            if target_type == "ligand":
                pi_lig = detect_pi_ligand_interaction(ring, target_atoms)
                if pi_lig:
                    results["pi_interactions"].append(pi_lig)
                    break
            elif target_resname in AROMATIC_RESIDUES:
                pi_stacks = detect_pi_stacking(residue_atoms, resname, target_atoms, target_resname)
                results["pi_interactions"].extend(pi_stacks)
                break

    # Hydrophobic contacts (for non-aromatic hydrophobic residues)
    if resname in HYDROPHOBIC_RESIDUES and resname not in AROMATIC_RESIDUES:
        hydro = detect_hydrophobic_contacts(residue_atoms, target_atoms)
        results["hydrophobic"] = hydro

    # Metal coordination
    metal = detect_metal_coordination(residue_atoms, target_atoms)
    results["metal_coord"] = metal

    # Charged interactions
    if resname in CHARGED_RESIDUES:
        charged = detect_charged_interactions(residue_atoms, resname, target_atoms)
        results["charged"] = charged

    return results


def count_favorable_interactions(interaction_results: Dict) -> int:
    """Count total number of favorable interactions from analysis results.

    Args:
        interaction_results: Output from analyze_residue_interactions

    Returns:
        Total count of interactions
    """
    count = 0
    count += len(interaction_results.get("hbonds_as_donor", []))
    count += len(interaction_results.get("hbonds_as_acceptor", []))
    count += len(interaction_results.get("pi_interactions", []))
    count += len(interaction_results.get("hydrophobic", []))
    count += len(interaction_results.get("metal_coord", []))
    count += len(interaction_results.get("charged", []))
    return count


def has_strong_interactions(interaction_results: Dict) -> bool:
    """Check if interaction results contain strong interactions (H-bonds or pi-stacking).

    Args:
        interaction_results: Output from analyze_residue_interactions

    Returns:
        True if H-bonds or pi-stacking detected
    """
    has_hbond = (len(interaction_results.get("hbonds_as_donor", [])) > 0 or
                 len(interaction_results.get("hbonds_as_acceptor", [])) > 0)
    has_pi = len(interaction_results.get("pi_interactions", [])) > 0

    return has_hbond or has_pi
