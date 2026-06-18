#!/usr/bin/env python3
"""Align catalytic residue coordinates from reference PDB to input PDB.

This script aligns ref_pdb to input_pdb via their ligands (expected identical geometry, RMSD~0),
identifies catalytic residues from REMARK 666 lines, detects crucial interactions (H-bonds, metal
coordination, charged interactions, hydrophobic/pi contacts), and transforms catres coordinates
from ref_pdb onto input_pdb based on whether backbone, sidechain, or both are important.

Usage:
    python align_catres.py --input_pdb pdb1.pdb --ref_pdb pdb0.pdb --outdir output/ [--catres_subset 1,3,5]
    python align_catres.py --test  # Run with test data

Outputs:
    - <outdir>/<basename>_aligned.pdb: Modified input_pdb with catres in ref_pdb geometry
    - <outdir>/<basename>_interactions.json: Interaction analysis for each catres_subset residue
"""
import argparse, json, logging, os, sys
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

# Add module_utils to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module_utils.pdb_utils import (parse_remark_666, parse_catres_subset, read_pdb_atoms,
                                     get_residue_atoms, get_ligand_atoms, atoms_to_coords,
                                     is_backbone_atom, format_atom_line, BACKBONE_ATOMS)

# ══════════════════════════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════════════════════════
TEST_INPUT_PDB = "/home/woodbuse/special_scripts/upgraded_fastMPNNdesign/modules/step01__catres_alignment/test/input_pdb.pdb"
TEST_REF_PDB = "/home/woodbuse/special_scripts/upgraded_fastMPNNdesign/modules/step01__catres_alignment/test/ref_pdb.pdb"
TEST_OUTDIR = "/home/woodbuse/special_scripts/upgraded_fastMPNNdesign/modules/step01__catres_alignment/test/output_dir"
TEST_CATRES_SUBSET = "1,2,3,4,5,6,7,8,9,10,11,13,15,16,17,18,19"

HBOND_DIST_MAX = 3.5           # Max distance (A) for hydrogen bond detection
METAL_COORD_DIST_MAX = 2.8     # Max distance (A) for metal coordination
CHARGED_DIST_MAX = 4.5         # Max distance (A) for charged interaction
HYDROPHOBIC_DIST_MAX = 4.5     # Max distance (A) for hydrophobic/pi interaction
COVALENT_DIST_MAX = 2.2        # Max distance (A) for covalent/post-translational mod
ACID_BASE_DIST_MAX = 1.5       # Max distance (A) for acid/base proton transfer (within bonding distance)

POLAR_SIDECHAIN_ATOMS = {"OG", "OG1", "OD1", "OD2", "OE1", "OE2", "ND1", "ND2", "NE", "NE1", "NE2",
                          "NH1", "NH2", "NZ", "OH", "SG", "SD", "OXT"}
BACKBONE_HBOND_DONORS = {"N", "H"}          # Backbone NH can donate H-bond
BACKBONE_HBOND_ACCEPTORS = {"O", "OXT"}     # Backbone CO can accept H-bond
METAL_ATOMS = {"ZN", "MG", "CA", "FE", "MN", "CU", "CO", "NI", "NA", "K"}
CHARGED_RESIDUES = {"ASP", "GLU", "LYS", "ARG", "HIS"}
# Only these atoms can participate in charged/ionic interactions (the actual charged moieties)
CHARGED_ATOMS_BY_RESIDUE = {
    "ASP": {"OD1", "OD2"},           # Carboxylate oxygens (negative)
    "GLU": {"OE1", "OE2"},           # Carboxylate oxygens (negative)
    "LYS": {"NZ"},                   # Terminal amine (positive)
    "ARG": {"NE", "NH1", "NH2"},     # Guanidinium nitrogens (positive)
    "HIS": {"ND1", "NE2"},           # Imidazole nitrogens (can be protonated)
}
AROMATIC_RESIDUES = {"PHE", "TYR", "TRP", "HIS"}
HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}

# Aromatic ring atom definitions (heavy atoms only)
AROMATIC_RING_ATOMS = {
    "PHE": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],  # 6-member ring
    "TYR": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],  # 6-member ring
    "HIS": [["CG", "ND1", "CD2", "CE1", "NE2"]],        # 5-member imidazole
    "TRP": [["CG", "CD1", "NE1", "CE2", "CD2"],         # 5-member pyrrole
            ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"]], # 6-member benzene
}

# Pi-stacking geometry thresholds
PI_CENTROID_DIST_MIN = 3.3   # Minimum centroid-centroid distance (A)
PI_CENTROID_DIST_MAX = 6.0   # Maximum centroid-centroid distance (A)
PI_PARALLEL_ANGLE_MAX = 30.0 # Max angle (degrees) for parallel stacking
PI_TSHAPE_ANGLE_MIN = 60.0   # Min angle (degrees) for T-shaped
PI_TSHAPE_ANGLE_MAX = 90.0   # Max angle (degrees) for T-shaped
PI_PERP_SEPARATION_MAX = 4.0 # Max perpendicular separation (A) for stacking
PI_OFFSET_FACE_TO_FACE = 2.0 # Max offset for face-to-face classification
PI_OFFSET_DISPLACED_MAX = 5.0 # Max offset for displaced stacking
PI_TSHAPE_CONTACT_MAX = 4.5  # Max atom-atom distance for T-shape contact check
NONPOLAR_SIDECHAIN_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "PHE", "PRO", "GLY"}  # No polar SC atoms

# Extremely flexible residues with special handling for coordinate movement
EXTREMELY_FLEXIBLE_RESIDUES = {"ARG", "LYS"}

# Tip atoms for extremely flexible residues (heavy atoms only)
# These are the atoms that actually participate in interactions - the flexible tip of the sidechain
FLEXIBLE_RESIDUE_TIP_HEAVY = {
    "LYS": {"NZ", "CE", "CD"},
    "ARG": {"NH1", "CZ", "NH2", "NE", "CD"},
}

# All hydrogens attached to tip heavy atoms (for coordinate movement)
# Includes both polar and non-polar hydrogens since we need to move them with their parent atoms
FLEXIBLE_RESIDUE_TIP_HYDROGENS = {
    "LYS": {"1HZ", "2HZ", "3HZ", "HZ1", "HZ2", "HZ3",  # NZ hydrogens (polar)
            "1HE", "2HE", "HE2", "HE3",                 # CE hydrogens (non-polar)
            "1HD", "2HD", "HD2", "HD3"},                # CD hydrogens (non-polar)
    "ARG": {"HE",                                       # NE hydrogen (polar)
            "1HH1", "2HH1", "HH11", "HH12",            # NH1 hydrogens (polar)
            "1HH2", "2HH2", "HH21", "HH22",            # NH2 hydrogens (polar)
            "1HD", "2HD", "HD2", "HD3"},                # CD hydrogens (non-polar)
}

# Polar hydrogens on tip atoms (for constraints - only polar H are constrained)
FLEXIBLE_RESIDUE_TIP_POLAR_H = {
    "LYS": {"1HZ", "2HZ", "3HZ", "HZ1", "HZ2", "HZ3"},  # NZ has polar H
    "ARG": {"HE", "1HH1", "2HH1", "1HH2", "2HH2", "HH11", "HH12", "HH21", "HH22"},  # NE, NH1, NH2 have polar H
}

# Atom classification for constraint recommendations
BACKBONE_HEAVY_ATOMS = {"N", "CA", "C", "O"}
BACKBONE_ALL_ATOMS = {"N", "CA", "C", "O", "H", "HA", "1H", "2H", "3H", "HA2", "HA3", "OXT"}
HETEROATOM_ELEMENTS = {"N", "O", "S"}  # Non-carbon, non-hydrogen

# Polar hydrogens by residue (H atoms bonded to N, O, S)
RESIDUE_POLAR_HYDROGENS = {
    "ALA": {"H"},
    "ARG": {"H", "HE", "1HH1", "2HH1", "1HH2", "2HH2", "HH11", "HH12", "HH21", "HH22"},
    "ASN": {"H", "1HD2", "2HD2", "HD21", "HD22"},
    "ASP": {"H"},
    "CYS": {"H", "HG"},
    "GLN": {"H", "1HE2", "2HE2", "HE21", "HE22"},
    "GLU": {"H"},
    "GLY": {"H"},
    "HIS": {"H", "HD1", "HE2"},  # Can have HD1 or HE2 depending on tautomer
    "ILE": {"H"},
    "LEU": {"H"},
    "LYS": {"H", "HZ1", "HZ2", "HZ3", "1HZ", "2HZ", "3HZ"},
    "MET": {"H"},
    "PHE": {"H"},
    "PRO": set(),  # No backbone H
    "SER": {"H", "HG"},
    "THR": {"H", "HG1"},
    "TRP": {"H", "HE1"},
    "TYR": {"H", "HH"},
    "VAL": {"H"},
}

# Acid/base residues and their catalytic atoms (can abstract or donate protons)
ACID_BASE_RESIDUES = {"ASP", "GLU", "HIS", "LYS", "ARG", "CYS", "TYR", "SER"}
ACID_BASE_ATOMS_BY_RESIDUE = {
    "ASP": {"OD1", "OD2"},    # Can act as base (abstract H) or acid (donate H)
    "GLU": {"OE1", "OE2"},    # Can act as base or acid
    "HIS": {"ND1", "NE2"},    # Can act as acid or base
    "LYS": {"NZ"},            # Can act as acid (donate H)
    "ARG": {"NE", "NH1", "NH2"},  # Can donate H
    "CYS": {"SG"},            # Thiol can be deprotonated
    "TYR": {"OH"},            # Phenol can be deprotonated
    "SER": {"OG"},            # Hydroxyl (less common)
}

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════════════════════════
# ALIGNMENT FUNCTIONS (Kabsch algorithm)
# ══════════════════════════════════════════════════════════════════════════════════════════════════
def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute optimal rotation matrix (R), translation (t), and RMSD to align P onto Q.

    Args:
        P: Nx3 coordinates to be transformed (mobile)
        Q: Nx3 target coordinates (fixed)

    Returns:
        R: 3x3 rotation matrix
        t: 3 translation vector (applied after rotation)
        rmsd: RMSD after alignment

    Transformed P' = (P - centroid_P) @ R + centroid_Q
    """
    centroid_P, centroid_Q = P.mean(axis=0), Q.mean(axis=0)
    P_centered, Q_centered = P - centroid_P, Q - centroid_Q
    H = P_centered.T @ Q_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    P_aligned = P_centered @ R + centroid_Q
    rmsd = np.sqrt(((P_aligned - Q) ** 2).sum(axis=1).mean())
    t = centroid_Q - centroid_P @ R
    return R, t, rmsd


def apply_transform(coords: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply rotation R and translation t to coordinates."""
    return coords @ R + t

# ══════════════════════════════════════════════════════════════════════════════════════════════════
# INTERACTION DETECTION
# ══════════════════════════════════════════════════════════════════════════════════════════════════
# H-bond geometry thresholds (angles in degrees)
HBOND_DONOR_ANGLE_MIN = 120.0    # D-H...A angle must be >= this (linearity of donation)
HBOND_ACCEPTOR_ANGLE_MIN = 100.0 # H...A-B angle must be >= this (lone pair orientation)
H_BOND_DIST_MAX = 1.4            # Max distance to consider H bonded to heavy atom
HEAVY_BOND_DIST_MAX = 1.8        # Max distance for heavy atom bonds


def distance(a1: Dict, a2: Dict) -> float:
    """Euclidean distance between two atoms."""
    return np.sqrt((a1["x"] - a2["x"])**2 + (a1["y"] - a2["y"])**2 + (a1["z"] - a2["z"])**2)


def get_coords(atom: Dict) -> np.ndarray:
    """Extract xyz coordinates from atom dict."""
    return np.array([atom["x"], atom["y"], atom["z"]])


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculate angle (in degrees) at p2 between vectors p2->p1 and p2->p3."""
    v1, v2 = p1 - p2, p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


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


def is_polar_atom(elem: str) -> bool:
    """Check if element is a polar heavy atom (N, O, S) that can participate in H-bonds."""
    return elem.upper() in {"N", "O", "S"}


def is_heteroatom(elem: str) -> bool:
    """Check if element is a non-carbon heteroatom (N, O, S, P, metals, halogens)."""
    return elem.upper() not in {"C", "H", ""}


def check_hbond_geometry(donor_atom: Dict, h_atom: Dict, acceptor_atom: Dict, acceptor_base: Dict) -> bool:
    """Check if H-bond geometry is valid (angles within thresholds).

    Args:
        donor_atom: Heavy atom the H is bonded to (D in D-H...A)
        h_atom: The hydrogen atom
        acceptor_atom: The acceptor atom (A in D-H...A)
        acceptor_base: Heavy atom bonded to acceptor (B in H...A-B)

    Returns:
        True if both D-H...A and H...A-B angles are within acceptable ranges
    """
    if donor_atom is None or acceptor_base is None:
        return False
    d_coords, h_coords = get_coords(donor_atom), get_coords(h_atom)
    a_coords, b_coords = get_coords(acceptor_atom), get_coords(acceptor_base)
    # D-H...A angle (at H): should be >= 120° for good linearity
    angle_dha = calculate_angle(d_coords, h_coords, a_coords)
    if angle_dha < HBOND_DONOR_ANGLE_MIN:
        return False
    # H...A-B angle (at A): should be >= 100° for lone pair access
    angle_hab = calculate_angle(h_coords, a_coords, b_coords)
    if angle_hab < HBOND_ACCEPTOR_ANGLE_MIN:
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


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# PI-STACKING GEOMETRY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════════════════════════
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


def detect_pi_ligand_interaction(ring_atoms: List[Dict], ligand_atoms: List[Dict]) -> Optional[float]:
    """Detect pi-ligand interaction based on carbon contacts to ring.

    Args:
        ring_atoms: Atoms of the aromatic ring
        ligand_atoms: All ligand atoms

    Returns:
        Minimum distance if interaction detected, None otherwise
    """
    min_dist = float("inf")
    ring_centroid, _ = compute_ring_geometry(ring_atoms)

    # Check for carbon atoms in ligand within range of ring centroid
    for lig_atom in ligand_atoms:
        elem = lig_atom.get("element", lig_atom["atom_name"][0])
        if elem.upper() != "C":
            continue
        lig_coords = get_coords(lig_atom)
        d = np.linalg.norm(lig_coords - ring_centroid)
        if d <= HYDROPHOBIC_DIST_MAX and d < min_dist:
            min_dist = d

    return min_dist if min_dist < float("inf") else None


def detect_pi_interactions_for_residue(catres_atoms: List[Dict], catres_resname: str,
                                        target_atoms: List[Dict], target_resname: Optional[str],
                                        target_type: str) -> List[Dict]:
    """Detect pi interactions between a catalytic residue and target.

    Args:
        catres_atoms: Atoms of the catalytic residue
        catres_resname: Residue name of catres
        target_atoms: Atoms of target (ligand or another residue)
        target_resname: Residue name of target (None for ligand)
        target_type: "ligand" or "catres_N"

    Returns:
        List of pi interaction dicts
    """
    if catres_resname not in AROMATIC_RESIDUES:
        return []

    catres_rings = get_ring_atoms(catres_atoms, catres_resname)
    if not catres_rings:
        return []

    interactions = []

    if target_type == "ligand":
        # Pi-ligand interaction
        for ring in catres_rings:
            min_dist = detect_pi_ligand_interaction(ring, target_atoms)
            if min_dist is not None:
                interactions.append({
                    "type": "pi_lig_interaction",
                    "target_type": target_type,
                    "min_distance": round(min_dist, 2),
                    "is_backbone": False
                })
                break  # One pi_lig_interaction per target is enough

    elif target_resname in AROMATIC_RESIDUES:
        # Pi-pi interaction between aromatic residues
        target_rings = get_ring_atoms(target_atoms, target_resname)
        best_interaction = None
        best_dist = float("inf")

        for cat_ring in catres_rings:
            for tgt_ring in target_rings:
                pi_type = classify_pi_pi_interaction(cat_ring, tgt_ring)
                if pi_type:
                    # Compute centroid distance for reporting
                    c1, _ = compute_ring_geometry(cat_ring)
                    c2, _ = compute_ring_geometry(tgt_ring)
                    d = np.linalg.norm(c2 - c1)
                    if d < best_dist:
                        best_dist = d
                        best_interaction = {
                            "type": pi_type,
                            "target_type": target_type,
                            "centroid_distance": round(d, 2),
                            "is_backbone": False
                        }

        if best_interaction:
            interactions.append(best_interaction)

    return interactions


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# CONSTRAINT RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════════════════════════
def get_recommended_constraint_atoms(residue_atoms: List[Dict], resname: str, interactions: List[Dict],
                                      backbone_important: bool, sidechain_important: bool,
                                      exclude_bb_only_hbond: bool = False,
                                      flex_res_constrain_all_sc: bool = False) -> List[str]:
    """Get list of atom names recommended for coordinate constraints.

    Args:
        residue_atoms: All atoms of the residue
        resname: Three-letter residue code
        interactions: List of detected interactions for this residue
        backbone_important: Whether backbone was flagged as important
        sidechain_important: Whether sidechain was flagged as important
        exclude_bb_only_hbond: If True, don't include backbone atoms when backbone is only
                               important due to backbone_hbond_* interactions
        flex_res_constrain_all_sc: If True, constrain entire sidechain for ARG/LYS instead of just tip atoms

    Returns:
        Sorted list of atom names to constrain
    """
    atoms_to_constrain = set()

    # Check if backbone is only important due to backbone-to-backbone H-bonds
    bb_interactions = [i for i in interactions if i.get("is_backbone", False)]
    bb_only_hbond = all(i["type"].startswith("backbone_hbond_") for i in bb_interactions) if bb_interactions else False

    # Determine if we should include backbone atoms
    include_backbone = backbone_important
    if exclude_bb_only_hbond and bb_only_hbond and backbone_important:
        # Backbone was only important due to backbone_hbond_*, skip backbone atoms for constraints
        include_backbone = False

    # Get polar hydrogens for this residue
    polar_h = RESIDUE_POLAR_HYDROGENS.get(resname, {"H"})

    # Check if this is an extremely flexible residue with sidechain-only importance
    is_flex_res_sc_only = (resname in EXTREMELY_FLEXIBLE_RESIDUES and
                           sidechain_important and not backbone_important)

    # For flexible residues with sidechain-only importance, use tip atoms unless flag is set
    if is_flex_res_sc_only and not flex_res_constrain_all_sc:
        tip_heavy = FLEXIBLE_RESIDUE_TIP_HEAVY.get(resname, set())
        tip_polar_h = FLEXIBLE_RESIDUE_TIP_POLAR_H.get(resname, set())

        for atom in residue_atoms:
            atom_name = atom["atom_name"]
            is_bb = is_backbone_atom(atom_name)

            if is_bb:
                if include_backbone:
                    elem = atom.get("element", atom_name[0]).upper()
                    if elem in HETEROATOM_ELEMENTS or atom_name in BACKBONE_HEAVY_ATOMS:
                        atoms_to_constrain.add(atom_name)
                    if elem == "H" and atom_name in polar_h:
                        atoms_to_constrain.add(atom_name)
            else:
                # Only constrain tip heavy atoms and their polar hydrogens
                if atom_name in tip_heavy:
                    atoms_to_constrain.add(atom_name)
                elif atom_name in tip_polar_h:
                    atoms_to_constrain.add(atom_name)

        return sorted(atoms_to_constrain)

    # Standard constraint logic for non-flexible residues or when full SC constraint is requested
    for atom in residue_atoms:
        atom_name = atom["atom_name"]
        elem = atom.get("element", atom_name[0]).upper()
        is_bb = is_backbone_atom(atom_name)

        if is_bb:
            if include_backbone:
                # Include all backbone heavy atoms
                if elem in HETEROATOM_ELEMENTS or atom_name in BACKBONE_HEAVY_ATOMS:
                    atoms_to_constrain.add(atom_name)
                # Include backbone polar H
                if elem == "H" and atom_name in polar_h:
                    atoms_to_constrain.add(atom_name)
        else:
            if sidechain_important:
                # Include all sidechain heavy atoms (C, N, O, S) to preserve geometry
                if elem not in {"H", ""}:
                    atoms_to_constrain.add(atom_name)
                # Include sidechain polar H
                if elem == "H" and atom_name in polar_h:
                    atoms_to_constrain.add(atom_name)

    return sorted(atoms_to_constrain)


def generate_constraint_recommendations(interaction_analysis: Dict[int, Dict],
                                         ref_atoms: List[Dict],
                                         exclude_bb_only_hbond: bool = False,
                                         flex_res_constrain_all_sc: bool = False) -> Dict[str, Dict]:
    """Generate constraint recommendations for each analyzed residue.

    Args:
        interaction_analysis: Output from analyze_catres_interactions
        ref_atoms: All atoms from ref_pdb
        exclude_bb_only_hbond: If True, exclude backbone atoms when only backbone_hbond_* present
        flex_res_constrain_all_sc: If True, constrain entire sidechain for ARG/LYS instead of just tip atoms

    Returns:
        Dict mapping block_index (str) -> {chain, resno, resname, constrain_atoms}
    """
    recommendations = {}

    for block_idx, analysis in interaction_analysis.items():
        chain, resno, resname = analysis["chain"], analysis["resno"], analysis["resname"]
        residue_atoms = get_residue_atoms(ref_atoms, chain, resno)

        constrain_atoms = get_recommended_constraint_atoms(
            residue_atoms, resname,
            analysis["interactions"],
            analysis["backbone_important"],
            analysis["sidechain_important"],
            exclude_bb_only_hbond=exclude_bb_only_hbond,
            flex_res_constrain_all_sc=flex_res_constrain_all_sc
        )

        # Determine if backbone is only important due to backbone-to-backbone H-bonds
        # True if: backbone has interactions AND all of them are backbone_hbond_* types
        # False if: backbone has no interactions OR has interactions to sidechain/ligand
        bb_interactions = [i for i in analysis["interactions"] if i.get("is_backbone", False)]
        if bb_interactions:
            bb_only_bb_hbond = all(i["type"].startswith("backbone_hbond_") for i in bb_interactions)
        else:
            bb_only_bb_hbond = False

        recommendations[str(block_idx)] = {
            "chain": chain,
            "resno": resno,
            "resname": resname,
            "constrain_atoms": constrain_atoms,
            "backbone_important": analysis["backbone_important"],
            "sidechain_important": analysis["sidechain_important"],
            "importance": analysis["importance"],
            "backbone_important_only_for_BB_BB_hbond": bb_only_bb_hbond
        }

    return recommendations


def detect_interactions(catres_atoms: List[Dict], target_atoms: List[Dict],
                         target_type: str, catres_resname: str, target_resname: str = None) -> List[Dict]:
    """Detect interactions between a catalytic residue and target atoms (ligand or other catres).

    Args:
        catres_atoms: All atoms of the catalytic residue
        target_atoms: Atoms of ligand or another catres
        target_type: "ligand", "catres_N", or "self_sidechain"
        catres_resname: Three-letter code of the catalytic residue
        target_resname: Three-letter code of target residue (None for ligand, catres_resname for self_sidechain)

    Returns:
        List of interaction dicts with keys: type, catres_atom, target_atom, distance, is_backbone
    """
    interactions = []
    is_charged = catres_resname in CHARGED_RESIDUES
    is_aromatic = catres_resname in AROMATIC_RESIDUES
    is_hydrophobic = catres_resname in HYDROPHOBIC_RESIDUES
    has_polar_sidechain = catres_resname not in NONPOLAR_SIDECHAIN_RESIDUES
    is_acid_base = catres_resname in ACID_BASE_RESIDUES
    acid_base_atoms = ACID_BASE_ATOMS_BY_RESIDUE.get(catres_resname, set())

    # Track acid_base_mod pairs to avoid double-counting as H-bonds
    acid_base_pairs = set()  # (catres_atom, target_atom) pairs that are acid_base_mod
    # Track metal_coord pairs to avoid double-counting as post_translational_mod
    metal_coord_pairs = set()  # (catres_atom, target_atom) pairs that are metal_coord

    # First pass: detect acid_base_mod (H within bonding distance of acid/base sidechain atom)
    if is_acid_base:
        for ca in catres_atoms:
            ca_name = ca["atom_name"]
            if ca_name not in acid_base_atoms:
                continue
            for ta in target_atoms:
                ta_name, ta_elem = ta["atom_name"], ta.get("element", ta["atom_name"][0])
                if ta_elem.upper() != "H":
                    continue
                d = distance(ca, ta)
                if d <= ACID_BASE_DIST_MAX:
                    interactions.append({"type": "acid_base_mod", "catres_atom": ca_name, "target_atom": ta_name,
                                         "distance": round(d, 2), "is_backbone": False, "target_type": target_type})
                    acid_base_pairs.add((ca_name, ta_name))

    for ca in catres_atoms:
        ca_name, ca_elem = ca["atom_name"], ca.get("element", ca["atom_name"][0])
        ca_is_bb = is_backbone_atom(ca_name)
        ca_is_polar_sc = ca_name in POLAR_SIDECHAIN_ATOMS
        ca_is_bb_acceptor = ca_name in BACKBONE_HBOND_ACCEPTORS

        for ta in target_atoms:
            ta_name, ta_elem = ta["atom_name"], ta.get("element", ta["atom_name"][0])
            d = distance(ca, ta)

            # ── H-BOND: Catres H donating to target acceptor ──
            # Catres H (backbone or sidechain) -> Target acceptor (N/O/S)
            if ca_elem.upper() == "H" and is_polar_atom(ta_elem) and d <= HBOND_DIST_MAX:
                is_valid, donor_atom = is_valid_hbond_donor(ca, catres_atoms)
                if is_valid and donor_atom:
                    acceptor_base = find_bonded_heavy_to_acceptor(ta, target_atoms)
                    if acceptor_base and check_hbond_geometry(donor_atom, ca, ta, acceptor_base):
                        # Check if target is also backbone (backbone-to-backbone H-bond)
                        ta_is_bb = is_backbone_atom(ta_name)
                        if ca_is_bb and ta_is_bb:
                            hbond_type = "backbone_hbond_bb_donor"
                        elif ca_is_bb:
                            hbond_type = "hbond_bb_donor"
                        else:
                            hbond_type = "hbond_sc_donor"
                        interactions.append({"type": hbond_type, "catres_atom": ca_name, "target_atom": ta_name,
                                             "distance": round(d, 2), "is_backbone": ca_is_bb, "target_type": target_type})

            # ── H-BOND: Catres acceptor receiving from target H ──
            # Target H -> Catres acceptor (backbone O or sidechain N/O/S)
            # Skip if this pair is already an acid_base_mod
            if ta_elem.upper() == "H" and d <= HBOND_DIST_MAX:
                if (ca_name, ta_name) in acid_base_pairs:
                    continue  # Don't double-count as H-bond
                ca_is_acceptor = ca_is_bb_acceptor or (has_polar_sidechain and ca_is_polar_sc)
                if ca_is_acceptor:
                    is_valid, donor_atom = is_valid_hbond_donor(ta, target_atoms)
                    if is_valid and donor_atom:
                        acceptor_base = find_bonded_heavy_to_acceptor(ca, catres_atoms)
                        if acceptor_base and check_hbond_geometry(donor_atom, ta, ca, acceptor_base):
                            # Check if donor (what H is attached to) is also backbone
                            donor_is_bb = is_backbone_atom(donor_atom["atom_name"])
                            if ca_is_bb and donor_is_bb:
                                hbond_type = "backbone_hbond_bb_acceptor"
                            elif ca_is_bb:
                                hbond_type = "hbond_bb_acceptor"
                            else:
                                hbond_type = "hbond_sc_acceptor"
                            interactions.append({"type": hbond_type, "catres_atom": ca_name, "target_atom": ta_name,
                                                 "distance": round(d, 2), "is_backbone": ca_is_bb, "target_type": target_type})

            # Metal coordination (polar atoms can coordinate metals)
            ca_is_polar = ca_is_polar_sc or ca_name in BACKBONE_HBOND_DONORS or ca_name in BACKBONE_HBOND_ACCEPTORS
            if ta_elem.upper() in METAL_ATOMS and ca_is_polar and d <= METAL_COORD_DIST_MAX:
                interactions.append({"type": "metal_coord", "catres_atom": ca_name, "target_atom": ta_name,
                                     "distance": round(d, 2), "is_backbone": ca_is_bb, "target_type": target_type})
                metal_coord_pairs.add((ca_name, ta_name))
            if ca_elem.upper() in METAL_ATOMS and d <= METAL_COORD_DIST_MAX:  # Catres has metal (rare)
                interactions.append({"type": "metal_coord", "catres_atom": ca_name, "target_atom": ta_name,
                                     "distance": round(d, 2), "is_backbone": ca_is_bb, "target_type": target_type})
                metal_coord_pairs.add((ca_name, ta_name))

            # Charged/ionic interaction (only actual charged atoms: NZ for LYS, OD1/OD2 for ASP, etc.)
            if is_charged and not ca_is_bb and is_heteroatom(ta_elem) and d <= CHARGED_DIST_MAX:
                charged_atoms = CHARGED_ATOMS_BY_RESIDUE.get(catres_resname, set())
                if ca_name in charged_atoms:
                    interactions.append({"type": "charged", "catres_atom": ca_name, "target_atom": ta_name,
                                         "distance": round(d, 2), "is_backbone": False, "target_type": target_type})

            # Hydrophobic interaction (non-aromatic hydrophobic residues only)
            # Aromatic pi interactions are handled separately with geometry-based detection
            if is_hydrophobic and not is_aromatic and not ca_is_bb and ca_elem.upper() == "C":
                if ta_elem.upper() == "C" and d <= HYDROPHOBIC_DIST_MAX:
                    interactions.append({"type": "hydrophobic", "catres_atom": ca_name, "target_atom": ta_name,
                                         "distance": round(d, 2), "is_backbone": False, "target_type": target_type})

            # Post-translational modification check (covalent distance between heavy atoms only)
            # Exclude hydrogens - they're close for H-bonding or acid/base, not covalent PTMs
            # Exclude metal_coord pairs - metal coordination is not a PTM
            if d <= COVALENT_DIST_MAX and not ca_is_bb and ca_elem.upper() not in {"H", ""} and ta_elem.upper() not in {"H", ""}:
                if (ca_name, ta_name) not in metal_coord_pairs:
                    interactions.append({"type": "post_translational_mod", "catres_atom": ca_name, "target_atom": ta_name,
                                         "distance": round(d, 2), "is_backbone": False, "target_type": target_type})

    # Add pi interactions (geometry-based detection for aromatic residues)
    pi_interactions = detect_pi_interactions_for_residue(catres_atoms, catres_resname,
                                                          target_atoms, target_resname, target_type)
    interactions.extend(pi_interactions)

    # Deduplicate while preserving order
    # For hydrophobic/charged: consolidate to one entry per target_type (not atom-by-atom)
    # Pi interactions are already consolidated in detect_pi_interactions_for_residue
    seen = set()
    unique = []
    consolidated_seen = set()  # Track (type, target_type) for consolidation
    for i in interactions:
        if i["type"] in ("hydrophobic", "charged"):
            key = (i["type"], i["target_type"])
            if key not in consolidated_seen:
                consolidated_seen.add(key)
                unique.append({"type": i["type"], "target_type": i["target_type"],
                               "min_distance": i["distance"], "is_backbone": False})
            else:  # Update min distance if closer
                for u in unique:
                    if u["type"] == i["type"] and u["target_type"] == i["target_type"]:
                        u["min_distance"] = min(u["min_distance"], i["distance"])
                        break
        elif i["type"] in ("pi_lig_interaction", "pi_pi_stacking", "pi_pi_stacking_displaced", "pi_pi_Tshape"):
            # Pi interactions are already one-per-target, just add directly
            unique.append(i)
        else:
            key = (i["type"], i["catres_atom"], i["target_atom"])
            if key not in seen:
                seen.add(key)
                unique.append(i)
    return unique


def analyze_catres_interactions(ref_atoms: List[Dict], lig_atoms: List[Dict],
                                 catres_entries: List[Dict], catres_subset_blocks: Set[int],
                                 strict_backbone_importance: bool = False
                                 ) -> Dict[int, Dict]:
    """Analyze interactions for each catres_subset residue.

    Args:
        ref_atoms: All atoms from ref_pdb (already aligned to input_pdb coordinate frame)
        lig_atoms: Ligand atoms
        catres_entries: Parsed REMARK 666 entries
        catres_subset_blocks: Set of block indices that are catres (not conserved)
        strict_backbone_importance: If True, backbone-to-backbone H-bonds alone are not enough
                                    to make backbone_important=True (need interaction to sidechain/ligand)

    Returns:
        Dict mapping block_index -> {residue_info, interactions, backbone_important, sidechain_important, importance}
    """
    results = {}
    subset_residues = {e["block_index"]: (e["motif_chain"], e["motif_resno"], e["motif_resname"])
                       for e in catres_entries if e["block_index"] in catres_subset_blocks}

    for block_idx, (chain, resno, resname) in subset_residues.items():
        LOGGER.info(f"  Analyzing block {block_idx}: {resname} {chain}{resno}")
        res_atoms = get_residue_atoms(ref_atoms, chain, resno)
        if not res_atoms:
            LOGGER.warning(f"    No atoms found for {chain}{resno}")
            results[block_idx] = {"chain": chain, "resno": resno, "resname": resname, "interactions": [],
                                  "backbone_important": False, "sidechain_important": True, "importance": "sidechain"}
            continue

        all_interactions = []

        # Check interactions with ligand (target_resname=None since ligand is not a standard residue)
        lig_interactions = detect_interactions(res_atoms, lig_atoms, "ligand", resname, target_resname=None)
        all_interactions.extend(lig_interactions)

        # Check interactions with other catres_subset residues
        for other_block, (other_chain, other_resno, other_resname) in subset_residues.items():
            if other_block == block_idx:
                continue
            other_atoms = get_residue_atoms(ref_atoms, other_chain, other_resno)
            catres_interactions = detect_interactions(res_atoms, other_atoms, f"catres_{other_block}", resname, target_resname=other_resname)
            # Filter: adjacent residues only count if meaningful (exclude backbone-backbone H-bonds)
            if abs(other_resno - resno) == 1 and other_chain == chain:
                allowed_types = {"hbond_sc_donor", "hbond_sc_acceptor", "metal_coord", "charged",
                                 "pi_lig_interaction", "pi_pi_stacking", "pi_pi_stacking_displaced", "pi_pi_Tshape"}
                catres_interactions = [i for i in catres_interactions if i["type"] in allowed_types]
            all_interactions.extend(catres_interactions)

        # Check intra-residue backbone-sidechain interactions (target_resname=resname since it's the same residue)
        bb_atoms = [a for a in res_atoms if is_backbone_atom(a["atom_name"])]
        sc_atoms = [a for a in res_atoms if not is_backbone_atom(a["atom_name"])]
        intra_interactions = detect_interactions(bb_atoms, sc_atoms, "self_sidechain", resname, target_resname=resname)
        all_interactions.extend([{**i, "is_backbone": True} for i in intra_interactions if i["type"].startswith("hbond")])

        # If residue has metal_coord, remove ALL charged interactions (metal coord is more specific/dominant)
        has_metal_coord = any(i["type"] == "metal_coord" for i in all_interactions)
        if has_metal_coord:
            all_interactions = [i for i in all_interactions if i["type"] != "charged"]

        # Determine importance
        bb_interactions = [i for i in all_interactions if i["is_backbone"]]
        sc_interactions = [i for i in all_interactions if not i["is_backbone"]]

        # In strict mode, backbone-to-backbone H-bonds don't count for backbone_important
        if strict_backbone_importance:
            bb_interactions_for_importance = [i for i in bb_interactions
                                               if not i["type"].startswith("backbone_hbond_")]
        else:
            bb_interactions_for_importance = bb_interactions

        backbone_important = len(bb_interactions_for_importance) > 0
        sidechain_important = len(sc_interactions) > 0 or len(all_interactions) == 0  # Default to sidechain if none

        if backbone_important and sidechain_important:
            importance = "both"
        elif backbone_important:
            importance = "backbone"
        else:
            importance = "sidechain"

        results[block_idx] = {"chain": chain, "resno": resno, "resname": resname, "interactions": all_interactions,
                              "backbone_important": backbone_important, "sidechain_important": sidechain_important,
                              "importance": importance}
        LOGGER.info(f"    Found {len(all_interactions)} interactions, importance: {importance}")

    return results

# ══════════════════════════════════════════════════════════════════════════════════════════════════
# COORDINATE TRANSFORMATION
# ══════════════════════════════════════════════════════════════════════════════════════════════════
def transform_catres_coords(input_lines: List[str], input_atoms: List[Dict], ref_atoms: List[Dict],
                             interaction_analysis: Dict[int, Dict], catres_entries: List[Dict],
                             flex_res_move_all_sc: bool = False
                             ) -> List[str]:
    """Transform catres_subset coordinates from ref_pdb geometry onto input_pdb.

    For each catres_subset residue:
    - If importance is "backbone" or "both": replace ALL atom coordinates with ref_pdb
    - If importance is "sidechain": replace only sidechain coordinates with ref_pdb
      - For extremely flexible residues (ARG, LYS): only move tip atoms by default
        (unless flex_res_move_all_sc is True)

    Args:
        input_lines: All lines from input_pdb
        input_atoms: Parsed atoms from input_pdb
        ref_atoms: Parsed atoms from ref_pdb (already in input_pdb coordinate frame)
        interaction_analysis: Output from analyze_catres_interactions
        catres_entries: Parsed REMARK 666 entries
        flex_res_move_all_sc: If True, move entire sidechain for ARG/LYS instead of just tip atoms

    Returns:
        Modified lines with transformed coordinates
    """
    output_lines = input_lines.copy()

    # Build lookup for ref atoms: (chain, resno, atom_name) -> atom
    ref_lookup = {(a["chain"], a["resno"], a["atom_name"]): a for a in ref_atoms}

    for block_idx, analysis in interaction_analysis.items():
        chain, resno, importance = analysis["chain"], analysis["resno"], analysis["importance"]
        resname = analysis["resname"]
        LOGGER.info(f"  Transforming block {block_idx} ({resname} {chain}{resno}): {importance}")

        # Check if this is an extremely flexible residue with sidechain-only importance
        is_flex_res_sc_only = (resname in EXTREMELY_FLEXIBLE_RESIDUES and
                               importance == "sidechain")

        # For flexible residues, get the tip atoms to move
        if is_flex_res_sc_only and not flex_res_move_all_sc:
            tip_heavy = FLEXIBLE_RESIDUE_TIP_HEAVY.get(resname, set())
            tip_hydrogens = FLEXIBLE_RESIDUE_TIP_HYDROGENS.get(resname, set())
            tip_atoms = tip_heavy | tip_hydrogens
            LOGGER.info(f"    Flexible residue: only moving tip atoms ({len(tip_atoms)} atoms)")

        for inp_atom in input_atoms:
            if inp_atom["chain"] != chain or inp_atom["resno"] != resno:
                continue

            atom_name = inp_atom["atom_name"]
            is_bb = is_backbone_atom(atom_name)

            # Decide whether to transform this atom
            should_transform = False
            if importance == "both" or importance == "backbone":
                should_transform = True
            elif importance == "sidechain" and not is_bb:
                if is_flex_res_sc_only and not flex_res_move_all_sc:
                    # Only transform tip atoms for flexible residues
                    should_transform = atom_name in tip_atoms
                else:
                    should_transform = True

            if not should_transform:
                continue

            # Find corresponding ref atom
            ref_key = (chain, resno, atom_name)
            if ref_key not in ref_lookup:
                LOGGER.debug(f"    Atom {atom_name} not found in ref_pdb, skipping")
                continue

            ref_atom = ref_lookup[ref_key]

            # Update coordinates in output line
            line_idx = inp_atom["line_idx"]
            old_line = output_lines[line_idx]
            new_line = old_line[:30] + f"{ref_atom['x']:8.3f}{ref_atom['y']:8.3f}{ref_atom['z']:8.3f}" + old_line[54:]
            output_lines[line_idx] = new_line

    return output_lines

# ══════════════════════════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════════════════════════
def validate_remark_666_consistency(ref_entries: List[Dict], inp_entries: List[Dict]) -> None:
    """Assert that REMARK 666 entries are consistent between ref and input PDBs."""
    if len(ref_entries) != len(inp_entries):
        raise ValueError(f"REMARK 666 count mismatch: ref={len(ref_entries)}, input={len(inp_entries)}")

    for r, i in zip(sorted(ref_entries, key=lambda x: x["block_index"]),
                    sorted(inp_entries, key=lambda x: x["block_index"])):
        if r["block_index"] != i["block_index"]:
            raise ValueError(f"Block index mismatch: ref={r['block_index']}, input={i['block_index']}")
        if r["motif_resname"] != i["motif_resname"]:
            raise ValueError(f"Residue type mismatch at block {r['block_index']}: ref={r['motif_resname']}, input={i['motif_resname']}")
        if r["motif_resno"] != i["motif_resno"]:
            raise ValueError(f"Residue number mismatch at block {r['block_index']}: ref={r['motif_resno']}, input={i['motif_resno']}")
    LOGGER.info(f"  REMARK 666 validation passed: {len(ref_entries)} consistent entries")


def identify_ligand(entries: List[Dict]) -> Tuple[str, str, int]:
    """Identify ligand from REMARK 666 TEMPLATE entries (non-protein residue)."""
    for e in entries:
        if "template_resname" in e and e["template_resname"] not in {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU",
                                                                       "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE",
                                                                       "PRO", "SER", "THR", "TRP", "TYR", "VAL"}:
            return e["template_chain"], e["template_resname"], e["template_resno"]
    raise ValueError("Could not identify ligand from REMARK 666 lines")


def main(input_pdb: str, ref_pdb: str, outdir: str, catres_subset: Optional[str] = None,
         outfile_bn: Optional[str] = None, strict_backbone_importance: bool = False,
         exclude_bb_only_hbond_constraints: bool = False,
         flex_res_move_all_sc: bool = False,
         flex_res_constrain_all_sc: bool = False) -> Tuple[str, str, str]:
    """Main pipeline for catalytic residue alignment.

    Args:
        input_pdb: Path to input PDB (structure prediction with ligand aligned in)
        ref_pdb: Path to reference PDB (original theozyme/template)
        outdir: Output directory
        catres_subset: Comma-separated block indices for catalytic motif (others become conserved_motif)
        outfile_bn: Optional custom basename for output files
        strict_backbone_importance: If True, backbone-to-backbone H-bonds alone don't count for backbone_important
        exclude_bb_only_hbond_constraints: If True, don't include backbone atoms in constraints when
                                           backbone is only important due to backbone_hbond_* interactions
        flex_res_move_all_sc: If True, move entire sidechain for ARG/LYS instead of just tip atoms
        flex_res_constrain_all_sc: If True, constrain entire sidechain for ARG/LYS instead of just tip atoms
                                   (implies flex_res_move_all_sc=True for consistency)

    Returns:
        Tuple of (output_pdb_path, output_json_path, output_constraint_json_path)
    """
    # If constraining full sidechain, must also move full sidechain for consistency
    if flex_res_constrain_all_sc:
        flex_res_move_all_sc = True
    os.makedirs(outdir, exist_ok=True)
    basename = outfile_bn or os.path.splitext(os.path.basename(input_pdb))[0]
    out_pdb = os.path.join(outdir, f"{basename}_aligned.pdb")
    out_json = os.path.join(outdir, f"{basename}_interactions.json")
    out_cst_json = os.path.join(outdir, f"{basename}_recommended_atom_cst.json")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Read PDB files and parse REMARK 666
    # ─────────────────────────────────────────────────────────────────────────
    LOGGER.info("Step 1: Reading PDB files and parsing REMARK 666 lines")
    ref_lines, ref_atoms = read_pdb_atoms(ref_pdb)
    inp_lines, inp_atoms = read_pdb_atoms(input_pdb)

    ref_r666 = parse_remark_666(ref_lines)
    inp_r666 = parse_remark_666(inp_lines)
    LOGGER.info(f"  ref_pdb: {len(ref_atoms)} atoms, {len(ref_r666)} REMARK 666 entries")
    LOGGER.info(f"  input_pdb: {len(inp_atoms)} atoms, {len(inp_r666)} REMARK 666 entries")

    validate_remark_666_consistency(ref_r666, inp_r666)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Parse catres_subset and classify residues
    # ─────────────────────────────────────────────────────────────────────────
    LOGGER.info("Step 2: Parsing catres_subset and classifying residues")
    max_block = max(e["block_index"] for e in ref_r666)
    subset_blocks = set(parse_catres_subset(catres_subset, max_block))
    conserved_blocks = set(range(1, max_block + 1)) - subset_blocks

    LOGGER.info(f"  catres_subset (catalytic_motif): {sorted(subset_blocks)}")
    LOGGER.info(f"  conserved_motif: {sorted(conserved_blocks)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Align ref_pdb to input_pdb via ligand
    # ─────────────────────────────────────────────────────────────────────────
    LOGGER.info("Step 3: Aligning ref_pdb to input_pdb via ligand coordinates")
    lig_chain, lig_name, lig_resno = identify_ligand(ref_r666)
    LOGGER.info(f"  Identified ligand: {lig_name} {lig_chain}{lig_resno}")

    ref_lig_atoms = get_ligand_atoms(ref_atoms, lig_chain, lig_resno)
    inp_lig_atoms = get_ligand_atoms(inp_atoms, lig_chain, lig_resno)

    if len(ref_lig_atoms) != len(inp_lig_atoms):
        raise ValueError(f"Ligand atom count mismatch: ref={len(ref_lig_atoms)}, input={len(inp_lig_atoms)}")

    # Sort by atom name for consistent ordering
    ref_lig_atoms = sorted(ref_lig_atoms, key=lambda a: a["atom_name"])
    inp_lig_atoms = sorted(inp_lig_atoms, key=lambda a: a["atom_name"])

    ref_lig_coords = atoms_to_coords(ref_lig_atoms)
    inp_lig_coords = atoms_to_coords(inp_lig_atoms)

    R, t, rmsd = kabsch_rotation(ref_lig_coords, inp_lig_coords)
    LOGGER.info(f"  Ligand alignment RMSD: {rmsd:.4f} A (expected ~0)")

    if rmsd > 0.1:
        LOGGER.warning(f"  WARNING: Ligand RMSD ({rmsd:.4f}) is higher than expected. Check ligand identity.")

    # Transform all ref atoms to input coordinate frame
    for atom in ref_atoms:
        old_coords = np.array([atom["x"], atom["y"], atom["z"]])
        new_coords = apply_transform(old_coords.reshape(1, -1), R, t).flatten()
        atom["x"], atom["y"], atom["z"] = new_coords[0], new_coords[1], new_coords[2]

    # Also update lig atoms for interaction analysis
    inp_lig_atoms_aligned = get_ligand_atoms(inp_atoms, lig_chain, lig_resno)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Detect interactions for catres_subset
    # ─────────────────────────────────────────────────────────────────────────
    LOGGER.info("Step 4: Detecting interactions for catres_subset residues")
    interaction_analysis = analyze_catres_interactions(ref_atoms, inp_lig_atoms_aligned, ref_r666, subset_blocks,
                                                        strict_backbone_importance=strict_backbone_importance)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Transform catres coordinates
    # ─────────────────────────────────────────────────────────────────────────
    LOGGER.info("Step 5: Transforming catres_subset coordinates based on interaction importance")
    output_lines = transform_catres_coords(inp_lines, inp_atoms, ref_atoms, interaction_analysis, ref_r666,
                                           flex_res_move_all_sc=flex_res_move_all_sc)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 6: Write outputs
    # ─────────────────────────────────────────────────────────────────────────
    LOGGER.info("Step 6: Writing output files")
    with open(out_pdb, "w") as f:
        f.writelines(output_lines)
    LOGGER.info(f"  Wrote aligned PDB: {out_pdb}")

    # Prepare JSON output
    json_output = {"input_pdb": input_pdb, "ref_pdb": ref_pdb,
                   "ligand": {"chain": lig_chain, "resname": lig_name, "resno": lig_resno},
                   "alignment_rmsd": round(rmsd, 6),
                   "catres_subset_blocks": sorted(subset_blocks), "conserved_motif_blocks": sorted(conserved_blocks),
                   "residue_analysis": {str(k): v for k, v in interaction_analysis.items()}}

    with open(out_json, "w") as f:
        json.dump(json_output, f, indent=2)
    LOGGER.info(f"  Wrote interaction analysis: {out_json}")

    # Generate constraint recommendations
    constraint_recommendations = generate_constraint_recommendations(
        interaction_analysis, ref_atoms,
        exclude_bb_only_hbond=exclude_bb_only_hbond_constraints,
        flex_res_constrain_all_sc=flex_res_constrain_all_sc
    )

    cst_json_output = {
        "input_pdb": input_pdb,
        "ref_pdb": ref_pdb,
        "output_pdb": out_pdb,
        "exclude_bb_only_hbond_constraints": exclude_bb_only_hbond_constraints,
        "flex_res_move_all_sc": flex_res_move_all_sc,
        "flex_res_constrain_all_sc": flex_res_constrain_all_sc,
        "residue_constraints": constraint_recommendations
    }

    with open(out_cst_json, "w") as f:
        json.dump(cst_json_output, f, indent=2)
    LOGGER.info(f"  Wrote constraint recommendations: {out_cst_json}")

    # Summary
    LOGGER.info("=" * 80)
    LOGGER.info("SUMMARY")
    LOGGER.info("=" * 80)
    for block_idx in sorted(interaction_analysis.keys()):
        a = interaction_analysis[block_idx]
        cst = constraint_recommendations[str(block_idx)]
        LOGGER.info(f"  Block {block_idx:2d}: {a['resname']} {a['chain']}{a['resno']:3d} | importance={a['importance']:9s} | interactions={len(a['interactions'])} | constrain={len(cst['constrain_atoms'])} atoms")

    return out_pdb, out_json, out_cst_json


def parse_args():
    p = argparse.ArgumentParser(description="Align catalytic residue coordinates from ref_pdb to input_pdb", formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_pdb", type=str, help="Input PDB file (structure prediction with ligand aligned)")
    p.add_argument("--ref_pdb", type=str, help="Reference PDB file (original theozyme/template)")
    p.add_argument("--outdir", type=str, help="Output directory")
    p.add_argument("--catres_subset", type=str, default=None, help="Comma-separated REMARK 666 block indices for catalytic motif (default: all)")
    p.add_argument("--outfile_bn", type=str, default=None, help="Custom basename for output files (default: input_pdb basename)")
    p.add_argument("--strict_backbone_importance", action="store_true",
                   help="Backbone-to-backbone H-bonds alone don't make backbone_important=True (need interaction to sidechain/ligand)")
    p.add_argument("--exclude_bb_only_hbond_constraints", action="store_true",
                   help="Don't include backbone atoms in constraint recommendations when backbone is only important due to backbone_hbond_* interactions")
    p.add_argument("--flex_res_move_all_sc", action="store_true",
                   help="For ARG/LYS with sidechain-only importance: move entire sidechain instead of just tip atoms (NZ/CE/CD for LYS, NH1/CZ/NH2/NE/CD for ARG)")
    p.add_argument("--flex_res_constrain_all_sc", action="store_true",
                   help="For ARG/LYS with sidechain-only importance: constrain entire sidechain instead of just tip atoms (implies --flex_res_move_all_sc)")
    p.add_argument("--test", action="store_true", help="Run with hardcoded test data")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.test:
        LOGGER.info("Running in TEST mode with hardcoded paths")
        input_pdb, ref_pdb, outdir, catres_subset = TEST_INPUT_PDB, TEST_REF_PDB, TEST_OUTDIR, TEST_CATRES_SUBSET
    else:
        if not all([args.input_pdb, args.ref_pdb, args.outdir]):
            LOGGER.error("Must provide --input_pdb, --ref_pdb, and --outdir (or use --test)")
            sys.exit(1)
        input_pdb, ref_pdb, outdir, catres_subset = args.input_pdb, args.ref_pdb, args.outdir, args.catres_subset

    out_pdb, out_json, out_cst_json = main(input_pdb, ref_pdb, outdir, catres_subset, args.outfile_bn,
                                            strict_backbone_importance=args.strict_backbone_importance,
                                            exclude_bb_only_hbond_constraints=args.exclude_bb_only_hbond_constraints,
                                            flex_res_move_all_sc=args.flex_res_move_all_sc,
                                            flex_res_constrain_all_sc=args.flex_res_constrain_all_sc)
    LOGGER.info(f"Done! Outputs: {out_pdb}, {out_json}, {out_cst_json}")
