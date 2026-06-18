"""
Two-tier contact detection between ligands/metals and protein residues.

Implements a hierarchical contact detection system:
- Metal contacts (2.6 A): Tight coordination with N/O/S
- Primary contacts (3.6 A): Heteroatom (N/O/S) involved
- Secondary contacts (4.2 A): Any heavy atom contacts

Prioritizes heteroatom contacts over carbon-carbon contacts.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set

from fastmpnndesign.config import Contact, CatalyticResidue, CatresCatresContact
from fastmpnndesign.ligand import Ligand, LigandAtom, detect_ligands_from_pdb, detect_metals_from_pdb
from fastmpnndesign.constants import (
    METALS, HETEROATOMS, METAL_CONTACT_CUTOFF, PRIMARY_CONTACT_CUTOFF,
    SECONDARY_CONTACT_CUTOFF, PRIORITY_METAL, PRIORITY_PRIMARY_HETERO,
    PRIORITY_PRIMARY_OTHER, PRIORITY_SECONDARY_HETERO, PRIORITY_SECONDARY_OTHER,
    PRIORITY_CARBON_CARBON, STANDARD_AA, NONSTANDARD_AA,
    # Catres-catres interaction constants
    HBOND_CUTOFF, SALT_BRIDGE_CUTOFF, PI_STACK_CUTOFF, PI_STACK_ANGLE_CUTOFF,
    POSITIVELY_CHARGED_AA, NEGATIVELY_CHARGED_AA, AROMATIC_AA,
    HBOND_DONORS, HBOND_ACCEPTORS, POSITIVE_CHARGE_ATOMS, NEGATIVE_CHARGE_ATOMS,
    AROMATIC_RING_ATOMS,
    PRIORITY_CATRES_HBOND, PRIORITY_CATRES_SALT_BRIDGE, PRIORITY_CATRES_PI_STACK
)
from fastmpnndesign.utils import iter_pdb_atoms, calculate_distance, get_element_from_atom_name
import math
from fastmpnndesign.logging_config import get_logger

logger = get_logger("contact_detection")


@dataclass
class ProteinAtom:
    """Represents a protein atom for contact detection."""
    name: str
    element: str
    x: float
    y: float
    z: float
    chain: str
    resnum: int
    resname: str
    icode: str = ""

    def coords(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def is_heteroatom(self) -> bool:
        """Check if this is a heteroatom (N, O, S)."""
        return self.element.upper() in HETEROATOMS

    @property
    def is_carbon(self) -> bool:
        """Check if this is a carbon atom."""
        return self.element.upper() == 'C'


def get_protein_atoms(
    pdb_path: Path,
    catres_only: bool = False,
    catres_list: Optional[List[CatalyticResidue]] = None
) -> List[ProteinAtom]:
    """
    Extract protein heavy atoms from PDB.

    Args:
        pdb_path: Path to PDB file.
        catres_only: If True, only return atoms from catalytic residues.
        catres_list: List of catalytic residues (required if catres_only=True).

    Returns:
        List of ProteinAtom objects.
    """
    # Build catres lookup
    catres_set: Set[Tuple[str, int]] = set()
    if catres_list:
        for cr in catres_list:
            catres_set.add((cr.chain, cr.resnum))

    atoms = []
    for atom in iter_pdb_atoms(pdb_path):
        # Only ATOM records (protein)
        if atom['record_type'] != 'ATOM':
            continue

        resname = atom['resname']
        if resname not in STANDARD_AA and resname not in NONSTANDARD_AA:
            continue

        # Filter to catres if requested
        if catres_only and (atom['chain'], atom['resnum']) not in catres_set:
            continue

        # Get element
        element = atom.get('element', '')
        if not element:
            element = get_element_from_atom_name(atom['name'])

        # Skip hydrogens
        if element.upper() == 'H':
            continue

        atoms.append(ProteinAtom(
            name=atom['name'],
            element=element.upper(),
            x=atom['x'],
            y=atom['y'],
            z=atom['z'],
            chain=atom['chain'],
            resnum=atom['resnum'],
            resname=resname,
            icode=atom.get('icode', '')
        ))

    return atoms


def detect_contacts(
    ligands: List[Ligand],
    metals: List[Ligand],
    protein_atoms: List[ProteinAtom],
    primary_cutoff: float = PRIMARY_CONTACT_CUTOFF,
    secondary_cutoff: float = SECONDARY_CONTACT_CUTOFF,
    metal_cutoff: float = METAL_CONTACT_CUTOFF
) -> List[Contact]:
    """
    Detect all contacts between ligands/metals and protein atoms.

    Implements two-tier detection:
    - Metal: ≤ metal_cutoff for coordination with N/O/S
    - Primary: ≤ primary_cutoff for close contacts
    - Secondary: ≤ secondary_cutoff for supporting contacts

    Args:
        ligands: List of ligand molecules.
        metals: List of metal ions.
        protein_atoms: List of protein atoms to check.
        primary_cutoff: Primary contact distance cutoff.
        secondary_cutoff: Secondary contact distance cutoff.
        metal_cutoff: Metal coordination distance cutoff.

    Returns:
        List of Contact objects, sorted by priority (highest first).
    """
    contacts = []

    # Process metal contacts first (highest priority)
    for metal in metals:
        for metal_atom in metal.atoms:
            for prot_atom in protein_atoms:
                # Metals only coordinate with N/O/S
                if not prot_atom.is_heteroatom:
                    continue

                dist = calculate_distance(
                    metal_atom.x, metal_atom.y, metal_atom.z,
                    prot_atom.x, prot_atom.y, prot_atom.z
                )

                if dist <= metal_cutoff:
                    contacts.append(Contact(
                        ligand_chain=metal.chain,
                        ligand_resnum=metal.resnum,
                        ligand_resname=metal.resname,
                        ligand_atom=metal_atom.name,
                        protein_chain=prot_atom.chain,
                        protein_resnum=prot_atom.resnum,
                        protein_resname=prot_atom.resname,
                        protein_atom=prot_atom.name,
                        distance=dist,
                        contact_type='metal',
                        priority=PRIORITY_METAL,
                        is_heteroatom_contact=True
                    ))

    # Process ligand contacts
    for ligand in ligands:
        for lig_atom in ligand.heavy_atoms:
            lig_is_hetero = lig_atom.element.upper() in HETEROATOMS

            for prot_atom in protein_atoms:
                dist = calculate_distance(
                    lig_atom.x, lig_atom.y, lig_atom.z,
                    prot_atom.x, prot_atom.y, prot_atom.z
                )

                # Skip if outside secondary cutoff
                if dist > secondary_cutoff:
                    continue

                # Determine contact type and priority
                is_hetero = lig_is_hetero or prot_atom.is_heteroatom
                is_carbon_carbon = (lig_atom.element.upper() == 'C' and
                                    prot_atom.is_carbon)

                if dist <= primary_cutoff:
                    contact_type = 'primary'
                    if is_hetero:
                        priority = PRIORITY_PRIMARY_HETERO
                    elif is_carbon_carbon:
                        priority = PRIORITY_CARBON_CARBON
                    else:
                        priority = PRIORITY_PRIMARY_OTHER
                else:
                    contact_type = 'secondary'
                    if is_hetero:
                        priority = PRIORITY_SECONDARY_HETERO
                    elif is_carbon_carbon:
                        priority = PRIORITY_CARBON_CARBON
                    else:
                        priority = PRIORITY_SECONDARY_OTHER

                contacts.append(Contact(
                    ligand_chain=ligand.chain,
                    ligand_resnum=ligand.resnum,
                    ligand_resname=ligand.resname,
                    ligand_atom=lig_atom.name,
                    protein_chain=prot_atom.chain,
                    protein_resnum=prot_atom.resnum,
                    protein_resname=prot_atom.resname,
                    protein_atom=prot_atom.name,
                    distance=dist,
                    contact_type=contact_type,
                    priority=priority,
                    is_heteroatom_contact=is_hetero
                ))

    # Sort by priority (highest first), then by distance (closest first)
    contacts.sort(key=lambda c: (-c.priority, c.distance))

    return contacts


def detect_contacts_from_pdb(
    pdb_path: Path,
    catres_list: Optional[List[CatalyticResidue]] = None,
    primary_cutoff: float = PRIMARY_CONTACT_CUTOFF,
    secondary_cutoff: float = SECONDARY_CONTACT_CUTOFF,
    metal_cutoff: float = METAL_CONTACT_CUTOFF,
    catres_only: bool = True
) -> List[Contact]:
    """
    Detect contacts between ligands/metals and protein in a PDB file.

    Args:
        pdb_path: Path to PDB file.
        catres_list: List of catalytic residues. If provided and catres_only=True,
                    only contacts to these residues are detected.
        primary_cutoff: Primary contact distance cutoff.
        secondary_cutoff: Secondary contact distance cutoff.
        metal_cutoff: Metal coordination distance cutoff.
        catres_only: If True, only detect contacts to catalytic residues.

    Returns:
        List of Contact objects.
    """
    pdb_path = Path(pdb_path)

    # Detect ligands and metals
    ligands = detect_ligands_from_pdb(pdb_path)
    metals = detect_metals_from_pdb(pdb_path)

    # Get protein atoms
    protein_atoms = get_protein_atoms(
        pdb_path,
        catres_only=catres_only,
        catres_list=catres_list
    )

    # Detect contacts
    contacts = detect_contacts(
        ligands, metals, protein_atoms,
        primary_cutoff=primary_cutoff,
        secondary_cutoff=secondary_cutoff,
        metal_cutoff=metal_cutoff
    )

    # Log summary
    metal_contacts = [c for c in contacts if c.contact_type == 'metal']
    primary_contacts = [c for c in contacts if c.contact_type == 'primary']
    secondary_contacts = [c for c in contacts if c.contact_type == 'secondary']

    logger.info(
        f"Detected {len(contacts)} contacts: "
        f"{len(metal_contacts)} metal, {len(primary_contacts)} primary, "
        f"{len(secondary_contacts)} secondary"
    )

    # Log top contacts
    if contacts:
        logger.debug("Top 10 contacts by priority:")
        for c in contacts[:10]:
            logger.debug(
                f"  {c.contact_type:9s} {c.ligand_resname:3s} {c.ligand_atom:4s} - "
                f"{c.protein_chain}{c.protein_resnum} {c.protein_resname:3s} "
                f"{c.protein_atom:4s} : {c.distance:.2f} A (priority {c.priority})"
            )

    return contacts


def get_unique_residues_from_contacts(contacts: List[Contact]) -> List[Tuple[str, int, str]]:
    """
    Extract unique protein residues involved in contacts.

    Returns:
        List of (chain, resnum, resname) tuples.
    """
    residues = set()
    for c in contacts:
        residues.add((c.protein_chain, c.protein_resnum, c.protein_resname))
    return sorted(list(residues), key=lambda x: (x[0], x[1]))


def filter_contacts_by_residue(
    contacts: List[Contact],
    chain: str,
    resnum: int
) -> List[Contact]:
    """Filter contacts to those involving a specific protein residue."""
    return [
        c for c in contacts
        if c.protein_chain == chain and c.protein_resnum == resnum
    ]


def get_best_contact_per_residue(contacts: List[Contact]) -> Dict[Tuple[str, int], Contact]:
    """
    Get the best (highest priority, shortest distance) contact for each residue.

    Returns:
        Dictionary mapping (chain, resnum) to best Contact.
    """
    best = {}
    for c in contacts:
        key = (c.protein_chain, c.protein_resnum)
        if key not in best or (
            c.priority > best[key].priority or
            (c.priority == best[key].priority and c.distance < best[key].distance)
        ):
            best[key] = c
    return best


def summarize_contacts(contacts: List[Contact]) -> Dict[str, Any]:
    """
    Generate a summary of detected contacts.

    Returns:
        Dictionary with contact statistics and categorization.
    """
    if not contacts:
        return {
            'total': 0,
            'by_type': {},
            'by_ligand': {},
            'unique_residues': 0
        }

    by_type = {}
    by_ligand = {}

    for c in contacts:
        # Count by type
        if c.contact_type not in by_type:
            by_type[c.contact_type] = {'count': 0, 'heteroatom': 0, 'carbon': 0}
        by_type[c.contact_type]['count'] += 1
        if c.is_heteroatom_contact:
            by_type[c.contact_type]['heteroatom'] += 1
        else:
            by_type[c.contact_type]['carbon'] += 1

        # Count by ligand
        lig_key = f"{c.ligand_chain}_{c.ligand_resname}_{c.ligand_resnum}"
        if lig_key not in by_ligand:
            by_ligand[lig_key] = 0
        by_ligand[lig_key] += 1

    unique_residues = len(get_unique_residues_from_contacts(contacts))

    return {
        'total': len(contacts),
        'by_type': by_type,
        'by_ligand': by_ligand,
        'unique_residues': unique_residues
    }


# ============================================================================
# Catres-Catres Interaction Detection
# ============================================================================

def _calculate_ring_centroid(
    atoms: List[ProteinAtom],
    ring_atom_names: Set[str]
) -> Optional[Tuple[float, float, float]]:
    """
    Calculate the centroid of aromatic ring atoms.

    Args:
        atoms: List of atoms from a residue.
        ring_atom_names: Set of atom names that form the ring.

    Returns:
        (x, y, z) centroid coordinates, or None if insufficient atoms found.
    """
    ring_atoms = [a for a in atoms if a.name in ring_atom_names]
    if len(ring_atoms) < 3:
        return None

    x = sum(a.x for a in ring_atoms) / len(ring_atoms)
    y = sum(a.y for a in ring_atoms) / len(ring_atoms)
    z = sum(a.z for a in ring_atoms) / len(ring_atoms)
    return (x, y, z)


def _calculate_ring_normal(
    atoms: List[ProteinAtom],
    ring_atom_names: Set[str]
) -> Optional[Tuple[float, float, float]]:
    """
    Calculate the normal vector to the aromatic ring plane.

    Uses the first three ring atoms to define the plane.

    Args:
        atoms: List of atoms from a residue.
        ring_atom_names: Set of atom names that form the ring.

    Returns:
        (nx, ny, nz) normalized normal vector, or None if insufficient atoms.
    """
    ring_atoms = [a for a in atoms if a.name in ring_atom_names]
    if len(ring_atoms) < 3:
        return None

    # Get first 3 atoms for plane calculation
    a1, a2, a3 = ring_atoms[0], ring_atoms[1], ring_atoms[2]

    # Vectors in the plane
    v1 = (a2.x - a1.x, a2.y - a1.y, a2.z - a1.z)
    v2 = (a3.x - a1.x, a3.y - a1.y, a3.z - a1.z)

    # Cross product for normal
    nx = v1[1] * v2[2] - v1[2] * v2[1]
    ny = v1[2] * v2[0] - v1[0] * v2[2]
    nz = v1[0] * v2[1] - v1[1] * v2[0]

    # Normalize
    length = math.sqrt(nx*nx + ny*ny + nz*nz)
    if length < 1e-6:
        return None

    return (nx / length, ny / length, nz / length)


def _angle_between_normals(
    n1: Tuple[float, float, float],
    n2: Tuple[float, float, float]
) -> float:
    """
    Calculate angle between two normal vectors in degrees.

    For pi-stacking, we care about both parallel and perpendicular orientations.

    Args:
        n1: First normal vector.
        n2: Second normal vector.

    Returns:
        Angle in degrees (0-90, accounting for parallel/antiparallel).
    """
    # Dot product gives cos(angle)
    dot = n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2]
    # Clamp to avoid numerical issues
    dot = max(-1.0, min(1.0, dot))
    angle = math.degrees(math.acos(abs(dot)))  # abs for parallel/antiparallel equivalence
    return angle


def detect_catres_hydrogen_bonds(
    catres_atoms: Dict[Tuple[str, int], List[ProteinAtom]],
    hbond_cutoff: float = HBOND_CUTOFF
) -> List[CatresCatresContact]:
    """
    Detect hydrogen bonds between catalytic residues.

    Detects N-H...O, O-H...N, N-H...N, O-H...O type hydrogen bonds.

    Args:
        catres_atoms: Dictionary mapping (chain, resnum) to list of atoms.
        hbond_cutoff: Distance cutoff for hydrogen bonds (donor-acceptor).

    Returns:
        List of CatresCatresContact objects for hydrogen bonds.
    """
    contacts = []
    catres_keys = list(catres_atoms.keys())

    for i, key1 in enumerate(catres_keys):
        atoms1 = catres_atoms[key1]
        res1 = atoms1[0] if atoms1 else None
        if not res1:
            continue

        resname1 = res1.resname
        donor_atoms1 = HBOND_DONORS.get(resname1, frozenset())
        acceptor_atoms1 = HBOND_ACCEPTORS.get(resname1, frozenset())

        for key2 in catres_keys[i+1:]:
            atoms2 = catres_atoms[key2]
            res2 = atoms2[0] if atoms2 else None
            if not res2:
                continue

            resname2 = res2.resname
            donor_atoms2 = HBOND_DONORS.get(resname2, frozenset())
            acceptor_atoms2 = HBOND_ACCEPTORS.get(resname2, frozenset())

            # Check res1 donor -> res2 acceptor
            for a1 in atoms1:
                if a1.name not in donor_atoms1:
                    continue
                for a2 in atoms2:
                    if a2.name not in acceptor_atoms2:
                        continue
                    dist = calculate_distance(a1.x, a1.y, a1.z, a2.x, a2.y, a2.z)
                    if dist <= hbond_cutoff:
                        contacts.append(CatresCatresContact(
                            chain1=a1.chain,
                            resnum1=a1.resnum,
                            resname1=a1.resname,
                            atom1=a1.name,
                            chain2=a2.chain,
                            resnum2=a2.resnum,
                            resname2=a2.resname,
                            atom2=a2.name,
                            distance=dist,
                            interaction_type='hbond',
                            priority=PRIORITY_CATRES_HBOND
                        ))

            # Check res2 donor -> res1 acceptor
            for a2 in atoms2:
                if a2.name not in donor_atoms2:
                    continue
                for a1 in atoms1:
                    if a1.name not in acceptor_atoms1:
                        continue
                    dist = calculate_distance(a2.x, a2.y, a2.z, a1.x, a1.y, a1.z)
                    if dist <= hbond_cutoff:
                        contacts.append(CatresCatresContact(
                            chain1=a2.chain,
                            resnum1=a2.resnum,
                            resname1=a2.resname,
                            atom1=a2.name,
                            chain2=a1.chain,
                            resnum2=a1.resnum,
                            resname2=a1.resname,
                            atom2=a1.name,
                            distance=dist,
                            interaction_type='hbond',
                            priority=PRIORITY_CATRES_HBOND
                        ))

    return contacts


def detect_catres_salt_bridges(
    catres_atoms: Dict[Tuple[str, int], List[ProteinAtom]],
    salt_bridge_cutoff: float = SALT_BRIDGE_CUTOFF
) -> List[CatresCatresContact]:
    """
    Detect salt bridges between charged catalytic residues.

    Detects interactions between:
    - Positively charged: LYS (NZ), ARG (NE, NH1, NH2), HIS (ND1, NE2)
    - Negatively charged: ASP (OD1, OD2), GLU (OE1, OE2)

    Args:
        catres_atoms: Dictionary mapping (chain, resnum) to list of atoms.
        salt_bridge_cutoff: Distance cutoff for salt bridges.

    Returns:
        List of CatresCatresContact objects for salt bridges.
    """
    contacts = []

    # Separate positive and negative residues
    positive_residues = []
    negative_residues = []

    for key, atoms in catres_atoms.items():
        if not atoms:
            continue
        resname = atoms[0].resname
        if resname in POSITIVELY_CHARGED_AA:
            positive_residues.append((key, atoms))
        if resname in NEGATIVELY_CHARGED_AA:
            negative_residues.append((key, atoms))

    # Check all positive-negative pairs
    for pos_key, pos_atoms in positive_residues:
        pos_resname = pos_atoms[0].resname
        pos_charge_atoms = POSITIVE_CHARGE_ATOMS.get(pos_resname, frozenset())

        for neg_key, neg_atoms in negative_residues:
            neg_resname = neg_atoms[0].resname
            neg_charge_atoms = NEGATIVE_CHARGE_ATOMS.get(neg_resname, frozenset())

            for p_atom in pos_atoms:
                if p_atom.name not in pos_charge_atoms:
                    continue
                for n_atom in neg_atoms:
                    if n_atom.name not in neg_charge_atoms:
                        continue

                    dist = calculate_distance(
                        p_atom.x, p_atom.y, p_atom.z,
                        n_atom.x, n_atom.y, n_atom.z
                    )
                    if dist <= salt_bridge_cutoff:
                        contacts.append(CatresCatresContact(
                            chain1=p_atom.chain,
                            resnum1=p_atom.resnum,
                            resname1=p_atom.resname,
                            atom1=p_atom.name,
                            chain2=n_atom.chain,
                            resnum2=n_atom.resnum,
                            resname2=n_atom.resname,
                            atom2=n_atom.name,
                            distance=dist,
                            interaction_type='salt_bridge',
                            priority=PRIORITY_CATRES_SALT_BRIDGE
                        ))

    return contacts


def detect_catres_pi_stacking(
    catres_atoms: Dict[Tuple[str, int], List[ProteinAtom]],
    pi_stack_cutoff: float = PI_STACK_CUTOFF,
    angle_cutoff: float = PI_STACK_ANGLE_CUTOFF
) -> List[CatresCatresContact]:
    """
    Detect pi-stacking interactions between aromatic catalytic residues.

    Detects face-to-face and edge-to-face stacking between:
    PHE, TYR, TRP, HIS

    Args:
        catres_atoms: Dictionary mapping (chain, resnum) to list of atoms.
        pi_stack_cutoff: Distance cutoff between ring centroids.
        angle_cutoff: Max angle deviation from parallel (for face-to-face).

    Returns:
        List of CatresCatresContact objects for pi-stacking.
    """
    contacts = []

    # Collect aromatic residues with their ring info
    aromatic_residues = []
    for key, atoms in catres_atoms.items():
        if not atoms:
            continue
        resname = atoms[0].resname
        if resname not in AROMATIC_AA:
            continue

        ring_atoms = AROMATIC_RING_ATOMS.get(resname)
        if not ring_atoms:
            continue

        centroid = _calculate_ring_centroid(atoms, ring_atoms)
        normal = _calculate_ring_normal(atoms, ring_atoms)
        if centroid and normal:
            aromatic_residues.append({
                'key': key,
                'atoms': atoms,
                'resname': resname,
                'centroid': centroid,
                'normal': normal,
                'ring_atoms': ring_atoms
            })

    # Check all aromatic pairs
    for i, res1 in enumerate(aromatic_residues):
        for res2 in aromatic_residues[i+1:]:
            # Calculate centroid distance
            c1, c2 = res1['centroid'], res2['centroid']
            centroid_dist = calculate_distance(c1[0], c1[1], c1[2], c2[0], c2[1], c2[2])

            if centroid_dist > pi_stack_cutoff:
                continue

            # Calculate angle between ring planes
            angle = _angle_between_normals(res1['normal'], res2['normal'])

            # Accept if roughly parallel (face-to-face) or perpendicular (T-shaped)
            # Parallel: angle < angle_cutoff
            # Perpendicular: angle > (90 - angle_cutoff)
            is_parallel = angle < angle_cutoff
            is_perpendicular = angle > (90 - angle_cutoff)

            if is_parallel or is_perpendicular:
                # Find closest ring atom pair for the constraint
                min_dist = float('inf')
                best_pair = None
                atoms1 = res1['atoms']
                atoms2 = res2['atoms']
                ring1 = res1['ring_atoms']
                ring2 = res2['ring_atoms']

                for a1 in atoms1:
                    if a1.name not in ring1:
                        continue
                    for a2 in atoms2:
                        if a2.name not in ring2:
                            continue
                        d = calculate_distance(a1.x, a1.y, a1.z, a2.x, a2.y, a2.z)
                        if d < min_dist:
                            min_dist = d
                            best_pair = (a1, a2)

                if best_pair:
                    a1, a2 = best_pair
                    contacts.append(CatresCatresContact(
                        chain1=a1.chain,
                        resnum1=a1.resnum,
                        resname1=a1.resname,
                        atom1=a1.name,
                        chain2=a2.chain,
                        resnum2=a2.resnum,
                        resname2=a2.resname,
                        atom2=a2.name,
                        distance=min_dist,
                        interaction_type='pi_stack',
                        priority=PRIORITY_CATRES_PI_STACK
                    ))

    return contacts


def detect_catres_catres_contacts(
    pdb_path: Path,
    catres_list: List[CatalyticResidue],
    hbond_cutoff: float = HBOND_CUTOFF,
    salt_bridge_cutoff: float = SALT_BRIDGE_CUTOFF,
    pi_stack_cutoff: float = PI_STACK_CUTOFF,
    pi_stack_angle_cutoff: float = PI_STACK_ANGLE_CUTOFF
) -> List[CatresCatresContact]:
    """
    Detect all interactions between catalytic residues.

    Detects:
    - Hydrogen bonds (N-H...O, O-H...N, etc.) within hbond_cutoff
    - Salt bridges (charged residue pairs) within salt_bridge_cutoff
    - Pi-stacking (aromatic residue pairs) within pi_stack_cutoff

    Args:
        pdb_path: Path to PDB file.
        catres_list: List of catalytic residues to analyze.
        hbond_cutoff: Distance cutoff for hydrogen bonds.
        salt_bridge_cutoff: Distance cutoff for salt bridges.
        pi_stack_cutoff: Distance cutoff for pi-stacking.
        pi_stack_angle_cutoff: Angle cutoff for pi-stacking geometry.

    Returns:
        List of CatresCatresContact objects sorted by priority.
    """
    if not catres_list or len(catres_list) < 2:
        logger.debug("Less than 2 catalytic residues, skipping catres-catres detection")
        return []

    pdb_path = Path(pdb_path)

    # Get atoms for each catalytic residue
    catres_atoms = get_protein_atoms(
        pdb_path,
        catres_only=True,
        catres_list=catres_list
    )

    # Group atoms by residue
    atoms_by_residue: Dict[Tuple[str, int], List[ProteinAtom]] = {}
    for atom in catres_atoms:
        key = (atom.chain, atom.resnum)
        if key not in atoms_by_residue:
            atoms_by_residue[key] = []
        atoms_by_residue[key].append(atom)

    # Detect all interaction types
    all_contacts = []

    # Hydrogen bonds
    hbond_contacts = detect_catres_hydrogen_bonds(atoms_by_residue, hbond_cutoff)
    all_contacts.extend(hbond_contacts)

    # Salt bridges
    salt_bridge_contacts = detect_catres_salt_bridges(atoms_by_residue, salt_bridge_cutoff)
    all_contacts.extend(salt_bridge_contacts)

    # Pi-stacking
    pi_stack_contacts = detect_catres_pi_stacking(
        atoms_by_residue, pi_stack_cutoff, pi_stack_angle_cutoff
    )
    all_contacts.extend(pi_stack_contacts)

    # Sort by priority (highest first), then by distance (closest first)
    all_contacts.sort(key=lambda c: (-c.priority, c.distance))

    # Log summary
    n_hbond = len([c for c in all_contacts if c.interaction_type == 'hbond'])
    n_salt = len([c for c in all_contacts if c.interaction_type == 'salt_bridge'])
    n_pi = len([c for c in all_contacts if c.interaction_type == 'pi_stack'])

    logger.info(
        f"Detected {len(all_contacts)} catres-catres contacts: "
        f"{n_hbond} hydrogen bonds, {n_salt} salt bridges, {n_pi} pi-stacking"
    )

    # Log individual contacts
    if all_contacts:
        logger.debug("Catres-catres contacts:")
        for c in all_contacts:
            logger.debug(
                f"  {c.interaction_type:12s} {c.resname1:3s} {c.chain1}{c.resnum1} "
                f"{c.atom1:4s} - {c.resname2:3s} {c.chain2}{c.resnum2} {c.atom2:4s} : "
                f"{c.distance:.2f} A"
            )

    return all_contacts


def summarize_catres_catres_contacts(
    contacts: List[CatresCatresContact]
) -> Dict[str, Any]:
    """
    Generate a summary of catres-catres contacts.

    Args:
        contacts: List of CatresCatresContact objects.

    Returns:
        Dictionary with contact statistics.
    """
    if not contacts:
        return {
            'total': 0,
            'by_type': {},
            'residue_pairs': []
        }

    by_type = {}
    residue_pairs = set()

    for c in contacts:
        # Count by type
        if c.interaction_type not in by_type:
            by_type[c.interaction_type] = 0
        by_type[c.interaction_type] += 1

        # Track unique residue pairs
        pair = tuple(sorted([c.res1_id, c.res2_id]))
        residue_pairs.add(pair)

    return {
        'total': len(contacts),
        'by_type': by_type,
        'residue_pairs': list(residue_pairs),
        'n_unique_pairs': len(residue_pairs)
    }
