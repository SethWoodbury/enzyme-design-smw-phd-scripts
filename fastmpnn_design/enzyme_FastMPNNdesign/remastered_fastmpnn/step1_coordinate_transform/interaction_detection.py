"""
Interaction detection for determining important residue components.

Analyzes interactions between catalytic residues and ligand/metal/other catres
to determine whether backbone, sidechain, or both are important for catalysis.

Detects:
1. Metal coordination (ZN, etc.) - always sidechain
2. Covalent bonds / PTM - determined by atom
3. Backbone H-bond donation (amide NH to ligand)
4. Backbone H-bond acceptance (amide C=O from ligand)
5. Sidechain H-bond donation/acceptance
6. Salt bridges (charged interactions)
7. Pi-stacking (aromatic interactions)
8. Hydrophobic contacts
9. Catres-catres interactions (supporting evidence)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

from remastered_fastmpnn.constants import (
    METALS,
    HETEROATOMS,
    BACKBONE_ATOMS,
    BACKBONE_AMIDE_DONOR,
    BACKBONE_AMIDE_ACCEPTOR,
    METAL_COORDINATION_CUTOFF,
    HBOND_CUTOFF,
    COVALENT_BOND_CUTOFF,
    SALT_BRIDGE_CUTOFF,
    PI_STACK_CUTOFF,
    HYDROPHOBIC_CUTOFF,
    SIDECHAIN_HBOND_DONORS,
    SIDECHAIN_HBOND_ACCEPTORS,
    METAL_COORDINATING_ATOMS,
    POSITIVE_CHARGE_ATOMS,
    NEGATIVE_CHARGE_ATOMS,
    AROMATIC_RING_ATOMS,
    PRIORITY_METAL_COORDINATION,
    PRIORITY_COVALENT_BOND,
    PRIORITY_SALT_BRIDGE,
    PRIORITY_HBOND_BACKBONE,
    PRIORITY_HBOND_SIDECHAIN,
    PRIORITY_CATRES_HBOND,
    PRIORITY_PI_STACK,
    PRIORITY_HYDROPHOBIC,
    DEFAULT_IMPORTANT_COMPONENT,
)
from remastered_fastmpnn.core.residue_data import (
    InteractionInfo,
    CatresSubsetInfo,
    ImportantComponent,
)
from remastered_fastmpnn.core.pdb_io import PDBParser, AtomRecord
from remastered_fastmpnn.core.coordinate_utils import atom_distance
from remastered_fastmpnn.logging_config import get_logger

logger = get_logger("interaction_detection")


class InteractionDetector:
    """
    Detects interactions between catalytic residues and ligand/metals.

    Determines whether backbone, sidechain, or both are important
    for each catres_subset residue based on detected interactions.
    """

    def __init__(
        self,
        metal_cutoff: float = METAL_COORDINATION_CUTOFF,
        hbond_cutoff: float = HBOND_CUTOFF,
        covalent_cutoff: float = COVALENT_BOND_CUTOFF,
        salt_bridge_cutoff: float = SALT_BRIDGE_CUTOFF,
        pi_stack_cutoff: float = PI_STACK_CUTOFF,
        hydrophobic_cutoff: float = HYDROPHOBIC_CUTOFF,
        verbose: bool = True,
    ):
        """
        Initialize the interaction detector.

        Args:
            metal_cutoff: Distance cutoff for metal coordination (A)
            hbond_cutoff: Distance cutoff for hydrogen bonds (A)
            covalent_cutoff: Distance cutoff for covalent/PTM bonds (A)
            salt_bridge_cutoff: Distance cutoff for salt bridges (A)
            pi_stack_cutoff: Distance cutoff for pi-stacking (A)
            hydrophobic_cutoff: Distance cutoff for hydrophobic contacts (A)
            verbose: Enable verbose logging
        """
        self.metal_cutoff = metal_cutoff
        self.hbond_cutoff = hbond_cutoff
        self.covalent_cutoff = covalent_cutoff
        self.salt_bridge_cutoff = salt_bridge_cutoff
        self.pi_stack_cutoff = pi_stack_cutoff
        self.hydrophobic_cutoff = hydrophobic_cutoff
        self.verbose = verbose

    def detect_all_interactions(
        self,
        residue_atoms: List[AtomRecord],
        ligand_atoms: List[AtomRecord],
        metal_atoms: List[AtomRecord],
        catres_atoms: Optional[Dict[str, List[AtomRecord]]] = None,
    ) -> Tuple[ImportantComponent, Dict[str, InteractionInfo], int, int]:
        """
        Detect all interactions for a single residue.

        Args:
            residue_atoms: Atoms of the residue to analyze
            ligand_atoms: Ligand atoms to check interactions with
            metal_atoms: Metal atoms to check interactions with
            catres_atoms: Dict mapping identifier -> atoms for other catres
                          (for catres-catres interaction detection)

        Returns:
            Tuple of:
                - ImportantComponent enum value
                - Dict of interaction_key -> InteractionInfo
                - Count of backbone interactions
                - Count of sidechain interactions
        """
        if not residue_atoms:
            return ImportantComponent.SIDECHAIN, {}, 0, 0

        resname = residue_atoms[0].resname
        identifier = residue_atoms[0].identifier

        interactions: Dict[str, InteractionInfo] = {}
        backbone_interactions: List[InteractionInfo] = []
        sidechain_interactions: List[InteractionInfo] = []

        # Check interactions with ligand and metals
        for res_atom in residue_atoms:
            atom_name = res_atom.name.strip()
            is_bb = self._is_backbone_atom(atom_name)

            # --- Metal coordination ---
            for metal in metal_atoms:
                dist = atom_distance(res_atom, metal)
                if dist <= self.metal_cutoff:
                    # Metal coordination - check if this atom can coordinate
                    if atom_name in METAL_COORDINATING_ATOMS.get(resname, set()):
                        key = f"metal_{atom_name}_{metal.name.strip()}"
                        interaction = InteractionInfo(
                            interaction_type="metal_coordination",
                            from_component="sidechain",
                            from_atom=atom_name,
                            to_entity="metal",
                            to_atom=metal.name.strip(),
                            distance=dist,
                            priority=PRIORITY_METAL_COORDINATION,
                        )
                        interactions[key] = interaction
                        sidechain_interactions.append(interaction)
                        logger.debug(
                            f"  Metal coordination: {atom_name} -> {metal.name.strip()} "
                            f"({dist:.2f} A)"
                        )

            # --- Ligand interactions ---
            for lig_atom in ligand_atoms:
                dist = atom_distance(res_atom, lig_atom)

                # Skip if too far for any interaction
                max_cutoff = max(
                    self.hbond_cutoff,
                    self.pi_stack_cutoff,
                    self.hydrophobic_cutoff
                )
                if dist > max_cutoff:
                    continue

                lig_element = lig_atom.element.strip().upper()
                res_element = res_atom.element.strip().upper()

                # --- Covalent bond (PTM) ---
                if dist <= self.covalent_cutoff:
                    key = f"covalent_{atom_name}_{lig_atom.name.strip()}"
                    component = "backbone" if is_bb else "sidechain"
                    interaction = InteractionInfo(
                        interaction_type="covalent_bond",
                        from_component=component,
                        from_atom=atom_name,
                        to_entity="ligand",
                        to_atom=lig_atom.name.strip(),
                        distance=dist,
                        priority=PRIORITY_COVALENT_BOND,
                    )
                    interactions[key] = interaction
                    if is_bb:
                        backbone_interactions.append(interaction)
                    else:
                        sidechain_interactions.append(interaction)
                    logger.debug(
                        f"  Covalent bond: {atom_name} -> {lig_atom.name.strip()} "
                        f"({dist:.2f} A)"
                    )
                    continue  # Covalent supersedes other interactions

                # --- Hydrogen bonds ---
                if dist <= self.hbond_cutoff:
                    # Backbone H-bond donation (N-H...acceptor)
                    if (
                        atom_name in BACKBONE_AMIDE_DONOR and
                        lig_element in HETEROATOMS
                    ):
                        key = f"hbond_bb_don_{atom_name}_{lig_atom.name.strip()}"
                        interaction = InteractionInfo(
                            interaction_type="hbond_donation",
                            from_component="backbone",
                            from_atom=atom_name,
                            to_entity="ligand",
                            to_atom=lig_atom.name.strip(),
                            distance=dist,
                            priority=PRIORITY_HBOND_BACKBONE,
                        )
                        interactions[key] = interaction
                        backbone_interactions.append(interaction)
                        logger.debug(
                            f"  BB H-bond donation: {atom_name} -> "
                            f"{lig_atom.name.strip()} ({dist:.2f} A)"
                        )

                    # Backbone H-bond acceptance (C=O...donor)
                    elif (
                        atom_name in BACKBONE_AMIDE_ACCEPTOR and
                        lig_element in {'N', 'O'}
                    ):
                        key = f"hbond_bb_acc_{atom_name}_{lig_atom.name.strip()}"
                        interaction = InteractionInfo(
                            interaction_type="hbond_acceptance",
                            from_component="backbone",
                            from_atom=atom_name,
                            to_entity="ligand",
                            to_atom=lig_atom.name.strip(),
                            distance=dist,
                            priority=PRIORITY_HBOND_BACKBONE,
                        )
                        interactions[key] = interaction
                        backbone_interactions.append(interaction)
                        logger.debug(
                            f"  BB H-bond acceptance: {atom_name} -> "
                            f"{lig_atom.name.strip()} ({dist:.2f} A)"
                        )

                    # Sidechain H-bond donation
                    elif (
                        not is_bb and
                        atom_name in SIDECHAIN_HBOND_DONORS.get(resname, set()) and
                        lig_element in HETEROATOMS
                    ):
                        key = f"hbond_sc_don_{atom_name}_{lig_atom.name.strip()}"
                        interaction = InteractionInfo(
                            interaction_type="hbond_donation",
                            from_component="sidechain",
                            from_atom=atom_name,
                            to_entity="ligand",
                            to_atom=lig_atom.name.strip(),
                            distance=dist,
                            priority=PRIORITY_HBOND_SIDECHAIN,
                        )
                        interactions[key] = interaction
                        sidechain_interactions.append(interaction)
                        logger.debug(
                            f"  SC H-bond donation: {atom_name} -> "
                            f"{lig_atom.name.strip()} ({dist:.2f} A)"
                        )

                    # Sidechain H-bond acceptance
                    elif (
                        not is_bb and
                        atom_name in SIDECHAIN_HBOND_ACCEPTORS.get(resname, set()) and
                        lig_element in {'N', 'O', 'H'}
                    ):
                        key = f"hbond_sc_acc_{atom_name}_{lig_atom.name.strip()}"
                        interaction = InteractionInfo(
                            interaction_type="hbond_acceptance",
                            from_component="sidechain",
                            from_atom=atom_name,
                            to_entity="ligand",
                            to_atom=lig_atom.name.strip(),
                            distance=dist,
                            priority=PRIORITY_HBOND_SIDECHAIN,
                        )
                        interactions[key] = interaction
                        sidechain_interactions.append(interaction)
                        logger.debug(
                            f"  SC H-bond acceptance: {atom_name} -> "
                            f"{lig_atom.name.strip()} ({dist:.2f} A)"
                        )

                # --- Salt bridges / charged interactions ---
                if dist <= self.salt_bridge_cutoff and not is_bb:
                    if (
                        atom_name in POSITIVE_CHARGE_ATOMS.get(resname, set()) or
                        atom_name in NEGATIVE_CHARGE_ATOMS.get(resname, set())
                    ):
                        # Check if ligand atom is likely charged (heteroatom)
                        if lig_element in HETEROATOMS:
                            key = f"charged_{atom_name}_{lig_atom.name.strip()}"
                            interaction = InteractionInfo(
                                interaction_type="charged_interaction",
                                from_component="sidechain",
                                from_atom=atom_name,
                                to_entity="ligand",
                                to_atom=lig_atom.name.strip(),
                                distance=dist,
                                priority=PRIORITY_SALT_BRIDGE,
                            )
                            interactions[key] = interaction
                            sidechain_interactions.append(interaction)
                            logger.debug(
                                f"  Charged interaction: {atom_name} -> "
                                f"{lig_atom.name.strip()} ({dist:.2f} A)"
                            )

                # --- Pi-stacking ---
                if dist <= self.pi_stack_cutoff and not is_bb:
                    if atom_name in AROMATIC_RING_ATOMS.get(resname, set()):
                        # Simplified: any contact with carbon on ligand
                        if lig_element == 'C':
                            key = f"pi_{atom_name}_{lig_atom.name.strip()}"
                            interaction = InteractionInfo(
                                interaction_type="pi_interaction",
                                from_component="sidechain",
                                from_atom=atom_name,
                                to_entity="ligand",
                                to_atom=lig_atom.name.strip(),
                                distance=dist,
                                priority=PRIORITY_PI_STACK,
                            )
                            interactions[key] = interaction
                            sidechain_interactions.append(interaction)

                # --- Hydrophobic contacts ---
                if dist <= self.hydrophobic_cutoff and not is_bb:
                    if res_element == 'C' and lig_element == 'C':
                        # Carbon-carbon contact
                        key = f"hydrophobic_{atom_name}_{lig_atom.name.strip()}"
                        if key not in interactions:  # Avoid duplicates
                            interaction = InteractionInfo(
                                interaction_type="hydrophobic_contact",
                                from_component="sidechain",
                                from_atom=atom_name,
                                to_entity="ligand",
                                to_atom=lig_atom.name.strip(),
                                distance=dist,
                                priority=PRIORITY_HYDROPHOBIC,
                            )
                            interactions[key] = interaction
                            sidechain_interactions.append(interaction)

        # --- Catres-catres interactions (supporting evidence) ---
        if catres_atoms:
            for other_id, other_atoms in catres_atoms.items():
                if other_id == identifier:
                    continue  # Skip self

                for res_atom in residue_atoms:
                    atom_name = res_atom.name.strip()
                    is_bb = self._is_backbone_atom(atom_name)

                    for other_atom in other_atoms:
                        dist = atom_distance(res_atom, other_atom)

                        if dist <= self.hbond_cutoff:
                            other_elem = other_atom.element.strip().upper()
                            res_elem = res_atom.element.strip().upper()

                            # Check for H-bond potential
                            if (
                                res_elem in HETEROATOMS and
                                other_elem in HETEROATOMS
                            ):
                                component = "backbone" if is_bb else "sidechain"
                                key = (
                                    f"catres_{atom_name}_{other_id}_"
                                    f"{other_atom.name.strip()}"
                                )
                                interaction = InteractionInfo(
                                    interaction_type="catres_interaction",
                                    from_component=component,
                                    from_atom=atom_name,
                                    to_entity="catres",
                                    to_atom=other_atom.name.strip(),
                                    to_residue=other_id,
                                    distance=dist,
                                    priority=PRIORITY_CATRES_HBOND,
                                )
                                interactions[key] = interaction
                                if is_bb:
                                    backbone_interactions.append(interaction)
                                else:
                                    sidechain_interactions.append(interaction)
                                logger.debug(
                                    f"  Catres interaction: {atom_name} -> "
                                    f"{other_id}:{other_atom.name.strip()} ({dist:.2f} A)"
                                )

        # Determine important component
        has_backbone = len(backbone_interactions) > 0
        has_sidechain = len(sidechain_interactions) > 0

        if has_backbone and has_sidechain:
            component = ImportantComponent.BOTH
        elif has_backbone:
            component = ImportantComponent.BACKBONE
        elif has_sidechain:
            component = ImportantComponent.SIDECHAIN
        else:
            # No interactions detected - default to sidechain
            component = ImportantComponent.SIDECHAIN
            logger.debug(f"  No interactions detected, defaulting to sidechain")

        if self.verbose:
            logger.info(
                f"  {identifier} ({resname}): {len(interactions)} interactions, "
                f"BB={len(backbone_interactions)}, SC={len(sidechain_interactions)} "
                f"-> {component.value}"
            )

        return (
            component,
            interactions,
            len(backbone_interactions),
            len(sidechain_interactions)
        )

    def analyze_catres_subset(
        self,
        parser: PDBParser,
        catres_identifiers: List[Tuple[str, int]],
    ) -> Dict[str, CatresSubsetInfo]:
        """
        Analyze interactions for all catres_subset residues.

        Args:
            parser: PDBParser with aligned coordinates
            catres_identifiers: List of (chain, resnum) tuples for catres

        Returns:
            Dict mapping identifier -> CatresSubsetInfo
        """
        # Get ligand and metal atoms
        ligand_atoms = parser.get_ligand_atoms()
        metal_atoms = parser.get_metal_atoms()

        logger.info(f"Analyzing {len(catres_identifiers)} catres_subset residues")
        logger.info(f"  Ligand atoms: {len(ligand_atoms)}")
        logger.info(f"  Metal atoms: {len(metal_atoms)}")

        # Collect all catres atoms for catres-catres interaction detection
        catres_atoms: Dict[str, List[AtomRecord]] = {}
        for chain, resnum in catres_identifiers:
            identifier = f"{chain}{resnum}"
            catres_atoms[identifier] = parser.get_residue_atoms(chain, resnum)

        # Analyze each catres
        results: Dict[str, CatresSubsetInfo] = {}

        for chain, resnum in catres_identifiers:
            identifier = f"{chain}{resnum}"
            residue_atoms = catres_atoms[identifier]

            if not residue_atoms:
                logger.warning(f"  No atoms found for {identifier}")
                continue

            logger.info(f"Analyzing {identifier} ({residue_atoms[0].resname})...")

            component, interactions, bb_count, sc_count = self.detect_all_interactions(
                residue_atoms=residue_atoms,
                ligand_atoms=ligand_atoms,
                metal_atoms=metal_atoms,
                catres_atoms=catres_atoms,
            )

            info = CatresSubsetInfo(
                important_component=component,
                interactions_found=interactions,
                backbone_interaction_count=bb_count,
                sidechain_interaction_count=sc_count,
                ref_pdb_chain=chain,
                ref_pdb_resnum=resnum,
            )
            results[identifier] = info

        return results

    @staticmethod
    def _is_backbone_atom(atom_name: str) -> bool:
        """Check if atom name is a backbone atom."""
        return atom_name in BACKBONE_ATOMS or atom_name in {'H', 'HA', 'HA2', 'HA3'}

    def summarize_interactions(
        self,
        catres_info: Dict[str, CatresSubsetInfo]
    ) -> str:
        """
        Generate a summary of all detected interactions.

        Args:
            catres_info: Dict mapping identifier -> CatresSubsetInfo

        Returns:
            Formatted summary string
        """
        lines = ["Interaction Analysis Summary:", "=" * 50]

        for identifier, info in sorted(catres_info.items()):
            lines.append(f"\n{identifier}:")
            lines.append(f"  Important component: {info.important_component.value}")
            lines.append(f"  Backbone interactions: {info.backbone_interaction_count}")
            lines.append(f"  Sidechain interactions: {info.sidechain_interaction_count}")

            if info.interactions_found:
                lines.append("  Interactions:")
                # Sort by priority (highest first)
                sorted_interactions = sorted(
                    info.interactions_found.values(),
                    key=lambda x: -x.priority
                )
                for interaction in sorted_interactions[:5]:  # Top 5
                    lines.append(
                        f"    - {interaction.interaction_type}: "
                        f"{interaction.from_atom} ({interaction.from_component}) -> "
                        f"{interaction.to_entity}:{interaction.to_atom} "
                        f"({interaction.distance:.2f} A)"
                    )
                if len(sorted_interactions) > 5:
                    lines.append(f"    ... and {len(sorted_interactions) - 5} more")

        return "\n".join(lines)
