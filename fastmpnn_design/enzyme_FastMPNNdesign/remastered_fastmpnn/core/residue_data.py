"""
Residue data structures for tracking per-residue information through the pipeline.

Provides dataclasses for:
- InteractionInfo: Details of a single interaction
- CatresSubsetInfo: Information specific to catres_subset residues
- ResidueInfo: Complete per-residue tracking
- ResidueRegistry: Collection of all tracked residues
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path
import json


class ImportantComponent(Enum):
    """Which component of the residue is important for catalysis."""
    BACKBONE = "backbone"
    SIDECHAIN = "sidechain"
    BOTH = "both"
    NONE = "none"  # No interactions detected


class ResidueCategory(Enum):
    """Categories for residue classification in the pipeline."""
    CATRES_SUBSET = "catres_subset"          # Constrained to exact geometry
    CONSERVED_MOTIF = "conserved_motif"      # Conserved sequence, free geometry
    DESIGN = "design"                         # Designable residue
    LIGAND = "ligand"                         # Ligand residue
    FIXED = "fixed"                           # Fixed but not catalytic


@dataclass
class InteractionInfo:
    """
    Details of a single interaction between a residue and ligand/metal/catres.

    Attributes:
        interaction_type: Type of interaction (e.g., "hbond_donation", "metal_coordination")
        from_component: Whether interaction is from "backbone" or "sidechain"
        from_atom: Atom name on the residue making the interaction
        to_entity: What the interaction is with ("ligand", "metal", or "catres")
        to_atom: Atom name on the target entity
        to_residue: For catres-catres, the target residue identifier
        distance: Distance in Angstroms
        priority: Priority level for ranking interactions
    """
    interaction_type: str
    from_component: str  # "backbone" or "sidechain"
    from_atom: str
    to_entity: str  # "ligand", "metal", or "catres"
    to_atom: str
    distance: float
    priority: int
    to_residue: Optional[str] = None  # For catres-catres interactions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "interaction_type": self.interaction_type,
            "from_component": self.from_component,
            "from_atom": self.from_atom,
            "to_entity": self.to_entity,
            "to_atom": self.to_atom,
            "distance": round(self.distance, 3),
            "priority": self.priority,
        }
        if self.to_residue:
            result["to_residue"] = self.to_residue
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InteractionInfo":
        """Create from dictionary."""
        return cls(
            interaction_type=data["interaction_type"],
            from_component=data["from_component"],
            from_atom=data["from_atom"],
            to_entity=data["to_entity"],
            to_atom=data["to_atom"],
            distance=data["distance"],
            priority=data["priority"],
            to_residue=data.get("to_residue"),
        )


@dataclass
class CatresSubsetInfo:
    """
    Detailed information for catres_subset residues.

    Tracks interaction analysis results and coordinate transformation status.
    """
    # Interaction analysis results
    important_component: ImportantComponent = ImportantComponent.NONE
    interactions_found: Dict[str, InteractionInfo] = field(default_factory=dict)

    # Interaction counts by component
    backbone_interaction_count: int = 0
    sidechain_interaction_count: int = 0

    # Coordinate transformation tracking
    backbone_coords_copied: bool = False
    sidechain_coords_copied: bool = False

    # Reference PDB information (for coordinate lookup)
    ref_pdb_resnum: Optional[int] = None
    ref_pdb_chain: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "important_component": self.important_component.value,
            "interactions_found": {
                k: v.to_dict() for k, v in self.interactions_found.items()
            },
            "backbone_interaction_count": self.backbone_interaction_count,
            "sidechain_interaction_count": self.sidechain_interaction_count,
            "backbone_coords_copied": self.backbone_coords_copied,
            "sidechain_coords_copied": self.sidechain_coords_copied,
            "ref_pdb_resnum": self.ref_pdb_resnum,
            "ref_pdb_chain": self.ref_pdb_chain,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CatresSubsetInfo":
        """Create from dictionary."""
        return cls(
            important_component=ImportantComponent(data["important_component"]),
            interactions_found={
                k: InteractionInfo.from_dict(v)
                for k, v in data.get("interactions_found", {}).items()
            },
            backbone_interaction_count=data.get("backbone_interaction_count", 0),
            sidechain_interaction_count=data.get("sidechain_interaction_count", 0),
            backbone_coords_copied=data.get("backbone_coords_copied", False),
            sidechain_coords_copied=data.get("sidechain_coords_copied", False),
            ref_pdb_resnum=data.get("ref_pdb_resnum"),
            ref_pdb_chain=data.get("ref_pdb_chain"),
        )

    def get_interactions_by_type(self, interaction_type: str) -> List[InteractionInfo]:
        """Get all interactions of a specific type."""
        return [
            i for i in self.interactions_found.values()
            if i.interaction_type == interaction_type
        ]

    def get_interactions_by_component(self, component: str) -> List[InteractionInfo]:
        """Get all interactions from a specific component (backbone/sidechain)."""
        return [
            i for i in self.interactions_found.values()
            if i.from_component == component
        ]


@dataclass
class ResidueInfo:
    """
    Complete information about a residue tracked through the pipeline.

    This is the primary data structure for per-residue tracking,
    providing the nested dictionary format requested by the user.
    """
    # Core identifiers
    chain: str
    residue_num: int
    identifier: str  # "A1", "A13", etc. (chain + resnum)
    res_type: str    # 3-letter code: "TRP", "HIS", etc.

    # REMARK 666 information (if from catalytic residue list)
    remark666_index: Optional[int] = None  # 1-indexed position in REMARK 666 lines
    cst_block: Optional[int] = None        # CST block number
    cst_variant: Optional[int] = None      # CST variant number

    # Categorization
    category: ResidueCategory = ResidueCategory.DESIGN
    is_catres_subset: bool = False
    is_conserved_motif: bool = False

    # Design probability (0.0 = never mutate, 1.0 = freely designable)
    probability_of_mutation: float = 1.0

    # Catres-specific information (only populated if is_catres_subset=True)
    catres_subset_info: Optional[CatresSubsetInfo] = None

    # Step completion tracking (for pipeline state)
    step1_complete: bool = False
    step2_complete: bool = False
    step3_complete: bool = False

    # Arbitrary metadata for extensibility
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and set up defaults after initialization."""
        # Ensure identifier matches chain + resnum
        expected_id = f"{self.chain}{self.residue_num}"
        if self.identifier != expected_id:
            self.identifier = expected_id

        # Set mutation probability based on category
        if self.is_catres_subset or self.is_conserved_motif:
            self.probability_of_mutation = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to the user-specified nested dictionary format.

        Example output:
        {
            "chain": "A",
            "residue_num": 13,
            "identifier": "A13",
            "res_type": "HIS",
            "catres_subset": True,
            "probability_of_mutation": 0.0,
            "catres_subset_info": {
                "important_component": "sidechain",
                "interactions_found": {...}
            }
        }
        """
        result = {
            "chain": self.chain,
            "residue_num": self.residue_num,
            "identifier": self.identifier,
            "res_type": self.res_type,
            "catres_subset": self.is_catres_subset,
            "conserved_motif": self.is_conserved_motif,
            "probability_of_mutation": self.probability_of_mutation,
        }

        if self.remark666_index is not None:
            result["remark666_index"] = self.remark666_index

        if self.cst_block is not None:
            result["cst_block"] = self.cst_block

        if self.catres_subset_info:
            result["catres_subset_info"] = self.catres_subset_info.to_dict()

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResidueInfo":
        """Create from dictionary."""
        catres_info = None
        if "catres_subset_info" in data:
            catres_info = CatresSubsetInfo.from_dict(data["catres_subset_info"])

        return cls(
            chain=data["chain"],
            residue_num=data["residue_num"],
            identifier=data["identifier"],
            res_type=data["res_type"],
            remark666_index=data.get("remark666_index"),
            cst_block=data.get("cst_block"),
            is_catres_subset=data.get("catres_subset", False),
            is_conserved_motif=data.get("conserved_motif", False),
            probability_of_mutation=data.get("probability_of_mutation", 1.0),
            catres_subset_info=catres_info,
            metadata=data.get("metadata", {}),
        )


@dataclass
class ResidueRegistry:
    """
    Registry of all residues tracked in the pipeline.

    Provides dictionary-style access, filtering methods, and serialization.
    """
    residues: Dict[str, ResidueInfo] = field(default_factory=dict)

    # Metadata about the registry
    input_pdb_path: Optional[str] = None
    ref_pdb_path: Optional[str] = None
    ligand_name: Optional[str] = None
    total_residue_count: int = 0

    def add(self, residue: ResidueInfo) -> None:
        """Add a residue to the registry."""
        self.residues[residue.identifier] = residue

    def get(self, identifier: str) -> Optional[ResidueInfo]:
        """Get a residue by identifier (e.g., 'A13')."""
        return self.residues.get(identifier)

    def get_by_chain_resnum(self, chain: str, resnum: int) -> Optional[ResidueInfo]:
        """Get a residue by chain and residue number."""
        identifier = f"{chain}{resnum}"
        return self.residues.get(identifier)

    def __contains__(self, identifier: str) -> bool:
        """Check if identifier is in registry."""
        return identifier in self.residues

    def __iter__(self):
        """Iterate over all residues."""
        return iter(self.residues.values())

    def __len__(self) -> int:
        """Return number of residues in registry."""
        return len(self.residues)

    def get_catres_subset(self) -> List[ResidueInfo]:
        """Get all catres_subset residues."""
        return [r for r in self.residues.values() if r.is_catres_subset]

    def get_conserved_motif(self) -> List[ResidueInfo]:
        """Get all conserved_motif residues."""
        return [r for r in self.residues.values() if r.is_conserved_motif]

    def get_designable(self) -> List[ResidueInfo]:
        """Get all designable residues (probability_of_mutation > 0)."""
        return [r for r in self.residues.values() if r.probability_of_mutation > 0]

    def get_by_category(self, category: ResidueCategory) -> List[ResidueInfo]:
        """Get all residues of a specific category."""
        return [r for r in self.residues.values() if r.category == category]

    def get_by_res_type(self, res_type: str) -> List[ResidueInfo]:
        """Get all residues of a specific type (e.g., 'HIS')."""
        return [r for r in self.residues.values() if r.res_type == res_type]

    def get_by_chain(self, chain: str) -> List[ResidueInfo]:
        """Get all residues in a specific chain."""
        return [r for r in self.residues.values() if r.chain == chain]

    def to_dict(self) -> Dict[str, Any]:
        """
        Export registry to dictionary format.

        Returns the user's desired nested dictionary structure.
        """
        return {
            "metadata": {
                "input_pdb_path": self.input_pdb_path,
                "ref_pdb_path": self.ref_pdb_path,
                "ligand_name": self.ligand_name,
                "total_residue_count": self.total_residue_count,
                "catres_subset_count": len(self.get_catres_subset()),
                "conserved_motif_count": len(self.get_conserved_motif()),
            },
            "residues": {k: v.to_dict() for k, v in self.residues.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResidueRegistry":
        """Create registry from dictionary."""
        registry = cls(
            input_pdb_path=data.get("metadata", {}).get("input_pdb_path"),
            ref_pdb_path=data.get("metadata", {}).get("ref_pdb_path"),
            ligand_name=data.get("metadata", {}).get("ligand_name"),
            total_residue_count=data.get("metadata", {}).get("total_residue_count", 0),
        )
        for identifier, residue_data in data.get("residues", {}).items():
            registry.add(ResidueInfo.from_dict(residue_data))
        return registry

    def save_json(self, path: Path, indent: int = 2) -> None:
        """
        Save registry to JSON file.

        Args:
            path: Output file path
            indent: JSON indentation level
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def load_json(cls, path: Path) -> "ResidueRegistry":
        """
        Load registry from JSON file.

        Args:
            path: Input file path

        Returns:
            ResidueRegistry instance
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def summary(self) -> str:
        """Generate a summary string of the registry contents."""
        catres = self.get_catres_subset()
        conserved = self.get_conserved_motif()
        designable = self.get_designable()

        lines = [
            f"ResidueRegistry Summary:",
            f"  Total residues: {len(self.residues)}",
            f"  Catres subset: {len(catres)}",
            f"  Conserved motif: {len(conserved)}",
            f"  Designable: {len(designable)}",
        ]

        if catres:
            lines.append(f"  Catres identifiers: {', '.join(r.identifier for r in catres)}")

        return "\n".join(lines)
