"""Data models for residue tracking and interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ResidueId:
    chain: str
    resno: int
    icode: str
    resname: str

    @property
    def identifier(self) -> str:
        icode = self.icode.strip()
        return f"{self.chain}{self.resno}{icode}"


@dataclass
class Interaction:
    interaction_type: str
    from_component: str
    from_atom: str
    to_entity: str
    to_atom: str
    distance: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interaction_type": self.interaction_type,
            "from_component": self.from_component,
            "from_atom": self.from_atom,
            "to_entity": self.to_entity,
            "to_atom": self.to_atom,
            "distance": round(self.distance, 3),
        }


@dataclass
class CatalyticResidueInfo:
    important_component: str
    interactions_found: List[Interaction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "important_component": self.important_component,
            "interactions_found": [interaction.to_dict() for interaction in self.interactions_found],
        }


@dataclass
class ResidueRecord:
    residue_id: ResidueId
    probability_of_mutation: float
    catres_subset: bool
    motif_label: str
    catres_subset_info: Optional[CatalyticResidueInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "chain": self.residue_id.chain,
            "residue_num": self.residue_id.resno,
            "identifier": self.residue_id.identifier,
            "res_type": self.residue_id.resname,
            "catres_subset": self.catres_subset,
            "probability_of_mutation": self.probability_of_mutation,
            "motif_label": self.motif_label,
        }
        if self.catres_subset_info:
            data["catres_subset_info"] = self.catres_subset_info.to_dict()
        return data
