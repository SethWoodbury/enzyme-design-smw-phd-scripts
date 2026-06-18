"""
Core utilities shared across pipeline steps.

Includes PDB I/O, residue data structures, coordinate utilities,
and REMARK 666 parsing.
"""

from remastered_fastmpnn.core.residue_data import (
    ResidueInfo,
    CatresSubsetInfo,
    InteractionInfo,
    ResidueRegistry,
)
from remastered_fastmpnn.core.pdb_io import PDBParser, PDBWriter
from remastered_fastmpnn.core.remark666 import parse_remark666_lines, Remark666Info
from remastered_fastmpnn.core.coordinate_utils import (
    kabsch_align,
    calculate_rmsd,
    transform_coordinates,
)

__all__ = [
    "ResidueInfo",
    "CatresSubsetInfo",
    "InteractionInfo",
    "ResidueRegistry",
    "PDBParser",
    "PDBWriter",
    "parse_remark666_lines",
    "Remark666Info",
    "kabsch_align",
    "calculate_rmsd",
    "transform_coordinates",
]
