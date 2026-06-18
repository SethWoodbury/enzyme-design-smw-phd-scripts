"""
Remastered FastMPNNdesign - Enzyme Design Pipeline

A modular pipeline for de novo enzyme design with catalytic residue
coordinate transformation and sequence optimization.
"""

__version__ = "0.1.0"
__author__ = "woodbuse"

from remastered_fastmpnn.core.residue_data import (
    ResidueInfo,
    CatresSubsetInfo,
    InteractionInfo,
    ResidueRegistry,
)

__all__ = [
    "ResidueInfo",
    "CatresSubsetInfo",
    "InteractionInfo",
    "ResidueRegistry",
]
