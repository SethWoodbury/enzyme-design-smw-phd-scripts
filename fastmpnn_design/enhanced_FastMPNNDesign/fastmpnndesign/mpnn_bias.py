"""
MPNN bias calculation for enhanced_fastmpnndesign.

Handles per-position amino acid bias calculation based on distance from ligand atoms.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add package directory to path for standalone execution
_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import BiasConfig
from constants import DEFAULT_BIAS_CUTS, DESIGN_UTILS_PATH
from logging_config import get_logger
from utils import format_residue_list

logger = get_logger("mpnn_bias")


def calculate_bias_positions(
    pose: Any,
    bias_atoms: List[str],
    bias_AAs: str,
    position_bias: float,
    keep_pos: List[int],
    ligand_seqpos: Optional[int] = None,
    cuts: tuple = DEFAULT_BIAS_CUTS
) -> Dict[str, Dict[str, float]]:
    """
    Calculate per-position amino acid bias.

    This replicates the bias calculation from original script lines 265-293.
    Biases positions close to specified ligand atoms towards/against certain amino acids.

    Args:
        pose: PyRosetta Pose object
        bias_atoms: Ligand atom names for bias calculation
        bias_AAs: Single-letter codes of amino acids to bias
        position_bias: Bias value (negative = favor, positive = disfavor)
        keep_pos: Positions to exclude from bias
        ligand_seqpos: Ligand sequence position (defaults to last residue)
        cuts: Distance cutoffs for layer detection

    Returns:
        Dictionary mapping position_id -> {aa: bias_value}
        Format: {'A150': {'K': -1.0, 'R': -1.0, ...}, ...}
    """
    if DESIGN_UTILS_PATH not in sys.path:
        sys.path.append(DESIGN_UTILS_PATH)

    import design_utils

    if ligand_seqpos is None:
        ligand_seqpos = pose.size()

    # Validate bias atoms
    ligand = pose.residue(ligand_seqpos)
    valid_atoms = []
    for atom in bias_atoms:
        if ligand.has(atom):
            valid_atoms.append(atom)
        else:
            logger.warning(f"Ligand does not have atom '{atom}'")

    if not valid_atoms:
        logger.warning("No valid bias atoms found")
        return {}

    # Get layer selections around bias atoms
    _, _, _, residues_bias = design_utils.get_layer_selections(
        pose, keep_pos, [], ligand_seqpos, valid_atoms,
        cuts=list(cuts)
    )

    # Positions in first two layers get bias
    bias_positions = list(set(residues_bias[0] + residues_bias[1]))

    # Special case: exclude positions closer to O1 than to H1 (from original lines 276-283)
    if "O1" not in bias_atoms and "H1" in bias_atoms and ligand.has("O1"):
        filtered_positions = []
        for pos in bias_positions:
            dist_o1 = (pose.residue(pos).xyz("CA") - ligand.xyz("O1")).norm()
            dists_bias = [
                (pose.residue(pos).xyz("CA") - ligand.xyz(a)).norm()
                for a in valid_atoms
            ]
            if min(dists_bias) < dist_o1:
                filtered_positions.append(pos)
        bias_positions = filtered_positions

    if not bias_positions:
        logger.info("No positions selected for bias")
        return {}

    logger.info(f"Bias positions for {bias_AAs}: {format_residue_list(bias_positions)}")

    # Build bias dictionary
    pdb_info = pose.pdb_info()
    bias_dict = {}

    for pos in bias_positions:
        chain = pdb_info.chain(pos)
        resnum = pdb_info.number(pos)
        key = f"{chain}{resnum}"

        bias_dict[key] = {aa: position_bias for aa in bias_AAs}

    return bias_dict


def apply_bias_config(
    pose: Any,
    config: BiasConfig,
    keep_pos: List[int],
    ligand_seqpos: Optional[int] = None
) -> Dict[str, Dict[str, float]]:
    """
    Apply bias configuration to get bias positions.

    Args:
        pose: PyRosetta Pose object
        config: Bias configuration
        keep_pos: Positions to exclude from bias
        ligand_seqpos: Ligand sequence position

    Returns:
        Bias dictionary for MPNN
    """
    if config.bias_atoms is None:
        return {}

    return calculate_bias_positions(
        pose=pose,
        bias_atoms=config.bias_atoms,
        bias_AAs=config.bias_AAs,
        position_bias=config.position_bias,
        keep_pos=keep_pos,
        ligand_seqpos=ligand_seqpos,
        cuts=config.bias_cuts
    )
