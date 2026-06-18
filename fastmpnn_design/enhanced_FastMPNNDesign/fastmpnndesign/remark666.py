"""
REMARK 666 parser for catalytic residue extraction.

Parses Rosetta match REMARK 666 lines from PDB files to extract
catalytic residue definitions.

Format:
    REMARK 666 MATCH TEMPLATE <chain> <resname> <resnum> MATCH MOTIF <chain> <resname> <resnum> <cst_block> <cst_var>

Example:
    REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A PHE  150  10  1

This extracts:
    - catres_index = 10 (the constraint block number)
    - chain = A
    - resnum = 150
    - resname = PHE
    - cst_block = 10
    - cst_var = 1
"""

import sys
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# Add package directory to path for standalone execution
_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import CatalyticResidue
from logging_config import get_logger

logger = get_logger("remark666")


def parse_remark666_line(line: str, fallback_index: int = 0) -> Optional[CatalyticResidue]:
    """
    Parse a single REMARK 666 line.

    Handles the format used by RosettaMatch/EnzDes:
    REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A PHE  150  10  1

    Where the fields are:
    - MATCH TEMPLATE: the ligand/substrate info (chain, resname, resnum)
    - MATCH MOTIF: the catalytic residue info (chain, resname, resnum, cst_block, cst_var)

    Args:
        line: The REMARK 666 line to parse
        fallback_index: Index to use if cst_block not found in line

    Returns:
        CatalyticResidue object or None if parse fails
    """
    line = line.strip()
    if not line.upper().startswith('REMARK 666'):
        return None

    try:
        # Split the line into parts
        # Format: REMARK 666 MATCH TEMPLATE <chain> <resname> <resnum> MATCH MOTIF <chain> <resname> <resnum> <cst_block> <cst_var>
        parts = line.split()

        # Find indices of key words
        # parts[0] = REMARK, parts[1] = 666, parts[2] = MATCH, parts[3] = TEMPLATE
        # parts[4] = template_chain, parts[5] = template_resname, parts[6] = template_resnum
        # parts[7] = MATCH, parts[8] = MOTIF
        # parts[9] = motif_chain, parts[10] = motif_resname, parts[11] = motif_resnum
        # parts[12] = cst_block, parts[13] = cst_var

        if len(parts) < 12:
            logger.warning(f"REMARK 666 line too short: {line}")
            return None

        # Template info
        template_chain = parts[4]
        template_resname = parts[5]
        template_resnum = int(parts[6])

        # Motif (catalytic residue) info
        motif_chain = parts[9]
        motif_resname = parts[10]
        motif_resnum = int(parts[11])

        # Constraint block and variant (optional)
        cst_block = None
        cst_var = None
        if len(parts) >= 14:
            cst_block = int(parts[12])
            cst_var = int(parts[13])

        # catres_index is the constraint block number
        catres_index = cst_block if cst_block is not None else fallback_index

        return CatalyticResidue(
            catres_index=catres_index,
            chain=motif_chain,
            resnum=motif_resnum,
            resname=motif_resname.upper(),
            icode="",  # Handle insertion codes if present
            cst_block=cst_block,
            cst_var=cst_var,
            raw_line=line,
            template_chain=template_chain,
            template_resname=template_resname,
            template_resnum=template_resnum
        )

    except (IndexError, ValueError) as e:
        logger.warning(f"Could not parse REMARK 666 line: {line} - {e}")
        return None


def parse_remark666_from_pdb(pdb_path: Path) -> List[CatalyticResidue]:
    """
    Parse all REMARK 666 lines from a PDB file.

    Args:
        pdb_path: Path to the PDB file

    Returns:
        List of CatalyticResidue objects, ordered by catres_index
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    catres_list = []
    line_index = 0

    with open(pdb_path, 'r') as f:
        for line in f:
            # Stop at ATOM records
            if line.startswith(('ATOM', 'HETATM')):
                break

            if line.upper().startswith('REMARK 666'):
                line_index += 1
                catres = parse_remark666_line(line, fallback_index=line_index)
                if catres:
                    catres_list.append(catres)

    # Sort by catres_index
    catres_list.sort(key=lambda x: x.catres_index)

    logger.info(f"Parsed {len(catres_list)} catalytic residues from REMARK 666 lines")
    for cr in catres_list:
        logger.debug(f"  Catres {cr.catres_index}: {cr.chain}{cr.resnum} {cr.resname}")

    return catres_list


def parse_remark666_from_pose(pose) -> List[CatalyticResidue]:
    """
    Parse REMARK 666 lines from a PyRosetta Pose.

    This matches the behavior of design_utils.get_matcher_residues(pose)
    from the original script.

    Args:
        pose: PyRosetta Pose object

    Returns:
        List of CatalyticResidue objects
    """
    import pyrosetta.distributed.io

    pdb_string = pyrosetta.distributed.io.to_pdbstring(pose)
    lines = pdb_string.split('\n')

    catres_list = []
    line_index = 0

    for line in lines:
        if line.startswith(('ATOM', 'HETATM')):
            break

        if line.upper().startswith('REMARK 666'):
            line_index += 1
            catres = parse_remark666_line(line, fallback_index=line_index)
            if catres:
                catres_list.append(catres)

    catres_list.sort(key=lambda x: x.catres_index)
    return catres_list


def parse_remark666_from_lines(lines: List[str]) -> List[CatalyticResidue]:
    """
    Parse REMARK 666 lines from a list of strings.

    Args:
        lines: List of PDB lines

    Returns:
        List of CatalyticResidue objects
    """
    catres_list = []
    line_index = 0

    for line in lines:
        if line.upper().startswith('REMARK 666'):
            line_index += 1
            catres = parse_remark666_line(line, fallback_index=line_index)
            if catres:
                catres_list.append(catres)

    catres_list.sort(key=lambda x: x.catres_index)
    return catres_list


def get_catres_as_dict(pdb_path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Parse REMARK 666 and return as dictionary keyed by seqpos.

    This provides compatibility with design_utils.get_matcher_residues()
    from the original script which returns:
    {seqpos: {'chain': ..., 'name3': ..., 'cst_no': ..., 'cst_no_var': ...}}

    Note: The seqpos key requires a pose to map chain/resnum to seqpos.
    This function uses resnum as the key instead.

    Args:
        pdb_path: Path to PDB file

    Returns:
        Dictionary keyed by resnum with catres info
    """
    catres_list = parse_remark666_from_pdb(pdb_path)

    matches = {}
    for cr in catres_list:
        matches[cr.resnum] = {
            'target_name': cr.template_resname,
            'target_chain': cr.template_chain,
            'target_resno': cr.template_resnum,
            'chain': cr.chain,
            'name3': cr.resname,
            'cst_no': cr.cst_block,
            'cst_no_var': cr.cst_var
        }

    return matches


def get_catres_subset(
    catres_list: List[CatalyticResidue],
    subset_indices: Optional[List[int]] = None
) -> Tuple[List[CatalyticResidue], List[CatalyticResidue]]:
    """
    Split catalytic residues into subset (tight constraints) and non-subset.

    Args:
        catres_list: Full list of catalytic residues
        subset_indices: Catres indices (from REMARK 666) to include in subset.
                       If None, all catres are in subset.

    Returns:
        Tuple of (subset_catres, non_subset_catres)
    """
    if subset_indices is None:
        return catres_list.copy(), []

    subset = []
    non_subset = []

    for cr in catres_list:
        if cr.catres_index in subset_indices:
            subset.append(cr)
        else:
            non_subset.append(cr)

    logger.info(f"Catres subset: {len(subset)} residues, non-subset: {len(non_subset)} residues")

    if subset:
        logger.debug(f"  Subset: {[cr.pdb_resid for cr in subset]}")
    if non_subset:
        logger.debug(f"  Non-subset: {[cr.pdb_resid for cr in non_subset]}")

    return subset, non_subset


def catres_to_fixed_residues(catres_list: List[CatalyticResidue]) -> List[str]:
    """
    Convert catalytic residues to list of residue IDs for MPNN fixed_residues.

    Returns list in format ['A150', 'A152', ...] for MPNN fixed_residues_multi JSON.
    """
    return [cr.pdb_resid for cr in catres_list]


def map_catres_to_seqpos(catres_list: List[CatalyticResidue], pose) -> Dict[int, int]:
    """
    Map catalytic residues to PyRosetta sequence positions.

    Args:
        catres_list: List of catalytic residues
        pose: PyRosetta Pose object

    Returns:
        Dictionary mapping seqpos -> catres_index
    """
    pdb_info = pose.pdb_info()
    catres_seqpos = {}

    for cr in catres_list:
        for i in range(1, pose.size() + 1):
            if (pdb_info.chain(i) == cr.chain and
                pdb_info.number(i) == cr.resnum):
                cr.seqpos = i
                catres_seqpos[i] = cr.catres_index
                logger.debug(f"Catres {cr.catres_index} ({cr.pdb_resid}) -> seqpos {i}")
                break
        else:
            logger.warning(f"Could not find seqpos for catres {cr.catres_index} ({cr.pdb_resid})")

    return catres_seqpos
