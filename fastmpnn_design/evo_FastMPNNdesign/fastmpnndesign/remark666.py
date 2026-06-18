"""
REMARK 666 parser for catalytic residue extraction.

Parses Rosetta match REMARK 666 lines from PDB files to extract
catalytic residue definitions.

Format:
    REMARK 666 MATCH TEMPLATE <chain> <resname> <resnum> MATCH MOTIF <chain> <resname> <resnum> <cst_block> <cst_var>

Example:
    REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A PHE  150  10  1

This extracts:
    - catres_index = 10
    - chain = A
    - resnum = 150
    - resname = PHE
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

from fastmpnndesign.config import CatalyticResidue
from fastmpnndesign.logging_config import get_logger

logger = get_logger("remark666")

# Pattern for REMARK 666 MATCH lines
# REMARK 666 MATCH TEMPLATE <chain> <resname> <resnum> MATCH MOTIF <chain> <resname> <resnum> [<cst_block> <cst_var>]
REMARK_666_PATTERN = re.compile(
    r'^REMARK\s+666\s+MATCH\s+TEMPLATE\s+'
    r'(\S+)\s+(\S+)\s+(\d+)\s+'  # Template: chain, resname, resnum
    r'MATCH\s+MOTIF\s+'
    r'(\S+)\s+(\S+)\s+(-?\d+)([A-Za-z]?)'  # Motif: chain, resname, resnum, optional icode
    r'(?:\s+(\d+)\s+(\d+))?'  # Optional: cst_block, cst_var
    r'\s*$',
    re.IGNORECASE
)

# Alternative pattern for simpler format
REMARK_666_SIMPLE_PATTERN = re.compile(
    r'^REMARK\s+666\s+MATCH\s+TEMPLATE\s+'
    r'(\S+)\s+(\S+)\s+(\d+)\s+'
    r'MATCH\s+MOTIF\s+'
    r'(\S+)\s+(\S+)\s+(-?\d+)',
    re.IGNORECASE
)


def parse_remark666_line(line: str, index: int = 0) -> Optional[CatalyticResidue]:
    """
    Parse a single REMARK 666 line.

    Args:
        line: The REMARK 666 line to parse.
        index: Default index if not found in line.

    Returns:
        CatalyticResidue object or None if parse fails.
    """
    line = line.strip()
    if not line.upper().startswith('REMARK 666'):
        return None

    # Try full pattern first
    match = REMARK_666_PATTERN.match(line)
    if match:
        groups = match.groups()
        # template_chain, template_resname, template_resnum = groups[0:3]
        motif_chain = groups[3]
        motif_resname = groups[4]
        motif_resnum = int(groups[5])
        motif_icode = groups[6] if groups[6] else ""
        cst_block = int(groups[7]) if groups[7] else None
        cst_var = int(groups[8]) if groups[8] else None

        # catres_index comes from cst_block if present, otherwise use provided index
        catres_idx = cst_block if cst_block is not None else index

        return CatalyticResidue(
            catres_index=catres_idx,
            chain=motif_chain,
            resnum=motif_resnum,
            resname=motif_resname.upper(),
            icode=motif_icode,
            cst_block=cst_block,
            cst_var=cst_var,
            raw_line=line
        )

    # Try simpler pattern
    match = REMARK_666_SIMPLE_PATTERN.match(line)
    if match:
        groups = match.groups()
        motif_chain = groups[3]
        motif_resname = groups[4]
        motif_resnum = int(groups[5])

        # Extract trailing numbers if present
        remaining = line[match.end():].strip()
        parts = remaining.split()
        cst_block = None
        cst_var = None
        icode = ""

        # Check if first character after resnum is an icode (letter)
        if parts:
            if parts[0].isalpha() and len(parts[0]) == 1:
                icode = parts[0]
                parts = parts[1:]
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                cst_block = int(parts[0])
                cst_var = int(parts[1])

        catres_idx = cst_block if cst_block is not None else index

        return CatalyticResidue(
            catres_index=catres_idx,
            chain=motif_chain,
            resnum=motif_resnum,
            resname=motif_resname.upper(),
            icode=icode,
            cst_block=cst_block,
            cst_var=cst_var,
            raw_line=line
        )

    logger.warning(f"Could not parse REMARK 666 line: {line}")
    return None


def parse_remark666_from_pdb(pdb_path: Path) -> List[CatalyticResidue]:
    """
    Parse all REMARK 666 lines from a PDB file.

    Args:
        pdb_path: Path to the PDB file.

    Returns:
        List of CatalyticResidue objects, ordered by catres_index.
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    catres_list = []
    line_index = 0

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.upper().startswith('REMARK 666'):
                line_index += 1
                catres = parse_remark666_line(line, index=line_index)
                if catres:
                    catres_list.append(catres)

    # Sort by catres_index
    catres_list.sort(key=lambda x: x.catres_index)

    logger.info(f"Parsed {len(catres_list)} catalytic residues from REMARK 666 lines")
    for cr in catres_list:
        logger.debug(f"  Catres {cr.catres_index}: {cr.chain}{cr.resnum} {cr.resname}")

    return catres_list


def parse_remark666_from_lines(lines: List[str]) -> List[CatalyticResidue]:
    """
    Parse REMARK 666 lines from a list of strings.

    Args:
        lines: List of PDB lines.

    Returns:
        List of CatalyticResidue objects.
    """
    catres_list = []
    line_index = 0

    for line in lines:
        if line.upper().startswith('REMARK 666'):
            line_index += 1
            catres = parse_remark666_line(line, index=line_index)
            if catres:
                catres_list.append(catres)

    catres_list.sort(key=lambda x: x.catres_index)
    return catres_list


def get_catres_subset(
    catres_list: List[CatalyticResidue],
    subset_indices: Optional[List[int]] = None
) -> Tuple[List[CatalyticResidue], List[CatalyticResidue]]:
    """
    Split catalytic residues into subset (tight constraints) and non-subset.

    Args:
        catres_list: Full list of catalytic residues.
        subset_indices: Indices to include in subset. If None, all are in subset.

    Returns:
        Tuple of (subset_catres, non_subset_catres).
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
    return subset, non_subset


def catres_to_fixed_residues(catres_list: List[CatalyticResidue]) -> List[str]:
    """
    Convert catalytic residues to list of residue IDs for MPNN fixed_residues.

    Returns list in format ['A150', 'A152', ...] for MPNN fixed_residues_multi JSON.
    """
    return [cr.pdb_resid for cr in catres_list]


def catres_to_rosetta_ids(catres_list: List[CatalyticResidue]) -> List[str]:
    """
    Convert catalytic residues to Rosetta residue selector format.

    Returns list in format ['150A', '152A', ...].
    """
    return [cr.rosetta_resid for cr in catres_list]


def validate_catres_in_pdb(
    catres_list: List[CatalyticResidue],
    pdb_path: Path
) -> Tuple[List[CatalyticResidue], List[CatalyticResidue]]:
    """
    Validate that catalytic residues exist in the PDB file.

    Args:
        catres_list: List of catalytic residues.
        pdb_path: Path to PDB file to validate against.

    Returns:
        Tuple of (found_catres, missing_catres).
    """
    from fastmpnndesign.utils import iter_pdb_atoms

    # Build set of (chain, resnum) present in PDB
    pdb_residues = set()
    for atom in iter_pdb_atoms(pdb_path):
        pdb_residues.add((atom['chain'], atom['resnum']))

    found = []
    missing = []

    for cr in catres_list:
        if (cr.chain, cr.resnum) in pdb_residues:
            found.append(cr)
        else:
            missing.append(cr)
            logger.warning(
                f"Catalytic residue {cr.chain}{cr.resnum} ({cr.resname}) "
                f"not found in PDB"
            )

    return found, missing
