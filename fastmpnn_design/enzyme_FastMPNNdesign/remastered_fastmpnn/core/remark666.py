"""
REMARK 666 line parsing for Rosetta enzyme design.

Rosetta's enzyme design / matcher workflow writes REMARK 666 lines into
the PDB header to describe how catalytic residues ("motif residues")
map to a constraint block in a CST file.

Example REMARK 666 line:
    REMARK 666 MATCH TEMPLATE B XDW 257 MATCH MOTIF A HIS 13 1 1

This means:
    - Template (ligand) residue: chain B, residue name XDW, residue 257
    - Motif (catalytic residue): chain A, residue type HIS, residue number 13
    - Block index: 1, Block variant: 1
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import re

from remastered_fastmpnn.logging_config import get_logger

logger = get_logger("remark666")


@dataclass
class Remark666Info:
    """
    Parsed information from a single REMARK 666 line.

    Attributes:
        line: Original REMARK 666 line
        template_chain: Chain ID for template (ligand/other catres)
        template_resname: Residue name for template
        template_resnum: Residue number for template (may be 0 for ligands)
        motif_chain: Chain ID for motif (catalytic residue)
        motif_resname: Residue name (3-letter code) for motif
        motif_resnum: Residue number for motif
        block_index: CST block index (1-indexed)
        block_variant: CST block variant (usually 1)
        line_index: Position in the list of REMARK 666 lines (1-indexed)
    """
    line: str
    template_chain: str
    template_resname: str
    template_resnum: int
    motif_chain: str
    motif_resname: str
    motif_resnum: int
    block_index: int
    block_variant: int
    line_index: int = 0  # Position in REMARK 666 list (1-indexed)

    @property
    def motif_identifier(self) -> str:
        """Return motif residue identifier (chain + resnum)."""
        return f"{self.motif_chain}{self.motif_resnum}"

    @property
    def template_identifier(self) -> str:
        """Return template residue identifier (chain + resnum)."""
        return f"{self.template_chain}{self.template_resnum}"

    def is_ligand_template(self) -> bool:
        """Check if template is a ligand (not a protein residue)."""
        # Ligands typically have non-standard 3-letter codes
        # and may have resnum 0 or a special residue number
        standard_aa = {
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
        }
        return self.template_resname not in standard_aa

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "line": self.line,
            "template_chain": self.template_chain,
            "template_resname": self.template_resname,
            "template_resnum": self.template_resnum,
            "motif_chain": self.motif_chain,
            "motif_resname": self.motif_resname,
            "motif_resnum": self.motif_resnum,
            "block_index": self.block_index,
            "block_variant": self.block_variant,
            "line_index": self.line_index,
        }


def parse_remark666_line(line: str, line_index: int = 0) -> Optional[Remark666Info]:
    """
    Parse a single REMARK 666 line.

    Uses keyword anchors (TEMPLATE, MOTIF) rather than hardcoded positions
    for robustness against format variations.

    Args:
        line: Raw REMARK 666 line
        line_index: Position in the REMARK 666 list (1-indexed)

    Returns:
        Remark666Info or None if parsing fails
    """
    # Validate line prefix
    if not line.startswith("REMARK 666"):
        logger.warning(f"Line does not start with 'REMARK 666': {line}")
        return None

    tokens = line.split()

    # Check for required keywords
    if "MATCH" not in tokens or "MOTIF" not in tokens:
        logger.warning(f"Line missing MATCH/MOTIF keywords: {line}")
        return None

    try:
        # Parse TEMPLATE section
        template_chain = ""
        template_resname = ""
        template_resnum = 0

        if "TEMPLATE" in tokens:
            template_idx = tokens.index("TEMPLATE")
            # TEMPLATE is followed by: chain resname resnum
            if template_idx + 3 < len(tokens):
                template_chain = tokens[template_idx + 1]
                template_resname = tokens[template_idx + 2]
                template_resnum = int(tokens[template_idx + 3])

        # Parse MOTIF section (required)
        motif_idx = tokens.index("MOTIF")

        # MOTIF is followed by: chain resname resnum
        if motif_idx + 3 >= len(tokens):
            logger.warning(f"Not enough tokens after MOTIF: {line}")
            return None

        motif_chain = tokens[motif_idx + 1]
        motif_resname = tokens[motif_idx + 2]
        motif_resnum = int(tokens[motif_idx + 3])

        # Block index and variant are typically the last two integers
        # Find them by scanning from the end
        block_index = 0
        block_variant = 0

        # The last two tokens should be block_index and block_variant
        if len(tokens) >= 2:
            try:
                block_variant = int(tokens[-1])
                block_index = int(tokens[-2])
            except ValueError:
                # Try alternative: may have extra whitespace tokens
                for i in range(len(tokens) - 1, motif_idx + 3, -1):
                    try:
                        if block_variant == 0:
                            block_variant = int(tokens[i])
                        elif block_index == 0:
                            block_index = int(tokens[i])
                            break
                    except ValueError:
                        continue

        # Validate parsed values
        if motif_resnum <= 0:
            logger.warning(f"Invalid motif_resnum {motif_resnum}: {line}")
            return None

        return Remark666Info(
            line=line,
            template_chain=template_chain,
            template_resname=template_resname,
            template_resnum=template_resnum,
            motif_chain=motif_chain,
            motif_resname=motif_resname,
            motif_resnum=motif_resnum,
            block_index=block_index,
            block_variant=block_variant,
            line_index=line_index,
        )

    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to parse REMARK 666 line: {line}: {e}")
        return None


def parse_remark666_lines(lines: List[str]) -> List[Remark666Info]:
    """
    Parse multiple REMARK 666 lines.

    Args:
        lines: List of REMARK 666 lines

    Returns:
        List of successfully parsed Remark666Info objects
    """
    results = []
    for i, line in enumerate(lines, start=1):
        info = parse_remark666_line(line, line_index=i)
        if info:
            results.append(info)
        else:
            logger.warning(f"Skipping unparseable REMARK 666 line {i}")

    logger.info(f"Parsed {len(results)} of {len(lines)} REMARK 666 lines")
    return results


def validate_remark666_consistency(
    ref_remarks: List[Remark666Info],
    input_remarks: List[Remark666Info],
) -> Tuple[bool, List[str]]:
    """
    Validate that REMARK 666 lines are consistent between ref and input PDBs.

    Checks:
    1. Same number of REMARK 666 lines
    2. Same motif residue types at each position
    3. Same motif chain/resnum at each position

    Args:
        ref_remarks: Parsed REMARK 666 from reference PDB
        input_remarks: Parsed REMARK 666 from input PDB

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # Check count
    if len(ref_remarks) != len(input_remarks):
        errors.append(
            f"REMARK 666 count mismatch: ref={len(ref_remarks)}, "
            f"input={len(input_remarks)}"
        )
        # Still try to validate what we can
        min_len = min(len(ref_remarks), len(input_remarks))
    else:
        min_len = len(ref_remarks)

    # Check each corresponding line
    for i in range(min_len):
        ref = ref_remarks[i]
        inp = input_remarks[i]

        # Check motif residue type
        if ref.motif_resname != inp.motif_resname:
            errors.append(
                f"Line {i+1}: motif resname mismatch: "
                f"ref={ref.motif_resname}, input={inp.motif_resname}"
            )

        # Check motif residue number (should match)
        if ref.motif_resnum != inp.motif_resnum:
            errors.append(
                f"Line {i+1}: motif resnum mismatch: "
                f"ref={ref.motif_resnum}, input={inp.motif_resnum}"
            )

        # Check motif chain
        if ref.motif_chain != inp.motif_chain:
            errors.append(
                f"Line {i+1}: motif chain mismatch: "
                f"ref={ref.motif_chain}, input={inp.motif_chain}"
            )

        # Check block index
        if ref.block_index != inp.block_index:
            errors.append(
                f"Line {i+1}: block_index mismatch: "
                f"ref={ref.block_index}, input={inp.block_index}"
            )

    is_valid = len(errors) == 0

    if is_valid:
        logger.info("REMARK 666 validation passed")
    else:
        for error in errors:
            logger.error(f"REMARK 666 validation: {error}")

    return is_valid, errors


def get_catres_from_remarks(
    remarks: List[Remark666Info],
    catres_subset_indices: Optional[List[int]] = None,
) -> Tuple[List[Remark666Info], List[Remark666Info]]:
    """
    Split REMARK 666 lines into catres_subset and conserved_motif groups.

    Args:
        remarks: Parsed REMARK 666 lines
        catres_subset_indices: 1-indexed positions of catres_subset residues
                               If None, all residues are catres_subset

    Returns:
        Tuple of (catres_subset_remarks, conserved_motif_remarks)
    """
    if catres_subset_indices is None:
        # All are catres_subset
        return remarks, []

    subset_indices = set(catres_subset_indices)
    catres_subset = []
    conserved_motif = []

    for remark in remarks:
        if remark.line_index in subset_indices:
            catres_subset.append(remark)
        else:
            conserved_motif.append(remark)

    logger.info(
        f"Split {len(remarks)} remarks: "
        f"{len(catres_subset)} catres_subset, {len(conserved_motif)} conserved_motif"
    )

    return catres_subset, conserved_motif


def get_unique_motif_residues(
    remarks: List[Remark666Info]
) -> List[Tuple[str, int, str]]:
    """
    Get unique motif residues from REMARK 666 lines.

    A residue may appear in multiple REMARK 666 lines (e.g., if it
    interacts with multiple CST blocks).

    Args:
        remarks: Parsed REMARK 666 lines

    Returns:
        List of (chain, resnum, resname) tuples for unique residues
    """
    seen = set()
    unique = []

    for remark in remarks:
        key = (remark.motif_chain, remark.motif_resnum)
        if key not in seen:
            seen.add(key)
            unique.append((remark.motif_chain, remark.motif_resnum, remark.motif_resname))

    return unique
