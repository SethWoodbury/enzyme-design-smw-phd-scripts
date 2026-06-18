"""PDB restoration utilities for post-MPNN processing.

MPNN outputs PDBs without:
- Hydrogens (only heavy atoms)
- REMARK 666 lines (catalytic residue/ligand definitions)
- Correct HIS tautomer names (always uses HIS, not HIS_D)

This module provides functions to restore these features from a reference PDB,
which is critical for:
1. Rosetta to properly add hydrogens based on tautomer state
2. Downstream scripts to identify catalytic residues via REMARK 666
3. Maintaining consistency with step01/step02 output

Key functions:
- extract_remark_lines: Get REMARK lines from reference PDB
- build_his_tautomer_map: Identify HIS_D vs HIS from reference hydrogens
- restore_pdb_features: Full restoration of MPNN output
"""
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add module_utils to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module_utils.pdb_utils import read_pdb_atoms, format_atom_line, get_residue_atoms

LOGGER = logging.getLogger(__name__)


# HIS tautomer determination based on proton placement
# HIS_D (delta tautomer): proton on ND1, has HD1 hydrogen
# HIS (epsilon tautomer): proton on NE2, has HE2 hydrogen
# HIP (doubly protonated): has both HD1 and HE2
HIS_TAUTOMER_ATOMS = {
    "HD1": "HIS_D",  # Delta tautomer
    "HE2": "HIS",    # Epsilon tautomer (default in most tools)
}


def extract_remark_lines(
    pdb_path: str,
    remark_types: Optional[List[str]] = None,
    preserve_all: bool = True,
) -> List[str]:
    """Extract REMARK lines from a PDB file.

    Args:
        pdb_path: Path to reference PDB file
        remark_types: List of REMARK types to extract (e.g., ["666", "900"])
                     If None, extracts all REMARK lines
        preserve_all: If True, preserve ALL lines of the specified remark types,
                     not just a subset. This ensures all REMARK 666 lines from
                     step01/step02 flow through the entire pipeline.

    Returns:
        List of REMARK lines (including newlines)
    """
    remark_lines = []
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("REMARK"):
                if remark_types is None:
                    remark_lines.append(line)
                else:
                    # Check if this REMARK type is in our list
                    for rtype in remark_types:
                        if line.startswith(f"REMARK {rtype}"):
                            remark_lines.append(line)
                            break

    if preserve_all and remark_lines:
        LOGGER.debug(f"Preserving all {len(remark_lines)} REMARK lines from {pdb_path}")

    return remark_lines


def build_his_tautomer_map(pdb_path: str) -> Dict[Tuple[str, int], str]:
    """Build map of HIS residue positions to their tautomer states.

    Determines tautomer state by checking which protons are present:
    - HD1 present → HIS_D (delta tautomer)
    - HE2 present → HIS (epsilon tautomer)
    - Both present → HIP (doubly protonated, but we'll use HIS for simplicity)
    - Neither present → HIS (default)

    Args:
        pdb_path: Path to reference PDB with hydrogens

    Returns:
        Dict mapping (chain, resno) -> tautomer name ("HIS" or "HIS_D")
    """
    _, atoms = read_pdb_atoms(pdb_path)

    # Find all HIS residues (including HIS_D, HIP, HIE, HID variants)
    his_residues: Dict[Tuple[str, int], Set[str]] = {}

    for atom in atoms:
        resname = atom["resname"]
        if resname in ("HIS", "HIS_D", "HIP", "HIE", "HID"):
            key = (atom["chain"], atom["resno"])
            if key not in his_residues:
                his_residues[key] = set()
            his_residues[key].add(atom["atom_name"])

    # Determine tautomer for each HIS
    tautomer_map = {}
    for (chain, resno), atom_names in his_residues.items():
        has_hd1 = "HD1" in atom_names
        has_he2 = "HE2" in atom_names

        if has_hd1 and not has_he2:
            tautomer_map[(chain, resno)] = "HIS_D"
            LOGGER.debug(f"HIS {chain}{resno}: HIS_D (delta tautomer, HD1 present)")
        elif has_he2 and not has_hd1:
            tautomer_map[(chain, resno)] = "HIS"
            LOGGER.debug(f"HIS {chain}{resno}: HIS (epsilon tautomer, HE2 present)")
        elif has_hd1 and has_he2:
            # Doubly protonated - use HIS as default (Rosetta will handle)
            tautomer_map[(chain, resno)] = "HIS"
            LOGGER.debug(f"HIS {chain}{resno}: HIS (doubly protonated, both HD1 and HE2)")
        else:
            # No protons found - default to HIS
            tautomer_map[(chain, resno)] = "HIS"
            LOGGER.debug(f"HIS {chain}{resno}: HIS (default, no tautomer protons found)")

    LOGGER.info(f"Built tautomer map for {len(tautomer_map)} HIS residues")
    return tautomer_map


def get_hydrogen_atoms_for_residue(
    ref_atoms: List[Dict],
    chain: str,
    resno: int,
) -> List[Dict]:
    """Get all hydrogen atoms for a specific residue from reference.

    Args:
        ref_atoms: List of atom dicts from reference PDB
        chain: Chain ID
        resno: Residue number

    Returns:
        List of hydrogen atom dicts
    """
    residue_atoms = get_residue_atoms(ref_atoms, chain, resno)
    hydrogens = []
    for atom in residue_atoms:
        # Element is H or atom name starts with H (and element is empty)
        if atom.get("element", "").strip() == "H":
            hydrogens.append(atom)
        elif atom["atom_name"].startswith("H") and not atom.get("element"):
            hydrogens.append(atom)
        elif atom["atom_name"] in ("1H", "2H", "3H"):  # N-terminal hydrogens
            hydrogens.append(atom)
    return hydrogens


def identify_mutated_positions(
    ref_pdb: str,
    mpnn_pdb: str,
) -> Set[Tuple[str, int]]:
    """Identify positions where MPNN changed the amino acid.

    Args:
        ref_pdb: Path to reference PDB
        mpnn_pdb: Path to MPNN output PDB

    Returns:
        Set of (chain, resno) tuples where mutations occurred
    """
    _, ref_atoms = read_pdb_atoms(ref_pdb)
    _, mpnn_atoms = read_pdb_atoms(mpnn_pdb)

    # Build residue name maps
    ref_residues = {}
    for atom in ref_atoms:
        if atom["record_type"] == "ATOM":
            key = (atom["chain"], atom["resno"])
            if key not in ref_residues:
                ref_residues[key] = atom["resname"]

    mpnn_residues = {}
    for atom in mpnn_atoms:
        if atom["record_type"] == "ATOM":
            key = (atom["chain"], atom["resno"])
            if key not in mpnn_residues:
                mpnn_residues[key] = atom["resname"]

    # Find mutations
    mutated = set()
    for key, ref_resname in ref_residues.items():
        mpnn_resname = mpnn_residues.get(key)
        if mpnn_resname and ref_resname != mpnn_resname:
            # Normalize HIS variants for comparison
            ref_norm = "HIS" if ref_resname in ("HIS", "HIS_D", "HIP", "HIE", "HID") else ref_resname
            mpnn_norm = "HIS" if mpnn_resname in ("HIS", "HIS_D", "HIP", "HIE", "HID") else mpnn_resname
            if ref_norm != mpnn_norm:
                mutated.add(key)
                LOGGER.debug(f"Mutation detected: {key} {ref_resname} -> {mpnn_resname}")

    LOGGER.info(f"Identified {len(mutated)} mutated positions")
    return mutated


def restore_pdb_features(
    mpnn_pdb: str,
    ref_pdb: str,
    output_pdb: str,
    restore_remarks: bool = True,
    restore_his_tautomers: bool = True,
    restore_hydrogens: bool = False,  # Usually let Rosetta handle this
    remark_types: Optional[List[str]] = None,
    preserve_all_remarks: bool = True,
) -> Dict:
    """Restore PDB features from reference to MPNN output.

    This function:
    1. Copies REMARK lines (especially 666) from reference
    2. Corrects HIS residue names to proper tautomer (HIS_D if needed)
    3. Optionally restores hydrogens for unchanged residues

    Note: For mutated positions, Rosetta should add hydrogens during relaxation.
    Restoring hydrogens from reference for mutated positions would be incorrect.

    IMPORTANT: REMARK 666 lines contain critical metadata about catalytic residues
    and ligand definitions from step01/step02. ALL REMARK 666 lines must be
    preserved throughout the pipeline, not just those matching a catres_subset.

    Args:
        mpnn_pdb: Path to MPNN output PDB (heavy atoms only)
        ref_pdb: Path to reference PDB (with hydrogens, REMARK lines)
        output_pdb: Path to write restored PDB
        restore_remarks: Whether to restore REMARK lines
        restore_his_tautomers: Whether to correct HIS names to proper tautomer
        restore_hydrogens: Whether to restore hydrogens for unchanged residues
        remark_types: Specific REMARK types to restore (default: ["666"])
        preserve_all_remarks: If True, preserve ALL REMARK lines of specified
                             types (important for REMARK 666 pipeline continuity)

    Returns:
        Dict with restoration statistics
    """
    if remark_types is None:
        remark_types = ["666"]

    stats = {
        "remark_lines_restored": 0,
        "his_tautomers_corrected": 0,
        "hydrogens_restored": 0,
        "mutated_positions": 0,
    }

    # Read both PDBs
    mpnn_lines, mpnn_atoms = read_pdb_atoms(mpnn_pdb)
    _, ref_atoms = read_pdb_atoms(ref_pdb)

    # Build tautomer map if needed
    his_tautomer_map = {}
    if restore_his_tautomers:
        his_tautomer_map = build_his_tautomer_map(ref_pdb)

    # Find mutated positions if restoring hydrogens
    mutated_positions = set()
    if restore_hydrogens:
        mutated_positions = identify_mutated_positions(ref_pdb, mpnn_pdb)
        stats["mutated_positions"] = len(mutated_positions)

    # Build output
    output_lines = []

    # First, add REMARK lines from reference
    # IMPORTANT: Preserve ALL REMARK 666 lines to maintain pipeline metadata
    if restore_remarks:
        remark_lines = extract_remark_lines(ref_pdb, remark_types, preserve_all=preserve_all_remarks)
        output_lines.extend(remark_lines)
        stats["remark_lines_restored"] = len(remark_lines)

    # Track which residues we've processed for hydrogen restoration
    processed_residues = set()
    residue_hydrogens: Dict[Tuple[str, int], List[Dict]] = {}

    if restore_hydrogens:
        # Pre-compute hydrogens for unchanged residues
        for atom in ref_atoms:
            if atom["record_type"] == "ATOM":
                key = (atom["chain"], atom["resno"])
                if key not in mutated_positions:
                    if key not in residue_hydrogens:
                        residue_hydrogens[key] = get_hydrogen_atoms_for_residue(
                            ref_atoms, atom["chain"], atom["resno"]
                        )

    # Process MPNN output lines
    current_residue = None
    for line in mpnn_lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            # Parse atom
            try:
                chain = line[21:22]
                resno = int(line[22:26])
                resname = line[17:20].strip()
                key = (chain, resno)

                # Check if we need to correct HIS tautomer
                if restore_his_tautomers and resname == "HIS":
                    tautomer = his_tautomer_map.get(key)
                    if tautomer == "HIS_D":
                        # For 5-char residue names like HIS_D, we carefully format the line.
                        # PDB format (0-indexed):
                        #   16=altLoc, 17-19=resName(3), 20=iCode, 21=chainID, 22-25=resNo
                        # Replace indices 16-20 (5 chars: altLoc+resName+iCode) with "HIS_D"
                        # This preserves chain ID at index 21 and resno at indices 22+
                        line = line[:16] + "HIS_D" + line[21:]
                        stats["his_tautomers_corrected"] += 1

                output_lines.append(line)

                # Check if we need to add hydrogens after this residue
                if restore_hydrogens:
                    if current_residue is not None and current_residue != key:
                        # Finished previous residue, add its hydrogens
                        if current_residue not in mutated_positions and current_residue in residue_hydrogens:
                            if current_residue not in processed_residues:
                                for h_atom in residue_hydrogens[current_residue]:
                                    output_lines.append(format_atom_line(h_atom))
                                    stats["hydrogens_restored"] += 1
                                processed_residues.add(current_residue)

                    current_residue = key

            except (ValueError, IndexError):
                output_lines.append(line)
        else:
            # Non-ATOM/HETATM line - check if it's a REMARK we should skip
            if line.startswith("REMARK") and restore_remarks:
                # Skip REMARK lines from MPNN output (we added reference ones)
                for rtype in remark_types:
                    if line.startswith(f"REMARK {rtype}"):
                        break
                else:
                    output_lines.append(line)
            else:
                output_lines.append(line)

    # Add hydrogens for last residue if needed
    if restore_hydrogens and current_residue is not None:
        if current_residue not in mutated_positions and current_residue in residue_hydrogens:
            if current_residue not in processed_residues:
                for h_atom in residue_hydrogens[current_residue]:
                    output_lines.append(format_atom_line(h_atom))
                    stats["hydrogens_restored"] += 1

    # Write output
    Path(output_pdb).parent.mkdir(parents=True, exist_ok=True)
    with open(output_pdb, "w") as f:
        f.writelines(output_lines)

    LOGGER.info(f"Restored PDB written to {output_pdb}")
    LOGGER.info(f"  REMARK lines restored: {stats['remark_lines_restored']}")
    LOGGER.info(f"  HIS tautomers corrected: {stats['his_tautomers_corrected']}")
    if restore_hydrogens:
        LOGGER.info(f"  Hydrogens restored: {stats['hydrogens_restored']}")
        LOGGER.info(f"  Mutated positions (no H restore): {stats['mutated_positions']}")

    return stats


def restore_ligand_and_hetatm(
    mpnn_pdb: str,
    ref_pdb: str,
    output_pdb: str,
) -> Dict:
    """Restore ligand/HETATM records from reference if missing in MPNN output.

    MPNN sometimes doesn't output HETATM records. This function checks if they're
    missing and restores them from the reference.

    Args:
        mpnn_pdb: Path to MPNN output PDB
        ref_pdb: Path to reference PDB
        output_pdb: Path to write output (can be same as mpnn_pdb)

    Returns:
        Dict with statistics
    """
    mpnn_lines, mpnn_atoms = read_pdb_atoms(mpnn_pdb)
    ref_lines, ref_atoms = read_pdb_atoms(ref_pdb)

    # Check for HETATM in MPNN output
    mpnn_hetatm = [a for a in mpnn_atoms if a["record_type"] == "HETATM"]
    ref_hetatm = [a for a in ref_atoms if a["record_type"] == "HETATM"]

    stats = {
        "hetatm_in_mpnn": len(mpnn_hetatm),
        "hetatm_in_ref": len(ref_hetatm),
        "hetatm_restored": 0,
    }

    if len(mpnn_hetatm) > 0:
        # HETATM already present, nothing to do
        LOGGER.info(f"HETATM records already present in MPNN output ({len(mpnn_hetatm)} atoms)")
        if output_pdb != mpnn_pdb:
            with open(output_pdb, "w") as f:
                f.writelines(mpnn_lines)
        return stats

    if len(ref_hetatm) == 0:
        LOGGER.info("No HETATM records in reference PDB")
        if output_pdb != mpnn_pdb:
            with open(output_pdb, "w") as f:
                f.writelines(mpnn_lines)
        return stats

    # Need to restore HETATM from reference
    LOGGER.info(f"Restoring {len(ref_hetatm)} HETATM atoms from reference")

    # Find where to insert HETATM (after last ATOM, before END/ENDMDL)
    output_lines = []
    inserted = False

    for i, line in enumerate(mpnn_lines):
        if not inserted and (line.startswith("END") or line.startswith("TER")):
            # Insert HETATM before END/TER
            for atom in ref_hetatm:
                output_lines.append(format_atom_line(atom))
                stats["hetatm_restored"] += 1
            inserted = True
        output_lines.append(line)

    # If we didn't find END/TER, append at the end
    if not inserted:
        for atom in ref_hetatm:
            output_lines.append(format_atom_line(atom))
            stats["hetatm_restored"] += 1

    with open(output_pdb, "w") as f:
        f.writelines(output_lines)

    LOGGER.info(f"Restored {stats['hetatm_restored']} HETATM atoms")
    return stats


def full_mpnn_output_restoration(
    mpnn_pdb: str,
    ref_pdb: str,
    output_pdb: str,
    restore_hydrogens_for_unchanged: bool = False,
    original_ref_pdb: Optional[str] = None,
    restore_his_tautomers: bool = True,
) -> Dict:
    """Complete restoration of MPNN output with all features.

    Performs in order:
    1. Restore ALL REMARK 666 lines (critical pipeline metadata)
    2. Correct HIS tautomers (if enabled)
    3. Optionally restore hydrogens for unchanged residues
    4. Restore ligand/HETATM if missing

    IMPORTANT: This function ensures MPNN outputs have:
    - All REMARK 666 lines from the original step01/step02 pipeline
    - Correct HIS tautomers (HIS_D vs HIS) - only if restore_his_tautomers=True
    - HETATM records for ligands

    This is critical when the protocol ends with an MPNN step (no Rosetta after),
    as the final outputs need these features for downstream analysis.

    IMPORTANT: For intermediate MPNN steps (when there will be more MPNN calls),
    set restore_his_tautomers=False because MPNN cannot properly handle 5-char
    residue names like HIS_D. Only restore HIS tautomers for the final output.

    Args:
        mpnn_pdb: Path to MPNN output PDB
        ref_pdb: Path to reference PDB (step02 relaxed output)
        output_pdb: Path to write fully restored PDB
        restore_hydrogens_for_unchanged: Whether to restore H for unchanged residues
        original_ref_pdb: Optional path to original step01/step02 PDB for REMARK 666
                         lines if they're not in ref_pdb (chain of custody)
        restore_his_tautomers: Whether to restore HIS tautomers (HIS_D). Set to False
                              for intermediate MPNN steps to avoid corruption.

    Returns:
        Dict with all restoration statistics
    """
    LOGGER.info(f"Starting full MPNN output restoration")
    LOGGER.info(f"  MPNN output: {mpnn_pdb}")
    LOGGER.info(f"  Reference: {ref_pdb}")

    # Determine best source for REMARK 666 lines
    # Prefer original_ref_pdb if provided, to ensure we get ALL original REMARK 666 lines
    remark_source = original_ref_pdb if original_ref_pdb else ref_pdb
    if original_ref_pdb:
        LOGGER.info(f"  Using original ref for REMARK 666: {original_ref_pdb}")

    # Step 1: Restore remarks, tautomers, and optionally hydrogens
    # Use preserve_all_remarks=True to ensure ALL REMARK 666 lines are kept
    stats = restore_pdb_features(
        mpnn_pdb=mpnn_pdb,
        ref_pdb=remark_source,  # Use source with all REMARK 666 lines
        output_pdb=output_pdb,
        restore_remarks=True,
        restore_his_tautomers=restore_his_tautomers,
        restore_hydrogens=restore_hydrogens_for_unchanged,
        remark_types=["666"],
        preserve_all_remarks=True,  # Preserve ALL REMARK 666 lines
    )

    # If we used a different source for REMARK lines, now apply HIS tautomers from ref_pdb
    if original_ref_pdb and original_ref_pdb != ref_pdb:
        # Re-run to get correct HIS tautomers from the actual ref_pdb
        # (original_ref_pdb might not have the right tautomers after relaxation)
        his_tautomer_map = build_his_tautomer_map(ref_pdb)
        # The HIS correction was already applied with remark_source's tautomers
        # This is fine since we want tautomers from the relaxed structure

    # Step 2: Restore ligand/HETATM if missing
    hetatm_stats = restore_ligand_and_hetatm(
        mpnn_pdb=output_pdb,  # Use intermediate output
        ref_pdb=ref_pdb,
        output_pdb=output_pdb,  # Overwrite
    )
    stats.update(hetatm_stats)

    LOGGER.info(f"Full restoration complete: {output_pdb}")
    return stats
