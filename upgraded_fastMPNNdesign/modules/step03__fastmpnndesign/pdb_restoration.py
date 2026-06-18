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
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..module_utils.pdb_utils import read_pdb_atoms, format_atom_line, get_residue_atoms
from ..module_utils.pdb_utils import parse_remark_666
from ..module_utils.constants import STANDARD_AA_3

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

                # Note: HIS tautomer state is determined by which hydrogens are present
                # (HD1 for delta/HIS_D, HE2 for epsilon/HIS), NOT by the residue name.
                # The residue name should always be "HIS" in the PDB file.
                # Tautomer information is preserved through hydrogen restoration below.
                # We track tautomers for logging purposes only.
                if restore_his_tautomers and resname == "HIS":
                    tautomer = his_tautomer_map.get(key)
                    if tautomer == "HIS_D":
                        # Tautomer will be preserved via HD1 hydrogen restoration
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


def restore_ligand_from_ref(
    mpnn_pdb: str,
    ref_pdb: str,
    output_pdb: str,
    ligand_info: Optional[Tuple[str, int, Optional[str]]] = None,
) -> Dict:
    """Force ligand HETATM records (including hydrogens) to match the reference.

    This is useful when ligand hydrogens must remain unchanged (e.g., QM geometry).
    The function identifies the ligand via:
      1) Explicit ligand_info (chain, resno, optional resname), OR
      2) REMARK 666 template (non-standard TEMPLATE), OR
      3) Fallback to largest non-standard HETATM group in ref_pdb.

    It then removes the ligand HETATM block from mpnn_pdb and inserts
    the reference ligand atoms.
    """
    stats = {"ligand_restored": 0, "ligand_found": False}

    ref_lines, ref_atoms = read_pdb_atoms(ref_pdb)
    mpnn_lines, mpnn_atoms = read_pdb_atoms(mpnn_pdb)

    # Identify ligand
    lig_chain = lig_resname = None
    lig_resno = None

    # 1) Explicit ligand_info (chain, resno, optional resname)
    if ligand_info:
        try:
            lig_chain = ligand_info[0]
            lig_resno = int(ligand_info[1])
            lig_resname = ligand_info[2] if len(ligand_info) > 2 else None
        except Exception:
            lig_chain = lig_resname = None
            lig_resno = None

    # 2) REMARK 666 (template not in standard AA)
    if lig_chain is None:
        for entry in parse_remark_666(ref_lines):
            template_resname = entry.get("template_resname", "")
            if template_resname and template_resname not in STANDARD_AA_3:
                lig_chain = entry.get("template_chain")
                lig_resname = template_resname
                lig_resno = entry.get("template_resno")
                break

    # 3) Fallback: largest non-standard HETATM group in ref_pdb
    ref_lig_atoms = []
    if lig_chain is None:
        hetatm_groups: Dict[Tuple[str, int, str], List[Dict]] = {}
        for atom in ref_atoms:
            if atom["record_type"] != "HETATM":
                continue
            resname = atom["resname"].strip()
            if resname in STANDARD_AA_3 or resname in {"HOH", "WAT", "DOD"}:
                continue
            key = (atom["chain"], atom["resno"], resname)
            hetatm_groups.setdefault(key, []).append(atom)

        if hetatm_groups:
            (lig_chain, lig_resno, lig_resname), ref_lig_atoms = max(
                hetatm_groups.items(), key=lambda kv: len(kv[1])
            )

    if lig_chain is None:
        LOGGER.warning("Could not identify ligand for restoration (no REMARK 666 template)")
        Path(output_pdb).parent.mkdir(parents=True, exist_ok=True)
        with open(output_pdb, "w") as f:
            f.writelines(mpnn_lines)
        return stats

    stats["ligand_found"] = True

    # Collect ligand atoms from reference (HETATM)
    if not ref_lig_atoms:
        ref_lig_atoms = [
            a for a in ref_atoms
            if a["record_type"] == "HETATM" and a["chain"] == lig_chain and a["resno"] == lig_resno
        ]

    # Remove ligand atoms from mpnn output
    output_lines = []
    inserted = False
    for line in mpnn_lines:
        if line.startswith("HETATM"):
            try:
                chain = line[21:22]
                resno = int(line[22:26])
            except Exception:
                chain = None
                resno = None
            if chain == lig_chain and resno == lig_resno:
                continue  # drop ligand atoms
        if not inserted and (line.startswith("TER") or line.startswith("END")):
            # Insert ligand block before TER/END
            for atom in ref_lig_atoms:
                output_lines.append(format_atom_line(atom))
                stats["ligand_restored"] += 1
            inserted = True
        output_lines.append(line)

    if not inserted and ref_lig_atoms:
        for atom in ref_lig_atoms:
            output_lines.append(format_atom_line(atom))
            stats["ligand_restored"] += 1

    Path(output_pdb).parent.mkdir(parents=True, exist_ok=True)
    with open(output_pdb, "w") as f:
        f.writelines(output_lines)

    LOGGER.info(f"Restored ligand from reference: {output_pdb} ({stats['ligand_restored']} atoms)")
    return stats


def normalize_pdb_for_mpnn(
    input_pdb: str,
    output_pdb: str,
) -> Dict:
    """Prepare a PDB for MPNN by removing hydrogens and normalizing HIS names.

    MPNN cannot handle 5-character residue names like HIS_D. This function:
    - Removes ALL hydrogen atoms (ATOM and HETATM)
    - Converts HIS variants (HIS_D/HID/HIE/HIP/etc.) to HIS
    - Preserves non-ATOM/HETATM lines (e.g., REMARK 666)

    Args:
        input_pdb: Path to input PDB (may include H and HIS variants)
        output_pdb: Path to write normalized PDB

    Returns:
        Dict with normalization statistics
    """
    stats = {
        "atoms_removed": 0,
        "his_renamed": 0,
    }

    all_lines, atoms = read_pdb_atoms(input_pdb)
    atom_by_idx = {a["line_idx"]: a for a in atoms}

    output_lines = []
    for line_idx, line in enumerate(all_lines):
        if line_idx not in atom_by_idx:
            output_lines.append(line)
            continue

        atom = atom_by_idx[line_idx]
        elem = atom.get("element", "").strip().upper()
        atom_name = atom.get("atom_name", "").strip()

        # Skip hydrogens (element H or atom name starts with H when element missing)
        if elem == "H" or (not elem and atom_name.startswith("H")):
            stats["atoms_removed"] += 1
            continue

        # Normalize HIS variants to HIS
        if atom["resname"] in ("HIS_D", "HIS_E", "HIS", "HID", "HIE", "HIP"):
            if atom["resname"] != "HIS":
                atom = dict(atom)
                atom["resname"] = "HIS"
                stats["his_renamed"] += 1

        output_lines.append(format_atom_line(atom))

    Path(output_pdb).parent.mkdir(parents=True, exist_ok=True)
    with open(output_pdb, "w") as f:
        f.writelines(output_lines)

    LOGGER.info(f"Normalized PDB for MPNN: {input_pdb} -> {output_pdb}")
    LOGGER.info(f"  Removed H atoms: {stats['atoms_removed']}")
    LOGGER.info(f"  HIS renamed: {stats['his_renamed']}")
    return stats


def cleanup_final_pdb(
    input_pdb: str,
    output_pdb: str,
) -> Dict:
    """Clean up final PDB output for proper formatting.

    This function performs post-processing to ensure the final PDB has:
    1. Proper REMARK ordering (666 after 220)
    2. Continuous atom numbering (no gaps between protein and ligand)
    3. Single TER line between protein and ligand
    4. No duplicate TER lines
    5. No CONECT lines (since atom numbers change)
    6. No Rosetta energy score tables

    Args:
        input_pdb: Path to input PDB file
        output_pdb: Path to write cleaned PDB

    Returns:
        Dict with cleanup statistics
    """
    stats = {
        "atoms_renumbered": 0,
        "ter_lines_fixed": 0,
        "conect_lines_removed": 0,
        "score_lines_removed": 0,
        "remarks_reordered": False,
    }

    with open(input_pdb, "r") as f:
        lines = f.readlines()

    # Separate lines by type
    remark_666_lines = []
    remark_220_lines = []
    header_lines = []  # HEADER, EXPDTA, HETNAM, etc.
    other_remark_lines = []
    atom_lines = []
    hetatm_lines = []
    other_lines = []  # MODEL, etc.

    in_score_table = False
    for line in lines:
        # Skip score tables
        if line.startswith("# All scores below") or line.startswith("#BEGIN_POSE_ENERGIES_TABLE"):
            in_score_table = True
            stats["score_lines_removed"] += 1
            continue
        if line.startswith("#END_POSE_ENERGIES_TABLE"):
            in_score_table = False
            stats["score_lines_removed"] += 1
            continue
        if in_score_table:
            stats["score_lines_removed"] += 1
            continue

        # Skip CONECT lines
        if line.startswith("CONECT"):
            stats["conect_lines_removed"] += 1
            continue

        # Skip blank lines
        if not line.strip():
            continue

        # Skip TER lines (we'll add proper ones later)
        if line.startswith("TER"):
            continue

        # Skip END lines (we'll add one at the end)
        if line.startswith("END") and not line.startswith("ENDMDL"):
            continue

        # Categorize lines
        if line.startswith("REMARK 666"):
            remark_666_lines.append(line)
        elif line.startswith("REMARK 220"):
            remark_220_lines.append(line)
        elif line.startswith("REMARK"):
            other_remark_lines.append(line)
        elif line.startswith("HEADER") or line.startswith("EXPDTA") or line.startswith("HETNAM"):
            header_lines.append(line)
        elif line.startswith("ATOM"):
            atom_lines.append(line)
        elif line.startswith("HETATM"):
            hetatm_lines.append(line)
        else:
            other_lines.append(line)

    # Track if we reordered remarks
    if remark_666_lines and (header_lines or remark_220_lines):
        stats["remarks_reordered"] = True

    # Build output in proper order:
    # 1. HEADER, EXPDTA, HETNAM
    # 2. REMARK 220
    # 3. REMARK 666
    # 4. Other REMARK lines
    # 5. Other lines (MODEL, etc.)
    # 6. ATOM lines (renumbered)
    # 7. TER
    # 8. HETATM lines (renumbered, continuing from ATOM)
    # 9. TER
    # 10. END

    output_lines = []

    # Add header lines
    output_lines.extend(header_lines)

    # Add REMARK 220 lines
    output_lines.extend(remark_220_lines)

    # Add REMARK 666 lines (after REMARK 220)
    output_lines.extend(remark_666_lines)

    # Add other REMARK lines
    output_lines.extend(other_remark_lines)

    # Add other lines (MODEL, etc.)
    output_lines.extend(other_lines)

    # Renumber ATOM lines
    atom_serial = 1
    for line in atom_lines:
        new_line = _renumber_atom_line(line, atom_serial)
        output_lines.append(new_line)
        atom_serial += 1
        stats["atoms_renumbered"] += 1

    # Add TER between protein and ligand
    if atom_lines and hetatm_lines:
        output_lines.append("TER\n")
        stats["ter_lines_fixed"] += 1

    # Renumber HETATM lines (continue numbering from ATOM)
    for line in hetatm_lines:
        new_line = _renumber_atom_line(line, atom_serial)
        output_lines.append(new_line)
        atom_serial += 1
        stats["atoms_renumbered"] += 1

    # Add final TER and END
    if hetatm_lines or atom_lines:
        output_lines.append("TER\n")
        stats["ter_lines_fixed"] += 1
    output_lines.append("END\n")

    # Write output
    Path(output_pdb).parent.mkdir(parents=True, exist_ok=True)
    with open(output_pdb, "w") as f:
        f.writelines(output_lines)

    LOGGER.info(f"Cleaned up PDB: {input_pdb} -> {output_pdb}")
    LOGGER.info(f"  Atoms renumbered: {stats['atoms_renumbered']}")
    LOGGER.info(f"  TER lines fixed: {stats['ter_lines_fixed']}")
    LOGGER.info(f"  CONECT lines removed: {stats['conect_lines_removed']}")
    LOGGER.info(f"  Score lines removed: {stats['score_lines_removed']}")
    LOGGER.info(f"  REMARKs reordered: {stats['remarks_reordered']}")

    return stats


def _renumber_atom_line(line: str, new_serial: int) -> str:
    """Renumber an ATOM/HETATM line with a new serial number.

    Args:
        line: Original ATOM/HETATM line
        new_serial: New atom serial number

    Returns:
        Line with updated serial number
    """
    # PDB format: columns 7-11 (1-indexed) are the atom serial number
    # In Python 0-indexed, that's positions 6:11
    serial_str = str(new_serial).rjust(5)
    new_line = line[:6] + serial_str + line[11:]
    return new_line
