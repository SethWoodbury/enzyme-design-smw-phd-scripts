"""Step 1: align by ligand and fix catalytic residues."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from utils.constants import (
    BACKBONE_ATOMS,
    CHARGED_RESIDUES,
    COVALENT_DISTANCE_MAX,
    DEFAULT_PROBABILITY_OF_MUTATION,
    ELECTROSTATIC_DISTANCE_MAX,
    HBOND_DISTANCE_MAX,
    HYDROPHOBIC_DISTANCE_MAX,
    METAL_DISTANCE_MAX,
    NONPOLAR_RESIDUES,
    PI_DISTANCE_MAX,
)
from utils.geometry import Transform, kabsch_align, rmsd
from utils.logging_utils import (
    configure_logging,
    is_verbose,
    print_dict_summary,
    print_key_value,
    print_list_item,
    print_section_header,
    print_subsection_header,
)
from utils.models import CatalyticResidueInfo, Interaction, ResidueId, ResidueRecord
from utils.pdbio import AtomRecord, PdbStructure, find_template_id, parse_remark_666

LOGGER = logging.getLogger(__name__)


def _apply_transform(atom: AtomRecord, transform: Transform) -> None:
    coord = np.array([atom.x, atom.y, atom.z])
    new_coord = transform.rotation @ coord + transform.translation
    atom.set_coord(new_coord)


def _split_indices(indices: Optional[str]) -> List[int]:
    if not indices:
        return []
    return [int(idx.strip()) for idx in indices.split(",") if idx.strip()]


def _guess_component(atom: AtomRecord) -> str:
    return "backbone" if atom.name in BACKBONE_ATOMS else "sidechain"


def _distance(a: AtomRecord, b: AtomRecord) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def _classify_interactions(
    atom: AtomRecord,
    partner: AtomRecord,
    residue_name: str,
    partner_resname: str,
    partner_entity: str,
    distance: float,
) -> List[str]:
    interactions: List[str] = []
    element = atom.element.upper()
    partner_element = partner.element.upper()

    if distance <= METAL_DISTANCE_MAX and ("ZN" in (element, partner_element) or "FE" in (element, partner_element)):
        interactions.append("metal_coordination")

    if distance <= HBOND_DISTANCE_MAX and element in {"N", "O", "S"} and partner_element in {"N", "O", "S"}:
        if element == "N" and partner_element in {"O", "N", "S"}:
            interactions.append("hbond_donation")
        if element in {"O", "S"} and partner_element == "N":
            interactions.append("hbond_acceptance")

    if distance <= ELECTROSTATIC_DISTANCE_MAX and residue_name in CHARGED_RESIDUES:
        if partner_element not in {"C", ""}:
            interactions.append("electrostatic")

    if (
        distance <= HYDROPHOBIC_DISTANCE_MAX
        and residue_name in NONPOLAR_RESIDUES
        and partner_element in {"C", "S"}
    ):
        interactions.append("nonpolar_interaction")

    if distance <= PI_DISTANCE_MAX and residue_name in {"PHE", "TYR", "TRP", "HIS"}:
        interactions.append("pi_interaction")

    if distance <= COVALENT_DISTANCE_MAX:
        if element in {"N", "O", "S"} and partner_element == "H":
            interactions.append("acid_base_mod")
        else:
            interactions.append("post_translational_mod")

    if not interactions:
        return []

    LOGGER.debug(
        "Interaction candidate %s-%s (%s) to %s (%s) at %.2f",
        residue_name,
        atom.name,
        atom.element,
        partner_entity,
        partner_element,
        distance,
    )
    return interactions


def _collect_interactions(
    residue_atoms: Sequence[AtomRecord],
    partner_atoms: Sequence[AtomRecord],
    partner_entity: str,
    verbose_residue_id: str = "",
) -> List[Interaction]:
    interactions: List[Interaction] = []
    residue_name = residue_atoms[0].resname
    for atom in residue_atoms:
        for partner in partner_atoms:
            dist = _distance(atom, partner)
            interaction_types = _classify_interactions(
                atom,
                partner,
                residue_name,
                partner.resname,
                partner_entity,
                dist,
            )
            for interaction_type in interaction_types:
                interactions.append(
                    Interaction(
                        interaction_type=interaction_type,
                        from_component=_guess_component(atom),
                        from_atom=atom.name,
                        to_entity=partner_entity,
                        to_atom=partner.name,
                        distance=dist,
                    )
                )

    if is_verbose() and interactions and verbose_residue_id:
        for interaction in interactions:
            LOGGER.debug(
                "  [%s] %s: %s-%s -> %s-%s (%.2f A)",
                verbose_residue_id,
                interaction.interaction_type,
                interaction.from_component,
                interaction.from_atom,
                interaction.to_entity,
                interaction.to_atom,
                interaction.distance,
            )

    return interactions


def _summarize_important_component(interactions: List[Interaction]) -> str:
    backbone = any(interaction.from_component == "backbone" for interaction in interactions)
    sidechain = any(interaction.from_component == "sidechain" for interaction in interactions)
    if backbone and sidechain:
        return "both"
    if backbone:
        return "backbone"
    if sidechain:
        return "sidechain"
    return "sidechain"


def _align_ref_to_input(
    ref_struct: PdbStructure,
    input_struct: PdbStructure,
    template_id: Tuple[str, str, int],
) -> Tuple[Transform, float]:
    chain, resname, resno = template_id

    if is_verbose():
        print_subsection_header("Ligand Alignment (Kabsch Algorithm)")
        print_key_value("Template Ligand", f"Chain {chain}, {resname} {resno}")

    ref_atoms = [
        atom for atom in ref_struct.residue_atoms(chain, resno) if atom.resname == resname
    ]
    input_atoms = [
        atom for atom in input_struct.residue_atoms(chain, resno) if atom.resname == resname
    ]

    if is_verbose():
        print_key_value("Reference Ligand Atoms", str(len(ref_atoms)))
        print_key_value("Input Ligand Atoms", str(len(input_atoms)))

    if not ref_atoms or not input_atoms:
        raise ValueError("Template ligand atoms not found in both PDBs.")

    ref_by_name = {atom.name: atom for atom in ref_atoms}
    input_by_name = {atom.name: atom for atom in input_atoms}
    common = sorted(set(ref_by_name) & set(input_by_name))

    if is_verbose():
        print_key_value("Common Atoms for Alignment", str(len(common)))
        if len(common) <= 10:
            print_key_value("Common Atom Names", ", ".join(common))

    if len(common) < 3:
        raise ValueError("Need at least 3 common ligand atoms for alignment.")

    ref_coords = np.array([ref_by_name[name].coord() for name in common])
    input_coords = np.array([input_by_name[name].coord() for name in common])

    if is_verbose():
        print()
        print("  Performing Kabsch superposition...")
        ref_centroid = ref_coords.mean(axis=0)
        input_centroid = input_coords.mean(axis=0)
        print_key_value("Reference Centroid", f"({ref_centroid[0]:.3f}, {ref_centroid[1]:.3f}, {ref_centroid[2]:.3f})")
        print_key_value("Input Centroid", f"({input_centroid[0]:.3f}, {input_centroid[1]:.3f}, {input_centroid[2]:.3f})")

    transform = kabsch_align(ref_coords, input_coords)
    aligned_coords = (transform.rotation @ ref_coords.T).T + transform.translation
    alignment_rmsd = rmsd(aligned_coords, input_coords)

    LOGGER.info("Ligand alignment RMSD: %.4f", alignment_rmsd)

    if is_verbose():
        print()
        print_key_value("Alignment RMSD", f"{alignment_rmsd:.4f} Angstroms")
        print()
        print("  Rotation Matrix:")
        for i, row in enumerate(transform.rotation):
            print(f"    [{row[0]:8.5f}, {row[1]:8.5f}, {row[2]:8.5f}]")
        print_key_value("Translation Vector",
                       f"({transform.translation[0]:.4f}, {transform.translation[1]:.4f}, {transform.translation[2]:.4f})")
        print()
        print(f"  Applying transform to all {len(ref_struct.atoms)} atoms in reference structure...")

    for atom in ref_struct.atoms:
        _apply_transform(atom, transform)

    if is_verbose():
        print("  Transform applied successfully.")

    return transform, alignment_rmsd


def _apply_catres_coordinates(
    input_struct: PdbStructure,
    ref_struct: PdbStructure,
    residue_id: ResidueId,
    important_component: str,
) -> None:
    """Transfer coordinates from reference to input for a catalytic residue.

    Args:
        input_struct: The input PDB structure (will be modified)
        ref_struct: The reference PDB structure (aligned)
        residue_id: The residue identifier
        important_component: Which component to transfer ("backbone", "sidechain", or "both")
    """
    input_atoms = input_struct.residue_atoms(residue_id.chain, residue_id.resno, residue_id.icode)
    ref_atoms = ref_struct.residue_atoms(residue_id.chain, residue_id.resno, residue_id.icode)

    if not input_atoms or not ref_atoms:
        LOGGER.warning(
            "Residue %s not found in one of the PDBs; skipping coordinate transfer.",
            residue_id.identifier,
        )
        return

    ref_by_name = {atom.name: atom for atom in ref_atoms}
    transferred_count = 0

    for atom in input_atoms:
        should_transfer = False
        if important_component in {"backbone", "both"}:
            if atom.name in ref_by_name:
                should_transfer = True
        else:  # sidechain only
            if atom.name not in BACKBONE_ATOMS and atom.name in ref_by_name:
                should_transfer = True

        if should_transfer:
            old_coord = atom.coord()
            new_coord = ref_by_name[atom.name].coord()
            atom.set_coord(new_coord)
            transferred_count += 1
            LOGGER.debug(
                "  Transferred %s %s: (%.3f, %.3f, %.3f) -> (%.3f, %.3f, %.3f)",
                residue_id.identifier,
                atom.name,
                old_coord[0], old_coord[1], old_coord[2],
                new_coord[0], new_coord[1], new_coord[2],
            )

    LOGGER.debug(
        "Transferred %d atoms for residue %s (component: %s)",
        transferred_count,
        residue_id.identifier,
        important_component,
    )


def _build_residue_records(
    input_struct: PdbStructure,
    ref_struct: PdbStructure,
    remark_entries: List[Dict[str, str]],
    catres_subset: List[int],
    ligand_atoms: List[AtomRecord],
) -> Dict[str, ResidueRecord]:
    if is_verbose():
        print_subsection_header("Building Residue Records")
        print_key_value("Total REMARK 666 Entries", str(len(remark_entries)))
        print_key_value("Catres Subset Filter", str(catres_subset) if catres_subset else "None (all)")
        print_key_value("Ligand Atoms for Interaction Analysis", str(len(ligand_atoms)))

    records: Dict[str, ResidueRecord] = {}
    catres_entries = []
    for entry in remark_entries:
        block_index = int(entry["block_index"])
        if catres_subset and block_index not in catres_subset:
            continue
        catres_entries.append(entry)

    if is_verbose():
        print_key_value("Catalytic Residue Entries (after filter)", str(len(catres_entries)))

    catres_keys = {
        (entry["motif_chain"], int(entry["motif_resno"]), entry.get("motif_resname", ""))
        for entry in catres_entries
    }

    if is_verbose():
        print()
        print("  Catalytic Residue Keys:")
        for chain, resno, resname in sorted(catres_keys):
            print(f"    Chain {chain}, {resname} {resno}")

    for (chain, resno, icode), atoms in input_struct.residues().items():
        resname = atoms[0].resname
        residue_id = ResidueId(chain=chain, resno=resno, icode=icode, resname=resname)
        identifier = residue_id.identifier
        is_catres = (chain, resno, resname) in catres_keys
        motif_label = "catalytic_motif" if is_catres else "conserved_motif"

        record = ResidueRecord(
            residue_id=residue_id,
            probability_of_mutation=DEFAULT_PROBABILITY_OF_MUTATION,
            catres_subset=is_catres,
            motif_label=motif_label,
        )
        records[identifier] = record

    if is_verbose():
        print()
        print_key_value("Total Residue Records Created", str(len(records)))

    catres_atoms_by_id: Dict[str, List[AtomRecord]] = {}
    for entry in catres_entries:
        chain = entry["motif_chain"]
        resno = int(entry["motif_resno"])
        atoms = input_struct.residue_atoms(chain, resno)
        if not atoms:
            if is_verbose():
                print(f"    Warning: No atoms found for {chain}{resno}")
            continue
        residue_id = ResidueId(chain=chain, resno=resno, icode="", resname=atoms[0].resname)
        catres_atoms_by_id[residue_id.identifier] = atoms

    if is_verbose():
        print_subsection_header("Interaction Analysis for Catalytic Residues")
        print(f"  Analyzing {len(catres_atoms_by_id)} catalytic residues...")
        print()

    for identifier, atoms in catres_atoms_by_id.items():
        if is_verbose():
            resname = atoms[0].resname
            print(f"  Processing {identifier} ({resname}, {len(atoms)} atoms)...")

        interaction_set: List[Interaction] = []

        # Collect ligand interactions
        ligand_interactions = _collect_interactions(atoms, ligand_atoms, "ligand", identifier)
        interaction_set.extend(ligand_interactions)

        if is_verbose():
            print(f"    Ligand interactions: {len(ligand_interactions)}")

        # Collect inter-residue and intra-residue interactions
        intra_count = 0
        inter_count = 0
        for other_id, other_atoms in catres_atoms_by_id.items():
            if other_id == identifier:
                backbone_atoms = [atom for atom in atoms if atom.name in BACKBONE_ATOMS]
                sidechain_atoms = [atom for atom in atoms if atom.name not in BACKBONE_ATOMS]
                intra_interactions = _collect_interactions(backbone_atoms, sidechain_atoms, identifier, identifier)
                interaction_set.extend(intra_interactions)
                intra_count = len(intra_interactions)
                continue
            inter_interactions = _collect_interactions(atoms, other_atoms, other_id, identifier)
            interaction_set.extend(inter_interactions)
            inter_count += len(inter_interactions)

        if is_verbose():
            print(f"    Intra-residue interactions: {intra_count}")
            print(f"    Inter-residue interactions: {inter_count}")
            print(f"    Total interactions: {len(interaction_set)}")

        important_component = _summarize_important_component(interaction_set)

        if is_verbose():
            print(f"    Important component: {important_component}")

        residue_record = records.get(identifier)
        if residue_record:
            residue_record.catres_subset = True
            residue_record.motif_label = "catalytic_motif"
            residue_record.catres_subset_info = CatalyticResidueInfo(
                important_component=important_component,
                interactions_found=interaction_set,
            )
        else:
            LOGGER.warning("Missing residue record for %s", identifier)

        if residue_record:
            if is_verbose():
                print(f"    Transferring coordinates from reference...")
            _apply_catres_coordinates(
                input_struct,
                ref_struct,
                residue_record.residue_id,
                important_component,
            )
            if is_verbose():
                print(f"    Coordinate transfer complete.")
        print()

    LOGGER.info(
        "Cataloged %d residues (%d catalytic).",
        len(records),
        len(catres_atoms_by_id),
    )

    if is_verbose():
        print_subsection_header("Residue Record Summary")
        print_key_value("Total Residues", str(len(records)))
        print_key_value("Catalytic Residues", str(len(catres_atoms_by_id)))

        # Summarize interaction types
        interaction_type_counts: Dict[str, int] = {}
        for identifier, atoms in catres_atoms_by_id.items():
            record = records.get(identifier)
            if record and record.catres_subset_info:
                for interaction in record.catres_subset_info.interactions_found:
                    itype = interaction.interaction_type
                    interaction_type_counts[itype] = interaction_type_counts.get(itype, 0) + 1

        if interaction_type_counts:
            print()
            print("  Interaction Type Summary:")
            for itype, count in sorted(interaction_type_counts.items(), key=lambda x: -x[1]):
                print(f"    {itype}: {count}")

    return records


def _validate_remark_consistency(
    ref_entries: List[Dict[str, str]],
    input_entries: List[Dict[str, str]],
) -> None:
    """Validate that REMARK 666 entries are consistent between reference and input PDBs."""
    ref_set = {
        (e["motif_chain"], e["motif_resname"], int(e["motif_resno"]), int(e["block_index"]))
        for e in ref_entries
    }
    input_set = {
        (e["motif_chain"], e["motif_resname"], int(e["motif_resno"]), int(e["block_index"]))
        for e in input_entries
    }

    if is_verbose():
        print()
        print("  Validating REMARK 666 consistency...")
        print_key_value("Reference Motifs", str(len(ref_set)))
        print_key_value("Input Motifs", str(len(input_set)))

    if ref_set != input_set:
        LOGGER.warning("REMARK 666 motif sets differ between ref and input.")
        if is_verbose():
            in_ref_only = ref_set - input_set
            in_input_only = input_set - ref_set
            if in_ref_only:
                print("  WARNING: Motifs only in reference:")
                for chain, resname, resno, block in sorted(in_ref_only):
                    print(f"    Block {block}: Chain {chain} {resname} {resno}")
            if in_input_only:
                print("  WARNING: Motifs only in input:")
                for chain, resname, resno, block in sorted(in_input_only):
                    print(f"    Block {block}: Chain {chain} {resname} {resno}")
    elif is_verbose():
        print("  REMARK 666 entries are consistent.")


def run_step1(
    input_pdb: Path,
    ref_pdb: Path,
    output_pdb: Path,
    output_json: Optional[Path],
    catres_subset: Optional[str],
    ligand_chain: Optional[str],
    ligand_resname: Optional[str],
    ligand_resno: Optional[int],
) -> None:
    """Execute Stage 1: Active Site Remastering.

    This stage:
    1. Loads input and reference PDB structures
    2. Parses REMARK 666 entries for motif information
    3. Aligns the reference to the input using the template ligand (Kabsch algorithm)
    4. Identifies and analyzes catalytic residue interactions
    5. Transfers coordinates from reference to input for catalytic residues
    6. Writes the modified PDB and a JSON catalog of catalytic residues
    """
    if is_verbose():
        print_section_header("STAGE 1: ACTIVE SITE REMASTERING")
        print_subsection_header("Stage 1 Initialization")
        print_key_value("Input PDB", str(input_pdb))
        print_key_value("Reference PDB", str(ref_pdb))
        print_key_value("Output PDB", str(output_pdb))
        print_key_value("Output JSON", str(output_json) if output_json else "None")
        print_key_value("Catres Subset", catres_subset if catres_subset else "All")
        if ligand_chain:
            print_key_value("Ligand Override", f"Chain={ligand_chain}, Name={ligand_resname}, No={ligand_resno}")

    # ==================== STEP 1.1: Load PDB Structures ====================
    if is_verbose():
        print_subsection_header("Step 1.1: Loading PDB Structures")

    LOGGER.info("Loading input PDB: %s", input_pdb)
    input_struct = PdbStructure.from_file(str(input_pdb))

    if is_verbose():
        print_key_value("Input PDB", str(input_pdb))
        print_key_value("  Total Lines", str(len(input_struct.lines)))
        print_key_value("  Total Atoms", str(len(input_struct.atoms)))
        print_key_value("  Total Residues", str(len(input_struct.residues())))
        chains = set(atom.chain for atom in input_struct.atoms)
        print_key_value("  Chains", ", ".join(sorted(chains)) if chains else "None")

    LOGGER.info("Loading reference PDB: %s", ref_pdb)
    ref_struct = PdbStructure.from_file(str(ref_pdb))

    if is_verbose():
        print_key_value("Reference PDB", str(ref_pdb))
        print_key_value("  Total Lines", str(len(ref_struct.lines)))
        print_key_value("  Total Atoms", str(len(ref_struct.atoms)))
        print_key_value("  Total Residues", str(len(ref_struct.residues())))
        chains = set(atom.chain for atom in ref_struct.atoms)
        print_key_value("  Chains", ", ".join(sorted(chains)) if chains else "None")

    # ==================== STEP 1.2: Parse REMARK 666 Entries ====================
    if is_verbose():
        print_subsection_header("Step 1.2: Parsing REMARK 666 Motif Information")

    input_entries = parse_remark_666(input_struct.lines)
    ref_entries = parse_remark_666(ref_struct.lines)

    if is_verbose():
        print_key_value("Input REMARK 666 Entries", str(len(input_entries)))
        print_key_value("Reference REMARK 666 Entries", str(len(ref_entries)))

        if input_entries:
            print()
            print("  Input Motif Entries (first 10):")
            for i, entry in enumerate(input_entries[:10]):
                print(f"    Block {entry['block_index']}: "
                      f"Chain {entry['motif_chain']} {entry['motif_resname']} {entry['motif_resno']}")
            if len(input_entries) > 10:
                print(f"    ... and {len(input_entries) - 10} more entries")

    if not input_entries or not ref_entries:
        raise ValueError("REMARK 666 lines not found in one or both PDBs.")

    _validate_remark_consistency(ref_entries, input_entries)

    # ==================== STEP 1.3: Identify Template Ligand ====================
    if is_verbose():
        print_subsection_header("Step 1.3: Identifying Template Ligand")

    template_id = find_template_id(input_entries)

    if is_verbose():
        if template_id:
            print_key_value("Auto-detected Template", f"Chain {template_id[0]}, {template_id[1]} {template_id[2]}")
        else:
            print("  No template auto-detected from REMARK 666")

    if ligand_chain and ligand_resname and ligand_resno:
        template_id = (ligand_chain, ligand_resname, ligand_resno)
        if is_verbose():
            print_key_value("Using Override Template", f"Chain {ligand_chain}, {ligand_resname} {ligand_resno}")

    if not template_id:
        raise ValueError("Template ligand ID not found; provide --ligand_chain/resname/resno.")

    LOGGER.info("Using ligand template: chain=%s resname=%s resno=%s", *template_id)

    # ==================== STEP 1.4: Align Reference to Input ====================
    if is_verbose():
        print_subsection_header("Step 1.4: Aligning Reference to Input")
        print("  Using Kabsch algorithm for optimal superposition")

    transform, alignment_rmsd = _align_ref_to_input(ref_struct, input_struct, template_id)

    # ==================== STEP 1.5: Extract Ligand Atoms ====================
    if is_verbose():
        print_subsection_header("Step 1.5: Extracting Ligand Atoms")

    ligand_atoms = [
        atom
        for atom in input_struct.residue_atoms(template_id[0], template_id[2])
        if atom.resname == template_id[1]
    ]

    if is_verbose():
        print_key_value("Ligand Atoms Found", str(len(ligand_atoms)))
        if ligand_atoms:
            unique_elements = set(atom.element for atom in ligand_atoms)
            print_key_value("Unique Elements", ", ".join(sorted(unique_elements)))
            # Show a few atom names
            atom_names = [atom.name for atom in ligand_atoms[:10]]
            print_key_value("Atom Names (first 10)", ", ".join(atom_names))

    if not ligand_atoms:
        raise ValueError("Ligand atoms not found in input PDB.")

    # ==================== STEP 1.6: Build Residue Records ====================
    if is_verbose():
        print_subsection_header("Step 1.6: Building Residue Records & Analyzing Interactions")

    subset_indices = _split_indices(catres_subset)
    if subset_indices:
        LOGGER.info("Using catres subset blocks: %s", ",".join(str(idx) for idx in subset_indices))
        if is_verbose():
            print_key_value("Catres Subset Blocks", ", ".join(str(idx) for idx in subset_indices))
    else:
        LOGGER.info("Using all REMARK 666 motif residues as catres subset.")
        if is_verbose():
            print("  Using ALL motif residues (no subset filter)")

    records = _build_residue_records(
        input_struct,
        ref_struct,
        input_entries,
        subset_indices,
        ligand_atoms,
    )

    # ==================== STEP 1.7: Write Output PDB ====================
    if is_verbose():
        print_subsection_header("Step 1.7: Writing Output PDB")
        print_key_value("Output Path", str(output_pdb))

    input_struct.write(str(output_pdb))
    LOGGER.info("Wrote modified PDB to %s", output_pdb)

    if is_verbose():
        if output_pdb.exists():
            print_key_value("File Written", "Yes")
            print_key_value("File Size", f"{output_pdb.stat().st_size:,} bytes")
        else:
            print_key_value("File Written", "ERROR - File not found after write!")

    # ==================== STEP 1.8: Write JSON Catalog ====================
    if output_json:
        if is_verbose():
            print_subsection_header("Step 1.8: Writing Catalytic Residue Catalog")
            print_key_value("Output Path", str(output_json))

        catres_only = {
            identifier: record
            for identifier, record in records.items()
            if record.catres_subset and record.catres_subset_info
        }

        if is_verbose():
            print_key_value("Catalytic Residues in Catalog", str(len(catres_only)))

        payload = {}
        for identifier, record in catres_only.items():
            payload[identifier] = {
                "chain": record.residue_id.chain,
                "residue_num": record.residue_id.resno,
                "identifier": record.residue_id.identifier,
                "res_type": record.residue_id.resname,
                "important_component": record.catres_subset_info.important_component,
                "interactions_found": [
                    interaction.to_dict()
                    for interaction in record.catres_subset_info.interactions_found
                ],
            }
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("Wrote catalytic residue catalog to %s", output_json)

        if is_verbose():
            if output_json.exists():
                print_key_value("File Written", "Yes")
                print_key_value("File Size", f"{output_json.stat().st_size:,} bytes")

                # Summarize output
                print()
                print("  Catalog Contents Summary:")
                for res_id, data in payload.items():
                    num_interactions = len(data.get("interactions_found", []))
                    component = data.get("important_component", "unknown")
                    print(f"    {res_id} ({data['res_type']}): "
                          f"{num_interactions} interactions, component: {component}")
            else:
                print_key_value("File Written", "ERROR - File not found after write!")

    # ==================== STAGE 1 COMPLETE ====================
    if is_verbose():
        print_section_header("STAGE 1 COMPLETE")
        print_key_value("Output PDB", str(output_pdb))
        if output_json:
            print_key_value("Output JSON", str(output_json))
        print_key_value("Alignment RMSD", f"{alignment_rmsd:.4f} Angstroms")
        print_key_value("Total Residues Processed", str(len(records)))
        catres_count = sum(1 for r in records.values() if r.catres_subset)
        print_key_value("Catalytic Residues", str(catres_count))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 1: fix catalytic residues.")
    parser.add_argument("--input_pdb", type=Path)
    parser.add_argument("--ref_pdb", type=Path)
    parser.add_argument("--output_pdb", type=Path)
    parser.add_argument("--output_json", type=Path)
    parser.add_argument("--test", action="store_true", help="Run stage using test config.")
    parser.add_argument(
        "--test_config",
        type=Path,
        default=Path("tests/stage1_test_config.json"),
        help="Path to stage 1 test config JSON.",
    )
    parser.add_argument("--catres_subset", type=str, help="Comma-separated block indices.")
    parser.add_argument("--ligand_chain", type=str)
    parser.add_argument("--ligand_resname", type=str)
    parser.add_argument("--ligand_resno", type=int)
    parser.add_argument("--verbose", action="count", default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Enable verbose by default for --test
    if args.test and args.verbose == 0:
        args.verbose = 1

    configure_logging(args.verbose)

    if args.test:
        if not args.test_config.exists():
            raise SystemExit(f"Test config not found: {args.test_config}")
        config = json.loads(args.test_config.read_text(encoding="utf-8"))
        args.input_pdb = Path(config["input_pdb"])
        args.ref_pdb = Path(config["ref_pdb"])
        outdir = Path(config.get("outdir", "tests/outdir_stage1_test"))
        outdir.mkdir(parents=True, exist_ok=True)
        args.output_pdb = outdir / "stage1_fixed.pdb"
        args.output_json = outdir / "stage1_catres_catalog.json"
        args.catres_subset = config.get("catres_subset")
        LOGGER.info("Loaded test config from %s", args.test_config)

        if is_verbose():
            print_section_header("STAGE 1 TEST MODE")
            print_subsection_header("Test Configuration")
            print_key_value("Config File", str(args.test_config))
            for key, value in config.items():
                print_key_value(key, str(value))

    if not args.input_pdb or not args.ref_pdb or not args.output_pdb:
        raise SystemExit("--input_pdb, --ref_pdb, and --output_pdb are required (unless --test).")

    run_step1(
        input_pdb=args.input_pdb,
        ref_pdb=args.ref_pdb,
        output_pdb=args.output_pdb,
        output_json=args.output_json,
        catres_subset=args.catres_subset,
        ligand_chain=args.ligand_chain,
        ligand_resname=args.ligand_resname,
        ligand_resno=args.ligand_resno,
    )


if __name__ == "__main__":
    main()
