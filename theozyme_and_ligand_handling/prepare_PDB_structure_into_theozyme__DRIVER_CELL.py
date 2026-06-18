#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prepare_PDB_structure_into_theozyme__DRIVER_CELL.py

Importable driver module for prepare_PDB_structure_into_theozyme.py.

Usage (Python / notebook):
    from prepare_PDB_structure_into_theozyme__DRIVER_CELL import build_command
    cmd = build_command(input_pdb="...", output_pdb="...", ...)

Usage (notebook cell — copy the template from
    prepare_PDB_structure_into_theozyme__DRIVER_CELL_TEMPLATE.txt)

build_command(**kwargs) -> list[str]
    Builds the CLI argument list.  Appends a flag ONLY when the corresponding
    variable is truthy / non-empty (preserve-by-default: omitting a flag is safe
    and changes nothing).  Prints the joined command and — when write_command_txt
    is set — writes the single-line command to that path for easy terminal paste.

Author: Seth M. Woodbury, David Baker Lab, UW (woodbuse@uw.edu)
"""

import os
import shlex

MAIN_SCRIPT = ("/home/woodbuse/special_scripts/theozyme_and_ligand_handling/"
               "prepare_PDB_structure_into_theozyme.py")


def build_command(
    # ── required ──────────────────────────────────────────────────────────────
    input_pdb,
    output_pdb,
    # ── REMARK 666 ordering & management ──────────────────────────────────────
    remark666_exclude_residues=None,    # list[str] e.g. ["A131"]  → --remark666_exclude_residues
    remark_front=None,                  # list[str] e.g. ["A55","A57"]  → --remark666_residue_front_order
    remark_back=None,                   # list[str] → --remark666_residue_back_order
    complete_remark666=False,           # bool → --complete_remark666
    force_regenerate_remark666=False,   # bool → --force_regenerate_remark666
    remark666_template_ligand=None,     # str e.g. "SUB" → --remark666_template_ligand
    remark666_template_chain=None,      # str → --remark666_template_chain
    remark666_template_resi=None,       # str → --remark666_template_resi
    clean_remarks=False,                # bool → --clean_remarks
    # ── filtering ─────────────────────────────────────────────────────────────
    residues_to_keep=None,              # list[str] selectors → --residues_to_keep  (mutually exclusive with throw_away)  e.g. ["A55","A57"]
    residues_to_throw_away=None,        # list[str] selectors → --residues_to_throw_away  e.g. ["A169"] or ["Z:HOH:100"]
    ligands_to_keep=None,               # list[str] selectors → --ligands_to_keep  e.g. ["Z:SUB:999"]
    ligands_to_throw_away=None,         # list[str] selectors → --ligands_to_throw_away  e.g. ["Z:HOH:100"]
    # ── ncAA handling ─────────────────────────────────────────────────────────
    frag_ncAA=None,                     # list[str] or [] → --frag_ncAA_into_cAA_plus_lig  ([] = all ncAAs)
    protect_ncAA_from_ligandization=False,  # bool → --protect_ncAA_from_ligandization
    leave_ncAA_as_ATOM=False,           # bool → --leave_ncAA_as_ATOM
    add_CA_to_labeled_frag=False,       # bool → --add_CA_to_labeled_frag
    protect_sidechain_polarH=None,      # list[str] or [] → --protect_sidechain_polarH  ([] = all)
    disable_intelligent_hstrip=False,   # bool → --disable_intelligent_hstrip
    # ── merge ligands ─────────────────────────────────────────────────────────
    merge_ligands_as=None,              # str e.g. "LIG" → --merge_ligands_as  (also enables --theozyme_layout)
    merged_ligand_chain=None,           # str → --merged_ligand_chain
    merged_ligand_resseq=None,          # str → --merged_ligand_resseq
    merge_only=None,                    # list[str] selectors → --merge_only  e.g. ["A901","Z:SUB:999"] — bare codes like "LIG" are invalid
    # ── legacy opt-in cleaning (preserve-by-default: OFF by default) ──────────
    strip_insertion_codes=False,        # bool → --strip_insertion_codes
    strip_protein_hydrogens=False,      # bool → --strip_protein_hydrogens
    blank_segid=False,                  # bool → --blank_segid
    strip_partial_charges=False,        # bool → --strip_partial_charges
    strip_formal_charges=False,         # bool → --strip_formal_charges
    # ── theozyme layout ───────────────────────────────────────────────────────
    theozyme_layout=False,              # bool → --theozyme_layout
    # ── renumber / checks ─────────────────────────────────────────────────────
    renumber_atoms=False,               # bool → --renumber_atoms
    no_ligand_code_checks=False,        # bool → --no_ligand_code_checks
    preserve_waters=False,              # bool → --preserve_waters (HOH special: keep + no-merge)
    rosetta_residue_types=None,         # str (path) → --rosetta_residue_types
    ccd_timeout=None,                   # float → --ccd_timeout
    verbose=False,                      # bool → --verbose
    # ── output ────────────────────────────────────────────────────────────────
    write_command_txt=None,             # str (path) → writes single-line command to this .txt
):
    """Build the CLI argument list for prepare_PDB_structure_into_theozyme.py.

    Each flag is only appended when the corresponding variable is truthy /
    non-empty (preserve-by-default: leaving a toggle unset changes nothing).

    Returns:
        list[str] — the full argv suitable for subprocess.run() or shlex.join().
    Prints:
        The joined command string.
    Side-effect (optional):
        When write_command_txt is set, writes the single-line command to that
        path (useful for terminal paste / SLURM script generation).
    """
    # ── Sanity: mutually-exclusive guards ────────────────────────────────────
    if residues_to_keep and residues_to_throw_away:
        raise ValueError(
            "residues_to_keep and residues_to_throw_away are mutually exclusive.")
    if ligands_to_keep and ligands_to_throw_away:
        raise ValueError(
            "ligands_to_keep and ligands_to_throw_away are mutually exclusive.")
    if protect_ncAA_from_ligandization and leave_ncAA_as_ATOM:
        raise ValueError(
            "protect_ncAA_from_ligandization and leave_ncAA_as_ATOM are mutually exclusive.")
    if frag_ncAA is not None and (protect_ncAA_from_ligandization or leave_ncAA_as_ATOM):
        raise ValueError(
            "frag_ncAA (fragmentation) cannot be combined with "
            "protect_ncAA_from_ligandization or leave_ncAA_as_ATOM.")

    # ── Base command ─────────────────────────────────────────────────────────
    cmd = ["python", MAIN_SCRIPT,
           "--input_pdb", input_pdb,
           "--output_pdb_path", output_pdb]

    # ── OPTIONAL REMARK 666 ORDERING ─────────────────────────────────────────
    if remark666_exclude_residues:
        cmd += ["--remark666_exclude_residues"] + list(remark666_exclude_residues)
    if remark_front:
        cmd += ["--remark666_residue_front_order"] + list(remark_front)
    if remark_back:
        cmd += ["--remark666_residue_back_order"] + list(remark_back)
    if complete_remark666:
        cmd += ["--complete_remark666"]
    if force_regenerate_remark666:
        cmd += ["--force_regenerate_remark666"]
    if remark666_template_ligand:
        cmd += ["--remark666_template_ligand", remark666_template_ligand]
    if remark666_template_chain:
        cmd += ["--remark666_template_chain", remark666_template_chain]
    if remark666_template_resi is not None:
        cmd += ["--remark666_template_resi", str(remark666_template_resi)]
    if clean_remarks:
        cmd += ["--clean_remarks"]

    # ── FILTERING ────────────────────────────────────────────────────────────
    if residues_to_keep:
        cmd += ["--residues_to_keep"] + list(residues_to_keep)
    if residues_to_throw_away:
        cmd += ["--residues_to_throw_away"] + list(residues_to_throw_away)
    if ligands_to_keep:
        cmd += ["--ligands_to_keep"] + list(ligands_to_keep)
    if ligands_to_throw_away:
        cmd += ["--ligands_to_throw_away"] + list(ligands_to_throw_away)

    # ── ncAA HANDLING ────────────────────────────────────────────────────────
    if frag_ncAA is not None:
        cmd += ["--frag_ncAA_into_cAA_plus_lig"]
        if frag_ncAA:                            # [] => fragment ALL; non-empty => specific residues
            cmd += list(frag_ncAA)
    if protect_ncAA_from_ligandization:
        cmd += ["--protect_ncAA_from_ligandization"]
    if leave_ncAA_as_ATOM:
        cmd += ["--leave_ncAA_as_ATOM"]
    if add_CA_to_labeled_frag:
        cmd += ["--add_CA_to_labeled_frag"]
    if protect_sidechain_polarH is not None:
        cmd += ["--protect_sidechain_polarH"]
        if protect_sidechain_polarH:             # [] => all protein residues; list => specific
            cmd += list(protect_sidechain_polarH)
    if disable_intelligent_hstrip:
        cmd += ["--disable_intelligent_hstrip"]

    # ── MERGE LIGANDS ────────────────────────────────────────────────────────
    if merge_ligands_as:
        cmd += ["--merge_ligands_as", merge_ligands_as]
    if merged_ligand_chain:
        cmd += ["--merged_ligand_chain", merged_ligand_chain]
    if merged_ligand_resseq is not None:
        cmd += ["--merged_ligand_resseq", str(merged_ligand_resseq)]
    if merge_only:
        cmd += ["--merge_only"] + list(merge_only)

    # ── LEGACY OPT-IN CLEANING ───────────────────────────────────────────────
    if strip_insertion_codes:
        cmd += ["--strip_insertion_codes"]
    if strip_protein_hydrogens:
        cmd += ["--strip_protein_hydrogens"]
    if blank_segid:
        cmd += ["--blank_segid"]
    if strip_partial_charges:
        cmd += ["--strip_partial_charges"]
    if strip_formal_charges:
        cmd += ["--strip_formal_charges"]

    # ── THEOZYME LAYOUT ───────────────────────────────────────────────────────
    if theozyme_layout:
        cmd += ["--theozyme_layout"]

    # ── RENUMBER / CHECKS ─────────────────────────────────────────────────────
    if renumber_atoms:
        cmd += ["--renumber_atoms"]
    if no_ligand_code_checks:
        cmd += ["--no_ligand_code_checks"]
    if preserve_waters:
        cmd += ["--preserve_waters"]
    if rosetta_residue_types:
        cmd += ["--rosetta_residue_types", rosetta_residue_types]
    if ccd_timeout is not None:
        cmd += ["--ccd_timeout", str(ccd_timeout)]
    if verbose:
        cmd += ["--verbose"]

    # ── BUILD COMMAND STRING & OUTPUT ────────────────────────────────────────
    command_str = " ".join(shlex.quote(c) for c in cmd)
    print(command_str)

    if write_command_txt:
        os.makedirs(os.path.dirname(write_command_txt) or ".", exist_ok=True)
        with open(write_command_txt, "w") as fh:
            fh.write(command_str + "\n")

    return cmd


if __name__ == "__main__":
    # Example: print a sample command for the TS1 theozyme
    build_command(
        input_pdb="/path/to/input.pdb",
        output_pdb="/path/to/output.pdb",
        remark_front=["A55", "A57"],
        remark666_exclude_residues=[],
    )
