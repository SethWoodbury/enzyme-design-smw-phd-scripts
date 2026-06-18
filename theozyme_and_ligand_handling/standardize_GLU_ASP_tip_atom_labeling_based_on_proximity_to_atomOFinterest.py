#!/usr/bin/env python3

"""
Script: standardize_GLU_ASP_tip_atom_labeling_based_on_proximity_to_atomOFinterest.py

Description:
    1. Reads a PDB file (--input_pdb).
    2. Finds a specific ligand (--ligand_code) and ligand atom
       (--ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp).
    3. For every ASP/GLU in the PDB:
       - Checks whether OD2/OE2 is closer to the ligand's reference atom than OD1/OE1.
       - If not, swaps OD1 <-> OD2 or OE1 <-> OE2.
    4. Writes out a new PDB (same lines in the same order, including remarks, etc.),
       except for changed ATOM/HETATM lines in ASP/GLU that got swapped.
    5. Logs messages about what it found and what it changed.

Example usage:
    python standardize_GLU_ASP_tip_atom_labeling_based_on_proximity_to_atomOFinterest.py \
        --input_pdb my_input.pdb \
        --ligand_code SZA \
        --ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp H1
"""

import os
import math
import argparse
import sys

def distance_squared(coord1, coord2):
    """Return the squared distance between two 3D coordinates."""
    return (
        (coord1[0] - coord2[0]) ** 2 +
        (coord1[1] - coord2[1]) ** 2 +
        (coord1[2] - coord2[2]) ** 2
    )

def _replace_atom_name(pdb_line, new_name):
    """
    Given an ATOM/HETATM line, replace the atom name in columns 12-16 with new_name.
    We'll right-pad with spaces if needed to preserve correct column formatting.
    """
    left = pdb_line[:13]
    right = pdb_line[17:]
    new_name_padded = new_name.ljust(4)  # ensure length 4
    return left + new_name_padded + right

def swap_atom_names_in_lines(atom_records, chain, resnum, atom1, atom2):
    """
    Swap the atom1 and atom2 names for the given chain/resnum in atom_records.
    We'll update the 'original_line' portion in place.
    
    atom_records is a dict:
        line_index -> {
            'resname': ...,
            'resnum': ...,
            'atomname': ...,
            'chain': ...,
            'coord': (x, y, z),
            'original_line': ...
        }
    """
    for i, record in atom_records.items():
        if record['chain'] == chain and record['resnum'] == resnum:
            if record['atomname'] == atom1:
                new_line = _replace_atom_name(record['original_line'], atom2)
                record['atomname'] = atom2
                record['original_line'] = new_line
            elif record['atomname'] == atom2:
                new_line = _replace_atom_name(record['original_line'], atom1)
                record['atomname'] = atom1
                record['original_line'] = new_line


def main():
    parser = argparse.ArgumentParser(
        description="Standardize ASP/GLU tip atom names (OD2/OE2) based on proximity to a given ligand atom."
    )
    parser.add_argument("--input_pdb", required=True, help="Path to the input PDB file.")
    parser.add_argument("--ligand_code", required=True, help="3-letter code for the ligand (e.g. SZA).")
    parser.add_argument(
        "--ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp",
        required=True,
        help="Ligand atom name (e.g. H1) to measure proximity for OE2/OD2."
    )

    args = parser.parse_args()

    pdb_file = args.input_pdb
    ligand_code = args.ligand_code
    ligand_atom_of_interest = args.ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp

    if not os.path.isfile(pdb_file):
        print(f"[ERROR] PDB file '{pdb_file}' does not exist.")
        sys.exit(1)

    # We'll read ALL lines, storing them in a list so we can rewrite them in order.
    with open(pdb_file, "r") as f:
        all_lines = f.readlines()

    # We'll keep a mapping from line_index -> record for ATOM/HETATM lines
    # record is { 'resname': str, 'resnum': str, 'atomname': str, 'chain': str,
    #             'coord': (x, y, z), 'original_line': str }
    atom_records = {}

    # Collect info about the ligand's atom of interest
    ligand_atom_coord = None

    # First pass: parse ATOM/HETATM lines, store them.
    for i, line in enumerate(all_lines):
        line_type = line[:6].strip()
        if line_type in ("ATOM", "HETATM"):
            atomname = line[12:16].strip()
            resname  = line[17:20].strip()
            chain    = line[21].strip()
            resnum   = line[22:26].strip()
            x_str    = line[30:38].strip()
            y_str    = line[38:46].strip()
            z_str    = line[46:54].strip()

            try:
                x = float(x_str)
                y = float(y_str)
                z = float(z_str)
            except ValueError:
                # Malformed coordinates, skip
                continue

            atom_records[i] = {
                'resname': resname,
                'resnum': resnum,
                'atomname': atomname,
                'chain': chain,
                'coord': (x, y, z),
                'original_line': line
            }

    # Find the coordinate of the specified ligand atom (just the first one we see)
    for i, record in atom_records.items():
        if record['resname'] == ligand_code and record['atomname'] == ligand_atom_of_interest:
            ligand_atom_coord = record['coord']
            print(f"### Found ligand {ligand_code} (resnum {record['resnum']}, chain {record['chain']}) "
                  f"with atom {ligand_atom_of_interest}. ###")
            break

    if ligand_atom_coord is None:
        print(f"### WARNING: Could not find atom '{ligand_atom_of_interest}' in ligand '{ligand_code}' in {pdb_file}.")
        print("### Distance checks won't be performed, but we'll still output a 'standardized' file. ###")

    # Build a dict of ASP/GLU residues keyed by (chain, resnum, resname).
    # Each value will hold the relevant atoms we find (e.g. OD1, OD2, OE1, OE2).
    residues_dict = {}
    for i, record in atom_records.items():
        rname = record['resname']
        if rname in ["ASP", "GLU"]:
            chain = record['chain']
            rnum  = record['resnum']
            key = (chain, rnum, rname)
            if key not in residues_dict:
                residues_dict[key] = {}
            # e.g. residues_dict[(A, "4", "ASP")]["OD1"] = (coord, line_index)
            aname = record['atomname']
            residues_dict[key][aname] = i  # store the line index

    # We'll track which ASP/GLU residues were found
    found_asp_glu = []

    # For each ASP/GLU, check distances
    for key, atoms_dict in residues_dict.items():
        chain, rnum, rname = key
        if rname == "ASP":
            found_asp_glu.append(f"ASP {rnum}")
            od1_idx = atoms_dict.get("OD1")
            od2_idx = atoms_dict.get("OD2")
            if od1_idx is not None and od2_idx is not None and ligand_atom_coord is not None:
                od1_coord = atom_records[od1_idx]['coord']
                od2_coord = atom_records[od2_idx]['coord']
                dist_od1_sq = distance_squared(od1_coord, ligand_atom_coord)
                dist_od2_sq = distance_squared(od2_coord, ligand_atom_coord)
                if dist_od2_sq <= dist_od1_sq:
                    print(f"### ASP {rnum} OD2 is already closer to {ligand_code} atom {ligand_atom_of_interest} | DOING NOTHING ###")
                else:
                    print(f"### ASP {rnum} OD1 is closer to {ligand_code} atom {ligand_atom_of_interest} than OD2 | INVERTING THE NAMES ###")
                    # Swap OD1 <-> OD2
                    swap_atom_names_in_lines(atom_records, chain, rnum, "OD1", "OD2")

        elif rname == "GLU":
            found_asp_glu.append(f"GLU {rnum}")
            oe1_idx = atoms_dict.get("OE1")
            oe2_idx = atoms_dict.get("OE2")
            if oe1_idx is not None and oe2_idx is not None and ligand_atom_coord is not None:
                oe1_coord = atom_records[oe1_idx]['coord']
                oe2_coord = atom_records[oe2_idx]['coord']
                dist_oe1_sq = distance_squared(oe1_coord, ligand_atom_coord)
                dist_oe2_sq = distance_squared(oe2_coord, ligand_atom_coord)
                if dist_oe2_sq <= dist_oe1_sq:
                    print(f"### GLU {rnum} OE2 is already closer to {ligand_code} atom {ligand_atom_of_interest} | DOING NOTHING ###")
                else:
                    print(f"### GLU {rnum} OE1 is closer to {ligand_code} atom {ligand_atom_of_interest} than OE2 | INVERTING THE NAMES ###")
                    # Swap OE1 <-> OE2
                    swap_atom_names_in_lines(atom_records, chain, rnum, "OE1", "OE2")

    if found_asp_glu:
        print("### found", ", ".join(found_asp_glu), "###")

    # Now write the output file. We'll replicate *all* lines from the original,
    # but if a line is in atom_records (and possibly swapped), we use the updated line.
    out_pdb_file = os.path.splitext(pdb_file)[0] + ".pdb"
    with open(out_pdb_file, "w") as out_f:
        for i, line in enumerate(all_lines):
            if i in atom_records:
                # Use updated (or unchanged) original_line
                out_f.write(atom_records[i]['original_line'])
            else:
                # Non-ATOM/HETATM line or unrecognized line index, just copy it verbatim
                out_f.write(line)

    print(f"### Wrote updated PDB to {out_pdb_file} ###")


if __name__ == "__main__":
    main()
