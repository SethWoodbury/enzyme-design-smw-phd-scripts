#!/usr/bin/env python3
"""
Author: Seth M. Woodbury

SCRIPT NAME
    make_FullyBonded_mol2_file_from_singleXYZ_ThatCanHave_multipleXYZinside.py

PURPOSE
    • Convert one or more molecules in an XYZ file into a MOL2 via Open Babel.
    • Detect any atoms with zero bonds and stitch them to their nearest heavy atom.
    • Ensure each molecule is a single connected graph by linking any disconnected fragments.
    • Standardize every atom’s residue ID and name to “1  UNL1.”
    • Rewrite the MOL2 in place: update the bond count and append all new bond lines.

INPUTS
    --input_xyz_that_may_contain_multipleXYZinside
        Path to an XYZ file.  Can contain multiple “MODEL” blocks.
    --output_mol2_file_basename
        Base filename (no extension) for the generated .mol2.
    --output_dir_IF_different_from_input_dir
        (optional) Directory in which to write the .mol2; defaults to the XYZ’s folder.

PRIMARY STEPS
  1) generate_mol2_file()
       • Builds output path {output_dir}/{basename}.mol2
       • Runs Open Babel CLI to convert XYZ → MOL2

  2) parse_mol2(mol2_file)
       • Reads the MOL2 and splits into per‑molecule blocks
       • In each block:
           – Parses ATOM lines → dict of atom_id → {name, coords, element, bonds=[]}
           – Parses BOND lines → populates mol["bonds"] and each atom’s bonds list

  3) gather_new_bonds(molecules)
       • For every atom with zero bonds, find nearest non‑H neighbor
       • Record and mutate in‑memory bond lists
       • Build new_bonds_dict: { mol_index → [ (a1,a2), … ] }

  4) connect_fragments(molecules, new_bonds_dict)
       • Build connected components of the atom–bond graph
       • If >1 component, for each adjacent pair select the closest heavy‑atom pair
       • Add one new bond per fragment pair and record it in new_bonds_dict

  5) partial_update_mol2(mol2_file, new_bonds_dict)
       • Read original lines[] of the MOL2
       • For each “@<TRIPOS>MOLECULE” header, increment the bond count if new bonds exist
       • For each “@<TRIPOS>BOND” block, append new bond lines with new IDs
       • Write updated lines back to the same .mol2

  6) standardize_residue_labels(mol2_file)
       • Post‑rewrite pass over the ATOM block
       • Replace every atom’s residue ID → `1` and residue name → `UNL1`
       • Preserve all original column alignment by slicing

  7) update_mol2_with_bonds(mol2_file)
       • Orchestrates steps 2→5  
       • Then calls standardize_residue_labels()

OUTPUT
    A single, in‑place overwritten .mol2 containing:
      • All original geometry & topology  
      • Updated bond counts  
      • All new bonds—no atom left unconnected  
      • Exactly one connected fragment per molecule  
      • Uniform residue labeling (“1  UNL1”) on every ATOM line

USAGE (CLI)
    python xyz_to_fully_bonded_mol2.py \
      --input_xyz_that_may_contain_multipleXYZinside path/to/file.xyz \
      --output_mol2_file_basename mymol2 \
      [--output_dir_IF_different_from_input_dir path/to/output/]

DEPENDENCIES
    • Python stdlib: os, argparse, math, re  
    • Open Babel CLI (`obabel`) installed and reachable (or adjust the hard‑coded path)
"""

import os
import argparse
from math import sqrt
from math import inf
import re

# --- locate repo root + shared external paths ---
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "repo_paths.py").is_file():
        _sys.path.insert(0, str(_anc)); break
import repo_paths

### NEW FUNCTION ADDED ###
def standardize_residue_labels(mol2_file, resid=1, resname="UNL1"):
    """
    After the file is fully written, rewrite every line in the ATOM section
    so that the 7th column is `resid` and the 8th column is `resname`.
    """
    with open(mol2_file, 'r') as f:
        lines = f.readlines()

    updated = []
    in_atom = False
    for line in lines:
        if line.strip().startswith('@<TRIPOS>ATOM'):
            in_atom = True
            updated.append(line)
            continue
        if in_atom and line.strip().startswith('@<TRIPOS>'):
            # end of ATOM block
            in_atom = False
            updated.append(line)
            continue

########## MAY NEED TO ADJUST SPACING HERE FOR LARGER SYSTEMS ##########
        if in_atom and line.strip():
            parts = line.split()
            # keep original spacing, only overwrite subst_id/subst_name
            prefix = line[:54]
            suffix = line[62:]
            new_line = f"{prefix}{resid:>2d}  {resname:<4s}{suffix}"
            updated.append(new_line)
        else:
            updated.append(line)

    with open(mol2_file, 'w') as f:
        f.writelines(updated)

### NEW FUNCTION ADDED ###
def connect_fragments(molecules, new_bonds_dict):
    """
    For each molecule, if its atoms split into >1 connected component,
    add one bond between the closest pair of non‑H atoms in each adjacent component.
    """
    for idx, mol in enumerate(molecules, start=1):
        atoms = mol["atoms"]
        # 1) build list of components via simple DFS/BFS
        visited = set()
        components = []
        for a0 in atoms:
            if a0 in visited: continue
            stack = [a0]
            comp = set()
            while stack:
                a = stack.pop()
                if a in visited: continue
                visited.add(a)
                comp.add(a)
                for nbr in atoms[a]["bonds"]:
                    if nbr not in visited:
                        stack.append(nbr)
            components.append(comp)

        # 2) if more than one, connect them in series
        if len(components) > 1:
            for compA, compB in zip(components, components[1:]):
                best_pair = None
                best_dist2 = inf
                for a in compA:
                    # skip H to avoid silly hydrogen bridges
                    if atoms[a]["element"].upper() == "H": continue
                    x1,y1,z1 = atoms[a]["coords"]
                    for b in compB:
                        if atoms[b]["element"].upper() == "H": continue
                        x2,y2,z2 = atoms[b]["coords"]
                        d2 = (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2
                        if d2 < best_dist2:
                            best_dist2 = d2
                            best_pair = (a,b)
                if best_pair:
                    a,b = best_pair
                    # mutate in‑memory
                    atoms[a]["bonds"].append(b)
                    atoms[b]["bonds"].append(a)
                    mol["bonds"].append((a,b))
                    # record for later file patching
                    new_bonds_dict[idx].append((a,b))
                    print(f"  Connected fragment of mol {idx}: adding bond {a}–{b}")
### END OF NEW FUNCTION ADDED ###

def parse_mol2(mol2_file):
    """
    Parse .mol2 file to extract atom and bond data for each molecule.
    Returns a list of molecule dicts:
       [
         {
           "atoms": {atom_id: {"name": ..., "coords": (x,y,z), "element": ..., "bonds": []}, ...},
           "bonds": [(a1, a2), (a1, a2), ...]
         },
         ...
       ]
    """
    molecules = []
    current_molecule = {"atoms": {}, "bonds": []}
    with open(mol2_file, "r") as file:
        in_atom_section = False
        in_bond_section = False

        for line in file:
            if line.startswith("@<TRIPOS>MOLECULE"):
                # Start of a new molecule
                if current_molecule["atoms"]:
                    molecules.append(current_molecule)
                current_molecule = {"atoms": {}, "bonds": []}
                in_atom_section = False
                in_bond_section = False

            elif line.startswith("@<TRIPOS>ATOM"):
                in_atom_section = True
                in_bond_section = False

            elif line.startswith("@<TRIPOS>BOND"):
                in_atom_section = False
                in_bond_section = True

            elif line.startswith("@<TRIPOS>"):
                in_atom_section = False
                in_bond_section = False

            elif in_atom_section:
                parts = line.split()
                if len(parts) >= 6:
                    atom_id = int(parts[0])
                    atom_name = parts[1]
                    x, y, z = map(float, parts[2:5])
                    element = parts[5]
                    current_molecule["atoms"][atom_id] = {
                        "name": atom_name,
                        "coords": (x, y, z),
                        "element": element,
                        "bonds": []
                    }
                else:
                    print(f"Warning: Malformed atom line: {line.strip()}")

            elif in_bond_section:
                parts = line.split()
                if len(parts) >= 3:
                    # The first column is bond ID (ignored), next two are atom IDs
                    try:
                        a1 = int(parts[1])
                        a2 = int(parts[2])
                        if a1 in current_molecule["atoms"] and a2 in current_molecule["atoms"]:
                            current_molecule["bonds"].append((a1, a2))
                            current_molecule["atoms"][a1]["bonds"].append(a2)
                            current_molecule["atoms"][a2]["bonds"].append(a1)
                        else:
                            print(f"Warning: Bond references non-existent atoms: {[a1, a2]}")
                    except ValueError:
                        print(f"Warning: Could not parse bond line: {line.strip()}")
                else:
                    print(f"Warning: Malformed bond line: {line.strip()}")

        # Append last molecule if it has data
        if current_molecule["atoms"]:
            molecules.append(current_molecule)

    return molecules


def find_nearest_heteroatom(target_atom, atoms):
    """
    Find the nearest non-hydrogen atom to the target_atom.
    Return its ID, or None if not found.
    """
    min_dist = float("inf")
    nearest_id = None
    tx, ty, tz = target_atom["coords"]
    # We'll compare with all other atoms
    for other_id, other_atom in atoms.items():
        if other_atom is target_atom:
            continue
        if other_atom["element"].upper() == "H":
            continue

        ox, oy, oz = other_atom["coords"]
        dist = sqrt((tx - ox)**2 + (ty - oy)**2 + (tz - oz)**2)
        if dist < min_dist:
            min_dist = dist
            nearest_id = other_id

    return nearest_id


def gather_new_bonds(molecules):
    """
    Ensure each atom has at least one bond.  If not, bond it to nearest non-H.
    Return a dict of *newly* added bonds for each molecule index (1-based):
      { 1: [(6,2), (some_atom, some_atom)], 2: [], 3: [...], ... }
    """
    new_bonds_dict = {}
    for i, mol in enumerate(molecules, start=1):
        new_bonds = []
        atoms = mol["atoms"]
        for atom_id, atom in atoms.items():
            if not atom["bonds"]:
                print(f"Molecule {i}: Atom {atom_id} ({atom['name']}) has no bonds.")
                nearest_het_id = find_nearest_heteroatom(atom, atoms)
                if nearest_het_id:
                    print(f"  Creating bond between atom {atom_id} and {nearest_het_id}.")
                    atom["bonds"].append(nearest_het_id)
                    atoms[nearest_het_id]["bonds"].append(atom_id)
                    mol["bonds"].append((atom_id, nearest_het_id))
                    new_bonds.append((atom_id, nearest_het_id))
        new_bonds_dict[i] = new_bonds
    return new_bonds_dict


def partial_update_mol2(mol2_file, new_bonds_dict):
    """
    Read the .mol2 line by line, modifying only:
      - The "atom/bond count" line (2nd line after @<TRIPOS>MOLECULE) if we have new bonds.
      - The BOND section to append new bond lines.

    Write updated result to "<mol2_file>.updated".
    """
    import re

    with open(mol2_file, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    line_index = 0
    total = len(lines)

    molecule_index = 0
    # Example of bond line: "    42     6     2    1"
    bond_line_pattern = re.compile(r'^\s*(\d+)\s+(\d+)\s+(\d+)\s+(.*)')

    while line_index < total:
        line = lines[line_index]
        stripped = line.strip()

        # Check for the MOLECULE block:
        if stripped.startswith('@<TRIPOS>MOLECULE'):
            # We have a new molecule!
            molecule_index += 1

            # 1) Append the "@<TRIPOS>MOLECULE" line itself
            updated_lines.append(line)
            line_index += 1

            # 2) Next line is typically the molecule name
            if line_index < total:
                mol_name_line = lines[line_index]
                updated_lines.append(mol_name_line)
                line_index += 1
            else:
                break  # No more lines

            # 3) Next line is the atom/bond count (like " 40  41  0  0  0")
            if line_index < total:
                mol_count_line = lines[line_index]
                parts = mol_count_line.strip().split()
                # Usually something like ["40","41","0","0","0"]
                if molecule_index in new_bonds_dict and new_bonds_dict[molecule_index]:
                    try:
                        old_bond_count = int(parts[1])
                        add_bonds = len(new_bonds_dict[molecule_index])
                        new_bond_count = old_bond_count + add_bonds
                        parts[1] = str(new_bond_count)  # e.g. "42"
                        # Rebuild that line, e.g. " 40 42 0 0 0\n"
                        updated_mol_count_line = " " + " ".join(parts) + "\n"
                        updated_lines.append(updated_mol_count_line)
                    except (IndexError, ValueError):
                        # If parsing fails, just copy the original line
                        updated_lines.append(mol_count_line)
                else:
                    # No new bonds => just copy
                    updated_lines.append(mol_count_line)

                line_index += 1
            else:
                break  # No more lines

            # Done handling the lines immediately after MOLECULE
            continue

        # Check for the BOND block:
        elif stripped.startswith('@<TRIPOS>BOND'):
            # We want to append this line
            updated_lines.append(line)
            line_index += 1

            # Gather all existing bond lines until next block
            bond_lines = []
            last_bond_id = 0
            while line_index < total:
                peek = lines[line_index]
                if peek.strip().startswith('@<TRIPOS>'):
                    break
                bond_lines.append(peek)
                # Parse existing bond ID
                match = bond_line_pattern.match(peek)
                if match:
                    b_id = int(match.group(1))
                    if b_id > last_bond_id:
                        last_bond_id = b_id
                line_index += 1

            # Append existing bond lines
            updated_lines.extend(bond_lines)

            # Append new bond lines (if any) for this molecule
            if molecule_index in new_bonds_dict:
                for (a1, a2) in new_bonds_dict[molecule_index]:
                    last_bond_id += 1
                    # Format something like: "    42     6     2    1"
                    new_bond_line = f" {last_bond_id:>5} {a1:>5} {a2:>5} {1:>4}\n"
                    updated_lines.append(new_bond_line)

            # Don't consume the next @<TRIPOS> marker line
            continue

        # Otherwise, just copy the line
        updated_lines.append(line)
        line_index += 1

    # Finally, write to .updated file
    out_file = mol2_file #+ ".updated"
    with open(out_file, 'w') as f:
        f.writelines(updated_lines)

    print(f"[partial_update_mol2] Wrote updated file => {out_file}")

def update_mol2_with_bonds(mol2_file):
    """
    1) Parse .mol2 to get molecules
    2) Identify new bonds (unbonded atoms -> nearest heteroatom)
    3) Partially rewrite .mol2: increment bond count, append new bond lines
    """
    molecules = parse_mol2(mol2_file)
    print(f"Parsed {len(molecules)} molecules from '{mol2_file}'.")
    new_bonds_dict = gather_new_bonds(molecules)
    # --- NEW: ensure single‐fragment connectivity across each molecule ---
    connect_fragments(molecules, new_bonds_dict)
    # -------------------------------------------------------------------

    partial_update_mol2(mol2_file, new_bonds_dict)
    # final pass: force every ATOM line to use residue 1 / UNL1
    standardize_residue_labels(mol2_file)

def generate_mol2_file(input_xyz, output_basename, output_dir=None):
    """
    Generates a .mol2 file from an .xyz file using Open Babel.
    """
    if output_dir is None:
        output_dir = os.path.dirname(input_xyz)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_mol2 = os.path.join(output_dir, f"{output_basename}.mol2")

    # Adjust path to obabel as needed:
    obabel_path = repo_paths.OBABEL
    cmd = f"{obabel_path} -ixyz {input_xyz} -omol2 -O {output_mol2}"
    print("[generate_mol2_file] Running:", cmd)
    os.system(cmd)
    print("[generate_mol2_file] Wrote:", output_mol2)
    return output_mol2


def main(input_xyz, output_basename, output_dir):
    # 1) Convert XYZ to .mol2
    mol2_path = generate_mol2_file(input_xyz, output_basename, output_dir)

    # 2) Fix missing bonds (partial rewrite)
    print("[main] Checking/fixing bonds in:", mol2_path)
    update_mol2_with_bonds(mol2_path)
    print("[main] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a fully bonded .mol2 from an .xyz (partial rewrite).")
    parser.add_argument("--input_xyz_that_may_contain_multipleXYZinside", type=str, required=True,
                        help="Path to input .xyz with possibly multiple molecules.")
    parser.add_argument("--output_mol2_file_basename", type=str, required=True,
                        help="Basename for output .mol2 file (without extension).")
    parser.add_argument("--output_dir_IF_different_from_input_dir", type=str, default=None,
                        help="Output directory (optional). If omitted, uses same dir as input XYZ.")

    args = parser.parse_args()
    main(
        input_xyz=args.input_xyz_that_may_contain_multipleXYZinside,
        output_basename=args.output_mol2_file_basename,
        output_dir=args.output_dir_IF_different_from_input_dir
    )
