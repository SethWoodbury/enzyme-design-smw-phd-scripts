#!/usr/bin/env python3

import argparse
import os
from math import sqrt

def parse_mol2(mol2_file):
    """Parse .mol2 file to get atoms and bonds."""
    atoms = {}
    bonds = []
    with open(mol2_file, "r") as file:
        in_atom_section = False
        in_bond_section = False
        for line in file:
            if line.startswith("@<TRIPOS>ATOM"):
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
                atom_id = int(parts[0])
                atom_name = parts[1]
                x, y, z = map(float, parts[2:5])
                element = parts[5]
                atoms[atom_id] = {"name": atom_name, "coords": (x, y, z), "element": element, "bonds": []}
            elif in_bond_section:
                bond_info = list(map(int, line.split()[1:3]))
                bonds.append(bond_info)
                atoms[bond_info[0]]["bonds"].append(bond_info[1])
                atoms[bond_info[1]]["bonds"].append(bond_info[0])
    return atoms, bonds

def find_nearest_heteroatom(target_atom, atoms):
    """Find the nearest non-hydrogen atom to the target atom."""
    min_distance = float("inf")
    nearest_atom_id = None
    tx, ty, tz = target_atom["coords"]

    for atom_id, atom in atoms.items():
        if atom["element"] != "H" and atom_id != target_atom:
            x, y, z = atom["coords"]
            distance = sqrt((tx - x) ** 2 + (ty - y) ** 2 + (tz - z) ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_atom_id = atom_id

    return nearest_atom_id

def main(input_pdb, ligand_code):
    obabel_path = "/home/woodbuse/conda_envs/openbabel_env/bin/obabel"
    output_mol2 = f"{ligand_code}.mol2"

    # Use Open Babel to convert PDB to MOL2 format
    os.system(f"{obabel_path} -ipdb {input_pdb} -omol2 -O {output_mol2}")
    print(f"\n### {output_mol2} created from {input_pdb} ###")

    # Parse the MOL2 file to gather atom and bond data
    atoms, bonds = parse_mol2(output_mol2)

    # Check for atoms without bonds
    for atom_id, atom in atoms.items():
        if not atom["bonds"]:
            print(f"### ATOM {atom_id}, {atom['name']}, IS MISSING A BOND... MAKING AN ARTIFICIAL ONE TO NEAREST HETEROATOM!")
            # Find the nearest heteroatom and create an artificial bond
            nearest_heteroatom_id = find_nearest_heteroatom(atom, atoms)
            if nearest_heteroatom_id:
                atom["bonds"].append(nearest_heteroatom_id)
                atoms[nearest_heteroatom_id]["bonds"].append(atom_id)
                bonds.append((atom_id, nearest_heteroatom_id))
                print(f"### Created artificial bond between ATOM {atom_id} and ATOM {nearest_heteroatom_id} ###")

    # Write updated .mol2 file with artificial bonds added
    with open(output_mol2, "w") as file:
        file.write("@<TRIPOS>MOLECULE\n")
        file.write(f"{ligand_code}.mol2\n")
        file.write(f" {len(atoms)} {len(bonds)} 0 0 0\n")  # Updated atom and bond counts
        file.write("SMALL\nGASTEIGER\n\n")
        file.write("@<TRIPOS>ATOM\n")
        for atom_id, atom in atoms.items():
            x, y, z = atom["coords"]
            file.write(f"{atom_id:>7} {atom['name']:<8} {x:>9.4f} {y:>9.4f} {z:>9.4f} {atom['element']:<5} 1 {ligand_code} 0.0000\n")
        file.write("@<TRIPOS>BOND\n")
        for i, (a1, a2) in enumerate(bonds, start=1):
            file.write(f"{i:>6} {a1:>5} {a2:>5} 1\n")

    print(f"\n### Updated {output_mol2} with artificial bonds where necessary ###")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ligand PDB to MOL2 file, check and add bonds as necessary.")
    parser.add_argument("-input_pdb", required=True, help="Input ligand PDB file to convert to MOL2")
    parser.add_argument("-ligand_code", required=True, help="3-letter code for the ligand, used as output filename prefix")

    args = parser.parse_args()
    main(args.input_pdb, args.ligand_code)
