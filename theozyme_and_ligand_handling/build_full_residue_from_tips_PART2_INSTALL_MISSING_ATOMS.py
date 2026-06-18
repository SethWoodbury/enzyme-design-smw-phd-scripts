import sys
import argparse
import math
from collections import defaultdict

### HELPER FUNCTIONS ###

def parse_pdb(file_path):
    """Parse a PDB file and return a dictionary of residues with their atoms, excluding hydrogens."""
    residues = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom = {
                    "atom_name": line[12:16].strip(),
                    "element": line[76:78].strip(),
                    "residue_name": line[17:20].strip(),
                    "chain_id": line[21].strip(),
                    "residue_number": int(line[22:26].strip()),
                    "x": float(line[30:38].strip()),
                    "y": float(line[38:46].strip()),
                    "z": float(line[46:54].strip()),
                    "line": line
                }
                if atom["element"] != "H":  # Exclude hydrogens
                    residues[(atom["chain_id"], atom["residue_number"])].append(atom)
    return residues

def calculate_distance(atom1, atom2):
    """Calculate the Euclidean distance between two atoms."""
    return math.sqrt((atom1["x"] - atom2["x"])**2 +
                     (atom1["y"] - atom2["y"])**2 +
                     (atom1["z"] - atom2["z"])**2)

def match_atoms(residue_dft, residue_rosetta):
    """Match atoms in the DFT residue to the closest atoms in the Rosetta residue, considering element types."""
    matched_atoms = {}
    used_atoms = set()

    for atom_dft in residue_dft:
        closest_atom = None
        min_distance = float("inf")
        for atom_rosetta in residue_rosetta:
            if atom_rosetta["atom_name"] in used_atoms:
                continue
            if atom_dft["element"] != atom_rosetta["element"]:
                continue
            distance = calculate_distance(atom_dft, atom_rosetta)
            if distance < min_distance:
                min_distance = distance
                closest_atom = atom_rosetta

        if closest_atom:
            matched_atoms[atom_dft["line"]] = closest_atom["atom_name"]
            used_atoms.add(closest_atom["atom_name"])

    return matched_atoms, [atom for atom in residue_rosetta if atom["atom_name"] not in used_atoms]

### MAIN SCRIPT ###

def main():
    parser = argparse.ArgumentParser(description="Match and merge PDB files.")
    parser.add_argument("-aligned_rosetta_single_pdb_of_residues", required=True, help="Rosetta PDB file.")
    parser.add_argument("-dft_tip_atoms_of_residues_only", required=True, help="DFT PDB file.")
    parser.add_argument("-output_pdb", required=True, help="Output PDB file name.")
    args = parser.parse_args()

    # Parse the input PDB files
    rosetta_residues = parse_pdb(args.aligned_rosetta_single_pdb_of_residues)
    dft_residues = parse_pdb(args.dft_tip_atoms_of_residues_only)

    output_lines = []

    # Process residues
    for residue_id, dft_atoms in dft_residues.items():
        rosetta_atoms = rosetta_residues.get(residue_id, [])
        if not rosetta_atoms:
            print(f"Warning: Residue {residue_id} in DFT file not found in Rosetta file.")
            output_lines.extend([atom["line"] for atom in dft_atoms])
            continue

        print(f"Processing residue {residue_id}...")
        matched_atoms, unmatched_rosetta_atoms = match_atoms(dft_atoms, rosetta_atoms)

        # Update atom names in DFT residue and write to output
        for line in dft_atoms:
            updated_line = line["line"]
            if line["line"] in matched_atoms:
                atom_name = matched_atoms[line["line"]]
                updated_line = updated_line[:12] + atom_name.ljust(4) + updated_line[16:]
                print(f"Matched {line['atom_name']} -> {atom_name}")
            output_lines.append(updated_line)

        # Append unmatched Rosetta atoms
        for atom in unmatched_rosetta_atoms:
            output_lines.append(atom["line"])
            print(f"Appending unmatched atom {atom['atom_name']} to residue {residue_id}.")

    # Write output PDB file
    with open(args.output_pdb, 'w') as output_file:
        output_file.writelines(output_lines)

    print(f"Output written to {args.output_pdb}.")

if __name__ == "__main__":
    main()
