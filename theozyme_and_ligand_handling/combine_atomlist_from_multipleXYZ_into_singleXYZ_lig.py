import os
import argparse

def parse_atom_ranges(range_string):
    """Parse a string of atom ranges, e.g., '1-40,62-100', into a list of integers."""
    ranges = range_string.split(',')
    atom_indices = []
    for r in ranges:
        start, end = map(int, r.split('-'))
        atom_indices.extend(range(start - 1, end))  # Adjust to zero-based indexing
    return atom_indices

def extract_atoms_from_xyz(file_path, atom_indices):
    """Extract specified atoms from a .xyz file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    num_atoms = int(lines[0].strip())
    if num_atoms < max(atom_indices) + 1:  # Adjust for zero-based indexing
        raise ValueError(f"File {file_path} has fewer atoms ({num_atoms}) than specified in the ranges.")

    selected_atoms = [lines[i + 2].strip() for i in atom_indices]
    return selected_atoms

def create_combined_xyz(output_file, xyz_files, selected_atoms_of_molecule, parse_this_xyz_first=None):
    """Create a new .xyz file combining selected atoms from multiple .xyz files."""
    atom_indices = parse_atom_ranges(selected_atoms_of_molecule)

    # Check if output directory exists; create it if not
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist. Creating it.")
        os.makedirs(output_dir)

    # Remove existing output file if it exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Deleting it to avoid conflicts.")
        os.remove(output_file)

    # Sort files alphanumerically
    xyz_files.sort()

    # If a specific file is to be parsed first, handle it
    if parse_this_xyz_first and parse_this_xyz_first in xyz_files:
        xyz_files.remove(parse_this_xyz_first)
        xyz_files.insert(0, parse_this_xyz_first)

    with open(output_file, 'w') as out:
        for xyz_file in xyz_files:
            try:
                selected_atoms = extract_atoms_from_xyz(xyz_file, atom_indices)
                num_atoms = len(selected_atoms)

                # Add header with number of atoms and file name (without extension)
                file_base = os.path.splitext(os.path.basename(xyz_file))[0]
                out.write(f"{num_atoms}\n{file_base}\n")

                # Write atom coordinates
                out.write("\n".join(selected_atoms) + "\n")
            except Exception as e:
                print(f"Skipping {xyz_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine selected atoms from multiple .xyz files.")
    parser.add_argument("--input_directory", type=str, required=True, help="Directory containing .xyz files.")
    parser.add_argument("--selected_atoms_of_molecule", type=str, required=True, help="Atom ranges in the format '1-40,62-100'.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output .xyz file.")
    parser.add_argument("--parse_this_xyz_first", type=str, help="Optional .xyz file to parse first.")

    args = parser.parse_args()

    # Find all .xyz files in the directory
    xyz_files = [os.path.join(args.input_directory, f) for f in os.listdir(args.input_directory) if f.endswith(".xyz")]

    if not xyz_files:
        print("No .xyz files found in the specified directory.")
        exit(1)

    # Create combined .xyz file
    create_combined_xyz(args.output_file, xyz_files, args.selected_atoms_of_molecule, args.parse_this_xyz_first)
    print(f"Combined .xyz file created at {args.output_file}")
