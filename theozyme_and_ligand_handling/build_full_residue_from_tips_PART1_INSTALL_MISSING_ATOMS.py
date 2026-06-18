import argparse
import os
from pathlib import Path

def combine_residues_into_pdb(aligned_rosetta_pdbs, output_file):
    """Combine multiple aligned Rosetta PDB files into a single PDB file."""
    residues = []

    for pdb_file in aligned_rosetta_pdbs:
        if not os.path.exists(pdb_file):
            print(f"Error: File {pdb_file} does not exist.")
            continue

        with open(pdb_file, 'r') as f:
            for line in f:
                # Include ATOM and HETATM lines only
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    residues.append(line)

    # Sort residues by residue number embedded in the file names
    def extract_residue_number(file_name):
        return int(''.join([c for c in file_name if c.isdigit()]))

    sorted_residues = sorted(aligned_rosetta_pdbs, key=extract_residue_number)

    # Write sorted residues to the output PDB file and renumber atoms
    atom_counter = 1
    with open(output_file, 'w') as output:
        for pdb_file in sorted_residues:
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        new_line = line[:6] + f"{atom_counter:5d}" + line[11:]
                        output.write(new_line)
                        atom_counter += 1

    print(f"Combined PDB written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine aligned Rosetta PDBs into a single PDB file.")
    parser.add_argument(
        "-aligned_rosetta_pdbs_for_residues",
        nargs='+',
        required=True,
        help="List of aligned Rosetta PDB files to combine.",
    )
    parser.add_argument(
        "-output_pdb",
        type=Path,
        required=True,
        help="Output file path for the combined PDB.",
    )

    args = parser.parse_args()

    combine_residues_into_pdb(args.aligned_rosetta_pdbs_for_residues, args.output_pdb)
