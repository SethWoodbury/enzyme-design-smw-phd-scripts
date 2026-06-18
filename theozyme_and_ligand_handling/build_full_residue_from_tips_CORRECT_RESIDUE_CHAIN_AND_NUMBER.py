import argparse

### SECTION HEADER ###
def parse_arguments():
    parser = argparse.ArgumentParser(description="Update residue, chain, and residue number in a PDB file.")
    parser.add_argument("-input_rosetta_residue", required=True, help="Path to the input PDB file containing the residue information.")
    parser.add_argument("-input_tip_atom_aligned_rosetta_residue", required=True, help="Path to the PDB file whose residue, chain, and number need updating.")
    return parser.parse_args()

### SECTION HEADER ###
def parse_residue_information(pdb_file):
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                res_name = line[17:20].strip()
                chain_id = line[21].strip()
                res_num = line[22:26].strip()
                print(f"Identified residue information: Residue={res_name}, Chain={chain_id}, Residue Number={res_num}")
                return res_name, chain_id, res_num
    raise ValueError("No ATOM lines found in the input_rosetta_residue file.")

### SECTION HEADER ###
def update_residue_chain_and_number(source_pdb, target_pdb, res_name, chain_id, res_num):
    updated_lines = []
    with open(target_pdb, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                original_line = line
                updated_line = (line[:17] + f"{res_name:>3}" + line[20] + f"{chain_id}" +
                                f"{res_num:>4}" + line[26:])
                updated_lines.append(updated_line)
                print(f"Updated line: \nOriginal: {original_line.strip()}\nUpdated: {updated_line.strip()}")
            else:
                updated_lines.append(line)
    return updated_lines

### SECTION HEADER ###
def write_updated_pdb(output_file, updated_lines):
    with open(output_file, 'w') as f:
        f.writelines(updated_lines)
    print(f"Updated PDB file written to {output_file}")

### SECTION HEADER ###
def main():
    args = parse_arguments()

    print(f"Parsing residue information from {args.input_rosetta_residue}")
    res_name, chain_id, res_num = parse_residue_information(args.input_rosetta_residue)

    print(f"Updating residue, chain, and residue number in {args.input_tip_atom_aligned_rosetta_residue}")
    updated_lines = update_residue_chain_and_number(
        args.input_rosetta_residue,
        args.input_tip_atom_aligned_rosetta_residue,
        res_name,
        chain_id,
        res_num
    )

    output_file = args.input_tip_atom_aligned_rosetta_residue
    write_updated_pdb(output_file, updated_lines)

if __name__ == "__main__":
    main()