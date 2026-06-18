import argparse

### FUNCTION DEFINITIONS ###

def fix_pdb_spacing(input_pdb, output_pdb):
    """
    Fixes the spacing of PDB lines in the input file and writes the corrected lines to the output file.
    
    Args:
        input_pdb (str): Path to the input PDB file.
        output_pdb (str): Path to the output PDB file.
    """
    # Define a PDB format string
    pdb_format = "{:<6}{:>5}  {:<3} {:<3} {:<1}{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}           {:<2} "

    with open(input_pdb, 'r') as infile, open(output_pdb, 'w') as outfile:
        for line in infile:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Extract columns based on PDB standard positions
                record = line[0:6].strip()
                atom_serial = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21:22].strip()
                res_seq = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                occupancy = float(line[54:60].strip()) if line[54:60].strip() else 1.00
                temp_factor = float(line[60:66].strip()) if line[60:66].strip() else 0.00
                element = line[76:78].strip()

                # Write corrected line
                outfile.write(
                    pdb_format.format(
                        record, atom_serial, atom_name, res_name, chain_id, res_seq, x, y, z, occupancy, temp_factor, element
                    ) + "\n"
                )
            else:
                # Write non-ATOM/HETATM lines as is
                outfile.write(line)

### ARGUMENT PARSING ###

def main():
    parser = argparse.ArgumentParser(description="Fix spacing in PDB files.")
    parser.add_argument("-input_pdb", required=True, help="Path to the input PDB file.")
    parser.add_argument("-output_pdb", required=True, help="Path to the output PDB file.")
    args = parser.parse_args()

    fix_pdb_spacing(args.input_pdb, args.output_pdb)

if __name__ == "__main__":
    main()