import argparse
from datetime import datetime
from collections import defaultdict


def parse_residues(pdb_lines):
    """
    Parse PDB lines to extract residue data grouped by residue number.
    """
    residues = defaultdict(list)
    for line in pdb_lines:
        if line.startswith(("ATOM", "HETATM")):
            chain = line[21].strip()
            res_num = int(line[22:26].strip())
            res_name = line[17:20].strip()
            residues[(chain, res_num, res_name)].append(line)
    return residues


def generate_remark666_lines(residues, ligand_name):
    """
    Generate REMARK 666 lines for each residue in the parsed residues dictionary.
    """
    remark_lines = []
    for i, ((chain, res_num, res_name), _) in enumerate(residues.items(), start=1):
        remark_lines.append(
            f"REMARK 666 MATCH TEMPLATE X {ligand_name:<4}   0 MATCH MOTIF {chain} {res_name}{res_num:>3}{res_num:>3}  1\n"
        )
    return remark_lines


def merge_and_renumber_pdb_files(residues_pdb, ligand_pdb, output_pdb):
    try:
        # Read the residues PDB file
        with open(residues_pdb, 'r') as residues_file:
            residues_lines = residues_file.readlines()
        
        # Read the ligand PDB file
        with open(ligand_pdb, 'r') as ligand_file:
            ligand_lines = ligand_file.readlines()
        
        # Parse residues from PDB lines
        parsed_residues = parse_residues(residues_lines)

        # Extract the ligand name from HETATM lines
        hetatm_lines = [line for line in ligand_lines if line.startswith("HETATM")]
        ligand_name = hetatm_lines[0][17:20].strip() if hetatm_lines else "UNK"  # Default to UNK if no ligand found

        # Generate REMARK 666 lines
        remark666_lines = generate_remark666_lines(parsed_residues, ligand_name)

        # Get the highest serial number from HETATM lines
        max_hetatm_serial = 0
        for line in hetatm_lines:
            try:
                serial = int(line[6:11].strip())
                max_hetatm_serial = max(max_hetatm_serial, serial)
            except ValueError:
                pass

        # Renumber the ATOM lines in residues PDB starting after the last HETATM serial
        next_serial = max_hetatm_serial + 1
        renumbered_residues_lines = []
        for line in residues_lines:
            if line.startswith("ATOM"):
                try:
                    line = f"{line[:6]}{next_serial:>5}{line[11:]}"
                    next_serial += 1
                except ValueError:
                    pass
            renumbered_residues_lines.append(line)
        
        # Get the current date in the required format (DD-MMM-YY)
        current_date = datetime.now().strftime("%d-%b-%y").upper()
        header_line = f"HEADER{' ' * 44}{current_date}   XXXX\n"

        # Create the merged PDB file
        with open(output_pdb, 'w') as output_file:
            # Write the HEADER line
            output_file.write(header_line)
            # Write REMARK 666 lines
            output_file.writelines(remark666_lines)
            # Write renumbered residues PDB lines
            output_file.writelines(renumbered_residues_lines)
            # Write TER after the residues
            output_file.write("TER\n")
            # Write HETATM lines from ligand PDB
            output_file.writelines(hetatm_lines)
            # Write TER after the HETATM lines
            output_file.write("TER\n")
        
        print(f"Merged and renumbered PDB file created successfully: {output_pdb}")
    
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and renumber residues and ligand PDB files.")
    parser.add_argument("-residues_pdb", required=True, help="Path to the residues PDB file")
    parser.add_argument("-ligand_pdb", required=True, help="Path to the ligand PDB file")
    parser.add_argument("-output_pdb", default="merged.pdb", help="Path to the output merged PDB file")
    
    args = parser.parse_args()
    merge_and_renumber_pdb_files(args.residues_pdb, args.ligand_pdb, args.output_pdb)
