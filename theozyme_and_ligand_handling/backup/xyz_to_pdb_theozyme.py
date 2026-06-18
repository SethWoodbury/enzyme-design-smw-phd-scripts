#!/usr/bin/env python3

import argparse
import os
import string

# Define the main function
def main(input_xyz, ligand_atom_ranges, lig_3letter_code, ligand_chain, tip_atom_residues_3letter, DO_NOT_pass_tip_atom_residues_3letter_to_help_identifier, DO_NOT_make_params_or_mol2_files, KEEP_temp_smiles_and_res_pdbs_for_debug):
    # Parse ligand atom ranges
    ligand_atoms = parse_atom_ranges(ligand_atom_ranges)
    
    # Generate temporary XYZ files for protein residues and ligand
    temp_residue_xyz_file = create_temp_xyz(input_xyz, ligand_atoms, exclude=True)
    temp_ligand_xyz_file = create_temp_xyz(input_xyz, ligand_atoms, exclude=False)
    
    # Convert residue XYZ to PDB using Open Babel
    residue_pdb_file = temp_residue_xyz_file.replace("TEMP.xyz", "residues_TEMP.pdb")
    obabel_path = "/home/woodbuse/conda_envs/openbabel_env/bin/obabel"
    os.system(f"{obabel_path} -ixyz {temp_residue_xyz_file} -opdb -O {residue_pdb_file}")
    
    # Convert ligand XYZ to PDB using Open Babel
    ligand_pdb_file = temp_ligand_xyz_file.replace("TEMP.xyz", "ligand_TEMP.pdb")
    os.system(f"{obabel_path} -ixyz {temp_ligand_xyz_file} -opdb -O {ligand_pdb_file}")
    
    # Clean up temporary XYZ files
    os.remove(temp_residue_xyz_file)
    os.remove(temp_ligand_xyz_file)
    
    # Clean both TEMP PDB files: remove COMPND/AUTHOR lines and convert to uppercase
    clean_pdb_file(residue_pdb_file)
    clean_pdb_file(ligand_pdb_file)
    
    # Modify residues PDB file to change HETATM to ATOM and add chain letter
    update_residues_pdb(residue_pdb_file)

    # Modify ligand PDB file: update HETATM, set 3-letter code, chain ID, unique atom names, and set all residue numbers to 9
    update_ligand_pdb(ligand_pdb_file, lig_3letter_code, ligand_chain)
    modify_pdb_atom_names(ligand_pdb_file)
    
    print(f"Temporary PDB files created:")
    print(f"  Residues only: {residue_pdb_file}")
    print(f"  Ligand only (updated): {ligand_pdb_file}")

    # Call the side script to renumber residues
    os.system(f"/software/containers/crispy.sif /home/woodbuse/special_scripts/theozyme_and_ligand_handling/renumber_pdb.py -input_pdb {residue_pdb_file}")

    print("PASSED RESIDUE RENUMBERING")
    print("")

    # Call the side script to identify residues with the -tip_atom_residues_3letter argument if the flag is set
    if DO_NOT_pass_tip_atom_residues_3letter_to_help_identifier:
        os.system(f"/software/containers/crispy.sif /home/woodbuse/special_scripts/theozyme_and_ligand_handling/identify_residues.py -input_pdb {residue_pdb_file} -tip_atom_residues_3letter {' '.join(tip_atom_residues_3letter)}")
    else:
        os.system(f"/software/containers/crispy.sif /home/woodbuse/special_scripts/theozyme_and_ligand_handling/identify_residues.py -input_pdb {residue_pdb_file}")

    print("PASSED RESIDUE IDENTIFICATION")
    print("")

    # Read the residue mapping and verify against tip_atom_residues_3letter
    with open("residue_map.txt", "r") as f:
        residue_map = [line.strip().split(' = ') for line in f]
        residue_mapping_dict = {int(res.split('_TEMP_smiles')[0][3:]): aa for res, aa in residue_map}

    # Print contents of residue_map.txt for verification
    print("\n### RESIDUE MAPPING FROM FILE ###")
    for res_num, aa_code in residue_mapping_dict.items():
        print(f"Residue {res_num}: {aa_code}")

    # Modify residue_pdb_file in place to replace "UNL" (or existing 3-letter code) with correct amino acid code
    with open(residue_pdb_file, "r") as infile:
        lines = infile.readlines()

    with open(residue_pdb_file, "w") as outfile:
        current_residue = None
        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Extract residue index from line
                residue_index = int(line[22:26].strip())
                if residue_index != current_residue:
                    current_residue = residue_index
                
                # Replace "UNL" (or current 3-letter code) with the mapped amino acid code
                aa_code = residue_mapping_dict.get(current_residue, "UNL")
                updated_line = line[:17] + f"{aa_code:<3}" + line[20:]
                outfile.write(updated_line)
            else:
                outfile.write(line)

    print("\n### {residue_pdb_file} has been updated with correct amino acid codes ###")
    print("")
    print("### MOVING ON TO MAKING .params FILE ###")

    if not DO_NOT_make_params_or_mol2_files:
        # Define the path to the new script and the ligand PDB file to be processed
        ligand_temp_pdb_file = f"{ligand_pdb_file}"
        params_script_path = "/home/woodbuse/special_scripts/theozyme_and_ligand_handling/make_params_file.py"

        # Call the make_params_file.py script to create a .mol2 file from the ligand PDB
        os.system(f"python {params_script_path} -input_pdb {ligand_temp_pdb_file} -ligand_code {lig_3letter_code}")

        print(f"\n### {params_script_path} has been called to generate {lig_3letter_code}.mol2 ###")
    else:
        print("\n### SKIPPING OVER MOL2 & PARAMS FILE GENERATION ###")

    # Remove temporary files unless the debug flag is set
    if not KEEP_temp_smiles_and_res_pdbs_for_debug:
        print("\n### CLEANING UP TEMPORARY FILES ###")

        # Remove residue_map.txt
        if os.path.exists("residue_map.txt"):
            os.remove("residue_map.txt")
            print("Deleted residue_map.txt")

        # Remove files ending with _TEMP_smiles.smi
        for file in os.listdir("."):
            if file.endswith("_TEMP_smiles.smi"):
                os.remove(file)
                print(f"Deleted {file}")

        # Remove files starting with pdb and ending with _TEMP.pdb
        for file in os.listdir("."):
            if file.startswith("pdb") and file.endswith("_TEMP.pdb"):
                os.remove(file)
                print(f"Deleted {file}")
    else:
        print("\n### KEEPING TEMPORARY SMILES AND RESIDUE PDB FILES FOR DEBUGGING ###")

def parse_atom_ranges(ligand_range):
    """Parse the ligand atom range string into a list of integers."""
    atoms = []
    for part in ligand_range.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            atoms.extend(range(start, end + 1))
        else:
            atoms.append(int(part))
    return set(atoms)

def create_temp_xyz(input_xyz, ligand_atoms, exclude=True):
    """Create a temporary XYZ file with specified atoms removed or kept."""
    suffix = "_TEMP.xyz" if exclude else "_ligand_TEMP.xyz"
    temp_xyz = input_xyz.replace(".xyz", suffix)
    with open(input_xyz, "r") as infile, open(temp_xyz, "w") as outfile:
        lines = infile.readlines()
        atom_count = int(lines[0].strip())
        
        # Filter lines based on exclusion or inclusion of ligand atoms
        selected_lines = [line for i, line in enumerate(lines[2:], start=1)
                          if (i in ligand_atoms) != exclude]
        
        # Write the new atom count and lines to the output file
        outfile.write(f"{len(selected_lines)}\n")
        outfile.write(lines[1])  # Copy the comment line
        outfile.writelines(selected_lines)
                
    return temp_xyz

def update_residues_pdb(residue_pdb_file):
    """Update residues PDB file to change HETATM to ATOM and replace chain number with letter."""
    # Create mapping from integer to alphabet letter (1 -> A, 2 -> B, ..., 26 -> Z)
    int_to_letter = {i: letter for i, letter in enumerate(string.ascii_uppercase, start=1)}
    
    with open(residue_pdb_file, "r") as file:
        lines = file.readlines()
    
    with open(residue_pdb_file, "w") as file:
        for line in lines:
            if line.startswith("HETATM"):
                line = "ATOM  " + line[6:]
                
            if line.startswith("ATOM  "):
                # Extract the chain position number and convert to corresponding letter
                chain_position = int(line[22:26].strip())
                chain_letter = int_to_letter.get(chain_position, " ")  # Default to space if out of range
                
                # Replace chain position with corresponding letter, maintaining formatting
                line = f"{line[:21]}{chain_letter}{chain_position:4d}{line[26:]}"
            
            file.write(line)

def update_ligand_pdb(ligand_pdb_file, lig_3letter_code, ligand_chain):
    """Update ligand PDB file to replace ATOM with HETATM, set 3-letter code, chain ID, set residue number to 9, and align ZN."""
    with open(ligand_pdb_file, "r") as file:
        lines = file.readlines()
    
    with open(ligand_pdb_file, "w") as file:
        for line in lines:
            # Replace 'ATOM  ' with 'HETATM' and update the 3-letter code and chain ID
            if line.startswith("ATOM  "):
                line = "HETATM" + line[6:]
            if line.startswith("HETATM"):
                # Properly align columns for HETATM entries with chain ID and set residue number to 9
                atom_serial = line[6:11].strip()
                atom_name = line[12:16].strip()
                res_name = lig_3letter_code
                chain_id = ligand_chain
                res_seq = 9  # Set residue sequence number to 9
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                occupancy = float(line[54:60].strip())
                temp_factor = float(line[60:66].strip())
                element = line[76:78].strip()
                
                # Reformat with correct spacing and residue number set to 9
                line = f"HETATM{int(atom_serial):5d} {atom_name:<3} {res_name} {chain_id}{res_seq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temp_factor:6.2f}           {element:<2}\n"
                
                # Only shift ZN if it is the element at the end of the line
                if line.endswith("ZN\n"):
                    line = line[:-5] + " ZN \n"
            file.write(line)

def modify_pdb_atom_names(input_pdb):
    """Modify atom names in the PDB file to ensure uniqueness."""
    with open(input_pdb, 'r') as infile:
        lines = infile.readlines()
    
    atom_counters = {}
    with open(input_pdb, 'w') as outfile:
        for line in lines:
            if line.startswith("HETATM"):
                atom_type = line[12:14].strip()  # Extract original atom type (e.g., C, O, ZN)
                if atom_type not in atom_counters:
                    atom_counters[atom_type] = 1
                else:
                    atom_counters[atom_type] += 1

                # Create a unique atom name by appending the counter to the atom type
                unique_atom_name = f"{atom_type}{atom_counters[atom_type]}"
                
                # Ensure exact PDB spacing and formatting
                line = f"{line[:12]} {unique_atom_name:<4}{line[16:]}"
            outfile.write(line)

def clean_pdb_file(pdb_file):
    """Remove lines starting with COMPND or AUTHOR and convert all text to uppercase."""
    with open(pdb_file, "r") as infile:
        lines = infile.readlines()
    
    with open(pdb_file, "w") as outfile:
        for line in lines:
            if not line.startswith(("COMPND", "AUTHOR")):
                outfile.write(line.upper())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert theozyme XYZ to formatted PDB with ligand and residue separation.")
    parser.add_argument("-input_xyz", required=True, help="Input XYZ file for theozyme structure")
    parser.add_argument("-ligand_atom_ranges", required=True, help="Ranges of ligand atoms, e.g., '1-77'")
    parser.add_argument("-ligand_3letter_code", required=True, help="3-letter code for the ligand in the PDB file")
    parser.add_argument("-ligand_chain", default="Z", help="Chain ID for the ligand (default: Z)")
    parser.add_argument("-tip_atom_residues_3letter", nargs="+", required=True, help="3-letter codes for residues with specified side chains")
    parser.add_argument("-DO_NOT_pass_tip_atom_residues_3letter_to_help_identifier", action="store_false", help="Flag to NOT pass tip_atom_residues_3letter to identify_residues.py")
    parser.add_argument("-DO_NOT_make_params_or_mol2_files", action="store_true", help="Flag to skip the .mol2 and parameter file generation")
    parser.add_argument("-KEEP_temp_smiles_and_res_pdbs_for_debug", action="store_true", help="Flag to keep temporary SMILES and residue PDB files for debugging purposes")

    args = parser.parse_args()
    main(
        args.input_xyz,
        args.ligand_atom_ranges,
        args.ligand_3letter_code,
        args.ligand_chain,
        args.tip_atom_residues_3letter,
        args.DO_NOT_pass_tip_atom_residues_3letter_to_help_identifier,
        args.DO_NOT_make_params_or_mol2_files,
        args.KEEP_temp_smiles_and_res_pdbs_for_debug,
    )
