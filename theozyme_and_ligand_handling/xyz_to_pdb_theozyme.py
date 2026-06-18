"""
Author: Seth M. Woodbury
Email: woodbuse@uw.edu
Date Created: 2024-12-04

### Script Overview ###
This script processes a theozyme .xyz file, separates it into residues and ligand files, modifies PDB formats, aligns residues, and prepares a final PDB file containing the theozyme with a backbone constructed from user-provided residue tips. The final output includes combined residues and ligand PDBs and ensures proper formatting for downstream applications. The script also provides options for debugging by retaining intermediate files. Additionally, the script can standardize the labeling of ASP/GLU tip atoms (OE1/OE2 or OD1/OD2) based on proximity to a specified ligand atom.

### Example Command ###
python path/to/xyz_to_pdb_theozyme.py -input_xyz theozyme_input.xyz -ligand_atom_ranges "1-77" -ligand_3letter_code LIG -ligand_chain Z -tip_atom_residues_3letter HIS GLU LYS --ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp H1 -KEEP_temp_residue_and_temp_ligand_files_for_debug

### Flags Explanation ###
- **-input_xyz** (required): The input XYZ file representing the theozyme structure.
  Example: `-input_xyz theozyme_input.xyz`

- **-ligand_atom_ranges** (required): A comma-separated range of ligand atoms from the input XYZ file. Atoms not included here are treated as residues.
  Example: `-ligand_atom_ranges "1-77"`

- **-ligand_3letter_code** (required): The 3-letter code for the ligand, used in the PDB file.
  Example: `-ligand_3letter_code LIG`

- **-ligand_chain** (optional): The chain ID to assign to the ligand. Default is "Z".
  Example: `-ligand_chain Z`

- **-tip_atom_residues_3letter** (required): A space-separated list of residue 3-letter codes corresponding to the tip residues used in the theozyme design.
  Example: `-tip_atom_residues_3letter HIS GLU LYS`

- **--ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp** (optional): Specifies an atom in the ligand (e.g., H1) to standardize the labeling of ASP/GLU atoms (OE1/OE2 or OD1/OD2). If provided, the script evaluates the proximity of ligand atom `H1` to OE1 and OE2 in GLU residues (or OD1 and OD2 in ASP residues) and adjusts atom names if necessary to ensure proper labeling. This step is skipped if the argument is not provided.

- **-DO_NOT_pass_tip_atom_residues_3letter_to_help_identifier** (optional): Disables passing the residue tips to the helper script. Default is `True`.

- **-make_params_or_mol2_files** (optional): Skips generation of .mol2 and .params files for the ligand. Default is `True` (files are generated).

- **-KEEP_temp_smiles_and_res_pdbs_for_debug** (optional): Retains intermediate SMILES and residue PDB files for debugging purposes. Default is `False`.

- **-KEEP_temp_residue_and_temp_ligand_files_for_debug** (optional): Retains intermediate residue and ligand files (e.g., `*_TEMP.pdb` files) for debugging purposes. Default is `False`.

### Required Commands ###
The following commands must always be provided:
- `-input_xyz`
- `-ligand_atom_ranges`
- `-ligand_3letter_code`
- `-tip_atom_residues_3letter`

### Optional Commands ###
The following commands can be omitted and will default to the specified behavior:
- `-ligand_chain`
- `--ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp`
- `-DO_NOT_pass_tip_atom_residues_3letter_to_help_identifier`
- `-make_params_or_mol2_files`
- `-KEEP_temp_smiles_and_res_pdbs_for_debug`
- `-KEEP_temp_residue_and_temp_ligand_files_for_debug`

### Script Workflow ###
1. **Input Parsing and Setup:**
   - Reads the provided .xyz file and separates it into residue and ligand files based on the specified atom ranges.

2. **File Conversion:**
   - Converts XYZ files into PDB format using Open Babel.
   - Cleans and modifies PDB files to ensure proper formatting (e.g., chain IDs, residue numbers).

3. **Residue and Ligand Alignment:**
   - Aligns residues using tip atoms and updates residue structures with missing atoms.

4. **Standardization of ASP/GLU Tip Atoms (Optional):**
   - If the argument `--ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp` is provided, the script evaluates the proximity of the specified ligand atom to OE1/OE2 in GLU or OD1/OD2 in ASP residues.
   - Based on the evaluation, the script updates the atom names in the final PDB file to ensure proper labeling.

5. **PDB Assembly:**
   - Combines residues and ligand PDB files into a single final theozyme PDB file.

6. **Debugging and Cleanup:**
   - Provides options to retain intermediate files for debugging.
   - Deletes temporary files unless flags are specified.

### Referenced Subscripts ###
The following subscripts are called during execution:
1. **build_full_residue_from_tips.py:** Reconstructs residues from tip atoms.
2. **build_full_residue_from_tips_OPTIMAL_SUPERIMPOSE_donghyo.py:** Aligns residues to their tip atoms.
3. **build_full_residue_from_tips_CORRECT_RESIDUE_CHAIN_AND_NUMBER.py:** Corrects residue chain IDs and numbering.
4. **build_full_residue_from_tips_PART1_INSTALL_MISSING_ATOMS.py:** Installs missing atoms into aligned residues.
5. **build_full_residue_from_tips_PART2_INSTALL_MISSING_ATOMS.py:** Combines residues and ligand for further processing.
6. **build_full_residue_from_tips_PART3_INSTALL_MISSING_ATOMS.py:** Finalizes residue construction with proper numbering.
7. **combine_residuesPDB_w_ligandPDB.py:** Combines residue and ligand PDB files into the final theozyme.
8. **identify_residues.py:** Identifies and maps residues using Rosetta.
9. **renumber_pdb.py:** Renumbers residue indices for consistency.
10. **standardize_GLU_ASP_tip_atom_labeling_based_on_proximity_to_atomOFinterest.py:** Adjusts ASP/GLU labeling based on ligand proximity.

### Author Notes ###
This script was developed to streamline theozyme design workflows, ensuring accurate residue reconstruction and PDB assembly. For inquiries or contributions, please contact Seth M. Woodbury at woodbuse@uw.edu.

DEBUG: Sometimes Rosetta residues may be incorrectly aligned to your "tip atoms." Rosetta standard residues come from energy minimized models, so sometimes the residues look "a little too symmetric" and thus it is hard to find the correct global alignment.
DEBUG (cont): A simple fix to attempt for this is to "make the residue look less symmetric" by altering the chi angles of the residue(s) that are not aligning correctly. To do this, go to the "build_full_residue_from_tips_INITIAL_FUNCTIONS.py" script and add it in (you'll find the section). It is coded like this for HIS, GLU, ASP, and TYR so far.
DEBUG_V2: Sometimes you need to adjust the "threshold" in the build_full_residue_from_tips_OPTIMAL_SUPERIMPOSE_donghyo.py in the is_reasonable_pairing function.
"""
#!/usr/bin/env python3

import argparse
import os
import string
import glob
import shutil
import sys


# Define the main function
def main(input_xyz, ligand_atom_ranges, lig_3letter_code, ligand_chain, tip_atom_residues_3letter, DO_NOT_pass_tip_atom_residues_3letter_to_help_identifier, make_params_or_mol2_files, KEEP_temp_smiles_and_res_pdbs_for_debug, KEEP_temp_residue_and_temp_ligand_files_for_debug, ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp=None):
    # Parse ligand atom ranges
    ligand_atoms = parse_atom_ranges(ligand_atom_ranges)
    
    # Generate temporary XYZ files for protein residues and ligand
    temp_residue_xyz_file = create_temp_xyz(input_xyz, ligand_atoms, exclude=True)
    temp_ligand_xyz_file = create_temp_xyz(input_xyz, ligand_atoms, exclude=False)
    
    # Convert residue XYZ to PDB using Open Babel
    residue_pdb_file = temp_residue_xyz_file.replace("TEMP.xyz", "residues_TEMP.pdb")
    obabel_path = "/home/woodbuse/conda_envs/openbabel_env/bin/obabel"
    #os.system(f"{obabel_path} -ixyz {temp_residue_xyz_file} -opdb -O {residue_pdb_file}") ## OLD ##
    
    ### NEW DEBUG SECTION ###########################################################################
    os.system(f"{obabel_path} -ixyz {temp_residue_xyz_file} --separate -opdb -O {residue_pdb_file}")
 
    # --- POST‑PROCESS the --separate output into a single, clean PDB ---
    processed = []
    with open(residue_pdb_file, 'r') as infile:
        model_idx   = 0
        atom_serial = 1

        for line in infile:
            if line.startswith("MODEL"):
                model_idx += 1
                continue
            if line.startswith("ENDMDL"):
                continue
            if line.startswith(("ATOM","HETATM")):
                # rebuild the line:
                #  1) record name        = cols  1–6 (line[:6])
                #  2) new atom serial    = cols  7–11
                #  3) original cols 12–22 = atom name, altLoc, resName, chainID (line[11:22])
                #  4) new residue seq    = cols 23–26
                #  5) remainder          = from col 27 onward (line[26:])
                record        = line[:6]
                atom_and_rest = line[11:22]
                tail          = line[26:]
                new_line = f"{record}{atom_serial:5d}{atom_and_rest}{model_idx:4d}{tail}"
                processed.append(new_line)
                atom_serial += 1

    # overwrite the PDB with only your cleaned ATOM/HETATM records
    with open(residue_pdb_file, 'w') as outfile:
        outfile.writelines(processed)

    # now you can sys.exit() or continue on with the cleaned PDB
    #sys.exit("Stopping here") 

    ### END DEBUG ###########################################################################

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

    print(f"### {residue_pdb_file} has been updated with correct amino acid codes ###")
    print("")
    print("### MOVING ON TO MAKING .params FILE ###")

    if not make_params_or_mol2_files:
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

    # Call the script to extract the residues from the residues_TEMP
    # Then build the corresponding rosetta pdbs (e.g., build a full his for histidine)
    # Then partial dock/align them
    build_residue_script = "/software/containers/crispy.sif /home/woodbuse/special_scripts/theozyme_and_ligand_handling/build_full_residue_from_tips.py"
    os.system(f"{build_residue_script} -input_pdb {residue_pdb_file}")
    print("\n### BUILT ROSETTA RESIDUE FILES FOR ALIGNMENT WITH THE CORRESPONDING TIPS ###\n")


    ### STOP HERE IF YOU WANT TO SEE ROSETTA PDBS FOR ADJUSTING CHI ANGLES ###
    # sys.exit("Stopping here") 
    ### STOP HERE IF YOU WANT TO SEE ROSETTA PDBS FOR ADJUSTING CHI ANGLES ###

    # Find all matching *_inputpdb_TEMP.pdb and *_rosetta_TEMP.pdb files
    input_files = glob.glob("*_inputpdb_TEMP.pdb")
    rosetta_files = glob.glob("*_rosetta_TEMP.pdb")

    # Match files based on the prefix (first 3 letters and numerical index)
    pairs = []
    for input_file in input_files:
        prefix = input_file.split("_inputpdb_TEMP.pdb")[0]
        rosetta_file = f"{prefix}_rosetta_TEMP.pdb"
        if rosetta_file in rosetta_files:
            pairs.append((input_file, rosetta_file))

    # Print all identified pairs
    print("\n### MATCHING PAIRS FOUND ###")
    for input_file, rosetta_file in pairs:
        print(f"Input: {input_file}, Rosetta: {rosetta_file}")

    ###################################### IMPORTANT NOTE ######################################
    # added --omit_backbone_but_keep_CA !! this prevents backbones from being part of the matching process
    # Execute the command for each pair
    build_script = "/net/software/containers/crispy.sif /home/woodbuse/special_scripts/theozyme_and_ligand_handling/build_full_residue_from_tips_OPTIMAL_SUPERIMPOSE_donghyo.py"
    for input_file, rosetta_file in pairs:
        command = f"{build_script} -i {input_file} -r {rosetta_file} -o aligned_rosetta.pdb --omit_backbone_but_keep_CA" # OR YOU CAN DO --omit_all_backbone
        os.system(command)
        print(f"\n### EXECUTED: {command} ###")
    print("")
    print("### ROSETTA RESIDUES SUCCESSFULLY ALIGNED WITH TIPS ###")
    ###################################### IMPORTANT NOTE ######################################


    # Find all *_rosetta_TEMP.pdb and *_inputpdb_TEMP_aligned_rosetta.pdb files
    rosetta_files = glob.glob("*_rosetta_TEMP.pdb")
    aligned_rosetta_files = glob.glob("*_inputpdb_TEMP_aligned_rosetta.pdb")

    # Match files based on the prefix (amino acid code and residue number)
    aligned_pairs = []
    for rosetta_file in rosetta_files:
        prefix = rosetta_file.split("_rosetta_TEMP.pdb")[0]
        aligned_file = f"{prefix}_inputpdb_TEMP_aligned_rosetta.pdb"
        if aligned_file in aligned_rosetta_files:
            aligned_pairs.append((rosetta_file, aligned_file))

    # Print all identified pairs
    print("\n### MATCHING ALIGNED PAIRS FOUND ###")
    for rosetta_file, aligned_file in aligned_pairs:
        print(f"Rosetta: {rosetta_file}, Aligned: {aligned_file}")

    # Execute the command for each pair
    correct_residue_script = "/home/woodbuse/special_scripts/theozyme_and_ligand_handling/build_full_residue_from_tips_CORRECT_RESIDUE_CHAIN_AND_NUMBER.py"
    for rosetta_file, aligned_file in aligned_pairs:
        command = f"python {correct_residue_script} -input_rosetta_residue {rosetta_file} -input_tip_atom_aligned_rosetta_residue {aligned_file}"
        os.system(command)
        print(f"\n### EXECUTED: {command} ###")

    print("")
    print("### RENAMED TIP ATOM ALIGNED ROSETTA RESIDUES ###")

    # Gather all aligned rosetta PDB files
    aligned_rosetta_files = glob.glob("*_inputpdb_TEMP_aligned_rosetta.pdb")

    # Print all gathered aligned_rosetta PDB files
    print("\n### ALIGNED ROSETTA PDB FILES ###")
    for aligned_file in aligned_rosetta_files:
        print(aligned_file)

    # Construct the command to install missing atoms
    install_missing_atoms_script = "/home/woodbuse/special_scripts/theozyme_and_ligand_handling/build_full_residue_from_tips_PART1_INSTALL_MISSING_ATOMS.py"
    output_pdb = "single_file_aligned_rosetta_pdb_residues_TEMP.pdb"
    aligned_files_argument = " ".join(aligned_rosetta_files)
    command = f"python {install_missing_atoms_script} -aligned_rosetta_pdbs_for_residues {aligned_files_argument} -output_pdb {output_pdb}"

    # Execute the command
    os.system(command)
    print(f"\n### EXECUTED: {command} ###")
    print(f"Output PDB: {output_pdb}")

    # Define the script path and fixed filenames
    part2_script = "/home/woodbuse/special_scripts/theozyme_and_ligand_handling/build_full_residue_from_tips_PART2_INSTALL_MISSING_ATOMS.py"
    aligned_rosetta_single_pdb = "single_file_aligned_rosetta_pdb_residues_TEMP.pdb"
    semi_final_output_pdb = "SEMI_FINAL_RESIDUE_CONSTRUCTION_TEMP.pdb"

    # Construct the command using the previously created residues_TEMP.pdb
    command = f"python {part2_script} -aligned_rosetta_single_pdb_of_residues {aligned_rosetta_single_pdb} -dft_tip_atoms_of_residues_only {residue_pdb_file} -output_pdb {semi_final_output_pdb}"

    # Execute the command
    os.system(command)
    print(f"\n### EXECUTED: {command} ###")
    print(f"Output PDB: {semi_final_output_pdb}")

    # Define the script path and filenames for PART3
    part3_script = "/home/woodbuse/special_scripts/theozyme_and_ligand_handling/build_full_residue_from_tips_PART3_INSTALL_MISSING_ATOMS.py"
    semi_final_pdb = "SEMI_FINAL_RESIDUE_CONSTRUCTION_TEMP.pdb"
    final_output_pdb = "final_residue_construction.pdb"

    # Construct and execute the PART3 command
    command = f"python {part3_script} -input_pdb {semi_final_pdb} -output_pdb {final_output_pdb}"
    os.system(command)
    print(f"\n### EXECUTED: {command} ###")
    print(f"Output PDB: {final_output_pdb}")

    # Make a copy of ligand_TEMP.pdb with "only" in its name
    new_ligand_pdb = ligand_pdb_file.replace("ligand_TEMP", "only")
    shutil.copy(ligand_pdb_file, new_ligand_pdb)
    print(f"\n### COPIED: {ligand_pdb_file} to {new_ligand_pdb} ###")

    # Define the script path for combining residues and ligand PDBs
    combine_script = "/home/woodbuse/special_scripts/theozyme_and_ligand_handling/combine_residuesPDB_w_ligandPDB.py"
    residues_pdb = "final_residue_construction.pdb"

    # Construct the output filename based on the input XYZ file
    output_pdb = os.path.basename(input_xyz).replace(".xyz", "_artificialBB_theozyme.pdb")

    # Construct and execute the combine command
    command = f"python {combine_script} -residues_pdb {residues_pdb} -ligand_pdb {ligand_pdb_file} -output_pdb {output_pdb}"
    os.system(command)
    print(f"\n### EXECUTED: {command} ###")
    print(f"Combined PDB Output: {output_pdb}")

    # Get the directory of the input XYZ file
    input_xyz_dir = os.path.dirname(input_xyz)

    # Create subdirectories within the input XYZ file's directory
    pdb_theozymes_folder = os.path.join(input_xyz_dir, "pdb_theozymes")
    pdb_ligands_only_folder = os.path.join(input_xyz_dir, "pdb_theozyme_ligands_only")

    os.makedirs(pdb_theozymes_folder, exist_ok=True)
    os.makedirs(pdb_ligands_only_folder, exist_ok=True)

    ### NEW CODE SECTION FOR OPTIONALLY STANDARDIZING OE2/OE1 & OD2/OD1 FOR EVERY GLU/ASP RESIDUE ###
    # If user provided a ligand atom for close proximity standardization
    if ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp:
        standardize_script = "/home/woodbuse/special_scripts/theozyme_and_ligand_handling/standardize_GLU_ASP_tip_atom_labeling_based_on_proximity_to_atomOFinterest.py"
        command = (
            f"python {standardize_script} "
            f"--input_pdb {output_pdb} "
            f"--ligand_code {lig_3letter_code} "
            f"--ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp {ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp}"
        )
        print(f"\n### EXECUTING STANDARDIZATION: {command}")
        os.system(command)
        print("### DONE WITH STANDARDIZATION ###\n")
        print("")

    # Move the output PDB file into the pdb_theozymes folder
    shutil.move(output_pdb, os.path.join(pdb_theozymes_folder, output_pdb))
    print(f"\n### MOVED: {output_pdb} to {pdb_theozymes_folder}/ ###")

    # Move the ligand "only" PDB file into the pdb_theozyme_ligands_only folder
    only_ligand_pdb = ligand_pdb_file.replace("ligand_TEMP", "only")
    shutil.move(only_ligand_pdb, os.path.join(pdb_ligands_only_folder, os.path.basename(only_ligand_pdb)))
    print(f"\n### MOVED: {only_ligand_pdb} to {pdb_ligands_only_folder}/ ###")

    # Cleanup intermediate files unless the debug flag is set
    if not KEEP_temp_residue_and_temp_ligand_files_for_debug:
        print("\n### CLEANING UP INTERMEDIATE FILES ###")

        # List of specific files to delete
        files_to_delete = [
            "SEMI_FINAL_RESIDUE_CONSTRUCTION_TEMP.pdb",
            "single_file_aligned_rosetta_pdb_residues_TEMP.pdb",
            "final_residue_construction.pdb",
            ligand_pdb_file,
            residue_pdb_file,
        ]

        # Add patterns for glob-based deletion
        patterns_to_delete = [
            "*_inputpdb_TEMP.pdb",
            "*_inputpdb_TEMP_aligned_rosetta.pdb",
            "*_rosetta_TEMP.pdb",
        ]

        # Delete specific files
        for file in files_to_delete:
            if os.path.exists(file):
                os.remove(file)
                print(f"Deleted: {file}")

        # Delete files matching patterns
        for pattern in patterns_to_delete:
            matching_files = glob.glob(pattern)
            for file in matching_files:
                os.remove(file)
                print(f"Deleted: {file}")
    else:
        print("\n### KEEPING INTERMEDIATE FILES FOR DEBUGGING ###")

    print("")
    print("### POST-SCRIPT CLARITY DEBUG V1: If stuff is not correctly aligned in outputs, read the top of this script. Try changing standard residue Chi angles in build_full_residue_from_tips_INITIAL_FUNCTIONS.py ###")
    print("### POST-SCRIPT CLARITY DEBUG V2: If the backbone of a residue is missing, read the top of this script. Try changing the threshold in the function 'is_reasonable_pairing' in the script build_full_residue_from_tips_OPTIMAL_SUPERIMPOSE_donghyo.py ###")
    print("")
    print("")
    print("### SCRIPT COMPLETED SUCCESSFULLY :D ###")

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
    parser.add_argument("-make_params_or_mol2_files", action="store_false", help="Flag to make the .mol2 and parameter file generation | NOT RECOMMENDED TO DO HERE... do with separate script")
    parser.add_argument("-KEEP_temp_smiles_and_res_pdbs_for_debug", action="store_true", help="Flag to keep temporary SMILES and residue PDB files for debugging purposes")
    parser.add_argument("-KEEP_temp_residue_and_temp_ligand_files_for_debug", action="store_true", help="Flag to keep intermediate residue and ligand files for debugging purposes")
    parser.add_argument("--ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp", default=None, help="Atom in the ligand used for proximity-based standardization of ASP/GLU tip atoms (e.g. H1).")

    args = parser.parse_args()
    main(
        args.input_xyz,
        args.ligand_atom_ranges,
        args.ligand_3letter_code,
        args.ligand_chain,
        args.tip_atom_residues_3letter,
        args.DO_NOT_pass_tip_atom_residues_3letter_to_help_identifier,
        args.make_params_or_mol2_files,
        args.KEEP_temp_smiles_and_res_pdbs_for_debug,
        args.KEEP_temp_residue_and_temp_ligand_files_for_debug,
        args.ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp,  # <-- new arg here
    )
