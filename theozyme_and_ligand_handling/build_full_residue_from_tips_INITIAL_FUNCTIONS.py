import os
from pyrosetta import *
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.conformation import ResidueFactory
import numpy as np

# Initialize PyRosetta
init('-ignore_unrecognized_res true -load_PDB_components false')

def read_pdb_residues(pdb_file):
    residues_info = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                res_name = line[17:20].strip()
                chain_id = line[21]
                res_num = int(line[22:26])
                residue_id = (chain_id, res_num)
                # Avoid duplicates
                if not any(res_info['id'] == residue_id for res_info in residues_info):
                    residues_info.append({
                        'resname': res_name,
                        'chain': chain_id,
                        'resnum': res_num,
                        'id': residue_id
                    })
    return residues_info

def generate_residue_pdbs(residues_info):
    residue_type_set = Pose().residue_type_set_for_pose()

    for res_info in residues_info:
        res_name = res_info['resname']
        chain_id = res_info['chain']
        res_num = res_info['resnum']

        try:
            res_type = residue_type_set.name_map(res_name)
        except Exception as e:
            print(f"Unknown residue type '{res_name}'. Skipping residue {res_num}{chain_id}.")
            continue
        
        # Create a residue and a new pose
        residue = ResidueFactory.create_residue(res_type)
        pose = Pose()
        pose.append_residue_by_jump(residue, 1)
        
        ### ADD MORE AMINO ACIDS IN HERE IF NEEDED ###
        ### SOME OF THE STANDARD RESIDUE GEOMETRIES ARE WEIRD FOR MAPPING ###
        ### FORCING CHI ANGLE CHANGES MAKES THE STRUCTURE DIVERSE ENOUGH ###
        ### FOR EFFICIENT MAPPING; COPY FORMAT BELOW ###

        # Rotate chi1 by 90 if the residue is HIS
        if res_name in ["HIS"]:
            current_chi1 = pose.chi(1, 1)
            pose.set_chi(1, 1, current_chi1 - 120.0)
            current_chi2 = pose.chi(2, 1)
            pose.set_chi(2, 1, current_chi2 - 120.0)
            print("##### !!!!!!!!!!!!!!!!!!!!!!! #####")
            print("##### STANDARD RESIDUE MODIFIED IN build_full_residue_from_tips_INITIAL_FUNCTIONS.py #####")

        # Rotate chi2 by 180 if the residue is ASP or GLU
        if res_name in ["ASP", "GLU"]:
            current_chi1 = pose.chi(1, 1)
            pose.set_chi(1, 1, current_chi1 - 90.0)
            current_chi2 = pose.chi(2, 1)
            pose.set_chi(2, 1, current_chi2 + 180.0)
            print("##### !!!!!!!!!!!!!!!!!!!!!!! #####")
            print("##### STANDARD RESIDUE MODIFIED IN build_full_residue_from_tips_INITIAL_FUNCTIONS.py #####")

        # Rotate chi2 by 90 if the residue is TYR
        if res_name in ["TYR"]:
            current_chi1 = pose.chi(1, 1)
            pose.set_chi(1, 1, current_chi1 - 120.0)
            current_chi2 = pose.chi(2, 1)
            pose.set_chi(2, 1, current_chi2 + 120.0)
            print("##### !!!!!!!!!!!!!!!!!!!!!!! #####")
            print("##### STANDARD RESIDUE MODIFIED IN build_full_residue_from_tips_INITIAL_FUNCTIONS.py #####")

        # Rotate chi2 by 90 if the residue is LYS
        if res_name in ["LYS"]:
            current_chi1 = pose.chi(1, 1)
            pose.set_chi(1, 1, current_chi1 - 180.0)
            current_chi2 = pose.chi(2, 1)
            pose.set_chi(2, 1, current_chi2 + 180.0)
            current_chi3 = pose.chi(3, 1)
            pose.set_chi(3, 1, current_chi3 + 180.0)
            current_chi4 = pose.chi(4, 1)
            pose.set_chi(4, 1, current_chi4 + 180.0)
            print("##### !!!!!!!!!!!!!!!!!!!!!!! #####")
            print("##### STANDARD RESIDUE MODIFIED IN build_full_residue_from_tips_INITIAL_FUNCTIONS.py #####")

        # Define the output filename
        output_filename = f"{res_name}{res_num}_rosetta_TEMP.pdb"

        # Write the PDB
        pose.dump_pdb(output_filename)

        # Manually modify the PDB file to set chain ID, residue number, and remove hydrogens
        modify_pdb_file(output_filename, chain_id, res_num)

        print(f"Generated standard residue PDB file: {output_filename}")

def generate_residue_pdbs_OUTDATED(residues_info):
    residue_type_set = Pose().residue_type_set_for_pose()
    for res_info in residues_info:
        res_name = res_info['resname']
        chain_id = res_info['chain']
        res_num = res_info['resnum']
        try:
            res_type = residue_type_set.name_map(res_name)
        except Exception as e:
            print(f"Unknown residue type '{res_name}'. Skipping residue {res_num}{chain_id}.")
            continue
        # Create a residue
        residue = ResidueFactory.create_residue(res_type)
        # Create a pose and append the residue
        pose = Pose()
        pose.append_residue_by_jump(residue, 1)
        # Define output filename for standard residue
        output_filename = f"{res_name}{res_num}_rosetta_TEMP.pdb"
        # Dump the pose to PDB
        pose.dump_pdb(output_filename)
        # Manually modify the PDB file to set chain ID and residue number, and remove hydrogens
        modify_pdb_file(output_filename, chain_id, res_num)
        print(f"Generated standard residue PDB file: {output_filename}")

def extract_residue_from_input_pdb(input_pdb, residues_info):
    # Read the entire input PDB file
    with open(input_pdb, 'r') as f:
        lines = f.readlines()
    for res_info in residues_info:
        res_name = res_info['resname']
        chain_id = res_info['chain']
        res_num = res_info['resnum']
        residue_lines = []
        atom_counts = {}
        for line in lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                line_res_num = int(line[22:26])
                line_chain_id = line[21]
                element = line[76:78].strip()
                if not element:
                    # Try to infer element from atom name
                    atom_name = line[12:16].strip()
                    element = ''.join(filter(str.isalpha, atom_name)).strip()
                    if len(element) > 1 and element[0].upper() == 'H':
                        element = 'H'
                    else:
                        element = element[0].upper()
                element = element.upper()
                if element == 'H':
                    continue  # Skip hydrogens
                if line_chain_id == chain_id and line_res_num == res_num:
                    if element not in atom_counts:
                        atom_counts[element] = 1
                    else:
                        atom_counts[element] += 1
                    temp_atom_name = f"{element}{atom_counts[element]}"
                    # Update the line with the temporary atom name
                    new_line = line[:12] + f"{temp_atom_name:<4}" + line[16:]
                    residue_lines.append(new_line)
        if residue_lines:
            output_filename = f"{res_name}{res_num}_inputpdb_TEMP.pdb"
            with open(output_filename, 'w') as f_out:
                f_out.writelines(residue_lines)
            print(f"Extracted residue from input PDB: {output_filename}")
        else:
            print(f"No lines found for residue {res_name}{res_num} in input PDB.")

def modify_pdb_file(pdb_filename, chain_id, res_num):
    with open(pdb_filename, 'r') as f:
        lines = f.readlines()
    with open(pdb_filename, 'w') as f:
        for line in lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # Skip hydrogens
                element = line[76:78].strip()
                if not element:
                    # Try to infer element from atom name
                    atom_name = line[12:16].strip()
                    element = ''.join(filter(str.isalpha, atom_name)).strip()
                    if len(element) > 1 and element[0].upper() == 'H':
                        element = 'H'
                    else:
                        element = element[0].upper()
                if element.upper() == 'H':
                    continue  # Skip writing hydrogens
                # Modify chain ID (column 22) and residue number (columns 23-26)
                line = line[:21] + chain_id + f"{res_num:>4}" + line[26:]
                f.write(line)
            else:
                f.write(line)

def read_pdb_atoms(pdb_filename, include_hydrogens=False):
    atoms = []
    with open(pdb_filename, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21]
                res_num = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                element = line[76:78].strip()
                if not element:
                    # Try to infer element from atom name
                    element = ''.join(filter(str.isalpha, atom_name)).strip()
                    if len(element) > 1 and element[0].upper() == 'H':
                        element = 'H'  # Handle cases like 'HG' for mercury vs 'HG' for hydrogen gamma
                    else:
                        element = element[0].upper()
                if not include_hydrogens and element.upper() == 'H':
                    continue  # Skip hydrogens
                atom = {
                    'atom_name': atom_name,
                    'element': element.upper(),
                    'coords': np.array([x, y, z]),
                    'line': line
                }
                atoms.append(atom)
    return atoms

def apply_transformation(atoms, rotation_matrix, translation_vector):
    transformed_atoms = []
    for atom in atoms:
        original_coords = atom['coords']
        transformed_coords = np.dot(original_coords, rotation_matrix) + translation_vector
        transformed_atom = atom.copy()
        transformed_atom['coords'] = transformed_coords
        transformed_atoms.append(transformed_atom)
    return transformed_atoms

def update_pdb_coordinates(pdb_filename, atoms):
    with open(pdb_filename, 'r') as f:
        lines = f.readlines()
    atom_index = 0
    with open(pdb_filename, 'w') as f:
        for line in lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # Skip hydrogens
                element = line[76:78].strip()
                if not element:
                    atom_name = line[12:16].strip()
                    element = ''.join(filter(str.isalpha, atom_name)).strip()
                    if len(element) > 1 and element[0].upper() == 'H':
                        element = 'H'
                    else:
                        element = element[0].upper()
                if element.upper() == 'H':
                    continue
                atom = atoms[atom_index]
                x, y, z = atom['coords']
                # Update the coordinates in the line
                new_line = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
                f.write(new_line)
                atom_index += 1
            else:
                f.write(line)
