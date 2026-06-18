'''
Author: Seth Woodbury
Date: 2024-07-15
Description: This script aligns ligand rotamers to an original crystal structure using a specified pair fitting method. 
It loads the original PDB file, aligns each rotamer to the original ligand, and saves the combined structures in new PDB files.

Input Arguments:
  --original_pdb: Path to the original crystal structure PDB file.
  --rotamer_dir: Directory containing the ligand rotamer PDB files.
  --output_dir: Directory to save the aligned PDB files.
  --atom_pairs: List of atom pairs for pair fitting in the format: atom1,[residue2,atom2] ...
  --ligands_to_remove_after_pair_fitting: Optional list of ligands (resn) to remove after pair fitting

Output: Combined PDB files saved in the specified output directory with the naming format "combined_<rotamer_name>.pdb".

EXAMPLE COMMAND USED BY SETH WOODBURY:
singularity exec /software/containers/crispy.sif python /home/woodbuse/special_scripts/theozyme_and_ligand_handling/align_ligand_rotamers_to_theozyme_pdb.py --original_pdb /home/woodbuse/for/don_hilvert/zinc_amidase_proj/theozymes/fluorogenic_amide_SM/dft_geom_opt/R_intermediate/cpa/cpa_8cpa_rfa_tip_atoms.pdb --rotamer_dir /home/woodbuse/for/don_hilvert/zinc_amidase_proj/theozymes/fluorogenic_amide_SM/dft_geom_opt/R_intermediate/cpa/theozyme_pdbs --output_dir /home/woodbuse/for/don_hilvert/zinc_amidase_proj/theozymes/fluorogenic_amide_SM/dft_geom_opt/R_intermediate/cpa/theozyme_pdbs/out --atom_pairs ZN1,[ZN,ZN] O3,[UNL,O2] O2,[UNL,O1] N1,[UNL,N1] C1,[UNL,C8] --ligands_to_remove_after_pair_fitting UNL ZN HOH
'''

import os
import argparse
from pymol import cmd

def parse_atom_pairs(atom_pairs_str):
    """
    Parse the atom pairs from the input string into a list of tuples.

    Args:
    - atom_pairs_str: List of strings in the format atom1,[residue2,atom2]

    Returns:
    - List of tuples in the format (atom1, residue2, atom2)
    """
    atom_pairs = []
    for pair in atom_pairs_str:
        atom1, rest = pair.split(",[")
        rest = rest.rstrip("]")
        residue2, atom2 = rest.split(",")
        atom_pairs.append((atom1, residue2, atom2))
    return atom_pairs

def pair_fit_command(rotamer, original, atom_pairs):
    """
    Generate the pair fit command for PyMOL.

    Args:
    - rotamer: Name of the rotamer object in PyMOL
    - original: Name of the original object in PyMOL
    - atom_pairs: List of tuples in the format (atom1, residue2, atom2)

    Returns:
    - A string representing the pair fit command for PyMOL
    """
    pair_fit_cmd = "pair_fit "
    for atom1, residue2, atom2 in atom_pairs:
        pair_fit_cmd += f"{rotamer} and name {atom1}, {original} and resn {residue2} and name {atom2}, "
    return pair_fit_cmd.rstrip(", ")

def align_and_save(original_pdb, rotamer_pdb, output_pdb, atom_pairs):
    """
    Aligns the rotamer PDB to the original PDB using the specified atom pairs and saves the combined structure.

    Args:
    - original_pdb: Path to the original PDB file
    - rotamer_pdb: Path to the rotamer PDB file
    - output_pdb: Path to save the combined PDB file
    - atom_pairs: List of tuples in the format (atom1, residue2, atom2)
    """
    cmd.reinitialize()
    cmd.load(original_pdb, "original")
    cmd.load(rotamer_pdb, "rotamer")

    print(f"Aligning {rotamer_pdb} to {original_pdb} using atom pairs: {atom_pairs}")
    pair_fit_cmd = pair_fit_command("rotamer", "original", atom_pairs)
    print(f"Executing pair fit command: {pair_fit_cmd}")
    print('')
    
    # Execute the pair fit command
    try:
        cmd.do(pair_fit_cmd)
        print("Pair fitting successful.")
    except Exception as e:
        print(f"Error during pair fitting: {e}")
        return

    # Combine structures
    cmd.create("combined", "original or rotamer")

    # Save the combined structure
    print(f"Saving combined structure to {output_pdb}")
    cmd.save(output_pdb, "combined")
    print('')

def manually_copy_protein_lines(original_pdb, output_pdb):
    """
    Manually copy protein lines from the original PDB to the output PDB if no protein is detected after alignment.

    Args:
    - original_pdb: Path to the original PDB file
    - output_pdb: Path to the output PDB file
    """
    print(f"Manually copying protein lines from {original_pdb} to {output_pdb}")
    with open(original_pdb, 'r') as orig_file:
        orig_lines = orig_file.readlines()
    
    with open(output_pdb, 'r') as output_file:
        output_lines = output_file.readlines()
    
    with open(output_pdb, 'w') as output_file:
        for line in orig_lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                output_file.write(line)
        for line in output_lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                if line[17:20].strip() == 'ROT':  # Assuming 'ROT' is the residue name for the rotamer
                    output_file.write(line)

def remove_ligands(output_pdb, ligands_to_remove):
    """
    Remove specified ligands from the PDB file.

    Args:
    - output_pdb: Path to the output PDB file
    - ligands_to_remove: List of ligand residue names to remove
    """
    print(f"Removing ligands {ligands_to_remove} from {output_pdb}")
    with open(output_pdb, 'r') as file:
        lines = file.readlines()
    
    with open(output_pdb, 'w') as file:
        for line in lines:
            if line.startswith("HETATM") and line[17:20].strip() in ligands_to_remove:
                continue
            file.write(line)

def process_files(original_pdb, rotamer_dir, output_dir, atom_pairs, ligands_to_remove_after_pair_fitting):
    """
    Process each rotamer file in the specified directory, align it to the original PDB, and save the combined structure.

    Args:
    - original_pdb: Path to the original PDB file
    - rotamer_dir: Directory containing the rotamer PDB files
    - output_dir: Directory to save the combined PDB files
    - atom_pairs: List of tuples in the format (atom1, residue2, atom2)
    - ligands_to_remove_after_pair_fitting: List of ligands to remove after pair fitting
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    total_rotamers = 0
    aligned_rotamers = 0

    for rotamer_file in os.listdir(rotamer_dir):
        if rotamer_file.endswith(".pdb"):
            total_rotamers += 1
            rotamer_path = os.path.join(rotamer_dir, rotamer_file)
            rotamer_name = os.path.splitext(rotamer_file)[0]
            output_pdb = os.path.join(output_dir, f"{rotamer_name}_theozyme.pdb")
            print('')
            print('##################### NEW BLOCK START #####################')
            print(f"Processing rotamer file: {rotamer_path}")
            align_and_save(original_pdb, rotamer_path, output_pdb, atom_pairs)
            
            # Verify protein presence
            try:
                cmd.load(output_pdb, "verify")
                protein_atoms = cmd.count_atoms("verify and polymer")
                cmd.delete("verify")
                if protein_atoms > 0:
                    print(f"Protein detected in {output_pdb} with {protein_atoms} atoms.")
                    print('###### LIGAND BLOCK ######')
                    aligned_rotamers += 1
                else:
                    print(f"No protein detected in {output_pdb}.")
            except Exception as e:
                print(f"Error checking protein presence: {e}")
                protein_atoms = 0
            
            if protein_atoms == 0:
                print(f"Manually copying protein lines to {output_pdb}.")
                manually_copy_protein_lines(original_pdb, output_pdb)
                print('')

            if ligands_to_remove_after_pair_fitting:
                print(f"Removing ligands: {ligands_to_remove_after_pair_fitting}")
                remove_ligands(output_pdb, ligands_to_remove_after_pair_fitting)
                print(f"Ligands removed from {output_pdb}.")
                print('######################## BLOCK END ########################')
                print('')
    
    print(f"Total rotamer PDBs: {total_rotamers}")
    print(f"Successfully aligned rotamer PDBs: {aligned_rotamers}")

def main():
    parser = argparse.ArgumentParser(description="Align ligand rotamers to an original crystal structure and save the combined structures.")
    parser.add_argument('--original_pdb', type=str, required=True, help='Path to the original crystal structure PDB file.')
    parser.add_argument('--rotamer_dir', type=str, required=True, help='Directory containing the ligand rotamer PDB files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the combined PDB files.')
    parser.add_argument('--atom_pairs', nargs='+', required=True, help='List of atom pairs for pair fitting in the format: atom1,[residue2,atom2] ...')
    parser.add_argument('--ligands_to_remove_after_pair_fitting', nargs='*', help='Optional list of ligands (resn) to remove after pair fitting (spaced list eg. ZN UNL HOH) remember that water is HOH')
    
    args = parser.parse_args()
    atom_pairs = parse_atom_pairs(args.atom_pairs)
    
    print("Starting ligand alignment process...")
    process_files(args.original_pdb, args.rotamer_dir, args.output_dir, atom_pairs, args.ligands_to_remove_after_pair_fitting)
    print("Ligand alignment process completed.")

if __name__ == "__main__":
    main()
