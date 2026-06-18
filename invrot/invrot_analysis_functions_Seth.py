"""
Created 2024-03-22 by Seth Woodbury (woodbuse@uw.edu)
This script is a living document meant for analysis functions to parse/filter/understand inverse rotamer outputs.
"""
import os
import json
import argparse
import pandas as pd
import multiprocessing
import glob
from pyrosetta import *
from pyrosetta.rosetta.core.import_pose import pose_from_file
from pyrosetta.rosetta.core.chemical import ChemicalManager

def init_pyrosetta_with_params(params_path):
    """
    Initialize PyRosetta with parameter files, specifically avoiding automatic termini addition.
    """
    options = f"-extra_res_fa {params_path} -ignore_unrecognized_res true -preserve_header true"
    init(options=options)
    print(f"PyRosetta initialized with parameters from: {params_path}")

def adjust_residue_types(pose):
    """
    Explicitly set all residues to be non-terminal if required. This function may need to
    manually adjust each residue's type based on your specific requirements.
    """
    # This section would need to be implemented based on the requirements and understanding
    # of how residues are treated in your specific PDB files and PyRosetta version.
    pass  # Placeholder for potential residue adjustment logic.

def process_pdb_file(pdb_path, atom_pairs):
    """
    Process a single PDB file to measure distances between specified atom pairs,
    ensuring residues are not incorrectly marked as termini.
    """
    pose = pose_from_pdb(pdb_path)
    adjust_residue_types(pose)  # Call adjust function if implementation is needed
    file_results = {}
    for pair_name, atoms in atom_pairs.items():
        try:
            atom1_info = atoms['atom1']
            atom2_info = atoms['atom2']
            atom1 = pose.residue(atom1_info['residue']).atom(atom1_info['atom_name'])
            atom2 = pose.residue(atom2_info['residue']).atom(atom2_info['atom_name'])
            distance = atom1.xyz().distance(atom2.xyz())
            file_results[pair_name] = distance
        except Exception as e:
            print(f"Error processing {pdb_path} for {pair_name}: {str(e)}")
            file_results[pair_name] = None
    file_results['pdb_path'] = pdb_path
    return file_results

def main():
    parser = argparse.ArgumentParser(description='Measure distances between atoms in PDB files using PyRosetta.')
    parser.add_argument('directory', type=str, help='Directory containing PDB files')
    parser.add_argument('atom_pairs_json', type=str, help='JSON string of atom pairs')
    parser.add_argument('params_path', type=str, help='Path to the .params file for non-standard residues')
    
    args = parser.parse_args()
    
    init_pyrosetta_with_params(args.params_path)
    
    with open(args.atom_pairs_json, 'r') as file:
        atom_pairs = json.load(file)
    
    pdb_files = glob.glob(os.path.join(args.directory, '*.pdb'))
    
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    tasks = [(pdb_file, atom_pairs) for pdb_file in pdb_files]
    all_results = pool.starmap(process_pdb_file, tasks)
    
    pool.close()
    pool.join()

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.directory, "invrot_analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results written to {csv_path}")

if __name__ == "__main__":
    main()
    
### Indrek Version ###
#df = pd.DataFrame()
#for i, p in enumerate(pdbs):
 #   pose = pyr.pose_from_file(p)
 #   dist1 = (pose.residue(pose.size()).xyz("O4") - pose.residue(5).xyz("N")).norm()
 #   dist2 = (pose.residue(pose.size()).xyz("O2") - pose.residue(5).xyz("O")).norm()
 #   df.at[i, "OO_dist"] = dist2
 #   df.at[i, "NO_dist"] = dist1
 #   df.at[i, "description"] = p

#with open("distances.csv", "w") as file:
 #   df.write_csv(file)