from Bio import PDB
from numpy.linalg import norm
import os, sys, glob
import json
import argparse
import numpy as np
import pandas as pd

def load_pdb(file_path):
    """Load a PDB file using Biopython."""
    parser = PDB.PDBParser(QUIET=True)
    return parser.get_structure(file_path.split("/")[-1], file_path)

def get_catalytic_residues(pdb_file):
    with open(pdb_file, 'r') as file:
        lines = file.readlines()
    
    catalytic_residues = []
    for line in lines:
        if "ATOM" in line:
            break
        if "REMARK 666" in line:
            parts = line.split()
            chain = parts[9]
            residue_name = parts[10]
            residue_number = int(parts[11])
            catalytic_residues.append({'res_num': residue_number, 'name3': residue_name, 'chain': chain})

    print(f"Catalytic residues identified in {pdb_file}: {catalytic_residues}")
    return catalytic_residues

def kabsch_alignment(coords1, coords2):
    """Align two sets of coordinates using the Kabsch algorithm."""
    # Center the coordinates by subtracting the mean
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)
    
    coords1_centered = coords1 - centroid1
    coords2_centered = coords2 - centroid2
    
    # Compute the covariance matrix
    H = np.dot(coords1_centered.T, coords2_centered)
    
    # Perform Singular Value Decomposition
    V, S, W = np.linalg.svd(H)
    
    # Ensure a proper rotation matrix (determinant should be +1)
    d = np.sign(np.linalg.det(V @ W))
    V[:, -1] *= d
    rotation_matrix = V @ W
    
    return rotation_matrix, centroid1, centroid2

def apply_rotation(coords, rotation_matrix, centroid1, centroid2):
    """Apply the rotation matrix to the coordinates and move the structure."""
    coords_centered = coords - centroid1
    coords_rotated = np.dot(coords_centered, rotation_matrix) + centroid2
    return coords_rotated

def calculate_rmsd(coords1, coords2):
    """Calculate RMSD between two sets of coordinates."""
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

def extract_coords_atom(structure, residue_name, atom_name):
    """Extract coordinates of a specific atom in a residue."""
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == residue_name:
                    for atom in residue:
                        if atom.get_name() == atom_name:
                            coords.append(atom.get_coord())
    return np.array(coords)

def extract_coords_res(structure, residue_num):
    """Extract coordinates of a specific atom in a residue."""
    coords, names = [], []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[1] == residue_num:
                    for atom in residue:
                        if atom.element == "H": 
                            continue
                        coords.append(atom.get_coord())
                        names.append(atom.get_name())
    return np.array(coords), names

def extract_ca_coords(structure):
    """Extract coordinates of all Cα atoms for alignment."""
    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_atom = residue['CA']
                    ca_coords.append(ca_atom.get_coord())
    return np.array(ca_coords)

def save_pdb(structure, output_file):
    """Save the structure to a PDB file."""
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_file)

def calculate_rmsd_after_alignment(structure1, structure2, catres_list, ligand_atom_mapping_list):
    """Main function to load, align, and calculate RMSD between two PDBs."""    
    # Extract Cα coordinates for alignment
    ca_coords1 = extract_ca_coords(structure1)
    ca_coords2 = extract_ca_coords(structure2)
    
    # Perform Kabsch alignment of Cα atoms
    rotation_matrix, centroid1, centroid2 = kabsch_alignment(ca_coords1, ca_coords2)
    for model in structure1:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom.set_coord(apply_rotation(np.array([atom.get_coord()]), rotation_matrix, centroid1, centroid2)[0])
                    
    # Save aligned PDB file if `dump_aligned_str` is True
    #if eval(args.dump_aligned_str):
    #    save_pdb(structure1, pdb1_path)
    
    # Calculate RMSD for the Ca atoms
    ca_coords1 = extract_ca_coords(structure1)
    ca_rmsd_value = calculate_rmsd(ca_coords1, ca_coords2)
    
    # Calculate RMSD for the catalytic residues
    catres_rmsds = []
    for catres in catres_list:     
        catres_coords1, catres_atom_name1 = extract_coords_res(structure1, catres["res_num"])
        catres_coords2, catres_atom_name2 = extract_coords_res(structure2, catres["res_num"])
        if catres_atom_name1 != catres_atom_name2:
            raise ValueError (f"Catalytic residue atom names in Chai-1 [{catres_atom_name1}] and reference structures [{catres_atom_name2}] are different.")
        catres_rmsds.append(calculate_rmsd(catres_coords1, catres_coords2))
    
    # Calculate RMSD for the ligand atoms
    ligand_rmsd_values = []
    
    for ligand_atom_mapping in ligand_atom_mapping_list:
        label, name3, atoms = ligand_atom_mapping["label"], ligand_atom_mapping["name3"], ligand_atom_mapping["atoms"]
        
        # Extract the coordinates for the specified atoms (from atom_mapping)
        ligand_coords1, ligand_coords2 = [], []
        for atom1, atom2 in zip(atoms[0], atoms[1]):
            res1, res2 = name3[0], name3[1]
            ligand_coords1.append(extract_coords_atom(structure1, res1, atom1))
            ligand_coords2.append(extract_coords_atom(structure2, res2, atom2))

        # Flatten the list of coordinates (there should be one coordinate set for each pair)
        ligand_coords1 = np.concatenate(ligand_coords1)
        ligand_coords2 = np.concatenate(ligand_coords2)

        # Calculate RMSD for the specified ligand atoms
        ligand_rmsd_values.append(calculate_rmsd(ligand_coords1, ligand_coords2))
    
    return ca_rmsd_value, catres_rmsds, ligand_rmsd_values

def get_plddt_per_residue_dic(chai_pdb):
    chai_pdbstr = open(chai_pdb, 'r').read().split("\n")
    plddt_dic = {}
    for line in chai_pdbstr:
        if not line[:6] in ["ATOM  ", "HETATM"]:
            continue
        if len(line.strip()) < 66:
            continue
        ch_id = line[21:23].strip()
        res = int(line[23:27].strip())
        key = (ch_id, res)
        
        if not key in plddt_dic: plddt_dic[key] = {}
        plddt_dic[key][line[13:16].strip()] = float(line[61:67])
    plddt_per_residue_dic = {key: np.average(list(plddt_dic[key].values())) for key in plddt_dic}
    return plddt_per_residue_dic

def get_avr_plddt(chai_pdb, ligand_groups_json):
    chai_pdbstr = open(chai_pdb, 'r').read().split("\n")
    plddt_list = []
    for ligand in args.ligand_groups_json:
        plddt_list.append([])
        for line in chai_pdbstr:
            if not line[:6] in ["ATOM  ", "HETATM"]:
                continue
            if len(line.strip()) < 66:
                continue
            atom = line[12:17].strip()
            name3 = line[17:21].strip()
            plddt = float(line[61:67])
            if (name3 == ligand["name3"][0] and atom in ligand["atoms"][0]):
                plddt_list[-1].append(plddt)
    return np.mean(plddt_list)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_pdb", help="Path of reference PDB.")
    parser.add_argument("--chai_dir", help="Path of that directory with five Chai-1 PDB predictions.")
    parser.add_argument("--outscr", type=str, help="Output score file path.")
    parser.add_argument("--ligand_groups_json", type=str, help='JSON string defining atom groups and labels.')
    args = parser.parse_args()

    args.ligand_groups_json = json.loads(args.ligand_groups_json)
    
    updated_score_dic = {"description": args.chai_dir}

    ref_structure = load_pdb(args.ref_pdb)
    catres_list = get_catalytic_residues(args.ref_pdb)

    chai_pdbs = sorted(glob.glob(os.path.join(args.chai_dir, "*.pdb")))
    for chai_pdb_num, chai_pdb in enumerate(chai_pdbs):
        chai_structure = load_pdb(chai_pdb)
        chai_ch_ids = [chain.id for model in chai_structure for chain in model]
        
        # Save other metrics from Chai-1 (iptm, ptm, aggregate_score)    
        with open(chai_pdb.replace("pred.", "scores.").replace(".pdb", ".json"), "r") as file:
            scores = json.load(file)
        updated_score_dic[f"iptm_idx_{chai_pdb_num}"] = scores["iptm"][0]
        updated_score_dic[f"ptm_idx_{chai_pdb_num}"] = scores["ptm"][0]
        updated_score_dic[f"aggregate_score_idx_{chai_pdb_num}"] = scores["aggregate_score"][0]

        for i, ch_id in enumerate(chai_ch_ids):
            updated_score_dic[f"chain{ch_id}_plddt_idx_{chai_pdb_num}"] = scores["per_chain_plddt"][0][i]
        
        ca_rmsd, catres_rmsds, ligand_rmsds = calculate_rmsd_after_alignment(chai_structure, ref_structure, catres_list, args.ligand_groups_json)
        ligand_plddts = get_avr_plddt(chai_pdb, args.ligand_groups_json)
        
        # Save Ca RMSD
        updated_score_dic[f"ca_rmsd_idx_{chai_pdb_num}"] = ca_rmsd
        
        # Save ligand RMSD
        for i, (ligand_rmsd, ligand_plddt) in enumerate(zip(ligand_rmsds, ligand_plddts)):
            updated_score_dic[f"{args.ligand_groups_json[i]['label']}_rmsd_idx_{chai_pdb_num}"] = ligand_rmsd
            updated_score_dic[f"{args.ligand_groups_json[i]['label']}_plddt_idx_{chai_pdb_num}"] = ligand_plddt
            
        # Save catalytic residue RMSD
        for i, catres in enumerate(catres_list):
            updated_score_dic[f"{catres['name3']}{i+1}_rmsd_idx_{chai_pdb_num}"] = catres_rmsds[i]
        updated_score_dic[f"catres_rmsd_idx_{chai_pdb_num}"] = np.mean(catres_rmsds[i])

        # Save catalytic residue plDDT
        plddt_per_residue_dic = get_plddt_per_residue_dic(chai_pdb)
        cat_res_plddts = []
        for i, catres in enumerate(catres_list):
            cat_res_plddt = np.mean([plddt_per_residue_dic[el] for el in filter(lambda el: el[1] == catres["res_num"], plddt_per_residue_dic.keys())])
            updated_score_dic[f"{catres['name3']}{i+1}_plddt_idx_{chai_pdb_num}"] = cat_res_plddt
            cat_res_plddts.append(cat_res_plddt)
        updated_score_dic[f"catres_plddt_idx_{chai_pdb_num}"] = np.mean(cat_res_plddts)
        
    score_df = pd.DataFrame([updated_score_dic])
    score_df.to_csv(args.outscr, index=False)
        
    print (score_df)
    