import pandas as pd
import argparse
import pyrosetta
from pyrosetta import rosetta

# Initialize PyRosetta
pyrosetta.init()

def get_hydrogen_bonds(pose):
    hbond_set = pose.get_hbonds()
    hbonds = []
    for i in range(1, hbond_set.nhbonds() + 1):
        hbond = hbond_set.hbond(i)
        donor_res = hbond.don_res()
        acceptor_res = hbond.acc_res()
        donor_atom = hbond.don_hatm()
        acceptor_atom = hbond.acc_atm()
        hbonds.append(hbond)
    return hbonds

def classify_hbond(hbond, pose):
    donor_res_num = hbond.don_res()
    acceptor_res_num = hbond.acc_res()
    donor_residue = pose.residue(donor_res_num)
    acceptor_residue = pose.residue(acceptor_res_num)
    
    donor_name = donor_residue.name3()
    acceptor_name = acceptor_residue.name3()
    
    prot_h2o_hbond = (donor_name != 'HOH' and acceptor_name == 'HOH') or (donor_name == 'HOH' and acceptor_name != 'HOH')
    h2o_h2o_hbond = donor_name == 'HOH' and acceptor_name == 'HOH'
    lig_h2o_hbond = (donor_residue.is_ligand() and acceptor_name == 'HOH') or (acceptor_residue.is_ligand() and donor_name == 'HOH')
    
    return prot_h2o_hbond, h2o_h2o_hbond, lig_h2o_hbond

def main(pdb_path, output_csv_path):
    # Load the pose
    pose = pyrosetta.pose_from_file(pdb_path)

    # Get all hydrogen bonds
    hbonds = get_hydrogen_bonds(pose)
    
    # Create empty lists to store the data
    donor_residues = []
    acceptor_residues = []
    donor_atoms = []
    acceptor_atoms = []
    hbond_energies = []
    donor_3letter = []
    acceptor_3letter = []
    prot_h2o_hbonds = []
    h2o_h2o_hbonds = []
    lig_h2o_hbonds = []
    
    for hbond in hbonds:
        donor_res_num = hbond.don_res()
        acceptor_res_num = hbond.acc_res()

        # Append residue numbers for donors and acceptors
        donor_residues.append(donor_res_num)
        acceptor_residues.append(acceptor_res_num)

        # Retrieve the 3-letter code for the donor and acceptor residues and append to the lists
        donor_3letter.append(pose.residue(donor_res_num).name3())
        acceptor_3letter.append(pose.residue(acceptor_res_num).name3())

        # Retrieve and append atom names involved in the hydrogen bonds
        donor_atoms.append(pose.residue(donor_res_num).atom_name(hbond.don_hatm()))
        acceptor_atoms.append(pose.residue(acceptor_res_num).atom_name(hbond.acc_atm()))

        # Append hydrogen bond energies
        hbond_energies.append(hbond.energy())
        
        # Classify hydrogen bonds
        prot_h2o, h2o_h2o, lig_h2o = classify_hbond(hbond, pose)
        prot_h2o_hbonds.append(prot_h2o)
        h2o_h2o_hbonds.append(h2o_h2o)
        lig_h2o_hbonds.append(lig_h2o)

    # Create the pandas DataFrame from the lists
    hbond_df = pd.DataFrame({
        'donor_residue': donor_residues,
        'donor_3letter': donor_3letter,
        'acceptor_residue': acceptor_residues,
        'acceptor_3letter': acceptor_3letter,
        'donor_atom': donor_atoms,
        'acceptor_atom': acceptor_atoms,
        'energy': hbond_energies,
        'prot_h2o_hbond': prot_h2o_hbonds,
        'h2o_h2o_hbond': h2o_h2o_hbonds,
        'lig_h2o_hbond': lig_h2o_hbonds
    })
    
    # Save to CSV
    hbond_df.to_csv(output_csv_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract interactions from a protein structure.')
    parser.add_argument('--pdb_path', type=str, required=True, help='Path to the input PDB file.')
    parser.add_argument('--output_csv_path', type=str, required=True, help='Path to the output CSV file.')
    
    args = parser.parse_args()
    main(args.pdb_path, args.output_csv_path)
