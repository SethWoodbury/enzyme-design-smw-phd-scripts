"""
Created 2024-02-07 by Seth Woodbury (woodbuse@uw.edu)
This script should do a pyfastrelax & output a new pdb with 
hydrogen bond information about the specified atoms, including # of H-bonds and the residue doing them
"""
from optparse import OptionParser
import os
import pyrosetta
import pyrosetta.distributed.tasks.rosetta_scripts as rosetta_scripts
from pyrosetta import *
from pyrosetta.rosetta.core.pose import num_chi_angles
import subprocess
import textwrap
import pandas as pd
import glob
import sys
from SimplePdbLib import *
from concurrent.futures import ThreadPoolExecutor
import re

# Parse command-line options
parser = OptionParser(usage="usage: %prog [options] FILE", version="0.1")
parser.add_option("--pdb", type="string", dest="pdb", help="Path to pdb you want to filter")
parser.add_option("--key_atoms", type="string", dest="key_atoms", help="check these atoms for hydrogen bonds, SHOULD BE A LIST (eg 'S1,O5,O6,N2' exactly in that format)")
parser.add_option("--params_dir", type="string", dest="params_dir", help="Directory containing params files")
parser.add_option("--out_dir", type="string", dest="out_dir", help="Directory to dump the relaxed PDBs")
parser.add_option("--ligand_exposed_atoms", type="string", dest="ligand_exposed_atoms", help="Ligand atoms to check for exposure. SHOULD BE A LIST (eg 'S1,O5,O6,N2' exactly in that format)")

(opts, args) = parser.parse_args()
parser.set_defaults()
print("Using the following arguments:")
print(opts)

# Correct way to form path and wildcard search
params_files = glob.glob(os.path.join(opts.params_dir, '*.params'))  # Note the comma between dir and pattern

# Initialize PyRosetta
pyrosetta.init(f"-mute all -beta -in:file:extra_res_fa {' '.join(params_files)} -dalphaball /net/software/lab/scripts/enzyme_design/DAlphaBall.gcc -mute all")

bn = os.path.basename(opts.pdb)
pose = pyrosetta.pose_from_file(opts.pdb)

def find_ligand_seqpos(pose):
    ligand_seqpos = None
    for res in pose.residues:
        if res.is_ligand() and not res.is_virtual_residue():
            ligand_seqpos = res.seqpos()
    return ligand_seqpos

def hbond_filter(bn, pose, key_atoms_list, out_dir, lig_res_num=''):
    '''
    Counts hydrogen bond between a design and its ligand, then does constrained fast relax
    where the key catalytic residues are constrained...hard-coded for glycosidases
    '''
    key_atoms_list = key_atoms_list.split(',') 
    filters_txt = ''
    hbond_filters = []  # List to store filter names and their corresponding target residues

    for atom in key_atoms_list:
        filter_name = f'{atom}_hbond'
        filters_txt += f'<SimpleHbondsToAtomFilter name="{filter_name}" n_partners="1" hb_e_cutoff="-0.1" target_atom_name="{atom}" confidence="0" res_num="{lig_res_num}" scorefxn="sfxn_design"/> \n          '
        hbond_filters.append((filter_name, []))

    protocols_txt = ''
    for atom in key_atoms_list:
        protocols_txt += f'<Add filter_name="{atom}_hbond" /> \n        '
   
    xml_script = f"""
    <ROSETTASCRIPTS>

      <SCOREFXNS>
          
          <ScoreFunction name="sfxn_design" weights="beta">
              <Reweight scoretype="arg_cation_pi" weight="3"/>
              <Reweight scoretype="angle_constraint" weight="1.0"/>
              <Reweight scoretype="angle_constraint" weight="1.0"/>
              <Reweight scoretype="coordinate_constraint" weight="1.0"/>
              <Reweight scoretype="dihedral_constraint" weight="1.0"/>
          </ScoreFunction>
          
          <ScoreFunction name="fa_csts" weights="beta">
              <Reweight scoretype="arg_cation_pi" weight="3"/>
              <Reweight scoretype="angle_constraint" weight="1.0"/>
              <Reweight scoretype="coordinate_constraint" weight="1.0"/>
              <Reweight scoretype="angle_constraint" weight="1.0"/>
              <Reweight scoretype="dihedral_constraint" weight="1.0"/>
          </ScoreFunction>
          
          <ScoreFunction name="sfxn" weights="beta" />
      </SCOREFXNS>
      
      <RESIDUE_SELECTORS>
          <Chain name="chainA" chains="A"/>
          <Chain name="chainB" chains="B"/>
      </RESIDUE_SELECTORS>
      
      <SIMPLE_METRICS>
          <TotalEnergyMetric name="total_energy" scorefxn="sfxn_design" />
          <SecondaryStructureMetric name="secondary_structure" dssp_reduced="false"/>
          <SecondaryStructureMetric name="secondary_structure_reduced" dssp_reduced="true"/>
          <SapScoreMetric name="spatial_aggregation_propensity_score"/>
          <ElectrostaticComplementarityMetric name="electrostatic_complementarity" ignore_radius="-1" interface_trim_radius="0" partially_solvated="1" jump="1" report_all_ec="0" />
      </SIMPLE_METRICS>
    
      <MOVERS>
            
      </MOVERS>
    
      <FILTERS>
          {filters_txt}
          <ContactMolecularSurface name="contact_molecular_surface" use_rosetta_radii="true" distance_weight="0.5" target_selector="chainB" binder_selector="chainA" confidence="0"/>
          <Ddg name="ddg_norepack"  threshold="0" jump="1" repeats="1" repack="0" confidence="0" scorefxn="sfxn_design"/>
          <Report name="ddg" filter="ddg_norepack"/>
          <LigInterfaceEnergy name="ligand_interface_energy"  scorefxn="sfxn_design" include_cstE="1" confidence="0"/>
          <ResidueCount name="total_residues_in_design_plus_ligand" max_residue_count="99999" min_residue_count="0" count_as_percentage="0" confidence="0"/>
          <ResidueCount name="hydrophobic_residues_in_design" include_property="HYDROPHOBIC" max_residue_count="99999" min_residue_count="0" count_as_percentage="0" confidence="0"/>
          <ResidueCount name="aliphatic_residues_in_design" include_property="ALIPHATIC" max_residue_count="99999" min_residue_count="0" count_as_percentage="0" confidence="0"/>
          <NetCharge name="net_charge_in_design_NOT_w_HIS" chain="1" confidence="0"/>
          <DSasa name="dSasa_fraction" lower_threshold="0.0" upper_threshold="1.0" confidence="0"/>
          <SecondaryStructureCount name="number_DSSP_helices_in_design" num_helix_sheet="0" num_helix="1" num_sheet="0" num_loop="0" filter_helix_sheet="0" filter_helix="1" filter_sheet="0" filter_loop="0" min_helix_length="3" max_helix_length="9999" min_sheet_length="3" max_sheet_length="9999" min_loop_length="1" max_loop_length="9999" return_total="true" confidence="0"/>
          <SecondaryStructureCount name="number_DSSP_sheets_in_design" num_helix_sheet="0" num_helix="0" num_sheet="1" num_loop="0" filter_helix_sheet="0" filter_helix="0" filter_sheet="1" filter_loop="0" min_helix_length="3" max_helix_length="9999" min_sheet_length="3" max_sheet_length="9999" min_loop_length="1" max_loop_length="9999" return_total="true" confidence="0"/>
          <SecondaryStructureCount name="number_DSSP_loops_in_design" num_helix_sheet="0" num_helix="0" num_sheet="0" num_loop="1" filter_helix_sheet="0" filter_helix="0" filter_sheet="0" filter_loop="1" min_helix_length="3" max_helix_length="9999" min_sheet_length="3" max_sheet_length="9999" min_loop_length="1" max_loop_length="9999" return_total="true" confidence="0"/>
          <Holes name="holes_in_design_lower_is_better" threshold="2" normalize_per_residue="false" exclude_bb_atoms="false" confidence="0"/>
          <InterfaceHoles name="interface_holes_at_ligand" jump="1" threshold="200" confidence="0"/>
          <ResInInterface name="num_residues_at_ligand_interface" residues="20" jump_number="1" confidence="0"/>
          <ShapeComplementarity name="shape_complementarity_interface_area" min_sc="0.5" min_interface="1" verbose="0" quick="0" jump="1" write_int_area="1" write_median_dist="0" max_median_dist="1000" residue_selector1="chainA" residue_selector2="chainB" confidence="0"/>
          <ShapeComplementarity name="shape_complementarity_median_distance_at_interface" min_sc="0.5" min_interface="1" verbose="0" quick="0" jump="1" write_int_area="0" write_median_dist="1" max_median_dist="1000" residue_selector1="chainA" residue_selector2="chainB" confidence="0"/>
          <ExposedHydrophobics name="hydrophobic_exposure_sasa_in_design" sasa_cutoff="20" threshold="-1" confidence="0"/>
          <Sasa name="sasa_ligand_interface" threshold="800" upper_threshold="1000000000000000" hydrophobic="0" polar="0" jump="1" confidence="0"/>
          <TotalSasa name="total_pose_sasa" threshold="800" upper_threshold="1000000000000000" hydrophobic="0" polar="0" confidence="0"/>
          <PreProline name="bad_torsion_preproline" use_statistical_potential="0" confidence="0"/>
          <LongestContinuousPolarSegment name="longest_cont_polar_seg" exclude_chain_termini="false" count_gly_as_polar="false" filter_out_high="false" cutoff="5" confidence="0"/>
          <LongestContinuousApolarSegment name="longest_cont_apolar_seg" exclude_chain_termini="false" filter_out_high="false" cutoff="5" confidence="0"/>

      </FILTERS>

      <PROTOCOLS>
          {protocols_txt}
         <Add filter="contact_molecular_surface"/>
         <Add filter="ddg"/>
         <Add filter="ligand_interface_energy"/>
         <Add filter="total_residues_in_design_plus_ligand"/>
         <Add filter="hydrophobic_residues_in_design"/>
         <Add filter="aliphatic_residues_in_design"/>
         <Add filter="net_charge_in_design_NOT_w_HIS"/>
         <Add filter="dSasa_fraction"/>
         <Add filter="number_DSSP_helices_in_design"/>
         <Add filter="number_DSSP_sheets_in_design"/>
         <Add filter="number_DSSP_loops_in_design"/>
         <Add filter="holes_in_design_lower_is_better"/>
         <Add filter="interface_holes_at_ligand"/>
         <Add filter="num_residues_at_ligand_interface"/>
         <Add filter="shape_complementarity_interface_area"/>
         <Add filter="shape_complementarity_median_distance_at_interface"/>
         <Add filter="hydrophobic_exposure_sasa_in_design"/>
         <Add filter="sasa_ligand_interface"/>
         <Add filter="total_pose_sasa"/>
         <Add filter="bad_torsion_preproline"/>
         <Add filter="longest_cont_polar_seg"/>
         <Add filter="longest_cont_apolar_seg"/>
        

         <Add metrics="total_energy,secondary_structure,secondary_structure_reduced,spatial_aggregation_propensity_score,electrostatic_complementarity" labels="total_rosetta_energy_metric,secondary_structure,secondary_structure_DSSP_reduced_alphabet,SAP_score,electrostatic_complementarity"/>

      </PROTOCOLS>
    
    </ROSETTASCRIPTS>
    """
    #         <InterfaceScoreCalculator name="interface_scores" chains="A,B" scorefxn="sfxn_design"/>
    #         <Add filter="interface_scores"/>
    #          <SpecificResiduesNearInterface name="residues_at_interface" task_operation="(&string)" confidence="0"/>
    #         <Add filter="residues_at_interface"/>

    task_relax = rosetta_scripts.SingleoutputRosettaScriptsTask(xml_script)
    task_relax.setup() # syntax check
    packed_pose = task_relax(pose)
    pose.dump_pdb(f'{out_dir}{bn}')

# RUN THE ABOVE FUNCTIONS
lig_res_num = find_ligand_seqpos(pose)
hbond_filter(bn, pose, opts.key_atoms, opts.out_dir, lig_res_num)

# Get all hydrogen bonds in the pose
hbonds = pose.get_hbonds()

# Create an empty DataFrame to store the hydrogen bond information
df = pd.DataFrame(columns=['donor_residue', 'donor_atom', 'acceptor_residue', 'acceptor_atom', 'distance'])

# Create empty lists to store the data
donor_residues = []
acceptor_residues = []
donor_atoms = []
acceptor_atoms = []
hbond_energies = []

# Loop over each HBond in the HBondSet and extract the information
for hbond in hbonds.hbonds():
    donor_residues.append(hbond.don_res())
    acceptor_residues.append(hbond.acc_res())
    donor_atoms.append(pose.residue(hbond.don_res()).atom_name(hbond.don_hatm()))
    acceptor_atoms.append(pose.residue(hbond.acc_res()).atom_name(hbond.acc_atm()))
    hbond_energies.append(hbond.energy())

# Create the pandas DataFrame from the lists
hbond_df = pd.DataFrame({
    'donor_residue': donor_residues,
    'acceptor_residue': acceptor_residues,
    'donor_atom': donor_atoms,
    'acceptor_atom': acceptor_atoms,
    'energy': hbond_energies
    })

# Filter the DataFrame to only include hydrogen bonds involving the ligand
mask = (hbond_df['donor_residue'] == lig_res_num) | (hbond_df['acceptor_residue'] == lig_res_num)
lig_hbond_df = hbond_df[mask]

# Function to Calculate SASA for Specified Ligand Atoms
def calculate_ligand_SASA(pose, lig_res_num, exposed_atoms):
    '''
    Calculates the SASA for specified atoms in the ligand
    '''
    lig_residue = pose.residue(lig_res_num)
    sasa_atoms = [lig_residue.atom_index(atom) for atom in exposed_atoms if lig_residue.has(atom)]
    exposed_sasa = getSASA(pose, resno=lig_res_num, SASA_atoms=sasa_atoms)
    return exposed_sasa

# Function to get SASA, copied from Indrek's script
def getSASA(pose, resno=None, SASA_atoms=None, ignore_sc=False):
    """
    Takes in a pose and calculates its SASA.
    Or calculates SASA of a given residue.
    Or calculates SASA of specified atoms in a given residue.

    Procedure by Brian Coventry
    """
    atoms = pyrosetta.rosetta.core.id.AtomID_Map_bool_t()
    atoms.resize(pose.size())

    n_ligands = 0
    for res in pose.residues:
        if res.is_ligand():
            n_ligands += 1

    for i, res in enumerate(pose.residues):
        if res.is_ligand():
            atoms.resize(i+1, res.natoms(), True)
        else:
            atoms.resize(i+1, res.natoms(), not(ignore_sc))
            if ignore_sc is True:
                for n in range(1, res.natoms()+1):
                    if res.atom_is_backbone(n) and not res.atom_is_hydrogen(n):
                        atoms[i+1][n] = True

    surf_vol = pyrosetta.rosetta.core.scoring.packing.get_surf_vol(pose, atoms, 1.4)

    if resno is not None:
        res_surf = 0.0
        for i in range(1, pose.residue(resno).natoms()+1):
            if SASA_atoms is not None and i not in SASA_atoms:
                continue
            res_surf += surf_vol.surf(resno, i)
        return res_surf
    else:
        return surf_vol

def copy_existing_remark666_lines(input_pdb, output_pdb):
    """
    Extracts lines starting with REMARK 666 from the input PDB file.
    Inserts these lines starting at the second line of the output PDB file.
    """
    with open(input_pdb, 'r') as pdb_file:
        remark666_lines = [line for line in pdb_file if line.startswith("REMARK 666")]

    with open(output_pdb, 'r') as pdb_file:
        pdb_lines = pdb_file.readlines()

    with open(output_pdb, 'w') as pdb_file:
        # Write the first line of the original PDB file
        pdb_file.write(pdb_lines[0])
        # Insert the REMARK 666 lines
        for line in remark666_lines:
            pdb_file.write(line)
        # Write the remaining lines of the original PDB file
        for line in pdb_lines[1:]:
            pdb_file.write(line)

# Run the new functionality if specified
if opts.ligand_exposed_atoms:
    exposed_atoms = opts.ligand_exposed_atoms.split(',')
    exposed_sasa = calculate_ligand_SASA(pose, lig_res_num, exposed_atoms)
    print(f"Exposed Ligand Atoms SASA: {exposed_sasa:.3f}")
    sasa_info = f"ligand_exposed_atoms_sasa {exposed_sasa:.3f}\n"

    # Append SASA information to the PDB file
    with open(f'{opts.out_dir}{bn}', 'a') as pdb_file:
        pdb_file.write(sasa_info)

# Append hydrogen bond information to the PDB file
with open(f'{opts.out_dir}{bn}', 'a') as pdb_file:
    pdb_file.write("# acceptor_atom, acceptor_residue, donor_atom, donor_residue, hbonding_energy \n")
    for index, row in lig_hbond_df.iterrows():
        pdb_file.write(f"{row['acceptor_atom']} {row['acceptor_residue']} {row['donor_atom']} {row['donor_residue']} {row['energy']}\n")

# Copy REMARK 666 lines from the input PDB file to the output PDB file
copy_existing_remark666_lines(opts.pdb, f'{opts.out_dir}{bn}')