"""
Authors: Donghyo Kim & Seth Woodbury

Description:
-------------
This script processes protein structure predictions by aligning them to a reference structure 
and computing various quality metrics. It performs the following steps:
    
1. **Reference Structure Loading and Catalytic Residue Identification**:
   - Loads a reference PDB file.
   - Extracts catalytic residues from the file header (via "REMARK 666" annotations).
   - When the optional N‑ and/or C‑terminal tag lengths are specified, the residue numbering 
     is adjusted so that the first N residues are ignored (i.e. the numbering is shifted by N) 
     and the last C residues are completely ignored (for protein residues only).
     
2. **Chai-1 Prediction Processing**:
   - Iterates through all predicted PDB files in a specified directory.
   - For each prediction:
     - Loads the structure and reads its associated JSON score file (containing metrics like iptm, ptm, 
       aggregate score, and per-chain pLDDT).
     - Aligns the prediction to the reference using the Kabsch algorithm based on Cα atoms extracted 
       from protein residues only (applying the ignore parameters), while ligand (HETATM) records 
       are never ignored.
     - Computes the RMSD for:
       - All Cα atoms (global structural alignment),
       - Catalytic residues (comparing non‐hydrogen atoms; using the shifted residue numbering),
       - User‐specified ligand atoms (as defined in a JSON mapping). 
         For ligand atoms, no terminal ignoring is applied.
     - Extracts average pLDDT scores for both the catalytic residues and the defined ligand atoms 
       (ignoring pLDDT from protein residues in the terminal tag regions but never ignoring ligand scores).

3. **Output Generation**:
   - All computed metrics (RMSD values, pLDDT scores, and other prediction metrics) are aggregated 
     into a single score dictionary.
   - The results are saved into a CSV file and printed to the console.

Inputs:
---------
--ref_pdb
    Path to the reference PDB file.

--chai_dir
    Path to the directory containing the Chai-1 predicted PDB files.

--outscr
    Output file path for the CSV file where the computed scores will be saved.

--ligand_groups_json
    A JSON string that defines ligand groups. Each ligand group should include:
      - "label": A label for the ligand group.
      - "name3": A list containing the residue names 
                 (one for the *predicted* structure and one for the reference structure).
      - "atoms": A list of two lists specifying the atom names to be used 
                 for the RMSD and pLDDT calculations for the query and reference structures, respectively.

--N_terminus_tag_length_to_ignore
    (Optional) An integer number of protein residues at the N-terminus to ignore.
    Ligand residues (HETATM) are never ignored.

--C_terminus_tag_length_to_ignore
    (Optional) An integer number of protein residues at the C-terminus to ignore.
    Ligand residues (HETATM) are never ignored.

Example Command:
-------------------
python calculate_alignment_and_rmsd.py \
    --ref_pdb /path/to/reference.pdb \
    --chai_dir /path/to/chai_predictions/ \
    --outscr /path/to/output_scores.csv \
    --ligand_groups_json '[{"label": "Ligand1", "name3": ["LIG", "LIG"], "atoms": [["C1", "N1"], ["C1", "N1"]]}]' \
    --N_terminus_tag_length_to_ignore 3 \
    --C_terminus_tag_length_to_ignore 2

Usage Notes:
-------------
- The reference PDB must include catalytic residue annotations in its header using "REMARK 666".
- The Chai-1 predictions directory should contain both the predicted PDB files and the corresponding JSON score files.
- The terminal tag lengths apply only to protein (ATOM) residues and never to ligand (HETATM) records.
- Ensure the ligand groups JSON string is properly formatted to define the atom mappings for ligand RMSD and pLDDT calculations.
- For ligand pLDDT, make sure the **first** name in `"name3": [...]` matches the actual residue name in the Chai (predicted) PDB.

Overall, this script automates the alignment of predicted protein structures to a reference model, 
calculates key RMSD and pLDDT quality metrics (ignoring terminal tag regions for protein residues only), 
and compiles the results for further analysis.
"""

"""
Script: calculate_alignment_and_rmsd.py

This script processes protein structure predictions by aligning them to a reference structure
and computing various quality metrics, including RMSD for Cα atoms, catalytic residues, and user‐defined ligands, 
plus pLDDT extraction for both the protein and ligands, with optional ignoring of N/C‐terminal tags for protein only.
"""

from Bio import PDB
import os, sys, glob
import json
import argparse
import numpy as np
import pandas as pd

########################################
### HELPER FUNCTIONS FOR PDB HANDLING ###
########################################

def load_pdb(file_path):
    """Load a PDB file using Biopython."""
    parser = PDB.PDBParser(QUIET=True)
    return parser.get_structure(os.path.basename(file_path), file_path)

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

##################################
### ALIGNMENT & RMSD FUNCTIONS ###
##################################

def kabsch_alignment(coords1, coords2):
    """Align two sets of coordinates using the Kabsch algorithm."""
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)
    coords1_centered = coords1 - centroid1
    coords2_centered = coords2 - centroid2
    
    H = np.dot(coords1_centered.T, coords2_centered)
    V, S, W = np.linalg.svd(H)
    d = np.sign(np.linalg.det(V @ W))
    V[:, -1] *= d
    rotation_matrix = V @ W
    return rotation_matrix, centroid1, centroid2

def apply_rotation(coords, rotation_matrix, centroid1, centroid2):
    """Apply the rotation matrix to the coords and move the structure."""
    coords_centered = coords - centroid1
    coords_rotated = np.dot(coords_centered, rotation_matrix) + centroid2
    return coords_rotated

def calculate_rmsd(coords1, coords2):
    """Calculate RMSD between two sets of coordinates."""
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

##############################
### RESIDUE EXTRACTION ETC ###
##############################

def extract_coords_atom(structure, residue_name, atom_name):
    """
    Extract coordinates of a specific atom in a residue (usually for ligands).
    We do NOT apply ignoring here (ligands should never be ignored).
    """
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == residue_name:
                    if atom_name in residue:
                        coords.append(residue[atom_name].get_coord())
    return np.array(coords)

def extract_coords_res(structure, residue_num, n_ignore=0, c_ignore=0):
    """
    Extract coordinates (and atom names) from a protein residue whose 'effective' index == residue_num.
    We skip the first n_ignore and the last c_ignore from the protein chain to match reference numbering.
    """
    coords, names = [], []
    for model in structure:
        for chain in model:
            protein_residues = [r for r in chain.get_residues() if r.get_id()[0] == ' ']
            effective_residues = protein_residues[n_ignore: len(protein_residues)-c_ignore] if c_ignore > 0 else protein_residues[n_ignore:]
            for i, residue in enumerate(effective_residues, start=1):
                if i == residue_num:
                    for atom in residue:
                        if atom.element == "H":
                            continue
                        coords.append(atom.get_coord())
                        names.append(atom.get_name())
    return np.array(coords), names

def extract_ca_coords(structure, n_ignore=0, c_ignore=0):
    """
    Extract Cα coordinates from protein residues, skipping the first n_ignore
    and the last c_ignore. Used for alignment & RMSD of the protein only.
    """
    ca_coords = []
    for model in structure:
        for chain in model:
            protein_residues = [r for r in chain.get_residues() if r.get_id()[0] == ' ']
            effective_residues = protein_residues[n_ignore: len(protein_residues)-c_ignore] if c_ignore > 0 else protein_residues[n_ignore:]
            for residue in effective_residues:
                if 'CA' in residue:
                    ca_coords.append(residue['CA'].get_coord())
    return np.array(ca_coords)

def save_pdb(structure, output_file):
    """Save the structure to a PDB file."""
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_file)

###########################################
### CALCULATE RMSD AFTER ALIGNMENT ETC ###
###########################################

def calculate_rmsd_after_alignment(structure1, structure2, catres_list, ligand_atom_mapping_list, n_ignore=0, c_ignore=0):
    """
    Align the Chai prediction (structure1) to the reference (structure2) using only protein residues.
    For structure1, apply ignoring (n_ignore, c_ignore). For structure2, use the full structure.
    Then compute:
      - CA RMSD
      - Catalytic residues RMSD
      - Ligand RMSD
    """
    # 1) Align
    ca_coords1 = extract_ca_coords(structure1, n_ignore, c_ignore)
    ca_coords2 = extract_ca_coords(structure2, 0, 0)
    rotation_matrix, centroid1, centroid2 = kabsch_alignment(ca_coords1, ca_coords2)
    
    # 2) Apply rotation
    for model in structure1:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    old_coord = atom.get_coord()
                    atom.set_coord(apply_rotation(np.array([old_coord]), rotation_matrix, centroid1, centroid2)[0])
    
    # 3) Re-extract CA coords and compute RMSD
    ca_coords1_aligned = extract_ca_coords(structure1, n_ignore, c_ignore)
    ca_rmsd_value = calculate_rmsd(ca_coords1_aligned, ca_coords2)

    # 4) Catalytic residue RMSD
    catres_rmsds = []
    for catres in catres_list:
        catres_coords1, catres_names1 = extract_coords_res(structure1, catres["res_num"], n_ignore, c_ignore)
        catres_coords2, catres_names2 = extract_coords_res(structure2, catres["res_num"], 0, 0)
        if catres_names1 != catres_names2:
            raise ValueError(f"Catalytic residue atom names differ: Chai-1 {catres_names1}, ref {catres_names2}")
        catres_rmsds.append(calculate_rmsd(catres_coords1, catres_coords2))

    # 5) Ligand RMSD (no ignoring for ligands)
    ligand_rmsd_values = []
    for ligand_atom_mapping in ligand_atom_mapping_list:
        label, name3, atoms = ligand_atom_mapping["label"], ligand_atom_mapping["name3"], ligand_atom_mapping["atoms"]
        lig_coords1, lig_coords2 = [], []
        for atom1, atom2 in zip(atoms[0], atoms[1]):
            lig_coords1.append(extract_coords_atom(structure1, name3[0], atom1))
            lig_coords2.append(extract_coords_atom(structure2, name3[1], atom2))
        lig_coords1 = np.concatenate(lig_coords1)
        lig_coords2 = np.concatenate(lig_coords2)
        ligand_rmsd_values.append(calculate_rmsd(lig_coords1, lig_coords2))

    return ca_rmsd_value, catres_rmsds, ligand_rmsd_values

##########################################
### pLDDT DICTIONARY WITH RES. IGNORING ###
##########################################

def get_plddt_per_residue_dic(chai_pdb, n_ignore=0, c_ignore=0):
    """
    Parse the Chai PDB and build a dictionary that maps (chain, effective_res_index) -> average pLDDT
    for protein lines, and (chain, raw_res_seq) for 'ligand' lines. We define 'ligand' lines here as:
      - Lines that start with 'HETATM', OR
      - Lines that start with 'ATOM' but whose resname is *not* a standard amino acid 
        (e.g. is 'LIG', 'SZA', 'ZN1', etc.)
    """
    # If you have multiple possible predicted ligand names (like "LIG", "SZA", "ZN1", etc.), 
    # you could define them in a set. For demonstration we just parse the lines and see if they're 
    # in the standard 20 amino acids or not.
    standard_aa = {
        "ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY","HIS","ILE",
        "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"
    }

    lines = open(chai_pdb, 'r').read().splitlines()
    chain_protein_resids = {}
    
    # Identify protein lines (true protein) by either 'ATOM' + standard residue name
    for line in lines:
        if len(line) < 80:
            continue
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        name3 = line[17:20].strip()
        if line.startswith("ATOM") and name3 in standard_aa:
            ch_id = line[21:23].strip()
            res_seq = int(line[23:27].strip())
            chain_protein_resids.setdefault(ch_id, []).append(res_seq)
    
    # Convert each chain's protein list to sorted unique
    for ch_id in chain_protein_resids:
        chain_protein_resids[ch_id] = sorted(set(chain_protein_resids[ch_id]))
    
    # Build an "effective" map for protein residues only
    eff_num_map = {}
    for ch_id, res_list in chain_protein_resids.items():
        real_prot_subset = res_list[n_ignore: len(res_list)-c_ignore] if c_ignore>0 else res_list[n_ignore:]
        for i, raw_res_seq in enumerate(real_prot_subset, start=1):
            eff_num_map[(ch_id, raw_res_seq)] = i

    # Now parse lines again to get B-factor => pLDDT
    plddt_dict = {}
    for line in lines:
        if len(line) < 80:
            continue
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        name3 = line[17:20].strip()
        ch_id = line[21:23].strip()
        raw_res_seq = int(line[23:27].strip())
        
        # parse B-factor
        try:
            plddt_val = float(line[61:67])
        except ValueError:
            continue

        # Distinguish if it's 'protein' or 'ligand'
        # If line.startswith("ATOM") and name3 in standard_aa => protein
        # else it's "ligand" or "non-standard"
        if line.startswith("ATOM") and name3 in standard_aa:
            # We only keep it if it's in eff_num_map
            if (ch_id, raw_res_seq) not in eff_num_map:
                continue
            eff_num = eff_num_map[(ch_id, raw_res_seq)]
            key = (ch_id, eff_num)  # effective numbering
        else:
            # Treat as ligand or non-standard
            # key is (chain, raw_res_seq)
            key = (ch_id, raw_res_seq)

        plddt_dict.setdefault(key, []).append(plddt_val)
    
    # average them
    for k in plddt_dict:
        plddt_dict[k] = float(np.mean(plddt_dict[k]))
    
    return plddt_dict

def get_avr_plddt(chai_pdb, ligand_groups_json, n_ignore=0, c_ignore=0):
    """
    Similar approach for ligand pLDDT. We skip protein lines if they're outside the ignoring region,
    but we do *not* skip lines if the residue name is 'LIG' or something not in the standard AAs set.
    """
    standard_aa = {
        "ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY","HIS","ILE",
        "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"
    }

    lines = open(chai_pdb, 'r').read().splitlines()
    # Identify *protein* lines for ignoring
    chain_prot_map = {}
    for line in lines:
        if len(line) < 80:
            continue
        if line.startswith("ATOM"):
            name3 = line[17:20].strip()
            if name3 in standard_aa:  # i.e. it is a standard AA
                ch_id = line[21:23].strip()
                res_seq = int(line[23:27].strip())
                chain_prot_map.setdefault(ch_id, []).append(res_seq)

    # Convert to sorted unique
    for ch_id in chain_prot_map:
        chain_prot_map[ch_id] = sorted(set(chain_prot_map[ch_id]))

    eff_map = {}
    for ch_id, raw_list in chain_prot_map.items():
        real_subset = raw_list[n_ignore: len(raw_list)-c_ignore] if c_ignore>0 else raw_list[n_ignore:]
        for i, raw_res_seq in enumerate(real_subset, start=1):
            eff_map[(ch_id, raw_res_seq)] = i

    # For each ligand group, find lines that match. We'll skip protein lines outside eff_map
    group_results = []
    for group in ligand_groups_json:
        pred_name = group["name3"][0]
        pred_atoms = group["atoms"][0]
        plddt_vals = []

        for line in lines:
            if len(line) < 80:
                continue
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            ch_id = line[21:23].strip()
            res_seq = int(line[23:27].strip())
            name3 = line[17:20].strip()
            atom_name = line[12:17].strip()

            # parse pLDDT
            try:
                val = float(line[61:67])
            except ValueError:
                continue

            # if it's an ATOM line with a standard_aa => check ignoring
            if line.startswith("ATOM") and name3 in standard_aa:
                if (ch_id, res_seq) not in eff_map:
                    continue
            
            # If the line matches the predicted ligand residue name and atom
            if name3 == pred_name and atom_name in pred_atoms:
                plddt_vals.append(val)
        print (chai_pdb)
        print (pred_name, pred_atoms, plddt_vals, np.mean(plddt_vals))

        if plddt_vals:
            group_results.append(float(np.mean(plddt_vals)))
        else:
            group_results.append(np.nan)

    return group_results

#######################
### MAIN SCRIPT FLOW ###
#######################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_pdb", help="Path of reference PDB.")
    parser.add_argument("--chai_dir", help="Path of the directory with Chai-1 PDB predictions.")
    parser.add_argument("--outscr", type=str, help="Output score file path.")
    parser.add_argument("--ligand_groups_json", type=str, help="JSON string defining atom groups and labels.")
    parser.add_argument("--N_terminus_tag_length_to_ignore", type=int, default=0,
                        help="Number of protein residues at the N-terminus to ignore for Chai PDBs.")
    parser.add_argument("--C_terminus_tag_length_to_ignore", type=int, default=0,
                        help="Number of protein residues at the C-terminus to ignore for Chai PDBs.")
    args = parser.parse_args()

    n_ignore = args.N_terminus_tag_length_to_ignore
    c_ignore = args.C_terminus_tag_length_to_ignore
    args.ligand_groups_json = json.loads(args.ligand_groups_json)

    updated_score_dic = {"description": args.chai_dir}

    # 1) Reference
    ref_structure = load_pdb(args.ref_pdb)
    catres_list = get_catalytic_residues(args.ref_pdb)

    # 2) Loop over Chai-1 PDBs
    chai_pdbs = sorted(glob.glob(os.path.join(args.chai_dir, "*.pdb")))
    for chai_pdb_num, chai_pdb in enumerate(chai_pdbs):
        chai_structure = load_pdb(chai_pdb)
        chai_ch_ids = [chain.id for model in chai_structure for chain in model]
        
        # Read JSON
        scores_file = chai_pdb.replace("pred.", "scores.").replace(".pdb", ".json")
        with open(scores_file, "r") as file:
            scores = json.load(file)
        updated_score_dic[f"iptm_idx_{chai_pdb_num}"] = scores["iptm"][0]
        updated_score_dic[f"ptm_idx_{chai_pdb_num}"] = scores["ptm"][0]
        updated_score_dic[f"aggregate_score_idx_{chai_pdb_num}"] = scores["aggregate_score"][0]

        # chain-level
        for i, ch_id in enumerate(chai_ch_ids):
            updated_score_dic[f"chain{ch_id}_plddt_idx_{chai_pdb_num}"] = scores["per_chain_plddt"][0][i]
        
        # Align & RMSD
        ca_rmsd, catres_rmsds, ligand_rmsds = calculate_rmsd_after_alignment(
            chai_structure, ref_structure, catres_list, args.ligand_groups_json,
            n_ignore=n_ignore, c_ignore=c_ignore)
        updated_score_dic[f"ca_rmsd_idx_{chai_pdb_num}"] = ca_rmsd

        # ligand pLDDT
        ligand_plddts = get_avr_plddt(chai_pdb, args.ligand_groups_json, n_ignore, c_ignore)
        
        # Save ligand RMSD & pLDDT
        for i, (ligand_rmsd, ligand_plddt) in enumerate(zip(ligand_rmsds, ligand_plddts)):
            label = args.ligand_groups_json[i]['label']
            updated_score_dic[f"{label}_rmsd_idx_{chai_pdb_num}"] = ligand_rmsd
            updated_score_dic[f"{label}_plddt_idx_{chai_pdb_num}"] = ligand_plddt
        
        # catalytic RMSD
        for i, catres in enumerate(catres_list):
            updated_score_dic[f"{catres['name3']}{i+1}_rmsd_idx_{chai_pdb_num}"] = catres_rmsds[i]
        updated_score_dic[f"catres_rmsd_idx_{chai_pdb_num}"] = np.mean(catres_rmsds)

        # catalytic pLDDT with ignoring
        plddt_dic = get_plddt_per_residue_dic(chai_pdb, n_ignore, c_ignore)
        cat_res_plddts = []
        for i, catres in enumerate(catres_list):
            key = (catres['chain'], catres['res_num'])
            cat_res_plddt = plddt_dic[key] if key in plddt_dic else np.nan
            updated_score_dic[f"{catres['name3']}{i+1}_plddt_idx_{chai_pdb_num}"] = cat_res_plddt
            cat_res_plddts.append(cat_res_plddt)
        updated_score_dic[f"catres_plddt_idx_{chai_pdb_num}"] = np.mean(cat_res_plddts)
    
    # 3) Output
    score_df = pd.DataFrame([updated_score_dic])
    score_df.to_csv(args.outscr, index=False)
    print(score_df)
