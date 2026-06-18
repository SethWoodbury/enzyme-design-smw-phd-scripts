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

def permute_symmetric_atoms(
    coords1,
    coords2,
    atom_names1,
    atom_names2,
    symmetric_groups,
    *,
    auto_include_fixed_from_atoms=True,
    debug=False,
    label=None
):
    """
    Calculate min RMSD across explicitly defined symmetric subgroups,
    and (optionally) auto-include all other atoms from `atoms` as a single
    fixed subgroup using positional mapping.

    - coords1/atom_names1: predicted (Chai) side, same order as atoms[0]
    - coords2/atom_names2: reference side,  same order as atoms[1]
    - symmetric_groups: list of groups, each with:
        {
          "lig1": [ ["A1","A2"], ["B1","B2","B3"], ... ],   # subgroups (pred names)
          "lig2": [ [ ["X1","X2"], ["X2","X1"] ],          # allowed perms for ref names
                    [ ["Y1","Y2","Y3"], ["Y2","Y3","Y1"], ... ] ]
        }

    If auto_include_fixed_from_atoms=True:
      - Any atom in atom_names1 *not* mentioned in any lig1 subgroup is added
        as a final subgroup with a single identity permutation, using the
        positional mapping declared by (atom_names1[i] <-> atom_names2[i]).

    Returns the minimum RMSD over the cartesian product of one permutation per subgroup.
    """
    from itertools import product

    name_to_idx1 = {name: i for i, name in enumerate(atom_names1)}
    name_to_idx2 = {name: i for i, name in enumerate(atom_names2)}

    min_rmsd = float("inf")

    for group_idx, group in enumerate(symmetric_groups, start=1):
        atoms1_subgroups = [list(sub) for sub in group["lig1"]]
        atoms2_perms_per_subgroup = [list(perms) for perms in group["lig2"]]

        # --- AUTO-INCLUDE FIXED ATOMS (identity subgroup built from `atoms`) ---
        if auto_include_fixed_from_atoms:
            # Collect all predicted names already covered by explicit symmetric subgroups
            covered_pred = set(a for sub in atoms1_subgroups for a in sub)

            # Remaining names in predicted list (preserve order from atom_names1)
            remaining_pred = [a for a in atom_names1 if a not in covered_pred]

            if remaining_pred:
                # Identity mapping for the remaining atoms uses *positional* pairing
                # declared by the top-level `atoms` lists.
                # i.e., atom_names1[i] <-> atom_names2[i]
                remaining_ref = []
                for a in remaining_pred:
                    i = name_to_idx1[a]
                    remaining_ref.append(atom_names2[i])

                # Append one fixed subgroup
                atoms1_subgroups.append(remaining_pred)
                atoms2_perms_per_subgroup.append([remaining_ref])  # single (identity) option

                if debug:
                    print(f"[symm][{label or ''}] Group {group_idx}: auto-added fixed subgroup")
                    print(f"  lig1 (pred): {remaining_pred}")
                    print(f"  lig2 (ref ): {remaining_ref}")

        # Sanity check
        if len(atoms1_subgroups) != len(atoms2_perms_per_subgroup):
            raise ValueError("Mismatch between number of lig1 subgroups and lig2 permutation sets.")

        # --- Evaluate all combinations of one permutation per subgroup ---
        for combo_idx, perm_combo in enumerate(product(*atoms2_perms_per_subgroup), start=1):
            coords1_concat, coords2_concat = [], []

            for atoms1_subgroup, atoms2_perm in zip(atoms1_subgroups, perm_combo):
                idxs1 = [name_to_idx1[a] for a in atoms1_subgroup]
                idxs2 = [name_to_idx2[a] for a in atoms2_perm]
                coords1_concat.extend(coords1[idxs1])
                coords2_concat.extend(coords2[idxs2])

            c1 = np.array(coords1_concat)
            c2 = np.array(coords2_concat)
            if c1.shape != c2.shape:
                raise ValueError("Symmetric RMSD: concatenated shapes differ.")

            rmsd = calculate_rmsd(c1, c2)
            if debug:
                print(f"[symm][{label or ''}] Group {group_idx} combo {combo_idx}: RMSD={rmsd:.4f}")
            if rmsd < min_rmsd:
                min_rmsd = rmsd

    return min_rmsd


##############################
### RESIDUE EXTRACTION ETC ###
##############################

def extract_coords_atom(structure, residue_name, atom_name, chain_id=None):
    """
    Extract coordinates of a specific atom in a residue (usually for ligands).
    Optional chain filter: if chain_id is provided, restrict to that chain.
    We do NOT apply terminal ignoring to ligands.
    """
    coords = []
    for model in structure:
        for chain in model:
            if chain_id is not None and str(chain.id) != str(chain_id):
                continue
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
    #catres_rmsds = []
    #for catres in catres_list:
     #   catres_coords1, catres_names1 = extract_coords_res(structure1, catres["res_num"], n_ignore, c_ignore)
      #  catres_coords2, catres_names2 = extract_coords_res(structure2, catres["res_num"], 0, 0)

        # ─── sort the two name‐lists before checking ───
       # sorted_names1 = sorted(catres_names1)
       # sorted_names2 = sorted(catres_names2)
       # if sorted_names1 != sorted_names2:
        #    raise ValueError(
        #        f"Catalytic residue atom names differ (after sorting): "
        #        f"Chai-1 {sorted_names1}, ref {sorted_names2}"
        #        )
        #catres_rmsds.append(calculate_rmsd(catres_coords1, catres_coords2))

    # 4) Catalytic residue RMSD (ncAA-robust: compare common atoms only)
    catres_rmsds = []
    ncaa_meta = []  # collect AF3-like metadata for extra atoms & naming

    for catres in catres_list:
        eff_idx = catres["res_num"]          # effective index by spec
        ch_id   = catres["chain"]

        coords1, names1 = extract_coords_res(structure1, eff_idx, n_ignore, c_ignore)  # Chai (pred)
        coords2, names2 = extract_coords_res(structure2, eff_idx, 0, 0)                 # Reference

        rmsd_val, pred_only, ref_only, common_used = rmsd_on_common_atoms(coords1, names1, coords2, names2)
        if not common_used:
            rmsd_val = float("nan")

        catres_rmsds.append(rmsd_val)

        # annotate residue names (pred may be ncAA)
        pred_resname = get_resname_by_chain_effindex(structure1, ch_id, eff_idx, n_ignore, c_ignore)
        ref_resname  = get_resname_by_chain_effindex(structure2, ch_id, eff_idx, 0, 0)

        # map to raw resseq in predicted PDB so we can fetch atom-level pLDDT later
        pred_raw_resseq = get_raw_resseq_for_effective_index(structure1, ch_id, eff_idx, n_ignore, c_ignore)

        ncaa_meta.append({
            "chain": ch_id,
            "eff_index": eff_idx,
            "pred_resname": pred_resname,
            "ref_resname": ref_resname,
            "pred_only_atoms": pred_only,          # atoms only in predicted (ncAA expansion)
            "ref_only_atoms": ref_only,
            "common_atoms_used": common_used,
            "pred_raw_resseq": pred_raw_resseq
        })

    # 5) Ligand RMSD (no ignoring for ligands)
#    ligand_rmsd_values = []
 #   for ligand_atom_mapping in ligand_atom_mapping_list:
 #       label, name3, atoms = ligand_atom_mapping["label"], ligand_atom_mapping["name3"], ligand_atom_mapping["atoms"]
 #       lig_coords1, lig_coords2 = [], []
 #       for atom1, atom2 in zip(atoms[0], atoms[1]):
 #           lig_coords1.append(extract_coords_atom(structure1, name3[0], atom1))
 #           lig_coords2.append(extract_coords_atom(structure2, name3[1], atom2))
 #       lig_coords1 = np.concatenate(lig_coords1)
 #       lig_coords2 = np.concatenate(lig_coords2)
 #       ligand_rmsd_values.append(calculate_rmsd(lig_coords1, lig_coords2))

    # 5) Ligand RMSD (no ignoring for ligands); optional symmetric handling
    ligand_rmsd_values = []
    for ligand_atom_mapping in ligand_atom_mapping_list:
        label = ligand_atom_mapping["label"]
        name3 = ligand_atom_mapping["name3"]          # [pred_resname, ref_resname]
        atoms = ligand_atom_mapping["atoms"]          # [pred_atom_names, ref_atom_names]

        # NEW: optional chain spec: ["pred_chain", "ref_chain"]
        chains = ligand_atom_mapping.get("chain", [None, None])
        pred_chain = chains[0] if chains and len(chains) > 0 else None
        ref_chain  = chains[1] if chains and len(chains) > 1 else None

        lig_coords1, lig_coords2 = [], []
        missing_pred, missing_ref = [], []

        for a1, a2 in zip(atoms[0], atoms[1]):
            # pass chain filters here
            c1 = extract_coords_atom(structure1, name3[0], a1, chain_id=pred_chain)
            c2 = extract_coords_atom(structure2, name3[1], a2, chain_id=ref_chain)
            if c1.size == 0:
                missing_pred.append(a1)
            if c2.size == 0:
                missing_ref.append(a2)
            lig_coords1.append(c1)
            lig_coords2.append(c2)

        if missing_pred:
            raise KeyError(f"[{label}] Missing atoms in predicted ({name3[0]} chain={pred_chain}): {', '.join(missing_pred)}")
        if missing_ref:
            raise KeyError(f"[{label}] Missing atoms in reference ({name3[1]} chain={ref_chain}): {', '.join(missing_ref)}")

        coords1 = np.concatenate(lig_coords1)
        coords2 = np.concatenate(lig_coords2)

        if "symmetric_atom_groups" in ligand_atom_mapping:
            min_rmsd = permute_symmetric_atoms(
                coords1, coords2, atoms[0], atoms[1],
                ligand_atom_mapping["symmetric_atom_groups"],
                auto_include_fixed_from_atoms=True,
                debug=True,
                label=label
            )
            ligand_rmsd_values.append(min_rmsd)
        else:
            ligand_rmsd_values.append(calculate_rmsd(coords1, coords2))

    return ca_rmsd_value, catres_rmsds, ligand_rmsd_values, ncaa_meta

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
        pred_name  = group["name3"][0]
        pred_atoms = group["atoms"][0]
        # NEW: optional chain on predicted side
        pred_chain = None
        if "chain" in group and isinstance(group["chain"], list) and len(group["chain"]) >= 1:
            pred_chain = group["chain"][0] or None

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

            # protein ignoring logic (unchanged)
            if line.startswith("ATOM") and name3 in standard_aa:
                if (ch_id, res_seq) not in eff_map:
                    continue

            # NEW: enforce predicted ligand chain if provided
            if pred_chain is not None and ch_id != str(pred_chain):
                continue

            # ligand name/atom match
            if name3 == pred_name and atom_name in pred_atoms:
                plddt_vals.append(val)

        print(f"\n{chai_pdb}")
        print(pred_name, pred_atoms, plddt_vals, np.mean(plddt_vals) if plddt_vals else np.nan)

        group_results.append(float(np.mean(plddt_vals)) if plddt_vals else np.nan)

        if plddt_vals:
            group_results.append(float(np.mean(plddt_vals)))
        else:
            group_results.append(np.nan)

    return group_results

############################################
### ncAA-robust overlap & pLDDT helpers  ###
############################################

# Favor backbone first when comparing common atoms
CANONICAL_BACKBONE_ORDER = ["N", "CA", "C", "O"]

def order_common_atoms(names1, names2):
    """
    Return a canonical ordering of the intersection of atom names:
    backbone first (N, CA, C, O) then remaining atoms alphabetically.
    """
    s1, s2 = set(names1), set(names2)
    inter = list(s1 & s2)
    if not inter:
        return []
    back = [a for a in CANONICAL_BACKBONE_ORDER if a in inter]
    rest = sorted([a for a in inter if a not in CANONICAL_BACKBONE_ORDER])
    return back + rest

def rmsd_on_common_atoms(coords1, names1, coords2, names2):
    """
    Compute RMSD using only common atom names (ordered canonically).
    Returns: (rmsd_value, only1_names, only2_names, common_used_names)
    """
    common = order_common_atoms(names1, names2)
    s1, s2 = set(names1), set(names2)
    only1 = sorted(list(s1 - s2))
    only2 = sorted(list(s2 - s1))
    if not common:
        return (float("nan"), only1, only2, [])
    idx1 = [names1.index(n) for n in common]
    idx2 = [names2.index(n) for n in common]
    diff = coords1[idx1] - coords2[idx2]
    rmsd = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
    return (rmsd, only1, only2, common)

def get_resname_by_chain_effindex(structure, chain_id, eff_index, n_ignore=0, c_ignore=0):
    """
    Return residue 3-letter name for the protein residue at a given *effective* index
    (after skipping n_ignore at N-term, c_ignore at C-term) on chain_id.
    """
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            protein = [r for r in chain.get_residues() if r.get_id()[0] == ' ']
            subset = protein[n_ignore: len(protein)-c_ignore] if c_ignore > 0 else protein[n_ignore:]
            if 1 <= eff_index <= len(subset):
                return subset[eff_index-1].get_resname()
    return None

def get_raw_resseq_for_effective_index(structure, chain_id, eff_index, n_ignore=0, c_ignore=0):
    """
    Map an *effective* index back to the raw PDB residue sequence number for chain_id.
    """
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            protein = [r for r in chain.get_residues() if r.get_id()[0] == ' ']
            subset = protein[n_ignore: len(protein)-c_ignore] if c_ignore > 0 else protein[n_ignore:]
            if 1 <= eff_index <= len(subset):
                return subset[eff_index-1].get_id()[1]
    return None

def get_plddt_for_specific_atoms(pdb_path, chain_id, raw_resseq, atom_names):
    """
    Average pLDDT for specific atom names in a residue identified by (chain_id, raw_resseq)
    from a Chai PDB. Returns np.nan if none found.
    """
    if not atom_names:
        return float("nan")
    vals = []
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            ch = line[21:23].strip()
            try:
                rn = int(line[23:27].strip())
            except ValueError:
                continue
            if ch != chain_id or rn != raw_resseq:
                continue
            atom = line[12:16].strip()
            if atom not in atom_names:
                continue
            try:
                vals.append(float(line[61:67]))
            except ValueError:
                pass
    return float(np.mean(vals)) if vals else float("nan")

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
        updated_score_dic[f"complex_plddt_idx_{chai_pdb_num}"] = scores["complex_plddt"][0]

        # chain-level
        for i, ch_id in enumerate(chai_ch_ids):
            updated_score_dic[f"chain{ch_id}_plddt_idx_{chai_pdb_num}"] = scores["per_chain_plddt"][0][i]
            for j, ch_id2 in enumerate(chai_ch_ids):
                updated_score_dic[f"chain{ch_id}_chain{ch_id2}_pae_idx_{chai_pdb_num}"] = scores["pae"][0][i][j]

        # ---- pairwise symmetric PAE means for all unordered chain pairs ----
        # scores["pae"][0] is assumed to be a square (n_chains x n_chains) matrix
        pae_mat = np.array(scores["pae"][0], dtype=float)
        n_chains = len(chai_ch_ids)

        for a in range(n_chains):
            for b in range(a + 1, n_chains):  # unordered pairs only
                ch_a = str(chai_ch_ids[a])
                ch_b = str(chai_ch_ids[b])

                # enforce lexicographic order in the output key (lower letter first)
                c_low, c_high = (ch_a, ch_b) if ch_a <= ch_b else (ch_b, ch_a)

                # symmetric mean from both directions (a->b and b->a)
                pair_mean = float(np.nanmean([pae_mat[a, b], pae_mat[b, a]]))

                key = f"chain{c_low}_chain{c_high}_pair_pae_mean_idx_{chai_pdb_num}"
                updated_score_dic[key] = pair_mean

        # Align & RMSD
        ca_rmsd, catres_rmsds, ligand_rmsds, ncaa_meta = calculate_rmsd_after_alignment(chai_structure, ref_structure, catres_list, args.ligand_groups_json, n_ignore=n_ignore, c_ignore=c_ignore)
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

        # --- NEW: per-catalytic-residue metrics for predicted-only (ncAA) atoms ---
        # For each catalytic residue, if Chai predicted extra atoms not in the reference
        # (e.g., KCX vs LYS), store their count and average pLDDT.
        for i, meta in enumerate(ncaa_meta):
            if not meta["pred_only_atoms"]:
                continue
            # Label with predicted residue name when available; fall back to reference name
            label_core = (meta["pred_resname"] or catres_list[i]["name3"]) + str(i+1)

            # count of predicted-only atoms
            updated_score_dic[f"{label_core}_extra_atoms_count_idx_{chai_pdb_num}"] = len(meta["pred_only_atoms"])

            # average pLDDT of those predicted-only atoms (by raw resseq in predicted PDB)
            avg_plddt_ncaa = get_plddt_for_specific_atoms(
                chai_pdb,
                meta["chain"],
                meta["pred_raw_resseq"],
                meta["pred_only_atoms"]
            )
            updated_score_dic[f"{label_core}_extra_atoms_plddt_idx_{chai_pdb_num}"] = avg_plddt_ncaa

            # (Optional) record which atoms were used for RMSD consistency
            used = meta["common_atoms_used"] or []
            updated_score_dic[f"{label_core}_common_atoms_used_idx_{chai_pdb_num}"] = "|".join(used)
    
    # 3) Output
    score_df = pd.DataFrame([updated_score_dic])
    score_df.to_csv(args.outscr, index=False)
    print(score_df)
