from Bio import PDB
from numpy.linalg import norm
import os, sys, glob
import json
import argparse
import numpy as np
import pandas as pd

def rapid_skip_if_sc_present(af3_dir, enabled):
    """
    If enabled and --af3_dir contains any *.sc files, print a note and exit 0.
    Do nothing otherwise.
    """
    if not enabled:
        return
    if not af3_dir or not os.path.isdir(af3_dir):
        return
    sc_hits = glob.glob(os.path.join(af3_dir, "*.sc"))
    if sc_hits:
        print(f"[RAPID] Found existing .sc file(s) in {af3_dir}:")
        for p in sc_hits[:5]:
            print(f"        - {os.path.basename(p)}")
        if len(sc_hits) > 5:
            print(f"        ... (+{len(sc_hits)-5} more)")
        print("[RAPID] Skipping processing and exiting cleanly (--rapid_mode_skip_sc_if_found).")
        sys.exit(0)

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

    print(f"\nCatalytic residues identified in {pdb_file}: {catalytic_residues}", f"\n")
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

def extract_coords_atom(structure, ch, residue_name, atom_name):
    """Extract coordinates of a specific atom in a residue."""
    coords = []
    for model in structure:
        for chain in model:
            if not chain.id == ch:
                continue
            for residue in chain:
                if residue.get_resname() == residue_name:
                    for atom in residue:
                        #print (chain.id , residue.get_resname(), atom.get_name())
                        if atom.get_name() == atom_name:
                            coords.append(atom.get_coord())
    return np.array(coords)

def extract_coords_res(structure, chain_id, residue_num):
    """Extract heavy-atom coords and names for a specific (chain, resnum)."""
    coords, names = [], []
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
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
    
    # Calculate RMSD for the catalytic residues     # NEW: be robust to ncAAs — compare only common atoms, collect AF3-only atoms
    catres_rmsds = []          # NEW: init
    ncaa_meta = []             # NEW: init

    for catres in catres_list:
        catres_coords1, catres_atom_name1 = extract_coords_res(structure1, catres["chain"], catres["res_num"])
        catres_coords2, catres_atom_name2 = extract_coords_res(structure2, catres["chain"], catres["res_num"])

        rmsd_val, af3_only_atoms, ref_only_atoms, common_used = rmsd_on_common_atoms(
            catres_coords1, catres_atom_name1, catres_coords2, catres_atom_name2
        )

        if not common_used:
            rmsd_val = float("nan")

        catres_rmsds.append(rmsd_val)

        # Capture ncAA / overlap metadata per residue
        af3_resname_here = get_resname_by_chain_resnum(structure1, catres["chain"], catres["res_num"])
        ref_resname_here = get_resname_by_chain_resnum(structure2, catres["chain"], catres["res_num"])
        ncaa_meta.append({
            "chain": catres["chain"],
            "resnum": catres["res_num"],
            "ref_resname": ref_resname_here,
            "af3_resname": af3_resname_here,
            "af3_only_atoms": af3_only_atoms,
            "ref_only_atoms": ref_only_atoms,
            "common_atoms_used": common_used
        })

    
    # Calculate RMSD for the ligand atoms
    ligand_rmsd_values = []
    
    for ligand_atom_mapping in ligand_atom_mapping_list:
        label, chain, name3, atoms = ligand_atom_mapping["label"], ligand_atom_mapping["chain"], ligand_atom_mapping["name3"], ligand_atom_mapping["atoms"]
        
        # Extract the coordinates for the specified atoms (from atom_mapping)
        ligand_coords1, ligand_coords2 = [], []
        for atom1, atom2 in zip(atoms[0], atoms[1]):
            res1, res2 = name3[0], name3[1]
            ch1, ch2 = chain[0], chain[1]
            ligand_coords1.append(extract_coords_atom(structure1, ch1, res1, atom1))
            ligand_coords2.append(extract_coords_atom(structure2, ch2, res2, atom2))

        # Flatten the list of coordinates (there should be one coordinate set for each pair)
        try:
            ligand_coords1 = np.concatenate(ligand_coords1)
        except ValueError:
            error_atoms = []
            for i, el in enumerate(ligand_coords1):
                if len(el) == 0:
                    error_atoms.append(atoms[0][i])
            raise KeyError(f"[{label}] Missing atoms in AF3({chain[0]}:{name3[0]}): {', '.join(error_atoms)} in file mapped to {atoms[0]}")
            
        try:
            ligand_coords2 = np.concatenate(ligand_coords2)
        except ValueError:
            error_atoms = []
            for i, el in enumerate(ligand_coords2):
                if len(el) == 0:
                    error_atoms.append(atoms[1][i])
            raise KeyError(f"[{label}] Missing atoms in AF3({chain[0]}:{name3[0]}): {', '.join(error_atoms)} in file mapped to {atoms[0]}")


        # Calculate RMSD for the specified ligand atoms
        #ligand_rmsd_values.append(calculate_rmsd(ligand_coords1, ligand_coords2))
        if "symmetric_atom_groups" in ligand_atom_mapping:
            min_rmsd = permute_symmetric_atoms(
                ligand_coords1,
                ligand_coords2,
                atoms[0], atoms[1],
                ligand_atom_mapping["symmetric_atom_groups"]
            )
            ligand_rmsd_values.append(min_rmsd)
        else:
            ligand_rmsd_values.append(calculate_rmsd(ligand_coords1, ligand_coords2))

    
    return ca_rmsd_value, catres_rmsds, ligand_rmsd_values, ncaa_meta

def permute_symmetric_atoms(coords1, coords2, atom_names1, atom_names2, symmetric_groups):
    """
    Calculates RMSD using explicitly defined symmetric atom subgroups.
    Each group contains multiple subgroups in atoms1 and corresponding allowed permutations in atoms2.
    """
    from itertools import product

    name_to_idx1 = {name: i for i, name in enumerate(atom_names1)}
    name_to_idx2 = {name: i for i, name in enumerate(atom_names2)}
    min_rmsd = float("inf")

    for group in symmetric_groups:
        atoms1_subgroups = group["lig1"]
        atoms2_permutations_per_subgroup = group["lig2"]

        if len(atoms1_subgroups) != len(atoms2_permutations_per_subgroup):
            raise ValueError("Mismatch in number of atoms1 subgroups and atoms2 permutation sets")

        # Create all combinations of one permutation per subgroup (like a product of choices)
        subgroup_perm_combos = product(*atoms2_permutations_per_subgroup)

        for atoms2_perm_combo in subgroup_perm_combos:
            coords1_concat = []
            coords2_concat = []
            for atoms1_subgroup, atoms2_perm in zip(atoms1_subgroups, atoms2_perm_combo):
                idxs1 = [name_to_idx1[a] for a in atoms1_subgroup]
                idxs2 = [name_to_idx2[a] for a in atoms2_perm]

                coords1_concat.extend(coords1[idxs1])
                coords2_concat.extend(coords2[idxs2])

            coords1_array = np.array(coords1_concat)
            coords2_array = np.array(coords2_concat)

            if coords1_array.shape != coords2_array.shape:
                raise ValueError("Mismatch in coords shape during symmetric RMSD calculation")

            rmsd = calculate_rmsd(coords1_array, coords2_array)
            min_rmsd = min(min_rmsd, rmsd)

    return min_rmsd

def get_plddt_per_residue_dic(af3_pdb):
    af3_pdbstr = open(af3_pdb, 'r').read().split("\n")
    plddt_dic = {}
    for line in af3_pdbstr:
        if not line[:6] in ["ATOM  ", "HETATM"]:
            continue
        if len(line.strip()) < 66:
            continue
        ch_id = line[21:23].strip()
        res = int(line[23:27].strip())
        key = (ch_id, res)
        
        if not key in plddt_dic: plddt_dic[key] = {}
        plddt_dic[key][line[12:16].strip()] = float(line[61:67])
    plddt_per_residue_dic = {key: np.average(list(plddt_dic[key].values())) for key in plddt_dic}
    return plddt_per_residue_dic

def get_avr_plddt(af3_pdb, ligand_groups_json):
    af3_pdbstr = open(af3_pdb, 'r').read().split("\n")
    avr_plddt_list = []
    for ligand in ligand_groups_json:
        plddt_list = []
        for line in af3_pdbstr:
            if not line[:6] in ["ATOM  ", "HETATM"]:
                continue
            if len(line.strip()) < 66:
                continue
            atom = line[12:16].strip()
            name3 = line[17:21].strip()
            chain = line[21:23].strip()
            plddt = float(line[61:67])
            if (name3 == ligand["name3"][0] and atom in ligand["atoms"][0] and chain == ligand["chain"][0]):
                plddt_list.append(plddt)
        avr_plddt_list.append(np.mean(plddt_list))
    return avr_plddt_list


### --- NEW HELPERS FOR PARTIAL-OVERLAP RESIDUE COMPARISON --- ###
# Prefer comparing backbone first, then sidechain, for a stable atom order.
CANONICAL_BACKBONE_ORDER = ["N", "CA", "C", "O"]

def order_common_atoms(names1, names2):
    """
    Return a reproducible, canonical ordering of the atom-name intersection
    favoring backbone first, then the remaining atoms alphabetically.
    """
    s1, s2 = set(names1), set(names2)
    inter = list(s1 & s2)
    if not inter:
        return []  # no overlap
    # backbone first (if present), then the rest sorted
    back = [a for a in CANONICAL_BACKBONE_ORDER if a in inter]
    rest = sorted([a for a in inter if a not in CANONICAL_BACKBONE_ORDER])
    return back + rest

def rmsd_on_common_atoms(coords1, names1, coords2, names2):
    """
    Compute RMSD using only common atom names. Returns:
      (rmsd_value, af3_only_atom_names, ref_only_atom_names, common_atom_names)
    """
    common = order_common_atoms(names1, names2)
    s1, s2 = set(names1), set(names2)
    af3_only = sorted(list(s1 - s2))
    ref_only = sorted(list(s2 - s1))
    if not common:
        return (float("nan"), af3_only, ref_only, [])
    idx1 = [names1.index(n) for n in common]
    idx2 = [names2.index(n) for n in common]
    return (calculate_rmsd(coords1[idx1], coords2[idx2]), af3_only, ref_only, common)

def get_resname_by_chain_resnum(structure, chain_id, resnum):
    """Grab the 3-letter residue name at (chain_id, resnum) from a Bio.PDB structure."""
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.get_id()[1] == resnum:
                    return residue.get_resname()
    return None  # not found

def get_plddt_for_specific_atoms(pdb_path, chain_id, resnum, atom_names):
    """
    Average pLDDT for specific atom names in a residue from a raw AF3 PDB file.
    Returns np.nan if none found.
    """
    if not atom_names:
        return float("nan")
    plddts = []
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if len(line.strip()) < 67:
                continue
            ch = line[21:23].strip()
            try:
                rn = int(line[23:27].strip())
            except ValueError:
                continue
            if ch != chain_id or rn != resnum:
                continue
            atom = line[12:16].strip()
            if atom in atom_names:
                try:
                    plddts.append(float(line[61:67]))
                except ValueError:
                    pass
    return float(np.mean(plddts)) if plddts else float("nan")


### PAE MATRIX WORK FUNCTIONS ###
def compute_interchain_pae_means_from_conf(conf_json_path, valid_chains=None, verbose=False, label=None):
    """
    Reads an AF3 *_confidences.json, extracts the NxN PAE matrix and token_chain_ids,
    and returns a dict mapping unordered chain pairs (a,b) with a<b to the mean
    interchain PAE across all AB and BA entries.

    If verbose=True, prints for each pair:
      - full PAE matrix (with row/col labels = token index : chain id)
      - boolean mask (1 where pair selected, 0 otherwise)
      - 'sliced' matrix of same shape with only AB/BA entries kept (others blank)
      - sum of selected entries, count used, and the mean value
    """
    try:
        with open(conf_json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}

    pae_list = data.get("pae")
    token_chains_list = data.get("token_chain_ids")
    if pae_list is None or token_chains_list is None:
        return {}

    # Build arrays (None -> NaN for safety)
    pae = np.array(
        [[(float(x) if x is not None else np.nan) for x in row] for row in pae_list],
        dtype=float,
    )
    token_chains = np.array(token_chains_list, dtype=str)

    # Shape checks
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1] or len(token_chains) != pae.shape[0]:
        return {}

    # Determine chain IDs to consider
    chain_set = sorted(set(token_chains.tolist()))
    if valid_chains is not None:
        valid = set(map(str, valid_chains))
        chain_list = [c for c in chain_set if c in valid]
    else:
        chain_list = chain_set

    # Nice labels for printing
    labels = [f"{i}:{c}" for i, c in enumerate(token_chains)]

    from itertools import combinations
    out = {}
    with np.errstate(all="ignore"):
        for a, b in combinations(chain_list, 2):
            rows_a = (token_chains == a)[:, None]  # Nx1
            cols_b = (token_chains == b)[None, :]  # 1xN
            rows_b = (token_chains == b)[:, None]
            cols_a = (token_chains == a)[None, :]

            mask = (rows_a & cols_b) | (rows_b & cols_a)  # AB ∪ BA
            vals = pae[mask]
            finite = np.isfinite(vals)
            count = int(np.count_nonzero(finite))
            sum_val = float(np.nansum(vals)) if vals.size > 0 else float("nan")
            mean_val = float(np.nanmean(vals)) if count > 0 else float("nan")

            out[(a, b)] = mean_val

            if verbose:
                # Pretty print with pandas; keep original shape for sliced matrix
                try:
                    import pandas as _pd
                    with _pd.option_context(
                        "display.max_rows", None,
                        "display.max_columns", None,
                        "display.width", 500,
                        "display.float_format", lambda x: f"{x:6.2f}"
                    ):
                        print("\n" + "="*80)
                        print(f"[VERBOSE] Token-level PAE  |  Pair {a}-{b}  |  {label or os.path.basename(conf_json_path)}")
                        print("- token_chain_ids:")
                        print(" ".join(token_chains.tolist()))

                        df_pae = _pd.DataFrame(pae, index=labels, columns=labels)
                        print("\n- PAE matrix:")
                        print(df_pae)

                        df_mask = _pd.DataFrame(mask.astype(int), index=labels, columns=labels)
                        print("\n- Boolean mask (1==selected AB/BA):")
                        print(df_mask)

                        sliced = np.full_like(pae, np.nan, dtype=float)
                        sliced[mask] = pae[mask]
                        # blank out NaNs for visibility (keep shape)
                        df_sliced = _pd.DataFrame(sliced, index=labels, columns=labels).fillna("")
                        print("\n- Sliced matrix (only AB/BA shown; others blank):")
                        print(df_sliced)

                        print(f"\n- Sum over mask: {sum_val:.6g}  |  Count: {count}  |  Mean: {mean_val:.6g}")
                        print("="*80)
                except Exception as _:
                    # Fallback: minimal prints without pandas
                    print("\n" + "="*80)
                    print(f"[VERBOSE] Pair {a}-{b}  |  {label or os.path.basename(conf_json_path)}")
                    print("- token_chain_ids:")
                    print(" ".join(token_chains.tolist()))
                    print("\n- Sum / Count / Mean:", sum_val, count, mean_val)
                    print("="*80)

    return out

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_pdb", help="Path of reference PDB.")
    parser.add_argument("--af3_dir", help="Path of that directory with five AF3 PDB predictions.")
    parser.add_argument("--outscr", type=str, help="Output score file path.")
    parser.add_argument("--ligand_groups_json", type=str, help='JSON string defining atom groups and labels.')
    parser.add_argument("--calculate_avg_pae_in_addition_to_pair", action="store_true",help="If set, also compute mean interchain PAE from *_confidences.json (token-level PAE).",)
    parser.add_argument("--verbose", action="store_true", help="Print token-level PAE debug info (PAE matrix, mask, sliced view, sum & count per chain pair).")
    parser.add_argument("--rapid_mode_skip_sc_if_found", action="store_true", help="If set, and any *.sc file exists in --af3_dir, exit immediately without doing any work.")


    args = parser.parse_args()
    rapid_skip_if_sc_present(args.af3_dir, args.rapid_mode_skip_sc_if_found)

    args.ligand_groups_json = json.loads(args.ligand_groups_json)
    
    updated_score_dic = {"description": args.af3_dir}

    ref_structure = load_pdb(args.ref_pdb)
    catres_list = get_catalytic_residues(args.ref_pdb)

    af3_pdbs = sorted(glob.glob(os.path.join(args.af3_dir, "*.pdb")))
    for af3_pdb_num, af3_pdb in enumerate(af3_pdbs):
        af3_structure = load_pdb(af3_pdb)
        af3_ch_ids = [chain.id for model in af3_structure for chain in model]
        
        # Save other metrics from AF3 (iptm, ptm)    
        with open(af3_pdb.replace("_model.pdb", "_summary_confidences.json"), "r") as file:
            summary_confidences = json.load(file)    
        with open(af3_pdb.replace("_model.pdb", "_confidences.json"), "r") as file:
            confidences = json.load(file)

        updated_score_dic[f"iptm_idx_{af3_pdb_num}"] = summary_confidences["iptm"]
        updated_score_dic[f"ptm_idx_{af3_pdb_num}"] = summary_confidences["ptm"]
        
        for i, ch_id in enumerate(af3_ch_ids):
            updated_score_dic[f"chain{ch_id}_plddt_idx_{af3_pdb_num}"] = np.mean([el[1] for el in filter(lambda el: el[0] == ch_id, zip(confidences["atom_chain_ids"], confidences["atom_plddts"]))])
            for j, ch_id2 in enumerate(af3_ch_ids):
                val = summary_confidences["chain_pair_pae_min"][i][j]
                if val is None:
                    val = float("nan")
                updated_score_dic[f"chain{ch_id}_chain{ch_id2}_pair_pae_min_idx_{af3_pdb_num}"] = val

        # --- NEW: symmetric PAE min averages per chain pair (off-diagonal only) ---
        # Must run inside the AF3 PDB loop, after you've filled updated_score_dic with
        # chain{X}_chain{Y}_pair_pae_min_idx_{af3_pdb_num} values.
        with np.errstate(all="ignore"):
            # iterate unique unordered pairs
            for i, ch_i in enumerate(af3_ch_ids):
                for j, ch_j in enumerate(af3_ch_ids):
                    if j <= i:
                        continue  # skip diagonal and reverse duplicates

                    # keys for both directions as already stored above
                    key_ij = f"chain{ch_i}_chain{ch_j}_pair_pae_min_idx_{af3_pdb_num}"
                    key_ji = f"chain{ch_j}_chain{ch_i}_pair_pae_min_idx_{af3_pdb_num}"

                    v_ij = updated_score_dic.get(key_ij, float("nan"))
                    v_ji = updated_score_dic.get(key_ji, float("nan"))

                    # average, ignoring NaNs (if both are NaN, result is NaN)
                    avg_val = np.nanmean([v_ij, v_ji])

                    # canonical name: lower/earlier chain id first (alphabetical)
                    a, b = sorted([str(ch_i), str(ch_j)])
                    avg_key = f"chain{a}_chain{b}_pair_pae_min_avg_idx_{af3_pdb_num}"

                    # store (keep as float; NaN will round-trip fine in the CSV)
                    updated_score_dic[avg_key] = float(avg_val) if np.isfinite(avg_val) else float("nan")


        # --- mean interchain PAE from token-level PAE matrix (optional, skip if unavailable) ---
        if args.calculate_avg_pae_in_addition_to_pair:
            conf_json_path = af3_pdb.replace("_model.pdb", "_confidences.json")
            try:
                if os.path.isfile(conf_json_path) and not conf_json_path.endswith("summary_confidences.json"):
                    pair_means = compute_interchain_pae_means_from_conf(
                        conf_json_path,
                        valid_chains=af3_ch_ids,
                        verbose=args.verbose,
                        label=f"idx {af3_pdb_num}"
                    )
                    if pair_means:
                        for (a, b), mean_val in pair_means.items():
                            x, y = sorted([str(a), str(b)])
                            key = f"chain{x}_chain{y}_pair_pae_mean_idx_{af3_pdb_num}"
                            updated_score_dic[key] = float(mean_val) if np.isfinite(mean_val) else float("nan")
            except Exception as e:
                print(f"[WARN] Skipping token-level PAE for idx {af3_pdb_num}: {e}")


        ca_rmsd, catres_rmsds, ligand_rmsds, ncaa_meta = calculate_rmsd_after_alignment(af3_structure, ref_structure, catres_list, args.ligand_groups_json)
        ligand_plddts = get_avr_plddt(af3_pdb, args.ligand_groups_json)
        
        # Save Ca RMSD
        updated_score_dic[f"ca_rmsd_idx_{af3_pdb_num}"] = ca_rmsd
        
        # Save ligand RMSD
        for i, (ligand_rmsd, ligand_plddt) in enumerate(zip(ligand_rmsds, ligand_plddts)):
            updated_score_dic[f"{args.ligand_groups_json[i]['label']}_rmsd_idx_{af3_pdb_num}"] = ligand_rmsd
            updated_score_dic[f"{args.ligand_groups_json[i]['label']}_plddt_idx_{af3_pdb_num}"] = ligand_plddt
            
        # Save catalytic residue RMSD
        for i, catres in enumerate(catres_list):
            updated_score_dic[f"{catres['name3']}{i+1}_rmsd_idx_{af3_pdb_num}"] = catres_rmsds[i]
        updated_score_dic[f"catres_rmsd_idx_{af3_pdb_num}"] = np.mean(catres_rmsds)

        # Save catalytic residue plDDT
        plddt_per_residue_dic = get_plddt_per_residue_dic(af3_pdb)
        cat_res_plddts = []
        for i, catres in enumerate(catres_list):
            cat_res_plddt = np.mean([plddt_per_residue_dic[el] for el in filter(lambda el: el == (catres["chain"], catres["res_num"]), plddt_per_residue_dic.keys())])
            updated_score_dic[f"{catres['name3']}{i+1}_plddt_idx_{af3_pdb_num}"] = cat_res_plddt
            cat_res_plddts.append(cat_res_plddt)
        updated_score_dic[f"catres_plddt_idx_{af3_pdb_num}"] = np.nanmean(cat_res_plddts)

        # --- NEW: per-catalytic-residue metrics for AF3-only (ncAA) atoms ---
        # For each catalytic residue, if AF3 had extra atoms not in the ref (e.g., KCX vs LYS),
        # record how many and their average pLDDT (no RMSD).
        for i, meta in enumerate(ncaa_meta):
            if not meta["af3_only_atoms"]:
                continue
            # We'll name using the AF3 residue name if present; fall back to reference name.
            label_core = (meta["af3_resname"] or catres_list[i]["name3"]) + str(i+1)
            # Count of AF3-only atoms:
            updated_score_dic[f"{label_core}_extra_atoms_count_idx_{af3_pdb_num}"] = len(meta["af3_only_atoms"])
            # Average pLDDT for those atoms:
            avg_plddt_ncaa = get_plddt_for_specific_atoms(
                af3_pdb,
                meta["chain"],
                meta["resnum"],
                meta["af3_only_atoms"]
            )
            updated_score_dic[f"{label_core}_extra_atoms_plddt_idx_{af3_pdb_num}"] = avg_plddt_ncaa
            # (Optional) For transparency, you can also record which atoms were used for RMSD:
            updated_score_dic[f"{label_core}_common_atoms_used_idx_{af3_pdb_num}"] = "|".join(meta["common_atoms_used"]) if meta["common_atoms_used"] else ""
        
    score_df = pd.DataFrame([updated_score_dic])
    score_df.to_csv(args.outscr, index=False)
        
    print (score_df)
    