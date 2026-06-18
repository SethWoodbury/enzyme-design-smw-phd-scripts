"""
Authors: Seth W. & Donghyo K.


"""
import re
import os
import glob
import math
import json
import time
import argparse
import queue, threading
import multiprocessing
import numpy as np
from Bio import PDB
from pathlib import Path
import shutil
import secrets


AA3to1 = {'CYS':'C', 'ASP':'D', 'SER':'S', 'GLN':'Q', 'LYS':'K', 'ILE':'I', 'PRO':'P', 'THR':'T', 'PHE':'F', 'ASN':'N', 'GLY':'G', 'HIS':'H', 'LEU':'L', 'ARG':'R', 'TRP':'W', 'ALA':'A', 'VAL':'V', 'GLU':'E', 'TYR':'Y', 'MET':'M'}

def get_protein_sequence(pdb_file, chain_id):
    # Initialize the PDB parser
    parser = PDB.PDBParser(QUIET=True)
    
    # Parse the structure from the PDB file
    structure = parser.get_structure('protein', pdb_file)
    
    sequence = ""
    for model in structure:
        for chain in model:
            if not chain.get_id() == chain_id:
                continue
            for residue in chain:
                # Check if the residue is a standard amino acid (excluding water, ions, etc.)
                if PDB.is_aa(residue):
                    # Add the 3-letter code of the amino acid to the sequence
                    sequence += AA3to1[residue.get_resname()]

    return sequence


###############################################
### REMARK 666 → PTM PARSING HELPER FUNCS   ###
###############################################

def parse_ptm_specs(specs_list):
    """
    Turn ['A/LYS/3:KCX', 'B/HIS/1:MSE'] into structured dicts.
    CHAIN/RES3/CATIDX:CCD
    """
    parsed = []
    for spec in specs_list:
        try:
            left, ccd = [s.strip() for s in spec.split(':', 1)]
            chain, res3, catidx = [s.strip() for s in left.split('/', 2)]
            if not chain or not res3 or not catidx or not ccd:
                raise ValueError
            parsed.append({
                "chain": chain,
                "resname": res3.upper(),
                "cat_idx": int(catidx),
                "ccd": ccd.upper()
            })
        except Exception as e:
            raise ValueError(
                f"Bad --ptm_from_remark666 entry '{spec}'. "
                "Expected CHAIN/RES3/CATIDX:CCD (e.g., A/LYS/3:KCX)."
            ) from e
    return parsed


def parse_remark666_catalog(pdb_path):
    """
    Parse REMARK 666 MOTIF lines.
    Robust to spacing; extracts: chain, resname, resnum, cat_idx.

    Matches e.g.:
    REMARK 666 MATCH TEMPLATE X YYE    0 MATCH MOTIF A LYS   48  3  1
    """
    catalog = []
    # Pattern: after 'MOTIF' → <chain> <res3> <resnum> <catidx> <group>
    # Allow flexible whitespace; capture optional insertion code on resnum.
    pat = re.compile(
        r"MOTIF\s+(\S+)\s+([A-Z]{3})\s+(\d+)([A-Z]?)\s+(\d+)\s+(\d+)",
        flags=re.IGNORECASE,
    )
    with open(pdb_path, 'r') as fh:
        for line in fh:
            if "REMARK 666" in line.upper() and "MOTIF" in line.upper():
                m = pat.search(line)
                if m:
                    chain = m.group(1)
                    resname = m.group(2).upper()
                    resnum = int(m.group(3))      # ignore optional insertion code m.group(4)
                    cat_idx = int(m.group(5))
                    catalog.append({
                        "chain": chain,
                        "resname": resname,
                        "resnum": resnum,
                        "cat_idx": cat_idx
                    })
    return catalog


def build_mods_by_chain_from_specs(catalog, ptm_specs, pdb_path_for_err=""):
    """
    Map requested PTMs to actual PDB residue numbers using the parsed catalog.
    Returns: { chain_id: [ {ptmType: CCD, ptmPosition: int}, ... ] }
    """
    mods = {}
    for spec in ptm_specs:
        matches = [
            r for r in catalog
            if r["chain"] == spec["chain"]
            and r["resname"] == spec["resname"]
            and r["cat_idx"] == spec["cat_idx"]
        ]
        if not matches:
            raise ValueError(
                f"PTM target not found in REMARK 666 for {pdb_path_for_err}: "
                f"{spec['chain']}/{spec['resname']}/{spec['cat_idx']} -> {spec['ccd']}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous PTM target in {pdb_path_for_err}: "
                f"{spec['chain']}/{spec['resname']}/{spec['cat_idx']} matches residues "
                f"{[m['resnum'] for m in matches]}."
            )
        resnum = matches[0]["resnum"]
        mods.setdefault(spec["chain"], []).append({
            "ptmType": spec["ccd"],
            "ptmPosition": resnum
        })
    return mods


def choose_base_seed(args) -> int:
    """
    Returns a 32-bit seed in the range 1..(2^32-1).
    Priority:
      1) --base_seed if provided
      2) if --no_random_seed: fixed seed = 1
      3) otherwise: random 32-bit seed
    """
    if args.base_seed is not None:
        s = int(args.base_seed)
        if not (1 <= s <= (2**32 - 1)):
            raise ValueError(f"--base_seed must be in [1, 2^32-1]. Got: {s}")
        return s

    if args.no_random_seed:
        return 1

    return secrets.randbelow(2**32 - 1) + 1


parser = argparse.ArgumentParser()

parser.add_argument("--pdb_path", type=str, required=True, help="Path of input PDB file")
parser.add_argument("--pdb_chain", type=str, nargs="+", required=True, help="PDB chains of proteins to predict structures in the input PDB file")
parser.add_argument("--ligand_chain", type=str, default=[], nargs="+", help="PDB chains of ligand. ex) B C")
parser.add_argument("--ligand_type", type=str, default=[], nargs="+", help="Types of ligand id. ex) ccdCodes smiles")
parser.add_argument("--ligand_id", type=str, default="", help="Ligand id. Make sure that smiles string is JSON-escaped. ex) 'ZN C(=O)(Oc1cc2c(cc1)c(cc(=O)o2)C)Cc1ccccc1'")
parser.add_argument("--json_path", type=str, required=True, help="Path of json file")
parser.add_argument("--json_basename", type=str, default="AF3_input", help="Name of json file")
parser.add_argument("--num_input_per_run", type=int, default=5, help="Number of inputs per AF3 run.")
parser.add_argument("--output_suffix", type=str, default="", help="Suffix to add output file")
parser.add_argument("--output_path", type=str, required=True, help="AF3 output path")
parser.add_argument('--check_made_output', action='store_true', help="Check already made outputs")
parser.add_argument('--cleanup_incomplete_outputs', action='store_true', help="Remove all incomplete outputs")
parser.add_argument("--ptm_from_remark666", type=str, nargs="*", default=[],help=("Optional PTM specs derived from REMARK 666. "
        "Format: CHAIN/RES3/CATIDX:CCD (e.g., 'A/LYS/3:KCX'). "
        "You can pass multiple entries separated by spaces."))
parser.add_argument("--base_seed", type=int, default=None, help=("Optional manual base seed for AF3 modelSeeds (32-bit). "
        "If provided, this exact seed is used for every input."),)
parser.add_argument("--no_random_seed", action="store_true", help=("Disable random base-seed generation. If --base_seed is not provided, "
        "a fixed deterministic seed is used (default: 1)."),)

args = parser.parse_args()
PTM_SPECS = parse_ptm_specs(args.ptm_from_remark666) if args.ptm_from_remark666 else []
args.ligand_id = args.ligand_id.split(" ")
if args.ligand_id == [""]:
    args.ligand_id = []
# Check input arguments
if not (len(args.ligand_chain) == len(args.ligand_type) and len(args.ligand_type) == len(args.ligand_id)):
    raise ValueError(f"Number of ligand_chain ({len(args.ligand_chain)}), ligand_type ({len(args.ligand_type)}), and ligand_id ({len(args.ligand_id)}) should be identical.")

if len(args.ligand_type) != args.ligand_type.count("ccdCodes") + args.ligand_type.count("smiles"):
    raise ValueError(f"Ligand type should be 'ccdCodes' or 'smiles'. Input: {args.ligand_type}")

for i, ligand_id in enumerate(args.ligand_id):
    if args.ligand_type[i] == "ccdCodes": continue
    splitted_ligand_id = ligand_id.split("/")
    if len(splitted_ligand_id) == 1:
        continue
    for j, el in splitted_ligand_id:
        if j == 0: continue
        if el != "":
            if splitted_ligand_id[j-1] != "":
                raise ValueError("Make sure that smiles string is JSON-escaped. In particular the backslash character must be escaped as two backslashes. Please check https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md")

pdb_dir = Path(args.pdb_path)
out_dir = Path(args.output_path)
suffix = args.output_suffix

# Collect input PDB files
all_input_pdbs = [p for p in pdb_dir.iterdir() if p.is_file() and p.suffix == ".pdb"]
print("Number of input PDBs:", len(all_input_pdbs))

#if args.check_made_output:
    # Map each input PDB to its expected output directory
 #   expdir_for_pdb = {p: (out_dir / (p.stem + suffix).lower()) for p in all_input_pdbs}
  #  existing_outputs = 0
   # incomplete_dirs = []
   # pdb_for_incomplete = []   # PDB files corresponding to incomplete outputs
   # to_run = []               # PDB files that still need to be processed

   # for pdb, expdir in expdir_for_pdb.items():
   #     if expdir.is_dir():
    #        existing_outputs += 1
     #       # Check if the expected "complete" subdirectory exists
      #      if not (expdir / "seed-1_sample-4").is_dir():
       #         incomplete_dirs.append(expdir)
        #        pdb_for_incomplete.append(pdb)
        #else:
         #   to_run.append(pdb)

    #print(f"Number of existing outputs: {existing_outputs}")

    # Handle incomplete outputs
    #if incomplete_dirs:
     #   example = str(incomplete_dirs[0])
      #  print(f"Number of incomplete outputs: {len(incomplete_dirs)} [Example: {example}]")
       # if args.cleanup_incomplete_outputs:
        #    for d in incomplete_dirs:
         #       shutil.rmtree(d, ignore_errors=True)
          #  to_run.extend(pdb_for_incomplete)
        #else:
         #   raise ValueError("Please use --cleanup_incomplete_outputs to remove all incomplete outputs")

    #print("Number of PDBs to run AF3:", len(to_run))
#else:
 #   to_run = all_input_pdbs

if args.check_made_output:
    # Map each input PDB to its expected output directory (no .lower())
    expdir_for_pdb = {p: (out_dir / (p.stem + suffix)) for p in all_input_pdbs}
    existing_outputs = 0
    incomplete_dirs = []
    pdb_for_incomplete = []   # PDB files corresponding to incomplete outputs
    to_run = []               # PDB files that still need to be processed

    for pdb, expdir in expdir_for_pdb.items():
        if expdir.is_dir():
            existing_outputs += 1
            samples_dir = expdir / "samples"

            # Incomplete if no samples dir or fewer than 5 .cif files
            cif_files = list(samples_dir.glob("*.cif")) if samples_dir.is_dir() else []
            if len(cif_files) < 5:
                incomplete_dirs.append(expdir)
                pdb_for_incomplete.append(pdb)
        else:
            to_run.append(pdb)

    print(f"Number of existing outputs: {existing_outputs}")

    # Handle incomplete outputs
    if incomplete_dirs:
        example = str(incomplete_dirs[0])
        print(f"Number of incomplete outputs: {len(incomplete_dirs)} [Example: {example}]")
        if args.cleanup_incomplete_outputs:
            for d in incomplete_dirs:
                shutil.rmtree(d, ignore_errors=True)
            to_run.extend(pdb_for_incomplete)
        else:
            # Safety: nothing is deleted unless --cleanup_incomplete_outputs is provided.
            raise ValueError("Please use --cleanup_incomplete_outputs to remove all incomplete outputs")

    print("Number of PDBs to run AF3:", len(to_run))
else:
    to_run = all_input_pdbs



the_queue = multiprocessing.Queue()  # Queue stores the iterables
manager = multiprocessing.Manager()

input_pdbs = []
for i, input_pdb in enumerate(to_run):
    if i % args.num_input_per_run == 0:
        if i != 0:
            the_queue.put((int(i/args.num_input_per_run), input_pdbs))
        input_pdbs = []
    input_pdbs.append(input_pdb)
the_queue.put((int(i/args.num_input_per_run), input_pdbs))
 
def process(q):
    while True:
        p = q.get(block=True)
        if p is None:
            return
        i, input_pdbs = p[0], p[1]

        # safer progress print (every power-of-10 index) or use modulo for regular cadence
        if i > 0 and float(np.log10(i)).is_integer():
            print(f"[{time.ctime()}] {i*args.num_input_per_run} PDBs processed.")

        AF3_input_list = []
        for input_pdb in input_pdbs:
            AF3_input = {
                "name": os.path.basename(input_pdb).replace(".pdb", "") + args.output_suffix,
                "sequences": []
            }

            # --- PTMs: resolve before building protein entries ---
            mods_by_chain = {}
            if PTM_SPECS:
                catalog = parse_remark666_catalog(input_pdb)
                mods_by_chain = build_mods_by_chain_from_specs(catalog, PTM_SPECS, pdb_path_for_err=str(input_pdb))

            # --- Proteins (attach modifications if present) ---
            for chain in args.pdb_chain:
                protein_sequence = get_protein_sequence(input_pdb, chain)
                if len(protein_sequence) == 0:
                    raise ValueError(
                        f"Protein sequence length of [{input_pdb}] at chain [{chain}] is 0."
                    )

                protein_obj = {
                    "id": chain,
                    "sequence": protein_sequence,
                    "unpairedMsa": "",
                    "pairedMsa": "",
                    "templates": "",
                }
                if chain in mods_by_chain:
                    protein_obj["modifications"] = mods_by_chain[chain]

                AF3_input["sequences"].append({"protein": protein_obj})

            # --- Ligands (unchanged) ---
            for ligand_id, ligand_type, ligand_chain in zip(args.ligand_id, args.ligand_type, args.ligand_chain):
                if ligand_type == "smiles":
                    AF3_input["sequences"].append({"ligand": {"id": ligand_chain, ligand_type: ligand_id}})
                else:
                    AF3_input["sequences"].append({"ligand": {"id": ligand_chain, ligand_type: eval(ligand_id)}})

            base_seed = choose_base_seed(args)
            AF3_input["modelSeeds"] = [base_seed]
            print(f"[SEED] {AF3_input['name']}: base_seed={base_seed}")

            AF3_input["dialect"] = "alphafold3"
            AF3_input["version"] = 1

            AF3_input_list.append(AF3_input)
            
        json.dump(AF3_input_list, open(os.path.join(args.json_path, f"{args.json_basename}_{i}.json"), 'w'))

# print(f"Performing analysis using {NPROC} processes")
pool = multiprocessing.Pool(os.cpu_count()-1, process, (the_queue, ))

# None to end each process
for _i in range(os.cpu_count()):
    the_queue.put(None)

# Closing the queue and the pool
the_queue.close()
the_queue.join_thread()
pool.close()
pool.join()
