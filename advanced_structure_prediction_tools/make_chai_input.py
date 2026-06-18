#!/usr/bin/env python3
"""
make_chai_input.py

Author: (Donghyo + patch by Seth Woodbury), 2025-06-04)

DESCRIPTION:
    Generates shell commands to run Chai-1 structure prediction for a set of protein PDB files, each with its ligand SMILES. 

    - Accepts either a directory (--pdb_path) of .pdb files,
      or a single .pdb file (--pdb_file).
    - For each input, extracts the protein sequence and writes
      a shell command for Chai-1 prediction to a commands file.
    - Supports ligand_smiles as a JSON string per PDB or per batch.
    - Handles skipping files if Chai-1 prediction already exists.
    - Can be run in parallel for batch processing.

USAGE EXAMPLES:
    # Run on a directory:
    python make_chai_input.py --pdb_path DIR --ligand_smiles '["[Zn+2]", "SOME_SMILES"]' --output_path OUTDIR --command_path CMDFILE

    # Run on a single file:
    python make_chai_input.py --pdb_file FILE --ligand_smiles '["[Zn+2]", "SOME_SMILES"]' --output_path OUTDIR --command_path CMDFILE

    # (You can call this script once for each PDB/SMILES pair.)

ARGUMENTS:
    --pdb_path         Directory of PDB files (batch mode)
    --pdb_file         Single PDB file (single mode, NEW)
    --fasta_path       Input fasta (alternative input, unchanged)
    --output_path      Output directory for Chai-1 predictions
    --ligand_smiles    JSON string with ligand SMILES
    --command_path     File to write Chai-1 shell commands to
    --check_made_output  Skip predictions already done
    --chai_shell_script Path to Chai-1 shell script
"""


import os
import time
import glob
import json
import math
import queue
import threading
import multiprocessing
import argparse

# --- locate repo root + shared external paths ---
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "repo_paths.py").is_file():
        _sys.path.insert(0, str(_anc)); break
import repo_paths

aa3to1 = {
    "ALA":'A', "ARG":'R', "ASN":'N', "ASP":'D', "CYS":'C',
    "GLN":'Q', "GLU":'E', "GLY":'G', "HIS":'H', "ILE":'I',
    "LEU":'L', "LYS":'K', "MET":'M', "PHE":'F', "PRO":'P',
    "SER":'S', "THR":'T', "TRP":'W', "TYR":'Y', "VAL":'V' }


# Function to extract the protein sequence from ATOM records in a PDB file
def extract_protein_sequence_from_pdb(q, seq_dic, chain='A'):
    while True:
        p = q.get(block=True)
        if p is None:
            return
        i, pdb_file = p[0], p[1]
    
        if (math.log10(i+1) % 1 == 0):
            print (f"[{time.ctime()}] {i+1} PDBs processed.")
        sequence = ""
        seen_residues = set()
        with open(pdb_file, 'r') as file:
            for line in file:
                if line.startswith("ATOM"):
                    residue_name = line[17:20].strip()  # Extract 3-letter residue name
                    residue_chain = line[21].strip()    # Extract chain ID
                    residue_number = line[22:26].strip()  # Extract residue number
                    
                    if residue_chain == chain and residue_number not in seen_residues:
                        seen_residues.add(residue_number)
                        if residue_name in aa3to1:
                            sequence += aa3to1[residue_name]
        seq_dic[pdb_file] = sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_path", help="Directory of input PDBs (batch mode).")
    parser.add_argument("--pdb_file", help="Path to a single input PDB (single mode).")   # <--- NEW ARGUMENT
    parser.add_argument("--fasta_path", help="Path of input fasta.")
    parser.add_argument("--output_path", help="Path of Chai-1 output.")
    parser.add_argument("--ligand_smiles", type=str, help='JSON string defining ligand smiles.')
    parser.add_argument("--command_path", type=str, help='Path of command file.')
    parser.add_argument("--check_made_output", action="store_true", default=False, help='Check whether Chai-1 predictions were already made.')
    parser.add_argument("--chai_shell_script", type=str, default=repo_paths.CHAI_RUN, help='Path of Chai-1 shell script.')
    args = parser.parse_args()

    output_dir = args.output_path
    ligand_smiles = json.loads(args.ligand_smiles)
    commands = args.command_path

    the_queue = multiprocessing.Queue()
    manager = multiprocessing.Manager()
    seq_dic = manager.dict()

    pdb_files = []

    # ---- PATCH: Accept either directory or single file ----
    if args.pdb_file and os.path.isfile(args.pdb_file):
        pdb_files = [args.pdb_file]
    elif args.pdb_path and os.path.isdir(args.pdb_path):
        pdb_files = glob.glob(os.path.join(args.pdb_path, "*.pdb"))
    elif args.fasta_path:
        # ... existing fasta logic ...
        f = open(args.fasta_path)
        data = f.read().strip().split(">")
        f.close()
        seq_dic = {}
        for da in data:
            if len(da) == 0: continue
            da = da.split()
            seq_id, seq = da[0], "".join(da[1:])
            seq_dic[seq_id] = seq
        if args.check_made_output:
            pdb_files = list(filter(lambda fi: not os.path.isfile(os.path.join(args.output_path, os.path.splitext(os.path.basename(fi))[0], "pred."+os.path.splitext(os.path.basename(fi))[0]+"_model_idx_4.pdb")), seq_dic.keys()))
            print (f"{len(pdb_files)} sequences found to run Chai-1.")
    else:
        raise ValueError ("Must specify either --pdb_file, --pdb_path, or --fasta_path.")

    # -- SEQUENCE EXTRACTION LOGIC --
    if pdb_files and (not args.fasta_path):
        if args.check_made_output:
            pdb_files = list(filter(lambda fi: not os.path.isfile(os.path.join(args.output_path, os.path.splitext(os.path.basename(fi))[0], "pred."+os.path.splitext(os.path.basename(fi))[0]+"_model_idx_4.pdb")), pdb_files))
            print (f"{len(pdb_files)} PDBs found to run Chai-1.")
        for i, pdb_file in enumerate(pdb_files):
            the_queue.put((i, pdb_file))
        pool = multiprocessing.Pool(os.cpu_count()-1, initializer=extract_protein_sequence_from_pdb, initargs=(the_queue, seq_dic))
        for _i in range(os.cpu_count()):
            the_queue.put(None)
        the_queue.close()
        the_queue.join_thread()
        pool.close()
        pool.join()

    # -- COMMAND GENERATION --
    cmd_num = 0
    with open(commands, 'w') as cmd_file:
        for i, pdb_file in enumerate(pdb_files):
            pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]
            pdb_output_dir = os.path.join(output_dir, pdb_name)
            if not os.path.exists(pdb_output_dir):
                os.makedirs(pdb_output_dir)
            cmd = f'{args.chai_shell_script} --name {pdb_name} --protein "[\'{seq_dic[pdb_file]}\']" --ligand "{ligand_smiles}" '
            cmd += f' --output_dir {pdb_output_dir} --export_arrays --export_mode json \n'
            cmd_file.write(cmd)
            cmd_num += 1

    print ()
    print ("Path of the command file:")
    print (commands)