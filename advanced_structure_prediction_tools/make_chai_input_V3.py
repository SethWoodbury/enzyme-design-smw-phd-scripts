#!/usr/bin/env python3
"""
Title: Chai-1 Input Command Generator
Authors: Donghyo Kim & Seth Woodbury

Description:
------------
This script prepares input commands for running Chai-1 predictions by processing protein data provided
either as PDB files (from a directory) or as a FASTA file. The primary steps are:

1. **Argument Parsing:**
   - The script accepts either a PDB directory (--pdb_path) or a FASTA file (--fasta_path) as input.
   - Other required arguments include:
       --output_path: The directory where Chai-1 outputs will be written.
       --ligand_smiles: A JSON string defining ligand SMILES.
       --command_path: The file where the generated Chai-1 commands will be saved.
       --chai_shell_script: The Chai-1 shell script to call for each prediction.
   - The flag --check_made_output is used to filter out inputs that already have Chai-1 predictions.
   - Two new optional flags are introduced:
       --n_terminus_tag: A string (e.g., "MSG") to prepend to the extracted protein sequence.
       --c_terminus_tag: A string (e.g., "GSAWSHPQFEK") to append to the extracted protein sequence.

2. **Protein Sequence Extraction:**
   - For **PDB input**:  
     A multiprocessing pool is used to process each PDB file in parallel. The function
     `extract_protein_sequence_from_pdb` reads each file’s ATOM records, extracts the 3-letter amino acid
     codes for the specified chain (default "A"), converts them to one-letter codes using a mapping (aa3to1),
     and builds a protein sequence.
   - For **FASTA input**:  
     The script simply parses the FASTA file and builds a dictionary of sequences keyed by the FASTA header.
   - If --check_made_output is enabled, the script filters out those inputs that already have an existing
     Chai-1 output file (based on a naming convention).

3. **Command File Generation:**
   - For each input (PDB or FASTA), the script:
       a. Creates an output directory if it does not already exist.
       b. Adds any provided N-terminal tag (using --n_terminus_tag) to the beginning of the sequence and
          any provided C-terminal tag (using --c_terminus_tag) to the end.
       c. Generates a command line that calls the specified Chai-1 shell script with parameters including the name,
          the modified protein sequence, the ligand SMILES, and output options.
       d. Writes each generated command to the specified command file.

Usage Examples:
---------------
- Using PDB files with both terminal tags:
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

# Mapping from three-letter to one-letter amino acid codes.
aa3to1 = {
    "ALA": 'A', "ARG": 'R', "ASN": 'N', "ASP": 'D', "CYS": 'C',
    "GLN": 'Q', "GLU": 'E', "GLY": 'G', "HIS": 'H', "ILE": 'I',
    "LEU": 'L', "LYS": 'K', "MET": 'M', "PHE": 'F', "PRO": 'P',
    "SER": 'S', "THR": 'T', "TRP": 'W', "TYR": 'Y', "VAL": 'V'
}

def extract_protein_sequence_from_pdb(q, seq_dic, chain='A'):
    """
    Extracts the protein sequence from a PDB file by processing its ATOM records.

    For each ATOM record in the file:
      - Extract the 3-letter residue code, chain ID, and residue number.
      - If the record belongs to the specified chain and the residue hasn't been added before,
        convert the 3-letter code to its one-letter equivalent and append it to the sequence.

    The resulting sequence is stored in the shared dictionary 'seq_dic' with the PDB file path as key.
    """
    while True:
        p = q.get(block=True)
        if p is None:
            return
        i, pdb_file = p[0], p[1]

        if (math.log10(i+1) % 1 == 0):
            print(f"[{time.ctime()}] {i+1} PDBs processed.")
        sequence = ""
        seen_residues = set()
        with open(pdb_file, 'r') as file:
            for line in file:
                if line.startswith("ATOM"):
                    residue_name = line[17:20].strip()  # 3-letter residue code
                    residue_chain = line[21].strip()    # Chain ID
                    residue_number = line[22:26].strip()  # Residue number
                    
                    if residue_chain == chain and residue_number not in seen_residues:
                        seen_residues.add(residue_number)
                        if residue_name in aa3to1:
                            sequence += aa3to1[residue_name]
        seq_dic[pdb_file] = sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Chai-1 input commands from PDB or FASTA input. "
                                                 "Authors: Donghyo Kim & Seth Woodbury")
    parser.add_argument("--pdb_path", help="Path of input PDB directory.")
    parser.add_argument("--fasta_path", help="Path of input FASTA file.")
    parser.add_argument("--output_path", help="Path for Chai-1 output.")
    parser.add_argument("--ligand_smiles", type=str, help='JSON string defining ligand SMILES.')
    parser.add_argument("--command_path", type=str, help='Path of the output command file.')
    parser.add_argument("--check_made_output", action="store_true", default=False, help='Check whether Chai-1 predictions were already made.')
    parser.add_argument("--chai_shell_script", type=str, default="/net/software/lab/chai/chai-lab/run_chai.sh", help='Path of Chai-1 shell script.')
    parser.add_argument("--n_terminus_tag", type=str, default="", help="Optional N-terminal tag (e.g. 'MSG') to prepend to the protein sequence.")
    parser.add_argument("--c_terminus_tag", type=str, default="", help="Optional C-terminal tag (e.g. 'GSAWSHPQFEK') to append to the protein sequence.")
    parser.add_argument("--do_not_make_separate_subdirectories", action="store_true", default=False, help="If specified, do not create separate subdirectories for each input. All outputs will be written directly to the output path.")
    args = parser.parse_args()
    
    output_dir = args.output_path
    ligand_smiles = json.loads(args.ligand_smiles)
    commands = args.command_path

    # Create a multiprocessing queue and a shared dictionary for storing sequences.
    the_queue = multiprocessing.Queue()
    manager = multiprocessing.Manager() 
    seq_dic = manager.dict()

    if bool(args.pdb_path) and not bool(args.fasta_path):
        input_dir = args.pdb_path
        pdb_files = glob.glob(os.path.join(input_dir, "*.pdb"))
        if args.check_made_output:
            pdb_files = list(filter(
                lambda fi: not os.path.isfile(os.path.join(args.output_path, os.path.splitext(os.path.basename(fi))[0],
                                      "pred." + os.path.splitext(os.path.basename(fi))[0] + "_model_idx_4.pdb")),
                pdb_files))
            print(f"{len(pdb_files)} PDBs found to run Chai-1.")
        for i, pdb_file in enumerate(pdb_files):
            the_queue.put((i, pdb_file))

        pool = multiprocessing.Pool(os.cpu_count()-1, initializer=extract_protein_sequence_from_pdb, initargs=(the_queue, seq_dic))

        # Signal the workers to exit by pushing None into the queue.
        for _i in range(os.cpu_count()):
            the_queue.put(None)

        the_queue.close()
        the_queue.join_thread()
        pool.close()
        pool.join()

    elif not bool(args.pdb_path) and bool(args.fasta_path):
        with open(args.fasta_path, 'r') as f:
            data = f.read().strip().split(">")
        # For FASTA input, use a regular dictionary.
        seq_dic = {}
        for da in data:
            if len(da) == 0:
                continue
            parts = da.split()
            seq_id, seq = parts[0], "".join(parts[1:])
            seq_dic[seq_id] = seq
        if args.check_made_output:
            pdb_files = list(filter(
                lambda fi: not os.path.isfile(os.path.join(args.output_path, os.path.splitext(os.path.basename(fi))[0],
                                      "pred." + os.path.splitext(os.path.basename(fi))[0] + "_model_idx_4.pdb")),
                seq_dic.keys()))
            print(f"{len(pdb_files)} sequences found to run Chai-1.")
    else:
        raise ValueError("'--pdb_path' or '--fasta_path' should be employed, but not both.")

    cmd_num = 0
    with open(commands, 'w') as cmd_file:
        for i, pdb_file in enumerate(pdb_files):
            pdb_name = os.path.splitext(os.path.basename(pdb_file))[0] if args.pdb_path else pdb_file
            if args.do_not_make_separate_subdirectories:
                pdb_output_dir = output_dir
            else:
                pdb_output_dir = os.path.join(output_dir, pdb_name)
                if not os.path.exists(pdb_output_dir):
                    os.makedirs(pdb_output_dir)
            
            # Construct the protein sequence by adding optional N- and C-terminal tags.
            protein_seq = f"{args.n_terminus_tag}{seq_dic[pdb_file]}{args.c_terminus_tag}"
            
            # Generate the command for this input.
            cmd = (f'{args.chai_shell_script} --name {pdb_name} --protein "[\'{protein_seq}\']" '
                   f'--ligand "{ligand_smiles}" --output_dir {pdb_output_dir} --export_arrays --export_mode json \n')
            cmd_file.write(cmd)
            cmd_num += 1

    print()
    print("Path of the command file:")
    print(commands)
