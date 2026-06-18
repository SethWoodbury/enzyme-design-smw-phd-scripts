#!/usr/bin/env python3
"""
Title: Chai-1 Input Command Generator (with Per-PDB Ligand Assignment)
Authors: Donghyo Kim, Seth Woodbury, OpenAI (ChatGPT)

Description:
------------
This script automates the creation of shell commands to run Chai-1 protein structure predictions
on a collection of inputs. Each input can be provided either as a PDB file (one or many in a directory)
or as an entry in a FASTA file. The script handles sequence extraction (if PDB input), optional tagging
of N- and C-termini, conditional skipping of already-predicted entries, and flexible assignment of
ligand SMILES strings—either a single ligand for all inputs or individual ligands per PDB based on
substring matching in filenames.

**Key Features:**
  1. **PDB vs. FASTA Input (mutually exclusive)**  
     - `--pdb_path <dir>`: Look for all `.pdb` files in the given directory. For each PDB, extract the
       one-letter amino acid sequence from chain “A” (default) using a multiprocessing pool (parallelized).
       If `--check_made_output` is specified, any PDB whose output directory already contains
       `pred.<basename>_model_idx_4.pdb` is skipped.  
     - `--fasta_path <file>`: Read a FASTA file; each FASTA header (the “>ID” line) is treated as a separate
       sequence. No PDB parsing is done. If `--check_made_output` is set, any FASTA ID whose output
       directory already contains `pred.<ID>_model_idx_4.pdb` is skipped.

  2. **Ligand SMILES Specification**  
     - **Global SMILES** (`--ligand_smiles '["[Zn+2]", "<SMILES>"]'`): A single 2-element JSON list of SMILES
       (e.g. `["[Zn+2]","CC(=O)Nc1nnc..."]`). Every input uses the same ligand.  
     - **Per-PDB Assignment** (`--ligand_assignment_json '<JSON object>'`): A JSON object mapping simple
       substrings → 2-element SMILES lists. For each PDB basename (no path, no .pdb), the script scans keys
       in insertion order. The first key K such that `K in basename` is used to select `ligand_assignment[K]`.
       If no key matches, but a `"default"` entry is present, it uses `ligand_assignment["default"]`. If
       neither a key nor “default” matches—and no global SMILES was provided—the script errors.

  3. **Terminal Tags**  
     - `--n_terminus_tag <string>`: If provided, this short peptide (e.g. `"MSG"`) is prepended to each
       extracted/provided sequence before calling Chai.  
     - `--c_terminus_tag <string>`: If provided, this peptide (e.g. `"GSAWSHPQFEK"`) is appended.

  4. **Skipping Completed Predictions**  
     - `--check_made_output`: When set, before generating any commands, the script checks for each input
       whether the file `output_path/<basename>/pred.<basename>_model_idx_4.pdb` already exists. If so,
       that PDB or FASTA ID is omitted from the command list.

  5. **Output Directory Organization**  
     - By default, for each input with basename B, a separate subdirectory `output_path/B/` is created.
       All Chai-1 results for B are written there.  
     - If `--do_not_make_separate_subdirectories` is specified, all inputs write directly to `output_path`
       (no per-input subfolders). In that mode, multiple inputs share the same folder; ensure unique B or
       manage naming collisions externally.

  6. **Final Command File**  
     - Each generated command is a line of the form:  
       ```
       <chai_shell_script> --name <basename> --protein "['<Ntag><sequence><Ctag>']" \
       --ligand "<SMILES_list>" --output_dir <output_path>/<basename> \
       --export_arrays --export_mode json
       ```
     - Those lines are written to `--command_path`. You can then submit them (e.g., as an array job) to
       your scheduler.

Command-Line Arguments:
-----------------------
  --pdb_path <directory>  
      Required if not using FASTA. Directory containing one or more `.pdb` files.  
      Example: `--pdb_path /data/myproject/pdb_inputs/`

  --fasta_path <file>  
      Required if not using PDB. Path to a FASTA file, where each header (`>ID`) is treated as an input.  
      Example: `--fasta_path sequences.fasta`

  --output_path <directory>  
      **Mandatory.** Root directory for all Chai-1 outputs.  
      If using separate subdirs (default), each input B writes into `output_path/B/`.  
      If `--do_not_make_separate_subdirectories` is set, all inputs write to `output_path` directly.  
      Example: `--output_path /data/myproject/chai_outputs/`

  --ligand_smiles <JSON string>  
      A JSON list of length 2: `["<charge>","<SMILES>"]`.  
      Example: `--ligand_smiles '["[Zn+2]","CC(=O)Nc1nnc..."]'`  
      Used **only if** `--ligand_assignment_json` is **not** provided.

  --ligand_assignment_json <JSON object>  
      **Overrides** `--ligand_smiles`. Must be a JSON object whose keys are **simple substrings** (no regex,
      no tuple syntax) and whose values are 2-element SMILES lists.  
      Example:  
      ```
      --ligand_assignment_json '{
        "group2": ["[Zn+2]","CC(=O)Nc1nnc(S([NH-])(=O)=O)s1"],
        "group3": ["[Zn+2]","[NH-]S(=O)(=O)c2ccc(NC(=S)OCCc1ccsc1)cc2"],
        "default": ["[Zn+2]","[NH-]S(=O)(=O)c1ccc(CO/N=C/CO)cc1"]
      }'
      ```
      - For a PDB basename B, the script checks:  
        1. Does `"group2"` appear anywhere in B? If so, use `["[Zn+2]", "..."]` from `"group2"`.  
        2. Else if `"group3"` appears in B, use its SMILES.  
        3. Else, use `"default"` (if present).  
        4. If no key matches and no `"default"` is provided, the script will **raise an error**.

  --command_path <file>  
      **Mandatory.** File path (absolute or relative) where all generated Chai-1 commands will be written.
      Example: `--command_path my_chai_commands.txt`

  --chai_shell_script <file>  
      Path to the Chai-1 wrapper script (shell script that actually invokes the predictor).  
      Default: `/net/software/lab/chai/chai-lab/run_chai.sh`

  --check_made_output  
      If specified, skip any input B whose output directory already contains `pred.<B>_model_idx_4.pdb`.
      This prevents re-running predictions that have already completed. Useful for resuming.

  --n_terminus_tag <string>  
      Optional peptide tag to prepend to every extracted or FASTA sequence. Example: `--n_terminus_tag MSG`

  --c_terminus_tag <string>  
      Optional peptide tag to append to every sequence. Example: `--c_terminus_tag GSAWSHPQFEK`

  --do_not_make_separate_subdirectories  
      If provided, all outputs are written directly into `--output_path` instead of
      `--output_path/<basename>/`. Use this mode if you do not want per-input folders.

Behavior Summary:
-----------------
1. **Early validation** ensures exactly one of `--pdb_path` or `--fasta_path` is given, and at least one of
   `--ligand_smiles` or `--ligand_assignment_json`.  
2. **PDB mode** (`--pdb_path`):  
   - Collects all `*.pdb` files under the specified directory, sorts them.  
   - If `--check_made_output` is used, filters out any PDB whose `output_path/<basename>/pred.<basename>_model_idx_4.pdb`
     already exists.  
   - Uses a `multiprocessing.Pool` to parallelize `extract_protein_sequence_from_pdb()`. Each worker reads an
     assigned PDB file’s ATOM records for chain “A,” converts 3-letter residues to 1-letter (via `aa3to1`),
     and stores the full sequence in a shared dictionary `seq_dic[pdb_path]`.  
3. **FASTA mode** (`--fasta_path`):  
   - Splits the FASTA by “>”. Each header line’s first token is the sequence ID. The remainder is the sequence.
   - If `--check_made_output` is used, filters out IDs whose `output_path/<ID>/pred.<ID>_model_idx_4.pdb`
     already exists.  
4. **Command generation**:  
   - For each retained input (PDB path or FASTA ID):  
     a. Determine `basename` = filename without extension (PDB) or FASTA ID.  
     b. Decide the output directory:  
        - If `--do_not_make_separate_subdirectories`, then `outdir = output_path`.  
        - Otherwise, `outdir = os.path.join(output_path, basename)` and create it if missing.  
     c. Build `protein_seq = n_terminus_tag + seq_dic[orig_key] + c_terminus_tag`.  
     d. Determine `ligand_smiles`:  
        - If `--ligand_assignment_json` is given, call `choose_ligand_smiles(basename, ligand_assignment, global_default=None)`.  
        - Else use the single `global_smiles` from `--ligand_smiles`.  
     e. Form the Chai-1 command line:  
        ```
        {chai_shell_script} \
          --name {basename} \
          --protein "['{protein_seq}']" \
          --ligand "{ligand_smiles}" \
          --output_dir {outdir} \
          --export_arrays --export_mode json
        ```
        (all on one line, with appropriate quoting).  
     f. Append that line to `--command_path`.  
5. **Final output**:  
   - After looping over all inputs, print:  
     ```
     [DONE] Wrote <N> commands to: <command_path>
     ```

Error Conditions:
-----------------
- If **both** `--pdb_path` and `--fasta_path` are specified, the script exits with an error.  
- If **neither** `--ligand_smiles` nor `--ligand_assignment_json` is provided, the script exits with an error.  
- If `--ligand_assignment_json` does not parse as a JSON object with keys as strings and values as 2-element lists, it exits with an error.  
- If **no key** in `ligand_assignment_json` matches a PDB basename and there is **no** `"default"` key or no global SMILES given, the script raises:
"""

import os
import glob
import json
import math
import multiprocessing
import argparse
import re

# Mapping from three-letter to one-letter amino acid codes.
aa3to1 = {
    "ALA": 'A', "ARG": 'R', "ASN": 'N', "ASP": 'D', "CYS": 'C',
    "GLN": 'Q', "GLU": 'E', "GLY": 'G', "HIS": 'H', "ILE": 'I',
    "LEU": 'L', "LYS": 'K', "MET": 'M', "PHE": 'F', "PRO": 'P',
    "SER": 'S', "THR": 'T', "TRP": 'W', "TYR": 'Y', "VAL": 'V'
}

def extract_protein_sequence_from_pdb(q, seq_dic, chain='A'):
    """
    Worker function to extract one-letter sequence from a PDB file’s ATOM records.
    Each process reads (index, pdb_path) from `q`, builds the sequence for chain A by default,
    and stores it into seq_dic[pdb_path].
    """
    while True:
        item = q.get(block=True)
        if item is None:
            return
        i, pdb_file = item[0], item[1]
        # Print progress every power of 10
        if (math.log10(i + 1) % 1 == 0):
            print(f"[{i+1} PDBs processed.]")
        sequence = ""
        seen_res = set()
        with open(pdb_file, 'r') as fh:
            for line in fh:
                if line.startswith("ATOM"):
                    resname = line[17:20].strip()
                    chainID = line[21].strip()
                    resid   = line[22:26].strip()
                    if chainID == chain and resid not in seen_res:
                        seen_res.add(resid)
                        if resname in aa3to1:
                            sequence += aa3to1[resname]
        seq_dic[pdb_file] = sequence

def choose_ligand_smiles(basename, ligand_assignment, global_default=None):
    """
    Given a PDB basename (no path, no .pdb), pick the correct SMILES:
      - Iterate over ligand_assignment keys in insertion order.
      - If key is "default" and nothing else matches, use that.
      - Otherwise, if key is a substring of basename, return ligand_assignment[key].
      - If no key matches:
          • If global_default is provided, return that.
          • Otherwise, raise an error.
    """
    for key, smiles in ligand_assignment.items():
        if key != "default" and (key in basename):
            return smiles
    # If no other key matched, check for an explicit "default" entry
    if "default" in ligand_assignment:
        return ligand_assignment["default"]
    # If no "default" key, fallback to global_default if given
    if global_default is not None:
        return global_default
    raise ValueError(f"No ligand_smiles found for '{basename}' (no mapping, no default).")

###############################################
### REMARK 666 → PTM PARSING (AF3-COMPATIBLE) ###
###############################################

def parse_ptm_specs(specs_list):
    """
    Turn ['A/LYS/3:KCX', ...] into structured dicts.
    CHAIN/RES3/CATIDX:CCD
    """
    parsed = []
    for spec in specs_list:
        try:
            left, ccd = [s.strip() for s in spec.split(':', 1)]
            chain, res3, catidx = [s.strip() for s in left.split('/', 3)]
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
    Parse REMARK 666 MOTIF lines to get:
      chain, resname, resnum, cat_idx
    """
    catalog = []
    pat = re.compile(
        r"MOTIF\s+(\S+)\s+([A-Z]{3})\s+(\d+)([A-Z]?)\s+(\d+)\s+(\d+)",
        flags=re.IGNORECASE,
    )
    with open(pdb_path, 'r') as fh:
        for line in fh:
            if "REMARK 666" in line.upper() and "MOTIF" in line.upper():
                m = pat.search(line)
                if m:
                    catalog.append({
                        "chain":   m.group(1),
                        "resname": m.group(2).upper(),
                        "resnum":  int(m.group(3)),  # insertion code ignored
                        "cat_idx": int(m.group(5)),
                    })
    return catalog


def _resnum_to_seq_index_map(pdb_path, chain='A'):
    """
    Build mapping {PDB_resnum(int) -> 1-based sequence index}
    using the same parsing logic as sequence extraction (ATOM lines, unique residue IDs).
    """
    mapping = {}
    seen_res = set()
    seq_idx = 0
    with open(pdb_path, 'r') as fh:
        for line in fh:
            if not line.startswith("ATOM"): 
                continue
            resname = line[17:20].strip()
            chainID = line[21].strip()
            resid   = line[22:26].strip()  # insertion code ignored intentionally (consistent with extractor)
            if chainID != chain:
                continue
            if resid in seen_res:
                continue
            seen_res.add(resid)
            # only count standard 20 aa (consistent with extractor)
            # (if desired, relax this check)
            if resname in {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS",
                           "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}:
                seq_idx += 1
                try:
                    mapping[int(resid)] = seq_idx
                except ValueError:
                    # non-integer resid (rare); skip to avoid incorrect mapping
                    pass
    return mapping


def map_specs_to_seq_indices(pdb_path, specs):
    """
    Resolve CHAIN/RES3/CATIDX → (chain, seq_index, ccd) using REMARK 666 catalog + PDB->seq index map.
    Returns: { chain: [ (seq_index, ccd), ... ] }
    """
    catalog = parse_remark666_catalog(pdb_path)
    out = {}
    # For each chain that appears in specs, precompute its resnum→seq index map
    chain_to_map = {}

    for spec in specs:
        matches = [
            r for r in catalog
            if r["chain"] == spec["chain"]
            and r["resname"] == spec["resname"]
            and r["cat_idx"] == spec["cat_idx"]
        ]
        if not matches:
            raise ValueError(
                f"PTM target not found in REMARK 666 for {pdb_path}: "
                f"{spec['chain']}/{spec['resname']}/{spec['cat_idx']} -> {spec['ccd']}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous PTM target in {pdb_path}: "
                f"{spec['chain']}/{spec['resname']}/{spec['cat_idx']} matches residues "
                f"{[m['resnum'] for m in matches]}."
            )
        chain = spec["chain"]
        resnum = matches[0]["resnum"]
        if chain not in chain_to_map:
            chain_to_map[chain] = _resnum_to_seq_index_map(pdb_path, chain=chain)
        seq_idx = chain_to_map[chain].get(resnum, None)
        if not seq_idx:
            raise ValueError(
                f"Could not map PDB resnum {resnum} on chain {chain} to a sequence index "
                f"(check numbering/gaps)."
            )
        out.setdefault(chain, []).append((seq_idx, spec["ccd"]))
    return out


def apply_inline_ptms_to_sequence(seq, ptm_list):
    """
    Replace the residue at 1-based seq_idx with a parenthesized CCD token,
    e.g., 'K' -> '(KCX)' so the final sequence looks like AAA(KCX)AAA.

    ptm_list: list of (seq_idx, ccd)
    """
    tokens = list(seq)  # start from 1-letter tokens
    for seq_idx, ccd in sorted(ptm_list, key=lambda x: x[0]):
        i = seq_idx - 1
        if i < 0 or i >= len(tokens):
            raise ValueError(f"PTM index {seq_idx} out of bounds for sequence of length {len(tokens)}.")
        tokens[i] = f"({ccd})"  # <-- parenthesized insertion for Chai-1
    return "".join(tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Chai-1 input commands from PDB or FASTA input, with optional per-input ligand assignment.")
    parser.add_argument("--pdb_path", help="Directory of input PDB files (mutually exclusive with --fasta_path).")
    parser.add_argument("--fasta_path", help="Path to input FASTA file (mutually exclusive with --pdb_path).")
    parser.add_argument("--output_path", required=True, help="Directory where Chai-1 outputs will be written.")
  #  parser.add_argument("--ligand_smiles", type=str,
   #                     help="JSON string defining one ligand SMILES pair (e.g. '[\"[Zn+2]\",\"CC(...)\"]'). "
    #                         "Used for all inputs if no per-PDB mapping provided.")
    parser.add_argument("--ligand_smiles", type=str,
        help="JSON string defining one or more ligand SMILES tokens (e.g. '[\"[Zn+2]\",\"CC(...)\"]'). "
             "Used for all inputs if no per-PDB mapping provided.")
    parser.add_argument("--ligand_assignment_json", type=str,
                        help="JSON object mapping substrings to SMILES lists. Example:\n"
                             "  '{\"group2\": [\"[Zn+2]\",\"SMILES2\"], \"group3\": [\"[Zn+2]\",\"SMILES3\"], \"default\": [\"[Zn+2]\",\"SMILES_def\"]}'\n"
                             "If provided, overrides --ligand_smiles. Keys are matched as substrings of each PDB’s basename.")
    parser.add_argument("--command_path", required=True, help="File where generated Chai-1 commands will be saved.")
    parser.add_argument("--chai_shell_script", default="/net/software/lab/chai/chai-lab/run_chai.sh",
                        help="Path to the Chai-1 shell script to invoke.")
    parser.add_argument("--check_made_output", action="store_true", default=False,
                        help="If set, skip any PDB/FASTA entry whose Chai-1 output already exists.")
    parser.add_argument("--n_terminus_tag", type=str, default="", help="Optional N-terminal tag to prepend.")
    parser.add_argument("--c_terminus_tag", type=str, default="", help="Optional C-terminal tag to append.")
    parser.add_argument("--do_not_make_separate_subdirectories", action="store_true", default=False,
                        help="If set, write all outputs directly into --output_path (no per-input subfolder).")
    parser.add_argument("--ptm_from_remark666", type=str, nargs="*", default=[],
                        help=("Optional PTM specs derived from REMARK 666. "
                              "Format: CHAIN/RES3/CATIDX:CCD (e.g., 'A/LYS/3:KCX'). "
                              "You can pass multiple entries separated by spaces."))
    args = parser.parse_args()

    if args.ptm_from_remark666 and not args.pdb_path:
        raise ValueError("--ptm_from_remark666 requires --pdb_path input (REMARK 666 must be read from PDBs).")

    # Verify mutual exclusivity
    if bool(args.pdb_path) and bool(args.fasta_path):
        raise ValueError("Cannot specify both --pdb_path and --fasta_path; choose one.")
    if not bool(args.pdb_path) and not bool(args.fasta_path):
        raise ValueError("Must specify either --pdb_path or --fasta_path.")

    output_dir = args.output_path
    commands_file = args.command_path

    # --- Build ligand_assignment dict (substring → SMILES list) ---
    ligand_assignment = None
    global_smiles = None
    if args.ligand_assignment_json:
        raw_map = json.loads(args.ligand_assignment_json)
        ligand_assignment = {}
        for k, v in raw_map.items():
            # Allow v to be any list of SMILES (no length restriction)
            #if not isinstance(v, list):
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise ValueError(f"Value for key '{k}' must be a list of SMILES strings, got: {v}")
            smiles_list = v[:]  # keep all entries

            if "|" in k:
                left, right = k.split("|", 1)
                prefix = None if left == "None" else left
                substring = None if right == "None" else right
                ligand_assignment[(prefix, substring)] = smiles_list
            else:
                # treat the entire string k as a prefix‐only rule (substring = None)
                prefix = k
                substring = None
                ligand_assignment[(prefix, substring)] = smiles_list

 #   elif args.ligand_smiles:
  #      global_smiles = json.loads(args.ligand_smiles)
   #     if not isinstance(global_smiles, list) or len(global_smiles) != 2:
    #        raise ValueError("--ligand_smiles must be a JSON list of length 2, e.g. '[\"[Zn+2]\",\"SMILES\"]'")
    elif args.ligand_smiles:
        global_smiles = json.loads(args.ligand_smiles)
        if not isinstance(global_smiles, list) or not all(isinstance(x, str) for x in global_smiles):
            raise ValueError("--ligand_smiles must be a JSON list of strings, e.g. '[\"[Zn+2]\",\"SMILES\"]'")
    else:
        raise ValueError("Must provide either --ligand_smiles or --ligand_assignment_json.")

    # --- Prepare multiprocessing for PDB → sequence extraction ---
    the_queue = multiprocessing.Queue()
    manager = multiprocessing.Manager()
    seq_dic = manager.dict()
    pdb_files = []

    # --- 1) If PDB mode, gather all .pdb files and run parallel sequence extraction ---
    if args.pdb_path:
        input_dir = args.pdb_path
        pdb_files = sorted(glob.glob(os.path.join(input_dir, "*.pdb")))
        if args.check_made_output:
            retained = []
            for fi in pdb_files:
                base = os.path.splitext(os.path.basename(fi))[0]
                out_pdb = os.path.join(output_dir, base, f"pred.{base}_model_idx_4.pdb")
                if not os.path.isfile(out_pdb):
                    retained.append(fi)
            pdb_files = retained
            print(f"{len(pdb_files)} PDBs found to run Chai-1 (after --check_made_output).")
        # Enqueue jobs
        for idx, pdb_path in enumerate(pdb_files):
            the_queue.put((idx, pdb_path))
        # Launch worker pool
        pool = multiprocessing.Pool(os.cpu_count() - 1,
                                    initializer=extract_protein_sequence_from_pdb,
                                    initargs=(the_queue, seq_dic))
        # Send termination signals
        for _ in range(os.cpu_count()):
            the_queue.put(None)
        the_queue.close()
        the_queue.join_thread()
        pool.close()
        pool.join()
    # --- 2) If FASTA mode, read sequences into seq_dic; pdb_files key = sequence IDs ---
    else:
        with open(args.fasta_path, 'r') as fh:
            data = fh.read().strip().split(">")
        seq_dic = {}
        for entry in data:
            if not entry:
                continue
            parts = entry.split()
            seq_id = parts[0]
            seq_body = "".join(parts[1:])
            seq_dic[seq_id] = seq_body
        if args.check_made_output:
            retained = []
            for seq_id in seq_dic.keys():
                out_pdb = os.path.join(output_dir, seq_id, f"pred.{seq_id}_model_idx_4.pdb")
                if not os.path.isfile(out_pdb):
                    retained.append(seq_id)
            pdb_files = retained
            print(f"{len(pdb_files)} sequences found to run Chai-1 (after --check_made_output).")
        else:
            pdb_files = list(seq_dic.keys())

    # --- 3) Write commands to the command file, tracking counts and unmatched ---
    counts = {k: 0 for k in (ligand_assignment or {})}
    if ligand_assignment and "default" in ligand_assignment:
        counts["default"] = 0
    unmatched = []

    cmd_count = 0
    with open(commands_file, 'w') as cf:
        for entry in pdb_files:
            if args.pdb_path:
                # entry is a full path to a .pdb file
                pdb_path = entry
                pdb_basename = os.path.splitext(os.path.basename(pdb_path))[0]
                seq_key = pdb_path
            else:
                # FASTA mode: entry is the sequence ID
                pdb_basename = entry
                seq_key = entry

            # Determine where to write this input’s Chai-1 output
            if args.do_not_make_separate_subdirectories:
                outdir = output_dir
            else:
                outdir = os.path.join(output_dir, pdb_basename)
                if not os.path.exists(outdir):
                    os.makedirs(outdir, exist_ok=True)

            # --- Build the final protein sequence with optional REMARK666-driven PTMs ---
            base_seq = seq_dic[seq_key]

            if args.ptm_from_remark666:
                # 1) parse specs (same syntax as AF3)
                _ptm_specs = parse_ptm_specs(args.ptm_from_remark666)

                # 2) resolve to sequence indices for this PDB (chain-aware)
                #    NOTE: we only have a single-chain sequence extracted (default 'A');
                #    specs for other chains will be ignored unless you also extract them.
                ptm_map = map_specs_to_seq_indices(pdb_path if args.pdb_path else "", _ptm_specs)

                # 3) apply PTMs that target the chain we extracted (default 'A')
                #    If you want a different chain, change the 'A' below or make it an arg.
                chain_key = 'A'
                if chain_key in ptm_map:
                    base_seq = apply_inline_ptms_to_sequence(base_seq, ptm_map[chain_key])

            # 4) prepend/append tags *after* PTM substitution (PTMs refer to untagged positions)
            protein_seq = f"{args.n_terminus_tag}{base_seq}{args.c_terminus_tag}"

            # Choose ligand_smiles for this input
            if ligand_assignment:
                used_key = None
                for (prefix, substring), smiles_pair in ligand_assignment.items():
                    prefix_matches = (prefix is None) or (prefix in pdb_basename)
                    substring_matches = (substring is None) or (substring in pdb_basename)
                    if prefix_matches and substring_matches:
                        used_key = (prefix, substring)
                        break

                if used_key is None:
                    # no (prefix,substring) matched → check if there is a default entry
                    if ("default", None) in ligand_assignment:
                        used_key = ("default", None)
                    else:
                        unmatched.append(pdb_basename)
                        ligand_smiles = ["", ""]
                        used_key = None

                if used_key is not None:
                    ligand_smiles = ligand_assignment[used_key]
                    counts[used_key] += 1
            else:
                ligand_smiles = global_smiles

            # --- NEW: serialize ligands as JSON only if not empty ---
            if ligand_smiles:
                ligand_smiles_json = json.dumps(ligand_smiles)
                ligand_flag = f"--ligand '{ligand_smiles_json}' "
            else:
                ligand_flag = ""   # no ligand option if list is empty

            # --- Generate the Chai-1 command line ---
            cmd_line = (
                f"{args.chai_shell_script} "
                f"--name {pdb_basename} "
                f"--protein \"['{protein_seq}']\" "
                f"{ligand_flag}"   # <-- conditionally included
                f"--output_dir {outdir} "
                f"--export_arrays --export_mode json\n"
            )

            # Generate the Chai-1 command line
#            cmd_line = (
 #               f"{args.chai_shell_script} "
  #              f"--name {pdb_basename} "
   #             f"--protein \"['{protein_seq}']\" "
    #            f"--ligand \"{ligand_smiles}\" "
     #           f"--output_dir {outdir} "
      #          f"--export_arrays --export_mode json\n"
       #     )
            cf.write(cmd_line)
            cmd_count += 1

    # ── sort the commands file alphanumerically ──
    with open(commands_file, "r") as f:
        lines = f.readlines()
    lines.sort()  # in‐place alphanumeric sort
    with open(commands_file, "w") as f:
        f.writelines(lines)

    # ── then print your final summary ──
    print()
    print(f"[ACTION] Sorting commands in alphanumerical order")

    # Print summary counts
    print()
    print(f"[DONE] Wrote {cmd_count} commands to: {commands_file}")
    if ligand_assignment:
        print("\nLigand‐assignment counts:")
        for key, cnt in counts.items():
            print(f"  {key!r}: {cnt} inputs matched")
        if unmatched:
            print(f"\nWARNING: {len(unmatched)} files did not match any ligand_assignment key and used an empty ligand:")
            for name in unmatched:
                print(f"  • {name}")
