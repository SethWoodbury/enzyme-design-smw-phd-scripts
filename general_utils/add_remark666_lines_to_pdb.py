"""
Created 2024-04-24 by Seth Woodbury (woodbuse@uw.edu)
This script maps and copies remark 666 + header lines from pdbs in a reference directory to a new directory.

# IMPORTANT NOTE: This script requires that the new pdbs have an added suffix that utilizes the old name (forward mapping).
# This cannot map backwards yet, as it is meant to propagate REMARK 666 + other important lines from older PDBs forward.
# Enjoy, and please contact me if you'd like to add anything as I want to keep it as general as possible.

# Example command using the --find_suffixes_from_random_sample flag:
# python your_script_name.py \
#   --new_pdb_dir "/path/to/new_pdbs" \
#   --reference_old_pdb_dir "/path/to/old_pdbs" \
#   --debug \
#   --find_suffixes_from_random_sample 30 \
#   --remove_unwanted_lines_from_new_pdbs "REMARK SomeUnwantedLine" "REMARK AnotherUnwantedLine" \
#   --additional_lines_from_old_pdbs_to_copy "DATE" "DIG" \
#   --remove_unwanted_lines_from_new_pdbs_that_STARTwith "REMARK 666" "HEADER" \
#   --clobber_existing_remark_header_lines
#
# The above command would:
#   1) Randomly select 30 old PDB files to identify suffixes.
#   2) Remove any lines in new PDBs that exactly match "REMARK SomeUnwantedLine" or "REMARK AnotherUnwantedLine".
#   3) Then remove any lines that *start with* "REMARK 666" or "HEADER".
#   4) Then, because --clobber_existing_remark_header_lines is specified, remove *all* lines that start with
#      "HEADER", "REMARK", "HETNAM", or "LINK" in the new PDB files.
#   5) Finally, copy the relevant lines from old PDBs to the new PDBs.
"""

import os
import glob
import multiprocessing
import time
from pathlib import Path
import argparse
import random

def collect_headers(old_file, additional_lines_prefixes):
    """
    Collects header lines from the old PDB file that start with specific prefixes.
    Optionally includes additional prefixes provided by the user.
    
    Parameters:
        old_file (str): Path to the old PDB file.
        additional_lines_prefixes (list): List of additional prefixes to include in headers.
    
    Returns:
        list: A list of header lines to be added to new PDB files.
    """
    headers_to_add = []
    seen_headers = set()
    with open(old_file, 'r') as file:
        for line in file:
            # We always look for these four plus any provided in additional_lines_prefixes
            if line.startswith(tuple(["HEADER", "REMARK", "HETNAM", "LINK"] + additional_lines_prefixes)):
                if line not in seen_headers:
                    headers_to_add.append(line)
                    seen_headers.add(line)
    return headers_to_add

def copy_remark_lines(
    old_file,
    suffixes,
    new_pdbs_dir,
    headers_to_add,
    unwanted_lines_exact,
    unwanted_lines_startwith,
    clobber_remark_header_lines
):
    """
    Copies necessary header lines from old PDB files to new PDB files, optionally removing lines
    based on exact matches, lines that start with certain prefixes, and/or clobbering all existing
    HEADER/REMARK/HETNAM/LINK lines if requested.
    
    Parameters:
        old_file (str): Path to the old PDB file.
        suffixes (dict): Dictionary mapping old file bases to new file suffixes.
        new_pdbs_dir (str): Directory containing new PDB files.
        headers_to_add (list): List of headers to add to the new files.
        unwanted_lines_exact (list): Lines to remove from new files if they exactly match.
        unwanted_lines_startwith (list or None): If provided, lines in new files that start with any
            of these prefixes will be removed.
        clobber_remark_header_lines (bool): If True, remove lines starting with "HEADER", "REMARK", 
            "HETNAM", or "LINK".
    
    Returns:
        list: A list of tuples with details on modifications for each file (file_path, was_modified, result_message).
    """
    old_base = Path(old_file).stem
    detailed_results = []
    
    for suffix in suffixes.get(old_base, []):
        new_file_path = Path(new_pdbs_dir) / f"{old_base}{suffix}.pdb"
        was_modified = False
        
        if new_file_path.exists():
            with open(new_file_path, 'r+') as new_file:
                # --------------------------
                # Step 1: Read existing lines
                # --------------------------
                content = new_file.readlines()

                # ------------------------------------------------
                # Step 2a: Remove lines that exactly match unwanted_lines_exact
                # ------------------------------------------------
                stripped_unwanted_exact = [ul.strip() for ul in unwanted_lines_exact]
                content = [
                    line
                    for line in content
                    if line.strip() not in stripped_unwanted_exact
                ]

                # ------------------------------------------------
                # Step 2b: Remove lines that start with any prefix
                #          in unwanted_lines_startwith (if provided)
                # ------------------------------------------------
                if unwanted_lines_startwith:
                    content = [
                        line for line in content
                        if not any(line.startswith(prefix) for prefix in unwanted_lines_startwith)
                    ]

                # ------------------------------------------------------
                # Step 2c: Clobber all lines that start with HEADER,
                #          REMARK, HETNAM, or LINK if the user requests
                # ------------------------------------------------------
                if clobber_remark_header_lines:
                    content = [
                        line for line in content
                        if not line.startswith(("HEADER", "REMARK", "HETNAM", "LINK"))
                    ]

                # ------------------------------------------------
                # Step 3: Now determine which lines from old file
                #         need to be inserted to avoid duplication
                # ------------------------------------------------
                content_set = set(content)
                headers_needed = [header for header in headers_to_add if header not in content_set]

                # We still deduplicate 'headers_needed' in case of duplicates within old_file
                unique_headers = []
                seen = set()
                for header in headers_needed:
                    if header not in seen:
                        seen.add(header)
                        unique_headers.append(header)

                # ------------------------------------------------
                # Step 4: Write back changes if new headers are needed
                # ------------------------------------------------
                if unique_headers:
                    new_file.seek(0, 0)
                    new_file.truncate()  # Clear file before writing updated content
                    new_file.writelines(unique_headers + content)
                    was_modified = True
                else:
                    # If we removed lines but have no lines to add, we should still check
                    # if the content changed relative to the original. So let's see if
                    # we actually changed the file in any way:
                    new_file.seek(0)
                    final_content = new_file.read().splitlines(keepends=True)
                    # Compare final_content to content. If different, it means we've removed something.
                    # But typically readlines -> writing -> reading again can differ in subtle ways
                    # (like newline endings). Instead, let's do a simpler approach: 
                    # We can do a "did we remove lines?" check:
                    new_file.seek(0, 0)
                    new_file.truncate()
                    new_file.writelines(content)
                    was_modified = True

            # --------------------------
            # Prepare log/return message
            # --------------------------
            if was_modified:
                if unique_headers:
                    msg = f"Headers copied from {old_file} to {new_file_path}:\n{''.join(unique_headers)}"
                else:
                    msg = f"File {new_file_path} modified (some lines removed or truncated, but no new headers were needed)."
            else:
                msg = f"No new headers needed copying for {new_file_path}."
                
            detailed_results.append((new_file_path, was_modified, msg))
        else:
            detailed_results.append((new_file_path, False, f"File not found: {new_file_path}"))
    
    return detailed_results

def add_remark666_lines_to_pdb_files(
    new_pdbs_dir,
    reference_old_pdbs_dir,
    debug,
    remove_unwanted_lines,
    additional_lines_prefixes,
    sample_size=None,
    remove_unwanted_lines_startwith=None,
    clobber_remark_header_lines=False
):
    """
    Main function to process PDB files, copying specific header lines from old to new PDBs, and
    optionally removing unwanted lines based on:
      1) Exact matches (remove_unwanted_lines)
      2) Lines that start with certain prefixes (remove_unwanted_lines_startwith)
      3) Full clobbering of 'HEADER', 'REMARK', 'HETNAM', 'LINK' lines if desired.
    
    Parameters:
        new_pdbs_dir (str): Directory containing new PDB files.
        reference_old_pdbs_dir (str): Directory containing old reference PDB files.
        debug (bool): Flag to enable detailed debug output.
        remove_unwanted_lines (list): List of lines to remove from new PDB files (exact match).
        additional_lines_prefixes (list): Additional line prefixes from old PDBs to copy to new PDB headers.
        sample_size (int or None): Optional number of old files to sample for suffix identification.
        remove_unwanted_lines_startwith (list or None): Optional list of prefixes. Lines in new PDBs that start
            with any of these prefixes will be removed.
        clobber_remark_header_lines (bool): If True, remove lines that start with 'HEADER', 'REMARK',
            'HETNAM', or 'LINK' in each new PDB before adding new lines from the old PDB.
    """
    print("\nStarting the processing of PDB files...")
    old_pdb_files = glob.glob(os.path.join(reference_old_pdbs_dir, '*.pdb'))
    new_pdb_files = glob.glob(os.path.join(new_pdbs_dir, '*.pdb'))
    print(f"Found {len(old_pdb_files)} old PDB files and {len(new_pdb_files)} new PDB files.\n")

    start_time = time.time()
    all_suffixes, cumulative_suffixes = identify_suffixes(old_pdb_files, new_pdb_files, debug, sample_size)
    
    num_cpus = multiprocessing.cpu_count()
    print(f"\nUsing {num_cpus} CPUs for parallel processing...")

    # Prepare arguments for the worker function in parallel
    work_args = []
    for old_file in old_pdb_files:
        headers_for_old = collect_headers(old_file, additional_lines_prefixes)
        work_args.append((
            old_file,
            all_suffixes,
            new_pdbs_dir,
            headers_for_old,
            remove_unwanted_lines,          # exact-match lines
            remove_unwanted_lines_startwith,
            clobber_remark_header_lines
        ))
    
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = pool.starmap(copy_remark_lines, work_args)

    # Flatten the results list of lists
    flattened_results = [item for sublist in results for item in sublist]

    total_modified = sum(1 for (file_path, was_modified, _) in flattened_results if was_modified)
    unmodified_files = [(fp, msg) for (fp, wm, msg) in flattened_results if not wm]

    # Reporting on file modifications
    print(f"\nTotal modified new PDB files: {total_modified}/{len(new_pdb_files)}")
    if total_modified == 0:
        print("No files were modified.")

    # Reporting on unmodified files
    if unmodified_files:
        print("\nUnmodified files:")
        for file, reason in unmodified_files:
            print(f"{file}")
            print(f"REASON: {reason}")
            print("")

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds.")
    print(f"Cumulative unique suffixes found across all samples: {sorted(cumulative_suffixes)}")
    print(f"Number of cumulative unique suffixes: {len(cumulative_suffixes)}")
    print(f"\nTotal modified new PDB files: {total_modified}/{len(new_pdb_files)}")

def identify_suffixes(old_files, new_files, debug, sample_size=None):
    """
    Identifies suffixes added to new PDB files based on old PDB file names. Optionally uses a
    random sample of old files for suffix identification.
    
    Parameters:
        old_files (list): List of old PDB file paths.
        new_files (list): List of new PDB file paths.
        debug (bool): Flag to enable debug output.
        sample_size (int): Optional size of random sample of old files to use for suffix identification.
    
    Returns:
        (dict, set):
           1) A dictionary where each old file's base name maps to the list of all unique suffixes found.
           2) A set of all unique suffixes identified across sampled files.
    """
    if sample_size is not None and sample_size < len(old_files):
        sampled_files = random.sample(old_files, sample_size)
        if debug:
            print(f"\nUsing a random sample of {sample_size} old files for suffix identification.")
    else:
        sampled_files = old_files
    
    start_time = time.time()
    all_suffixes = {}
    cumulative_suffixes = set()
    
    # First, discover suffixes from the chosen subset of old files
    for old_file in sampled_files:
        old_base = Path(old_file).stem
        suffixes_found = []
        for new_file in new_files:
            new_base = Path(new_file).stem
            if new_base.startswith(old_base):
                suffix = new_base[len(old_base):]
                suffixes_found.append(suffix)
                cumulative_suffixes.add(suffix)
        all_suffixes[old_base] = suffixes_found
        if debug:
            print(f"Old file base '{old_base}' mapped with suffixes: {suffixes_found}")
    
    # Next, map every old file base (not just the sample) to the set of all discovered suffixes
    for old_file in old_files:
        old_base = Path(old_file).stem
        all_suffixes[old_base] = list(cumulative_suffixes)

    print(f"\nSuffix identification completed in {time.time() - start_time:.2f} seconds.")
    return all_suffixes, cumulative_suffixes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy specific lines from old PDBs to new PDBs, optionally removing or clobbering lines in the new PDBs beforehand.")

    parser.add_argument(
        '--new_pdb_dir',
        type=str,
        required=True,
        help='Directory with new PDB files.'
    )
    parser.add_argument(
        '--reference_old_pdb_dir',
        type=str,
        required=True,
        help='Directory with reference old PDB files.'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable verbose debugging output.'
    )
    parser.add_argument(
        '--remove_unwanted_lines_from_new_pdbs',
        nargs='*',
        default=["REMARK AtomGroup Unnamed + Unnamed"],
        help='List of unwanted lines to remove from new PDB files by exact match. '
             'Example usage: --remove_unwanted_lines_from_new_pdbs "BAD LINE" "TERRIBLE LINE". '
             'Default is ["REMARK AtomGroup Unnamed + Unnamed"].'
    )
    parser.add_argument(
        '--additional_lines_from_old_pdbs_to_copy',
        nargs='*',
        default=[],
        help='Additional line prefixes from old PDBs to copy to new PDB headers. '
             'Example usage: --additional_lines_from_old_pdbs_to_copy "DATE" "DIG"'
    )
    parser.add_argument(
        '--find_suffixes_from_random_sample',
        type=int,
        help='Optionally find suffixes from a random sample of old PDB files. Provide an integer to specify sample size.'
    )
    # ---------------- NEW FLAG #1 ----------------
    parser.add_argument(
        '--remove_unwanted_lines_from_new_pdbs_that_STARTwith',
        nargs='*',
        default=None,
        help='Optional list of prefixes to remove from new PDB files if a line starts with one of these. '
             'For example: --remove_unwanted_lines_from_new_pdbs_that_STARTwith "REMARK 666" "HEADER". '
             'These lines are removed after the exact-match removal but before any new lines are added.'
    )
    # ---------------- NEW FLAG #2 ----------------
    parser.add_argument(
        '--clobber_existing_remark_header_lines',
        action='store_true',
        help='If specified, remove lines starting with "HEADER", "REMARK", "HETNAM", or "LINK" in the new PDB '
             'files after the other removals but before adding lines from the old PDB. This effectively '
             'overwrites existing remarks/headers in the new PDB.'
    )

    args = parser.parse_args()

    add_remark666_lines_to_pdb_files(
        new_pdbs_dir=args.new_pdb_dir,
        reference_old_pdbs_dir=args.reference_old_pdb_dir,
        debug=args.debug,
        remove_unwanted_lines=args.remove_unwanted_lines_from_new_pdbs,
        additional_lines_prefixes=args.additional_lines_from_old_pdbs_to_copy,
        sample_size=args.find_suffixes_from_random_sample,
        remove_unwanted_lines_startwith=args.remove_unwanted_lines_from_new_pdbs_that_STARTwith,
        clobber_remark_header_lines=args.clobber_existing_remark_header_lines
    )
