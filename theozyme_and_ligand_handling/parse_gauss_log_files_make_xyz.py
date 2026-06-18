"""
Author: Seth M. Woodbury
Date: 2024-11-25
Description: This script processes Gaussian log files to identify transition states, verify frequencies, convert files to XYZ format using Open Babel, and optionally reorder atoms in the XYZ output based on a specified list of ranges. It creates an organized output for further theozyme analysis.

Example Command:
python script_name.py /path/to/log/files --reorder_ligand_atoms_first 1-18,43-43,51-71

Options:
- `input_dir` (required): Directory containing Gaussian .log files.
- `--reorder_ligand_atoms_first` (optional): Comma-separated list of number ranges to reorder ligand atoms first in the XYZ output.
- `--ignore_TS_warning` (optional): If set, process files with >1 imaginary frequency, with warnings.
"""

import os
import re
import subprocess

# --- locate repo root + shared external paths ---
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "repo_paths.py").is_file():
        _sys.path.insert(0, str(_anc)); break
import repo_paths

def parse_gaussian_log(log_file):
    """
    Parse a Gaussian log file to verify transition state and extract frequencies.

    Args:
        log_file (str): Path to the Gaussian log file.

    Returns:
        dict: Contains 'is_valid_ts' (bool) and 'imaginary_freqs' (list).
    """
    frequencies = []

    with open(log_file, 'r') as file:
        for line in file:
            # Extract frequencies
            if "Frequencies" in line:
                freqs = [float(f) for f in re.findall(r"-?\d+\.\d+", line)]
                frequencies.extend(freqs)

    # Validate transition state
    imaginary_freqs = [f for f in frequencies if f < 0]
    is_valid_ts = len(imaginary_freqs) == 1

    return {
        "is_valid_ts": is_valid_ts,
        "imaginary_freqs": imaginary_freqs
    }

def run_openbabel(log_file, output_file):
    """
    Convert Gaussian log file to XYZ format using Open Babel.

    Args:
        log_file (str): Path to the Gaussian log file.
        output_file (str): Path to the output XYZ file.

    Returns:
        bool: True if conversion is successful, False otherwise.
    """
    obabel_path = repo_paths.OBABEL
    try:
        subprocess.run([obabel_path, "-ig09", log_file, "-oxyz", "-O", output_file], check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def reorder_atoms(xyz_file, atom_ranges):
    """
    Reorder atoms in an XYZ file so that specified ranges come first.

    Args:
        xyz_file (str): Path to the input XYZ file.
        atom_ranges (list): List of atom number ranges to reorder first.
    """
    with open(xyz_file, 'r') as file:
        lines = file.readlines()

    header = lines[:2]  # First two lines are XYZ header
    atoms = lines[2:]   # Remaining lines are atomic coordinates

    # Parse ranges into a set of atom indices
    ranges = []
    for r in atom_ranges:
        start, end = map(int, r.split('-'))
        ranges.extend(range(start, end + 1))

    ligand_atoms = [atoms[i - 1] for i in ranges if 1 <= i <= len(atoms)]
    other_atoms = [atom for i, atom in enumerate(atoms, start=1) if i not in ranges]

    # Write reordered XYZ
    with open(xyz_file, 'w') as file:
        file.write(header[0])
        file.write(header[1])
        file.writelines(ligand_atoms + other_atoms)

    # Print ligand atom range
    if ligand_atoms:
        print(f"### LIGAND ATOM RANGE: 1 - {len(ligand_atoms)} ###")

def process_log_files(input_dir, atom_reorder_ranges, ignore_TS_warning=False):
    """
    Process all Gaussian log files in the specified directory.

    Args:
        input_dir (str): Path to the directory containing Gaussian log files.
        atom_reorder_ranges (list): Atom ranges for reordering in the XYZ output.
        ignore_TS_warning (bool): Whether to process files with >1 imaginary frequency.
    """
    output_dir = os.path.join(input_dir, "xyz_from_parsed_logs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_files = [f for f in os.listdir(input_dir) if f.endswith(".log")]
    if not log_files:
        print("No .log files found in the specified directory.")
        return

    for log_file in log_files:
        log_path = os.path.join(input_dir, log_file)
        print(f"### Parsing {log_file} ###")

        result = parse_gaussian_log(log_path)
        imaginary_freqs = result["imaginary_freqs"]

        proceed = result["is_valid_ts"]
        proceed_due_to_flag = False

        if len(imaginary_freqs) > 1:
            print("### WARNING: >1 IMAGINARY FREQUENCY DETECTED, PROCEED WITH CAUTION & VERIFY THAT THE OTHERS HAVE LOW MAGNITUDES ###")
            # Check if any imaginary freq is "small" (<10 cm-1 in magnitude)
            has_small_imag = any(abs(freq) < 10 for freq in imaginary_freqs)
            if has_small_imag:
                print("### KEEP CALM: At least one imaginary frequency has magnitude < 10. (Usually numerical artifact) ###")
            if ignore_TS_warning:
                print("### --ignore_TS_warning FLAG IS SET: WILL CONTINUE PROCESSING THIS FILE ###")
                proceed = True
                proceed_due_to_flag = True

        if proceed:
            if len(imaginary_freqs) == 0:
                freq_str = "none found"
            else:
                freq_str = ", ".join([f"{f:.2f}" for f in imaginary_freqs])
            if result["is_valid_ts"]:
                print(f"### 1 NEGATIVE FREQUENCY CONFIRMED: {freq_str} ###")
            elif proceed_due_to_flag:
                print(f"### FILE HAS >1 IMAGINARY FREQUENCY, BUT --ignore_TS_warning WAS USED. FREQUENCIES: {freq_str} ###")

            base_name = os.path.splitext(log_file)[0]
            xyz_path = os.path.join(output_dir, f"{base_name}.xyz")
            success = run_openbabel(log_path, xyz_path)
            if success:
                print(f"### XYZ FILE CREATED: {xyz_path} ###")

                if atom_reorder_ranges:
                    reorder_atoms(xyz_path, atom_reorder_ranges)
                    print(f"### REORDERED XYZ FILE OVERWRITTEN: {xyz_path} ###\n")
                else:
                    print(f"### NO REORDERING SPECIFIED ###\n")
            else:
                print(f"### FAILED TO CREATE XYZ FILE FOR {log_file} ###\n")
        else:
            print(f"### SKIPPED {log_file}: NOT A VALID TRANSITION STATE ###\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process Gaussian log files to extract XYZ geometries.")
    parser.add_argument("input_dir", type=str, help="Directory containing Gaussian .log files.")
    parser.add_argument("--reorder_ligand_atoms_first", type=str, default=None, help="Comma-separated list of atom ranges to reorder ligand atoms first in the XYZ output (e.g., '1-18,43-43,51-71').")
    parser.add_argument("--ignore_TS_warning", action="store_true", help="If set, process files with >1 imaginary frequency, with warnings.")
    args = parser.parse_args()

    atom_reorder_ranges = args.reorder_ligand_atoms_first.split(",") if args.reorder_ligand_atoms_first else None

    process_log_files(args.input_dir, atom_reorder_ranges, ignore_TS_warning=args.ignore_TS_warning)
