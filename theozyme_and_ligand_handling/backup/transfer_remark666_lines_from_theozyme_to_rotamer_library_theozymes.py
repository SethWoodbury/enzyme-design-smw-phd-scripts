"""
Author: Seth Woodbury
Date: 2024-07-15
Description: This script transfers REMARK 666 and HEADER lines from a theozyme PDB file to multiple rotamer PDB files.
It loads the theozyme PDB file, extracts the specified lines, and inserts them at the top of the rotamer PDB files.

Input Arguments:
  --theozyme_pdb_with_remark666_lines: Path to the theozyme PDB file containing REMARK 666 lines.
  --pdbs_without_remark666_lines: Pattern to match rotamer PDB files without REMARK 666 lines.

Output: Rotamer PDB files with inserted REMARK 666 lines saved in their original location.

EXAMPLE COMMAND:
singularity exec /software/containers/crispy.sif python /home/woodbuse/special_scripts/theozyme_and_ligand_handling/transfer_remark666_lines_from_theozyme_to_rotamer_library_theozymes.py --theozyme_pdb_with_remark666_lines /home/woodbuse/for/don_hilvert/zinc_amidase_proj/theozymes/fluorogenic_amide_SM/cpa/theozymes_r_s/test/RFA_0001_theozyme.pdb --pdbs_without_remark666_lines "/home/woodbuse/for/don_hilvert/zinc_amidase_proj/theozymes/fluorogenic_amide_SM/cpa/theozymes_r_s/test/*RFA*.pdb"
"""

import glob
import argparse
import os

def transfer_remark666_lines(theozyme_pdb_with_remark666_lines, pdbs_without_remark666_lines):
    """
    Transfer REMARK 666 and HEADER lines from a theozyme PDB file to multiple rotamer PDB files.

    Args:
    - theozyme_pdb_with_remark666_lines: Path to the theozyme PDB file containing REMARK 666 lines.
    - pdbs_without_remark666_lines: Pattern to match rotamer PDB files without REMARK 666 lines.

    Output: Modified rotamer PDB files with inserted REMARK 666 lines.
    """
    # Get the list of PDB files
    theozyme_files = glob.glob(theozyme_pdb_with_remark666_lines)
    rotamer_files = glob.glob(pdbs_without_remark666_lines)

    # Ensure the theozyme file is not in the list of rotamer files
    for theozyme_file in theozyme_files:
        if theozyme_file in rotamer_files:
            rotamer_files.remove(theozyme_file)

    print('')
    print('Theozyme PDB files with REMARK 666 lines:')
    for file in theozyme_files:
        print(file)
    print('')
    
    print('PDB files without REMARK 666 lines:')
    for file in rotamer_files:
        print(file)
    print('')

    for theozyme_file in theozyme_files:
        with open(theozyme_file, 'r') as f:
            lines = f.readlines()
        
        remark666_lines = [line for line in lines if line.startswith('REMARK 666') or line.startswith('HEADER')]
        
        if not remark666_lines:
            print(f"No REMARK 666 or HEADER lines found in {theozyme_file}")
            continue

        print(f'Found REMARK 666 or HEADER lines in {theozyme_file}:')
        for line in remark666_lines:
            print(line.strip())

        for rotamer_file in rotamer_files:
            with open(rotamer_file, 'r') as f:
                rotamer_lines = f.readlines()

            with open(rotamer_file, 'w') as f:
                f.writelines(remark666_lines + rotamer_lines)

            print('')
            print(f'Inserted REMARK 666 lines into {rotamer_file}')
            print('')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer REMARK 666 lines from one PDB to multiple PDBs')
    parser.add_argument('--theozyme_pdb_with_remark666_lines', type=str, required=True, help='Theozyme PDB file with REMARK 666 lines')
    parser.add_argument('--pdbs_without_remark666_lines', type=str, required=True, help='Pattern to match PDB files without REMARK 666 lines')
    args = parser.parse_args()

    transfer_remark666_lines(args.theozyme_pdb_with_remark666_lines, args.pdbs_without_remark666_lines)
