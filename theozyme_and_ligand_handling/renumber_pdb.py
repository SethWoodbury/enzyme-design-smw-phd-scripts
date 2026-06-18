#!/usr/bin/env python3

from collections import defaultdict

def parse_pdb(input_pdb):
    """Parse the PDB file, separating ATOM, CONECT, and MASTER lines."""
    atoms = []
    conects = []
    master_line = ""
    
    with open(input_pdb, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atoms.append(line)
            elif line.startswith("CONECT"):
                conects.append(line)
            elif line.startswith("MASTER"):
                master_line = line
    
    return atoms, conects, master_line

def group_and_renumber_atoms(atoms):
    """Group atoms by residue number, renumber them, adjust UNL spacing, and create a mapping of old to new atom numbers."""
    grouped_atoms = []
    atom_mapping = {}
    new_atom_number = 1

    # Group atoms by residue number and sort by residue and original atom number
    residues = defaultdict(list)
    for line in atoms:
        residue_num = int(line[22:26].strip())
        residues[residue_num].append(line)
    
    sorted_residue_numbers = sorted(residues.keys())

    # Renumber atoms, build mapping, and adjust spacing for UNL
    for residue_num in sorted_residue_numbers:
        for line in residues[residue_num]:
            old_atom_number = int(line[6:11].strip())
            atom_mapping[old_atom_number] = new_atom_number

            # Move 'UNL' back by one space by slicing everything from index 16
            modified_line = line[:16] + line[17:]  # This removes the space behind 'UNL'
            new_line = f"{modified_line[:6]}{new_atom_number:>5} {modified_line[11:]}"
        
            grouped_atoms.append(new_line)
            new_atom_number += 1


    return grouped_atoms, atom_mapping

def update_conect(conects, atom_mapping):
    """Update CONECT lines with renumbered atom numbers based on atom_mapping, ensuring proper spacing."""
    new_conects = []
    
    for line in conects:
        parts = line.split()
        updated_parts = ["CONECT"]
        
        for part in parts[1:]:
            atom_num = int(part)
            updated_parts.append(f"{atom_mapping.get(atom_num, atom_num):>4}")
        
        new_line = " ".join(updated_parts).ljust(80) + "\n"
        new_conects.append(new_line)
    
    return new_conects

def update_master_line(master_line, atom_count):
    """Update the MASTER line with the new atom count."""
    updated_master = f"{master_line[:30]}{atom_count:>5}{master_line[35:40]}{atom_count:>5}{master_line[45:]}"
    return updated_master

def write_pdb(input_pdb, atoms, conects, master_line):
    """Overwrite the input PDB file with renumbered atoms, updated CONECT records, and modified MASTER line."""
    with open(input_pdb, 'w') as f:
        f.writelines(atoms)
        f.writelines(conects)
        f.write(master_line)
        f.write("END\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Group atoms by residue, renumber them, adjust spacing, and update CONECT and MASTER lines in a PDB file.")
    parser.add_argument("-input_pdb", required=True, help="Input PDB file to overwrite")

    args = parser.parse_args()

    # Parse PDB and separate ATOM, CONECT, and MASTER records
    atoms, conects, master_line = parse_pdb(args.input_pdb)

    # Group and renumber atoms by residue and get atom mapping
    grouped_atoms, atom_mapping = group_and_renumber_atoms(atoms)

    # Update CONECT records with the new atom numbers and ensure proper spacing
    new_conects = update_conect(conects, atom_mapping)

    # Update MASTER line with the new atom count
    updated_master_line = update_master_line(master_line, len(grouped_atoms))

    # Overwrite the input PDB file
    write_pdb(args.input_pdb, grouped_atoms, new_conects, updated_master_line)

    print(f"PDB file {args.input_pdb} has been overwritten with grouped, renumbered atoms, updated CONECT lines, and adjusted MASTER line.")
