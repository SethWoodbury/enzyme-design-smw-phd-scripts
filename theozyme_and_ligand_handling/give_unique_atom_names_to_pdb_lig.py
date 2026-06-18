import sys

def modify_pdb_atom_names(input_pdb, output_pdb):
    with open(input_pdb, 'r') as infile, open(output_pdb, 'w') as outfile:
        atom_counters = {}
        for line in infile:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_type = line[76:78].strip()
                if atom_type not in atom_counters:
                    atom_counters[atom_type] = 1
                else:
                    atom_counters[atom_type] += 1

                unique_atom_name = f"{atom_type}{atom_counters[atom_type]}"
                line = line[:12] + unique_atom_name.ljust(4) + line[16:]
            outfile.write(line)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python modify_pdb_atom_names.py input.pdb output.pdb")
    else:
        input_pdb = sys.argv[1]
        output_pdb = sys.argv[2]
        modify_pdb_atom_names(input_pdb, output_pdb)