import os
import argparse
import numpy as np
from Bio import PDB


def random_ORI_move(input_pdb, output_pdb, radius=5, initial_ORI_coord=None):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', input_pdb)

    if initial_ORI_coord:
        # Check the validity of initial_ORI_coord
        #try:
        initial_ORI_coord = eval(initial_ORI_coord)
        x, y, z = [float(el) for el in initial_ORI_coord]
        
        # Add ORI residue and atom.
        last_residue = [res for res in structure[0].get_residues()][-1]
        ori_res = PDB.Residue.Residue(('H', last_residue.id[1]+1, ' '), "ORI", ' ')
        
        ori_atom = PDB.Atom.Atom("ORI", (x, y, z), 1.0, 1.0, ' ', 'ORI', ori_res, element='ORI')
        ori_res.add(ori_atom)
        
        structure[0].add(PDB.Chain.Chain("X"))
        structure[0]["X"].add(ori_res)
        
        """
        except:
            print("Invalid format for initial_ORI_coord. Example:")
            print("--initial_ORI_coord [0, 2.2, -1]")
            return
        """

    else:
        # Locate the ORI atom on chain Z
        ori_atom = None
        for atom in structure.get_atoms():
            if atom.name == 'ORI' and atom.parent.parent.id == 'X':
                ori_atom = atom
                break

        if not ori_atom:
            print("No ORI atom found on chain Z in the PDB file.")
            print("Use --initial_ORI_coord for the PDB file without ORI atom.")
            return

    ori_position = ori_atom.coord

    # Generate random spherical coordinates
    phi = np.random.uniform(0, 2 * np.pi)
    costheta = np.random.uniform(-1, 1)
    u = np.random.uniform(0, 1)
    
    # Convert spherical coordinates to Cartesian coordinates
    theta = np.arccos(costheta)
    r = radius * (u ** (1/3))  # Correct for uniform distribution in 3D sphere
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    new_position = ori_position + np.array([x, y, z])
    ori_atom.coord = new_position
    
    # Save updated structure
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)
    print(f"Moved ORI atom to new position within {radius}Å radius and saved to {output_pdb}")

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomize the ORI token coordination.")
    parser.add_argument('--input_pdb', type=str, required=True, help='Input PDB file including an initial ORI token.')
    parser.add_argument('--output_pdb', type=str, required=True, help='Output PDB file including an initial ORI token.')
    parser.add_argument('--radius', type=float, default=5, help='Radius to randomize ORI token.')
    parser.add_argument('--initial_ORI_coord', type=str, default=None, help='Coordination to define an initial ORI token. --initialize_ORI [0,0,0]')

    args = parser.parse_args()
    random_ORI_move(args.input_pdb, args.output_pdb, radius=args.radius, initial_ORI_coord=args.initial_ORI_coord)
