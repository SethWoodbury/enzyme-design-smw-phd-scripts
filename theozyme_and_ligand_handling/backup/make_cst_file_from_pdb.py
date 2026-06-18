#!/usr/bin/env python3
"""
Builds a Rosetta constraint file from an input PDB and a list of constraint specs.
Each spec defines two residues (or a residue and a ligand fragment) by index, three atom names,
and flags for primary/covalent. The script measures the 6D geometry (distance, angles,
dihedrals) between the two triples of atoms and emits a .cst file pinning those atoms
within given tolerances.

EXAMPLE USAGE:
    csts2build = [  # define your constraints here... ]
    pdb = 'path/to/structure.pdb'
    constraint_file_str_from_spec(pdb, csts2build, write=True, outfile='out.cst')
"""
import sys
from textwrap import dedent
import math
import numpy as np

# --- PDB parsing classes ---
class AtomRecord:
    def __init__(self, name, resSeq, coord):
        self.name = name.strip()
        self.resSeq = resSeq
        self.coord = coord

    @classmethod
    def from_str(cls, line):
        # PDB ATOM/HETATM fixed-column parsing
        name = line[12:16]
        resSeq = int(line[22:26])
        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        return cls(name, resSeq, np.array([x, y, z]))

class Residue:
    def __init__(self, atom_records):
        self.atom_records = atom_records
        self.coords = [atom.coord for atom in atom_records]

    @classmethod
    def from_records(cls, records):
        return cls(records)

# --- Geometry measurement ---
def measure_distance(xyz1, xyz2):
    return math.sqrt((xyz1 - xyz2).dot(xyz1 - xyz2))

def measure_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return math.degrees(math.acos(max(min(cosine_angle, 1.0), -1.0)))

def measure_dihedral(p0, p1, p2, p3):
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return math.degrees(math.atan2(y, x))

def measure_geometry(r1_xyz, r2_xyz):
    r1a1, r1a2, r1a3 = r1_xyz
    r2a1, r2a2, r2a3 = r2_xyz
    d   = measure_distance(r1a1, r2a1)
    aA  = measure_angle(r1a2, r1a1, r2a1)
    aB  = measure_angle(r1a1, r2a1, r2a2)
    dA  = measure_dihedral(r1a3, r1a2, r1a1, r2a1)
    dAB = measure_dihedral(r1a2, r1a1, r2a1, r2a2)
    dB  = measure_dihedral(r1a1, r2a1, r2a2, r2a3)
    return d, aA, aB, dA, dAB, dB

# --- PDB reader ---
def read_in_stubs_file(fname: str):
    models = []
    model = []
    curr_res = []
    resSeq = None
    detected_multimodel = False
    with open(fname) as f:
        for line in f:
            if line.startswith('MODEL'):
                detected_multimodel = True
                model = []
                curr_res = []
                resSeq = None
                continue
            if line.startswith('ENDMDL'):
                model.append(Residue.from_records(curr_res))
                models.append(model)
                continue
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom = AtomRecord.from_str(line)
                if resSeq is None or atom.resSeq != resSeq:
                    if curr_res:
                        model.append(Residue.from_records(curr_res))
                    curr_res = [atom]
                    resSeq = atom.resSeq
                else:
                    curr_res.append(atom)
        if not detected_multimodel and curr_res:
            model.append(Residue.from_records(curr_res))
            models.append(model)
    return models

# --- Core constraint writing ---
def get_xyz(model, resi, atom_name):
    # 1-based indexing; allow negative to index from end
    if resi < 0:
        res_obj = model[resi]
    else:
        res_obj = model[resi-1]
    for i, atom in enumerate(res_obj.atom_records):
        if atom.name == atom_name:
            return atom.coord
    sys.exit(f"ERROR: atom {atom_name} not found in residue {resi}")


def write_constraint(name, r1_n, r2_n, r1_atms, r2_atms, g6D,
                     primary=True, covalent=False, dist_tol=0.1, angle_tol=5.0):
    g6D_fmt = [str(round(v,1)).rjust(7) for v in g6D]
    dist_t = str(dist_tol).rjust(4)
    ang_t  = str(angle_tol).rjust(4)
    return dedent(f"""
    # {name}
    CST::BEGIN
        TEMPLATE::   ATOM_MAP: 1 atom_name: {' '.join(r1_atms)}
        TEMPLATE::   ATOM_MAP: 1 residue3:  {r1_n}

        TEMPLATE::   ATOM_MAP: 2 atom_name: {' '.join(r2_atms)}
        TEMPLATE::   ATOM_MAP: 2 residue1:  {r2_n}

        CONSTRAINT:: distanceAB: {g6D_fmt[0]} {dist_t}  100    {'1' if covalent else '0'}  1
        CONSTRAINT::    angle_A: {g6D_fmt[1]} {ang_t}   50  360. 1
        CONSTRAINT::    angle_B: {g6D_fmt[2]} {ang_t}   50  360. 1
        CONSTRAINT::  torsion_A: {g6D_fmt[3]} {ang_t}   50  360. 1
        CONSTRAINT:: torsion_AB: {g6D_fmt[4]} {ang_t}   50  360. 1
        CONSTRAINT::  torsion_B: {g6D_fmt[5]} {ang_t}   50  360. 1
        {'' if primary else 'ALGORITHM_INFO:: match\n    SECONDARY_MATCH: DOWNSTREAM\nALGORITHM_INFO::END'}
    CST::END
    """
    )

def constraint_file_str_from_spec(pdb, csts2build, write=True, outfile=''):
    file_str = ''
    model = read_in_stubs_file(pdb)[0]
    for spec in csts2build:
        r1_xyz = [get_xyz(model, spec['r1_i'], a) for a in spec['r1_atms']]
        r2_xyz = [get_xyz(model, spec['r2_i'], a) for a in spec['r2_atms']]
        g6 = measure_geometry(r1_xyz, r2_xyz)
        file_str += write_constraint(
            spec['name'], spec['r1_n'], spec['r2_n'],
            spec['r1_atms'], spec['r2_atms'], g6,
            primary=spec.get('primary', True),
            covalent=spec.get('covalent', False)
        )
    if write:
        assert outfile, 'Specify outfile'
        with open(outfile, 'w') as f:
            f.write(file_str)
    return file_str

# Example call:
# cst_text = constraint_file_str_from_spec(pdb, csts2build, write=True, outfile='example.cst')
