#!/usr/bin/env python3
"""
Purpose:
    Automatically enforce the correct protonation state for catalytic histidines
    annotated via “REMARK 666” in a PDB file, by autodetecting whether each
    ND1 or NE2 ring nitrogen is coordinating (closest) to any Zn(II) HETATM
    (or, if no Zn is present, to the nearest non-H HETATM). Whichever nitrogen
    is closer is treated as the coordinating site and “loses” its proton; the
    opposite nitrogen receives the δ- (HD1) or ε-proton (HE2) accordingly.
    Optionally, this initial choice can be refined via simple steric-clash and
    H-bond geometry checks. All ATOM/HETATM records are then renumbered in a
    clean, logical order.

Inputs:
    --pdb <path>
        Path to the input PDB file containing “REMARK 666” annotations for HIS.
    --out <filename>
        (Optional) Name of the output PDB. Defaults to “modified_output.pdb”.
    --autodetect_his_hbonds
        (Optional) If set, refine the initial ND1/NE2 choice for each catalytic
        histidine by checking:
          • steric clashes of candidate HD1/HE2 positions against nearby atoms
          • potential H-bonds to nearby acceptors (O/N/S, non-metals)

Logic:
    1. Read all lines of the input PDB.
    2. Scan for `REMARK 666 ... HIS chain resnum` lines in file order and collect
       (chain, resnum) pairs identifying catalytic histidines.
    3. Parse every ATOM/HETATM line into an AtomRecord, preserving original text,
       atom names, element, and 3D coordinates.
    4. For each catalytic HIS:
       a. Extract its ring atoms (ND1, NE2, CG, CE1, CD2) and any existing HD1/HE2.
       b. Collect all Zn(II) HETATM coordinates; if none are present, fall back
          to all non-H HETATM coordinates.
       c. Compute the minimum ND1–Zn and NE2–Zn distances (or ND1–het / NE2–het).
       d. Whichever nitrogen is closer is treated as coordinating and has its
          proton removed; the other receives a newly placed proton (HD1 if ND1,
          HE2 if NE2) using a ring-normal bisector + dihedral placement procedure.
       e. If both hydrogens are present, remove the undesired one; if the desired
          one is missing, insert a new hydrogen immediately after the last HIS atom.
       f. If --autodetect_his_hbonds is set, compute candidate HD1 and HE2
          positions, then:
             • reject any candidate that clashes with nearby hydrophobic atoms
               (tight cutoff) or metals (more generous cutoff);
             • otherwise, score each candidate based on proximity to potential
               H-bond acceptors;
             • pick the non-clashing candidate with the better H-bond score; if
               they are effectively tied, keep the original Zn/heteroatom-based
               choice.
    5. Gather all surviving AtomRecords (excluding any flagged removals), sort by:
       (1) HETATM without “ORI ORI”, (2) ATOM, (3) HETATM with “ORI ORI”,
       then assign sequential serial numbers.
    6. Reconstruct the PDB: emit all non-ATOM/HETATM lines unchanged, interleave
       rewritten ATOM/HETATM lines in original file order, and write to the output.

Outputs:
    A modified PDB file with:
      - Exactly one proton (HD1 or HE2) on each catalytic histidine,
        assigned by proximity to Zn(II) (or nearest heteroatom), optionally
        refined by clash/H-bond geometry.
      - All other atoms unchanged.
      - Clean, sequential renumbering of all ATOM/HETATM records with the
        specified HETATM/ATOM/HETATM(“ORI ORI”) ordering.

Example:
    python change_cat_remark666_histidines_to_Hepsilon_or_Hdelta__AUTODETECT.py \\
        --pdb input_structure.pdb \\
        --out adjusted_structure.pdb \\
        --autodetect_his_hbonds

Key points:
  - No manual E/D flags required—protonation state is autodetected from
    Zn/heteroatom proximity.
  - Optional --autodetect_his_hbonds further refines HD1/HE2 placement using
    simple steric-clash and H-bond heuristics.
  - Proton placement uses a ring-normal bisector + 180° dihedral adjustment
    relative to the opposite ring nitrogen.
  - Final PDB is renumbered: HETATM (no “ORI ORI”), then ATOM, then
    HETATM (with “ORI ORI”).
"""

import sys
import math
import argparse
from typing import List, Tuple, Dict, Optional

def parse_args():
    parser = argparse.ArgumentParser(
        description="Modify catalytic histidines to have either HE2 or HD1 according to REMARK 666 lines and a user-provided E/D sequence."
    )
    parser.add_argument("--pdb", required=True, help="Path to the input PDB file.")
    parser.add_argument(
        "--out", required=False, default="modified_output.pdb",
        help="Output PDB file name (default: modified_output.pdb)"
    )
    parser.add_argument(
        "--autodetect_his_hbonds",
        action="store_true",
        help=(
            "If set, refine HID/HIE choice by checking steric clashes and potential "
            "HD1/HE2 hydrogen bonds to nearby acceptors."
        )
    )
    return parser.parse_args()

########################
# Basic Vector Utilities
########################

def v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def v_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def v_dot(a, b):
    return (a[0]*b[0] + a[1]*b[1] + a[2]*b[2])

def v_scale(a, s):
    return (a[0]*s, a[1]*s, a[2]*s)

def v_len(a):
    return math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])

def v_norm(a):
    length = v_len(a)
    if length < 1e-12:
        return (0.0, 0.0, 0.0)
    return (a[0]/length, a[1]/length, a[2]/length)

def v_cross(a, b):
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])

def dihedral(pA, pB, pC, pD):
    """
    Calculate dihedral angle (in degrees) for the 4 points A-B-C-D
    with the angle about the axis B->C. Returns angle in [-180, 180].
    """
    AB = v_sub(pA, pB)
    CB = v_sub(pC, pB)
    DC = v_sub(pD, pC)

    n1 = v_norm(v_cross(AB, CB))  # plane A-B-C
    n2 = v_norm(v_cross(DC, CB))  # plane D-C-B
    m1 = v_cross(n1, v_norm(CB))

    x = v_dot(n1, n2)
    y = v_dot(m1, n2)
    return math.degrees(math.atan2(y, x))

########################
# Histidine Geometry
########################

def place_histidine_h(
    central_atom_coord,  # ND1 or NE2
    ref1_coord,          # CE1
    ref2_coord,          # CG or CD2
    other_atom_coord,    # NE2 or ND1 (the opposite ring atom)
    bond_length=1.0,
    dihedral_target=180.0
):
    """
    1) Place H on the bisector between (central_atom->ref1) and (central_atom->ref2),
       at ~ bond_length from central_atom_coord.
    2) Rotate around the ring normal to achieve dihedral ~180° with 'other_atom_coord'.
    """
    B = central_atom_coord
    A = ref1_coord
    C = ref2_coord
    D = other_atom_coord

    # Step 1: direct bisector approach
    BA = v_sub(A, B)
    BC = v_sub(C, B)
    BA_n = v_norm(BA)
    BC_n = v_norm(BC)

    bis = v_add(BA_n, BC_n)
    if v_len(bis) < 1e-9:
        # fallback if collinear
        bis = (1.0, 0.0, 0.0)
    bis = v_norm(bis)
    # place H at ~ bond_length
    H0 = v_add(B, v_scale(bis, bond_length))

    # Step 2: ring normal
    ring_normal = v_cross(BA, BC)
    ring_normal = v_norm(ring_normal)
    if v_len(ring_normal) < 1e-9:
        # near-collinear, fallback
        return H0

    # Step 3: adjust dihedral of (D, A, B, H0) to ~180
    current_dihed = dihedral(D, A, B, H0)
    delta = dihedral_target - current_dihed
    if abs(delta) < 0.1:
        return H0

    delta_rad = math.radians(delta)
    BH0 = v_sub(H0, B)

    k = ring_normal
    cos_t = math.cos(delta_rad)
    sin_t = math.sin(delta_rad)
    kxv = v_cross(k, BH0)
    kdotv = v_dot(k, BH0)

    BH_rot = (
        BH0[0]*cos_t + kxv[0]*sin_t + k[0]*kdotv*(1 - cos_t),
        BH0[1]*cos_t + kxv[1]*sin_t + k[1]*kdotv*(1 - cos_t),
        BH0[2]*cos_t + kxv[2]*sin_t + k[2]*kdotv*(1 - cos_t),
    )

    return v_add(B, BH_rot)

########################
# PDB Parsing
########################

def parse_remark666_for_histidines(pdb_lines: List[str]) -> List[Tuple[str, str]]:
    """
    Collect (chain, resnum) for any REMARK 666 lines that mention HIS.
    Maintains order found in the file.
    """
    out = []
    for line in pdb_lines:
        if line.startswith("REMARK 666") and "HIS" in line:
            parts = line.split()
            if "HIS" in parts:
                i = parts.index("HIS")
                chain = parts[i - 1]
                resnum = parts[i + 1]
                out.append((chain, resnum))
    return out

class AtomRecord:
    """
    Holds info about a single ATOM/HETATM line, plus flags for removal.
    We preserve original_line except columns 6-11 (serial) and 30-54 (coords).
    """
    def __init__(self, original_line: str, line_index: int):
        self.original_line = original_line
        self.line_index = line_index
        self.record_type = original_line[0:6].strip()  # "ATOM" or "HETATM"
        self.atom_name = original_line[12:16].strip()
        self.alt_loc = original_line[16]
        self.res_name = original_line[17:20].strip()
        self.chain_id = original_line[21]
        self.res_seq = original_line[22:26].strip()
        self.i_code = original_line[26]
        try:
            self.x = float(original_line[30:38])
            self.y = float(original_line[38:46])
            self.z = float(original_line[46:54])
        except ValueError:
            self.x, self.y, self.z = 0.0, 0.0, 0.0
        self.element = original_line[76:78].strip() or self.atom_name[0]
        self.to_remove = False

    def rewrite_line(self, new_serial: int) -> str:
        """
        Return a new line with updated serial (6-11) and coords (30-54).
        """
        serial_str = f"{new_serial:5d}"
        coord_str = f"{self.x:8.3f}{self.y:8.3f}{self.z:8.3f}"
        new_line = (
            self.original_line[:6]
            + serial_str
            + self.original_line[11:30]
            + coord_str
            + self.original_line[54:]
        )
        return new_line

    def set_xyz(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def autodetect_EorD(
    nd1: AtomRecord,
    ne2: AtomRecord,
    all_records: List[AtomRecord]
) -> str:
    """
    1) If any ZN HETATM’s exist, measure the minimum ZN–ND1 and ZN–NE2 distances
       and decide based on those.
    2) Otherwise fall back to measuring against all non-H HETATM’s.
    """
    # 1) Gather ZN coordinates
    zn_coords = [
        (r.x, r.y, r.z)
        for r in all_records
        if r.record_type == "HETATM" and r.element.upper() == "ZN"
    ]

    if zn_coords:
        # compute shortest Zn–N distance
        dist_nd1 = min(math.dist((nd1.x, nd1.y, nd1.z), zc) for zc in zn_coords)
        dist_ne2 = min(math.dist((ne2.x, ne2.y, ne2.z), zc) for zc in zn_coords)
        print(f"\nFOUND ZINC -- using this")
    else:
        # 2) fallback: any non-H HETATM
        het_coords = [
            (r.x, r.y, r.z)
            for r in all_records
            if r.record_type == "HETATM" and r.element.upper() != "H"
        ]
        if het_coords:
            dist_nd1 = min(math.dist((nd1.x, nd1.y, nd1.z), hc) for hc in het_coords)
            dist_ne2 = min(math.dist((ne2.x, ne2.y, ne2.z), hc) for hc in het_coords)
        else:
            dist_nd1 = dist_ne2 = float('inf')

    # if NE2 is physically closer to ZN (or any heteroatom), assume it’s coordinating → attach HD1 ("D")
    return "D" if dist_ne2 < dist_nd1 else "E"

def create_new_h_atom_line(
    template_atom: AtomRecord, atom_name: str, x: float, y: float, z: float
) -> AtomRecord:
    """
    Create a new AtomRecord for a newly added hydrogen. We override:
      - record_type=ATOM
      - columns 6-11 => placeholder
      - columns 12-16 => new atom name
      - columns 30-54 => new coords
      - columns 76-78 => " H"
    """
    base = list(template_atom.original_line)
    base[0:6] = list("ATOM  ")
    base[6:11] = list("XXXXX")
    nm = atom_name.rjust(4, " ")
    for i, c in enumerate(nm):
        base[12 + i] = c
    coord_str = f"{x:8.3f}{y:8.3f}{z:8.3f}"
    for i, c in enumerate(coord_str):
        base[30 + i] = c
    base[76:78] = list(" H")
    new_line = "".join(base)
    ar = AtomRecord(new_line, line_index=-1)
    ar.record_type = "ATOM"
    ar.atom_name = atom_name
    ar.res_name = template_atom.res_name
    ar.chain_id = template_atom.chain_id
    ar.res_seq = template_atom.res_seq
    ar.i_code = template_atom.i_code
    ar.x = x
    ar.y = y
    ar.z = z
    ar.element = "H"
    return ar

################################
### LOOK AT HBOND CANDIDATES ###
################################
def compute_candidate_histidine_H_coords(
    nd1: AtomRecord,
    ne2: AtomRecord,
    ce1: AtomRecord,
    cg: AtomRecord,
    cd2: AtomRecord,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute hypothetical positions for:
      - 'D' state: HD1 on ND1
      - 'E' state: HE2 on NE2
    using the same ring-normal geometry as later insertion.
    """
    # Candidate D (HD1)
    nd1_c = (nd1.x, nd1.y, nd1.z)
    ce1_c = (ce1.x, ce1.y, ce1.z)
    cg_c  = (cg.x,  cg.y,  cg.z)
    ne2_c = (ne2.x, ne2.y, ne2.z)
    H_D   = place_histidine_h(
        central_atom_coord=nd1_c,
        ref1_coord=ce1_c,
        ref2_coord=cg_c,
        other_atom_coord=ne2_c,
        bond_length=1.0,
        dihedral_target=180.0
    )

    # Candidate E (HE2)
    ne2_c = (ne2.x, ne2.y, ne2.z)
    cd2_c = (cd2.x, cd2.y, cd2.z)
    nd1_c = (nd1.x, nd1.y, nd1.z)
    H_E   = place_histidine_h(
        central_atom_coord=ne2_c,
        ref1_coord=ce1_c,
        ref2_coord=cd2_c,
        other_atom_coord=nd1_c,
        bond_length=1.0,
        dihedral_target=180.0
    )

    return {"D": H_D, "E": H_E}


def has_clash(
    H_coord: Tuple[float, float, float],
    neighbors: List[AtomRecord],
    clash_cutoff: float = 1.0
) -> bool:
    """
    Return True if H is closer than clash_cutoff Å to any neighbor heavy atom.
    """
    hx, hy, hz = H_coord
    for r in neighbors:
        d = math.dist((hx, hy, hz), (r.x, r.y, r.z))
        if d < clash_cutoff:
            return True
    return False


def hbond_score(
    H_coord: Tuple[float, float, float],
    acceptors: List[AtomRecord],
    hbond_cutoff: float = 2.8
) -> float:
    """
    Very simple H-bond scoring: for any acceptor within hbond_cutoff Å of H,
    assign a score proportional to (cutoff - distance). Return best score.
    """
    hx, hy, hz = H_coord
    best = 0.0
    for r in acceptors:
        d = math.dist((hx, hy, hz), (r.x, r.y, r.z))
        if d <= hbond_cutoff:
            score = hbond_cutoff - d
            if score > best:
                best = score
    return best


def refine_his_orientation_by_hbonds(
    desired_type: str,
    nd1: AtomRecord,
    ne2: AtomRecord,
    ce1: AtomRecord,
    cg: AtomRecord,
    cd2: AtomRecord,
    his_atoms: List[AtomRecord],
    all_atom_records: List[AtomRecord],
    clash_cutoff: float = 1.0,
    hbond_cutoff: float = 2.8,
) -> str:
    """
    Given an initial 'desired_type' ('D' or 'E') from Zn proximity, refine it by:
      1) Checking steric clashes for the candidate HD1 and HE2 hydrogens.
      2) Checking which candidate makes a better potential H-bond to nearby
         acceptors (O/N/S, excluding metals).

    Rules:
      - If only one candidate clashes, pick the non-clashing one.
      - Otherwise, pick the one with higher hbond_score().
      - If scores are effectively tied, keep the original desired_type.
    """
    candidates = compute_candidate_histidine_H_coords(nd1, ne2, ce1, cg, cd2)
    H_D = candidates["D"]
    H_E = candidates["E"]

    # Identify this residue so we can avoid self-collisions
    his_res_id = (nd1.chain_id, nd1.res_seq, nd1.i_code)

    metal_elems = {"ZN", "MG", "FE", "CU", "MN", "CO", "NI", "CA"}
    hydrophobic_elems = {"C", "H"}  # mostly nonpolar stuff you care about for clashes

    # All heavy neighbors except this histidine
    all_heavy_neighbors = [
        r for r in all_atom_records
        if r.element.upper() != "H"
        and (r.chain_id, r.res_seq, r.i_code) != his_res_id
    ]

    # Split into hydrophobic vs metal neighbors
    hydro_neighbors = [
        r for r in all_heavy_neighbors
        if r.element.upper() in hydrophobic_elems
    ]
    metal_neighbors = [
        r for r in all_heavy_neighbors
        if r.element.upper() in metal_elems
    ]

    # Potential H-bond acceptors: O/N/S, non-metals, and not the same residue
    acceptors = [
        r for r in all_heavy_neighbors
        if r.element.upper() in ("O", "N", "S")
        and r.element.upper() not in metal_elems
    ]

    # Clash checks:
    # - hydrophobic clashes: tighter cutoff (e.g. 1.2 Å)
    # - metal "clashes": more conservative cutoff (e.g. 3.0 Å), because
    #   we *never* want to protonate the coordinating N toward a metal.
    clash_D       = has_clash(H_D, hydro_neighbors, clash_cutoff=clash_cutoff)
    clash_D_metal = has_clash(H_D, metal_neighbors, clash_cutoff=3.0)

    clash_E       = has_clash(H_E, hydro_neighbors, clash_cutoff=clash_cutoff)
    clash_E_metal = has_clash(H_E, metal_neighbors, clash_cutoff=3.0)

    # 1) Clash-based override (including metal-side clashes)
    if (clash_D or clash_D_metal) and not (clash_E or clash_E_metal):
        return "E"
    if (clash_E or clash_E_metal) and not (clash_D or clash_D_metal):
        return "D"

    # 2) H-bond-based refinement
    score_D = hbond_score(H_D, acceptors, hbond_cutoff=hbond_cutoff)
    score_E = hbond_score(H_E, acceptors, hbond_cutoff=hbond_cutoff)

    if score_D > score_E + 1e-3:
        return "D"
    if score_E > score_D + 1e-3:
        return "E"

    # 3) Tie → fall back
    return desired_type

################
### DEF MAIN ###
################
def main():
    args = parse_args()

    # 1) Read lines
    with open(args.pdb, "r") as f:
        pdb_lines = f.readlines()

    # 2) Histidines from REMARK 666
    histidines_in_remark666 = parse_remark666_for_histidines(pdb_lines)

    # 3) (no more user-supplied E/D flags → auto-detect only)
    # Print summary
    msg = "### IDENTIFIED "
    for (chain, resnum) in histidines_in_remark666:
        msg += f"{{HIS {resnum}, CHAIN {chain}}} "
    msg += "AS CATALYTIC HISTIDINE RESIDUES ###"
    print(msg)

    # 4) Parse lines -> AtomRecord or dict
    parsed_records = []
    for idx, line in enumerate(pdb_lines):
        if line.startswith(("ATOM", "HETATM")):
            parsed_records.append(AtomRecord(line, idx))
        else:
            parsed_records.append({"type": "OTHER", "line": line, "line_index": idx})

    # 5) For each HIS from REMARK 666, auto-detect E vs D
    for (chain, resnum) in histidines_in_remark666:
        # collect all HIS atoms for this residue
        his_atoms = [
            r for r in parsed_records
            if isinstance(r, AtomRecord)
            and r.res_name == "HIS"
            and r.chain_id.strip() == chain.strip()
            and r.res_seq.strip()   == resnum.strip()
            and not r.to_remove
        ]

        # pick out the two ring nitrogens
        nd1 = next((r for r in his_atoms if r.atom_name == "ND1"), None)
        ne2 = next((r for r in his_atoms if r.atom_name == "NE2"), None)
        if nd1 is None or ne2 is None:
            print(f"WARNING: Missing ND1 or NE2 for HIS {resnum} chain {chain}. Skipping.")
            continue

        # now autodetect D vs E
        desired_type = autodetect_EorD(
            nd1, ne2,
            [r for r in parsed_records if isinstance(r, AtomRecord)]
        )
        print(f"HIS {resnum} chain {chain}: autodetected protonation → {desired_type}")

        # gather relevant HIS lines
        his_atoms = [
            r for r in parsed_records
            if isinstance(r, AtomRecord)
            and r.res_name == "HIS"
            and r.chain_id.strip() == chain.strip()
            and r.res_seq.strip() == resnum.strip()
            and not r.to_remove
        ]
        if not his_atoms:
            print(f"WARNING: No HIS lines found for {chain} {resnum}")
            continue

        # we need ND1, NE2, CE1, CG, CD2
        nd1 = next((x for x in his_atoms if x.atom_name == "ND1"), None)
        ne2 = next((x for x in his_atoms if x.atom_name == "NE2"), None)
        ce1 = next((x for x in his_atoms if x.atom_name == "CE1"), None)
        cg  = next((x for x in his_atoms if x.atom_name == "CG"),  None)
        cd2 = next((x for x in his_atoms if x.atom_name == "CD2"), None)
        if not (nd1 and ne2 and ce1 and cg and cd2):
            print(f"WARNING: Missing ring atoms for HIS {resnum}, chain {chain}. Skipping.")
            continue

        # Optionally refine D/E choice by checking steric clashes and H-bond geometry
        if args.autodetect_his_hbonds:
            all_atoms_for_refine = [
                r for r in parsed_records
                if isinstance(r, AtomRecord)
            ]
            prev_type = desired_type
            desired_type = refine_his_orientation_by_hbonds(
                desired_type,
                nd1, ne2, ce1, cg, cd2,
                his_atoms,
                all_atoms_for_refine
            )
            if desired_type != prev_type:
                print(
                    f"HIS {resnum} chain {chain}: "
                    f"refined protonation {prev_type} → {desired_type} based on clashes/H-bonds."
                )
            else:
                print(
                    f"HIS {resnum} chain {chain}: "
                    f"H-bond/clash refinement kept protonation as {desired_type}."
                )


        # existing hydrogens
        hd1 = next((x for x in his_atoms if x.atom_name == "HD1"), None)
        he2 = next((x for x in his_atoms if x.atom_name == "HE2"), None)

        # remove undesired if both exist
        if hd1 and he2:
            if desired_type == "D":
                print(f"HIS {resnum} chain {chain}: Found both HD1 & HE2; removing HE2.")
                he2.to_remove = True
                he2 = None
            else:
                print(f"HIS {resnum} chain {chain}: Found both HD1 & HE2; removing HD1.")
                hd1.to_remove = True
                hd1 = None

        # place the correct hydrogen if missing
        if desired_type == "D":
            # We want HD1
            if hd1 and not hd1.to_remove:
                print(f"HIS {resnum} chain {chain}: Already has HD1.")
            else:
                print(f"HIS {resnum} chain {chain}: Adding HD1 with ring-normal geometry.")
                nd1_c = (nd1.x, nd1.y, nd1.z)
                ce1_c = (ce1.x, ce1.y, ce1.z)
                cg_c  = (cg.x,  cg.y,  cg.z)
                ne2_c = (ne2.x, ne2.y, ne2.z)

                # place new HD1
                Hcoord = place_histidine_h(
                    central_atom_coord=nd1_c,
                    ref1_coord=ce1_c,
                    ref2_coord=cg_c,
                    other_atom_coord=ne2_c,
                    bond_length=1.0,
                    dihedral_target=180.0
                )
                new_atom = create_new_h_atom_line(nd1, "HD1", Hcoord[0], Hcoord[1], Hcoord[2])

                # Insert after last HIS atom
                line_indices = [a.line_index for a in his_atoms if not a.to_remove]
                insert_after = max(line_indices)
                insert_pos = None
                for jj, rr in enumerate(parsed_records):
                    if isinstance(rr, AtomRecord) and rr.line_index == insert_after:
                        insert_pos = jj + 1
                        break
                if insert_pos is None:
                    insert_pos = len(parsed_records)
                parsed_records.insert(insert_pos, new_atom)
        else:
            # We want HE2
            if he2 and not he2.to_remove:
                print(f"HIS {resnum} chain {chain}: Already has HE2.")
            else:
                print(f"HIS {resnum} chain {chain}: Adding HE2 with ring-normal geometry.")
                ne2_c = (ne2.x, ne2.y, ne2.z)
                ce1_c = (ce1.x, ce1.y, ce1.z)
                cd2_c = (cd2.x, cd2.y, cd2.z)
                nd1_c = (nd1.x, nd1.y, nd1.z)

                # place new HE2
                Hcoord = place_histidine_h(
                    central_atom_coord=ne2_c,
                    ref1_coord=ce1_c,
                    ref2_coord=cd2_c,
                    other_atom_coord=nd1_c,
                    bond_length=1.0,
                    dihedral_target=180.0
                )
                new_atom = create_new_h_atom_line(ne2, "HE2", Hcoord[0], Hcoord[1], Hcoord[2])

                # Insert after last HIS atom
                line_indices = [a.line_index for a in his_atoms if not a.to_remove]
                insert_after = max(line_indices)
                insert_pos = None
                for jj, rr in enumerate(parsed_records):
                    if isinstance(rr, AtomRecord) and rr.line_index == insert_after:
                        insert_pos = jj + 1
                        break
                if insert_pos is None:
                    insert_pos = len(parsed_records)
                parsed_records.insert(insert_pos, new_atom)

    # 6) Renumber lines in the order: (1) HETATM w/o "ORI ORI", (2) ATOM, (3) HETATM with "ORI ORI"
    atom_records = [r for r in parsed_records if isinstance(r, AtomRecord) and not r.to_remove]

    def numbering_priority(rec: AtomRecord) -> int:
        has_ori = ("ORI ORI" in rec.original_line)
        if rec.record_type == "HETATM" and not has_ori:
            return 1
        if rec.record_type == "ATOM":
            return 2
        if rec.record_type == "HETATM" and has_ori:
            return 3
        return 999

    sorted_for_numbering = sorted(atom_records, key=lambda r: numbering_priority(r))
    rec_to_serial = {}
    serial_counter = 1
    for r in sorted_for_numbering:
        rec_to_serial[r] = serial_counter
        serial_counter += 1

    # 7) Build final lines in original order
    final_lines = []
    for item in parsed_records:
        if isinstance(item, dict) and item["type"] == "OTHER":
            final_lines.append(item["line"])
        elif isinstance(item, AtomRecord):
            if item.to_remove:
                continue
            new_serial = rec_to_serial[item]
            out_line = item.rewrite_line(new_serial)
            final_lines.append(out_line)

    # 8) Don't append END if not present

    # 9) Write out
    with open(args.out, "w") as f:
        f.writelines(final_lines)

    print(f"Done. Wrote modified PDB to {args.out}")


if __name__ == "__main__":
    main()
