#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prepare_PDB_structure_into_theozyme.py — one-stop theozyme/ligand PDB prep.
Author: Seth M. Woodbury, David Baker Lab, UW (woodbuse@uw.edu)
See prepare_PDB_structure_into_theozyme__DESIGN.md for the full spec.
Preserve-by-default: no destructive change unless an explicit flag is given.
"""
import argparse, math, os, re, string, sys

# --- locate repo root + shared external paths ---
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "repo_paths.py").is_file():
        _sys.path.insert(0, str(_anc)); break
import repo_paths

PROTEIN_RESNAMES = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS",
    "MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
}

###############################################################################
# ncAA DICTIONARY (EDIT THIS SECTION TO ADD/CHANGE NON-CANONICAL AAs)
###############################################################################

# Each entry:
#   key = ncAA residue name (3-letter)
#   value = {
#       "canonical_resname": <cAA 3-letter>,
#       "atom_name_map": { <ncAA_atom_name>: <cAA_atom_name> or None },
#       "ligand_resname": <3-letter ligand name for "extra" atoms>
#   }
#
# Any ncAA atoms whose names are in atom_name_map with a non-None value will
# be turned into canonical ATOM lines (same chain / residue number).
# All remaining atoms from that ncAA residue will be turned into HETATM with
# a new (chain, resseq) and residue name ligand_resname.
#
# NOTE: This KCX mapping is a starting point; tweak as needed.
NONCANONICAL_AA_MAP = {
    "KCX": {
        "canonical_resname": "LYS",
        "atom_name_map": {
            # Backbone
            "N": "N", "CA": "CA", "C": "C", "O": "O",
            # Side chain
            "CB": "CB", "CG": "CG", "CD": "CD", "CE": "CE", "NZ": "NZ",
            # Add / edit more as needed, e.g. "H": "H", etc.
        },
        "ligand_resname": "LIG",
    },
}

###############################################################################
# PROTEIN / SIDECHAIN POLAR HYDROGEN DICTIONARIES
# (EDIT THIS SECTION TO TUNE WHICH POLAR Hs ARE CANONICAL / PROTECTED)
###############################################################################

# Mapping: resname -> heavy_atom_name -> list of canonical H atom names
# (You can extend/edit this any time.)
PROTECTED_POLAR_H_MAP = {
    "SER": {"OG":  ["HG"]},
    "THR": {"OG1": ["HG1"]},
    "TYR": {"OH":  ["HH"]},
    "CYS": {"SG":  ["HG"]},
    "TRP": {"NE1": ["HE1"]},
    "HIS": {"ND1": ["HD1"], "NE2": ["HE2"]},
    "LYS": {"NZ":  ["HZ1", "HZ2", "HZ3"]},
    "ARG": {
        "NE":  ["HE"],
        "NH1": ["HH11", "HH12"],
        "NH2": ["HH21", "HH22"],
    },
    "ASN": {"ND2": ["HD21", "HD22"]},
    "GLN": {"NE2": ["HE21", "HE22"]},
    "ASP": {"OD1": ["HD1"], "OD2": ["HD2"]},
    "GLU": {"OE1": ["HE1"], "OE2": ["HE2"]},
}

###############################################################################
# COLUMN HELPERS
###############################################################################

def pad_line(line):
    line = line.rstrip("\n")
    return line if len(line) >= 80 else line.ljust(80)

def get_field(line, start, end):
    return pad_line(line)[start:end]

def replace_field(line, start, end, new):
    if len(new) != (end - start):
        raise ValueError(f"replace_field length mismatch: {end-start} vs {len(new)}")
    line = pad_line(line)
    return line[:start] + new + line[end:]

def is_atom_record(line):
    return line[:6] in ("ATOM  ", "HETATM")

def atom_name_field(line):
    """Return the raw 4-char atom-name field (cols 13-16), spaces intact."""
    return get_field(line, 12, 16)

def parse_charge_fields(line):
    """Return (element, formal_charge_str). Element from cols 77-78; charge
    from 79-80 (e.g. '1-'). Falls back to atom-name first alpha char for
    element only when col 77-78 is blank. Never misreads 'N1-' as element."""
    p = pad_line(line)
    elem = p[76:78].strip()
    chg = p[78:80].strip()
    if not elem:
        nm = atom_name_field(line).strip()
        elem = "".join(c for c in nm if c.isalpha())[:2].upper()
    return elem, chg


###############################################################################
# ncAA TRANSFORM FUNCTIONS (verbatim port from MAIN.py with 4 patches)
###############################################################################

def strip_insertion_code(line):
    """Remove insertion-code letter (column 27) for ATOM/HETATM, e.g. 132N -> 132 ."""
    if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
        return line
    if line[26] != "":  # column 27
        if line[26] != " ":
            line = line[:26] + " " + line[27:]
    return line


def detect_ncaa(lines):
    """Return dict of ncAA residues found: {(resname, chain, resseq, icode): {'record_types': {ATOM/HETATM}}}."""
    ncaa_residues = {}
    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        resname = line[17:20].strip()
        if resname not in NONCANONICAL_AA_MAP:
            continue
        chain = line[21]
        resseq = line[22:26].strip()
        icode = (line[26].strip() or " ")
        key = (resname, chain, resseq, icode)
        info = ncaa_residues.setdefault(key, {"record_types": set()})
        info["record_types"].add(line[0:6].strip())
    return ncaa_residues


def gather_used_positions(lines):
    """Collect all (chain, resseq, icode) positions used by ATOM/HETATM."""
    used = set()
    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        chain = line[21]
        resseq = line[22:26].strip()
        icode = (line[26].strip() or " ")
        used.add((chain, resseq, icode))
    return used


def new_lig_position_generator(used):
    """Yield new (chain, resseq, icode) combos not in 'used', deterministic A1..Z1, A2..Z2, etc."""
    for resseq_int in range(1, 9999):
        resseq_str = str(resseq_int)
        for chain in string.ascii_uppercase:
            key = (chain, resseq_str, " ")
            if key not in used:
                used.add(key)
                yield key


def build_frag_filters(arg_list, wl=None):
    """
    Translate --frag_ncAA_into_cAA_plus_lig argument into:
      None                 -> no fragmentation
      {'all': True, ...}   -> fragment all ncAA
      {'all': False, 'targets': {(chain, resseq), ...}} -> fragment these ncAA only
    """
    if arg_list is None:
        return None
    if len(arg_list) == 0:
        return {"all": True, "targets": set()}

    targets = set()
    for token in arg_list:
        token = token.strip()
        if not token:
            continue
        chain = token[0]
        resseq = "".join(ch for ch in token[1:] if ch.isdigit())
        if not resseq:
            # Patch 4: route through wl if provided, else fallback
            msg = f"Could not parse residue identifier '{token}' (expected like A169); skipping."
            if wl is not None:
                wl.warn(msg, category="ncaa")
            else:
                print(f"[WARN] {msg}", file=sys.stderr)
            continue
        targets.add((chain, resseq))
    return {"all": False, "targets": targets}


def frag_ncaa(lines, frag_filters, enable_intelligent_hstrip=True, wl=None):
    """
    Fragment ncAA residues into canonical AA + ligand, based on NONCANONICAL_AA_MAP.

    If enable_intelligent_hstrip is True:
      - For each fragmented ncAA residue:
          * Hydrogens whose nearest heavy atom is canonical -> STRIPPED (deleted).
          * Hydrogens whose nearest heavy atom is ligand-side -> LIGANDIZED.

    Patch 1: element detection uses parse_charge_fields() instead of line[76:78].strip().
    Patch 3: no-newline string convention (lines stored without trailing newline).
    Patch 4: print calls replaced by wl.info/wl.warn (wl is last param).
    """
    if frag_filters is None:
        return lines

    used = gather_used_positions(lines)
    pos_gen = new_lig_position_generator(used)
    lig_pos_for_res = {}  # key: (chain, resseq, icode) of ncAA -> (chain, resseq, icode) for ligand

    target_all = frag_filters.get("all", False)
    targets = frag_filters.get("targets", set())

    # ------------------------------------------------------------------
    # Precompute hydrogen handling: which Hs to strip vs force into ligand
    # ------------------------------------------------------------------
    strip_H = set()              # indices of lines to drop
    force_lig_H_to_ligand = set()  # indices of lines forced to ligand branch

    if enable_intelligent_hstrip:
        groups = {}  # key: (resname, chain, resseq, icode) -> [(idx, line), ...]
        for idx, line in enumerate(lines):
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            resname = line[17:20].strip()
            if resname not in NONCANONICAL_AA_MAP:
                continue
            chain = line[21]
            resseq = line[22:26].strip()
            icode = (line[26].strip() or " ")
            if not (target_all or (chain, resseq) in targets):
                continue
            key = (resname, chain, resseq, icode)
            groups.setdefault(key, []).append((idx, line))

        for (resname, chain, resseq, icode), grp in groups.items():
            rule = NONCANONICAL_AA_MAP[resname]
            canon_map = rule["atom_name_map"]

            heavy = []      # (idx, x, y, z, is_canonical_heavy)
            hydrogens = []  # (idx, x, y, z)

            for idx, line in grp:
                atom_name = line[12:16].strip()
                # Patch 1: safe element detection via parse_charge_fields
                elem = parse_charge_fields(line)[0]

                # Coordinates
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    # If coords are weird, skip for H-handling
                    continue

                if elem == "H":
                    hydrogens.append((idx, x, y, z))
                else:
                    is_canon = atom_name in canon_map and canon_map[atom_name] is not None
                    heavy.append((idx, x, y, z, is_canon))

            if not heavy:  # no heavy atoms to anchor to; skip special H handling
                continue

            for h_idx, hx, hy, hz in hydrogens:
                best_d2 = None
                best_is_canon = False
                for a_idx, ax, ay, az, is_canon in heavy:
                    d2 = (hx - ax)**2 + (hy - ay)**2 + (hz - az)**2
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2
                        best_is_canon = is_canon
                if best_d2 is None:
                    continue
                if best_is_canon:
                    strip_H.add(h_idx)  # belongs to canonical fragment -> strip
                else:
                    force_lig_H_to_ligand.add(h_idx)  # belongs to ligand fragment -> ligandize

    # ------------------------------------------------------------------
    # Main fragmentation pass
    # ------------------------------------------------------------------
    new_lines = []
    for idx, line in enumerate(lines):
        if not line.startswith(("ATOM  ", "HETATM")):
            new_lines.append(line)
            continue

        # Drop hydrogens we decided to strip
        if idx in strip_H:
            continue

        resname = line[17:20].strip()
        if resname not in NONCANONICAL_AA_MAP:
            new_lines.append(line)
            continue

        chain = line[21]
        resseq = line[22:26].strip()
        icode = (line[26].strip() or " ")
        res_key_for_filter = (chain, resseq)

        if not (target_all or res_key_for_filter in targets):
            new_lines.append(line)
            continue

        rule = NONCANONICAL_AA_MAP[resname]
        canon = rule["canonical_resname"]
        canon_map = rule["atom_name_map"]
        ligand_resname = rule["ligand_resname"]
        atom_name = line[12:16].strip()

        # Decide if this atom should be canonical or ligand
        if idx in force_lig_H_to_ligand or atom_name not in canon_map or canon_map[atom_name] is None:
            # -----------------------------
            # Ligand branch
            # -----------------------------
            lig_key = (chain, resseq, icode)
            if lig_key not in lig_pos_for_res:
                lig_pos_for_res[lig_key] = next(pos_gen)
            lig_chain, lig_resseq, lig_icode = lig_pos_for_res[lig_key]
            resseq_field = lig_resseq.rjust(4)

            line2 = line
            # Patch 2: use Task 2 replace_field (length-checked + padding)
            line2 = replace_field(line2, 0, 6, "HETATM")
            line2 = replace_field(line2, 17, 20, ligand_resname.rjust(3))
            line2 = replace_field(line2, 21, 22, lig_chain)
            line2 = replace_field(line2, 22, 26, resseq_field)
            line2 = replace_field(line2, 26, 27, lig_icode)
            new_lines.append(line2)
        else:
            # -----------------------------
            # Canonical AA branch
            # -----------------------------
            new_name = canon_map[atom_name].rjust(4)
            line2 = line
            # Patch 2: use Task 2 replace_field
            line2 = replace_field(line2, 0, 6, "ATOM  ")
            line2 = replace_field(line2, 12, 16, new_name)
            line2 = replace_field(line2, 17, 20, canon.rjust(3))
            new_lines.append(line2)

    return new_lines


def convert_ncaa_hetatm_to_atom(lines, wl=None):
    """
    For ncAAs, convert HETATM -> ATOM and return:
      new_lines, protected_atoms
    where protected_atoms is a set of (chain, resseq, icode, atom_name).
    Patch 2: use Task 2 replace_field.
    """
    protected_atoms = set()
    new_lines = []
    for line in lines:
        if not line.startswith("HETATM"):
            new_lines.append(line)
            continue
        resname = line[17:20].strip()
        if resname not in NONCANONICAL_AA_MAP:
            new_lines.append(line)
            continue
        chain = line[21]
        resseq = line[22:26].strip()
        icode = (line[26].strip() or " ")
        atom_name = line[12:16].strip()
        protected_atoms.add((chain, resseq, icode, atom_name))
        line2 = replace_field(line, 0, 6, "ATOM  ")
        new_lines.append(line2)
    return new_lines, protected_atoms


def revert_protected_ncaa_atom_to_hetatm(lines, protected_atoms, wl=None):
    """Revert ncAA atoms we previously forced to ATOM back to HETATM.
    Patch 2: use Task 2 replace_field.
    """
    if not protected_atoms:
        return lines
    new_lines = []
    for line in lines:
        if not line.startswith("ATOM  "):
            new_lines.append(line)
            continue
        chain = line[21]
        resseq = line[22:26].strip()
        icode = (line[26].strip() or " ")
        atom_name = line[12:16].strip()
        if (chain, resseq, icode, atom_name) in protected_atoms:
            line = replace_field(line, 0, 6, "HETATM")
        new_lines.append(line)
    return new_lines


def protect_sidechain_polarH(lines, residue_tokens, wl=None):
    """
    Rename selected sidechain polar Hs from H* -> Q* so they survive step1/2, then
    later Q* will be reverted to H*.

    residue_tokens:
      - None => feature OFF
      - []   => apply to ALL protein residues
      - ["A32", "A25", ...] => apply only to those (chain,resseq) combos

    Patch 1: element detection via parse_charge_fields().
    Patch 3: no-newline convention (lines have no trailing \n).
    Patch 4: print -> wl.info/wl.warn.
    """
    if residue_tokens is None:
        return lines

    # Interpret residue list
    if len(residue_tokens) == 0:
        apply_all = True
        target_res = set()
        if wl:
            wl.info("protect_sidechain_polarH: applying to ALL protein residues.")
        else:
            print("[INFO] protect_sidechain_polarH: applying to ALL protein residues.")
    else:
        apply_all = False
        target_res = set()
        for token in residue_tokens:
            token = token.strip()
            if not token:
                continue
            chain = token[0]
            resseq = "".join(ch for ch in token[1:] if ch.isdigit())
            if not resseq:
                msg = f"protect_sidechain_polarH: could not parse residue '{token}' (expected like A32); skipping."
                if wl:
                    wl.warn(msg, category="ncaa")
                else:
                    print(f"[WARN] {msg}", file=sys.stderr)
                continue
            target_res.add((chain, resseq))
        if wl:
            wl.info(f"protect_sidechain_polarH: applying to residues: {sorted(target_res)}")
        else:
            print(f"[INFO] protect_sidechain_polarH: applying to residues: {sorted(target_res)}")

    # Group per residue: collect heavy + H atoms
    residues = {}  # key: (chain, resseq, icode, resname) -> dict(heavy=[...], H=[...])
    for idx, line in enumerate(lines):
        if not line.startswith("ATOM  "):  # protein only
            continue
        resname = line[17:20].strip()
        if resname not in PROTEIN_RESNAMES:
            continue
        chain = line[21]
        resseq = line[22:26].strip()
        icode = (line[26].strip() or " ")
        if not (apply_all or (chain, resseq) in target_res):
            continue

        atom_name = line[12:16].strip()
        # Patch 1: safe element detection via parse_charge_fields
        elem = parse_charge_fields(line)[0]

        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue

        key = (chain, resseq, icode, resname)
        bucket = residues.setdefault(key, {"heavy": [], "H": []})
        if elem == "H":
            bucket["H"].append((idx, x, y, z, atom_name))
        else:
            bucket["heavy"].append((idx, x, y, z, atom_name, elem))

    # Decide which Hs to protect and build atom-name updates
    name_updates = {}
    protected_count = 0

    for (chain, resseq, icode, resname), grp in residues.items():
        heavy = grp["heavy"]
        hydrogens = grp["H"]
        if not heavy or not hydrogens:
            continue

        for h_idx, hx, hy, hz, h_name in hydrogens:
            # Find nearest heavy atom (ANY heavy), then decide if it's allowed
            best_d2 = None
            best_heavy_name = None
            best_heavy_elem = None
            for a_idx, ax, ay, az, a_name, a_elem in heavy:
                d2 = (hx - ax)**2 + (hy - ay)**2 + (hz - az)**2
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_heavy_name = a_name
                    best_heavy_elem = a_elem

            if best_heavy_name is None:
                continue

            # If the *nearest* heavy atom is carbon or backbone N/C/CA/O (except PRO N),
            # then do NOT treat this as a polar sidechain H -> skip it.
            if best_heavy_elem == "C":
                continue
            if best_heavy_name in ("N", "C", "CA", "O") and not (resname == "PRO" and best_heavy_name == "N"):
                continue

            # Try to canonicalize H name based on which heavy atom it is near
            canonical_base = h_name  # default: keep current
            polar_map = PROTECTED_POLAR_H_MAP.get(resname, {})
            canon_list = polar_map.get(best_heavy_name, [])

            if canon_list:
                if h_name in canon_list:
                    canonical_base = h_name
                elif len(canon_list) == 1:
                    canonical_base = canon_list[0]
                else:
                    msg = (f"protect_sidechain_polarH: {resname} {chain}{resseq}{icode} "
                           f"H '{h_name}' near {best_heavy_name} not recognized among "
                           f"canonical {canon_list}; leaving name as-is.")
                    if wl:
                        wl.warn(msg, category="ncaa")
                    else:
                        print(f"[WARN] {msg}")
            else:
                # No canonical mapping known for this heavy atom; warn once per H and keep name
                msg = (f"protect_sidechain_polarH: no canonical polar-H mapping for "
                       f"{resname} {chain}{resseq}{icode} heavy '{best_heavy_name}'; "
                       f"keeping H name '{h_name}'.")
                if wl:
                    wl.warn(msg, category="ncaa")
                else:
                    print(f"[WARN] {msg}")

            # Build Q* name (first letter forced to Q)
            base = canonical_base.strip()
            if not base:
                base = h_name.strip() or "H"
            if base[0] == "H":
                q_base = "Q" + base[1:]
            else:
                q_base = "Q" + base
            q_name = q_base[:4].rjust(4)

            name_updates[h_idx] = q_name
            protected_count += 1

    if protected_count:
        msg = f"protect_sidechain_polarH: protected {protected_count} sidechain polar hydrogens (H* -> Q*)."
        if wl:
            wl.info(msg)
        else:
            print(f"[INFO] {msg}")
    else:
        msg = "protect_sidechain_polarH: no sidechain polar hydrogens met criteria; nothing changed."
        if wl:
            wl.info(msg)
        else:
            print(f"[INFO] {msg}")

    # Apply name updates
    new_lines = []
    for idx, line in enumerate(lines):
        if idx in name_updates:
            # Patch 2: use Task 2 replace_field
            line = replace_field(line, 12, 16, name_updates[idx])
        new_lines.append(line)

    return new_lines


def revert_protected_sidechain_polarH(lines, wl=None):
    """
    Revert Q* sidechain polar hydrogens back to H* (Q -> H in atom name prefix).
    Any ATOM/HETATM whose atom name starts with 'Q' is mapped back to 'H'.
    Patch 2: use Task 2 replace_field.
    """
    new_lines = []
    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM")):
            new_lines.append(line)
            continue
        atom_name = line[12:16].strip()
        if atom_name.startswith("Q"):
            h_base = "H" + atom_name[1:]
            new_name = h_base[:4].rjust(4)
            line = replace_field(line, 12, 16, new_name)
        new_lines.append(line)
    return new_lines


def add_CA_to_labeled_frag(lines, ca_cb_bond_length=1.53, wl=None):
    """
    For canonical protein residues (PROTEIN_RESNAMES) that are missing a CA atom:
      - Require that a CB atom exists.
      - Require that at least one hydrogen is CB-bound (nearest heavy atom in same residue is CB).
      - For each CB-bound H, propose a CA position by extending CB->H to a typical CA-CB bond length.
      - Choose the candidate CA position that maximizes the minimum distance to all other heavy atoms
        (to minimize clashes).
      - Replace that H line with a CA carbon at the chosen coordinates.

    Patch 1: element detection via parse_charge_fields().
    Patch 2: use Task 2 replace_field.
    Patch 3: no-newline convention — pad to 80 but do NOT preserve/append newline char.
    Patch 4: print -> wl.info.
    """
    # Gather global heavy-atom coordinates for clash checking
    heavy_global = []
    for idx, line in enumerate(lines):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        # Patch 1: safe element detection
        elem = parse_charge_fields(line)[0]
        if elem == "H":
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        heavy_global.append((idx, x, y, z))

    # Group per canonical residue: only ATOM records, PROTEIN_RESNAMES
    residues = {}  # key: (chain, resseq, icode, resname) -> list of atom dicts
    for idx, line in enumerate(lines):
        if not line.startswith("ATOM  "):
            continue
        resname = line[17:20].strip()
        if resname not in PROTEIN_RESNAMES:
            continue
        chain = line[21]
        resseq = line[22:26].strip()
        icode = (line[26].strip() or " ")
        atom_name = line[12:16].strip()
        # Patch 1: safe element detection
        elem = parse_charge_fields(line)[0]
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        key = (chain, resseq, icode, resname)
        residues.setdefault(key, []).append({
            "idx": idx, "atom_name": atom_name, "elem": elem,
            "x": x, "y": y, "z": z
        })

    new_lines = list(lines)
    for (chain, resseq, icode, resname), atoms in residues.items():
        # Skip residues that already have CA
        if any(a["atom_name"] == "CA" for a in atoms):
            continue

        # Require CB
        cb_atoms = [a for a in atoms if a["atom_name"] == "CB"]
        if not cb_atoms:
            raise RuntimeError(
                f"add_CA_to_labeled_frag: residue {resname} {chain}{resseq}{icode} "
                f"is missing CA and CB; cannot construct CA.")

        cb = cb_atoms[0]
        heavy_same = [a for a in atoms if a["elem"] != "H"]
        H_atoms = [a for a in atoms if a["elem"] == "H"]

        if not H_atoms:
            raise RuntimeError(
                f"add_CA_to_labeled_frag: residue {resname} {chain}{resseq}{icode} "
                f"is missing CA and has no hydrogens; cannot construct CA.")

        # Identify hydrogens whose nearest heavy atom in this residue is CB
        cb_hydrogens = []
        for a in H_atoms:
            hx, hy, hz = a["x"], a["y"], a["z"]
            best_d2 = None
            best_heavy = None
            for h in heavy_same:
                dx = hx - h["x"]
                dy = hy - h["y"]
                dz = hz - h["z"]
                d2 = dx*dx + dy*dy + dz*dz
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_heavy = h
            if best_heavy and best_heavy["atom_name"] == "CB":
                cb_hydrogens.append(a)

        if not cb_hydrogens:
            raise RuntimeError(
                f"add_CA_to_labeled_frag: residue {resname} {chain}{resseq}{icode} "
                f"is missing CA and has no CB-bound hydrogens; cannot construct CA.")

        # For each CB-bound H, propose a CA position and score by min distance to other heavy atoms
        best_choice = None  # (h_idx, ca_x, ca_y, ca_z, min_d)
        for h in cb_hydrogens:
            hx, hy, hz = h["x"], h["y"], h["z"]
            dx = hx - cb["x"]
            dy = hy - cb["y"]
            dz = hz - cb["z"]
            d = math.sqrt(dx*dx + dy*dy + dz*dz)
            if d == 0.0:
                continue
            scale = ca_cb_bond_length / d
            ca_x = cb["x"] + dx * scale
            ca_y = cb["y"] + dy * scale
            ca_z = cb["z"] + dz * scale

            # Compute minimum distance to all other heavy atoms
            min_d = None
            for (g_idx, gx, gy, gz) in heavy_global:
                if g_idx == cb["idx"] or g_idx == h["idx"]:
                    continue
                ddx = ca_x - gx
                ddy = ca_y - gy
                ddz = ca_z - gz
                dist = math.sqrt(ddx*ddx + ddy*ddy + ddz*ddz)
                if min_d is None or dist < min_d:
                    min_d = dist

            if min_d is None:
                continue
            if best_choice is None or min_d > best_choice[4]:
                best_choice = (h["idx"], ca_x, ca_y, ca_z, min_d)

        if best_choice is None:
            raise RuntimeError(
                f"add_CA_to_labeled_frag: could not find a non-clashing CA placement "
                f"for residue {resname} {chain}{resseq}{icode}.")

        h_idx, ca_x, ca_y, ca_z, min_d = best_choice

        # Patch 3: no-newline convention — pad to >=80, no newline handling
        line = pad_line(new_lines[h_idx])

        # Patch 2: use Task 2 replace_field for all field updates
        ca_name_field = "CA".rjust(4)
        line = replace_field(line, 12, 16, ca_name_field)
        line = replace_field(line, 30, 38, f"{ca_x:8.3f}")
        line = replace_field(line, 38, 46, f"{ca_y:8.3f}")
        line = replace_field(line, 46, 54, f"{ca_z:8.3f}")
        line = replace_field(line, 76, 78, "C ")

        new_lines[h_idx] = line
        # Patch 4: route through wl if provided
        msg = (f"add_CA_to_labeled_frag: inserted CA for {resname} {chain}{resseq}{icode} "
               f"using former H at index {h_idx} (min heavy-atom distance ~ {min_d:.2f} A).")
        if wl:
            wl.info(msg)
        else:
            print(f"[INFO] {msg}")

    return new_lines


###############################################################################
# STRUCTURE SCANNER
###############################################################################

class Structure:
    def __init__(self):
        self.lines = []
        self.residues = {}      # (chain,resseq,icode) -> {resname, indices:[...], is_het}
        self.ligands = {}       # same key, HETATM & non-water non-protein
        self.ncaa = {}          # key -> resname (in NONCANONICAL_AA_MAP)
        self.remark666 = []     # [{chain,res,resseq,idx,var,ligand,raw}]
        self.remark665_header_present = False
        self.conect = []        # raw CONECT lines (index positions)
        self.master_index = None
        self.has_partial_charges = False
        self.has_formal_charges = False
        self.hydrogen_atom_indices = set()
        self.multi_model = False
        self.overflow_warn = []

R666_RE = re.compile(
    r"^REMARK 666 MATCH TEMPLATE\s+(\S+)\s+(\S+)\s+(\S+)\s+MATCH MOTIF\s+"
    r"(\S)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)")

def _is_r665_666_header(line):
    """Return True if this line is one of the two canonical REMARK 665 header
    lines that introduce the REMARK 666 block.  Used by scan_structure (to set
    remark665_header_present) and by remark666_manager trigger_g (to count
    how many such lines are present and detect missing or duplicated headers).
    Canonical predicate: starts with 'REMARK 665' AND contains 'REMARK 666'.
    """
    return line.startswith("REMARK 665") and "REMARK 666" in line


def _has_exact_r665_header_pair(lines):
    """Return True iff the EXACT two R665_HEADER strings are each present
    exactly once, and no other REMARK-665-containing-REMARK-666 lines exist.
    This is the DEFECT-2 exact-pair validation replacing the loose count-of-2.
    """
    # Count occurrences of each canonical line
    c0 = sum(1 for l in lines if l == R665_HEADER[0])
    c1 = sum(1 for l in lines if l == R665_HEADER[1])
    # Count all header-predicate-matching lines
    total = sum(1 for l in lines if _is_r665_666_header(l))
    return c0 == 1 and c1 == 1 and total == 2

def scan_structure(lines):
    s = Structure()
    s.lines = [l.rstrip("\n") for l in lines]
    seen_model = False
    for i, line in enumerate(s.lines):
        rec = line[:6]
        if rec == "MODEL ":
            if seen_model:
                s.multi_model = True
            seen_model = True
        # remark665_header_present is set after the loop (needs all lines)
        # to apply exact-pair validation (DEFECT 2 fix)
        if line.startswith("REMARK 666 MATCH TEMPLATE"):
            m = R666_RE.match(line)
            if m:
                s.remark666.append({
                    "tch": m.group(1), "ligand": m.group(2), "tresi": m.group(3),
                    "chain": m.group(4), "res": m.group(5), "resseq": m.group(6),
                    "idx": int(m.group(7)), "var": int(m.group(8)), "raw": line})
            else:
                s._malformed_666 = getattr(s, "_malformed_666", [])
                s._malformed_666.append(line)
        if rec == "CONECT":
            s.conect.append(i)
        if rec == "MASTER":
            s.master_index = i
        if is_atom_record(line):
            chain = line[21]
            resseq = get_field(line, 22, 26).strip()
            icode = (get_field(line, 26, 27).strip() or " ")
            resname = get_field(line, 17, 20).strip()
            key = (chain, resseq, icode)
            bucket = s.residues.setdefault(
                key, {"resname": resname, "resseq": resseq, "chain": chain,
                      "icode": icode, "indices": [], "is_het": rec == "HETATM"})
            bucket["indices"].append(i)
            elem, chg = parse_charge_fields(line)
            if elem == "H" or atom_name_field(line).strip().startswith("H"):
                s.hydrogen_atom_indices.add(i)
            bcol = get_field(line, 60, 66).strip()
            if bcol not in ("", "0.00", "0.0", "0"):
                try:
                    if abs(float(bcol)) > 0:
                        s.has_partial_charges = True
                except ValueError:
                    pass
            if chg:
                s.has_formal_charges = True
            try:
                if int(get_field(line, 6, 11)) > 99999:
                    s.overflow_warn.append(f"serial overflow near line {i}")
            except ValueError:
                pass
            if rec == "HETATM" and resname != "HOH":
                s.ligands[key] = bucket  # intentional alias: same dict object as s.residues[key]
            if resname in NONCANONICAL_AA_MAP:
                s.ncaa[key] = resname
    # DEFECT-2 fix: remark665_header_present uses exact-pair validation
    # (both canonical R665_HEADER strings, each exactly once, no extras).
    s.remark665_header_present = _has_exact_r665_header_pair(s.lines)
    return s


###############################################################################
# WARNING LOG
###############################################################################

class WarningLog:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.warnings = []  # list[(category, msg)]
    @property
    def count(self):
        return len(self.warnings)
    def info(self, msg):
        if self.verbose:
            print(f"[INFO] {msg}")
    def warn(self, msg, category="general"):
        self.warnings.append((category, msg))
        print(f"[WARN] {msg}")
    def render_summary(self):
        bar = "=" * 79
        lines = [bar, "  RUN SUMMARY", bar]
        if not self.warnings:
            lines.append("  No warnings. All operations completed cleanly.")
        else:
            lines.append(f"  {len(self.warnings)} warning(s):")
            for cat, msg in self.warnings:
                lines.append(f"   [{cat}] {msg}")
        lines.append(bar)
        return "\n".join(lines)


###############################################################################
# CLI PARSER + VALIDATION
###############################################################################

def parse_cli(argv=None):
    p = argparse.ArgumentParser(description="One-stop PDB→theozyme prep (preserve-by-default).")
    p.add_argument("--input_pdb", required=True)
    p.add_argument("--output_pdb_path", required=True)
    # REMARK 666
    p.add_argument("--force_regenerate_remark666", action="store_true")
    p.add_argument("--complete_remark666", action="store_true")
    p.add_argument("--remark666_exclude_residues", nargs="*", default=[])
    p.add_argument("--remark666_template_ligand", default=None)
    p.add_argument("--remark666_template_chain", default="X")
    p.add_argument("--remark666_template_resi", default="0")
    p.add_argument("--remark666_residue_front_order", nargs="*", default=[])
    p.add_argument("--remark666_residue_back_order", nargs="*", default=[])
    p.add_argument("--clean_remarks", action="store_true")
    # filtering
    p.add_argument("--residues_to_keep", nargs="*", default=None)
    p.add_argument("--residues_to_throw_away", nargs="*", default=None)
    p.add_argument("--ligands_to_keep", nargs="*", default=None)
    p.add_argument("--ligands_to_throw_away", nargs="*", default=None)
    # ncAA
    p.add_argument("--frag_ncAA_into_cAA_plus_lig", nargs="*", default=None)
    p.add_argument("--protect_ncAA_from_ligandization", action="store_true")
    p.add_argument("--leave_ncAA_as_ATOM", action="store_true")
    p.add_argument("--add_CA_to_labeled_frag", action="store_true")
    p.add_argument("--protect_sidechain_polarH", nargs="*", default=None)
    p.add_argument("--disable_intelligent_hstrip", action="store_true")
    # legacy opt-ins
    p.add_argument("--strip_insertion_codes", action="store_true")
    p.add_argument("--strip_protein_hydrogens", action="store_true")
    p.add_argument("--blank_segid", action="store_true")
    p.add_argument("--strip_partial_charges", action="store_true")
    p.add_argument("--strip_formal_charges", action="store_true")
    p.add_argument("--merge_ligands_as", default=None)
    p.add_argument("--merged_ligand_chain", default="Z")
    p.add_argument("--merged_ligand_resseq", default="999")
    p.add_argument("--merge_only", nargs="*", default=None)
    p.add_argument("--theozyme_layout", action="store_true")
    # renumber / checks / debug
    p.add_argument("--renumber_atoms", action="store_true")
    p.add_argument("--rosetta_residue_types",
        default=repo_paths.ROSETTA_RESIDUE_TYPES)
    p.add_argument("--ccd_timeout", type=float, default=4.0)
    p.add_argument("--no_ligand_code_checks", action="store_true")
    p.add_argument("--preserve_waters", action="store_true",
                   help="Treat HOH specially: always keep (bypass ligand "
                        "filters) and never merge. Default: HOH is a normal "
                        "HETATM ligand, subject to filtering/merging.")
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args(argv)

def _die(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(2)

def validate_options(args):
    if args.residues_to_keep is not None and args.residues_to_throw_away is not None:
        _die("--residues_to_keep and --residues_to_throw_away are mutually exclusive.")
    if args.ligands_to_keep is not None and args.ligands_to_throw_away is not None:
        _die("--ligands_to_keep and --ligands_to_throw_away are mutually exclusive.")
    if args.protect_ncAA_from_ligandization and args.leave_ncAA_as_ATOM:
        _die("--protect_ncAA_from_ligandization and --leave_ncAA_as_ATOM are mutually exclusive.")
    if args.frag_ncAA_into_cAA_plus_lig is not None and (
        args.protect_ncAA_from_ligandization or args.leave_ncAA_as_ATOM):
        _die("--frag_ncAA_into_cAA_plus_lig cannot combine with ncAA protect/leave.")
    if args.merge_ligands_as is not None:
        args.theozyme_layout = True
    return args


###############################################################################
# TASK 7: filter_structure (residue + ligand keep/throw, selector parsing)
###############################################################################

def parse_selector(tok):
    """'A55' -> ('A','55',None) ; 'Z:SUB:999' -> ('Z','999','SUB').
    Returns (chain, resseq, resname_or_None)."""
    tok = tok.strip()
    if ":" in tok:
        parts = tok.split(":")
        if len(parts) == 3:
            return parts[0], parts[2], parts[1]
        raise ValueError(f"bad selector '{tok}', expected CHAIN:RESNAME:RESSEQ")
    chain = tok[0]
    resseq = "".join(c for c in tok[1:] if c.isdigit())
    if not resseq:
        raise ValueError(f"bad selector '{tok}', expected like A169")
    return chain, resseq, None

def _match(line, sels):
    chain = line[21]
    resseq = get_field(line, 22, 26).strip()
    resname = get_field(line, 17, 20).strip()
    for c, rs, rn in sels:
        if c == chain and rs == resseq and (rn is None or rn == resname):
            return True
    return False

def filter_structure(lines, residues_to_keep, residues_to_throw_away,
                     ligands_to_keep, ligands_to_throw_away, wl,
                     preserve_waters=False):
    dropped = {"residues": set(), "ligands": set()}
    keep_sel = [parse_selector(t) for t in (residues_to_keep or [])]
    throw_sel = [parse_selector(t) for t in (residues_to_throw_away or [])]
    lkeep_sel = [parse_selector(t) for t in (ligands_to_keep or [])]
    lthrow_sel = [parse_selector(t) for t in (ligands_to_throw_away or [])]
    out = []
    for line in lines:
        if not is_atom_record(line):
            out.append(line); continue
        rec = line[:6]
        is_het = rec == "HETATM"
        resname = get_field(line, 17, 20).strip()
        chain = line[21]; resseq = get_field(line, 22, 26).strip()
        icode = get_field(line, 26, 27)
        key = (chain, resseq, icode if icode != " " else " ")
        # ligand record handling. By default HOH is a normal HETATM ligand
        # (subject to keep/throw selectors). --preserve_waters opts back into
        # the old special handling: HOH always kept, bypassing ligand filters.
        if is_het:
            if resname == "HOH" and preserve_waters:
                out.append(line); continue
            if lkeep_sel and not _match(line, lkeep_sel):
                dropped["ligands"].add(key); continue
            if lthrow_sel and _match(line, lthrow_sel):
                dropped["ligands"].add(key); continue
            out.append(line); continue
        # protein ATOM record
        if keep_sel:
            if _match(line, keep_sel):
                out.append(line)
            else:
                dropped["residues"].add(key)
            continue
        if throw_sel and _match(line, throw_sel):
            dropped["residues"].add(key); continue
        out.append(line)
    for k in sorted(dropped["residues"]):
        wl.warn(f"residue {k[0]}{k[1]} removed by filter", category="filter")
    for k in sorted(dropped["ligands"]):
        wl.warn(f"ligand {k[0]}{k[1]} removed by filter", category="filter")
    # NOTE: REMARK 666 lines are intentionally left untouched here.
    # filter_structure is ONLY responsible for ATOM/HETATM records.
    # remark666_manager is the SOLE owner of REMARK 666 lines; it receives
    # `dropped` and uses trigger_a to detect + drop stale anchors, emit
    # per-anchor warnings, and contiguously reindex survivors.
    return out, dropped


###############################################################################
# TASK 8: remark666_manager
###############################################################################

R665_HEADER = (
    "REMARK 665 REMARK 666 = Rosetta enzyme-matcher catalytic-motif anchors",
    "REMARK 665 fmt: REMARK 666 MATCH TEMPLATE <tCH tNAME tRESI> MATCH MOTIF <mCH mRESN mRESI IDX VAR>",
)

def _fmt_666(chain, res, resseq, idx, ligand, tch, tresi, var=1):
    return (f"REMARK 666 MATCH TEMPLATE {tch:<3}{ligand:<3}{int(tresi):>5} "
            f"MATCH MOTIF {chain} {res:<3} {int(resseq):>4}{idx:>4}{var:>4}")

def _norm_sel(tok):
    chain = tok[0]; rs = "".join(c for c in tok[1:] if c.isdigit())
    return (chain, rs)

def _partial_coverage_warning(s, entries, excl_set, wl):
    """Emit a [WARN] if any protein residue in the structure lacks a REMARK 666
    entry.  Shared by both the preserve-path and rebuild-path of
    remark666_manager so the logic cannot drift between the two callers.
    Returns the set of missing (chain, resseq) pairs (may be empty)."""
    prot = {}
    for (chain, resseq, icode), info in s.residues.items():
        if (not info["is_het"]) and info["resname"] in PROTEIN_RESNAMES:
            prot[(chain, resseq)] = info["resname"]
    covered = {(e["chain"], e["resseq"]) for e in entries}
    missing = [(c, rs) for (c, rs) in prot
               if (c, rs) not in covered and (c, rs) not in excl_set]
    if missing:
        wl.warn("partial REMARK 666 coverage: protein residues lacking a 666 "
                "line: " + ", ".join(f"{c}{rs}" for c, rs in sorted(missing)),
                category="remark666")
    return set(missing)

def remark666_manager(lines, s, dropped, args, wl):
    # 1. Collect existing parsed entries (dedupe by chain,resseq).
    seen = {}
    for e in s.remark666:
        seen.setdefault((e["chain"], e["resseq"]), e)
    has_duplicates = (len(seen) != len(s.remark666))
    has_malformed = bool(getattr(s, "_malformed_666", None))
    if has_duplicates:
        wl.warn("duplicate REMARK 666 entries de-duplicated by (chain,resseq)",
                category="remark666")
    if has_malformed:
        wl.warn(f"{len(s._malformed_666)} malformed REMARK 666 line(s) ignored",
                category="remark666")
    entries = list(seen.values())
    template = (args.remark666_template_ligand or
                (entries[0]["ligand"] if entries else None))
    tch = args.remark666_template_chain
    tresi = args.remark666_template_resi

    # --- Compute trigger conditions (a)-(g) for preserve-by-default ---
    # (a) filtering removed an anchored residue
    drop_keys = set()
    if dropped:
        drop_keys = {(c, rs) for (c, rs, _ic) in dropped.get("residues", set())}
    trigger_a = bool(drop_keys and any(
        (e["chain"], e["resseq"]) in drop_keys for e in entries))
    # (b) ordering flags given (non-empty)
    trigger_b = bool(args.remark666_residue_front_order or
                     args.remark666_residue_back_order)
    # (c) --remark666_exclude_residues matches an existing entry
    excl_set = {_norm_sel(t) for t in args.remark666_exclude_residues}
    trigger_c = bool(excl_set and any(
        (e["chain"], e["resseq"]) in excl_set for e in entries))
    # (d) --force_regenerate_remark666
    trigger_d = bool(args.force_regenerate_remark666)
    # (e) --complete_remark666
    trigger_e = bool(args.complete_remark666)
    # (f) duplicate or malformed existing 666 entries
    trigger_f = has_duplicates or has_malformed
    # (g) REMARK 665 header not exactly the canonical pair (DEFECT-2 fix: exact-pair check)
    trigger_g = not _has_exact_r665_header_pair(lines)
    # (h) DEFECT-1 fix: no existing 666 entries but protein residues present
    #     → fresh generation required even though no other trigger fired.
    prot_present = any(
        (not info["is_het"]) and info["resname"] in PROTEIN_RESNAMES
        for info in s.residues.values()
    )
    trigger_h = (len(entries) == 0 and prot_present
                 and not args.force_regenerate_remark666)

    any_trigger = (trigger_a or trigger_b or trigger_c or trigger_d or
                   trigger_e or trigger_f or trigger_g or trigger_h)

    # --- PRESERVE PATH: no trigger fired → return verbatim (but still warn) ---
    if not any_trigger:
        # Compute partial-coverage warning even in preserve path (informational).
        # Uses shared helper to prevent logic drift vs rebuild path.
        _partial_coverage_warning(s, entries, excl_set, wl)
        # DEFECT-3 fix: apply --clean_remarks even on preserve path.
        # Strips non-665/666 REMARK lines while keeping 665+666 intact.
        if getattr(args, "clean_remarks", False):
            return [l for l in lines
                    if not l.startswith("REMARK")
                    or l.startswith("REMARK 666")
                    or _is_r665_666_header(l)]
        # Return lines as-is; 665 header is already correct (trigger_g is False)
        return list(lines)

    # --- REBUILD PATH: at least one trigger fired ---

    # 2. Drop entries whose residue was removed by filtering.
    if drop_keys:
        kept = []
        for e in entries:
            if (e["chain"], e["resseq"]) in drop_keys:
                wl.warn(f"REMARK 666 for {e['chain']}{e['resseq']} dropped "
                        f"(residue removed); reindexing", category="remark666")
            else:
                kept.append(e)
        entries = kept

    # 3. Exclusion list.
    excl = excl_set
    present_keys = {(e["chain"], e["resseq"]) for e in entries}
    for c, rs in excl:
        if (c, rs) not in present_keys and not args.complete_remark666:
            wl.warn(f"--remark666_exclude_residues {c}{rs}: no current 666 line",
                    category="remark666")
    entries = [e for e in entries if (e["chain"], e["resseq"]) not in excl]

    # 4. Determine protein residues present in structure (for coverage/complete).
    prot = {}
    for (chain, resseq, icode), info in s.residues.items():
        if (not info["is_het"]) and info["resname"] in PROTEIN_RESNAMES:
            prot[(chain, resseq)] = info["resname"]

    force = args.force_regenerate_remark666
    if force:
        entries = []
        missing = [(c, rs) for (c, rs) in prot if (c, rs) not in excl]
    else:
        # Use shared helper to compute (and emit) partial-coverage warning.
        # This avoids duplicating the predicate logic vs the preserve path.
        # DEFECT-1 fix: suppress partial-coverage warning when trigger_h fired
        # (zero existing 666 → fresh generation will cover all residues anyway).
        missing_set = _partial_coverage_warning(s, entries, excl, wl) \
                      if (not args.complete_remark666 and not trigger_h) else set()
        # Build the list form needed by the complete path below.
        covered = {(e["chain"], e["resseq"]) for e in entries}
        missing = [(c, rs) for (c, rs) in prot if (c, rs) not in covered
                   and (c, rs) not in excl]

    # DEFECT-1 fix: fresh-generation when trigger_h fired (zero existing 666,
    # protein present, not force — force already cleared entries above).
    # Treat this identically to complete_remark666 for template inference + build.
    fresh_gen = args.complete_remark666 or force or trigger_h
    if fresh_gen:
        if template is None and len(prot) and not entries:
            # need a ligand token; infer from single ligand else error / warn
            lig_names = {info["resname"] for k, info in s.ligands.items()}
            if len(lig_names) == 1:
                template = next(iter(lig_names))
            elif len(lig_names) == 0:
                template = "LIG"
                wl.warn("no ligands in structure; using default template token 'LIG' "
                        "for fresh REMARK 666 generation", category="remark666")
            else:
                _die("ambiguous ligand for fresh REMARK 666 (multiple ligands present); "
                     "pass --remark666_template_ligand CODE")
        for (c, rs) in missing:
            entries.append({"chain": c, "res": prot[(c, rs)], "resseq": rs,
                            "ligand": template or "LIG", "idx": 0, "var": 1})

    # 5. Ordering: front (listed), middle (sorted), back (listed, NOT reversed).
    by_key = {(e["chain"], e["resseq"]): e for e in entries}
    front = [k for k in (_norm_sel(t) for t in args.remark666_residue_front_order)]
    back = [k for k in (_norm_sel(t) for t in args.remark666_residue_back_order)]
    for k in front + back:
        if k not in by_key:
            wl.warn(f"REMARK 666 order: {k[0]}{k[1]} not among entries; skipped",
                    category="remark666")
    front = [k for k in front if k in by_key]
    back = [k for k in back if k in by_key]
    if args.remark666_residue_back_order:
        wl.info("back_order applied in LISTED order (not reversed; differs "
                "from legacy STEP2).")
    mid = sorted([k for k in by_key if k not in front and k not in back],
                 key=lambda x: (x[0], int(x[1])))
    final = front + mid + back

    # 6. Re-emit with contiguous IDX, plus ensured single 665 header.
    new_666 = []
    if final and (template is not None):
        for i, k in enumerate(final, start=1):
            e = by_key[k]
            new_666.append(_fmt_666(e["chain"], e["res"], e["resseq"], i,
                                    e.get("ligand", template), tch, tresi,
                                    e.get("var", 1)))
    elif final and (template is None):
        # Anchors exist but no ligand template could be determined; warn instead
        # of silently dropping all anchors.
        anchor_list = ", ".join(f"{by_key[k]['chain']}{by_key[k]['resseq']}" for k in final)
        wl.warn(
            f"remark666_manager: {len(final)} anchor(s) ({anchor_list}) dropped "
            f"because no ligand template could be determined (no REMARK 666 entries "
            f"and --remark666_template_ligand not given); pass --remark666_template_ligand.",
            category="remark666")
    # Strip every existing REMARK 665(=666 header) + REMARK 666 line, reinsert.
    # Use _is_r665_666_header (shared predicate) for the 665-header strip.
    # DEFECT-3 fix: also drop all other REMARK lines when --clean_remarks.
    if getattr(args, "clean_remarks", False):
        body = [l for l in lines
                if not l.startswith("REMARK 666")
                and not _is_r665_666_header(l)
                and not l.startswith("REMARK")]
    else:
        body = [l for l in lines
                if not l.startswith("REMARK 666")
                and not _is_r665_666_header(l)]
    if not new_666:
        return body
    # Insert header+666 immediately before first ATOM/HETATM, else after last REMARK.
    insert_at = next((i for i, l in enumerate(body) if is_atom_record(l)), len(body))
    return body[:insert_at] + list(R665_HEADER) + new_666 + body[insert_at:]


###############################################################################
# TASK 9: apply_legacy_cleaning (opt-in strip H / segID / charges / merge)
###############################################################################

def apply_legacy_cleaning(lines, args, wl):
    out = list(lines)
    if args.strip_protein_hydrogens:
        n0 = len(out)
        kept = []
        for l in out:
            if l.startswith("ATOM  ") and atom_name_field(l).strip().startswith("H"):
                continue
            kept.append(l)
        out = kept
        wl.warn(f"stripped {n0-len(out)} protein hydrogen atom(s)", category="clean")
    if args.strip_partial_charges:
        out = [replace_field(l, 60, 66, "  0.00") if is_atom_record(l) else l
               for l in out]
        wl.warn("partial charges (B-factor col) zeroed", category="clean")
    if args.strip_formal_charges:
        out = [replace_field(l, 78, 80, "  ") if is_atom_record(l) else l
               for l in out]
        wl.warn("formal charge column (79-80) cleared", category="clean")
    if args.blank_segid:
        out = [replace_field(l, 72, 76, "    ") if is_atom_record(l) else l
               for l in out]
    if args.merge_ligands_as:
        out = _merge_ligands(out, args, wl)
    return out

def _merge_ligands(lines, args, wl):
    code = args.merge_ligands_as
    if len(code) != 3:
        wl.warn(f"merge ligand code '{code}' is not 3 chars", category="ligand")
    ch = args.merged_ligand_chain
    rs = f"{int(args.merged_ligand_resseq):>4}"
    sel = None
    if args.merge_only:
        sel = [parse_selector(t) for t in args.merge_only]
    out = []
    elem_counts = {}
    n = 0
    for l in lines:
        if not l.startswith("HETATM"):
            out.append(l); continue
        rn = get_field(l, 17, 20).strip()
        if rn == "HOH" and getattr(args, "preserve_waters", False):
            out.append(l); continue       # special handling opted in
        if sel is not None and not _match(l, sel):
            out.append(l); continue
        elem = parse_charge_fields(l)[0] or "X"
        elem_counts[elem] = elem_counts.get(elem, 0) + 1
        name = f"{elem}{elem_counts[elem]}"
        l = replace_field(l, 12, 16, f"{name:<4}"[:4])
        l = replace_field(l, 17, 20, f"{code:>3}")
        l = replace_field(l, 21, 22, ch[0])
        l = replace_field(l, 22, 26, rs)
        l = replace_field(l, 26, 27, " ")
        out.append(l); n += 1
    if n:
        wl.warn(f"merged {n} HETATM atom(s) into ligand '{code}' "
                f"({ch} {rs.strip()}); atom names regenerated (may break "
                f"pre-existing Rosetta params)", category="ligand")
    return out


###############################################################################
# TASK 10: connectivity_repair + renumber_atoms
###############################################################################

def _atom_serials(lines):
    """Return the set of valid atom serial numbers present in ATOM/HETATM records."""
    s = {}
    for l in lines:
        if is_atom_record(l):
            try:
                s[int(l[6:11])] = True
            except ValueError:
                pass
    return set(s)


def _recount_master(lines):
    """If a MASTER record exists, rewrite numCoord (cols 51-55, slice [50:55])
    to the current ATOM+HETATM count and numConect (cols 41-45, slice [40:45])
    to the current CONECT count.  Leave all other MASTER fields untouched.
    No-op if no MASTER record.  Returns possibly-updated lines.

    PDB MASTER field layout (0-based Python slices, PDB v3.3 spec):
      [0:6]   record name "MASTER"
      [6:10]  numRemark
      [10:15] 0 (always zero)
      [15:20] numHet
      [20:25] numHelix
      [25:30] numSheet
      [30:35] numTurn (deprecated)
      [35:40] numSite
      [40:45] numXform / numConect  ← CONECT count written here
      [45:50] numCoord field A (unused in our scheme)
      [50:55] numCoord (ATOM+HETATM)  ← atom count written here
      [55:60] numTer
      [60:65] numSeqres
    """
    natom = sum(1 for l in lines if is_atom_record(l))
    nconect = sum(1 for l in lines if l.startswith("CONECT"))
    new_lines = []
    for l in lines:
        if l.startswith("MASTER"):
            l = pad_line(l)
            l = replace_field(l, 40, 45, f"{nconect:>5}")
            l = replace_field(l, 50, 55, f"{natom:>5}")
        new_lines.append(l)
    return new_lines


def connectivity_repair(lines, wl):
    """Prune dangling CONECT references (atoms not in current serial set).
    Then unconditionally recounts MASTER (numCoord + numConect) via
    _recount_master so MASTER is always accurate even without --renumber_atoms.
    Returns (new_lines, changed_bool).  Safe to always run — cheap no-op when
    nothing is dangling.  Does NOT renumber; see renumber_atoms for that."""
    valid = _atom_serials(lines)
    out = []
    changed = False
    for l in lines:
        if l.startswith("CONECT"):
            toks = l.split()
            nums = [t for t in toks[1:] if t.isdigit()]
            if not nums or int(nums[0]) not in valid:
                # Primary atom not present → drop the whole CONECT record
                changed = True
                continue
            kept = [nums[0]] + [t for t in nums[1:] if int(t) in valid]
            if len(kept) != len(nums):
                changed = True
            out.append("CONECT" + "".join(f"{int(t):>5}" for t in kept))
        else:
            out.append(l)
    if changed:
        wl.warn("CONECT records repaired (dangling refs pruned)",
                category="connect")
    # Always recount MASTER so it reflects the current ATOM+HETATM and CONECT
    # counts, even on filter/merge runs where --renumber_atoms was not given.
    out = _recount_master(out)
    return out, changed


def renumber_atoms(lines, wl):
    """Renumber atom serial numbers (cols 7–11, i.e. [6:11]) sequentially,
    grouped by (chain, resseq, icode) in first-seen file order.
    Updates CONECT records via old→new mapping; prunes unmapped (dangling) refs.
    MASTER recount is delegated to _recount_master (called at end of
    connectivity_repair, which main() always runs after renumber_atoms).
    NEVER touches any other columns (col-16/17 region untouched — the old
    renumber_pdb.py UNL hack was a bug and is NOT reproduced here)."""
    # First pass: collect group order and old serial→new serial mapping
    order = []
    groups = {}
    for i, l in enumerate(lines):
        if is_atom_record(l):
            key = (l[21], get_field(l, 22, 26), get_field(l, 26, 27))
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(i)

    # Build old→new serial mapping and rewrite serial fields in-place
    mapping = {}
    new_serial = 1
    new_lines = list(lines)
    for key in order:
        for idx in groups[key]:
            old = None
            try:
                old = int(lines[idx][6:11])
            except ValueError:
                pass
            if old is not None:
                mapping[old] = new_serial
            new_lines[idx] = replace_field(lines[idx], 6, 11, f"{new_serial:>5}")
            new_serial += 1

    # Second pass: rewrite CONECT via mapping; prune dangling (unmapped) refs.
    # NOTE: MASTER recount is NOT done here — it is delegated to _recount_master
    # which is called unconditionally by connectivity_repair after renumber_atoms.
    out = []
    for l in new_lines:
        if l.startswith("CONECT"):
            toks = l.split()
            nums = [int(t) for t in toks[1:] if t.lstrip("-").isdigit()]
            mapped = [mapping[n] for n in nums if n in mapping]
            if len(mapped) >= 2:
                out.append("CONECT" + "".join(f"{m:>5}" for m in mapped))
            else:
                # Fewer than 2 surviving serials → whole record is dangling
                wl.warn("CONECT dropped during renumber (dangling)",
                        category="connect")
        else:
            out.append(l)

    wl.info(f"renumbered {new_serial - 1} atom serial(s) grouped by residue")
    return out


###############################################################################
# TASK 11: check_ligand_codes (always-on, non-fatal CCD + Rosetta checks)
###############################################################################

def _ccd_status(code, timeout):
    """Return 'exists' / 'missing' / 'unknown' via RCSB ligand CIF endpoint.
    Never raises; network errors or non-200/404 → 'unknown'."""
    url = f"https://files.rcsb.org/ligands/view/{code}.cif"
    try:
        import requests
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return "exists"
        if r.status_code == 404:
            return "missing"
        return "unknown"
    except Exception:
        return "unknown"


def _rosetta_status(code, path):
    """Return 'present' / 'absent' / 'unknown' by scanning residue_types.txt.
    'unknown' if the file path is absent/unreadable (non-fatal — offline or
    Rosetta not installed).  Matches: filename `CODE.params` OR standalone
    token `CODE` (word boundary).
    Returns a 2-tuple: (status_str, has_loose_match_bool).
    has_loose_match is True when a line contains both '.params' AND the code
    anywhere (even non-adjacent) without a stronger filename/standalone match.
    This mirrors the original check_if_ligand_3string_code_exists_in_rosetta.py
    [warning] category (DEFECT-5 fix)."""
    if not path or not os.path.isfile(path):
        return "unknown", False
    try:
        fn_re = re.compile(rf"{re.escape(code)}\.params\b")
        sa_re = re.compile(rf"\b{re.escape(code)}\b")
        strong_match = False
        loose_match = False
        with open(path) as fh:
            for raw in fh:
                has_fn = bool(fn_re.search(raw))
                has_sa = bool(sa_re.search(raw)) and not has_fn
                if has_fn or has_sa:
                    strong_match = True
                elif ".params" in raw and code in raw:
                    loose_match = True
        if strong_match:
            return "present", False
        return "absent", loose_match
    except Exception:
        return "unknown", False


def check_ligand_codes(ligand_codes, rosetta_path, ccd_timeout, wl):
    """Check each ligand code against PDB CCD and Rosetta residue_types.txt.
    Always-on but non-fatal: never sys.exit / raise.  Returns a dict:
      {code: {"ccd": "exists"|"missing"|"unknown",
              "rosetta": "present"|"absent"|"unknown"}}
    Each code gets ONE consolidated line stating BOTH databases. Problems
    (Rosetta collision, or a status that couldn't be verified) are wl.warn so
    they appear in SUMMARY; a code clear in both DBs is wl.info (not a scary
    SUMMARY warning). The loose .params match keeps its own category."""
    results = {}
    for code in sorted(set(ligand_codes)):
        if len(code) != 3:
            wl.warn(f"ligand code '{code}' is not 3 characters", category="ligand")
        ccd = _ccd_status(code, ccd_timeout)
        # DEFECT-5 fix: _rosetta_status now returns (status, has_loose_match)
        ros, ros_loose = _rosetta_status(code, rosetta_path)
        ros_status = ("present" if ros == "present"
                      else ("absent" if ros == "absent" else "unknown"))
        results[code] = {"ccd": ccd, "rosetta": ros_status}

        ccd_txt = {"exists": "in CCD",
                   "missing": "NOT in CCD",
                   "unknown": "CCD unknown (offline?)"}[ccd]
        ros_txt = {"present": "ALREADY in Rosetta DB",
                   "absent": "not in Rosetta DB",
                   "unknown": "Rosetta unknown (residue_types.txt "
                              "unreachable)"}[ros_status]
        line = f"ligand '{code}': {ccd_txt}, {ros_txt}"

        if ros_status == "present":
            wl.warn(f"{line} — NAME COLLISION RISK in Rosetta; pick a "
                    f"different code", category="ligand")
        elif ccd == "unknown" or ros_status == "unknown":
            wl.warn(f"{line} — could not fully verify; check manually before "
                    f"relying on this code", category="ligand")
        elif ccd == "missing" and ros_status == "absent":
            # the GOOD case: novel code, free to use — not a problem.
            wl.info(f"{line} — free/novel code, OK to use")
        else:  # in CCD, not in Rosetta — a normal real PDB component
            wl.info(f"{line} — OK (known PDB component, no Rosetta collision)")

        # DEFECT-5: loose .params-line match keeps its own non-fatal category
        if ros_loose:
            wl.warn(
                f"ligand '{code}' also appears on a .params line in Rosetta "
                f"DB (loose match — not a direct filename/standalone "
                f"collision, but verify manually to be safe)",
                category="rosetta_loose")
    return results


###############################################################################
# TASK 12: I/O helpers + _apply_theozyme_layout + main pipeline (DESIGN §4)
###############################################################################

def _read(path):
    """Read a PDB file; strip trailing newlines; return list of strings."""
    with open(path) as fh:
        return [l.rstrip("\n") for l in fh]


def _write(path, lines):
    """Write lines (no trailing newline on each) to path, adding \\n per line."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _apply_theozyme_layout(lines):
    """Reorder records into canonical theozyme layout while preserving ALL records.

    Output order:
      1. All records that are NOT (REMARK / ATOM / HETATM / TER / CONECT /
         MASTER / END), in their original relative order.  This preserves
         HEADER, CRYST1, LINK, MODEL/ENDMDL, SEQRES, SSBOND, HELIX, SHEET,
         ANISOU, etc. at the top.
      2. Non-665/666 REMARK lines (original order).
      3. REMARK 665/666 header lines + REMARK 666 anchor lines (original order).
      4. ATOM block.
      5. TER  (one, always — after the ATOM block).
      6. HETATM block (if non-empty).
      7. TER  (one, only if there was ≥1 HETATM; no spurious TER when empty).
      8. CONECT, then MASTER, then END — original order — at the tail.

    Original TER lines are replaced by the two structured TERs; all other
    record classes are preserved verbatim.

    Applied only when --theozyme_layout is given (auto-enabled by
    --merge_ligands_as)."""
    # Bucket each line
    top = []       # non-REMARK / non-atom / non-TER / non-tail records (HEADER, CRYST1, LINK, …)
    rem = []       # plain REMARK (non-665/666)
    hdr = []       # REMARK 665 header lines + REMARK 666 lines
    atoms = []     # ATOM
    hets = []      # HETATM
    tail = []      # CONECT / MASTER / END

    _tail_prefixes = ("CONECT", "MASTER", "END")
    _skip_prefixes = ("REMARK", "ATOM  ", "HETATM", "TER", "CONECT", "MASTER", "END")

    for l in lines:
        rec6 = l[:6]
        if rec6 == "ATOM  ":
            atoms.append(l)
        elif rec6 == "HETATM":
            hets.append(l)
        elif rec6 in ("CONECT", "MASTER"):
            tail.append(l)
        elif l == "END" or l.startswith("END"):
            tail.append(l)
        elif rec6 == "TER   " or l.startswith("TER"):
            pass  # original TER lines are dropped; structured TERs inserted below
        elif l.startswith("REMARK"):
            if _is_r665_666_header(l) or l.startswith("REMARK 666"):
                hdr.append(l)
            else:
                rem.append(l)
        else:
            top.append(l)

    out = top + rem + hdr + atoms + ["TER"]
    if hets:
        out += hets + ["TER"]
    out += tail
    return out


def main(argv=None):
    """Entry point — implements the DESIGN §4 canonical pipeline order:
    1. read + scan_structure (before any destructive transform)
    2. validate_options
    3. (opt) insertion-code strip
    4. ncAA transforms
    5. (opt) residue/ligand filtering
    6. (opt) legacy cleaning
    7. RE-SCAN → remark666_manager(lines, s, dropped, args, wl)
    8. connectivity_repair (always)
    9. (opt) renumber_atoms
    10. (opt) _apply_theozyme_layout
    11. ncAA reverts
    12. (unless --no_ligand_code_checks) check_ligand_codes
    13. write + print wl.render_summary()
    """
    args = parse_cli(argv)
    validate_options(args)
    wl = WarningLog(verbose=args.verbose)

    # Step 1: read + initial scan
    lines = _read(args.input_pdb)
    s0 = scan_structure(lines)
    if s0.multi_model:
        wl.warn("multi-MODEL PDB; all models processed as a flat atom list",
                category="structure")
    for w in s0.overflow_warn:
        wl.warn(w, category="structure")

    # Step 3: (opt) insertion-code strip (early; residue IDs must be stable
    # before filters and REMARK matching)
    if args.strip_insertion_codes:
        lines = [strip_insertion_code(l) for l in lines]
        wl.info("stripped residue insertion codes (opt-in).")

    # Step 4: ncAA transforms
    if args.frag_ncAA_into_cAA_plus_lig is not None:
        lines = frag_ncaa(lines,
                          build_frag_filters(args.frag_ncAA_into_cAA_plus_lig),
                          not args.disable_intelligent_hstrip, wl)
    protected = set()
    if args.protect_ncAA_from_ligandization or args.leave_ncAA_as_ATOM:
        lines, protected = convert_ncaa_hetatm_to_atom(lines, wl)
    if args.add_CA_to_labeled_frag:
        lines = add_CA_to_labeled_frag(lines, wl=wl)  # DEFECT-4 fix: pass wl as kwarg
    if args.protect_sidechain_polarH is not None:
        lines = protect_sidechain_polarH(lines, args.protect_sidechain_polarH, wl)

    # Step 5: (opt) residue/ligand filtering — before REMARK 666 so stale
    # anchors are detected by remark666_manager via `dropped`.
    dropped = None
    if any(x is not None for x in (args.residues_to_keep,
                                    args.residues_to_throw_away,
                                    args.ligands_to_keep,
                                    args.ligands_to_throw_away)):
        lines, dropped = filter_structure(
            lines,
            args.residues_to_keep, args.residues_to_throw_away,
            args.ligands_to_keep, args.ligands_to_throw_away,
            wl, preserve_waters=args.preserve_waters)

    # Step 6: (opt) legacy cleaning
    lines = apply_legacy_cleaning(lines, args, wl)

    # Step 7: RE-SCAN (after all residue/ligand mutations) then remark666_manager.
    # `dropped` is passed through so remark666_manager can detect dropped anchors
    # via trigger_a and emit per-anchor warnings + reindex IDX contiguously.
    # When no filtering happened, dropped=None (manager treats it as no drops).
    s = scan_structure(lines)
    lines = remark666_manager(lines, s, dropped, args, wl)

    # Step 9: (opt) renumber_atoms — remap CONECT serials before repair
    if args.renumber_atoms:
        lines = renumber_atoms(lines, wl)

    # Step 8/10: connectivity_repair (always; cheap no-op when nothing is
    # dangling).  Runs AFTER renumber_atoms so _recount_master sees the final
    # serial set and the final CONECT list.  On non-renumber runs this still
    # ensures MASTER reflects any atom/ligand removal done by filter/merge.
    lines, _ = connectivity_repair(lines, wl)

    # Step 10: (opt) theozyme layout reorder
    if args.theozyme_layout:
        lines = _apply_theozyme_layout(lines)

    # Step 11: ncAA reverts (after layout — reverts are invisible to 666/CONECT)
    if args.protect_ncAA_from_ligandization and protected:
        lines = revert_protected_ncaa_atom_to_hetatm(lines, protected)
    if args.protect_sidechain_polarH is not None:
        lines = revert_protected_sidechain_polarH(lines)

    # Step 12: (unless --no_ligand_code_checks) ligand code checks
    if not args.no_ligand_code_checks:
        s2 = scan_structure(lines)
        codes = sorted({info["resname"] for info in s2.ligands.values()})
        if args.merge_ligands_as:
            codes = sorted(set(codes) | {args.merge_ligands_as})
        check_ligand_codes(codes, args.rosetta_residue_types,
                           args.ccd_timeout, wl)

    # Step 13: write + summary
    _write(args.output_pdb_path, lines)
    wl.info(f"wrote {args.output_pdb_path}")
    print(wl.render_summary())


if __name__ == "__main__":
    main()
