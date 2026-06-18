#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Original filtering logic: Indrek Kalvet (ikalvet@uw.edu)
Reworked / refactored: Seth M. Woodbury

Post-process and filter aa_diffusion / HAL designs using PyRosetta.

High-level:
    - Takes diffusion outputs as CIF(.gz) plus companion JSON (TRB-like) files
    - Loads the corresponding reference PDB(s) used for diffusion
    - Builds a mapping between reference and design residues from diffused_index_map
    - Applies a set of geometric / environment filters to reject bad backbones
    - Optionally dumps only the “passing” designs as PDBs with annotations
    - Writes a Rosetta-style scorefile summarizing all per-design metrics

Main filters and metrics:
    - Backbone sanity:
        • max CA–CA distance across sequential residues (chainbreak)
        • min non-adjacent CA–CA distance (rCA_nonadj)
    - Ligand environment:
        • min backbone–ligand heavy-atom distance (lig_dist)
        • ligand SASA and relative burial vs. free ligand (SASA, SASA_rel)
        • optional SASA for specific ligand atoms (SASA_exposed_atoms)
        • distance from N- and C-termini to ligand (term_mindist)
    - Motif geometry (when tip-atom diffusion was used):
        • sidechain bond-length deviations vs. ideal (bondlen_dev)
        • cart_bonded and fa_dun scores at motif residues (cart_bonded_avg, fa_dun_avg)
    - Global scaffold quality:
        • loop fraction (loop_frac)
        • longest helix length (longest_helix)
        • radius of gyration (rog)
        • optional penalty if catalytic residues sit between loops (loop_at_motif)

Reference / design mapping:
    - Uses TRB["diffused_index_map"] to map reference residues (e.g. A94) to
      HAL numbering (e.g. A101), in both full and partial diffusion modes
    - Can restrict mapping to user-specified --ref_catres (e.g. “A94-96 B101”)
    - Optionally infers catalytic residues from REMARK 666 lines in the reference
      if --ref_catres is not provided

REMARK 666 handling:
    - Reads REMARK 666 MATCH TEMPLATE / MATCH MOTIF lines from the reference PDB
    - Remaps motif and template residues into HAL numbering using diffused_index_map
    - Re-inserts adjusted REMARK 666 lines into each passing design PDB
    - With --fix_unmatched_remark_lines_to_lig, attempts to repair “unmapped”
      MATCH TEMPLATE targets by assigning them to a unique residue (protein or
      ligand) in the design with the same residue name3, avoiding ambiguous cases

Other behavior:
    - Supports both full-protein diffusion and partial diffusion onto a fixed
      reference scaffold (with optional alignment + ligand grafting)
    - Uses multiprocessing to process many designs in parallel
    - Writes a single Rosetta-style scorefile (diffusion_analysis.sc by default)
      containing all metrics and pass/fail status for each backbone
"""


import os
import sys
import glob
import time
import json
import copy
import gzip
import itertools
import tempfile
import multiprocessing
import argparse
from shutil import copy2  # kept in case future logic relies on it

import numpy as np
import pandas as pd

import pyrosetta as pyr
import pyrosetta.rosetta
import pyrosetta.distributed.io
from pyrosetta.rosetta.core.scoring import score_type_from_name

# --- locate repo root + shared external paths ---
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "repo_paths.py").is_file():
        _sys.path.insert(0, str(_anc)); break
import repo_paths

# =============================================================================
# GLOBAL CONSTANTS & SIMPLE UTILITIES
# =============================================================================

aa3to1 = {
    "ALA": 'A', "ARG": 'R', "ASN": 'N', "ASP": 'D', "CYS": 'C',
    "GLN": 'Q', "GLU": 'E', "GLY": 'G', "HIS": 'H', "ILE": 'I',
    "LEU": 'L', "LYS": 'K', "MET": 'M', "PHE": 'F', "PRO": 'P',
    "SER": 'S', "THR": 'T', "TRP": 'W', "TYR": 'Y', "VAL": 'V'
}
aa1to3 = {val: k for k, val in aa3to1.items()}

# Toggle this manually if you want extra debug prints
DEBUG = True


def debug_print(msg: str):
    """Print debug information if DEBUG is enabled."""
    if DEBUG:
        print(f"[DEBUG] {msg}")


# =============================================================================
# I/O & POSE-LEVEL UTILITIES
# =============================================================================

def load_cif_to_pose(cif_file: str) -> pyr.Pose:
    """
    Load a CIF (or CIF.GZ) into a PyRosetta Pose, preserving fold tree.

    Steps:
      - Reads the CIF or CIF.GZ
      - Appends an empty citation title (to avoid certain CIF issues)
      - Uses Rosetta's CIF importer
      - Rebuilds a canonical pose chain-by-chain
    """
    debug_print(f"Loading CIF file into pose: {cif_file}")

    if cif_file.endswith(".cif"):
        with open(cif_file, "r") as fh:
            lines = fh.readlines()
    elif cif_file.endswith(".cif.gz"):
        with gzip.open(cif_file, "rt") as fh:
            lines = fh.readlines()
    else:
        raise ValueError(f"Unsupported file extension for CIF input: {cif_file}")

    lines.append('_citation.title  ""\n')

    tempcif = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + ".cif")
    with open(tempcif, "w") as fh:
        fh.write("".join(lines))

    pose = pyrosetta.rosetta.core.import_pose.pose_from_file(
        tempcif,
        read_fold_tree=True,
        type=pyrosetta.rosetta.core.import_pose.FileType.CIF_file
    )

    pose2 = pyrosetta.rosetta.core.pose.Pose()
    for chain in range(1, pose.num_chains() + 1):
        pyrosetta.rosetta.core.pose.append_subpose_to_pose(
            pose2, pose, pose.chain_begin(chain), pose.chain_end(chain), True
        )

    pdb_string = pyrosetta.distributed.io.to_pdbstring(pose2)
    pose3 = pyr.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose3, pdb_string)

    os.remove(tempcif)
    debug_print(f"Finished loading CIF into pose: {cif_file}")
    return pose3


def add_matcher_line_to_pose(pose, ref_pose, tgt_residues, ref_residues):
    """
    Adjust and insert REMARK 666 lines into the given pose.

    Parameters
    ----------
    pose : pyrosetta.Pose
        Target pose where REMARK 666 lines should be inserted.
    ref_pose : pyrosetta.Pose
        Reference pose (unused here but kept for interface consistency).
    tgt_residues : dict
        Dictionary keyed by residue number in the target pose; each value is
        the REMARK 666 info for that residue (chain, name3, target info, etc.).
    ref_residues : dict
        Original REMARK 666 dictionary from the reference PDB.
    """
    if len(tgt_residues) == 0:
        debug_print("No target residues provided for matcher remark insertion; returning original pose.")
        return pose

    ligand_name = pose.residue(pose.size()).name3()
    debug_print(f"Inserting REMARK 666 lines into pose for ligand name3={ligand_name}, n_lines={len(tgt_residues)}")

    new_remarks = []
    for resno, info in tgt_residues.items():
        new_remarks.append(
            f"REMARK 666 MATCH TEMPLATE {info['target_chain']} {info['target_name']}  {info['target_resno']:>3} MATCH MOTIF {info['chain']} {info['name3']}  {resno:>3}  {info['cst_no']}  {info['cst_no_var']}               \n"
        )

    pdb_str = pyrosetta.distributed.io.to_pdbstring(pose)
    pdb_lines = pdb_str.split("\n")

    new_pdb_lines = []
    if "ATOM" in pdb_lines[0]:
        for lr in new_remarks:
            new_pdb_lines.append(lr)
        new_pdb_lines.extend(pdb_lines)
    else:
        for line in pdb_lines:
            if "HEADER" in line:
                new_pdb_lines.append(line)
                for lr in new_remarks:
                    new_pdb_lines.append(lr)
            elif "REMARK 666" in line:
                continue
            else:
                new_pdb_lines.append(line)

    pose2 = pyr.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose2, "\n".join(new_pdb_lines))
    debug_print("Finished inserting REMARK 666 lines into pose.")
    return pose2


def get_matcher_residues(filename: str) -> dict:
    """
    Parse REMARK 666 matcher lines from a PDB and return a dictionary:

    {
      resno: {
        'target_name': ...,
        'target_chain': ...,
        'target_resno': ...,
        'chain': ...,
        'name3': ...,
        'cst_no': ...,
        'cst_no_var': ...
      },
      ...
    }
    """
    matches = {}
    with open(filename, "r") as fh:
        for line in fh:
            if "ATOM" in line:
                break
            if "REMARK 666" in line:
                lspl = line.split()
                resno = int(lspl[11])
                matches[resno] = {
                    "target_name": lspl[5],
                    "target_chain": lspl[4],
                    "target_resno": int(lspl[6]),
                    "chain": lspl[9],
                    "name3": lspl[10],
                    "cst_no": int(lspl[12]),
                    "cst_no_var": int(lspl[13]),
                }
    debug_print(f"Parsed {len(matches)} matcher residues from {filename}")
    return matches


# =============================================================================
# GEOMETRY / METRIC UTILITIES
# =============================================================================

def getSASA(pose, resno=None, SASA_atoms=None, ignore_sc=False):
    """
    Calculate SASA for a pose, a residue, or selected atoms in a residue.

    Returns either a surf_vol object (if resno is None) or a float.
    """
    atoms = pyr.rosetta.core.id.AtomID_Map_bool_t()
    atoms.resize(pose.size())

    for i, res in enumerate(pose.residues):
        if res.is_ligand():
            atoms.resize(i + 1, res.natoms(), True)
        else:
            atoms.resize(i + 1, res.natoms(), not ignore_sc)
            if ignore_sc:
                for n in range(1, res.natoms() + 1):
                    if res.atom_is_backbone(n) and not res.atom_is_hydrogen(n):
                        atoms[i + 1][n] = True

    surf_vol = pyr.rosetta.core.scoring.packing.get_surf_vol(pose, atoms, 1.4)

    if resno is not None:
        if isinstance(resno, int):
            res_surf = 0.0
            for i in range(1, pose.residue(resno).natoms() + 1):
                if SASA_atoms is not None and i not in SASA_atoms:
                    continue
                res_surf += surf_vol.surf(resno, i)
            return res_surf

        elif isinstance(resno, list):
            res_surf = 0.0
            for rn in resno:
                for i in range(1, pose.residue(rn).natoms() + 1):
                    if SASA_atoms is not None and i not in SASA_atoms:
                        continue
                    res_surf += surf_vol.surf(rn, i)
            return res_surf

    return surf_vol


def get_ROG(pose) -> float:
    """Compute a simple radius of gyration measure based on protein CA atoms only."""
    ca_coords = [res.xyz("CA") for res in pose.residues if res.is_protein()]
    if not ca_coords:
        return 0.0
    centroid = np.array([np.mean([c.__getattribute__(axis) for c in ca_coords]) for axis in "xyz"])
    ROG = max(np.linalg.norm(centroid - np.array([c.x, c.y, c.z])) for c in ca_coords)
    return float(ROG)


def sidechain_connectivity(res):
    """
    Evaluate the physical correctness of sidechain bond lengths of a residue.

    Compares distances against a reference A-X-A pose (same residue type) and
    returns the maximum absolute deviation of bond lengths.
    """
    ref_res_pose = pyr.pose_from_sequence("A" + res.name1() + "A")
    ref_res = ref_res_pose.residue(2)
    bondlen_deviations = []

    for an in range(1, res.natoms() + 1):
        if res.atom_type(an).element() == "H":
            continue
        for nn in res.bonded_neighbor(an):
            if res.atom_type(nn).element() == "H":
                continue
            bondlen_deviations.append(abs((res.xyz(an) - res.xyz(nn)).norm() - (ref_res.xyz(an) - ref_res.xyz(nn)).norm()))

    return max(bondlen_deviations) if bondlen_deviations else 0.0


def thread_seq_to_pose(pose, sequence, skip_resnos=None):
    """
    Simple sequence threading onto a pose backbone, skipping ligand residues
    and any positions specified in skip_resnos.
    """
    if skip_resnos is None:
        skip_resnos = []

    pose2 = pose.clone()
    for i, aa in enumerate(sequence):
        seqpos = i + 1
        if seqpos in skip_resnos:
            continue
        if pose.residue(seqpos).is_ligand():
            continue

        mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
        mutres.set_target(seqpos)
        mutres.set_res_name(aa1to3[aa])
        mutres.apply(pose2)

    return pose2


def get_rosetta_scores(pose, sfx, sfx_cart, catres_list):
    """
    Get Rosetta per-residue sidechain quality scores (cart_bonded and fa_dun)
    for a set of residues in catres_list.
    """
    sfx(pose)
    sfx_cart(pose)

    scoredict = {}
    for term in ["cart_bonded", "fa_dun"]:
        scoredict[term] = {
            res.seqpos(): pose.energies().residue_total_energies(res.seqpos()).get(score_type_from_name(term))
            for res in pose.residues
            if res.seqpos() in catres_list
        }

    averages = {term: np.average(list(scores.values())) if scores else 0.0 for term, scores in scoredict.items()}
    return averages, scoredict


# =============================================================================
# SCOREFILE I/O
# =============================================================================

def dump_scorefile(df: pd.DataFrame, filename: str):
    """
    Dump a Rosetta-style scorefile from a DataFrame.
    """
    widths = {}
    for k in df.keys():
        if k in ["SCORE:", "description", "name"]:
            widths[k] = 0
        elif len(k) >= 12:
            widths[k] = len(k) + 1
        else:
            widths[k] = 12

    with open(filename, "w") as fh:
        title = ""
        for k in df.keys():
            if k == "SCORE:":
                title += k
            elif k in ["description", "name"]:
                title += f" {k}"
            else:
                title += f"{k:>{widths[k]}}"
        if all([t not in df.keys() for t in ["description", "name"]]):
            title += " description"
        fh.write(title + "\n")

        for index, row in df.iterrows():
            line = ""
            for k in df.keys():
                if isinstance(row[k], (float, np.floating)):
                    val = f"{row[k]:.3f}"
                else:
                    val = row[k]
                if k == "SCORE:":
                    line += val
                elif k in ["description", "name"]:
                    line += f" {val}"
                else:
                    line += f"{val:>{widths[k]}}"
            if all([t not in df.keys() for t in ["description", "name"]]):
                line += f" {index}"
            fh.write(line + "\n")


# =============================================================================
# MAIN PIPELINE HELPERS
# =============================================================================

def parse_ref_catres(ref_catres_args):
    """
    Parse reference catalytic residue identifiers.

    Accepts a list like: ["A94-96", "B101"] and returns a flat list:
    ["A94", "A95", "A96", "B101"]
    """
    ref_catres = []
    if ref_catres_args is None:
        return ref_catres

    for r in ref_catres_args:
        if "-" in r:
            start_token, end_token = r.split("-")[0], r.split("-")[1]
            ch = r[0]
            for n in range(int(start_token[1:]), int(end_token) + 1):
                ref_catres.append(f"{ch}{n}")
        else:
            ref_catres.append(r)
    return ref_catres


def resolve_pdbfiles(args):
    """
    Resolve the list of design CIF/CIF.GZ files from --pdb or --pdbpath.
    """
    assert any([x is not None for x in [args.pdb, args.pdbpath]]), "Need to provide either --pdb or --pdbpath"

    if args.pdb is not None:
        pdbfiles = args.pdb
    else:
        pdbfiles = []
        for pth in args.pdbpath:
            pdbfiles.extend(glob.glob(os.path.join(pth, "*.cif.gz")))
    pdbfiles = sorted(pdbfiles)
    debug_print(f"Resolved {len(pdbfiles)} design CIF files.")
    return pdbfiles


def resolve_ref_pdbs(args):
    """
    Resolve reference PDBs either from:
      - --ref
      - --ref_path
      - or defer to JSON ("specification.input") if both are None.
    """
    if args.ref is not None:
        debug_print(f"Using explicitly provided reference PDBs (--ref), n={len(args.ref)}")
        return args.ref
    elif args.ref_path is not None:
        ref_pdbs = sorted(glob.glob(os.path.join(args.ref_path, "*.pdb")))
        debug_print(f"Found {len(ref_pdbs)} reference PDBs in --ref_path.")
        return ref_pdbs
    else:
        debug_print("No --ref or --ref_path given; will use TRB 'specification.input'.")
        return None


def ensure_output_dir(path):
    """Create output directory if it doesn't exist (ignore permission errors)."""
    try:
        if not os.path.exists(path):
            os.mkdir(path)
            debug_print(f"Created output directory: {path}")
    except PermissionError:
        print(f"WARNING: Could not create output directory (permission denied): {path}")


def init_pyrosetta_with_params(params):
    """
    Initialize PyRosetta with provided params and DAlphaBall path.
    """
    extra_res_fa = ""
    if params is not None:
        extra_res_fa = "-extra_res_fa"
        for p in params:
            extra_res_fa += f" {p}"

    DAB = repo_paths.DALPHABALL
    if not os.path.exists(DAB):
        DAB = None

    assert DAB is not None, (
        "Please compile DAlphaBall.gcc and manually provide a path to it in this script under the variable `DAB`.\n"
        "For more info on DAlphaBall, visit: "
        "https://www.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Filters/HolesFilter"
    )

    init_flags = f"{extra_res_fa} -mute all -dalphaball {DAB} -run:preserve_header -in:fast_restyping true"
    print(f"Initializing PyRosetta with flags: {init_flags}")
    pyr.init(init_flags)


def compute_datacolumns(args):
    """
    Build the list of data columns for the final DataFrame.
    """
    columns = ["chainbreak", "rCA_nonadj", "lig_dist", "bondlen_dev"]
    if args.cart_bonded is not None:
        columns.append("cart_bonded_avg")
    if args.fa_dun is not None:
        columns.append("fa_dun_avg")

    columns.extend(["loop_frac", "longest_helix", "rog"])

    if args.loop_catres is True:
        columns.append("loop_at_motif")

    columns.append("term_mindist")
    columns.extend(["SASA", "SASA_rel"])

    if args.ligand_exposed_atoms is not None:
        columns.append("SASA_exposed_atoms")

    debug_print(f"Initial datacolumns: {columns}")
    return columns


def determine_num_processes(args):
    """
    Determine number of CPU cores to use, preserving the original search order.
    """
    if args.nproc is not None:
        return args.nproc
    if "SLURM_CPUS_ON_NODE" in os.environ:
        return int(os.environ["SLURM_CPUS_ON_NODE"])
    if "OMP_NUM_THREADS" in os.environ:
        return int(os.environ["OMP_NUM_THREADS"])
    return os.cpu_count()


def load_trb(jsonfile, design_name):
    """Safe JSON load with diagnostics."""
    try:
        with open(jsonfile, "r") as fh:
            trb = json.load(fh)
        debug_print(f"Loaded TRB JSON for {design_name}: {jsonfile}")
        return trb
    except Exception as e:
        print(f"[ERROR] Failed to load TRB JSON for {design_name} at {jsonfile}: {e}")
        return None


def select_reference_pdb(trb, ref_pdbs_local, pdbfile, ref_poses_local):
    """
    Decide which reference PDB to use for a given design and cache the Pose.
    """
    if ref_pdbs_local is not None:
        matching_refs = [
            r for r in ref_pdbs_local
            if os.path.basename(r).replace(".cif.gz", "_") in os.path.basename(pdbfile)
        ]
        if len(matching_refs) != 1:
            print(f"[ERROR] Bad number of reference PDBs found for {pdbfile}: matches={matching_refs}")
            return None, None
        ref_pdb = matching_refs[0]
    else:
        ref_pdb = trb["specification"]["input"]

    if ref_pdb not in ref_poses_local.keys():
        debug_print(f"Loading reference PDB into pose cache: {ref_pdb}")
        with open(ref_pdb, "r") as fh:
            pdb_lines = [l for l in fh if "ORI" not in l]
        _pose = pyrosetta.rosetta.core.pose.Pose()
        pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(_pose, "".join(pdb_lines))
        if len(_pose.sequence()) == 0:
            print(f"[ERROR] Reference pose appears to be empty for {ref_pdb}")
            return None, None
        ref_poses_local[ref_pdb] = _pose.clone()

    return ref_pdb, ref_poses_local[ref_pdb].clone()


def build_fixed_positions(trb, args, ref_catres, matched_residues, ref_pose):
    """
    Build mapping between reference residues and HAL (design) residues.

    Returns
    -------
    fixed_positions_from_JSON : dict[str, str]
        Mapping from ref (e.g. 'A106') -> hal (e.g. 'A42')
    fixed_pos_in_hal : list[int]
        HAL residue indices (1-based, no chain prefix).
    fixed_pos_in_ref : list[tuple[str,int]]
        Reference (chain, resno) pairs.
    _ref_catres : list[str]
        Effective catalytic residues used for mapping.
    """
    if args.partial:
        # ------------------------ PARTIAL DIFFUSION -------------------------
        debug_print("Building fixed position mapping in PARTIAL diffusion mode.")
        if "diffused_index_map" not in trb:
            fixed_positions_from_JSON = {}
            contig = trb["specification"]["contig"]
            debug_print(f"No diffused_index_map found; building from contig={contig}")
            for x in contig.split(","):
                if not x or not x[0].isalpha():
                    continue
                if "-" in x:
                    start_token, end_token = x.split("-")[0], x.split("-")[1]
                    ch = x[0]
                    for n in range(int(start_token[1:]), int(end_token) + 1):
                        fixed_positions_from_JSON[f"{ch}{n}"] = f"{ch}{n}"
                else:
                    fixed_positions_from_JSON[x] = x
        else:
            fixed_positions_from_JSON = trb["diffused_index_map"]

        fixed_pos_in_hal = [int(x[1:]) for x in fixed_positions_from_JSON.values()]
        fixed_pos_in_ref = [(x[0], int(x[1:])) for x in fixed_positions_from_JSON.keys()]
        _ref_catres = [f"{ch}{rn}" for ch, rn in fixed_pos_in_ref]
        print(f"### REFERENCE CATALYTIC RESIDUES (PARTIAL, V2) ###\n{_ref_catres}\n")

    else:
        # ------------------------- FULL DIFFUSION ---------------------------
        debug_print("Building fixed position mapping in FULL diffusion mode.")
        full_map = trb["diffused_index_map"]  # all diffused residues from JSON

        if args.ref_catres is not None:
            # Enforce that --ref_catres is a subset of diffused_index_map keys
            requested = list(args.ref_catres)
            available_keys = set(full_map.keys())
            missing = [r for r in requested if r not in available_keys]

            if missing:
                msg = (
                    "ERROR: The following --ref_catres entries are NOT present in "
                    "diffused_index_map (TRB['diffused_index_map']) and this is "
                    "not allowed:\n"
                    f"    {', '.join(sorted(missing))}\n"
                    "Every residue passed in --ref_catres must appear as a key in "
                    "diffused_index_map. Please fix your input or the TRB and rerun."
                )
                print(msg)
                sys.exit(1)

            # Use ONLY the subset of diffused_index_map that was explicitly requested
            fixed_positions_from_JSON = {
                k: v for k, v in full_map.items() if k in requested
            }

            # Preserve user-specified order for catalytic residues
            _ref_catres = requested.copy()
            print("### REFERENCE CATALYTIC RESIDUES (FULL, from --ref_catres) ###")
            print(f"{_ref_catres}\n")

        else:
            # No explicit --ref_catres: keep original behavior
            fixed_positions_from_JSON = full_map
            fixed_pos_in_hal_tmp = [int(x[1:]) for x in fixed_positions_from_JSON.values()]
            fixed_pos_in_ref_tmp = [(x[0], int(x[1:])) for x in fixed_positions_from_JSON.keys()]

            _ref_catres = ref_catres.copy()
            print(f"### REFERENCE CATALYTIC RESIDUES (FULL, V3 - initial) ###\n{_ref_catres}\n")

            # Optional: infer from REMARK 666 if user did not pass --ref_catres
            if len(_ref_catres) == 0 and len(matched_residues) > 0:
                inferred = []
                pdbinfo = ref_pose.pdb_info()

                for rn, d in matched_residues.items():
                    chain = d["chain"]  # e.g. 'A'

                    # Map (chain, PDB residue number) -> Rosetta seqpos
                    seqpos = None
                    for i in range(1, ref_pose.size() + 1):
                        if pdbinfo.chain(i) == chain and pdbinfo.number(i) == rn:
                            seqpos = i
                            break

                    if seqpos is None:
                        debug_print(
                            f"[build_fixed_positions] WARNING: "
                            f"could not map REMARK 666 residue {chain}{rn} into ref_pose; skipping."
                        )
                        continue

                    if ref_pose.residue(seqpos).is_protein():
                        inferred.append(f"{chain}{rn}")

                if inferred:
                    _ref_catres = inferred
                    print("### REFERENCE CATALYTIC RESIDUES (FULL, V4 - inferred from REMARK 666) ###")
                    print(f"{_ref_catres}\n")
                else:
                    print("### WARNING: No protein REMARK 666 residues found to infer catalytic residues; using empty list. ###")
                    _ref_catres = []

        # Now that fixed_positions_from_JSON is finalized (either subset or full),
        # compute the numeric lists.
        fixed_pos_in_hal = [int(x[1:]) for x in fixed_positions_from_JSON.values()]
        fixed_pos_in_ref = [(x[0], int(x[1:])) for x in fixed_positions_from_JSON.keys()]

    debug_print(f"Fixed positions from JSON (n={len(fixed_positions_from_JSON)}): {fixed_positions_from_JSON}")
    return fixed_positions_from_JSON, fixed_pos_in_hal, fixed_pos_in_ref, _ref_catres


def compute_backbone_metrics(pose, scores_entry, args, pdbfile, pose_label="pose"):
    """
    Compute chainbreak and non-adjacent CA-CA distances, update scores_entry.

    Returns True if passes filters, False if should be filtered out.
    """
    # Chainbreak
    dists = []
    for n in range(1, pose.size()):
        if pose.residue(n).is_ligand():
            continue
        if pose.residue(n + 1).is_ligand():
            continue
        if pose.chain(n) != pose.chain(n + 1):
            continue
        dists.append((pose.residue(n).xyz("CA") - pose.residue(n + 1).xyz("CA")).norm())
    if not dists:
        print(f"[WARNING] No chainbreak distances computed for {pdbfile} ({pose_label}); skipping chainbreak filter.")
        scores_entry["chainbreak"] = np.nan
    else:
        scores_entry["chainbreak"] = max(dists)
        if not args.analyze and scores_entry["chainbreak"] > 4.5:
            print(f"{pdbfile}: chainbreak found! max_CA_CA={scores_entry['chainbreak']:.2f}")
            return False

    # Non-adjacent CA-CA
    nonadjacentCAs = []
    for (r1, r2) in itertools.combinations(pose.residues, 2):
        if r1.is_ligand() or r2.is_ligand():
            continue
        if r1.is_virtual_residue() or r2.is_virtual_residue():
            continue
        if not r1.is_protein() or not r2.is_protein():
            continue
        if abs(r1.seqpos() - r2.seqpos()) == 1:
            continue
        nonadjacentCAs.append((r1.xyz("CA") - r2.xyz("CA")).norm())
    if not nonadjacentCAs:
        print(f"[WARNING] No non-adjacent CA-CA distances computed for {pdbfile} ({pose_label}); skipping rCA_nonadj filter.")
        scores_entry["rCA_nonadj"] = np.nan
    else:
        scores_entry["rCA_nonadj"] = min(nonadjacentCAs)
        if not args.analyze and scores_entry["rCA_nonadj"] < 3.0:
            print(f"{pdbfile}: some residues are too close to each other: min_CA_CA={scores_entry['rCA_nonadj']:.2f}")
            return False

    return True


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main(args):
    """
    Main entry point.

    Steps:
      1) Resolve input design CIF files and reference PDBs
      2) Initialize PyRosetta & scoring functions
      3) Build multiprocessing queue & shared data structures
      4) Spawn workers that parse and filter each design (via Pool initializer loop)
      5) Aggregate scores into a DataFrame and write a scorefile
    """
    pdbfiles = resolve_pdbfiles(args)
    params = args.params

    ref_pdbs = resolve_ref_pdbs(args)
    ref_catres = parse_ref_catres(args.ref_catres)

    if args.ligand_exposed_atoms is not None and args.exposed_atom_SASA is None:
        sys.exit("Defined --ligand_exposed_atoms but not --exposed_atom_SASA")
    if args.exposed_atom_SASA is not None and args.ligand_exposed_atoms is None:
        sys.exit("Defined --exposed_atom_SASA but not --ligand_exposed_atoms")

    filtered_dir = args.outdir
    ensure_output_dir(filtered_dir)

    init_pyrosetta_with_params(params)

    # ImportPose options (currently unused but kept for future flexibility)
    opts = pyrosetta.rosetta.core.import_pose.ImportPoseOptions()
    opts.set_fast_restyping(False)

    sfx = pyr.get_fa_scorefxn()
    sfx_cart = sfx.clone()
    sfx_cart.set_weight(score_type_from_name("cart_bonded"), 0.5)
    sfx_cart.set_weight(score_type_from_name("pro_close"), 0.0)

    start = time.time()

    the_queue = multiprocessing.Queue()
    manager = multiprocessing.Manager()

    ref_poses = manager.dict()   # reference pose cache across workers
    scores = manager.dict()      # per-design scores across workers

    print(f"{len(pdbfiles)} designs to analyze.")
    print("Building multiprocessing queue of designs.")

    for idx, pdbfile in enumerate(pdbfiles):
        scores[idx] = manager.dict()
        the_queue.put((idx, pdbfile))

    datacolumns = manager.list()
    datacolumns += compute_datacolumns(args)

    # -------------------------------------------------------------------------
    # INNER WORKER FUNCTION
    # -------------------------------------------------------------------------

    def process_worker(q, ref_pdbs_local, ref_poses_local):
        """
        Worker loop that processes designs until it receives a sentinel (None).

        NOTE: This function is used as the Pool initializer and never returns
        until it consumes a sentinel from the shared queue.
        """
        while True:
            try:
                job = q.get(block=True)
            except Exception as err:
                print(f"[ERROR] Worker encountered queue error: {err}")
                return

            if job is None:
                debug_print("Worker received sentinel None; exiting.")
                return

            i, pdbfile = job
            design_name = os.path.basename(pdbfile)
            scores[i]["description"] = pdbfile
            scores[i]["passed"] = False if not args.analyze else np.nan  # will only be True for passing designs

            if args.analyze:
                print(f"\n===== PROCESSING DESIGN [{i}] {design_name} =====")

            jsonfile = pdbfile.replace(".cif.gz", ".json")
            debug_print(f"[{i}] JSON metadata path: {jsonfile}")

            trb = load_trb(jsonfile, design_name)
            if trb is None:
                scores[i]["error"] = "failed_to_load_trb"
                continue

            ref_pdb, ref_pose = select_reference_pdb(trb, ref_pdbs_local, pdbfile, ref_poses_local)
            if ref_pdb is None or ref_pose is None:
                scores[i]["error"] = "failed_to_load_ref_pose"
                continue

            matched_residues = get_matcher_residues(ref_pdb)
            print(f"### MATCHED RESIDUES FOR {design_name} ###\n{matched_residues}\n")

            fixed_positions_from_JSON, fixed_pos_in_hal, fixed_pos_in_ref, _ref_catres = build_fixed_positions(
                trb, args, ref_catres, matched_residues, ref_pose
            )

            # Load design pose from CIF
            print(f"[{design_name}] Loading CIF into Pose...")
            try:
                pose = load_cif_to_pose(pdbfile)
            except Exception as e:
                print(f"[ERROR] Error loading CIF for {pdbfile}: {e}")
                scores[i]["error"] = "failed_to_load_cif"
                continue

            poses_to_parse = [pose]  # kept for future extension with multiple states/trajectories

            for traj_idx, pose in enumerate(poses_to_parse):
                pose2 = pose.clone()
                debug_print(f"[{design_name}] Processing trajectory index {traj_idx}.")

                # Partial diffusion: align protein and append ligand from reference if needed
                if args.partial and any(res.is_ligand() for res in ref_pose.residues) and not any(res.is_ligand() for res in pose2.residues):
                    print(f"[{design_name}] Aligning partial diffusion output to reference and appending ligand(s).")
                    align_map = pyrosetta.rosetta.std.map_core_id_AtomID_core_id_AtomID()
                    aln_atoms = ['N', 'CA', 'C', 'O']

                    for template_i, target_i in zip(trb["con_ref_idx0"], trb["con_hal_idx0"]):
                        res_template_i = ref_pose.residue(template_i + 1)
                        res_target_i = pose2.residue(target_i + 1)
                        for aname in aln_atoms:
                            template_atom_idx = res_template_i.atom_index(aname)
                            target_atom_idx = res_target_i.atom_index(aname)
                            atom_id_template = pyrosetta.rosetta.core.id.AtomID(template_atom_idx, template_i + 1)
                            atom_id_target = pyrosetta.rosetta.core.id.AtomID(target_atom_idx, target_i + 1)
                            align_map[atom_id_target] = atom_id_template

                    rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(pose2, ref_pose, align_map)
                    print(f"[{design_name}] Alignment RMSD (partial diffusion) = {rmsd:.3f}")

                    ligands_ref = [res for res in ref_pose.residues if res.is_ligand()]
                    for lig in ligands_ref:
                        pyrosetta.rosetta.core.pose.append_subpose_to_pose(pose2, ref_pose, lig.seqpos(), lig.seqpos(), True)

                # --------------------------------------------------------------
                # 1) BACKBONE CHAINBREAKS & NON-ADJ CA-CA
                # --------------------------------------------------------------
                passed_backbone = compute_backbone_metrics(pose, scores[i], args, pdbfile, pose_label=f"traj{traj_idx}")
                if not passed_backbone:
                    scores[i]["failed_stage"] = "backbone"
                    break

                # --------------------------------------------------------------
                # 2) LIGAND CLASH CHECK
                # --------------------------------------------------------------
                ligands = [res for res in pose2.residues if res.is_ligand()]
                if ligands:
                    lig_dists = []
                    for lig in ligands:
                        ligand_HAs = [n for n in range(1, lig.natoms() + 1) if not lig.atom_is_hydrogen(n)]
                        for res in pose2.residues:
                            if (res.nbr_atom_xyz() - lig.nbr_atom_xyz()).norm() > 15.0:
                                continue
                            if res.is_ligand():
                                continue
                            for lha in ligand_HAs:
                                if args.exclude_clash_atoms is not None and lig.atom_name(lha).strip() in args.exclude_clash_atoms:
                                    continue
                                for n in range(1, min(5, res.natoms() + 1)):
                                    lig_dists.append((res.xyz(n) - lig.xyz(lha)).norm())
                    if not lig_dists:
                        lig_dists = [9.9]
                        print(f"[WARNING] {pdbfile} no clashcheck-valid residues around ligand found; defaulting lig_dist=9.9.")
                    scores[i]["lig_dist"] = min(lig_dists)
                    if not args.analyze and scores[i]["lig_dist"] < args.lig_dist:
                        print(f"{pdbfile}: ligand is too close to the backbone (min ligand-backbone dist={scores[i]['lig_dist']:.2f})")
                        scores[i]["failed_stage"] = "ligand_clash"
                        break
                else:
                    scores[i]["lig_dist"] = np.nan
                    debug_print(f"[{design_name}] No ligands found in pose; skipping ligand clash check.")

                # --------------------------------------------------------------
                # 3) TIP-ATOM SIDECHAIN CONNECTIVITY & CART_BONDED / FA_DUN
                # --------------------------------------------------------------
                if "select_fixed_atoms" in trb.get("specification", {}):
                    motif_res_bond_deviations = []
                    for res in fixed_positions_from_JSON.values():
                        resno = int(res[1:])
                        motif_res_bond_deviations.append(sidechain_connectivity(pose2.residue(resno + 1)))
                    scores[i]["bondlen_dev"] = max(motif_res_bond_deviations) if motif_res_bond_deviations else 0.0
                    if not args.analyze and scores[i]["bondlen_dev"] > args.bondlen_dev:
                        print(f"{pdbfile}: motif residue geometry too distorted (bondlen_dev={scores[i]['bondlen_dev']:.2f})")
                        scores[i]["failed_stage"] = "bondlen_dev"
                        break

                    if args.cart_bonded is not None or args.fa_dun is not None:
                        catres_seqpos = [int(r[1:]) for r in fixed_positions_from_JSON.values()]
                        averages, scoredict = get_rosetta_scores(pose, sfx, sfx_cart, catres_seqpos)
                        for term in averages:
                            scores[i][f"{term}_avg"] = averages[term]

                        if not args.analyze and args.cart_bonded is not None and scores[i]["cart_bonded_avg"] > args.cart_bonded:
                            print(f"{pdbfile}: motif sidechain geometry too distorted (cart_bonded_avg={scores[i]['cart_bonded_avg']:.2f})")
                            scores[i]["failed_stage"] = "cart_bonded"
                            break

                        if not args.analyze and args.fa_dun is not None and scores[i]["fa_dun_avg"] > args.fa_dun:
                            print(f"{pdbfile}: motif rotamers suboptimal (fa_dun_avg={scores[i]['fa_dun_avg']:.2f})")
                            scores[i]["failed_stage"] = "fa_dun"
                            break
                else:
                    scores[i]["bondlen_dev"] = np.nan
                    debug_print(f"[{design_name}] No 'select_fixed_atoms' in specification; skipping motif connectivity and rotamer checks.")

                # --------------------------------------------------------------
                # 4) SUBJECTIVE SCAFFOLD QUALITY (Loops, helices, ROG, loops@motif)
                # --------------------------------------------------------------
                dssp = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose2)
                secstruct = dssp.get_dssp_secstruct()

                loop_frac = secstruct.count("L") / max(1, pose2.size())
                scores[i]["loop_frac"] = loop_frac
                if not args.analyze and loop_frac > args.loop_limit:
                    if loop_frac > 0.9 and traj_idx > 0:
                        print(f"{pdbfile}: unusual trajectory loopiness? loop_frac={loop_frac:.3f}")
                    else:
                        print(f"{pdbfile}: protein too loopy (loop_frac={loop_frac:.3f})")
                    scores[i]["failed_stage"] = "loop_frac"
                    break

                if "H" in secstruct:
                    longest_helix = max(len(x.replace("E", "")) for x in secstruct.split("L") if "H" in x)
                else:
                    longest_helix = 0
                scores[i]["longest_helix"] = longest_helix
                if not args.analyze and longest_helix > args.longest_helix:
                    print(f"{pdbfile}: longest helix too long (longest_helix={longest_helix})")
                    scores[i]["failed_stage"] = "longest_helix"
                    break

                if "metrics" in trb and "radius_of_gyration" in trb["metrics"]:
                    scores[i]["rog"] = trb["metrics"]["radius_of_gyration"]
                    debug_print(f"[{design_name}] Using ROG from TRB: {scores[i]['rog']:.3f}")
                else:
                    scores[i]["rog"] = get_ROG(pose2)
                    debug_print(f"[{design_name}] Using computed ROG: {scores[i]['rog']:.3f}")

                if not args.analyze and scores[i]["rog"] > args.rog:
                    print(f"{pdbfile}: radius of gyration too high (rog={scores[i]['rog']:.1f})")
                    scores[i]["failed_stage"] = "rog"
                    break

                if args.loop_catres:
                    loops_next_to_catres = False
                    for r_hal, r_ref in zip(fixed_pos_in_hal, fixed_pos_in_ref):
                        if f"{r_ref[0]}{r_ref[1]}" not in _ref_catres:
                            continue
                        left = secstruct[max(0, r_hal - 3):max(0, r_hal - 1)]
                        right = secstruct[r_hal:r_hal + 2]
                        if left == "LL" and right == "LL":
                            loops_next_to_catres = True
                            break
                    scores[i]["loop_at_motif"] = int(loops_next_to_catres)
                    if not args.analyze and loops_next_to_catres:
                        print(f"{pdbfile}: catalytic residue sits between loops (loop_at_motif=1)")
                        scores[i]["failed_stage"] = "loop_at_motif"
                        break
                else:
                    scores[i]["loop_at_motif"] = np.nan

                # --------------------------------------------------------------
                # 5) TERMINI DISTANCE TO LIGAND
                # --------------------------------------------------------------
                if ligands:
                    ligands_design = [res for res in pose2.residues if res.is_ligand()]
                    term_mindists = []
                    for lig in ligands_design:
                        lig_HAs = [n + 1 for n in range(lig.natoms()) if lig.atom_type(n + 1).element() != "H"]
                        d_Nt_lig = min((pose2.residue(1).xyz("CA") - lig.xyz(a)).norm() for a in lig_HAs)
                        d_Ct_lig = min((pose2.residue(pose2.size() - len(ligands_design)).xyz("CA") - lig.xyz(a)).norm() for a in lig_HAs)
                        term_mindists.append(min(d_Nt_lig, d_Ct_lig))
                    scores[i]["term_mindist"] = min(term_mindists) if term_mindists else np.nan
                    if not args.analyze and scores[i]["term_mindist"] < args.term_limit:
                        print(f"{pdbfile}: terminus too close to ligand (term_mindist={scores[i]['term_mindist']:.2f})")
                        scores[i]["failed_stage"] = "term_mindist"
                        break
                else:
                    scores[i]["term_mindist"] = np.nan

                # --------------------------------------------------------------
                # 6) MAP REFERENCE RESIDUES TO HAL NUMBERING (Mutate ALL diffused)
                # --------------------------------------------------------------
                ref_catres_nos = []
                hal_catres_nos = []
                mapping_failure = False

                # Always mutate ALL residues in diffused_index_map, regardless of --ref_catres
                mutate_keys = list(fixed_positions_from_JSON.keys())  # e.g. ["A94", "A96", "A106", "A119", ...]

                for r in mutate_keys:
                    ch = r[0]
                    ref_resno = int(r[1:])

                    if (ch, ref_resno) not in fixed_pos_in_ref:
                        print(f"[ERROR] Cannot find reference residue {r} in diffused_index_map keys {fixed_pos_in_ref}")
                        mapping_failure = True
                        break

                    # Find this residue in the reference pose by chain + PDB number
                    ref_resno_in_pose = None
                    for res in ref_pose.residues:
                        chain = ref_pose.pdb_info().chain(res.seqpos())
                        pdb_no = ref_pose.pdb_info().number(res.seqpos())
                        if chain == ch and pdb_no == ref_resno:
                            ref_resno_in_pose = (chain, res.seqpos())
                            break

                    if ref_resno_in_pose is None:
                        print(f"[ERROR] Could not determine ref_pose residue number for reference residue {r}")
                        mapping_failure = True
                        break

                    ref_catres_nos.append(ref_resno_in_pose)

                    # Map from (chain, ref_resno) → HAL seqpos using fixed_pos_in_ref / fixed_pos_in_hal
                    hal_catres_nos.append(fixed_pos_in_hal[fixed_pos_in_ref.index((ch, ref_resno))])

                if mapping_failure:
                    scores[i]["error"] = "catres_mapping_failed"
                    break

                for j, ref_res in enumerate(ref_catres_nos):
                    ref_catres_no = ref_res[1]
                    catres_seqpos = hal_catres_nos[j]
                    catres_AA = ref_pose.residue(ref_catres_no).name().split(":")[0]
                    if "ProteinFull" in pose2.residue(catres_seqpos).name():
                        catres_AA = catres_AA + ":" + pose2.residue(catres_seqpos).name().split(":")[1]
                    mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
                    mutres.set_res_name(catres_AA)
                    mutres.set_target(catres_seqpos)
                    mutres.apply(pose2)

                # --------------------------------------------------------------
                # 7) LIGAND SASA & EXPOSURE
                # --------------------------------------------------------------
                if ligands:
                    free_ligands = {}
                    for lig in ligands:
                        tmp_pose = pyrosetta.rosetta.core.pose.Pose()
                        tmp_pose.append_residue_by_jump(lig, 0)
                        free_ligands[lig.name3()] = tmp_pose.clone()

                    free_ligand_SASA = sum(getSASA(p, resno=1) for p in free_ligands.values())
                    scores[i]["SASA"] = getSASA(pose2, resno=[lig.seqpos() for lig in ligands]) if ligands else 0.0
                    scores[i]["SASA_rel"] = scores[i]["SASA"] / free_ligand_SASA if free_ligand_SASA > 0 else np.nan

                    if not args.analyze and not np.isnan(scores[i]["SASA_rel"]) and scores[i]["SASA_rel"] > args.SASA_limit:
                        print(f"{pdbfile}: ligand too exposed (SASA_rel={scores[i]['SASA_rel']:.3f})")
                        scores[i]["failed_stage"] = "SASA_rel_high"
                        break
                    if not args.analyze and not np.isnan(scores[i]["SASA_rel"]) and scores[i]["SASA_rel"] < 0.01:
                        print(f"{pdbfile}: ligand too buried (SASA_rel={scores[i]['SASA_rel']:.3f})")
                        scores[i]["failed_stage"] = "SASA_rel_low"
                        break

                    if args.ligand_exposed_atoms is not None and args.exposed_atom_SASA is not None:
                        target_ligand = [res for res in pose2.residues if all(res.has(a) for a in args.ligand_exposed_atoms)]
                        if not target_ligand:
                            print(f"[WARNING] Cannot find a ligand containing atoms {args.ligand_exposed_atoms} in {design_name}; skipping exposed_atom_SASA check.")
                            scores[i]["SASA_exposed_atoms"] = np.nan
                        else:
                            indexes = [target_ligand[0].atom_index(x) for x in args.ligand_exposed_atoms]
                            surf_vol_nosc = getSASA(pose2, ignore_sc=True)
                            scores[i]["SASA_exposed_atoms"] = sum(surf_vol_nosc.surf(pose2.size(), idx) for idx in indexes)

                            if not args.analyze and scores[i]["SASA_exposed_atoms"] < args.exposed_atom_SASA:
                                print(f"{pdbfile}: ligand atoms {args.ligand_exposed_atoms} too buried (SASA_exposed_atoms={scores[i]['SASA_exposed_atoms']:.3f})")
                                scores[i]["failed_stage"] = "SASA_exposed_atoms"
                                break
                else:
                    scores[i]["SASA"] = np.nan
                    scores[i]["SASA_rel"] = np.nan
                    scores[i]["SASA_exposed_atoms"] = np.nan

                # --------------------------------------------------------------
                # 8) MARK AS PASSED (IF NOT ANALYZE-ONLY)
                # --------------------------------------------------------------
                if not args.analyze:
                    scores[i]["passed"] = True
                    print(f"{pdbfile}: GOOD design (passed all filters).")

                # --------------------------------------------------------------
                # 9) ADJUST REMARK 666 MATCHER LINES IN OUTPUT PDB
                # --------------------------------------------------------------
                if not args.analyze and len(matched_residues) != 0:
                    matched_residues_in_design = {}

                    # fixed_positions_from_JSON: ref 'A94' -> hal 'A101' (for example)
                    ref2hal = fixed_positions_from_JSON

                    for ref_resno, info in matched_residues.items():
                        # MATCH MOTIF residue in reference (PDB numbering)
                        ref_motif_key = f"{info['chain']}{ref_resno}"
                        if ref_motif_key not in ref2hal:
                            # motif residue not in diffused region; skip remapping for this line
                            continue

                        hal_motif = ref2hal[ref_motif_key]  # e.g. 'A101'
                        hal_chain, hal_resno = hal_motif[0], int(hal_motif[1:])

                        new_info = copy.deepcopy(info)
                        new_info["chain"] = hal_chain
                        # Track whether we *successfully* remapped the MATCH TEMPLATE target
                        new_info["_target_mapped"] = False

                        tgt_resno_orig = info["target_resno"]
                        tgt_chain_orig = info["target_chain"]

                        if tgt_resno_orig != 0:
                            # Map MATCH TEMPLATE residue (chain + PDB number) to ref_pose seqpos
                            tgt_seqpos = None
                            for res in ref_pose.residues:
                                chain = ref_pose.pdb_info().chain(res.seqpos())
                                pdb_no = ref_pose.pdb_info().number(res.seqpos())
                                if chain == tgt_chain_orig and pdb_no == tgt_resno_orig:
                                    tgt_seqpos = res.seqpos()
                                    break

                            if tgt_seqpos is None:
                                print(f"[WARNING] Could not map MATCH TEMPLATE residue {tgt_chain_orig}{tgt_resno_orig} into ref_pose; "
                                    "leaving target_* fields unchanged in REMARK 666.")
                                matched_residues_in_design[hal_resno] = new_info
                                continue

                            # If MATCH TEMPLATE residue is a protein, try to remap using diffused_index_map
                            if ref_pose.residue(tgt_seqpos).is_protein():
                                tgt_key = f"{tgt_chain_orig}{tgt_resno_orig}"  # uses PDB numbering
                                if tgt_key in ref2hal:
                                    hal_tgt = ref2hal[tgt_key]  # e.g. 'A56'
                                    new_info["target_chain"] = hal_tgt[0]
                                    new_info["target_resno"] = int(hal_tgt[1:])
                                    new_info["_target_mapped"] = True
                                # else: no mapping in diffused_index_map; keep original target_* as-is

                            # If MATCH TEMPLATE residue is a ligand, point it at the matching ligand in the design
                            elif ref_pose.residue(tgt_seqpos).is_ligand():
                                ligands2_in_design = [
                                    res2 for res2 in pose2.residues
                                    if res2.name3() == info["target_name"]
                                ]
                                if len(ligands2_in_design) == 1:
                                    lig_seqpos = ligands2_in_design[0].seqpos()
                                    new_info["target_chain"] = pose2.pdb_info().chain(lig_seqpos)
                                    new_info["target_resno"] = pose2.pdb_info().number(lig_seqpos)
                                    new_info["_target_mapped"] = True
                                else:
                                    print("[WARNING] Multiple ligands with same name in system; REMARK 666 target remap may be incorrect; leaving "
                                        "target_* unchanged.")
                                    matched_residues_in_design[hal_resno] = new_info
                                    continue

                        # If tgt_resno_orig == 0, we *never* tried to map it above,
                        # so _target_mapped stays False; those are "unmatched" and
                        # may be fixed later if unique by residue name.
                        matched_residues_in_design[hal_resno] = new_info

                    # At this point matched_residues_in_design contains REMARK 666 entries
                    # for the design, including some where target_chain/resno might still
                    # be dummy values (e.g. X / 0) OR that we consciously left unmapped.
                    # We marked "mapped" ones with _target_mapped=True.

                    # Optional: try to fix unresolved MATCH TEMPLATE targets where the
                    # target_name corresponds to a unique residue (protein or ligand)
                    # in the design. If multiple residues share the same name3, we
                    # leave the REMARK line unchanged to avoid ambiguity.
                    if args.fix_unmatched_remark_lines_to_lig:
                        # Build mapping: residue name3 -> list of (chain, resno) in pose2
                        res_pos_by_name = {}
                        pdbinfo_design = pose2.pdb_info()
                        for res in pose2.residues:
                            if res.is_virtual_residue():
                                continue
                            name3 = res.name3()
                            chain = pdbinfo_design.chain(res.seqpos())
                            resno = pdbinfo_design.number(res.seqpos())
                            res_pos_by_name.setdefault(name3, []).append((chain, resno))

                        n_fixed = 0
                        for hal_resno, info in matched_residues_in_design.items():
                            # Only attempt to fix entries that were NOT successfully mapped earlier
                            if info.get("_target_mapped", False):
                                continue

                            tname = info.get("target_name", "").strip()
                            if not tname:
                                continue

                            candidates = res_pos_by_name.get(tname, [])

                            # Only fix if there is exactly one residue instance for tname
                            # (protein or ligand). If there are multiple (e.g. many HIS),
                            # we cannot safely disambiguate, so we skip.
                            if len(candidates) == 1:
                                chain, resno = candidates[0]
                                info["target_chain"] = chain
                                info["target_resno"] = resno
                                info["_target_mapped"] = True
                                n_fixed += 1

                        if n_fixed > 0:
                            print(f"[{design_name}] fix_unmatched_remark_lines_to_lig: fixed {n_fixed} REMARK 666 MATCH TEMPLATE targets "
                                f"based on unique residue positions.")

                    print(f"\n[{design_name}] REMARK 666: {len(matched_residues)} in ref, {len(matched_residues_in_design)} mapped into design.")
                    pose2 = add_matcher_line_to_pose(pose2, ref_pose, matched_residues_in_design, matched_residues)
                
                # Only annotate/dump PDBs if not in analyze-only mode
                if not args.analyze:
                    # Add 'motif' label to motif residues
                    for rn in fixed_positions_from_JSON.values():
                        pose2.pdb_info().add_reslabel(int(rn[1:]), "motif")

                    # Decide output file path (PDB instead of CIF)
                    outfile = os.path.join(filtered_dir, os.path.basename(pdbfile))
                    if outfile.endswith(".cif.gz"):
                        outfile = outfile.replace(".cif.gz", ".pdb")
                    elif outfile.endswith(".cif"):
                        outfile = outfile.replace(".cif", ".pdb")

                    debug_print(f"Dumping PDB to: {outfile}")
                    pose2.dump_pdb(outfile)
                else:
                    debug_print(f"[ANALYZE ONLY] Skipping PDB dump for {pdbfile}")

                # Only one trajectory per file currently; break out of loop
                break

    # -------------------------------------------------------------------------
    # MULTIPROCESSING: START WORKERS
    # -------------------------------------------------------------------------
    N_PROCESSES = determine_num_processes(args)
    print(f"Using {N_PROCESSES} processes for analysis/filtering.")

    pool = multiprocessing.Pool(
        processes=N_PROCESSES,
        initializer=process_worker,
        initargs=(the_queue, ref_pdbs, ref_poses),
    )

    for _ in range(N_PROCESSES):
        the_queue.put(None)

    the_queue.close()
    the_queue.join_thread()
    pool.close()
    pool.join()

    end = time.time()
    print(f"Analyzing diffusion outputs took {end - start:.3f} seconds.")

    # -------------------------------------------------------------------------
    # BUILD DATAFRAME FROM SCORES
    # -------------------------------------------------------------------------
    df = pd.DataFrame()

    if not scores:
        print("[WARNING] No scores collected; nothing to write.")
        return

    keylens = {i: len(scores[i]) for i in scores.keys()}
    all_keys_idx = max(keylens, key=keylens.get)

    for k in scores[all_keys_idx].keys():
        if k == "description":
            continue
        df[k] = float
    df["description"] = str

    for i in scores.keys():
        for k in scores[i].keys():
            if k == "description":
                continue
            df.at[i, k] = scores[i][k]
        df.at[i, "description"] = scores[i]["description"]

    for k in datacolumns:
        if k not in df.keys():
            df[k] = np.nan

    for k in df.keys():
        if k == "description":
            continue
        df = df.sort_values(k)

    if len(df) < 200:
        print(df)
    else:
        print("Too many structures, only printing rows with non-NaN metrics:")
        print(df.dropna(how="all", subset=[c for c in df.columns if c not in ["description"]]))

    if args.analyze is False and "passed" in df.columns:
        n_passed = int(df.loc[df.passed == 1.0].shape[0])
        print(f"##### {n_passed}/{len(df)} backbones passed all filters  #######")
        print(f"Backbones and TRBs that pass filters have been copied to `{args.outdir}`")

    print(f"Saving analysis scores of each backbone into `{args.scorefile_out}`")
    dump_scorefile(df, args.scorefile_out)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdb", nargs="+", type=str, help="Input PDB/CIFs from aa_diffusion (CIF.GZ expected).")
    parser.add_argument("--pdbpath", nargs="+", type=str, help="Directories where input CIF.GZ files can be found.")
    parser.add_argument("--ref", nargs="+", type=str, help="Reference PDBs used as input for diffusion.")
    parser.add_argument("--ref_path", type=str, help="Path where reference PDBs can be found.")
    parser.add_argument("--analyze", action="store_true", default=False, help="Analyze only; do not filter/move files.")

    parser.add_argument("--params", nargs="+", type=str, help="Params files of ligands and noncanonicals.")

    parser.add_argument("--lig_dist", default=2.5, type=float, help="Cutoff for smallest allowed backbone-to-ligand heavy-atom distance.")
    parser.add_argument("--SASA_limit", default=0.20, type=float, help="Cutoff for ligand relative SASA.")
    parser.add_argument("--loop_limit", default=0.30, type=float, help="Cutoff for maximum allowed loop content.")
    parser.add_argument("--longest_helix", default=30, type=int, help="Longest allowed helix length.")
    parser.add_argument("--rog", default=30.0, type=float, help="Largest allowed radius of gyration.")
    parser.add_argument("--term_limit", default=15.0, type=float, help="Cutoff for how close the termini can be to any ligand heavy atom.")
    parser.add_argument("--bondlen_dev", default=0.1, type=float, help="Maximum allowed sidechain bondlength deviation for tip-atom diffusion.")
    parser.add_argument("--exclude_clash_atoms", type=str, nargs="+", help="Ligand atom names to exclude from ligand clash checking.")
    parser.add_argument("--ligand_exposed_atoms", type=str, nargs="+", help="Ligand atoms that must have SASA above --exposed_atom_SASA.")
    parser.add_argument("--exposed_atom_SASA", type=float, help="Per-atom SASA cutoff for --ligand_exposed_atoms.")
    parser.add_argument("--ref_catres", type=str, nargs="+", help="Catalytic residue positions in reference structure (e.g. A94-96).")
    parser.add_argument("--loop_catres", action="store_false", default=True, help="If True, filter out designs with catalytic residues between loops.")
    parser.add_argument("--cart_bonded", type=float, help="Cutoff for cart_bonded sidechain quality at motif residues.")
    parser.add_argument("--fa_dun", type=float, help="Cutoff for fa_dun rotamer quality at motif residues.")

    parser.add_argument("--scorefile_out", type=str, default="diffusion_analysis.sc", help="Output Rosetta-style scorefile.")
    parser.add_argument("--outdir", type=str, default="filtered_structures", help="Directory for filtered output PDBs.")
    parser.add_argument("--partial", action="store_true", default=False, help="Set if running on partial diffusion output.")
    parser.add_argument("--nproc", type=int, help="# of CPU cores used.")
    parser.add_argument("--fix_unmatched_remark_lines_to_lig", action="store_true", default=False, help="If set, try to fix REMARK 666 lines whose MATCH TEMPLATE target could not be mapped earlier by assigning them to a unique residue (protein or ligand) in the design with the same residue name3.")

    args = parser.parse_args()
    main(args)
