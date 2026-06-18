#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Seth M. Woodbury & Donghyo Kim
/net/software/container/universal.sif

PDB/CSV → Sequence Toolkit (Robust, Modular, Verbose)
=====================================================

This script ingests either:
  (A) One or more PDB files (via --input_pdbs or --input_pdb_dir), extracts the
      single-chain protein sequence (1-letter) from ATOM records only, and builds a dataframe
      with columns [design_basename, design_aa_seq].

  (B) A preexisting CSV from the "John Bercow" route (via --input_john_bercow_csv),
      which is then column-filtered/renamed to a standardized schema.

From either route, you can optionally add N/C-terminal tags to produce expressed_aa_seq,
compute protein properties (extinction coefficient @ 280nm and molecular weight), and
export results to CSV/XLSX.

Key Invariants & Rules
----------------------
• Exactly *one* input mode must be used: [--input_pdbs | --input_pdb_dir | --input_john_bercow_csv].
• For PDB inputs: the structure must contain *exactly one* chain with ATOM records.
  - Multiple chains are allowed only if exactly ONE chain contains ATOM records (others may
    have only HETATM/water/etc). If >1 chain has ATOM, the script exits with an error.
• Only standard amino-acid residues from ATOM lines are converted. Ligands/HETATM/solvent are ignored.
• If both --prespecified_plasmid and any of --n_terminus_tag/--c_terminus_tag are provided,
  a loud warning is printed and the explicit tags take precedence.
• Outputs: By default both CSV and XLSX are written. Use --csv_output_only or --xlsx_output_only
  to limit outputs.

Inputs
------
Mandatory (choose exactly one route):
  --input_pdbs            One or more PDB paths (space-separated) or a glob, e.g. /path/to/*pdb
  --input_pdb_dir         A directory to scan for *.pdb (non-recursive unless --recursive)
  --input_john_bercow_csv Path to a preprocessed CSV with targeted columns to filter/rename

Common optional flags:
  --n_terminus_tag        Optional N-terminal amino-acid tag sequence (1-letter)
  --c_terminus_tag        Optional C-terminal amino-acid tag sequence (1-letter)
  --prespecified_plasmid  Shortcut key for hard-coded plasmid tag sets (e.g., pDT1)
  --path_length_nanodrop  Optical path length (cm) for Nanodrop A280 context (default: 1)
  --split_xlsx_by_plate   If present and 'plate_id' exists, also write one sheet per plate.
  --single_plate          If present, drop 'plate_id' and 'plate_well_position' from outputs.
  --minimize_xlsx_size    If present, do not create the 'important_info' sheet.
  --output_dir            Output directory (default: current working directory)
  --output_basename       Basename for outputs (without extension). If extension is supplied,
                          it is stripped and the correct one is appended.
  --csv_output_only       Only write CSV output
  --xlsx_output_only      Only write XLSX output
  --recursive             When using --input_pdb_dir, search recursively for *.pdb
  --quiet                 Reduce verbosity

Outputs
-------
• <basename>.csv  (default)
• <basename>.xlsx (default; engine-agnostic with graceful fallback if no Excel engine)

Final dataframe columns (union; some may be blank depending on route):
  - design_basename (str)
  - design_aa_seq (str)
  - design_aa_length (int)
  - expressed_aa_seq (str)
  - expressed_aa_length (int)
  - gene_frag_seq (str)              [CSV route]
  - gene_frag_length (int/float)     [CSV route]
  - idt_score (float/str)            [CSV route]
  - eblock_order_name (str)          [CSV route]
  - design_id (str)                  [CSV route]
  - plate_well_position (str)        [CSV route; from 'position']
  - plate_id (str)                   [CSV route]
  - well_position (str)              [CSV route]
  - plasmid (str)
  - n_term_tag (str)
  - c_term_tag (str)
  - ɛ(280nm)_M-1_cm-1 (int)          [computed from expressed_aa_seq]
  - expressed_aa_MW_Da (float)                    [computed from expressed_aa_seq]
  - theoretical_pI (float)           [computed]
  - instability_index (float)        [dimensionless]
  - aromaticity (float)
  - GRAVY (float)
  - helix_frac (float)
  - turn_frac (float)
  - sheet_frac (float)

Examples
--------
# From PDB files listed explicitly
auth$ python pdb_or_csv_to_sequences_and_properties.py \
    --input_pdbs /path/to/A001.pdb /path/to/A002.pdb \
    --n_terminus_tag MSG --c_terminus_tag GSAWSHPQFEK \
    --output_basename my_designs

# From a directory of PDBs (recursive) with prespecified plasmid tag
$ python pdb_or_csv_to_sequences_and_properties.py \
    --input_pdb_dir /path/to/pdbs --recursive \
    --prespecified_plasmid pDT1

# From the John Bercow CSV route (filter/rename columns)
$ python pdb_or_csv_to_sequences_and_properties.py \
    --input_john_bercow_csv /path/to/input.csv \
    --output_dir ./out --output_basename processed

"""

###########################################################################
### IMPORTS ###
###########################################################################
import os
import sys
import re
import glob
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    from Bio.Data import IUPACData
except Exception as e:
    print("[FATAL] Biopython is required (pip install biopython). Error:", e, file=sys.stderr)
    sys.exit(1)


###########################################################################
### EASY-TO-EDIT PRESETS (TOP OF FILE) ###
###########################################################################
# !!!!!!!!!!!!!!!!!!!!!! IMPORTANT: Hard-coded plasmid tags !!!!!!!!!!!!!!!!!!!!!!
# Add more in this dict: key → (N_tag, C_tag)
PLASMID_TAGS: Dict[str, Tuple[str, str]] = {
    "pDT1": ("MSG", "GSAWSHPQFEK"),
}

# Optional per-column rounding (decimals) and display name remapping
# (None means do not round). Keys are *internal* column names produced by the pipeline;
# remap names are what will appear in the Excel/CSV outputs.
COLUMN_SPEC = [
    ("design_basename",            "design_basename",          None),
    ("plate_id",                   "plate_id",                 None),
    ("plate_well_position",        "plate_well_position",      None),
    ("well_position",              "well_position",            None),
    ("eblock_order_name",          "eblock_order_name",        None),
    ("design_id",                  "design_id",                None),
    ("gene_frag_seq",              "gene_frag_seq",            None),
    ("gene_frag_length",           "gene_frag_length",         None),
    ("idt_score",                  "idt_score",                None),
    ("design_aa_seq",              "design_aa_seq",            None),
    ("design_aa_length",           "design_aa_length",         None),
    ("plasmid",                    "plasmid",                  None),
    ("n_term_tag",                 "n_term_tag",               None),
    ("c_term_tag",                 "c_term_tag",               None),
    ("expressed_aa_seq",           "expressed_aa_seq",         None),
    ("expressed_aa_length",        "expressed_aa_length",      None),
    ("ec_280_M_minus1_cm_minus1",  "ɛ(280nm)_M-1_cm-1",        None),
    ("mw_Da",                      "expressed_aa_MW_Da",       3),
    ("pI_theoretical",             "theoretical_pI",           3),
    ("instability_index",          "instability_index",        3),  # dimensionless
    ("aromaticity",                "aromaticity",              4),
    ("gravy",                      "GRAVY",                    4),
    ("helix_frac",                 "helix_frac",               4),
    ("turn_frac",                  "turn_frac",                4),
    ("sheet_frac",                 "sheet_frac",               4),
]

# Column width hints for Excel (approx. characters). Fallback uses header length.
WIDTH_HINTS = {
    "design_basename": 36,
    "design_aa_seq": 64,
    "expressed_aa_seq": 64,
    "gene_frag_seq": 64,
}


###########################################################################
### CONSTANTS & RESIDUE MAPS ###
###########################################################################
# Map 3-letter codes to 1-letter codes, including common alternates.
THREE_TO_ONE: Dict[str, str] = {k.upper(): v for k, v in IUPACData.protein_letters_3to1.items()}
THREE_TO_ONE.update({
    "MSE": "M",   # Selenomethionine
    "SEC": "U",   # Selenocysteine (if present; may not be desired—keep for completeness)
    "HIS": "H",   # Normalized histidine
})

# Residues to ignore outright
IGNORE_RESN = {"HOH", "WAT", "DOD"}

# Valid AA for ProtParam computations
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


###########################################################################
### LOGGING SETUP ###
###########################################################################
class _Formatter(logging.Formatter):
    def format(self, record):
        lvl = record.levelname
        msg = super().format(record)
        return f"[{lvl}] {msg}"

def setup_logger(quiet: bool = False) -> logging.Logger:
    logger = logging.getLogger("pdb_csv_seq_tool")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO if quiet else logging.DEBUG)
    fmt = _Formatter("%(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


###########################################################################
### PDB PARSING UTILITIES ###
###########################################################################
PDB_ATOM_RE = re.compile(r"^(ATOM  )")
PDB_HETATM_RE = re.compile(r"^(HETATM)")

# PDB fixed-width columns (PDB format v3.3-like):
#  1-6  Record name  "ATOM  " or "HETATM"
# 17-20 Residue name
# 22    Chain ID
# 23-26 Residue sequence number
# 27    Insertion code

def _safe_slice(line: str, start: int, end: int) -> str:
    return line[start-1:end] if len(line) >= end else line[start-1:]

def parse_pdb_sequence_single_chain(pdb_path: Path, logger: logging.Logger) -> Tuple[str, str]:
    """
    Parse a PDB to extract a single-chain amino-acid sequence derived from ATOM records only.
    Returns (chain_id, sequence_1letter).

    Exits with an error if more than one chain contains ATOM records.
    """
    if not pdb_path.exists():
        raise FileNotFoundError(f"Missing PDB: {pdb_path}")

    chains_atoms: Dict[str, List[Tuple[int, str, str]]] = {}

    with pdb_path.open("r", errors="ignore") as fh:
        for line in fh:
            if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
                continue

            rec = _safe_slice(line, 1, 6).strip()
            resn = _safe_slice(line, 18, 20).strip().upper()
            chain = _safe_slice(line, 22, 22).strip() or "_"
            resi = _safe_slice(line, 23, 26).strip()
            icode = _safe_slice(line, 27, 27).strip()

            if rec == "HETATM":
                continue

            if rec == "ATOM":
                if resn in IGNORE_RESN:
                    continue
                chains_atoms.setdefault(chain, []).append((int(resi) if resi else 0, icode, resn))

    if not chains_atoms:
        raise ValueError(f"No ATOM residues found in {pdb_path}")

    chains_with_atoms = [c for c, rows in chains_atoms.items() if rows]
    if len(chains_with_atoms) != 1:
        raise ValueError(
            f"PDB has {len(chains_with_atoms)} chains with ATOM records (expected exactly 1): {chains_with_atoms} in {pdb_path}"
        )

    chain_id = chains_with_atoms[0]
    rows = chains_atoms[chain_id]

    rows_sorted = sorted(rows, key=lambda x: (x[0], x[1]))

    sequence_letters: List[str] = []
    seen_keys = set()
    for resi, icode, resn in rows_sorted:
        key = (resi, icode)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        aa = THREE_TO_ONE.get(resn)
        if not aa:
            logger.debug(f"Ignoring non-standard/unknown residue {resn} at {chain_id}:{resi}{icode} in {pdb_path.name}")
            continue
        sequence_letters.append(aa)

    seq = "".join(sequence_letters)
    if not seq:
        raise ValueError(f"Empty sequence derived from ATOM records in {pdb_path}")

    logger.debug(f"Parsed {pdb_path.name}: chain={chain_id}, length={len(seq)}")
    return chain_id, seq


def collect_pdb_sequences(pdb_files: List[Path], logger: logging.Logger) -> pd.DataFrame:
    records = []
    for p in pdb_files:
        try:
            _, seq = parse_pdb_sequence_single_chain(p, logger)
            records.append({"design_basename": p.stem, "design_aa_seq": seq})
        except Exception as e:
            logger.error(f"Failed on {p}: {e}")
            sys.exit(1)
    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError("No sequences produced from PDB inputs.")
    return df


###########################################################################
### CSV ROUTE: FILTER & RENAME ###
###########################################################################
CSV_KEEP = [
    "design_name",
    "Sequence",
    "length_eblock",
    "design_aa_seq",
    "idt_score",
    "order_name",
    "design_id",
    "position",
    "plate_id",
    "Well Position",
]

CSV_RENAME = {
    "design_name": "design_basename",
    "Sequence": "gene_frag_seq",
    "length_eblock": "gene_frag_length",
    "order_name": "eblock_order_name",
    "Well Position": "well_position",
    "position": "plate_well_position",
}


def process_john_bercow_csv(csv_path: Path, logger: logging.Logger) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.debug(f"Loaded CSV with columns: {list(df.columns)}")

    missing = [c for c in CSV_KEEP if c not in df.columns]
    if missing:
        logger.error(f"Missing expected columns in CSV: {missing}")
        raise ValueError("CSV missing required columns.")

    df = df[CSV_KEEP].copy()
    df.rename(columns=CSV_RENAME, inplace=True)

    if "design_basename" not in df.columns:
        raise ValueError("Column rename failed for design_name → design_basename")

    logger.debug("Filtered and renamed CSV columns → standardized schema.")
    return df


###########################################################################
### TAG APPLICATION & PROTEIN PROPERTIES ###
###########################################################################

def resolve_tags(args, logger: logging.Logger) -> Tuple[str, str, str]:
    n_tag = args.n_terminus_tag or ""
    c_tag = args.c_terminus_tag or ""
    plasmid = args.prespecified_plasmid or "unspecified"

    if args.prespecified_plasmid:
        preset = PLASMID_TAGS.get(args.prespecified_plasmid)
        if not preset:
            logger.error(f"Unknown prespecified plasmid '{args.prespecified_plasmid}'. Known: {list(PLASMID_TAGS.keys())}")
            sys.exit(1)
        pre_n, pre_c = preset
        if args.n_terminus_tag or args.c_terminus_tag:
            logger.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logger.warning("Both --prespecified_plasmid AND explicit --n_terminus_tag/--c_terminus_tag were provided.")
            logger.warning("Explicit tags take precedence; plasmid presets are IGNORED for tag sequences.")
            logger.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            n_tag, c_tag = pre_n, pre_c

    for tag_val, tag_name in [(n_tag, "n_terminus_tag"), (c_tag, "c_terminus_tag")]:
        if tag_val and not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYUX\*]*", tag_val):
            logger.error(f"Tag '{tag_name}' contains non-standard characters: '{tag_val}'")
            sys.exit(1)

    logger.info(f"Resolved tags → N-term: '{n_tag or '(none)'}' | C-term: '{c_tag or '(none)'}' | plasmid: {plasmid}")
    return n_tag, c_tag, plasmid


def _sanitize_for_protparam(seq: str, logger: logging.Logger) -> str:
    s = seq.replace("*", "").upper()
    bad = sorted({ch for ch in s if ch not in VALID_AA})
    if bad:
        logger.debug(f"Sanitizing sequence for ProtParam: removing characters {bad}")
    return "".join(ch for ch in s if ch in VALID_AA)


def compute_properties(seq: str, logger: logging.Logger) -> Tuple[int, float, float, float, float, float, float, float]:
    clean = _sanitize_for_protparam(seq, logger)
    if not clean:
        return (0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))

    sa = ProteinAnalysis(clean)
    ec = sa.molar_extinction_coefficient()[1]  # includes cystines
    mw = sa.molecular_weight()  # Daltons
    pI = sa.isoelectric_point()
    instab = sa.instability_index()
    arom = sa.aromaticity()
    gravy = sa.gravy()
    helix, turn, sheet = sa.secondary_structure_fraction()
    return (int(ec), float(mw), float(pI), float(instab), float(arom), float(gravy), float(helix), float(turn), float(sheet))


def add_tags_and_properties(df: pd.DataFrame, n_tag: str, c_tag: str, plasmid: str, path_length_cm: float, logger: logging.Logger) -> pd.DataFrame:
    if "design_aa_seq" not in df.columns:
        raise ValueError("Input dataframe must contain 'design_aa_seq' before tagging.")

    df = df.copy()
    df["n_term_tag"] = n_tag
    df["c_term_tag"] = c_tag
    df["plasmid"] = plasmid

    df["design_aa_seq"] = df["design_aa_seq"].fillna("").astype(str)
    df["design_aa_length"] = df["design_aa_seq"].str.len().astype(int)

    df["expressed_aa_seq"] = df["design_aa_seq"].apply(lambda s: f"{n_tag}{s}{c_tag}")
    df["expressed_aa_length"] = df["expressed_aa_seq"].str.len().astype(int)

    logger.info(f"Computing ProtParam metrics for {len(df)} design(s)...")

    ecs, mws, pIs, instabs, aroms, gravys, hels, turns, sheets = ([] for _ in range(9))
    for s in df["expressed_aa_seq"].astype(str):
        try:
            ec, mw, pI, instab, arom, gravy, helix, turn, sheet = compute_properties(s, logger)
        except Exception:
            ec, mw, pI, instab, arom, gravy, helix, turn, sheet = (0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
        ecs.append(ec); mws.append(mw); pIs.append(pI); instabs.append(instab); aroms.append(arom); gravys.append(gravy); hels.append(helix); turns.append(turn); sheets.append(sheet)

    df["ec_280_M_minus1_cm_minus1"] = ecs
    df["mw_Da"] = mws
    df["pI_theoretical"] = pIs
    df["instability_index"] = instabs  # dimensionless
    df["aromaticity"] = aroms
    df["gravy"] = gravys
    df["helix_frac"] = hels
    df["turn_frac"] = turns
    df["sheet_frac"] = sheets

    logger.debug(f"Path length (Nanodrop context) = {path_length_cm} cm")
    return df


###########################################################################
### IO HELPERS ###
###########################################################################

def coerce_basename(basename: Optional[str]) -> str:
    if not basename:
        return "pdb_or_csv_sequences"
    base = basename
    for ext in (".csv", ".xlsx"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
    return base or "pdb_or_csv_sequences"


def _apply_rounding_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    for key, display, decimals in COLUMN_SPEC:
        if key in df.columns:
            series = df[key]
            if decimals is not None:
                series = pd.to_numeric(series, errors="coerce").round(decimals)
            out[display] = series
    return out


def _auto_set_col_widths(writer, sheet_name: str, df: pd.DataFrame):
    try:
        book = writer.book
        sheet = writer.sheets[sheet_name]
        # Determine engine capabilities
        is_xlsxwriter = book.__class__.__module__.startswith("xlsxwriter")
        is_openpyxl = book.__class__.__module__.startswith("openpyxl")

        for idx, col in enumerate(df.columns, 1 if is_openpyxl else 0):
            header = str(col)
            width = max(len(header) + 2, WIDTH_HINTS.get(col, len(header) + 2))
            # Extra-wide for sequences
            if any(k in header.lower() for k in ["seq", "ɛ", "expressed_aa_MW_Da", "design_basename"]):
                width = max(width, 40)
            if is_xlsxwriter:
                sheet.set_column(idx, idx, width)
            elif is_openpyxl:
                from openpyxl.utils import get_column_letter
                col_letter = get_column_letter(idx)
                sheet.column_dimensions[col_letter].width = width
    except Exception:
        pass  # Keep output even if column sizing fails


def _write_xlsx_with_fallback(df_full: pd.DataFrame, df_important: Optional[pd.DataFrame], xlsx_path: Path, logger: logging.Logger, split_by_plate: bool, minimize_xlsx_size: bool) -> Optional[Path]:
    try:
        with pd.ExcelWriter(xlsx_path) as writer:  # auto engine
            # Main sheet
            df_full.to_excel(writer, index=False, sheet_name="Complete_Summary")
            _auto_set_col_widths(writer, "Complete_Summary", df_full)

            # Optional: important_info sheet
            if (not minimize_xlsx_size) and df_important is not None and not df_important.empty:
                df_important.to_excel(writer, index=False, sheet_name="Important_Info")
                _auto_set_col_widths(writer, "Important_Info", df_important)

            # Optional split by plate
            if split_by_plate and "plate_id" in df_full.columns:
                unique_plates = list(pd.unique(df_full["plate_id"]))
                logger.info(f"Writing per-plate sheets for {len(unique_plates)} plate(s)...")
                for plate in unique_plates:
                    sub = df_full[df_full["plate_id"] == plate].copy()
                    # Make IDT-style minimal sheet if possible
                    cols = []
                    if "well_position" in sub.columns:
                        cols.append("well_position")
                    if "design_basename" in sub.columns:
                        cols.append("design_basename")
                    seq_col = None
                    for c in ("gene_frag_seq", "expressed_aa_seq", "design_aa_seq"):
                        if c in sub.columns:
                            seq_col = c; break
                    if seq_col:
                        cols.append(seq_col)
                    plate_df = sub[cols] if cols else sub
                    # Rename for IDT look
                    plate_df = plate_df.rename(columns={
                        "well_position": "Well Position",
                        "design_basename": "Name",
                        seq_col or "": "Sequence",
                    })
                    sname = f"plate_{str(plate)}"[:31]
                    plate_df.to_excel(writer, index=False, sheet_name=sname)
                    _auto_set_col_widths(writer, sname, plate_df)
        return xlsx_path
    except Exception as e:
        logger.warning(
            "Could not write XLSX (no Excel engine like 'openpyxl'/'xlsxwriter'?). "
            f"Skipping XLSX. Install one to enable Excel output. Details: {e}"
        )
        return None


def write_outputs(df_pretty: pd.DataFrame, df_important: Optional[pd.DataFrame], out_dir: Path, base: str, only_csv: bool, only_xlsx: bool, logger: logging.Logger, split_by_plate: bool, minimize_xlsx_size: bool) -> Tuple[Optional[Path], Optional[Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{base}.csv"
    xlsx_path = out_dir / f"{base}.xlsx"

    if only_csv and only_xlsx:
        logger.error("Cannot specify both --csv_output_only and --xlsx_output_only.")
        sys.exit(1)

    if not only_xlsx:
        df_pretty.to_csv(csv_path, index=False)
        logger.info(f"Wrote CSV: {csv_path}")
    else:
        csv_path = None

    if not only_csv:
        xlsx_written = _write_xlsx_with_fallback(df_pretty, df_important, xlsx_path, logger, split_by_plate=split_by_plate, minimize_xlsx_size=minimize_xlsx_size)
        if xlsx_written:
            logger.info(f"Wrote XLSX: {xlsx_written}")
        else:
            xlsx_path = None
    else:
        xlsx_path = None

    return csv_path, xlsx_path


###########################################################################
### ARGUMENTS & MAIN LOGIC ###
###########################################################################

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="get_design_seqs_and_biophysical_properties__MAIN.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Extract sequences from PDB(s) or process a preexisting CSV, optionally apply N/C-terminal tags, "
            "compute protein properties, and export to CSV/XLSX."
        ),
    )

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input_pdbs", nargs="+", help="One or more PDB paths (supports shell globs if expanded by shell)")
    g.add_argument("--input_pdb_dir", type=str, help="Directory containing PDB files (use --recursive for nested)")
    g.add_argument("--input_john_bercow_csv", type=str, help="Path to the preprocessed CSV to filter/rename")

    p.add_argument("--recursive", action="store_true", help="Recursively search for *.pdb in --input_pdb_dir")

    p.add_argument("--n_terminus_tag", type=str, default="", help="Optional N-terminal tag (1-letter AA)")
    p.add_argument("--c_terminus_tag", type=str, default="", help="Optional C-terminal tag (1-letter AA)")
    p.add_argument("--prespecified_plasmid", type=str, default="", help=f"Use preset tags by plasmid key (options: {list(PLASMID_TAGS.keys())})")

    p.add_argument("--path_length_nanodrop", type=float, default=1.0, help="Path length in cm (default: 1.0)")

    p.add_argument("--output_dir", type=str, default=".", help="Output directory (default: current dir)")
    p.add_argument("--output_basename", type=str, default="pdb_or_csv_sequences", help="Output basename (extension stripped)")
    p.add_argument("--csv_output_only", action="store_true", help="Write only CSV output")
    p.add_argument("--xlsx_output_only", action="store_true", help="Write only XLSX output")
    p.add_argument("--split_xlsx_by_plate", action="store_true", help="Also add a sheet per plate_id if present")
    p.add_argument("--single_plate", action="store_true", help="Drop 'plate_id' and 'plate_well_position' columns from outputs")
    p.add_argument("--minimize_xlsx_size", action="store_true", help="Do not create the 'important_info' sheet")

    p.add_argument("--quiet", action="store_true", help="Reduce logging verbosity")

    return p


def discover_pdbs_in_dir(dir_path: Path, recursive: bool) -> List[Path]:
    return sorted(dir_path.rglob("*.pdb") if recursive else dir_path.glob("*.pdb"))


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logger = setup_logger(args.quiet)

    logger.info("==================== RUN CONFIGURATION ====================")
    logger.info(f"Route selection: {'PDBs' if args.input_pdbs else ('PDB_DIR' if args.input_pdb_dir else 'CSV')}")

    n_tag, c_tag, plasmid = resolve_tags(args, logger)

    if args.input_pdbs:
        expanded: List[Path] = []
        for item in args.input_pdbs:
            matches = glob.glob(item)
            expanded.extend(Path(m) for m in matches) if matches else expanded.append(Path(item))
        pdb_list = [p for p in expanded if p.suffix.lower() == ".pdb"]
        if not pdb_list:
            logger.error("No PDB files found from --input_pdbs")
            return 2
        logger.info(f"Found {len(pdb_list)} PDB(s) to process.")
        df = collect_pdb_sequences(pdb_list, logger)

    elif args.input_pdb_dir:
        dpath = Path(args.input_pdb_dir)
        if not dpath.is_dir():
            logger.error(f"--input_pdb_dir is not a directory: {dpath}")
            return 2
        pdb_list = discover_pdbs_in_dir(dpath, args.recursive)
        if not pdb_list:
            logger.error("No *.pdb files found in directory (check --recursive if needed).")
            return 2
        logger.info(f"Discovered {len(pdb_list)} PDB(s) in directory.")
        df = collect_pdb_sequences(pdb_list, logger)

    else:
        csv_path = Path(args.input_john_bercow_csv)
        df = process_john_bercow_csv(csv_path, logger)
        if "design_aa_seq" not in df.columns:
            logger.warning("CSV lacks 'design_aa_seq'; expressed_aa_seq will be composed from an empty base unless provided.")
            df["design_aa_seq"] = ""

    logger.info(f"Total designs/rows: {len(df)}")

    df = add_tags_and_properties(df, n_tag=n_tag, c_tag=c_tag, plasmid=plasmid, path_length_cm=args.path_length_nanodrop, logger=logger)

    # Optionally drop plate columns
    if args.single_plate:
        for drop_col in ("plate_id", "plate_well_position"):
            if drop_col in df.columns:
                df.drop(columns=[drop_col], inplace=True)
        logger.info("--single_plate enabled → dropped 'plate_id' and 'plate_well_position'.")

    # Build pretty/rounded dataframe with display names/ordering
    df_pretty = _apply_rounding_and_rename(df)

    # Build important_info sheet (subset) unless minimized
    important_keys = [
        "design_basename", "design_aa_seq", "design_aa_length",
        "expressed_aa_seq", "expressed_aa_length",
        "ec_280_M_minus1_cm_minus1", "mw_Da", "pI_theoretical",
    ]
    df_important = None
    if not args.minimize_xlsx_size:
        sub = {}
        for k, display, decimals in COLUMN_SPEC:
            if k in important_keys and k in df.columns:
                series = df[k]
                if decimals is not None:
                    series = pd.to_numeric(series, errors="coerce").round(decimals)
                sub[display if k != "ec_280_M_minus1_cm_minus1" else "ɛ(280nm)_M-1_cm-1"] = series
        df_important = pd.DataFrame(sub)

    out_dir = Path(args.output_dir).resolve()
    base = coerce_basename(args.output_basename)
    csv_path, xlsx_path = write_outputs(
        df_pretty, df_important, out_dir, base,
        only_csv=args.csv_output_only, only_xlsx=args.xlsx_output_only,
        logger=logger, split_by_plate=args.split_xlsx_by_plate,
        minimize_xlsx_size=args.minimize_xlsx_size,
    )

    preview_cols = [c for c in ["design_basename", "expressed_aa_seq", "ɛ(280nm)_M-1_cm-1", "expressed_aa_MW_Da", "theoretical_pI"] if c in df_pretty.columns]
    logger.info("==================== PREVIEW (first 10 rows) ====================")
    logger.info(df_pretty[preview_cols].head(10).to_string(index=False))

    logger.info("==================== DONE ====================")
    if csv_path:
        logger.info(f"CSV → {csv_path}")
    if xlsx_path:
        logger.info(f"XLSX → {xlsx_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
