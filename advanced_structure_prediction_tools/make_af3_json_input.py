#!/usr/bin/env python3
"""
AlphaFold3 JSON Input Generator

Authors: Seth W. & Donghyo K.
Refactored for performance, clarity, and user experience.

This script generates AF3-compatible JSON input files from PDB structures.
Features:
  - Efficient batch processing with progress reporting
  - Smart detection of completed/incomplete output directories
  - PTM support via REMARK 666 parsing
  - Multi-ligand support (CCD codes and SMILES)
  - Safe cleanup of incomplete outputs with confirmation
"""

import re
import os
import ast
import sys
import json
import math
import time
import random
import shutil
import secrets
import argparse
import multiprocessing
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from Bio import PDB


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

AA3TO1 = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'
}

# Progress reporting thresholds
PROGRESS_THRESHOLDS = [10, 100, 500, 1000, 2500, 5000, 10000, 20000, 50000]

# Default minimum CIF files for a complete AF3 output
DEFAULT_MIN_CIF_FILES = 5

# Default number of files to cross-check with BioPython (--verify_fast_parser)
DEFAULT_VERIFY_FAST_PARSER = 200

# Compiled REMARK 666 MOTIF pattern, shared by the fast parser and the BioPython path.
REMARK666_PATTERN = re.compile(
    r"MOTIF\s+(\S+)\s+([A-Z]{3})\s+(\d+)([A-Z]?)\s+(\d+)\s+(\d+)",
    flags=re.IGNORECASE,
)


def str2bool(value) -> bool:
    """Argparse type that parses true/false-ish strings into a bool.

    Used with ``nargs='?', const=True`` so both ``--flag`` and ``--flag true|false`` work.
    """
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "t", "yes", "y", "1"):
        return True
    if s in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value (true/false), got: {value!r}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProgressTracker:
    """Tracks progress with timing metrics and ETA calculation."""

    total: int
    operation_name: str = "Processing"
    start_time: float = field(default_factory=time.time)
    current: int = 0
    _last_report_threshold: int = 0

    def update(self, count: int = 1) -> Optional[str]:
        """Update progress and return a status message if threshold is crossed."""
        self.current += count

        # Check if we've crossed a reporting threshold
        for threshold in PROGRESS_THRESHOLDS:
            if self._last_report_threshold < threshold <= self.current:
                self._last_report_threshold = threshold
                return self._format_progress()

        # Also report at completion
        if self.current >= self.total:
            return self._format_progress()

        return None

    def _format_progress(self) -> str:
        """Format a progress message with timing information."""
        elapsed = time.time() - self.start_time
        pct = (self.current / self.total * 100) if self.total > 0 else 100

        # Calculate rate and ETA
        rate = self.current / elapsed if elapsed > 0 else 0
        remaining = self.total - self.current
        eta_seconds = remaining / rate if rate > 0 else 0

        eta_str = self._format_duration(eta_seconds) if eta_seconds > 0 else "done"
        elapsed_str = self._format_duration(elapsed)

        return (
            f"[{time.strftime('%H:%M:%S')}] {self.operation_name}: "
            f"{self.current:,}/{self.total:,} ({pct:.1f}%) | "
            f"Elapsed: {elapsed_str} | Rate: {rate:.1f}/s | ETA: {eta_str}"
        )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds into human-readable duration."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def summary(self) -> str:
        """Return a final summary of the operation."""
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        return (
            f"  → Completed {self.current:,} items in {self._format_duration(elapsed)} "
            f"({rate:.1f}/s)"
        )


class Logger:
    """Simple logger with step tracking and visual formatting."""

    def __init__(self):
        self.step_count = 0
        self.start_time = time.time()

    def header(self, title: str):
        """Print a prominent header."""
        width = 70
        print("\n" + "═" * width)
        print(f"  {title}")
        print("═" * width)

    def step(self, message: str):
        """Print a numbered step."""
        self.step_count += 1
        print(f"\n[Step {self.step_count}] {message}")
        print("-" * 50)

    def info(self, message: str):
        """Print an info message."""
        print(f"  ℹ {message}")

    def success(self, message: str):
        """Print a success message."""
        print(f"  ✓ {message}")

    def warning(self, message: str):
        """Print a warning message."""
        print(f"  ⚠ {message}")

    def error(self, message: str):
        """Print an error message."""
        print(f"  ✗ {message}")

    def progress(self, message: str):
        """Print a progress update."""
        print(f"    {message}")

    def metric(self, label: str, value, unit: str = ""):
        """Print a metric with label and value."""
        unit_str = f" {unit}" if unit else ""
        print(f"    • {label}: {value:,}{unit_str}" if isinstance(value, int) else f"    • {label}: {value}{unit_str}")

    def final_summary(self):
        """Print final execution summary."""
        elapsed = time.time() - self.start_time
        print("\n" + "═" * 70)
        print(f"  Execution completed in {ProgressTracker._format_duration(elapsed)}")
        print("═" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  PDB PARSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_protein_sequence(pdb_file: Path, chain_id: str) -> str:
    """
    Extract the amino acid sequence from a specific chain in a PDB file.

    Args:
        pdb_file: Path to the PDB file
        chain_id: Chain identifier to extract

    Returns:
        Single-letter amino acid sequence string
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', str(pdb_file))

    sequence = ""
    for model in structure:
        for chain in model:
            if chain.get_id() != chain_id:
                continue
            for residue in chain:
                if PDB.is_aa(residue):
                    resname = residue.get_resname()
                    if resname in AA3TO1:
                        sequence += AA3TO1[resname]

    return sequence


def fast_parse_pdb(
    pdb_path: Path,
    chain_ids,
    need_remark666: bool,
) -> Tuple[Dict[str, str], List[Dict]]:
    """
    Fast single-pass PDB parser. Reads the file exactly ONCE and extracts:
      - the 1-letter sequence for each requested protein chain (standard 20 AAs only,
        in residue order, deduped by residue id), and
      - the REMARK 666 MOTIF catalog (only when need_remark666 is True).

    This replaces BioPython's PDBParser on the hot path. It is provably equivalent to
    get_protein_sequence for the data this tool processes: the current BioPython filter
    ``is_aa(residue) and resname in AA3TO1`` reduces exactly to ``resname in AA3TO1``
    (the 20 standard AAs are a strict subset of BioPython's is_aa set).

    Assumption (validated at runtime by --verify_fast_parser): a requested protein chain
    does not contain a HETATM whose resname is one of the 20 standard AAs at the same
    (chain, resSeq, iCode) as an ATOM residue. This holds for these structures, where
    chain A is the ATOM protein and ligands live in their own HETATM chains.

    Args:
        pdb_path: Path to the PDB file.
        chain_ids: Iterable of protein chain IDs to extract.
        need_remark666: Whether to also collect the REMARK 666 catalog.

    Returns:
        (sequences, catalog) where sequences maps each requested chain_id to its 1-letter
        string (empty if the chain is absent), and catalog is the REMARK 666 dict list
        (empty when need_remark666 is False).
    """
    chain_set = set(chain_ids)
    residues: Dict[str, List[str]] = {c: [] for c in chain_set}
    last_key: Dict[str, Optional[str]] = {c: None for c in chain_set}
    catalog: List[Dict] = []

    with open(pdb_path, "r") as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                if len(line) < 27:
                    continue
                chain = line[21]
                if chain not in chain_set:
                    continue
                res_key = line[22:27]  # resSeq (cols 23-26) + iCode (col 27)
                if res_key == last_key[chain]:
                    continue
                last_key[chain] = res_key
                aa = AA3TO1.get(line[17:20].strip())
                if aa is not None:
                    residues[chain].append(aa)
            elif need_remark666 and "REMARK 666" in line and "MOTIF" in line.upper():
                match = REMARK666_PATTERN.search(line)
                if match:
                    catalog.append({
                        "chain": match.group(1),
                        "resname": match.group(2).upper(),
                        "resnum": int(match.group(3)),
                        "cat_idx": int(match.group(5)),
                    })

    sequences = {c: "".join(residues[c]) for c in chain_set}
    return sequences, catalog


def biopython_parse(
    pdb_path: Path,
    chain_ids,
    need_remark666: bool,
) -> Tuple[Dict[str, str], List[Dict]]:
    """BioPython-based parse with the same return contract as fast_parse_pdb.

    Used both for --use_biopython mode and as the per-file safety-net fallback.
    """
    sequences = {c: get_protein_sequence(pdb_path, c) for c in chain_ids}
    catalog = parse_remark666_catalog(pdb_path) if need_remark666 else []
    return sequences, catalog


# ═══════════════════════════════════════════════════════════════════════════════
#  PDB FILE COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

def collect_pdb_files(
    pdb_dir: Path,
    recursive: bool,
    max_depth: Optional[int],
    specific_depth: Optional[int],
    logger: Logger,
    pdb_prefixes: Optional[List[str]] = None,
) -> List[Path]:
    """
    Collect PDB files from a directory, optionally searching subdirectories.

    Depth convention:
      - depth 0: files directly in pdb_dir
      - depth 1: files in direct subdirectories of pdb_dir
      - depth N: files N directory levels below pdb_dir

    When recursive=False, only depth 0 is searched (original behavior).
    When recursive=True:
      - No depth flags: search all depths (unlimited)
      - max_depth=N: search depths 0 through N (inclusive)
      - specific_depth=N: search only at exactly depth N

    Args:
        pdb_dir: Root directory to search
        recursive: Whether to search subdirectories
        max_depth: Maximum search depth (inclusive), or None for unlimited
        specific_depth: Exact depth to search, or None for range-based search
        logger: Logger instance for progress reporting

    Returns:
        Sorted list of PDB file paths with verified-unique stems

    Raises:
        SystemExit: If duplicate PDB stems are detected across directories
    """
    if not recursive:
        # Original flat behavior
        logger.info(f"Flat search in: {pdb_dir}")
        all_pdbs = sorted(
            p for p in pdb_dir.iterdir()
            if p.is_file() and p.suffix == ".pdb"
        )

        # Apply prefix filter if specified
        if pdb_prefixes:
            before_count = len(all_pdbs)
            all_pdbs = [
                p for p in all_pdbs
                if any(p.stem.startswith(pfx) for pfx in pdb_prefixes)
            ]
            logger.info(
                f"Prefix filter ({', '.join(pdb_prefixes)}): "
                f"{len(all_pdbs)} of {before_count} PDB files matched"
            )
            if not all_pdbs:
                logger.warning(
                    f"No PDB files matched any of the specified prefixes: "
                    f"{pdb_prefixes}"
                )

        return all_pdbs

    # Recursive mode
    logger.info(f"Recursive search in: {pdb_dir}")
    if specific_depth is not None:
        logger.info(f"Searching at exact depth: {specific_depth}")
    elif max_depth is not None:
        logger.info(f"Searching up to depth: {max_depth}")
    else:
        logger.info("Searching all depths (no depth limit)")

    all_pdbs: List[Path] = []
    stem_to_paths: Dict[str, List[Path]] = {}

    def _search_dir(current_dir: Path, current_depth: int) -> None:
        """Recursively search directories for PDB files."""
        # Determine whether to collect PDB files at this depth
        if specific_depth is not None:
            collect_here = (current_depth == specific_depth)
        elif max_depth is not None:
            collect_here = (current_depth <= max_depth)
        else:
            collect_here = True

        # Determine whether to recurse deeper
        if specific_depth is not None:
            recurse_deeper = (current_depth < specific_depth)
        elif max_depth is not None:
            recurse_deeper = (current_depth < max_depth)
        else:
            recurse_deeper = True

        try:
            entries = list(current_dir.iterdir())
        except PermissionError:
            logger.warning(f"Permission denied: {current_dir}")
            return

        for entry in entries:
            if collect_here and entry.is_file() and entry.suffix == ".pdb":
                all_pdbs.append(entry)
                stem_to_paths.setdefault(entry.stem, []).append(entry)
            elif recurse_deeper and entry.is_dir():
                _search_dir(entry, current_depth + 1)

    _search_dir(pdb_dir, 0)

    # Check for stem collisions (AF3 output naming uses pdb.stem)
    collisions = {stem: paths for stem, paths in stem_to_paths.items() if len(paths) > 1}
    if collisions:
        logger.error("Duplicate PDB stems detected across directories!")
        logger.error(
            "AF3 output naming uses pdb.stem, so duplicates would cause "
            "silent overwrites or mismatched outputs."
        )
        for stem, paths in sorted(collisions.items())[:20]:
            logger.error(f"  '{stem}' found in:")
            for p in paths:
                logger.error(f"    {p.parent}")
        if len(collisions) > 20:
            logger.error(f"  ... and {len(collisions) - 20} more collisions")
        logger.error(
            "Resolution: rename PDB files to have unique stems, or use "
            "--specific_depth / --max_depth to narrow the search."
        )
        # ── Verbose remediation hint for the chisel-design directory layout. ──
        # Common cause here: each design subdirectory holds a copied "seed"/input PDB
        # whose basename == the subdirectory name, plus design outputs named
        # <subdir>_chisel_NN.pdb. A child run that was seeded by a design shares that
        # design's stem, so the seed copy and the original design output collide.
        seed_collisions = sum(
            1 for paths in collisions.values() for p in paths if p.stem == p.parent.name
        )
        logger.error("")
        if seed_collisions:
            logger.error(
                f"Likely cause: {seed_collisions} colliding file(s) are copied 'seed'/input PDBs "
                "(basename == their subdirectory name) that share a stem with a design output elsewhere."
            )
        logger.error(
            "To make every stem unique WITHOUT dropping any file, rename the RIGHTMOST 'chisel' -> "
            "'chiseli2' in each design-output basename, leaving the seed copies (basename == "
            "subdirectory name) untouched, then re-run this command. Suggested one-off rename:"
        )
        print()
        print("python3 - <<'EOF'")
        print("import os")
        print(f"ROOT = {str(pdb_dir)!r}")
        print("OLD, NEW = 'chisel', 'chiseli2'")
        print("renamed = 0")
        print("for d in os.scandir(ROOT):")
        print("    if not d.is_dir(): continue")
        print("    for f in os.listdir(d.path):")
        print("        if not f.endswith('.pdb'): continue")
        print("        stem = f[:-4]")
        print("        if stem == d.name or NEW in stem: continue   # skip seed copy / already renamed")
        print("        i = stem.rfind(OLD)                          # RIGHTMOST 'chisel'")
        print("        if i < 0: continue")
        print("        os.rename(os.path.join(d.path, f),")
        print("                  os.path.join(d.path, stem[:i] + NEW + stem[i+len(OLD):] + '.pdb'))")
        print("        renamed += 1")
        print("print('renamed', renamed)")
        print("EOF")
        print()
        sys.exit(1)

    all_pdbs.sort()

    # Apply prefix filter if specified
    if pdb_prefixes:
        before_count = len(all_pdbs)
        all_pdbs = [
            p for p in all_pdbs
            if any(p.stem.startswith(pfx) for pfx in pdb_prefixes)
        ]
        logger.info(
            f"Prefix filter ({', '.join(pdb_prefixes)}): "
            f"{len(all_pdbs)} of {before_count} PDB files matched"
        )
        if not all_pdbs:
            logger.warning(
                f"No PDB files matched any of the specified prefixes: "
                f"{pdb_prefixes}"
            )

    return all_pdbs


# ═══════════════════════════════════════════════════════════════════════════════
#  PTM / REMARK 666 PARSING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PTMSpec:
    """Specification for a post-translational modification."""
    chain: str
    resname: str
    cat_idx: int
    ccd: str


def parse_ptm_specs(specs_list: List[str]) -> List[PTMSpec]:
    """
    Parse PTM specifications from command-line arguments.

    Format: CHAIN/RES3/CATIDX:CCD (e.g., 'A/LYS/3:KCX')

    Args:
        specs_list: List of PTM specification strings

    Returns:
        List of parsed PTMSpec objects
    """
    parsed = []
    for spec in specs_list:
        try:
            left, ccd = [s.strip() for s in spec.split(':', 1)]
            chain, res3, catidx = [s.strip() for s in left.split('/', 2)]
            if not all([chain, res3, catidx, ccd]):
                raise ValueError("Empty field")
            parsed.append(PTMSpec(
                chain=chain,
                resname=res3.upper(),
                cat_idx=int(catidx),
                ccd=ccd.upper()
            ))
        except Exception as e:
            raise ValueError(
                f"Bad --ptm_from_remark666 entry '{spec}'. "
                "Expected CHAIN/RES3/CATIDX:CCD (e.g., A/LYS/3:KCX)."
            ) from e
    return parsed


def parse_remark666_catalog(pdb_path: Path) -> List[Dict]:
    """
    Parse REMARK 666 MOTIF lines from a PDB file.

    Args:
        pdb_path: Path to PDB file

    Returns:
        List of dictionaries with chain, resname, resnum, cat_idx
    """
    catalog = []
    pattern = REMARK666_PATTERN

    with open(pdb_path, 'r') as fh:
        for line in fh:
            upper_line = line.upper()
            if "REMARK 666" in upper_line and "MOTIF" in upper_line:
                match = pattern.search(line)
                if match:
                    catalog.append({
                        "chain": match.group(1),
                        "resname": match.group(2).upper(),
                        "resnum": int(match.group(3)),
                        "cat_idx": int(match.group(5))
                    })
    return catalog


def build_mods_by_chain(
    catalog: List[Dict],
    ptm_specs: List[PTMSpec],
    pdb_path: str = ""
) -> Dict[str, List[Dict]]:
    """
    Map PTM specifications to actual residue numbers using the REMARK 666 catalog.

    Args:
        catalog: Parsed REMARK 666 catalog
        ptm_specs: List of PTM specifications
        pdb_path: PDB path for error messages

    Returns:
        Dictionary mapping chain IDs to lists of modifications
    """
    mods = {}
    for spec in ptm_specs:
        matches = [
            r for r in catalog
            if (r["chain"] == spec.chain and
                r["resname"] == spec.resname and
                r["cat_idx"] == spec.cat_idx)
        ]

        if not matches:
            raise ValueError(
                f"PTM target not found in REMARK 666 for {pdb_path}: "
                f"{spec.chain}/{spec.resname}/{spec.cat_idx} -> {spec.ccd}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous PTM target in {pdb_path}: "
                f"{spec.chain}/{spec.resname}/{spec.cat_idx} matches residues "
                f"{[m['resnum'] for m in matches]}."
            )

        resnum = matches[0]["resnum"]
        mods.setdefault(spec.chain, []).append({
            "ptmType": spec.ccd,
            "ptmPosition": resnum
        })

    return mods


# ═══════════════════════════════════════════════════════════════════════════════
#  SEED GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def choose_base_seed(base_seed: Optional[int], no_random_seed: bool) -> int:
    """
    Generate or return a seed for AF3 modelSeeds.

    Priority:
      1) base_seed if provided
      2) if no_random_seed: fixed seed = 1
      3) otherwise: random 32-bit seed

    Returns:
        A 32-bit seed in range [1, 2^32-1]
    """
    if base_seed is not None:
        if not (1 <= base_seed <= (2**32 - 1)):
            raise ValueError(f"--base_seed must be in [1, 2^32-1]. Got: {base_seed}")
        return base_seed

    if no_random_seed:
        return 1

    return secrets.randbelow(2**32 - 1) + 1


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT DIRECTORY SCANNING (OPTIMIZED)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OutputStatus:
    """Status of an AF3 output directory."""
    exists: bool = False
    complete: bool = False
    cif_count: int = 0
    path: Optional[Path] = None


def scan_output_directory(
    output_dir: Path,
    logger: Logger,
    min_cif_files: int = DEFAULT_MIN_CIF_FILES
) -> Dict[str, OutputStatus]:
    """
    Efficiently scan all output directories in a single pass.

    This is the KEY OPTIMIZATION - instead of checking each PDB's expected
    output individually (O(n) filesystem ops), we scan once and build a
    lookup table (O(1) lookups thereafter).

    Args:
        output_dir: The AF3 output directory to scan
        logger: Logger instance for progress reporting
        min_cif_files: Minimum CIF files required to consider output complete

    Returns:
        Dictionary mapping directory names (lowercase) to their status
    """
    logger.info(f"Scanning output directory: {output_dir}")
    logger.info(f"Completeness threshold: {min_cif_files} CIF files")

    output_status: Dict[str, OutputStatus] = {}

    if not output_dir.exists():
        logger.info("Output directory does not exist yet - all PDBs will be processed")
        return output_status

    # First pass: collect all subdirectories
    try:
        subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
    except PermissionError as e:
        logger.error(f"Permission denied accessing {output_dir}: {e}")
        return output_status

    total_dirs = len(subdirs)
    logger.info(f"Found {total_dirs:,} existing output directories to check")

    if total_dirs == 0:
        return output_status

    # Initialize progress tracker
    progress = ProgressTracker(total_dirs, "Scanning outputs")

    def check_directory_completeness(subdir: Path) -> Tuple[str, OutputStatus]:
        """Check if a directory has complete AF3 output."""
        samples_dir = subdir / "samples"
        cif_count = 0

        if samples_dir.is_dir():
            # Count CIF files efficiently using scandir
            try:
                cif_count = sum(
                    1 for entry in os.scandir(samples_dir)
                    if entry.is_file() and entry.name.endswith('.cif')
                )
            except (PermissionError, OSError):
                cif_count = 0

        status = OutputStatus(
            exists=True,
            complete=(cif_count >= min_cif_files),
            cif_count=cif_count,
            path=subdir
        )

        return subdir.name, status

    # Process directories with threading for faster I/O
    # Using threads because this is I/O bound (filesystem operations)
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 4)) as executor:
        futures = {executor.submit(check_directory_completeness, d): d for d in subdirs}

        for future in as_completed(futures):
            try:
                dir_name, status = future.result()
                output_status[dir_name] = status

                # Report progress
                msg = progress.update()
                if msg:
                    logger.progress(msg)

            except Exception as e:
                subdir = futures[future]
                logger.warning(f"Error checking {subdir}: {e}")

    logger.progress(progress.summary())

    # Summary statistics
    complete_count = sum(1 for s in output_status.values() if s.complete)
    incomplete_count = sum(1 for s in output_status.values() if s.exists and not s.complete)

    logger.metric("Complete outputs", complete_count)
    logger.metric("Incomplete outputs", incomplete_count)

    return output_status


def classify_pdbs_for_processing(
    all_pdbs: List[Path],
    output_status: Dict[str, OutputStatus],
    suffix: str,
    logger: Logger
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Classify PDB files based on their output status.

    This uses O(1) dictionary lookups instead of filesystem operations.

    Args:
        all_pdbs: All input PDB files
        output_status: Pre-scanned output status dictionary
        suffix: Output suffix to append to PDB names
        logger: Logger instance

    Returns:
        Tuple of (to_run, incomplete_pdbs, complete_pdbs)
    """
    logger.info(f"Classifying {len(all_pdbs):,} PDB files...")

    to_run = []
    incomplete_pdbs = []
    complete_pdbs = []

    progress = ProgressTracker(len(all_pdbs), "Classifying PDBs")

    for pdb in all_pdbs:
        expected_name = pdb.stem + suffix
        status = output_status.get(expected_name, OutputStatus())

        if not status.exists:
            to_run.append(pdb)
        elif status.complete:
            complete_pdbs.append(pdb)
        else:
            incomplete_pdbs.append(pdb)

        msg = progress.update()
        if msg:
            logger.progress(msg)

    logger.progress(progress.summary())

    return to_run, incomplete_pdbs, complete_pdbs


# ═══════════════════════════════════════════════════════════════════════════════
#  CLEANUP FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def cleanup_incomplete_outputs(
    incomplete_pdbs: List[Path],
    output_status: Dict[str, OutputStatus],
    suffix: str,
    logger: Logger
) -> List[Path]:
    """
    Remove incomplete output directories safely.

    Args:
        incomplete_pdbs: PDB files with incomplete outputs
        output_status: Output status dictionary
        suffix: Output suffix
        logger: Logger instance

    Returns:
        List of directories that were removed
    """
    if not incomplete_pdbs:
        return []

    logger.info(f"Cleaning up {len(incomplete_pdbs):,} incomplete output directories...")

    removed = []
    progress = ProgressTracker(len(incomplete_pdbs), "Cleanup")

    for pdb in incomplete_pdbs:
        expected_name = pdb.stem + suffix
        status = output_status.get(expected_name)

        if status and status.path and status.path.exists():
            try:
                shutil.rmtree(status.path, ignore_errors=True)
                removed.append(status.path)
            except Exception as e:
                logger.warning(f"Failed to remove {status.path}: {e}")

        msg = progress.update()
        if msg:
            logger.progress(msg)

    logger.progress(progress.summary())
    logger.success(f"Removed {len(removed):,} incomplete directories")

    return removed


# ═══════════════════════════════════════════════════════════════════════════════
#  JSON GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AF3Config:
    """Configuration for AF3 JSON generation."""
    pdb_chains: List[str]
    ligand_chains: List[str]
    ligand_types: List[str]
    ligand_ids: List[str]
    output_suffix: str
    ptm_specs: List[PTMSpec]
    base_seed: Optional[int]
    no_random_seed: bool
    n_terminus_tag: str = ""
    c_terminus_tag: str = ""


@dataclass
class ParsedPDB:
    """Lightweight result of parsing one PDB file (no BioPython objects retained)."""
    path: Path
    stem: str
    sequences: Dict[str, str]                 # chain_id -> raw 1-letter seq (no tags)
    mods_by_chain: Dict[str, List[Dict]]      # resolved PTMs, pre-tag-offset
    total_length: int                          # sum of protein chain lengths (sort key)
    error: Optional[str] = None                # set => skip this file, report it
    used_fallback: bool = False                # True if the BioPython safety net was used


# ── Parse-worker globals (set once per worker via the Pool initializer so the
#    AF3Config is not re-pickled for every one of >100k tasks). ──
_PARSE_CONFIG: Optional["AF3Config"] = None
_PARSE_USE_BIOPYTHON: bool = False


def _init_parse_worker(config: "AF3Config", use_biopython: bool) -> None:
    global _PARSE_CONFIG, _PARSE_USE_BIOPYTHON
    _PARSE_CONFIG = config
    _PARSE_USE_BIOPYTHON = use_biopython


def parse_one_pdb(pdb_path: Path) -> ParsedPDB:
    """Parse a single PDB into a ParsedPDB. Errors are isolated to this file.

    Uses the fast parser by default; falls back to BioPython for this one file if the
    fast parser raises or returns an empty chain (so an odd file is never silently
    dropped). With --use_biopython, BioPython is used directly.
    """
    config = _PARSE_CONFIG
    use_biopython = _PARSE_USE_BIOPYTHON
    stem = pdb_path.stem
    try:
        chains = config.pdb_chains
        need_remark = bool(config.ptm_specs)
        used_fallback = False

        if use_biopython:
            sequences, catalog = biopython_parse(pdb_path, chains, need_remark)
        else:
            try:
                sequences, catalog = fast_parse_pdb(pdb_path, chains, need_remark)
                if any(len(sequences.get(c, "")) == 0 for c in chains):
                    raise ValueError("fast parser returned an empty chain")
            except Exception:
                # Per-file safety net: retry this one file with BioPython.
                sequences, catalog = biopython_parse(pdb_path, chains, need_remark)
                used_fallback = True

        # Genuine empty-chain error (even BioPython could not fill it): same text as before.
        for chain in chains:
            if len(sequences.get(chain, "")) == 0:
                return ParsedPDB(
                    pdb_path, stem, {}, {}, 0,
                    error=f"Protein sequence length of [{pdb_path}] at chain [{chain}] is 0.",
                )

        mods_by_chain: Dict[str, List[Dict]] = {}
        if config.ptm_specs:
            mods_by_chain = build_mods_by_chain(catalog, config.ptm_specs, str(pdb_path))

        total_length = sum(len(sequences[c]) for c in chains)
        return ParsedPDB(pdb_path, stem, sequences, mods_by_chain, total_length,
                         used_fallback=used_fallback)
    except Exception as e:
        return ParsedPDB(pdb_path, stem, {}, {}, 0, error=f"{pdb_path}: {e}")


def parse_all_pdbs(
    to_run: List[Path],
    config: "AF3Config",
    num_workers: int,
    use_biopython: bool,
    logger: Logger,
) -> Tuple[List[ParsedPDB], List[str]]:
    """Parse every PDB in parallel. Returns (ok_results, error_messages).

    Results come back unordered; deterministic ordering is imposed later by the sort phase.
    """
    total = len(to_run)
    parser_name = "BioPython" if use_biopython else "fast parser"
    logger.info(f"Parsing {total:,} PDB files with the {parser_name} using {num_workers} workers")

    chunksize = max(1, total // (num_workers * 8)) if num_workers else 1
    progress = ProgressTracker(total, "Parsing PDBs")
    ok: List[ParsedPDB] = []
    errors: List[str] = []
    fallback_count = 0

    with multiprocessing.Pool(
        num_workers, initializer=_init_parse_worker, initargs=(config, use_biopython)
    ) as pool:
        for result in pool.imap_unordered(parse_one_pdb, to_run, chunksize=chunksize):
            if result.error:
                errors.append(result.error)
            else:
                ok.append(result)
                if result.used_fallback:
                    fallback_count += 1

            msg = progress.update()
            if msg:
                logger.progress(msg)

    logger.progress(progress.summary())

    if fallback_count:
        logger.warning(
            f"{fallback_count:,} file(s) fell back to BioPython because the fast parser "
            f"failed or returned an empty chain. These were still parsed correctly; consider "
            f"--use_biopython (or reporting these files) if the count is unexpectedly high."
        )

    return ok, errors


def sort_parsed_for_grouping(parsed_list: List[ParsedPDB]) -> List[ParsedPDB]:
    """Deterministically order by (total protein length ASC, full path ASC).

    Ligands and terminus tags are constant across all inputs, so they do not affect the
    relative ordering and are excluded from the key. Grouping equal/similar lengths into
    the same JSON minimizes AF3's per-token-count JAX recompilation.
    """
    return sorted(parsed_list, key=lambda r: (r.total_length, str(r.path)))


def build_af3_input_from_parsed(parsed: ParsedPDB, config: AF3Config) -> Dict:
    """
    Build a single AF3 input dictionary from an already-parsed PDB (no file I/O).

    Args:
        parsed: Pre-parsed PDB data (sequences + resolved PTMs)
        config: AF3 configuration

    Returns:
        AF3-compatible input dictionary
    """
    af3_input = {
        "name": parsed.stem + config.output_suffix,
        "sequences": []
    }

    n_tag_len = len(config.n_terminus_tag)

    # Add protein chains
    for chain in config.pdb_chains:
        # Apply terminus tags
        sequence = f"{config.n_terminus_tag}{parsed.sequences[chain]}{config.c_terminus_tag}"

        protein_obj = {
            "id": chain,
            "sequence": sequence,
            "unpairedMsa": "",
            "pairedMsa": "",
            "templates": "",
        }

        if chain in parsed.mods_by_chain:
            mods = parsed.mods_by_chain[chain]
            # Offset ptmPosition by N-terminus tag length
            if n_tag_len > 0:
                mods = [
                    {"ptmType": m["ptmType"], "ptmPosition": m["ptmPosition"] + n_tag_len}
                    for m in mods
                ]
            protein_obj["modifications"] = mods

        af3_input["sequences"].append({"protein": protein_obj})

    # Add ligands
    for ligand_id, ligand_type, ligand_chain in zip(
        config.ligand_ids, config.ligand_types, config.ligand_chains
    ):
        if ligand_type == "smiles":
            ligand_obj = {"id": ligand_chain, ligand_type: ligand_id}
        else:
            # ccdCodes expects a list (e.g. "['ZN']"); literal_eval is safe vs eval.
            ligand_obj = {"id": ligand_chain, ligand_type: ast.literal_eval(ligand_id)}

        af3_input["sequences"].append({"ligand": ligand_obj})

    # Set seed (random per input by default; unchanged semantics)
    base_seed = choose_base_seed(config.base_seed, config.no_random_seed)
    af3_input["modelSeeds"] = [base_seed]

    # Metadata
    af3_input["dialect"] = "alphafold3"
    af3_input["version"] = 1

    return af3_input


# ── Write-worker globals (set once per worker via the Pool initializer). ──
_WRITE_CONFIG: Optional[AF3Config] = None
_WRITE_JSON_PATH: Optional[str] = None
_WRITE_BASENAME: Optional[str] = None


def _init_write_worker(config: AF3Config, json_path: str, json_basename: str) -> None:
    global _WRITE_CONFIG, _WRITE_JSON_PATH, _WRITE_BASENAME
    _WRITE_CONFIG = config
    _WRITE_JSON_PATH = json_path
    _WRITE_BASENAME = json_basename


def write_batch(args: Tuple) -> Optional[str]:
    """
    Build AF3 inputs for one batch from already-parsed data and write the JSON file.

    Args:
        args: Tuple of (batch_idx, parsed_list)

    Returns:
        Error message if failed, None if successful
    """
    batch_idx, parsed_list = args
    try:
        af3_inputs = [build_af3_input_from_parsed(p, _WRITE_CONFIG) for p in parsed_list]
        output_file = Path(_WRITE_JSON_PATH) / f"{_WRITE_BASENAME}_{batch_idx}.json"
        with open(output_file, 'w') as f:
            json.dump(af3_inputs, f)
        return None
    except Exception as e:
        return f"Batch {batch_idx}: {str(e)}"


def write_json_files(
    ordered: List[ParsedPDB],
    json_path: Path,
    json_basename: str,
    num_per_run: int,
    config: AF3Config,
    logger: Logger,
    num_workers: int,
) -> Tuple[int, List[str]]:
    """
    Write AF3 JSON input files from the (already sorted) parsed list using multiprocessing.

    Returns:
        Tuple of (successful_count, error_list)
    """
    # Create output directory if needed
    json_path.mkdir(parents=True, exist_ok=True)

    # Split into batches of consecutive slices (batch_idx fixed by slice index, so the
    # output file names are deterministic regardless of worker completion order).
    batches = [
        (i // num_per_run, ordered[i:i + num_per_run])
        for i in range(0, len(ordered), num_per_run)
    ]

    total_batches = len(batches)
    logger.info(f"Processing {len(ordered):,} PDBs in {total_batches:,} batches")
    logger.info(f"Using {num_per_run} inputs per JSON file")

    if total_batches == 0:
        logger.info("No batches to process")
        return 0, []

    logger.info(f"Using {num_workers} worker processes")

    progress = ProgressTracker(total_batches, "Generating JSONs")
    errors = []
    successful = 0

    with multiprocessing.Pool(
        num_workers,
        initializer=_init_write_worker,
        initargs=(config, str(json_path), json_basename),
    ) as pool:
        for result in pool.imap_unordered(write_batch, batches):
            if result is None:
                successful += 1
            else:
                errors.append(result)

            msg = progress.update()
            if msg:
                logger.progress(msg)

    logger.progress(progress.summary())

    return successful, errors


# ═══════════════════════════════════════════════════════════════════════════════
#  JSON DIRECTORY SAFETY GATE
# ═══════════════════════════════════════════════════════════════════════════════

def check_json_directory_safety(
    json_path: Path,
    json_basename: str,
    num_to_run: int,
    num_per_run: int,
    args: argparse.Namespace,
    logger: Logger,
) -> None:
    """
    Guard the JSON output directory before any writing.

    Two scenarios are detected against the JSON files this run plans to write
    (planned = {json_basename}_{0..num_batches-1}.json):
      - Scenario 2 (name clash): planned outputs already exist. Halt by default; allow
        --clobber true to overwrite ONLY the clashing files, or --dangerous_clobber true
        to delete ALL .json first.
      - Scenario 1 (unrelated jsons): non-clashing .json files exist. Warn and require
        --other_nonclash_jsons_okay_in_json_dir true (or an interactive y/N) to proceed.
    May print, prompt, or sys.exit().
    """
    num_batches = math.ceil(num_to_run / num_per_run) if num_per_run > 0 else 0
    planned = {f"{json_basename}_{i}.json" for i in range(num_batches)}

    if not json_path.exists():
        return

    try:
        existing_files = sorted(p for p in json_path.glob("*.json") if p.is_file())
    except PermissionError as e:
        logger.error(f"Permission denied scanning {json_path}: {e}")
        sys.exit(1)

    existing = {p.name for p in existing_files}
    if not existing:
        return

    clashing = sorted(planned & existing)
    nonclashing = sorted(existing - planned)

    def _print_list(names: List[str], limit: int = 40) -> None:
        for n in names[:limit]:
            logger.info(f"    - {n}")
        if len(names) > limit:
            logger.info(f"    ... and {len(names) - limit:,} more")

    # ── --dangerous_clobber: highest precedence; wipe ALL json then proceed. ──
    if args.dangerous_clobber:
        logger.header("DANGEROUS CLOBBER ENABLED — DELETING ALL .json FILES")
        logger.warning(f"Deleting ALL {len(existing):,} .json file(s) in: {json_path}")
        deleted = 0
        for p in existing_files:
            logger.info(f"    deleting {p.name}")
            try:
                p.unlink()
                deleted += 1
            except Exception as e:
                logger.warning(f"Failed to delete {p}: {e}")
        logger.success(f"Deleted {deleted:,} .json file(s); continuing.")
        return

    # ── Scenario 2: planned output names already exist. ──
    if clashing:
        if not args.clobber:
            logger.header("OUTPUT NAME CLASH DETECTED — HALTING")
            logger.error(
                f"{len(clashing):,} of the {num_batches:,} JSON file(s) this run would write "
                f"already exist in:"
            )
            logger.error(f"      {json_path}")
            logger.error("Clashing output names that already exist:")
            _print_list(clashing)
            logger.warning("Refusing to overwrite. Choose ONE of:")
            logger.info("    1) Point --json_path at a different (empty) directory")
            logger.info("    2) Clear the existing JSON files in that directory yourself")
            logger.info("    3) --clobber true            -> overwrite ONLY the clashing files")
            logger.info("    4) --dangerous_clobber true  -> delete ALL .json in the dir first")
            if nonclashing:
                logger.warning(
                    f"Also note: {len(nonclashing):,} UNRELATED .json file(s) are present. With "
                    "--clobber you would additionally hit the unrelated-files gate and need "
                    "--other_nonclash_jsons_okay_in_json_dir true (or a y/N confirm)."
                )
            sys.exit(1)
        else:
            logger.header("CLOBBER ENABLED — OVERWRITING CLASHING OUTPUTS")
            logger.warning(
                f"writing {num_batches:,} files, {len(clashing):,} clash and will be overwritten"
            )
            logger.info("Files that will be overwritten:")
            _print_list(clashing)
            # Fall through to the unrelated-files gate for any non-clashing strangers.

    # ── Scenario 1: unrelated json files present (won't be overwritten). ──
    if nonclashing:
        logger.header("UNRELATED JSON FILES PRESENT IN OUTPUT DIRECTORY")
        logger.warning(
            f"{len(nonclashing):,} .json file(s) in {json_path} do NOT clash with this run's "
            "output names and will be LEFT IN PLACE:"
        )
        _print_list(nonclashing)

        if args.other_nonclash_jsons_okay_in_json_dir:
            logger.info(
                "--other_nonclash_jsons_okay_in_json_dir true -> continuing, leaving "
                f"{len(nonclashing):,} unrelated file(s) untouched."
            )
            return

        if sys.stdin.isatty():
            try:
                resp = input("  Proceed and leave these files in place? [y/N]: ").strip().lower()
            except EOFError:
                resp = ""
            if resp in ("y", "yes"):
                logger.info("Proceeding per user confirmation.")
                return
            logger.error("Aborted by user.")
            sys.exit(1)
        else:
            logger.error(
                "Non-interactive session: refusing to proceed with unrelated JSON files present."
            )
            logger.info(
                "Re-run with --other_nonclash_jsons_okay_in_json_dir true to continue anyway, "
                "or use a different/empty --json_path."
            )
            sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  FAST-PARSER VERIFICATION (vs BioPython)
# ═══════════════════════════════════════════════════════════════════════════════

def verify_fast_parser(
    ok_parsed: List[ParsedPDB],
    config: AF3Config,
    n: int,
    logger: Logger,
) -> None:
    """
    Cross-check the fast parser against BioPython on a random sample of already-parsed
    files, BEFORE any JSON is written. Aborts the run on any mismatch.

    Validates the exact in-memory data that will be written (parsed.sequences /
    parsed.mods_by_chain). The sample is drawn with a fixed-seed RNG so the checked set
    is reproducible across runs.
    """
    if n <= 0 or not ok_parsed:
        return

    sample_n = min(n, len(ok_parsed))
    rng = random.Random(0)
    sample = rng.sample(ok_parsed, sample_n)
    logger.info(
        f"Verifying fast parser against BioPython on {sample_n:,} randomly sampled file(s)..."
    )

    need_remark = bool(config.ptm_specs)
    mismatches = 0

    for parsed in sample:
        for chain in config.pdb_chains:
            bio_seq = get_protein_sequence(parsed.path, chain)
            fast_seq = parsed.sequences.get(chain, "")
            if bio_seq != fast_seq:
                mismatches += 1
                idx = next(
                    (i for i, (a, b) in enumerate(zip(bio_seq, fast_seq)) if a != b),
                    min(len(bio_seq), len(fast_seq)),
                )
                logger.error(f"SEQUENCE MISMATCH: {parsed.path} chain {chain}")
                logger.error(
                    f"  BioPython len={len(bio_seq)} | parser len={len(fast_seq)} | "
                    f"first diff @ index {idx}"
                )
                logger.error(f"  BioPython: ...{bio_seq[max(0, idx-10):idx+10]}...")
                logger.error(f"  parser:    ...{fast_seq[max(0, idx-10):idx+10]}...")

        if need_remark:
            bio_catalog = parse_remark666_catalog(parsed.path)
            try:
                bio_mods = build_mods_by_chain(bio_catalog, config.ptm_specs, str(parsed.path))
            except Exception as e:
                bio_mods = f"ERROR: {e}"
            if bio_mods != parsed.mods_by_chain:
                mismatches += 1
                logger.error(f"PTM MISMATCH: {parsed.path}")
                logger.error(f"  BioPython: {bio_mods}")
                logger.error(f"  parser:    {parsed.mods_by_chain}")

    if mismatches:
        logger.error(
            f"Fast-parser verification FAILED with {mismatches} mismatch(es). "
            "Aborting BEFORE writing any JSON."
        )
        logger.error(
            "Re-run with --use_biopython true to bypass the fast parser, and please report this."
        )
        sys.exit(1)

    logger.success(f"Fast-parser verification passed on {sample_n:,} file(s) — no mismatches.")


# ═══════════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate AlphaFold3 JSON input files from PDB structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage:
    %(prog)s --pdb_path ./pdbs --pdb_chain A --json_path ./jsons --output_path ./af3_out

  With ligands:
    %(prog)s --pdb_path ./pdbs --pdb_chain A --json_path ./jsons --output_path ./af3_out \\
             --ligand_chain B C --ligand_type smiles smiles --ligand_id "[Zn+2] [OH-]"

  Check and cleanup incomplete outputs:
    %(prog)s --pdb_path ./pdbs --pdb_chain A --json_path ./jsons --output_path ./af3_out \\
             --check_made_output --cleanup_incomplete_outputs

  Recursive search for PDBs in subdirectories (e.g., ProteinMPNN output):
    %(prog)s --pdb_path ./mpnn_out --pdb_chain A --json_path ./jsons --output_path ./af3_out \\
             --recursive --specific_depth 1

  Filter by filename prefix (process only files starting with a specific prefix):
    %(prog)s --pdb_path ./pdbs --pdb_chain A --json_path ./jsons --output_path ./af3_out \\
             --pdb_prefix ZAPP_i1_KCX_SW

  Disable length grouping (use legacy collection order instead of grouping by size):
    %(prog)s --pdb_path ./pdbs --pdb_chain A --json_path ./jsons --output_path ./af3_out \\
             --optimize_grouping_by_size false

  Overwrite only the clashing JSON outputs in a non-empty json dir:
    %(prog)s --pdb_path ./pdbs --pdb_chain A --json_path ./jsons --output_path ./af3_out \\
             --clobber true --other_nonclash_jsons_okay_in_json_dir true
        """
    )

    # Required arguments
    parser.add_argument(
        "--pdb_path", type=str, required=True,
        help="Directory containing input PDB files"
    )
    parser.add_argument(
        "--pdb_chain", type=str, nargs="+", required=True,
        help="Protein chain(s) to include (e.g., A or A B)"
    )
    parser.add_argument(
        "--json_path", type=str, required=True,
        help="Output directory for generated JSON files"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="AF3 output directory (for checking completed runs)"
    )

    # Optional arguments
    parser.add_argument(
        "--json_basename", type=str, default="AF3_input",
        help="Base name for output JSON files (default: AF3_input)"
    )
    parser.add_argument(
        "--num_input_per_run", type=int, default=5,
        help="Number of inputs per JSON file (default: 5)"
    )
    parser.add_argument(
        "--output_suffix", type=str, default="",
        help="Suffix to add to output names"
    )

    # Terminus tag arguments
    parser.add_argument(
        "--n_terminus_tag", type=str, default="",
        help="Amino acid sequence to prepend to all protein chains (e.g., MSG)"
    )
    parser.add_argument(
        "--c_terminus_tag", type=str, default="",
        help="Amino acid sequence to append to all protein chains (e.g., GSA)"
    )

    # Ligand arguments
    parser.add_argument(
        "--ligand_chain", type=str, nargs="+", default=[],
        help="Ligand chain ID(s) (e.g., B C)"
    )
    parser.add_argument(
        "--ligand_type", type=str, nargs="+", default=[],
        help="Ligand type(s): 'ccdCodes' or 'smiles'"
    )
    parser.add_argument(
        "--ligand_id", type=str, default="",
        help="Space-separated ligand identifiers"
    )

    # PTM arguments
    parser.add_argument(
        "--ptm_from_remark666", type=str, nargs="*", default=[],
        help="PTM specs from REMARK 666 (format: CHAIN/RES3/CATIDX:CCD)"
    )

    # Seed arguments
    parser.add_argument(
        "--base_seed", type=int, default=None,
        help="Fixed seed for AF3 modelSeeds (32-bit integer)"
    )
    parser.add_argument(
        "--no_random_seed", action="store_true",
        help="Use deterministic seed (1) instead of random"
    )

    # Output checking arguments
    parser.add_argument(
        "--check_made_output", action="store_true",
        help="Check for existing AF3 outputs and skip completed ones"
    )
    parser.add_argument(
        "--cleanup_incomplete_outputs", action="store_true",
        help="Remove incomplete output directories for reprocessing"
    )
    parser.add_argument(
        "--min_cif_files", type=int, default=DEFAULT_MIN_CIF_FILES,
        help=f"Minimum CIF files in samples/ to consider output complete (default: {DEFAULT_MIN_CIF_FILES})"
    )

    # Recursive search arguments
    parser.add_argument(
        "--recursive", action="store_true",
        help="Recursively search subdirectories for PDB files"
    )
    parser.add_argument(
        "--max_depth", type=int, default=None,
        help="Maximum directory depth to search (0=pdb_path itself, 1=direct subdirs, etc.). "
             "Only used with --recursive. Default: unlimited"
    )
    parser.add_argument(
        "--specific_depth", type=int, default=None,
        help="Search ONLY at this exact directory depth (0=pdb_path itself, 1=direct subdirs, etc.). "
             "Only used with --recursive. Mutually exclusive with --max_depth"
    )

    # Prefix filter argument
    parser.add_argument(
        "--pdb_prefix", type=str, nargs="+", default=None,
        help="Only process PDB files whose basename starts with one of these prefixes. "
             "Multiple prefixes accepted. Default: process all PDB files."
    )

    # Grouping / performance arguments
    parser.add_argument(
        "--optimize_grouping_by_size", type=str2bool, nargs="?", const=True, default=True,
        help="Group PDBs by protein length (smallest->largest, then path) before batching so "
             "each JSON holds same/similar-length structures, minimizing AF3 JAX recompilation. "
             "Deterministic. Default: true. Set 'false' for legacy collection order."
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Number of worker processes for parsing/writing (default: all CPUs)."
    )

    # Fast-parser verification / fallback arguments
    parser.add_argument(
        "--verify_fast_parser", type=int, default=DEFAULT_VERIFY_FAST_PARSER,
        help=f"After parsing, cross-check this many randomly sampled PDBs against BioPython and "
             f"abort before writing on any mismatch (default: {DEFAULT_VERIFY_FAST_PARSER}; 0 disables)."
    )
    parser.add_argument(
        "--use_biopython", type=str2bool, nargs="?", const=True, default=False,
        help="Use BioPython for the entire parse instead of the fast parser (slower; full "
             "override). Skips fast-parser verification. Default: false."
    )

    # JSON-directory safety arguments
    parser.add_argument(
        "--other_nonclash_jsons_okay_in_json_dir", type=str2bool, nargs="?", const=True,
        default=False,
        help="Auto-continue (no prompt) when UNRELATED, non-clashing .json files already exist "
             "in the json dir. Default: false."
    )
    parser.add_argument(
        "--clobber", type=str2bool, nargs="?", const=True, default=False,
        help="Overwrite ONLY the JSON files whose names clash with this run's planned outputs "
             "(prints exactly which). Default: false."
    )
    parser.add_argument(
        "--dangerous_clobber", type=str2bool, nargs="?", const=True, default=False,
        help="DANGEROUS: delete ALL .json files in the json dir before writing (prints each). "
             "Highest precedence. Default: false."
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace, logger: Logger) -> None:
    """Validate argument combinations and values."""
    # Parse ligand IDs
    ligand_ids = args.ligand_id.split(" ") if args.ligand_id else []
    ligand_ids = [lid for lid in ligand_ids if lid]  # Remove empty strings

    # Check ligand argument counts match
    if not (len(args.ligand_chain) == len(args.ligand_type) == len(ligand_ids)):
        raise ValueError(
            f"Ligand argument count mismatch: "
            f"ligand_chain={len(args.ligand_chain)}, "
            f"ligand_type={len(args.ligand_type)}, "
            f"ligand_id={len(ligand_ids)}"
        )

    # Validate ligand types
    valid_types = {"ccdCodes", "smiles"}
    invalid_types = set(args.ligand_type) - valid_types
    if invalid_types:
        raise ValueError(
            f"Invalid ligand type(s): {invalid_types}. Must be 'ccdCodes' or 'smiles'"
        )

    # Validate SMILES escaping
    for i, ligand_id in enumerate(ligand_ids):
        if args.ligand_type[i] != "smiles":
            continue
        parts = ligand_id.split("/")
        if len(parts) > 1:
            for j, part in enumerate(parts[1:], 1):
                if part and parts[j-1]:
                    raise ValueError(
                        "SMILES string may need JSON escaping. "
                        "Backslashes must be escaped as \\\\. "
                        "See: https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md"
                    )

    # Store parsed ligand IDs
    args.ligand_ids_parsed = ligand_ids

    # Validate paths
    pdb_path = Path(args.pdb_path)
    if not pdb_path.exists():
        raise ValueError(f"PDB path does not exist: {pdb_path}")
    if not pdb_path.is_dir():
        raise ValueError(f"PDB path is not a directory: {pdb_path}")

    # Validate recursive search arguments
    if args.max_depth is not None and args.specific_depth is not None:
        raise ValueError(
            "--max_depth and --specific_depth are mutually exclusive. "
            "Use --max_depth N for depths 0..N, or --specific_depth N for exactly depth N."
        )

    if args.max_depth is not None and args.max_depth < 0:
        raise ValueError(f"--max_depth must be non-negative. Got: {args.max_depth}")

    if args.specific_depth is not None and args.specific_depth < 0:
        raise ValueError(f"--specific_depth must be non-negative. Got: {args.specific_depth}")

    if not args.recursive and (args.max_depth is not None or args.specific_depth is not None):
        logger.warning(
            "Depth flags (--max_depth / --specific_depth) imply --recursive. "
            "Enabling recursive mode automatically."
        )
        args.recursive = True

    # Validate terminus tags contain only valid amino acid characters
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    for flag_name in ("n_terminus_tag", "c_terminus_tag"):
        val = getattr(args, flag_name)
        if val:
            invalid_chars = set(val.upper()) - valid_aa
            if invalid_chars:
                raise ValueError(
                    f"--{flag_name} contains invalid amino acid characters: {invalid_chars}. "
                    f"Must contain only standard single-letter amino acid codes."
                )
            setattr(args, flag_name, val.upper())

    # Validate performance / verification arguments
    if args.verify_fast_parser < 0:
        raise ValueError(f"--verify_fast_parser must be non-negative. Got: {args.verify_fast_parser}")

    if args.num_workers is not None and args.num_workers < 1:
        raise ValueError(f"--num_workers must be >= 1. Got: {args.num_workers}")

    if args.num_input_per_run < 1:
        raise ValueError(f"--num_input_per_run must be >= 1. Got: {args.num_input_per_run}")

    if args.clobber and args.dangerous_clobber:
        logger.warning(
            "Both --clobber and --dangerous_clobber set; --dangerous_clobber takes precedence "
            "(ALL .json files will be deleted)."
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point."""
    logger = Logger()
    logger.header("AlphaFold3 JSON Input Generator")

    # Parse arguments
    logger.step("Parsing arguments")
    args = parse_arguments()

    try:
        validate_arguments(args, logger)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Log configuration
    logger.metric("PDB directory", args.pdb_path)
    logger.metric("JSON output", args.json_path)
    logger.metric("AF3 output", args.output_path)
    logger.metric("Protein chains", ", ".join(args.pdb_chain))
    logger.metric("Inputs per JSON", args.num_input_per_run)

    if args.ligand_chain:
        logger.metric("Ligand chains", ", ".join(args.ligand_chain))
        logger.metric("Ligand types", ", ".join(args.ligand_type))

    if args.ptm_from_remark666:
        logger.metric("PTM specs", ", ".join(args.ptm_from_remark666))

    if args.n_terminus_tag:
        logger.metric("N-terminus tag", args.n_terminus_tag)

    if args.c_terminus_tag:
        logger.metric("C-terminus tag", args.c_terminus_tag)

    if args.check_made_output:
        logger.metric("Min CIF files for complete", args.min_cif_files)

    if args.recursive:
        logger.metric("Search mode", "recursive")
        if args.specific_depth is not None:
            logger.metric("Search depth", f"exactly {args.specific_depth}")
        elif args.max_depth is not None:
            logger.metric("Search depth", f"0 to {args.max_depth}")
        else:
            logger.metric("Search depth", "unlimited")

    if args.pdb_prefix:
        logger.metric("PDB prefix filter", ", ".join(args.pdb_prefix))

    # Resolve worker count (default: all CPUs)
    num_workers = args.num_workers if args.num_workers else (os.cpu_count() or 1)

    logger.metric("Length grouping", "on" if args.optimize_grouping_by_size else "off (legacy order)")
    logger.metric("Parser", "BioPython" if args.use_biopython else "fast parser")
    logger.metric("Worker processes", num_workers)
    if not args.use_biopython:
        logger.metric(
            "Verify fast parser",
            f"{args.verify_fast_parser} sampled" if args.verify_fast_parser > 0 else "disabled",
        )

    # Parse PTM specs
    ptm_specs = parse_ptm_specs(args.ptm_from_remark666) if args.ptm_from_remark666 else []

    # Create configuration object
    config = AF3Config(
        pdb_chains=args.pdb_chain,
        ligand_chains=args.ligand_chain,
        ligand_types=args.ligand_type,
        ligand_ids=args.ligand_ids_parsed,
        output_suffix=args.output_suffix,
        ptm_specs=ptm_specs,
        base_seed=args.base_seed,
        no_random_seed=args.no_random_seed,
        n_terminus_tag=args.n_terminus_tag,
        c_terminus_tag=args.c_terminus_tag,
    )

    # Collect input PDB files
    logger.step("Collecting input PDB files")
    pdb_dir = Path(args.pdb_path)
    all_pdbs = collect_pdb_files(
        pdb_dir,
        recursive=args.recursive,
        max_depth=args.max_depth,
        specific_depth=args.specific_depth,
        logger=logger,
        pdb_prefixes=args.pdb_prefix,
    )
    logger.metric("Total PDB files found", len(all_pdbs))

    if len(all_pdbs) == 0:
        if args.recursive:
            depth_info = ""
            if args.specific_depth is not None:
                depth_info = f" at depth {args.specific_depth}"
            elif args.max_depth is not None:
                depth_info = f" up to depth {args.max_depth}"
            logger.warning(f"No PDB files found recursively{depth_info} in: {pdb_dir}")
        else:
            logger.warning("No PDB files found in input directory")
        logger.final_summary()
        return

    # Determine which PDBs need processing
    if args.check_made_output:
        logger.step("Scanning existing outputs")
        output_dir = Path(args.output_path)
        output_status = scan_output_directory(output_dir, logger, args.min_cif_files)

        logger.step("Classifying PDB files")
        to_run, incomplete_pdbs, complete_pdbs = classify_pdbs_for_processing(
            all_pdbs, output_status, args.output_suffix, logger
        )

        logger.info("Classification results:")
        logger.metric("Already complete", len(complete_pdbs))
        logger.metric("Incomplete (need rerun)", len(incomplete_pdbs))
        logger.metric("Not yet started", len(to_run))

        # Handle incomplete outputs
        if incomplete_pdbs:
            if args.cleanup_incomplete_outputs:
                logger.step("Cleaning up incomplete outputs")

                # Show examples before cleanup
                examples = incomplete_pdbs[:3]
                logger.info("Examples of incomplete outputs:")
                for ex in examples:
                    expected_name = ex.stem + args.output_suffix
                    status = output_status.get(expected_name)
                    if status:
                        logger.info(f"  {expected_name}: {status.cif_count} CIF files (need {args.min_cif_files})")

                cleanup_incomplete_outputs(incomplete_pdbs, output_status, args.output_suffix, logger)
                to_run.extend(incomplete_pdbs)
            else:
                logger.error(
                    f"Found {len(incomplete_pdbs)} incomplete outputs. "
                    "Use --cleanup_incomplete_outputs to remove them for reprocessing."
                )
                example = incomplete_pdbs[0].stem + args.output_suffix
                logger.info(f"Example incomplete output: {example}")
                sys.exit(1)
    else:
        to_run = all_pdbs
        logger.info("--check_made_output not specified; processing all PDBs")

    # Generate JSON files
    logger.step("Preparing to generate AF3 JSON files")
    logger.metric("PDBs to process", len(to_run))

    if len(to_run) == 0:
        logger.success("All PDBs already have complete outputs - nothing to do!")
        logger.final_summary()
        return

    json_path = Path(args.json_path)

    # Auto-derive json_basename from pdb_prefix if user didn't override
    json_basename = args.json_basename
    if args.pdb_prefix and args.json_basename == "AF3_input":
        json_basename = "AF3_input_" + "_".join(args.pdb_prefix)

    # Safety gate: inspect the JSON output dir BEFORE any expensive parsing/writing.
    # Planned file names depend only on json_basename + ceil(len(to_run)/num_per_run).
    logger.step("Checking JSON output directory")
    check_json_directory_safety(
        json_path, json_basename, len(to_run), args.num_input_per_run, args, logger
    )

    # Parse phase (parallel, all CPUs)
    logger.step("Parsing PDB files")
    ok_parsed, parse_errors = parse_all_pdbs(
        to_run, config, num_workers, args.use_biopython, logger
    )
    logger.metric("Successfully parsed", len(ok_parsed))
    if parse_errors:
        logger.warning(f"Skipped {len(parse_errors):,} file(s) due to parse/PTM errors:")
        for err in parse_errors[:10]:
            logger.error(f"  {err}")
        if len(parse_errors) > 10:
            logger.error(f"  ... and {len(parse_errors) - 10:,} more")

    if not ok_parsed:
        logger.error("No PDB files parsed successfully - nothing to write.")
        logger.final_summary()
        sys.exit(1)

    # Verify phase (sampled, before any write; skipped under --use_biopython)
    if not args.use_biopython and args.verify_fast_parser > 0:
        logger.step("Verifying fast parser against BioPython")
        verify_fast_parser(ok_parsed, config, args.verify_fast_parser, logger)

    # Sort phase (deterministic)
    logger.step("Ordering PDBs")
    if args.optimize_grouping_by_size:
        ordered = sort_parsed_for_grouping(ok_parsed)
        logger.info("Grouped by protein length (smallest -> largest, then path).")
        logger.metric("Length range", f"{ordered[0].total_length} - {ordered[-1].total_length} residues")
    else:
        # Preserve legacy collection order exactly (imap returned results unordered).
        by_path = {r.path: r for r in ok_parsed}
        ordered = [by_path[p] for p in to_run if p in by_path]
        logger.info("Length grouping disabled; using legacy collection order.")

    # Write phase (parallel)
    logger.step("Generating AF3 JSON files")
    successful, write_errors = write_json_files(
        ordered, json_path, json_basename,
        args.num_input_per_run, config, logger, num_workers
    )

    # Report results
    logger.step("Summary")
    logger.metric("JSON files created", successful)
    logger.metric("Inputs written", len(ordered))
    if parse_errors:
        logger.metric("Files skipped (parse errors)", len(parse_errors))

    if write_errors:
        logger.warning(f"Encountered {len(write_errors)} write error(s):")
        for err in write_errors[:10]:  # Show first 10 errors
            logger.error(f"  {err}")
        if len(write_errors) > 10:
            logger.error(f"  ... and {len(write_errors) - 10} more errors")
    elif parse_errors:
        logger.success("All parsed inputs written successfully (some files were skipped above).")
    else:
        logger.success("All batches processed successfully!")

    logger.final_summary()


if __name__ == "__main__":
    main()
