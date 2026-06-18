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
import sys
import json
import time
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
    pattern = re.compile(
        r"MOTIF\s+(\S+)\s+([A-Z]{3})\s+(\d+)([A-Z]?)\s+(\d+)\s+(\d+)",
        flags=re.IGNORECASE,
    )

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


def build_af3_input(pdb_path: Path, config: AF3Config) -> Dict:
    """
    Build a single AF3 input dictionary from a PDB file.

    Args:
        pdb_path: Path to input PDB file
        config: AF3 configuration

    Returns:
        AF3-compatible input dictionary
    """
    af3_input = {
        "name": pdb_path.stem + config.output_suffix,
        "sequences": []
    }

    # Resolve PTMs if specified
    mods_by_chain = {}
    if config.ptm_specs:
        catalog = parse_remark666_catalog(pdb_path)
        mods_by_chain = build_mods_by_chain(catalog, config.ptm_specs, str(pdb_path))

    # Add protein chains
    for chain in config.pdb_chains:
        sequence = get_protein_sequence(pdb_path, chain)
        if len(sequence) == 0:
            raise ValueError(
                f"Protein sequence length of [{pdb_path}] at chain [{chain}] is 0."
            )

        protein_obj = {
            "id": chain,
            "sequence": sequence,
            "unpairedMsa": "",
            "pairedMsa": "",
            "templates": "",
        }

        if chain in mods_by_chain:
            protein_obj["modifications"] = mods_by_chain[chain]

        af3_input["sequences"].append({"protein": protein_obj})

    # Add ligands
    for ligand_id, ligand_type, ligand_chain in zip(
        config.ligand_ids, config.ligand_types, config.ligand_chains
    ):
        if ligand_type == "smiles":
            ligand_obj = {"id": ligand_chain, ligand_type: ligand_id}
        else:
            # ccdCodes expects a list
            ligand_obj = {"id": ligand_chain, ligand_type: eval(ligand_id)}

        af3_input["sequences"].append({"ligand": ligand_obj})

    # Set seed
    base_seed = choose_base_seed(config.base_seed, config.no_random_seed)
    af3_input["modelSeeds"] = [base_seed]

    # Metadata
    af3_input["dialect"] = "alphafold3"
    af3_input["version"] = 1

    return af3_input


def process_batch(args: Tuple) -> Optional[str]:
    """
    Process a batch of PDB files and write JSON output.

    Args:
        args: Tuple of (batch_idx, pdb_list, json_path, json_basename, config)

    Returns:
        Error message if failed, None if successful
    """
    batch_idx, pdb_list, json_path, json_basename, config = args

    try:
        af3_inputs = []
        for pdb_path in pdb_list:
            af3_input = build_af3_input(pdb_path, config)
            af3_inputs.append(af3_input)

        output_file = Path(json_path) / f"{json_basename}_{batch_idx}.json"
        with open(output_file, 'w') as f:
            json.dump(af3_inputs, f)

        return None
    except Exception as e:
        return f"Batch {batch_idx}: {str(e)}"


def generate_json_files(
    pdbs_to_process: List[Path],
    json_path: Path,
    json_basename: str,
    num_per_run: int,
    config: AF3Config,
    logger: Logger
) -> Tuple[int, List[str]]:
    """
    Generate AF3 JSON input files using multiprocessing.

    Args:
        pdbs_to_process: List of PDB files to process
        json_path: Output directory for JSON files
        json_basename: Base name for JSON files
        num_per_run: Number of inputs per JSON file
        config: AF3 configuration
        logger: Logger instance

    Returns:
        Tuple of (successful_count, error_list)
    """
    # Create output directory if needed
    json_path.mkdir(parents=True, exist_ok=True)

    # Split into batches
    batches = []
    for i in range(0, len(pdbs_to_process), num_per_run):
        batch_pdbs = pdbs_to_process[i:i + num_per_run]
        batch_idx = i // num_per_run
        batches.append((batch_idx, batch_pdbs, str(json_path), json_basename, config))

    total_batches = len(batches)
    logger.info(f"Processing {len(pdbs_to_process):,} PDBs in {total_batches:,} batches")
    logger.info(f"Using {num_per_run} inputs per JSON file")

    if total_batches == 0:
        logger.info("No batches to process")
        return 0, []

    # Use multiprocessing for CPU-bound work
    num_workers = max(1, (os.cpu_count() or 4) - 1)
    logger.info(f"Using {num_workers} worker processes")

    progress = ProgressTracker(total_batches, "Generating JSONs")
    errors = []
    successful = 0

    with multiprocessing.Pool(num_workers) as pool:
        for result in pool.imap_unordered(process_batch, batches):
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

    if args.check_made_output:
        logger.metric("Min CIF files for complete", args.min_cif_files)

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
        no_random_seed=args.no_random_seed
    )

    # Collect input PDB files
    logger.step("Collecting input PDB files")
    pdb_dir = Path(args.pdb_path)
    all_pdbs = sorted([p for p in pdb_dir.iterdir() if p.is_file() and p.suffix == ".pdb"])
    logger.metric("Total PDB files found", len(all_pdbs))

    if len(all_pdbs) == 0:
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
    logger.step("Generating AF3 JSON files")
    logger.metric("PDBs to process", len(to_run))

    if len(to_run) == 0:
        logger.success("All PDBs already have complete outputs - nothing to do!")
        logger.final_summary()
        return

    json_path = Path(args.json_path)
    successful, errors = generate_json_files(
        to_run, json_path, args.json_basename,
        args.num_input_per_run, config, logger
    )

    # Report results
    logger.step("Summary")
    logger.metric("JSON files created", successful)
    logger.metric("Total inputs written", len(to_run))

    if errors:
        logger.warning(f"Encountered {len(errors)} errors:")
        for err in errors[:10]:  # Show first 10 errors
            logger.error(f"  {err}")
        if len(errors) > 10:
            logger.error(f"  ... and {len(errors) - 10} more errors")
    else:
        logger.success("All batches processed successfully!")

    logger.final_summary()


if __name__ == "__main__":
    main()
