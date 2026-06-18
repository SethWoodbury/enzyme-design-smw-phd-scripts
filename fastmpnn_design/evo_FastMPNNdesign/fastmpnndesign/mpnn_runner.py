"""
LigandMPNN/EnhancedMPNN subprocess runner.

Handles command construction and execution for MPNN sequence design,
with support for direct execution and Apptainer containerized execution.
"""

import json
import subprocess
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re

from fastmpnndesign.config import MPNNConfig, CatalyticResidue
from fastmpnndesign.remark666 import catres_to_fixed_residues
from fastmpnndesign.utils import ensure_dir, read_json, write_json
from fastmpnndesign.logging_config import get_logger

logger = get_logger("mpnn_runner")


@dataclass
class MPNNResult:
    """Result from an MPNN run."""
    success: bool
    output_dir: Path
    sequences_dir: Optional[Path] = None
    packed_dir: Optional[Path] = None
    n_sequences: int = 0
    sequences: List[Dict[str, Any]] = None
    command: str = ""
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""

    def __post_init__(self):
        if self.sequences is None:
            self.sequences = []


def build_fixed_residues_json(
    pdb_path: Path,
    fixed_residues: List[str],
    output_path: Path
) -> Path:
    """
    Create fixed_residues_multi JSON file for MPNN.

    Args:
        pdb_path: Path to input PDB.
        fixed_residues: List of residue IDs in format ['A150', 'A152', ...].
        output_path: Path to write JSON file.

    Returns:
        Path to created JSON file.
    """
    # MPNN expects the full path as the key, not just the filename
    pdb_key = str(pdb_path)

    # Format for MPNN: {"/full/path/to/pdb.pdb": ["A150", "A152", ...]}
    data = {pdb_key: fixed_residues}

    write_json(output_path, data)
    logger.debug(f"Wrote fixed residues JSON: {output_path}")
    logger.debug(f"  PDB key: {pdb_key}")
    logger.debug(f"  Fixed residues ({len(fixed_residues)} total): {fixed_residues[:10]}{'...' if len(fixed_residues) > 10 else ''}")

    return output_path


def build_mpnn_command(
    pdb_path: Path,
    output_dir: Path,
    fixed_residues_json: Optional[Path],
    config: MPNNConfig,
    packed_suffix: str = "_packed_"
) -> List[str]:
    """
    Build MPNN command line.

    Args:
        pdb_path: Path to input PDB.
        output_dir: Output directory for MPNN results.
        fixed_residues_json: Path to fixed residues JSON file.
        config: MPNN configuration.
        packed_suffix: Suffix for packed PDB files.

    Returns:
        List of command arguments.
    """
    cmd = []

    # Apptainer wrapper if requested
    if config.use_apptainer:
        cmd.extend([
            "apptainer", "exec",
            str(config.apptainer_image),
            "python"
        ])
    else:
        cmd.append("python")

    # MPNN runner script
    cmd.append(str(config.mpnn_runner))

    # Required arguments
    cmd.extend([
        "--model_type", config.model_type,
        "--pdb_path", str(pdb_path),
        "--out_folder", str(output_dir),
        "--temperature", str(config.temperature),
        "--number_of_batches", str(config.number_of_batches),
        "--batch_size", str(config.batch_size),
    ])

    # Side chain packing
    cmd.extend([
        "--pack_side_chains", "1" if config.pack_side_chains else "0",
        "--sc_num_denoising_steps", str(config.sc_num_denoising_steps),
    ])

    # Fixed residues
    if fixed_residues_json and fixed_residues_json.exists():
        cmd.extend(["--fixed_residues_multi", str(fixed_residues_json)])

    # Enhancement model
    if config.enhance_model:
        cmd.extend(["--enhance", config.enhance_model])

    # Omit amino acids
    if config.omit_AA:
        cmd.extend(["--omit_AA", config.omit_AA])

    # Side chain context
    cmd.extend([
        "--ligand_mpnn_use_side_chain_context",
        "1" if config.ligand_mpnn_use_side_chain_context else "0"
    ])

    # Packed suffix
    cmd.extend(["--packed_suffix", packed_suffix])

    # Don't repack everything (we want specific fixed residues)
    cmd.extend(["--repack_everything", "0"])

    return cmd


def run_mpnn(
    pdb_path: Path,
    output_dir: Path,
    catres_list: Optional[List[CatalyticResidue]] = None,
    config: Optional[MPNNConfig] = None,
    additional_fixed: Optional[List[str]] = None,
    dry_run: bool = False
) -> MPNNResult:
    """
    Run LigandMPNN/EnhancedMPNN on a PDB file.

    Args:
        pdb_path: Path to input PDB.
        output_dir: Output directory for results.
        catres_list: Catalytic residues to fix (optional).
        config: MPNN configuration.
        additional_fixed: Additional residues to fix.
        dry_run: If True, print command without executing.

    Returns:
        MPNNResult with execution details.
    """
    if config is None:
        config = MPNNConfig()

    pdb_path = Path(pdb_path).resolve()
    output_dir = Path(output_dir).resolve()
    ensure_dir(output_dir)

    # Build fixed residues list
    fixed_residues = []
    if catres_list:
        fixed_residues.extend(catres_to_fixed_residues(catres_list))
    if additional_fixed:
        fixed_residues.extend(additional_fixed)

    # Create fixed residues JSON
    fixed_json = None
    if fixed_residues:
        fixed_json = output_dir / "fixed_residues.json"
        build_fixed_residues_json(pdb_path, fixed_residues, fixed_json)

    # Build command
    cmd = build_mpnn_command(
        pdb_path, output_dir, fixed_json, config
    )
    cmd_str = shlex.join(cmd)

    logger.info(f"MPNN command: {cmd_str}")

    if dry_run:
        logger.info("Dry run - command not executed")
        return MPNNResult(
            success=True,
            output_dir=output_dir,
            command=cmd_str
        )

    # Execute command
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        success = result.returncode == 0

        if not success:
            logger.error(f"MPNN failed with return code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")

        # Find output directories
        seqs_dir = output_dir / "seqs"
        packed_dir = output_dir / "packed"

        # Parse sequences if available
        sequences = []
        n_sequences = 0
        if seqs_dir.exists():
            sequences = parse_mpnn_sequences(seqs_dir)
            n_sequences = len(sequences)

        return MPNNResult(
            success=success,
            output_dir=output_dir,
            sequences_dir=seqs_dir if seqs_dir.exists() else None,
            packed_dir=packed_dir if packed_dir.exists() else None,
            n_sequences=n_sequences,
            sequences=sequences,
            command=cmd_str,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr
        )

    except subprocess.TimeoutExpired:
        logger.error("MPNN timed out after 1 hour")
        return MPNNResult(
            success=False,
            output_dir=output_dir,
            command=cmd_str,
            returncode=-1,
            stderr="Timeout after 1 hour"
        )
    except Exception as e:
        logger.error(f"MPNN execution failed: {e}")
        return MPNNResult(
            success=False,
            output_dir=output_dir,
            command=cmd_str,
            returncode=-1,
            stderr=str(e)
        )


def parse_mpnn_sequences(seqs_dir: Path) -> List[Dict[str, Any]]:
    """
    Parse MPNN output sequences from FASTA files.

    Args:
        seqs_dir: Path to MPNN seqs/ output directory.

    Returns:
        List of sequence dictionaries with 'name', 'sequence', 'score'.
    """
    sequences = []

    for fasta_file in seqs_dir.glob("*.fa"):
        with open(fasta_file, 'r') as f:
            content = f.read()

        # Parse FASTA entries
        entries = content.strip().split('>')
        for entry in entries:
            if not entry.strip():
                continue

            lines = entry.strip().split('\n')
            header = lines[0]
            sequence = ''.join(lines[1:])

            # Parse header for score
            # Format: name, score=X.XXX, ...
            score = None
            score_match = re.search(r'score=([-\d.]+)', header)
            if score_match:
                score = float(score_match.group(1))

            sequences.append({
                'name': header.split(',')[0].strip(),
                'sequence': sequence,
                'score': score,
                'source_file': str(fasta_file)
            })

    logger.info(f"Parsed {len(sequences)} sequences from MPNN output")
    return sequences


def get_packed_pdbs(output_dir: Path) -> List[Path]:
    """
    Get list of packed PDB files from MPNN output.

    Args:
        output_dir: MPNN output directory.

    Returns:
        List of paths to packed PDB files.
    """
    packed_dir = output_dir / "packed"
    if not packed_dir.exists():
        return []

    pdbs = list(packed_dir.glob("*.pdb"))
    logger.debug(f"Found {len(pdbs)} packed PDB files")
    return sorted(pdbs)


def parse_mpnn_scores_json(output_dir: Path) -> Dict[str, Any]:
    """
    Parse MPNN scores.json if available.

    Args:
        output_dir: MPNN output directory.

    Returns:
        Dictionary with score information.
    """
    scores_file = output_dir / "scores.json"
    if not scores_file.exists():
        return {}

    try:
        return read_json(scores_file)
    except Exception as e:
        logger.warning(f"Could not parse scores.json: {e}")
        return {}


def validate_mpnn_setup(config: MPNNConfig) -> Tuple[bool, List[str]]:
    """
    Validate MPNN configuration and availability.

    Args:
        config: MPNN configuration.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors = []

    # Check MPNN runner exists
    if not config.mpnn_runner.exists():
        errors.append(f"MPNN runner not found: {config.mpnn_runner}")

    # Check Apptainer image if using Apptainer
    if config.use_apptainer:
        if not config.apptainer_image.exists():
            errors.append(f"Apptainer image not found: {config.apptainer_image}")

        # Check apptainer is available
        try:
            result = subprocess.run(
                ["apptainer", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                errors.append("Apptainer not available or not working")
        except Exception:
            errors.append("Apptainer command not found")

    is_valid = len(errors) == 0
    return is_valid, errors
