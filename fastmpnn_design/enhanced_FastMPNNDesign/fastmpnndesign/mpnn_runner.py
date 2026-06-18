"""
MPNN runner for enhanced_fastmpnndesign.

Handles LigandMPNN/EnhancedMPNN execution via subprocess or apptainer.
Provides safe command construction with full logging.
"""

import sys
import subprocess
import shlex
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add package directory to path for standalone execution
_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import MPNNConfig, CatalyticResidue
from constants import (
    DEFAULT_MPNN_RUNNER, DEFAULT_MODEL_TYPE, DEFAULT_ENHANCE_MODEL,
    DEFAULT_APPTAINER_IMAGE
)
from logging_config import get_logger

logger = get_logger("mpnn_runner")


@dataclass
class MPNNResult:
    """Result from an MPNN run."""
    success: bool
    output_dir: Path
    sequences: Optional[List[Dict[str, Any]]] = None
    packed_pdbs: Optional[List[Path]] = None
    command: str = ""
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def build_fixed_residues_json(
    pdb_path: Path,
    fixed_residues: List[str],
    output_path: Path
) -> Path:
    """
    Create fixed_residues_multi JSON file for MPNN.

    Args:
        pdb_path: Path to PDB file (used as key in JSON)
        fixed_residues: List of residue IDs in format ['A150', 'A152', ...]
        output_path: Path to write JSON file

    Returns:
        Path to created JSON file
    """
    # Use PDB filename as key
    pdb_key = str(pdb_path)

    data = {pdb_key: fixed_residues}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.debug(f"Wrote fixed residues JSON: {output_path} ({len(fixed_residues)} residues)")
    return output_path


def build_mpnn_command(
    pdb_path: Path,
    output_dir: Path,
    fixed_residues_json: Optional[Path],
    config: MPNNConfig,
    packed_suffix: Optional[str] = None
) -> List[str]:
    """
    Build MPNN command line with safe argument handling.

    Exposes all key LigandMPNN args as specified:
    - model_type, input_pdb, output_dir, temperature
    - number_of_batches, batch_size, pack_side_chains
    - fixed_residues_multi, sc_num_denoising_steps
    - repack_everything, omit_AA, packed_suffix
    - ligand_mpnn_use_side_chain_context

    Args:
        pdb_path: Path to input PDB
        output_dir: Output directory
        fixed_residues_json: Path to fixed residues JSON (optional)
        config: MPNN configuration
        packed_suffix: Suffix for packed output files (optional)

    Returns:
        Command as list of strings
    """
    cmd = []

    # Apptainer wrapper if requested
    if config.use_apptainer:
        apptainer_image = config.apptainer_image or Path(DEFAULT_APPTAINER_IMAGE)
        cmd.extend([
            "apptainer", "exec",
            str(apptainer_image),
            "python"
        ])
        logger.debug(f"Using apptainer image: {apptainer_image}")
    else:
        cmd.append("python")

    # MPNN runner script
    mpnn_runner = config.mpnn_runner or Path(DEFAULT_MPNN_RUNNER)
    cmd.append(str(mpnn_runner))

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

    # EnhancedMPNN
    if config.use_enhanced_mpnn and config.enhance_model:
        cmd.extend(["--enhance", config.enhance_model])
        logger.debug(f"Using EnhancedMPNN model: {config.enhance_model}")

    # Omit amino acids
    if config.omit_AA:
        cmd.extend(["--omit_AA", config.omit_AA])

    # Side chain context
    cmd.extend([
        "--ligand_mpnn_use_side_chain_context",
        "1" if config.ligand_mpnn_use_side_chain_context else "0"
    ])

    # Repack everything flag
    cmd.extend(["--repack_everything", "1" if config.repack_everything else "0"])

    # Packed suffix
    suffix = packed_suffix or config.packed_suffix
    if suffix:
        cmd.extend(["--packed_suffix", suffix])

    return cmd


def run_mpnn(
    pdb_path: Path,
    output_dir: Path,
    catres_list: Optional[List[CatalyticResidue]] = None,
    config: Optional[MPNNConfig] = None,
    additional_fixed: Optional[List[str]] = None,
    dry_run: bool = False,
    timeout: int = 3600
) -> MPNNResult:
    """
    Run LigandMPNN/EnhancedMPNN on a PDB file.

    Provides safe command construction with full logging.

    Args:
        pdb_path: Path to input PDB file
        output_dir: Directory for output files
        catres_list: Catalytic residues to fix (optional)
        config: MPNN configuration (uses defaults if None)
        additional_fixed: Additional residue IDs to fix
        dry_run: If True, print command but don't execute
        timeout: Timeout in seconds (default 1 hour)

    Returns:
        MPNNResult with success status and outputs
    """
    if config is None:
        config = MPNNConfig()

    pdb_path = Path(pdb_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build fixed residues list
    fixed_residues = []

    if catres_list:
        fixed_residues.extend([cr.pdb_resid for cr in catres_list])

    if additional_fixed:
        fixed_residues.extend(additional_fixed)

    # Remove duplicates while preserving order
    seen = set()
    fixed_residues = [x for x in fixed_residues if not (x in seen or seen.add(x))]

    # Create fixed residues JSON if needed
    fixed_json = None
    if fixed_residues:
        fixed_json = output_dir / "fixed_residues.json"
        build_fixed_residues_json(pdb_path, fixed_residues, fixed_json)
        logger.info(f"Fixed residues: {', '.join(fixed_residues)}")

    # Build command
    cmd = build_mpnn_command(pdb_path, output_dir, fixed_json, config)
    cmd_str = shlex.join(cmd)

    logger.info(f"MPNN command:\n  {cmd_str}")

    if dry_run:
        logger.info("Dry run - command not executed")
        return MPNNResult(success=True, output_dir=output_dir, command=cmd_str)

    # Execute with subprocess
    try:
        logger.info("Running MPNN...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        success = result.returncode == 0

        if success:
            logger.info("MPNN completed successfully")
        else:
            logger.error(f"MPNN failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"STDERR: {result.stderr[:500]}")

        return MPNNResult(
            success=success,
            output_dir=output_dir,
            command=cmd_str,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr
        )

    except subprocess.TimeoutExpired:
        logger.error(f"MPNN timed out after {timeout} seconds")
        return MPNNResult(
            success=False,
            output_dir=output_dir,
            command=cmd_str,
            stderr=f"Timeout after {timeout} seconds"
        )

    except Exception as e:
        logger.error(f"MPNN execution failed: {e}")
        return MPNNResult(
            success=False,
            output_dir=output_dir,
            command=cmd_str,
            stderr=str(e)
        )


def run_mpnn_with_library(
    pose,
    fixed_residues: List[str],
    config: Optional[MPNNConfig] = None,
    name: str = "design"
) -> Optional[Dict[str, Any]]:
    """
    Run MPNN using the FastMPNNDesign library (Python API).

    This matches the pattern from original script lines 361-363, 405-413.

    Args:
        pose: PyRosetta Pose object
        fixed_residues: List of residue IDs to fix
        config: MPNN configuration
        name: Name for the design

    Returns:
        Dictionary with MPNN outputs or None if failed
    """
    if config is None:
        config = MPNNConfig()

    try:
        # Import FastMPNNdesign library
        from constants import FASTMPNN_DESIGN_PATH
        if FASTMPNN_DESIGN_PATH not in sys.path:
            sys.path.append(FASTMPNN_DESIGN_PATH)

        import FastMPNNdesign.fusedmpnn as fusedmpnn
        import pyrosetta.distributed.io

        # Create MPNN runner
        mpnnrunner = fusedmpnn.MPNNRunner(
            config.model_type,
            verbose=True,
            pack_sc=config.pack_side_chains,
            ligand_mpnn_use_side_chain_context=config.ligand_mpnn_use_side_chain_context
        )

        # Create input
        mpnn_input = mpnnrunner.MPNN_Input()
        mpnn_input.fixed_residues = fixed_residues
        mpnn_input.omit_AA = list(config.omit_AA)
        mpnn_input.pdb = pyrosetta.distributed.io.to_pdbstring(pose)
        mpnn_input.name = name
        mpnn_input.number_of_batches = config.number_of_batches
        mpnn_input.batch_size = config.batch_size
        mpnn_input.temperature = config.temperature

        # Run
        logger.info(f"Running MPNN via library API (T={config.temperature})")
        mpnn_out = mpnnrunner.run(mpnn_input)

        logger.info(f"MPNN generated {len(mpnn_out.get('generated_sequences', []))} sequences")
        return mpnn_out

    except Exception as e:
        logger.error(f"MPNN library execution failed: {e}")
        return None


def get_example_command(config: Optional[MPNNConfig] = None) -> str:
    """
    Generate an example MPNN command for documentation.

    Args:
        config: MPNN configuration (uses defaults if None)

    Returns:
        Example command string
    """
    if config is None:
        config = MPNNConfig()
        config.use_apptainer = True

    cmd = build_mpnn_command(
        pdb_path=Path("<input.pdb>"),
        output_dir=Path("<output_dir>"),
        fixed_residues_json=Path("<fixed_residues.json>"),
        config=config
    )

    return shlex.join(cmd)
