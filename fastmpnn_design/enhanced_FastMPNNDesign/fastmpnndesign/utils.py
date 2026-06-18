"""
Utility functions for enhanced_fastmpnndesign.

General-purpose utilities for path handling, file I/O, and common operations.
"""

import sys
import os
from pathlib import Path
from typing import List, Optional, Any, Dict
import pandas as pd

# Add package directory to path for standalone execution
_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from logging_config import get_logger

logger = get_logger("utils")


def validate_pdb_path(path: Path) -> Path:
    """
    Validate that a PDB file exists and is readable.

    Args:
        path: Path to PDB file

    Returns:
        Resolved absolute path

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a .pdb file
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDB file not found: {path}")
    if not path.suffix.lower() == '.pdb':
        logger.warning(f"File does not have .pdb extension: {path}")
    return path


def validate_params_files(params: List[Path]) -> List[Path]:
    """
    Validate that all params files exist.

    Args:
        params: List of paths to params files

    Returns:
        List of resolved absolute paths

    Raises:
        FileNotFoundError: If any file doesn't exist
    """
    validated = []
    for p in params:
        p = Path(p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Params file not found: {p}")
        validated.append(p)
    return validated


def get_nproc() -> int:
    """
    Get number of available processors.

    Checks OMP_NUM_THREADS and SLURM_CPUS_ON_NODE environment variables,
    falls back to os.cpu_count().

    Returns:
        Number of processors to use
    """
    nproc = os.cpu_count() or 1

    if "OMP_NUM_THREADS" in os.environ:
        try:
            nproc = int(os.environ["OMP_NUM_THREADS"])
        except ValueError:
            pass

    if "SLURM_CPUS_ON_NODE" in os.environ:
        try:
            nproc = int(os.environ["SLURM_CPUS_ON_NODE"])
        except ValueError:
            pass

    return nproc


def dump_scorefile(
    scores_df: pd.DataFrame,
    filename: str,
    append: bool = True
) -> None:
    """
    Dump scores DataFrame to a scorefile.

    Matches the format from scoring_utils.dump_scorefile in the original script.

    Args:
        scores_df: DataFrame with score columns
        filename: Path to scorefile
        append: If True, append to existing file; otherwise overwrite
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists and has header
    write_header = not filename.exists() or not append

    mode = 'a' if append else 'w'

    scores_df.to_csv(
        filename,
        mode=mode,
        sep='\t',
        index=False,
        header=write_header
    )
    logger.debug(f"Wrote scores to {filename}")


def get_pdb_basename(pdb_path: Path) -> str:
    """
    Get the base name of a PDB file without extension.

    Args:
        pdb_path: Path to PDB file

    Returns:
        Base name without .pdb extension
    """
    return Path(pdb_path).stem


def ensure_output_dirs(output_dir: Path, do_2nd_layer: bool = False) -> Dict[str, Path]:
    """
    Ensure output directories exist.

    Args:
        output_dir: Base output directory
        do_2nd_layer: If True, also create seqs/ directory

    Returns:
        Dictionary with paths to 'scores' and optionally 'seqs' directories
    """
    output_dir = Path(output_dir)
    dirs = {}

    # Scores directory
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    dirs['scores'] = scores_dir

    # Seqs directory for 2nd layer MPNN
    if do_2nd_layer:
        seqs_dir = output_dir / "seqs"
        seqs_dir.mkdir(parents=True, exist_ok=True)
        dirs['seqs'] = seqs_dir

    logger.debug(f"Created output directories: {list(dirs.values())}")
    return dirs


def format_residue_list(residues: List[int]) -> str:
    """
    Format a list of residue numbers for display.

    Args:
        residues: List of residue sequence positions

    Returns:
        String with residues joined by '+'
    """
    return "+".join([str(x) for x in sorted(residues)])


def get_pdb_chain_resnum(pose: Any, seqpos: int) -> str:
    """
    Get chain and residue number from a PyRosetta pose.

    Args:
        pose: PyRosetta Pose object
        seqpos: Sequence position

    Returns:
        String in format 'A150' (chain + resnum)
    """
    pdb_info = pose.pdb_info()
    chain = pdb_info.chain(seqpos)
    resnum = pdb_info.number(seqpos)
    icode = pdb_info.icode(seqpos)
    if icode and icode.strip():
        return f"{chain}{resnum}{icode}"
    return f"{chain}{resnum}"


def seqpos_to_pdb_resid(pose: Any, seqpos_list: List[int]) -> List[str]:
    """
    Convert list of sequence positions to PDB residue IDs.

    Args:
        pose: PyRosetta Pose object
        seqpos_list: List of sequence positions

    Returns:
        List of strings in format 'A150'
    """
    return [get_pdb_chain_resnum(pose, s) for s in seqpos_list]


def pdb_resid_to_seqpos(pose: Any, chain: str, resnum: int, icode: str = "") -> Optional[int]:
    """
    Convert PDB chain/resnum to sequence position.

    Args:
        pose: PyRosetta Pose object
        chain: Chain identifier
        resnum: Residue number
        icode: Insertion code (optional)

    Returns:
        Sequence position, or None if not found
    """
    pdb_info = pose.pdb_info()
    for i in range(1, pose.size() + 1):
        if (pdb_info.chain(i) == chain and
            pdb_info.number(i) == resnum):
            res_icode = pdb_info.icode(i)
            if icode == res_icode or (not icode and not res_icode.strip()):
                return i
    return None


def read_pdb_header(pdb_path: Path) -> List[str]:
    """
    Read header lines (before ATOM) from a PDB file.

    Args:
        pdb_path: Path to PDB file

    Returns:
        List of header lines
    """
    header_lines = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                break
            header_lines.append(line.rstrip())
    return header_lines


def write_pdb_with_header(
    pose: Any,
    output_path: Path,
    header_lines: Optional[List[str]] = None
) -> None:
    """
    Write PDB with custom header lines.

    Args:
        pose: PyRosetta Pose object
        output_path: Output path
        header_lines: Optional list of header lines to prepend
    """
    import pyrosetta.distributed.io

    pdb_str = pyrosetta.distributed.io.to_pdbstring(pose)

    if header_lines:
        # Split PDB string and insert header
        pdb_lines = pdb_str.split('\n')
        # Find first ATOM/HETATM line
        insert_idx = 0
        for i, line in enumerate(pdb_lines):
            if line.startswith(('ATOM', 'HETATM')):
                insert_idx = i
                break

        # Insert header before ATOM lines
        new_lines = pdb_lines[:insert_idx] + header_lines + pdb_lines[insert_idx:]
        pdb_str = '\n'.join(new_lines)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(pdb_str)
