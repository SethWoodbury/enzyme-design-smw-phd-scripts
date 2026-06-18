"""MPNN execution module for sequence design.

This module provides:
- MPNNInput dataclass for configuring MPNN runs
- MPNNResult dataclass for parsing MPNN outputs
- MPNNRunner class for executing MPNN via subprocess

MPNN outputs are saved to a folder structure:
  out_folder/
    seqs/           # FASTA files with designed sequences
    backbones/      # PDB files with backbone only (no sidechains)
    packed/         # PDB files with packed sidechains (if pack_side_chains=True)
    stats/          # JSON files with statistics (if save_stats=True)

Key features:
- Subprocess execution via apptainer container
- Support for fixed_residues and redesigned_residues
- Per-residue amino acid biases
- Side chain packing with denoising
"""
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add module_utils to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module_utils.constants import (
    DEFAULT_MPNN_RUNNER,
    DEFAULT_MODEL_TYPE,
    DEFAULT_ENHANCE_MODEL,
    DEFAULT_OMIT_AA,
    DEFAULT_APPTAINER_IMAGE,
    DEFAULT_LIGAND_MPNN_USE_SC_CONTEXT,
    DEFAULT_REPACK_EVERYTHING,
    DEFAULT_PACK_SIDE_CHAINS,
    DEFAULT_SC_DENOISING_STEPS,
    DEFAULT_MPNN_BATCH_SIZE,
    DEFAULT_MPNN_NUMBER_OF_BATCHES,
    DEFAULT_MPNN_TEMPERATURE,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class MPNNInput:
    """Input configuration for an MPNN design run.

    Attributes:
        pdb_path: Path to input PDB structure
        out_folder: Output directory for MPNN results
        fixed_residues: List of residue IDs to keep fixed (e.g., ["A12", "A13"])
        redesigned_residues: List of residue IDs to redesign (everything else fixed)
        temperature: Sampling temperature (lower = more conservative)
        batch_size: Number of sequences per batch (1 for max diversity)
        number_of_batches: Number of batches to generate
        omit_aa: Amino acids to never design (e.g., "CM")
        bias_aa: Global amino acid biases (e.g., {"A": -0.5, "C": -2.0})
        bias_aa_per_residue: Per-residue biases (e.g., {"A12": {"G": -1.0}})
        model_type: MPNN model type (default: "ligand_mpnn")
        enhance: Enhancement model (e.g., "plddt_3_20240930-f9c9ea0f")
        use_sc_context: Use side chain context for ligand MPNN
        pack_side_chains: Pack side chains after design
        repack_everything: Repack all residues vs only designed ones
        sc_denoising_steps: Side chain denoising steps
        seed: Random seed (0 for random)
    """
    pdb_path: str
    out_folder: str

    # Residue specification (use one or the other, not both)
    fixed_residues: List[str] = field(default_factory=list)
    redesigned_residues: List[str] = field(default_factory=list)

    # Sampling parameters
    temperature: float = DEFAULT_MPNN_TEMPERATURE
    batch_size: int = DEFAULT_MPNN_BATCH_SIZE
    number_of_batches: int = DEFAULT_MPNN_NUMBER_OF_BATCHES

    # Amino acid restrictions
    omit_aa: str = DEFAULT_OMIT_AA
    bias_aa: Dict[str, float] = field(default_factory=dict)
    bias_aa_per_residue: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Model configuration
    model_type: str = DEFAULT_MODEL_TYPE
    enhance: Optional[str] = None  # e.g., "plddt_3_20240930-f9c9ea0f"
    use_sc_context: int = DEFAULT_LIGAND_MPNN_USE_SC_CONTEXT

    # Side chain packing
    pack_side_chains: int = DEFAULT_PACK_SIDE_CHAINS
    repack_everything: int = DEFAULT_REPACK_EVERYTHING
    sc_denoising_steps: int = DEFAULT_SC_DENOISING_STEPS

    # Other
    seed: int = 0
    save_stats: int = 1
    verbose: int = 1

    def validate(self) -> None:
        """Validate input configuration."""
        if not os.path.exists(self.pdb_path):
            raise FileNotFoundError(f"PDB file not found: {self.pdb_path}")

        if self.fixed_residues and self.redesigned_residues:
            raise ValueError("Specify either fixed_residues OR redesigned_residues, not both")

        if self.temperature <= 0:
            raise ValueError(f"Temperature must be positive: {self.temperature}")

        if self.batch_size < 1:
            raise ValueError(f"Batch size must be >= 1: {self.batch_size}")


@dataclass
class MPNNResult:
    """Results from an MPNN design run.

    Attributes:
        sequences: List of designed sequences (1-letter codes)
        seq_files: List of paths to FASTA files
        backbone_files: List of paths to backbone PDB files
        packed_files: List of paths to packed PDB files (if pack_side_chains)
        stats_files: List of paths to stats JSON files (if save_stats)
        recovery_scores: List of sequence recovery scores
        mpnn_scores: List of MPNN log probability scores
    """
    sequences: List[str] = field(default_factory=list)
    seq_files: List[str] = field(default_factory=list)
    backbone_files: List[str] = field(default_factory=list)
    packed_files: List[str] = field(default_factory=list)
    stats_files: List[str] = field(default_factory=list)
    recovery_scores: List[float] = field(default_factory=list)
    mpnn_scores: List[float] = field(default_factory=list)

    @property
    def num_designs(self) -> int:
        """Number of designs generated."""
        return len(self.sequences)

    @property
    def output_pdbs(self) -> List[str]:
        """Primary PDB outputs (packed if available, else backbone)."""
        if self.packed_files:
            return self.packed_files
        return self.backbone_files


class MPNNRunner:
    """Execute MPNN sequence design via subprocess.

    This class handles:
    - Building MPNN command line arguments
    - Running MPNN in apptainer container
    - Parsing output files

    Usage:
        runner = MPNNRunner()
        result = runner.run(mpnn_input)
        for pdb, seq in zip(result.output_pdbs, result.sequences):
            print(f"{pdb}: {seq[:20]}...")
    """

    def __init__(
        self,
        mpnn_runner_script: str = DEFAULT_MPNN_RUNNER,
        container_image: str = DEFAULT_APPTAINER_IMAGE,
        use_container: bool = True,
        timeout: int = 600,
    ):
        """Initialize MPNN runner.

        Args:
            mpnn_runner_script: Path to MPNN run.py script
            container_image: Path to apptainer/singularity image
            use_container: Whether to run in container (recommended)
            timeout: Maximum execution time in seconds
        """
        self.mpnn_runner_script = mpnn_runner_script
        self.container_image = container_image
        self.use_container = use_container
        self.timeout = timeout

    def run(self, mpnn_input: MPNNInput) -> MPNNResult:
        """Execute MPNN design.

        Args:
            mpnn_input: Configuration for MPNN run

        Returns:
            MPNNResult with designed sequences and output files
        """
        mpnn_input.validate()

        # Create output directory
        out_folder = Path(mpnn_input.out_folder)
        out_folder.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = self._build_command(mpnn_input)
        LOGGER.info(f"Running MPNN: {' '.join(cmd[:5])}...")

        # Execute
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(out_folder.parent),
            )

            if result.returncode != 0:
                LOGGER.error(f"MPNN failed with return code {result.returncode}")
                LOGGER.error(f"STDERR: {result.stderr[-2000:]}")
                raise RuntimeError(f"MPNN execution failed: {result.stderr[-500:]}")

            if mpnn_input.verbose:
                LOGGER.debug(f"MPNN STDOUT: {result.stdout[-1000:]}")

        except subprocess.TimeoutExpired:
            raise TimeoutError(f"MPNN execution timed out after {self.timeout}s")

        # Parse results
        return self._parse_results(mpnn_input)

    def _build_command(self, mpnn_input: MPNNInput) -> List[str]:
        """Build MPNN command line arguments."""
        args = [
            "--pdb_path", mpnn_input.pdb_path,
            "--out_folder", mpnn_input.out_folder,
            "--model_type", mpnn_input.model_type,
            "--temperature", str(mpnn_input.temperature),
            "--batch_size", str(mpnn_input.batch_size),
            "--number_of_batches", str(mpnn_input.number_of_batches),
            "--ligand_mpnn_use_side_chain_context", str(mpnn_input.use_sc_context),
            "--pack_side_chains", str(mpnn_input.pack_side_chains),
            "--repack_everything", str(mpnn_input.repack_everything),
            "--sc_num_denoising_steps", str(mpnn_input.sc_denoising_steps),
            "--save_stats", str(mpnn_input.save_stats),
            "--verbose", str(mpnn_input.verbose),
        ]

        if mpnn_input.seed:
            args.extend(["--seed", str(mpnn_input.seed)])

        if mpnn_input.omit_aa:
            args.extend(["--omit_AA", mpnn_input.omit_aa])

        if mpnn_input.enhance:
            args.extend(["--enhance", mpnn_input.enhance])

        # Residue specification
        if mpnn_input.fixed_residues:
            args.extend(["--fixed_residues", " ".join(mpnn_input.fixed_residues)])
        elif mpnn_input.redesigned_residues:
            args.extend(["--redesigned_residues", " ".join(mpnn_input.redesigned_residues)])

        # Amino acid biases
        if mpnn_input.bias_aa:
            bias_str = ",".join(f"{aa}:{val}" for aa, val in mpnn_input.bias_aa.items())
            args.extend(["--bias_AA", bias_str])

        if mpnn_input.bias_aa_per_residue:
            # Write to temp file
            bias_file = os.path.join(mpnn_input.out_folder, "_bias_per_residue.json")
            with open(bias_file, "w") as f:
                json.dump(mpnn_input.bias_aa_per_residue, f)
            args.extend(["--bias_AA_per_residue", bias_file])

        # Build full command with container if needed
        if self.use_container:
            cmd = [
                "apptainer", "exec",
                "--nv",  # Enable GPU
                self.container_image,
                "python", self.mpnn_runner_script,
            ] + args
        else:
            cmd = ["python", self.mpnn_runner_script] + args

        return cmd

    def _parse_results(self, mpnn_input: MPNNInput) -> MPNNResult:
        """Parse MPNN output files into MPNNResult."""
        out_folder = Path(mpnn_input.out_folder)
        result = MPNNResult()

        # Parse sequence files (FASTA format)
        seqs_dir = out_folder / "seqs"
        if seqs_dir.exists():
            for fasta_file in sorted(seqs_dir.glob("*.fa")):
                result.seq_files.append(str(fasta_file))
                sequences = self._parse_fasta(fasta_file)
                result.sequences.extend(sequences)

        # Parse backbone PDBs
        backbones_dir = out_folder / "backbones"
        if backbones_dir.exists():
            for pdb_file in sorted(backbones_dir.glob("*.pdb")):
                result.backbone_files.append(str(pdb_file))

        # Parse packed PDBs
        packed_dir = out_folder / "packed"
        if packed_dir.exists():
            for pdb_file in sorted(packed_dir.glob("*.pdb")):
                result.packed_files.append(str(pdb_file))

        # Parse stats files
        stats_dir = out_folder / "stats"
        if stats_dir.exists():
            for stats_file in sorted(stats_dir.glob("*.json")):
                result.stats_files.append(str(stats_file))
                stats = self._parse_stats(stats_file)
                if "recovery" in stats:
                    result.recovery_scores.append(stats["recovery"])
                if "score" in stats:
                    result.mpnn_scores.append(stats["score"])

        LOGGER.info(f"Parsed {result.num_designs} MPNN designs from {out_folder}")
        return result

    def _parse_fasta(self, fasta_path: Path) -> List[str]:
        """Parse sequences from FASTA file."""
        sequences = []
        current_seq = []

        with open(fasta_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_seq:
                        sequences.append("".join(current_seq))
                        current_seq = []
                else:
                    current_seq.append(line)

            if current_seq:
                sequences.append("".join(current_seq))

        return sequences

    def _parse_stats(self, stats_path: Path) -> Dict:
        """Parse MPNN stats JSON file."""
        try:
            with open(stats_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}


def create_mpnn_input_from_classifier(
    pdb_path: str,
    out_folder: str,
    residue_classifier,  # ResidueClassifier from residue_classifier.py
    design_spheres: List[str] = None,
    temperature: float = DEFAULT_MPNN_TEMPERATURE,
    num_designs: int = DEFAULT_MPNN_NUMBER_OF_BATCHES,
    batch_size: int = DEFAULT_MPNN_BATCH_SIZE,
    omit_aa: str = DEFAULT_OMIT_AA,
    additional_fixed: List[str] = None,
    bias_aa: Dict[str, float] = None,
) -> MPNNInput:
    """Create MPNNInput from a ResidueClassifier.

    This helper function extracts fixed and redesigned residues from the
    classifier's sphere assignments.

    Args:
        pdb_path: Path to input PDB
        out_folder: Output directory
        residue_classifier: ResidueClassifier with classified residues
        design_spheres: Which spheres to design (default: ["primary"])
        temperature: Sampling temperature
        num_designs: Number of designs to generate
        batch_size: Batch size (1 for max diversity)
        omit_aa: Amino acids to omit
        additional_fixed: Extra residues to fix (e.g., conserved mutations)
        bias_aa: Global amino acid biases

    Returns:
        MPNNInput configured for the classifier's design region
    """
    try:
        from .residue_classifier import DesignSphere
    except ImportError:
        from residue_classifier import DesignSphere

    if design_spheres is None:
        design_spheres = ["primary"]

    # Map string names to enum values
    sphere_map = {
        "primary": DesignSphere.PRIMARY,
        "secondary": DesignSphere.SECONDARY,
    }
    target_spheres = {sphere_map.get(s, s) for s in design_spheres}

    # Get residues to design vs fix
    fixed_residues = []
    redesigned_residues = []

    LOGGER.debug(f"Target spheres for design: {target_spheres}")
    LOGGER.debug(f"Total residues to classify: {len(residue_classifier.residues)}")

    for res_info in residue_classifier.residues.values():
        res_id = f"{res_info.chain}{res_info.resno}"

        if res_info.is_fixed:
            # Catalytic or conserved motif - always fixed
            fixed_residues.append(res_id)
        elif res_info.is_protected:
            # GLY/PRO protected
            fixed_residues.append(res_id)
        elif res_info.sphere in target_spheres:
            # In design sphere - redesign
            redesigned_residues.append(res_id)
            LOGGER.debug(f"  Design: {res_id} ({res_info.resname}, sphere={res_info.sphere})")
        else:
            # Outside design sphere - fixed
            fixed_residues.append(res_id)

    # Fallback: if primary-only yielded 0 designable residues, expand to secondary
    if len(redesigned_residues) == 0 and DesignSphere.SECONDARY not in target_spheres:
        LOGGER.warning("No designable residues in primary sphere, expanding to include secondary sphere")
        target_spheres.add(DesignSphere.SECONDARY)
        # Re-scan residues that were marked as fixed (non-catalytic, non-protected)
        new_redesigned = []
        new_fixed = []
        for res_info in residue_classifier.residues.values():
            res_id = f"{res_info.chain}{res_info.resno}"
            if res_info.is_fixed:
                new_fixed.append(res_id)
            elif res_info.is_protected:
                new_fixed.append(res_id)
            elif res_info.sphere in target_spheres:
                new_redesigned.append(res_id)
                LOGGER.debug(f"  Design (expanded): {res_id} ({res_info.resname}, sphere={res_info.sphere})")
            else:
                new_fixed.append(res_id)
        redesigned_residues = new_redesigned
        fixed_residues = new_fixed

    # Add any additional fixed residues
    if additional_fixed:
        for res_id in additional_fixed:
            if res_id not in fixed_residues:
                fixed_residues.append(res_id)
            if res_id in redesigned_residues:
                redesigned_residues.remove(res_id)

    LOGGER.info(f"Design setup: {len(redesigned_residues)} redesign, {len(fixed_residues)} fixed")

    return MPNNInput(
        pdb_path=pdb_path,
        out_folder=out_folder,
        fixed_residues=fixed_residues,
        temperature=temperature,
        batch_size=batch_size,
        number_of_batches=num_designs,
        omit_aa=omit_aa,
        bias_aa=bias_aa or {},
        pack_side_chains=DEFAULT_PACK_SIDE_CHAINS,
        use_sc_context=DEFAULT_LIGAND_MPNN_USE_SC_CONTEXT,
    )


def run_mpnn_batch(
    pdb_paths: List[str],
    out_folder: str,
    fixed_residues_map: Dict[str, List[str]],
    temperature: float = DEFAULT_MPNN_TEMPERATURE,
    num_designs: int = DEFAULT_MPNN_NUMBER_OF_BATCHES,
    batch_size: int = DEFAULT_MPNN_BATCH_SIZE,
    omit_aa: str = DEFAULT_OMIT_AA,
    runner: Optional[MPNNRunner] = None,
) -> Dict[str, MPNNResult]:
    """Run MPNN on multiple PDBs with per-PDB fixed residues.

    This uses MPNN's multi-PDB mode for efficiency.

    Args:
        pdb_paths: List of PDB file paths
        out_folder: Base output directory
        fixed_residues_map: Dict mapping PDB path to list of fixed residue IDs
        temperature: Sampling temperature
        num_designs: Number of designs per PDB
        batch_size: Batch size
        omit_aa: Amino acids to omit
        runner: MPNNRunner instance (creates new if None)

    Returns:
        Dict mapping PDB path to MPNNResult
    """
    if runner is None:
        runner = MPNNRunner()

    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    # Write multi-PDB JSON files
    pdb_json_path = out_folder / "_pdb_paths.json"
    fixed_json_path = out_folder / "_fixed_residues.json"

    pdb_dict = {p: "" for p in pdb_paths}
    with open(pdb_json_path, "w") as f:
        json.dump(pdb_dict, f)

    with open(fixed_json_path, "w") as f:
        json.dump(fixed_residues_map, f)

    # Build command for multi-PDB mode
    # Note: This requires modifying the runner to support multi-PDB
    # For now, we'll run sequentially
    results = {}
    for pdb_path in pdb_paths:
        pdb_name = Path(pdb_path).stem
        pdb_out = str(out_folder / pdb_name)

        mpnn_input = MPNNInput(
            pdb_path=pdb_path,
            out_folder=pdb_out,
            fixed_residues=fixed_residues_map.get(pdb_path, []),
            temperature=temperature,
            batch_size=batch_size,
            number_of_batches=num_designs,
            omit_aa=omit_aa,
        )

        try:
            result = runner.run(mpnn_input)
            results[pdb_path] = result
        except Exception as e:
            LOGGER.error(f"MPNN failed for {pdb_path}: {e}")
            results[pdb_path] = MPNNResult()  # Empty result

    return results
