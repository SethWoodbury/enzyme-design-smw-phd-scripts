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

Batch size handling:
- num_designs specifies the target number of unique sequences
- batch_size controls sequences generated per batch (higher = faster, potentially less diverse)
- number_of_batches is calculated as ceil(num_designs / batch_size)
- If more than num_designs unique sequences are generated, clustering is used to
  select the most diverse subset
"""
import json
import logging
import math
import os
import socket
import struct
import subprocess
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..module_utils.constants import (
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
    DEFAULT_MPNN_SERVER_HOST,
    DEFAULT_MPNN_SERVER_PORT,
)

LOGGER = logging.getLogger(__name__)


def _auto_detect_gpu() -> bool:
    """Best-effort GPU detection for deciding whether to use --nv.

    Returns True if a GPU is likely available, False otherwise.
    """
    env_val = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_val is not None:
        if env_val.strip() in ("", "none", "None", "-1"):
            return False

    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass

    # Fallback check for NVIDIA device files
    if os.path.exists("/dev/nvidia0") or os.path.exists("/proc/driver/nvidia/gpus"):
        return True

    return False


class MPNNServerClient:
    """Client for communicating with persistent MPNN server.

    The MPNN server keeps model weights loaded in GPU memory, eliminating
    the ~15-20s model loading overhead per call.
    """

    def __init__(
        self,
        host: str = DEFAULT_MPNN_SERVER_HOST,
        port: int = DEFAULT_MPNN_SERVER_PORT,
        timeout: float = 600.0,
        auto_start: bool = True,
        container_image: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        model_type: str = "ligand_mpnn",
    ):
        """Initialize MPNN server client.

        Args:
            host: Server hostname
            port: Server port
            timeout: Request timeout in seconds
            auto_start: Whether to auto-start server if not running
            container_image: Apptainer image for auto-starting server
            model_type: Model type for auto-starting server
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.auto_start = auto_start
        self.container_image = container_image
        self.use_gpu = use_gpu
        self.model_type = model_type
        self._server_process = None

    def is_server_running(self) -> bool:
        """Check if MPNN server is running and responsive."""
        try:
            response = self._send_request({"type": "health"}, timeout=5.0)
            return response.get("status") == "success"
        except Exception:
            return False

    def start_server(self) -> bool:
        """Start the MPNN server in background.

        Returns:
            True if server started successfully, False otherwise.
        """
        if self.is_server_running():
            LOGGER.info("MPNN server already running")
            return True

        LOGGER.info(f"Starting MPNN server on {self.host}:{self.port}...")

        server_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "mpnn_server.py"
        )

        cmd = []
        if self.container_image:
            cmd = [
                "apptainer", "exec",
                self.container_image,
                "python", server_script,
            ]
            if self.use_gpu:
                cmd.insert(2, "--nv")
        else:
            cmd = ["python", server_script]

        cmd.extend([
            "--host", self.host,
            "--port", str(self.port),
            "--model_type", self.model_type,
            "--pack_side_chains",
        ])

        try:
            # Start server in background
            self._server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )

            # Wait for server to become responsive (up to 120 seconds for model loading)
            startup_timeout = 120
            for i in range(startup_timeout):
                time.sleep(1)
                if self.is_server_running():
                    LOGGER.info(f"MPNN server started successfully after {i+1}s")
                    return True
                # Check if process died
                if self._server_process.poll() is not None:
                    stderr = self._server_process.stderr.read().decode() if self._server_process.stderr else ""
                    LOGGER.error(f"MPNN server process died: {stderr[:500]}")
                    return False

            LOGGER.error(f"MPNN server failed to start within {startup_timeout}s")
            return False

        except Exception as e:
            LOGGER.error(f"Failed to start MPNN server: {e}")
            return False

    def run(self, mpnn_input: "MPNNInput") -> "MPNNResult":
        """Send design request to server.

        Args:
            mpnn_input: MPNN input configuration

        Returns:
            MPNNResult with designed sequences and files

        Raises:
            RuntimeError: If server communication fails
        """
        # Build request from MPNNInput
        request = {
            "type": "design",
            "pdb_path": mpnn_input.pdb_path,
            "out_folder": mpnn_input.out_folder,
            "temperature": mpnn_input.temperature,
            "batch_size": mpnn_input.batch_size,
            "number_of_batches": mpnn_input.number_of_batches,
            "fixed_residues": mpnn_input.fixed_residues,
            "redesigned_residues": mpnn_input.redesigned_residues,
            "omit_aa": mpnn_input.omit_aa,
            "bias_aa": mpnn_input.bias_aa,
            "bias_aa_per_residue": mpnn_input.bias_aa_per_residue,
            "model_type": mpnn_input.model_type,
            "enhance": mpnn_input.enhance,
            "use_sc_context": mpnn_input.use_sc_context,
            "pack_side_chains": mpnn_input.pack_side_chains,
            "repack_everything": mpnn_input.repack_everything,
            "sc_denoising_steps": mpnn_input.sc_denoising_steps,
            "packed_suffix": mpnn_input.packed_suffix,
            "seed": mpnn_input.seed,
            "save_stats": mpnn_input.save_stats,
            "verbose": mpnn_input.verbose,
        }

        # Create output directory
        Path(mpnn_input.out_folder).mkdir(parents=True, exist_ok=True)

        response = self._send_request(request)

        if response.get("status") == "error":
            raise RuntimeError(
                f"MPNN server error: {response.get('error_message')}\n"
                f"{response.get('traceback', '')}"
            )

        # Build MPNNResult from response
        result = MPNNResult(
            sequences=response.get("sequences", []),
            seq_files=response.get("seq_files", []),
            backbone_files=response.get("backbone_files", []),
            packed_files=response.get("packed_files", []),
            stats_files=response.get("stats_files", []),
            recovery_scores=response.get("recovery_scores", []),
            mpnn_scores=response.get("mpnn_scores", []),
        )

        elapsed = response.get("elapsed_time", 0)
        LOGGER.info(f"Server returned {result.num_designs} designs in {elapsed:.2f}s")

        return result

    def _send_request(
        self,
        request: dict,
        timeout: Optional[float] = None,
    ) -> dict:
        """Send request to server and get response."""
        timeout = timeout or self.timeout

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect((self.host, self.port))

            # Send length-prefixed message
            msg_bytes = json.dumps(request).encode("utf-8")
            sock.sendall(struct.pack(">I", len(msg_bytes)))
            sock.sendall(msg_bytes)

            # Receive response
            length_bytes = self._recv_exact(sock, 4)
            msg_length = struct.unpack(">I", length_bytes)[0]
            response_bytes = self._recv_exact(sock, msg_length)

            return json.loads(response_bytes.decode("utf-8"))

    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        """Receive exactly n bytes."""
        data = b""
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed by server")
            data += chunk
        return data

    def shutdown(self) -> None:
        """Request server shutdown."""
        try:
            self._send_request({"type": "shutdown"}, timeout=5.0)
        except Exception:
            pass


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
        packed_suffix: Suffix appended to packed PDB filenames (prevents collisions)
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
    packed_suffix: Optional[str] = None

    # Other
    seed: int = 0
    save_stats: int = 1
    verbose: int = 1

    # Diversity filtering (for handling overshoot when batch_size > 1)
    target_num_designs: Optional[int] = None  # Original requested count
    reference_sequence: Optional[str] = None  # For diversity calculation

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
    """Execute MPNN sequence design via subprocess or persistent server.

    This class handles:
    - Building MPNN command line arguments
    - Running MPNN via persistent server (default, ~5-10x faster)
    - Fallback to apptainer subprocess if server fails
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
        use_gpu: Optional[bool] = None,
        use_mpnn_server: bool = True,
        mpnn_server_host: str = DEFAULT_MPNN_SERVER_HOST,
        mpnn_server_port: int = DEFAULT_MPNN_SERVER_PORT,
        auto_start_server: bool = True,
    ):
        """Initialize MPNN runner.

        Args:
            mpnn_runner_script: Path to MPNN run.py script
            container_image: Path to apptainer/singularity image
            use_container: Whether to run in container (recommended)
            timeout: Maximum execution time in seconds
            use_gpu: Force GPU usage (None = auto-detect)
            use_mpnn_server: Use persistent MPNN server (default: True)
            mpnn_server_host: MPNN server hostname
            mpnn_server_port: MPNN server port
            auto_start_server: Auto-start server if not running
        """
        self.mpnn_runner_script = mpnn_runner_script
        self.container_image = container_image
        self.use_container = use_container
        self.timeout = timeout
        self.use_gpu = _auto_detect_gpu() if use_gpu is None else bool(use_gpu)

        # Server configuration
        self.use_mpnn_server = use_mpnn_server
        self._server_client: Optional[MPNNServerClient] = None

        if use_mpnn_server:
            self._server_client = MPNNServerClient(
                host=mpnn_server_host,
                port=mpnn_server_port,
                timeout=timeout,
                auto_start=auto_start_server,
                container_image=container_image if use_container else None,
                use_gpu=self.use_gpu if use_container else False,
                model_type="ligand_mpnn",
            )

    def run(self, mpnn_input: MPNNInput) -> MPNNResult:
        """Execute MPNN design.

        Tries to use persistent MPNN server first (much faster), then falls
        back to subprocess if server is unavailable.

        Args:
            mpnn_input: Configuration for MPNN run

        Returns:
            MPNNResult with designed sequences and output files
        """
        mpnn_input.validate()

        # Try server first if enabled
        if self.use_mpnn_server and self._server_client:
            try:
                result = self._run_via_server(mpnn_input)
                return self._apply_diversity_filter(mpnn_input, result)
            except Exception as e:
                LOGGER.warning(f"MPNN server failed, falling back to subprocess: {e}")

        # Fallback to subprocess execution
        return self._run_via_subprocess(mpnn_input)

    def _run_via_server(self, mpnn_input: MPNNInput) -> MPNNResult:
        """Run MPNN via persistent server."""
        # Auto-start server if needed
        if self._server_client.auto_start and not self._server_client.is_server_running():
            if not self._server_client.start_server():
                raise RuntimeError("Failed to start MPNN server")

        LOGGER.info("Running MPNN via server...")
        return self._server_client.run(mpnn_input)

    def _run_via_subprocess(self, mpnn_input: MPNNInput) -> MPNNResult:
        """Run MPNN via subprocess (original implementation)."""
        # Create output directory
        out_folder = Path(mpnn_input.out_folder)
        out_folder.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = self._build_command(mpnn_input)
        LOGGER.info(f"Running MPNN via subprocess: {' '.join(cmd[:5])}...")

        # Execute
        try:
            stdout_path = out_folder / "mpnn.stdout"
            stderr_path = out_folder / "mpnn.stderr"

            def _read_tail(path: Path, max_chars: int = 2000) -> str:
                try:
                    with open(path, "r") as f:
                        f.seek(0, os.SEEK_END)
                        size = f.tell()
                        f.seek(max(size - max_chars, 0))
                        return f.read()
                except Exception:
                    return ""

            with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
                result = subprocess.run(
                    cmd,
                    stdout=out_f,
                    stderr=err_f,
                    text=True,
                    timeout=self.timeout,
                    cwd=str(out_folder.parent),
                )
            LOGGER.info(f"MPNN logs: {stdout_path.name}, {stderr_path.name}")

            if result.returncode != 0:
                LOGGER.error(f"MPNN failed with return code {result.returncode}")
                err_tail = _read_tail(stderr_path, 2000)
                LOGGER.error(f"STDERR: {err_tail}")
                raise RuntimeError(f"MPNN execution failed: {err_tail[-500:]}")

            if mpnn_input.verbose:
                out_tail = _read_tail(stdout_path, 1000)
                if out_tail:
                    LOGGER.debug(f"MPNN STDOUT: {out_tail}")

        except subprocess.TimeoutExpired:
            raise TimeoutError(f"MPNN execution timed out after {self.timeout}s")

        # Parse results
        result = self._parse_results(mpnn_input)

        # Apply diversity filtering
        return self._apply_diversity_filter(mpnn_input, result)

    def _apply_diversity_filter(
        self,
        mpnn_input: MPNNInput,
        result: MPNNResult,
    ) -> MPNNResult:
        """Apply diversity filtering if we overshot target count."""
        if (mpnn_input.target_num_designs is not None and
                mpnn_input.reference_sequence is not None and
                result.num_designs > mpnn_input.target_num_designs):
            result = filter_to_target_count(
                result,
                mpnn_input.target_num_designs,
                mpnn_input.reference_sequence,
            )
        return result

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
        if mpnn_input.packed_suffix:
            args.extend(["--packed_suffix", mpnn_input.packed_suffix])

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
            ]
            if self.use_gpu:
                cmd.append("--nv")
            cmd.extend([
                self.container_image,
                "python", self.mpnn_runner_script,
            ])
            cmd += args
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
        """Parse sequences from FASTA file.

        Note: LigandMPNN outputs the reference sequence as sample_0 (or similar),
        followed by designed sequences. We skip any entry marked as reference/wildtype.
        """
        sequences = []
        entries = []  # List of (header, sequence) tuples

        current_header = ""
        current_seq = []

        with open(fasta_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    # Save previous entry if any
                    if current_header and current_seq:
                        entries.append((current_header, "".join(current_seq)))
                    current_header = line
                    current_seq = []
                else:
                    current_seq.append(line)

            # Save final entry
            if current_header and current_seq:
                entries.append((current_header, "".join(current_seq)))

        # Filter out reference/wildtype sequences (sample_0, sample 0, seq_0, etc.)
        for header, seq in entries:
            is_reference = (
                "sample_0" in header.lower() or
                "sample 0" in header.lower() or
                ", sample=0" in header.lower() or
                "_0," in header or
                header.rstrip().endswith("_0")
            )
            if not is_reference:
                sequences.append(seq)

        return sequences

    def _parse_stats(self, stats_path: Path) -> Dict:
        """Parse MPNN stats JSON file."""
        try:
            with open(stats_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}


def sequence_distance(seq1: str, seq2: str) -> int:
    """Compute Hamming distance between two sequences (count mismatches).

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Number of positions where sequences differ.
        If lengths differ, returns max(len) as penalty.
    """
    if len(seq1) != len(seq2):
        return max(len(seq1), len(seq2))  # Penalize length mismatch
    return sum(a != b for a, b in zip(seq1, seq2))


def cluster_sequences_hierarchical(
    sequences: List[str],
    n_clusters: int,
) -> List[List[int]]:
    """Cluster sequences into n_clusters groups using hierarchical clustering.

    Uses scipy.cluster.hierarchy with Hamming distance matrix.
    Falls back to greedy assignment if scipy unavailable.

    Args:
        sequences: List of sequences to cluster
        n_clusters: Target number of clusters

    Returns:
        List of clusters, each cluster is a list of sequence indices.
    """
    if len(sequences) <= n_clusters:
        return [[i] for i in range(len(sequences))]

    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        import numpy as np

        # Build distance matrix
        n = len(sequences)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = sequence_distance(sequences[i], sequences[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Hierarchical clustering
        condensed = squareform(dist_matrix)
        Z = linkage(condensed, method='average')
        labels = fcluster(Z, n_clusters, criterion='maxclust')

        # Group by cluster label
        clusters = [[] for _ in range(n_clusters)]
        for idx, label in enumerate(labels):
            clusters[label - 1].append(idx)

        # Remove empty clusters (scipy can return fewer than n_clusters)
        clusters = [c for c in clusters if c]
        if len(clusters) < n_clusters:
            LOGGER.warning(
                f"scipy clustering returned {len(clusters)} clusters < {n_clusters}; "
                "falling back to deterministic greedy clustering"
            )
            return _greedy_cluster(sequences, n_clusters)
        return clusters

    except ImportError:
        LOGGER.warning("scipy not available, falling back to greedy clustering")
        return _greedy_cluster(sequences, n_clusters)


def _greedy_cluster(sequences: List[str], n_clusters: int) -> List[List[int]]:
    """Greedy clustering fallback when scipy unavailable.

    Args:
        sequences: List of sequences to cluster
        n_clusters: Target number of clusters

    Returns:
        List of clusters, each cluster is a list of sequence indices.
    """
    if len(sequences) <= n_clusters:
        return [[i] for i in range(len(sequences))]

    # Initialize centroids as first n_clusters sequences
    clusters = [[i] for i in range(n_clusters)]

    # Assign remaining sequences to nearest centroid
    for i in range(n_clusters, len(sequences)):
        min_dist = float('inf')
        best_cluster = 0
        for c_idx, cluster in enumerate(clusters):
            centroid_idx = cluster[0]  # Use first member as centroid
            dist = sequence_distance(sequences[i], sequences[centroid_idx])
            if dist < min_dist:
                min_dist = dist
                best_cluster = c_idx
        clusters[best_cluster].append(i)

    return clusters


def filter_to_target_count(
    result: MPNNResult,
    target_count: int,
    reference_sequence: str,
) -> MPNNResult:
    """Filter MPNN results to target_count by clustering and keeping most diverse.

    Algorithm:
        1. If num_sequences <= target_count: return as-is
        2. Cluster sequences into target_count clusters
        3. For each cluster with >1 member: keep sequence most different from reference

    Args:
        result: MPNNResult with sequences and files
        target_count: Desired number of outputs
        reference_sequence: Original input sequence for diversity calculation

    Returns:
        Filtered MPNNResult with at most target_count designs
    """
    if result.num_designs <= target_count:
        return result

    LOGGER.info(f"Filtering {result.num_designs} designs to {target_count} using clustering")

    # Cluster sequences
    clusters = cluster_sequences_hierarchical(result.sequences, target_count)

    # For each cluster, keep most diverse (max distance from reference)
    keep_indices = []
    for cluster in clusters:
        if len(cluster) == 1:
            keep_indices.append(cluster[0])
        else:
            # Find member most different from reference
            best_idx = max(cluster, key=lambda i: sequence_distance(result.sequences[i], reference_sequence))
            keep_indices.append(best_idx)

    keep_indices.sort()  # Preserve original order

    # Build filtered result
    # Note: With batch_size > 1, there may be fewer PDB files than sequences
    # Only include files where index is valid
    def safe_index(arr, indices):
        if not arr:
            return []
        return [arr[i] for i in indices if i < len(arr)]

    filtered = MPNNResult(
        sequences=[result.sequences[i] for i in keep_indices],
        seq_files=safe_index(result.seq_files, keep_indices),
        backbone_files=safe_index(result.backbone_files, keep_indices),
        packed_files=safe_index(result.packed_files, keep_indices),
        stats_files=safe_index(result.stats_files, keep_indices),
        recovery_scores=safe_index(result.recovery_scores, keep_indices),
        mpnn_scores=safe_index(result.mpnn_scores, keep_indices),
    )

    LOGGER.info(f"Kept {len(keep_indices)} diverse designs from {len(clusters)} clusters")
    return filtered


def create_mpnn_input_from_classifier(
    pdb_path: str,
    out_folder: str,
    residue_classifier,  # ResidueClassifier from residue_classifier.py
    design_spheres: List[str] = None,
    temperature: float = DEFAULT_MPNN_TEMPERATURE,
    num_designs: int = DEFAULT_MPNN_NUMBER_OF_BATCHES,
    batch_size: int = DEFAULT_MPNN_BATCH_SIZE,
    omit_aa: str = DEFAULT_OMIT_AA,
    enhance: Optional[str] = None,
    additional_fixed: List[str] = None,
    bias_aa: Dict[str, float] = None,
    bias_aa_per_residue: Dict[str, Dict[str, float]] = None,
    use_sc_context: int = DEFAULT_LIGAND_MPNN_USE_SC_CONTEXT,
    pack_side_chains: int = DEFAULT_PACK_SIDE_CHAINS,
    repack_everything: int = DEFAULT_REPACK_EVERYTHING,
    sc_denoising_steps: int = DEFAULT_SC_DENOISING_STEPS,
    reference_sequence: Optional[str] = None,
    packed_suffix: Optional[str] = None,
) -> MPNNInput:
    """Create MPNNInput from a ResidueClassifier.

    This helper function extracts fixed and redesigned residues from the
    classifier's sphere assignments.

    Batch calculation:
        number_of_batches = ceil(num_designs / batch_size)

        This ensures we generate at least num_designs sequences. If more unique
        sequences are generated than requested (overshoot), diversity-based
        clustering is applied to select the most diverse subset.

    Args:
        pdb_path: Path to input PDB
        out_folder: Output directory
        residue_classifier: ResidueClassifier with classified residues
        design_spheres: Which spheres to design (default: ["primary"])
        temperature: Sampling temperature
        num_designs: Target number of unique designs (not batches)
        batch_size: Sequences per batch (higher = faster, potentially less diverse)
        omit_aa: Amino acids to omit
        enhance: Optional enhancement model (e.g., "plddt_3_20240930-f9c9ea0f")
        additional_fixed: Extra residues to fix (e.g., conserved mutations)
        bias_aa: Global amino acid biases
        reference_sequence: Original input sequence for diversity filtering
            (if None, tries to extract from PDB)
        packed_suffix: Suffix for packed PDB filenames (prevents collisions)

    Returns:
        MPNNInput configured for the classifier's design region
    """
    from .residue_classifier import DesignSphere

    if design_spheres is None:
        design_spheres = ["primary"]

    # Map string names to enum values (supports both new and legacy names)
    sphere_map = {
        # New names
        "core": DesignSphere.DESIGN_CORE,
        "shell": DesignSphere.DESIGN_SHELL,
        "design_core": DesignSphere.DESIGN_CORE,
        "design_shell": DesignSphere.DESIGN_SHELL,
        "flex": DesignSphere.FLEX,
        "frozen": DesignSphere.FROZEN,
        # Legacy names (backwards compatibility)
        "primary": DesignSphere.DESIGN_CORE,
        "secondary": DesignSphere.DESIGN_SHELL,
        "repack_primary": DesignSphere.FLEX,
        "repack_secondary": DesignSphere.FLEX,
        "distant": DesignSphere.FROZEN,
    }

    target_spheres: Set[DesignSphere] = set()
    design_all = False
    for s in design_spheres:
        s = s.strip().lower()
        if s in ("all", "global"):
            design_all = True
        elif s in ("repack", "flex"):
            target_spheres.add(DesignSphere.FLEX)
        else:
            mapped = sphere_map.get(s)
            if mapped:
                target_spheres.add(mapped)
            else:
                # Allow passing DesignSphere directly
                if isinstance(s, DesignSphere):
                    target_spheres.add(s)
                else:
                    LOGGER.warning(f"Unknown sphere '{s}' - ignoring")

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
        elif design_all or res_info.sphere in target_spheres:
            # In design sphere - redesign
            redesigned_residues.append(res_id)
            LOGGER.debug(f"  Design: {res_id} ({res_info.resname}, sphere={res_info.sphere})")
        else:
            # Outside design sphere - fixed
            fixed_residues.append(res_id)

    # Fallback: if core-only yielded 0 designable residues, expand to shell
    if len(redesigned_residues) == 0 and DesignSphere.DESIGN_SHELL not in target_spheres:
        LOGGER.warning("No designable residues in design core, expanding to include design shell")
        target_spheres.add(DesignSphere.DESIGN_SHELL)
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

    # Calculate number of batches to ensure we get at least num_designs total
    # total_generated = batch_size * number_of_batches >= num_designs
    number_of_batches = math.ceil(num_designs / batch_size)
    total_expected = batch_size * number_of_batches
    if total_expected > num_designs:
        LOGGER.info(f"Batch calculation: {num_designs} requested, batch_size={batch_size}, "
                   f"running {number_of_batches} batches (may generate up to {total_expected})")

    return MPNNInput(
        pdb_path=pdb_path,
        out_folder=out_folder,
        fixed_residues=fixed_residues,
        temperature=temperature,
        batch_size=batch_size,
        number_of_batches=number_of_batches,
        omit_aa=omit_aa,
        enhance=enhance,
        bias_aa=bias_aa or {},
        bias_aa_per_residue=bias_aa_per_residue or {},
        pack_side_chains=pack_side_chains,
        repack_everything=repack_everything,
        use_sc_context=use_sc_context,
        sc_denoising_steps=sc_denoising_steps,
        packed_suffix=packed_suffix,
        target_num_designs=num_designs,
        reference_sequence=reference_sequence,
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
    # Calculate number of batches to ensure we get at least num_designs total
    number_of_batches = math.ceil(num_designs / batch_size)

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
            number_of_batches=number_of_batches,
            omit_aa=omit_aa,
            target_num_designs=num_designs,
        )

        try:
            result = runner.run(mpnn_input)
            results[pdb_path] = result
        except Exception as e:
            LOGGER.error(f"MPNN failed for {pdb_path}: {e}")
            results[pdb_path] = MPNNResult()  # Empty result

    return results
