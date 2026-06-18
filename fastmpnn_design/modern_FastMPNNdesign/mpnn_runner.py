#!/usr/bin/env python3
"""
MPNN Runner - Subprocess-based interface for ProteinMPNN/LigandMPNN execution.

This module provides a unified interface for running MPNN via subprocess calls
to /net/software/lab/fused_mpnn/seth_temp/run.py, which handles model loading
and inference efficiently.

The packed PDB outputs serve as initial guesses for Rosetta repacking.
"""

import os
import sys
import glob
import subprocess
# --- locate repo root + shared external paths ---
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "repo_paths.py").is_file():
        _sys.path.insert(0, str(_anc)); break
import repo_paths
import tempfile
import json
import shutil
from typing import List, Dict, Optional, Union

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from constants import (
    DEFAULT_MPNN_RUNNER, DEFAULT_MODEL_TYPE, DEFAULT_OMIT_AA, DEFAULT_APPTAINER_IMAGE,
    DEFAULT_LIGAND_MPNN_USE_SC_CONTEXT, DEFAULT_REPACK_EVERYTHING, DEFAULT_PACK_SIDE_CHAINS,
    DEFAULT_SC_DENOISING_STEPS, DEFAULT_ENHANCE_MODEL, CONTAINER_ENV_VAR,
    DEFAULT_MPNN_SERVER_PORT, DEFAULT_MPNN_SERVER_HOST
)


def is_inside_container() -> bool:
    """Detect if we're running inside an Apptainer/Singularity container."""
    if os.environ.get(CONTAINER_ENV_VAR):
        return True
    if os.environ.get("SINGULARITY_CONTAINER") or os.environ.get("APPTAINER_CONTAINER"):
        return True
    if os.path.exists("/.singularity.d"):
        return True
    return False

# =============================================================================
# Amino Acid Mappings (exported for use by other modules)
# =============================================================================

restype_1to3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
    'X': 'UNK'
}

restype_3to1 = {v: k for k, v in restype_1to3.items()}


# =============================================================================
# MPNNServerClient - Client for socket-based MPNN server
# =============================================================================

class MPNNServerClient:
    """
    Client for connecting to a running MPNN server.

    The server loads model weights once, so subsequent requests avoid
    the ~2-5 second model loading overhead.

    Usage:
        client = MPNNServerClient(host="localhost", port=5000)
        if client.is_available():
            result = client.run(mpnn_input)
        else:
            # Fall back to subprocess mode
            runner = MPNNRunner()
            result = runner.run(mpnn_input)
    """

    def __init__(self, host: str = DEFAULT_MPNN_SERVER_HOST,
                 port: int = DEFAULT_MPNN_SERVER_PORT,
                 timeout: float = 600.0,
                 verbose: bool = True):
        """
        Initialize the MPNN server client.

        Args:
            host: Server hostname
            port: Server port
            timeout: Socket timeout in seconds
            verbose: Print debug information
        """
        self._host = host
        self._port = port
        self._timeout = timeout
        self._verbose = verbose

    def _log(self, message: str):
        """Print log message if verbose mode is enabled."""
        if self._verbose:
            print(f"[MPNNServerClient] {message}")

    def is_available(self) -> bool:
        """Check if the server is available and accepting connections."""
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((self._host, self._port))
            sock.close()
            return True
        except (socket.error, socket.timeout):
            return False

    def _recv_message(self, sock) -> Optional[bytes]:
        """Receive a length-prefixed message from socket."""
        import struct
        # Read 4-byte length prefix
        length_data = b''
        while len(length_data) < 4:
            chunk = sock.recv(4 - len(length_data))
            if not chunk:
                return None
            length_data += chunk

        message_length = struct.unpack('>I', length_data)[0]

        # Read message
        data = b''
        while len(data) < message_length:
            chunk = sock.recv(min(8192, message_length - len(data)))
            if not chunk:
                return None
            data += chunk

        return data

    def _send_message(self, sock, data: bytes):
        """Send a length-prefixed message over socket."""
        import struct
        length = struct.pack('>I', len(data))
        sock.sendall(length + data)

    def run(self, input_obj: 'MPNNRunner.MPNN_Input', pack_sc: bool = True) -> Dict:
        """
        Send an MPNN request to the server.

        Args:
            input_obj: MPNN_Input object with design configuration
            pack_sc: Whether to pack side chains

        Returns:
            Dict with keys:
                - success: bool
                - generated_sequences: List of sequences
                - packed_pdbs: List of packed PDB strings
                - packed: Dict mapping index to list of PDB strings (compatibility)
                - error: Error message if failed
        """
        import socket

        self._log(f"Connecting to server at {self._host}:{self._port}")

        try:
            # Connect to server
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self._timeout)
            sock.connect((self._host, self._port))

            # Build request
            request = {
                'pdb': input_obj.pdb,
                'name': input_obj.name or 'unnamed',
                'temperature': input_obj.temperature,
                'batch_size': input_obj.batch_size,
                'number_of_batches': input_obj.number_of_batches,
                'fixed_residues': input_obj.fixed_residues or [],
                'omit_AA': ''.join(input_obj.omit_AA) if input_obj.omit_AA else '',
                'bias_AA': input_obj.bias_AA,
                'bias_AA_per_residue': input_obj.bias_AA_per_residue,
                'pack_side_chains': pack_sc,
                'repack_everything': input_obj.repack_everything,
                'sc_num_denoising_steps': input_obj.sc_num_denoising_steps,
            }

            # Send request
            request_data = json.dumps(request).encode('utf-8')
            self._send_message(sock, request_data)
            self._log(f"Sent request ({len(request_data)} bytes)")

            # Receive response
            response_data = self._recv_message(sock)
            if not response_data:
                raise RuntimeError("No response from server")

            response = json.loads(response_data.decode('utf-8'))
            self._log(f"Received response: success={response.get('success')}")

            sock.close()

            # Convert to expected format
            packed_pdbs = response.get('packed_pdbs', [])
            result = {
                'success': response.get('success', False),
                'generated_sequences': response.get('generated_sequences', []),
                'packed_pdbs': packed_pdbs,
                'packed': {i: [pdb] for i, pdb in enumerate(packed_pdbs)},
                'native_sequence': None,
                'output_dir': None,
            }

            if not response.get('success'):
                result['error'] = response.get('error', 'Unknown error')
                self._log(f"Server error: {result['error']}")

            return result

        except socket.timeout:
            self._log(f"Timeout connecting to server")
            return {
                'success': False,
                'error': 'Connection timeout',
                'generated_sequences': [],
                'packed_pdbs': [],
                'packed': {},
            }
        except socket.error as e:
            self._log(f"Socket error: {e}")
            return {
                'success': False,
                'error': str(e),
                'generated_sequences': [],
                'packed_pdbs': [],
                'packed': {},
            }
        except Exception as e:
            self._log(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'generated_sequences': [],
                'packed_pdbs': [],
                'packed': {},
            }


# =============================================================================
# MPNNRunner Class - In-process execution with model caching
# =============================================================================

# Global model cache - shared across all MPNNRunner instances
_MPNN_MODEL_CACHE = {
    'model': None,
    'model_sc': None,
    'model_type': None,
    'enhance_model': None,
    'fused_mpnn': None,
}


class MPNNRunner:
    """
    MPNN Runner with automatic model caching.

    By default, models are loaded once on the first MPNN call and cached
    in memory for subsequent calls, eliminating the ~2-5 second loading
    overhead for each call.

    Falls back to subprocess execution if running outside a container
    where fused_mpnn is not available.

    Usage:
        runner = MPNNRunner("ligand_mpnn")
        inp = runner.MPNN_Input()
        inp.pdb = "/path/to/structure.pdb"  # or PDB string
        inp.temperature = 0.1
        inp.batch_size = 5
        inp.fixed_residues = ["A12", "A13", "A14"]
        result = runner.run(inp)
        # result["packed_pdbs"] contains list of packed PDB file contents
    """

    def __init__(self, model_type: str = DEFAULT_MODEL_TYPE,
                 checkpoint_path: str = None,
                 ligand_mpnn_use_side_chain_context: bool = True,
                 pack_sc: bool = True,
                 seed: int = None,
                 verbose: bool = False,
                 enhance_model: str = None,
                 container_image: str = DEFAULT_APPTAINER_IMAGE,
                 runner_script: str = DEFAULT_MPNN_RUNNER,
                 use_server: bool = False,
                 server_host: str = DEFAULT_MPNN_SERVER_HOST,
                 server_port: int = DEFAULT_MPNN_SERVER_PORT,
                 cache_models: bool = True):
        """
        Initialize MPNNRunner.

        Args:
            model_type: Model type (e.g., 'ligand_mpnn', 'protein_mpnn')
            checkpoint_path: Optional custom checkpoint path
            ligand_mpnn_use_side_chain_context: Use side chain context for ligand MPNN
            pack_sc: Enable side-chain packing (default: True)
            seed: Random seed for reproducibility
            verbose: Print debug information
            enhance_model: Enhanced model name (e.g., 'plddt_3_20240930-f9c9ea0f')
            container_image: Apptainer image path
            runner_script: Path to run.py script
            use_server: If True, try connecting to external MPNN server
            server_host: MPNN server hostname
            server_port: MPNN server port
            cache_models: If True (default), load models once and cache in memory
        """
        self._model_type = model_type
        self._checkpoint_path = checkpoint_path
        self._ligand_mpnn_use_side_chain_context = ligand_mpnn_use_side_chain_context
        self._pack_sc = pack_sc
        self._seed = seed
        self._verbose = verbose
        self._enhance_model = enhance_model if enhance_model else DEFAULT_ENHANCE_MODEL
        self._container_image = container_image
        self._runner_script = runner_script
        self._use_server = use_server
        self._server_host = server_host
        self._server_port = server_port
        self._server_client = None
        self._cache_models = cache_models
        self._use_cached_models = False  # Will be set True after successful cache load

        # Initialize server client if server mode enabled
        if self._use_server:
            self._server_client = MPNNServerClient(
                host=server_host,
                port=server_port,
                verbose=verbose
            )

        if self._verbose:
            print(f"[MPNNRunner] Initialized with model_type={model_type}, enhance={self._enhance_model}")
            print(f"[MPNNRunner] Model caching: {'enabled' if cache_models else 'disabled'}")
            if self._use_server:
                print(f"[MPNNRunner] External server mode: {server_host}:{server_port}")

    def _load_models_if_needed(self):
        """Load and cache models on first use. Returns True if models are available."""
        global _MPNN_MODEL_CACHE

        # Check if we already have cached models of the right type
        if (_MPNN_MODEL_CACHE['model'] is not None and
            _MPNN_MODEL_CACHE['model_type'] == self._model_type and
            _MPNN_MODEL_CACHE['enhance_model'] == self._enhance_model):
            self._use_cached_models = True
            return True

        # Try to import and load fused_mpnn
        try:
            if _MPNN_MODEL_CACHE['fused_mpnn'] is None:
                if self._verbose:
                    print(f"[MPNNRunner] Importing fused_mpnn...")
                sys.path.insert(0, str(_Path(repo_paths.FUSED_MPNN_RUN).parent))
                import fused_mpnn
                _MPNN_MODEL_CACHE['fused_mpnn'] = fused_mpnn

            fused_mpnn = _MPNN_MODEL_CACHE['fused_mpnn']

            if self._verbose:
                print(f"[MPNNRunner] Loading MPNN models (this happens once)...")
                print(f"[MPNNRunner]   Model type: {self._model_type}")
                print(f"[MPNNRunner]   Enhance: {self._enhance_model}")

            # Load main model
            _MPNN_MODEL_CACHE['model'] = fused_mpnn.load_model(
                model_type=self._model_type,
                enhance=self._enhance_model
            )

            # Load side-chain packer model
            _MPNN_MODEL_CACHE['model_sc'] = fused_mpnn.load_sc_model()

            _MPNN_MODEL_CACHE['model_type'] = self._model_type
            _MPNN_MODEL_CACHE['enhance_model'] = self._enhance_model

            if self._verbose:
                print(f"[MPNNRunner] Models loaded and cached successfully")

            self._use_cached_models = True
            return True

        except ImportError as e:
            if self._verbose:
                print(f"[MPNNRunner] Cannot import fused_mpnn: {e}")
                print(f"[MPNNRunner] Will use subprocess mode (slower)")
            self._use_cached_models = False
            return False
        except Exception as e:
            if self._verbose:
                print(f"[MPNNRunner] Failed to load models: {e}")
                print(f"[MPNNRunner] Will use subprocess mode (slower)")
            self._use_cached_models = False
            return False

    def _run_with_cached_models(self, input_obj: 'MPNNRunner.MPNN_Input', pack_sc: bool) -> Dict:
        """Run MPNN using cached in-memory models."""
        global _MPNN_MODEL_CACHE

        fused_mpnn = _MPNN_MODEL_CACHE['fused_mpnn']
        model = _MPNN_MODEL_CACHE['model']
        model_sc = _MPNN_MODEL_CACHE['model_sc'] if pack_sc else None

        # Create temporary directory
        tmp_dir = tempfile.mkdtemp(prefix="mpnn_cached_")

        try:
            # Write PDB to temp file
            pdb_path = input_obj.pdb
            if pdb_path is None:
                raise ValueError("input_obj.pdb must be set")

            # Check if input is a PDB string
            if '\n' in pdb_path or pdb_path.startswith('HEADER') or pdb_path.startswith('ATOM') or pdb_path.startswith('REMARK'):
                tmp_pdb = os.path.join(tmp_dir, "input.pdb")
                with open(tmp_pdb, 'w') as f:
                    f.write(pdb_path)
                pdb_path = tmp_pdb

            # Prepare bias_AA_per_residue file if needed
            bias_per_residue_path = None
            if input_obj.bias_AA_per_residue:
                bias_per_residue_path = os.path.join(tmp_dir, "bias_per_residue.json")
                with open(bias_per_residue_path, 'w') as f:
                    json.dump(input_obj.bias_AA_per_residue, f)

            # Prepare arguments
            fixed_residues = " ".join(input_obj.fixed_residues) if input_obj.fixed_residues else None
            omit_AA = "".join(input_obj.omit_AA) if input_obj.omit_AA else None
            bias_AA = None
            if input_obj.bias_AA:
                bias_AA = ",".join(f"{aa}:{val}" for aa, val in input_obj.bias_AA.items())

            if self._verbose:
                print(f"[MPNNRunner] Running with cached models: T={input_obj.temperature}, batch={input_obj.batch_size}x{input_obj.number_of_batches}")

            # Run inference
            result = fused_mpnn.run_inference(
                model=model,
                model_sc=model_sc,
                pdb_path=pdb_path,
                out_folder=tmp_dir,
                temperature=input_obj.temperature,
                batch_size=input_obj.batch_size,
                number_of_batches=input_obj.number_of_batches,
                fixed_residues=fixed_residues,
                omit_AA=omit_AA,
                bias_AA=bias_AA,
                bias_AA_per_residue=bias_per_residue_path,
                pack_side_chains=pack_sc,
                repack_everything=input_obj.repack_everything,
                sc_num_denoising_steps=input_obj.sc_num_denoising_steps,
                ligand_mpnn_use_side_chain_context=self._ligand_mpnn_use_side_chain_context,
            )

            # Parse outputs
            return self._parse_outputs(tmp_dir, pack_sc)

        except Exception as e:
            if self._verbose:
                print(f"[MPNNRunner] Cached model run failed: {e}")
                import traceback
                traceback.print_exc()
            return {
                "generated_sequences": [],
                "packed_pdbs": [],
                "packed": {},
                "native_sequence": None,
                "output_dir": tmp_dir,
                "success": False
            }

    @property
    def model_type(self) -> str:
        return self._model_type

    class MPNN_Input:
        """Input configuration for MPNN design."""

        def __init__(self, obj=None):
            # PDB input (file path or string)
            self.pdb = None
            self.name = None

            # Design parameters
            self.temperature = 0.1
            self.batch_size = 1
            self.number_of_batches = 1

            # Residue specification - list of "CHAIN+RESNO" strings (e.g., ["A45", "B30"])
            self.fixed_residues = []
            self.design_residues = []
            self.chains_to_design = None

            # Amino acid biases
            self.omit_AA = []  # List of AA letters to omit
            self.bias_AA = None  # Dict of {AA: bias_value}
            self.bias_AA_per_residue = None  # Dict of {residue: {AA: bias}}

            # Side chain packing
            self.repack_everything = False  # Only repack designed residues
            self.sc_num_denoising_steps = DEFAULT_SC_DENOISING_STEPS
            self.number_of_packs_per_design = 1

            # Clone from another object if provided
            if obj is not None:
                for attr in dir(obj):
                    if not attr.startswith("_") and not callable(getattr(obj, attr)):
                        try:
                            setattr(self, attr, getattr(obj, attr))
                        except AttributeError:
                            pass

        def copy(self):
            """Create a copy of this input object."""
            new_obj = MPNNRunner.MPNN_Input()
            for attr in dir(self):
                if not attr.startswith("_") and not callable(getattr(self, attr)):
                    try:
                        val = getattr(self, attr)
                        if isinstance(val, (list, dict)):
                            import copy
                            setattr(new_obj, attr, copy.deepcopy(val))
                        else:
                            setattr(new_obj, attr, val)
                    except AttributeError:
                        pass
            return new_obj

    def run(self, input_obj: 'MPNNRunner.MPNN_Input', pack_sc: bool = None, **kwargs) -> Dict:
        """
        Run MPNN design with automatic model caching.

        By default, models are loaded once on the first call and cached in memory.
        This eliminates the ~2-5 second model loading overhead for subsequent calls.

        Falls back to subprocess mode if:
        - Running outside container (fused_mpnn not importable)
        - cache_models=False was set
        - External server mode is enabled and available

        Args:
            input_obj: MPNN_Input object with design configuration
            pack_sc: Override pack_sc setting (default: use instance setting)

        Returns:
            Dict with keys:
                - generated_sequences: List of designed sequence strings
                - packed_pdbs: List of packed PDB file contents (strings)
                - packed: Dict mapping index to list of PDB strings (for compatibility)
                - native_sequence: Original sequence (if available)
                - output_dir: Path to output directory
        """
        if pack_sc is None:
            pack_sc = self._pack_sc

        # Try external server mode first if enabled
        if self._use_server and self._server_client is not None:
            if self._server_client.is_available():
                if self._verbose:
                    print(f"[MPNNRunner] Using external server ({self._server_host}:{self._server_port})")
                result = self._server_client.run(input_obj, pack_sc=pack_sc)
                if result.get('success', False):
                    return result
                else:
                    if self._verbose:
                        print(f"[MPNNRunner] Server request failed, falling back to local execution")
            else:
                if self._verbose:
                    print(f"[MPNNRunner] External server not available, using local execution")

        # Try cached model mode (default) - loads models once, reuses them
        if self._cache_models:
            if self._load_models_if_needed():
                return self._run_with_cached_models(input_obj, pack_sc)
            # If model loading failed, fall through to subprocess mode

        # Fallback: subprocess mode (slower - loads models each time)
        if self._verbose:
            print(f"[MPNNRunner] Using subprocess mode")

        # Create temporary directory for MPNN output
        tmp_dir = tempfile.mkdtemp(prefix="mpnn_")
        tmp_pdb = None

        try:
            # Handle PDB input - write string to temp file if needed
            pdb_path = input_obj.pdb
            if pdb_path is None:
                raise ValueError("input_obj.pdb must be set to a PDB path or string")

            # Check if input is a PDB string
            if '\n' in pdb_path or pdb_path.startswith('HEADER') or pdb_path.startswith('ATOM') or pdb_path.startswith('REMARK'):
                tmp_pdb = os.path.join(tmp_dir, "input.pdb")
                with open(tmp_pdb, 'w') as f:
                    f.write(pdb_path)
                pdb_path = tmp_pdb

            # Build command
            cmd = self._build_command(pdb_path, tmp_dir, input_obj, pack_sc)

            if self._verbose:
                in_container = is_inside_container()
                print(f"[MPNNRunner] Inside container: {in_container}")
                print(f"[MPNNRunner] Output dir: {tmp_dir}")

            # Always print the command being executed
            print(f"[MPNNRunner] Executing command:")
            print(f"  {' '.join(cmd)}")

            # Execute
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout

            if result.returncode != 0:
                print(f"[MPNNRunner] ERROR: MPNN failed (return code {result.returncode})")
                if result.stderr:
                    print(f"[MPNNRunner] stderr: {result.stderr[:1000]}")
                if result.stdout:
                    print(f"[MPNNRunner] stdout: {result.stdout[:500]}")
                return {
                    "generated_sequences": [],
                    "packed_pdbs": [],
                    "packed": {},
                    "native_sequence": None,
                    "output_dir": tmp_dir,
                    "success": False
                }

            # Parse outputs
            return self._parse_outputs(tmp_dir, pack_sc)

        except subprocess.TimeoutExpired:
            print(f"[MPNNRunner] ERROR: MPNN timed out after 600 seconds")
            return {
                "generated_sequences": [],
                "packed_pdbs": [],
                "packed": {},
                "native_sequence": None,
                "output_dir": tmp_dir,
                "success": False
            }

        except Exception as e:
            print(f"[MPNNRunner] Exception: {e}")
            import traceback
            traceback.print_exc()
            return {
                "generated_sequences": [],
                "packed_pdbs": [],
                "packed": {},
                "native_sequence": None,
                "output_dir": tmp_dir,
                "success": False
            }

    def _build_command(self, pdb_path: str, output_dir: str, input_obj: 'MPNNRunner.MPNN_Input',
                       pack_sc: bool) -> List[str]:
        """Build the subprocess command for run.py."""
        # Check if we're already inside a container - if so, run python directly
        if is_inside_container():
            cmd = [
                "python", self._runner_script,
            ]
        else:
            cmd = [
                "apptainer", "exec", self._container_image,
                "python", self._runner_script,
            ]

        cmd.extend([
            "--model_type", self._model_type,
            "--pdb_path", pdb_path,
            "--out_folder", output_dir,
            "--temperature", str(input_obj.temperature),
            "--number_of_batches", str(input_obj.number_of_batches),
            "--batch_size", str(input_obj.batch_size),
            "--pack_side_chains", "1" if pack_sc else "0",
            "--sc_num_denoising_steps", str(input_obj.sc_num_denoising_steps),
            "--repack_everything", "1" if input_obj.repack_everything else "0",
            "--ligand_mpnn_use_side_chain_context", "1" if self._ligand_mpnn_use_side_chain_context else "0",
        ])

        # Add enhance model
        if self._enhance_model:
            cmd.extend(["--enhance", self._enhance_model])

        # Add custom checkpoint if specified
        if self._checkpoint_path:
            cmd.extend(["--checkpoint_path", self._checkpoint_path])

        # Add seed if specified
        if self._seed is not None:
            cmd.extend(["--seed", str(self._seed)])

        # Add omit_AA
        if input_obj.omit_AA:
            omit_str = "".join(input_obj.omit_AA) if isinstance(input_obj.omit_AA, list) else input_obj.omit_AA
            cmd.extend(["--omit_AA", omit_str])

        # Add fixed residues
        if input_obj.fixed_residues:
            # Convert to space-separated format expected by run.py
            fixed_str = " ".join(input_obj.fixed_residues)
            cmd.extend(["--fixed_residues", fixed_str])

        # Add design residues (redesigned_residues in run.py)
        if input_obj.design_residues:
            design_str = " ".join(input_obj.design_residues)
            cmd.extend(["--redesigned_residues", design_str])

        # Add chains to design
        if input_obj.chains_to_design:
            if isinstance(input_obj.chains_to_design, list):
                chains_str = ",".join(input_obj.chains_to_design)
            else:
                chains_str = input_obj.chains_to_design
            cmd.extend(["--chains_to_design", chains_str])

        # Add bias_AA
        if input_obj.bias_AA:
            # Format: "A:-1.024,P:2.34,C:-12.34"
            bias_parts = [f"{aa}:{val}" for aa, val in input_obj.bias_AA.items()]
            cmd.extend(["--bias_AA", ",".join(bias_parts)])

        # Add bias_AA_per_residue via temp JSON file
        if input_obj.bias_AA_per_residue:
            bias_file = os.path.join(os.path.dirname(output_dir), "bias_per_residue.json")
            with open(bias_file, 'w') as f:
                json.dump(input_obj.bias_AA_per_residue, f)
            cmd.extend(["--bias_AA_per_residue", bias_file])

        # Add suffix based on name and temperature
        if input_obj.name:
            suffix = f"_{input_obj.name}_T{input_obj.temperature:.2f}_"
            cmd.extend(["--packed_suffix", suffix])

        return cmd

    def _parse_outputs(self, output_dir: str, pack_sc: bool) -> Dict:
        """Parse MPNN outputs from the output directory.

        We primarily use the packed PDB structures and extract sequences from them
        to ensure 1:1 correspondence between sequences and structures.
        """
        result = {
            "generated_sequences": [],
            "packed_pdbs": [],
            "packed": {},
            "native_sequence": None,
            "output_dir": output_dir,
            "success": True
        }

        # Parse packed PDB files - this is our primary output
        if pack_sc:
            packed_dir = os.path.join(output_dir, "packed")
            if os.path.exists(packed_dir):
                pdb_files = sorted(glob.glob(os.path.join(packed_dir, "*.pdb")))
                print(f"[MPNNRunner] Found {len(pdb_files)} packed structures in {packed_dir}")

                for idx, pdb_file in enumerate(pdb_files):
                    with open(pdb_file, 'r') as f:
                        pdb_content = f.read()
                    result["packed_pdbs"].append(pdb_content)
                    # Also populate "packed" dict for compatibility with old interface
                    result["packed"][idx] = [pdb_content]

                    # Extract sequence from PDB to ensure correspondence
                    seq = self._extract_sequence_from_pdb(pdb_content)
                    if seq:
                        result["generated_sequences"].append(seq)

        # Clean up unnecessary subdirectories - we only need packed/
        for subdir in ["seqs", "backbones", "stats"]:
            subdir_path = os.path.join(output_dir, subdir)
            if os.path.exists(subdir_path):
                shutil.rmtree(subdir_path, ignore_errors=True)

        # Clean up input.pdb if it exists
        input_pdb = os.path.join(output_dir, "input.pdb")
        if os.path.exists(input_pdb):
            os.remove(input_pdb)

        if self._verbose:
            print(f"[MPNNRunner] Parsed {len(result['packed_pdbs'])} packed structures")
            print(f"[MPNNRunner] Extracted {len(result['generated_sequences'])} sequences from PDBs")

        return result

    def _extract_sequence_from_pdb(self, pdb_content: str) -> str:
        """Extract protein sequence from PDB content."""
        residues = {}  # (chain, resno) -> resname

        for line in pdb_content.split('\n'):
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                chain = line[21]
                try:
                    resno = int(line[22:26].strip())
                    resname = line[17:20].strip()
                    residues[(chain, resno)] = resname
                except ValueError:
                    continue

        # Sort by chain then residue number and convert to 1-letter codes
        sorted_residues = sorted(residues.items(), key=lambda x: (x[0][0], x[0][1]))

        sequence = ""
        for (chain, resno), resname in sorted_residues:
            one_letter = restype_3to1.get(resname, 'X')
            sequence += one_letter

        return sequence

    def cleanup_output(self, output_dir: str):
        """Clean up temporary output directory."""
        if output_dir and os.path.exists(output_dir) and output_dir.startswith(tempfile.gettempdir()):
            shutil.rmtree(output_dir, ignore_errors=True)


# =============================================================================
# MPNNConfig class for backward compatibility
# =============================================================================

class MPNNConfig:
    """Configuration for subprocess-based MPNN execution (backward compatibility)."""

    def __init__(self, runner: str = DEFAULT_MPNN_RUNNER, model_type: str = DEFAULT_MODEL_TYPE,
                 omit_aa: str = DEFAULT_OMIT_AA, apptainer_image: str = DEFAULT_APPTAINER_IMAGE,
                 use_sc_context: int = DEFAULT_LIGAND_MPNN_USE_SC_CONTEXT,
                 repack_everything: int = DEFAULT_REPACK_EVERYTHING,
                 pack_side_chains: int = DEFAULT_PACK_SIDE_CHAINS,
                 sc_denoising_steps: int = DEFAULT_SC_DENOISING_STEPS):
        self.runner = runner
        self.model_type = model_type
        self.omit_aa = omit_aa
        self.apptainer_image = apptainer_image
        self.use_sc_context = use_sc_context
        self.repack_everything = repack_everything
        self.pack_side_chains = pack_side_chains
        self.sc_denoising_steps = sc_denoising_steps

    def to_dict(self) -> Dict:
        return self.__dict__.copy()

    @classmethod
    def from_args(cls, args) -> "MPNNConfig":
        return cls(
            runner=getattr(args, 'mpnn_runner', DEFAULT_MPNN_RUNNER),
            model_type=getattr(args, 'mpnn_model_type', DEFAULT_MODEL_TYPE),
            omit_aa=getattr(args, 'mpnn_omit_aa', DEFAULT_OMIT_AA),
            apptainer_image=getattr(args, 'apptainer_image', DEFAULT_APPTAINER_IMAGE),
            use_sc_context=getattr(args, 'mpnn_use_sc_context', DEFAULT_LIGAND_MPNN_USE_SC_CONTEXT),
            repack_everything=getattr(args, 'mpnn_repack_everything', DEFAULT_REPACK_EVERYTHING),
            pack_side_chains=getattr(args, 'mpnn_pack_side_chains', DEFAULT_PACK_SIDE_CHAINS)
        )


# =============================================================================
# Standalone functions for backward compatibility
# =============================================================================

def run_mpnn(pdb_path: str, config: MPNNConfig, output_dir: str, name: str,
             temperature: float = 0.1, batch_size: int = 2, num_batches: int = 1,
             fixed_residues: Optional[List[str]] = None, design_residues: Optional[List[str]] = None,
             bias_aa_per_residue: Optional[Dict] = None, verbose: bool = True) -> Dict:
    """
    Execute MPNN design via subprocess (standalone function).

    Returns dict with 'success', 'output_files', 'sequences', 'stdout', 'stderr'.
    """
    runner = MPNNRunner(
        model_type=config.model_type,
        container_image=config.apptainer_image,
        runner_script=config.runner,
        verbose=verbose
    )

    inp = runner.MPNN_Input()
    inp.pdb = pdb_path
    inp.name = name
    inp.temperature = temperature
    inp.batch_size = batch_size
    inp.number_of_batches = num_batches
    inp.fixed_residues = fixed_residues or []
    inp.design_residues = design_residues or []
    inp.omit_AA = list(config.omit_aa) if config.omit_aa else []
    inp.repack_everything = bool(config.repack_everything)
    inp.sc_num_denoising_steps = config.sc_denoising_steps

    if bias_aa_per_residue:
        inp.bias_AA_per_residue = bias_aa_per_residue

    result = runner.run(inp, pack_sc=bool(config.pack_side_chains))

    # Convert to old format
    return {
        "success": result.get("success", True),
        "output_files": glob.glob(os.path.join(result["output_dir"], "packed", "*.pdb")),
        "sequences": result["generated_sequences"],
        "stdout": "",
        "stderr": ""
    }


def run_mpnn_from_pose(pose, config: MPNNConfig, output_dir: str, name: str, **kwargs) -> Dict:
    """Run MPNN from a PyRosetta pose."""
    import pyrosetta
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        tmp_pdb = f.name
        pose.dump_pdb(tmp_pdb)
    try:
        return run_mpnn(tmp_pdb, config, output_dir, name, **kwargs)
    finally:
        if os.path.exists(tmp_pdb):
            os.remove(tmp_pdb)


def run_mpnn_from_pdbstring(pdb_string: str, config: MPNNConfig, output_dir: str, name: str, **kwargs) -> Dict:
    """Run MPNN from a PDB string."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_string)
        tmp_pdb = f.name
    try:
        return run_mpnn(tmp_pdb, config, output_dir, name, **kwargs)
    finally:
        if os.path.exists(tmp_pdb):
            os.remove(tmp_pdb)


# =============================================================================
# CLI for standalone execution
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MPNN Runner - Execute ProteinMPNN/LigandMPNN")
    parser.add_argument("--pdb", required=True, help="Input PDB file")
    parser.add_argument("--output_dir", default=".", help="Output directory")
    parser.add_argument("--name", default="mpnn_output", help="Output name prefix")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_batches", type=int, default=1, help="Number of batches")
    parser.add_argument("--fixed_residues", type=str, help="Space-separated fixed residues")
    parser.add_argument("--design_residues", type=str, help="Space-separated design residues")
    parser.add_argument("--model_type", default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--omit_aa", default=DEFAULT_OMIT_AA)
    parser.add_argument("--apptainer_image", default=DEFAULT_APPTAINER_IMAGE)
    parser.add_argument("--pack_side_chains", type=int, default=DEFAULT_PACK_SIDE_CHAINS)
    parser.add_argument("--repack_everything", type=int, default=DEFAULT_REPACK_EVERYTHING)
    args = parser.parse_args()

    config = MPNNConfig(
        model_type=args.model_type,
        omit_aa=args.omit_aa,
        apptainer_image=args.apptainer_image,
        pack_side_chains=args.pack_side_chains,
        repack_everything=args.repack_everything
    )

    fixed = args.fixed_residues.split() if args.fixed_residues else None
    design = args.design_residues.split() if args.design_residues else None

    result = run_mpnn(args.pdb, config, args.output_dir, args.name, args.temperature,
                      args.batch_size, args.num_batches, fixed, design)

    print(f"Success: {result['success']}")
    print(f"Sequences: {len(result['sequences'])}")
    print(f"Output files: {len(result['output_files'])}")

    sys.exit(0 if result["success"] else 1)
