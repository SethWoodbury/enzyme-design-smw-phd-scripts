#!/usr/bin/env python3
"""
MPNN Server - Socket-based server for ProteinMPNN/LigandMPNN inference.

This server loads model weights once at startup and serves multiple requests,
eliminating the ~2-5 second overhead of weight loading for each MPNN call.

Usage:
    # Start server (inside container):
    apptainer exec --network host /software/containers/universal.sif \
        python mpnn_server.py --port 5000 --model_type ligand_mpnn

    # Connect from client:
    from mpnn_runner import MPNNServerClient
    client = MPNNServerClient(host="localhost", port=5000)
    result = client.run(mpnn_input)
"""

import os
import sys
import json
import socket
import struct
import tempfile
import argparse
import traceback
import threading
from typing import Dict, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from constants import (
    DEFAULT_MODEL_TYPE, DEFAULT_ENHANCE_MODEL,
    DEFAULT_MPNN_SERVER_PORT, DEFAULT_MPNN_SERVER_HOST,
    DEFAULT_SC_DENOISING_STEPS
)


class MPNNServer:
    """
    Socket-based MPNN server that loads models once and serves multiple requests.

    The server:
    1. Loads MPNN model weights at startup (expensive operation)
    2. Listens for incoming JSON requests
    3. Runs inference using pre-loaded models
    4. Returns results via socket

    Protocol:
    - Client sends: 4-byte length prefix (big-endian) + JSON request
    - Server responds: 4-byte length prefix (big-endian) + JSON response
    """

    def __init__(self, model_type: str = DEFAULT_MODEL_TYPE,
                 enhance_model: str = DEFAULT_ENHANCE_MODEL,
                 port: int = DEFAULT_MPNN_SERVER_PORT,
                 host: str = "0.0.0.0",
                 verbose: bool = True):
        """
        Initialize the MPNN server.

        Args:
            model_type: Model type (e.g., 'ligand_mpnn', 'protein_mpnn')
            enhance_model: Enhanced model name
            port: Port to listen on
            host: Host to bind to (0.0.0.0 for all interfaces)
            verbose: Print detailed logs
        """
        self._model_type = model_type
        self._enhance_model = enhance_model
        self._port = port
        self._host = host
        self._verbose = verbose

        self._socket = None
        self._model = None
        self._model_sc = None
        self._running = False

        # Import and setup will happen in start()
        self._fused_mpnn = None

    def _log(self, message: str):
        """Print log message if verbose mode is enabled."""
        if self._verbose:
            print(f"[MPNNServer] {message}")

    def _load_models(self):
        """Load MPNN models at startup. This is the expensive operation we want to do once."""
        self._log(f"Loading models: {self._model_type} (enhance: {self._enhance_model})")

        # Import fused_mpnn - this should be available in the container
        try:
            sys.path.insert(0, "/net/software/lab/fused_mpnn/seth_temp")
            import fused_mpnn
            self._fused_mpnn = fused_mpnn
        except ImportError as e:
            self._log(f"Failed to import fused_mpnn: {e}")
            self._log("Make sure you're running inside the appropriate container")
            raise

        # Load the main model
        self._log("Loading main MPNN model...")
        self._model = self._fused_mpnn.load_model(
            model_type=self._model_type,
            enhance=self._enhance_model
        )
        self._log(f"Main model loaded: {type(self._model)}")

        # Load the side-chain packer model
        self._log("Loading side-chain packer model...")
        self._model_sc = self._fused_mpnn.load_sc_model()
        self._log(f"Side-chain packer loaded: {type(self._model_sc)}")

        self._log("All models loaded successfully!")

    def start(self):
        """Start the server - load models and begin listening."""
        self._log(f"Starting MPNN server on {self._host}:{self._port}")

        # Load models first
        self._load_models()

        # Setup socket
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self._host, self._port))
        self._socket.listen(5)

        self._running = True
        self._log(f"Server listening on {self._host}:{self._port}")
        self._log("Press Ctrl+C to stop")

        try:
            while self._running:
                try:
                    client_socket, address = self._socket.accept()
                    self._log(f"Connection from {address}")
                    # Handle each client in a separate thread for basic concurrency
                    thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, address)
                    )
                    thread.daemon = True
                    thread.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    self._log(f"Accept error: {e}")
        except KeyboardInterrupt:
            self._log("Shutting down...")
        finally:
            self.stop()

    def stop(self):
        """Stop the server."""
        self._running = False
        if self._socket:
            self._socket.close()
            self._socket = None
        self._log("Server stopped")

    def _recv_message(self, sock: socket.socket) -> Optional[bytes]:
        """Receive a length-prefixed message from socket."""
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

    def _send_message(self, sock: socket.socket, data: bytes):
        """Send a length-prefixed message over socket."""
        length = struct.pack('>I', len(data))
        sock.sendall(length + data)

    def _handle_client(self, client_socket: socket.socket, address):
        """Handle a client connection."""
        try:
            # Receive request
            request_data = self._recv_message(client_socket)
            if not request_data:
                self._log(f"Empty request from {address}")
                return

            # Parse JSON request
            request = json.loads(request_data.decode('utf-8'))
            self._log(f"Received request: {request.get('name', 'unnamed')}")

            # Process request
            response = self._handle_request(request)

            # Send response
            response_data = json.dumps(response).encode('utf-8')
            self._send_message(client_socket, response_data)
            self._log(f"Sent response to {address}")

        except Exception as e:
            self._log(f"Error handling client {address}: {e}")
            traceback.print_exc()
            try:
                error_response = {'success': False, 'error': str(e)}
                self._send_message(client_socket, json.dumps(error_response).encode('utf-8'))
            except:
                pass
        finally:
            client_socket.close()

    def _handle_request(self, request: Dict) -> Dict:
        """
        Process an MPNN inference request.

        Args:
            request: Dict with keys:
                - pdb: PDB string
                - name: Output name
                - temperature: Sampling temperature
                - batch_size: Batch size
                - number_of_batches: Number of batches
                - fixed_residues: List of fixed residue strings
                - omit_AA: String of AAs to omit
                - bias_AA: Dict of AA biases
                - bias_AA_per_residue: Dict of per-residue AA biases
                - pack_side_chains: Whether to pack side chains
                - repack_everything: Whether to repack all residues
                - sc_num_denoising_steps: Denoising steps for SC packing

        Returns:
            Dict with keys:
                - success: bool
                - generated_sequences: List of sequences
                - packed_pdbs: List of packed PDB strings
                - error: Error message if failed
        """
        try:
            # Create temporary directory for MPNN
            with tempfile.TemporaryDirectory(prefix="mpnn_server_") as tmp_dir:
                # Write PDB to temp file
                pdb_path = os.path.join(tmp_dir, "input.pdb")
                with open(pdb_path, 'w') as f:
                    f.write(request['pdb'])

                # Extract parameters
                temperature = request.get('temperature', 0.1)
                batch_size = request.get('batch_size', 1)
                number_of_batches = request.get('number_of_batches', 1)
                fixed_residues = request.get('fixed_residues', [])
                omit_AA = request.get('omit_AA', '')
                bias_AA = request.get('bias_AA')
                bias_AA_per_residue = request.get('bias_AA_per_residue')
                pack_side_chains = request.get('pack_side_chains', True)
                repack_everything = request.get('repack_everything', False)
                sc_num_denoising_steps = request.get('sc_num_denoising_steps', DEFAULT_SC_DENOISING_STEPS)

                # Write bias per residue if provided
                bias_per_residue_path = None
                if bias_AA_per_residue:
                    bias_per_residue_path = os.path.join(tmp_dir, "bias_per_residue.json")
                    with open(bias_per_residue_path, 'w') as f:
                        json.dump(bias_AA_per_residue, f)

                # Run MPNN using fused_mpnn with pre-loaded models
                self._log(f"Running MPNN: T={temperature}, batch={batch_size}x{number_of_batches}")

                result = self._fused_mpnn.run_inference(
                    model=self._model,
                    model_sc=self._model_sc if pack_side_chains else None,
                    pdb_path=pdb_path,
                    out_folder=tmp_dir,
                    temperature=temperature,
                    batch_size=batch_size,
                    number_of_batches=number_of_batches,
                    fixed_residues=" ".join(fixed_residues) if fixed_residues else None,
                    omit_AA=omit_AA if omit_AA else None,
                    bias_AA=",".join(f"{aa}:{val}" for aa, val in bias_AA.items()) if bias_AA else None,
                    bias_AA_per_residue=bias_per_residue_path,
                    pack_side_chains=pack_side_chains,
                    repack_everything=repack_everything,
                    sc_num_denoising_steps=sc_num_denoising_steps,
                )

                # Collect results
                generated_sequences = result.get('sequences', [])
                packed_pdbs = []

                # Read packed PDB files
                packed_dir = os.path.join(tmp_dir, "packed")
                if os.path.exists(packed_dir):
                    import glob
                    for pdb_file in sorted(glob.glob(os.path.join(packed_dir, "*.pdb"))):
                        with open(pdb_file, 'r') as f:
                            packed_pdbs.append(f.read())

                self._log(f"Generated {len(generated_sequences)} sequences, {len(packed_pdbs)} packed structures")

                return {
                    'success': True,
                    'generated_sequences': generated_sequences,
                    'packed_pdbs': packed_pdbs,
                }

        except Exception as e:
            self._log(f"Inference error: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'generated_sequences': [],
                'packed_pdbs': []
            }


def main():
    """Main entry point for the MPNN server."""
    parser = argparse.ArgumentParser(
        description="MPNN Server - Load models once, serve multiple requests"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_MPNN_SERVER_PORT,
        help=f"Port to listen on (default: {DEFAULT_MPNN_SERVER_PORT})"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--model_type", type=str, default=DEFAULT_MODEL_TYPE,
        help=f"Model type (default: {DEFAULT_MODEL_TYPE})"
    )
    parser.add_argument(
        "--enhance", type=str, default=DEFAULT_ENHANCE_MODEL,
        help=f"Enhanced model name (default: {DEFAULT_ENHANCE_MODEL})"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity"
    )
    args = parser.parse_args()

    server = MPNNServer(
        model_type=args.model_type,
        enhance_model=args.enhance,
        port=args.port,
        host=args.host,
        verbose=not args.quiet
    )

    server.start()


if __name__ == "__main__":
    main()
