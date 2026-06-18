#!/usr/bin/env python3
"""MPNN Server - Persistent model for fast sequence design.

This server keeps ProteinMPNN model weights loaded in GPU memory between calls,
eliminating the ~15-20s model loading overhead per invocation.

Usage:
    # Start server manually (typically auto-started by MPNNRunner)
    apptainer exec --nv /net/software/containers/universal.sif \
        python /path/to/mpnn_server.py --host localhost --port 5000

    # With custom model type
    apptainer exec --nv container.sif python mpnn_server.py \
        --model_type ligand_mpnn --pack_side_chains

Protocol:
    - TCP socket with length-prefixed JSON messages
    - Request types: "design", "health", "shutdown"
    - Server auto-starts on first MPNN call if not running

Author: FastMPNN Pipeline
"""
import argparse
import copy
import json
import logging
import os
import random
import signal
import socket
import struct
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# MPNN imports (available inside container)
from data_utils import (
    featurize,
    get_score,
    get_seq_rec,
    make_pair_bias,
    parse_PDB,
    save_output_structure,
)
from model_utils import ProteinMPNN
from mpnn.datasets.transforms import (
    BuildMPNNInput,
    Compose,
    ExcludeByResName,
    GetAssembly,
    GetChainInfo,
    LoadFile,
    SeperateProteinLigand,
)
from sc_utils import Packer, pack_side_chains

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("mpnn_server")


class MPNNServer:
    """Persistent MPNN server with models loaded in GPU memory."""

    # Model checkpoint paths (from run.py defaults)
    CHECKPOINT_PATHS = {
        "protein_mpnn": "/databases/mpnn/vanilla_model_weights/v_48_020.pt",
        "ligand_mpnn": "/databases/mpnn/ligand_mpnn_model_weights/s25_r010_t300_p.pt",
        "per_residue_label_membrane_mpnn": "/databases/mpnn/tmd_per_residue_weights/tmd_v_48_020.pt",
        "global_label_membrane_mpnn": "/databases/mpnn/tmd_weights/v_48_020.pt",
        "soluble_mpnn": "/databases/mpnn/no_transmembrane/v_48_020.pt",
        "pssm_mpnn": "/databases/mpnn/pssm_model_weights/v_48_020.pt",
        "antibody_mpnn": "/databases/mpnn/antibody_mpnn_model_weights/v_48_020_bias_005.pt",
        "msa_mpnn": "/projects/ml/struc2seq/msa_mpnn_models/dropout_v1/last.pt",
    }
    CHECKPOINT_SC = "/projects/ml/struc2seq/ligandMPNN_models/b_v1/s_300756.pt"

    # Amino acid mappings
    RESTYPE_STR_TO_INT = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
        'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20
    }
    RESTYPE_INT_TO_STR = {v: k for k, v in RESTYPE_STR_TO_INT.items()}
    ALPHABET = list(RESTYPE_STR_TO_INT.keys())

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5000,
        model_type: str = "ligand_mpnn",
        pack_side_chains: bool = True,
    ):
        """Initialize MPNN server.

        Args:
            host: Server hostname
            port: Server port
            model_type: MPNN model type to load
            pack_side_chains: Whether to load side-chain packer model
        """
        self.host = host
        self.port = port
        self.model_type = model_type
        self.enable_packing = pack_side_chains

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        LOGGER.info(f"Using device: {self.device}")

        # Statistics
        self.requests_served = 0
        self.start_time = time.time()
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()

        # Model references
        self.model = None
        self.model_sc = None
        self.checkpoint_path = None

        # Load models
        self._load_models()

        # Initialize feature pipeline
        self._init_feature_pipeline()

    def _load_models(self) -> None:
        """Load MPNN and optional packer models to GPU."""
        LOGGER.info(f"Loading {self.model_type} model...")

        # Get checkpoint path
        self.checkpoint_path = self.CHECKPOINT_PATHS.get(self.model_type)
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found for {self.model_type}: {self.checkpoint_path}"
            )

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Model configuration based on type (mirrors run.py lines 107-118)
        if self.model_type == "ligand_mpnn":
            atom_context_num = 25
            k_neighbors = 32
            ligand_mpnn_use_side_chain_context = 1
        elif self.model_type in ("antibody_mpnn", "msa_mpnn"):
            atom_context_num = 1
            ligand_mpnn_use_side_chain_context = 0
            k_neighbors = 48
        else:
            atom_context_num = 1
            ligand_mpnn_use_side_chain_context = 0
            k_neighbors = checkpoint.get("num_edges", 48)

        # Initialize model (mirrors run.py lines 121-130)
        self.model = ProteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=k_neighbors,
            device=self.device,
            atom_context_num=atom_context_num,
            model_type=self.model_type,
            ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
        )

        # Load weights
        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint['model_state_dict'], strict=False
        )
        if missing_keys:
            LOGGER.warning(f"Missing keys in checkpoint: {missing_keys}")

        self.model.to(self.device)
        self.model.eval()
        LOGGER.info(f"Loaded {self.model_type} model to {self.device}")

        # Load side-chain packer if enabled (mirrors run.py lines 144-167)
        if self.enable_packing:
            LOGGER.info("Loading side-chain packer model...")
            self.model_sc = Packer(
                node_features=128,
                edge_features=128,
                num_positional_embeddings=16,
                num_chain_embeddings=16,
                num_rbf=16,
                hidden_dim=128,
                num_encoder_layers=3,
                num_decoder_layers=3,
                atom_context_num=16,
                lower_bound=0.0,
                upper_bound=20.0,
                top_k=32,
                dropout=0.0,
                augment_eps=0.0,
                atom37_order=False,
                device=self.device,
                num_mix=3,
            )
            checkpoint_sc = torch.load(self.CHECKPOINT_SC, map_location=self.device)
            self.model_sc.load_state_dict(checkpoint_sc['model_state_dict'])
            self.model_sc.to(self.device)
            self.model_sc.eval()
            LOGGER.info("Loaded packer model")

        # Log GPU memory usage
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated(self.device) / 1024**2
            LOGGER.info(f"GPU memory used after model loading: {mem_used:.0f} MB")

    def _init_feature_pipeline(self) -> None:
        """Initialize the feature extraction pipeline (mirrors run.py lines 258-271)."""
        self.loader = Compose([
            LoadFile(),
            GetAssembly(),
            ExcludeByResName(exclude_res_list=["HOH", "NA", "CL", "K", "BR", "UNX"]),
            GetChainInfo(),
            SeperateProteinLigand(
                protein_backbone_occ_cutoff=0.,
                protein_side_chain_occ_cutoff=0.,
                ligand_occ_cutoff=0.,
            ),
            BuildMPNNInput(lig_nn_num=25),
        ])

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a single request."""
        request_id = request.get("request_id", str(uuid.uuid4()))
        request_type = request.get("type", "design")

        try:
            if request_type == "health":
                return self._handle_health(request_id)
            elif request_type == "shutdown":
                self._shutdown_event.set()
                return {"status": "success", "request_id": request_id, "message": "Shutdown initiated"}
            elif request_type == "design":
                return self._handle_design(request, request_id)
            else:
                return {
                    "status": "error",
                    "request_id": request_id,
                    "error_type": "ValueError",
                    "error_message": f"Unknown request type: {request_type}",
                }
        except Exception as e:
            LOGGER.error(f"Request {request_id} failed: {e}")
            return {
                "status": "error",
                "request_id": request_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            }

    def _handle_health(self, request_id: str) -> Dict[str, Any]:
        """Return server health status."""
        gpu_mem_used = 0
        gpu_mem_total = 0
        if torch.cuda.is_available():
            gpu_mem_used = torch.cuda.memory_allocated(self.device) / 1024**2
            gpu_mem_total = torch.cuda.get_device_properties(self.device).total_memory / 1024**2

        return {
            "status": "success",
            "request_id": request_id,
            "type": "health",
            "gpu_memory_used_mb": int(gpu_mem_used),
            "gpu_memory_total_mb": int(gpu_mem_total),
            "model_loaded": self.model is not None,
            "packer_loaded": self.model_sc is not None,
            "model_type": self.model_type,
            "requests_served": self.requests_served,
            "uptime_seconds": int(time.time() - self.start_time),
        }

    def _handle_design(self, request: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Handle MPNN design request - mirrors run.py main() logic."""
        start_time = time.time()

        # Extract parameters
        pdb_path = request["pdb_path"]
        out_folder = request["out_folder"]
        temperature = request.get("temperature", 0.1)
        batch_size = request.get("batch_size", 1)
        number_of_batches = request.get("number_of_batches", 8)
        fixed_residues = request.get("fixed_residues", [])
        redesigned_residues = request.get("redesigned_residues", [])
        omit_aa = request.get("omit_aa", "")
        bias_aa_dict = request.get("bias_aa", {})
        bias_aa_per_residue_dict = request.get("bias_aa_per_residue", {})
        use_sc_context = request.get("use_sc_context", 1)
        do_pack_side_chains = request.get("pack_side_chains", 1)
        repack_everything = request.get("repack_everything", 0)
        sc_denoising_steps = request.get("sc_denoising_steps", 3)
        seed = request.get("seed", 0)
        save_stats = request.get("save_stats", 1)
        verbose = request.get("verbose", 1)
        enhance = request.get("enhance", None)

        # Handle enhance checkpoint override
        checkpoint_path = self.checkpoint_path
        if enhance and self.model_type == "ligand_mpnn":
            enhancer = enhance.lower()
            if enhancer.startswith("plddt"):
                checkpoint_path = f'/net/scratch/faxue/share/mpnn_enhancer/{enhancer}.pth'
                # Re-load model with enhanced weights
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    LOGGER.info(f"Loaded enhanced model: {enhancer}")

        # Set seed
        if not seed:
            seed = int(np.random.randint(0, high=99999, size=1, dtype=int)[0])
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Create output directories
        base_folder = Path(out_folder)
        base_folder.mkdir(parents=True, exist_ok=True)
        (base_folder / "seqs").mkdir(exist_ok=True)
        (base_folder / "backbones").mkdir(exist_ok=True)
        if do_pack_side_chains:
            (base_folder / "packed").mkdir(exist_ok=True)
        if save_stats:
            (base_folder / "stats").mkdir(exist_ok=True)

        # Load PDB and extract features
        if verbose:
            LOGGER.info(f"Designing PDB: {pdb_path}")

        protein_dict = self.loader(dict(cif_path=pdb_path, assembly_id=None))

        # Build residue encoding (mirrors run.py lines 299-307)
        icodes = protein_dict["ins_codes"]
        R_idx_list = protein_dict["R_idx"]
        chain_letters_list = list(protein_dict["chain_letters"])
        encoded_residues = []
        for i in range(len(R_idx_list)):
            tmp = str(chain_letters_list[i]) + str(R_idx_list[i]) + icodes[i]
            encoded_residues.append(tmp)
        encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
        encoded_residue_dict_rev = dict(zip(list(range(len(encoded_residues))), encoded_residues))

        # Build amino acid bias tensor (mirrors run.py lines 209-217)
        bias_AA = torch.zeros([21], device=self.device, dtype=torch.float32)
        if bias_aa_dict:
            for aa, val in bias_aa_dict.items():
                if aa in self.RESTYPE_STR_TO_INT:
                    bias_AA[self.RESTYPE_STR_TO_INT[aa]] = val

        # Build per-residue bias tensor (mirrors run.py lines 309-318)
        bias_AA_per_residue = torch.zeros([len(encoded_residues), 21], device=self.device, dtype=torch.float32)
        if bias_aa_per_residue_dict:
            for residue_name, v1 in bias_aa_per_residue_dict.items():
                if residue_name in encoded_residue_dict:
                    i1 = encoded_residue_dict[residue_name]
                    for amino_acid, v2 in v1.items():
                        if amino_acid in self.ALPHABET:
                            j1 = self.RESTYPE_STR_TO_INT[amino_acid]
                            bias_AA_per_residue[i1, j1] = v2

        # Build omit_AA tensor (mirrors run.py lines 253-256)
        omit_AA = torch.tensor(
            np.array([AA in omit_aa for AA in self.ALPHABET]).astype(np.float32),
            device=self.device
        )

        # Build fixed/redesigned positions (mirrors run.py lines 334-336)
        fixed_positions = torch.tensor(
            [int(item not in fixed_residues) for item in encoded_residues],
            device=self.device
        )
        redesigned_positions = torch.tensor(
            [int(item not in redesigned_residues) for item in encoded_residues],
            device=self.device
        )

        # Create chain mask (mirrors run.py lines 368-382)
        chain_mask = torch.ones(len(encoded_residues), device=self.device, dtype=torch.int32)
        if redesigned_residues:
            protein_dict["chain_mask"] = chain_mask * (1 - redesigned_positions)
        elif fixed_residues:
            protein_dict["chain_mask"] = chain_mask * fixed_positions
        else:
            protein_dict["chain_mask"] = chain_mask

        protein_dict["side_chain_mask"] = protein_dict["chain_mask"].clone()

        # Membrane labels (default zeros)
        protein_dict["membrane_per_residue_labels"] = torch.zeros_like(fixed_positions)
        protein_dict["pssm"] = torch.zeros([fixed_positions.shape[0], 20], device=self.device)

        # Build feature dict (mirrors run.py lines 449-485)
        with torch.no_grad():
            feature_dict = dict()
            for k, v in protein_dict.items():
                if isinstance(v, np.ndarray):
                    if not np.issubdtype(v.dtype, np.str_):
                        v = torch.from_numpy(v)[None, ]
                elif isinstance(v, torch.Tensor):
                    v = v[None, ]
                feature_dict[k] = v

            # Move to device
            for k, v in feature_dict.items():
                if isinstance(v, torch.Tensor):
                    feature_dict[k] = v.to(self.device)

            feature_dict['xyz_37'] = feature_dict['X']
            feature_dict['xyz_37_m'] = feature_dict['X_m']
            feature_dict['X'] = feature_dict['X'][:, :, :4]
            feature_dict['X_m'] = feature_dict['X_m'][:, :, :4]

            # Chain mask for FASTA output
            sorted_chain_letters = sorted(set(protein_dict["chain_letters"]))
            mask_c = np.array([protein_dict['chain_letters'] == c for c in sorted_chain_letters])
            mask_c = torch.tensor(mask_c, dtype=bool)

            feature_dict["batch_size"] = batch_size
            B, L, _, _ = feature_dict["X"].shape

            # Add additional keys
            feature_dict["temperature"] = temperature
            omit_AA_per_residue = torch.zeros([L, 21], device=self.device, dtype=torch.float32)
            feature_dict["bias"] = (
                (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1]) +
                bias_AA_per_residue[None] -
                1e8 * omit_AA_per_residue[None]
            )
            feature_dict["symmetry_residues"] = [[]]
            feature_dict["symmetry_weights"] = [[]]

            # Sampling loop (mirrors run.py lines 494-523)
            sampling_probs_list = []
            log_probs_list = []
            decoding_order_list = []
            S_list = []
            loss_list = []
            loss_per_residue_list = []
            loss_XY_list = []

            for _ in range(number_of_batches):
                feature_dict["randn"] = torch.randn(
                    [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
                    device=self.device
                )
                # Main inference
                output_dict = self.model.sample(feature_dict)

                # Compute confidence scores
                loss, loss_per_residue = get_score(
                    output_dict["S"],
                    output_dict["log_probs"],
                    feature_dict["mask"] * feature_dict["chain_mask"]
                )
                if self.model_type == "ligand_mpnn":
                    combined_mask = feature_dict["mask"] * feature_dict["mask_XY"] * feature_dict["chain_mask"]
                else:
                    combined_mask = feature_dict["mask"] * feature_dict["chain_mask"]
                loss_XY, _ = get_score(output_dict["S"], output_dict["log_probs"], combined_mask)

                S_list.append(output_dict["S"])
                log_probs_list.append(output_dict["log_probs"])
                sampling_probs_list.append(output_dict["sampling_probs"])
                decoding_order_list.append(output_dict["decoding_order"])
                loss_list.append(loss)
                loss_per_residue_list.append(loss_per_residue)
                loss_XY_list.append(loss_XY)

            S_stack = torch.cat(S_list, 0)
            log_probs_stack = torch.cat(log_probs_list, 0)
            sampling_probs_stack = torch.cat(sampling_probs_list, 0)
            decoding_order_stack = torch.cat(decoding_order_list, 0)
            loss_stack = torch.cat(loss_list, 0)
            loss_per_residue_stack = torch.cat(loss_per_residue_list, 0)
            loss_XY_stack = torch.cat(loss_XY_list, 0)
            rec_mask = feature_dict["mask"][:1] * feature_dict["chain_mask"][:1]
            rec_stack = get_seq_rec(feature_dict["S"][:1], S_stack, rec_mask)

            # Side chain packing (mirrors run.py lines 528-719)
            X_stack_list = []
            X_m_stack_list = []
            b_factor_stack_list = []

            if do_pack_side_chains and self.model_sc is not None:
                sc_feature_dict = copy.deepcopy(feature_dict)
                B_pack = batch_size
                for k, v in sc_feature_dict.items():
                    if k != "S":
                        try:
                            num_dim = len(v.shape)
                            if num_dim == 2:
                                sc_feature_dict[k] = v.repeat(B_pack, 1)
                            elif num_dim == 3:
                                sc_feature_dict[k] = v.repeat(B_pack, 1, 1)
                            elif num_dim == 4:
                                sc_feature_dict[k] = v.repeat(B_pack, 1, 1, 1)
                            elif num_dim == 5:
                                sc_feature_dict[k] = v.repeat(B_pack, 1, 1, 1, 1)
                        except:
                            pass

                # Single pack per design
                X_list = []
                X_m_list = []
                b_factor_list = []
                for c in range(number_of_batches):
                    sc_feature_dict["S"] = S_list[c]

                    # Freeze fixed residues during packing
                    if fixed_residues:
                        freeze_idxs = []
                        for tok in fixed_residues:
                            tok = str(tok).strip()
                            if tok in encoded_residue_dict:
                                freeze_idxs.append(encoded_residue_dict[tok])
                        if freeze_idxs:
                            sc_feature_dict["chain_mask"][:, freeze_idxs] = 0
                            if "side_chain_mask" in sc_feature_dict:
                                sc_feature_dict["side_chain_mask"][:, freeze_idxs] = 0

                    sc_dict = pack_side_chains(
                        sc_feature_dict,
                        self.model_sc,
                        sc_denoising_steps,
                        16,  # sc_num_samples
                        repack_everything
                    )
                    X_list.append(sc_dict["X"])
                    X_m_list.append(sc_dict["X_m"])
                    b_factor_list.append(sc_dict["b_factors"])

                X_stack = torch.cat(X_list, 0)
                X_m_stack = torch.cat(X_m_list, 0)
                b_factor_stack = torch.cat(b_factor_list, 0)
                X_stack_list.append(X_stack)
                X_m_stack_list.append(X_m_stack)
                b_factor_stack_list.append(b_factor_stack)

            # Write outputs (mirrors run.py lines 723-795)
            name = os.path.basename(pdb_path)
            if name.endswith(".pdb"):
                name = name[:-4]

            # Native sequence
            native_seq = "".join([self.RESTYPE_INT_TO_STR[AA] for AA in feature_dict["S"][0].cpu().numpy()])
            seq_np = np.array(list(native_seq))
            seq_out_str = []
            for mask in mask_c:
                seq_out_str += list(seq_np[mask.cpu().numpy()])
                seq_out_str += ['/']
            seq_out_str = "".join(seq_out_str)[:-1]

            # Write FASTA
            output_fasta = str(base_folder / "seqs" / f"{name}.fa")
            with open(output_fasta, 'w') as f:
                f.write(
                    f'>{name}, T={temperature}, seed={seed}, num_res={torch.sum(rec_mask).cpu().numpy()}, '
                    f'num_ligand_res={torch.sum(combined_mask[:1]).cpu().numpy()}, '
                    f'use_ligand_context={use_sc_context}, batch_size={batch_size}, '
                    f'number_of_batches={number_of_batches}, model_path={checkpoint_path}\n{seq_out_str}\n'
                )
                for ix in range(S_stack.shape[0]):
                    ix_suffix = ix + 1
                    seq_rec_print = np.format_float_positional(rec_stack[ix].cpu().numpy(), unique=False, precision=4)
                    loss_np = np.format_float_positional(np.exp(-loss_stack[ix].cpu().numpy()), unique=False, precision=4)
                    loss_XY_np = np.format_float_positional(np.exp(-loss_XY_stack[ix].cpu().numpy()), unique=False, precision=4)
                    seq = "".join([self.RESTYPE_INT_TO_STR[AA] for AA in S_stack[ix].cpu().numpy()])

                    # Format with chain separators
                    seq_np_i = np.array(list(seq))
                    seq_out_str_i = []
                    for mask in mask_c:
                        seq_out_str_i += list(seq_np_i[mask.cpu().numpy()])
                        seq_out_str_i += ['/']
                    seq_out_str_i = "".join(seq_out_str_i)[:-1]

                    f.write(
                        f'>{name}, id={ix_suffix}, T={temperature}, seed={seed}, '
                        f'overall_confidence={loss_np}, ligand_confidence={loss_XY_np}, '
                        f'seq_rec={seq_rec_print}\n{seq_out_str_i}\n'
                    )

                    # Write packed PDBs
                    if do_pack_side_chains and X_stack_list:
                        X_stack_out = X_stack_list[0]
                        X_m_stack_out = X_m_stack_list[0]
                        b_factor_stack_out = b_factor_stack_list[0]

                        save_path = str(base_folder / "packed" / f"{name}_packed_{ix_suffix}_1.pdb")
                        save_output_structure(
                            save_path,
                            X_stack_out[ix].cpu().numpy(),
                            X_m_stack_out[ix].cpu().numpy(),
                            b_factor_stack_out[ix].cpu().numpy(),
                            protein_dict["R_idx"],
                            protein_dict["chain_letters"],
                            S_stack[ix].cpu().numpy(),
                            other_atoms=protein_dict.get("other_atoms"),
                            icodes=icodes,
                            force_hetatm=1
                        )

            # Save stats
            if save_stats:
                output_stats_path = str(base_folder / "stats" / f"{name}.pt")
                out_dict = {
                    "generated_sequences": S_stack.cpu(),
                    "sampling_probs": sampling_probs_stack.cpu(),
                    "log_probs": log_probs_stack.cpu(),
                    "decoding_order": decoding_order_stack.cpu(),
                    "native_sequence": feature_dict["S"][0].cpu(),
                    "mask": feature_dict["mask"][0].cpu(),
                    "chain_mask": feature_dict["chain_mask"][0].cpu(),
                    "seed": seed,
                    "temperature": temperature,
                }
                torch.save(out_dict, output_stats_path)

        # Update statistics
        with self._lock:
            self.requests_served += 1

        elapsed_time = time.time() - start_time
        LOGGER.info(f"Design completed in {elapsed_time:.2f}s, generated {S_stack.shape[0]} sequences")

        # Parse and return results
        return self._build_response(request_id, base_folder, elapsed_time)

    def _build_response(
        self,
        request_id: str,
        out_folder: Path,
        elapsed_time: float
    ) -> Dict[str, Any]:
        """Build response from generated outputs."""
        result = {
            "status": "success",
            "request_id": request_id,
            "sequences": [],
            "seq_files": [],
            "backbone_files": [],
            "packed_files": [],
            "stats_files": [],
            "recovery_scores": [],
            "mpnn_scores": [],
            "elapsed_time": elapsed_time,
        }

        # Parse FASTA files
        seqs_dir = out_folder / "seqs"
        for fasta_file in sorted(seqs_dir.glob("*.fa")):
            result["seq_files"].append(str(fasta_file))
            result["sequences"].extend(self._parse_fasta(fasta_file))

        # Parse packed PDBs
        packed_dir = out_folder / "packed"
        if packed_dir.exists():
            for pdb in sorted(packed_dir.glob("*.pdb")):
                result["packed_files"].append(str(pdb))

        # Parse stats
        stats_dir = out_folder / "stats"
        if stats_dir.exists():
            for stats_file in sorted(stats_dir.glob("*.pt")):
                result["stats_files"].append(str(stats_file))

        return result

    def _parse_fasta(self, fasta_path: Path) -> List[str]:
        """Parse sequences from FASTA, skipping reference sequence."""
        sequences = []
        current_header = ""
        current_seq = []

        with open(fasta_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_header and current_seq:
                        if not self._is_reference_sequence(current_header):
                            sequences.append("".join(current_seq))
                    current_header = line
                    current_seq = []
                else:
                    current_seq.append(line)

            if current_header and current_seq:
                if not self._is_reference_sequence(current_header):
                    sequences.append("".join(current_seq))

        return sequences

    def _is_reference_sequence(self, header: str) -> bool:
        """Check if FASTA header indicates reference/wildtype sequence."""
        header_lower = header.lower()
        return (
            "sample_0" in header_lower or
            "sample 0" in header_lower or
            ", sample=0" in header_lower or
            "_0," in header or
            header.rstrip().endswith("_0") or
            ", id=" not in header_lower  # First entry has no id=
        )

    def serve(self) -> None:
        """Start the server and listen for connections."""
        # Set up signal handlers
        signal.signal(signal.SIGTERM, lambda s, f: self._shutdown_event.set())
        signal.signal(signal.SIGINT, lambda s, f: self._shutdown_event.set())

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            server_socket.settimeout(1.0)  # Check shutdown every second

            LOGGER.info(f"MPNN Server listening on {self.host}:{self.port}")
            LOGGER.info(f"Model: {self.model_type}, Packer: {self.enable_packing}")

            while not self._shutdown_event.is_set():
                try:
                    client_socket, addr = server_socket.accept()
                    LOGGER.info(f"Connection from {addr}")
                    # Handle synchronously (GPU can only handle one at a time)
                    self._handle_client(client_socket)
                except socket.timeout:
                    continue
                except Exception as e:
                    LOGGER.error(f"Accept error: {e}")

        LOGGER.info("Server shutdown complete")

    def _handle_client(self, client_socket: socket.socket) -> None:
        """Handle a single client connection."""
        try:
            with client_socket:
                client_socket.settimeout(600.0)  # 10 minute timeout

                # Read length-prefixed message
                length_bytes = self._recv_exact(client_socket, 4)
                if not length_bytes:
                    return
                msg_length = struct.unpack(">I", length_bytes)[0]

                # Read message
                msg_bytes = self._recv_exact(client_socket, msg_length)
                if not msg_bytes:
                    return

                request = json.loads(msg_bytes.decode("utf-8"))
                LOGGER.info(f"Received request: type={request.get('type', 'design')}")

                # Handle request
                response = self.handle_request(request)

                # Send response
                response_bytes = json.dumps(response).encode("utf-8")
                client_socket.sendall(struct.pack(">I", len(response_bytes)))
                client_socket.sendall(response_bytes)

        except Exception as e:
            LOGGER.error(f"Client handling error: {e}")
            traceback.print_exc()

    def _recv_exact(self, sock: socket.socket, n: int) -> Optional[bytes]:
        """Receive exactly n bytes from socket."""
        data = b""
        while len(data) < n:
            try:
                chunk = sock.recv(n - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                return None
        return data


def main():
    parser = argparse.ArgumentParser(
        description="MPNN Server - Persistent model for fast sequence design",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--host", default="localhost", help="Server hostname")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument(
        "--model_type", default="ligand_mpnn",
        choices=list(MPNNServer.CHECKPOINT_PATHS.keys()),
        help="MPNN model type"
    )
    parser.add_argument(
        "--pack_side_chains", action="store_true", default=True,
        help="Load side-chain packer model"
    )
    parser.add_argument(
        "--no_pack_side_chains", action="store_true",
        help="Don't load side-chain packer model"
    )

    args = parser.parse_args()
    pack = args.pack_side_chains and not args.no_pack_side_chains

    server = MPNNServer(
        host=args.host,
        port=args.port,
        model_type=args.model_type,
        pack_side_chains=pack,
    )
    server.serve()


if __name__ == "__main__":
    main()
