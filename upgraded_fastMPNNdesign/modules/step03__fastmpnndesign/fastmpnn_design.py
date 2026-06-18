#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step03: FastMPNN Design with Rosetta Refinement

Main orchestrator for iterative MPNN sequence design with Rosetta relaxation.

This module:
1. Takes step02 output (relaxed PDB + metrics JSON) as input
2. Classifies residues into design spheres (primary, secondary, etc.)
3. Runs iterative MPNN design + Rosetta relaxation
4. Tracks metrics throughout the protocol
5. Optionally conserves favorable interactions with catalytic residues (protocol-driven)
6. Outputs optimized enzyme variants

Usage:
    python fastmpnn_design.py --step02_json <path> --params <file> [options]

Example:
    # Uses protocols/default.json by default
    python fastmpnn_design.py \\
        --step02_json step02_outputs/relaxed_metrics.json \\
        --params params/LIG.params \\
        --output_dir output/

    # Use a different protocol from protocols/ directory
    python fastmpnn_design.py \\
        --step02_json step02_outputs/relaxed_metrics.json \\
        --params params/LIG.params \\
        --protocol my_custom_protocol \\
        --output_dir output/

    # Use an explicit protocol file path (outside protocols/ directory)
    python fastmpnn_design.py \\
        --step02_json step02_outputs/relaxed_metrics.json \\
        --params params/LIG.params \\
        --protocol_file /path/to/my_protocol.json \\
        --output_dir output/

    # Protocol text file (.txt) using compact syntax
    python fastmpnn_design.py \\
        --step02_json step02_outputs/relaxed_metrics.json \\
        --params params/LIG.params \\
        --protocol_file /path/to/my_protocol.txt \\
        --output_dir output/

Verbosity levels:
    --quiet   : Minimal output (errors and final summary only)
    (default) : Moderate output (step progress, key metrics)
    --verbose : Detailed output (timing, all metrics, debug info)
"""
import argparse
import hashlib
from collections import defaultdict, Counter
from dataclasses import dataclass, field, replace
import threading
import uuid
import json
import logging
import math
import os
import subprocess
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add module_utils to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.dirname(SCRIPT_DIR)
PROTOCOLS_DIR = os.path.join(SCRIPT_DIR, "protocols")
DEFAULT_PROTOCOL = "default"  # Basename without .json extension
MAX_FILENAME_LEN = 200  # conservative max to avoid filesystem/MPNN path issues
sys.path.insert(0, MODULE_DIR)

# Local imports
from .protocol_parser import ProtocolParser, ProtocolFileParser, ProtocolValidationError, ProtocolStep
from .protocol_parser import (
    MPNNStep,
    MPNNMultiStep,
    CartRelaxStep,
    TorsionalRelaxStep,
    MinimizeStep,
    RepackStep,
    SelectBestStep,
    ScaleScoreTermStep,
    SetOptionsStep,
    KeepInteractionsStep,
    ClusterStep,
    KeepClusterBestStep,
    TaskOperationStep,
    TimeCheckStep,
    FinalDiversifyStep,
    StepType,
)
from .residue_classifier import ResidueClassifier, DesignSphere
from .pdb_restoration import (
    full_mpnn_output_restoration,
    normalize_pdb_for_mpnn,
    restore_ligand_from_ref,
    build_his_tautomer_map,
    cleanup_final_pdb,
)
from .mpnn_runner import MPNNRunner, MPNNInput, MPNNResult, create_mpnn_input_from_classifier
from .interaction_analyzer import InteractionAnalyzer, MutationInteraction
from .metrics import MetricsCalculator, round_metrics

from module_utils.pdb_utils import parse_remark_666, read_pdb_atoms


def in_container() -> bool:
    """Detect if we're running inside a Singularity/Apptainer container."""
    return (
        os.path.exists('/.singularity.d') or
        'APPTAINER_CONTAINER' in os.environ or
        'SINGULARITY_CONTAINER' in os.environ
    )
from module_utils.sequence_utils import (
    get_sequence_from_atoms,
    remove_duplicate_sequences,
    calculate_sequence_identity,
)
from module_utils.constants import (
    DEFAULT_MPNN_TEMPERATURE,
    DEFAULT_MPNN_NUMBER_OF_BATCHES,
    DEFAULT_MPNN_BATCH_SIZE,
    DEFAULT_OMIT_AA,
    DEFAULT_LAYER_CUTS,
    DEFAULT_COORD_CST_WEIGHT,
    DEFAULT_COORD_CST_STDEV,
    DEFAULT_BOND_LENGTH_TOLERANCE,
    DEFAULT_BOND_ANGLE_TOLERANCE,
    DEFAULT_APPTAINER_IMAGE,
    DEFAULT_PYROSETTA_IMAGE,
    SCOREFUNCTION_CART,
    SCOREFUNCTION_TORSIONAL,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def log_step_header(step_num: int, step_type: str, total_steps: int = None) -> None:
    """Log a formatted step header."""
    if total_steps:
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"Step {step_num}/{total_steps}: {step_type}")
        LOGGER.info(f"{'='*60}")
    else:
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"Step {step_num}: {step_type}")
        LOGGER.info(f"{'='*60}")


class FastMPNNDesigner:
    """Orchestrator for iterative MPNN design with Rosetta refinement.

    Implements a phased protocol:
    1. Geometry optimization phase (cartesian relax until converged)
    2. Design phase (iterative MPNN + torsional relax)
    3. Optimization phase (torsional relax for scoring)
    """

    def __init__(
        self,
        step02_json_path: str,
        params_files: List[str],
        output_dir: str,
        protocol: Optional[str] = None,  # Basename in protocols/ dir (default: "default")
        protocol_file: Optional[str] = None,  # Explicit file path (.json/.txt)
        # Design scope
        catres_subset: Optional[str] = None,
        design_secondary_sphere: bool = False,
        design_gly_pro: bool = False,
        layer_cuts: Optional[List[float]] = None,
        mpnn_spheres: Optional[str] = None,
        # MPNN settings
        mpnn_temperature: float = DEFAULT_MPNN_TEMPERATURE,
        mpnn_num_designs: int = DEFAULT_MPNN_NUMBER_OF_BATCHES,
        mpnn_num_designs_after_first: Optional[int] = None,
        mpnn_batch_size: int = DEFAULT_MPNN_BATCH_SIZE,
        mpnn_omit_aa: str = DEFAULT_OMIT_AA,
        # Rosetta settings
        coord_cst_weight: float = DEFAULT_COORD_CST_WEIGHT,
        coord_cst_stdev: float = DEFAULT_COORD_CST_STDEV,
        global_coord_cst_weight: float = 0.0,
        global_coord_cst_stdev: float = 0.5,
        scorefunction_cart: str = SCOREFUNCTION_CART,
        scorefunction_torsional: str = SCOREFUNCTION_TORSIONAL,
        fa_rep_weight: Optional[float] = None,
        cart_bonded_weight: float = 2.0,
        relax_rounds: int = 5,
        relax_inner_cycles: Optional[int] = None,
        # Convergence
        bond_length_tolerance: float = DEFAULT_BOND_LENGTH_TOLERANCE,
        bond_angle_tolerance: float = DEFAULT_BOND_ANGLE_TOLERANCE,
        # References
        step01_pdb: Optional[str] = None,
        # Protocol options
        skip_initial_cart_relax: bool = False,
        # Interaction conservation
        conserve_favorable_interactions: bool = False,
        conservation_probability: float = 0.5,
        # Constraint options
        include_bb_hbond_constraints: bool = False,
        # Output
        num_final_designs: Optional[int] = None,
        # Runtime
        max_runtime: int = 7200,
        rosetta_timeout: int = 7200,
        cart_relax_max_rounds: int = 5,
        pyrosetta_image: str = DEFAULT_PYROSETTA_IMAGE,
        no_container: bool = False,
        rosetta_in_process: bool = False,
        pyrosetta_dir: Optional[str] = None,
        debug: bool = False,
        test: bool = False,
        keep_intermediates: bool = False,
        # MPNN execution settings
        mpnn_use_container: Optional[bool] = None,
        mpnn_use_gpu: Optional[bool] = None,
        mpnn_container_image: Optional[str] = None,
        # MPNN Server settings
        use_mpnn_server: bool = True,
        mpnn_server_host: str = "localhost",
        mpnn_server_port: int = 5000,
        auto_start_mpnn_server: bool = True,
    ):
        """Initialize FastMPNN designer.

        Args:
            step02_json_path: Path to step02 output JSON
            params_files: List of ligand .params files
            output_dir: Output directory
            protocol: Protocol name (basename of JSON file in protocols/)
            protocol_file: Path to protocol file (.json or .txt) - overrides protocol if provided
            catres_subset: Override catalytic residue subset
            design_secondary_sphere: Include secondary sphere in design
            design_gly_pro: Allow GLY/PRO redesign
            layer_cuts: Distance cutoffs for sphere classification
            mpnn_spheres: Override MPNN design spheres globally (comma-separated, e.g. "primary,secondary" or "global")
            mpnn_temperature: MPNN sampling temperature
            mpnn_num_designs: Number of MPNN designs per round
            mpnn_num_designs_after_first: Override num designs after first MPNN step
            mpnn_batch_size: MPNN batch size (1 for max diversity)
            mpnn_omit_aa: Amino acids to never design
            coord_cst_weight: Coordinate constraint weight for catalytic residues
            coord_cst_stdev: Coordinate constraint stdev for catalytic residues
            global_coord_cst_weight: Coordinate constraint weight for ALL protein atoms (0 = disabled)
            global_coord_cst_stdev: Stdev for global constraints (looser than catres)
            scorefunction_cart: Scorefunction for cartesian relax
            scorefunction_torsional: Scorefunction for torsional relax
            fa_rep_weight: Override fa_rep weight (default 0.55, try 0.3-1.0)
            relax_rounds: Number of FastRelax rounds/outer cycles (default 5)
            relax_inner_cycles: Number of inner cycles (default varies by mode)
            bond_length_tolerance: Bond length convergence tolerance
            bond_angle_tolerance: Bond angle convergence tolerance
            step01_pdb: Original step01 PDB for RMSD tracking
            skip_initial_cart_relax: Skip first cart_relax if input is from step02
                                     (step02 already did cartesian relaxation)
            conserve_favorable_interactions: Enable interaction conservation (default off)
            conservation_probability: Probability to conserve favorable mutations (used only if enabled)
            include_bb_hbond_constraints: Include backbone atoms (N, CA, C, O, H) in constraints
                                         for residues with backbone_important_only_for_BB_BB_hbond=True
                                         (default: False, excludes these atoms)
            num_final_designs: Number of final designs to output
            max_runtime: Maximum runtime in seconds
            rosetta_timeout: Timeout per Rosetta subprocess (seconds)
            cart_relax_max_rounds: Max outer rounds when until_converged=True
            pyrosetta_image: Path to PyRosetta apptainer image (pyrosetta.sif)
            no_container: Run commands directly without apptainer wrapper
                         (use when already inside a container or when deps are available)
            rosetta_in_process: Run PyRosetta in-process (no container) if available
            pyrosetta_dir: Override host PyRosetta install path (fallback for in-process)
            debug: Enable debug output
            test: Run in test mode (faster)
            keep_intermediates: Keep intermediate files/directories (default: clean up)
            mpnn_use_container: Force MPNN to use container runtime (None = auto)
            mpnn_use_gpu: Force MPNN to use GPU (None = auto-detect)
        """
        self.step02_json_path = step02_json_path
        self.params_files = params_files
        self.output_dir = Path(output_dir).resolve()  # Use absolute path to avoid nesting issues
        self.protocol_str = protocol
        self.protocol_file = protocol_file
        self.catres_subset_override = catres_subset
        self.design_secondary_sphere = design_secondary_sphere
        self.design_gly_pro = design_gly_pro
        self.layer_cuts = layer_cuts or DEFAULT_LAYER_CUTS
        self.mpnn_spheres_override = mpnn_spheres
        self.mpnn_temperature = mpnn_temperature
        self.mpnn_num_designs = mpnn_num_designs
        self.mpnn_num_designs_after_first = mpnn_num_designs_after_first
        self.mpnn_batch_size = mpnn_batch_size
        self.mpnn_omit_aa = mpnn_omit_aa
        self.coord_cst_weight = coord_cst_weight
        self.coord_cst_stdev = coord_cst_stdev
        self.global_coord_cst_weight = global_coord_cst_weight
        self.global_coord_cst_stdev = global_coord_cst_stdev
        self.scorefunction_cart = scorefunction_cart
        self.scorefunction_torsional = scorefunction_torsional
        self.fa_rep_weight = fa_rep_weight
        self.cart_bonded_weight = cart_bonded_weight
        self.relax_rounds = relax_rounds
        self.relax_inner_cycles = relax_inner_cycles
        self.bond_length_tolerance = bond_length_tolerance
        self.bond_angle_tolerance = bond_angle_tolerance
        self.step01_pdb = step01_pdb
        self.skip_initial_cart_relax = skip_initial_cart_relax
        self.conserve_favorable_interactions = conserve_favorable_interactions
        self.conservation_probability = conservation_probability
        self.include_bb_hbond_constraints = include_bb_hbond_constraints
        self.max_runtime = max_runtime
        self.rosetta_timeout = rosetta_timeout
        self.cart_relax_max_rounds = cart_relax_max_rounds
        self.pyrosetta_image = pyrosetta_image
        self.no_container = no_container
        self.rosetta_in_process = rosetta_in_process
        self.pyrosetta_dir = pyrosetta_dir
        self.debug = debug

        # Auto-detect container environment and runtime availability
        runtime_available = bool(shutil.which("apptainer") or shutil.which("singularity"))
        if in_container() and not runtime_available:
            if not self.no_container:
                LOGGER.warning(
                    "Detected container environment without apptainer/singularity. "
                    "Forcing --no-container mode for in-process execution."
                )
                self.no_container = True
        elif not self.no_container and in_container():
            LOGGER.warning(
                "Detected container environment. Consider using --no-container "
                "if PyRosetta and MPNN are available in this container."
            )
        self.test = test
        self.keep_intermediates = keep_intermediates

        # MPNN Server settings
        self.use_mpnn_server = use_mpnn_server
        self.mpnn_server_host = mpnn_server_host
        self.mpnn_server_port = mpnn_server_port
        self.auto_start_mpnn_server = auto_start_mpnn_server
        # MPNN container/GPU settings (resolved below)
        self.mpnn_use_container = None
        self.mpnn_use_gpu = mpnn_use_gpu
        self.mpnn_container_image = mpnn_container_image or DEFAULT_APPTAINER_IMAGE
        self.num_final_designs = num_final_designs
        self.num_final_designs_explicit = num_final_designs is not None
        self.protocol_target_count: Optional[int] = None

        # Resolve MPNN container usage
        if mpnn_use_container is None:
            self.mpnn_use_container = not self.no_container
        else:
            self.mpnn_use_container = mpnn_use_container
        if self.mpnn_use_container and not runtime_available:
            LOGGER.warning(
                "MPNN container requested but apptainer/singularity not found. "
                "Falling back to no-container execution for MPNN."
            )
            self.mpnn_use_container = False

        # Runtime state
        self._mpnn_has_run = False
        self._score_term_overrides: Dict[str, float] = {}
        self._score_term_overrides_next: Optional[Dict[str, float]] = None
        self._score_term_overrides_next_reset: Optional[set] = None
        self._mpnn_call_index: int = 0
        self._pdb_lineage: Dict[str, str] = {}
        self._pdb_lineage_lock = threading.Lock()
        self._cluster_assignments: Dict[str, int] = {}
        self._cluster_source: Optional[Set[str]] = None
        self._last_cluster_step: Optional[ClusterStep] = None
        self._dynamic_options: Dict[str, Any] = {}
        self._metrics_cache: Dict[str, Dict[str, Any]] = {}
        self._metrics_cache_lock = threading.Lock()

        if debug:
            logging.getLogger().setLevel(logging.DEBUG)

        # Parse protocol
        # Priority: protocol_file (explicit path) > protocol (basename)
        if protocol_file:
            # Explicit file path overrides --protocol
            LOGGER.info(f"Loading protocol from file: {protocol_file}")
            self.protocol_steps = ProtocolFileParser.load_from_file(protocol_file)
            self.protocol_str = f"file:{protocol_file}"
        else:
            # Load from protocols/ directory by basename (default: "default")
            protocol_name = protocol if protocol else DEFAULT_PROTOCOL
            protocol_path = os.path.join(PROTOCOLS_DIR, f"{protocol_name}.json")
            if not os.path.exists(protocol_path):
                available = [f.stem for f in Path(PROTOCOLS_DIR).glob("*.json")]
                raise FileNotFoundError(
                    f"Protocol '{protocol_name}' not found at {protocol_path}. "
                    f"Available protocols: {', '.join(available)}"
                )
            LOGGER.info(f"Loading protocol: {protocol_name} ({protocol_path})")
            self.protocol_steps = ProtocolFileParser.load_from_file(protocol_path)
            self.protocol_str = f"protocol:{protocol_name}"

        # Will be set during initialization
        self.step02_json = None
        self.step02_pdb = None
        self.ref_pdb = None
        self.residue_classifier = None
        self.ligand_info = None
        self.constrained_atoms = {}
        self.catres_positions = []
        self.motif_positions = []
        self.mpnn_runner = None
        self.interaction_analyzer = None
        self.start_time = None
        self.metrics_history = []
        self.designs = []  # List of design dicts
        self._last_step_was_mpnn = False  # Track if protocol ended with MPNN

    @staticmethod
    def _is_mpnn_step(step: ProtocolStep) -> bool:
        return isinstance(step, (MPNNStep, MPNNMultiStep))

    @staticmethod
    def _iter_mpnn_strategies(step: ProtocolStep) -> List[MPNNStep]:
        if isinstance(step, MPNNMultiStep):
            return list(step.strategies)
        if isinstance(step, MPNNStep):
            return [step]
        return []

    def _resolve_mpnn_multi_workers(
        self,
        requested: Optional[int],
        minimum: Optional[int],
    ) -> int:
        """Resolve worker count for mpnn_multi based on environment and request."""
        env_candidates = [
            os.environ.get("SLURM_CPUS_PER_TASK"),
            os.environ.get("SLURM_CPUS_ON_NODE"),
            os.environ.get("OMP_NUM_THREADS"),
        ]
        available = None
        for val in env_candidates:
            if not val:
                continue
            try:
                parsed = int(val)
                if parsed > 0:
                    available = parsed
                    break
            except ValueError:
                continue
        if available is None:
            available = os.cpu_count() or 1
        if requested is not None:
            try:
                requested = int(requested)
                if requested > 0:
                    available = min(available, requested)
            except Exception:
                pass
        if minimum is not None:
            try:
                minimum = int(minimum)
                if minimum > 1:
                    available = max(available, minimum)
            except Exception:
                pass
        return max(1, available)

    def initialize(self) -> None:
        """Load step02 data and initialize components."""
        LOGGER.info("=" * 70)
        LOGGER.info("STEP03: FASTMPNN DESIGN")
        LOGGER.info("=" * 70)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load step02 JSON
        LOGGER.info(f"Loading step02 JSON: {self.step02_json_path}")
        with open(self.step02_json_path, "r") as f:
            self.step02_json = json.load(f)

        # Get PDB paths from JSON
        metadata = self.step02_json.get("metadata", {})
        self.step02_pdb = metadata.get("output_pdb") or metadata.get("input_pdb")
        if not self.step02_pdb or not os.path.exists(self.step02_pdb):
            raise FileNotFoundError(f"Step02 PDB not found: {self.step02_pdb}")

        self.ref_pdb = metadata.get("ref_pdb") or self.step02_pdb

        if self.step01_pdb is None:
            self.step01_pdb = metadata.get("step01_pdb") or metadata.get("input_pdb")

        LOGGER.info(f"  Step02 PDB: {self.step02_pdb}")
        LOGGER.info(f"  Reference PDB: {self.ref_pdb}")
        if self.step01_pdb:
            LOGGER.info(f"  Step01 PDB: {self.step01_pdb}")
        LOGGER.info(f"  Rosetta timeout: {self.rosetta_timeout}s")
        LOGGER.info(f"  Cart relax max rounds: {self.cart_relax_max_rounds}")
        LOGGER.info(f"  PyRosetta image: {self.pyrosetta_image}")

        # Build HIS tautomer map from reference PDB
        # This is used to correct HIS tautomers after PyRosetta adds hydrogens
        self.his_tautomer_map = build_his_tautomer_map(self.step02_pdb)
        LOGGER.info(f"  HIS tautomer map: {len(self.his_tautomer_map)} residues")

        # Get constrained atoms from step02 JSON
        self._load_constraints_from_json()

        # Initialize residue classifier
        LOGGER.info("Classifying residues into design spheres...")
        self.residue_classifier = ResidueClassifier(
            step02_pdb_path=self.step02_pdb,
            step02_json_path=self.step02_json_path,
            catres_subset=self.catres_subset_override,
            layer_cuts=self.layer_cuts,
            design_gly_pro=self.design_gly_pro,
        )
        self.residue_classifier.classify()

        # Get ligand info
        self.ligand_info = self.residue_classifier.ligand
        self.catres_positions = self.residue_classifier.get_catres_positions()
        self.motif_positions = self.residue_classifier.get_motif_positions()

        # Log classification summary
        summary = self.residue_classifier.get_summary()
        LOGGER.info(f"  Catalytic residues: {summary['num_catalytic']}")
        LOGGER.info(f"  Primary sphere: {summary['num_primary']}")
        LOGGER.info(f"  Secondary sphere: {summary['num_secondary']}")
        LOGGER.info(f"  Total fixed: {summary['num_fixed']}")

        # Initialize MPNN runner
        self.mpnn_runner = MPNNRunner(
            container_image=self.mpnn_container_image,
            use_container=self.mpnn_use_container,
            use_gpu=self.mpnn_use_gpu,
            use_mpnn_server=self.use_mpnn_server,
            mpnn_server_host=self.mpnn_server_host,
            mpnn_server_port=self.mpnn_server_port,
            auto_start_server=self.auto_start_mpnn_server,
        )

        # Initialize interaction analyzer
        if self.conserve_favorable_interactions:
            self.interaction_analyzer = InteractionAnalyzer(
                conservation_probability=self.conservation_probability,
            )

        # Validate and potentially modify protocol
        self._validate_protocol()

        # Apply global MPNN sphere overrides
        self._apply_mpnn_sphere_overrides()

        # Log protocol
        if self.mpnn_spheres_override:
            LOGGER.info(f"MPNN spheres override active: {self.mpnn_spheres_override}")
        LOGGER.info(f"\nProtocol ({self.protocol_str}):")
        LOGGER.info(ProtocolParser.describe_protocol(self.protocol_steps))

    def _validate_protocol(self) -> None:
        """Validate protocol and warn about potential issues.

        Checks:
        1. If protocol ends with MPNN (no relaxation after) - warn user
        2. If first step is cart_relax on step02 output - optionally skip it
        3. Protocol has at least one design step
        """
        if not self.protocol_steps:
            LOGGER.warning("Protocol is empty! No design steps defined.")
            return

        # Check if first step is cart_relax and should be skipped
        if self.skip_initial_cart_relax and len(self.protocol_steps) > 0:
            first_step = self.protocol_steps[0]
            if isinstance(first_step, CartRelaxStep):
                LOGGER.info("Skipping initial cart_relax step (--skip_initial_cart_relax)")
                LOGGER.info("  Reason: Step02 already performed cartesian relaxation")
                self.protocol_steps = self.protocol_steps[1:]

        # Check if protocol ends with MPNN (no Rosetta after)
        last_step = self.protocol_steps[-1] if self.protocol_steps else None
        if self._is_mpnn_step(last_step):
            LOGGER.warning("=" * 60)
            LOGGER.warning("PROTOCOL WARNING: Ending with MPNN step (no relaxation)")
            LOGGER.warning("  Final outputs will go through full_mpnn_output_restoration()")
            LOGGER.warning("  to restore REMARK 666, HIS tautomers, and HETATM records.")
            LOGGER.warning("  Consider adding a torsional_relax step for better geometry.")
            LOGGER.warning("=" * 60)
            self._last_step_was_mpnn = True

        # Check if there's at least one MPNN step
        has_mpnn = any(self._is_mpnn_step(s) for s in self.protocol_steps)
        if not has_mpnn:
            LOGGER.warning("Protocol has no MPNN design steps - only relaxation will occur")

        # Check for sensible ordering
        for i, step in enumerate(self.protocol_steps[:-1]):
            next_step = self.protocol_steps[i + 1]
            # Warn if two MPNN steps in a row without relaxation
            if self._is_mpnn_step(step) and self._is_mpnn_step(next_step):
                LOGGER.warning(f"Protocol has consecutive MPNN steps at positions {i+1}-{i+2}")
                LOGGER.warning("  Consider adding relaxation between MPNN steps")

        # Capture protocol target_count (last FinalDiversifyStep with explicit target_count)
        self.protocol_target_count = None
        for step in self.protocol_steps:
            if isinstance(step, FinalDiversifyStep) and step.target_count is not None:
                self.protocol_target_count = int(step.target_count)

        # If CLI num_final_designs is explicit, override protocol target_count to avoid over-generation
        if self.num_final_designs_explicit and self.num_final_designs is not None:
            for step in self.protocol_steps:
                if isinstance(step, FinalDiversifyStep):
                    step.target_count = int(self.num_final_designs)
            self.protocol_target_count = int(self.num_final_designs)
            LOGGER.info(f"Overriding protocol target_count with --num_final_designs={self.num_final_designs}")

    def _apply_mpnn_sphere_overrides(self) -> None:
        """Apply global MPNN sphere overrides from CLI flags."""
        # If design_secondary_sphere is set and no explicit override, default to primary+secondary
        if self.design_secondary_sphere and not self.mpnn_spheres_override:
            self.mpnn_spheres_override = "primary,secondary"

        if not self.mpnn_spheres_override:
            return

        spheres = [s.strip() for s in self.mpnn_spheres_override.split(",") if s.strip()]
        if not spheres:
            return

        LOGGER.info(f"Applying global MPNN sphere override: {spheres}")
        for step in self.protocol_steps:
            for mpnn_step in self._iter_mpnn_strategies(step):
                mpnn_step.design_spheres = spheres

    def _load_constraints_from_json(self) -> None:
        """Load constraint information from step02 JSON, with fallback to step01 JSON.

        Constraint loading priority:
        1. Check step02 JSON for 'residue_constraints' key
        2. If not found, fall back to step01 JSON referenced in metadata.step01_json

        Backbone atom handling:
        - If backbone_important_only_for_BB_BB_hbond=True, backbone atoms (N, CA, C, O, H)
          are excluded from constraints by default
        - Use --include_bb_hbond_constraints flag to include them anyway
        """
        LOGGER.info("=" * 60)
        LOGGER.info("LOADING COORDINATE CONSTRAINTS")
        LOGGER.info("=" * 60)

        residue_constraints = {}
        constraint_source = None

        # Backbone atoms to potentially exclude
        BACKBONE_ATOMS = {"N", "CA", "C", "O", "H", "HA", "HA2", "HA3"}

        # =====================================================================
        # STEP 1: Try to load from step02 JSON (check both top-level and metadata)
        # =====================================================================
        # Check top-level first, then metadata.residue_constraints
        rc = None
        if "residue_constraints" in self.step02_json:
            rc = self.step02_json["residue_constraints"]
            rc_location = "top-level"
        elif "metadata" in self.step02_json and "residue_constraints" in self.step02_json["metadata"]:
            rc = self.step02_json["metadata"]["residue_constraints"]
            rc_location = "metadata.residue_constraints"

        if rc is not None:
            if isinstance(rc, dict) and len(rc) > 0:
                residue_constraints = rc
                constraint_source = f"step02_json ({rc_location})"
                LOGGER.info(f"  [OK] Found residue_constraints in step02 JSON")
                LOGGER.info(f"       Location: {rc_location}")
                LOGGER.info(f"       Number of constraint entries: {len(rc)}")
            elif isinstance(rc, list) and len(rc) > 0:
                for res_cst in rc:
                    key = res_cst.get("resno", len(residue_constraints))
                    residue_constraints[str(key)] = res_cst
                constraint_source = f"step02_json ({rc_location})"
                LOGGER.info(f"  [OK] Found residue_constraints in step02 JSON (list format)")
                LOGGER.info(f"       Number of constraint entries: {len(residue_constraints)}")
            else:
                LOGGER.warning(f"  [WARN] residue_constraints in step02 JSON is empty or invalid type")
        else:
            LOGGER.info(f"  [INFO] No residue_constraints key in step02 JSON (checked top-level and metadata)")

        # =====================================================================
        # STEP 2: Fall back to step01 JSON if needed
        # =====================================================================
        if not residue_constraints:
            step01_json_path = self.step02_json.get("metadata", {}).get("step01_json")
            LOGGER.info(f"  [FALLBACK] Attempting to load from step01 JSON")
            LOGGER.info(f"             Path from metadata.step01_json: {step01_json_path}")

            if step01_json_path and os.path.exists(step01_json_path):
                LOGGER.info(f"  [OK] step01 JSON file exists, loading...")
                with open(step01_json_path) as f:
                    step01_json = json.load(f)

                rc = step01_json.get("residue_constraints", {})
                if isinstance(rc, dict) and len(rc) > 0:
                    residue_constraints = rc
                    constraint_source = f"step01_json: {step01_json_path}"
                    LOGGER.info(f"  [OK] Loaded {len(rc)} constraint entries from step01 JSON")
                else:
                    LOGGER.error(f"  [ERROR] step01 JSON has no valid residue_constraints!")
            elif step01_json_path:
                LOGGER.error(f"  [ERROR] step01 JSON path specified but file not found: {step01_json_path}")
            else:
                LOGGER.error(f"  [ERROR] No step01_json path in step02 metadata!")

        # =====================================================================
        # STEP 3: Process constraints - handle backbone filtering
        # =====================================================================
        if not residue_constraints:
            LOGGER.error("=" * 60)
            LOGGER.error("CRITICAL: NO CONSTRAINTS LOADED!")
            LOGGER.error("Relaxation will run WITHOUT coordinate constraints!")
            LOGGER.error("Catalytic geometry may be compromised!")
            LOGGER.error("=" * 60)
            return

        LOGGER.info(f"  Processing {len(residue_constraints)} constraint entries...")
        LOGGER.info(f"  Backbone exclusion for BB-BB hbond only: {not self.include_bb_hbond_constraints}")

        ligand_key = None
        catres_count = 0
        bb_excluded_count = 0

        for key, res_cst in residue_constraints.items():
            chain = res_cst.get("chain", "A")
            resno = res_cst.get("resno")
            resname = res_cst.get("resname", "UNK")
            atoms = res_cst.get("constrain_atoms", [])
            is_ligand = res_cst.get("is_ligand", False)
            bb_only_hbond = res_cst.get("backbone_important_only_for_BB_BB_hbond", False)
            bb_important = res_cst.get("backbone_important", False)
            sc_important = res_cst.get("sidechain_important", False)

            if resno is None:
                LOGGER.warning(f"    Skipping entry {key}: no resno specified")
                continue

            if not atoms:
                LOGGER.warning(f"    Skipping {chain}:{resno} ({resname}): no constrain_atoms")
                continue

            cst_key = f"{chain}:{resno}"

            # Handle ligand specially - ALWAYS constrain ALL atoms (including H) by default
            if is_ligand or key == "ligand":
                ligand_key = cst_key
                # Always use ALL_ATOMS for ligand - this includes hydrogens
                # We ignore the JSON-specified atoms and constrain everything
                self.constrained_atoms[cst_key] = "ALL_ATOMS"
                LOGGER.info(f"    [LIGAND] {cst_key} ({resname}): ALL_ATOMS constrained (default behavior)")
                LOGGER.debug(f"             (JSON specified {len(atoms)} atoms, but using ALL_ATOMS)")
                continue

            # Filter backbone atoms if bb_only_hbond is True and flag not set
            filtered_atoms = list(atoms)
            if bb_only_hbond and not self.include_bb_hbond_constraints:
                original_count = len(filtered_atoms)
                filtered_atoms = [a for a in filtered_atoms if a not in BACKBONE_ATOMS]
                removed = original_count - len(filtered_atoms)
                if removed > 0:
                    bb_excluded_count += removed
                    LOGGER.debug(f"    {cst_key}: Excluded {removed} backbone atoms (bb_only_hbond=True)")

            if filtered_atoms:
                self.constrained_atoms[cst_key] = filtered_atoms
                catres_count += 1
                importance = res_cst.get("importance", "unknown")
                LOGGER.info(f"    [CATRES] {cst_key} ({resname}): {len(filtered_atoms)} atoms, importance={importance}")
                LOGGER.debug(f"             Atoms: {filtered_atoms}")
            else:
                LOGGER.warning(f"    {cst_key} ({resname}): All atoms filtered out!")

        # =====================================================================
        # STEP 4: Ensure ligand is constrained (ALL_ATOMS by default)
        # =====================================================================
        if not ligand_key:
            LOGGER.info(f"  [INFO] No explicit ligand in constraints, checking residue classifier...")
            if hasattr(self, 'residue_classifier') and self.residue_classifier:
                lig = self.residue_classifier.ligand_info
                if lig:
                    ligand_key = f"{lig['chain']}:{lig['resno']}"
                    self.constrained_atoms[ligand_key] = "ALL_ATOMS"
                    LOGGER.info(f"    [LIGAND] Added from classifier: {ligand_key} (ALL_ATOMS, including H)")

        # =====================================================================
        # SUMMARY
        # =====================================================================
        LOGGER.info("=" * 60)
        LOGGER.info("CONSTRAINT LOADING SUMMARY")
        LOGGER.info("=" * 60)
        LOGGER.info(f"  Source: {constraint_source}")
        LOGGER.info(f"  Total constrained residues: {len(self.constrained_atoms)}")
        LOGGER.info(f"  Catalytic residues: {catres_count}")
        LOGGER.info(f"  Ligand: {ligand_key or 'NOT FOUND'}")
        LOGGER.info(f"  Backbone atoms excluded: {bb_excluded_count}")
        if not self.include_bb_hbond_constraints:
            LOGGER.info(f"  (Use --include_bb_hbond_constraints to include backbone atoms)")
        LOGGER.info("=" * 60)

        # Final sanity check
        if len(self.constrained_atoms) == 0:
            LOGGER.error("CRITICAL: No constraints loaded! Check input JSON files!")
        elif not ligand_key:
            LOGGER.warning("WARNING: No ligand constraints found!")

    def run_protocol(self) -> List[Dict]:
        """Execute the design protocol.

        Returns:
            List of design result dicts
        """
        self.start_time = time.time()
        total_steps = len(self.protocol_steps)

        # Current working structures (start with step02 output)
        current_pdbs = [self.step02_pdb]
        current_fixed = self.residue_classifier.get_fixed_residues()

        # Track timing for each step type
        step_timings = defaultdict(list)

        step_num = 0
        for step in self.protocol_steps:
            step_num += 1
            step_start = time.time()
            elapsed = step_start - self.start_time

            if elapsed > self.max_runtime:
                LOGGER.warning(f"Max runtime exceeded ({format_time(elapsed)} > {format_time(self.max_runtime)})")
                break

            log_step_header(step_num, step.step_type.value, total_steps)
            LOGGER.info(f"  Elapsed: {format_time(elapsed)} | Structures in: {len(current_pdbs)}")

            if isinstance(step, MPNNStep):
                current_pdbs = self._run_mpnn_step(step, current_pdbs, current_fixed)
                self._last_step_was_mpnn = True
            elif isinstance(step, MPNNMultiStep):
                current_pdbs = self._run_mpnn_multi_step(step, current_pdbs, current_fixed)
                self._last_step_was_mpnn = True
            elif isinstance(step, CartRelaxStep):
                current_pdbs = self._run_cart_relax_step(step, current_pdbs)
                self._last_step_was_mpnn = False
            elif isinstance(step, TorsionalRelaxStep):
                current_pdbs = self._run_torsional_relax_step(step, current_pdbs)
                self._last_step_was_mpnn = False
            elif isinstance(step, MinimizeStep):
                current_pdbs = self._run_minimize_step(step, current_pdbs)
                self._last_step_was_mpnn = False
            elif isinstance(step, RepackStep):
                current_pdbs = self._run_repack_step(step, current_pdbs)
                self._last_step_was_mpnn = False
            elif isinstance(step, SelectBestStep):
                current_pdbs = self._run_select_best_step(step, current_pdbs)
            elif isinstance(step, ScaleScoreTermStep):
                self._apply_scale_step(step)
            elif isinstance(step, SetOptionsStep):
                self._apply_set_options_step(step)
            elif isinstance(step, KeepInteractionsStep):
                current_fixed = self._run_keep_interactions_step(step, current_pdbs, current_fixed)
            elif isinstance(step, ClusterStep):
                self._run_cluster_step(step, current_pdbs)
            elif isinstance(step, KeepClusterBestStep):
                current_pdbs = self._run_keep_cluster_best_step(step, current_pdbs)
            elif isinstance(step, TaskOperationStep):
                current_fixed = self._run_task_operation_step(step, current_pdbs, current_fixed)
            elif isinstance(step, FinalDiversifyStep):
                current_pdbs = self._run_final_diversify_step(step, current_pdbs, current_fixed)
                self._mpnn_has_run = True
            elif isinstance(step, TimeCheckStep):
                triggered = self._time_check_triggered(step, elapsed)
                if triggered:
                    LOGGER.warning("Time check triggered; executing fallback steps")
                    fallback_steps = step.then_steps or []
                    if step.target_total_designs:
                        fallback_steps = self._apply_quota_to_steps(
                            fallback_steps, step.target_total_designs, len(current_pdbs)
                        )
                    total_steps = total_steps + len(fallback_steps)
                    current_pdbs, current_fixed, step_num = self._run_steps_sequence(
                        fallback_steps,
                        current_pdbs,
                        current_fixed,
                        step_timings,
                        step_num,
                        total_steps,
                        label_prefix="time_check",
                    )
                    if step.mode.lower() == "replace_remaining":
                        break

            # Update fixed residues if conserving favorable interactions
            if self.conserve_favorable_interactions and self._is_mpnn_step(step):
                current_fixed = self._update_fixed_from_interactions(current_pdbs, current_fixed)

            step_duration = time.time() - step_start
            step_timings[step.step_type.value].append(step_duration)
            LOGGER.info(f"  Step completed in {format_time(step_duration)} | Structures out: {len(current_pdbs)}")

            # Record metrics snapshot for diagnostics
            try:
                self._record_metrics_snapshot(step_num, step, current_pdbs, step_duration)
            except Exception as e:
                LOGGER.warning(f"Failed to record metrics snapshot: {e}")

        # Log timing summary
        total_elapsed = time.time() - self.start_time
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info("PROTOCOL TIMING SUMMARY")
        LOGGER.info(f"{'='*60}")
        for step_type, times in step_timings.items():
            avg_time = sum(times) / len(times)
            LOGGER.info(f"  {step_type}: {len(times)} runs, avg {format_time(avg_time)}, total {format_time(sum(times))}")
        LOGGER.info(f"  TOTAL: {format_time(total_elapsed)}")

        # Final selection
        final_designs = self._finalize_designs(current_pdbs)
        return final_designs

    def _time_check_triggered(self, step: TimeCheckStep, elapsed: float) -> bool:
        """Return True if time-check condition is met."""
        if step.max_elapsed is not None and elapsed >= float(step.max_elapsed):
            return True
        if step.min_remaining is not None:
            remaining = max(self.max_runtime - elapsed, 0)
            if remaining <= float(step.min_remaining):
                return True
        if step.max_runtime_fraction is not None:
            if self.max_runtime > 0 and (elapsed / self.max_runtime) >= float(step.max_runtime_fraction):
                return True
        return False

    def _apply_quota_to_steps(
        self,
        steps: List[ProtocolStep],
        target_total: int,
        current_count: int,
    ) -> List[ProtocolStep]:
        """Inject num_designs into first MPNN step without explicit num_designs."""
        if target_total is None or target_total <= 0:
            return steps
        per_input = max(1, int(math.ceil(target_total / max(1, current_count))))
        updated: List[ProtocolStep] = []
        applied = False
        for step in steps:
            if not applied and isinstance(step, MPNNStep):
                if step.num_designs is None:
                    updated.append(replace(step, num_designs=per_input))
                    applied = True
                else:
                    updated.append(step)
            elif not applied and isinstance(step, MPNNMultiStep):
                new_strategies: List[MPNNStep] = []
                for strat in step.strategies:
                    if not applied and strat.num_designs is None:
                        new_strategies.append(replace(strat, num_designs=per_input))
                        applied = True
                    else:
                        new_strategies.append(strat)
                if applied:
                    updated.append(replace(step, strategies=new_strategies))
                else:
                    updated.append(step)
            else:
                updated.append(step)
        if applied:
            LOGGER.info(f"  Time check quota: target={target_total}, per_input={per_input}")
        return updated

    def _run_steps_sequence(
        self,
        steps: List[ProtocolStep],
        current_pdbs: List[str],
        current_fixed: List[str],
        step_timings: Dict[str, List[float]],
        step_num_start: int,
        total_steps: int,
        label_prefix: str = "",
    ) -> Tuple[List[str], List[str], int]:
        """Run a list of protocol steps (used for time-check fallbacks)."""
        step_num = step_num_start
        for step in steps:
            step_num += 1
            step_start = time.time()

            step_label = step.step_type.value
            if label_prefix:
                step_label = f"{label_prefix}:{step_label}"
            log_step_header(step_num, step_label, total_steps)
            LOGGER.info(f"  Elapsed: {format_time(time.time() - self.start_time)} | Structures in: {len(current_pdbs)}")

            if isinstance(step, MPNNStep):
                current_pdbs = self._run_mpnn_step(step, current_pdbs, current_fixed)
                self._last_step_was_mpnn = True
            elif isinstance(step, CartRelaxStep):
                current_pdbs = self._run_cart_relax_step(step, current_pdbs)
                self._last_step_was_mpnn = False
            elif isinstance(step, TorsionalRelaxStep):
                current_pdbs = self._run_torsional_relax_step(step, current_pdbs)
                self._last_step_was_mpnn = False
            elif isinstance(step, MinimizeStep):
                current_pdbs = self._run_minimize_step(step, current_pdbs)
                self._last_step_was_mpnn = False
            elif isinstance(step, RepackStep):
                current_pdbs = self._run_repack_step(step, current_pdbs)
                self._last_step_was_mpnn = False
            elif isinstance(step, SelectBestStep):
                current_pdbs = self._run_select_best_step(step, current_pdbs)
            elif isinstance(step, ScaleScoreTermStep):
                self._apply_scale_step(step)
            elif isinstance(step, SetOptionsStep):
                self._apply_set_options_step(step)
            elif isinstance(step, KeepInteractionsStep):
                current_fixed = self._run_keep_interactions_step(step, current_pdbs, current_fixed)
            elif isinstance(step, ClusterStep):
                self._run_cluster_step(step, current_pdbs)
            elif isinstance(step, KeepClusterBestStep):
                current_pdbs = self._run_keep_cluster_best_step(step, current_pdbs)
            elif isinstance(step, TaskOperationStep):
                current_fixed = self._run_task_operation_step(step, current_pdbs, current_fixed)
            elif isinstance(step, FinalDiversifyStep):
                current_pdbs = self._run_final_diversify_step(step, current_pdbs, current_fixed)
                self._mpnn_has_run = True
            elif isinstance(step, TimeCheckStep):
                LOGGER.warning("Nested time_check step ignored in fallback sequence")

            if self.conserve_favorable_interactions and self._is_mpnn_step(step):
                current_fixed = self._update_fixed_from_interactions(current_pdbs, current_fixed)

            step_duration = time.time() - step_start
            step_timings[step.step_type.value].append(step_duration)
            LOGGER.info(f"  Step completed in {format_time(step_duration)} | Structures out: {len(current_pdbs)}")
            self._warn_on_basename_collisions(current_pdbs, step_label)

            try:
                self._record_metrics_snapshot(step_num, step, current_pdbs, step_duration)
            except Exception as e:
                LOGGER.warning(f"Failed to record metrics snapshot: {e}")

        return current_pdbs, current_fixed, step_num

    def _run_mpnn_step(
        self,
        step: MPNNStep,
        input_pdbs: List[str],
        fixed_residues: List[str],
        mpnn_runner_override: Optional[MPNNRunner] = None,
        mark_mpnn_run: bool = True,
        mpnn_has_run_override: Optional[bool] = None,
        dedupe_output: bool = True,
        strategy_tag: Optional[str] = None,
    ) -> List[str]:
        """Run MPNN design step."""
        temperature = step.temperature if step.temperature is not None else self.mpnn_temperature
        num_designs = step.num_designs if step.num_designs is not None else self.mpnn_num_designs
        batch_size = step.batch_size if step.batch_size is not None else self.mpnn_batch_size
        omit_aa = step.omit_aa if step.omit_aa is not None else self.mpnn_omit_aa
        repack_everything = step.repack_everything if step.repack_everything is not None else False

        call_index, packed_suffix = self._next_mpnn_packed_suffix(strategy_tag=strategy_tag)

        # Optional branching reduction after first MPNN step
        mpnn_has_run = self._mpnn_has_run if mpnn_has_run_override is None else mpnn_has_run_override
        if mpnn_has_run and self.mpnn_num_designs_after_first is not None:
            if not getattr(step, "num_designs_explicit", False):
                num_designs = self.mpnn_num_designs_after_first

        LOGGER.info(f"  Temperature: {temperature}, Designs: {num_designs}")
        LOGGER.info(f"  Spheres: {step.design_spheres}")
        LOGGER.info(f"  Packed suffix: {packed_suffix}")

        output_pdbs = []
        expected_total = len(input_pdbs) * (num_designs or 0)

        for input_pdb in input_pdbs:
            # Create unique output folder
            mpnn_out = self.output_dir / f"mpnn_{call_index:03d}_{time.time_ns()}_{uuid.uuid4().hex[:6]}"

            # Normalize input for MPNN (remove H, normalize HIS)
            mpnn_input_pdb = str(mpnn_out / "input_for_mpnn.pdb")
            try:
                normalize_pdb_for_mpnn(input_pdb, mpnn_input_pdb)
            except Exception as e:
                LOGGER.warning(f"Failed to normalize PDB for MPNN ({input_pdb}): {e}")
                mpnn_input_pdb = input_pdb
            else:
                LOGGER.info(f"  MPNN input normalized: {mpnn_input_pdb}")

            # Build MPNN input
            # Get reference sequence for diversity filtering (handles batch_size overshoot)
            ref_seq = self._get_sequence_for_pdb(input_pdb)

            mpnn_input = create_mpnn_input_from_classifier(
                pdb_path=mpnn_input_pdb,
                out_folder=str(mpnn_out),
                residue_classifier=self.residue_classifier,
                design_spheres=step.design_spheres,
                temperature=temperature,
                num_designs=num_designs,
                batch_size=batch_size,
                omit_aa=omit_aa,
                enhance=step.enhance,
                additional_fixed=fixed_residues,
                bias_aa=step.bias_aa,
                bias_aa_per_residue=step.bias_aa_per_residue,
                use_sc_context=1 if step.use_sc_context else 0,
                pack_side_chains=1 if step.pack_side_chains else 0,
                repack_everything=1 if repack_everything else 0,
                sc_denoising_steps=step.sc_denoising_steps,
                reference_sequence=ref_seq,
                packed_suffix=packed_suffix,
            )

            # Run MPNN
            try:
                runner = mpnn_runner_override or self.mpnn_runner
                result = runner.run(mpnn_input)

                # Restore PDB features for each output and add hydrogens.
                # We restore HIS tautomers immediately, then hydrate with PyRosetta
                # so all residues have hydrogens before any subsequent relax.
                for pdb in result.output_pdbs:
                    restored_pdb = str(Path(pdb).with_suffix(".restored.pdb"))
                    full_mpnn_output_restoration(
                        mpnn_pdb=pdb,
                        ref_pdb=self.step02_pdb,
                        output_pdb=restored_pdb,
                        original_ref_pdb=self.ref_pdb or self.step02_pdb,
                        restore_his_tautomers=True,
                    )
                    LOGGER.info(f"  Restored REMARK/HIS: {restored_pdb}")

                    # Hydrate via PyRosetta (adds hydrogens)
                    hydrated = self._run_rosetta_relax(
                        [restored_pdb],
                        mode="hydrate",
                        repeats=1,
                        ramp_stages=1,
                    )
                    hydrated_pdb = hydrated[0] if hydrated else restored_pdb
                    LOGGER.info(f"  Hydrated via PyRosetta: {hydrated_pdb}")

                    # Sanity check: ensure hydrogens exist after hydrate
                    try:
                        _, h_atoms = read_pdb_atoms(hydrated_pdb)
                        h_count = sum(
                            1 for a in h_atoms
                            if a.get("record_type") in ("ATOM", "HETATM")
                            and (
                                a.get("element", "").strip() == "H"
                                or (not a.get("element") and a.get("atom_name", "").startswith("H"))
                            )
                        )
                        if h_count == 0:
                            LOGGER.error(f"  Hydration check FAILED: no H atoms found in {hydrated_pdb}")
                        else:
                            LOGGER.info(f"  Hydration check: {h_count} H atoms present")
                    except Exception as e:
                        LOGGER.warning(f"  Could not verify hydrogens in {hydrated_pdb}: {e}")

                    # Re-add REMARK 666 lines after PyRosetta dump (and ensure HETATM)
                    if hydrated_pdb.endswith(".ligfixed.pdb"):
                        final_pdb = hydrated_pdb.replace(".ligfixed.pdb", ".restored_h.pdb")
                    else:
                        final_pdb = hydrated_pdb.replace(".hydrate.pdb", ".restored_h.pdb")
                    try:
                        full_mpnn_output_restoration(
                            mpnn_pdb=hydrated_pdb,
                            ref_pdb=self.step02_pdb,
                            output_pdb=final_pdb,
                            original_ref_pdb=self.ref_pdb or self.step02_pdb,
                            restore_his_tautomers=True,
                        )
                        LOGGER.info(f"  Final REMARK-restored PDB: {final_pdb}")

                        # Force ligand (including hydrogens) to match reference
                        ligand_fixed = final_pdb.replace(".pdb", ".ligfixed.pdb")
                        restore_ligand_from_ref(
                            mpnn_pdb=final_pdb,
                            ref_pdb=self.step02_pdb,
                            output_pdb=ligand_fixed,
                            ligand_info=(
                                self.ligand_info.chain,
                                self.ligand_info.resno,
                                self.ligand_info.resname,
                            ) if self.ligand_info else None,
                        )
                        final_pdb = ligand_fixed
                        LOGGER.info(f"  Ligand forced to reference: {final_pdb}")
                    except Exception as e:
                        LOGGER.warning(f"Failed to re-add REMARK/HIS after hydrate: {e}")
                        final_pdb = hydrated_pdb

                    output_pdbs.append(final_pdb)
                    with self._pdb_lineage_lock:
                        self._pdb_lineage[final_pdb] = input_pdb

            except Exception as e:
                LOGGER.error(f"MPNN failed for {input_pdb}: {e}")

        # Mark that we've run MPNN at least once
        if output_pdbs and mark_mpnn_run:
            self._mpnn_has_run = True

        # Deduplicate sequences after MPNN (optional)
        if dedupe_output:
            before = len(output_pdbs)
            output_pdbs = self._dedupe_pdbs_by_sequence(output_pdbs)
            after = len(output_pdbs)
            if before and after < before:
                LOGGER.info(f"  Deduped MPNN outputs: {before} -> {after}")

        if not output_pdbs:
            LOGGER.warning("  MPNN produced 0 outputs; falling back to input PDBs")
            return input_pdbs
        if expected_total and len(output_pdbs) < max(1, expected_total // 2):
            LOGGER.warning(
                f"  MPNN outputs lower than expected (got {len(output_pdbs)}, expected ~{expected_total}). "
                "This may be due to deduping, strong constraints, or small design region."
            )
        return output_pdbs

    def _run_mpnn_multi_step(
        self,
        step: MPNNMultiStep,
        input_pdbs: List[str],
        fixed_residues: List[str],
    ) -> List[str]:
        """Run multiple MPNN strategies in parallel and pool outputs."""
        if not step.strategies:
            LOGGER.warning("MPNN multi-step has no strategies; skipping")
            return input_pdbs
        base_has_run = self._mpnn_has_run
        LOGGER.info(f"  MPNN strategies: {len(step.strategies)} (pooling results)")

        run_parallel = bool(step.parallel)
        max_workers = self._resolve_mpnn_multi_workers(step.max_workers, step.min_workers)
        if not run_parallel or max_workers <= 1:
            if run_parallel:
                LOGGER.info("  Parallel mpnn_multi requested, but only 1 worker available; running sequentially")
            run_parallel = False
            max_workers = 1

        if run_parallel and self.rosetta_in_process:
            LOGGER.warning("  Parallel mpnn_multi disabled: rosetta_in_process is True")
            run_parallel = False
            max_workers = 1

        use_mpnn_server = step.use_mpnn_server if step.use_mpnn_server is not None else self.use_mpnn_server
        if run_parallel and use_mpnn_server:
            LOGGER.warning("  Parallel mpnn_multi forces MPNN server off to avoid port contention")
            use_mpnn_server = False

        pooled: List[str] = []

        def _run_strategy(strat: MPNNStep, index: int) -> List[str]:
            LOGGER.info(f"  Strategy {index}/{len(step.strategies)}: {strat.step_type.value}")
            runner = self.mpnn_runner
            needs_custom_runner = run_parallel or (
                step.use_mpnn_server is not None and step.use_mpnn_server != self.use_mpnn_server
            )
            if needs_custom_runner:
                runner = MPNNRunner(
                    container_image=self.mpnn_container_image,
                    use_container=self.mpnn_use_container,
                    use_gpu=self.mpnn_use_gpu,
                    use_mpnn_server=use_mpnn_server,
                    mpnn_server_host=self.mpnn_server_host,
                    mpnn_server_port=self.mpnn_server_port,
                    auto_start_server=self.auto_start_mpnn_server,
                )
            return self._run_mpnn_step(
                strat,
                input_pdbs,
                fixed_residues,
                mpnn_runner_override=runner,
                mark_mpnn_run=False,
                mpnn_has_run_override=base_has_run,
                dedupe_output=False,
                strategy_tag=f"s{index:02d}",
            )

        if run_parallel:
            LOGGER.info(f"  Parallel execution enabled (workers={max_workers})")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(_run_strategy, strat, idx): idx
                    for idx, strat in enumerate(step.strategies, 1)
                }
                for future in as_completed(future_map):
                    try:
                        pooled.extend(future.result())
                    except Exception as e:
                        LOGGER.error(f"  Strategy {future_map[future]} failed: {e}")
        else:
            for idx, strat in enumerate(step.strategies, 1):
                pooled.extend(_run_strategy(strat, idx))

        if pooled:
            self._mpnn_has_run = True
        else:
            self._mpnn_has_run = base_has_run

        if step.dedupe_pool:
            before = len(pooled)
            pooled = self._dedupe_pdbs_by_sequence(pooled)
            after = len(pooled)
            if before and after < before:
                LOGGER.info(f"  mpnn_multi dedupe: {before} -> {after}")

        return pooled if pooled else input_pdbs

    def _run_cart_relax_step(
        self,
        step: CartRelaxStep,
        input_pdbs: List[str],
    ) -> List[str]:
        """Run cartesian relaxation step."""
        # Use CLI override if provided, otherwise use step's value
        cart_bonded_weight = self.cart_bonded_weight if self.cart_bonded_weight != 2.0 else step.cart_bonded_weight
        LOGGER.info(f"  Repeats: {step.repeats}, Stages: {step.stages}")
        LOGGER.info(f"  Cart bonded weight: {cart_bonded_weight}")
        if step.fa_rep_weight is not None:
            LOGGER.info(f"  fa_rep weight: {step.fa_rep_weight}")
        if step.global_coord_cst_weight > 0:
            LOGGER.info(f"  Global constraints: weight={step.global_coord_cst_weight}, stdev={step.global_coord_cst_stdev}")

        if step.until_converged:
            return self._run_cart_relax_until_converged(step, input_pdbs, cart_bonded_weight)

        return self._run_rosetta_relax(
            input_pdbs,
            mode="cartesian",
            repeats=step.repeats,
            ramp_stages=step.stages,
            cart_bonded_weight=cart_bonded_weight,
            until_converged=step.until_converged,
            scorefunction=step.scorefunction,
            coord_cst_weight=step.coord_cst_weight,
            coord_cst_stdev=step.coord_cst_stdev,
            global_coord_cst_weight=step.global_coord_cst_weight,
            global_coord_cst_stdev=step.global_coord_cst_stdev,
            fa_rep_weight=step.fa_rep_weight,
            score_term_weights=step.score_term_weights,
            relax_rounds=step.relax_rounds,
            relax_inner_cycles=step.relax_inner_cycles,
            enable_bond_geometry_min=step.enable_bond_geometry_min,
        )

    def _run_cart_relax_until_converged(
        self,
        step: CartRelaxStep,
        input_pdbs: List[str],
        cart_bonded_weight: float,
    ) -> List[str]:
        """Run cartesian relax iteratively until bond/angle convergence."""
        LOGGER.info(f"  Running until converged (max rounds={self.cart_relax_max_rounds})")
        output_pdbs = []

        for input_pdb in input_pdbs:
            current = input_pdb
            rounds = 0
            while True:
                rounds += 1
                LOGGER.info(f"  [cart_relax] round {rounds} on {Path(current).name}")

                out = self._run_rosetta_relax(
                    [current],
                    mode="cartesian",
                    repeats=step.repeats,
                    ramp_stages=step.stages,
                    cart_bonded_weight=cart_bonded_weight,
                    scorefunction=step.scorefunction,
                    coord_cst_weight=step.coord_cst_weight,
                    coord_cst_stdev=step.coord_cst_stdev,
                    global_coord_cst_weight=step.global_coord_cst_weight,
                    global_coord_cst_stdev=step.global_coord_cst_stdev,
                    fa_rep_weight=step.fa_rep_weight,
                    relax_rounds=step.relax_rounds,
                    relax_inner_cycles=step.relax_inner_cycles,
                    enable_bond_geometry_min=step.enable_bond_geometry_min,
                )
                current = out[0] if out else current

                # Check convergence
                try:
                    calc = MetricsCalculator(
                        designed_pdb=current,
                        step02_pdb=self.step02_pdb,
                        step01_pdb=self.step01_pdb,
                        params_files=self.params_files,
                        constrained_atoms=self.constrained_atoms,
                        catres_positions=self.catres_positions,
                        motif_positions=self.motif_positions,
                        ligand_info=(self.ligand_info.chain, self.ligand_info.resname, self.ligand_info.resno)
                        if self.ligand_info else None,
                        pyrosetta_image=self.pyrosetta_image,
                        use_container_fallback=not self.no_container,
                        container_timeout=self.rosetta_timeout,
                    )
                    geom = calc.calculate_bond_geometry()
                    bond_max = geom["bond_length_geometry"]["unconstrained_only"]["max"]
                    angle_max = geom["bond_angle_geometry"]["unconstrained_only"]["max"]
                    LOGGER.info(f"    Bond max={bond_max:.4f}A | Angle max={angle_max:.2f}deg")
                    if bond_max <= step.bond_length_tolerance and angle_max <= step.bond_angle_tolerance:
                        LOGGER.info("    Converged: bond/angle tolerances satisfied")
                        break
                except Exception as e:
                    LOGGER.warning(f"    Could not evaluate convergence: {e}")
                    break

                # Stop if runtime exceeded or max rounds reached
                if rounds >= self.cart_relax_max_rounds:
                    LOGGER.warning("    Max convergence rounds reached")
                    break
                if self.start_time and (time.time() - self.start_time) > self.max_runtime:
                    LOGGER.warning("    Max runtime exceeded during cart_relax convergence loop")
                    break

            output_pdbs.append(current)

        return output_pdbs

    def _run_torsional_relax_step(
        self,
        step: TorsionalRelaxStep,
        input_pdbs: List[str],
    ) -> List[str]:
        """Run torsional relaxation step."""
        LOGGER.info(f"  Repeats: {step.repeats}, Stages: {step.stages}")
        if step.fa_rep_weight is not None:
            LOGGER.info(f"  fa_rep weight: {step.fa_rep_weight}")
        if step.global_coord_cst_weight > 0:
            LOGGER.info(f"  Global constraints: weight={step.global_coord_cst_weight}, stdev={step.global_coord_cst_stdev}")

        return self._run_rosetta_relax(
            input_pdbs,
            mode="torsional",
            repeats=step.repeats,
            ramp_stages=step.stages,
            scorefunction=step.scorefunction,
            coord_cst_weight=step.coord_cst_weight,
            coord_cst_stdev=step.coord_cst_stdev,
            global_coord_cst_weight=step.global_coord_cst_weight,
            global_coord_cst_stdev=step.global_coord_cst_stdev,
            fa_rep_weight=step.fa_rep_weight,
            score_term_weights=step.score_term_weights,
            relax_rounds=step.relax_rounds,
            relax_inner_cycles=step.relax_inner_cycles,
        )

    def _run_minimize_step(
        self,
        step: MinimizeStep,
        input_pdbs: List[str],
    ) -> List[str]:
        """Run minimization step."""
        LOGGER.info(f"  Tolerance: {step.tolerance}")

        mobile_residues = None
        if step.minimize_scope:
            mobile_residues = self._get_mobile_residues_for_scope(step.minimize_scope, input_pdbs)
            if mobile_residues is not None:
                LOGGER.info(f"  Minimize scope: {step.minimize_scope} ({len(mobile_residues)} residues)")

        return self._run_rosetta_relax(
            input_pdbs,
            mode="minimize",
            repeats=1,
            ramp_stages=1,
            min_tolerance=step.tolerance,
            min_max_iter=step.max_iter,
            min_cartesian=step.cartesian,
            min_backbone_rmsd_cutoff=step.min_backbone_rmsd_cutoff,
            scorefunction=step.scorefunction,
            coord_cst_weight=step.coord_cst_weight,
            coord_cst_stdev=step.coord_cst_stdev,
            global_coord_cst_weight=step.global_coord_cst_weight,
            global_coord_cst_stdev=step.global_coord_cst_stdev,
            fa_rep_weight=step.fa_rep_weight,
            cart_bonded_weight=step.cart_bonded_weight,
            score_term_weights=step.score_term_weights,
            mobile_residues=mobile_residues,
        )

    def _run_repack_step(
        self,
        step: RepackStep,
        input_pdbs: List[str],
    ) -> List[str]:
        """Run repacking step."""
        mobile_residues = None
        if step.repack_shell is not None:
            mobile_residues = self._get_mobile_residues_for_shell(step.repack_shell)
            LOGGER.info(f"  Repack shell: {step.repack_shell}A ({len(mobile_residues)} residues)")
        else:
            scope = step.repack_scope or "core_shell_flex"
            mobile_residues = self._get_mobile_residues_for_scope(scope, input_pdbs)
            if mobile_residues is not None:
                LOGGER.info(f"  Repack scope: {scope} ({len(mobile_residues)} residues)")

        if mobile_residues is not None:
            mobile_residues = self._filter_out_catres(mobile_residues)
            if len(mobile_residues) == 0:
                LOGGER.warning("  Repack: no mobile residues after catres filtering; skipping repack")
                return input_pdbs

        return self._run_rosetta_relax(
            input_pdbs,
            mode="repack",
            repeats=1,
            ramp_stages=1,
            mobile_residues=mobile_residues,
            scorefunction=step.scorefunction,
            coord_cst_weight=step.coord_cst_weight,
            coord_cst_stdev=step.coord_cst_stdev,
            global_coord_cst_weight=step.global_coord_cst_weight,
            global_coord_cst_stdev=step.global_coord_cst_stdev,
            fa_rep_weight=step.fa_rep_weight,
            score_term_weights=step.score_term_weights,
        )

    def _dedupe_pdbs_by_sequence(self, pdbs: List[str]) -> List[str]:
        """Remove duplicate sequences, keeping the best geometry per sequence."""
        if not pdbs or len(pdbs) <= 1:
            return pdbs

        sequences: List[str] = []
        metrics: List[Dict[str, Any]] = []

        for pdb in pdbs:
            seq = None
            try:
                _, atoms = read_pdb_atoms(pdb)
                seq = get_sequence_from_atoms(atoms)
            except Exception:
                seq = None

            # If sequence extraction failed, keep it unique to avoid over-deduping
            if not seq:
                seq = f"__missing__{pdb}"
            sequences.append(seq)

            metric_entry: Dict[str, Any] = {}
            try:
                cached = self._get_cached_bond_rmsd_metrics(pdb)
                bond_max = cached.get("bond_max")
                angle_max = cached.get("angle_max")
                if bond_max is not None:
                    metric_entry.setdefault(
                        "bond_length_geometry",
                        {"unconstrained_only": {"max": bond_max}},
                    )
                if angle_max is not None:
                    metric_entry.setdefault(
                        "bond_angle_geometry",
                        {"unconstrained_only": {"max": angle_max}},
                    )
            except Exception:
                metric_entry = {}
            metrics.append(metric_entry)

        try:
            _, unique_pdbs, _ = remove_duplicate_sequences(
                sequences, pdbs, metrics=metrics, keep_best_geometry=True
            )
            return unique_pdbs if unique_pdbs else pdbs
        except Exception as e:
            LOGGER.warning(f"Failed to dedupe sequences: {e}")
            return pdbs

    def _apply_scale_step(self, step: ScaleScoreTermStep) -> None:
        """Apply score term scaling to subsequent Rosetta steps."""
        def normalize(term: str) -> str:
            t = term.strip().lower()
            aliases = {
                "coord_cst": "coordinate_constraint",
                "coord_cst_weight": "coordinate_constraint",
                "coordinate_constraint": "coordinate_constraint",
                "fa_rep": "fa_rep",
                "rep": "fa_rep",
                "cart_bonded": "cart_bonded",
                "global_coord_cst": "global_coord_constraint",
                "global_cst": "global_coord_constraint",
                "global_coord_constraint": "global_coord_constraint",
            }
            return aliases.get(t, t)

        normalized: Dict[str, float] = {}
        reset_terms = []
        for k, v in (step.terms or {}).items():
            key = normalize(k)
            if isinstance(v, str) and v.strip().lower() in ("reset", "default", "none", "null"):
                reset_terms.append(key)
                continue
            if v is None:
                reset_terms.append(key)
                continue
            try:
                normalized[key] = float(v)
            except Exception:
                continue

        scope = (step.scope or "global").lower()
        if scope == "next":
            if reset_terms:
                if self._score_term_overrides_next_reset is None:
                    self._score_term_overrides_next_reset = set()
                self._score_term_overrides_next_reset.update(reset_terms)
                LOGGER.info(f"  One-shot reset of score term overrides: {sorted(reset_terms)}")
            if normalized:
                self._score_term_overrides_next = normalized
                LOGGER.info(f"  Applied one-shot score term overrides: {normalized}")
            if not reset_terms and not normalized:
                return
        else:
            if reset_terms:
                for term in reset_terms:
                    self._score_term_overrides.pop(term, None)
                LOGGER.info(f"  Reset global score term overrides: {sorted(reset_terms)}")
            if normalized:
                self._score_term_overrides.update(normalized)
                LOGGER.info(f"  Updated global score term overrides: {normalized}")
            if not reset_terms and not normalized:
                return

    def _apply_set_options_step(self, step: SetOptionsStep) -> None:
        """Update runtime defaults from set-options step."""
        def coerce(val: Any) -> Any:
            if isinstance(val, (int, float, bool)):
                return val
            if isinstance(val, str):
                low = val.lower()
                if low in ("true", "yes", "y", "1"):
                    return True
                if low in ("false", "no", "n", "0"):
                    return False
                try:
                    if "." in val:
                        return float(val)
                    return int(val)
                except ValueError:
                    return val
            return val

        for key, value in step.options.items():
            val = coerce(value)
            if hasattr(self, key):
                setattr(self, key, val)
                self._dynamic_options[key] = val
                LOGGER.info(f"  Set option {key} = {val}")
            else:
                LOGGER.warning(f"  Unknown option '{key}' in set step")

    def _get_sequence_for_pdb(self, pdb: str) -> str:
        try:
            _, atoms = read_pdb_atoms(pdb)
            return get_sequence_from_atoms(atoms)
        except Exception:
            return ""

    def _get_reference_sequence(self, pdb: str, ref: str = "parent") -> str:
        ref = (ref or "parent").lower()
        if ref == "parent":
            parent = self._pdb_lineage.get(pdb)
            if parent:
                return self._get_sequence_for_pdb(parent)
        if ref == "step01" and self.step01_pdb and os.path.exists(self.step01_pdb):
            return self._get_sequence_for_pdb(self.step01_pdb)
        # default to step02
        return self._get_sequence_for_pdb(self.step02_pdb)

    def _run_keep_interactions_step(
        self,
        step: KeepInteractionsStep,
        pdbs: List[str],
        current_fixed: List[str],
    ) -> List[str]:
        """Add fixed residues based on interaction analysis."""
        from .interaction_analyzer import InteractionAnalyzer, InteractionConfig, InteractionType

        if not pdbs:
            return current_fixed

        # Determine target residue set
        target = step.target.lower()
        catres_positions: List[Tuple[str, int]] = []
        include_ligand = step.include_ligand_interactions
        include_catres = step.include_catres_interactions

        if target in ("catres", "catres_subset"):
            catres_positions = self.catres_positions
            include_catres = True
        elif target in ("motif", "all_motif"):
            catres_positions = self.motif_positions
            include_catres = True
        elif target in ("catres_or_motif", "motif_or_catres"):
            catres_positions = list({*self.catres_positions, *self.motif_positions})
            include_catres = True
        elif target == "ligand":
            include_ligand = True
            include_catres = False

        interaction_types = [InteractionType.from_string(t) for t in step.interaction_types]
        config = InteractionConfig(
            interaction_types=interaction_types,
            include_ligand_interactions=include_ligand,
            include_catres_interactions=include_catres,
            conservation_probability=step.probability,
        )
        # Atom scope filters
        config.mutator_atoms = step.mutator_atoms
        config.target_atoms = step.target_atoms
        if step.hbond_accept_probability is not None:
            config.hbond_accept_probability = step.hbond_accept_probability
        if step.strong_interaction_types:
            config.strong_interaction_types = [InteractionType.from_string(t) for t in step.strong_interaction_types]

        analyzer = InteractionAnalyzer(config=config)

        updated_fixed = list(current_fixed)
        for pdb in pdbs:
            try:
                result = analyzer.analyze_mutations(
                    designed_pdb=pdb,
                    original_pdb=self.step02_pdb,
                    catres_positions=catres_positions,
                    ligand_chain=self.ligand_info.chain if (self.ligand_info and include_ligand) else None,
                    ligand_resname=self.ligand_info.resname if (self.ligand_info and include_ligand) else None,
                )
                updated_fixed = analyzer.update_fixed_residues(updated_fixed, result)
            except Exception as e:
                LOGGER.warning(f"Interaction analysis failed for {pdb}: {e}")

        return updated_fixed

    def _run_task_operation_step(
        self,
        step: TaskOperationStep,
        pdbs: List[str],
        current_fixed: List[str],
    ) -> List[str]:
        """Run a custom task operation to add fixed residues."""
        if not step.module:
            LOGGER.warning("Task operation step missing module")
            return current_fixed

        try:
            import importlib
            import importlib.util

            if step.module.endswith(".py") and os.path.exists(step.module):
                spec = importlib.util.spec_from_file_location("taskop_module", step.module)
                module = importlib.util.module_from_spec(spec)
                assert spec and spec.loader
                spec.loader.exec_module(module)
            else:
                module = importlib.import_module(step.module)

            func = getattr(module, step.function)
        except Exception as e:
            LOGGER.warning(f"Failed to load task operation {step.module}:{step.function}: {e}")
            return current_fixed

        updated_fixed = list(current_fixed)
        context = {
            "step02_pdb": self.step02_pdb,
            "step01_pdb": self.step01_pdb,
            "catres_positions": self.catres_positions,
            "motif_positions": self.motif_positions,
            "ligand_info": self.ligand_info,
            "residue_classifier": self.residue_classifier,
            "params_files": self.params_files,
        }
        for pdb in pdbs:
            try:
                result = func(pdb, context, **(step.args or {}))
            except TypeError:
                # Fallback: call with context only
                try:
                    result = func(context)
                except Exception as e:
                    LOGGER.warning(f"Task operation failed for {pdb}: {e}")
                    continue
            except Exception as e:
                LOGGER.warning(f"Task operation failed for {pdb}: {e}")
                continue

            # Accept list of residue IDs or dict mapping
            if isinstance(result, dict):
                res_ids = result.get(pdb, result.get("fixed_residues", []))
            else:
                res_ids = result or []
            for res_id in res_ids:
                if res_id not in updated_fixed:
                    updated_fixed.append(res_id)
        return updated_fixed

    def _run_cluster_step(self, step: ClusterStep, pdbs: List[str]) -> None:
        """Cluster structures by sequence or structure."""
        if not pdbs:
            return

        method = step.method.lower()
        sequences = [self._get_sequence_for_pdb(p) for p in pdbs]

        if method == "sequence":
            # Deterministic clustering by sequence identity
            def identity(i: int, j: int) -> float:
                return calculate_sequence_identity(sequences[i], sequences[j])

            centers: List[int] = []
            if step.n_clusters:
                # Build unique sequence representatives
                seq_to_indices: Dict[str, List[int]] = {}
                for i, seq in enumerate(sequences):
                    seq_to_indices.setdefault(seq, []).append(i)
                unique_indices = [indices[0] for indices in seq_to_indices.values()]

                if len(unique_indices) <= step.n_clusters:
                    # Each unique sequence becomes its own cluster
                    seq_to_cluster = {seq: idx for idx, seq in enumerate(seq_to_indices.keys())}
                    assignments = {pdbs[i]: seq_to_cluster[sequences[i]] for i in range(len(pdbs))}
                    self._cluster_assignments = assignments
                    self._cluster_source = set(pdbs)
                    self._last_cluster_step = step
                    LOGGER.info(
                        f"  Clustered {len(pdbs)} structures into {len(set(assignments.values()))} sequence clusters"
                    )
                    return

                # Farthest-first on unique sequences, with deterministic tie-breaks
                centers = [unique_indices[0]]
                while len(centers) < step.n_clusters and len(centers) < len(unique_indices):
                    best_idx = None
                    best_score = None
                    for idx in unique_indices:
                        if idx in centers:
                            continue
                        max_sim = max(identity(idx, c) for c in centers)
                        if best_score is None:
                            best_score = max_sim
                            best_idx = idx
                            continue
                        if max_sim < best_score - 1e-12:
                            best_score = max_sim
                            best_idx = idx
                        elif abs(max_sim - best_score) <= 1e-12 and idx < best_idx:
                            best_idx = idx
                    if best_idx is None:
                        break
                    centers.append(best_idx)
            else:
                # threshold-based clustering
                threshold = step.threshold if step.threshold is not None else 0.9
                for i in range(len(pdbs)):
                    placed = False
                    for c in centers:
                        if identity(i, c) >= threshold:
                            placed = True
                            break
                    if not placed:
                        centers.append(i)

            assignments: Dict[str, int] = {}
            for i, pdb in enumerate(pdbs):
                if not centers:
                    assignments[pdb] = 0
                    continue
                # assign to closest center (highest identity), deterministic tie-break
                best_center = None
                best_sim = None
                for k, c in enumerate(centers):
                    sim = identity(i, c)
                    if best_sim is None or sim > best_sim + 1e-12:
                        best_sim = sim
                        best_center = k
                    elif abs(sim - best_sim) <= 1e-12 and k < best_center:
                        best_center = k
                assignments[pdb] = best_center
            self._cluster_assignments = assignments
            self._cluster_source = set(pdbs)
            self._last_cluster_step = step
            LOGGER.info(f"  Clustered {len(pdbs)} structures into {len(set(assignments.values()))} sequence clusters")
            if step.n_clusters and len(sequences) >= step.n_clusters and len(set(assignments.values())) < step.n_clusters:
                LOGGER.warning(
                    f"  Cluster count lower than requested ({len(set(assignments.values()))} < {step.n_clusters}). "
                    "This suggests near-identical sequences or distance ties."
                )
        else:
            LOGGER.warning("Structure clustering not available without PyRosetta; falling back to sequence clustering")
            step_seq = ClusterStep(method="sequence", n_clusters=step.n_clusters, threshold=step.threshold)
            self._run_cluster_step(step_seq, pdbs)

    def _run_keep_cluster_best_step(self, step: KeepClusterBestStep, pdbs: List[str]) -> List[str]:
        """Keep best structures per cluster based on metric."""
        if not pdbs:
            return pdbs
        if not self._cluster_assignments:
            if self._last_cluster_step is not None:
                LOGGER.warning("  No cluster assignments found; reclustering before keep_cluster_best")
                self._run_cluster_step(self._last_cluster_step, pdbs)
            else:
                return pdbs
        if self._cluster_source is None or not set(pdbs).issubset(self._cluster_source):
            if self._last_cluster_step is not None:
                LOGGER.warning("  Cluster assignments stale; reclustering before keep_cluster_best")
                self._run_cluster_step(self._last_cluster_step, pdbs)
            else:
                LOGGER.warning("  No prior cluster step found; skipping keep_cluster_best")
                return pdbs

        clusters: Dict[int, List[str]] = {}
        for pdb in pdbs:
            cid = self._cluster_assignments.get(pdb, 0)
            clusters.setdefault(cid, []).append(pdb)

        selected: List[str] = []
        for cid, members in clusters.items():
            selected.extend(self._select_best_structures(members, step.n, step.metric))

        LOGGER.info(f"  Kept {len(selected)} structures from {len(clusters)} clusters")
        if len(selected) == 0:
            LOGGER.warning("  keep_cluster_best produced 0 outputs; returning inputs")
            return pdbs
        return selected

    def _get_cached_bond_rmsd_metrics(self, pdb: str) -> Dict[str, Any]:
        """Get cached bond/RMSD metrics for a PDB, computing if needed."""
        with self._metrics_cache_lock:
            entry = self._metrics_cache.get(pdb)
            if entry and "bond_max" in entry and "angle_max" in entry and "ca_rmsd" in entry:
                return entry

        calc = MetricsCalculator(
            designed_pdb=pdb,
            step02_pdb=self.step02_pdb,
            step01_pdb=self.step01_pdb,
            params_files=self.params_files,
            constrained_atoms=self.constrained_atoms,
            catres_positions=self.catres_positions,
            motif_positions=self.motif_positions,
            ligand_info=(self.ligand_info.chain, self.ligand_info.resname, self.ligand_info.resno)
            if self.ligand_info else None,
            bond_length_tolerance=self.bond_length_tolerance,
            bond_angle_tolerance=self.bond_angle_tolerance,
            pyrosetta_image=self.pyrosetta_image,
            use_container_fallback=not self.no_container,
            container_timeout=self.rosetta_timeout,
        )
        geom = calc.calculate_bond_geometry()
        rmsd = calc.calculate_rmsd_metrics()

        bond_max = None
        angle_max = None
        if "error" not in geom:
            bond_max = geom.get("bond_length_geometry", {}).get("unconstrained_only", {}).get("max")
            angle_max = geom.get("bond_angle_geometry", {}).get("unconstrained_only", {}).get("max")

        ca_rmsd = None
        if "error" not in rmsd:
            ca_rmsd = rmsd.get("global_ca_vs_step01")

        payload = {
            "bond_max": bond_max,
            "angle_max": angle_max,
            "ca_rmsd": ca_rmsd,
        }
        with self._metrics_cache_lock:
            entry = self._metrics_cache.setdefault(pdb, {})
            entry.update(payload)
            return entry

    def _get_cached_rosetta_score(self, pdb: str, scorefxn: str) -> Optional[float]:
        """Get cached Rosetta score for a PDB/scorefunction, computing if needed."""
        key = f"rosetta_score::{scorefxn}"
        with self._metrics_cache_lock:
            entry = self._metrics_cache.get(pdb)
            if entry and key in entry:
                return entry[key]

        calc = MetricsCalculator(
            designed_pdb=pdb,
            step02_pdb=self.step02_pdb,
            step01_pdb=self.step01_pdb,
            params_files=self.params_files,
            constrained_atoms=self.constrained_atoms,
            catres_positions=self.catres_positions,
            motif_positions=self.motif_positions,
            ligand_info=(self.ligand_info.chain, self.ligand_info.resname, self.ligand_info.resno)
            if self.ligand_info else None,
            bond_length_tolerance=self.bond_length_tolerance,
            bond_angle_tolerance=self.bond_angle_tolerance,
            pyrosetta_image=self.pyrosetta_image,
            use_container_fallback=not self.no_container,
            container_timeout=self.rosetta_timeout,
        )
        score = calc.calculate_rosetta_score(scorefunction=scorefxn)
        val = None
        if "error" not in score:
            val = score.get("total_score")
        with self._metrics_cache_lock:
            entry = self._metrics_cache.setdefault(pdb, {})
            entry[key] = val
        return val

    def _select_best_structures(
        self,
        pdbs: List[str],
        n: int,
        metric: str,
        step: Optional[SelectBestStep] = None,
    ) -> List[str]:
        """Select best N structures by advanced metrics."""
        if not pdbs:
            return pdbs

        metric = (metric or "geometry").lower()
        tol_bond = step.bond_length_tolerance if step and step.bond_length_tolerance is not None else self.bond_length_tolerance
        tol_angle = step.bond_angle_tolerance if step and step.bond_angle_tolerance is not None else self.bond_angle_tolerance
        geom_eps = step.geom_similarity_epsilon if step else 0.005
        rmsd_eps = step.rmsd_similarity_epsilon if step else 0.05
        seq_ref = step.sequence_ref if step else "parent"
        scorefxn = step.scorefunction if step and step.scorefunction else "ref2015_cart"

        def get_metrics(pdb: str) -> Dict[str, Any]:
            cached = self._get_cached_bond_rmsd_metrics(pdb)
            bond_max = cached.get("bond_max")
            angle_max = cached.get("angle_max")
            ca_rmsd = cached.get("ca_rmsd")

            rosetta_score = None
            if metric in ("score", "rosetta_score", "smart", "best", "legacy"):
                rosetta_score = self._get_cached_rosetta_score(pdb, scorefxn)

            seq_identity = None
            if "sequence" in metric or metric.startswith("seq_"):
                ref_seq = self._get_reference_sequence(pdb, ref=seq_ref)
                seq = self._get_sequence_for_pdb(pdb)
                try:
                    seq_identity = calculate_sequence_identity(ref_seq, seq)
                except Exception:
                    seq_identity = None

            return {
                "bond_max": bond_max,
                "angle_max": angle_max,
                "ca_rmsd": ca_rmsd,
                "rosetta_score": rosetta_score,
                "seq_identity": seq_identity,
            }

        # Precompute metrics
        metrics_map: Dict[str, Dict[str, Any]] = {}
        for pdb in pdbs:
            try:
                metrics_map[pdb] = get_metrics(pdb)
            except Exception as e:
                LOGGER.warning(f"  Failed metrics for {pdb}: {e}")
                metrics_map[pdb] = {}

        if metric in ("smart", "best", "legacy"):
            # Filter by geometry cutoffs if possible
            def geom_pass(m: Dict[str, Any]) -> bool:
                if m.get("bond_max") is None or m.get("angle_max") is None:
                    return False
                return m["bond_max"] <= tol_bond and m["angle_max"] <= tol_angle

            passing = [p for p in pdbs if geom_pass(metrics_map[p])]
            candidates = passing if passing else pdbs

            if not passing:
                # Choose closest to cutoffs
                def violation(m: Dict[str, Any]) -> float:
                    if m.get("bond_max") is None or m.get("angle_max") is None:
                        return float("inf")
                    v_bond = max(0.0, m["bond_max"] - tol_bond)
                    v_angle = max(0.0, m["angle_max"] - tol_angle)
                    return max(v_bond / tol_bond, v_angle / tol_angle)

                violations = {p: violation(metrics_map[p]) for p in candidates}
                min_v = min(violations.values()) if violations else float("inf")
                candidates = [p for p in candidates if violations.get(p, float("inf")) <= min_v + geom_eps]

            # Break ties by CA RMSD, then Rosetta score
            def rmsd_key(p: str) -> float:
                val = metrics_map[p].get("ca_rmsd")
                return val if val is not None else float("inf")

            candidates_sorted = sorted(candidates, key=rmsd_key)
            if len(candidates_sorted) > 1:
                best = candidates_sorted[0]
                second = candidates_sorted[1]
                best_r = rmsd_key(best)
                second_r = rmsd_key(second)
                if not (math.isfinite(best_r) and math.isfinite(second_r)):
                    # RMSD unavailable; use Rosetta score
                    def score_key(p: str) -> float:
                        val = metrics_map[p].get("rosetta_score")
                        return val if val is not None else float("inf")
                    candidates_sorted = sorted(candidates_sorted, key=score_key)
                elif abs(best_r - second_r) <= rmsd_eps:
                    # Use Rosetta score as tiebreaker
                    def score_key(p: str) -> float:
                        val = metrics_map[p].get("rosetta_score")
                        return val if val is not None else float("inf")
                    candidates_sorted = sorted(candidates_sorted, key=score_key)

            return candidates_sorted[:n]

        if metric in ("geometry", "bond_geometry"):
            def geom_key(p: str) -> float:
                m = metrics_map[p]
                return m.get("bond_max") if m.get("bond_max") is not None else float("inf")
            return sorted(pdbs, key=geom_key)[:n]

        if metric in ("score", "rosetta_score"):
            def score_key(p: str) -> float:
                m = metrics_map[p]
                return m.get("rosetta_score") if m.get("rosetta_score") is not None else float("inf")
            return sorted(pdbs, key=score_key)[:n]

        if metric in ("sequence_similarity_high", "seq_similarity_high"):
            def seq_key(p: str) -> float:
                m = metrics_map[p]
                return -(m.get("seq_identity") if m.get("seq_identity") is not None else -1.0)
            return sorted(pdbs, key=seq_key)[:n]

        if metric in ("sequence_similarity_low", "seq_similarity_low", "sequence_diversity"):
            def seq_key(p: str) -> float:
                m = metrics_map[p]
                return m.get("seq_identity") if m.get("seq_identity") is not None else float("inf")
            return sorted(pdbs, key=seq_key)[:n]

        if metric in ("ca_rmsd", "ca_rmsd_step01"):
            def rmsd_key(p: str) -> float:
                m = metrics_map[p]
                return m.get("ca_rmsd") if m.get("ca_rmsd") is not None else float("inf")
            return sorted(pdbs, key=rmsd_key)[:n]

        # Default fallback
        return pdbs[:n]

    def _get_mobile_residues_for_shell(self, shell_radius: float) -> List[str]:
        """Get residues within a CA distance shell of the ligand (chain:resno)."""
        if not self.residue_classifier:
            return []

        mobile = []
        for res in self.residue_classifier.residues.values():
            if res.is_fixed:
                continue
            if res.ca_distance_to_ligand <= shell_radius:
                mobile.append(f"{res.chain}:{res.resno}")

        return mobile

    def _get_mobile_residues_for_scope(self, scope: str, input_pdbs: List[str]) -> List[str]:
        """Get repackable residues for a scope, excluding fixed/catres when possible."""
        if self.residue_classifier:
            return self.residue_classifier.get_repack_residues_by_scope(scope)

        if not input_pdbs:
            return []

        LOGGER.warning("No residue classifier available; repack scope will use PDB-derived residues")
        return self._get_nonfixed_residues_from_pdb(input_pdbs[0])

    def _get_nonfixed_residues_from_pdb(self, pdb_path: str) -> List[str]:
        """Fallback: get non-fixed residues from PDB ATOM records (chain:resno)."""
        _, atoms = read_pdb_atoms(pdb_path)
        residues = {(a["chain"], a["resno"]) for a in atoms if a.get("record_type") == "ATOM"}
        fixed = set(self.catres_positions or []) | set(self.motif_positions or [])
        mobile = sorted(residues - fixed, key=lambda x: (x[0], x[1]))
        return [f"{chain}:{resno}" for chain, resno in mobile]

    def _filter_out_catres(self, residues: List[str]) -> List[str]:
        """Remove catalytic residues from a residue list."""
        if not residues or not self.catres_positions:
            return residues
        catres_norm = {f"{ch}{resno}" for ch, resno in self.catres_positions}

        def normalize(res_id: str) -> str:
            return str(res_id).replace(":", "").strip()

        return [r for r in residues if normalize(r) not in catres_norm]

    def _resolve_final_target_count(self, step: Optional[FinalDiversifyStep] = None) -> int:
        """Resolve target count for final outputs.

        Priority:
        1) CLI --num_final_designs (explicit)
        2) final_diversify target_count (if provided)
        3) Protocol target_count (if provided)
        4) Fallback default (10)
        """
        if self.num_final_designs_explicit and self.num_final_designs is not None:
            return int(self.num_final_designs)
        if step and step.target_count is not None:
            return int(step.target_count)
        if self.protocol_target_count is not None:
            return int(self.protocol_target_count)
        return 10

    def _run_final_diversify_step(
        self,
        step: FinalDiversifyStep,
        input_pdbs: List[str],
        fixed_residues: List[str],
    ) -> List[str]:
        """Run final diversification: multi-temp MPNN + clustering to target count.

        Algorithm:
        1. For each temperature, run MPNN (shell_only)
        2. Cross-temperature deduplication (keep best geometry)
        3. If short of target: run additional MPNN rounds
        4. If over target: cluster and pick best geometry from each cluster
        5. Assert no duplicate sequences in final output

        Target count resolution (in priority order):
        1. CLI --num_final_designs (explicit)
        2. Step JSON target_count (if specified)
        3. Default: 10
        """
        target_count = self._resolve_final_target_count(step)

        # Dynamic scaling: if expected output is below threshold, scale up
        effective_designs_per_temp = step.designs_per_temp
        effective_batch_size = step.batch_size

        max_possible = len(input_pdbs) * len(step.temperatures) * effective_designs_per_temp
        threshold_count = step.overshoot_threshold * target_count

        while max_possible < threshold_count:
            # Double batch_size and designs_per_temp (keeps number_of_batches constant)
            effective_batch_size *= 2
            effective_designs_per_temp *= 2
            max_possible = len(input_pdbs) * len(step.temperatures) * effective_designs_per_temp
            LOGGER.info(f"  Scaling up: designs_per_temp={effective_designs_per_temp}, "
                        f"batch_size={effective_batch_size} (max_possible={max_possible})")

        LOGGER.info(f"\n{'='*50}")
        LOGGER.info(f"Final diversify: temps={step.temperatures}, target={target_count}")
        if effective_designs_per_temp != step.designs_per_temp:
            LOGGER.info(f"  Dynamic scaling: {step.designs_per_temp}→{effective_designs_per_temp} designs/temp "
                        f"(batch={effective_batch_size})")
        LOGGER.info(f"{'='*50}")

        # Phase 1: Multi-temperature sampling
        all_samples = []  # List of (sequence, pdb, temperature)

        for temp in step.temperatures:
            LOGGER.info(f"  T={temp} sampling (scope={step.design_scope})...")
            mpnn_step = MPNNStep(
                temperature=temp,
                num_designs=effective_designs_per_temp,
                design_scope=step.design_scope,
                design_spheres=step.design_spheres,
                omit_aa=step.omit_aa,
                enhance=step.enhance,
                use_sc_context=step.use_sc_context,
                pack_side_chains=step.pack_side_chains,
                sc_denoising_steps=step.sc_denoising_steps,
                batch_size=effective_batch_size,
            )

            output_pdbs = self._run_mpnn_step(mpnn_step, input_pdbs, fixed_residues)

            for pdb in output_pdbs:
                try:
                    _, atoms = read_pdb_atoms(pdb)
                    seq = get_sequence_from_atoms(atoms)
                    all_samples.append((seq, pdb, temp))
                except Exception:
                    pass

            LOGGER.info(f"    Generated {len(output_pdbs)} structures")

        # Phase 2: Cross-temperature deduplication
        unique_seqs, unique_pdbs, unique_metrics = self._cross_temp_dedupe(all_samples)
        LOGGER.info(f"  Cross-temp dedupe: {len(all_samples)} -> {len(unique_seqs)} unique")

        # Phase 3: Expansion loop (if needed)
        # Sort input_pdbs by geometry (best first) for expansion priority
        if len(unique_seqs) < target_count:
            input_pdbs_sorted = self._sort_pdbs_by_geometry(input_pdbs)
        else:
            input_pdbs_sorted = input_pdbs

        iteration = 0
        while len(unique_seqs) < target_count and iteration < step.max_iterations:
            iteration += 1
            prev_count = len(unique_seqs)
            needed = target_count - len(unique_seqs)

            LOGGER.info(f"  Expansion round {iteration} (have {len(unique_seqs)}, need {needed} more)")

            # Calculate how many designs per input to reach target
            # Use all inputs, sorted by geometry (best first)
            num_designs_per_input = max(1, (needed + len(input_pdbs_sorted) - 1) // len(input_pdbs_sorted))

            mpnn_step = MPNNStep(
                temperature=max(step.temperatures),  # Use highest temp for max diversity
                num_designs=num_designs_per_input,
                design_scope=step.design_scope,
                design_spheres=step.design_spheres,
                omit_aa=step.omit_aa,
                enhance=step.enhance,
                use_sc_context=step.use_sc_context,
                pack_side_chains=step.pack_side_chains,
                sc_denoising_steps=step.sc_denoising_steps,
            )

            new_pdbs = self._run_mpnn_step(mpnn_step, input_pdbs_sorted, fixed_residues)
            new_samples = []
            for pdb in new_pdbs:
                try:
                    _, atoms = read_pdb_atoms(pdb)
                    seq = get_sequence_from_atoms(atoms)
                    new_samples.append((seq, pdb, max(step.temperatures)))
                except Exception:
                    pass

            # Merge and re-dedupe (keeps existing unique sequences + new ones)
            all_samples = [(s, p, 0.0) for s, p in zip(unique_seqs, unique_pdbs)]
            all_samples.extend(new_samples)
            unique_seqs, unique_pdbs, unique_metrics = self._cross_temp_dedupe(all_samples)

            LOGGER.info(f"    Generated {len(new_pdbs)} structures, now have {len(unique_seqs)} unique")

            if len(unique_seqs) <= prev_count:
                LOGGER.warning(f"    No progress in expansion round {iteration}, stopping")
                break

        # Phase 3b: Flex fallback (if still short after max_iterations)
        if len(unique_seqs) < target_count and step.fallback_include_flex:
            needed = target_count - len(unique_seqs)
            LOGGER.info(f"  Flex fallback: still need {needed} more, trying shell+flex design")

            # Use shell+flex spheres for more designable positions
            num_designs_per_input = max(1, (needed + len(input_pdbs_sorted) - 1) // len(input_pdbs_sorted))

            mpnn_step = MPNNStep(
                temperature=max(step.temperatures),
                num_designs=num_designs_per_input,
                design_scope="shell_flex",
                design_spheres=["shell", "flex"],
                omit_aa=step.omit_aa,
                enhance=step.enhance,
                use_sc_context=step.use_sc_context,
                pack_side_chains=step.pack_side_chains,
                sc_denoising_steps=step.sc_denoising_steps,
                batch_size=step.batch_size,
            )

            new_pdbs = self._run_mpnn_step(mpnn_step, input_pdbs_sorted, fixed_residues)
            new_samples = []
            for pdb in new_pdbs:
                try:
                    _, atoms = read_pdb_atoms(pdb)
                    seq = get_sequence_from_atoms(atoms)
                    new_samples.append((seq, pdb, max(step.temperatures)))
                except Exception:
                    pass

            # Merge and re-dedupe
            all_samples = [(s, p, 0.0) for s, p in zip(unique_seqs, unique_pdbs)]
            all_samples.extend(new_samples)
            unique_seqs, unique_pdbs, unique_metrics = self._cross_temp_dedupe(all_samples)

            LOGGER.info(f"    Flex fallback generated {len(new_pdbs)} structures, now have {len(unique_seqs)} unique")

        # Phase 4: Clustering (if over target)
        if len(unique_seqs) > target_count:
            from .mpnn_runner import cluster_sequences_hierarchical

            LOGGER.info(f"  Clustering {len(unique_seqs)} -> {target_count}")
            clusters = cluster_sequences_hierarchical(unique_seqs, target_count)

            selected_indices = []
            for cluster in clusters:
                if len(cluster) == 1:
                    selected_indices.append(cluster[0])
                else:
                    # Pick best geometry from cluster
                    best_idx = min(cluster, key=lambda i: unique_metrics[i].get(
                        "bond_geometry", {}).get("bond_length_geometry", {}).get(
                        "unconstrained_only", {}).get("max", float('inf')))
                    selected_indices.append(best_idx)

            unique_seqs = [unique_seqs[i] for i in selected_indices]
            unique_pdbs = [unique_pdbs[i] for i in selected_indices]

        # Phase 5: Final assertion
        final_set = set(unique_seqs)
        if len(final_set) != len(unique_seqs):
            LOGGER.error(f"  DUPLICATE SEQUENCES in final output!")
            # Auto-dedupe
            seen = set()
            deduped_pdbs = []
            for seq, pdb in zip(unique_seqs, unique_pdbs):
                if seq not in seen:
                    seen.add(seq)
                    deduped_pdbs.append(pdb)
            unique_pdbs = deduped_pdbs

        if len(unique_pdbs) < target_count:
            LOGGER.warning(
                f"  Final diversify produced {len(unique_pdbs)} unique sequences "
                f"(target={target_count}). Consider increasing temperatures, "
                "designable region, or reducing constraints."
            )
        LOGGER.info(f"  Final diversify output: {len(unique_pdbs)} unique sequences")
        return unique_pdbs

    def _sort_pdbs_by_geometry(self, pdbs: List[str]) -> List[str]:
        """Sort PDBs by geometry score (best/lowest first)."""
        if not pdbs:
            return pdbs

        scored = []
        for pdb in pdbs:
            try:
                cached = self._get_cached_bond_rmsd_metrics(pdb)
                score = cached.get("bond_max")
                if score is None:
                    score = float('inf')
                scored.append((score, pdb))
            except Exception:
                scored.append((float('inf'), pdb))

        scored.sort(key=lambda x: x[0])
        return [pdb for _, pdb in scored]

    def _cross_temp_dedupe(
        self,
        samples: List[Tuple[str, str, float]],
    ) -> Tuple[List[str], List[str], List[Dict]]:
        """Deduplicate samples across temperatures, keeping best geometry."""
        from collections import defaultdict

        seq_to_candidates = defaultdict(list)
        for seq, pdb, temp in samples:
            seq_to_candidates[seq].append((pdb, temp))

        unique_seqs, unique_pdbs, unique_metrics = [], [], []
        for seq, candidates in seq_to_candidates.items():
            best_pdb = None
            best_metrics: Dict[str, Any] = {}
            best_score = float("inf")
            best_angle = float("inf")
            best_rmsd = float("inf")

            for pdb, _temp in candidates:
                try:
                    cached = self._get_cached_bond_rmsd_metrics(pdb)
                    bond_score = cached.get("bond_max")
                    angle_score = cached.get("angle_max")
                    rmsd_score = cached.get("ca_rmsd")
                except Exception:
                    bond_score = None
                    angle_score = None
                    rmsd_score = None

                b = bond_score if bond_score is not None else float("inf")
                a = angle_score if angle_score is not None else float("inf")
                r = rmsd_score if rmsd_score is not None else float("inf")

                if (
                    best_pdb is None
                    or b < best_score - 1e-12
                    or (abs(b - best_score) <= 1e-12 and a < best_angle - 1e-12)
                    or (abs(b - best_score) <= 1e-12 and abs(a - best_angle) <= 1e-12 and r < best_rmsd - 1e-12)
                    or (
                        abs(b - best_score) <= 1e-12
                        and abs(a - best_angle) <= 1e-12
                        and abs(r - best_rmsd) <= 1e-12
                        and str(pdb) < str(best_pdb)
                    )
                ):
                    best_score, best_angle, best_rmsd = b, a, r
                    best_pdb = pdb

            if best_pdb is None:
                continue

            try:
                calc = MetricsCalculator(
                    designed_pdb=best_pdb,
                    step02_pdb=self.step02_pdb,
                    step01_pdb=self.step01_pdb,
                    params_files=self.params_files,
                    constrained_atoms=self.constrained_atoms,
                    catres_positions=self.catres_positions,
                    motif_positions=self.motif_positions,
                    ligand_info=(self.ligand_info.chain, self.ligand_info.resname, self.ligand_info.resno)
                    if self.ligand_info else None,
                    bond_length_tolerance=self.bond_length_tolerance,
                    bond_angle_tolerance=self.bond_angle_tolerance,
                    pyrosetta_image=self.pyrosetta_image,
                    use_container_fallback=not self.no_container,
                    container_timeout=self.rosetta_timeout,
                )
                best_metrics = calc.calculate_all_metrics()
            except Exception:
                # Provide minimal metrics for downstream clustering
                if math.isfinite(best_score):
                    best_metrics = {
                        "bond_geometry": {
                            "bond_length_geometry": {
                                "unconstrained_only": {"max": best_score}
                            }
                        }
                    }
                else:
                    best_metrics = {}

            unique_seqs.append(seq)
            unique_pdbs.append(best_pdb)
            unique_metrics.append(best_metrics)

        return unique_seqs, unique_pdbs, unique_metrics

    def _pdb_path_tag(self, pdb_path: str) -> str:
        """Return a short deterministic tag for a PDB path (used for safe filenames)."""
        return hashlib.md5(str(pdb_path).encode("utf-8")).hexdigest()[:8]

    def _make_tagged_output_path(
        self,
        input_pdb: str,
        stage: str,
        out_dir: Optional[Path] = None,
        ext: str = ".pdb",
    ) -> str:
        """Create a deterministic, collision-safe output path.

        The filename grows with protocol history because the stem already contains
        prior stage tags. We append a short path hash to avoid collisions when
        stems repeat across different inputs or parallel branches.
        """
        stem = Path(input_pdb).stem
        tag = self._pdb_path_tag(input_pdb)
        stage_str = str(stage) if stage is not None else "stage"
        filename = f"{stem}.{stage_str}.{tag}{ext}"
        if len(filename) > MAX_FILENAME_LEN:
            stem_hash = hashlib.md5(stem.encode("utf-8")).hexdigest()[:10]
            stage_hash = hashlib.md5(stage_str.encode("utf-8")).hexdigest()[:6]
            short_stem = stem[:32]
            short_stage = stage_str[:20]
            filename = f"{short_stem}~{stem_hash}.{short_stage}~{stage_hash}.{tag}{ext}"
            if len(filename) > MAX_FILENAME_LEN:
                filename = f"{stem_hash}.{stage_hash}.{tag}{ext}"
            LOGGER.warning(
                f"[naming] Shortened filename for stage={stage_str} (len>{MAX_FILENAME_LEN}) -> {filename}"
            )
        return str((out_dir or Path(input_pdb).parent) / filename)

    def _warn_on_basename_collisions(self, pdbs: List[str], context: str) -> None:
        """Warn if multiple PDBs share the same basename (collision risk)."""
        if not pdbs or len(pdbs) < 2:
            return
        stems = [Path(p).stem for p in pdbs]
        counts = Counter(stems)
        dupes = {k: v for k, v in counts.items() if v > 1}
        if dupes:
            top = sorted(dupes.items(), key=lambda x: x[1], reverse=True)[:5]
            LOGGER.warning(
                f"[naming] Potential basename collisions after {context}: "
                f"{', '.join([f'{k}x{v}' for k, v in top])}"
            )

    def _next_mpnn_packed_suffix(self, strategy_tag: Optional[str] = None) -> Tuple[int, str]:
        """Generate a deterministic packed_suffix for MPNN outputs."""
        self._mpnn_call_index += 1
        call_index = self._mpnn_call_index
        suffix = f"_m{call_index:03d}"
        if strategy_tag:
            suffix += f"_{strategy_tag}"
        return call_index, suffix

    def _run_rosetta_relax(
        self,
        input_pdbs: List[str],
        mode: str,
        repeats: int,
        ramp_stages: int,
        cart_bonded_weight: float = 2.0,
        until_converged: bool = False,
        scorefunction: Optional[str] = None,
        coord_cst_weight: Optional[float] = None,
        coord_cst_stdev: Optional[float] = None,
        global_coord_cst_weight: Optional[float] = None,
        global_coord_cst_stdev: Optional[float] = None,
        fa_rep_weight: Optional[float] = None,
        score_term_weights: Optional[Dict[str, float]] = None,
        relax_rounds: Optional[int] = None,
        relax_inner_cycles: Optional[int] = None,
        enable_bond_geometry_min: Optional[bool] = None,
        min_tolerance: Optional[float] = None,
        min_max_iter: Optional[int] = None,
        min_cartesian: Optional[bool] = None,
        min_backbone_rmsd_cutoff: Optional[float] = None,
        mobile_residues: Optional[List[str]] = None,
    ) -> List[str]:
        """Run Rosetta relaxation via subprocess."""
        output_pdbs = []
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        def _read_tail(path: Path, max_chars: int = 2000) -> str:
            try:
                with open(path, "r") as f:
                    f.seek(0, os.SEEK_END)
                    size = f.tell()
                    f.seek(max(size - max_chars, 0))
                    return f.read()
            except Exception:
                return ""

        # Use step-specific or global defaults
        if coord_cst_weight is None:
            coord_cst_weight = self.coord_cst_weight
        if coord_cst_stdev is None:
            coord_cst_stdev = self.coord_cst_stdev
        if global_coord_cst_weight is None:
            global_coord_cst_weight = self.global_coord_cst_weight
        if global_coord_cst_stdev is None:
            global_coord_cst_stdev = self.global_coord_cst_stdev
        if fa_rep_weight is None:
            fa_rep_weight = self.fa_rep_weight
        if relax_rounds is None:
            relax_rounds = self.relax_rounds
        if relax_inner_cycles is None:
            relax_inner_cycles = self.relax_inner_cycles

        # Merge score term overrides (global + one-shot + step-specific)
        merged_score_terms: Dict[str, float] = {}
        if self._score_term_overrides:
            merged_score_terms.update(self._score_term_overrides)
        if self._score_term_overrides_next:
            merged_score_terms.update(self._score_term_overrides_next)
            self._score_term_overrides_next = None
        if self._score_term_overrides_next_reset:
            for term in self._score_term_overrides_next_reset:
                merged_score_terms.pop(term, None)
            self._score_term_overrides_next_reset = None
        if score_term_weights:
            merged_score_terms.update(score_term_weights)

        # Apply special score term overrides to explicit args
        if "coordinate_constraint" in merged_score_terms:
            coord_cst_weight = float(merged_score_terms.pop("coordinate_constraint"))
        if "fa_rep" in merged_score_terms:
            fa_rep_weight = float(merged_score_terms.pop("fa_rep"))
        if "cart_bonded" in merged_score_terms:
            cart_bonded_weight = float(merged_score_terms.pop("cart_bonded"))
        if "global_coord_constraint" in merged_score_terms:
            global_coord_cst_weight = float(merged_score_terms.pop("global_coord_constraint"))

        used_outputs: Set[str] = set()
        for input_pdb in input_pdbs:
            tag = self._pdb_path_tag(input_pdb)
            stage = mode
            output_pdb = self._make_tagged_output_path(input_pdb, stage, out_dir=Path(input_pdb).parent)
            if output_pdb in used_outputs:
                stage = f"{mode}_{len(used_outputs)}"
                output_pdb = self._make_tagged_output_path(input_pdb, stage, out_dir=Path(input_pdb).parent)
            used_outputs.add(output_pdb)
            LOGGER.debug(f"  [{mode}] output_pdb={output_pdb}")

            # Write constraints JSON in output directory (avoid read-only input paths)
            cst_dir = self.output_dir / "constraints"
            cst_dir.mkdir(parents=True, exist_ok=True)
            cst_json = str(cst_dir / f"{Path(input_pdb).stem}.{stage}.{tag}.cst.json")
            with open(cst_json, "w") as f:
                json.dump(self.constrained_atoms, f)

            # Write HIS tautomer map JSON
            # Convert (chain, resno) tuple keys to "chain:resno" string keys for JSON
            his_json = str(cst_dir / f"{Path(input_pdb).stem}.{stage}.{tag}.his_tautomer.json")
            his_map_for_json = {
                f"{chain}:{resno}": tautomer
                for (chain, resno), tautomer in self.his_tautomer_map.items()
            }
            with open(his_json, "w") as f:
                json.dump(his_map_for_json, f)

            # Resolve scorefunction default
            if scorefunction is None:
                if mode == "cartesian":
                    scorefunction = self.scorefunction_cart
                else:
                    scorefunction = self.scorefunction_torsional

            # Decide whether to attempt in-process PyRosetta
            use_in_process = self.rosetta_in_process
            if not use_in_process:
                if not self.pyrosetta_image or not os.path.exists(self.pyrosetta_image):
                    use_in_process = True

            if use_in_process:
                try:
                    from module_utils.pyrosetta_utils import try_import_pyrosetta
                    fallback_paths = [self.pyrosetta_dir] if self.pyrosetta_dir else None
                    if not try_import_pyrosetta(fallback_paths=fallback_paths):
                        raise ImportError("PyRosetta not importable in-process")
                    from . import rosetta_relax as rr
                    result = rr.run_relaxation(
                        pdb_path=input_pdb,
                        params_files=self.params_files,
                        output_path=output_pdb,
                        mode=mode,
                        constraints=self.constrained_atoms,
                        coord_cst_weight=coord_cst_weight,
                        coord_cst_stdev=coord_cst_stdev,
                        global_coord_cst_weight=global_coord_cst_weight,
                        global_coord_cst_stdev=global_coord_cst_stdev,
                        cart_bonded_weight=cart_bonded_weight,
                        repeats=repeats,
                        ramp_stages=ramp_stages,
                        enable_bond_geometry_min=enable_bond_geometry_min if enable_bond_geometry_min is not None else True,
                        scorefunction=scorefunction,
                        fa_rep_weight=fa_rep_weight,
                        score_term_weights=merged_score_terms if merged_score_terms else None,
                        relax_rounds=relax_rounds,
                        relax_inner_cycles=relax_inner_cycles,
                        mobile_residues=mobile_residues,
                        nstruct=1,
                        min_tolerance=min_tolerance or 0.01,
                        min_max_iter=min_max_iter or 200,
                        min_cartesian=min_cartesian,
                        min_backbone_rmsd_cutoff=min_backbone_rmsd_cutoff,
                        his_tautomer_map=self.his_tautomer_map,
                    )
                    result = result  # for symmetry with subprocess path
                    if os.path.exists(output_pdb):
                        final_pdb = output_pdb
                        try:
                            if self.step02_pdb:
                                ligfixed = output_pdb.replace(".pdb", ".ligfixed.pdb")
                                restore_ligand_from_ref(
                                    mpnn_pdb=output_pdb,
                                    ref_pdb=self.step02_pdb,
                                    output_pdb=ligfixed,
                                    ligand_info=(
                                        self.ligand_info.chain,
                                        self.ligand_info.resno,
                                        self.ligand_info.resname,
                                    ) if self.ligand_info else None,
                                )
                                final_pdb = ligfixed
                        except Exception as e:
                            LOGGER.warning(f"Ligand restore after relax failed: {e}")
                        output_pdbs.append(final_pdb)
                        self._pdb_lineage[final_pdb] = input_pdb
                        continue
                except Exception as e:
                    if not self.pyrosetta_image or not os.path.exists(self.pyrosetta_image):
                        LOGGER.error(f"PyRosetta unavailable and container image missing: {e}")
                        output_pdbs.append(input_pdb)
                        continue
                    LOGGER.warning(f"In-process Rosetta failed, falling back to subprocess: {e}")

            # Build command - optionally use apptainer with PyRosetta container
            rosetta_script = os.path.join(SCRIPT_DIR, "rosetta_relax.py")
            if self.no_container:
                cmd = ["python", rosetta_script]
            else:
                cmd = ["apptainer", "exec", self.pyrosetta_image, "python", rosetta_script]

            cmd.extend([
                "--pdb", input_pdb,
                "--output", output_pdb,
                "--mode", mode,
                "--constraints_json", cst_json,
                "--his_tautomer_json", his_json,
                "--coord_cst_weight", str(coord_cst_weight),
                "--coord_cst_stdev", str(coord_cst_stdev),
                "--global_coord_cst_weight", str(global_coord_cst_weight),
                "--global_coord_cst_stdev", str(global_coord_cst_stdev),
                "--repeats", str(repeats),
                "--ramp_stages", str(ramp_stages),
                "--relax_rounds", str(relax_rounds),
            ])

            if relax_inner_cycles is not None:
                cmd.extend(["--relax_inner_cycles", str(relax_inner_cycles)])

            if self.params_files:
                cmd.extend(["--params"] + self.params_files)

            if mobile_residues:
                cmd.extend(["--mobile_residues", ",".join(mobile_residues)])

            # Scorefunction selection
            if scorefunction is None:
                if mode == "cartesian":
                    scorefunction = self.scorefunction_cart
                else:
                    scorefunction = self.scorefunction_torsional
            cmd.extend(["--scorefunction", scorefunction])

            if mode == "cartesian":
                cmd.extend(["--cart_bonded_weight", str(cart_bonded_weight)])

            if fa_rep_weight is not None:
                cmd.extend(["--fa_rep_weight", str(fa_rep_weight)])

            if merged_score_terms:
                score_json = str(cst_dir / f"{Path(input_pdb).stem}.score_terms.json")
                with open(score_json, "w") as f:
                    json.dump(merged_score_terms, f)
                cmd.extend(["--score_term_weights_json", score_json])

            if enable_bond_geometry_min is True:
                cmd.append("--enable_bond_geometry_min")
            elif enable_bond_geometry_min is False:
                cmd.append("--disable_bond_geometry_min")

            if mode == "minimize":
                if min_tolerance is not None:
                    cmd.extend(["--min_tolerance", str(min_tolerance)])
                if min_max_iter is not None:
                    cmd.extend(["--min_max_iter", str(min_max_iter)])
                if min_cartesian:
                    cmd.append("--min_cartesian")
                if min_backbone_rmsd_cutoff is not None:
                    cmd.extend(["--min_backbone_rmsd_cutoff", str(min_backbone_rmsd_cutoff)])

            # Run via subprocess
            try:
                # Respect overall max_runtime if available
                timeout = self.rosetta_timeout
                if self.start_time:
                    remaining = self.max_runtime - (time.time() - self.start_time)
                    if remaining > 0:
                        timeout = min(timeout, remaining)

                log_prefix = f"rosetta_{Path(input_pdb).stem}_{mode}_{time.time_ns()}"
                stdout_path = log_dir / f"{log_prefix}.out"
                stderr_path = log_dir / f"{log_prefix}.err"

                with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
                    result = subprocess.run(
                        cmd,
                        stdout=out_f,
                        stderr=err_f,
                        text=True,
                        timeout=timeout,
                    )
                LOGGER.info(f"  Rosetta logs: {stdout_path.name}, {stderr_path.name}")

                if result.returncode == 0 and os.path.exists(output_pdb):
                    final_pdb = output_pdb
                    # Force ligand atoms (incl. H) to match reference after any PyRosetta output
                    try:
                        if self.step02_pdb:
                            ligfixed = output_pdb.replace(".pdb", ".ligfixed.pdb")
                            restore_ligand_from_ref(
                                mpnn_pdb=output_pdb,
                                ref_pdb=self.step02_pdb,
                                output_pdb=ligfixed,
                                ligand_info=(
                                    self.ligand_info.chain,
                                    self.ligand_info.resno,
                                    self.ligand_info.resname,
                                ) if self.ligand_info else None,
                            )
                            final_pdb = ligfixed
                    except Exception as e:
                        LOGGER.warning(f"Ligand restore after relax failed: {e}")

                    output_pdbs.append(final_pdb)
                    self._pdb_lineage[final_pdb] = input_pdb
                else:
                    LOGGER.error(f"Rosetta relax failed for {input_pdb}")
                    err_tail = _read_tail(stderr_path, 1000)
                    LOGGER.error(err_tail if err_tail else "No stderr")
                    output_pdbs.append(input_pdb)  # Keep input on failure

            except subprocess.TimeoutExpired:
                LOGGER.error(f"Rosetta relax timed out for {input_pdb} (timeout={self.rosetta_timeout}s or remaining runtime)")
                output_pdbs.append(input_pdb)

        return output_pdbs

    def _run_select_best_step(
        self,
        step: SelectBestStep,
        input_pdbs: List[str],
    ) -> List[str]:
        """Select best N structures by specified metric."""
        LOGGER.info(f"  Selecting best {step.n} by {step.metric}")

        if len(input_pdbs) <= step.n:
            return input_pdbs
        selected = self._select_best_structures(
            input_pdbs,
            step.n,
            step.metric,
            step,
        )

        LOGGER.info(f"  Selected {len(selected)} structures")
        return selected

    def _record_metrics_snapshot(
        self,
        step_num: int,
        step: ProtocolStep,
        pdbs: List[str],
        duration: float,
    ) -> None:
        """Record per-step metrics for diagnostics."""
        def serialize_value(value: Any) -> Any:
            if isinstance(value, StepType):
                return value.value
            if isinstance(value, MPNNStep):
                return {k: serialize_value(v) for k, v in value.__dict__.items()}
            if isinstance(value, MPNNMultiStep):
                return {k: serialize_value(v) for k, v in value.__dict__.items()}
            if isinstance(value, list):
                return [serialize_value(v) for v in value]
            if isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            return value

        snapshot = {
            "step_num": step_num,
            "step_type": step.step_type.value,
            "num_structures": len(pdbs),
            "duration_seconds": round(duration, 2),
            "step_params": serialize_value(step.__dict__),
            "structures": [],
        }

        for pdb in pdbs:
            try:
                calc = MetricsCalculator(
                    designed_pdb=pdb,
                    step02_pdb=self.step02_pdb,
                    step01_pdb=self.step01_pdb,
                    params_files=self.params_files,
                    constrained_atoms=self.constrained_atoms,
                    catres_positions=self.catres_positions,
                    motif_positions=self.motif_positions,
                    ligand_info=(self.ligand_info.chain, self.ligand_info.resname, self.ligand_info.resno)
                    if self.ligand_info else None,
                    bond_length_tolerance=self.bond_length_tolerance,
                    bond_angle_tolerance=self.bond_angle_tolerance,
                    pyrosetta_image=self.pyrosetta_image,
                    use_container_fallback=not self.no_container,
                    container_timeout=self.rosetta_timeout,
                )
                metrics = calc.calculate_comprehensive_metrics() if self.debug else calc.calculate_all_metrics()
                snapshot["structures"].append({
                    "pdb": pdb,
                    "metrics": metrics,
                })
            except Exception as e:
                snapshot["structures"].append({
                    "pdb": pdb,
                    "metrics": {"error": str(e)},
                })

        self.metrics_history.append(round_metrics(snapshot))

    def _update_fixed_from_interactions(
        self,
        current_pdbs: List[str],
        current_fixed: List[str],
    ) -> List[str]:
        """Update fixed residues based on favorable interactions."""
        if not self.interaction_analyzer or not current_pdbs:
            return current_fixed

        # Analyze first design (representative)
        try:
            result = self.interaction_analyzer.analyze_mutations(
                designed_pdb=current_pdbs[0],
                original_pdb=self.step02_pdb,
                catres_positions=self.catres_positions,
            )

            new_fixed = self.interaction_analyzer.update_fixed_residues(
                current_fixed, result
            )

            if len(new_fixed) > len(current_fixed):
                added = len(new_fixed) - len(current_fixed)
                LOGGER.info(f"  Added {added} positions to fixed list (interaction conservation)")

            return new_fixed

        except Exception as e:
            LOGGER.warning(f"Interaction analysis failed: {e}")
            return current_fixed

    def _finalize_designs(self, current_pdbs: List[str]) -> List[Dict]:
        """Finalize designs: remove duplicates, calculate metrics, select best.

        If the protocol ended with an MPNN step (no Rosetta after), this method
        ensures all outputs go through full_mpnn_output_restoration() to get:
        - REMARK 666 lines from step02
        - Correct HIS tautomers
        - HETATM records (ligands)
        """
        LOGGER.info("\n" + "=" * 50)
        LOGGER.info("Finalizing designs")
        LOGGER.info("=" * 50)
        self._warn_on_basename_collisions(current_pdbs, "finalize start")

        # Apply final restoration to ensure HIS tautomers are correct
        # This is needed because intermediate MPNN steps don't restore HIS tautomers
        # (MPNN can't handle 5-char residue names like HIS_D)
        LOGGER.info("  Applying final HIS tautomer restoration...")
        restored_pdbs = []
        restored_paths = set()
        collision_count = 0
        for pdb in current_pdbs:
            # Determine output path for final restoration (write into output_dir)
            stem = Path(pdb).stem
            # Avoid basename collisions across MPNN runs (same stem in different folders)
            tag = hashlib.md5(str(pdb).encode("utf-8")).hexdigest()[:8]
            restored_pdb = str(self.output_dir / f"{stem}.{tag}.final.pdb")
            if restored_pdb in restored_paths:
                collision_count += 1
            restored_paths.add(restored_pdb)
            try:
                full_mpnn_output_restoration(
                    mpnn_pdb=pdb,
                    ref_pdb=self.step02_pdb,
                    output_pdb=restored_pdb,
                    original_ref_pdb=self.ref_pdb or self.step02_pdb,  # Ensure ALL REMARK 666 lines
                    restore_his_tautomers=True,  # Always restore HIS tautomers for final output
                )
                restored_pdbs.append(restored_pdb)
                LOGGER.debug(f"    Final restored: {pdb} -> {restored_pdb}")
            except Exception as e:
                LOGGER.warning(f"    Failed to restore {pdb}: {e}")
                restored_pdbs.append(pdb)
        current_pdbs = restored_pdbs
        if collision_count:
            LOGGER.warning(f"  Final restore path collisions: {collision_count}")
        LOGGER.info(f"  Final restored {len(current_pdbs)} structures")

        # Remove duplicate sequences (keep best geometry) using cached metrics
        before = len(current_pdbs)
        unique_pdbs = self._dedupe_pdbs_by_sequence(current_pdbs)
        after = len(unique_pdbs)
        if before and after < before:
            LOGGER.info(f"  Final dedupe: {before} -> {after}")

        # Get sequences for logging/output
        seq_by_pdb: Dict[str, str] = {}
        for pdb in unique_pdbs:
            try:
                _, atoms = read_pdb_atoms(pdb)
                seq_by_pdb[pdb] = get_sequence_from_atoms(atoms)
            except Exception:
                seq_by_pdb[pdb] = ""

        LOGGER.info(f"  Unique sequences: {len(unique_pdbs)}")

        # Select top N by geometry (bond_max)
        scored = []
        for i, pdb in enumerate(unique_pdbs):
            try:
                cached = self._get_cached_bond_rmsd_metrics(pdb)
                score = cached.get("bond_max")
                if score is None:
                    score = float("inf")
                scored.append((score, i, pdb))
            except Exception:
                scored.append((float("inf"), i, pdb))

        scored.sort(key=lambda x: x[0])
        num_to_select = self._resolve_final_target_count(None)
        if len(unique_pdbs) < num_to_select:
            LOGGER.warning(
                f"  Only {len(unique_pdbs)} unique designs available for selection "
                f"(target={num_to_select})."
            )
        selected = scored[:num_to_select]

        # Build final design list
        final_designs = []
        for rank, (score, idx, pdb) in enumerate(selected):
            # Copy to output directory with clean name
            output_pdb = self.output_dir / f"design_{rank:02d}.pdb"

            try:
                import shutil
                shutil.copy(pdb, output_pdb)
                # Clean up the PDB (renumber atoms, fix TER lines, remove CONECT/score tables)
                cleanup_final_pdb(str(output_pdb), str(output_pdb))
            except Exception as e:
                LOGGER.warning(f"Failed to process design {rank}: {e}")
                output_pdb = pdb

            # Compute full metrics only for selected designs
            met = {}
            try:
                calc = MetricsCalculator(
                    designed_pdb=str(output_pdb),
                    step02_pdb=self.step02_pdb,
                    step01_pdb=self.step01_pdb,
                    params_files=self.params_files,
                    constrained_atoms=self.constrained_atoms,
                    catres_positions=self.catres_positions,
                    motif_positions=self.motif_positions,
                    ligand_info=(self.ligand_info.chain, self.ligand_info.resname, self.ligand_info.resno)
                    if self.ligand_info else None,
                    bond_length_tolerance=self.bond_length_tolerance,
                    bond_angle_tolerance=self.bond_angle_tolerance,
                    pyrosetta_image=self.pyrosetta_image,
                    use_container_fallback=not self.no_container,
                    container_timeout=self.rosetta_timeout,
                )
                met = calc.calculate_all_metrics()
            except Exception as e:
                LOGGER.warning(f"Failed to compute metrics for design {rank}: {e}")

            seq = seq_by_pdb.get(pdb, "")
            design = {
                "rank": rank,
                "pdb_path": str(output_pdb),
                "sequence": seq,
                "metrics": met,
            }
            final_designs.append(design)

            LOGGER.info(
                f"  Design {rank}: {met.get('sequence_metrics', {}).get('num_mutations', '?')} mutations, "
                f"bond_dev={score:.4f}"
            )

        return final_designs

    def save_results(self, designs: List[Dict]) -> str:
        """Save final results JSON."""
        elapsed = time.time() - self.start_time

        results = {
            "metadata": {
                "step02_json": os.path.abspath(self.step02_json_path),
                "step02_pdb": os.path.abspath(self.step02_pdb),
                "step01_pdb": os.path.abspath(self.step01_pdb) if self.step01_pdb else None,
                "output_dir": str(self.output_dir.absolute()),
                "protocol": self.protocol_str,
                "mpnn_spheres_override": self.mpnn_spheres_override,
                "pyrosetta_image": self.pyrosetta_image,
                "runtime_seconds": round(elapsed, 1),
            },
            "residue_classification": self.residue_classifier.get_summary() if self.residue_classifier else {},
            "output_designs": designs,
            "metrics_history": self.metrics_history,
        }

        results = round_metrics(results)

        output_json = self.output_dir / "fastmpnn_design_results.json"
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)

        # Print final summary
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info("FINAL SUMMARY")
        LOGGER.info(f"{'='*60}")
        LOGGER.info(f"  Total runtime: {format_time(elapsed)}")
        LOGGER.info(f"  Output designs: {len(designs)}")
        if designs:
            # Show mutation statistics
            mutations = [d.get('metrics', {}).get('sequence_metrics', {}).get('num_mutations', 0) for d in designs]
            if mutations:
                LOGGER.info(f"  Mutations per design: min={min(mutations)}, max={max(mutations)}, avg={sum(mutations)/len(mutations):.1f}")
            # Show geometry statistics
            geom_scores = []
            for d in designs:
                try:
                    score = d.get('metrics', {}).get('bond_geometry', {}).get('bond_length_geometry', {}).get('unconstrained_only', {}).get('max', None)
                    if score is not None:
                        geom_scores.append(score)
                except:
                    pass
            if geom_scores:
                LOGGER.info(f"  Bond deviation (max): min={min(geom_scores):.4f}A, max={max(geom_scores):.4f}A")
        LOGGER.info(f"  Results: {output_json}")
        LOGGER.info(f"{'='*60}")

        return str(output_json)

    def cleanup_intermediates(self) -> Dict:
        """Remove intermediate files and directories.

        Cleans up:
        - mpnn_* directories (MPNN intermediate outputs)
        - constraints/ directory
        - Intermediate .pdb and .metrics.*.json files (not design_*.pdb or fastmpnn_design_results.json)

        Returns:
            Dict with cleanup statistics
        """
        import shutil

        stats = {
            "dirs_removed": 0,
            "files_removed": 0,
        }

        if not self.output_dir.exists():
            return stats

        # Files to keep (final outputs)
        keep_patterns = {
            "design_",  # design_00.pdb, design_01.pdb, etc.
            "fastmpnn_design_results.json",
        }

        # Remove mpnn_* directories
        for item in self.output_dir.iterdir():
            if item.is_dir() and item.name.startswith("mpnn_"):
                try:
                    shutil.rmtree(item)
                    stats["dirs_removed"] += 1
                    LOGGER.debug(f"Removed directory: {item}")
                except Exception as e:
                    LOGGER.warning(f"Failed to remove {item}: {e}")

        # Remove constraints directory
        constraints_dir = self.output_dir / "constraints"
        if constraints_dir.exists():
            try:
                shutil.rmtree(constraints_dir)
                stats["dirs_removed"] += 1
                LOGGER.debug(f"Removed directory: {constraints_dir}")
            except Exception as e:
                LOGGER.warning(f"Failed to remove {constraints_dir}: {e}")

        # Remove intermediate files (not matching keep patterns)
        for item in self.output_dir.iterdir():
            if item.is_file():
                # Check if this file should be kept
                should_keep = any(pattern in item.name for pattern in keep_patterns)
                if not should_keep:
                    try:
                        item.unlink()
                        stats["files_removed"] += 1
                        LOGGER.debug(f"Removed file: {item}")
                    except Exception as e:
                        LOGGER.warning(f"Failed to remove {item}: {e}")

        LOGGER.info(f"Cleanup: removed {stats['dirs_removed']} directories, {stats['files_removed']} files")
        return stats

    def run(self) -> str:
        """Main execution entry point.

        Returns:
            Path to output JSON file
        """
        self.initialize()
        designs = self.run_protocol()
        output_json = self.save_results(designs)

        # Clean up intermediate files unless --keep_intermediates was specified
        if not self.keep_intermediates:
            self.cleanup_intermediates()

        return output_json


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Step03: FastMPNN Design with Rosetta Refinement"
    )

    # Required
    parser.add_argument("--step02_json", required=True,
                       help="Step02 metrics JSON (PDB path extracted from it)")
    parser.add_argument("--params", nargs="+", required=True,
                       help="Ligand .params files")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory")

    # Protocol
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL,
                       help=f"Protocol name (basename of JSON file in protocols/ directory). "
                            f"E.g., '--protocol default' loads protocols/default.json. "
                            f"Default: {DEFAULT_PROTOCOL}")
    parser.add_argument("--protocol_file", default=None,
                       help="Explicit path to protocol file (.json or .txt). "
                            "Overrides --protocol if provided.")

    # Design scope
    parser.add_argument("--catres_subset", default=None,
                       help="Override catres subset (comma-separated block indices)")
    parser.add_argument("--design_secondary_sphere", action="store_true",
                       help="Also design secondary sphere")
    parser.add_argument("--design_gly_pro", action="store_true",
                       help="Allow GLY/PRO redesign (default: protected)")
    parser.add_argument("--layer_cuts", type=float, nargs=3,
                       default=None,
                       help="Distance cutoffs for sphere classification [core shell flex]. "
                            "Defines boundaries: DESIGN_CORE (0-core), DESIGN_SHELL (core-shell), "
                            "FLEX (shell-flex), FROZEN (>flex). Default: [6.0, 8.0, 12.0]")
    parser.add_argument("--mpnn_spheres", default=None,
                       help="Override MPNN design spheres globally (comma-separated, e.g. "
                            "'primary,secondary', 'repack', or 'global')")

    # MPNN settings
    parser.add_argument("--mpnn_temperature", type=float,
                       default=DEFAULT_MPNN_TEMPERATURE)
    parser.add_argument("--mpnn_num_designs", type=int,
                       default=DEFAULT_MPNN_NUMBER_OF_BATCHES)
    parser.add_argument("--mpnn_num_designs_after_first", type=int, default=None,
                       help="Override num designs after first MPNN step (legacy branching reduction)")
    parser.add_argument("--mpnn_batch_size", type=int,
                       default=DEFAULT_MPNN_BATCH_SIZE)
    parser.add_argument("--mpnn_omit_aa", default=DEFAULT_OMIT_AA)

    # Rosetta settings
    parser.add_argument("--coord_cst_weight", type=float,
                       default=DEFAULT_COORD_CST_WEIGHT,
                       help="Coordinate constraint weight for catalytic residues (default 750.0)")
    parser.add_argument("--coord_cst_stdev", type=float,
                       default=DEFAULT_COORD_CST_STDEV,
                       help="Coordinate constraint stdev for catalytic residues (default 0.01)")
    parser.add_argument("--global_coord_cst_weight", type=float, default=0.0,
                       help="Coordinate constraint weight for ALL protein atoms (default 0, meaning no global constraints)")
    parser.add_argument("--global_coord_cst_stdev", type=float, default=0.5,
                       help="Stdev for global constraints (default 0.5, looser than catres constraints)")
    parser.add_argument("--scorefunction_cart", default=SCOREFUNCTION_CART,
                       choices=["ref2015_cart", "beta_nov16_cart"],
                       help="Scorefunction for cartesian relax (default ref2015_cart)")
    parser.add_argument("--scorefunction_torsional", default=SCOREFUNCTION_TORSIONAL,
                       choices=["beta_jan25", "ref2015", "beta_nov16"],
                       help="Scorefunction for torsional relax (default beta_jan25)")
    parser.add_argument("--fa_rep_weight", type=float, default=None,
                       help="Override fa_rep weight (default 0.55, try 0.3-1.0 for clash tolerance)")
    parser.add_argument("--cart_bonded_weight", type=float, default=2.0,
                       help="Cartesian bonded weight for cart_bonded score term (default 2.0)")
    parser.add_argument("--relax_rounds", type=int, default=5,
                       help="Number of FastRelax rounds/outer cycles (default 5)")
    parser.add_argument("--relax_inner_cycles", type=int, default=None,
                       help="Number of inner cycles (default varies by mode)")

    # Convergence
    parser.add_argument("--bond_length_tolerance", type=float,
                       default=DEFAULT_BOND_LENGTH_TOLERANCE)
    parser.add_argument("--bond_angle_tolerance", type=float,
                       default=DEFAULT_BOND_ANGLE_TOLERANCE)

    # References
    parser.add_argument("--step01_pdb", default=None,
                       help="Original step01 PDB (for CA RMSD)")

    # Protocol options
    parser.add_argument("--skip_initial_cart_relax", action="store_true",
                       help="Skip initial cart_relax step if present (step02 already did cartesian relaxation)")

    # Interaction conservation (protocol-driven via keep_interactions steps)

    # Constraint options
    parser.add_argument("--include_bb_hbond_constraints", action="store_true",
                        help="Include backbone atoms (N, CA, C, O, H) in constraints for residues "
                             "with backbone_important_only_for_BB_BB_hbond=True (default: exclude them)")

    # Output
    parser.add_argument("--num_final_designs", type=int, default=None,
                       help="Number of final designs to output (overrides protocol target_count)")

    # Runtime
    parser.add_argument("--max_runtime", type=int, default=7200)
    parser.add_argument("--rosetta_timeout", type=int, default=7200,
                       help="Timeout per Rosetta subprocess (seconds)")
    parser.add_argument("--cart_relax_max_rounds", type=int, default=5,
                       help="Max outer rounds when cart_relax until_converged is enabled")
    parser.add_argument("--pyrosetta_image", default=DEFAULT_PYROSETTA_IMAGE,
                       help="Path to PyRosetta apptainer image (pyrosetta.sif)")
    parser.add_argument("--pyrosetta_dir", default=None,
                       help="Override host PyRosetta install path for in-process fallback")
    parser.add_argument("--rosetta_in_process", action="store_true",
                       help="Run Rosetta in-process (no container) if PyRosetta is available")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging (DEBUG level)")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce logging (WARNING level)")
    parser.add_argument("--test", action="store_true",
                       help="Run in test mode (faster)")
    parser.add_argument("--keep_intermediates", action="store_true",
                       help="Keep intermediate files/directories (default: clean up)")
    parser.add_argument("--no-container", action="store_true",
                       help="Run commands directly without apptainer wrapper "
                            "(use when already inside a container or when deps are available)")

    # MPNN execution controls
    parser.add_argument("--mpnn_use_container", action="store_true",
                       help="Force MPNN to use container runtime (apptainer)")
    parser.add_argument("--mpnn_no_container", action="store_true",
                       help="Disable container for MPNN (run run.py directly)")
    parser.add_argument("--mpnn_use_gpu", action="store_true",
                       help="Force MPNN to use GPU (--nv)")
    parser.add_argument("--mpnn_no_gpu", action="store_true",
                       help="Force MPNN to run without GPU (no --nv)")
    parser.add_argument("--mpnn_container_image", default=None,
                       help="Apptainer image for MPNN (default: universal.sif)")

    # MPNN server controls (persistent model for faster design)
    parser.add_argument("--no-mpnn-server", action="store_true",
                       help="Disable MPNN server (use subprocess for each MPNN call). "
                            "Server mode keeps model weights in memory for ~5-10x speedup.")
    parser.add_argument("--mpnn-server-host", default="localhost",
                       help="MPNN server hostname (default: localhost)")
    parser.add_argument("--mpnn-server-port", type=int, default=5000,
                       help="MPNN server port (default: 5000)")
    parser.add_argument("--no-auto-start-mpnn-server", action="store_true",
                       help="Do not auto-start MPNN server on first call")

    args = parser.parse_args()

    # Logging verbosity
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose or args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate MPNN container/GPU flags
    if args.mpnn_use_container and args.mpnn_no_container:
        sys.exit("Error: --mpnn_use_container and --mpnn_no_container are mutually exclusive")
    if args.mpnn_use_gpu and args.mpnn_no_gpu:
        sys.exit("Error: --mpnn_use_gpu and --mpnn_no_gpu are mutually exclusive")

    mpnn_use_container = None
    if args.mpnn_use_container:
        mpnn_use_container = True
    elif args.mpnn_no_container:
        mpnn_use_container = False

    mpnn_use_gpu = None
    if args.mpnn_use_gpu:
        mpnn_use_gpu = True
    elif args.mpnn_no_gpu:
        mpnn_use_gpu = False

    designer = FastMPNNDesigner(
        step02_json_path=args.step02_json,
        params_files=args.params,
        output_dir=args.output_dir,
        protocol=args.protocol,
        protocol_file=args.protocol_file,
        catres_subset=args.catres_subset,
        design_secondary_sphere=args.design_secondary_sphere,
        design_gly_pro=args.design_gly_pro,
        layer_cuts=args.layer_cuts,
        mpnn_spheres=args.mpnn_spheres,
        mpnn_temperature=args.mpnn_temperature,
        mpnn_num_designs=args.mpnn_num_designs,
        mpnn_num_designs_after_first=args.mpnn_num_designs_after_first,
        mpnn_batch_size=args.mpnn_batch_size,
        mpnn_omit_aa=args.mpnn_omit_aa,
        coord_cst_weight=args.coord_cst_weight,
        coord_cst_stdev=args.coord_cst_stdev,
        global_coord_cst_weight=args.global_coord_cst_weight,
        global_coord_cst_stdev=args.global_coord_cst_stdev,
        scorefunction_cart=args.scorefunction_cart,
        scorefunction_torsional=args.scorefunction_torsional,
        fa_rep_weight=args.fa_rep_weight,
        cart_bonded_weight=args.cart_bonded_weight,
        relax_rounds=args.relax_rounds,
        relax_inner_cycles=args.relax_inner_cycles,
        bond_length_tolerance=args.bond_length_tolerance,
        bond_angle_tolerance=args.bond_angle_tolerance,
        step01_pdb=args.step01_pdb,
        skip_initial_cart_relax=args.skip_initial_cart_relax,
        include_bb_hbond_constraints=args.include_bb_hbond_constraints,
        num_final_designs=args.num_final_designs,
        max_runtime=args.max_runtime,
        rosetta_timeout=args.rosetta_timeout,
        cart_relax_max_rounds=args.cart_relax_max_rounds,
        pyrosetta_image=args.pyrosetta_image,
        no_container=getattr(args, 'no_container', False),
        rosetta_in_process=args.rosetta_in_process,
        pyrosetta_dir=args.pyrosetta_dir,
        debug=args.debug,
        test=args.test,
        keep_intermediates=args.keep_intermediates,
        mpnn_use_container=mpnn_use_container,
        mpnn_use_gpu=mpnn_use_gpu,
        mpnn_container_image=args.mpnn_container_image,
        use_mpnn_server=not args.no_mpnn_server,
        mpnn_server_host=args.mpnn_server_host,
        mpnn_server_port=args.mpnn_server_port,
        auto_start_mpnn_server=not args.no_auto_start_mpnn_server,
    )

    output_json = designer.run()
    LOGGER.info(f"\nStep03 complete! Results: {output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
