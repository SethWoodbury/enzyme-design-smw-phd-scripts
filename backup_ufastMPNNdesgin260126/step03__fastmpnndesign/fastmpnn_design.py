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
5. Conserves favorable interactions with catalytic residues
6. Outputs optimized enzyme variants

Usage:
    python fastmpnn_design.py --step02_json <path> --params <file> [options]

Example:
    python fastmpnn_design.py \\
        --step02_json step02_outputs/relaxed_metrics.json \\
        --params params/LIG.params \\
        --output_dir output/ \\
        --preset balanced \\
        --num_final_designs 10

Verbosity levels:
    --quiet   : Minimal output (errors and final summary only)
    (default) : Moderate output (step progress, key metrics)
    --verbose : Detailed output (timing, all metrics, debug info)
"""
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

# Add module_utils to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, MODULE_DIR)
sys.path.insert(0, SCRIPT_DIR)

# Local imports - support both direct CLI and module execution
try:
    from .protocol_parser import ProtocolParser, ProtocolFileParser, ProtocolValidationError
    from .protocol_parser import PRESETS, MPNNStep, CartRelaxStep, TorsionalRelaxStep
    from .protocol_parser import MinimizeStep, RepackStep, SelectBestStep, StepType
    from .residue_classifier import ResidueClassifier, DesignSphere
    from .pdb_restoration import full_mpnn_output_restoration
    from .mpnn_runner import MPNNRunner, MPNNInput, MPNNResult, create_mpnn_input_from_classifier
    from .interaction_analyzer import InteractionAnalyzer, MutationInteraction
    from .metrics import MetricsCalculator, round_metrics
except ImportError:
    from protocol_parser import ProtocolParser, ProtocolFileParser, ProtocolValidationError
    from protocol_parser import PRESETS, MPNNStep, CartRelaxStep, TorsionalRelaxStep
    from protocol_parser import MinimizeStep, RepackStep, SelectBestStep, StepType
    from residue_classifier import ResidueClassifier, DesignSphere
    from pdb_restoration import full_mpnn_output_restoration
    from mpnn_runner import MPNNRunner, MPNNInput, MPNNResult, create_mpnn_input_from_classifier
    from interaction_analyzer import InteractionAnalyzer, MutationInteraction
    from metrics import MetricsCalculator, round_metrics

from module_utils.pdb_utils import parse_remark_666, read_pdb_atoms
from module_utils.sequence_utils import (
    get_sequence_from_atoms,
    remove_duplicate_sequences,
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
        preset: str = "balanced",
        protocol: Optional[str] = None,
        protocol_file: Optional[str] = None,
        # Design scope
        catres_subset: Optional[str] = None,
        design_secondary_sphere: bool = False,
        design_gly_pro: bool = False,
        layer_cuts: Optional[List[float]] = None,
        # MPNN settings
        mpnn_temperature: float = DEFAULT_MPNN_TEMPERATURE,
        mpnn_num_designs: int = DEFAULT_MPNN_NUMBER_OF_BATCHES,
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
        conserve_favorable_interactions: bool = True,
        conservation_probability: float = 0.5,
        # Constraint options
        include_bb_hbond_constraints: bool = False,
        # Output
        num_final_designs: int = 10,
        # Runtime
        max_runtime: int = 7200,
        debug: bool = False,
        test: bool = False,
    ):
        """Initialize FastMPNN designer.

        Args:
            step02_json_path: Path to step02 output JSON
            params_files: List of ligand .params files
            output_dir: Output directory
            preset: Protocol preset name
            protocol: Custom protocol string (overrides preset)
            protocol_file: Path to protocol file (.json or .txt) - overrides both
                          preset and protocol string if provided
            catres_subset: Override catalytic residue subset
            design_secondary_sphere: Include secondary sphere in design
            design_gly_pro: Allow GLY/PRO redesign
            layer_cuts: Distance cutoffs for sphere classification
            mpnn_temperature: MPNN sampling temperature
            mpnn_num_designs: Number of MPNN designs per round
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
            conserve_favorable_interactions: Enable H-bond keeper
            conservation_probability: Probability to conserve favorable mutations
            include_bb_hbond_constraints: Include backbone atoms (N, CA, C, O, H) in constraints
                                         for residues with backbone_important_only_for_BB_BB_hbond=True
                                         (default: False, excludes these atoms)
            num_final_designs: Number of final designs to output
            max_runtime: Maximum runtime in seconds
            debug: Enable debug output
            test: Run in test mode (faster)
        """
        self.step02_json_path = step02_json_path
        self.params_files = params_files
        self.output_dir = Path(output_dir).resolve()  # Use absolute path to avoid nesting issues
        self.preset = preset
        self.protocol_str = protocol
        self.protocol_file = protocol_file
        self.catres_subset_override = catres_subset
        self.design_secondary_sphere = design_secondary_sphere
        self.design_gly_pro = design_gly_pro
        self.layer_cuts = layer_cuts or DEFAULT_LAYER_CUTS
        self.mpnn_temperature = mpnn_temperature
        self.mpnn_num_designs = mpnn_num_designs
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
        self.num_final_designs = num_final_designs
        self.max_runtime = max_runtime
        self.debug = debug
        self.test = test

        if debug:
            logging.getLogger().setLevel(logging.DEBUG)

        # Parse protocol (priority: protocol_file > protocol string > preset)
        if protocol_file:
            LOGGER.info(f"Loading protocol from file: {protocol_file}")
            self.protocol_steps = ProtocolFileParser.load_from_file(protocol_file)
            self.protocol_str = f"file:{protocol_file}"
        elif protocol:
            self.protocol_steps = ProtocolParser.parse(protocol)
        else:
            self.protocol_steps = ProtocolParser.parse(preset)

        # Will be set during initialization
        self.step02_json = None
        self.step02_pdb = None
        self.ref_pdb = None
        self.residue_classifier = None
        self.ligand_info = None
        self.constrained_atoms = {}
        self.catres_positions = []
        self.mpnn_runner = None
        self.interaction_analyzer = None
        self.start_time = None
        self.metrics_history = []
        self.designs = []  # List of design dicts
        self._last_step_was_mpnn = False  # Track if protocol ended with MPNN

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

        # Log classification summary
        summary = self.residue_classifier.get_summary()
        LOGGER.info(f"  Catalytic residues: {summary['num_catalytic']}")
        LOGGER.info(f"  Primary sphere: {summary['num_primary']}")
        LOGGER.info(f"  Secondary sphere: {summary['num_secondary']}")
        LOGGER.info(f"  Total fixed: {summary['num_fixed']}")

        # Initialize MPNN runner
        self.mpnn_runner = MPNNRunner()

        # Initialize interaction analyzer
        if self.conserve_favorable_interactions:
            self.interaction_analyzer = InteractionAnalyzer(
                conservation_probability=self.conservation_probability,
            )

        # Validate and potentially modify protocol
        self._validate_protocol()

        # Log protocol
        LOGGER.info(f"\nProtocol ({self.preset}):")
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
        if isinstance(last_step, MPNNStep):
            LOGGER.warning("=" * 60)
            LOGGER.warning("PROTOCOL WARNING: Ending with MPNN step (no relaxation)")
            LOGGER.warning("  Final outputs will go through full_mpnn_output_restoration()")
            LOGGER.warning("  to restore REMARK 666, HIS tautomers, and HETATM records.")
            LOGGER.warning("  Consider adding a torsional_relax step for better geometry.")
            LOGGER.warning("=" * 60)
            self._last_step_was_mpnn = True

        # Check if there's at least one MPNN step
        has_mpnn = any(isinstance(s, MPNNStep) for s in self.protocol_steps)
        if not has_mpnn:
            LOGGER.warning("Protocol has no MPNN design steps - only relaxation will occur")

        # Check for sensible ordering
        for i, step in enumerate(self.protocol_steps[:-1]):
            next_step = self.protocol_steps[i + 1]
            # Warn if two MPNN steps in a row without relaxation
            if isinstance(step, MPNNStep) and isinstance(next_step, MPNNStep):
                LOGGER.warning(f"Protocol has consecutive MPNN steps at positions {i+1}-{i+2}")
                LOGGER.warning("  Consider adding relaxation between MPNN steps")

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

            # Update fixed residues if conserving favorable interactions
            if self.conserve_favorable_interactions and isinstance(step, MPNNStep):
                current_fixed = self._update_fixed_from_interactions(current_pdbs, current_fixed)

            step_duration = time.time() - step_start
            step_timings[step.step_type.value].append(step_duration)
            LOGGER.info(f"  Step completed in {format_time(step_duration)} | Structures out: {len(current_pdbs)}")

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

    def _run_mpnn_step(
        self,
        step: MPNNStep,
        input_pdbs: List[str],
        fixed_residues: List[str],
    ) -> List[str]:
        """Run MPNN design step."""
        LOGGER.info(f"  Temperature: {step.temperature}, Designs: {step.num_designs}")
        LOGGER.info(f"  Spheres: {step.design_spheres}")

        output_pdbs = []

        for input_pdb in input_pdbs:
            # Create unique output folder
            mpnn_out = self.output_dir / f"mpnn_{time.time():.0f}"

            # Build MPNN input
            mpnn_input = create_mpnn_input_from_classifier(
                pdb_path=input_pdb,
                out_folder=str(mpnn_out),
                residue_classifier=self.residue_classifier,
                design_spheres=step.design_spheres,
                temperature=step.temperature,
                num_designs=step.num_designs,
                batch_size=step.batch_size,
                omit_aa=self.mpnn_omit_aa,
                additional_fixed=fixed_residues,
            )

            # Run MPNN
            try:
                result = self.mpnn_runner.run(mpnn_input)

                # Restore PDB features for each output
                # NOTE: For intermediate MPNN steps, we do NOT restore HIS tautomers
                # because MPNN cannot properly handle 5-char residue names like HIS_D.
                # HIS tautomers will be restored in _finalize_designs() at the end.
                for pdb in result.output_pdbs:
                    restored_pdb = str(Path(pdb).with_suffix(".restored.pdb"))
                    full_mpnn_output_restoration(
                        mpnn_pdb=pdb,
                        ref_pdb=input_pdb,
                        output_pdb=restored_pdb,
                        restore_his_tautomers=False,  # Don't restore for intermediate steps
                    )
                    output_pdbs.append(restored_pdb)

            except Exception as e:
                LOGGER.error(f"MPNN failed for {input_pdb}: {e}")

        return output_pdbs if output_pdbs else input_pdbs

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
            relax_rounds=step.relax_rounds,
            relax_inner_cycles=step.relax_inner_cycles,
        )

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

        return self._run_rosetta_relax(
            input_pdbs,
            mode="minimize",
            repeats=1,
            ramp_stages=1,
        )

    def _run_repack_step(
        self,
        step: RepackStep,
        input_pdbs: List[str],
    ) -> List[str]:
        """Run repacking step."""
        return self._run_rosetta_relax(
            input_pdbs,
            mode="repack",
            repeats=1,
            ramp_stages=1,
        )

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
        relax_rounds: Optional[int] = None,
        relax_inner_cycles: Optional[int] = None,
    ) -> List[str]:
        """Run Rosetta relaxation via subprocess."""
        output_pdbs = []

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

        for input_pdb in input_pdbs:
            output_pdb = str(Path(input_pdb).with_suffix(f".{mode}.pdb"))

            # Write constraints JSON
            cst_json = str(Path(input_pdb).with_suffix(".cst.json"))
            with open(cst_json, "w") as f:
                json.dump(self.constrained_atoms, f)

            # Build command - use apptainer with PyRosetta container
            rosetta_script = os.path.join(SCRIPT_DIR, "rosetta_relax.py")
            cmd = [
                "apptainer", "exec", DEFAULT_PYROSETTA_IMAGE,
                "python", rosetta_script,
                "--pdb", input_pdb,
                "--output", output_pdb,
                "--mode", mode,
                "--constraints_json", cst_json,
                "--coord_cst_weight", str(coord_cst_weight),
                "--coord_cst_stdev", str(coord_cst_stdev),
                "--global_coord_cst_weight", str(global_coord_cst_weight),
                "--global_coord_cst_stdev", str(global_coord_cst_stdev),
                "--repeats", str(repeats),
                "--ramp_stages", str(ramp_stages),
                "--relax_rounds", str(relax_rounds),
            ]

            if relax_inner_cycles is not None:
                cmd.extend(["--relax_inner_cycles", str(relax_inner_cycles)])

            if self.params_files:
                cmd.extend(["--params"] + self.params_files)

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

            # Run via subprocess
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )

                if result.returncode == 0 and os.path.exists(output_pdb):
                    output_pdbs.append(output_pdb)
                else:
                    LOGGER.error(f"Rosetta relax failed for {input_pdb}")
                    LOGGER.error(result.stderr[-500:] if result.stderr else "No stderr")
                    output_pdbs.append(input_pdb)  # Keep input on failure

            except subprocess.TimeoutExpired:
                LOGGER.error(f"Rosetta relax timed out for {input_pdb}")
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

        # Calculate metrics for each
        scored = []
        for pdb in input_pdbs:
            try:
                calc = MetricsCalculator(
                    designed_pdb=pdb,
                    step02_pdb=self.step02_pdb,
                    params_files=self.params_files,
                )

                if step.metric == "geometry":
                    geom = calc.calculate_bond_geometry()
                    if "error" not in geom:
                        score = geom["bond_length_geometry"]["unconstrained_only"]["max"]
                        scored.append((score, pdb))
                elif step.metric == "score":
                    # Would need PyRosetta - use geometry as fallback
                    geom = calc.calculate_bond_geometry()
                    if "error" not in geom:
                        score = geom["bond_length_geometry"]["unconstrained_only"]["max"]
                        scored.append((score, pdb))
                else:
                    scored.append((0.0, pdb))

            except Exception as e:
                LOGGER.warning(f"Could not score {pdb}: {e}")
                scored.append((float("inf"), pdb))

        # Fallback: if no structures could be scored (e.g., PyRosetta not available),
        # just take the first N structures rather than selecting none
        if not scored and input_pdbs:
            LOGGER.warning(f"  No structures could be scored (PyRosetta may not be available)")
            LOGGER.warning(f"  Falling back to first {step.n} structures")
            selected = input_pdbs[:step.n]
        else:
            # Sort and select best
            scored.sort(key=lambda x: x[0])
            selected = [pdb for score, pdb in scored[:step.n]]

        LOGGER.info(f"  Selected {len(selected)} structures")
        return selected

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

        # Apply final restoration to ensure HIS tautomers are correct
        # This is needed because intermediate MPNN steps don't restore HIS tautomers
        # (MPNN can't handle 5-char residue names like HIS_D)
        LOGGER.info("  Applying final HIS tautomer restoration...")
        restored_pdbs = []
        for pdb in current_pdbs:
            # Determine output path for final restoration
            if pdb.endswith(".restored.pdb"):
                # Replace .restored.pdb with .final.pdb
                restored_pdb = pdb.replace(".restored.pdb", ".final.pdb")
            else:
                restored_pdb = str(Path(pdb).with_suffix(".final.pdb"))
            try:
                full_mpnn_output_restoration(
                    mpnn_pdb=pdb,
                    ref_pdb=self.step02_pdb,
                    output_pdb=restored_pdb,
                    original_ref_pdb=self.step02_pdb,  # Ensure ALL REMARK 666 lines
                    restore_his_tautomers=True,  # Always restore HIS tautomers for final output
                )
                restored_pdbs.append(restored_pdb)
                LOGGER.debug(f"    Final restored: {pdb} -> {restored_pdb}")
            except Exception as e:
                LOGGER.warning(f"    Failed to restore {pdb}: {e}")
                restored_pdbs.append(pdb)
        current_pdbs = restored_pdbs
        LOGGER.info(f"  Final restored {len(current_pdbs)} structures")

        # Get sequences
        sequences = []
        for pdb in current_pdbs:
            try:
                _, atoms = read_pdb_atoms(pdb)
                seq = get_sequence_from_atoms(atoms)
                sequences.append(seq)
            except:
                sequences.append("")

        # Remove duplicates (keep best geometry)
        metrics_list = []
        for pdb in current_pdbs:
            try:
                calc = MetricsCalculator(
                    designed_pdb=pdb,
                    step02_pdb=self.step02_pdb,
                    params_files=self.params_files,
                )
                metrics = calc.calculate_all_metrics()
                metrics_list.append(metrics)
            except:
                metrics_list.append({})

        unique_seqs, unique_pdbs, unique_metrics = remove_duplicate_sequences(
            sequences, current_pdbs, metrics_list, keep_best_geometry=True
        )

        LOGGER.info(f"  Unique sequences: {len(unique_seqs)}")

        # Select top N by geometry
        scored = []
        for i, (seq, pdb, met) in enumerate(zip(unique_seqs, unique_pdbs, unique_metrics)):
            try:
                geom = met.get("bond_geometry", {})
                score = geom.get("bond_length_geometry", {}).get("unconstrained_only", {}).get("max", float("inf"))
                scored.append((score, i, seq, pdb, met))
            except:
                scored.append((float("inf"), i, seq, pdb, met))

        scored.sort(key=lambda x: x[0])
        selected = scored[:self.num_final_designs]

        # Build final design list
        final_designs = []
        for rank, (score, idx, seq, pdb, met) in enumerate(selected):
            # Copy to output directory with clean name
            output_pdb = self.output_dir / f"design_{rank:02d}.pdb"

            try:
                import shutil
                shutil.copy(pdb, output_pdb)
            except:
                output_pdb = pdb

            design = {
                "rank": rank,
                "pdb_path": str(output_pdb),
                "sequence": seq,
                "metrics": met,
            }
            final_designs.append(design)

            LOGGER.info(f"  Design {rank}: {met.get('sequence_metrics', {}).get('num_mutations', '?')} mutations, "
                       f"bond_dev={score:.4f}")

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
                "protocol": self.preset if not self.protocol_str else self.protocol_str,
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

    def run(self) -> str:
        """Main execution entry point.

        Returns:
            Path to output JSON file
        """
        self.initialize()
        designs = self.run_protocol()
        output_json = self.save_results(designs)
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
    parser.add_argument("--preset", default="balanced",
                       choices=list(PRESETS.keys()),
                       help="Protocol preset (default: balanced)")
    parser.add_argument("--protocol", default=None,
                       help="Custom protocol string (overrides preset)")
    parser.add_argument("--protocol_file", default=None,
                       help="Path to protocol file (.json or .txt). Overrides both "
                            "--preset and --protocol if provided. See ProtocolFileParser "
                            "for format documentation.")

    # Design scope
    parser.add_argument("--catres_subset", default=None,
                       help="Override catres subset (comma-separated block indices)")
    parser.add_argument("--design_secondary_sphere", action="store_true",
                       help="Also design secondary sphere")
    parser.add_argument("--design_gly_pro", action="store_true",
                       help="Allow GLY/PRO redesign (default: protected)")
    parser.add_argument("--layer_cuts", type=float, nargs=4,
                       default=None,
                       help="Distance cuts for sphere classification")

    # MPNN settings
    parser.add_argument("--mpnn_temperature", type=float,
                       default=DEFAULT_MPNN_TEMPERATURE)
    parser.add_argument("--mpnn_num_designs", type=int,
                       default=DEFAULT_MPNN_NUMBER_OF_BATCHES)
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

    # Interaction conservation
    parser.add_argument("--conserve_favorable_interactions", action="store_true",
                       default=True,
                       help="Conserve mutations with H-bonds/pi-stacks")
    parser.add_argument("--no_conserve_interactions", action="store_true",
                       help="Disable interaction conservation")
    parser.add_argument("--conservation_probability", type=float, default=0.5)

    # Constraint options
    parser.add_argument("--include_bb_hbond_constraints", action="store_true",
                        help="Include backbone atoms (N, CA, C, O, H) in constraints for residues "
                             "with backbone_important_only_for_BB_BB_hbond=True (default: exclude them)")

    # Output
    parser.add_argument("--num_final_designs", type=int, default=10)

    # Runtime
    parser.add_argument("--max_runtime", type=int, default=7200)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test", action="store_true",
                       help="Run in test mode (faster)")

    args = parser.parse_args()

    # Handle conserve_interactions flag
    conserve = args.conserve_favorable_interactions and not args.no_conserve_interactions

    designer = FastMPNNDesigner(
        step02_json_path=args.step02_json,
        params_files=args.params,
        output_dir=args.output_dir,
        preset=args.preset,
        protocol=args.protocol,
        protocol_file=args.protocol_file,
        catres_subset=args.catres_subset,
        design_secondary_sphere=args.design_secondary_sphere,
        design_gly_pro=args.design_gly_pro,
        layer_cuts=args.layer_cuts,
        mpnn_temperature=args.mpnn_temperature,
        mpnn_num_designs=args.mpnn_num_designs,
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
        conserve_favorable_interactions=conserve,
        conservation_probability=args.conservation_probability,
        include_bb_hbond_constraints=args.include_bb_hbond_constraints,
        num_final_designs=args.num_final_designs,
        max_runtime=args.max_runtime,
        debug=args.debug,
        test=args.test,
    )

    output_json = designer.run()
    LOGGER.info(f"\nStep03 complete! Results: {output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
