"""
Main orchestrator for enhanced_fastmpnndesign pipeline.

Coordinates the entire design workflow, matching the original script behavior.
"""

import sys
import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add package directory to path for standalone execution
_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import RunConfig, CatalyticResidue
from logging_config import get_logger, setup_logging
from remark666 import parse_remark666_from_pdb, get_catres_subset, map_catres_to_seqpos
from catres import get_fixed_catres, get_catres_seqpos_list, build_keep_pos_list
from rosetta_init import initialize_pyrosetta
from scorefunction import setup_scorefunction
from constraints import CSTs, setup_constraints
from layer_detection import detect_design_residues, get_ligand_heavyatoms, get_2nd_layer_fixed_pos
from relax import run_pre_relaxation, thread_sequence_to_pose, fix_catalytic_residue_rotamers, repack_sidechains
from mpnn_bias import apply_bias_config
from protocol import parse_protocol
from task_operations import create_hbond_keeper
from scoring import load_scoring_module, score_design, filter_design, save_scores
from mpnn_runner import run_mpnn_with_library
from utils import ensure_output_dirs, get_pdb_basename, format_residue_list
from constants import FASTMPNN_DESIGN_PATH, DEFAULT_HBOND_ACCEPT_PROBABILITY

logger = get_logger("orchestrator")


@dataclass
class PipelineState:
    """Tracks state across pipeline execution."""
    config: RunConfig
    pose: Any = None
    pre_relaxed_pose: Any = None
    catres_all: List[CatalyticResidue] = field(default_factory=list)
    catres_subset: List[CatalyticResidue] = field(default_factory=list)
    catres_non_subset: List[CatalyticResidue] = field(default_factory=list)
    catres_seqpos: Dict[int, int] = field(default_factory=dict)
    keep_pos: List[int] = field(default_factory=list)
    design_residues: List[int] = field(default_factory=list)
    repack_residues: List[int] = field(default_factory=list)
    do_not_touch_residues: List[int] = field(default_factory=list)
    bias_dict: Dict[str, Dict[str, float]] = field(default_factory=dict)
    scorefxn: Any = None
    cst_mover: Optional[CSTs] = None
    protocol_steps: List = field(default_factory=list)
    task_operations: Dict[str, Any] = field(default_factory=dict)
    scoring_module: Any = None
    ligand_seqpos: int = 0
    heavyatoms: List[str] = field(default_factory=list)


def initialize_pipeline(config: RunConfig) -> PipelineState:
    """
    Initialize the pipeline.

    This corresponds to original script lines 77-359.

    1. Initialize PyRosetta
    2. Setup scorefunction with constraint weights
    3. Parse REMARK 666 for catalytic residues
    4. Setup design/repack layers
    5. Run pre-relaxation
    6. Calculate MPNN bias
    7. Parse protocol
    8. Setup task operations
    """
    logger.info("=" * 70)
    logger.info("  ENHANCED FASTMPNN DESIGN - INITIALIZATION")
    logger.info("=" * 70)

    state = PipelineState(config=config)

    # Step 1: Initialize PyRosetta
    logger.info("Step 1: Initializing PyRosetta")
    initialize_pyrosetta(
        params=config.rosetta.params,
        scorefunction=config.rosetta.scorefunction,
        dalphaball=config.rosetta.dalphaball_path,
        multithreading=config.rosetta.multithreading,
        preserve_header=config.rosetta.preserve_header
    )

    import pyrosetta

    # Step 2: Setup scorefunction
    logger.info("Step 2: Setting up scorefunction")
    state.scorefxn = setup_scorefunction(
        config.rosetta.scorefunction,
        config.protocol.cstfile
    )

    # Step 3: Setup constraints if cstfile provided
    if config.protocol.cstfile:
        logger.info("Step 3: Loading enzyme constraints")
        state.cst_mover = setup_constraints(config.protocol.cstfile, state.scorefxn)

    # Step 4: Load pose
    logger.info(f"Step 4: Loading structure from {config.pdb}")
    state.pose = pyrosetta.pose_from_file(str(config.pdb))

    # Get ligand info
    state.ligand_seqpos = state.pose.size()
    assert state.pose.residue(state.ligand_seqpos).is_ligand(), \
        f"Last residue (seqpos {state.ligand_seqpos}) is not a ligand"
    state.heavyatoms = get_ligand_heavyatoms(state.pose)

    # Step 5: Parse REMARK 666 for catalytic residues
    logger.info("Step 5: Parsing catalytic residues from REMARK 666")
    state.catres_all = parse_remark666_from_pdb(config.pdb)
    state.catres_seqpos = map_catres_to_seqpos(state.catres_all, state.pose)

    # Handle catres subset
    state.catres_subset, state.catres_non_subset = get_catres_subset(
        state.catres_all,
        config.catres.catres_subset
    )

    # Build keep_pos list
    user_keep_pos = config.keep_pos or []
    state.keep_pos = build_keep_pos_list(
        state.catres_all, user_keep_pos, state.pose
    )

    # Step 6: Detect design layers
    logger.info("Step 6: Detecting design/repack layers")
    state.design_residues, state.repack_residues, state.do_not_touch_residues = \
        detect_design_residues(
            state.pose,
            state.keep_pos,
            config.design_pos,
            state.ligand_seqpos,
            config.detect_pocket
        )

    # Step 7: Run pre-relaxation
    logger.info("Step 7: Running pre-relaxation")
    state.pre_relaxed_pose = run_pre_relaxation(
        state.pose,
        state.scorefxn,
        state.cst_mover,
        state.keep_pos,
        config.relax,
        state.ligand_seqpos
    )

    # Step 8: Calculate MPNN bias
    if config.bias.bias_atoms:
        logger.info("Step 8: Calculating MPNN bias positions")
        state.bias_dict = apply_bias_config(
            state.pre_relaxed_pose,
            config.bias,
            state.keep_pos,
            state.ligand_seqpos
        )

    # Step 9: Parse protocol
    logger.info("Step 9: Parsing design protocol")
    if config.protocol.protocol_file:
        state.protocol_steps = parse_protocol(config.protocol.protocol_file)
    elif config.protocol.protocol_text:
        state.protocol_steps = parse_protocol(config.protocol.protocol_text)
    else:
        state.protocol_steps = parse_protocol(None)  # Default protocol

    # Step 10: Setup task operations
    logger.info("Step 10: Setting up task operations")
    catres_seqpos_list = get_catres_seqpos_list(state.catres_all, state.pose)
    hbond_keeper = create_hbond_keeper(
        name="keep_hbonds_to_ligand_and_catres",
        ligand_seqpos=state.ligand_seqpos,
        catres_seqpos=catres_seqpos_list,
        accept_probability=config.protocol.hbond_accept_probability
    )
    state.task_operations["keep_hbonds_to_ligand_and_catres"] = hbond_keeper

    # Step 11: Load scoring module
    if config.scoring.scoring_script:
        logger.info("Step 11: Loading scoring module")
        state.scoring_module = load_scoring_module(config.scoring.scoring_script)

    logger.info("=" * 70)
    logger.info("  INITIALIZATION COMPLETE")
    logger.info("=" * 70)

    return state


def run_design_iteration(
    state: PipelineState,
    iteration: int
) -> List[Any]:
    """
    Run a single design iteration.

    This corresponds to original script lines 364-424.

    Args:
        state: Pipeline state
        iteration: Iteration number (0-indexed)

    Returns:
        List of output poses that pass filters
    """
    config = state.config
    pdbname = get_pdb_basename(config.pdb)
    suffix = f"_{config.suffix}" if config.suffix else ""

    logger.info(f"\n{'='*70}")
    logger.info(f"  DESIGN ITERATION {iteration + 1}/{config.nstruct}")
    logger.info(f"{'='*70}")

    # Check if outputs already exist
    existing = glob.glob(str(config.output_dir / f"{pdbname}{suffix}_{iteration}_*.pdb"))
    if existing:
        logger.info(f"Outputs already exist for iteration {iteration}, skipping")
        return []

    # Clone the pre-relaxed pose
    pose = state.pre_relaxed_pose.clone()

    # Apply constraints
    if state.cst_mover:
        state.cst_mover.remove_cst(pose)
        state.cst_mover.add_cst(pose)

    # Import and run FastMPNNdesign
    if FASTMPNN_DESIGN_PATH not in sys.path:
        sys.path.append(FASTMPNN_DESIGN_PATH)

    import FastMPNNdesign

    # Create FastMPNNdesign instance
    model_type = "enhanced_mpnn_V2" if config.mpnn.use_enhanced_mpnn else config.mpnn.model_type

    fmd = FastMPNNdesign.FastMPNNdesign(
        model_type=model_type,
        params=[str(p) for p in config.rosetta.params],
        scorefxn=state.scorefxn,
        script_file=state.protocol_steps,
        design_positions=state.design_residues,
        repack_positions=state.repack_residues,
        do_not_repack_positions=state.do_not_touch_residues,
        cst_io=state.cst_mover.cst_io() if state.cst_mover else None,
        omit_AA=config.mpnn.omit_AA
    )

    # Apply bias if configured
    if state.bias_dict:
        fmd.set_mpnn_bias_per_residue(state.bias_dict)

    # Add task operations
    for name, taskop in state.task_operations.items():
        fmd.add_task_operation(taskop)

    # Run design
    logger.info("Running FastMPNNdesign...")
    poses = fmd.apply(pose)
    logger.info(f"Generated {len(poses)} designs")

    # Process outputs
    import pyrosetta

    output_poses = []
    for i, p in enumerate(poses):
        design_name = f"{pdbname}{suffix}_{iteration}_{i}"

        # Add constraints
        if state.cst_mover:
            state.cst_mover.add_cst(p)

        # Score the design
        catres_seqpos_list = list(state.catres_seqpos.keys())
        scores_df = score_design(
            p, pyrosetta.get_fa_scorefxn(),
            catres_seqpos_list, state.scoring_module
        )

        # Apply filter if requested
        if config.scoring.apply_filter:
            if not filter_design(scores_df, state.scoring_module):
                logger.info(f"FILTERED: {design_name}")
                continue

        # Save scores
        scores_df.at[0, "description"] = design_name
        scorefile = config.output_dir / f"scores/{pdbname}{suffix}.sc"
        save_scores(scores_df, design_name, scorefile)

        # Dump PDB
        output_path = config.output_dir / f"{design_name}.pdb"
        p.dump_pdb(str(output_path))
        logger.info(f"Saved: {output_path}")

        output_poses.append(p)

        # Run 2nd layer MPNN if requested
        if config.second_layer_mpnn.enabled:
            run_2nd_layer_mpnn(
                p, state, pdbname, suffix, iteration, i
            )

    return output_poses


def run_2nd_layer_mpnn(
    pose: Any,
    state: PipelineState,
    pdbname: str,
    suffix: str,
    iteration: int,
    design_idx: int
) -> None:
    """
    Run 2nd layer MPNN on successful outputs.

    This corresponds to original script lines 401-424.
    """
    config = state.config
    import pyrosetta.distributed.io

    # Get fixed residues for 2nd layer
    fixed_residues = get_2nd_layer_fixed_pos(
        pose, state.ligand_seqpos, state.heavyatoms, state.keep_pos
    )

    # Run at multiple temperatures
    for T in config.second_layer_mpnn.temperatures:
        mpnn_config = config.mpnn
        mpnn_config.temperature = T
        mpnn_config.number_of_batches = config.second_layer_mpnn.number_of_batches
        mpnn_config.batch_size = config.second_layer_mpnn.batch_size

        mpnn_out = run_mpnn_with_library(
            pose, fixed_residues, mpnn_config,
            name=f"{pdbname}{suffix}_{iteration}_{design_idx}"
        )

        if mpnn_out is None:
            continue

        # Thread sequences onto backbone
        for seqid, seq in enumerate(mpnn_out.get("generated_sequences", [])):
            threaded = thread_sequence_to_pose(pose, seq)
            threaded = fix_catalytic_residue_rotamers(
                threaded, pose, state.catres_seqpos
            )

            if state.cst_mover:
                state.cst_mover.add_cst(threaded)

            threaded = repack_sidechains(threaded, state.scorefxn)
            state.scorefxn(threaded)

            # Save
            output_path = config.output_dir / f"seqs/{pdbname}{suffix}_{iteration}_{design_idx}_T{T}_{seqid}.pdb"
            threaded.dump_pdb(str(output_path))
            logger.debug(f"Saved 2nd layer: {output_path}")


def run_pipeline(config: RunConfig) -> PipelineState:
    """
    Run the complete design pipeline.

    Args:
        config: Run configuration

    Returns:
        Final pipeline state
    """
    # Setup logging
    setup_logging(verbose=config.verbose)

    logger.info("Starting Enhanced FastMPNN Design Pipeline")
    logger.info(f"Input: {config.pdb}")
    logger.info(f"Output: {config.output_dir}")

    # Ensure output directories
    ensure_output_dirs(
        config.output_dir,
        do_2nd_layer=config.second_layer_mpnn.enabled
    )

    # Initialize
    state = initialize_pipeline(config)

    # Run design iterations
    for n_iter in range(config.nstruct):
        run_design_iteration(state, n_iter)

    logger.info("\n" + "=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 70)

    return state
