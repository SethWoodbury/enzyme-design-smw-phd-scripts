"""
Enhanced FastMPNN Design

A modular, CLI-friendly package for enzyme active site design using
LigandMPNN/EnhancedMPNN with PyRosetta.

This package is a modernized version of fastmpnn_ZnEsterase_SETH_LINKED.py,
providing the same protocol logic with improved:
- Parameterization and CLI options
- Modularity and code organization
- Logging and verbosity
- Robustness and error handling

Usage:
    # As a command-line tool
    enhanced-fastmpnndesign --pdb input.pdb --nstruct 5

    # As a Python library
    from fastmpnndesign import run_pipeline, RunConfig
    config = RunConfig(pdb=Path("input.pdb"), nstruct=5)
    state = run_pipeline(config)
"""

__version__ = "0.1.0"
__author__ = "Enhanced FastMPNN Design Team"

# Core configuration
from .config import (
    RunConfig,
    MPNNConfig,
    CatresConfig,
    BiasConfig,
    RelaxConfig,
    RosettaConfig,
    ProtocolConfig,
    ScoringConfig,
    SecondLayerMPNNConfig,
    LayerConfig,
    CatalyticResidue,
)

# Main orchestrator
from .orchestrator import (
    run_pipeline,
    initialize_pipeline,
    run_design_iteration,
    PipelineState,
)

# REMARK 666 and catalytic residue handling
from .remark666 import (
    parse_remark666_from_pdb,
    parse_remark666_from_pose,
    get_catres_subset,
    catres_to_fixed_residues,
    map_catres_to_seqpos,
)

from .catres import (
    get_fixed_catres,
    get_catres_seqpos_list,
    build_keep_pos_list,
)

# PyRosetta integration
from .rosetta_init import (
    initialize_pyrosetta,
    is_initialized,
    get_init_options,
)

from .scorefunction import (
    create_scorefunction,
    setup_scorefunction,
    setup_cartesian_scorefunction,
)

# Constraints
from .constraints import (
    CSTs,
    setup_constraints,
)

# Layer detection
from .layer_detection import (
    detect_design_residues,
    get_layer_selections,
    get_ligand_heavyatoms,
    get_2nd_layer_fixed_pos,
)

# Relaxation
from .relax import (
    run_pre_relaxation,
    find_clashing_residues,
    mutate_residues,
    repack_sidechains,
    thread_sequence_to_pose,
    fix_catalytic_residue_rotamers,
)

# MPNN
from .mpnn_runner import (
    run_mpnn,
    run_mpnn_with_library,
    build_mpnn_command,
    build_fixed_residues_json,
    MPNNResult,
)

from .mpnn_bias import (
    calculate_bias_positions,
    apply_bias_config,
)

# Protocol
from .protocol import (
    parse_protocol,
    get_default_protocol,
    ProtocolStep,
    ProtocolStepType,
)

from .task_operations import (
    SelectHBondsToResidue,
    create_hbond_keeper,
)

# Scoring
from .scoring import (
    load_scoring_module,
    score_design,
    filter_design,
    save_scores,
    ScoringModule,
)

# Utilities
from .utils import (
    validate_pdb_path,
    get_nproc,
    dump_scorefile,
    ensure_output_dirs,
    get_pdb_basename,
)

from .logging_config import (
    setup_logging,
    get_logger,
    set_level,
    set_quiet,
    set_debug,
)

# Constants
from .constants import (
    DEFAULT_MPNN_RUNNER,
    DEFAULT_MODEL_TYPE,
    DEFAULT_ENHANCE_MODEL,
    DEFAULT_SCOREFUNCTION,
    DEFAULT_PROTOCOL,
    BETA_SCOREFUNCTIONS,
)

# CLI
from .cli import main, create_parser

__all__ = [
    # Version
    "__version__",

    # Configuration
    "RunConfig",
    "MPNNConfig",
    "CatresConfig",
    "BiasConfig",
    "RelaxConfig",
    "RosettaConfig",
    "ProtocolConfig",
    "ScoringConfig",
    "SecondLayerMPNNConfig",
    "LayerConfig",
    "CatalyticResidue",

    # Orchestrator
    "run_pipeline",
    "initialize_pipeline",
    "run_design_iteration",
    "PipelineState",

    # REMARK 666
    "parse_remark666_from_pdb",
    "parse_remark666_from_pose",
    "get_catres_subset",
    "catres_to_fixed_residues",
    "map_catres_to_seqpos",
    "get_fixed_catres",
    "get_catres_seqpos_list",
    "build_keep_pos_list",

    # PyRosetta
    "initialize_pyrosetta",
    "is_initialized",
    "get_init_options",
    "create_scorefunction",
    "setup_scorefunction",
    "setup_cartesian_scorefunction",

    # Constraints
    "CSTs",
    "setup_constraints",

    # Layer detection
    "detect_design_residues",
    "get_layer_selections",
    "get_ligand_heavyatoms",
    "get_2nd_layer_fixed_pos",

    # Relaxation
    "run_pre_relaxation",
    "find_clashing_residues",
    "mutate_residues",
    "repack_sidechains",
    "thread_sequence_to_pose",
    "fix_catalytic_residue_rotamers",

    # MPNN
    "run_mpnn",
    "run_mpnn_with_library",
    "build_mpnn_command",
    "build_fixed_residues_json",
    "MPNNResult",
    "calculate_bias_positions",
    "apply_bias_config",

    # Protocol
    "parse_protocol",
    "get_default_protocol",
    "ProtocolStep",
    "ProtocolStepType",
    "SelectHBondsToResidue",
    "create_hbond_keeper",

    # Scoring
    "load_scoring_module",
    "score_design",
    "filter_design",
    "save_scores",
    "ScoringModule",

    # Utilities
    "validate_pdb_path",
    "get_nproc",
    "dump_scorefile",
    "ensure_output_dirs",
    "get_pdb_basename",
    "setup_logging",
    "get_logger",
    "set_level",
    "set_quiet",
    "set_debug",

    # Constants
    "DEFAULT_MPNN_RUNNER",
    "DEFAULT_MODEL_TYPE",
    "DEFAULT_ENHANCE_MODEL",
    "DEFAULT_SCOREFUNCTION",
    "DEFAULT_PROTOCOL",
    "BETA_SCOREFUNCTIONS",

    # CLI
    "main",
    "create_parser",
]
