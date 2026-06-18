"""
Centralized constants, defaults, and presets for the enzyme design pipeline.

This file contains all configuration values used by run_pipeline.py.
Modify values here to change default behavior without touching the runner code.

To add a new step or modify existing ones, see the README.md section:
"Adapting to Step Script Changes"
"""

from pathlib import Path
from typing import Dict, Any

# =============================================================================
# Container Configuration
# =============================================================================
DEFAULT_CONTAINER_RUNTIME = "apptainer"
# universal.sif contains both PyRosetta (2026.03) and ProteinMPNN
# It has beta_jan25 scorefunction, multi-threading, and serialization support
# Using a single container simplifies execution and avoids nested container overhead
DEFAULT_UNIVERSAL_CONTAINER = "/net/software/containers/universal.sif"
DEFAULT_PYROSETTA_CONTAINER = "/net/software/containers/universal.sif"  # Default to universal.sif

# =============================================================================
# Step Script Paths (relative to project root)
# =============================================================================
STEP1_SCRIPT = "modules/step01__catres_alignment/align_catres.py"
STEP2_SCRIPT = "modules/step02__constrained_cart_relax/constrained_cart_relax.py"
STEP3_SCRIPT = "modules/step03__fastmpnndesign/fastmpnn_design.py"
STEP3_DEFAULT_PROTOCOL_FILE = "modules/step03__fastmpnndesign/protocols/default.json"
STEP3_FAST_PROTOCOL_FILE = "modules/step03__fastmpnndesign/protocols/fast.json"
STEP3_THOROUGH_PROTOCOL_FILE = "modules/step03__fastmpnndesign/protocols/thorough.json"
STEP3_AGGRESSIVE_PROTOCOL_FILE = "modules/step03__fastmpnndesign/protocols/aggressive.json"

# Container mapping for each step
STEP_CONTAINERS = {
    "step1": "universal",   # Uses universal.sif
    "step2": "pyrosetta",   # Uses pyrosetta.sif
    "step3": "pyrosetta",   # Uses pyrosetta.sif
}

# =============================================================================
# Step 1 Defaults: Catalytic Residue Alignment
# =============================================================================
STEP1_DEFAULTS: Dict[str, Any] = {
    "strict_backbone_importance": False,
    "exclude_bb_only_hbond_constraints": False,
    "flex_res_move_all_sc": False,
    "flex_res_constrain_all_sc": False,
}

# Step 1 CLI argument mapping: pipeline_arg -> step_script_arg
STEP1_ARG_MAPPING = {
    "step1_strict_backbone_importance": "strict_backbone_importance",
    "step1_exclude_bb_only_hbond_constraints": "exclude_bb_only_hbond_constraints",
    "step1_flex_res_move_all_sc": "flex_res_move_all_sc",
    "step1_flex_res_constrain_all_sc": "flex_res_constrain_all_sc",
}

# =============================================================================
# Step 2 Defaults: Constrained Cartesian Relaxation
# =============================================================================
STEP2_DEFAULTS: Dict[str, Any] = {
    "coord_cst_weight": 750.0,
    "coord_cst_stdev": 0.01,
    "cart_bonded_weight": 3.0,
    "mobile_radius": 10.0,
    "fastrelax_repeats": 3,
    "fastrelax_ramp_stages": 3,
    "bond_length_tolerance": 0.05,
    "bond_angle_tolerance": 10.0,
    "sequence_neighbor_buffer": 5,
    "max_adaptive_rounds": 10,
    "scorefunction": "ref2015_cart",
    "max_runtime": 3600,
    "cart_bonded_scale_factor": 1.5,
    "cart_bonded_max": 4.0,
    "fa_rep_scale": 0.5,
    "fa_atr_scale": 1.0,
    "fa_elec_scale": 1.0,
    "ramp_fa_rep": False,
    "fa_rep_min_scale": 0.2,
    "auto_expand_mobile": True,
    "expansion_radius": 5.0,
    "max_expansions": 3,
    "catres_bond_tolerance": 0.05,
    "catres_angle_tolerance": 10.0,
    "require_catres_converged": True,
    "enable_bond_geometry_min": False,
}

# Step 2 CLI argument mapping: pipeline_arg -> step_script_arg
STEP2_ARG_MAPPING = {
    "step2_coord_cst_weight": "coord_cst_weight",
    "step2_coord_cst_stdev": "coord_cst_stdev",
    "step2_cart_bonded_weight": "cart_bonded_weight",
    "step2_mobile_radius": "mobile_radius",
    "step2_fastrelax_repeats": "fastrelax_repeats",
    "step2_fastrelax_ramp_stages": "fastrelax_ramp_stages",
    "step2_bond_length_tolerance": "bond_length_tolerance",
    "step2_bond_angle_tolerance": "bond_angle_tolerance",
    "step2_sequence_neighbor_buffer": "sequence_neighbor_buffer",
    "step2_max_adaptive_rounds": "max_adaptive_rounds",
    "step2_scorefunction": "scorefunction",
    "step2_max_runtime": "max_runtime",
    "step2_cart_bonded_scale_factor": "cart_bonded_scale_factor",
    "step2_cart_bonded_max": "cart_bonded_max",
    "step2_fa_rep_scale": "fa_rep_scale",
    "step2_fa_atr_scale": "fa_atr_scale",
    "step2_fa_elec_scale": "fa_elec_scale",
    "step2_ramp_fa_rep": "ramp_fa_rep",
    "step2_fa_rep_min_scale": "fa_rep_min_scale",
    "step2_expansion_radius": "expansion_radius",
    "step2_max_expansions": "max_expansions",
    "step2_catres_bond_tolerance": "catres_bond_tolerance",
    "step2_catres_angle_tolerance": "catres_angle_tolerance",
    "step2_preset": "preset",
}

# =============================================================================
# Step 3 Defaults: FastMPNN Design with Rosetta Refinement
# =============================================================================
STEP3_DEFAULTS: Dict[str, Any] = {
    # Design settings
    "design_secondary_sphere": False,
    "design_gly_pro": False,
    "layer_cuts": None,  # Default [6.0, 8.0, 12.0] in step03
    # MPNN settings
    "mpnn_spheres": None,  # Default: core only
    "mpnn_temperature": 0.1,
    "mpnn_num_designs": 8,
    "mpnn_num_designs_after_first": None,  # Use same as mpnn_num_designs
    "mpnn_batch_size": 1,
    "mpnn_omit_aa": "CM",
    # Constraint settings
    "coord_cst_weight": 750.0,
    "coord_cst_stdev": 0.01,
    "global_coord_cst_weight": 0.0,
    "global_coord_cst_stdev": 0.5,
    # Scorefunction settings (beta_jan25 requires -corrections:beta_jan25 flag, auto-enabled)
    "scorefunction_cart": "ref2015_cart",
    "scorefunction_torsional": "beta_jan25",
    "fa_rep_weight": None,  # Use scorefunction default
    "cart_bonded_weight": 2.0,
    # Relaxation settings
    "relax_rounds": 5,
    "relax_inner_cycles": None,  # Use default
    "bond_length_tolerance": None,  # Use step03 default
    "bond_angle_tolerance": None,  # Use step03 default
    # Backbone H-bond constraints
    "include_bb_hbond_constraints": False,
    # Workflow settings
    "skip_initial_cart_relax": False,
    # Output settings
    "num_final_designs": None,
    "max_runtime": 7200,
    "rosetta_timeout": 7200,
    "cart_relax_max_rounds": 5,
    "keep_intermediates": False,
}

# Step 3 CLI argument mapping: pipeline_arg -> step_script_arg
STEP3_ARG_MAPPING = {
    # Protocol settings
    "step3_protocol": "protocol",
    "step3_protocol_file": "protocol_file",
    # Design settings
    "step3_design_secondary_sphere": "design_secondary_sphere",
    "step3_design_gly_pro": "design_gly_pro",
    "step3_layer_cuts": "layer_cuts",  # 3 floats: [core, shell, flex] cutoffs
    # MPNN settings
    "step3_mpnn_spheres": "mpnn_spheres",
    "step3_mpnn_temperature": "mpnn_temperature",
    "step3_mpnn_num_designs": "mpnn_num_designs",
    "step3_mpnn_num_designs_after_first": "mpnn_num_designs_after_first",
    "step3_mpnn_batch_size": "mpnn_batch_size",
    "step3_mpnn_omit_aa": "mpnn_omit_aa",
    "step3_mpnn_use_gpu": "mpnn_use_gpu",
    "step3_mpnn_no_gpu": "mpnn_no_gpu",
    "step3_mpnn_container_image": "mpnn_container_image",
    # Constraint settings
    "step3_coord_cst_weight": "coord_cst_weight",
    "step3_coord_cst_stdev": "coord_cst_stdev",
    "step3_global_coord_cst_weight": "global_coord_cst_weight",
    "step3_global_coord_cst_stdev": "global_coord_cst_stdev",
    # Scorefunction settings
    "step3_scorefunction_cart": "scorefunction_cart",
    "step3_scorefunction_torsional": "scorefunction_torsional",
    "step3_fa_rep_weight": "fa_rep_weight",
    "step3_cart_bonded_weight": "cart_bonded_weight",
    # Relaxation settings
    "step3_relax_rounds": "relax_rounds",
    "step3_relax_inner_cycles": "relax_inner_cycles",
    "step3_bond_length_tolerance": "bond_length_tolerance",
    "step3_bond_angle_tolerance": "bond_angle_tolerance",
    # Backbone H-bond constraints
    "step3_include_bb_hbond_constraints": "include_bb_hbond_constraints",
    # Workflow settings
    "step3_skip_initial_cart_relax": "skip_initial_cart_relax",
    # Output settings
    "step3_num_final_designs": "num_final_designs",
    "step3_max_runtime": "max_runtime",
    "step3_rosetta_timeout": "rosetta_timeout",
    "step3_cart_relax_max_rounds": "cart_relax_max_rounds",
    "step3_keep_intermediates": "keep_intermediates",
    # MPNN server controls
    "step3_no_mpnn_server": "no-mpnn-server",
    "step3_mpnn_server_host": "mpnn-server-host",
    "step3_mpnn_server_port": "mpnn-server-port",
    "step3_no_auto_start_mpnn_server": "no-auto-start-mpnn-server",
}

# =============================================================================
# Output File Patterns
# =============================================================================
# Glob patterns for finding output files from each step
STEP1_OUTPUT_PATTERNS = {
    "aligned_pdb": "*_aligned.pdb",
    "interactions_json": "*_interactions.json",
    "constraints_json": "*_recommended_atom_cst.json",
}

STEP2_OUTPUT_PATTERNS = {
    "relaxed_pdb": "*_relaxed.pdb",
    "metrics_json": "*_relaxed_metrics.json",
}

STEP3_OUTPUT_PATTERNS = {
    "designs_pdb": "design_*.pdb",
    "results_json": "fastmpnn_design_results.json",
}

# =============================================================================
# Console Output Formatting
# =============================================================================
HEADER_WIDTH = 80
SECTION_CHAR = "="
SUBSECTION_CHAR = "-"

# =============================================================================
# Scorefunction Choices
# =============================================================================
STEP2_SCOREFUNCTION_CHOICES = ["ref2015_cart", "beta_nov16_cart"]
STEP3_SCOREFUNCTION_CART_CHOICES = ["ref2015_cart", "beta_nov16_cart"]
STEP3_SCOREFUNCTION_TORSIONAL_CHOICES = ["beta_jan25", "ref2015", "beta_nov16"]

# =============================================================================
# Helper Functions
# =============================================================================

def get_project_root() -> Path:
    """Get the project root directory (where this file is located)."""
    return Path(__file__).parent.resolve()


def get_step_script_path(step: int) -> Path:
    """Get the absolute path to a step script."""
    scripts = {
        1: STEP1_SCRIPT,
        2: STEP2_SCRIPT,
        3: STEP3_SCRIPT,
    }
    return get_project_root() / scripts[step]


def get_container_type(step: int) -> str:
    """Get the container type for a step ('universal' or 'pyrosetta')."""
    return STEP_CONTAINERS[f"step{step}"]


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


# =============================================================================
# Sweep Configuration: Parameter Metadata
# =============================================================================
# Complete parameter definitions for sweep system.
# Each parameter includes: type, default, description, and valid choices (if applicable)

SWEEP_PARAMETERS: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # Step 1 Parameters
    # -------------------------------------------------------------------------
    "step1_strict_backbone_importance": {
        "type": "bool",
        "default": False,
        "description": "BB-to-BB H-bonds alone don't make backbone important",
        "step": 1,
    },
    "step1_exclude_bb_only_hbond_constraints": {
        "type": "bool",
        "default": False,
        "description": "Exclude BB atoms from constraints for BB-only H-bonds",
        "step": 1,
    },
    "step1_flex_res_move_all_sc": {
        "type": "bool",
        "default": False,
        "description": "For ARG/LYS: move entire sidechain",
        "step": 1,
    },
    "step1_flex_res_constrain_all_sc": {
        "type": "bool",
        "default": False,
        "description": "For ARG/LYS: constrain entire sidechain",
        "step": 1,
    },

    # -------------------------------------------------------------------------
    # Step 2 Parameters
    # -------------------------------------------------------------------------
    "step2_preset": {
        "type": "choice",
        "default": None,
        "choices": ["fast", "balanced", "thorough", "aggressive"],
        "description": "Step 2 preset (overrides individual params)",
        "step": 2,
    },
    "step2_coord_cst_weight": {
        "type": "float",
        "default": 750.0,
        "description": "Coordinate constraint weight",
        "range": [100.0, 2000.0],
        "step": 2,
    },
    "step2_coord_cst_stdev": {
        "type": "float",
        "default": 0.01,
        "description": "Constraint stdev (Angstroms)",
        "range": [0.001, 0.1],
        "step": 2,
    },
    "step2_cart_bonded_weight": {
        "type": "float",
        "default": 3.0,
        "description": "Cart_bonded term weight",
        "range": [0.5, 5.0],
        "step": 2,
    },
    "step2_mobile_radius": {
        "type": "float",
        "default": 10.0,
        "description": "Mobile region radius (Angstroms)",
        "range": [5.0, 20.0],
        "step": 2,
    },
    "step2_fastrelax_repeats": {
        "type": "int",
        "default": 3,
        "description": "FastRelax repeats (M in MxN cycles)",
        "range": [1, 10],
        "step": 2,
    },
    "step2_fastrelax_ramp_stages": {
        "type": "int",
        "default": 3,
        "description": "FastRelax ramp stages (N in MxN cycles)",
        "range": [1, 10],
        "step": 2,
    },
    "step2_bond_length_tolerance": {
        "type": "float",
        "default": 0.05,
        "description": "Bond length convergence tolerance (Angstroms)",
        "range": [0.01, 0.2],
        "step": 2,
    },
    "step2_bond_angle_tolerance": {
        "type": "float",
        "default": 10.0,
        "description": "Bond angle convergence tolerance (degrees)",
        "range": [1.0, 20.0],
        "step": 2,
    },
    "step2_sequence_neighbor_buffer": {
        "type": "int",
        "default": 5,
        "description": "Include residues +/- N from catres",
        "range": [0, 10],
        "step": 2,
    },
    "step2_max_adaptive_rounds": {
        "type": "int",
        "default": 10,
        "description": "Max adaptive relaxation rounds",
        "range": [1, 30],
        "step": 2,
    },
    "step2_scorefunction": {
        "type": "choice",
        "default": "ref2015_cart",
        "choices": STEP2_SCOREFUNCTION_CHOICES,
        "description": "Rosetta scorefunction for Cartesian relaxation",
        "step": 2,
    },
    "step2_max_runtime": {
        "type": "int",
        "default": 3600,
        "description": "Max runtime in seconds",
        "range": [300, 14400],
        "step": 2,
    },
    "step2_cart_bonded_scale_factor": {
        "type": "float",
        "default": 1.5,
        "description": "Cart_bonded scale factor when not converging",
        "range": [1.0, 3.0],
        "step": 2,
    },
    "step2_cart_bonded_max": {
        "type": "float",
        "default": 4.0,
        "description": "Maximum cart_bonded weight cap",
        "range": [3.0, 6.0],
        "step": 2,
    },
    "step2_fa_rep_scale": {
        "type": "float",
        "default": 0.5,
        "description": "Scale factor for fa_rep term",
        "range": [0.1, 1.0],
        "step": 2,
    },
    "step2_fa_atr_scale": {
        "type": "float",
        "default": 1.0,
        "description": "Scale factor for fa_atr term",
        "range": [0.5, 1.5],
        "step": 2,
    },
    "step2_fa_elec_scale": {
        "type": "float",
        "default": 1.0,
        "description": "Scale factor for fa_elec term",
        "range": [0.5, 1.5],
        "step": 2,
    },
    "step2_ramp_fa_rep": {
        "type": "bool",
        "default": False,
        "description": "Ramp fa_rep across adaptive rounds",
        "step": 2,
    },
    "step2_fa_rep_min_scale": {
        "type": "float",
        "default": 0.2,
        "description": "Starting fa_rep scale when ramping",
        "range": [0.1, 0.5],
        "step": 2,
    },
    "step2_auto_expand_mobile": {
        "type": "bool",
        "default": True,
        "description": "Enable automatic mobile region expansion",
        "step": 2,
    },
    "step2_expansion_radius": {
        "type": "float",
        "default": 5.0,
        "description": "Radius for mobile region expansion",
        "range": [2.0, 10.0],
        "step": 2,
    },
    "step2_max_expansions": {
        "type": "int",
        "default": 3,
        "description": "Maximum mobile region expansions",
        "range": [1, 6],
        "step": 2,
    },
    "step2_catres_bond_tolerance": {
        "type": "float",
        "default": 0.05,
        "description": "Catres-specific bond tolerance (Angstroms)",
        "range": [0.02, 0.1],
        "step": 2,
    },
    "step2_catres_angle_tolerance": {
        "type": "float",
        "default": 10.0,
        "description": "Catres-specific angle tolerance (degrees)",
        "range": [2.0, 15.0],
        "step": 2,
    },
    "step2_require_catres_converged": {
        "type": "bool",
        "default": True,
        "description": "Require catres geometry convergence",
        "step": 2,
    },
    "step2_enable_bond_geometry_min": {
        "type": "bool",
        "default": False,
        "description": "Enable bond geometry minimization",
        "step": 2,
    },

    # -------------------------------------------------------------------------
    # Step 3 Parameters
    # -------------------------------------------------------------------------
    "step3_protocol": {
        "type": "str",
        "default": None,
        "description": "Step 3 protocol name (JSON basename in protocols/)",
        "step": 3,
    },
    "step3_design_secondary_sphere": {
        "type": "bool",
        "default": False,
        "description": "Include secondary sphere in design",
        "step": 3,
    },
    "step3_design_gly_pro": {
        "type": "bool",
        "default": False,
        "description": "Allow GLY/PRO redesign",
        "step": 3,
    },
    "step3_layer_cuts": {
        "type": "list_float",
        "default": [6.0, 8.0, 12.0],
        "description": "Layer cutoffs [core, shell, flex] in Angstroms",
        "step": 3,
    },
    "step3_mpnn_spheres": {
        "type": "str",
        "default": None,
        "description": "Override design spheres (e.g., 'core,shell')",
        "step": 3,
    },
    "step3_mpnn_temperature": {
        "type": "float",
        "default": 0.1,
        "description": "MPNN sampling temperature",
        "range": [0.05, 1.0],
        "step": 3,
    },
    "step3_mpnn_num_designs": {
        "type": "int",
        "default": 8,
        "description": "Designs per MPNN round",
        "range": [1, 64],
        "step": 3,
    },
    "step3_mpnn_num_designs_after_first": {
        "type": "int",
        "default": None,
        "description": "Designs per subsequent MPNN rounds (default: same as mpnn_num_designs)",
        "range": [1, 64],
        "step": 3,
    },
    "step3_mpnn_batch_size": {
        "type": "int",
        "default": 1,
        "description": "MPNN batch size",
        "range": [1, 64],
        "step": 3,
    },
    "step3_mpnn_omit_aa": {
        "type": "str",
        "default": "CM",
        "description": "Amino acids to exclude from design",
        "step": 3,
    },
    "step3_coord_cst_weight": {
        "type": "float",
        "default": 750.0,
        "description": "Coordinate constraint weight",
        "range": [100.0, 2000.0],
        "step": 3,
    },
    "step3_coord_cst_stdev": {
        "type": "float",
        "default": 0.01,
        "description": "Constraint stdev",
        "range": [0.001, 0.1],
        "step": 3,
    },
    "step3_global_coord_cst_weight": {
        "type": "float",
        "default": 0.0,
        "description": "Global coordinate constraint weight (all atoms)",
        "range": [0.0, 100.0],
        "step": 3,
    },
    "step3_global_coord_cst_stdev": {
        "type": "float",
        "default": 0.5,
        "description": "Global coordinate constraint stdev",
        "range": [0.1, 2.0],
        "step": 3,
    },
    "step3_scorefunction_cart": {
        "type": "choice",
        "default": "ref2015_cart",
        "choices": STEP3_SCOREFUNCTION_CART_CHOICES,
        "description": "Cartesian scorefunction",
        "step": 3,
    },
    "step3_scorefunction_torsional": {
        "type": "choice",
        "default": "beta_jan25",
        "choices": STEP3_SCOREFUNCTION_TORSIONAL_CHOICES,
        "description": "Torsional scorefunction (beta_jan25 = improved LJ params)",
        "step": 3,
    },
    "step3_fa_rep_weight": {
        "type": "float",
        "default": None,
        "description": "Override fa_rep term weight",
        "range": [0.1, 2.0],
        "step": 3,
    },
    "step3_cart_bonded_weight": {
        "type": "float",
        "default": 2.0,
        "description": "Cart_bonded term weight for Cartesian relaxation",
        "range": [0.5, 5.0],
        "step": 3,
    },
    "step3_relax_rounds": {
        "type": "int",
        "default": 5,
        "description": "Number of FastRelax repeats per relax step",
        "range": [1, 10],
        "step": 3,
    },
    "step3_relax_inner_cycles": {
        "type": "int",
        "default": None,
        "description": "FastRelax inner cycles (default: auto)",
        "range": [1, 10],
        "step": 3,
    },
    "step3_bond_length_tolerance": {
        "type": "float",
        "default": None,
        "description": "Bond length deviation tolerance for geometry checks",
        "range": [0.01, 0.2],
        "step": 3,
    },
    "step3_bond_angle_tolerance": {
        "type": "float",
        "default": None,
        "description": "Bond angle deviation tolerance for geometry checks",
        "range": [1.0, 20.0],
        "step": 3,
    },
    "step3_include_bb_hbond_constraints": {
        "type": "bool",
        "default": False,
        "description": "Include backbone atoms in constraints for BB-BB H-bonds",
        "step": 3,
    },
    "step3_skip_initial_cart_relax": {
        "type": "bool",
        "default": False,
        "description": "Skip initial Cartesian relaxation",
        "step": 3,
    },
    "step3_num_final_designs": {
        "type": "int",
        "default": 10,
        "description": "Number of final designs to output",
        "range": [1, 50],
        "step": 3,
    },
    "step3_max_runtime": {
        "type": "int",
        "default": 7200,
        "description": "Max runtime in seconds",
        "range": [300, 28800],
        "step": 3,
    },
    "step3_rosetta_timeout": {
        "type": "int",
        "default": 7200,
        "description": "Timeout for individual Rosetta subprocess calls",
        "range": [300, 28800],
        "step": 3,
    },
    "step3_cart_relax_max_rounds": {
        "type": "int",
        "default": 5,
        "description": "Max rounds for adaptive Cartesian relaxation",
        "range": [1, 20],
        "step": 3,
    },
    "step3_keep_intermediates": {
        "type": "bool",
        "default": False,
        "description": "Keep intermediate files from step3",
        "step": 3,
    },
}

# =============================================================================
# Example Protocols for Sweeps
# =============================================================================
EXAMPLE_PROTOCOLS: Dict[str, str] = {
    # Basic protocols
    "single_mpnn": "mpnn:T0.2:N8",
    "mpnn_with_relax": "mpnn:T0.2:N8 -> torsional_relax:R1S2",
    "double_mpnn": "mpnn:T0.2:N8 -> torsional_relax:R1S2 -> mpnn:T0.1:N8",

    # Temperature annealing
    "temp_anneal_3step": "mpnn:T0.3:N8 -> torsional_relax:R1S2 -> mpnn:T0.2:N8 -> torsional_relax:R1S2 -> mpnn:T0.1:N8",
    "temp_anneal_2step": "mpnn:T0.25:N16 -> torsional_relax:R2S3 -> mpnn:T0.1:N16",

    # Cartesian vs torsional
    "cart_then_mpnn": "cart_relax:R2S3 -> mpnn:T0.2:N16",
    "torsional_then_mpnn": "torsional_relax:R2S3 -> mpnn:T0.2:N16",
    "mpnn_cart_mpnn": "mpnn:T0.2:N8 -> cart_relax:R2S3 -> mpnn:T0.1:N8",

    # More designs
    "high_diversity": "mpnn:T0.3:N32 -> torsional_relax:R1S2 -> select_best:N16",
    "very_high_diversity": "mpnn:T0.4:N64 -> torsional_relax:R1S2 -> select_best:N32 -> mpnn:T0.1:N16",

    # Conservative
    "conservative": "mpnn:T0.1:N4 -> torsional_relax:R2S3",
    "very_conservative": "mpnn:T0.05:N4 -> cart_relax:R3S5",

    # Thorough
    "thorough_design": "cart_relax:R3S5 -> mpnn:T0.2:N16 -> torsional_relax:R2S3 -> mpnn:T0.1:N16 -> cart_relax:R2S3",

    # Secondary sphere
    "with_secondary": "mpnn:T0.2:N8:spheres=primary,secondary -> torsional_relax:R2S3",
    "global_design": "mpnn:T0.3:N16:spheres=global -> torsional_relax:R1S2 -> select_best:N8",
}

# =============================================================================
# Metrics for Sweep Analysis
# =============================================================================
# Metrics to extract from results for ranking and analysis
SWEEP_METRICS: Dict[str, Dict[str, Any]] = {
    # Geometry metrics (lower is better)
    "max_bond_deviation": {
        "source": "step2",
        "path": ["geometry", "unconstrained_only", "bond_lengths", "max_deviation"],
        "direction": "minimize",
        "description": "Maximum bond length deviation (Angstroms)",
    },
    "max_angle_deviation": {
        "source": "step2",
        "path": ["geometry", "unconstrained_only", "bond_angles", "max_deviation"],
        "direction": "minimize",
        "description": "Maximum bond angle deviation (degrees)",
    },
    "mean_bond_deviation": {
        "source": "step2",
        "path": ["geometry", "unconstrained_only", "bond_lengths", "mean_deviation"],
        "direction": "minimize",
        "description": "Mean bond length deviation (Angstroms)",
    },
    "mean_angle_deviation": {
        "source": "step2",
        "path": ["geometry", "unconstrained_only", "bond_angles", "mean_deviation"],
        "direction": "minimize",
        "description": "Mean bond angle deviation (degrees)",
    },

    # Design metrics
    "num_mutations": {
        "source": "step3",
        "path": ["best_design", "num_mutations"],
        "direction": "neutral",
        "description": "Number of mutations in best design",
    },
    "sequence_identity": {
        "source": "step3",
        "path": ["best_design", "sequence_identity"],
        "direction": "neutral",
        "description": "Sequence identity to original",
    },
    "total_score": {
        "source": "step3",
        "path": ["best_design", "total_score"],
        "direction": "minimize",
        "description": "Rosetta total score",
    },
    "ca_rmsd": {
        "source": "step3",
        "path": ["best_design", "ca_rmsd"],
        "direction": "minimize",
        "description": "CA RMSD to reference",
    },

    # Timing
    "step2_duration": {
        "source": "step2",
        "path": ["duration"],
        "direction": "minimize",
        "description": "Step 2 runtime (seconds)",
    },
    "step3_duration": {
        "source": "step3",
        "path": ["duration"],
        "direction": "minimize",
        "description": "Step 3 runtime (seconds)",
    },
    "total_duration": {
        "source": "pipeline",
        "path": ["timings", "total"],
        "direction": "minimize",
        "description": "Total pipeline runtime (seconds)",
    },
}
