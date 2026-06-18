"""
FastMPNN Design - Iterative protein active-site design pipeline.

A modular Python package for iterative protein design using:
- LigandMPNN/EnhancedMPNN for sequence design
- Constrained Rosetta/PyRosetta relaxation for structure refinement
- Automated constraint generation from REMARK 666 catalytic residues
- Multi-cycle optimization with diversity-aware selection

Example usage:
    from fastmpnndesign import run_pipeline, RunConfig

    config = RunConfig(
        pdb=Path("input.pdb"),
        params=[Path("ligand.params")],
        output_dir=Path("./output")
    )
    state = run_pipeline(config)

Command-line usage:
    fastmpnndesign --pdb input.pdb --params ligand.params --output_dir ./output
"""

__version__ = "0.1.0"
__author__ = "Woodbuse"

# Core configuration
from fastmpnndesign.config import (
    RunConfig,
    MPNNConfig,
    ConstraintConfig,
    RelaxConfig,
    CatresConfig,
    PipelineConfig,
    SlurmConfig,
    CatalyticResidue,
    Contact,
    DesignCandidate,
    CycleResult,
)

# REMARK 666 parsing
from fastmpnndesign.remark666 import (
    parse_remark666_from_pdb,
    parse_remark666_line,
    get_catres_subset,
    catres_to_fixed_residues,
)

# Ligand detection
from fastmpnndesign.ligand import (
    Ligand,
    LigandAtom,
    detect_ligands_from_pdb,
    detect_metals_from_pdb,
    is_ligand_residue,
)

# Contact detection
from fastmpnndesign.contact_detection import (
    detect_contacts_from_pdb,
    detect_contacts,
    get_unique_residues_from_contacts,
)

# Constraint generation
from fastmpnndesign.constraints import (
    ConstraintSet,
    CoordinateConstraint,
    DistanceConstraint,
    generate_constraints,
    generate_constraints_self_derived,
)

# Alignment
from fastmpnndesign.alignment import (
    extract_ligand_coords,
    compute_alignment_transform,
    transform_coordinates,
    align_ref_to_input_by_ligand,
    get_alignment_rmsd,
    detect_ligand_resname,
)

# MPNN runner
from fastmpnndesign.mpnn_runner import (
    run_mpnn,
    MPNNResult,
    build_fixed_residues_json,
    get_packed_pdbs,
)

# Relax runner
from fastmpnndesign.relax_runner import (
    relax_structure,
    RelaxResult,
    init_pyrosetta,
)

# Protonation state handling
from fastmpnndesign.protonation import (
    detect_histidine_protonation_states,
    HistidineProtonation,
    get_protonation_state_dict,
    verify_protonation_states,
    protonation_states_to_json,
    protonation_states_from_json,
)

# Metrics
from fastmpnndesign.metrics import (
    compute_all_metrics,
    compute_geometry_metrics,
    compute_sequence_metrics,
    calculate_sequence_distance,
    CandidateMetrics,
    GeometryMetrics,
    SequenceMetrics,
    SequenceDistanceMetrics,
)

# Filtering
from fastmpnndesign.filtering import (
    select_best,
    RankingStrategy,
    FilterCriteria,
    diversify_selection,
)

# Orchestration
from fastmpnndesign.orchestrator import (
    run_pipeline,
    initialize_pipeline,
    run_design_cycle,
    PipelineState,
)

# SLURM
from fastmpnndesign.slurm import (
    generate_sbatch_script,
    generate_array_sbatch,
    submit_job,
)

# Logging
from fastmpnndesign.logging_config import (
    setup_logging,
    get_logger,
)

# CLI
from fastmpnndesign.cli import main

__all__ = [
    # Version
    "__version__",
    # Config
    "RunConfig",
    "MPNNConfig",
    "ConstraintConfig",
    "RelaxConfig",
    "CatresConfig",
    "PipelineConfig",
    "SlurmConfig",
    "CatalyticResidue",
    "Contact",
    "DesignCandidate",
    "CycleResult",
    # REMARK 666
    "parse_remark666_from_pdb",
    "parse_remark666_line",
    "get_catres_subset",
    "catres_to_fixed_residues",
    # Ligand
    "Ligand",
    "LigandAtom",
    "detect_ligands_from_pdb",
    "detect_metals_from_pdb",
    "is_ligand_residue",
    # Contact
    "detect_contacts_from_pdb",
    "detect_contacts",
    "get_unique_residues_from_contacts",
    # Constraints
    "ConstraintSet",
    "CoordinateConstraint",
    "DistanceConstraint",
    "generate_constraints",
    "generate_constraints_self_derived",
    # Alignment
    "extract_ligand_coords",
    "compute_alignment_transform",
    "transform_coordinates",
    "align_ref_to_input_by_ligand",
    "get_alignment_rmsd",
    "detect_ligand_resname",
    # MPNN
    "run_mpnn",
    "MPNNResult",
    "build_fixed_residues_json",
    "get_packed_pdbs",
    # Relax
    "relax_structure",
    "RelaxResult",
    "init_pyrosetta",
    # Protonation
    "detect_histidine_protonation_states",
    "HistidineProtonation",
    "get_protonation_state_dict",
    "verify_protonation_states",
    "protonation_states_to_json",
    "protonation_states_from_json",
    # Metrics
    "compute_all_metrics",
    "compute_geometry_metrics",
    "compute_sequence_metrics",
    "calculate_sequence_distance",
    "CandidateMetrics",
    "GeometryMetrics",
    "SequenceMetrics",
    "SequenceDistanceMetrics",
    # Filtering
    "select_best",
    "RankingStrategy",
    "FilterCriteria",
    "diversify_selection",
    # Orchestration
    "run_pipeline",
    "initialize_pipeline",
    "run_design_cycle",
    "PipelineState",
    # SLURM
    "generate_sbatch_script",
    "generate_array_sbatch",
    "submit_job",
    # Logging
    "setup_logging",
    "get_logger",
    # CLI
    "main",
]
