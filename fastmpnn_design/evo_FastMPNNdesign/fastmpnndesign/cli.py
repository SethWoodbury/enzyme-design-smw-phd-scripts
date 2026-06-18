#!/usr/bin/env python
"""
Command-line interface for fastmpnndesign.

Entry point for the iterative protein design pipeline.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional

# Add package directory to path for direct script execution
_SCRIPT_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _SCRIPT_DIR.parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

from fastmpnndesign.config import (
    RunConfig, MPNNConfig, ConstraintConfig, RelaxConfig,
    CatresConfig, PipelineConfig, SlurmConfig
)
from fastmpnndesign.constants import (
    DEFAULT_MPNN_RUNNER, DEFAULT_MODEL_TYPE, DEFAULT_ENHANCE_MODEL,
    DEFAULT_TEMPERATURE, DEFAULT_BATCHES, DEFAULT_BATCH_SIZE, DEFAULT_OMIT_AA,
    DEFAULT_SC_DENOISING_STEPS, DEFAULT_APPTAINER_IMAGE,
    DEFAULT_ROSETTA_PATH, DEFAULT_PYROSETTA_PATH, DEFAULT_SCOREFUNCTION,
    DEFAULT_PYROSETTA_IMAGE,
    PRIMARY_CONTACT_CUTOFF, SECONDARY_CONTACT_CUTOFF, METAL_CONTACT_CUTOFF,
    COORD_CST_WEIGHT, COORD_CST_STDEV, MOBILE_RADIUS, CART_BONDED_WEIGHT,
    FASTRELAX_CYCLES, LIGAND_CST_STDEV, ALLOW_CATRES_BB,
    DEFAULT_N_CYCLES, DEFAULT_N_CANDIDATES, DEFAULT_N_KEEP,
    DEFAULT_N_FINAL, DEFAULT_SLURM_TIME, DEFAULT_SLURM_CPUS, DEFAULT_SLURM_MEM
)
from fastmpnndesign.logging_config import setup_logging, get_logger
from fastmpnndesign.orchestrator import run_pipeline, validate_inputs
from fastmpnndesign.slurm import generate_sbatch_script

__version__ = "0.1.0"


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with all options."""
    parser = argparse.ArgumentParser(
        prog="fastmpnndesign",
        description=(
            "Iterative protein active-site design pipeline: "
            "LigandMPNN → Constrained Rosetta Relax → Filter/Rank → Repeat"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with single ligand
  fastmpnndesign --pdb input.pdb --params ligand.params

  # With custom output directory and prefix
  fastmpnndesign --pdb input.pdb --params ligand.params --output_dir ./results --prefix design1

  # Using Apptainer for MPNN execution
  fastmpnndesign --pdb input.pdb --params ligand.params --use_apptainer

  # Run with specific catres subset for tight constraints
  fastmpnndesign --pdb input.pdb --params ligand.params --catres_subset 1,2,6,10

  # Generate sbatch script instead of running
  fastmpnndesign --pdb input.pdb --params ligand.params --generate_sbatch

  # Dry run to see commands without executing
  fastmpnndesign --pdb input.pdb --params ligand.params --dry_run
"""
    )

    # Required arguments
    required = parser.add_argument_group("Required Arguments")
    required.add_argument(
        "--pdb", type=Path, required=True,
        help="Input PDB file with REMARK 666 lines defining catalytic residues"
    )
    required.add_argument(
        "--params", type=Path, action="append", required=True,
        help="Ligand params file(s) for Rosetta (can specify multiple times: --params X.params --params Y.params)"
    )

    # Output options
    output = parser.add_argument_group("Output Options")
    output.add_argument(
        "--output_dir", type=Path, default=Path("./fastmpnn_output"),
        help="Output directory (default: ./fastmpnn_output)"
    )
    output.add_argument(
        "--prefix", type=str, default="design",
        help="Output file prefix (default: design)"
    )

    # MPNN options
    mpnn = parser.add_argument_group("MPNN Options")
    mpnn.add_argument(
        "--mpnn_runner", type=Path, default=Path(DEFAULT_MPNN_RUNNER),
        help=f"Path to MPNN run.py (default: {DEFAULT_MPNN_RUNNER})"
    )
    mpnn.add_argument(
        "--model_type", type=str, default=DEFAULT_MODEL_TYPE,
        help=f"MPNN model type (default: {DEFAULT_MODEL_TYPE})"
    )
    mpnn.add_argument(
        "--enhance", type=str, default=DEFAULT_ENHANCE_MODEL,
        help=f"Enhancement model (default: {DEFAULT_ENHANCE_MODEL})"
    )
    mpnn.add_argument(
        "--no_enhance", action="store_true",
        help="Disable enhancement model"
    )
    mpnn.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help=f"MPNN sampling temperature (default: {DEFAULT_TEMPERATURE})"
    )
    mpnn.add_argument(
        "--number_of_batches", type=int, default=DEFAULT_BATCHES,
        help=f"Number of MPNN batches (default: {DEFAULT_BATCHES})"
    )
    mpnn.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"MPNN batch size (default: {DEFAULT_BATCH_SIZE})"
    )
    mpnn.add_argument(
        "--pack_side_chains", action="store_true", default=True,
        help="Enable side chain packing (default: True)"
    )
    mpnn.add_argument(
        "--no_pack_side_chains", action="store_true",
        help="Disable side chain packing"
    )
    mpnn.add_argument(
        "--sc_num_denoising_steps", type=int, default=DEFAULT_SC_DENOISING_STEPS,
        help=f"Side chain denoising steps (default: {DEFAULT_SC_DENOISING_STEPS})"
    )
    mpnn.add_argument(
        "--omit_AA", type=str, default=DEFAULT_OMIT_AA,
        help=f"Amino acids to omit (default: {DEFAULT_OMIT_AA})"
    )
    mpnn.add_argument(
        "--use_apptainer", action="store_true",
        help="Execute MPNN via Apptainer container"
    )
    mpnn.add_argument(
        "--apptainer_image", type=Path, default=Path(DEFAULT_APPTAINER_IMAGE),
        help=f"Apptainer image path (default: {DEFAULT_APPTAINER_IMAGE})"
    )

    # Catalytic residue options
    catres = parser.add_argument_group("Catalytic Residue Options")
    catres.add_argument(
        "--catres_subset", type=str, default=None,
        help="Comma-separated catres indices for tight constraints (default: ALL)"
    )
    catres.add_argument(
        "--redesign_non_subset_catres", action="store_true",
        help="Allow redesigning catres not in subset (default: False)"
    )

    # Constraint options
    constraints = parser.add_argument_group("Constraint Options")
    constraints.add_argument(
        "--ref_pdb", type=Path, default=None,
        help="Reference PDB for ideal geometry (optional)"
    )
    constraints.add_argument(
        "--cst_file", type=Path, default=None,
        help="Legacy Rosetta constraint file (optional)"
    )
    constraints.add_argument(
        "--primary_contact_cutoff", type=float, default=PRIMARY_CONTACT_CUTOFF,
        help=f"Primary contact distance cutoff in Angstroms (default: {PRIMARY_CONTACT_CUTOFF})"
    )
    constraints.add_argument(
        "--secondary_contact_cutoff", type=float, default=SECONDARY_CONTACT_CUTOFF,
        help=f"Secondary contact distance cutoff in Angstroms (default: {SECONDARY_CONTACT_CUTOFF})"
    )
    constraints.add_argument(
        "--metal_cutoff", type=float, default=METAL_CONTACT_CUTOFF,
        help=f"Metal coordination cutoff in Angstroms (default: {METAL_CONTACT_CUTOFF})"
    )
    constraints.add_argument(
        "--contact_cutoff", type=float, default=None,
        help="Set both primary and secondary contact cutoffs"
    )
    constraints.add_argument(
        "--coord_cst_weight", type=float, default=COORD_CST_WEIGHT,
        help=f"Coordinate constraint weight (default: {COORD_CST_WEIGHT})"
    )
    constraints.add_argument(
        "--coord_cst_stdev", type=float, default=COORD_CST_STDEV,
        help=f"Coordinate constraint stdev in Angstroms (default: {COORD_CST_STDEV})"
    )

    # Relax options
    relax = parser.add_argument_group("Relaxation Options")
    relax.add_argument(
        "--rosetta_path", type=Path, default=Path(DEFAULT_ROSETTA_PATH),
        help=f"Rosetta installation path (default: {DEFAULT_ROSETTA_PATH})"
    )
    relax.add_argument(
        "--pyrosetta_path", type=Path, default=Path(DEFAULT_PYROSETTA_PATH),
        help=f"PyRosetta path (default: {DEFAULT_PYROSETTA_PATH})"
    )
    relax.add_argument(
        "--scorefunction", type=str, default=DEFAULT_SCOREFUNCTION,
        help=f"Rosetta scorefunction (default: {DEFAULT_SCOREFUNCTION})"
    )
    relax.add_argument(
        "--use_pyrosetta", action="store_true", default=True,
        help="Use PyRosetta for relaxation (default: True)"
    )
    relax.add_argument(
        "--use_pyrosetta_image", action="store_true", default=True,
        help="Run relax inside pyrosetta.sif container for beta_jan25 support (default: True)"
    )
    relax.add_argument(
        "--no_pyrosetta_image", action="store_true",
        help="Disable container-based relax (use local PyRosetta)"
    )
    relax.add_argument(
        "--pyrosetta_image", type=Path, default=Path(DEFAULT_PYROSETTA_IMAGE),
        help=f"PyRosetta container image path (default: {DEFAULT_PYROSETTA_IMAGE})"
    )
    relax.add_argument(
        "--fastrelax_cycles", type=int, default=FASTRELAX_CYCLES,
        help=f"FastRelax cycles (default: {FASTRELAX_CYCLES})"
    )
    relax.add_argument(
        "--mobile_radius", type=float, default=MOBILE_RADIUS,
        help=f"Mobile region radius in Angstroms (default: {MOBILE_RADIUS})"
    )
    relax.add_argument(
        "--cart_bonded_weight", type=float, default=CART_BONDED_WEIGHT,
        help=f"Cartesian bonded weight for geometry penalty (default: {CART_BONDED_WEIGHT})"
    )
    relax.add_argument(
        "--ligand_cst_stdev", type=float, default=LIGAND_CST_STDEV,
        help=f"Constraint stdev for ligand freezing in Angstroms (default: {LIGAND_CST_STDEV})"
    )
    relax.add_argument(
        "--allow_catres_bb", action="store_true", default=ALLOW_CATRES_BB,
        help="Allow backbone movement for catalytic residues"
    )
    relax.add_argument(
        "--weight_flexibility", action="store_true",
        help="Use B-factor/pLDDT for flexibility weighting"
    )

    # Pipeline options
    pipeline = parser.add_argument_group("Pipeline Options")
    pipeline.add_argument(
        "--n_cycles", type=int, default=DEFAULT_N_CYCLES,
        help=f"Number of design cycles (default: {DEFAULT_N_CYCLES})"
    )
    pipeline.add_argument(
        "--n_candidates", type=int, default=DEFAULT_N_CANDIDATES,
        help=f"Candidates per MPNN run (default: {DEFAULT_N_CANDIDATES})"
    )
    pipeline.add_argument(
        "--n_keep", type=int, default=DEFAULT_N_KEEP,
        help=f"Candidates to keep per cycle (default: {DEFAULT_N_KEEP})"
    )
    pipeline.add_argument(
        "--n_final", type=int, default=DEFAULT_N_FINAL,
        help=f"Final output sequences (default: {DEFAULT_N_FINAL})"
    )
    pipeline.add_argument(
        "--design_shell_radius", type=float, default=12.0,
        help="Radius around ligand for design (residues outside are fixed, default: 12.0 A)"
    )
    pipeline.add_argument(
        "--no_fix_outside_shell", action="store_true",
        help="Allow redesigning all residues (don't fix outside shell)"
    )

    # Execution options
    execution = parser.add_argument_group("Execution Options")
    execution.add_argument(
        "--single_thread", action="store_true",
        help="Disable on-node parallelism"
    )
    execution.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing"
    )
    execution.add_argument(
        "--verbose", action="store_true", default=True,
        help="Verbose output (default: True)"
    )
    execution.add_argument(
        "--quiet", action="store_true",
        help="Minimal output"
    )
    execution.add_argument(
        "--log_file", type=Path, default=None,
        help="Log file path"
    )
    execution.add_argument(
        "--cleanup", action="store_true",
        help="Delete intermediate directories (cycle_*/) after successful completion"
    )

    # SLURM options
    slurm = parser.add_argument_group("SLURM Options")
    slurm.add_argument(
        "--generate_sbatch", action="store_true",
        help="Generate sbatch script instead of running"
    )
    slurm.add_argument(
        "--slurm_partition", type=str, default=None,
        help="SLURM partition"
    )
    slurm.add_argument(
        "--slurm_time", type=str, default=DEFAULT_SLURM_TIME,
        help=f"SLURM time limit (default: {DEFAULT_SLURM_TIME})"
    )
    slurm.add_argument(
        "--slurm_cpus", type=int, default=DEFAULT_SLURM_CPUS,
        help=f"SLURM CPUs (default: {DEFAULT_SLURM_CPUS})"
    )
    slurm.add_argument(
        "--slurm_mem", type=str, default=DEFAULT_SLURM_MEM,
        help=f"SLURM memory (default: {DEFAULT_SLURM_MEM})"
    )

    # Version
    parser.add_argument(
        "--version", action="version",
        version=f"fastmpnndesign {__version__}"
    )

    return parser


def parse_catres_subset(value: Optional[str]) -> Optional[List[int]]:
    """Parse comma-separated catres indices."""
    if value is None or value.upper() == "ALL":
        return None
    try:
        return [int(x.strip()) for x in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid catres_subset: {value}. Expected comma-separated integers."
        )


def build_config(args: argparse.Namespace) -> RunConfig:
    """Build RunConfig from parsed arguments."""
    # Parse catres subset
    catres_subset = parse_catres_subset(args.catres_subset)

    # Handle contact_cutoff shorthand
    primary_cutoff = args.primary_contact_cutoff
    secondary_cutoff = args.secondary_contact_cutoff
    if args.contact_cutoff is not None:
        primary_cutoff = args.contact_cutoff
        secondary_cutoff = args.contact_cutoff

    # Handle enhance model
    enhance_model = args.enhance if not args.no_enhance else None

    # Handle pack_side_chains
    pack_side_chains = args.pack_side_chains and not args.no_pack_side_chains

    # Handle verbose/quiet
    verbose = args.verbose and not args.quiet

    # Handle pyrosetta_image mode
    use_pyrosetta_image = args.use_pyrosetta_image and not args.no_pyrosetta_image

    # Resolve ref_pdb and cst_file to absolute paths if provided
    ref_pdb_resolved = args.ref_pdb.resolve() if args.ref_pdb else None
    cst_file_resolved = args.cst_file.resolve() if args.cst_file else None

    return RunConfig(
        pdb=args.pdb.resolve(),
        params=[p.resolve() for p in args.params],
        output_dir=args.output_dir.resolve(),
        prefix=args.prefix,
        mpnn=MPNNConfig(
            mpnn_runner=args.mpnn_runner,
            model_type=args.model_type,
            enhance_model=enhance_model,
            temperature=args.temperature,
            number_of_batches=args.number_of_batches,
            batch_size=args.batch_size,
            pack_side_chains=pack_side_chains,
            sc_num_denoising_steps=args.sc_num_denoising_steps,
            omit_AA=args.omit_AA,
            use_apptainer=args.use_apptainer,
            apptainer_image=args.apptainer_image
        ),
        constraints=ConstraintConfig(
            primary_contact_cutoff=primary_cutoff,
            secondary_contact_cutoff=secondary_cutoff,
            metal_cutoff=args.metal_cutoff,
            coord_cst_weight=args.coord_cst_weight,
            coord_cst_stdev=args.coord_cst_stdev,
            ref_pdb=ref_pdb_resolved,
            cst_file=cst_file_resolved
        ),
        relax=RelaxConfig(
            rosetta_path=args.rosetta_path,
            pyrosetta_path=args.pyrosetta_path,
            scorefunction=args.scorefunction,
            use_pyrosetta=args.use_pyrosetta,
            fastrelax_cycles=args.fastrelax_cycles,
            mobile_radius=args.mobile_radius,
            cart_bonded_weight=args.cart_bonded_weight,
            coord_cst_weight=args.coord_cst_weight,
            ligand_cst_stdev=args.ligand_cst_stdev,
            allow_catres_bb=args.allow_catres_bb,
            weight_flexibility=args.weight_flexibility,
            use_pyrosetta_image=use_pyrosetta_image,
            pyrosetta_image=args.pyrosetta_image
        ),
        catres=CatresConfig(
            catres_subset=catres_subset,
            redesign_non_subset_catres=args.redesign_non_subset_catres
        ),
        pipeline=PipelineConfig(
            n_cycles=args.n_cycles,
            n_candidates=args.n_candidates,
            n_keep=args.n_keep,
            n_final=args.n_final,
            design_shell_radius=args.design_shell_radius,
            fix_outside_shell=not args.no_fix_outside_shell
        ),
        slurm=SlurmConfig(
            partition=args.slurm_partition,
            time=args.slurm_time,
            cpus=args.slurm_cpus,
            mem=args.slurm_mem
        ),
        single_thread=args.single_thread,
        dry_run=args.dry_run,
        verbose=verbose,
        log_file=args.log_file,
        generate_sbatch=args.generate_sbatch,
        cleanup=args.cleanup
    )


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Build configuration
    try:
        config = build_config(args)
    except Exception as e:
        print(f"Error parsing arguments: {e}", file=sys.stderr)
        return 1

    # Setup logging
    log_file = config.output_dir / "fastmpnndesign.log" if config.log_file is None else config.log_file
    logger = setup_logging(
        verbose=config.verbose,
        log_file=log_file
    )

    logger.info(f"FastMPNN Design v{__version__}")
    logger.info(f"Input PDB: {config.pdb}")
    logger.info(f"Output directory: {config.output_dir}")

    # Validate inputs
    errors = validate_inputs(config)
    if errors:
        for err in errors:
            logger.error(err)
        return 1

    # Generate sbatch if requested
    if config.generate_sbatch:
        sbatch_path = generate_sbatch_script(config)
        logger.info(f"Generated sbatch script: {sbatch_path}")
        logger.info("Submit with: sbatch " + str(sbatch_path))
        return 0

    # Run pipeline
    try:
        state = run_pipeline(config)

        # Summary
        logger.info("=" * 60)
        logger.info("Pipeline Summary")
        logger.info("=" * 60)
        logger.info(f"Total candidates generated: {len(state.all_candidates)}")
        logger.info(f"Final candidates selected: {len(state.final_candidates)}")
        logger.info(f"Outputs in: {config.output_dir}")

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
