#!/usr/bin/env python3
"""
Command-line interface for enhanced_fastmpnndesign.

Provides a fully parameterized CLI matching the original script arguments
plus new MPNN integration options.

Usage:
    apptainer exec /net/software/containers/universal.sif python /path/to/cli.py --pdb input.pdb --nstruct 5
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional

# Add package directory to path for standalone execution
_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from config import (
    RunConfig, MPNNConfig, CatresConfig, BiasConfig,
    RelaxConfig, RosettaConfig, ProtocolConfig, ScoringConfig,
    SecondLayerMPNNConfig, LayerConfig
)
from constants import (
    DEFAULT_MPNN_RUNNER, DEFAULT_MODEL_TYPE, DEFAULT_ENHANCE_MODEL,
    DEFAULT_TEMPERATURE, DEFAULT_BATCHES, DEFAULT_BATCH_SIZE, DEFAULT_OMIT_AA,
    DEFAULT_SC_DENOISING_STEPS, DEFAULT_APPTAINER_IMAGE,
    DEFAULT_BIAS_VALUE, DEFAULT_BIAS_AAS, DEFAULT_HBOND_ACCEPT_PROBABILITY,
    DEFAULT_SCOREFUNCTION, SECOND_LAYER_TEMPERATURES,
    DEFAULT_LAYER_DIST_BB, DEFAULT_LAYER_DIST_SC
)
from orchestrator import run_pipeline
from logging_config import setup_logging, get_logger

__version__ = "0.1.0"


def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser.

    Matches original script lines 77-99 and adds new MPNN options.
    """
    parser = argparse.ArgumentParser(
        prog="enhanced-fastmpnndesign",
        description="Enhanced FastMPNN Design for enzyme active site optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic design run
  python cli.py --pdb input.pdb --nstruct 5

  # With custom params and constraint file
  python cli.py --pdb input.pdb --params ligand.params --cstfile design.cst

  # Using apptainer for MPNN execution
  apptainer exec /net/software/containers/universal.sif python cli.py --pdb input.pdb --use_apptainer

  # Full example with all options
  apptainer exec /net/software/containers/universal.sif python cli.py \\
    --pdb input.pdb \\
    --params ligand.params \\
    --output_dir ./output \\
    --prefix my_design \\
    --temperature 0.1 \\
    --number_of_batches 10 \\
    --catres_subset 1,2,3,4,5 \\
    --scorefunction beta_jan25 \\
    --use_apptainer
        """
    )

    # Required arguments
    required = parser.add_argument_group("Required Arguments")
    required.add_argument(
        "--pdb", type=Path, required=True,
        help="Input PDB file with ligand and REMARK 666 lines"
    )

    # Structure arguments (from original script)
    structure = parser.add_argument_group("Structure Options")
    structure.add_argument(
        "--nstruct", type=int, default=1,
        help="Number of design iterations (default: 1)"
    )
    structure.add_argument(
        "--prefix", type=str, default="",
        help="Prefix for output filenames"
    )
    structure.add_argument(
        "--suffix", type=str, default="",
        help="Suffix for output filenames"
    )
    structure.add_argument(
        "--params", type=Path, nargs="+", action="append",
        help="Ligand and NCAA params file(s) - can be specified multiple times"
    )
    structure.add_argument(
        "--cstfile", type=Path,
        help="Matcher/enzdes constraint file"
    )
    structure.add_argument(
        "--ref_pdb", type=Path,
        help="Reference PDB for alignment/comparison"
    )

    # Design position arguments (from original)
    design = parser.add_argument_group("Design Position Options")
    design.add_argument(
        "--design_pos", type=int, nargs="+",
        help="Specific positions to redesign"
    )
    design.add_argument(
        "--keep_pos", type=int, nargs="+",
        help="Positions to keep fixed (repack allowed)"
    )
    design.add_argument(
        "--detect_pocket", action="store_true",
        help="Algorithmically detect designable positions"
    )

    # Catalytic residue options (NEW)
    catres = parser.add_argument_group("Catalytic Residue Options")
    catres.add_argument(
        "--catres_subset", type=str,
        help="Comma-separated catres indices for tight constraints (default: ALL)"
    )
    catres.add_argument(
        "--redesign_non_subset_catres", action="store_true",
        help="Allow redesigning catres not in subset (catres are fixed by default)"
    )

    # MPNN options (NEW - expanded per spec)
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
        help=f"EnhancedMPNN model (default: {DEFAULT_ENHANCE_MODEL})"
    )
    mpnn.add_argument(
        "--no_enhance", action="store_true",
        help="Disable EnhancedMPNN"
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
        "--pack_side_chains", type=int, default=1, choices=[0, 1],
        help="Enable side chain packing (default: 1)"
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
        "--repack_everything", type=int, default=0, choices=[0, 1],
        help="Repack all sidechains (default: 0)"
    )
    mpnn.add_argument(
        "--use_apptainer", action="store_true",
        help="Execute MPNN via Apptainer container"
    )
    mpnn.add_argument(
        "--apptainer_image", type=Path, default=Path(DEFAULT_APPTAINER_IMAGE),
        help=f"Apptainer image path (default: {DEFAULT_APPTAINER_IMAGE})"
    )
    mpnn.add_argument(
        "--ligand_mpnn_use_side_chain_context", type=int, default=1, choices=[0, 1],
        help="Use side chain context for ligand MPNN (default: 1)"
    )
    mpnn.add_argument(
        "--packed_suffix", type=str, default="_packed_",
        help="Suffix for packed output files (default: _packed_)"
    )

    # Bias options (from original)
    bias = parser.add_argument_group("MPNN Bias Options")
    bias.add_argument(
        "--position_bias", type=float, default=DEFAULT_BIAS_VALUE,
        help=f"Bias for polar AAs near bias_atoms (default: {DEFAULT_BIAS_VALUE})"
    )
    bias.add_argument(
        "--bias_atoms", nargs="+", type=str,
        help="Ligand atom names for bias calculation"
    )
    bias.add_argument(
        "--bias_AAs", type=str, default=DEFAULT_BIAS_AAS,
        help=f"Amino acids to bias (default: {DEFAULT_BIAS_AAS})"
    )

    # Contact/constraint parameters (from evo_FastMPNNdesign)
    constraint = parser.add_argument_group("Constraint Options")
    constraint.add_argument(
        "--primary_contact_cutoff", type=float, default=3.6,
        help="Primary contact distance cutoff in Angstroms (default: 3.6)"
    )
    constraint.add_argument(
        "--secondary_contact_cutoff", type=float, default=4.2,
        help="Secondary contact distance cutoff in Angstroms (default: 4.2)"
    )
    constraint.add_argument(
        "--metal_cutoff", type=float, default=2.6,
        help="Metal coordination distance cutoff in Angstroms (default: 2.6)"
    )
    constraint.add_argument(
        "--coord_cst_stdev", type=float, default=0.01,
        help="Coordinate constraint standard deviation (default: 0.01)"
    )
    constraint.add_argument(
        "--mobile_radius", type=float, default=10.0,
        help="Radius for mobile residues around ligand (default: 10.0)"
    )

    # Protocol options
    protocol = parser.add_argument_group("Protocol Options")
    protocol.add_argument(
        "--protocol", type=Path,
        help="Protocol file defining design steps"
    )
    protocol.add_argument(
        "--hbond_accept_probability", type=float, default=DEFAULT_HBOND_ACCEPT_PROBABILITY,
        help=f"Probability of keeping H-bond contacts (default: {DEFAULT_HBOND_ACCEPT_PROBABILITY})"
    )

    # Relaxation options
    relax = parser.add_argument_group("Relaxation Options")
    relax.add_argument(
        "--fastrelax_cycles", type=int, default=3,
        help="Number of FastRelax cycles (default: 3)"
    )
    relax.add_argument(
        "--cartesian", action="store_true", default=True,
        help="Use Cartesian FastRelax (default: True)"
    )
    relax.add_argument(
        "--no_cartesian", action="store_true",
        help="Disable Cartesian FastRelax"
    )

    # Pipeline options (from evo_FastMPNNdesign)
    pipeline = parser.add_argument_group("Pipeline Options")
    pipeline.add_argument(
        "--n_cycles", type=int, default=3,
        help="Number of design cycles (default: 3)"
    )
    pipeline.add_argument(
        "--n_candidates", type=int, default=10,
        help="Number of candidates per cycle (default: 10)"
    )
    pipeline.add_argument(
        "--n_keep", type=int, default=2,
        help="Number of designs to keep per cycle (default: 2)"
    )
    pipeline.add_argument(
        "--n_final", type=int, default=10,
        help="Final number of designs to output (default: 10)"
    )

    # Scoring options
    scoring = parser.add_argument_group("Scoring Options")
    scoring.add_argument(
        "--scoring", type=Path,
        help="Scoring script for design evaluation"
    )
    scoring.add_argument(
        "--filter", action="store_true",
        help="Only dump outputs meeting filter criteria"
    )
    scoring.add_argument(
        "--mpnn", action="store_true",
        help="Perform additional 2nd layer MPNN on successful outputs"
    )

    # Rosetta options (NEW)
    rosetta = parser.add_argument_group("Rosetta Options")
    rosetta.add_argument(
        "--scorefunction", type=str, default=DEFAULT_SCOREFUNCTION,
        help=f"Rosetta scorefunction (default: {DEFAULT_SCOREFUNCTION})"
    )

    # Execution options
    execution = parser.add_argument_group("Execution Options")
    execution.add_argument(
        "--output_dir", type=Path, default=Path("."),
        help="Output directory (default: current directory)"
    )
    execution.add_argument(
        "--verbose", action="store_true", default=True,
        help="Verbose output (default: True)"
    )
    execution.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    execution.add_argument(
        "--dry_run", action="store_true",
        help="Print configuration without running"
    )

    parser.add_argument(
        "--version", action="version",
        version=f"enhanced-fastmpnndesign {__version__}"
    )

    return parser


def parse_catres_subset(value: Optional[str]) -> Optional[List[int]]:
    """Parse comma-separated catres indices."""
    if value is None or value.upper() == "ALL":
        return None
    return [int(x.strip()) for x in value.split(",")]


def flatten_params(params_list: Optional[List[List[Path]]]) -> List[Path]:
    """Flatten nested params list from argparse append action."""
    if params_list is None:
        return []
    result = []
    for item in params_list:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result


def build_config(args: argparse.Namespace) -> RunConfig:
    """Build RunConfig from parsed arguments."""

    # Handle verbose/quiet
    verbose = args.verbose and not args.quiet

    # Handle cartesian flag
    use_cartesian = args.cartesian and not args.no_cartesian

    # Flatten params list
    params = flatten_params(args.params)

    return RunConfig(
        pdb=args.pdb.resolve(),
        output_dir=args.output_dir.resolve(),
        prefix=args.prefix or "",
        suffix=args.suffix or "",
        nstruct=args.nstruct,
        design_pos=args.design_pos,
        keep_pos=args.keep_pos,
        detect_pocket=args.detect_pocket,

        mpnn=MPNNConfig(
            mpnn_runner=args.mpnn_runner,
            model_type=args.model_type,
            enhance_model=args.enhance if not args.no_enhance else None,
            use_enhanced_mpnn=not args.no_enhance,
            temperature=args.temperature,
            number_of_batches=args.number_of_batches,
            batch_size=args.batch_size,
            pack_side_chains=bool(args.pack_side_chains),
            sc_num_denoising_steps=args.sc_num_denoising_steps,
            omit_AA=args.omit_AA,
            repack_everything=bool(args.repack_everything),
            use_apptainer=args.use_apptainer,
            apptainer_image=args.apptainer_image,
            ligand_mpnn_use_side_chain_context=bool(args.ligand_mpnn_use_side_chain_context),
            packed_suffix=args.packed_suffix
        ),

        catres=CatresConfig(
            catres_subset=parse_catres_subset(args.catres_subset),
            redesign_non_subset_catres=args.redesign_non_subset_catres
        ),

        bias=BiasConfig(
            bias_atoms=args.bias_atoms,
            position_bias=args.position_bias,
            bias_AAs=args.bias_AAs
        ),

        relax=RelaxConfig(
            cartesian=use_cartesian,
            crude=True
        ),

        rosetta=RosettaConfig(
            scorefunction=args.scorefunction,
            params=[p.resolve() for p in params] if params else []
        ),

        protocol=ProtocolConfig(
            protocol_file=args.protocol.resolve() if args.protocol else None,
            cstfile=args.cstfile.resolve() if args.cstfile else None,
            hbond_accept_probability=args.hbond_accept_probability
        ),

        scoring=ScoringConfig(
            scoring_script=args.scoring.resolve() if args.scoring else None,
            apply_filter=args.filter
        ),

        second_layer_mpnn=SecondLayerMPNNConfig(
            enabled=args.mpnn,
            temperatures=SECOND_LAYER_TEMPERATURES
        ),

        verbose=verbose,
        dry_run=args.dry_run
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Build configuration
    config = build_config(args)

    # Setup logging
    logger = setup_logging(verbose=config.verbose)
    logger.info(f"Enhanced FastMPNN Design v{__version__}")
    logger.info(f"Script location: {_SCRIPT_DIR}")

    # Dry run - just print config
    if config.dry_run:
        import json
        print("\nConfiguration (dry run):")
        print(json.dumps(config.to_dict(), indent=2))
        return 0

    try:
        run_pipeline(config)
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
