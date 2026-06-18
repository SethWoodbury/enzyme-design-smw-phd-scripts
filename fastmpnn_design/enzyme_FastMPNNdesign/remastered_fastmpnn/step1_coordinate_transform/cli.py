"""
CLI for Step 1: Coordinate Transformation.

Usage:
    python -m remastered_fastmpnn.step1_coordinate_transform.cli \
        --input_pdb input.pdb \
        --ref_pdb reference.pdb \
        --catres_subset 1,2,3,4,5 \
        --output_pdb output.pdb \
        --verbose
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from remastered_fastmpnn.step1_coordinate_transform.coordinate_transformer import (
    CoordinateTransformer,
)
from remastered_fastmpnn.constants import (
    METAL_COORDINATION_CUTOFF,
    HBOND_CUTOFF,
    DEFAULT_OUTPUT_PDB,
)
from remastered_fastmpnn.logging_config import setup_logging, get_logger

__version__ = "0.1.0"


def parse_catres_subset(value: str) -> Optional[List[int]]:
    """
    Parse comma-separated catres subset indices.

    Args:
        value: Comma-separated string of 1-indexed positions, or "ALL"

    Returns:
        List of integers, or None for all catres
    """
    if not value or value.upper() == "ALL":
        return None

    try:
        indices = [int(x.strip()) for x in value.split(",") if x.strip()]
        # Validate all positive
        for idx in indices:
            if idx < 1:
                raise ValueError(f"Index must be >= 1, got {idx}")
        return indices
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid catres_subset format: {e}")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for Step 1 CLI."""
    parser = argparse.ArgumentParser(
        prog="step1-coordinate-transform",
        description=(
            "Step 1: Coordinate Transformation for Remastered FastMPNNdesign\n\n"
            "Transforms catalytic residue coordinates from structure prediction\n"
            "to match ground-truth theozyme geometry."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (all REMARK 666 residues as catres_subset)
    python -m remastered_fastmpnn.step1_coordinate_transform.cli \\
        --input_pdb alphafold_output.pdb \\
        --ref_pdb theozyme.pdb \\
        --output_pdb transformed.pdb

    # With catres subset selection (exclude blocks 12 and 14)
    python -m remastered_fastmpnn.step1_coordinate_transform.cli \\
        --input_pdb alphafold_output.pdb \\
        --ref_pdb theozyme.pdb \\
        --catres_subset 1,2,3,4,5,6,7,8,9,10,11,13,15,16,17,18,19 \\
        --output_pdb transformed.pdb \\
        --output_json residue_registry.json \\
        --verbose

    # With custom distance cutoffs
    python -m remastered_fastmpnn.step1_coordinate_transform.cli \\
        --input_pdb input.pdb \\
        --ref_pdb ref.pdb \\
        --metal_cutoff 2.8 \\
        --hbond_cutoff 3.2 \\
        --output_pdb output.pdb
        """,
    )

    # Required arguments
    required = parser.add_argument_group("Required Arguments")
    required.add_argument(
        "--input_pdb",
        type=Path,
        required=True,
        help="Input PDB file (structure prediction with ligand aligned)",
    )
    required.add_argument(
        "--ref_pdb",
        type=Path,
        required=True,
        help="Reference PDB file (theozyme with ground truth geometry)",
    )

    # Output arguments
    output = parser.add_argument_group("Output Options")
    output.add_argument(
        "--output_pdb",
        type=Path,
        default=Path(DEFAULT_OUTPUT_PDB),
        help=f"Output PDB file path (default: {DEFAULT_OUTPUT_PDB})",
    )
    output.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Output JSON file for residue registry (optional)",
    )
    output.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (if not specified in output paths)",
    )

    # Catres subset arguments
    catres = parser.add_argument_group("Catalytic Residue Options")
    catres.add_argument(
        "--catres_subset",
        type=str,
        default=None,
        metavar="INDICES",
        help=(
            "Comma-separated 1-indexed REMARK 666 line positions for catres_subset. "
            "Residues not in this list become conserved_motif. "
            "Use 'ALL' or omit for all residues (default: ALL)"
        ),
    )

    # Interaction detection cutoffs
    detection = parser.add_argument_group("Interaction Detection Cutoffs (Angstroms)")
    detection.add_argument(
        "--metal_cutoff",
        type=float,
        default=METAL_COORDINATION_CUTOFF,
        metavar="DIST",
        help=f"Metal coordination cutoff (default: {METAL_COORDINATION_CUTOFF})",
    )
    detection.add_argument(
        "--hbond_cutoff",
        type=float,
        default=HBOND_CUTOFF,
        metavar="DIST",
        help=f"Hydrogen bond cutoff (default: {HBOND_CUTOFF})",
    )

    # Execution options
    execution = parser.add_argument_group("Execution Options")
    execution.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (INFO level) logging",
    )
    execution.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging (more detailed than verbose)",
    )
    execution.add_argument(
        "--dry_run",
        action="store_true",
        help="Parse and analyze only, do not write output files",
    )
    execution.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Write logs to file in addition to stdout",
    )

    # Version
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate parsed arguments."""
    # Check input files exist
    if not args.input_pdb.exists():
        raise FileNotFoundError(f"Input PDB not found: {args.input_pdb}")
    if not args.ref_pdb.exists():
        raise FileNotFoundError(f"Reference PDB not found: {args.ref_pdb}")

    # Handle output directory
    if args.output_dir:
        args.output_dir = Path(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Adjust output paths if they're just filenames
        if not args.output_pdb.is_absolute():
            args.output_pdb = args.output_dir / args.output_pdb
        if args.output_json and not args.output_json.is_absolute():
            args.output_json = args.output_dir / args.output_json

    # Ensure output directory exists
    args.output_pdb.parent.mkdir(parents=True, exist_ok=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for Step 1 CLI.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Setup logging first
    setup_logging(
        verbose=args.verbose,
        debug=args.debug,
        log_file=args.log_file,
    )

    logger = get_logger("cli")

    try:
        # Validate arguments
        validate_args(args)

        logger.info(f"Step 1: Coordinate Transformation v{__version__}")
        logger.info(f"Input PDB: {args.input_pdb}")
        logger.info(f"Reference PDB: {args.ref_pdb}")

        # Parse catres subset
        catres_subset = parse_catres_subset(args.catres_subset) if args.catres_subset else None
        if catres_subset:
            logger.info(f"Catres subset indices: {catres_subset}")
        else:
            logger.info("Catres subset: ALL (no filtering)")

        # Create transformer
        transformer = CoordinateTransformer(
            input_pdb=args.input_pdb,
            ref_pdb=args.ref_pdb,
            catres_subset_indices=catres_subset,
            metal_cutoff=args.metal_cutoff,
            hbond_cutoff=args.hbond_cutoff,
            verbose=args.verbose or args.debug,
        )

        if args.dry_run:
            logger.info("Dry run mode - parsing and analyzing only")
            # Run everything except output
            transformer._parse_pdbs()
            transformer._align_by_ligand()
            transformer._parse_and_validate_remarks()
            transformer._build_registry()
            transformer._analyze_interactions()
            transformer._transform_coordinates()
            transformer._log_summary()
            logger.info("Dry run completed (no files written)")
        else:
            # Full run
            output_pdb, registry = transformer.run(args.output_pdb)
            logger.info(f"Output PDB: {output_pdb}")

            # Save JSON if requested
            if args.output_json:
                registry.save_json(args.output_json)
                logger.info(f"Saved residue registry: {args.output_json}")

        logger.info("Step 1 completed successfully")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
