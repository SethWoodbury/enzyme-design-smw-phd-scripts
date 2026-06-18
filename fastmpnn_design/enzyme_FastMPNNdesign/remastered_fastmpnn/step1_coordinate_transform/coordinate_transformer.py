"""
Coordinate transformation module for Step 1.

Orchestrates the complete workflow:
1. Parse input and reference PDB files
2. Align reference to input by ligand
3. Parse and validate REMARK 666 lines
4. Categorize residues as catres_subset or conserved_motif
5. Analyze interactions to determine important components
6. Transform coordinates based on analysis
7. Output modified PDB and residue registry
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import copy

from remastered_fastmpnn.constants import (
    LIGAND_RMSD_WARNING,
    BACKBONE_ATOMS,
)
from remastered_fastmpnn.core.residue_data import (
    ResidueInfo,
    ResidueRegistry,
    CatresSubsetInfo,
    ImportantComponent,
    ResidueCategory,
)
from remastered_fastmpnn.core.pdb_io import PDBParser, PDBWriter, AtomRecord
from remastered_fastmpnn.core.remark666 import (
    parse_remark666_lines,
    validate_remark666_consistency,
    get_catres_from_remarks,
    Remark666Info,
)
from remastered_fastmpnn.core.coordinate_utils import (
    align_by_ligand,
    calculate_residue_rmsd,
)
from remastered_fastmpnn.step1_coordinate_transform.interaction_detection import (
    InteractionDetector,
)
from remastered_fastmpnn.logging_config import get_logger, log_section, log_key_value

logger = get_logger("coordinate_transformer")


class CoordinateTransformer:
    """
    Transform catalytic residue coordinates from input to match reference.

    Main orchestrator for Step 1 of the pipeline.
    """

    def __init__(
        self,
        input_pdb: Path,
        ref_pdb: Path,
        catres_subset_indices: Optional[List[int]] = None,
        metal_cutoff: float = 2.6,
        hbond_cutoff: float = 3.5,
        verbose: bool = True,
    ):
        """
        Initialize the coordinate transformer.

        Args:
            input_pdb: Path to input PDB (structure prediction with ligand)
            ref_pdb: Path to reference PDB (theozyme with ground truth)
            catres_subset_indices: 1-indexed REMARK 666 line positions for catres_subset
                                   If None, all catres are subset
            metal_cutoff: Metal coordination distance cutoff (A)
            hbond_cutoff: Hydrogen bond distance cutoff (A)
            verbose: Enable verbose logging
        """
        self.input_pdb = Path(input_pdb)
        self.ref_pdb = Path(ref_pdb)
        self.catres_subset_indices = catres_subset_indices
        self.verbose = verbose

        # Will be populated during run()
        self.input_parser: Optional[PDBParser] = None
        self.ref_parser: Optional[PDBParser] = None
        self.aligned_ref_parser: Optional[PDBParser] = None
        self.registry: ResidueRegistry = ResidueRegistry()

        # Alignment results
        self.ligand_rmsd: float = 0.0
        self.rotation_matrix = None
        self.translation_vector = None

        # Interaction detector
        self.interaction_detector = InteractionDetector(
            metal_cutoff=metal_cutoff,
            hbond_cutoff=hbond_cutoff,
            verbose=verbose,
        )

        # Tracking
        self.catres_subset_remarks: List[Remark666Info] = []
        self.conserved_motif_remarks: List[Remark666Info] = []

    def run(self, output_pdb: Path) -> Tuple[Path, ResidueRegistry]:
        """
        Execute the complete coordinate transformation workflow.

        Args:
            output_pdb: Path for output PDB file

        Returns:
            Tuple of (output_pdb_path, residue_registry)
        """
        log_section(logger, "Step 1: Coordinate Transformation")

        # 1. Parse PDB files
        self._parse_pdbs()

        # 2. Align reference to input by ligand
        self._align_by_ligand()

        # 3. Parse and validate REMARK 666 lines
        self._parse_and_validate_remarks()

        # 4. Build residue registry and categorize
        self._build_registry()

        # 5. Analyze interactions for catres_subset
        self._analyze_interactions()

        # 6. Transform coordinates
        self._transform_coordinates()

        # 7. Write output
        self._write_output(output_pdb)

        # 8. Generate summary
        self._log_summary()

        return output_pdb, self.registry

    def _parse_pdbs(self) -> None:
        """Parse input and reference PDB files."""
        logger.info("Parsing PDB files...")

        log_key_value(logger, "Input PDB", str(self.input_pdb))
        self.input_parser = PDBParser(self.input_pdb)

        log_key_value(logger, "Reference PDB", str(self.ref_pdb))
        self.ref_parser = PDBParser(self.ref_pdb)

        # Log basic stats
        log_key_value(logger, "Input atoms", str(len(self.input_parser.atoms)))
        log_key_value(logger, "Reference atoms", str(len(self.ref_parser.atoms)))
        log_key_value(
            logger, "Input REMARK 666",
            str(len(self.input_parser.remark666_lines))
        )
        log_key_value(
            logger, "Ref REMARK 666",
            str(len(self.ref_parser.remark666_lines))
        )

        # Store ligand name
        self.registry.ligand_name = self.input_parser.get_ligand_name()
        log_key_value(logger, "Ligand name", self.registry.ligand_name or "Unknown")

    def _align_by_ligand(self) -> None:
        """Align reference PDB to input PDB by ligand atoms."""
        logger.info("Aligning reference to input by ligand...")

        # Make a deep copy of ref_parser for alignment
        # (we need the original for coordinate extraction)
        self.aligned_ref_parser = copy.deepcopy(self.ref_parser)

        self.aligned_ref_parser, self.ligand_rmsd, R, t = align_by_ligand(
            self.aligned_ref_parser,
            self.input_parser,
            ligand_name=self.registry.ligand_name,
        )

        self.rotation_matrix = R
        self.translation_vector = t

        log_key_value(logger, "Ligand alignment RMSD", f"{self.ligand_rmsd:.6f} A")

        if self.ligand_rmsd > LIGAND_RMSD_WARNING:
            logger.warning(
                f"Ligand RMSD ({self.ligand_rmsd:.4f} A) exceeds threshold "
                f"({LIGAND_RMSD_WARNING} A). Ligand geometries may differ!"
            )

    def _parse_and_validate_remarks(self) -> None:
        """Parse REMARK 666 lines and validate consistency."""
        logger.info("Parsing REMARK 666 lines...")

        ref_remarks = parse_remark666_lines(self.ref_parser.remark666_lines)
        input_remarks = parse_remark666_lines(self.input_parser.remark666_lines)

        log_key_value(logger, "Parsed ref remarks", str(len(ref_remarks)))
        log_key_value(logger, "Parsed input remarks", str(len(input_remarks)))

        # Validate consistency
        is_valid, errors = validate_remark666_consistency(ref_remarks, input_remarks)
        if not is_valid:
            for error in errors:
                logger.error(f"  {error}")
            raise ValueError(
                f"REMARK 666 validation failed with {len(errors)} errors"
            )

        # Split into catres_subset and conserved_motif
        self.catres_subset_remarks, self.conserved_motif_remarks = get_catres_from_remarks(
            input_remarks,
            self.catres_subset_indices,
        )

        log_key_value(
            logger, "Catres subset count",
            str(len(self.catres_subset_remarks))
        )
        log_key_value(
            logger, "Conserved motif count",
            str(len(self.conserved_motif_remarks))
        )

    def _build_registry(self) -> None:
        """Build residue registry from parsed data."""
        logger.info("Building residue registry...")

        self.registry.input_pdb_path = str(self.input_pdb)
        self.registry.ref_pdb_path = str(self.ref_pdb)

        # Track which residues are catres
        catres_identifiers: Set[str] = set()

        # Add catres_subset residues
        for remark in self.catres_subset_remarks:
            identifier = remark.motif_identifier
            catres_identifiers.add(identifier)

            # Create CatresSubsetInfo placeholder
            catres_info = CatresSubsetInfo(
                ref_pdb_chain=remark.motif_chain,
                ref_pdb_resnum=remark.motif_resnum,
            )

            residue = ResidueInfo(
                chain=remark.motif_chain,
                residue_num=remark.motif_resnum,
                identifier=identifier,
                res_type=remark.motif_resname,
                remark666_index=remark.line_index,
                cst_block=remark.block_index,
                cst_variant=remark.block_variant,
                category=ResidueCategory.CATRES_SUBSET,
                is_catres_subset=True,
                is_conserved_motif=False,
                probability_of_mutation=0.0,
                catres_subset_info=catres_info,
            )

            # Only add if not already present (residue may appear in multiple REMARK 666)
            if identifier not in self.registry:
                self.registry.add(residue)
            else:
                # Update existing entry with additional REMARK 666 info
                existing = self.registry.get(identifier)
                if existing and existing.remark666_index is None:
                    existing.remark666_index = remark.line_index

        # Add conserved_motif residues
        for remark in self.conserved_motif_remarks:
            identifier = remark.motif_identifier

            if identifier in catres_identifiers:
                # This residue is already catres_subset from another REMARK 666 line
                continue

            residue = ResidueInfo(
                chain=remark.motif_chain,
                residue_num=remark.motif_resnum,
                identifier=identifier,
                res_type=remark.motif_resname,
                remark666_index=remark.line_index,
                cst_block=remark.block_index,
                category=ResidueCategory.CONSERVED_MOTIF,
                is_catres_subset=False,
                is_conserved_motif=True,
                probability_of_mutation=0.0,
            )

            if identifier not in self.registry:
                self.registry.add(residue)

        # Count total residues in input PDB
        self.registry.total_residue_count = len(
            set(a.identifier for a in self.input_parser.atoms if a.is_protein())
        )

        log_key_value(
            logger, "Registry entries",
            str(len(self.registry))
        )

    def _analyze_interactions(self) -> None:
        """Analyze interactions for catres_subset residues."""
        logger.info("Analyzing interactions for catres_subset residues...")

        catres_list = self.registry.get_catres_subset()
        if not catres_list:
            logger.warning("No catres_subset residues to analyze")
            return

        # Build list of (chain, resnum) for all catres
        catres_identifiers = [
            (r.chain, r.residue_num) for r in catres_list
        ]

        # Run interaction analysis on aligned reference
        catres_info = self.interaction_detector.analyze_catres_subset(
            self.aligned_ref_parser,
            catres_identifiers,
        )

        # Update registry with analysis results
        for identifier, info in catres_info.items():
            residue = self.registry.get(identifier)
            if residue and residue.catres_subset_info:
                residue.catres_subset_info.important_component = info.important_component
                residue.catres_subset_info.interactions_found = info.interactions_found
                residue.catres_subset_info.backbone_interaction_count = info.backbone_interaction_count
                residue.catres_subset_info.sidechain_interaction_count = info.sidechain_interaction_count

        # Log summary
        if self.verbose:
            summary = self.interaction_detector.summarize_interactions(catres_info)
            for line in summary.split('\n'):
                logger.debug(line)

    def _transform_coordinates(self) -> None:
        """Transform coordinates based on interaction analysis."""
        logger.info("Transforming coordinates...")

        transformed_count = 0
        backbone_count = 0
        sidechain_only_count = 0

        for residue in self.registry.get_catres_subset():
            if not residue.catres_subset_info:
                continue

            component = residue.catres_subset_info.important_component
            chain = residue.chain
            resnum = residue.residue_num
            identifier = residue.identifier

            # Get atoms from input (to modify) and aligned reference (source)
            input_atoms = self.input_parser.get_residue_atoms(chain, resnum)
            ref_atoms = self.aligned_ref_parser.get_residue_atoms(chain, resnum)

            if not input_atoms or not ref_atoms:
                logger.warning(f"  Skipping {identifier}: atoms not found")
                continue

            # Build coordinate lookup from aligned reference
            ref_coords = {a.name.strip(): (a.x, a.y, a.z) for a in ref_atoms}

            # Determine which atoms to copy
            if component in (ImportantComponent.BACKBONE, ImportantComponent.BOTH):
                # Copy ALL atom coordinates
                updated = self._copy_all_coords(input_atoms, ref_coords)
                residue.catres_subset_info.backbone_coords_copied = True
                residue.catres_subset_info.sidechain_coords_copied = True
                backbone_count += 1
                logger.info(
                    f"  {identifier}: Copied ALL coords ({updated} atoms) "
                    f"[{component.value}]"
                )

            elif component == ImportantComponent.SIDECHAIN:
                # Copy only sidechain coordinates
                updated = self._copy_sidechain_coords(input_atoms, ref_coords)
                residue.catres_subset_info.sidechain_coords_copied = True
                sidechain_only_count += 1
                logger.info(
                    f"  {identifier}: Copied SIDECHAIN coords ({updated} atoms) "
                    f"[{component.value}]"
                )

            else:  # NONE - default to sidechain
                updated = self._copy_sidechain_coords(input_atoms, ref_coords)
                residue.catres_subset_info.sidechain_coords_copied = True
                sidechain_only_count += 1
                logger.info(
                    f"  {identifier}: Copied SIDECHAIN coords ({updated} atoms) "
                    f"[default - no interactions]"
                )

            transformed_count += 1
            residue.step1_complete = True

        log_key_value(logger, "Residues transformed", str(transformed_count))
        log_key_value(logger, "All coords copied", str(backbone_count))
        log_key_value(logger, "Sidechain only", str(sidechain_only_count))

    def _copy_all_coords(
        self,
        input_atoms: List[AtomRecord],
        ref_coords: Dict[str, Tuple[float, float, float]],
    ) -> int:
        """
        Copy all atom coordinates from reference to input.

        Args:
            input_atoms: Input atoms to modify
            ref_coords: Reference coordinates by atom name

        Returns:
            Number of atoms updated
        """
        updated = 0
        for atom in input_atoms:
            name = atom.name.strip()
            if name in ref_coords:
                atom.x, atom.y, atom.z = ref_coords[name]
                updated += 1
        return updated

    def _copy_sidechain_coords(
        self,
        input_atoms: List[AtomRecord],
        ref_coords: Dict[str, Tuple[float, float, float]],
    ) -> int:
        """
        Copy only sidechain atom coordinates from reference to input.

        Args:
            input_atoms: Input atoms to modify
            ref_coords: Reference coordinates by atom name

        Returns:
            Number of atoms updated
        """
        updated = 0
        for atom in input_atoms:
            name = atom.name.strip()
            # Skip backbone atoms
            if name in BACKBONE_ATOMS or name in {'H', 'HA', 'HA2', 'HA3'}:
                continue
            if name in ref_coords:
                atom.x, atom.y, atom.z = ref_coords[name]
                updated += 1
        return updated

    def _write_output(self, output_pdb: Path) -> None:
        """Write transformed PDB to file."""
        logger.info(f"Writing output PDB: {output_pdb}")

        PDBWriter.write(
            self.input_parser,
            output_pdb,
            preserve_remarks=True,
            preserve_other=True,
            renumber_atoms=False,
        )

    def _log_summary(self) -> None:
        """Log summary of the transformation."""
        log_section(logger, "Step 1 Summary")

        catres = self.registry.get_catres_subset()
        conserved = self.registry.get_conserved_motif()

        log_key_value(logger, "Input PDB", str(self.input_pdb.name))
        log_key_value(logger, "Reference PDB", str(self.ref_pdb.name))
        log_key_value(logger, "Ligand RMSD", f"{self.ligand_rmsd:.6f} A")
        log_key_value(logger, "Total residues", str(self.registry.total_residue_count))
        log_key_value(logger, "Catres subset", str(len(catres)))
        log_key_value(logger, "Conserved motif", str(len(conserved)))

        # Count by important component
        backbone_count = sum(
            1 for r in catres
            if r.catres_subset_info and
            r.catres_subset_info.important_component == ImportantComponent.BACKBONE
        )
        sidechain_count = sum(
            1 for r in catres
            if r.catres_subset_info and
            r.catres_subset_info.important_component == ImportantComponent.SIDECHAIN
        )
        both_count = sum(
            1 for r in catres
            if r.catres_subset_info and
            r.catres_subset_info.important_component == ImportantComponent.BOTH
        )

        log_key_value(logger, "Backbone important", str(backbone_count))
        log_key_value(logger, "Sidechain important", str(sidechain_count))
        log_key_value(logger, "Both important", str(both_count))

        logger.info("Step 1 completed successfully")
