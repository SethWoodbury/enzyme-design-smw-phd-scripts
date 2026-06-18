"""
Pipeline orchestration for iterative design cycles.

Manages the iterative workflow:
1. LigandMPNN sequence design
2. Constrained Rosetta relaxation
3. Metrics computation and filtering
4. Repeat for N cycles
5. Final amplification and output
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

from fastmpnndesign.config import (
    RunConfig, CatalyticResidue, CycleResult, PipelineConfig, CatresCatresContact
)
from fastmpnndesign.remark666 import (
    parse_remark666_from_pdb, get_catres_subset, validate_catres_in_pdb
)
from fastmpnndesign.ligand import detect_ligands_from_pdb, detect_metals_from_pdb, Ligand
from fastmpnndesign.protonation import (
    detect_histidine_protonation_states, HistidineProtonation,
    protonation_states_to_json, verify_protonation_states
)
from fastmpnndesign.constraints import generate_constraints, ConstraintSet
from fastmpnndesign.contact_detection import detect_catres_catres_contacts
from fastmpnndesign.mpnn_runner import run_mpnn, get_packed_pdbs, MPNNResult
from fastmpnndesign.relax_runner import relax_structure, RelaxResult
from fastmpnndesign.metrics import compute_all_metrics, CandidateMetrics, summarize_metrics
from fastmpnndesign.filtering import (
    select_best, RankingStrategy, FilterCriteria, diversify_selection
)
from fastmpnndesign.utils import (
    ensure_dir, write_json, copy_file, get_pdb_sequence,
    get_residues_outside_shell, get_ligand_atom_coords
)
from fastmpnndesign.logging_config import get_logger

logger = get_logger("orchestrator")


@dataclass
class PipelineState:
    """Tracks state across pipeline execution."""
    config: RunConfig
    catres_all: List[CatalyticResidue] = field(default_factory=list)
    catres_subset: List[CatalyticResidue] = field(default_factory=list)
    catres_non_subset: List[CatalyticResidue] = field(default_factory=list)
    constraint_set: Optional[ConstraintSet] = None
    cycle_results: List[CycleResult] = field(default_factory=list)
    all_candidates: List[CandidateMetrics] = field(default_factory=list)
    final_candidates: List[CandidateMetrics] = field(default_factory=list)
    # Histidine protonation states detected from input PDB
    # Key: (chain, resnum, icode), Value: HistidineProtonation
    protonation_states: Dict[Tuple[str, int, str], HistidineProtonation] = field(default_factory=dict)
    # Detected metals (stored for protonation state detection and reuse)
    metals: List[Ligand] = field(default_factory=list)
    # Catres-catres interactions (hydrogen bonds, salt bridges, pi-stacking)
    catres_catres_contacts: List[CatresCatresContact] = field(default_factory=list)


def initialize_pipeline(config: RunConfig) -> PipelineState:
    """
    Initialize pipeline state by parsing inputs and generating constraints.

    Args:
        config: Run configuration.

    Returns:
        Initialized PipelineState.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("  FASTMPNN DESIGN PIPELINE - INITIALIZATION")
    logger.info("=" * 60)

    state = PipelineState(config=config)

    # Create output directory
    ensure_dir(config.output_dir)

    # Parse REMARK 666 for catalytic residues
    logger.info("")
    logger.info("  Parsing Catalytic Residues (REMARK 666)")
    logger.info("-" * 40)
    state.catres_all = parse_remark666_from_pdb(config.pdb)

    if not state.catres_all:
        logger.warning("    No REMARK 666 catalytic residues found in PDB")

    # Validate catres exist in PDB
    found, missing = validate_catres_in_pdb(state.catres_all, config.pdb)
    if missing:
        logger.error(f"    {len(missing)} catalytic residues not found in PDB structure")

    # Split into subset and non-subset
    state.catres_subset, state.catres_non_subset = get_catres_subset(
        state.catres_all,
        config.catres.catres_subset
    )

    logger.info(f"    Total catres: {len(state.catres_all)}")
    logger.info(f"    Subset (tight constraints): {len(state.catres_subset)}")
    for cr in state.catres_subset:
        logger.info(f"      {cr.catres_index}: {cr.chain}{cr.resnum} {cr.resname}")

    # Detect ligands
    logger.info("")
    logger.info("  Detecting Ligands & Metals")
    logger.info("-" * 40)
    ligands = detect_ligands_from_pdb(config.pdb)
    metals = detect_metals_from_pdb(config.pdb)
    state.metals = metals  # Store for later use
    logger.info(f"    Ligands: {len(ligands)}")
    for lig in ligands:
        logger.info(f"      {lig.resname} chain {lig.chain} res {lig.resnum} ({len(lig.heavy_atoms)} heavy atoms)")
    logger.info(f"    Metals: {len(metals)}")
    for met in metals:
        logger.info(f"      {met.resname} chain {met.chain} res {met.resnum}")

    # Detect histidine protonation states
    logger.info("")
    logger.info("  Detecting Histidine Protonation States")
    logger.info("-" * 40)
    state.protonation_states = detect_histidine_protonation_states(
        config.pdb, metals=metals
    )
    n_his_d = sum(1 for p in state.protonation_states.values() if p.protonation_state == 'HIS_D')
    n_his = sum(1 for p in state.protonation_states.values() if p.protonation_state == 'HIS')
    logger.info(f"    Total histidines: {len(state.protonation_states)}")
    logger.info(f"    HIS (standard): {n_his}")
    logger.info(f"    HIS_D (delta-protonated): {n_his_d}")
    for key, prot in state.protonation_states.items():
        if prot.protonation_state == 'HIS_D':
            logger.info(f"      {prot.resid}: HIS_D - {prot.reason}")

    # Generate constraints
    logger.info("")
    logger.info("  Generating Constraints")
    logger.info("-" * 40)
    if config.constraints.ref_pdb and config.constraints.ref_pdb.exists():
        logger.info(f"    Using ref_pdb for ideal geometry: {config.constraints.ref_pdb}")
    else:
        logger.info("    Using self-derived constraints (input PDB geometry)")
        if config.constraints.ref_pdb:
            logger.warning(f"    ref_pdb provided but not found: {config.constraints.ref_pdb}")
    state.constraint_set = generate_constraints(
        config.pdb,
        catres_list=state.catres_subset,
        config=config.constraints,
        output_dir=config.output_dir
    )
    logger.info(f"    Coordinate constraints: {len(state.constraint_set.coordinate_constraints)}")
    logger.info(f"    Distance constraints: {len(state.constraint_set.distance_constraints)}")

    # Detect and log catres-catres interactions
    logger.info("")
    logger.info("  Detecting Catres-Catres Interactions")
    logger.info("-" * 40)
    catres_source_pdb = config.constraints.ref_pdb if (
        config.constraints.ref_pdb and config.constraints.ref_pdb.exists()
    ) else config.pdb
    if len(state.catres_subset) >= 2:
        state.catres_catres_contacts = detect_catres_catres_contacts(
            catres_source_pdb,
            state.catres_subset
        )
        # Count by type
        n_hbond = sum(1 for c in state.catres_catres_contacts if c.interaction_type == 'hbond')
        n_salt = sum(1 for c in state.catres_catres_contacts if c.interaction_type == 'salt_bridge')
        n_pi = sum(1 for c in state.catres_catres_contacts if c.interaction_type == 'pi_stack')

        logger.info(f"    Source: {catres_source_pdb}")
        logger.info(f"    Total catres-catres contacts: {len(state.catres_catres_contacts)}")
        logger.info(f"      Hydrogen bonds: {n_hbond}")
        logger.info(f"      Salt bridges: {n_salt}")
        logger.info(f"      Pi-stacking: {n_pi}")

        # Log individual interactions
        for contact in state.catres_catres_contacts:
            logger.info(
                f"      {contact.interaction_type:12s}: "
                f"{contact.resname1} {contact.chain1}{contact.resnum1} {contact.atom1} - "
                f"{contact.resname2} {contact.chain2}{contact.resnum2} {contact.atom2} "
                f"({contact.distance:.2f} A)"
            )
    else:
        state.catres_catres_contacts = []
        logger.info("    Less than 2 catalytic residues, skipping catres-catres detection")

    # Save run configuration
    config.save(config.output_dir / "run_config.json")
    logger.info("")
    logger.info(f"  Configuration saved to: {config.output_dir / 'run_config.json'}")

    return state


def run_design_cycle(
    state: PipelineState,
    cycle_num: int,
    input_pdb: Path
) -> CycleResult:
    """
    Run a single design cycle.

    Args:
        state: Pipeline state.
        cycle_num: Cycle number (1-indexed).
        input_pdb: Input PDB for this cycle.

    Returns:
        CycleResult with cycle outputs.
    """
    config = state.config
    cycle_dir = config.output_dir / f"cycle_{cycle_num:02d}"
    ensure_dir(cycle_dir)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  CYCLE {cycle_num}/{config.pipeline.n_cycles}")
    logger.info("=" * 60)

    # Determine fixed residues for MPNN
    # Start with catres (always fixed unless redesign_non_subset_catres is set)
    if config.catres.redesign_non_subset_catres:
        fixed_catres = state.catres_subset
    else:
        fixed_catres = state.catres_all

    # Build additional fixed residues list
    additional_fixed = []

    # Fix residues outside the design shell (keep most of the scaffold)
    if config.pipeline.fix_outside_shell:
        ligand_coords = get_ligand_atom_coords(input_pdb)
        if ligand_coords:
            outside_shell = get_residues_outside_shell(
                input_pdb,
                ligand_coords=ligand_coords,
                shell_radius=config.pipeline.design_shell_radius
            )
            # Don't double-add catres
            catres_resids = set(cr.pdb_resid for cr in fixed_catres)
            outside_shell = [r for r in outside_shell if r not in catres_resids]
            additional_fixed.extend(outside_shell)

            # Calculate designable residues
            all_fixed = set(outside_shell) | catres_resids
            from fastmpnndesign.utils import iter_pdb_atoms
            from fastmpnndesign.constants import STANDARD_AA, NONSTANDARD_AA
            all_protein_residues = set()
            for atom in iter_pdb_atoms(input_pdb):
                if atom['resname'] in STANDARD_AA or atom['resname'] in NONSTANDARD_AA:
                    chain = atom['chain'] or 'A'
                    all_protein_residues.add(f"{chain}{atom['resnum']}")
            designable_residues = sorted(all_protein_residues - all_fixed, key=lambda x: (x[0], int(x[1:])))

            logger.info("")
            logger.info(f"  Design Shell: {config.pipeline.design_shell_radius} A radius from any ligand atom")
            logger.info(f"    Ligand heavy atoms: {len(ligand_coords)}")
            logger.info(f"    Total protein residues: {len(all_protein_residues)}")
            logger.info("")
            logger.info(f"    FIXED residues ({len(all_fixed)} total):")
            logger.info(f"      - Catres (always fixed): {len(catres_resids)}")
            catres_list_str = ", ".join(sorted(catres_resids, key=lambda x: (x[0], int(x[1:]))))
            logger.info(f"        {catres_list_str}")
            logger.info(f"      - Outside shell (fixed): {len(outside_shell)}")
            if len(outside_shell) <= 30:
                logger.info(f"        {', '.join(outside_shell)}")
            else:
                logger.info(f"        {', '.join(outside_shell[:15])} ... {', '.join(outside_shell[-15:])}")
            logger.info("")
            logger.info(f"    DESIGNABLE residues ({len(designable_residues)} total):")
            if len(designable_residues) <= 30:
                logger.info(f"        {', '.join(designable_residues)}")
            else:
                logger.info(f"        {', '.join(designable_residues[:15])} ... {', '.join(designable_residues[-15:])}")
        else:
            logger.warning("  Could not find ligand atoms - designing all residues")

    # Step 1: Run MPNN
    logger.info("")
    logger.info("-" * 40)
    logger.info("  Step 1: Running LigandMPNN")
    logger.info("-" * 40)

    mpnn_dir = cycle_dir / "mpnn"
    ensure_dir(mpnn_dir)

    mpnn_result = run_mpnn(
        input_pdb,
        mpnn_dir,
        catres_list=fixed_catres,
        config=config.mpnn,
        additional_fixed=additional_fixed,
        dry_run=config.dry_run
    )

    if not mpnn_result.success:
        logger.error(f"  MPNN failed: {mpnn_result.stderr}")
        return CycleResult(
            cycle_num=cycle_num,
            input_pdb=input_pdb,
            mpnn_output_dir=mpnn_dir,
            relax_output_dir=cycle_dir / "relax",
            selected_pdbs=[],
            metrics={'error': 'MPNN failed'}
        )

    logger.info(f"    Generated {mpnn_result.n_sequences} sequences")

    # Get packed PDBs for relaxation
    packed_pdbs = get_packed_pdbs(mpnn_dir)
    logger.info(f"    Found {len(packed_pdbs)} packed PDB structures")

    if not packed_pdbs:
        logger.warning("    No packed PDBs generated")

    # Step 2: Relax each candidate
    logger.info("")
    logger.info("-" * 40)
    logger.info("  Step 2: Constrained Relaxation")
    logger.info("-" * 40)

    relax_dir = cycle_dir / "relax"
    ensure_dir(relax_dir)

    relax_results = []
    n_to_relax = min(len(packed_pdbs), config.pipeline.n_candidates)
    for i, pdb in enumerate(packed_pdbs[:n_to_relax]):
        output_pdb = relax_dir / f"relaxed_{i:03d}.pdb"
        logger.info(f"    [{i+1}/{n_to_relax}] Relaxing {pdb.name}")

        result = relax_structure(
            pdb,
            output_pdb,
            state.constraint_set,
            config.params,
            config.relax,
            catres_list=state.catres_all,
            protonation_states=state.protonation_states,
            dry_run=config.dry_run
        )
        relax_results.append(result)

    successful_relax = [r for r in relax_results if r.success]
    logger.info(f"    Completed: {len(successful_relax)}/{len(relax_results)} successful")

    # Step 3: Compute metrics
    logger.info("")
    logger.info("-" * 40)
    logger.info("  Step 3: Computing Metrics")
    logger.info("-" * 40)

    candidates = []
    for i, relax_result in enumerate(relax_results):
        if not relax_result.success or not relax_result.output_pdb:
            continue

        # Get MPNN score from sequences
        mpnn_score = None
        if i < len(mpnn_result.sequences):
            mpnn_score = mpnn_result.sequences[i].get('score')

        metrics = compute_all_metrics(
            relax_result.output_pdb,
            constraint_set=state.constraint_set,
            native_pdb=config.pdb,
            catres_list=state.catres_all,
            params_files=config.params,
            mpnn_score=mpnn_score,
            cycle=cycle_num,
            input_pdb=input_pdb
        )
        candidates.append(metrics)

    logger.info(f"    Metrics computed for {len(candidates)} candidates")

    # Step 4: Select best candidates
    logger.info("")
    logger.info("-" * 40)
    logger.info("  Step 4: Selecting Best Candidates")
    logger.info("-" * 40)
    filter_criteria = FilterCriteria(
        max_mean_displacement=0.5,
        max_max_displacement=1.0
    )

    selected = select_best(
        candidates,
        n_keep=config.pipeline.n_keep,
        strategy=RankingStrategy.GEOMETRY_QUALITY,
        filter_criteria=filter_criteria
    )

    # Copy selected to cycle directory
    selected_pdbs = []
    for i, cand in enumerate(selected):
        dest = cycle_dir / f"selected_{i:03d}.pdb"
        copy_file(cand.pdb_path, dest)
        selected_pdbs.append(dest)

    # Save cycle metrics
    cycle_metrics = {
        'cycle': cycle_num,
        'n_mpnn_sequences': mpnn_result.n_sequences,
        'n_packed_pdbs': len(packed_pdbs),
        'n_relaxed': len(successful_relax),
        'n_candidates': len(candidates),
        'n_selected': len(selected),
        'summary': summarize_metrics(candidates),
        'selected': [c.to_dict() for c in selected]
    }
    write_json(cycle_dir / "cycle_metrics.json", cycle_metrics)

    # Update state
    state.all_candidates.extend(candidates)

    result = CycleResult(
        cycle_num=cycle_num,
        input_pdb=input_pdb,
        mpnn_output_dir=mpnn_dir,
        relax_output_dir=relax_dir,
        selected_pdbs=selected_pdbs,
        metrics=cycle_metrics
    )
    state.cycle_results.append(result)

    return result


def run_pipeline(config: RunConfig) -> PipelineState:
    """
    Run the complete iterative design pipeline.

    Args:
        config: Run configuration.

    Returns:
        Final PipelineState with all results.
    """
    # Initialize
    state = initialize_pipeline(config)

    # Current input PDB (starts with original)
    current_input = config.pdb

    # Run design cycles
    for cycle_num in range(1, config.pipeline.n_cycles + 1):
        cycle_result = run_design_cycle(state, cycle_num, current_input)

        if not cycle_result.selected_pdbs:
            logger.warning(f"No candidates selected in cycle {cycle_num}, stopping")
            break

        # Use best selected as input for next cycle
        current_input = cycle_result.selected_pdbs[0]

    # Final amplification
    logger.info("=" * 60)
    logger.info("Final Amplification")
    logger.info("=" * 60)

    state.final_candidates = amplify_sequences(
        state,
        n_final=config.pipeline.n_final
    )

    # Generate final outputs
    generate_final_outputs(state)

    # Cleanup intermediate directories if requested
    if config.cleanup:
        cleanup_intermediates(state)

    logger.info("=" * 60)
    logger.info("Pipeline Complete")
    logger.info("=" * 60)

    return state


def amplify_sequences(
    state: PipelineState,
    n_final: int
) -> List[CandidateMetrics]:
    """
    Amplify to reach final sequence quota with diversity.

    Args:
        state: Pipeline state.
        n_final: Number of final sequences to produce.

    Returns:
        List of final candidates.
    """
    # Get all candidates sorted by quality
    all_ranked = select_best(
        state.all_candidates,
        n_keep=len(state.all_candidates),
        strategy=RankingStrategy.GEOMETRY_QUALITY
    )

    # Diversify selection
    final = diversify_selection(
        all_ranked,
        n_select=n_final,
        min_hamming_distance=2
    )

    logger.info(f"Selected {len(final)} diverse final candidates from {len(all_ranked)} total")

    return final


def generate_final_outputs(state: PipelineState) -> None:
    """
    Generate final output files.

    Final PDBs go directly in output_dir/ with naming:
        {prefix}_final_001.pdb, {prefix}_final_002.pdb, etc.
    Metrics summary goes to {prefix}_metrics.json at top level.

    Args:
        state: Pipeline state with final candidates.
    """
    config = state.config

    # Copy final PDBs directly to output_dir (not a subdirectory)
    for i, cand in enumerate(state.final_candidates):
        dest = config.output_dir / f"{config.prefix}_final_{i+1:03d}.pdb"
        copy_file(cand.pdb_path, dest)

    # Generate metrics summary at top level
    summary = {
        'input_pdb': str(config.pdb),
        'n_cycles': config.pipeline.n_cycles,
        'n_total_candidates': len(state.all_candidates),
        'n_final_candidates': len(state.final_candidates),
        'catres': [cr.to_dict() for cr in state.catres_all],
        'catres_subset': [cr.to_dict() for cr in state.catres_subset],
        'protonation_states': protonation_states_to_json(state.protonation_states),
        'cycles': [cr.to_dict() for cr in state.cycle_results],
        'final_candidates': [c.to_dict() for c in state.final_candidates],
        'overall_summary': summarize_metrics(state.all_candidates)
    }
    metrics_path = config.output_dir / f"{config.prefix}_metrics.json"
    write_json(metrics_path, summary)

    logger.info(f"Final outputs written to {config.output_dir}")
    logger.info(f"  {len(state.final_candidates)} PDB structures: {config.prefix}_final_001.pdb ...")
    logger.info(f"  Metrics: {metrics_path}")


def cleanup_intermediates(state: PipelineState) -> None:
    """
    Delete intermediate directories after successful completion.

    Keeps only:
    - Final PDBs ({prefix}_final_*.pdb)
    - Metrics ({prefix}_metrics.json)
    - Run config (run_config.json)
    - Constraint files (*.cst)
    - Log file

    Deletes:
    - cycle_*/ directories

    Args:
        state: Pipeline state.
    """
    config = state.config

    logger.info("Cleaning up intermediate directories...")

    # Find and delete cycle directories
    cycle_dirs = list(config.output_dir.glob("cycle_*"))
    for cycle_dir in cycle_dirs:
        if cycle_dir.is_dir():
            logger.info(f"  Removing {cycle_dir.name}/")
            shutil.rmtree(cycle_dir)

    logger.info(f"Cleaned up {len(cycle_dirs)} intermediate directories")


def validate_inputs(config: RunConfig) -> List[str]:
    """
    Validate all input files and configuration.

    Returns:
        List of error messages (empty if valid).
    """
    errors = []

    # Check PDB exists
    if not config.pdb.exists():
        errors.append(f"Input PDB not found: {config.pdb}")

    # Check params files exist
    for p in config.params:
        if not p.exists():
            errors.append(f"Params file not found: {p}")

    # Check MPNN runner
    if not config.mpnn.mpnn_runner.exists():
        errors.append(f"MPNN runner not found: {config.mpnn.mpnn_runner}")

    # Check Apptainer image if using
    if config.mpnn.use_apptainer and not config.mpnn.apptainer_image.exists():
        errors.append(f"Apptainer image not found: {config.mpnn.apptainer_image}")

    # Check ref_pdb if provided
    if config.constraints.ref_pdb and not config.constraints.ref_pdb.exists():
        errors.append(f"Reference PDB (ref_pdb) not found: {config.constraints.ref_pdb}")

    return errors
