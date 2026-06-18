Ready to code?

 Here is Claude's plan:
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 fastMPNNdesign Implementation Plan

 Overview

 Create a modular Python package for iterative protein active-site design:
 LigandMPNN → Constrained Rosetta Relax → Filter/Rank → Repeat 3-4 cycles

 Target: /home/woodbuse/special_scripts/fastmpnn_design/evo_FastMPNNdesign

 ---
 1. Architecture & Data Flow

 Input: PDB (with REMARK 666) + Params files
               │
               ▼
 ┌─────────────────────────────────────┐
 │  1. PARSING & INITIALIZATION        │
 │  • Parse REMARK 666 → catres list   │
 │  • Detect ligands via is_ligand()   │
 │  • Validate params files exist      │
 └─────────────────────────────────────┘
               │
               ▼
 ┌─────────────────────────────────────┐
 │  2. CONTACT DETECTION               │
 │  • Two-tier: 3.6Å primary, 4.2Å sec │
 │  • Metal: 2.6Å cutoff               │
 │  • Prioritize N/O/S over C atoms    │
 └─────────────────────────────────────┘
               │
               ▼
 ┌─────────────────────────────────────┐
 │  3. CONSTRAINT GENERATION           │
 │  • Coordinate csts (ligand + catres)│
 │  • Distance csts from contacts      │
 │  • Output: .cst file + JSON summary │
 └─────────────────────────────────────┘
               │
               ▼
 ┌─────────────────────────────────────────────────────┐
 │  4. ITERATIVE DESIGN LOOP (3-4 cycles)              │
 │  ┌───────────────────────────────────────────────┐  │
 │  │ A. LigandMPNN Design                          │  │
 │  │    • Fixed residues: all catres (default)    │  │
 │  │    • Generate N candidates                    │  │
 │  │    • Subprocess or Apptainer execution        │  │
 │  └───────────────────────────────────────────────┘  │
 │                        │                            │
 │                        ▼                            │
 │  ┌───────────────────────────────────────────────┐  │
 │  │ B. Constrained Cartesian FastRelax            │  │
 │  │    • Apply coordinate + distance constraints  │  │
 │  │    • MoveMap: mobile=10Å sphere, freeze lig   │  │
 │  │    • Backbone movement allowed (controlled)   │  │
 │  └───────────────────────────────────────────────┘  │
 │                        │                            │
 │                        ▼                            │
 │  ┌───────────────────────────────────────────────┐  │
 │  │ C. Metrics & Filtering                        │  │
 │  │    • Constraint satisfaction (displacement)   │  │
 │  │    • Rosetta scores                           │  │
 │  │    • Select best 1-2 by geometry quality      │  │
 │  └───────────────────────────────────────────────┘  │
 └─────────────────────────────────────────────────────┘
               │
               ▼
 ┌─────────────────────────────────────┐
 │  5. FINAL AMPLIFICATION             │
 │  • Diversity-aware sequence expand  │
 │  • Spherical neighborhood criteria  │
 └─────────────────────────────────────┘
               │
               ▼
 Outputs: PDBs, FASTAs, metrics.json, run_config.json

 ---
 2. Package Layout

 evo_FastMPNNdesign/
 ├── fastmpnndesign/
 │   ├── __init__.py           # Version, exports
 │   ├── cli.py                # Entry point, argparse
 │   ├── config.py             # Dataclasses (RunConfig, MPNNConfig, etc.)
 │   ├── remark666.py          # REMARK 666 parser → CatalyticResidue
 │   ├── ligand.py             # Ligand detection utilities
 │   ├── contact_detection.py  # Two-tier contact detection
 │   ├── constraints.py        # Constraint generation + file output
 │   ├── mpnn_runner.py        # MPNN subprocess wrapper (+Apptainer)
 │   ├── relax_runner.py       # PyRosetta relax execution
 │   ├── metrics.py            # Quality metrics computation
 │   ├── filtering.py          # Candidate ranking/selection
 │   ├── orchestrator.py       # Pipeline cycle management
 │   ├── slurm.py              # Sbatch template generation
 │   ├── logging_config.py     # Logging setup
 │   ├── constants.py          # Element sets, defaults
 │   └── utils.py              # Path handling, I/O utilities
 ├── tests/
 │   ├── test_remark666.py
 │   ├── test_contact_detection.py
 │   ├── test_constraints.py
 │   ├── test_mpnn_runner.py
 │   └── fixtures/             # Test PDBs, params
 ├── pyproject.toml
 ├── design_goal.md            # (existing spec)
 └── progress_and_updates.md   # (implementation journal)

 ---
 3. CLI Specification

 fastmpnndesign --pdb INPUT.pdb --params LIGAND.params [OPTIONS]

 Required:
   --pdb PATH              Input PDB with REMARK 666 lines
   --params PATH [PATH...] Ligand params file(s)

 Output:
   --output_dir PATH       Output directory (default: ./fastmpnn_output)
   --prefix STR            Output file prefix

 MPNN Options:
   --mpnn_runner PATH      MPNN run.py path (default: /net/software/lab/fused_mpnn/seth_temp/run.py)
   --model_type STR        Model type (default: ligand_mpnn)
   --enhance STR           Enhancement model (default: plddt_3_20240930-f9c9ea0f)
   --no_enhance            Disable enhancement
   --temperature FLOAT     MPNN temperature (default: 0.1)
   --number_of_batches INT Batches (default: 10)
   --batch_size INT        Batch size (default: 1)
   --pack_side_chains      Enable SC packing (default: True)
   --sc_num_denoising_steps INT  Denoising steps (default: 3)
   --omit_AA STR           Amino acids to omit (default: CM)
   --use_apptainer         Execute via Apptainer
   --apptainer_image PATH  Image path (default: /software/containers/universal.sif)

 Catalytic Residues:
   --catres_subset STR     Comma-separated indices for tight constraints (default: ALL)
   --redesign_non_subset_catres  Allow redesigning non-subset catres

 Constraints:
   --ref_pdb PATH          Reference PDB for ideal geometry (optional)
   --cst_file PATH         Legacy constraint file (optional)
   --primary_contact_cutoff FLOAT   Primary cutoff (default: 3.6 A)
   --secondary_contact_cutoff FLOAT Secondary cutoff (default: 4.2 A)
   --metal_cutoff FLOAT    Metal coordination cutoff (default: 2.6 A)
   --contact_cutoff FLOAT  Set both primary+secondary
   --coord_cst_weight FLOAT  Coordinate constraint weight (default: 100.0)
   --coord_cst_stdev FLOAT   Constraint stdev (default: 0.01 A)

 Relax:
   --rosetta_path PATH     Rosetta installation (default: /software/rosetta/latest)
   --pyrosetta_path PATH   PyRosetta path (default: /software/pyrosetta/latest)
   --scorefunction STR     Scorefunction (default: beta_jan25)
   --use_pyrosetta         Use PyRosetta (default: True)
   --fastrelax_cycles INT  FastRelax cycles (default: 2)
   --mobile_radius FLOAT   Mobile region radius (default: 10.0 A)
   --cart_bonded_weight FLOAT  Cartesian bonded weight (default: 0.5)
   --weight_flexibility    Use B-factor/pLDDT for flexibility weighting

 Pipeline:
   --n_cycles INT          Design cycles (default: 3)
   --n_candidates INT      Candidates per MPNN run (default: 10)
   --n_keep INT            Keep per cycle (default: 2)
   --n_final INT           Final output quota (default: 10)

 Execution:
   --single_thread         Disable on-node parallelism
   --dry_run               Print commands without executing
   --verbose / --quiet     Verbosity control (default: verbose)
   --log_file PATH         Log file path

 Slurm:
   --generate_sbatch       Generate sbatch script instead of running
   --slurm_partition STR   Slurm partition
   --slurm_time STR        Time limit (default: 4:00:00)
   --slurm_cpus INT        CPUs (default: 8)
   --slurm_mem STR         Memory (default: 16G)

 ---
 4. Key Implementation Details

 4.1 REMARK 666 Parsing

 Parse format: REMARK 666 MATCH TEMPLATE <chain> <resname> <resnum> MATCH MOTIF <chain> <resname> <resnum> <cst_block> <cst_var>

 Extract: catres_index (1-indexed order), chain, resnum, resname, raw_line

 4.2 Contact Detection (Two-Tier)

 - Metal contacts (2.6 A): Zn/Fe/Mg/Ca coordination with N/O/S - priority 100
 - Primary contacts (3.6 A): Heteroatom (N/O/S) involved - priority 30-50
 - Secondary contacts (4.2 A): Any heavy atom - priority 5-10
 - Carbon-carbon: Deprioritized (priority 5)
 - Ignore hydrogens entirely

 4.3 Constraint Generation (Self-Derived Default)

 If no --ref_pdb: derive constraints from input structure geometry:
 1. Coordinate constraints: All ligand heavy atoms + catres contact atoms
   - HarmonicFunc(0.0, stdev=0.01) for tight preservation
 2. Distance constraints: From detected contacts
   - Metal: stdev 0.005 A (extra tight)
   - Primary: stdev 0.01 A
   - Secondary: stdev 0.05 A

 Output files:
 - constraints.cst (human-readable Rosetta format)
 - constraints_summary.json (structured metadata)

 4.4 MPNN Execution

 Build command with subprocess, support Apptainer wrapper:
 [apptainer exec IMAGE] python /path/run.py \
   --model_type ligand_mpnn --pdb_path X --out_folder Y \
   --temperature 0.1 --number_of_batches 10 --batch_size 1 \
   --pack_side_chains 1 --fixed_residues_multi fixed.json \
   --enhance plddt_3_20240930-f9c9ea0f --omit_AA CM \
   --ligand_mpnn_use_side_chain_context 1

 Fixed residues JSON format: {"pdb_path": ["A150", "A152", ...]}

 4.5 PyRosetta Relax

 - Scorefunction: beta_jan25 with cart_bonded=0.5, coordinate_constraint=100.0, pro_close=0.0
 - Cartesian FastRelax with lbfgs_armijo_nonmonotone minimizer
 - MoveMap: enable BB+CHI for mobile region (10 A sphere), freeze ligands
 - Apply coordinate constraints via CoordinateConstraint(AtomID, AtomID(1,1), xyz, HarmonicFunc)

 4.6 Metrics & Filtering

 Key metrics:
 - mean_displacement, max_displacement for constraint atoms
 - pct_within_0.1A - percentage of constrained atoms within 0.1 A
 - cart_bonded score (should be < 2.0)
 - Total Rosetta score

 Selection: Sort by geometry quality (displacement metrics), keep best N

 ---
 5. Output Structure

 output_dir/
 ├── run_config.json           # Complete configuration snapshot
 ├── constraints.cst           # Generated constraint file
 ├── constraints_summary.json  # Constraint metadata
 ├── cycle_01/
 │   ├── mpnn/                 # MPNN outputs (seqs/, packed/)
 │   ├── relax/                # Relaxed structures
 │   ├── selected_000.pdb      # Best candidate(s)
 │   └── cycle_metrics.json
 ├── cycle_02/
 │   └── ...
 ├── cycle_03/
 │   └── ...
 ├── final/                    # Amplified final outputs
 │   ├── final_001.pdb
 │   └── final_sequences.fasta
 └── pipeline_summary.json     # Master summary across all cycles

 ---
 6. Implementation Milestones

 Milestone 1: Foundation (First)

 - Package structure + pyproject.toml
 - config.py - All dataclasses
 - remark666.py - Parser with tests
 - logging_config.py + utils.py
 - cli.py - Argument parsing shell

 Milestone 2: Contact Detection & Constraints

 - ligand.py - Ligand detection
 - contact_detection.py - Two-tier detection
 - constraints.py - Constraint generation + file output
 - Tests for contact detection and constraint generation

 Milestone 3: MPNN Integration

 - mpnn_runner.py - Subprocess + Apptainer
 - Fixed residues JSON generation
 - Result parsing (sequences, scores, packed PDBs)
 - Tests with dry_run mode

 Milestone 4: Relax Integration

 - relax_runner.py - PyRosetta initialization
 - Constraint application to pose
 - MoveMap generation
 - Cartesian FastRelax execution
 - Metric computation (displacement, scores)

 Milestone 5: Orchestration & Polish

 - metrics.py - Quality assessment
 - filtering.py - Candidate selection
 - orchestrator.py - Full cycle management
 - slurm.py - Sbatch template generation
 - Integration tests with test PDB

 ---
 7. Verification Plan

 1. Unit Tests: Each module has dedicated test file
   - test_remark666.py: Various REMARK 666 formats, edge cases
   - test_contact_detection.py: Verify cutoffs, heteroatom priority
   - test_constraints.py: Constraint file format validation
   - test_mpnn_runner.py: Command building, dry_run mode
 2. Integration Test: Full pipeline with test PDB
   - Verify output directory structure created
   - Check run_config.json contains all settings
   - Validate constraints.cst is valid Rosetta format
   - Confirm cycle directories contain expected outputs
 3. Manual Validation:
   - Run on known ZnEsterase PDB with REMARK 666
   - Compare constraint atom selection to manual inspection
   - Verify relaxed structures maintain catres geometry

 ---
 8. Agent Division (3 Agents)

 Agent A: Core Infrastructure

 - Package setup, config.py, remark666.py, cli.py, logging_config.py, utils.py, slurm.py
 - Output: Working CLI that parses args, loads config, generates sbatch

 Agent B: Contact Detection & Constraints

 - ligand.py, contact_detection.py, constraints.py
 - Output: Standalone constraint generation tested with sample PDBs

 Agent C: Execution & Orchestration

 - mpnn_runner.py, relax_runner.py, metrics.py, filtering.py, orchestrator.py
 - Output: Complete working pipeline

 Coordination: Agent A defines interfaces first, A+B work in parallel, C integrates both.

 ---
 9. Key Design Decisions

 1. PyRosetta over Rosetta binaries: Easier constraint application, metric computation, programmatic control
 2. Coordinate constraints as primary mechanism: Fixes atoms in absolute space, simpler than distance/angle networks
 3. Self-derived constraints default: More robust than requiring ref_pdb; preserves input geometry
 4. Cartesian FastRelax: Required for accurate constraint satisfaction with backbone movement
 5. Two-tier contact detection: Balances tight key-interaction preservation with flexible secondary packing

 ---
 10. Critical Files to Reference

 - Spec: /home/woodbuse/special_scripts/fastmpnn_design/evo_FastMPNNdesign/design_goal.md
 - Old script (patterns to avoid): /home/woodbuse/special_scripts/fastmpnn_design/fastmpnn_ZnEsterase_SETH_LINKED.py
 - Geometry preservation pattern: /home/woodbuse/special_scripts/scaffold_handling/idealize_rfdiffusion3_geometry.py
 - MPNN runner: /net/software/lab/fused_mpnn/seth_temp/run.py
