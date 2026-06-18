You are Claude Opus acting as a senior Python engineer + computational structural biology tooling expert (protein design pipelines, LigandMPNN/EnhancedMPNN, Rosetta/PyRosetta, constraints, HPC/Slurm, Apptainer). 

We are creating a new, clean, modular codebase called `fastMPNNdesign` with package name + working directory:
  /home/woodbuse/special_scripts/fastmpnn_design/evo_FastMPNNdesign

It will replace an older archaic script located at:
  /home/woodbuse/special_scripts/fastmpnn_design/fastmpnn_ZnEsterase_SETH_LINKED.py

Project docs workflow:
- My spec will live at:
  /home/woodbuse/special_scripts/fastmpnn_design/evo_FastMPNNdesign/design_goal.md
- You MUST keep an ongoing changelog / implementation journal at:
  /home/woodbuse/special_scripts/fastmpnn_design/evo_FastMPNNdesign/progress_and_updates.md
  (append updates as you implement, including decisions, assumptions, TODOs, and test notes; you can add additional internal notes files if useful but keep this updated)

The new codebase must be:
- parameterized with a robust CLI (helpful --help, sane defaults, validation, clear errors)
- modular (multi-file package layout from the start; importable functions; minimal monolithic logic)
- clean, readable, user-friendly, and “intelligently verbose” (logging with verbosity levels)
- robust on HPC (paths, subprocess safety, optional apptainer execution)
- Slurm-oriented (designed to submit/run under Slurm; optionally parallelize on-node unless --single_thread)
- designed for iterative cycles of:
    LigandMPNN sequence design -> Rosetta constrained relax -> filtering/ranking -> iterate 3–4 cycles
  -> output diverse high-quality sequences/structures + metrics

VERBOSITY / DEBUGGING REQUIREMENTS:
- Be extremely verbose by default (but controllable with --verbose/--quiet).
- Always log:
  * detected ligands and how they were identified
  * parsed REMARK 666 catres list and chosen catres_subset
  * detected interactions (counts, categories, and top N examples)
  * how constraint atoms were chosen (including why a given atom was selected; especially hetero-atom prioritization)
  * constraint tiers and weights (primary vs secondary; metal; any flexibility weighting)
  * exact Rosetta/MPNN/apptainer commands invoked
  * output paths created and summary of produced artifacts
- Emit a JSON “run_config” snapshot and a “constraints_summary.json” for every run/cycle/candidate.
- Handle edge cases gracefully (multiple ligands, multiple params, missing/odd PDB fields, unusual residue names like KCX, metals, extra chains, insertion codes).


CRITICAL CLARIFICATION: the unit of work is a SINGLE PDB (one input structure per run). 
Batching and Slurm arrays are supported by running multiple single-PDB jobs, not by one CLI invocation processing a directory.

========================
0) KEY HIGH-LEVEL INTENT
========================

Inputs are AF3-predicted backbones where I have copied a transition state ligand into the structure.
We redesign the active site around the ligand(s), but preserve catalytic residue identities from REMARK 666.

We do iterative design+relax:
- MPNN redesign (catres fixed by default)
- constrained relaxation that preserves/recovers key interaction geometry (very tight near ligand/catres network, esp. metal/H-bond networks)
- backbone movement is REQUIRED (but controlled; don’t let it go insane)
- filter and select best 1–2 per cycle primarily by geometry quality (modular ranking system for future changes)
- repeat cycles 3–4 (configurable)
- then amplify sequences to reach a final quota while maintaining diversity (spherical neighborhood around ligand)

========================
1) MPNN INTEGRATION SPECS
========================

Default MPNN runner path:
  /net/software/lab/fused_mpnn/seth_temp/run.py
…but provide a CLI argument to override it.

EnhancedMPNN:
- Must support toggling enhancedMPNN and selecting enhancement model.
- Default: enhancement ON with:
    --enhance plddt_3_20240930-f9c9ea0f
- Allow turning off or changing model via CLI.

Expose key LigandMPNN args via CLI including (at minimum):
- --model_type (default ligand_mpnn)
- --input_pdb (single PDB path; canonical unit of work)
- --output_dir
- --temperature
- --number_of_batches
- --batch_size
- --pack_side_chains
- --fixed_residues_multi (JSON file)
- --sc_num_denoising_steps
- --repack_everything
- --omit_AA (string like "MCX" or list) clean CLI
- --packed_suffix
- --ligand_mpnn_use_side_chain_context (default true/1, but parameterized)

Must support executing MPNN directly OR via apptainer:
- CLI flags like:
  --use_apptainer, --apptainer_image (default /software/containers/universal.sif)
- Command construction must be safe, logged, and robust.

We often run something like:
apptainer exec /software/containers/universal.sif python /net/software/lab/fused_mpnn/seth_temp/run.py \
  --model_type ligand_mpnn \
  --pdb_path <pdb> \
  --out_folder <out> \
  --temperature 0.1 \
  --number_of_batches 10 \
  --batch_size 1 \
  --pack_side_chains 1 \
  --fixed_residues_multi <json> \
  --sc_num_denoising_steps 3 \
  --repack_everything 0 \
  --enhance plddt_3_20240930-f9c9ea0f \
  --omit_AA MCX \
  --packed_suffix _eV1_T0_10_ \
  --ligand_mpnn_use_side_chain_context 1

======================================
2) CATALYTIC RESIDUES FROM REMARK 666
======================================

Critical rule: PDB contains REMARK 666 lines defining catalytic residues (catres).
Those residues should NEVER be redesigned during sequence design by default.

Example:
  REMARK 666 MATCH TEMPLATE B XDW  257 MATCH MOTIF A PHE  150  10  1

Interpretation:
- catres_index = 10
- chain A, residue number 150, residue name PHE

Parse ALL REMARK 666 lines into an ordered list of catres entries with fields:
- catres_index (1..N)
- chain
- resnum
- resname (3-letter)
- any extra fields if present (store raw line too)

CLI:
- --catres_subset 1,2,6,10
  Meaning: only those catres (by REMARK 666 index) participate in “tight geometry preservation constraints” during relax.
  Default: subset = ALL catres.
- --redesign_non_subset_catres
  If set: catres NOT in subset may be redesigned (otherwise they remain fixed too).
  IMPORTANT: catres are fixed by default; this flag is the only way to redesign catres outside subset.

=========================================
3) ROSETTA / PYROSETTA RELAX + CONSTRAINTS
=========================================

Use up-to-date Rosetta:
  /software/rosetta/latest
Default scorefunction:
  beta_jan25
Allow CLI override.
Apptainer Rosetta may be outdated; prefer /software/rosetta/latest if present.

PyRosetta option:
  /software/pyrosetta/latest
Support either:
- calling Rosetta binaries (preferred for HPC stability), OR
- using PyRosetta (helpful for geometry measurement and building constraint sets)
Make selectable via CLI.

Implementation choice: You do not need to pick a specific Rosetta executable preference from me. Choose what is easiest to code, modular, controllable, generalizable, and efficient. (RosettaScripts vs relax binary etc. is your choice.)

REMOVE dependency on "scorefile workflow". Do not require reading an external scorefile. Compute and output all metrics internally.

Constrained relax options:
A) Legacy: optional --cst_file for Rosetta constraint relax (still supported).
B) Ref-based: optional --ref_pdb that encodes ideal catalytic geometry relative to ligand(s).
C) DEFAULT IF NO REF:
   If --ref_pdb is NOT provided, then treat the current input PDB’s catres_subset geometry as the ideal geometry.
   In that mode, constraints are derived from the input itself (so relax preserves these atoms/interactions rather than trying to match an external reference).
   This is effectively “keep these interactions constant” and is easier/robust.

You should take inspiration from how to do “idealize/preserve geometry” from this script:
  /home/woodbuse/special_scripts/scaffold_handling/idealize_rfdiffusion3_geometry.py
(You do not need to copy its exact implementation; but use its conceptual approach for protecting geometry and/or idealizing.)

PARAMS / LIGANDS:
- A Rosetta params file WILL be specified via CLI.
- There may be multiple ligands; the tool should support multiple constraints per ligand to ensure correctness (do not assume one ligand).
- Matching assumptions are strong:
  - ref_pdb has same ligand residue name(s) + same ligand atom names as the input
  - ref_pdb has same catalytic residue names and uses canonical AA atom names
  - ligand atom naming is guaranteed identical between systems
So mapping can be name-based.

Constraint type preference:
- Prefer distance/angle/dihedral constraints relative to ligands/catres (because coordinate frames/backbones may not superimpose).
- You MAY optionally implement coordinate constraints as an internal representation, but only if you transform them correctly so they act relative to the ligand environment (i.e., do not assume global superposition).
- Rosetta doesn’t handle constraining hydrogens well; for H-bonds, constrain the attached heteroatom rather than the hydrogen itself.
- Output BOTH:
  1) a human-readable generated constraint file on disk (for debugging and reproducibility)
  2) internal constraint objects / structured JSON summary (for programmatic use)

How to choose “meaningful interacting atoms” (IMPORTANT):
- Use heuristic atom selection by default (Option C):
  - Automatically identify near-contact interactions between:
    * catres_subset <-> ligand(s)
    * catres_subset <-> catres_subset (network constraints)
  - Build constraint frames using ~3 atoms from each partner (as in Rosetta cst conventions).
  - Prioritize non-carbon atoms (N/O/S/metal) as the lead interacting atoms when multiple close atoms exist, especially for more distant constraints.
  - Prefer heavy atoms; ignore hydrogens for constraints.
- Make the system modular so a future explicit mapping/override file can be added, but not required now.

CONTACT / INTERACTION DETECTION (heuristic):
- Use a two-tier detection system:
  * primary_contact_cutoff default 3.6 Å (for contacts involving any non-carbon atom N/O/S/metal/halogen OR chemically important residues/ligand atoms)
  * secondary_contact_cutoff default 4.2 Å (general heavy-atom contacts for buttressing/packing/network support)
- Metal coordination detection: default metal_cutoff 2.6 Å for Zn–(N/O/S) type contacts (element-specific window is ok but keep it simple).
- The algorithm must:
  * consider only heavy atoms for detection and constraints (ignore hydrogens)
  * prioritize non-carbon atoms as lead interacting atoms when picking the “frame” atoms, especially for longer/secondary constraints
  * downweight or deprioritize pure carbon–carbon contacts unless no other atoms exist near that interaction

CLI flexibility:
- Allow overriding contact cutoffs via CLI:
  --primary_contact_cutoff, --secondary_contact_cutoff, --metal_cutoff
  (also allow a single --contact_cutoff as shorthand that sets both primary & secondary if provided)

PARAMS handling:
- Support single or multiple params:
  --params file1.params file2.params ...
- Robustly pass these through to Rosetta/PyRosetta invocations (and validate files exist).

Constraint tightness:
- For key contacts (closest atoms to ligand and key catres-catres networks like His/Glu/Zn coordination/H-bond networks), target extremely tight tolerances:
  - distance: aim ~<0.01 Å deviation for key constraints
  - angle: ~1–2 degrees
  - dihedral: similarly tight
- However: do NOT overconstrain everything immediately. Start with sane defaults for:
  - clashes
  - non-key geometry
  so that we can sample diverse space. Make constraint strengths tiered:
  - “primary” (very tight) vs “secondary” (looser) interaction sets.

Flexibility weighting (optional feature):
- Add an optional flag like --weight_flexibility that uses per-residue numeric fields from the PDB to guide Rosetta flexibility.
- It should detect if the PDB contains meaningful variation in that column.
- It should infer whether it is:
  - pLDDT-like (HIGH = confident/rigid), or
  - B-factor-like (LOW = confident/rigid).
- Then use it to modulate constraint strength and/or coordinate constraint weights or move-map flexibility heuristics.
- If detection fails or numbers are uniform, disable and warn.

Relax method:
- Must allow backbone movement (required).
- Default should be robust and controlled (cartesian FastRelax is likely good). Support both cartesian and non-cartesian relax modes if feasible.
- Repacking policy must be configurable. Default often:
  “repack everything except catres_subset”
- catres not in subset: allowed to relax freely.

=========================================
4) ITERATIVE PIPELINE / CYCLING / FILTERING
=========================================

Pipeline is iterative:
- Cycle 1: run LigandMPNN designs (catres fixed by default) -> candidates
- For each candidate: relax with constraints (subset; using ref_pdb if given, else self-derived ideal geometry) -> compute metrics
- Early filtering: keep only designs where catalytic geometry constraints are reasonably satisfied and match the ideal well.
- Pick best 1–2 after each cycle primarily based on geometry quality for now.
  (Ranking must be modular: later can incorporate ESM perplexity, etc.)
- Repeat 3–4 cycles total (configurable).
- Converge on strong catres geometry while producing diverse sequences.

After convergence:
- Amplify sequences to reach N_final quota while maintaining diversity.
- Diversity definition: spherical neighborhood around ligand (residues within X Å of ligand).
- Optionally expand to second sphere later; design should support both.

=========================================
5) METRICS / OUTPUTS
=========================================

Metrics should be cleaner than old script but cover:
- MPNN outputs: whatever is available (scores/probs, enhancement outputs, etc.)
- Rosetta: total score, selected score terms, constraint score/violations
- Geometry match to ideal for catres_subset:
  - distance/angle/dihedral deltas per interaction
  - summary metrics (RMS, max, percent within tolerance)
  - handle metal coordination + KCX reasonably (don’t overengineer)
- Structural: ligand RMSD if meaningful, local clash metrics around active site, Dunbrack/rotamer probability where available
- Sequence: identity, diversity metrics, clustering on residues within X Å of ligand

Outputs:
- organized output directory tree by cycle and candidate
- a single master CSV/JSON summarizing all candidates across cycles
- provenance: command lines, config snapshot, version stamp, environment hints
- final output: selected structures + sequences + summary report explaining selection

=========================================
6) IMPLEMENTATION REQUIREMENTS
=========================================

- Keep dependencies minimal. Argparse or Typer are ok.
- Use python logging with levels; support --verbose and --quiet; log to console + file.
- Use pathlib.
- Multi-file repo from the start. Clear separation of concerns:
  - parse REMARK 666 and catres definitions
  - MPNN command construction/execution (subprocess + apptainer option)
  - ligand identification + params integration
  - constraint generation:
      - ref_pdb-based AND self-derived (if no ref_pdb)
      - heuristic atom selection with hetero-atom priority
      - writes constraint file + JSON summary
  - relax execution (rosetta binary or pyrosetta)
  - metric computation
  - ranking/filtering + diversity selection
  - orchestrator for cycles
  - Slurm integration helpers (generate sbatch scripts, or provide templates)
- On-node parallelism: take advantage of cores by default unless --single_thread.

=========================================
7) MULTI-AGENT PLAN (NOT TOO MANY)
=========================================

Propose a 2–3 agent plan (max 3) to implement this efficiently:
- clear division of responsibilities
- mergeable outputs
- minimal coordination overhead

=========================================
8) WHAT YOU SHOULD PRODUCE
=========================================

Do NOT handwave. Produce concrete implementation-ready artifacts:

A) Short design doc (1–2 pages) architecture + data flow.
B) Proposed CLI spec: full `--help` style listing with defaults.
C) Repository/package layout for `evo_FastMPNNdesign` (files + responsibilities).
D) Core code skeleton (real code, not pseudocode) implementing:
   - CLI entrypoint
   - config dataclasses
   - REMARK 666 parsing
   - MPNN runner wrapper (subprocess + apptainer option)
   - constraint generation interfaces (ref-based + self-derived)
   - relax runner interfaces (rosetta binary + optional pyrosetta path)
   - metric computation stubs with clear return schemas
   - orchestration of cycles (3–4 default)
   - Slurm helper module (sbatch template generation for single-PDB jobs)
E) Incremental implementation plan (milestones) + where unit tests should exist.
F) Brief 2–3 agent division-of-labor plan.

If anything is ambiguous, make reasonable assumptions, state them explicitly, and proceed.
