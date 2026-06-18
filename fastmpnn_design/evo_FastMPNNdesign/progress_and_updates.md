# FastMPNN Design - Implementation Progress

## Overview

This document tracks the implementation progress of the `fastmpnndesign` package,
a modular Python tool for iterative protein active-site design.

---

## Implementation Status: COMPLETE

All core modules have been implemented as of the initial release.

---

## Package Structure

```
evo_FastMPNNdesign/
├── fastmpnndesign/
│   ├── __init__.py           ✓ Package exports and version
│   ├── cli.py                ✓ CLI entry point with argparse
│   ├── config.py             ✓ All dataclasses (RunConfig, MPNNConfig, etc.)
│   ├── constants.py          ✓ Element sets, distance cutoffs, defaults
│   ├── remark666.py          ✓ REMARK 666 parser
│   ├── ligand.py             ✓ Ligand detection utilities
│   ├── contact_detection.py  ✓ Two-tier contact detection
│   ├── constraints.py        ✓ Constraint generation + file output
│   ├── mpnn_runner.py        ✓ MPNN subprocess wrapper (+Apptainer)
│   ├── relax_runner.py       ✓ PyRosetta relax execution
│   ├── metrics.py            ✓ Quality metrics computation
│   ├── filtering.py          ✓ Candidate ranking/selection
│   ├── orchestrator.py       ✓ Pipeline cycle management
│   ├── slurm.py              ✓ Sbatch template generation
│   ├── logging_config.py     ✓ Logging setup
│   └── utils.py              ✓ Path handling, I/O utilities
├── tests/
│   ├── __init__.py           ✓
│   ├── test_remark666.py     ✓ REMARK 666 parser tests
│   ├── test_contact_detection.py ✓ Contact detection tests
│   ├── test_constraints.py   ✓ Constraint generation tests
│   ├── test_mpnn_runner.py   ✓ MPNN runner tests
│   └── fixtures/
│       └── sample.pdb        ✓ Test PDB file
├── pyproject.toml            ✓ Package configuration
├── design_goal.md            ✓ Original specification
└── progress_and_updates.md   ✓ This file
```

---

## Implementation Notes

### 1. REMARK 666 Parsing (remark666.py)

- Parses REMARK 666 MATCH lines from PDB files
- Extracts catalytic residue information: chain, resnum, resname, cst_block, cst_var
- Supports insertion codes
- Returns ordered list of CatalyticResidue objects

### 2. Ligand Detection (ligand.py)

- Identifies ligands from HETATM records
- Filters out solvents (HOH, WAT), buffers (GOL, PEG, etc.), and metal ions
- Separate detection for metals (ZN, FE, MG, CA, etc.)
- Provides center_of_mass calculation for ligands

### 3. Contact Detection (contact_detection.py)

- Two-tier detection system:
  - Metal contacts: 2.6 Å cutoff
  - Primary contacts: 3.6 Å cutoff
  - Secondary contacts: 4.2 Å cutoff
- Prioritizes heteroatom (N/O/S) contacts over carbon-carbon
- Returns Contact objects with priority scoring

### 4. Constraint Generation (constraints.py)

- Generates coordinate constraints for ligand atoms and catres contact atoms
- Generates distance constraints from detected contacts
- Self-derived mode: constraints from input PDB geometry
- Ref-based mode: constraints from reference PDB
- Outputs both .cst file (Rosetta format) and .json summary

### 5. MPNN Runner (mpnn_runner.py)

- Builds command line for LigandMPNN/EnhancedMPNN
- Supports direct execution and Apptainer wrapper
- Creates fixed_residues_multi JSON file
- Parses output sequences and scores

### 6. Relax Runner (relax_runner.py)

- PyRosetta-based Cartesian FastRelax
- Applies coordinate constraints
- Creates MoveMap with mobile region around ligand
- Computes displacement metrics after relaxation

### 7. Metrics (metrics.py)

- Geometry metrics: displacement from constraints, RMSD
- Sequence metrics: identity, mutations
- Scoring metrics: Rosetta energy terms

### 8. Filtering (filtering.py)

- Multiple ranking strategies: geometry quality, Rosetta score, MPNN score, combined
- Filter criteria for displacement and scoring
- Diversity-aware selection

### 9. Orchestrator (orchestrator.py)

- Manages iterative design cycles
- Coordinates MPNN → Relax → Filter → Repeat
- Handles final amplification and output generation

### 10. SLURM Integration (slurm.py)

- Generates sbatch scripts for single jobs
- Generates array job scripts for batch processing

---

## Design Decisions

1. **PyRosetta over Rosetta binaries**: Easier constraint application, metric computation, programmatic control

2. **Coordinate constraints as primary mechanism**: Fixes atoms in absolute space, simpler than complex distance/angle networks

3. **Self-derived constraints default**: More robust than requiring ref_pdb; preserves input geometry

4. **Cartesian FastRelax**: Required for accurate constraint satisfaction with backbone movement

5. **Two-tier contact detection**: Balances tight key-interaction preservation with flexible secondary packing

---

## Usage Examples

### Basic CLI Usage

```bash
# Basic run
fastmpnndesign --pdb input.pdb --params ligand.params

# With options
fastmpnndesign --pdb input.pdb --params ligand.params \
    --output_dir ./results \
    --n_cycles 4 \
    --n_candidates 20 \
    --catres_subset 1,2,6

# Generate sbatch script
fastmpnndesign --pdb input.pdb --params ligand.params --generate_sbatch

# Dry run
fastmpnndesign --pdb input.pdb --params ligand.params --dry_run
```

### Python API Usage

```python
from pathlib import Path
from fastmpnndesign import run_pipeline, RunConfig

config = RunConfig(
    pdb=Path("input.pdb"),
    params=[Path("ligand.params")],
    output_dir=Path("./output"),
    prefix="design"
)

state = run_pipeline(config)
print(f"Generated {len(state.final_candidates)} designs")
```

---

## Testing

Run tests with:

```bash
cd /home/woodbuse/special_scripts/fastmpnn_design/evo_FastMPNNdesign
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Known Limitations

1. PyRosetta must be installed and accessible in PYTHONPATH
2. MPNN runner path must exist (default: /net/software/lab/fused_mpnn/seth_temp/run.py)
3. Apptainer image required if using --use_apptainer

---

## Future Enhancements

- [ ] ESM perplexity integration for ranking
- [ ] Explicit atom mapping file for custom constraint frames
- [ ] Multiple scoring function support
- [ ] Web interface for job monitoring
