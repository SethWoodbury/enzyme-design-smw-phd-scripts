# Enhanced FastMPNN Design

A modular, CLI-friendly package for enzyme active site design using LigandMPNN/EnhancedMPNN with PyRosetta.

This package is a modernized version of `fastmpnn_ZnEsterase_SETH_LINKED.py`, providing the same protocol logic with improved parameterization, modularity, logging, and robustness.

## Features

- **Fully parameterized CLI** with all MPNN options exposed
- **REMARK 666 parsing** for catalytic residue handling with subset support
- **EnhancedMPNN support** with model selection
- **Apptainer execution** for containerized MPNN runs
- **beta_jan25 scorefunction** (newest PyRosetta)
- **Modular architecture** for easy future extensions (ESM, etc.)
- **Comprehensive logging** with verbosity control
- **JSON configuration** save/load for reproducibility

## Installation

```bash
cd enhanced_FastMPNNDesign
pip install -e .
```

## Quick Start

```bash
# Basic design run
enhanced-fastmpnndesign --pdb input.pdb --nstruct 5

# With custom params and constraint file
enhanced-fastmpnndesign --pdb input.pdb --params ligand.params --cstfile design.cst

# Using apptainer for MPNN execution
enhanced-fastmpnndesign --pdb input.pdb --use_apptainer

# Specifying catres subset for tight geometry constraints
enhanced-fastmpnndesign --pdb input.pdb --catres_subset 1,2,6,10
```

## CLI Arguments

### Required
- `--pdb` - Input PDB file with ligand and REMARK 666 lines

### Structure Options
- `--nstruct` - Number of design iterations (default: 1)
- `--suffix` - Suffix for output filenames
- `--params` - Ligand/NCAA params files
- `--cstfile` - Matcher/enzdes constraint file

### Design Position Options
- `--design_pos` - Specific positions to redesign
- `--keep_pos` - Positions to keep fixed (repack allowed)
- `--detect_pocket` - Auto-detect designable positions

### Catalytic Residue Options
- `--catres_subset` - Comma-separated catres indices for tight geometry constraints (e.g., `1,2,6,10`)
- `--redesign_non_subset_catres` - Allow redesigning catres not in subset

### MPNN Options
- `--mpnn_runner` - Path to MPNN run.py
- `--model_type` - MPNN model type (default: ligand_mpnn)
- `--enhance` - EnhancedMPNN model (default: plddt_3_20240930-f9c9ea0f)
- `--no_enhance` - Disable EnhancedMPNN
- `--temperature` - Sampling temperature
- `--number_of_batches` - Number of batches
- `--batch_size` - Batch size
- `--pack_side_chains` - Enable side chain packing
- `--sc_num_denoising_steps` - Denoising steps
- `--omit_AA` - Amino acids to omit (default: CM)
- `--use_apptainer` - Use Apptainer container
- `--apptainer_image` - Container image path
- `--ligand_mpnn_use_side_chain_context` - Use SC context

### Bias Options
- `--position_bias` - Bias value for polar AAs
- `--bias_atoms` - Ligand atoms for bias calculation
- `--bias_AAs` - Amino acids to bias (default: KREDYQWSTH)

### Protocol Options
- `--protocol` - Protocol file defining design steps
- `--hbond_accept_probability` - Probability of keeping H-bond contacts

### Scoring Options
- `--scoring` - Custom scoring script
- `--filter` - Only output designs passing filters
- `--mpnn` - Run 2nd layer MPNN on outputs

### Execution Options
- `--scorefunction` - Rosetta scorefunction (default: beta_jan25)
- `--output_dir` - Output directory
- `--verbose` / `--quiet` - Verbosity control
- `--dry_run` - Print config without running

## Python API

```python
from fastmpnndesign import run_pipeline, RunConfig, MPNNConfig
from pathlib import Path

config = RunConfig(
    pdb=Path("input.pdb"),
    nstruct=5,
    mpnn=MPNNConfig(
        use_enhanced_mpnn=True,
        use_apptainer=True
    )
)

state = run_pipeline(config)
```

## Package Structure

```
fastmpnndesign/
├── __init__.py          # Package exports
├── cli.py               # CLI entry point
├── config.py            # Dataclass configurations
├── constants.py         # Centralized defaults
├── logging_config.py    # Logging setup
│
├── remark666.py         # REMARK 666 parsing
├── catres.py            # Catalytic residue handling
│
├── mpnn_runner.py       # MPNN subprocess/apptainer
├── mpnn_bias.py         # Position bias calculation
│
├── rosetta_init.py      # PyRosetta initialization
├── scorefunction.py     # Scorefunction setup
├── constraints.py       # CSTs class
├── layer_detection.py   # Design layer detection
├── relax.py             # Pre-relaxation
│
├── protocol.py          # Protocol parsing
├── task_operations.py   # HBondKeeper
│
├── scoring.py           # Scoring interface
├── orchestrator.py      # Main workflow
└── utils.py             # Utilities
```

## Key Differences from Original Script

1. **Modular architecture**: 19 focused modules vs 1 monolithic script
2. **Type-safe configuration**: Dataclasses with serialization
3. **CLI parameterization**: All MPNN options exposed
4. **beta_jan25 scorefunction**: Newest PyRosetta
5. **Apptainer support**: Containerized MPNN execution
6. **catres_subset**: Fine-grained catalytic residue control
7. **Comprehensive logging**: Verbosity control
8. **JSON reproducibility**: Save/load configurations

## Outputs

```
<output_dir>/
├── scores/<pdbname><suffix>.sc    # Scorefile
├── seqs/                          # 2nd layer MPNN outputs (if --mpnn)
└── <pdbname><suffix>_<iter>_<i>.pdb  # Designed structures
```

## Dependencies

- Python >= 3.9
- PyRosetta
- numpy
- pandas
- FastMPNNDesign (internal library)
