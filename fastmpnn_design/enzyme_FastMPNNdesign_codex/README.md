# Remastered FastMPNNdesign

Modular pipeline for theozyme-guided enzyme design.

## Layout

- `utils/`: shared helpers (constants, logging, PDB IO, geometry, models).
- `stages/`: pipeline stages (each stage is isolated in its own folder).
- `run_pipeline.py`: subprocess orchestrator for stages.
- `tests/`: unit test scaffolding.

## Stage 1 usage

```bash
python stages/stage1_activesite_remaster/run.py \
  --input_pdb /path/to/input.pdb \
  --ref_pdb /path/to/ref.pdb \
  --output_pdb /path/to/output.pdb \
  --output_json /path/to/catres_catalog.json \
  --catres_subset 1,3,5 \
  --ligand_chain A \
  --ligand_resname XDW \
  --ligand_resno 0 \
  --verbose
```

## Pipeline wrapper usage

```bash
python run_pipeline.py \
  --input_pdb /path/to/input.pdb \
  --ref_pdb /path/to/ref.pdb \
  --catres_subset 1,3,5 \
  --outdir /path/to/outdir
```

Outputs are saved into `--outdir`, including `pipeline_state.json`.

## Pipeline test run

```bash
python run_pipeline.py --test
```

## Stage 1 test run

```bash
python stages/stage1_activesite_remaster/run.py --test
```

Stage-specific Apptainer settings:

```bash
python run_pipeline.py \
  --stage1_image /software/containers/universal.sif \
  --stage1_bind /net:/net,/tmp:/tmp \
  --stage1_env OMP_NUM_THREADS=8,CUDA_VISIBLE_DEVICES=0 \
  --input_pdb /path/to/input.pdb \
  --ref_pdb /path/to/ref.pdb \
  --outdir /path/to/outdir
```
