# enzyme-design-smw-phd-scripts

Computational enzyme-design scripts and utilities developed by **Seth M. Woodbury**
during his PhD in the Baker Lab / Institute for Protein Design (IPD), University of
Washington. The collection covers theozyme and ligand handling, MPNN-based sequence
design, structure-prediction input/output processing, design filtering and metrics,
and a range of PDB/sequence utilities.

> **Environment note.** These scripts were written to run on the **IPD compute
> cluster** (SLURM scheduler, shared `/net/software`, `/software/containers/*.sif`,
> `/databases`, PyRosetta, etc.). Lab members on that cluster can run them as-is.
> Paths to shared cluster software are documented and centralized (see
> *Path configuration*, added alongside the path refactor). Off-cluster users will
> need the corresponding tools installed and the relevant paths pointed at them.

## Usage

Scripts are designed to be called directly from the command line and/or submitted to
SLURM. Most analysis tools expose a `--help`:

```bash
python theozyme_and_ligand_handling/make_cst_file_from_pdb__MAIN.py --help
```

Many multi-step tools follow a `*__MAIN.py` (orchestrator) + `*__STEP1_*.py` /
`*__STEP2_*.py` (workers) convention, where the `__MAIN` script dispatches the steps.

## Layout

| Directory | Contents |
|---|---|
| `advanced_structure_prediction_tools/` | AF3 / Chai input generation and output (CIF/PDB) processing |
| `af2_analysis_and_tools/` | AlphaFold2 output analysis, RMSD, ligand copying |
| `chemnet/` | ChemNet input prep and output processing |
| `design_filtering/` | Design metrics: contact counting, fpocket, size/shape, `metric_monster` |
| `ESM/` | ESM / ESM-C / SaProt scoring and mutation suggestion |
| `experimental_processing_scripts/` | Biophysical property calculations from designed sequences |
| `fastmpnn_design/`, `upgraded_fastMPNNdesign/` | MPNN-based enzyme sequence-design pipelines |
| `fast_relax_scripts/` | Constrained Rosetta FastRelax protocols |
| `general_utils/` | PDB/sequence utilities (REMARK 666, clustering, dedup, renaming) |
| `invrot/` | Inverse-rotamer analysis |
| `msa_tools/` | MSA generation |
| `notebook_functions/` | Notebook helpers and SLURM submission utilities |
| `pymol_rc_scripts/` | PyMOL helper scripts |
| `scaffold_handling/` | RFdiffusion output processing and geometry idealization |
| `site_saturation_mutagenesis/` | Golden Gate SSM design and assembly validation |
| `theozyme_and_ligand_handling/` | Theozyme / ligand / params / CST file generation |

Top-level libraries: `SimplePdbLib.py`, `SimpleXyzMath3.py`, `hbonding_network.py`,
and the `metrics_and_hbond_rosetta_*` scripts.

## For lab members (contributing)

This is the shared lab copy of these scripts. To collaborate:

```bash
git clone git@github.com:SethWoodbury/enzyme-design-smw-phd-scripts.git
# work on a branch, then open a PR / push:
git checkout -b my-feature
git push -u origin my-feature
```

Please branch for changes and open a pull request so changes are reviewable, rather
than committing directly to `main`.

## Acknowledgements

Several scripts were contributed by or adapted from colleagues in the Baker Lab / IPD
and remain credited to their authors:

- `from_indrek/` — Indrek Kalvet
- `from_declan/` — Declan (diffusion analysis)
- `*_donghyo*` (e.g. `theozyme_and_ligand_handling/random_ORI_donghyo.py`) — Donghyo
- `pymol_rc_scripts/shajesh_scripts_dir/` — Shajesh

The MPNN sequence-design engine (**`fused_mpnn`**, by Justas Dauparas) is an
**external dependency** and is **not** included in this repository. On the IPD
cluster it is available at
`/net/software/lab/scripts/enzyme_design/fused_mpnn_api`; set the `FUSED_MPNN_DIR`
environment variable to point at another location.

## License

MIT — see [LICENSE](LICENSE). Note that third-party/collaborator-contributed scripts
remain the property of their respective authors.
