# Protocol Files

JSON-based protocol files for customizable, reproducible design workflows.

## Quick Start

```bash
# Use a protocol file
python -m modules.step03__fastmpnndesign.fastmpnn_design \
    --step02_json metrics.json \
    --params ligand.params \
    --protocol_file protocols/default.json \
    --output_dir results/
```

## Available Protocols

| File | Description | Time |
|------|-------------|------|
| `default.json` | Legacy‑inspired: early cart‑min cleanup + controlled clustering | ~20–30 min |
| `fast.json` | Quick prototyping | ~3 min |
| `balanced.json` | Moderate optimization with early geometry cleanup | ~8–12 min |
| `thorough.json` | Multi-round extensive optimization | ~20 min |
| `aggressive.json` | High diversity with lighter optimization | ~6–10 min |
| `design_only.json` | Single-round design + torsional relax | ~3–5 min |
| `geometry_only.json` | Cart relax only, no design | ~5–8 min |
| `breadth.json` | High-diversity exploration | ~10–15 min |
| `depth.json` | Low-temperature refinement | ~10–15 min |
| `iterative_refine.json` | Multi-stage refinement with selection | ~12–18 min |
| `progressive.json` | Core → shell expansion | ~10–14 min |
| `geometry_first.json` | Cart relax → design on optimized geometry | ~12–18 min |
| `design_secondary_shell.json` | Shell-focused design | ~8–12 min |
| `progressive_expansion.json` | Dynamic scope expansion example | ~12 min |

## Protocol Schema

```json
{
  "name": "protocol_name",
  "description": "What this protocol does",
  "version": "2.2",
  "layer_cuts": [6.0, 8.0, 12.0],
  "steps": [
    {"type": "step_type", "param1": "value1", ...}
  ]
}
```

## Sphere System

Distance-based classification from ligand:

| Sphere | Default Range | Behavior |
|--------|--------------|----------|
| `DESIGN_CORE` | 0-6Å | Redesigned by MPNN |
| `DESIGN_SHELL` | 6-8Å | Redesigned (with CB check) |
| `FLEX` | 8-12Å | Sidechains repack, sequence fixed |
| `FROZEN` | >12Å | Completely fixed |

## Step Types

### Design Steps

```json
{"type": "design_core", "temperature": 0.1, "num_designs": 8}
{"type": "design_core_shell", "temperature": 0.2, "num_designs": 16}
{"type": "design_shell_only", "temperature": 0.1, "num_designs": 4}
{"type": "design_global", "temperature": 0.3, "num_designs": 32}
```

### Multi‑Strategy MPNN (JSON only)

```json
{
  "type": "mpnn_multi",
  "strategies": [
    {"type": "design_core_shell", "temperature": 0.2, "num_designs": 2},
    {"type": "design_core_shell_flex", "temperature": 0.2, "num_designs": 2}
  ]
}
```

Outputs are pooled and de‑duplicated before continuing to the next step.

### Relaxation Steps

```json
{"type": "cart_relax", "repeats": 2, "stages": 3, "until_converged": true}
{"type": "torsional_relax", "repeats": 1, "stages": 3}
{"type": "minimize", "tolerance": 0.01, "cartesian": false}
{"type": "repack", "repack_scope": "core_shell_flex"}
```

Repack scopes:
- `core` - Only DESIGN_CORE (0-6Å)
- `core_shell` - DESIGN_CORE + DESIGN_SHELL (0-8Å)
- `core_shell_flex` - All within flex boundary (0-12Å) [default]
- `global` - All non-fixed residues

Notes:
- Repacking never includes catalytic residues (catres_subset) or other fixed residues.
- If `repack_shell` is provided (legacy), it overrides `repack_scope`.

### Selection Steps

```json
{"type": "select_best", "n": 1, "metric": "geometry"}
{"type": "select_best", "n": 5, "metric": "score"}
{"type": "select_best", "n": 3, "metric": "smart"}
```

Metrics: `geometry`, `score`, `smart`, `ca_rmsd`, `sequence_similarity_high`, `sequence_similarity_low`

### Configuration Steps

```json
{"type": "set_layer_cuts", "core_cutoff": 5.0, "shell_cutoff": 7.0, "flex_cutoff": 11.0}
{"type": "scale", "terms": {"coordinate_constraint": 500.0, "fa_rep": 0.3}}
{"type": "set", "options": {"mpnn_temperature": 0.15}}
{"type": "time_check", "max_elapsed": 3600, "then": [{"type": "cluster", "method": "sequence", "n_clusters": 10}]}
```

### Interaction Conservation

```json
{"type": "keep_interactions", "target": "ligand", "types": ["hbond"], "probability": 0.75}
```

## CLI Layer Cuts Override

```bash
# Override layer cuts from CLI
python -m modules.step03__fastmpnndesign.fastmpnn_design \
    --step02_json metrics.json \
    --params ligand.params \
    --protocol_file protocols/default.json \
    --layer_cuts 5.0 7.0 11.0 \
    --output_dir results/
```

## Creating Custom Protocols

1. Copy an existing protocol as a starting point
2. Modify steps and parameters
3. Add comments for documentation:
   ```json
   {"type": "comment", "text": "This explains the next step"}
   ```
4. Validate with:
   ```bash
   PYTHONPYCACHEPREFIX=/tmp python -m modules.step03__fastmpnndesign.test.test_protocol_file_parser
   ```

## Best Practices

1. **Always follow MPNN with relaxation** - prevents clashes
2. **Use cart_relax for geometry-critical work** - optimizes bond geometry
3. **Temperature annealing** - start high (0.3), end low (0.1)
4. **Select best after cart_relax** - reduce branching before expensive steps
5. **Progressive scope expansion** - start core, expand to shell
