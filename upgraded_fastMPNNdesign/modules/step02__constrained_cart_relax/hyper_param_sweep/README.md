# Hyperparameter Sweep for Constrained Cartesian Relaxation

This directory contains tools for running hyperparameter sweeps on `constrained_cart_relax.py`.

## Final Optimized Defaults

Based on comprehensive sweeps (268 conditions across 3 sweep campaigns), these defaults are now set in the main script:

| Parameter | Optimized Value | Rationale |
|-----------|-----------------|-----------|
| `fa_rep_scale` | **0.5** | Reduced clash penalty allows better constraint satisfaction |
| `cart_bonded_weight` | **3.0** | Higher weight enforces better bond geometry |
| `cart_bonded_max` | **4.0** | Ceiling for adaptive scaling |
| `fastrelax_repeats` | **3** | Sufficient cycles |
| `fastrelax_ramp_stages` | **3** | 3×3=9 internal rounds (optimal) |
| `enable_bond_geometry_min` | **False** | Explicit minimization not needed |

**Expected results with defaults:**
- Max bond deviation: ~0.045 Å (target: <0.05 Å)
- Max angle deviation: ~15° (A116 ILE is inherently strained)
- Ligand RMSD: 0.00 Å (perfect constraint satisfaction)
- Constrained atom RMSD: 0.00 Å (perfect)
- CA RMSD: ~0.87 Å (target: <1.0 Å)
- Runtime: ~30 minutes

---

## Sweep History Summary

### Initial Sweep (108 jobs)
- Tested: scorefunctions, cart_bonded weights, FastRelax configs
- Key finding: **ref2015_cart** is 34% better than beta_jan25_cart

### Focused Sweep (160 jobs)
- Tested: cart_bonded 1.5-3.0, FastRelax 1×3 to 3×5, 5 replicates each
- Key finding: **cart_bonded=2.5**, minimal cycles sufficient

### Comprehensive Sweep (1536 jobs)
- Tested: fa_rep scaling, ramping, cart_bonded, FastRelax configs
- Key findings:
  - **fa_rep_scale=0.5** gives best bond geometry
  - **ramp_fa_rep=False** slightly better than ramping
  - **cart_bonded=3.0** optimal balance
  - **bond_geometry_min=False** slightly better

### Persistent Issue: A116 ILE

One catalytic residue (A116 ILE) fails angle tolerance in 100% of runs:
- Best achieved: ~14.75° (tolerance: 10°)
- Root cause: Inherent geometric strain from theozyme placement
- Recommendation: Accept this or relax `catres_angle_tolerance` to 15°

---

## Directory Structure

```
hyper_param_sweep/
├── README.md                    # This file
├── ANALYSIS.md                  # Initial sweep detailed analysis
├── generate_sweep.py            # Initial sweep generator
├── analyze_sweep.py             # Initial sweep analysis
├── submit_sweep.sh              # Initial sweep SLURM script
├── focused_sweep/
│   ├── generate_focused_sweep.py
│   ├── analyze_focused_sweep.py
│   └── submit_sweep.sh
└── comprehensive_sweep/
    ├── COMPREHENSIVE_ANALYSIS.md  # Full results from final sweep
    ├── generate_comprehensive_sweep.py
    ├── analyze_comprehensive_sweep.py
    └── submit_sweep.sh
```

---

## Running a New Sweep

### 1. Generate Commands

```bash
cd hyper_param_sweep/comprehensive_sweep
python generate_comprehensive_sweep.py
```

This creates:
- `cmds/all_sweep_commands.txt` - One command per line
- `cmds/job_ids.txt` - Job ID mapping
- `outputs/` directory for results
- `logs/` directory for SLURM output

### 2. Customize Parameters (Optional)

Edit `generate_comprehensive_sweep.py` to modify:

```python
# Score term scaling
FA_REP_SCALES = [0.3, 0.5, 0.7, 1.0]
RAMP_FA_REP = [True, False]

# Cart_bonded settings
CART_BONDED_WEIGHTS = [2.0, 2.5, 3.0, 3.5]
CART_BONDED_MAX = [4.0, 5.0]

# FastRelax configurations (repeats, stages)
FASTRELAX_CONFIGS = [
    (1, 3),   # 3 internal rounds
    (2, 3),   # 6 internal rounds
    (3, 3),   # 9 internal rounds
    (3, 5),   # 15 internal rounds
]

# Replicates per condition
N_REPLICATES = 3
```

### 3. Submit to SLURM

```bash
sbatch submit_sweep.sh
```

Monitor progress:
```bash
squeue -u $USER
sacct -j <JOBID> --format=State --noheader | sort | uniq -c
```

### 4. Analyze Results

After all jobs complete:

```bash
python analyze_comprehensive_sweep.py
```

This generates `COMPREHENSIVE_ANALYSIS.md` with:
- Top 30 configurations ranked by composite score
- Parameter effect analysis
- Summary statistics
- Recommended configuration

---

## SLURM Configuration

Default settings in `submit_sweep.sh`:

```bash
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem=4g           # ~1.3 GB actual usage
#SBATCH -t 02:00:00        # 2 hours (most jobs finish in 10-45 min)
#SBATCH -a 1-2304%100      # Array with 100 concurrent jobs max
#SBATCH --exclude=c1127    # Exclude problematic node
```

## Container

Use `/net/software/containers/universal.sif` (PyRosetta 2026.03):
- Has threading support (cxx11thread)
- Has serialization support
- Full path avoids symlink issues on compute nodes

```bash
apptainer exec /net/software/containers/universal.sif python constrained_cart_relax.py ...
```

Adjust `-a` range based on number of generated commands.

---

## Optimization Priorities

The analysis script ranks configurations by composite score:

1. **Bond geometry** (highest priority)
   - Target: max deviation < 0.05 Å

2. **Angle geometry**
   - Target: max deviation < 10° (often limited by A116)

3. **Catres failures**
   - Target: minimize failing catalytic residues

4. **Constraint satisfaction** (critical)
   - Ligand RMSD must be ~0.00 Å
   - Constrained atom RMSD must be ~0.00 Å

5. **CA RMSD**
   - Target: < 1.0 Å, ideally < 0.75 Å

6. **Runtime** (lower priority)

7. **Clashes** (lowest priority)

---

## Tips

- **Start small**: Test with 1-2 replicates first to verify setup
- **Check constraints**: All runs should have ligand/constrained RMSD ≈ 0.00
- **Monte Carlo variance**: Use N≥3 replicates for reliable statistics
- **Memory**: 4 GB is sufficient (actual usage ~1.3 GB)
- **Runtime**: 1×3 configs take ~10 min, 3×5 configs take ~45 min
