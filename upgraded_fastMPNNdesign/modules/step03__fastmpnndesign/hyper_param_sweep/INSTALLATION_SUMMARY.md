# SLURM Job Submission Infrastructure - Installation Summary

## Created Files

### Core SLURM Scripts (New)

1. **`sbatch_template.sh`** (3.2 KB)
   - SLURM array job template with resource specifications
   - Defines: 4 CPUs, 16GB RAM, 4-hour time limit, batch partition
   - Executes commands from `cmds/all_sweep_commands.txt`
   - Auto-populated by `submit_array.sh` before submission

2. **`submit_array.sh`** (5.5 KB)
   - Main submission script for SLURM array jobs
   - Supports command-line customization of resources
   - Auto-generates commands if needed
   - Creates `.sbatch_submit.sh` with proper job array settings
   - Options: `--max-parallel`, `--time`, `--partition`, `--mem`, `--cpus`, `--dry-run`

3. **`monitor_jobs.sh`** (8.1 KB)
   - Real-time job monitoring and status tracking
   - Parses SLURM queue and log files
   - Shows completed/failed/running counts
   - Options: `--job-id`, `--watch`, `--failed`, `--detailed`

4. **`collect_results.sh`** (6.0 KB)
   - Results aggregation and summary generation
   - Creates CSV and JSON output files
   - Integrates with `analyze_sweep.py`
   - Options: `--output`, `--format`, `--min-complete`

### Documentation (New)

5. **`README_SLURM.md`** (7.6 KB)
   - Complete documentation for SLURM infrastructure
   - Usage examples and troubleshooting guide
   - Resource requirements and workflow examples

6. **`QUICKSTART.md`** (2.3 KB)
   - Quick reference card for common commands
   - 30-second start guide
   - Troubleshooting table

### Existing Files (Not Modified)

- `generate_sweep.py` - Creates hyperparameter sweep commands
- `analyze_sweep.py` - Analyzes results and creates summary tables
- `submit_sweep.sh` - Alternative local execution (non-SLURM)

## File Permissions

All scripts are executable (`chmod +x` applied):
```
-rwxr-xr-x  sbatch_template.sh
-rwxr-xr-x  submit_array.sh
-rwxr-xr-x  monitor_jobs.sh
-rwxr-xr-x  collect_results.sh
```

## Directory Structure

```
hyper_param_sweep/
├── sbatch_template.sh          # SLURM array job template
├── submit_array.sh             # Submit jobs to SLURM (NEW)
├── monitor_jobs.sh             # Monitor job progress (NEW)
├── collect_results.sh          # Collect results (NEW)
├── README_SLURM.md             # Full documentation (NEW)
├── QUICKSTART.md               # Quick reference (NEW)
├── generate_sweep.py           # Generate sweep commands (existing)
├── analyze_sweep.py            # Analyze results (existing)
├── submit_sweep.sh             # Local execution (existing)
├── cmds/
│   └── all_sweep_commands.txt  # 21 generated commands
├── logs/                       # SLURM output logs (empty)
│   └── sweep_JOBID_TASKID.{out,err}  # Created by SLURM
└── outputs/                    # Job results (empty)
    └── jobNNN__description/    # Created by each job
```

## Current Sweep Configuration

**Total Jobs**: 21 commands generated

**Parameter Sweeps**:
- Protocols (JSON): fast, balanced, thorough, aggressive, design_only (5 jobs)
- MPNN temperatures: 0.1, 0.2, 0.3 (3 jobs)
- MPNN num designs: 4, 8, 16 (3 jobs)
- Design scope: primary only, primary+secondary (2 jobs)
- Conservation: on, off (2 jobs)
- Scorefunction combos: 2 combinations (2 jobs)
- Combined sweeps: protocol × temperature (4 jobs)

## Usage Workflow

### 1. Submit Jobs to SLURM
```bash
cd /home/woodbuse/special_scripts/upgraded_fastMPNNdesign/modules/step03__fastmpnndesign/hyper_param_sweep

# Submit with default settings (20 jobs in parallel)
./submit_array.sh

# Or customize resources
./submit_array.sh --max-parallel 10 --time 8:00:00 --mem 32G
```

### 2. Monitor Progress
```bash
# Quick status check
./monitor_jobs.sh

# Watch mode (auto-refresh every 10 seconds)
./monitor_jobs.sh --watch

# Show failed jobs with error messages
./monitor_jobs.sh --failed
```

### 3. Collect Results
```bash
# When jobs complete
./collect_results.sh

# Creates:
#   - sweep_results.csv
#   - sweep_results.json
```

### 4. Analyze Results
```bash
# View summary table
python analyze_sweep.py

# Or open CSV in spreadsheet
# (download sweep_results.csv)
```

## Default Resource Allocation

Per Job:
- **CPUs**: 4 cores
- **Memory**: 16 GB
- **Time Limit**: 4 hours
- **Partition**: batch (CPU-only, no GPU)

Total Resources (with 20 parallel):
- **CPUs**: 80 cores
- **Memory**: 320 GB
- **Wall Time**: ~4.2 hours (21 jobs ÷ 20 parallel × 4 hours)

## Customization Examples

### Run fewer jobs in parallel
```bash
./submit_array.sh --max-parallel 5
```

### Use more resources per job
```bash
./submit_array.sh --mem 32G --cpus 8
```

### Use different partition
```bash
./submit_array.sh --partition long --time 12:00:00
```

### Test before submitting
```bash
./submit_array.sh --dry-run
```

## SLURM Job Array Configuration

The submission creates a SLURM array job with:
```bash
#SBATCH --array=1-21%20
```

This means:
- **Total tasks**: 21 (one per sweep command)
- **Max parallel**: 20 (configurable via `--max-parallel`)
- **Task IDs**: 1-21 (each reads one line from commands file)

Each task:
- Gets unique `SLURM_ARRAY_TASK_ID` (1-21)
- Reads corresponding command from `cmds/all_sweep_commands.txt`
- Writes logs to `logs/sweep_JOBID_TASKID.{out,err}`
- Creates output in `outputs/jobNNN__description/`

## Log Files

Format: `logs/sweep_JOBID_TASKID.{out,err}`

Example:
```
logs/sweep_12345_1.out      # Output for task 1
logs/sweep_12345_1.err      # Error for task 1
logs/sweep_12345_2.out      # Output for task 2
...
```

Each log contains:
- Job metadata (ID, node, resources)
- Command being executed
- Output from fastmpnn_design.py
- Completion status and exit code

## Output Files

Format: `outputs/jobNNN__description/`

Example:
```
outputs/job001__protocol_fast/
├── fastmpnn_design_results.json    # Main results
├── designs/                         # PDB files
│   ├── design_001.pdb
│   └── ...
└── intermediate/                    # Intermediate files
```

## Result Collection

The `collect_results.sh` script:
1. Finds all `fastmpnn_design_results.json` files
2. Uses `analyze_sweep.py` to extract metrics
3. Creates `sweep_results.csv` with columns:
   - job_name
   - protocol
   - num_designs
   - avg_mutations
   - avg_seq_identity
   - avg_max_bond_dev
   - avg_ca_rmsd
   - best_bond_dev
   - best_mutations
   - runtime

## Verification

Test the infrastructure without submitting:
```bash
# Dry run to see what would be submitted
./submit_array.sh --dry-run

# Check generated SLURM script
cat .sbatch_submit.sh

# Verify commands exist
wc -l cmds/all_sweep_commands.txt
# Output: 21 cmds/all_sweep_commands.txt
```

## Next Steps

1. **Verify Cluster Configuration**
   ```bash
   # Check available partitions
   sinfo

   # Check partition limits
   scontrol show partition batch
   ```

2. **Test with Small Job**
   ```bash
   # Submit just 1 job to test
   # (edit .sbatch_submit.sh to set --array=1-1)
   ```

3. **Full Submission**
   ```bash
   # Submit all jobs
   ./submit_array.sh
   ```

4. **Monitor and Collect**
   ```bash
   # Watch progress
   ./monitor_jobs.sh --watch

   # Collect when done
   ./collect_results.sh
   ```

## Troubleshooting

See `README_SLURM.md` for detailed troubleshooting, including:
- Jobs fail immediately
- Jobs don't start
- Wrong partition
- Resource limits
- No results found

## Support

For help with any script:
```bash
./submit_array.sh --help
./monitor_jobs.sh --help
./collect_results.sh --help
```

For SLURM help:
```bash
man sbatch
man squeue
man scancel
```

## Files Created Summary

| File | Size | Purpose |
|------|------|---------|
| `sbatch_template.sh` | 3.2 KB | SLURM array job template |
| `submit_array.sh` | 5.5 KB | Submit jobs with customization |
| `monitor_jobs.sh` | 8.1 KB | Monitor job progress |
| `collect_results.sh` | 6.0 KB | Collect and summarize results |
| `README_SLURM.md` | 7.6 KB | Complete documentation |
| `QUICKSTART.md` | 2.3 KB | Quick reference guide |
| **Total** | **33.7 KB** | **6 new files** |

All scripts are executable and ready to use!
