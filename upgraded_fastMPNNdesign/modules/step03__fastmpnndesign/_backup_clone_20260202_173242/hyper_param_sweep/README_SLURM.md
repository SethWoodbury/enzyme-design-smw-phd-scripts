# SLURM Job Submission Infrastructure

This directory contains infrastructure for running Step03 FastMPNN design hyperparameter sweeps on SLURM clusters.

## Overview

The sweep infrastructure consists of:

1. **Command Generation**: Creates parameter sweep commands
2. **Job Submission**: Submits jobs as SLURM array tasks
3. **Monitoring**: Tracks job progress and identifies failures
4. **Results Collection**: Aggregates and analyzes results

## Quick Start

```bash
# 1. Generate sweep commands (if not already done)
python generate_sweep.py

# 2. Submit jobs to SLURM (20 jobs in parallel by default)
./submit_array.sh

# 3. Monitor progress
./monitor_jobs.sh

# 4. Collect results when complete
./collect_results.sh

# 5. Analyze results
python analyze_sweep.py
```

## Files

### Core Scripts

- **`sbatch_template.sh`**: SLURM array job template
  - Defines resource requirements (CPUs, memory, time)
  - Executes commands from the sweep
  - Do not run directly - use `submit_array.sh`

- **`submit_array.sh`**: Job submission script
  - Generates commands if needed
  - Configures and submits SLURM array job
  - Supports customization via command-line options

- **`monitor_jobs.sh`**: Job monitoring script
  - Shows SLURM queue status
  - Analyzes log files for success/failure
  - Can watch in real-time or show failed jobs

- **`collect_results.sh`**: Results collection script
  - Finds all output JSON files
  - Extracts key metrics
  - Creates summary CSV and JSON files

### Supporting Files

- **`generate_sweep.py`**: Generates parameter sweep commands
- **`analyze_sweep.py`**: Analyzes results and creates summary tables
- **`submit_sweep.sh`**: Alternative local execution (non-SLURM)

## Usage Details

### Submitting Jobs

Basic submission with defaults:
```bash
./submit_array.sh
```

Custom submission options:
```bash
# Run 10 jobs in parallel
./submit_array.sh --max-parallel 10

# Custom time limit and partition
./submit_array.sh --time 8:00:00 --partition long

# More memory per job
./submit_array.sh --mem 32G --cpus 8

# Dry run to see what would be submitted
./submit_array.sh --dry-run
```

Full options:
```
--max-parallel N     Maximum parallel array tasks (default: 20)
--time HH:MM:SS      Time limit per job (default: 4:00:00)
--partition PART     SLURM partition (default: batch)
--mem MEM            Memory per job (default: 16G)
--cpus N             CPUs per task (default: 4)
--dry-run            Show what would be submitted
--help               Show help message
```

### Monitoring Jobs

Basic monitoring:
```bash
./monitor_jobs.sh
```

Advanced monitoring:
```bash
# Monitor specific job
./monitor_jobs.sh --job-id 12345

# Continuously monitor (refreshes every 10 seconds)
./monitor_jobs.sh --watch

# Show failed jobs with error messages
./monitor_jobs.sh --failed

# Show detailed status for each task
./monitor_jobs.sh --detailed
```

Using SLURM commands directly:
```bash
# View queue status
squeue -u $USER

# View specific job
squeue -j JOB_ID

# Cancel job
scancel JOB_ID

# View log file
tail -f logs/sweep_JOBID_TASKID.out
```

### Collecting Results

Basic collection:
```bash
./collect_results.sh
```

Custom collection:
```bash
# Custom output file
./collect_results.sh --output my_results.csv

# JSON only
./collect_results.sh --format json

# CSV only
./collect_results.sh --format csv

# Require minimum completed jobs
./collect_results.sh --min-complete 10
```

Output files:
- **`sweep_results.csv`**: Spreadsheet-compatible results
- **`sweep_results.json`**: Machine-readable results

### Analyzing Results

```bash
# Show summary table
python analyze_sweep.py

# Export to custom CSV
python analyze_sweep.py --export_csv custom_results.csv

# Analyze specific output directory
python analyze_sweep.py --output_dir /path/to/outputs
```

## Directory Structure

```
hyper_param_sweep/
├── sbatch_template.sh          # SLURM array job template
├── submit_array.sh             # Submit jobs to SLURM
├── monitor_jobs.sh             # Monitor job progress
├── collect_results.sh          # Collect and summarize results
├── generate_sweep.py           # Generate sweep commands
├── analyze_sweep.py            # Analyze results
├── submit_sweep.sh             # Alternative local execution
├── cmds/
│   └── all_sweep_commands.txt  # Generated commands (one per line)
├── logs/
│   ├── sweep_JOBID_1.out       # Output logs for each array task
│   ├── sweep_JOBID_1.err       # Error logs for each array task
│   └── ...
└── outputs/
    ├── job001__preset_fast/    # Output directory for each job
    │   ├── fastmpnn_design_results.json
    │   ├── designs/
    │   └── ...
    └── ...
```

## Resource Requirements

Default settings (adjust as needed for your cluster):

- **CPUs**: 4 cores per job
- **Memory**: 16 GB per job
- **Time**: 4 hours per job
- **Partition**: batch (CPU-only, no GPU needed)

These can be customized with `submit_array.sh` options.

## Workflow Example

Complete workflow from start to finish:

```bash
# 1. Navigate to sweep directory
cd /home/woodbuse/special_scripts/upgraded_fastMPNNdesign/modules/step03__fastmpnndesign/hyper_param_sweep

# 2. Generate commands (creates cmds/all_sweep_commands.txt)
python generate_sweep.py

# 3. Submit to SLURM with 15 jobs in parallel
./submit_array.sh --max-parallel 15

# 4. Monitor progress (Ctrl+C to exit)
./monitor_jobs.sh --watch

# 5. Check for failures
./monitor_jobs.sh --failed

# 6. Collect results when jobs complete
./collect_results.sh

# 7. View results
cat sweep_results.csv
# or
python analyze_sweep.py

# 8. Open in spreadsheet
# Download sweep_results.csv and open in Excel or Google Sheets
```

## Troubleshooting

### Jobs fail immediately
- Check partition name: `sinfo` to see available partitions
- Check resource limits: `scontrol show partition PARTITION_NAME`
- View error log: `cat logs/sweep_JOBID_TASKID.err`

### Jobs don't start
- Check queue: `squeue -u $USER`
- Check job limit: `scontrol show job JOBID`
- Reduce `--max-parallel` if cluster has job limits

### No results found
- Wait for jobs to complete: `./monitor_jobs.sh`
- Check if jobs failed: `./monitor_jobs.sh --failed`
- Verify output directory exists: `ls outputs/`

### Commands file not found
- Generate commands: `python generate_sweep.py`
- Verify file exists: `ls cmds/all_sweep_commands.txt`

## Performance Tips

1. **Parallelization**: Adjust `--max-parallel` based on cluster policies
   - Too high: May hit job limits
   - Too low: Underutilizes cluster resources

2. **Resource Allocation**: Match resources to job requirements
   - Monitor actual usage: `sacct -j JOBID --format=JobID,MaxRSS,Elapsed`
   - Adjust `--mem` and `--cpus` if jobs are killed or idle

3. **Time Limits**: Set realistic time limits
   - Check actual runtimes: `./monitor_jobs.sh --detailed`
   - Set time slightly higher than expected runtime

4. **Checkpoint Results**: Jobs write results incrementally
   - Partial results available even if job times out
   - Use `--max_runtime` in sweep commands to control execution time

## Notes

- Jobs run on CPU only (no GPU needed for this sweep)
- Each job is independent and can be rerun if it fails
- Log files use format: `sweep_JOBID_TASKID.{out,err}`
- Results are written to: `outputs/jobNNN__description/fastmpnn_design_results.json`
- SLURM array tasks are 1-indexed (task IDs start at 1)

## Getting Help

For script usage:
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
