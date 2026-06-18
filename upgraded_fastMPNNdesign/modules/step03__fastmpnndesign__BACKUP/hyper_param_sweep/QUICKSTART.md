# SLURM Hyperparameter Sweep - Quick Start

## 30-Second Start

```bash
cd /home/woodbuse/special_scripts/upgraded_fastMPNNdesign/modules/step03__fastmpnndesign/hyper_param_sweep

# Submit jobs
./submit_array.sh

# Monitor progress
./monitor_jobs.sh --watch

# Collect results (when done)
./collect_results.sh
```

## Common Commands

### Submit Jobs
```bash
# Default: 20 jobs in parallel
./submit_array.sh

# Custom parallelism
./submit_array.sh --max-parallel 10

# Custom resources
./submit_array.sh --time 8:00:00 --mem 32G --cpus 8

# Test without submitting
./submit_array.sh --dry-run
```

### Monitor Jobs
```bash
# Quick status check
./monitor_jobs.sh

# Watch continuously (Ctrl+C to exit)
./monitor_jobs.sh --watch

# Show failed jobs
./monitor_jobs.sh --failed

# Detailed per-task status
./monitor_jobs.sh --detailed
```

### SLURM Commands
```bash
# View your jobs
squeue -u $USER

# View specific job
squeue -j JOB_ID

# Cancel job
scancel JOB_ID

# View log
tail -f logs/sweep_JOBID_TASKID.out
```

### Collect Results
```bash
# Default: creates sweep_results.csv and sweep_results.json
./collect_results.sh

# Custom output
./collect_results.sh --output my_results.csv

# Analyze results
python analyze_sweep.py
```

## Typical Workflow

1. **Submit**: `./submit_array.sh`
2. **Get Job ID**: Note the job ID from output (e.g., 12345)
3. **Monitor**: `./monitor_jobs.sh --watch`
4. **Check Logs**: `tail -f logs/sweep_12345_*.out`
5. **Collect**: `./collect_results.sh` (when jobs finish)
6. **Analyze**: `python analyze_sweep.py`

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Jobs don't start | Check partition: `sinfo` |
| Jobs fail immediately | View error: `cat logs/sweep_*_1.err` |
| Wrong partition | Use: `./submit_array.sh --partition YOUR_PARTITION` |
| Need more time | Use: `./submit_array.sh --time 8:00:00` |
| Out of memory | Use: `./submit_array.sh --mem 32G` |

## File Locations

- **Commands**: `cmds/all_sweep_commands.txt`
- **Logs**: `logs/sweep_JOBID_TASKID.{out,err}`
- **Results**: `outputs/jobNNN__description/`
- **Summary**: `sweep_results.csv`, `sweep_results.json`

## Need Help?

```bash
./submit_array.sh --help
./monitor_jobs.sh --help
./collect_results.sh --help
```

See [README_SLURM.md](README_SLURM.md) for full documentation.
