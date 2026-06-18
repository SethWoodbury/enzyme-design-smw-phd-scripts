#!/bin/bash
#SBATCH --job-name=step03_sweep
#SBATCH --output=LOGS_DIR_PLACEHOLDER/sweep_%A_%a.out
#SBATCH --error=LOGS_DIR_PLACEHOLDER/sweep_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=batch
#SBATCH --array=1-NUM_JOBS%MAX_PARALLEL

# =============================================================================
# SLURM Array Job Template for Step03 Hyperparameter Sweep
# =============================================================================
#
# This template runs hyperparameter sweep jobs as a SLURM array job.
# Each array task executes one command from the all_sweep_commands.txt file.
#
# Jobs run on CPU only (no GPU needed for this sweep).
#
# USAGE:
#   Do not run this template directly. Use submit_array.sh instead, which will:
#   1. Generate sweep commands if needed
#   2. Count the number of jobs
#   3. Replace NUM_JOBS and MAX_PARALLEL placeholders
#   4. Submit to SLURM with sbatch
#
# SLURM ARRAY TASK ID:
#   SLURM_ARRAY_TASK_ID ranges from 1 to NUM_JOBS
#   Each task reads the corresponding line from all_sweep_commands.txt
#
# RESOURCE REQUIREMENTS:
#   - Time: 4 hours per job (adjust if needed)
#   - CPUs: 4 cores per job
#   - Memory: 16 GB per job
#   - Partition: batch (change if your cluster uses different partition names)
#
# LOGS:
#   - Output: logs/sweep_<JOB_ID>_<TASK_ID>.out
#   - Error:  logs/sweep_<JOB_ID>_<TASK_ID>.err
#
# =============================================================================

# SWEEP_DIR and CMDS_FILE are set by submit_array.sh when generating this script
# These must be absolute paths, not derived from BASH_SOURCE
SWEEP_DIR="SWEEP_DIR_PLACEHOLDER"
CMDS_FILE="CMDS_FILE_PLACEHOLDER"

# Verify commands file exists
if [ ! -f "$CMDS_FILE" ]; then
    echo "ERROR: Commands file not found: $CMDS_FILE"
    echo "Run generate_sweep.py first to create the commands file."
    exit 1
fi

# Get the command for this array task
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CMDS_FILE")

if [ -z "$CMD" ]; then
    echo "ERROR: No command found for array task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

# Print job information
echo "============================================================================="
echo "SLURM Array Job - Task ${SLURM_ARRAY_TASK_ID}"
echo "============================================================================="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Array Task:   ${SLURM_ARRAY_TASK_ID}"
echo "Node:         ${SLURMD_NODENAME}"
echo "CPUs:         ${SLURM_CPUS_PER_TASK}"
echo "Memory:       ${SLURM_MEM_PER_NODE} MB"
echo "Started:      $(date)"
echo "-----------------------------------------------------------------------------"
echo "Command (first 150 chars):"
echo "${CMD:0:150}..."
echo "============================================================================="
echo ""

# Execute the command
eval "$CMD"
STATUS=$?

# Print completion information
echo ""
echo "============================================================================="
echo "Task ${SLURM_ARRAY_TASK_ID} Completed"
echo "============================================================================="
echo "Finished:     $(date)"
echo "Exit status:  $STATUS"
echo "============================================================================="

exit $STATUS
