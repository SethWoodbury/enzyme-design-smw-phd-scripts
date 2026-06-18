#!/bin/bash
#SBATCH --job-name=step03_sweep
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=batch
# Note: Array range will be set dynamically by submit_array.sh

# ============================================================================
# Step03 FastMPNN Design Hyperparameter Sweep - SLURM Array Job
# ============================================================================
#
# This script runs one job from the hyperparameter sweep as a SLURM array task.
# Jobs are CPU-only (no GPU required).
#
# Metrics being optimized:
# 1. Constrained atom RMSD (should be ~0)
# 2. Bond geometry for unconstrained catres (<0.05A bond, <7.5 deg angle)
# 3. CA RMSD vs step01 (minimize to stay close to AlphaFold3)
# 4. No clashes involving catalytic residues
# 5. No mutations to catalytic residues
# 6. HIS tautomer preservation
# 7. Good secondary structure and Dunbrack rotamers
# 8. Runtime efficiency
#
# ============================================================================

set -e

# Get script directory
SWEEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SWEEP_DIR")")")"
CMDS_FILE="${SWEEP_DIR}/cmds/all_sweep_commands.txt"

# Logging
echo "=============================================="
echo "Step03 FastMPNN Sweep - Job ${SLURM_ARRAY_TASK_ID}"
echo "=============================================="
echo "Hostname: $(hostname)"
echo "Started: $(date)"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}"
echo ""

# Verify commands file exists
if [ ! -f "$CMDS_FILE" ]; then
    echo "ERROR: Commands file not found: $CMDS_FILE"
    exit 1
fi

# Get the command for this array task
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CMDS_FILE")

if [ -z "$CMD" ]; then
    echo "ERROR: No command found for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "Command (truncated): ${CMD:0:150}..."
echo ""

# Change to project root (commands are relative to this)
cd "$PROJECT_ROOT"

# Set environment
export PYTHONUNBUFFERED=1

# Execute the command
echo "Starting execution..."
START_TIME=$(date +%s)

eval "$CMD"
STATUS=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "Job ${SLURM_ARRAY_TASK_ID} completed"
echo "Exit status: $STATUS"
echo "Duration: ${DURATION} seconds ($((DURATION/60)) minutes)"
echo "Finished: $(date)"
echo "=============================================="

exit $STATUS
