#!/bin/bash
#SBATCH -J cart_sweep
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem=4g
#SBATCH -t 02:00:00
#SBATCH --exclude=c1127
#SBATCH -o logs/sweep_%a.stdout
#SBATCH -e logs/sweep_%a.stderr
#SBATCH -a 1-2304%100

# ============================================================================
# COMPREHENSIVE HYPERPARAMETER SWEEP
# ============================================================================
#
# This sweep tests fa_rep scaling, cart_bonded settings, and FastRelax configs.
# See generate_comprehensive_sweep.py for parameter details.
#
# Jobs: 2304 (768 conditions × 3 replicates)
# Estimated time per job: ~15-30 min
# Estimated total CPU time: ~576-1150 hours
#
# Array limit %100 means max 100 jobs run simultaneously.
# Adjust based on cluster queue limits.
#
# Memory: 4 GB per job (based on previous sweep profiling)
# ============================================================================

# Get command for this array task
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SLURM_SUBMIT_DIR:-$(pwd)}/cmds/all_sweep_commands.txt")

echo "=============================================="
echo "Job ${SLURM_ARRAY_TASK_ID} starting at $(date)"
echo "Host: $(hostname)"
echo "=============================================="
echo ""
echo "Command:"
echo "${CMD}"
echo ""
echo "=============================================="

# Run the command
eval "${CMD}"

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Job ${SLURM_ARRAY_TASK_ID} finished at $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=============================================="

exit ${EXIT_CODE}
