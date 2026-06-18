#!/bin/bash
#SBATCH -J cart_relax_sweep
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem=4g
#SBATCH -t 02:00:00
#SBATCH -o logs/sweep_%a.stdout
#SBATCH -e logs/sweep_%a.stderr
#SBATCH -a 1-108%50

# Each job runs one command from the commands file
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SLURM_SUBMIT_DIR:-$(pwd)}/cmds/all_sweep_commands.txt")

echo "Job ${SLURM_ARRAY_TASK_ID} starting at $(date)"
echo "Command: ${CMD}"
echo "=========================================="

# Run the command
eval "${CMD}"

echo "=========================================="
echo "Job ${SLURM_ARRAY_TASK_ID} finished at $(date)"
