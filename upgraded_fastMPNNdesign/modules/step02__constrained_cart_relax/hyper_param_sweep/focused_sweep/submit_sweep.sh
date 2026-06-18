#!/bin/bash
#SBATCH -J cart_focused
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem=4g
#SBATCH -t 02:00:00
#SBATCH -o /home/woodbuse/special_scripts/upgraded_fastMPNNdesign/modules/step02__constrained_cart_relax/hyper_param_sweep/focused_sweep/logs/sweep_%a.stdout
#SBATCH -e /home/woodbuse/special_scripts/upgraded_fastMPNNdesign/modules/step02__constrained_cart_relax/hyper_param_sweep/focused_sweep/logs/sweep_%a.stderr
#SBATCH -a 1-160%50

# Each job runs one command from the commands file
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /home/woodbuse/special_scripts/upgraded_fastMPNNdesign/modules/step02__constrained_cart_relax/hyper_param_sweep/focused_sweep/cmds/all_sweep_commands.txt)

echo "Job ${SLURM_ARRAY_TASK_ID} starting at $(date)"
echo "Command: ${CMD}"
echo "=========================================="

# Run the command
eval "${CMD}"

echo "=========================================="
echo "Job ${SLURM_ARRAY_TASK_ID} finished at $(date)"
