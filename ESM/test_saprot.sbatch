#!/bin/bash
#SBATCH --job-name=saprot_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=test_saprot.%j.out
#SBATCH --error=test_saprot.%j.err

set -euo pipefail

ESMC_SIF="${ESMC_SIF:-/net/software/containers/users/woodbuse/esmc.sif}"
HUGGINGFACE="${HUGGINGFACE:-/net/databases/huggingface}"
SIF="$ESMC_SIF"
SCRIPT="${SLURM_SUBMIT_DIR:-$(pwd)}/test_saprot.py"
HF="$HUGGINGFACE/saprot"

echo "host: $(hostname), gpu: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

apptainer exec --nv \
    --env HF_HOME="$HF" \
    --env HF_HUB_CACHE="$HF/hub" \
    "$SIF" python "$SCRIPT"
