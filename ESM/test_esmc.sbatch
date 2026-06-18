#!/bin/bash
#SBATCH --job-name=esmc_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=test_esmc.%j.out
#SBATCH --error=test_esmc.%j.err

set -euo pipefail

ESMC_SIF="${ESMC_SIF:-/net/software/containers/users/woodbuse/esmc.sif}"
HUGGINGFACE="${HUGGINGFACE:-/net/databases/huggingface}"
SIF="$ESMC_SIF"
SCRIPT="${SLURM_SUBMIT_DIR:-$(pwd)}/test_esmc.py"
HF="$HUGGINGFACE/esmc"

echo "host: $(hostname), gpu: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

apptainer exec --nv \
    --env HF_HOME="$HF" \
    --env HF_HUB_CACHE="$HF/hub" \
    "$SIF" python "$SCRIPT"
