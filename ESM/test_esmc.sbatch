#!/bin/bash
#SBATCH --job-name=esmc_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/woodbuse/special_scripts/ESM/test_esmc.%j.out
#SBATCH --error=/home/woodbuse/special_scripts/ESM/test_esmc.%j.err

set -euo pipefail

SIF=/net/software/containers/users/woodbuse/esmc.sif
SCRIPT=/home/woodbuse/special_scripts/ESM/test_esmc.py
HF=/net/databases/huggingface/esmc

echo "host: $(hostname), gpu: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

apptainer exec --nv \
    --env HF_HOME="$HF" \
    --env HF_HUB_CACHE="$HF/hub" \
    "$SIF" python "$SCRIPT"
