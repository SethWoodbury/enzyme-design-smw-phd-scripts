#!/bin/bash
#SBATCH --job-name=saprot_dl
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/woodbuse/special_scripts/ESM/download_saprot.%j.out
#SBATCH --error=/home/woodbuse/special_scripts/ESM/download_saprot.%j.err

set -euo pipefail

SIF=/net/software/containers/users/woodbuse/esmc.sif
SCRIPT=/home/woodbuse/special_scripts/ESM/download_saprot.py

apptainer exec \
    --env HF_HUB_ENABLE_HF_TRANSFER=0 \
    "$SIF" python "$SCRIPT"

du -sh /net/databases/huggingface/saprot/hub/* || true
