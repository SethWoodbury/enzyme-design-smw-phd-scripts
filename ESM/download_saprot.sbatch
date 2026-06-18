#!/bin/bash
#SBATCH --job-name=saprot_dl
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=download_saprot.%j.out
#SBATCH --error=download_saprot.%j.err

set -euo pipefail

ESMC_SIF="${ESMC_SIF:-/net/software/containers/users/woodbuse/esmc.sif}"
HUGGINGFACE="${HUGGINGFACE:-/net/databases/huggingface}"
SIF="$ESMC_SIF"
SCRIPT="${SLURM_SUBMIT_DIR:-$(pwd)}/download_saprot.py"

apptainer exec \
    --env HF_HUB_ENABLE_HF_TRANSFER=0 \
    "$SIF" python "$SCRIPT"

du -sh "$HUGGINGFACE"/saprot/hub/* || true
