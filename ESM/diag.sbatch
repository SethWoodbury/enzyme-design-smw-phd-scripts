#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --time=00:05:00
#SBATCH --mem=2G
#SBATCH --output=/home/woodbuse/special_scripts/ESM/diag.%j.out
#SBATCH --error=/home/woodbuse/special_scripts/ESM/diag.%j.err

echo "host: $(hostname)"
echo "--- /net mount? ---"
mount | grep -E "/net|/mnt/net" | head
echo "--- ls /net/software/containers/users/ ---"
ls /net/software/containers/users/ 2>&1 | head
echo "--- ls /net/software/containers/users/woodbuse ---"
ls -la /net/software/containers/users/woodbuse 2>&1
echo "--- readlink chain ---"
readlink -f /net/software/containers/users/woodbuse 2>&1
echo "--- stat .sif ---"
stat /net/software/containers/users/woodbuse/esmc.sif 2>&1 | head
echo "--- /mnt/net visible? ---"
ls -la /mnt/net 2>&1 | head -5
echo "--- /mnt/net/software/containers/users/woodbuse ---"
ls -la /mnt/net/software/containers/users/woodbuse 2>&1 | head
echo "--- existing user dir: jbutch ---"
ls -la /mnt/net/software/containers/users/jbutch 2>&1 | head -3
ls -la /net/software/containers/users/jbutch 2>&1 | head -3
