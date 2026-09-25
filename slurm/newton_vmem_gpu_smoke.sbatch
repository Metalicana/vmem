#!/bin/bash -l
#SBATCH --job-name=vmem_smoke
#SBATCH --partition=highgpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:10:00
#SBATCH --output=vmem_smoke_%j.out
#SBATCH --error=vmem_smoke_%j.err

set -euo pipefail
source "${VMEM_ROOT:-$HOME/vmem}/slurm/newton_vmem_common.sh"
exec timeout 300s "$VMEM_PYTHON" -u scripts/vmem_slurm.py \
    --output "${VMEM_RESULTS_ROOT:-$HOME/vmem_results}/gpu_smoke/${SLURM_JOB_ID}.json"
