#!/usr/bin/env bash
#SBATCH --job-name=vmem_mem_smoke
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1

set -euo pipefail

VMEM_REPO_ROOT="${VMEM_REPO_ROOT:-$HOME/vmem}"
cd "$VMEM_REPO_ROOT"

if [ -n "${CONDA_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

bash scripts/run_vmem_memory_policy_smoke.sh
