#!/usr/bin/env bash
#SBATCH --job-name=vmem_demo_60s
#SBATCH --output=slurm/%x_%A_%a.out
#SBATCH --error=slurm/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

set -euo pipefail

VMEM_REPO_ROOT="${VMEM_REPO_ROOT:-$HOME/vmem}"
cd "$VMEM_REPO_ROOT"

if [ -n "${CONDA_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

MANIFEST="${MANIFEST:-manifests/vmem_60s_unbounded_smoke.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/vmem_results/demo_actions_60s}"
DEVICE="${DEVICE:-cuda}"

python scripts/run_vmem_demo_manifest.py "$MANIFEST" \
  --job-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --output-root "$OUTPUT_ROOT" \
  --device "$DEVICE"
