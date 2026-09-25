#!/bin/bash -l
#SBATCH --job-name=vmem_setup
#SBATCH --partition=highgpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=vmem_setup_%j.out
#SBATCH --error=vmem_setup_%j.err

set -euo pipefail
source "${VMEM_ROOT:-$HOME/vmem}/slurm/newton_vmem_common.sh"
: "${VMEM_CUDA_WHEEL:?Set cu126 or cu128 matching the loaded nvcc toolkit}"
command -v nvcc
nvcc --version
# setup_vmem_newton.py resolves CC/CXX and compiles a preflight before pip.
ARGS=()
if [[ "${VMEM_FETCH_WEIGHTS:-no}" == yes ]]; then ARGS+=(--fetch-weights); fi
exec "$VMEM_PYTHON" -u scripts/setup_vmem_newton.py \
    --cuda "$VMEM_CUDA_WHEEL" "${ARGS[@]}" \
    --output "${VMEM_RESULTS_ROOT:-$HOME/vmem_results}/environment"
