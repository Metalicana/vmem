# Sourced only inside a Slurm allocation. No GPU selection or module purge.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit via sbatch or run inside an allocated compute shell}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must provide a GPU mask}"
cd "${VMEM_ROOT:-$HOME/vmem}"
unset PYTHONHOME PYTHONPATH
export VMEM_PYTHON="${VMEM_PYTHON:-$HOME/.conda/envs/vmem/bin/python}"
test -x "$VMEM_PYTHON"
export PATH="$(dirname "$VMEM_PYTHON"):$PATH"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
if [[ -n "${VMEM_COMPILER_MODULE:-}" ]]; then
    module load "$VMEM_COMPILER_MODULE"
fi
if [[ -n "${VMEM_CUDA_MODULE:-}" ]]; then
    module load "$VMEM_CUDA_MODULE"
fi
if type module >/dev/null 2>&1; then module list; fi
for name in SLURM_JOB_ID SLURM_JOB_NODELIST SLURM_JOB_GPUS CUDA_VISIBLE_DEVICES; do
    printf '%s=%s\n' "$name" "${!name:-unset}"
done
git rev-parse HEAD
git status --short
readlink -f "$VMEM_PYTHON"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
# nvidia-smi above is node diagnostics only; admission uses CUDA logical device 0.
