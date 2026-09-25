#!/bin/bash -l
#SBATCH --job-name=vmem_results
#SBATCH --partition=highgpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --no-requeue
#SBATCH --output=vmem_results_%j.out
#SBATCH --error=vmem_results_%j.err

set -euo pipefail
source "${VMEM_ROOT:-$HOME/vmem}/slurm/newton_vmem_common.sh"
ROOT="${VMEM_RESULTS_ROOT:-$HOME/vmem_results}"
MODE="${VMEM_MODE:-smoke}"
COMMON=(run --gpu inherit --min-free-mib "${VMEM_MIN_FREE_MIB:-70000}")
case "$MODE" in
    smoke)
        ARGS=(--manifest manifests/vmem_newton_smoke_v1.jsonl --generation-only
              --output "$ROOT/newton_smoke_v1/reporting"
              --generation-root "$ROOT/newton_smoke_v1/generation"
              --lock "$ROOT/newton_smoke_v1/experiment_lock.json")
        ;;
    oxford60|suite60)
        if [[ "${VMEM_APPROVE_LONG:-}" != yes ]]; then
            printf '%s\n' 'Review the completed smoke report first, then explicitly set VMEM_APPROVE_LONG=yes.' >&2
            exit 2
        fi
        CUDA_VISIBLE_DEVICES="" "$VMEM_PYTHON" scripts/profile_vmem_newton.py \
            --output "$ROOT/newton_smoke_v1/reporting" --approve-long
        ARGS=(--manifest manifests/vmem_transfer_v3.jsonl
              --output "$ROOT/transfer_v3/${MODE}_reporting"
              --generation-root "$ROOT/transfer_v3/generation"
              --lock "$ROOT/transfer_v3/experiment_lock.json"
              --vbench-root "${VBENCH_ROOT:-$HOME/VBench}"
              --vbench-python "${VBENCH_PYTHON:-$HOME/.conda/envs/vbench/bin/python}")
        if [[ "$MODE" == suite60 ]]; then ARGS+=(--all-cases); fi
        ;;
    *) printf 'Unknown VMEM_MODE: %s\n' "$MODE" >&2; exit 2 ;;
esac
timeout 300s "$VMEM_PYTHON" scripts/vmem_slurm.py \
    --output "$ROOT/gpu_smoke/${SLURM_JOB_ID}.json"
# exec keeps the controller as the batch PID. B:USR1 reaches its checkpoint-aware handler.
exec "$VMEM_PYTHON" -u scripts/run_vmem_results.py "${COMMON[@]}" "${ARGS[@]}"
