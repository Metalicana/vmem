#!/usr/bin/env bash
#SBATCH --job-name=vmem_mem_grid
#SBATCH --output=slurm/%x_%A_%a.out
#SBATCH --error=slurm/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --array=0-63

set -euo pipefail

VMEM_REPO_ROOT="${VMEM_REPO_ROOT:-$HOME/vmem}"
cd "$VMEM_REPO_ROOT"

if [ -n "${CONDA_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

IFS=',' read -r -a SCENE_ARRAY <<< "${SCENES:-${SCENE:-}}"
IFS=',' read -r -a DURATION_ARRAY <<< "${DURATIONS:-180}"
IFS=',' read -r -a POLICY_ARRAY <<< "${POLICIES:-unbounded,fifo,rarity_irreplaceability,slam_covisibility}"
IFS=',' read -r -a BUDGET_ARRAY <<< "${BUDGETS:-16,32,64,128}"

COMBOS=()
for scene in "${SCENE_ARRAY[@]}"; do
  scene="${scene//[[:space:]]/}"
  [ -n "$scene" ] || continue
  for duration in "${DURATION_ARRAY[@]}"; do
    duration="${duration//[[:space:]]/}"
    [ -n "$duration" ] || continue
    for policy in "${POLICY_ARRAY[@]}"; do
      policy="${policy//[[:space:]]/}"
      [ -n "$policy" ] || continue
      if [ "$policy" = "unbounded" ]; then
        COMBOS+=("$scene $duration $policy none")
      else
        for budget in "${BUDGET_ARRAY[@]}"; do
          budget="${budget//[[:space:]]/}"
          [ -n "$budget" ] || continue
          COMBOS+=("$scene $duration $policy $budget")
        done
      fi
    done
  done
done

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK_ID" -ge "${#COMBOS[@]}" ]; then
  echo "Task $TASK_ID is outside ${#COMBOS[@]} configured combos; exiting."
  exit 0
fi

read -r scene duration policy budget <<< "${COMBOS[$TASK_ID]}"
export SCENE="$scene"
export DURATION_SECONDS="$duration"
export MEMORY_POLICY="$policy"
if [ "$budget" = "none" ]; then
  unset MEMORY_BUDGET || true
else
  export MEMORY_BUDGET="$budget"
fi

echo "Running combo $TASK_ID/${#COMBOS[@]}: scene=$SCENE duration=$DURATION_SECONDS policy=$MEMORY_POLICY budget=${MEMORY_BUDGET:-none}"
bash scripts/run_vmem_memory_policy_smoke.sh
