#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IFS=',' read -r -a SCENE_ARRAY <<< "${SCENES:-${SCENE:-}}"
IFS=',' read -r -a DURATION_ARRAY <<< "${DURATIONS:-180}"
IFS=',' read -r -a POLICY_ARRAY <<< "${POLICIES:-unbounded,fifo,rarity_irreplaceability,slam_covisibility}"
IFS=',' read -r -a BUDGET_ARRAY <<< "${BUDGETS:-16,32,64,128}"

if [ "${#SCENE_ARRAY[@]}" -eq 0 ] || [ -z "${SCENE_ARRAY[0]//[[:space:]]/}" ]; then
  echo "Set SCENE or SCENES to one or more Context-as-Memory scene ids" >&2
  exit 2
fi

run_one() {
  local scene="$1"
  local duration="$2"
  local policy="$3"
  local budget="$4"

  scene="${scene//[[:space:]]/}"
  duration="${duration//[[:space:]]/}"
  policy="${policy//[[:space:]]/}"
  budget="${budget//[[:space:]]/}"

  echo
  echo "=== VMem scene=$scene duration=${duration}s policy=$policy budget=${budget:-none} ==="

  if [ "$policy" = "unbounded" ]; then
    SCENE="$scene" DURATION_SECONDS="$duration" MEMORY_POLICY="$policy" \
      bash "$SCRIPT_DIR/run_vmem_memory_policy_smoke.sh"
  else
    SCENE="$scene" DURATION_SECONDS="$duration" MEMORY_POLICY="$policy" MEMORY_BUDGET="$budget" \
      bash "$SCRIPT_DIR/run_vmem_memory_policy_smoke.sh"
  fi
}

for scene in "${SCENE_ARRAY[@]}"; do
  for duration in "${DURATION_ARRAY[@]}"; do
    for policy in "${POLICY_ARRAY[@]}"; do
      policy="${policy//[[:space:]]/}"
      [ -n "$policy" ] || continue
      if [ "$policy" = "unbounded" ]; then
        run_one "$scene" "$duration" "$policy" ""
      else
        for budget in "${BUDGET_ARRAY[@]}"; do
          budget="${budget//[[:space:]]/}"
          [ -n "$budget" ] || continue
          run_one "$scene" "$duration" "$policy" "$budget"
        done
      fi
    done
  done
done
