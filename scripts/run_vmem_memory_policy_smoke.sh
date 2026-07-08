#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VMEM_REPO_ROOT="${VMEM_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$VMEM_REPO_ROOT"

DATA_ROOT="${DATA_ROOT:-${CONTEXT_MEMORY_ROOT:-}}"
if [ -z "$DATA_ROOT" ]; then
  echo "Set DATA_ROOT to Context-as-Memory-Dataset/Context-as-Memory-Dataset" >&2
  exit 2
fi

SCENE="${SCENE:-}"
if [ -z "$SCENE" ]; then
  echo "Set SCENE to a Context-as-Memory scene id" >&2
  exit 2
fi

MEMORY_POLICY="${MEMORY_POLICY:-unbounded}"
MEMORY_BUDGET="${MEMORY_BUDGET:-}"
MEMORY_SCOPE="${MEMORY_SCOPE:-surfel_indexed_view_memory}"

if [ "$MEMORY_POLICY" != "unbounded" ] && [ -z "$MEMORY_BUDGET" ]; then
  echo "MEMORY_POLICY=$MEMORY_POLICY requires MEMORY_BUDGET" >&2
  exit 2
fi

FPS="${FPS:-13}"
DURATION_SECONDS="${DURATION_SECONDS:-180}"
NUM_FRAMES="${NUM_FRAMES:-$((DURATION_SECONDS * FPS + 1))}"
START_FRAME="${START_FRAME:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-4}"

STORAGE_ROOT="${STORAGE_ROOT:-/data/ab575577/vmem}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$STORAGE_ROOT/outputs/memory_policy}"

DEVICE="${DEVICE:-cuda}"
CAMERA_CONVENTION="${CAMERA_CONVENTION:-unreal}"
POSE_SCALE="${POSE_SCALE:-0.01}"
ROTATION_ORDER="${ROTATION_ORDER:-xyz}"
NATIVE_STEP_SIZE="${NATIVE_STEP_SIZE:-0.025}"
CONFIG="${CONFIG:-configs/inference/inference.yaml}"
SEED="${SEED:-42}"

cmd=(
  python scripts/run_context_memory_vmem.py "$DATA_ROOT"
  --scene "$SCENE"
  --start-frame "$START_FRAME"
  --num-frames "$NUM_FRAMES"
  --chunk-size "$CHUNK_SIZE"
  --fps "$FPS"
  --output-root "$OUTPUT_ROOT"
  --memory-policy "$MEMORY_POLICY"
  --memory-scope "$MEMORY_SCOPE"
  --config "$CONFIG"
  --device "$DEVICE"
  --seed "$SEED"
  --camera-convention "$CAMERA_CONVENTION"
  --pose-scale "$POSE_SCALE"
  --rotation-order "$ROTATION_ORDER"
  --native-step-size "$NATIVE_STEP_SIZE"
)

if [ -n "$MEMORY_BUDGET" ]; then
  cmd+=(--memory-budget "$MEMORY_BUDGET")
fi
if [ -n "${INFERENCE_STEPS:-}" ]; then
  cmd+=(--inference-steps "$INFERENCE_STEPS")
fi
if [ -n "${SURFEL_NITER:-}" ]; then
  cmd+=(--surfel-niter "$SURFEL_NITER")
fi
if [ -n "${SURFEL_RECONSTRUCTION_WINDOW:-}" ]; then
  cmd+=(--surfel-reconstruction-window "$SURFEL_RECONSTRUCTION_WINDOW")
fi
if [ "${SAVE_GT_VIDEO:-false}" = "true" ]; then
  cmd+=(--save-gt-video)
fi
if [ "${SAVE_FRAMES:-false}" = "true" ]; then
  cmd+=(--save-frames)
fi
if [ "${VISUALIZE_INTERMEDIATES:-false}" = "true" ]; then
  cmd+=(--visualize-intermediates)
fi

echo "VMem repo: $VMEM_REPO_ROOT"
echo "Data root: $DATA_ROOT"
echo "Scene: $SCENE"
echo "Policy: $MEMORY_POLICY"
echo "Budget: ${MEMORY_BUDGET:-none}"
echo "Scope: $MEMORY_SCOPE"
echo "Frames: $NUM_FRAMES at ${FPS}fps"
echo "Output root: $OUTPUT_ROOT"

if [ "${DRY_RUN:-false}" = "true" ] || [ "${1:-}" = "--dry-run" ]; then
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
