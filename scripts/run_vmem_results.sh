#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${VMEM_PYTHON:-$HOME/miniconda3/envs/vmem/bin/python}"
exec "$PYTHON" scripts/run_vmem_results.py "$@"
