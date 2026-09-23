#!/usr/bin/env python
"""Launch VMem demo-action runs from a JSONL manifest."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_vmem_demo_actions.py"

SUPPORTED_KEYS = {
    "image": "--image",
    "run_id": "--run-id",
    "trajectory": "--trajectory",
    "pattern": "--pattern",
    "duration_seconds": "--duration-seconds",
    "num_actions": "--num-actions",
    "fps": "--fps",
    "step_size": "--step-size",
    "frames_per_action": "--frames-per-action",
    "seed": "--seed",
    "rng_mode": "--rng-mode",
    "clip_attention": "--clip-attention",
    "cut3r_attention": "--cut3r-attention",
    "memory_policy": "--memory-policy",
    "memory_budget": "--memory-budget",
    "memory_scope": "--memory-scope",
    "inference_steps": "--inference-steps",
    "surfel_niter": "--surfel-niter",
    "surfel_reconstruction_window": "--surfel-reconstruction-window",
    "save_frames": "--save-frames",
    "visualize_intermediates": "--visualize-intermediates",
    "checkpoint_every": "--checkpoint-every",
    "frame_storage": "--frame-storage",
}


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            row.setdefault("_manifest_line", line_number)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} did not contain any runnable manifest rows")
    return rows


def _append_arg(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    command.extend([flag, str(value)])


def _command_for_row(
    row: dict[str, Any],
    *,
    output_root: Path,
    config: Path,
    device: str,
    dry_run: bool,
    experiment_lock: Path | None = None,
    resume_from: Path | None = None,
) -> list[str]:
    unknown_keys = sorted(
        key for key in row if key not in SUPPORTED_KEYS and not key.startswith("_")
    )
    if unknown_keys:
        raise ValueError(f"Unsupported manifest keys: {unknown_keys}")
    if "image" not in row:
        raise ValueError(f"Manifest line {row.get('_manifest_line')} is missing image")

    command = [
        sys.executable,
        str(RUNNER),
        "--output-root",
        str(output_root),
        "--config",
        str(config),
        "--device",
        device,
    ]
    for key, flag in SUPPORTED_KEYS.items():
        _append_arg(command, flag, row.get(key))
    if dry_run:
        command.append("--dry-run")
    if experiment_lock is not None:
        command.extend(["--experiment-lock", str(experiment_lock)])
    if resume_from is not None:
        command.extend(["--resume-from", str(resume_from)])
    return command


def _selected_indices(args, rows: list[dict[str, Any]]) -> list[int]:
    if getattr(args, "job_indices", None) is not None:
        if args.all or args.job_index is not None:
            raise ValueError("Use only one of --job-indices, --job-index or --all")
        indices = args.job_indices
        if len(set(indices)) != len(indices) or any(index < 0 or index >= len(rows) for index in indices):
            raise ValueError("Job indices must be unique and within the manifest")
        return indices
    if args.all:
        return list(range(len(rows)))
    job_index = args.job_index
    if job_index is None:
        raw_task_id = args.slurm_array_task_id
        job_index = int(raw_task_id) if raw_task_id is not None else 0
    if job_index < 0 or job_index >= len(rows):
        raise IndexError(
            f"job index {job_index} is outside manifest range 0-{len(rows) - 1}"
        )
    return [job_index]


def run_logged(command, *, log_dir, index):
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = f"job{index:03d}_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}"
    log_path = log_dir / (stem + ".log")
    status_path = log_dir / (stem + ".json")
    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    record = {"command": command, "job_index": index, "log": str(log_path),
              "started_at": datetime.now().isoformat(), "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES")}
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        record.update(status="running", pid=process.pid)
        status_path.write_text(json.dumps(record, indent=2))
        print(f"VMem PID {process.pid}; follow progress with: tail -f {log_path}", flush=True)
        try:
            returncode = process.wait()
        finally:
            record.update(returncode=process.poll(), finished_at=datetime.now().isoformat(),
                          status="exited" if process.poll() is not None else "controller_interrupted")
            status_path.write_text(json.dumps(record, indent=2))
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--job-indices", type=int, nargs="+", help="Run only these rows sequentially.")
    parser.add_argument(
        "--slurm-array-task-id",
        default=os.environ.get("SLURM_ARRAY_TASK_ID"),
        help="Usually inherited from SLURM_ARRAY_TASK_ID by the sbatch wrapper.",
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/demo_actions_60s"))
    parser.add_argument("--config", type=Path, default=Path("configs/inference/inference.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--experiment-lock", type=Path)
    parser.add_argument("--resume-from", type=Path, help="Resume one selected row into a new run directory.")
    parser.add_argument("--log-dir", type=Path, help="Persistent child stdout/stderr logs; defaults under output-root.")
    args = parser.parse_args()

    rows = _load_manifest(args.manifest)
    indices = _selected_indices(args, rows)
    if args.resume_from is not None and len(indices) != 1:
        raise ValueError("--resume-from requires exactly one selected manifest row")
    for index in indices:
        row = rows[index]
        command = _command_for_row(
            row,
            output_root=args.output_root,
            config=args.config,
            device=args.device,
            dry_run=args.dry_run,
            experiment_lock=args.experiment_lock,
            resume_from=args.resume_from,
        )
        print(
            json.dumps(
                {
                    "manifest": str(args.manifest),
                    "job_index": index,
                    "manifest_line": row.get("_manifest_line"),
                    "run_id": row.get("run_id"),
                    "command": command,
                },
                indent=2,
            ),
            flush=True,
        )
        if args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        else:
            run_logged(command, log_dir=args.log_dir or args.output_root / "launcher_logs", index=index)


if __name__ == "__main__":
    main()
