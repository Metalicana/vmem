#!/usr/bin/env python
"""Summarize and sanity-check a Context-as-Memory VMem run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Iterable


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path(value: str | None, *, run_dir: Path, default_name: str) -> Path:
    if value:
        path = Path(value)
        return path if path.is_absolute() else run_dir / path
    return run_dir / default_name


def _as_int_list(value: Iterable | None) -> list[int]:
    return [int(item) for item in value or []]


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _record_brief(record: dict) -> dict:
    return {
        "global_step": record.get("global_step"),
        "num_stored_views": record.get("num_stored_views"),
        "target_frame_indices": record.get("target_frame_indices", []),
        "target_dataset_frame_indices": record.get("target_dataset_frame_indices", []),
        "allowed_memory_indices": record.get("allowed_memory_indices", []),
        "selected_context_indices": record.get("selected_context_indices", []),
        "selected_dataset_frame_indices": record.get("selected_dataset_frame_indices", []),
        "raw_candidate_count": record.get("raw_candidate_count", 0),
        "bounded_candidate_count": record.get("bounded_candidate_count", 0),
        "fallback_used": record.get("fallback_used", False),
    }


def summarize_run(path: Path) -> dict:
    metadata_path = path if path.name == "metadata.json" else path / "metadata.json"
    run_dir = metadata_path.parent
    metadata = _read_json(metadata_path)

    trace_path = _resolve_path(
        metadata.get("retrieval_trace"),
        run_dir=run_dir,
        default_name="retrieval_trace.json",
    )
    trace = _read_json(trace_path)

    selected_outside_allowed = []
    fifo_window_violations = []
    selected_sizes = []
    unique_selected_sizes = []
    allowed_sizes = []
    raw_candidate_counts = []
    bounded_candidate_counts = []
    fallback_steps = []
    duplicate_context_records = []

    policy = metadata.get("memory_policy")
    budget = metadata.get("memory_budget")
    budget = int(budget) if budget is not None else None

    for record_index, record in enumerate(trace):
        allowed = _as_int_list(record.get("allowed_memory_indices"))
        selected = _as_int_list(record.get("selected_context_indices"))
        allowed_set = set(allowed)

        outside = [idx for idx in selected if idx not in allowed_set]
        if outside:
            selected_outside_allowed.append(
                {
                    "record_index": record_index,
                    "global_step": record.get("global_step"),
                    "outside_indices": outside,
                    "allowed_memory_indices": allowed,
                    "selected_context_indices": selected,
                }
            )

        if policy == "fifo" and budget is not None:
            num_stored_views = int(record.get("num_stored_views", len(allowed)))
            expected = list(range(max(0, num_stored_views - budget), num_stored_views))
            if allowed != expected:
                fifo_window_violations.append(
                    {
                        "record_index": record_index,
                        "global_step": record.get("global_step"),
                        "num_stored_views": num_stored_views,
                        "expected_allowed_memory_indices": expected,
                        "actual_allowed_memory_indices": allowed,
                    }
                )

        selected_sizes.append(len(selected))
        unique_selected_size = len(set(selected))
        unique_selected_sizes.append(unique_selected_size)
        if len(selected) != unique_selected_size:
            duplicate_context_records.append(
                {
                    "record_index": record_index,
                    "global_step": record.get("global_step"),
                    "selected_context_indices": selected,
                    "unique_selected_context_size": unique_selected_size,
                }
            )
        allowed_sizes.append(len(allowed))
        raw_candidate_counts.append(float(record.get("raw_candidate_count", 0)))
        bounded_candidate_counts.append(float(record.get("bounded_candidate_count", 0)))
        if record.get("fallback_used", False):
            fallback_steps.append(record.get("global_step"))

    generated_video = _resolve_path(
        metadata.get("generated_video"),
        run_dir=run_dir,
        default_name="generated.mp4",
    )
    ground_truth_video = _resolve_path(
        metadata.get("ground_truth_video"),
        run_dir=run_dir,
        default_name="ground_truth.mp4",
    )

    summary = {
        "run_dir": str(run_dir),
        "scene_id": metadata.get("scene_id"),
        "num_frames": metadata.get("num_frames"),
        "chunk_size": metadata.get("chunk_size"),
        "fps": metadata.get("fps"),
        "memory_policy": policy,
        "memory_budget": budget,
        "memory_scope": metadata.get("memory_scope"),
        "memory_unit": metadata.get("memory_unit"),
        "generated_video": str(generated_video),
        "generated_video_exists": generated_video.exists(),
        "ground_truth_video": str(ground_truth_video),
        "ground_truth_video_exists": ground_truth_video.exists(),
        "trace_path": str(trace_path),
        "num_trace_records": len(trace),
        "max_allowed_memory_size": max(allowed_sizes) if allowed_sizes else 0,
        "max_selected_context_size": max(selected_sizes) if selected_sizes else 0,
        "max_unique_selected_context_size": (
            max(unique_selected_sizes) if unique_selected_sizes else 0
        ),
        "duplicate_context_record_count": len(duplicate_context_records),
        "mean_raw_candidate_count": _mean(raw_candidate_counts),
        "mean_bounded_candidate_count": _mean(bounded_candidate_counts),
        "fallback_step_count": len(fallback_steps),
        "fallback_steps": fallback_steps,
        "selected_outside_allowed_count": len(selected_outside_allowed),
        "fifo_window_violation_count": len(fifo_window_violations),
        "fifo_verified": (
            len(selected_outside_allowed) == 0
            and len(fifo_window_violations) == 0
        ),
        "first_record": _record_brief(trace[0]) if trace else None,
        "last_record": _record_brief(trace[-1]) if trace else None,
        "duplicate_context_sample": duplicate_context_records[:5],
        "selected_outside_allowed_sample": selected_outside_allowed[:5],
        "fifo_window_violation_sample": fifo_window_violations[:5],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_path",
        type=Path,
        help="Run directory or metadata.json from scripts/run_context_memory_vmem.py",
    )
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit nonzero if selected context frames violate the configured FIFO window.",
    )
    args = parser.parse_args()

    summary = summarize_run(args.run_path)
    print(json.dumps(summary, indent=2))

    if args.fail_on_violation and not summary["fifo_verified"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
