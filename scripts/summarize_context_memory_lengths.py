#!/usr/bin/env python
"""Summarize Context-as-Memory sequence lengths."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_adapters import ContextMemoryDataset


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    weight = pos - lo
    return float(sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight)


def _describe(values: Iterable[int]) -> dict:
    collected = sorted(float(value) for value in values)
    if not collected:
        return {
            "count": 0,
            "min": 0,
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0,
        }
    return {
        "count": len(collected),
        "min": int(collected[0]),
        "mean": float(sum(collected) / len(collected)),
        "median": _quantile(collected, 0.5),
        "p10": _quantile(collected, 0.1),
        "p90": _quantile(collected, 0.9),
        "p95": _quantile(collected, 0.95),
        "max": int(collected[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        help="Path to Context-as-Memory-Dataset/Context-as-Memory-Dataset",
    )
    parser.add_argument("--scene", action="append", dest="scenes")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument(
        "--fps",
        type=float,
        default=13.0,
        help="FPS used only to convert frame counts to seconds. VMem demo uses 13.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4,
        help="Generation chunk size H, used to report full chunks per sequence.",
    )
    parser.add_argument(
        "--minute-threshold",
        type=float,
        default=60.0,
        help="Seconds threshold for counting minute-long sequences.",
    )
    args = parser.parse_args()

    dataset = ContextMemoryDataset(args.root)
    scene_ids = tuple(args.scenes) if args.scenes else dataset.scene_ids()
    if args.max_scenes is not None:
        scene_ids = scene_ids[: args.max_scenes]

    rows = []
    for sequence in dataset.sequences(scene_ids):
        num_frames = len(sequence.frame_indices)
        num_generated_frames_after_anchor = max(0, num_frames - 1)
        rows.append(
            {
                "scene_id": sequence.scene_id,
                "num_frames": num_frames,
                "duration_seconds": num_frames / args.fps,
                "num_full_chunks_after_anchor": num_generated_frames_after_anchor
                // args.chunk_size,
                "remaining_frames_after_full_chunks": num_generated_frames_after_anchor
                % args.chunk_size,
            }
        )

    minute_frames = math.ceil(args.minute_threshold * args.fps)
    payload = {
        "root": str(args.root),
        "fps": args.fps,
        "chunk_size": args.chunk_size,
        "minute_threshold_seconds": args.minute_threshold,
        "minute_threshold_frames": minute_frames,
        "num_scenes": len(rows),
        "frame_count": _describe(row["num_frames"] for row in rows),
        "duration_seconds": _describe(int(row["duration_seconds"]) for row in rows),
        "full_chunks_after_anchor": _describe(
            row["num_full_chunks_after_anchor"] for row in rows
        ),
        "num_scenes_at_least_threshold": sum(
            1 for row in rows if row["num_frames"] >= minute_frames
        ),
        "shortest_scenes": sorted(rows, key=lambda row: row["num_frames"])[:10],
        "longest_scenes": sorted(rows, key=lambda row: row["num_frames"], reverse=True)[:10],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
