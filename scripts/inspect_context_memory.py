#!/usr/bin/env python
"""Inspect Context-as-Memory dataset layout and sequence consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_adapters import ContextMemoryDataset


def _small_tuple(values: tuple[int, ...], limit: int) -> list[int]:
    return list(values[:limit])


def _summary_to_dict(summary: Any, *, sample_limit: int) -> dict:
    return {
        "scene_id": summary.scene_id,
        "ok": summary.ok,
        "num_frames": summary.num_frames,
        "num_camera_entries": summary.num_camera_entries,
        "num_overlap_files": summary.num_overlap_files,
        "num_caption_segments": summary.num_caption_segments,
        "missing_camera_count": len(summary.missing_camera_indices),
        "missing_camera_sample": _small_tuple(summary.missing_camera_indices, sample_limit),
        "missing_overlap_count": len(summary.missing_overlap_indices),
        "missing_overlap_sample": _small_tuple(summary.missing_overlap_indices, sample_limit),
        "overlap_references_missing_frame_count": len(
            summary.overlap_references_missing_frames
        ),
        "overlap_references_missing_frame_sample": [
            list(pair)
            for pair in summary.overlap_references_missing_frames[:sample_limit]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        help="Path to Context-as-Memory-Dataset/Context-as-Memory-Dataset",
    )
    parser.add_argument("--scene", action="append", dest="scenes")
    parser.add_argument("--max-scenes", type=int, default=5)
    parser.add_argument("--sample-limit", type=int, default=10)
    parser.add_argument(
        "--skip-overlap-reference-check",
        action="store_true",
        help="Skip checking whether every overlap reference points to an existing frame.",
    )
    args = parser.parse_args()

    dataset = ContextMemoryDataset(args.root)
    scene_ids = tuple(args.scenes) if args.scenes else dataset.scene_ids()[: args.max_scenes]

    output = {
        "root": str(args.root),
        "num_indexed_scenes": len(dataset.scene_ids()),
        "inspected_scenes": [],
    }

    for sequence in dataset.sequences(scene_ids):
        summary = sequence.validate(
            check_overlap_references=not args.skip_overlap_reference_check
        )
        data = _summary_to_dict(summary, sample_limit=args.sample_limit)

        if sequence.frame_indices:
            first_frame = sequence.frame_indices[0]
            data["first_frame"] = {
                "index": first_frame,
                "path": str(sequence.frame_path(first_frame)),
                "camera": sequence.camera(first_frame).__dict__,
            }
            if sequence.overlap_dir is not None:
                overlap = sequence.overlap_label(first_frame)
                data["first_frame"]["num_overlaps"] = len(overlap.overlapping_frames)
                data["first_frame"]["overlap_sample"] = list(
                    overlap.overlapping_frames[: args.sample_limit]
                )

        output["inspected_scenes"].append(data)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
