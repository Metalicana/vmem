#!/usr/bin/env python
"""Analyze online overlap structure in Context-as-Memory sequences.

This script treats overlap labels as supervision for relevant memory. Because the
labels can point to future frames, all online memory metrics use only overlaps
with frame index < current frame index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_adapters import ContextMemoryDataset, ContextMemorySequence


def _parse_budgets(value: str) -> tuple[int, ...]:
    budgets = tuple(int(part) for part in value.split(",") if part.strip())
    if not budgets or any(budget <= 0 for budget in budgets):
        raise argparse.ArgumentTypeError("budgets must be positive integers, e.g. 1,2,4,8")
    return budgets


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


def _describe(values: Iterable[int | float]) -> dict:
    collected = [float(value) for value in values]
    if not collected:
        return {
            "count": 0,
            "min": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }

    sorted_values = sorted(collected)
    return {
        "count": len(sorted_values),
        "min": float(sorted_values[0]),
        "mean": float(sum(sorted_values) / len(sorted_values)),
        "median": _quantile(sorted_values, 0.5),
        "p90": _quantile(sorted_values, 0.9),
        "p95": _quantile(sorted_values, 0.95),
        "p99": _quantile(sorted_values, 0.99),
        "max": float(sorted_values[-1]),
    }


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _top_counts(counts: dict[int, int], *, limit: int) -> list[dict]:
    return [
        {"frame_index": frame_index, "count": count}
        for frame_index, count in sorted(
            counts.items(),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )[:limit]
    ]


def analyze_sequence(
    sequence: ContextMemorySequence,
    *,
    budgets: Sequence[int],
    sample_limit: int,
) -> dict:
    frame_indices = sequence.frame_indices
    frame_index_set = set(frame_indices)
    frame_position = {frame_index: pos for pos, frame_index in enumerate(frame_indices)}

    all_counts: list[int] = []
    past_counts: list[int] = []
    future_counts: list[int] = []
    same_frame_counts: list[int] = []
    past_overlap_lags: list[int] = []
    nearest_past_overlap_lags: list[int] = []
    past_count_by_frame: dict[int, int] = {}
    future_count_by_frame: dict[int, int] = {}
    first_frame_with_past_overlap: int | None = None

    budget_values: dict[int, dict[str, list[float]]] = {
        budget: {
            "recency_recall": [],
            "recency_precision": [],
            "recency_any_hit": [],
            "oracle_recall": [],
        }
        for budget in budgets
    }

    for frame_index in frame_indices:
        label = sequence.overlap_label(frame_index)
        overlaps = set(label.overlapping_frames)
        missing = overlaps.difference(frame_index_set)
        if missing:
            raise ValueError(
                f"{sequence.scene_id}: frame {frame_index} overlap references missing frames: "
                f"{sorted(missing)[:sample_limit]}"
            )

        past_overlaps = {idx for idx in overlaps if idx < frame_index}
        future_overlaps = {idx for idx in overlaps if idx > frame_index}
        same_frame_overlaps = {idx for idx in overlaps if idx == frame_index}

        all_counts.append(len(overlaps))
        past_counts.append(len(past_overlaps))
        future_counts.append(len(future_overlaps))
        same_frame_counts.append(len(same_frame_overlaps))
        past_count_by_frame[frame_index] = len(past_overlaps)
        future_count_by_frame[frame_index] = len(future_overlaps)

        if past_overlaps:
            lags = [frame_index - overlap_index for overlap_index in past_overlaps]
            past_overlap_lags.extend(lags)
            nearest_past_overlap_lags.append(min(lags))
            if first_frame_with_past_overlap is None:
                first_frame_with_past_overlap = frame_index

        if not past_overlaps:
            continue

        pos = frame_position[frame_index]
        for budget in budgets:
            memory = set(frame_indices[max(0, pos - budget) : pos])
            hits = len(memory.intersection(past_overlaps))
            memory_size = len(memory)

            budget_values[budget]["recency_recall"].append(hits / len(past_overlaps))
            budget_values[budget]["recency_precision"].append(
                hits / memory_size if memory_size else 0.0
            )
            budget_values[budget]["recency_any_hit"].append(1.0 if hits else 0.0)
            budget_values[budget]["oracle_recall"].append(
                min(budget, len(past_overlaps)) / len(past_overlaps)
            )

    frames_with_any_overlap = sum(1 for count in all_counts if count > 0)
    frames_with_past_overlap = sum(1 for count in past_counts if count > 0)
    frames_with_future_overlap = sum(1 for count in future_counts if count > 0)

    return {
        "scene_id": sequence.scene_id,
        "num_frames": len(frame_indices),
        "frames_with_any_overlap": frames_with_any_overlap,
        "frames_with_past_overlap": frames_with_past_overlap,
        "frames_with_future_overlap": frames_with_future_overlap,
        "first_frame_with_past_overlap": first_frame_with_past_overlap,
        "all_overlap_count": _describe(all_counts),
        "past_overlap_count": _describe(past_counts),
        "future_overlap_count": _describe(future_counts),
        "same_frame_overlap_count": _describe(same_frame_counts),
        "past_overlap_lag": _describe(past_overlap_lags),
        "nearest_past_overlap_lag": _describe(nearest_past_overlap_lags),
        "top_past_overlap_frames": _top_counts(
            past_count_by_frame,
            limit=sample_limit,
        ),
        "top_future_overlap_frames": _top_counts(
            future_count_by_frame,
            limit=sample_limit,
        ),
        "budget_baselines": [
            {
                "budget": budget,
                "eligible_frames": len(values["recency_recall"]),
                "recency_mean_recall": _mean(values["recency_recall"]),
                "recency_mean_precision": _mean(values["recency_precision"]),
                "recency_any_hit_rate": _mean(values["recency_any_hit"]),
                "oracle_mean_recall": _mean(values["oracle_recall"]),
            }
            for budget, values in budget_values.items()
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
    parser.add_argument("--max-scenes", type=int, default=1)
    parser.add_argument("--sample-limit", type=int, default=10)
    parser.add_argument(
        "--budgets",
        type=_parse_budgets,
        default=_parse_budgets("1,2,4,8,16,32,64,128"),
        help="Comma-separated memory budgets measured in number of past frames.",
    )
    args = parser.parse_args()

    dataset = ContextMemoryDataset(args.root)
    scene_ids = tuple(args.scenes) if args.scenes else dataset.scene_ids()[: args.max_scenes]

    output = {
        "root": str(args.root),
        "assumption": "online metrics use only overlap labels with overlap_frame < current_frame",
        "budgets": list(args.budgets),
        "scenes": [
            analyze_sequence(
                sequence,
                budgets=args.budgets,
                sample_limit=args.sample_limit,
            )
            for sequence in dataset.sequences(scene_ids)
        ],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
