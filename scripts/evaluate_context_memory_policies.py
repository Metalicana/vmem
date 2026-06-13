#!/usr/bin/env python
"""Evaluate fixed-budget memory policies on Context-as-Memory overlap labels.

This is a non-generation simulator. It scores whether each online memory policy
keeps frames that the dataset labels as overlapping with the current frame.

Budget unit: number of remembered frame indices.
Relevance: overlap labels restricted to overlap_frame < current_frame.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import sys
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_adapters import ContextMemoryDataset, ContextMemorySequence


PolicyResult = dict[str, float | int | str]


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


def _summarize_metric(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
        }
    sorted_values = sorted(values)
    return {
        "mean": float(sum(sorted_values) / len(sorted_values)),
        "median": _quantile(sorted_values, 0.5),
        "p10": _quantile(sorted_values, 0.1),
        "p90": _quantile(sorted_values, 0.9),
    }


def _uniform_temporal_memory(past_frames: Sequence[int], budget: int) -> set[int]:
    if len(past_frames) <= budget:
        return set(past_frames)
    if budget == 1:
        return {past_frames[-1]}

    max_pos = len(past_frames) - 1
    positions = {
        round(i * max_pos / (budget - 1))
        for i in range(budget)
    }
    return {past_frames[pos] for pos in positions}


def _hybrid_recent_uniform_memory(past_frames: Sequence[int], budget: int) -> set[int]:
    if len(past_frames) <= budget:
        return set(past_frames)

    recent_budget = max(1, budget // 2)
    uniform_budget = budget - recent_budget
    recent = set(past_frames[-recent_budget:])

    older_frames = past_frames[: max(0, len(past_frames) - recent_budget)]
    uniform = _uniform_temporal_memory(older_frames, uniform_budget) if uniform_budget else set()
    memory = recent | uniform

    if len(memory) > budget:
        # Prefer recency if a uniform pick collides near the recent boundary.
        extras = len(memory) - budget
        removable = sorted(memory.difference(recent))
        for frame_index in removable[:extras]:
            memory.remove(frame_index)
    return memory


def _summarize_records(
    *,
    scene_id: str,
    policy: str,
    budget: int,
    records: Sequence[dict[str, float]],
) -> PolicyResult:
    recalls = [record["recall"] for record in records]
    precisions = [record["precision"] for record in records]
    any_hits = [record["any_hit"] for record in records]
    memory_sizes = [record["memory_size"] for record in records]

    recall = _summarize_metric(recalls)
    precision = _summarize_metric(precisions)

    return {
        "scene_id": scene_id,
        "policy": policy,
        "budget": budget,
        "eligible_frames": len(records),
        "mean_recall": recall["mean"],
        "median_recall": recall["median"],
        "p10_recall": recall["p10"],
        "p90_recall": recall["p90"],
        "mean_precision": precision["mean"],
        "median_precision": precision["median"],
        "any_hit_rate": float(sum(any_hits) / len(any_hits)) if any_hits else 0.0,
        "mean_memory_size": float(sum(memory_sizes) / len(memory_sizes)) if memory_sizes else 0.0,
    }


def _flatten_scene_results(scene_results: Sequence[dict]) -> list[PolicyResult]:
    rows: list[PolicyResult] = []
    for scene in scene_results:
        rows.extend(scene["policy_results"])
    return rows


def _weighted_mean(rows: Sequence[PolicyResult], key: str) -> float:
    total_weight = sum(float(row["eligible_frames"]) for row in rows)
    if total_weight == 0:
        return 0.0
    return float(
        sum(float(row[key]) * float(row["eligible_frames"]) for row in rows)
        / total_weight
    )


def _aggregate_policy_results(rows: Sequence[PolicyResult]) -> list[PolicyResult]:
    grouped: dict[tuple[str, int], list[PolicyResult]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["policy"]), int(row["budget"]))].append(row)

    summary_rows: list[PolicyResult] = []
    for (policy, budget), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        summary_rows.append(
            {
                "scene_id": "ALL",
                "policy": policy,
                "budget": budget,
                "num_scenes": len(group),
                "eligible_frames": int(sum(int(row["eligible_frames"]) for row in group)),
                "mean_recall": _weighted_mean(group, "mean_recall"),
                "macro_mean_recall": float(
                    sum(float(row["mean_recall"]) for row in group) / len(group)
                ),
                "median_recall": _weighted_mean(group, "median_recall"),
                "p10_recall": _weighted_mean(group, "p10_recall"),
                "p90_recall": _weighted_mean(group, "p90_recall"),
                "mean_precision": _weighted_mean(group, "mean_precision"),
                "median_precision": _weighted_mean(group, "median_precision"),
                "any_hit_rate": _weighted_mean(group, "any_hit_rate"),
                "mean_memory_size": _weighted_mean(group, "mean_memory_size"),
            }
        )
    return summary_rows


def _write_csv(rows: Sequence[PolicyResult]) -> None:
    fieldnames = [
        "scene_id",
        "policy",
        "budget",
        "num_scenes",
        "eligible_frames",
        "mean_recall",
        "macro_mean_recall",
        "median_recall",
        "p10_recall",
        "p90_recall",
        "mean_precision",
        "median_precision",
        "any_hit_rate",
        "mean_memory_size",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def _score_memory(memory: set[int], relevant: set[int]) -> dict[str, float]:
    hits = len(memory.intersection(relevant))
    return {
        "recall": hits / len(relevant),
        "precision": hits / len(memory) if memory else 0.0,
        "any_hit": 1.0 if hits else 0.0,
        "memory_size": float(len(memory)),
    }


def evaluate_sequence(
    sequence: ContextMemorySequence,
    *,
    budgets: Sequence[int],
    seed: int,
    include_oracle: bool,
) -> list[PolicyResult]:
    frame_indices = sequence.frame_indices
    rng_by_budget = {budget: random.Random(seed + budget) for budget in budgets}
    reservoir_memory: dict[int, list[int]] = {budget: [] for budget in budgets}
    seen_count = 0

    records: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)

    for position, frame_index in enumerate(frame_indices):
        relevant = set(sequence.overlap_label(frame_index, past_only=True).overlapping_frames)

        if relevant:
            past_frames = frame_indices[:position]
            for budget in budgets:
                memories = {
                    "recency": set(past_frames[-budget:]),
                    "uniform_temporal": _uniform_temporal_memory(past_frames, budget),
                    "hybrid_recent_uniform": _hybrid_recent_uniform_memory(past_frames, budget),
                    "reservoir_random": set(reservoir_memory[budget]),
                }
                if include_oracle:
                    memories["oracle_upper_bound"] = set(
                        sorted(relevant, reverse=True)[:budget]
                    )

                for policy, memory in memories.items():
                    records[(policy, budget)].append(_score_memory(memory, relevant))

        # Update online reservoir memories after scoring the current frame.
        seen_count += 1
        for budget in budgets:
            memory = reservoir_memory[budget]
            rng = rng_by_budget[budget]
            if len(memory) < budget:
                memory.append(frame_index)
            else:
                replace_pos = rng.randrange(seen_count)
                if replace_pos < budget:
                    memory[replace_pos] = frame_index

    results: list[PolicyResult] = []
    policy_order = [
        "recency",
        "uniform_temporal",
        "hybrid_recent_uniform",
        "reservoir_random",
    ]
    if include_oracle:
        policy_order.append("oracle_upper_bound")

    for budget in budgets:
        for policy in policy_order:
            results.append(
                _summarize_records(
                    scene_id=sequence.scene_id,
                    policy=policy,
                    budget=budget,
                    records=records[(policy, budget)],
                )
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        help="Path to Context-as-Memory-Dataset/Context-as-Memory-Dataset",
    )
    parser.add_argument("--scene", action="append", dest="scenes")
    parser.add_argument("--max-scenes", type=int, default=1)
    parser.add_argument(
        "--budgets",
        type=_parse_budgets,
        default=_parse_budgets("1,2,4,8,16,32,64,128"),
        help="Comma-separated memory budgets measured in number of past frames.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-oracle",
        action="store_true",
        help="Do not include the non-deployable oracle upper-bound policy.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Aggregate rows by policy and budget across scenes.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="Output format.",
    )
    args = parser.parse_args()

    dataset = ContextMemoryDataset(args.root)
    scene_ids = tuple(args.scenes) if args.scenes else dataset.scene_ids()[: args.max_scenes]

    scene_results = []
    for sequence in dataset.sequences(scene_ids):
        scene_results.append(
            {
                "scene_id": sequence.scene_id,
                "policy_results": evaluate_sequence(
                    sequence,
                    budgets=args.budgets,
                    seed=args.seed,
                    include_oracle=not args.no_oracle,
                ),
            }
        )

    flat_rows = _flatten_scene_results(scene_results)
    rows = _aggregate_policy_results(flat_rows) if args.summary_only else flat_rows

    if args.format == "csv":
        _write_csv(rows)
        return

    payload = {
        "root": str(args.root),
        "assumptions": [
            "budget is measured in remembered frame indices",
            "relevance uses only overlap labels with overlap_frame < current_frame",
            "oracle_upper_bound is non-deployable and uses current-frame labels",
            "reservoir_random is online and updated only after scoring the current frame",
            "summary rows use eligible-frame-weighted averages plus macro_mean_recall",
        ],
        "budgets": list(args.budgets),
        "seed": args.seed,
    }
    if args.summary_only:
        payload["summary_policy_results"] = rows
    else:
        payload["scenes"] = scene_results
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
