#!/usr/bin/env python
"""Audit how many rows of one or more manifests actually completed.

Candidate dirs are found via output_root/<run_id>_*/ (run_vmem_demo_manifest.py
names each run dir "<run_id>_<trajectory>_A<n>_<policy>[_B<budget>]_
<timestamp>"), but a prefix glob alone is NOT enough to confirm a match:
run_ids across manifests can be prefixes of each other (e.g. unbounded's
"oxford_pan_45_60s" is a strict prefix of fifo64's "oxford_pan_45_60s_fifo64"),
so a naive glob silently double-counts another policy's output as this row's
completion. Each candidate's metadata.json is checked to confirm its
memory_policy (and memory_budget, when the row expects one) actually matches
what this manifest row asked for before counting it complete.

Usage:
  python scripts/audit_manifest_completion.py \
      --output-root outputs/demo_actions_60s \
      manifests/vmem_60s_constrained_15_*.jsonl

Prints one row per manifest (policy/budget parsed from the filename) with
expected vs completed counts, plus incomplete run_ids for follow-up.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


POLICY_BUDGET_RE = re.compile(
    r"vmem_60s_constrained_15_(?P<label>[a-z]+?)(?P<budget>\d*)\.jsonl$"
)
LABEL_TO_POLICY = {
    "unbounded": "unbounded",
    "fifo": "fifo",
    "slam": "slam_covisibility",
    "ri": "rarity_irreplaceability",
    "kcenter": "kcenter_coreset",
    "mce": "mce",
}


def _load_manifest_rows(manifest_path: Path) -> list[dict]:
    rows = []
    with manifest_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


def _metadata_matches_row(metadata: dict, row: dict) -> bool:
    if metadata.get("memory_policy") != row.get("memory_policy", "unbounded"):
        return False
    expected_budget = row.get("memory_budget")
    if expected_budget is not None and metadata.get("memory_budget") != expected_budget:
        return False
    return True


def _is_complete(output_root: Path, row: dict) -> bool:
    run_id = row["run_id"]
    for candidate in output_root.glob(f"{run_id}_*"):
        video_path = candidate / "generated.mp4"
        metadata_path = candidate / "metadata.json"
        if not video_path.exists() or not metadata_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if _metadata_matches_row(metadata, row):
            return True
    return False


def _policy_budget_from_filename(manifest_path: Path) -> tuple[str, str]:
    match = POLICY_BUDGET_RE.search(manifest_path.name)
    if not match:
        return manifest_path.stem, ""
    label = match.group("label")
    budget = match.group("budget")
    policy = LABEL_TO_POLICY.get(label, label)
    return policy, budget or "-"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/demo_actions_60s"))
    parser.add_argument(
        "--show-incomplete",
        action="store_true",
        help="List incomplete run_ids under each manifest row.",
    )
    args = parser.parse_args()

    header = f"{'policy':<26}{'budget':<8}{'completed':<12}{'expected':<10}{'manifest'}"
    print(header)
    print("-" * len(header))

    total_completed = 0
    total_expected = 0
    incomplete_by_manifest: dict[str, list[str]] = {}

    for manifest_path in sorted(args.manifests):
        rows = _load_manifest_rows(manifest_path)
        completed_ids = [row["run_id"] for row in rows if _is_complete(args.output_root, row)]
        incomplete_ids = [row["run_id"] for row in rows if row["run_id"] not in completed_ids]
        policy, budget = _policy_budget_from_filename(manifest_path)

        print(
            f"{policy:<26}{budget:<8}{len(completed_ids):<12}{len(rows):<10}{manifest_path.name}"
        )
        total_completed += len(completed_ids)
        total_expected += len(rows)
        if incomplete_ids:
            incomplete_by_manifest[manifest_path.name] = incomplete_ids

    print("-" * len(header))
    print(f"{'TOTAL':<26}{'':<8}{total_completed:<12}{total_expected:<10}")

    if args.show_incomplete and incomplete_by_manifest:
        print("\nIncomplete run_ids:")
        for name, run_ids in incomplete_by_manifest.items():
            print(f"  {name}:")
            for rid in run_ids:
                print(f"    - {rid}")


if __name__ == "__main__":
    main()
