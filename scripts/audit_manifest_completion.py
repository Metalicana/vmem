#!/usr/bin/env python
"""Audit how many rows of one or more manifests actually completed.

Matches each manifest row's run_id against output_root/<run_id>_*/ dirs
(run_vmem_demo_manifest.py names each run dir "<run_id>_<trajectory>_A<n>_
<policy>[_B<budget>]_<timestamp>", so a prefix match on run_id is enough).
A row counts as complete only if its run dir contains generated.mp4.

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


def _load_manifest_run_ids(manifest_path: Path) -> list[str]:
    run_ids = []
    with manifest_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            run_ids.append(row["run_id"])
    return run_ids


def _is_complete(output_root: Path, run_id: str) -> bool:
    matches = list(output_root.glob(f"{run_id}_*"))
    return any((match / "generated.mp4").exists() for match in matches)


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
        run_ids = _load_manifest_run_ids(manifest_path)
        completed = [rid for rid in run_ids if _is_complete(args.output_root, rid)]
        incomplete = [rid for rid in run_ids if rid not in completed]
        policy, budget = _policy_budget_from_filename(manifest_path)

        print(
            f"{policy:<26}{budget:<8}{len(completed):<12}{len(run_ids):<10}{manifest_path.name}"
        )
        total_completed += len(completed)
        total_expected += len(run_ids)
        if incomplete:
            incomplete_by_manifest[manifest_path.name] = incomplete

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
