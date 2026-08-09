#!/usr/bin/env python
"""Self-consistency paired LPIPS/SSIM for VMem's constrained-revisit suite.

None of the demo-actions trajectories (pan_45, pan_90, local_loop) have a
ground-truth video -- they're scripted paths from a single conditioning
image, so there's no real reference frame to pair against at most poses.
What IS a legitimate paired comparison: every trajectory in this suite is
built to return near the starting pose periodically (pan_* nets to yaw 0
every trajectory-period actions; local_loop nets to ~identity pose every 16
actions), and the starting pose's real frame is the input conditioning
image itself. So for each run, we find the action whose logged current_pose
was closest to the identity start pose, take the frame generated at that
action, and score it against frame 0 (the real anchor) with LPIPS/SSIM.

This measures "does memory let the model recognize it's back home," not
absolute fidelity against unseen ground truth -- report it as that.

Requires: pip install lpips  (not currently in requirements.txt)

Usage:
  python scripts/score_revisit_lpips.py \
      --output-root outputs/demo_actions_60s \
      manifests/vmem_60s_constrained_15_*.jsonl \
      --csv-out revisit_lpips_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np


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
MIN_ACTION_INDEX_FOR_RETURN = 10  # skip the trivial near-zero distance at the very start


def _policy_budget_from_filename(manifest_path: Path) -> tuple[str, str]:
    match = POLICY_BUDGET_RE.search(manifest_path.name)
    if not match:
        return manifest_path.stem, ""
    label = match.group("label")
    budget = match.group("budget")
    return LABEL_TO_POLICY.get(label, label), (budget or "-")


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


def _find_run_dir(output_root: Path, row: dict) -> Path | None:
    # A prefix glob on run_id alone is not enough: run_ids across manifests
    # can be prefixes of each other (unbounded's "oxford_pan_45_60s" is a
    # strict prefix of fifo64's "oxford_pan_45_60s_fifo64"), so each
    # candidate's metadata.json must confirm it's actually this row's policy
    # /budget, not just a differently-suffixed sibling run.
    run_id = row["run_id"]
    matches = []
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
            matches.append(candidate)
    if not matches:
        return None
    # Prefer the most recently completed if a row somehow matches twice
    # (e.g. a crashed-then-restarted row).
    return max(matches, key=lambda p: p.stat().st_mtime)


def _pose_distance_to_identity(pose: list) -> float:
    pose = np.asarray(pose, dtype=np.float64)
    translation = pose[:3, 3]
    rotation = pose[:3, :3]
    translation_dist = float(np.linalg.norm(translation))
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    rotation_dist = float(np.arccos(cosine))
    return translation_dist + rotation_dist


def _find_return_frame_index(actions_path: Path) -> int | None:
    with actions_path.open() as handle:
        records = json.load(handle)
    candidates = [r for r in records if r["action_index"] >= MIN_ACTION_INDEX_FOR_RETURN]
    if not candidates:
        return None
    best = min(candidates, key=lambda r: _pose_distance_to_identity(r["current_pose"]))
    return int(best["total_frames_after"]) - 1


def _read_video_frame(video_path: Path, frame_index: int):
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(video_path))
    try:
        return reader.get_data(frame_index)
    finally:
        reader.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/demo_actions_60s"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args()

    try:
        import lpips
    except ImportError:
        print(
            "The `lpips` package is required (pip install lpips) -- not currently in "
            "requirements.txt. Aborting before scoring anything.",
            file=sys.stderr,
        )
        raise

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from modeling.metrics import calculate_score

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lpips_loss_fn = lpips.LPIPS(net="alex").to(device)

    rows_out = []
    for manifest_path in sorted(args.manifests):
        policy, budget = _policy_budget_from_filename(manifest_path)
        for row in _load_manifest_rows(manifest_path):
            run_id = row["run_id"]
            run_dir = _find_run_dir(args.output_root, row)
            if run_dir is None:
                rows_out.append(
                    {"run_id": run_id, "policy": policy, "budget": budget, "status": "missing"}
                )
                continue

            actions_path = run_dir / "actions.json"
            video_path = run_dir / "generated.mp4"
            if not actions_path.exists():
                rows_out.append(
                    {"run_id": run_id, "policy": policy, "budget": budget, "status": "no_actions_json"}
                )
                continue

            return_frame_index = _find_return_frame_index(actions_path)
            if return_frame_index is None:
                rows_out.append(
                    {"run_id": run_id, "policy": policy, "budget": budget, "status": "no_return_frame"}
                )
                continue

            try:
                anchor_frame = _read_video_frame(video_path, 0)
                return_frame = _read_video_frame(video_path, return_frame_index)
            except Exception as exc:  # noqa: BLE001 - report and continue the sweep
                rows_out.append(
                    {
                        "run_id": run_id,
                        "policy": policy,
                        "budget": budget,
                        "status": f"read_error: {exc}",
                    }
                )
                continue

            from PIL import Image

            anchor_pil = Image.fromarray(anchor_frame)
            return_pil = Image.fromarray(return_frame)
            psnr, lpips_value, ssim_value = calculate_score(
                anchor_pil, return_pil, lpips_loss_fn, device=device
            )
            rows_out.append(
                {
                    "run_id": run_id,
                    "policy": policy,
                    "budget": budget,
                    "status": "ok",
                    "return_frame_index": return_frame_index,
                    "revisit_lpips_alex": lpips_value,
                    "revisit_ssim": ssim_value,
                    "revisit_psnr": psnr,
                }
            )

    # Print a per-(policy, budget) summary.
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for row in rows_out:
        by_cell.setdefault((row["policy"], row["budget"]), []).append(row)

    header = f"{'policy':<26}{'budget':<8}{'scored':<10}{'mean_lpips':<14}{'mean_ssim':<12}"
    print(header)
    print("-" * len(header))
    for (policy, budget), cell_rows in sorted(by_cell.items()):
        ok_rows = [r for r in cell_rows if r["status"] == "ok"]
        mean_lpips = np.mean([r["revisit_lpips_alex"] for r in ok_rows]) if ok_rows else float("nan")
        mean_ssim = np.mean([r["revisit_ssim"] for r in ok_rows]) if ok_rows else float("nan")
        print(f"{policy:<26}{budget:<8}{len(ok_rows):<10}{mean_lpips:<14.4f}{mean_ssim:<12.4f}")

    if args.csv_out:
        fieldnames = sorted({key for row in rows_out for key in row.keys()})
        with args.csv_out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"\nWrote per-run detail to {args.csv_out}")


if __name__ == "__main__":
    main()
