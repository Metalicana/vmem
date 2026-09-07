#!/usr/bin/env python
"""Score all verified returns to the demo runner's identity starting pose.

Reports RGB MSE in [0, 1] and optional full-resolution AlexNet LPIPS, plus
anchor availability/selection from the retrieval trace. This is commanded-pose
self-consistency, not a measurement of reconstructed camera accuracy or GT
video quality. Unlike score_revisit_lpips.py, a non-return is never replaced
with the nearest pose. Existing historical metric outputs are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_rows(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def find_returns(records, *, translation_tolerance=1e-4, rotation_tolerance_deg=0.1):
    returns = []
    away = False
    for record in records:
        pose = np.asarray(record["current_pose"], dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("current_pose must be a finite 4x4 matrix")
        translation = float(np.linalg.norm(pose[:3, 3]))
        cosine = np.clip((np.trace(pose[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        rotation_deg = float(np.degrees(np.arccos(cosine)))
        at_start = translation <= translation_tolerance and rotation_deg <= rotation_tolerance_deg
        if away and at_start and record["num_generated_frames"] > 0:
            returns.append({
                "action_index": int(record["action_index"]),
                "frame_index": int(record["total_frames_after"]) - 1,
                "translation_error": translation,
                "rotation_error_deg": rotation_deg,
            })
        # Count one return per excursion, not each frame of a stationary hold.
        away = not at_start
    return returns


def metadata_matches(metadata, row):
    defaults = {
        "memory_policy": "unbounded", "memory_budget": None,
        "memory_scope": "surfel_indexed_view_memory", "trajectory": "pattern",
        "fps": 13.0, "step_size": 0.1, "frames_per_action": 4,
    }
    for key, default in defaults.items():
        if metadata.get(key, default) != row.get(key, default):
            return False
    for key in ("run_id", "image"):
        if metadata.get(key) != row.get(key):
            return False
    if "seed" in metadata and metadata["seed"] != row.get("seed", 42):
        return False
    if row.get("trajectory", "pattern") == "pattern":
        if metadata.get("pattern") != row.get("pattern", "forward"):
            return False
    num_actions = row.get("num_actions")
    if num_actions is None:
        duration = row.get("duration_seconds")
        frames_per_action = row.get("frames_per_action", 4)
        new_frames = max(1, round(duration * row.get("fps", 13))) if duration else None
        num_actions = max(1, (new_frames + frames_per_action - 1) // frames_per_action) if new_frames else 3
    if metadata.get("num_actions") != num_actions:
        return False
    expected_frames = 1 + num_actions * row.get("frames_per_action", 4)
    if metadata.get("actual_frames") != expected_frames:
        return False
    overrides = metadata.get("config_overrides", {})
    for key in ("inference_steps", "surfel_niter", "surfel_reconstruction_window"):
        if overrides.get(key) != row.get(key):
            return False
    return True


def find_run(output_root, row):
    matches = []
    for path in output_root.glob(f"{row['run_id']}_*"):
        try:
            metadata = json.loads((path / "metadata.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if metadata_matches(metadata, row):
            matches.append((path, metadata))
    return max(matches, key=lambda item: (item[0] / "metadata.json").stat().st_mtime) if matches else None


def retrieval_for_frame(trace, frame_index):
    events = [event for event in trace if frame_index in event["target_frame_indices"]]
    if len(events) > 1:
        raise ValueError(f"Ambiguous retrieval trace for frame {frame_index}")
    if not events:
        return {"retrieval_logged": False}
    event = events[0]
    return {
        "retrieval_logged": True,
        "anchor_retained": 0 in event["allowed_memory_indices"],
        "anchor_selected": 0 in event["selected_context_indices"],
        "eligible_frames": len(event["allowed_memory_indices"]),
        "retained_surfels": event.get("num_retained_surfels"),
        "fallback_used": event.get("fallback_used"),
    }


def read_frame(run_dir, frame_index, source):
    if source == "png":
        from PIL import Image
        with Image.open(run_dir / "generated_frames" / f"{frame_index:04d}.png") as image:
            return np.asarray(image.convert("RGB"))
    import imageio.v2 as imageio
    reader = imageio.get_reader(str(run_dir / "generated.mp4"))
    try:
        return reader.get_data(frame_index)
    finally:
        reader.close()


def image_metrics(anchor, frame, lpips_model=None, device="cpu"):
    if anchor.shape != frame.shape:
        raise ValueError("Anchor and return frame must have the same shape")
    anchor = anchor.astype(np.float32) / 255.0
    frame = frame.astype(np.float32) / 255.0
    mse = float(np.mean((anchor - frame) ** 2))
    metrics = {"rgb_mse": mse}
    if lpips_model is not None:
        import torch
        images = [
            torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
            for array in (anchor, frame)
        ]
        with torch.inference_mode():
            metrics["lpips_alex_fullres"] = float(lpips_model(*images).item())
    return metrics


def summarize_returns(returns):
    summary = {"num_returns": len(returns)}
    for key in ("rgb_mse", "lpips_alex_fullres", "anchor_retained", "anchor_selected"):
        values = [record[key] for record in returns if key in record]
        if values:
            summary[f"mean_{key}"] = float(np.mean(values))
    for key in ("rgb_mse", "lpips_alex_fullres", "frame_index", "time_seconds"):
        if returns and key in returns[-1]:
            summary[f"last_{key}"] = returns[-1][key]
    return summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, default=Path("revisits.csv"))
    parser.add_argument("--translation-tolerance", type=float, default=1e-4)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=0.1)
    parser.add_argument("--lpips", action="store_true", help="Also score LPIPS at saved resolution.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--traces-only", action="store_true", help="Inspect returns/retrieval without decoding video.")
    args = parser.parse_args()
    if not all(np.isfinite(value) and value >= 0 for value in (
        args.translation_tolerance, args.rotation_tolerance_deg,
    )):
        parser.error("Pose tolerances must be finite and nonnegative")
    if args.lpips and args.traces_only:
        parser.error("--lpips cannot be combined with --traces-only")

    lpips_model = None
    if args.lpips:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(args.device).eval()

    summaries, all_returns = [], []
    for manifest in args.manifests:
        for row in load_rows(manifest):
            base = {
                "run_id": row["run_id"], "policy": row.get("memory_policy", "unbounded"),
                "budget": row.get("memory_budget"), "image": row["image"],
                "seed": row.get("seed", 42),
                "translation_tolerance": args.translation_tolerance,
                "rotation_tolerance_deg": args.rotation_tolerance_deg,
            }
            match = find_run(args.output_root, row)
            if match is None:
                summaries.append({**base, "status": "missing"})
                continue
            run_dir, metadata = match
            base.update(run_dir=str(run_dir), seed_verified="seed" in metadata)
            try:
                actions = json.loads((run_dir / "actions.json").read_text())
                if (len(actions) != metadata["num_actions"] or not actions
                        or actions[-1]["total_frames_after"] != metadata["actual_frames"]):
                    raise ValueError("Action log does not match completed run metadata")
                returns = find_returns(
                    actions, translation_tolerance=args.translation_tolerance,
                    rotation_tolerance_deg=args.rotation_tolerance_deg,
                )
                trace_path = run_dir / "retrieval_trace.json"
                trace = json.loads(trace_path.read_text()) if trace_path.exists() else []
                if not returns:
                    summaries.append({**base, "status": "no_verified_return", "num_returns": 0})
                    continue
                # Use the same encoding for the anchor and every return in a run.
                indices = [0] + [record["frame_index"] for record in returns]
                source = "png" if all(
                    (run_dir / "generated_frames" / f"{idx:04d}.png").is_file() for idx in indices
                ) else "mp4"
                base["frame_source"] = "none" if args.traces_only else source
                anchor = None if args.traces_only else read_frame(run_dir, 0, source)
                for record in returns:
                    record["time_seconds"] = record["frame_index"] / metadata["fps"]
                    record.update(retrieval_for_frame(trace, record["frame_index"]))
                    if not args.traces_only:
                        frame = read_frame(run_dir, record["frame_index"], source)
                        record.update(image_metrics(anchor, frame, lpips_model, args.device))
                all_returns.extend({**base, **record} for record in returns)
                summaries.append({**base, "status": "ok", **summarize_returns(returns)})
            except (OSError, ValueError, KeyError, IndexError) as exc:
                summaries.append({**base, "status": "error", "error": str(exc)})

    write_csv(args.csv_out, summaries)
    returns_path = args.csv_out.with_name(args.csv_out.stem + ".returns.csv")
    write_csv(returns_path, all_returns)
    for summary in summaries:
        print(json.dumps(summary))
    print(f"Wrote {args.csv_out} and {returns_path}")
    if any(row["status"] == "error" for row in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
