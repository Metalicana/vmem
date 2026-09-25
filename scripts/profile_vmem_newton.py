"""Summarize the short matched pilot and gate an explicitly approved long run."""

import argparse
import math
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def action_profile(actions, steps):
    warmup = {row["step"] for row in steps if row.get("warmup")}
    timed = [row for row in actions if isinstance(row, dict) and isinstance(row.get("wall_seconds"), (int, float))
             and math.isfinite(row["wall_seconds"]) and row["wall_seconds"] > 0]
    steady = [row["wall_seconds"] for row in timed if row["action_index"] not in warmup]
    late = [row["wall_seconds"] for row in timed[-3:]]
    return {"timed_actions": len(timed), "warmup_action_indices": sorted(warmup),
            "median_action_seconds_excluding_warmup": statistics.median(steady) if steady else None,
            "last_three_median_action_seconds": statistics.median(late) if late else None,
            "first_action_seconds": timed[0]["wall_seconds"] if timed else None,
            "sum_completed_action_seconds": sum(row["wall_seconds"] for row in timed),
            "constant_late_rate_195_actions_hours": statistics.median(late) * 195 / 3600 if late else None,
            "timing_scope": "Action wall times exclude load/export, frame-save/checkpoint I/O and lost/replayed work. The 195-action constant-rate projection ignores unbounded growth and is NOT an ETA or fit guarantee."}


def verify_smoke(output):
    from run_vmem_results import read, digest
    from run_vmem_demo_actions import _generation_provenance
    from run_vmem_demo_manifest import _load_manifest
    from audit_vmem_runs import inspect_attempt
    from audit_vmem_pairing import compare_runs
    from plot_vmem_resources import read_steps
    from report_vmem_results import prefix_passed, resource_summary
    from vmem_protocol import expected_settings, verify_lock
    from vmem_slurm import environment_identity

    if read(output / "workflow_status.json")["status"] != "complete":
        raise ValueError("Smoke workflow is not complete")
    spec = read(output / "workflow_spec.json")
    if spec.get("vmem_environment") != environment_identity():
        raise ValueError("VMem environment differs from the smoke; review before a long launch")
    rows = [{k: v for k, v in row.items() if k != "_manifest_line"}
            for row in _load_manifest(REPO / "manifests/vmem_newton_smoke_v1.jsonl")]
    if spec["selected_rows"] != rows or spec["dimensions"] != []:
        raise ValueError("Expected the predefined generation-only Newton smoke pair")
    lock_path = Path(spec["lock"])
    lock = read(lock_path)
    if digest(lock_path) != spec["lock_sha256"]:
        raise ValueError("Smoke lock changed")
    for row in rows:
        verify_lock(lock_path, expected_settings(row), _generation_provenance(Path(row["image"]), Path(lock["config"])))
    long_rows = _load_manifest(REPO / "manifests/vmem_transfer_v3.jsonl")[:2]
    for smoke, long in zip(rows, long_rows):
        ignore = {"run_id", "num_actions"}
        if ({k: v for k, v in expected_settings(smoke).items() if k not in ignore}
                != {k: v for k, v in expected_settings(long).items() if k not in ignore}):
            raise ValueError("Long Oxford settings differ from the tested smoke beyond IDs/duration")
    inventory = read(output / "inventory.json")
    if len(inventory["pairs"]) != 1 or not inventory["pairs"][0]["validated_pair"]:
        raise ValueError("Smoke lacks a validated pair")
    arms = {arm["policy"]: arm for arm in inventory["pairs"][0]["arms"]}
    checked, profiles = [], []
    for row in rows:
        arm = arms[row["memory_policy"]]
        path = Path(arm["run_dir"])
        item = inspect_attempt(path, row, lock_path, lock)
        if item["status"] != "validated" or item["video"]["frames"] != 73:
            raise ValueError(f"Smoke artifact validation failed: {item}")
        checked.append(item)
        metadata = read(path / "metadata.json")
        expected = 32 if row["memory_policy"] == "slam_covisibility" else 73
        payload = metadata["frame_payloads"]
        for component in ("pil_frames", "latents", "encoder_embeddings", "Ks", "surfel_depths"):
            if payload["resident_counts"].get(component) != expected:
                raise ValueError(f"Wrong smoke residency for {component}: {path}")
        if payload["pending_evictions"]:
            raise ValueError("Smoke has pending evictions")
        steps, truncated = read_steps(path / "resource_trace.jsonl")
        if truncated or len(steps) != 18:
            raise ValueError("Incomplete smoke resource trace")
        profile = {**resource_summary(arm, metadata, steps), **action_profile(read(path / "actions.json"), steps)}
        if profile["timed_actions"] != 18 or profile["max_recorded_cuda_peak_gib"] is None:
            raise ValueError("Smoke lacks required timing/peak measurements")
        profiles.append(profile)
    if checked[0]["checkpoints"] != checked[1]["checkpoints"] or checked[0]["environment"] != checked[1]["environment"]:
        raise ValueError("Smoke checkpoint/runtime mismatch")
    pairing = compare_runs(Path(arms["unbounded"]["run_dir"]), Path(arms["slam_covisibility"]["run_dir"]), compare_pixels=True)
    if not prefix_passed(pairing) or pairing.get("first_eviction") is None:
        raise ValueError("Smoke pre-eviction pairing or budget-crossing gate failed")
    return {"schema": "vmem_newton_profile_v1", "status": "validated", "lock_sha256": digest(lock_path),
            "profiles": profiles, "source_sha256": lock["source_sha256"], "config_sha256": lock["config_sha256"],
            "claim": "18-action H100 smoke, not evidence that 195 actions fit 80GB or four hours."}


def main():
    from run_vmem_results import save
    import json
    import os
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True, help="Completed smoke workflow directory")
    cli.add_argument("--approve-long", action="store_true", help="Explicit user approval after reviewing short-run costs")
    args = cli.parse_args()
    os.chdir(REPO)
    result = verify_smoke(args.output.resolve())
    save(args.output / "newton_profile.json", result)
    print(json.dumps(result, indent=2))
    if args.approve_long:
        print("Short smoke revalidated. Long run approved explicitly, not certified to fit.")


if __name__ == "__main__":
    main()
