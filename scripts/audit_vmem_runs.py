#!/usr/bin/env python
"""Freeze generation inputs or validate every attempt in a matched run inventory."""

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

from run_vmem_demo_actions import _expand_trajectory_actions, _generation_provenance
from run_vmem_demo_manifest import _load_manifest
from vmem_protocol import commanded_path, expected_settings, verify_lock


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def freeze(args):
    rows = [{key: value for key, value in row.items() if key != "_manifest_line"}
            for row in _load_manifest(args.manifest)]
    if len(rows) % 2 or len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("Expected adjacent unique unbounded/GeoCov pairs")
    paths = {}
    for i in range(0, len(rows), 2):
        left, right = [expected_settings(row) for row in rows[i:i + 2]]
        ignored = {"run_id", "memory_policy", "memory_budget"}
        if ({k: v for k, v in left.items() if k not in ignored}
                != {k: v for k, v in right.items() if k not in ignored}):
            raise ValueError(f"Mismatched pair at rows {i}, {i + 1}")
        if left["memory_policy"] != "unbounded" or right["memory_policy"] != "slam_covisibility" or right["memory_budget"] != 32:
            raise ValueError("This transfer protocol requires unbounded / GeoCov-32 pairs")
        if left["memory_budget"] is not None or left["surfel_reconstruction_window"] is not None:
            raise ValueError("Use no unbounded budget and no reconstruction-window override")
        if left["frames_per_action"] != 4 or left["memory_scope"] != "surfel_indexed_view_memory":
            raise ValueError("This protocol requires four-frame actions and indexed-view memory")
        if rows[i].get("_case_id") != rows[i + 1].get("_case_id"):
            raise ValueError("Paired case identifiers differ")
        actions = _expand_trajectory_actions(SimpleNamespace(**left))
        path = commanded_path(actions, left["step_size"], left["frames_per_action"], left["fps"])
        returns = [record["frame_index"] for record in path if record["at_origin"]]
        if not returns:
            raise ValueError(f"No commanded return in {left['run_id']}")
        if left["trajectory"] == "fixed_region_v1" and path[-1]["max_radius"] > 8 * left["step_size"] + 1e-6:
            raise ValueError("Fixed-region path exceeded its specified radius")
        if left["trajectory"] == "expanding_excursions_v1" and path[-1]["max_radius"] <= 8 * left["step_size"]:
            raise ValueError("Expanding path did not exceed the first excursion")
        paths[rows[i].get("_case_id", left["run_id"])] = {
            "actions": actions, "commanded_path": path, "return_frames": returns,
            "max_commanded_radius": path[-1]["max_radius"],
            "units": "Navigator units; not reconstructed coverage or meters",
        }
    first = _generation_provenance(Path(rows[0]["image"]), args.config)
    lock = {"schema": "vmem_experiment_lock_v3", "manifest": str(args.manifest),
            "manifest_sha256": file_hash(args.manifest), "config": str(args.config),
            "config_sha256": first["config_sha256"], "source_sha256": first["source_sha256"],
            "git_commit": first["git_commit"], "rows": rows,
            "image_sha256": {row["image"]: file_hash(row["image"]) for row in rows},
            "path_validation": paths}
    write_json(args.output, lock, exclusive=True)
    print(f"Frozen {len(rows)} runs in {args.output}; no models were loaded.")
    for case, details in paths.items():
        print(f"{case}: returns={len(details['return_frames'])}, radius={details['max_commanded_radius']:.4f}")


def probe_video(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
         "stream=codec_type,width,height,avg_frame_rate,nb_read_frames:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    if result.stderr.strip():
        raise ValueError("ffprobe reported decoding errors: " + result.stderr.strip())
    data = json.loads(result.stdout)
    stream = next(item for item in data["streams"] if item["codec_type"] == "video")
    return {"frames": int(stream["nb_read_frames"]), "width": int(stream["width"]),
            "height": int(stream["height"]), "fps": float(Fraction(stream["avg_frame_rate"])),
            "seconds": float(data["format"]["duration"])}


def validate_actions(actions, settings):
    expected = _expand_trajectory_actions(SimpleNamespace(**settings))
    planned = commanded_path(expected, settings["step_size"], settings["frames_per_action"], settings["fps"])
    if len(actions) != len(expected):
        raise ValueError("Action count differs from frozen protocol")
    for index, (actual, plan) in enumerate(zip(actions, planned)):
        if actual["action"] != plan["action"] or actual["action_index"] != index:
            raise ValueError("Action sequence differs from frozen protocol")
        if actual["total_frames_after"] != plan["frame_index"] + 1:
            raise ValueError("Action frame count differs from frozen protocol")
        if actual["num_generated_frames"] != settings["frames_per_action"]:
            raise ValueError("Action did not generate the requested number of frames")
        if len(actual["current_pose"]) != 4 or any(len(row) != 4 for row in actual["current_pose"]):
            raise ValueError("Malformed logged camera pose")
        for row, reference in zip(actual["current_pose"], plan["current_pose"]):
            if any(not math.isfinite(a) or abs(a - b) > 1e-5 for a, b in zip(row, reference)):
                raise ValueError("Logged camera pose differs from commanded path")


def read_resource_steps(path):
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not events or events[0].get("schema") not in {"vmem_resources_v1", "vmem_resources_v2"}:
        raise ValueError("Missing or unknown resource trace schema")
    return [event for event in events if event["event"] == "step"]


def validate_resources(steps, settings):
    if len(steps) != settings["num_actions"]:
        raise ValueError("Resource trace does not cover every generation step")
    for index, step in enumerate(steps):
        if step["step"] != index or step["frame_count"] != 1 + (index + 1) * settings["frames_per_action"]:
            raise ValueError("Resource step/frame indices differ from protocol")
        session_start = step.get("session_start_step", 0)
        if not isinstance(session_start, int) or not 0 <= session_start <= index:
            raise ValueError("Invalid profiling session boundary")
        if step["warmup"] != (index < session_start + settings["profile_warmup_steps"]):
            raise ValueError("Inconsistent warm-up classification")
        if step["policy"] != settings["memory_policy"] or step["budget"] != settings["memory_budget"] or step["scope"] != settings["memory_scope"]:
            raise ValueError("Resource trace policy, budget or scope mismatch")
        for phase in ("generation", "retrieval", "reconstruction", "memory_update"):
            seconds = step["phase_seconds"][phase]
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError("Invalid stage timing")
        if not math.isfinite(step["accounting_seconds"]) or step["accounting_seconds"] < 0:
            raise ValueError("Invalid accounting overhead")
        if abs(step["video_seconds"] - step["frame_count"] / settings["fps"]) > 1e-6:
            raise ValueError("Resource duration mismatch")
        for snapshot in (step["before_update"], step["after_update"]):
            validate_snapshot(snapshot)
        memory = step["after_update"]
        eligible = memory["eligible_frame_indices"]
        if len(set(eligible)) != len(eligible) or memory["eligible_frames"] != len(eligible):
            raise ValueError("Invalid eligible bank accounting")
        if not all(0 <= idx < step["frame_count"] for idx in eligible):
            raise ValueError("Invalid eligible frame index")
        if settings["memory_policy"] == "unbounded":
            if set(eligible) != set(range(step["frame_count"])):
                raise ValueError("Unbounded archive unexpectedly lost eligible frames")
        elif len(eligible) > settings["memory_budget"]:
            raise ValueError("Eligible bank exceeded B, including protected frames")
        if not memory["protected_frames_eligible"] or any(idx not in eligible for idx in memory["protected_frame_indices"]):
            raise ValueError("Protected endpoint or anchor missing")
        expected_protected = {step["frame_count"] - 1}
        if settings["memory_policy"] == "slam_covisibility":
            expected_protected.add(0)
        if set(memory["protected_frame_indices"]) != expected_protected:
            raise ValueError("Protected-frame accounting differs from policy")
        if settings["memory_scope"] == "surfel_indexed_view_memory" and not memory["surfel_references_within_eligible"]:
            raise ValueError("Surfel references leaked outside the eligible bank")
        if settings["memory_policy"] == "slam_covisibility":
            if sum(memory["appearance_descriptor_sources"].values()) <= 0:
                raise ValueError("GeoCov appearance descriptor was not logged")
        mode = settings.get("frame_storage", "legacy")
        if memory.get("frame_payloads", {}).get("mode", "legacy") != mode:
            raise ValueError("Frame storage mode differs from the frozen protocol")
        if mode == "resident":
            if step.get("after_update_boundary") != "durable_output_and_payload_release":
                raise ValueError("Resident snapshot was taken before payload release")
            seconds = step["phase_seconds"]["payload_eviction"]
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError("Missing or invalid payload-eviction timing")
            previous_count = step["frame_count"] - settings["frames_per_action"]
            previous_bank = set(steps[index - 1]["after_update"]["eligible_frame_indices"]) if index else {0}
            prospective = previous_bank | set(range(previous_count, step["frame_count"]))
            validate_payload_snapshot(step["before_update"], prospective, previous_count)
            validate_payload_snapshot(memory, set(eligible), step["frame_count"])
            if memory["components"]["navigator_frames"]["live_items"]:
                raise ValueError("Navigator retained frame aliases")


def validate_payload_snapshot(snapshot, expected_indices, durable_frames):
    payload = snapshot["frame_payloads"]
    if payload["schema"] != "vmem_resident_frames_v1" or payload["mode"] != "resident":
        raise ValueError("Missing resident-payload accounting")
    fields = {"pil_frames", "latents", "encoder_embeddings", "Ks", "surfel_depths"}
    if set(payload["resident_indices"]) != fields or set(payload["logical_bytes"]) != fields:
        raise ValueError("Incomplete resident-payload accounting")
    for name in fields:
        ids = payload["resident_indices"][name]
        if len(ids) != len(set(ids)) or set(ids) != expected_indices:
            raise ValueError(f"Resident {name} leaked or lost a required frame")
        if payload["resident_counts"][name] != len(ids) or snapshot["components"][name]["live_items"] != len(ids):
            raise ValueError(f"Resident {name} count mismatch")
        if not isinstance(payload["logical_bytes"][name], int) or payload["logical_bytes"][name] <= 0:
            raise ValueError(f"Missing resident {name} bytes")
    if payload["total_logical_bytes"] != sum(payload["logical_bytes"].values()):
        raise ValueError("Resident-payload byte total mismatch")
    if not payload["arrays_own_storage"] or payload["pending_evictions"]:
        raise ValueError("Shared backing storage or uncommitted evictions remain")
    if payload["durable_frames"] != durable_frames:
        raise ValueError("Durable output does not match the storage boundary")


def validate_frame_manifest(path, expected_count):
    data = json.loads((path / "frame_manifest.json").read_text())
    if data.get("schema") != "vmem_output_frames_v1" or len(data["sha256"]) != expected_count:
        raise ValueError("Incomplete disk-output frame manifest")
    for index, expected_hash in enumerate(data["sha256"]):
        if file_hash(path / "generated_frames" / f"{index:04d}.png") != expected_hash:
            raise ValueError(f"Durable output frame {index} differs from its recorded hash")


def validate_snapshot(memory):
    if not isinstance(memory, dict):
        raise ValueError("Missing before/after-update resource snapshot")
    required_components = {"pil_frames", "latents", "encoder_embeddings", "c2ws", "Ks",
                           "surfel_depths", "surfel_Ks", "surfels", "surfel_to_timestep",
                           "memory_buffer", "retrieval_trace", "memory_events"}
    components = memory["components"]
    if not required_components.issubset(components):
        raise ValueError("Persistent component accounting is incomplete")
    byte_fields = ("python_bytes", "cpu_backing_bytes", "cuda_backing_bytes", "pil_pixel_bytes_estimate")
    for component in components.values():
        for key in (*byte_fields, "items", "live_items", "estimated_bytes", "backing_allocations"):
            if not isinstance(component[key], int) or component[key] < 0:
                raise ValueError(f"Invalid component accounting: {key}")
        if component["estimated_bytes"] != sum(component[key] for key in byte_fields):
            raise ValueError("Component byte total mismatch")
    if memory["estimated_component_bytes"] != sum(c["estimated_bytes"] for c in components.values()):
        raise ValueError("Snapshot byte total mismatch")
    for key in ("surfel_count", "surfel_reference_count", "process_rss_bytes",
                "cuda_allocated_bytes", "cuda_reserved_bytes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
        if not isinstance(memory[key], int) or memory[key] < 0:
            raise ValueError(f"Missing or invalid CECSL measurement: {key}")
    if memory["cuda_allocated_bytes"] > memory["cuda_reserved_bytes"]:
        raise ValueError("Allocated CUDA bytes exceed reserved bytes")


def validate_retrieval(trace, steps):
    if len(trace) != len(steps):
        raise ValueError("Retrieval trace is incomplete")
    previous_count = 1
    previous_bank = {0}
    for index, event in enumerate(trace):
        if event["global_step"] != index:
            raise ValueError("Retrieval step index mismatch")
        if set(event["allowed_memory_indices"]) != previous_bank:
            raise ValueError("Retrieval bank differs from previous update")
        if not all(idx in previous_bank for idx in event["selected_context_indices"]):
            raise ValueError("Retriever selected an ineligible frame")
        if event["target_frame_indices"] != list(range(previous_count, steps[index]["frame_count"])):
            raise ValueError("Retrieval targets do not match generated frames")
        if steps[index]["policy"] == "unbounded" or steps[index]["scope"] == "view_context":
            expected_reconstruction = set(range(steps[index]["frame_count"]))
        else:
            expected_reconstruction = previous_bank | set(event["target_frame_indices"])
        if steps[index]["reconstruction_input_indices"] != sorted(expected_reconstruction):
            raise ValueError("Reconstruction inputs differ from the no-window transfer protocol")
        previous_count = steps[index]["frame_count"]
        previous_bank = set(steps[index]["after_update"]["eligible_frame_indices"])


def normalized_config(path):
    from omegaconf import OmegaConf
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    config.pop("visualization_dir", None)
    config["model"].pop("samples_dir", None)
    return config


def inspect_attempt(path, row, lock_path, lock):
    item = {"run_id": row["run_id"], "run_dir": str(path), "policy": row["memory_policy"],
            "case_id": row.get("_case_id", row["run_id"]), "status": "unverifiable",
            "provenance_verified": False}
    try:
        spec = json.loads((path / "run_spec.json").read_text())
        verify_lock(lock_path, spec["arguments"], spec["provenance"])
        if spec["arguments"]["run_id"] != row["run_id"]:
            raise ValueError("Run ID mismatch")
        if spec["provenance"].get("experiment_lock_sha256") != file_hash(lock_path):
            raise ValueError("Run was not executed under this experiment lock")
        item["provenance_verified"] = True
        item["resumed"] = bool(spec.get("recovery", {}).get("resume"))
        status_path = path / "run_status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get("status") != "complete":
            item.update(status=status["status"] if status.get("status") in {"failed", "paused"} else "unfinished", details=status)
            return item
        metadata = json.loads((path / "metadata.json").read_text())
        settings = expected_settings(row)
        if metadata.get("frame_storage", "legacy") != settings["frame_storage"]:
            raise ValueError("Metadata frame storage differs from frozen settings")
        for key in ("run_id", "seed", "image", "trajectory", "fps", "step_size", "frames_per_action", "num_actions", "memory_policy", "memory_budget", "memory_scope"):
            if metadata.get(key) != settings[key]:
                raise ValueError(f"Metadata mismatch: {key}")
        if metadata["provenance"] != spec["provenance"]:
            raise ValueError("Metadata and run spec provenance differ")
        if metadata.get("recovery") != spec.get("recovery"):
            raise ValueError("Metadata and run spec recovery lineage differ")
        if set(spec["provenance"].get("checkpoint_sha256", {})) != {"vmem", "cut3r"}:
            raise ValueError("Missing VMem/CUT3R checkpoint hashes")
        if file_hash(lock["config"]) != lock["config_sha256"]:
            raise ValueError("Frozen source config file has changed")
        expected_config = normalized_config(lock["config"])
        if settings["inference_steps"] is not None:
            expected_config["model"]["inference_num_steps"] = settings["inference_steps"]
        if settings["surfel_niter"] is not None:
            expected_config["surfel"]["niter"] = settings["surfel_niter"]
        expected_config["inference"]["visualize"] = settings["visualize_intermediates"]
        if normalized_config(path / "generation_config.yaml") != expected_config:
            raise ValueError("Effective generation config differs from frozen settings")
        actions = json.loads((path / "actions.json").read_text())
        validate_actions(actions, settings)
        steps = read_resource_steps(path / "resource_trace.jsonl")
        validate_resources(steps, settings)
        validate_retrieval(json.loads((path / "retrieval_trace.json").read_text()), steps)
        video = probe_video(path / "generated.mp4")
        expected_frames = 1 + settings["num_actions"] * settings["frames_per_action"]
        if settings["frame_storage"] == "resident":
            validate_frame_manifest(path, expected_frames)
            if metadata["frame_payloads"] != steps[-1]["after_update"]["frame_payloads"]:
                raise ValueError("Final resident-payload metadata differs from trace")
        if metadata["actual_frames"] != expected_frames or video["frames"] != expected_frames:
            raise ValueError("Decoded video frame count mismatch")
        if (video["width"], video["height"]) != (expected_config["model"]["width"], expected_config["model"]["height"]):
            raise ValueError("Decoded video dimensions mismatch")
        if abs(video["fps"] - settings["fps"]) > 1e-6 or abs(video["seconds"] - expected_frames / settings["fps"]) > 1 / settings["fps"]:
            raise ValueError("Decoded video fps/duration mismatch")
        item.update(status="validated", video=video, checkpoints=spec["provenance"]["checkpoint_sha256"],
                    environment={key: spec.get(key) for key in ("torch_version", "cuda_version")})
    except (OSError, ValueError, KeyError, TypeError, StopIteration, subprocess.CalledProcessError) as exc:
        item.update(status="invalid", error=str(exc))
    return item


def inventory(args):
    lock = json.loads(args.lock.read_text())
    attempts = []
    for row in lock["rows"]:
        candidates = sorted(args.output_root.glob(row["run_id"] + "_*"))
        if not candidates:
            attempts.append({"run_id": row["run_id"], "case_id": row.get("_case_id"), "policy": row["memory_policy"], "status": "missing"})
        for path in candidates:
            attempts.append(inspect_attempt(path, row, args.lock, lock))
    pairs = []
    for index in range(0, len(lock["rows"]), 2):
        rows = lock["rows"][index:index + 2]
        selected = []
        for row in rows:
            good = [item for item in attempts if item["run_id"] == row["run_id"] and item["status"] == "validated"]
            selected.append(max(good, key=lambda item: (Path(item["run_dir"]) / "metadata.json").stat().st_mtime) if good else None)
        valid = all(selected)
        reason = "matched" if valid else "missing validated arm"
        if valid and (selected[0]["checkpoints"] != selected[1]["checkpoints"] or selected[0]["environment"] != selected[1]["environment"]):
            valid, reason = False, "checkpoint or runtime environment mismatch"
        pairs.append({"case_id": rows[0].get("_case_id", rows[0]["run_id"]), "validated_pair": bool(valid),
                      "uninterrupted_resource_pair": bool(valid and not any(arm.get("resumed") for arm in selected)),
                      "reason": reason, "arms": selected})
    write_json(args.output, {"schema": "vmem_inventory_v1", "lock": str(args.lock),
                             "attempts": attempts, "pairs": pairs, "quality_results": "pending user evaluation"})
    print(f"Validated pairs: {sum(p['validated_pair'] for p in pairs)}/{len(pairs)}. All attempts saved to {args.output}.")


def storage_smoke(args):
    path = args.run_dir
    spec = json.loads((path / "run_spec.json").read_text())
    metadata = json.loads((path / "metadata.json").read_text())
    status = json.loads((path / "run_status.json").read_text())
    settings = expected_settings(spec["arguments"])
    if (status.get("status") != "complete" or settings["frame_storage"] != "resident"
            or settings["memory_policy"] != "slam_covisibility" or settings["memory_budget"] != 32
            or settings["num_actions"] != 12 or settings["frames_per_action"] != 4):
        raise ValueError("Expected a completed 12-action, resident GeoCov-32 smoke run")
    current = _generation_provenance(Path(settings["image"]), Path(spec["arguments"]["config"]))
    for key in ("source_sha256", "config_sha256", "image_sha256"):
        if current[key] != spec["provenance"][key]:
            raise ValueError(f"Smoke run does not match current {key}")
    if metadata["provenance"] != spec["provenance"] or metadata["recovery"] != spec["recovery"]:
        raise ValueError("Smoke metadata and specification differ")
    if spec["recovery"]["schema"] != "vmem_recovery_v2":
        raise ValueError("Smoke run does not use resident-aware recovery")
    resume = spec["recovery"].get("resume")
    if resume is None or resume["completed_actions"] != 10:
        raise ValueError("Pause after 10 actions and resume to exercise recovery after eviction")
    validate_actions(json.loads((path / "actions.json").read_text()), settings)
    steps = read_resource_steps(path / "resource_trace.jsonl")
    validate_resources(steps, settings)
    validate_retrieval(json.loads((path / "retrieval_trace.json").read_text()), steps)
    validate_frame_manifest(path, 49)
    video = probe_video(path / "generated.mp4")
    if (video["frames"] != 49 or metadata["actual_frames"] != 49
            or (video["width"], video["height"]) != (576, 576)
            or abs(video["fps"] - settings["fps"]) > 1e-6
            or abs(video["seconds"] - 49 / settings["fps"]) > 1 / settings["fps"]):
        raise ValueError("Smoke video does not contain all 49 frames at the expected format")
    payload = steps[-1]["after_update"]["frame_payloads"]
    if metadata["frame_storage"] != "resident" or metadata["frame_payloads"] != payload:
        raise ValueError("Smoke final payload metadata differs from trace")
    print(json.dumps({"status": "validated", "video": video, "resumed_after_actions": 10,
                      "resident_counts": payload["resident_counts"], "durable_frames": payload["durable_frames"],
                      "arrays_own_storage": payload["arrays_own_storage"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    lock_parser = sub.add_parser("freeze")
    lock_parser.add_argument("manifest", type=Path)
    lock_parser.add_argument("--config", type=Path, default=Path("configs/inference/inference.yaml"))
    lock_parser.add_argument("--output", type=Path, required=True)
    audit_parser = sub.add_parser("inventory")
    audit_parser.add_argument("--lock", type=Path, required=True)
    audit_parser.add_argument("--output-root", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    smoke_parser = sub.add_parser("storage-smoke")
    smoke_parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    {"freeze": freeze, "inventory": inventory, "storage-smoke": storage_smoke}[args.command](args)


if __name__ == "__main__":
    main()
