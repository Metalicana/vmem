"""Pure-Python commanded-path and frozen-setting helpers; no model imports."""

import hashlib
import json
import math
from pathlib import Path


def scaling_actions(trajectory, num_actions):
    if trajectory not in {"fixed_region_v1", "expanding_excursions_v1"}:
        raise ValueError(f"Not a scaling trajectory: {trajectory}")
    actions = []
    excursion = 1
    while len(actions) < num_actions:
        length = 8 if trajectory == "fixed_region_v1" else 8 * excursion
        actions.extend(["forward"] * length + ["backward"] * length)
        excursion += 1
    return actions[:num_actions]


def commanded_path(actions, step_size, frames_per_action=4, fps=13):
    x, z, yaw, distance, max_radius = 0.0, 0.0, 0.0, 0.0, 0.0
    records = []
    for index, action in enumerate(actions):
        angle = {"left5": 5, "right5": -5, "left10": 10, "right10": -10}.get(action)
        if angle is not None:
            yaw += math.radians(angle)
            behavior = "rotation"
        elif action in {"forward", "backward"}:
            before = math.hypot(x, z)
            sign = -1 if action == "forward" else 1
            x += sign * step_size * math.sin(yaw)
            z += sign * step_size * math.cos(yaw)
            distance += abs(step_size)
            radius = math.hypot(x, z)
            behavior = "new_commanded_extent" if radius > max_radius + 1e-8 else "return" if radius < before - 1e-8 else "repeat"
            max_radius = max(max_radius, radius)
        else:
            raise ValueError(f"Unknown camera action: {action}")
        c, s = math.cos(yaw), math.sin(yaw)
        pose = [[c, 0, s, x], [0, 1, 0, 0], [-s, 0, c, z], [0, 0, 0, 1]]
        records.append({
            "action_index": index, "action": action,
            "frame_index": (index + 1) * frames_per_action,
            "time_seconds": (index + 1) * frames_per_action / fps,
            "current_pose": pose, "behavior": behavior,
            "path_length": distance, "max_radius": max_radius,
            "at_origin": math.hypot(x, z) < 1e-6 and abs(math.sin(yaw / 2)) < 1e-6,
        })
    return records


def expected_settings(row):
    values = {
        "image": str(row["image"]), "run_id": row["run_id"],
        "trajectory": row.get("trajectory", "pattern"), "pattern": row.get("pattern", "forward"),
        "step_size": row.get("step_size", 0.1), "fps": row.get("fps", 13.0),
        "frames_per_action": row.get("frames_per_action", 4), "seed": row.get("seed", 42),
        "memory_policy": row.get("memory_policy", "unbounded"), "memory_budget": row.get("memory_budget"),
        "memory_scope": row.get("memory_scope", "surfel_indexed_view_memory"),
        "inference_steps": row.get("inference_steps"), "surfel_niter": row.get("surfel_niter"),
        "surfel_reconstruction_window": row.get("surfel_reconstruction_window"),
        "resource_trace": True, "profile_warmup_steps": 2,
        "checkpoint_every": row.get("checkpoint_every", 5),
        "frame_storage": row.get("frame_storage", "legacy"),
        "visualize_intermediates": row.get("visualize_intermediates", False),
    }
    num_actions = row.get("num_actions")
    if num_actions is None:
        if row.get("duration_seconds") is None:
            num_actions = 3
        else:
            new_frames = max(1, round(row["duration_seconds"] * values["fps"]))
            num_actions = max(1, (new_frames + values["frames_per_action"] - 1) // values["frames_per_action"])
    values["num_actions"] = num_actions
    return values


def verify_lock(lock_path, arguments, provenance):
    lock = json.loads(Path(lock_path).read_text())
    for key in ("source_sha256", "config_sha256"):
        if provenance[key] != lock[key]:
            raise ValueError(f"Experiment lock mismatch: {key}")
    expected = next((row for row in lock["rows"] if row["run_id"] == arguments["run_id"]), None)
    if expected is None:
        raise ValueError("Run ID is not in the experiment lock")
    if provenance["image_sha256"] != lock["image_sha256"][expected["image"]]:
        raise ValueError("Experiment lock mismatch: input image")
    for key, value in expected_settings(expected).items():
        if key == "checkpoint_every" and lock.get("schema") not in {"vmem_experiment_lock_v2", "vmem_experiment_lock_v3"}:
            continue
        if key == "frame_storage" and lock.get("schema") != "vmem_experiment_lock_v3":
            if arguments.get(key, "legacy") != "legacy" or value != "legacy":
                raise ValueError("Resident storage requires a new v3 experiment lock")
            continue
        actual = arguments.get(key)
        if isinstance(actual, Path):
            actual = str(actual)
        if actual != value:
            raise ValueError(f"Experiment lock mismatch: {key}: {actual!r} != {value!r}")
    return hashlib.sha256(Path(lock_path).read_bytes()).hexdigest()
