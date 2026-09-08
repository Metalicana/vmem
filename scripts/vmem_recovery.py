"""Action-boundary recovery for locally created VMem runs.

Checkpoints are trusted local pickle files, not portable/safe model downloads.
Weights are reloaded separately. Restore creates a new attempt, never rewrites
the source attempt, and restores RNG only after model/state initialization.
"""

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
from PIL import Image


SCHEMA = "vmem_recovery_v1"
STATE_FIELDS = (
    "latents", "encoder_embeddings", "poses", "c2ws", "Ks", "focal_lengths",
    "surfel_depths", "surfel_Ks", "surfels", "surfel_to_timestep",
    "memory_buffer", "memory_policy", "memory_budget", "memory_scope",
    "surfel_reconstruction_window", "global_step", "initial_threshold",
    "memory_descriptor_sources", "_active_target_frame_indices",
    "retrieval_trace", "memory_events", "_dino_feature_cache",
)
TRACE_FILES = ("resource_trace.jsonl", "actions.partial.jsonl", "recovery_trace.jsonl")
IDENTITY_ARGS = (
    "image", "run_id", "trajectory", "pattern", "num_actions", "fps", "step_size",
    "frames_per_action", "seed", "memory_policy", "memory_budget", "memory_scope",
    "inference_steps", "surfel_niter", "surfel_reconstruction_window",
    "visualize_intermediates", "resource_trace", "profile_warmup_steps", "checkpoint_every",
)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        # CECSL uses Linux: persist the rename as well as the file contents.
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path, value):
    atomic_write(path, lambda handle: handle.write(json.dumps(value, indent=2, allow_nan=False).encode()))


def identity(arguments, provenance, torch_module):
    return {
        "arguments": {key: str(arguments[key]) if isinstance(arguments[key], Path) else arguments[key]
                      for key in IDENTITY_ARGS},
        "provenance": {key: provenance.get(key) for key in (
            "source_sha256", "config_sha256", "image_sha256", "checkpoint_sha256", "experiment_lock_sha256",
        )},
        "torch_version": str(torch_module.__version__), "cuda_version": torch_module.version.cuda,
    }


def save_new_frames(run_dir, frames, hashes):
    started = time.perf_counter()
    for index in range(len(hashes), len(frames)):
        path = Path(run_dir) / "generated_frames" / f"{index:04d}.png"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite a durable frame: {path}")
        atomic_write(path, lambda handle: frames[index].save(handle, format="PNG"))
        hashes.append(digest_file(path))
    return time.perf_counter() - started


def save_checkpoint(run_dir, pipeline, navigator, action_records, frame_hashes, run_identity, torch_module):
    started = time.perf_counter()
    if len(frame_hashes) != len(pipeline.pil_frames):
        raise ValueError("Save every frame before committing its checkpoint")
    frame_ids = {id(frame): index for index, frame in enumerate(pipeline.pil_frames)}
    navigator_ids = [frame_ids[id(frame)] for frame in navigator.frames]
    device = pipeline.device
    if str(device).startswith("cuda"):
        torch_module.cuda.synchronize(device)
    state = {
        "schema": SCHEMA, "identity": run_identity,
        "completed_actions": len(action_records), "frame_count": len(frame_hashes),
        "pipeline": {name: getattr(pipeline, name) for name in STATE_FIELDS if hasattr(pipeline, name)},
        "navigator": {"current_pose": navigator.current_pose, "current_K": navigator.current_K,
                      "pose_history": navigator.pose_history, "frame_indices": navigator_ids},
        "action_records": action_records, "frame_sha256": frame_hashes,
        "has_dino_extractor": getattr(pipeline, "_dino_extractor", None) is not None,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch_cpu": torch_module.get_rng_state(),
                "torch_cuda": torch_module.cuda.get_rng_state(device) if str(device).startswith("cuda") else None},
        "trace_prefixes": {},
    }
    for name in TRACE_FILES:
        path = Path(run_dir) / name
        if path.exists():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            state["trace_prefixes"][name] = {"bytes": path.stat().st_size, "sha256": digest_file(path)}
    path = Path(run_dir) / "recovery" / "latest.pt"
    atomic_write(path, lambda handle: torch_module.save(state, handle, pickle_protocol=4))
    return {"event": "checkpoint", "completed_actions": len(action_records),
            "frame_count": len(frame_hashes), "checkpoint_seconds": time.perf_counter() - started,
            "checkpoint_bytes": path.stat().st_size}


def load_checkpoint(source_dir, run_identity, torch_module, device):
    path = Path(source_dir) / "recovery" / "latest.pt"
    if not path.is_file():
        raise FileNotFoundError(f"No recovery checkpoint at {path}; action logs alone cannot resume a run")
    # Open once: a concurrent atomic checkpoint replacement cannot change this read.
    with path.open("rb") as handle:
        sha = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
        handle.seek(0)
        def map_storage(storage, location):
            if location.startswith("cuda") and str(device).startswith("cuda"):
                return storage.cuda(device=device)
            return storage
        state = torch_module.load(handle, map_location=map_storage, weights_only=False)
    if state.get("schema") != SCHEMA or state.get("identity") != run_identity:
        raise ValueError("Recovery source/config/input/settings/model/runtime identity mismatch")
    cuda_state = state["rng"]["torch_cuda"]
    if (cuda_state is not None) != str(device).startswith("cuda"):
        raise ValueError("Recovery cannot switch between CPU and CUDA execution")
    count = state["completed_actions"]
    frame_count = state["frame_count"]
    if count != len(state["action_records"]) or frame_count != 1 + count * run_identity["arguments"]["frames_per_action"]:
        raise ValueError("Inconsistent recovery action/frame counts")
    if state["pipeline"]["global_step"] != count:
        raise ValueError("Recovery supports one four-frame generation step per action")
    if len(state["frame_sha256"]) != frame_count:
        raise ValueError("Incomplete recovery frame hashes")
    return state, sha.hexdigest()


def copy_prefix(source, target, prefix):
    digest = hashlib.sha256()
    remaining = prefix["bytes"]
    with source.open("rb") as src, target.open("xb") as dst:
        while remaining:
            block = src.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError(f"Incomplete checkpoint trace: {source}")
            dst.write(block)
            digest.update(block)
            remaining -= len(block)
    if digest.hexdigest() != prefix["sha256"]:
        raise ValueError(f"Checkpoint trace hash mismatch: {source}")


def restore_into_new_attempt(source_dir, run_dir, state, pipeline, navigator):
    source_dir, run_dir = Path(source_dir), Path(run_dir)
    if source_dir.resolve() == run_dir.resolve():
        raise ValueError("Recovery must create a new attempt, preserving the original")
    frame_dir = run_dir / "generated_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, expected_hash in enumerate(state["frame_sha256"]):
        source = source_dir / "generated_frames" / f"{index:04d}.png"
        if digest_file(source) != expected_hash:
            raise ValueError(f"Recovery frame hash mismatch: {source}")
        # Copy, not hard-link: later modifications must not mutate the parent run.
        target = frame_dir / source.name
        with target.open("xb") as dst, source.open("rb") as src:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        with Image.open(target) as image:
            frames.append(image.copy())
    directory = os.open(frame_dir, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    for name, prefix in state["trace_prefixes"].items():
        if name not in TRACE_FILES:
            raise ValueError(f"Unknown checkpoint trace: {name}")
        copy_prefix(source_dir / name, run_dir / name, prefix)
    for name, value in state.pop("pipeline").items():
        if name not in STATE_FIELDS:
            raise ValueError(f"Unknown pipeline state: {name}")
        setattr(pipeline, name, value)
    pipeline.pil_frames = frames
    saved_navigator = state.pop("navigator")
    navigator.current_pose = saved_navigator["current_pose"]
    navigator.current_K = saved_navigator["current_K"]
    navigator.pose_history = saved_navigator["pose_history"]
    navigator.frames = [frames[index] for index in saved_navigator["frame_indices"]]
    if state["has_dino_extractor"]:
        pipeline._dino_extractor_instance()
    return state.pop("action_records"), state.pop("frame_sha256"), state.pop("rng")


def restore_rng(rng, torch_module, device):
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch_module.set_rng_state(rng["torch_cpu"])
    if rng["torch_cuda"] is not None:
        torch_module.cuda.set_rng_state(rng["torch_cuda"], device)


def append_event(run_dir, event):
    with (Path(run_dir) / "recovery_trace.jsonl").open("a") as handle:
        handle.write(json.dumps(event, allow_nan=False) + "\n")
