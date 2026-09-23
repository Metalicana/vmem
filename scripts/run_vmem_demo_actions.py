#!/usr/bin/env python
"""Run VMem by chaining the same small actions used by the Gradio demo."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MEMORY_POLICIES = (
    "unbounded",
    "fifo",
    "rarity_irreplaceability",
    "slam_covisibility",
    "mce",
    "kcenter_coreset",
)
BUDGETED_MEMORY_POLICIES = (
    "fifo",
    "rarity_irreplaceability",
    "slam_covisibility",
    "mce",
    "kcenter_coreset",
)
ACTION_ALIASES = {
    "f": "forward",
    "forward": "forward",
    "b": "backward",
    "backward": "backward",
    "l5": "left5",
    "left5": "left5",
    "left": "left5",
    "l10": "left10",
    "left10": "left10",
    "r5": "right5",
    "right5": "right5",
    "right": "right5",
    "r10": "right10",
    "right10": "right10",
}
TRAJECTORY_ALIASES = {
    "pattern": "pattern",
    "forward": "forward",
    "out_and_back": "out_and_back",
    "square": "square_walk",
    "square_walk": "square_walk",
    "pan": "pan_180",
    "pan_180": "pan_180",
    "left_right_180": "pan_180",
    "sweep_180": "pan_180",
    "pan_90": "pan_90",
    "left_right_90": "pan_90",
    "sweep_90": "pan_90",
    "pan_45": "pan_45",
    "left_right_45": "pan_45",
    "sweep_45": "pan_45",
    "spin": "spin_360",
    "spin_360": "spin_360",
    "rotate_360": "spin_360",
    "local_loop": "local_loop",
    "random": "random_walk",
    "random_walk": "random_walk",
    "fixed_region_v1": "fixed_region_v1",
    "expanding_excursions_v1": "expanding_excursions_v1",
}


def _load_runtime_dependencies() -> None:
    global Navigator
    global OmegaConf
    global VMemPipeline
    global get_default_intrinsics
    global load_img_and_K
    global np
    global torch
    global transform_img_and_K

    import numpy as np
    from omegaconf import OmegaConf
    import torch

    from modeling.pipeline import VMemPipeline
    from navigation import Navigator
    from utils import get_default_intrinsics, load_img_and_K, transform_img_and_K


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(inner) for inner in value]
    return value


def _save_pil_video(frames: Iterable, path: Path, *, fps: float) -> None:
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Saving MP4 videos requires imageio. Install the project requirements "
            "in the VMem environment before running generation."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB")))
    finally:
        writer.close()


def _load_vmem_image(path: Path, *, config, device):
    image, _ = load_img_and_K(str(path), None, K=None, device=device)
    image, _ = transform_img_and_K(
        image,
        (config.model.height, config.model.width),
        mode="crop",
        K=None,
    )
    return image


def _parse_pattern(pattern: str) -> list[str]:
    actions = []
    for raw_action in pattern.split(","):
        key = raw_action.strip().lower()
        if not key:
            continue
        if key not in ACTION_ALIASES:
            raise ValueError(
                f"Unsupported action {raw_action!r}. "
                f"Expected one of {sorted(ACTION_ALIASES)}"
            )
        actions.append(ACTION_ALIASES[key])
    if not actions:
        raise ValueError("--pattern must include at least one action")
    return actions


def _repeat_to_length(base: Sequence[str], *, num_actions: int) -> list[str]:
    if not base:
        raise ValueError("trajectory action template must include at least one action")
    return [base[idx % len(base)] for idx in range(num_actions)]


def _expand_actions(pattern: str, *, num_actions: int) -> list[str]:
    base = _parse_pattern(pattern)
    return _repeat_to_length(base, num_actions=num_actions)


def _canonical_trajectory(name: str) -> str:
    key = name.strip().lower()
    if key not in TRAJECTORY_ALIASES:
        raise ValueError(
            f"Unsupported trajectory {name!r}. "
            f"Expected one of {sorted(TRAJECTORY_ALIASES)}"
        )
    return TRAJECTORY_ALIASES[key]


def _expand_trajectory_actions(args) -> list[str]:
    trajectory = _canonical_trajectory(args.trajectory)
    if trajectory in {"fixed_region_v1", "expanding_excursions_v1"}:
        from scripts.vmem_protocol import scaling_actions
        return scaling_actions(trajectory, args.num_actions)
    if trajectory == "pattern":
        return _expand_actions(args.pattern, num_actions=args.num_actions)
    if trajectory == "forward":
        return _repeat_to_length(["forward"], num_actions=args.num_actions)
    if trajectory == "out_and_back":
        base = ["forward"] * 20 + ["backward"] * 20
        return _repeat_to_length(base, num_actions=args.num_actions)
    if trajectory == "square_walk":
        side = ["forward"] * 10
        right_angle_turn = ["right10"] * 9
        base = []
        for _ in range(4):
            base.extend(side)
            base.extend(right_angle_turn)
        return _repeat_to_length(base, num_actions=args.num_actions)
    if trajectory == "pan_180":
        base = ["left10"] * 18 + ["right10"] * 36 + ["left10"] * 18
        return _repeat_to_length(base, num_actions=args.num_actions)
    if trajectory == "pan_90":
        base = ["left10"] * 9 + ["right10"] * 18 + ["left10"] * 9
        return _repeat_to_length(base, num_actions=args.num_actions)
    if trajectory == "pan_45":
        base = ["left5"] * 9 + ["right5"] * 18 + ["left5"] * 9
        return _repeat_to_length(base, num_actions=args.num_actions)
    if trajectory == "spin_360":
        return _repeat_to_length(["right10"] * 36, num_actions=args.num_actions)
    if trajectory == "local_loop":
        base = ["forward"] * 4 + ["left5"] * 4 + ["backward"] * 4 + ["right5"] * 4
        return _repeat_to_length(base, num_actions=args.num_actions)
    if trajectory == "random_walk":
        rng = random.Random(args.seed)
        population = [
            "forward",
            "backward",
            "left5",
            "right5",
            "left10",
            "right10",
        ]
        weights = [0.58, 0.04, 0.16, 0.16, 0.03, 0.03]
        return [
            rng.choices(population, weights=weights, k=1)[0]
            for _ in range(args.num_actions)
        ]
    raise ValueError(f"Unsupported trajectory: {args.trajectory}")


def _action_histogram(actions: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(actions).items()))


def _num_actions_from_duration(duration_seconds: float, *, fps: float, frames_per_action: int) -> int:
    target_new_frames = max(1, int(round(float(duration_seconds) * float(fps))))
    return max(1, (target_new_frames + frames_per_action - 1) // frames_per_action)


def _apply_action(navigator, action: str):
    if action == "forward":
        return navigator.move_forward(1)
    if action == "backward":
        return navigator.move_backward(1)
    if action == "left5":
        return navigator.turn_left(5)
    if action == "left10":
        return navigator.turn_left(10)
    if action == "right5":
        return navigator.turn_right(5)
    if action == "right10":
        return navigator.turn_right(10)
    raise ValueError(f"Unsupported action: {action}")


def _run_name(args, *, num_actions: int) -> str:
    image_stem = (args.run_id or args.image.stem).replace(" ", "_")
    policy = args.memory_policy
    if args.memory_policy in BUDGETED_MEMORY_POLICIES:
        policy = f"{policy}_B{args.memory_budget}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{image_stem}_{args.trajectory}_A{num_actions}_{policy}_{timestamp}"


def _generation_provenance(image_path: Path, config_path: Path) -> dict:
    source_paths = (
        "scripts/run_vmem_demo_actions.py", "navigation.py", "modeling/pipeline.py",
        "modeling/memory_policies.py", "modeling/sampling.py", "utils/util.py",
        "modeling/resource_audit.py", "scripts/vmem_protocol.py",
        "scripts/vmem_recovery.py",
        "frame_storage.py",
        "generation_debug.py", "generation_rng.py", "clip_attention.py",
        "modeling/modules/autoencoder.py", "modeling/modules/conditioner.py",
        "extern/CUT3R/surfel_inference.py", "extern/CUT3R/src/dust3r/inference.py",
        "extern/CUT3R/src/dust3r/blocks.py", "extern/CUT3R/src/dust3r/model.py",
    )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "git_commit": commit,
        "source_sha256": {
            path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
            for path in source_paths
        },
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "python_version": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _checkpoint_hashes(pipeline):
    result = {}
    for name, path in pipeline.checkpoint_paths.items():
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result[name] = digest.hexdigest()
    return result


def _install_failure_log(run_dir, runtime):
    from scripts.vmem_recovery import atomic_json
    previous_hook = sys.excepthook

    def on_exception(error_type, error, traceback):
        status = {"status": "failed", "phase": runtime["phase"],
                  "error_type": error_type.__name__, "error": str(error)}
        try:
            pipeline = runtime.get("pipeline")
            if pipeline is not None:
                status["actual_frames"] = len(pipeline.pil_frames)
                profiler = getattr(pipeline, "resource_profiler", None)
                if profiler is not None:
                    profiler.failure(error)
            status["pid"] = os.getpid()
            atomic_json(run_dir / "run_status.json", status)
        finally:
            previous_hook(error_type, error, traceback)

    sys.excepthook = on_exception
    atomic_json(run_dir / "run_status.json", {"status": "running", "pid": os.getpid()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/demo_actions"))
    parser.add_argument("--config", type=Path, default=Path("configs/inference/inference.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=13.0)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--num-actions", type=int)
    parser.add_argument(
        "--trajectory",
        choices=sorted(TRAJECTORY_ALIASES),
        default="pattern",
        help=(
            "Named demo-action trajectory. Use pattern to repeat --pattern. "
            "Other presets are generated from the same Navigator actions."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="forward",
        help=(
            "Comma-separated demo actions repeated until num-actions is reached. "
            "Choices: forward, backward, left5, left10, right5, right10."
        ),
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.1,
        help="Navigator step size. The Gradio demo uses 0.1.",
    )
    parser.add_argument(
        "--frames-per-action",
        type=int,
        default=4,
        help="Interpolated frames per action. The Gradio demo uses 4.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rng-mode", choices=("legacy", "isolated"), default="legacy",
                        help="Phase/action RNG isolation without diagnostic hashing; supports long runs and recovery.")
    parser.add_argument("--generation-debug", choices=("observe", "isolated"),
                        help="Short fresh-run fingerprints; isolated additionally uses phase/action RNG. Requires --checkpoint-every 0.")
    parser.add_argument("--clip-attention", choices=("native", "math"), default="native",
                        help="CLIP-only attention dispatch; math requires isolated RNG or a short debug run.")
    parser.add_argument("--cut3r-attention", choices=("native", "math"), default="native",
                        help="CUT3R inference-only attention dispatch; math requires isolated RNG or a short debug run.")
    parser.add_argument(
        "--memory-policy",
        choices=MEMORY_POLICIES,
        default="unbounded",
    )
    parser.add_argument("--memory-budget", type=int)
    parser.add_argument(
        "--memory-scope",
        choices=("surfel_indexed_view_memory", "view_context"),
        default="surfel_indexed_view_memory",
    )
    parser.add_argument("--inference-steps", type=int)
    parser.add_argument("--surfel-niter", type=int)
    parser.add_argument("--surfel-reconstruction-window", type=int)
    parser.add_argument("--frame-storage", choices=("legacy", "resident"), default="legacy",
                        help="resident releases evicted payloads after durable PNG output; legacy is eligibility-only.")
    parser.add_argument("--save-frames", action="store_true",
                        help="Compatibility flag: this runner now always saves incremental PNG frames.")
    parser.add_argument("--visualize-intermediates", action="store_true")
    parser.add_argument("--resource-trace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile-warmup-steps", type=int, default=2)
    parser.add_argument("--experiment-lock", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="Commit recovery state every N actions; 0 disables state checkpoints, not PNG saving.")
    parser.add_argument("--resume-from", type=Path, help="Trusted local run directory containing recovery/latest.pt.")
    parser.add_argument("--stop-after-actions", type=int,
                        help="Pause this process after N additional actions with a committed checkpoint.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.trajectory = _canonical_trajectory(args.trajectory)

    if args.num_actions is None and args.duration_seconds is None:
        args.num_actions = 3
    if args.num_actions is None:
        args.num_actions = _num_actions_from_duration(
            args.duration_seconds,
            fps=args.fps,
            frames_per_action=args.frames_per_action,
        )
    if args.num_actions <= 0:
        raise ValueError("--num-actions must be positive")
    from generation_debug import validate_debug_args
    from generation_rng import PhaseRNG, execution_settings
    validate_debug_args(args)
    if args.profile_warmup_steps < 0:
        raise ValueError("--profile-warmup-steps must be nonnegative")
    if args.frames_per_action <= 0:
        raise ValueError("--frames-per-action must be positive")
    if args.frame_storage == "resident" and (
        args.frames_per_action != 4 or args.memory_scope != "surfel_indexed_view_memory"
        or args.visualize_intermediates or args.surfel_reconstruction_window is not None
    ):
        raise ValueError("Resident storage requires four-frame actions, indexed memory, no window and no full-history visualization")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be nonnegative")
    if args.checkpoint_every and args.frames_per_action != 4:
        raise ValueError("Recovery currently requires four-frame actions; use --checkpoint-every 0 otherwise")
    if args.resume_from is not None and not (args.resume_from / "recovery" / "latest.pt").is_file():
        raise ValueError("No recovery/latest.pt in --resume-from; old action logs cannot resume generation")
    if args.stop_after_actions is not None and (args.stop_after_actions <= 0 or not args.checkpoint_every):
        raise ValueError("--stop-after-actions requires a positive count and enabled checkpoints")
    if args.memory_policy in BUDGETED_MEMORY_POLICIES and (
        args.memory_budget is None or args.memory_budget <= 0
    ):
        raise ValueError(f"--memory-policy {args.memory_policy} requires --memory-budget")
    if (
        args.memory_policy in {"rarity_irreplaceability", "slam_covisibility"}
        and args.memory_budget is not None
        and args.memory_budget < 2
    ):
        raise ValueError(f"--memory-policy {args.memory_policy} requires --memory-budget >= 2")

    actions = _expand_trajectory_actions(args)
    expected_frames = 1 + args.num_actions * args.frames_per_action
    expected_seconds = expected_frames / args.fps
    if args.dry_run:
        print(
            json.dumps(
                {
                    "image": str(args.image),
                    "output_root": str(args.output_root),
                    "run_id": args.run_id,
                    "num_actions": args.num_actions,
                    "trajectory": args.trajectory,
                    "pattern": args.pattern,
                    "expanded_action_sample": actions[:20],
                    "action_histogram": _action_histogram(actions),
                    "expected_frames": expected_frames,
                    "expected_seconds": expected_seconds,
                    "memory_policy": args.memory_policy,
                    "memory_budget": args.memory_budget,
                    "frame_storage": args.frame_storage,
                    "clip_attention": args.clip_attention,
                    "cut3r_attention": args.cut3r_attention,
                    "execution": execution_settings(vars(args)),
                },
                indent=2,
            )
        )
        return

    if args.memory_policy in BUDGETED_MEMORY_POLICIES and args.frame_storage == "legacy":
        print("WARNING: legacy mode bounds eligibility only. Use --frame-storage resident to release evicted payloads.", flush=True)
    from scripts.vmem_protocol import commanded_path, verify_lock
    provenance = _generation_provenance(args.image, args.config)
    if args.experiment_lock is not None:
        provenance["experiment_lock_sha256"] = verify_lock(args.experiment_lock, vars(args), provenance)

    _load_runtime_dependencies()
    from scripts.vmem_recovery import (
        append_event, atomic_json, identity, load_checkpoint, restore_into_new_attempt,
        restore_rng, save_checkpoint, save_new_frames,
        iter_saved_frames, SCHEMA as RECOVERY_SCHEMA,
    )
    from frame_storage import release_evicted_payloads, validate_resident_payloads
    from modeling.resource_audit import SCHEMA_VERSION as RESOURCE_SCHEMA

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = args.output_root / _run_name(args, num_actions=args.num_actions)
    run_dir.mkdir(parents=True, exist_ok=False)

    config = OmegaConf.load(args.config)
    config.model.clip_attention = args.clip_attention
    config.surfel.cut3r_attention = args.cut3r_attention
    config.inference.rng_mode = args.rng_mode
    if args.inference_steps is not None:
        config.model.inference_num_steps = args.inference_steps
    if args.surfel_niter is not None:
        config.surfel.niter = args.surfel_niter
    config.visualization_dir = str(run_dir / "visualization")
    config.model.samples_dir = str(run_dir / "visualization")
    config.inference.visualize = bool(args.visualize_intermediates)

    with (run_dir / "commanded_path.json").open("w", encoding="utf-8") as handle:
        json.dump(commanded_path(actions, args.step_size, args.frames_per_action, args.fps), handle, indent=2)
    OmegaConf.save(config, run_dir / "generation_config.yaml")
    with (run_dir / "run_spec.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe({
            "arguments": vars(args), "actions": actions, "provenance": provenance,
            "execution": execution_settings(vars(args)),
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        }), handle, indent=2)

    runtime = {"phase": "model_load"}
    _install_failure_log(run_dir, runtime)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    debug = None
    if args.generation_debug is not None:
        from generation_debug import GenerationDebug
        debug = GenerationDebug(run_dir / "generation_debug.jsonl", seed=args.seed,
                                mode=args.generation_debug, device=device, torch_module=torch)
        atomic_json(run_dir / "generation_environment.json", debug.environment())
    control = debug
    if control is None and args.rng_mode == "isolated":
        from generation_debug import generation_environment
        control = PhaseRNG(seed=args.seed, device=device, torch_module=torch)
        atomic_json(run_dir / "generation_environment.json", generation_environment(torch, control.devices))
    with control.phase("model_load", -1) if control is not None else nullcontext():
        pipeline = VMemPipeline(config, device)
    pipeline.generation_debug = debug
    pipeline.generation_control = control
    runtime["pipeline"] = pipeline
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - started_at
    runtime["phase"] = "checkpoint_hashing"
    provenance["checkpoint_sha256"] = _checkpoint_hashes(pipeline)
    with (run_dir / "run_spec.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe({
            "arguments": vars(args), "actions": actions, "provenance": provenance,
            "execution": execution_settings(vars(args)),
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "resource_schema": RESOURCE_SCHEMA if args.resource_trace else None,
        }), handle, indent=2)
    pipeline.configure_memory_budget(
        policy=args.memory_policy,
        budget=args.memory_budget,
        scope=args.memory_scope,
    )
    pipeline.configure_surfel_reconstruction(window=args.surfel_reconstruction_window)
    pipeline.configure_frame_storage(args.frame_storage)

    navigator = Navigator(
        pipeline,
        step_size=args.step_size,
        num_interpolation_frames=args.frames_per_action,
        retain_frame_history=args.frame_storage != "resident",
    )
    initial_image = _load_vmem_image(args.image, config=config, device=device)
    initial_pose = np.eye(4, dtype=np.float32)
    initial_K = np.array(get_default_intrinsics()[0])
    action_records = []
    frame_hashes = []
    resume_info = None
    rng_to_restore = None
    run_identity = identity(vars(args), provenance, torch)
    if args.resume_from is not None:
        runtime["phase"] = "recovery_load"
        state, state_hash = load_checkpoint(args.resume_from, run_identity, torch, device)
        completed_actions = state["completed_actions"]
        action_records, frame_hashes, rng_to_restore = restore_into_new_attempt(
            args.resume_from, run_dir, state, pipeline, navigator,
        )
        resume_info = {"parent_run": str(args.resume_from.resolve()), "checkpoint_sha256": state_hash,
                       "completed_actions": completed_actions, "inherited_frames": len(frame_hashes)}
        del state
    start_action = len(action_records)
    if start_action > args.num_actions:
        raise ValueError("Recovery checkpoint exceeds the planned action count")
    spec_path = run_dir / "run_spec.json"
    spec = json.loads(spec_path.read_text())
    spec["recovery"] = {"schema": RECOVERY_SCHEMA, "checkpoint_every": args.checkpoint_every,
                        "resume": resume_info, "durable_frames": True}
    atomic_json(spec_path, spec)
    if args.resource_trace:
        from modeling.resource_audit import ResourceProfiler
        pipeline.resource_profiler = ResourceProfiler(
            run_dir / "resource_trace.jsonl", device=device, torch_module=torch,
            fps=args.fps, warmup_steps=args.profile_warmup_steps,
            session_start_step=start_action, session_id=run_dir.name,
        )
        pipeline.resource_profiler.extra_components = lambda: {
            "navigator_frames": navigator.frames, "navigator_poses": navigator.pose_history,
            "navigator_current_pose": navigator.current_pose, "navigator_current_K": navigator.current_K,
            "runner_action_records": action_records, "runner_planned_actions": actions,
            "runner_initial_image": initial_image,
            "runner_frame_hashes": frame_hashes,
        }
    runtime["phase"] = "initialization"
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    initialization_started = time.perf_counter()
    if args.resume_from is None:
        with control.phase("initialization", -1) if control is not None else nullcontext():
            if debug is not None:
                debug.record("initial_input", -1, image=initial_image, pose=initial_pose, K=initial_K)
            navigator.initialize(initial_image, initial_pose, initial_K)
            if debug is not None:
                debug.record("initial_encoding", -1, latents=pipeline.latents,
                             embeddings=pipeline.encoder_embeddings, poses=pipeline.c2ws, Ks=pipeline.Ks)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    initialization_seconds = time.perf_counter() - initialization_started
    if args.frame_storage == "resident":
        initial_image = None
    frame_save_seconds = save_new_frames(run_dir, pipeline.pil_frames, frame_hashes)
    release_evicted_payloads(pipeline, len(frame_hashes))
    if args.resource_trace:
        pipeline.resource_profiler.initial_snapshot(pipeline)
    if rng_to_restore is not None:
        restore_rng(rng_to_restore, torch, device)
        del rng_to_restore
    if args.checkpoint_every:
        runtime["phase"] = "checkpoint"
        append_event(run_dir, save_checkpoint(run_dir, pipeline, navigator, action_records, frame_hashes, run_identity, torch))
    if resume_info is not None:
        append_event(run_dir, {"event": "resume", **resume_info})
    append_event(run_dir, {"event": "frame_save", "completed_actions": start_action,
                           "frame_count": len(frame_hashes), "frame_save_seconds": frame_save_seconds})

    runtime["phase"] = "generation"
    autocast_context = (
        torch.autocast("cuda") if device.type == "cuda" else nullcontext()
    )
    with torch.no_grad(), autocast_context:
        for action_index in range(start_action, len(actions)):
            action = actions[action_index]
            runtime["phase"] = "generation"
            print(
                f"[{action_index + 1}/{len(actions)}] action={action} "
                f"total_frames={len(pipeline.pil_frames)} eligible_frames={len(pipeline.get_allowed_memory_indices())} "
                f"frame_storage={args.frame_storage}",
                flush=True,
            )
            action_started_at = time.perf_counter()
            previous_frame_count = len(pipeline.pil_frames)
            # Do not retain the returned batch across action boundaries.
            _apply_action(navigator, action)
            action_records.append(
                {
                    "action_index": action_index,
                    "action": action,
                    "num_generated_frames": len(pipeline.pil_frames) - previous_frame_count,
                    "total_frames_after": len(pipeline.pil_frames),
                    "current_pose": navigator.current_pose.tolist(),
                    "wall_seconds": time.perf_counter() - action_started_at,
                }
            )
            with (run_dir / "actions.partial.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(action_records[-1]) + "\n")
            runtime["phase"] = "frame_save"
            frame_save_seconds = save_new_frames(run_dir, pipeline.pil_frames, frame_hashes)
            append_event(run_dir, {"event": "frame_save", "completed_actions": len(action_records),
                                   "frame_count": len(frame_hashes), "frame_save_seconds": frame_save_seconds})
            if args.frame_storage == "resident":
                runtime["phase"] = "payload_eviction"
                if navigator.frames:
                    raise RuntimeError("Navigator retained frame aliases in resident mode")
                if args.resource_trace:
                    pipeline.resource_profiler.begin_phase("payload_eviction")
                release_evicted_payloads(pipeline, len(frame_hashes))
                if args.resource_trace:
                    pipeline.resource_profiler.end_phase("payload_eviction")
                    pipeline.resource_profiler.finish_step(pipeline, storage_ready=True)
            pause = (args.stop_after_actions is not None
                     and len(action_records) - start_action >= args.stop_after_actions
                     and len(action_records) < len(actions))
            if args.checkpoint_every and (len(action_records) % args.checkpoint_every == 0 or pause
                                          or len(action_records) == len(actions)):
                runtime["phase"] = "checkpoint"
                append_event(run_dir, save_checkpoint(run_dir, pipeline, navigator, action_records, frame_hashes, run_identity, torch))
            atomic_json(run_dir / "run_status.json", {
                "status": "paused" if pause else "running", "pid": os.getpid(),
                "completed_actions": len(action_records), "durable_frames": len(frame_hashes),
                "updated_at": datetime.now().isoformat(),
            })
            if pause:
                print(json.dumps({"run_dir": str(run_dir), "status": "paused",
                                  "completed_actions": len(action_records)}, indent=2), flush=True)
                return

    runtime["phase"] = "export"
    generated_video_path = run_dir / "generated.mp4"
    atomic_json(run_dir / "frame_manifest.json", {"schema": "vmem_output_frames_v1", "sha256": frame_hashes})
    _save_pil_video(iter_saved_frames(run_dir, frame_hashes), generated_video_path, fps=args.fps)

    actions_path = run_dir / "actions.json"
    with actions_path.open("w", encoding="utf-8") as handle:
        json.dump(action_records, handle, indent=2)

    retrieval_trace_path = run_dir / "retrieval_trace.json"
    memory_trace_path = run_dir / "memory_trace.json"
    pipeline.save_retrieval_trace(str(retrieval_trace_path))
    pipeline.save_memory_trace(str(memory_trace_path))

    metadata = {
        "image": args.image,
        "run_id": args.run_id,
        "seed": args.seed,
        "generation_debug": args.generation_debug,
        "clip_attention": args.clip_attention,
        "cut3r_attention": args.cut3r_attention,
        "rng_mode": args.rng_mode,
        "execution": execution_settings(vars(args)),
        "provenance": provenance,
        "model_load_seconds": model_load_seconds,
        "initialization_seconds": initialization_seconds,
        "recovery": spec["recovery"],
        "wall_time_scope": "current process only, including load, recovery, checkpoint I/O and export",
        "resource_trace": run_dir / "resource_trace.jsonl" if args.resource_trace else None,
        "profile_warmup_steps": args.profile_warmup_steps,
        "generation_config": run_dir / "generation_config.yaml",
        "wall_seconds_including_load_and_export": time.perf_counter() - started_at,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "cuda_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
        "fps": args.fps,
        "num_actions": args.num_actions,
        "trajectory": args.trajectory,
        "pattern": args.pattern,
        "action_histogram": _action_histogram(actions),
        "expanded_action_prefix": actions[: min(40, len(actions))],
        "expected_frames": expected_frames,
        "actual_frames": len(pipeline.pil_frames),
        "actual_seconds": len(pipeline.pil_frames) / args.fps,
        "step_size": args.step_size,
        "frames_per_action": args.frames_per_action,
        "memory_policy": args.memory_policy,
        "memory_budget": args.memory_budget,
        "memory_scope": pipeline.memory_scope,
        "frame_storage": args.frame_storage,
        "frame_payloads": validate_resident_payloads(pipeline),
        "generated_video": generated_video_path,
        "actions": actions_path,
        "retrieval_trace": retrieval_trace_path,
        "memory_trace": memory_trace_path,
        "config_overrides": {
            "clip_attention": args.clip_attention,
            "cut3r_attention": args.cut3r_attention,
            "rng_mode": args.rng_mode,
            "inference_steps": args.inference_steps,
            "surfel_niter": args.surfel_niter,
            "surfel_reconstruction_window": args.surfel_reconstruction_window,
            "visualize_intermediates": args.visualize_intermediates,
        },
    }
    metadata_path = run_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(metadata), handle, indent=2)
    atomic_json(run_dir / "run_status.json", {"status": "complete", "pid": os.getpid()})

    print(json.dumps(_json_safe({
        "run_dir": run_dir,
        "generated_video": generated_video_path,
        "metadata": metadata_path,
        "actions": actions_path,
        "retrieval_trace": retrieval_trace_path,
        "memory_trace": memory_trace_path,
    }), indent=2))


if __name__ == "__main__":
    main()
