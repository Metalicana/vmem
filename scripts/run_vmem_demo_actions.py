#!/usr/bin/env python
"""Run VMem by chaining the same small actions used by the Gradio demo."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
import json
from pathlib import Path
import random
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MEMORY_POLICIES = (
    "unbounded",
    "fifo",
    "rarity_irreplaceability",
    "slam_covisibility",
)
BUDGETED_MEMORY_POLICIES = (
    "fifo",
    "rarity_irreplaceability",
    "slam_covisibility",
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


def _save_pil_video(frames: Sequence, path: Path, *, fps: float) -> None:
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
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--visualize-intermediates", action="store_true")
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
    if args.frames_per_action <= 0:
        raise ValueError("--frames-per-action must be positive")
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
                },
                indent=2,
            )
        )
        return

    _load_runtime_dependencies()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = args.output_root / _run_name(args, num_actions=args.num_actions)
    run_dir.mkdir(parents=True, exist_ok=False)

    config = OmegaConf.load(args.config)
    if args.inference_steps is not None:
        config.model.inference_num_steps = args.inference_steps
    if args.surfel_niter is not None:
        config.surfel.niter = args.surfel_niter
    config.visualization_dir = str(run_dir / "visualization")
    config.model.samples_dir = str(run_dir / "visualization")
    config.inference.visualize = bool(args.visualize_intermediates)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pipeline = VMemPipeline(config, device)
    pipeline.configure_memory_budget(
        policy=args.memory_policy,
        budget=args.memory_budget,
        scope=args.memory_scope,
    )
    pipeline.configure_surfel_reconstruction(window=args.surfel_reconstruction_window)

    navigator = Navigator(
        pipeline,
        step_size=args.step_size,
        num_interpolation_frames=args.frames_per_action,
    )
    initial_image = _load_vmem_image(args.image, config=config, device=device)
    initial_pose = np.eye(4, dtype=np.float32)
    initial_K = np.array(get_default_intrinsics()[0])
    navigator.initialize(initial_image, initial_pose, initial_K)

    action_records = []
    autocast_context = (
        torch.autocast("cuda") if device.type == "cuda" else nullcontext()
    )
    with torch.no_grad(), autocast_context:
        for action_index, action in enumerate(actions):
            print(
                f"[{action_index + 1}/{len(actions)}] action={action} "
                f"stored_frames={len(pipeline.pil_frames)}",
                flush=True,
            )
            frames = _apply_action(navigator, action)
            action_records.append(
                {
                    "action_index": action_index,
                    "action": action,
                    "num_generated_frames": 0 if frames is None else len(frames),
                    "total_frames_after": len(pipeline.pil_frames),
                    "current_pose": navigator.current_pose.tolist(),
                }
            )

    generated_video_path = run_dir / "generated.mp4"
    _save_pil_video(pipeline.pil_frames, generated_video_path, fps=args.fps)

    if args.save_frames:
        frame_dir = run_dir / "generated_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx, frame in enumerate(pipeline.pil_frames):
            frame.save(frame_dir / f"{frame_idx:04d}.png")

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
        "generated_video": generated_video_path,
        "actions": actions_path,
        "retrieval_trace": retrieval_trace_path,
        "memory_trace": memory_trace_path,
        "config_overrides": {
            "inference_steps": args.inference_steps,
            "surfel_niter": args.surfel_niter,
            "surfel_reconstruction_window": args.surfel_reconstruction_window,
            "visualize_intermediates": args.visualize_intermediates,
        },
    }
    metadata_path = run_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(metadata), handle, indent=2)

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
