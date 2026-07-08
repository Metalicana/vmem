#!/usr/bin/env python
"""Run vanilla VMem from one initial image along a scripted camera path."""

from __future__ import annotations

import argparse
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


def _load_runtime_dependencies() -> None:
    global OmegaConf
    global VMemPipeline
    global get_default_intrinsics
    global load_img_and_K
    global np
    global tensor_to_pil
    global torch
    global transform_img_and_K

    import numpy as np
    from omegaconf import OmegaConf
    import torch

    from modeling.pipeline import VMemPipeline
    from utils import get_default_intrinsics, load_img_and_K, tensor_to_pil, transform_img_and_K


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


def _yaw_matrix(degrees: float) -> np.ndarray:
    angle = np.radians(float(degrees))
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def _pose(position, *, yaw_degrees: float = 0.0) -> np.ndarray:
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = _yaw_matrix(yaw_degrees)
    c2w[:3, 3] = np.asarray(position, dtype=np.float32)
    return c2w


def _build_forward(num_frames: int, *, step_size: float) -> list[np.ndarray]:
    return [
        _pose((0.0, 0.0, -step_size * frame_idx))
        for frame_idx in range(num_frames)
    ]


def _build_out_and_back(num_frames: int, *, step_size: float) -> list[np.ndarray]:
    if num_frames <= 1:
        return [_pose((0.0, 0.0, 0.0))]
    midpoint = (num_frames - 1) / 2.0
    c2ws = []
    for frame_idx in range(num_frames):
        distance = step_size * (midpoint - abs(frame_idx - midpoint))
        yaw = 0.0 if frame_idx <= midpoint else 180.0
        c2ws.append(_pose((0.0, 0.0, -distance), yaw_degrees=yaw))
    return c2ws


def _build_square_loop(num_frames: int, *, step_size: float) -> list[np.ndarray]:
    if num_frames <= 1:
        return [_pose((0.0, 0.0, 0.0))]
    side = max(step_size * (num_frames - 1) / 4.0, step_size)
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [side, 0.0, 0.0],
            [side, 0.0, -side],
            [0.0, 0.0, -side],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    yaws = [90.0, 180.0, -90.0, 0.0]
    c2ws = []
    for frame_idx in range(num_frames):
        progress = frame_idx / max(num_frames - 1, 1) * 4.0
        segment = min(int(progress), 3)
        local = progress - segment
        position = (1.0 - local) * corners[segment] + local * corners[segment + 1]
        c2ws.append(_pose(position, yaw_degrees=yaws[segment]))
    return c2ws


def _build_arc(num_frames: int, *, step_size: float, yaw_degrees: float) -> list[np.ndarray]:
    if num_frames <= 1:
        return [_pose((0.0, 0.0, 0.0))]
    c2ws = []
    radius = max(step_size * num_frames / 2.0, step_size)
    for frame_idx in range(num_frames):
        t = frame_idx / max(num_frames - 1, 1)
        yaw = yaw_degrees * t
        angle = np.radians(yaw)
        x = radius * np.sin(angle)
        z = -radius * (1.0 - np.cos(angle))
        c2ws.append(_pose((x, 0.0, z), yaw_degrees=yaw))
    return c2ws


def _build_trajectory(args) -> list[np.ndarray]:
    if args.trajectory == "forward":
        return _build_forward(args.num_frames, step_size=args.step_size)
    if args.trajectory == "out_and_back":
        return _build_out_and_back(args.num_frames, step_size=args.step_size)
    if args.trajectory == "square_loop":
        return _build_square_loop(args.num_frames, step_size=args.step_size)
    if args.trajectory == "arc":
        return _build_arc(
            args.num_frames,
            step_size=args.step_size,
            yaw_degrees=args.yaw_degrees,
        )
    raise ValueError(f"Unsupported trajectory: {args.trajectory}")


def _run_name(args) -> str:
    image_stem = args.image.stem.replace(" ", "_")
    policy = args.memory_policy
    if args.memory_policy in BUDGETED_MEMORY_POLICIES:
        policy = f"{policy}_B{args.memory_budget}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{image_stem}_{args.trajectory}_N{args.num_frames}_{policy}_{timestamp}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/manual_vmem"))
    parser.add_argument("--config", type=Path, default=Path("configs/inference/inference.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=13.0)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=17,
        help="Total frames including the initial image.",
    )
    parser.add_argument(
        "--trajectory",
        choices=("forward", "arc", "out_and_back", "square_loop"),
        default="arc",
    )
    parser.add_argument("--step-size", type=float, default=0.035)
    parser.add_argument("--yaw-degrees", type=float, default=35.0)
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

    if args.num_frames < 2:
        raise ValueError("--num-frames must include the initial image plus at least one target frame")
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
    if args.dry_run:
        print(
            json.dumps(
                {
                    "image": str(args.image),
                    "output_root": str(args.output_root),
                    "num_frames": args.num_frames,
                    "trajectory": args.trajectory,
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

    run_dir = args.output_root / _run_name(args)
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

    c2ws = _build_trajectory(args)
    K = get_default_intrinsics()[0].detach().cpu().numpy()
    Ks = [K for _ in c2ws]

    initial_image = _load_vmem_image(args.image, config=config, device=device)
    pipeline.initialize(initial_image, c2ws[0], Ks[0])

    autocast_context = (
        torch.autocast("cuda") if device.type == "cuda" else nullcontext()
    )
    with torch.no_grad(), autocast_context:
        pipeline.generate_trajectory_frames(c2ws[1:], Ks[1:])

    frames = pipeline.pil_frames[: args.num_frames]
    generated_video_path = run_dir / "generated.mp4"
    _save_pil_video(frames, generated_video_path, fps=args.fps)

    if args.save_frames:
        frame_dir = run_dir / "generated_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx, frame in enumerate(frames):
            frame.save(frame_dir / f"{frame_idx:04d}.png")

    trajectory_path = run_dir / "trajectory.json"
    with trajectory_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "c2ws": [c2w.tolist() for c2w in c2ws],
                "Ks": [K.tolist() for K in Ks],
            },
            handle,
            indent=2,
        )

    retrieval_trace_path = run_dir / "retrieval_trace.json"
    memory_trace_path = run_dir / "memory_trace.json"
    pipeline.save_retrieval_trace(str(retrieval_trace_path))
    pipeline.save_memory_trace(str(memory_trace_path))

    metadata = {
        "image": args.image,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "trajectory": args.trajectory,
        "step_size": args.step_size,
        "yaw_degrees": args.yaw_degrees,
        "memory_policy": args.memory_policy,
        "memory_budget": args.memory_budget,
        "memory_scope": pipeline.memory_scope,
        "generated_video": generated_video_path,
        "trajectory_path": trajectory_path,
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
        "trajectory": trajectory_path,
        "retrieval_trace": retrieval_trace_path,
        "memory_trace": memory_trace_path,
    }), indent=2))


if __name__ == "__main__":
    main()
