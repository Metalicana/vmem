#!/usr/bin/env python
"""Run VMem on a Context-as-Memory trajectory with an optional FIFO memory budget.

This is the first experimental runner, not the final benchmark harness. It keeps
VMem's generation and retrieval structure intact, but can restrict which stored
view memories are eligible for context retrieval.
"""

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


DEFAULT_OUTPUT_ROOT = Path("/data/ab575577/vmem_context_memory_runs")


def _load_runtime_dependencies() -> None:
    global ContextMemoryDataset
    global OmegaConf
    global Rotation
    global VMemPipeline
    global get_default_intrinsics
    global load_img_and_K
    global np
    global tensor_to_pil
    global torch
    global transform_img_and_K

    import numpy as np
    from omegaconf import OmegaConf
    from scipy.spatial.transform import Rotation
    import torch

    from data_adapters import ContextMemoryDataset
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


def _load_vmem_image(path: Path, *, config, device) -> torch.Tensor:
    image, _ = load_img_and_K(str(path), None, K=None, device=device)
    image, _ = transform_img_and_K(
        image,
        (config.model.height, config.model.width),
        mode="crop",
        K=None,
    )
    return image


def _camera_to_c2w(
    camera,
    *,
    origin: np.ndarray,
    pose_scale: float,
    rotation_order: str,
    camera_convention: str,
) -> np.ndarray:
    raw_position = (np.array(camera.position, dtype=np.float64) - origin) * pose_scale
    raw_rotation = np.array(camera.rotation, dtype=np.float64)
    if camera_convention == "scipy_euler":
        position = raw_position
        rotation = Rotation.from_euler(
            rotation_order,
            raw_rotation,
            degrees=True,
        ).as_matrix()
    elif camera_convention == "unreal":
        # Context-as-Memory camera metadata appears to come from Unreal.
        # Unreal world: X forward, Y right, Z up.
        # VMem navigation convention: X right, Y up, camera forward is -Z.
        unreal_to_vmem = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        roll, pitch, yaw = raw_rotation
        unreal_rotation = Rotation.from_euler(
            "ZYX",
            [yaw, pitch, roll],
            degrees=True,
        ).as_matrix()
        position = unreal_to_vmem @ raw_position
        rotation = unreal_to_vmem @ unreal_rotation @ unreal_to_vmem.T
    else:
        raise ValueError(f"Unsupported camera convention: {camera_convention}")

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rotation.astype(np.float32)
    c2w[:3, 3] = position.astype(np.float32)
    return c2w


def _build_native_forward_trajectory(
    frame_indices: Sequence[int],
    *,
    native_step_size: float,
) -> list[np.ndarray]:
    c2ws = []
    for local_idx, _ in enumerate(frame_indices):
        c2w = np.eye(4, dtype=np.float32)
        c2w[2, 3] = -float(native_step_size) * local_idx
        c2ws.append(c2w)
    return c2ws


def _build_camera_trajectory(
    sequence: ContextMemorySequence,
    frame_indices: Sequence[int],
    *,
    pose_scale: float,
    rotation_order: str,
    camera_convention: str,
    native_step_size: float,
) -> list[np.ndarray]:
    if camera_convention == "native_forward":
        return _build_native_forward_trajectory(
            frame_indices,
            native_step_size=native_step_size,
        )

    origin = np.array(sequence.camera(frame_indices[0]).position, dtype=np.float64)
    return [
        _camera_to_c2w(
            sequence.camera(frame_index),
            origin=origin,
            pose_scale=pose_scale,
            rotation_order=rotation_order,
            camera_convention=camera_convention,
        )
        for frame_index in frame_indices
    ]


def _build_run_dir(args) -> Path:
    budget_part = "unbounded"
    if args.memory_policy == "fifo":
        budget_part = f"fifo_B{args.memory_budget}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (
        f"{args.scene}_start{args.start_frame:04d}_N{args.num_frames}_"
        f"H{args.chunk_size}_{budget_part}_{args.camera_convention}_{timestamp}"
    )
    return args.output_root / name


def _local_to_dataset_indices(local_indices: Sequence[int], frame_indices: Sequence[int]) -> list[int]:
    dataset_indices = []
    for local_index in local_indices:
        if 0 <= local_index < len(frame_indices):
            dataset_indices.append(frame_indices[local_index])
    return dataset_indices


def _annotate_retrieval_trace(records: Sequence[dict], frame_indices: Sequence[int]) -> None:
    for record in records:
        record["target_dataset_frame_indices"] = _local_to_dataset_indices(
            record["target_frame_indices"],
            frame_indices,
        )
        record["allowed_dataset_frame_indices"] = _local_to_dataset_indices(
            record["allowed_memory_indices"],
            frame_indices,
        )
        record["selected_dataset_frame_indices"] = _local_to_dataset_indices(
            record["selected_context_indices"],
            frame_indices,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        help="Path to Context-as-Memory-Dataset/Context-as-Memory-Dataset",
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=17,
        help="Total output frames including the anchor frame.",
    )
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--fps", type=float, default=13.0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="High-volume output root. Defaults to /data/ab575577.",
    )
    parser.add_argument(
        "--memory-policy",
        choices=("unbounded", "fifo"),
        default="unbounded",
    )
    parser.add_argument("--memory-budget", type=int)
    parser.add_argument("--config", type=Path, default=Path("configs/inference/inference.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--inference-steps",
        type=int,
        help="Override config.model.inference_num_steps for cheap smoke tests.",
    )
    parser.add_argument(
        "--surfel-niter",
        type=int,
        help="Override config.surfel.niter for cheap smoke tests.",
    )
    parser.add_argument(
        "--surfel-reconstruction-window",
        type=int,
        help=(
            "Fast visual mode: reconstruct surfels from only the most recent N "
            "generated/conditioning frames instead of all frames so far."
        ),
    )
    parser.add_argument(
        "--pose-scale",
        type=float,
        default=0.01,
        help="Scale applied to dataset positions after subtracting the anchor position.",
    )
    parser.add_argument(
        "--camera-convention",
        choices=("scipy_euler", "unreal", "native_forward"),
        default="scipy_euler",
        help=(
            "How to convert Context-as-Memory cameras to VMem poses. "
            "Use native_forward to ignore dataset poses for a VMem sanity check."
        ),
    )
    parser.add_argument(
        "--rotation-order",
        default="xyz",
        help="Euler order for --camera-convention scipy_euler.",
    )
    parser.add_argument(
        "--native-step-size",
        type=float,
        default=0.025,
        help="Per-frame forward distance for --camera-convention native_forward.",
    )
    parser.add_argument("--save-gt-video", action="store_true")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument(
        "--visualize-intermediates",
        action="store_true",
        help="Keep VMem's per-frame PNG/GIF visualization outputs enabled.",
    )
    args = parser.parse_args()

    if args.num_frames < 2:
        raise ValueError("--num-frames must include anchor plus at least one target frame")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.memory_policy == "fifo" and (args.memory_budget is None or args.memory_budget <= 0):
        raise ValueError("--memory-policy fifo requires --memory-budget")

    _load_runtime_dependencies()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = ContextMemoryDataset(args.root)
    sequence = dataset.sequence(args.scene)
    frame_indices = list(
        range(args.start_frame, args.start_frame + args.num_frames)
    )
    available_frame_indices = set(sequence.frame_indices)
    missing_frames = [
        frame_index
        for frame_index in frame_indices
        if frame_index not in available_frame_indices
    ]
    if missing_frames:
        raise ValueError(f"Requested frames do not exist: {missing_frames[:10]}")

    if (args.num_frames - 1) % args.chunk_size != 0:
        print(
            "Warning: num_frames - 1 is not divisible by chunk_size; final chunk will be partial/padded.",
            file=sys.stderr,
        )

    run_dir = _build_run_dir(args)
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
        scope="view_context",
    )
    pipeline.configure_surfel_reconstruction(window=args.surfel_reconstruction_window)

    c2ws = _build_camera_trajectory(
        sequence,
        frame_indices,
        pose_scale=args.pose_scale,
        rotation_order=args.rotation_order,
        camera_convention=args.camera_convention,
        native_step_size=args.native_step_size,
    )
    K = get_default_intrinsics()[0].detach().cpu().numpy()
    Ks = [K for _ in frame_indices]

    initial_image = _load_vmem_image(
        sequence.frame_path(frame_indices[0]),
        config=config,
        device=device,
    )
    pipeline.initialize(initial_image, c2ws[0], Ks[0])
    partial_trace_path = run_dir / "retrieval_trace.partial.json"

    autocast_context = (
        torch.autocast("cuda") if device.type == "cuda" else nullcontext()
    )
    with torch.no_grad(), autocast_context:
        for chunk_start in range(1, args.num_frames, args.chunk_size):
            chunk_end = min(args.num_frames, chunk_start + args.chunk_size)
            target_c2ws = c2ws[chunk_start:chunk_end]
            target_Ks = Ks[chunk_start:chunk_end]
            print(
                f"Generating frames {chunk_start}-{chunk_end - 1} / {args.num_frames - 1}",
                flush=True,
            )
            pipeline.generate_trajectory_frames(
                target_c2ws,
                target_Ks,
                use_non_maximum_suppression=None,
            )
            _annotate_retrieval_trace(pipeline.retrieval_trace, frame_indices)
            pipeline.save_retrieval_trace(str(partial_trace_path))

    generated_frames = pipeline.pil_frames[: args.num_frames]
    generated_video_path = run_dir / "generated.mp4"
    _save_pil_video(generated_frames, generated_video_path, fps=args.fps)

    gt_video_path = None
    if args.save_gt_video:
        gt_frames = [
            tensor_to_pil(
                _load_vmem_image(
                    sequence.frame_path(frame_index),
                    config=config,
                    device=device,
                )
            )
            for frame_index in frame_indices
        ]
        gt_video_path = run_dir / "ground_truth.mp4"
        _save_pil_video(gt_frames, gt_video_path, fps=args.fps)

    if args.save_frames:
        frame_dir = run_dir / "generated_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for local_idx, frame in enumerate(generated_frames):
            frame.save(frame_dir / f"{local_idx:04d}.png")

    _annotate_retrieval_trace(pipeline.retrieval_trace, frame_indices)

    trace_path = run_dir / "retrieval_trace.json"
    pipeline.save_retrieval_trace(str(trace_path))

    metadata = {
        "scene_id": args.scene,
        "frame_indices": frame_indices,
        "num_frames": args.num_frames,
        "chunk_size": args.chunk_size,
        "fps": args.fps,
        "memory_policy": args.memory_policy,
        "memory_budget": args.memory_budget,
        "memory_scope": pipeline.memory_scope,
        "memory_unit": "stored generated/conditioning view frame",
        "generated_video": generated_video_path,
        "ground_truth_video": gt_video_path,
        "retrieval_trace": trace_path,
        "camera_conversion": {
            "camera_convention": args.camera_convention,
            "pose_scale": args.pose_scale,
            "rotation_order": args.rotation_order,
            "native_step_size": args.native_step_size,
            "position_origin_frame": frame_indices[0],
            "position_origin": sequence.camera(frame_indices[0]).position,
            "assumption": (
                "scipy_euler uses raw dataset rotations as Rotation.from_euler(rotation_order). "
                "unreal treats rotations as roll,pitch,yaw and maps Unreal X/Y/Z to VMem -Z/X/Y. "
                "native_forward ignores dataset camera metadata."
            ),
        },
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
        "ground_truth_video": gt_video_path,
        "metadata": metadata_path,
        "retrieval_trace": trace_path,
    }), indent=2))


if __name__ == "__main__":
    main()
