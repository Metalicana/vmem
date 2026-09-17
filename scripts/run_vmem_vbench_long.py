"""VBench-Long entry point with the MemCam/WorldMem custom-input grouping fix.

The upstream checkout and metric definitions are not modified. Aggregation back
to each staged video is corrected, and a video-only PyAV writer is supplied when
TorchVision no longer exports write_video. Semantic splitting is excluded.
"""

import argparse
from collections import defaultdict
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import random
import runpy
import sys
import tempfile


DIMENSIONS = (
    "aesthetic_quality", "imaging_quality", "subject_consistency",
    "background_consistency", "motion_smoothness", "dynamic_degree",
)


def write_video_pyav(filename, video_array, fps, video_codec="libx264", options=None):
    """Video-only writer for VBench's integer-fps RGB clips.

    Uses the libx264/yuv420p defaults of torchvision 0.23's PyAV writer.
    Audio, fractional fps and other codecs are deliberately unsupported.
    """
    rate = float(fps)
    if not math.isfinite(rate) or rate <= 0 or not rate.is_integer():
        raise ValueError("VMem's video writer requires positive integer fps")
    if video_codec != "libx264":
        raise ValueError("VMem's video writer supports only libx264")
    import av
    from av.video.frame import PictureType
    import torch

    frames = torch.as_tensor(video_array, dtype=torch.uint8).detach().cpu().numpy()
    if frames.ndim != 4 or frames.shape[-1] != 3 or any(size <= 0 for size in frames.shape):
        raise ValueError("Expected nonempty RGB video with shape [T, H, W, 3]")
    _, height, width, _ = frames.shape
    if height % 2 or width % 2:
        raise ValueError("yuv420p encoding requires even frame dimensions")
    with av.open(str(filename), mode="w") as container:
        stream = container.add_stream(video_codec, rate=int(rate))
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.options = dict(options or {})
        for pixels in frames:
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pict_type = PictureType.NONE
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def install_video_writer():
    """Patch only the current evaluator process, and only a missing API."""
    import torchvision
    from torchvision import io
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("VBench video encoding needs PyAV. In the vbench environment, run: python -m pip install av") from exc
    backend = "torchvision.io.write_video"
    if not callable(getattr(io, "write_video", None)):
        io.write_video = write_video_pyav
        backend = "vmem_pyav_video_only_v1"
    elif io.write_video is write_video_pyav:
        backend = "vmem_pyav_video_only_v1"
    return {"video_writer": backend, "torchvision": torchvision.__version__,
            "pyav": av.__version__, "ffmpeg_libraries": av.library_versions}


def check_video_writer():
    """Exercise the selected writer and RGB decoding on a tiny CPU-only clip."""
    import av
    import numpy as np
    from torchvision.io import write_video

    frames = np.empty((26, 16, 16, 3), dtype=np.uint8)
    frames[:, :, :, 0] = np.arange(26, dtype=np.uint8)[:, None, None] * 7 + 32
    frames[:, :, :, 1] = 96
    frames[:, :, :, 2] = 16
    with tempfile.TemporaryDirectory(prefix="vmem-vbench-writer-") as tmp:
        path = Path(tmp) / "check.mp4"
        write_video(str(path), frames, fps=13)
        with av.open(str(path)) as container:
            fps = float(container.streams.video[0].average_rate)
            decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    if len(decoded) != 26 or fps != 13 or np.asarray(decoded).shape != frames.shape:
        raise ValueError("Video writer preflight failed frame-count/fps/shape checks")
    # Lossy H.264 may change values slightly, but not channels or frame order.
    if np.max(np.abs(np.asarray(decoded, dtype=np.int16) - frames.astype(np.int16))) > 12:
        raise ValueError("Video writer preflight failed RGB/frame-order checks")
    return {"frames": 26, "fps": 13, "height": 16, "width": 16, "status": "passed"}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def source_video(clip_path):
    clip = Path(clip_path)
    stem, separator, index = clip.stem.rpartition("_")
    if (clip.parent.parent.name != "split_clip" or not separator
            or not index.isascii() or not index.isdigit() or stem != clip.parent.name):
        raise ValueError(f"Unsupported VBench-Long clip layout: {clip}")
    video = clip.parent.parent.parent / (stem + ".mp4")
    if not video.is_file():
        raise ValueError(f"Missing staged source video: {video}")
    return str(video)


def reorganize_clips_results(detailed_results, dimension=None):
    if dimension is not None and dimension not in DIMENSIONS:
        raise ValueError(f"Unsupported dimension: {dimension}")
    if not detailed_results:
        raise ValueError("Cannot aggregate empty VBench-Long clip results")
    grouped, seen = defaultdict(list), set()
    for result in detailed_results:
        path, score = result["video_path"], result["video_results"]
        if path in seen:
            raise ValueError(f"Duplicate clip result: {path}")
        seen.add(path)
        if not isinstance(score, bool) and not finite(score):
            raise ValueError(f"Invalid clip score: {path}: {score}")
        grouped[source_video(path)].append(score)
    # Preserve the siblings' equal-clip means and upstream imaging scaling.
    videos = [{"video_path": path, "video_results": sum(scores) / len(scores)}
              for path, scores in grouped.items()]
    overall = sum(row["video_results"] for row in videos) / len(videos)
    if dimension == "imaging_quality":
        overall /= 100
    return overall, detailed_results, videos


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--vbench-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Check imports and CPU video encoding, without loading scoring models.")
    parser.add_argument("--check-report", type=Path)
    args, upstream_args = parser.parse_known_args()
    check = argparse.ArgumentParser(add_help=False)
    check.add_argument("--mode", required=not args.check, choices=["long_custom_input"])
    check.add_argument("--dimension", nargs="+", required=not args.check, choices=DIMENSIONS, default=list(DIMENSIONS))
    check.add_argument("--use_semantic_splitting", action="store_true")
    check.add_argument("--static_filter_flag", action="store_true")
    options, _ = check.parse_known_args(upstream_args)
    if options.use_semantic_splitting or options.static_filter_flag:
        check.error("The grouping adapter does not support semantic/static filtering")
    root = args.vbench_root.resolve()
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    runtime = install_video_writer()
    print(json.dumps(runtime), flush=True)
    from vbench2_beta_long import utils

    if args.check:
        for dimension in options.dimension:
            importlib.import_module(f"vbench2_beta_long.{dimension}")
        runtime.update(status="passed", dimensions=options.dimension, torch=str(torch.__version__),
                       cuda=torch.version.cuda, writer_check=check_video_writer())
        if args.check_report is not None:
            args.check_report.write_text(json.dumps(runtime, indent=2) + "\n")
        print(json.dumps(runtime, indent=2), flush=True)
        return

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    original_hash = hashlib.sha256(inspect.getsource(utils.reorganize_clips_results).encode()).hexdigest()
    print(f"VMem custom-input grouping adapter; upstream function SHA256: {original_hash}", flush=True)
    print(f"torch={torch.__version__}, CUDA={torch.version.cuda}", flush=True)
    utils.reorganize_clips_results = reorganize_clips_results
    entry = root / "vbench2_beta_long/eval_long.py"
    sys.argv = [str(entry), *upstream_args]
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
