"""VBench-Long entry point with the MemCam/WorldMem custom-input grouping fix.

The upstream checkout and metric definitions are not modified. Only aggregation
back to each original staged video is corrected; semantic splitting is excluded.
"""

import argparse
from collections import defaultdict
import hashlib
import inspect
import math
from pathlib import Path
import random
import runpy
import sys


DIMENSIONS = (
    "aesthetic_quality", "imaging_quality", "subject_consistency",
    "background_consistency", "motion_smoothness", "dynamic_degree",
)


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
    args, upstream_args = parser.parse_known_args()
    check = argparse.ArgumentParser(add_help=False)
    check.add_argument("--mode", required=True, choices=["long_custom_input"])
    check.add_argument("--dimension", nargs="+", required=True, choices=DIMENSIONS)
    check.add_argument("--use_semantic_splitting", action="store_true")
    check.add_argument("--static_filter_flag", action="store_true")
    options, _ = check.parse_known_args(upstream_args)
    if options.use_semantic_splitting or options.static_filter_flag:
        check.error("The grouping adapter does not support semantic/static filtering")
    root = args.vbench_root.resolve()
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from vbench2_beta_long import utils

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
