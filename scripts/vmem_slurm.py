"""Allocation-aware GPU checks and a fail-closed entry point for GPU children."""

import argparse
import importlib
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]


def allocation(environment=None):
    environment = os.environ if environment is None else environment
    mask = environment.get("CUDA_VISIBLE_DEVICES", "")
    if not environment.get("SLURM_JOB_ID") or not mask or mask == "-1":
        raise ValueError("inherit requires a Slurm allocation and its nonempty CUDA_VISIBLE_DEVICES")
    return {"hostname": socket.gethostname(), **{key: environment.get(key) for key in (
        "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES")}}


def child_environment(gpu, enabled, environment=None):
    result = dict(os.environ if environment is None else environment)
    if enabled and gpu == "inherit":
        allocation(result)
        # The exact scheduler mask is authoritative, including cgroup remapping.
    else:
        result["CUDA_VISIBLE_DEVICES"] = str(gpu) if enabled else ""
    return result


def environment_identity():
    from importlib.metadata import distributions
    return {"python_version": sys.version, "packages": sorted(
        [d.metadata["Name"], d.version] for d in distributions() if d.metadata.get("Name"))}


def check_launch(action, gpu, environment=None):
    environment = os.environ if environment is None else environment
    if action not in {"run", "start"}:
        return
    if environment.get("SLURM_JOB_ID"):
        if action != "run" or gpu != "inherit":
            raise ValueError("Under Slurm use foreground 'run --gpu inherit', never detached start/physical selection")
    if gpu == "inherit":
        allocation(environment)


def probe():
    record = allocation()
    import torch
    torch.cuda.init()
    if torch.cuda.device_count() != 1:
        raise ValueError("Expected exactly one visible allocated GPU")
    torch.cuda.set_device(0)
    x = torch.ones((256, 256), device="cuda:0")
    y = x @ x
    torch.cuda.synchronize()
    if y[0, 0].item() != 256:
        raise RuntimeError("CUDA matrix multiplication failed")
    properties = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    return {**record, "status": "passed", "device": "cuda:0", "name": properties.name,
            "uuid": str(properties.uuid) if hasattr(properties, "uuid") else None,
            "capability": [properties.major, properties.minor],
            "free_mib": free // 2**20, "total_mib": total // 2**20,
            "torch": torch.__version__, "cuda_build": torch.version.cuda,
            "python": sys.executable, "scope": "CUDA logical device 0 within inherited Slurm mask"}


def query_allocated_gpu():
    allocation()
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--gpu-only"],
                            capture_output=True, text=True, check=True, timeout=180)
    return json.loads(result.stdout)


def check_capacity(snapshot, minimum):
    if minimum > snapshot["total_mib"]:
        raise ValueError("--min-free-mib exceeds allocated GPU capacity")
    if minimum > snapshot["free_mib"]:
        raise ValueError("Allocated GPU has insufficient free memory at admission; no other job will be killed")


def model_smoke():
    import torch
    import numpy as np
    from PIL import Image
    import imageio.v2 as imageio
    import curope
    tokens = torch.ones((1, 4, 2, 16), device="cuda:0")
    positions = torch.arange(8, device="cuda:0", dtype=torch.int64).reshape(1, 4, 2)
    curope.rope_2d(tokens, positions, 100.0, 1.0)
    if not torch.isfinite(tokens).all() or torch.equal(tokens, torch.ones_like(tokens)):
        raise RuntimeError("curope forward kernel failed")
    curope.rope_2d(tokens, positions, 100.0, -1.0)
    torch.cuda.synchronize()
    torch.testing.assert_close(tokens, torch.ones_like(tokens), atol=1e-5, rtol=1e-5)
    for name in ("modeling.pipeline", "navigation", "open_clip", "diffusers"):
        importlib.import_module(name)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        frame = np.full((32, 32, 3), 128, dtype=np.uint8)
        Image.fromarray(frame).save(path / "test.png")
        with Image.open(path / "test.png") as decoded:
            np.testing.assert_array_equal(np.asarray(decoded), frame)
        imageio.mimwrite(path / "test.mp4", [frame] * 4, fps=13)
        reader = imageio.get_reader(path / "test.mp4")
        try:
            if reader.count_frames() != 4 or reader.get_data(0).shape != frame.shape:
                raise RuntimeError("Video I/O check failed")
        finally:
            reader.close()
    return {"curope_cuda_forward_inverse": "passed", "vmem_cut3r_imports": "passed", "image_video_io": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    result = probe()
    if args.execute:
        target = Path(args.execute[0]).resolve()
        if target.parent != REPO / "scripts" or target == Path(__file__).resolve():
            raise ValueError("Only repository runner scripts may be executed")
        print(json.dumps({"allocated_gpu": result}), flush=True)
        sys.argv = [str(target), *args.execute[1:]]
        runpy.run_path(str(target), run_name="__main__")
        return
    if not args.gpu_only:
        result.update(model_smoke())
    if args.output:
        from run_vmem_results import save
        save(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
