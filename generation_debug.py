"""Opt-in short-run fingerprints and phase-local RNG; no model imports.

Observation never reseeds. Isolation is an experimental control, not a promise
of deterministic CUDA kernels. Hashing copies tensors to CPU and affects timing.
"""

from contextlib import contextmanager, nullcontext
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys

import numpy as np
from generation_rng import PhaseRNG, RNG_SCHEMA, phase_seed


SCHEMA = "vmem_generation_debug_v1"


def validate_debug_args(args):
    rng_mode = getattr(args, "rng_mode", "legacy")
    if rng_mode not in ("legacy", "isolated"):
        raise ValueError("Unknown RNG mode")
    if args.generation_debug is not None and rng_mode != "legacy":
        raise ValueError("Use --generation-debug isolated OR --rng-mode isolated, not both")
    for name in ("clip_attention", "cut3r_attention"):
        if getattr(args, name, "native") not in ("native", "math"):
            raise ValueError(f"Unknown attention profile: {name}")
        if getattr(args, name, "native") != "native" and args.generation_debug is None and rng_mode != "isolated":
            raise ValueError(f"--{name.replace('_', '-')} math requires --rng-mode isolated or a short --generation-debug run")
    if args.generation_debug is None:
        return
    if args.experiment_lock or args.resume_from or args.checkpoint_every:
        raise ValueError("Generation debug requires a fresh unlocked run with --checkpoint-every 0; no resume")
    if args.frames_per_action != 4 or not 1 <= args.num_actions <= 12:
        raise ValueError("Generation debug is limited to 1-12 four-frame actions")


class GenerationDebug:
    def __init__(self, path, *, seed, mode, device, torch_module):
        if mode not in {"observe", "isolated"}:
            raise ValueError(f"Unknown generation debug mode: {mode}")
        self.path = Path(path)
        self.seed, self.mode, self.torch = seed, mode, torch_module
        self.device = torch_module.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Generation debug supports CPU and one selected CUDA device")
        self.devices = []
        if self.device.type == "cuda":
            index = self.device.index
            self.devices = [torch_module.cuda.current_device() if index is None else index]
        self.phase_rng = PhaseRNG(seed=seed, device=device, torch_module=torch_module)
        # Refuse to append to another attempt's diagnostic.
        with self.path.open("x") as handle:
            handle.write(json.dumps({"event": "schema", "schema": SCHEMA, "mode": mode,
                                     "seed": seed, "rng_schema": RNG_SCHEMA if mode == "isolated" else "legacy",
                                     "timing_scope": "debug hashes synchronize/copy tensors; not benchmark timings"}) + "\n")

    def fingerprint(self, value):
        if isinstance(value, self.torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            raw = tensor.reshape(-1).view(self.torch.uint8).numpy().tobytes()
            return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                    "sha256": hashlib.sha256(raw).hexdigest()}
        if isinstance(value, np.ndarray):
            return {"shape": list(value.shape), "dtype": str(value.dtype),
                    "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
        if isinstance(value, dict):
            return {key: self.fingerprint(inner) for key, inner in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.fingerprint(inner) for inner in value]
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise TypeError(f"Unsupported debug value: {type(value).__name__}")

    def record(self, event, step, **values):
        row = {"event": event, "step": step, **self.fingerprint(values)}
        with self.path.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    def rng_state(self):
        return {"cpu": self.torch.get_rng_state(),
                "cuda": self.torch.cuda.get_rng_state(self.devices[0]) if self.devices else None}

    @contextmanager
    def phase(self, name, step):
        derived = phase_seed(self.seed, name, step) if self.mode == "isolated" else None
        with self.phase_rng.phase(name, step) if derived is not None else nullcontext():
            self.record("phase_start", step, phase=name, phase_seed=derived, rng=self.rng_state())
            try:
                yield
            except BaseException as error:
                self.record("phase_error", step, phase=name, error_type=type(error).__name__)
                raise
            else:
                self.record("phase_end", step, phase=name, rng=self.rng_state())

    def environment(self):
        return generation_environment(self.torch, self.devices)


def generation_environment(torch, devices):
    packages = {}
    for name in ("numpy", "Pillow", "torch", "torchvision", "diffusers", "transformers",
                 "open_clip_torch", "kornia", "huggingface_hub", "safetensors", "xformers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    cuda = None
    if devices:
        index = devices[0]
        cuda = {"logical_device": index, "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index))}
    return {"python": sys.version, "packages": packages, "torch_build": torch.__config__.show(),
            "cuda_version": torch.version.cuda, "selected_gpu": cuda,
            "env": {key: os.environ.get(key) for key in (
                "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED",
                "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")},
            "backends": {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                         "cudnn_benchmark": torch.backends.cudnn.benchmark,
                         "cudnn_deterministic": torch.backends.cudnn.deterministic,
                         "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                         "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                         "float32_matmul_precision": torch.get_float32_matmul_precision()}}
