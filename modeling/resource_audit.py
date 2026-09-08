"""Observational memory accounting and synchronized stage timing for VMem.

Component bytes are estimates, not RSS: Python sizes plus unique backing
storage and logical PIL pixels. Tensor/NumPy views and aliased frames are
deduplicated across components, in the reported component order.
"""

from functools import wraps
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image


SCHEMA_VERSION = "vmem_resources_v1"
HISTORY_COMPONENTS = (
    "pil_frames", "latents", "encoder_embeddings", "c2ws", "Ks",
    "surfel_depths", "surfel_Ks", "surfels", "surfel_to_timestep",
    "poses", "focal_lengths", "_dino_feature_cache",
    "memory_buffer", "retrieval_trace", "memory_events",
    "denoiser", "sampler", "_active_target_frame_indices",
)


class StorageSizer:
    def __init__(self, torch_module=None):
        self.torch = torch_module
        self.objects = set()
        self.storages = set()
        self.totals = {key: 0 for key in (
            "python_bytes", "cpu_backing_bytes", "cuda_backing_bytes", "pil_pixel_bytes_estimate",
        )}

    def visit(self, value):
        if value is None or id(value) in self.objects:
            return
        self.objects.add(id(value))
        size = sys.getsizeof(value)
        if isinstance(value, np.ndarray):
            self.totals["python_bytes"] += max(0, size - (value.nbytes if value.flags.owndata else 0))
            if value.base is not None:
                self.visit(value.base)
            else:
                key = ("cpu", value.__array_interface__["data"][0], int(value.nbytes))
                if key not in self.storages:
                    self.storages.add(key)
                    self.totals["cpu_backing_bytes"] += int(value.nbytes)
            if value.dtype.hasobject:
                for item in value.flat:
                    self.visit(item)
            return
        self.totals["python_bytes"] += size
        if self.torch is not None and isinstance(value, self.torch.Tensor):
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr(), storage.nbytes())
            if key not in self.storages:
                self.storages.add(key)
                kind = "cuda_backing_bytes" if value.device.type == "cuda" else "cpu_backing_bytes"
                self.totals[kind] += storage.nbytes()
            return
        if isinstance(value, Image.Image):
            key = ("pil", id(value.im))
            if key not in self.storages:
                self.storages.add(key)
                pixel_bytes = 4 if value.mode in {"I", "F"} else 2 if value.mode.startswith("I;16") else len(value.getbands())
                self.totals["pil_pixel_bytes_estimate"] += value.width * value.height * pixel_bytes
            return
        if isinstance(value, dict):
            for key, item in value.items():
                self.visit(key)
                self.visit(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                self.visit(item)
        elif isinstance(value, memoryview):
            self.visit(value.obj)
        elif hasattr(value, "__dict__"):
            self.visit(vars(value))

    def component(self, value):
        before = dict(self.totals)
        before_storages = len(self.storages)
        self.visit(value)
        result = {key: int(self.totals[key] - before[key]) for key in before}
        result["estimated_bytes"] = sum(result.values())
        result["backing_allocations"] = len(self.storages) - before_storages
        result["items"] = len(value) if hasattr(value, "__len__") else int(value is not None)
        result["live_items"] = (sum(item is not None for item in value)
                                if isinstance(value, (list, tuple)) else result["items"])
        return result


def memory_snapshot(pipeline, extra_components=None, torch_module=None):
    sizer = StorageSizer(torch_module)
    components = {
        name: sizer.component(getattr(pipeline, name, None)) for name in HISTORY_COMPONENTS
    }
    for name, value in (extra_components or {}).items():
        components[name] = sizer.component(value)
    weights = {}
    for name in ("model", "vae", "image_encoder", "surfel_model", "_dino_extractor"):
        model = getattr(pipeline, name, None)
        if name == "_dino_extractor" and model is not None:
            model = model.model
        if model is not None:
            weights["weights_" + name] = tuple(model.parameters()) + tuple(model.buffers())
    for name, tensors in weights.items():
        components[name] = sizer.component(tensors)
    eligible = pipeline.get_allowed_memory_indices()
    eligible_set = set(eligible)
    newest = len(pipeline.pil_frames) - 1
    protected = sorted(pipeline._pinned_memory_frames() | ({newest} if newest >= 0 else set()))
    return {
        "components": components,
        "estimated_component_bytes": sum(c["estimated_bytes"] for c in components.values()),
        "eligible_frames": len(eligible),
        "eligible_frame_indices": eligible,
        "surfel_count": len(pipeline.surfels),
        "surfel_reference_count": sum(len(v) for v in pipeline.surfel_to_timestep.values()),
        "protected_frame_indices": protected,
        "protected_frames_eligible": all(idx in eligible for idx in protected),
        "surfel_references_within_eligible": all(
            idx in eligible_set for values in pipeline.surfel_to_timestep.values() for idx in values
        ),
        "appearance_descriptor_sources": getattr(pipeline, "memory_descriptor_sources", {}),
    }


def process_memory(torch_module=None, device="cpu"):
    rss = None
    try:
        # Linux current RSS, not resource.getrusage's lifetime maximum.
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        rss = resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        try:
            import psutil
            rss = psutil.Process().memory_info().rss
        except ImportError:
            pass
    result = {"process_rss_bytes": rss, "pid": os.getpid()}
    if torch_module is not None and str(device).startswith("cuda"):
        result.update(
            cuda_allocated_bytes=torch_module.cuda.memory_allocated(device),
            cuda_reserved_bytes=torch_module.cuda.memory_reserved(device),
            cuda_peak_allocated_bytes=torch_module.cuda.max_memory_allocated(device),
            cuda_peak_reserved_bytes=torch_module.cuda.max_memory_reserved(device),
        )
    return result


class ResourceProfiler:
    def __init__(self, path, *, device="cpu", torch_module=None, fps=13.0, warmup_steps=2,
                 session_start_step=0, session_id=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.torch = torch_module
        self.fps = fps
        self.warmup_steps = warmup_steps
        self.session_start_step = session_start_step
        self.session_id = session_id
        self.active = False
        self.extra_components = lambda: {}
        self.current_phase = None
        self.failed_phase = None
        self.write({"event": "session" if self.path.exists() else "schema", "schema": SCHEMA_VERSION,
                    "session_start_step": session_start_step, "session_id": session_id,
                    "timing": "perf_counter wall time with selected CUDA device synchronized at each stage boundary",
                    "warmup_steps": warmup_steps,
                    "bytes": "unique backing storage + sys.getsizeof + estimated logical PIL pixels; excludes native allocator overhead",
                    "cuda_scope": "this process PyTorch allocator; peaks cumulative since process start",
                    "component_attribution": "first owning component wins; shared payload counted once"})

    def write(self, record):
        # Append and close each event so failed jobs leave usable raw records.
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")

    def synchronize(self):
        if self.torch is not None and str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    def begin_step(self, pipeline):
        self.active = True
        self.step = int(pipeline.global_step)
        self.times = {}
        self.before_update = None
        self.reconstruction_input_indices = []
        self.accounting_seconds = 0.0
        self.current_phase = None

    def begin_phase(self, name):
        if not self.active:
            return
        if self.current_phase is not None:
            raise RuntimeError(f"Nested resource timing: {self.current_phase} / {name}")
        self.current_phase = name
        self.synchronize()
        self.phase_started = time.perf_counter()

    def end_phase(self, name):
        if not self.active:
            return
        if self.current_phase != name:
            raise RuntimeError(f"Mismatched resource timing: {self.current_phase} / {name}")
        self.synchronize()
        self.times[name] = self.times.get(name, 0.0) + time.perf_counter() - self.phase_started
        self.current_phase = None

    def capture(self, pipeline):
        started = time.perf_counter()
        data = memory_snapshot(pipeline, self.extra_components(), self.torch)
        data.update(process_memory(self.torch, self.device))
        self.accounting_seconds += time.perf_counter() - started
        return data

    def capture_before_update(self, pipeline):
        self.before_update = self.capture(pipeline)

    def finish_step(self, pipeline):
        after_update = self.capture(pipeline)
        self.write({
            "event": "step", "schema": SCHEMA_VERSION, "step": self.step,
            "warmup": self.step < self.session_start_step + self.warmup_steps,
            "session_start_step": self.session_start_step, "session_id": self.session_id,
            "frame_count": len(pipeline.pil_frames), "video_seconds": len(pipeline.pil_frames) / self.fps,
            "policy": pipeline.memory_policy, "budget": pipeline.memory_budget,
            "scope": pipeline.memory_scope, "phase_seconds": self.times,
            "accounting_seconds": self.accounting_seconds,
            "reconstruction_input_indices": self.reconstruction_input_indices,
            "before_update": self.before_update, "after_update": after_update,
        })
        self.active = False

    def initial_snapshot(self, pipeline):
        self.accounting_seconds = 0.0
        self.write({"event": "initial", "frame_count": len(pipeline.pil_frames),
                    "after_update": self.capture(pipeline)})

    def failure(self, exc):
        self.write({"event": "failure", "step": getattr(self, "step", None),
                    "phase": self.failed_phase or self.current_phase,
                    "error_type": type(exc).__name__, "error": str(exc)})


def profiled_phase(name):
    def decorate(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            profiler = getattr(self, "resource_profiler", None)
            if profiler is None or not profiler.active:
                return function(self, *args, **kwargs)
            profiler.begin_phase(name)
            try:
                result = function(self, *args, **kwargs)
            except BaseException:
                profiler.failed_phase = name
                raise
            profiler.end_phase(name)
            return result
        return wrapped
    return decorate
