"""Phase/action RNG isolation shared by diagnostics and normal generation."""

from contextlib import contextmanager, ExitStack
import hashlib
import json
import random

import numpy as np


RNG_SCHEMA = "vmem_phase_action_rng_v1"
EXECUTION_DEFAULTS = {"rng_mode": "legacy", "clip_attention": "native", "cut3r_attention": "native"}


def phase_seed(seed, phase, step):
    payload = json.dumps([RNG_SCHEMA, int(seed), phase, int(step)], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2 ** 63)


def execution_settings(arguments):
    values = {key: arguments.get(key, default) for key, default in EXECUTION_DEFAULTS.items()}
    if arguments.get("generation_debug") == "isolated":
        values["rng_mode"] = "isolated"
    values["rng_schema"] = RNG_SCHEMA if values["rng_mode"] == "isolated" else "legacy"
    return values


class PhaseRNG:
    def __init__(self, *, seed, device, torch_module):
        self.seed, self.torch = seed, torch_module
        self.device = torch_module.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Phase RNG supports CPU and one selected CUDA device")
        self.devices = []
        if self.device.type == "cuda":
            index = self.device.index
            self.devices = [torch_module.cuda.current_device() if index is None else index]

    @contextmanager
    def phase(self, name, step):
        torch = self.torch
        derived = phase_seed(self.seed, name, step)
        with ExitStack() as stack:
            stack.callback(random.setstate, random.getstate())
            stack.callback(np.random.set_state, np.random.get_state())
            stack.enter_context(torch.random.fork_rng(devices=self.devices))
            random.seed(derived)
            np.random.seed(derived % (2 ** 32))
            # Seed only CPU and the selected GPU, preserving other devices.
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(derived).get_state())
            if self.devices:
                selected = torch.device("cuda", self.devices[0])
                state = torch.Generator(device=selected).manual_seed(derived).get_state()
                torch.cuda.set_rng_state(state, selected)
            yield
