#!/usr/bin/env python
"""Probe initial CLIP encoding without loading VMem/VAE/CUT3R weights or generating video."""

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import random
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "vmem_clip_probe_v1"


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def difference(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("Probe arrays must have matching shapes and dtypes; no conversion/resizing")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Nonfinite probe arrays")
    error = right.astype(np.float64) - left.astype(np.float64)
    norm = float(np.linalg.norm(left.astype(np.float64)))
    return {"equal": bool(np.array_equal(left, right)),
            "max_abs": float(np.abs(error).max()), "mean_abs": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "relative_l2": float(np.linalg.norm(error) / norm) if norm else None,
            "changed_element_fraction": float(np.mean(left != right))}


def model_fingerprint(model, debug):
    entries = {"parameter:" + name: debug.fingerprint(value)
               for name, value in model.named_parameters()}
    entries.update({"buffer:" + name: debug.fingerprint(value)
                    for name, value in model.named_buffers()})
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"sha256": digest, "tensors": entries}


@contextmanager
def attention_profile(profile, torch):
    if profile == "native":
        yield
        return
    if profile != "math":
        raise ValueError(f"Unknown attention profile: {profile}")
    from torch.nn.attention import SDPBackend, sdpa_kernel

    previous = torch.backends.mha.get_fastpath_enabled()
    try:
        # MHA's native fast path can bypass the SDPA backend selection.
        torch.backends.mha.set_fastpath_enabled(False)
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def resolve_repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def run(args):
    reference = json.loads((args.reference_run / "run_spec.json").read_text())
    with (args.reference_run / "generation_debug.jsonl").open() as handle:
        old_events = [json.loads(line) for line in handle]
    input_record = next(row for row in old_events if row["event"] == "initial_input")
    encoding_record = next(row for row in old_events if row["event"] == "initial_encoding")
    image_path = resolve_repo_path(reference["arguments"]["image"])
    config_path = resolve_repo_path(reference["arguments"]["config"])
    for path, key in ((image_path, "image_sha256"), (config_path, "config_sha256")):
        if digest_file(path) != reference["provenance"][key]:
            raise ValueError(f"Reference {key} does not match current file: {path}")
    if not 2 <= args.repeats <= 5:
        raise ValueError("Use 2-5 encoder repetitions")
    args.output.mkdir(parents=True, exist_ok=False)

    from generation_debug import GenerationDebug
    from scripts import run_vmem_demo_actions as runner
    # Preserve the real import side effects (including backend flags), but do
    # not instantiate a VMemPipeline or load its video/VAE/CUT3R checkpoints.
    runner._load_runtime_dependencies()
    torch = runner.torch
    from modeling.modules.conditioner import CLIPConditioner
    seed = reference["arguments"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    debug = GenerationDebug(args.output / "probe_trace.jsonl", seed=seed, mode="observe",
                            device=device, torch_module=torch)
    config = runner.OmegaConf.load(config_path)
    image = runner._load_vmem_image(image_path, config=config, device=device)
    input_fingerprint = debug.fingerprint(image)
    if input_fingerprint != input_record["image"]:
        raise ValueError("Reloaded input differs from the reference tensor; do not interpret as an encoder-only comparison")
    encoder = CLIPConditioner().eval()
    weights = model_fingerprint(encoder, debug)
    encoder = encoder.to(device=device, dtype=torch.float32).eval()
    if any(module.training for module in encoder.modules()):
        raise ValueError("An encoder module is still in training mode")

    sources = {str(path.relative_to(ROOT)): digest_file(path) for path in (
        Path(__file__).resolve(), ROOT / "modeling/modules/conditioner.py", ROOT / "utils/util.py")}
    import kornia
    for label, obj in (("installed_clip", type(encoder.module)),
                       ("installed_visual", type(encoder.module.visual)),
                       ("installed_preprocess_resize", kornia.geometry.resize)):
        path = inspect.getsourcefile(obj)
        sources[label] = digest_file(path) if path is not None else None
    environment = debug.environment()
    reference_environment = json.loads((args.reference_run / "generation_environment.json").read_text())
    environment_matches_reference = environment == reference_environment
    environment["mha_fastpath_enabled"] = torch.backends.mha.get_fastpath_enabled()
    environment["grad_enabled"] = torch.is_grad_enabled()
    preprocessed, embeddings = [], []
    with attention_profile(args.attention, torch):
        for index in range(args.repeats):
            debug.record("before_encode", index, rng=debug.rng_state())
            # These are the same two operations as CLIPConditioner.forward.
            processed = encoder.preprocess(image.to(device=device, dtype=torch.float32))
            embedding = encoder.module.encode_image(processed)
            preprocessed.append(processed.detach().cpu().numpy().copy())
            embeddings.append(embedding.detach().cpu().numpy()[0].copy())
            debug.record("after_encode", index, processed=processed, embedding=embedding, rng=debug.rng_state())
            del processed, embedding
    artifact = args.output / "samples.npz"
    np.savez(artifact, preprocessed=np.stack(preprocessed), embeddings=np.stack(embeddings))
    report = {"schema": SCHEMA, "status": "complete", "reference_run": str(args.reference_run),
              "seed": seed, "attention": args.attention, "repeats": args.repeats,
              "environment_matches_reference": environment_matches_reference,
              "environment": environment, "source_sha256": sources,
              "input": input_fingerprint, "weights": weights, "samples_sha256": digest_file(artifact),
              "matches_reference_embedding": [debug.fingerprint(value) == encoding_record["embeddings"][0]
                                              for value in embeddings],
              "within_process": {"preprocessed": [difference(preprocessed[0], value) for value in preprocessed[1:]],
                                 "embeddings": [difference(embeddings[0], value) for value in embeddings[1:]]},
              "scope": "Standalone FP32 initial encoder, no autocast or inference-mode override. Imports match VMem, but allocator/model-load history differs. Math profile changes attention dispatch only; it is not a deterministic-output guarantee or a benchmark change."}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"output": str(args.output), "attention": args.attention,
                      "environment_matches_reference": environment_matches_reference,
                      "matches_reference_embedding": report["matches_reference_embedding"],
                      "within_process": report["within_process"]}, indent=2))


def load_probe(path):
    report = json.loads((path / "report.json").read_text())
    if report.get("schema") != SCHEMA or report.get("status") != "complete":
        raise ValueError(f"Not a completed encoder probe: {path}")
    artifact = path / "samples.npz"
    if digest_file(artifact) != report["samples_sha256"]:
        raise ValueError(f"Probe sample hash mismatch: {path}")
    with np.load(artifact, allow_pickle=False) as arrays:
        values = {key: arrays[key].copy() for key in ("preprocessed", "embeddings")}
    if any(value.shape[0] != report["repeats"] for value in values.values()):
        raise ValueError(f"Incomplete repetitions: {path}")
    return report, values


def compare(left, right):
    a, av = load_probe(left)
    b, bv = load_probe(right)
    if a["repeats"] != b["repeats"]:
        raise ValueError("Compare probes with the same repetition count")
    return {"left": str(left), "right": str(right), "attention_profiles": [a["attention"], b["attention"]],
            "input_equal": a["input"] == b["input"], "weights_equal": a["weights"] == b["weights"],
            "sources_equal": a["source_sha256"] == b["source_sha256"],
            "environment_equal": a["environment"] == b["environment"],
            "comparisons": {key: [difference(x, y) for x, y in zip(av[key], bv[key])]
                            for key in av},
            "note": "Array errors are encoder/preprocessing units, not video quality or GT fidelity. Math/native disagreement is not evidence of a faulty kernel. Stable standalone probes do not prove full-pipeline reproducibility."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("run")
    probe.add_argument("--reference-run", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--device", default="cuda")
    probe.add_argument("--attention", choices=("native", "math"), default="native")
    probe.add_argument("--repeats", type=int, default=3)
    diff = commands.add_parser("compare")
    diff.add_argument("--left", type=Path, required=True)
    diff.add_argument("--right", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        print(json.dumps(compare(args.left, args.right), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
