#!/usr/bin/env python
"""Repeat first-action reconstruction from saved RGB frames; load only CUT3R weights."""

import argparse
from contextlib import ExitStack, nullcontext
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.probe_vmem_clip import digest_file, model_fingerprint
from clip_attention import attention_profile

SCHEMA = "vmem_reconstruction_probe_v1"
STAGES = ("preprocessed", "predictions", "aligned")
MATH_ATTENTION = {"mha_fastpath": False, "flash_sdp": False,
                  "memory_efficient_sdp": False, "math_sdp": True, "cudnn_sdp": False}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def attention_state(torch):
    return {
        "mha_fastpath": torch.backends.mha.get_fastpath_enabled(),
        "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
        "memory_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp": torch.backends.cuda.math_sdp_enabled(),
        "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled(),
    }


def load_reference(path):
    from PIL import Image

    if read_json(path / "run_status.json").get("status") != "complete":
        raise ValueError("Reference generation must be complete")
    spec = read_json(path / "run_spec.json")
    args = spec["arguments"]
    if args["frames_per_action"] != 4 or args.get("resume_from"):
        raise ValueError("Use a fresh four-frame-action reference")
    actions = read_json(path / "actions.json")
    if not actions or actions[0]["action_index"] != 0 or actions[0]["total_frames_after"] != 5:
        raise ValueError("Expected five frames after the first action")
    if actions[0]["action"] != spec["actions"][0]:
        raise ValueError("First action disagrees with the run spec")
    with (path / "resource_trace.jsonl").open() as handle:
        first = next((row for line in handle if (row := json.loads(line)).get("event") == "step"), None)
    if first is None or first["step"] != 0 or first["reconstruction_input_indices"] != list(range(5)):
        raise ValueError("Reference must reconstruct frames 0-4 on the first action")
    manifest = read_json(path / "frame_manifest.json")
    if manifest.get("schema") != "vmem_output_frames_v1" or len(manifest["sha256"]) != 1 + 4 * args["num_actions"]:
        raise ValueError("Missing or incomplete frame manifest")
    images = []
    for index in range(5):
        frame = path / "generated_frames" / f"{index:04d}.png"
        encoded = frame.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != manifest["sha256"][index]:
            raise ValueError(f"Reference frame hash mismatch: {frame}")
        with Image.open(io.BytesIO(encoded)) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError("Expected saved RGB PNGs, without mode conversion")
            if images and image.size != images[0].size:
                raise ValueError("Reference frame sizes differ; no resizing is allowed")
            images.append(image.copy())
    provenance = {
        "frame_indices": list(range(5)), "frame_sha256": manifest["sha256"][:5],
        "records_sha256": {name: digest_file(path / name) for name in (
            "run_spec.json", "actions.json", "resource_trace.jsonl", "frame_manifest.json",
            "generation_config.yaml", "generation_environment.json")},
    }
    return spec, actions[0], images, provenance


def first_action_poses(runner, spec, action):
    captured = []

    def collect(poses, intrinsics, **kwargs):
        captured.extend(poses)
        return []

    args = spec["arguments"]
    navigator = runner.Navigator(SimpleNamespace(generate_trajectory_frames=collect),
                                 step_size=args["step_size"], num_interpolation_frames=4,
                                 retain_frame_history=False)
    navigator.current_pose = np.eye(4, dtype=np.float32)
    navigator.current_K = np.array(runner.get_default_intrinsics()[0])
    runner._apply_action(navigator, action["action"])
    if len(captured) != 4 or not np.allclose(navigator.current_pose, action["current_pose"], rtol=0, atol=1e-8):
        raise ValueError("Replayed Navigator action does not match the reference endpoint")
    # The pipeline stores generated poses after its float32 tensor conversion.
    poses = np.asarray([np.eye(4)] + captured, dtype=np.float32)
    return runner.VMemPipeline.get_transformed_c2ws(None, poses)


def snapshot(value, torch):
    arrays, entries = {}, {}

    def visit(node, path):
        if isinstance(node, torch.Tensor):
            node = node.detach().cpu().numpy()
        if isinstance(node, np.ndarray):
            if node.dtype.kind not in "buif":
                raise ValueError(f"Unsupported diagnostic array dtype: {node.dtype}")
            key = f"array_{len(arrays):04d}"
            arrays[key] = np.ascontiguousarray(node)
            entries[key] = {"path": path, "shape": list(node.shape), "dtype": str(node.dtype),
                            "sha256": hashlib.sha256(arrays[key].tobytes()).hexdigest()}
            return {"array": key}
        if isinstance(node, dict):
            return {key: visit(node[key], path + [key]) for key in sorted(node)}
        if isinstance(node, (list, tuple)):
            return [visit(item, path + [index]) for index, item in enumerate(node)]
        if isinstance(node, np.generic):
            node = node.item()
        if node is None or isinstance(node, (str, bool, int, float)):
            return node
        raise TypeError(f"Unsupported diagnostic value at {path}: {type(node).__name__}")

    tree = visit(value, [])
    return {"tree": tree, "arrays": entries}, arrays


def observe_reconstruction(reconstruction, inference_module, model, images, *, poses,
                           lr, niter, device, capture, debug, attention="native",
                           torch_module=None, attention_trace=None):
    """Intercept existing call boundaries only; return original objects unchanged."""
    if attention not in ("native", "math"):
        raise ValueError(f"Unknown prediction attention profile: {attention}")
    if torch_module is None and (attention != "native" or attention_trace is not None):
        raise ValueError("Prediction attention control/recording requires PyTorch")
    prepare = reconstruction.prepare_input_from_pil
    infer = inference_module.inference
    align = reconstruction.prepare_output
    rng = {"start": debug.fingerprint(debug.rng_state())}

    def observed_prepare(*args, **kwargs):
        views = prepare(*args, **kwargs)
        capture("preprocessed", views)
        return views

    def observed_infer(*args, **kwargs):
        if attention_trace is not None:
            if attention_trace:
                raise ValueError("Expected one CUT3R inference call per reconstruction")
            attention_trace.update(profile=attention, before=attention_state(torch_module))
        try:
            # Restore dispatch before capture/alignment, including on failure.
            with attention_profile(attention, torch_module):
                if attention_trace is not None:
                    attention_trace["active"] = attention_state(torch_module)
                result = infer(*args, **kwargs)
        finally:
            if attention_trace is not None:
                attention_trace["after"] = attention_state(torch_module)
        capture("predictions", result[0]["pred"])
        return result

    def observed_align(*args, **kwargs):
        rng["before_alignment"] = debug.fingerprint(debug.rng_state())
        return align(*args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(reconstruction, "prepare_input_from_pil", observed_prepare))
        stack.enter_context(patch.object(inference_module, "inference", observed_infer))
        stack.enter_context(patch.object(reconstruction, "prepare_output", observed_align))
        result = reconstruction.run_inference_from_pil(
            images, model, poses=poses, depths=None, lr=lr, niter=niter,
            device=device, size=512, visualize=False, save_flag=False)
        capture("aligned", {key: result[key] for key in (
            "point_clouds", "depths", "confidences", "camera_info")})
    rng["end"] = debug.fingerprint(debug.rng_state())
    return rng


def array_difference(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return {"comparable": False, "left_shape": list(left.shape), "right_shape": list(right.shape),
                "left_dtype": str(left.dtype), "right_dtype": str(right.dtype)}
    finite = np.isfinite(left) & np.isfinite(right)
    error = right[finite].astype(np.float64) - left[finite].astype(np.float64)
    equal = (left == right) | (np.isnan(left) & np.isnan(right))
    return {"comparable": True, "values_equal_nan_aware": bool(equal.all()),
            "changed_element_fraction": float(np.mean(~equal)) if equal.size else 0.0,
            "nonfinite_positions_equal": bool(np.array_equal(np.isfinite(left), np.isfinite(right))),
            "finite_elements_compared": int(finite.sum()),
            "max_abs_finite": float(np.abs(error).max()) if error.size else None,
            "rmse_finite": float(np.sqrt(np.mean(error ** 2))) if error.size else None}


def compare_stage(left_root, right_root, left, right, repeat, stage, *, right_repeat=None):
    right_repeat = repeat if right_repeat is None else right_repeat
    paths = [left_root / f"repeat_{repeat:02d}" / f"{stage}.npz",
             right_root / f"repeat_{right_repeat:02d}" / f"{stage}.npz"]
    for path, record in zip(paths, (left, right)):
        if digest_file(path) != record["artifact_sha256"]:
            raise ValueError(f"Changed stage artifact: {path}")
    differing, first = [], None
    with np.load(paths[0], allow_pickle=False) as a, np.load(paths[1], allow_pickle=False) as b:
        for values, record in ((a, left), (b, right)):
            if not values.files or set(values.files) != set(record["arrays"]):
                raise ValueError("Stage array inventory does not match its report")
            for key, entry in record["arrays"].items():
                value = values[key]
                if (list(value.shape) != entry["shape"] or str(value.dtype) != entry["dtype"]
                        or hashlib.sha256(value.tobytes()).hexdigest() != entry["sha256"]):
                    raise ValueError("Stage array values do not match their recorded fingerprints")
        for key in sorted(set(left["arrays"]) | set(right["arrays"])):
            la, rb = left["arrays"].get(key), right["arrays"].get(key)
            if la != rb:
                differing.append(key)
                if first is None:
                    first = {"left": la, "right": rb}
                    if la is not None and rb is not None:
                        first["errors"] = array_difference(a[key], b[key])
    tree_equal = left["tree"] == right["tree"]
    return {"equal": tree_equal and not differing, "structure_equal": tree_equal,
            "array_count": len(left["arrays"]), "different_array_count": len(differing),
            "first_difference": first}


def load_report(path):
    report = read_json(path / "report.json")
    if report.get("schema") != SCHEMA or report.get("status") != "complete":
        raise ValueError(f"Not a completed reconstruction probe: {path}")
    for key in ("reference_inputs", "settings", "weights", "source_sha256", "environment", "reconstruction_environment"):
        if not report.get(key):
            raise ValueError(f"Missing reconstruction provenance: {key}")
    count = report["settings"]["repeats"]
    profile = report["settings"].get("prediction_attention", "native")
    if profile not in ("native", "math"):
        raise ValueError("Unknown recorded prediction attention profile")
    if len(report["repetitions"]) != count or count not in (2, 3):
        raise ValueError("Incomplete reconstruction repetitions")
    for index, record in enumerate(report["repetitions"]):
        if record["index"] != index or set(record["stages"]) != set(STAGES):
            raise ValueError("Incomplete reconstruction stages")
        rng = record.get("rng", {})
        if set(rng) != {"start", "before_alignment", "end"} or not all(rng.values()):
            raise ValueError("Incomplete reconstruction RNG records")
        if "prediction_attention" in report["settings"]:
            trace = record.get("prediction_attention", {})
            if (not isinstance(trace, dict) or trace.get("profile") != profile or any(
                    not isinstance(trace.get(key), dict) or set(trace[key]) != set(MATH_ATTENTION)
                    or any(type(value) is not bool for value in trace[key].values())
                    for key in ("before", "active", "after"))):
                raise ValueError("Incomplete prediction attention records")
            expected = trace["before"] if profile == "native" else MATH_ATTENTION
            if trace["active"] != expected or trace["after"] != trace["before"]:
                raise ValueError("Prediction attention profile was not applied/restored")
    return report


def compare(left_path, right_path):
    a, b = load_report(left_path), load_report(right_path)
    checks = {key + "_equal": a[key] == b[key] for key in (
        "reference_inputs", "settings", "weights", "source_sha256", "environment", "reconstruction_environment")}
    checks["weights_unchanged"] = a["weights_unchanged"] and b["weights_unchanged"]
    if len(a["repetitions"]) != len(b["repetitions"]):
        raise ValueError("Compare probes with the same repetition count")
    rows = []
    for index, (ar, br) in enumerate(zip(a["repetitions"], b["repetitions"])):
        stages = {stage: compare_stage(left_path, right_path, ar["stages"][stage], br["stages"][stage], index, stage)
                  for stage in STAGES}
        rows.append({"repeat": index, "rng_equal": ar["rng"] == br["rng"], "stages": stages,
                     "prediction_attention_equal": ar.get("prediction_attention") == br.get("prediction_attention"),
                     "first_different_stage": next((stage for stage in STAGES if not stages[stage]["equal"]), None)})
    return {"schema": SCHEMA + "_comparison", "left": str(left_path), "right": str(right_path),
            **checks, "matched_control": all(checks.values()) and all(
                row["rng_equal"] and row["prediction_attention_equal"] for row in rows),
            "prediction_attention": [record["settings"].get("prediction_attention", "native") for record in (a, b)],
            "environment_matches_reference": [a.get("environment_matches_reference"), b.get("environment_matches_reference")],
            "comparisons": rows,
            "note": "Standalone first-action reconstruction, not a replay of original RNG/allocator state. No surfel merging or quality measurement. Errors use each field's native units; expected NaN ray-map placeholders are compared explicitly."}


def run(args):
    profile = getattr(args, "attention", "native")
    if profile not in ("native", "math"):
        raise ValueError(f"Unknown prediction attention profile: {profile}")
    if args.output.resolve().is_relative_to(args.reference_run.resolve()):
        raise ValueError("Probe output must be outside the reference run")
    spec, first_action, images, reference_inputs = load_reference(args.reference_run)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", {"schema": SCHEMA, "status": "running", "prediction_attention": profile})
    try:
        _run_loaded(args, spec, first_action, images, reference_inputs)
    except BaseException as error:
        failure = {"schema": SCHEMA, "status": "failed", "prediction_attention": profile,
                   "error_type": type(error).__name__, "error": str(error)}
        write_json(args.output / "failure.json", failure)
        write_json(args.output / "report.json", failure)
        raise
    finally:
        for image in images:
            image.close()


def _run_loaded(args, spec, first_action, images, reference_inputs):
    from generation_debug import GenerationDebug, RNG_SCHEMA
    from scripts import run_vmem_demo_actions as runner
    # Preserve VMem import side effects, but never construct VMemPipeline/CLIP/VAE.
    runner._load_runtime_dependencies()
    torch = runner.torch
    profile = getattr(args, "attention", "native")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    config = runner.OmegaConf.load(args.reference_run / "generation_config.yaml")
    poses = first_action_poses(runner, spec, first_action)
    lr, niter = float(config.surfel.lr), int(config.surfel.niter)
    if niter <= 0 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("Invalid saved reconstruction settings")
    override = spec["arguments"].get("surfel_niter")
    if override is not None and override != niter:
        raise ValueError("Effective reconstruction steps disagree with the run spec")
    seed = spec["arguments"]["seed"]
    debug = GenerationDebug(args.output / "load_trace.jsonl", seed=seed, mode="isolated",
                            device=device, torch_module=torch)
    environment = debug.environment()
    from huggingface_hub import hf_hub_download
    from modeling.pipeline import ARCroco3DStereo, add_path_to_dust3r
    checkpoint = hf_hub_download(repo_id=config.surfel.model_path, filename="cut3r_512_dpt_4_64.pth")
    checkpoint_hash = digest_file(checkpoint)
    if checkpoint_hash != spec["provenance"]["checkpoint_sha256"]["cut3r"]:
        raise ValueError("CUT3R checkpoint differs from the reference generation")
    with debug.phase("reconstruction_probe_load", -1):
        add_path_to_dust3r(checkpoint)
        model = ARCroco3DStereo.from_pretrained(checkpoint).to(device).eval()
    if any(module.training for module in model.modules()):
        raise ValueError("CUT3R has a submodule in training mode")
    weights = model_fingerprint(model, debug)
    reconstruction = importlib.import_module("extern.CUT3R.surfel_inference")
    inference_module = importlib.import_module("src.dust3r.inference")
    # Initialize the aligner imports before repeating, as the full pipeline does.
    importlib.import_module("extern.CUT3R.cloud_opt.dust3r_opt")
    importlib.import_module("cloud_opt.dust3r_opt")
    reconstruction_environment = debug.environment()
    reconstruction_environment["attention"] = attention_state(torch)
    source_paths = [Path(__file__).resolve(), ROOT / "scripts/probe_vmem_clip.py", ROOT / "generation_debug.py",
                    ROOT / "clip_attention.py",
                    ROOT / "navigation.py", ROOT / "modeling/pipeline.py", ROOT / "scripts/run_vmem_demo_actions.py"]
    source_paths.extend(sorted((ROOT / "extern/CUT3R").rglob("*.py")))
    sources = {str(path.relative_to(ROOT)): digest_file(path) for path in source_paths}
    repetitions = []
    for index in range(args.repeats):
        target = args.output / f"repeat_{index:02d}"
        target.mkdir()
        current_debug = GenerationDebug(target / "rng_trace.jsonl", seed=seed, mode="isolated",
                                        device=device, torch_module=torch)
        stages, attention_trace = {}, {}

        def capture(stage, value):
            if stage in stages or stage not in STAGES:
                raise ValueError(f"Unexpected/duplicate reconstruction stage: {stage}")
            record, arrays = snapshot(value, torch)
            path = target / f"{stage}.npz"
            np.savez_compressed(path, **arrays)
            stages[stage] = {**record, "artifact_sha256": digest_file(path)}

        print(f"Reconstruction-only repetition {index + 1}/{args.repeats}: five saved frames, attention={profile}", flush=True)
        # Same derived phase seed on every repetition. This is not the unknown
        # original post-diffusion RNG state; attention is controlled separately.
        with current_debug.phase("reconstruction_probe", 0), torch.no_grad(), (
            torch.autocast("cuda") if device.type == "cuda" else nullcontext()
        ):
            rng = observe_reconstruction(reconstruction, inference_module, model, images,
                                         poses=poses.copy(), lr=lr, niter=niter, device=device,
                                         capture=capture, debug=current_debug, attention=profile,
                                         torch_module=torch, attention_trace=attention_trace)
        if set(stages) != set(STAGES):
            raise ValueError("Missing reconstruction observation stage")
        repetitions.append({"index": index, "stages": stages, "rng": rng,
                            "prediction_attention": attention_trace})
        write_json(target / "stages.json", repetitions[-1])
    settings = {"seed": seed, "repeats": args.repeats, "rng_schema": RNG_SCHEMA,
                "prediction_attention": profile,
                "rng_phase": "reconstruction_probe", "rng_step": 0,
                "size": 512, "lr": lr, "niter": niter, "device": str(device),
                "outer_cuda_autocast": device.type == "cuda", "prior_depths": None,
                "outer_cuda_autocast_dtype": str(torch.get_autocast_dtype("cuda")) if device.type == "cuda" else None,
                "poses": {"fingerprint": debug.fingerprint(poses), "values": poses.tolist()},
                "checkpoint_sha256": checkpoint_hash}
    report = {"schema": SCHEMA, "status": "complete", "reference_run": str(args.reference_run),
              "reference_inputs": reference_inputs, "settings": settings, "source_sha256": sources,
              "environment": environment,
              "reconstruction_environment": reconstruction_environment,
              "environment_matches_reference": environment == read_json(args.reference_run / "generation_environment.json"),
              "weights": weights, "weights_unchanged": weights == model_fingerprint(model, debug),
              "repetitions": repetitions,
              "scope": "Existing reconstruction function only. Fixed saved RGB frames and replayed Navigator poses; controlled probe RNG, original ambient autocast. Optional math attention applies only to CUT3R inference and restores before alignment. Environment matching describes ambient flags, not an unchanged execution profile. No video/VAE/CLIP weights or surfel merging. Source hashes exclude compiled extensions and installed dependencies."}
    # Compare repeats without copying or relabelling their on-disk artifacts.
    within = []
    for index in range(1, args.repeats):
        stages = {}
        for stage in STAGES:
            a, b = repetitions[0]["stages"][stage], repetitions[index]["stages"][stage]
            stages[stage] = compare_stage(args.output, args.output, a, b, 0, stage, right_repeat=index)
        within.append({"repeat": index, "rng_equal": repetitions[0]["rng"] == repetitions[index]["rng"],
                       "prediction_attention_equal": repetitions[0]["prediction_attention"] == repetitions[index]["prediction_attention"],
                       "stages": stages,
                       "first_different_stage": next((stage for stage in STAGES if not stages[stage]["equal"]), None)})
    write_json(args.output / "within_process.json", within)
    write_json(args.output / "report.json", report)
    load_report(args.output)
    print(json.dumps({"output": str(args.output), "environment_matches_reference": report["environment_matches_reference"],
                      "prediction_attention": profile, "weights_unchanged": report["weights_unchanged"],
                      "within_process": within}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("run")
    probe.add_argument("--reference-run", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    probe.add_argument("--repeats", choices=(2, 3), type=int, default=2)
    probe.add_argument("--attention", choices=("native", "math"), default="native",
                       help="Probe-only CUT3R inference dispatch; restores before alignment")
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
