#!/usr/bin/env python
"""Compare two completed short generation diagnostics, including same-policy repeats."""

import argparse
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def read_run(path):
    if read_json(path / "run_status.json").get("status") != "complete":
        raise ValueError(f"Diagnostic is not complete: {path}")
    spec = read_json(path / "run_spec.json")
    args = spec["arguments"]
    with (path / "generation_debug.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle]
    if not rows or rows[0].get("schema") != "vmem_generation_debug_v1":
        raise ValueError(f"Unexpected generation diagnostic schema: {path}")
    if rows[0]["mode"] != args["generation_debug"] or rows[0]["seed"] != args["seed"]:
        raise ValueError(f"Diagnostic header disagrees with run spec: {path}")
    events = {}
    for row in rows[1:]:
        key = (row["event"], row["step"], row.get("phase"), row.get("sampler_step"))
        if key in events or row["event"] == "phase_error":
            raise ValueError(f"Duplicate or failed diagnostic event: {key} in {path}")
        events[key] = row
    required = [("initial_input", -1, None, None), ("initial_encoding", -1, None, None)]
    phases = [(phase, -1) for phase in ("model_load", "initialization")]
    for step in range(args["num_actions"]):
        required.extend((name, step, None, None) for name in (
            "conditioning", "initial_noise", "diffusion_latents", "decoded_samples"))
        phases.extend((phase, step) for phase in ("diffusion", "reconstruction"))
        indices = [row["sampler_step"] for row in rows[1:]
                   if row["event"] == "sampler_noise" and row["step"] == step]
        if not indices or indices != list(range(len(indices))):
            raise ValueError(f"Incomplete/unordered sampler noise at step {step}: {path}")
        if args.get("inference_steps") is not None and len(indices) != args["inference_steps"]:
            raise ValueError(f"Sampler noise count disagrees with configured steps: {path}")
    required.extend((event, step, phase, None) for phase, step in phases
                    for event in ("phase_start", "phase_end"))
    if any(key not in events for key in required):
        raise ValueError(f"Missing diagnostic events: {path}")
    return {"spec": spec, "rows": rows[1:], "events": events,
            "environment": read_json(path / "generation_environment.json")}


def compare(left_path, right_path):
    left, right = read_run(left_path), read_run(right_path)
    a, b = left["spec"]["arguments"], right["spec"]["arguments"]
    settings = ("generation_debug", "seed", "num_actions", "frames_per_action", "trajectory", "pattern",
                "step_size", "fps", "frame_storage", "memory_scope", "inference_steps", "surfel_niter",
                "surfel_reconstruction_window", "visualize_intermediates")
    mismatches = [f"arguments.{key}" for key in settings if a.get(key) != b.get(key)]
    missing = [f"arguments.{key}" for key in settings if key not in a or key not in b]
    for key in ("image_sha256", "config_sha256", "source_sha256", "checkpoint_sha256"):
        values = [run["spec"].get("provenance", {}).get(key) for run in (left, right)]
        if not all(values):
            missing.append(f"provenance.{key}")
        elif values[0] != values[1]:
            mismatches.append(f"provenance.{key}")
    differences = []
    keys = list(left["events"]) + [key for key in right["events"] if key not in left["events"]]
    for key in keys:
        lrow, rrow = left["events"].get(key), right["events"].get(key)
        if lrow != rrow:
            fields = sorted(name for name in set(lrow or {}) | set(rrow or {})
                            if (lrow or {}).get(name) != (rrow or {}).get(name))
            differences.append({"event": key[0], "step": key[1], "phase": key[2],
                                "sampler_step": key[3], "different_fields": fields})
    first_by_event = {}
    for diff in differences:
        name = diff["event"] + (":" + diff["phase"] if diff["phase"] else "")
        first_by_event.setdefault(name, diff)
    return {"schema": "vmem_generation_debug_comparison_v1",
            "left": str(left_path), "right": str(right_path),
            "policies": [a["memory_policy"], b["memory_policy"]],
            "settings_mismatches": mismatches, "missing_provenance": missing,
            "environment_equal": left["environment"] == right["environment"],
            "all_recorded_events_equal": not differences,
            "first_difference": differences[0] if differences else None,
            "first_difference_by_event": first_by_event,
            "different_event_count": len(differences),
            "note": "Hashes identify where values first differ, not why. Matching noise does not guarantee deterministic inference; debug timings are not benchmark timings."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.left, args.right), indent=2))


if __name__ == "__main__":
    main()
