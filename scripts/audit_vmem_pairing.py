#!/usr/bin/env python
"""Read saved run records to locate divergence before memory eviction.

No models, video decoding, checkpoint loading, or generation are involved.
This supplements the inventory validator; it is not a quality metric or a
claim of deterministic generation.
"""

import argparse
import json
from pathlib import Path


SETTINGS = (
    "seed", "fps", "num_actions", "trajectory", "pattern", "step_size",
    "frames_per_action", "memory_scope", "frame_storage", "inference_steps",
    "surfel_niter", "surfel_reconstruction_window", "visualize_intermediates",
)
PROVENANCE = ("image_sha256", "config_sha256", "source_sha256", "checkpoint_sha256")


def read_json(path):
    return json.loads(path.read_text())


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return {"index": index, "unbounded": a, "bounded": b}
    if len(left) != len(right):
        index = min(len(left), len(right))
        return {"index": index,
                "unbounded": left[index] if index < len(left) else None,
                "bounded": right[index] if index < len(right) else None}
    return None


def read_run(path):
    spec = read_json(path / "run_spec.json")
    retrieval = read_json(path / "retrieval_trace.json")
    actions = read_json(path / "actions.json")
    resources = []
    with (path / "resource_trace.jsonl").open() as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event") == "step":
                after = event["after_update"]
                resources.append({
                    "step": event["step"],
                    "reconstruction_input_indices": event["reconstruction_input_indices"],
                    "surfel_count": after["surfel_count"],
                    "appearance_descriptor_sources": after.get("appearance_descriptor_sources"),
                    "resident_counts": after.get("frame_payloads", {}).get("resident_counts"),
                })
    count = spec["arguments"]["num_actions"]
    if (count < 1 or len(actions) != count or len(retrieval) != count
            or len(resources) != count):
        raise ValueError(f"Expected complete action/retrieval/resource records: {path}")
    for rows, key in ((actions, "action_index"), (retrieval, "global_step"), (resources, "step")):
        if [row[key] for row in rows] != list(range(count)):
            raise ValueError(f"Missing, duplicate or unordered {key}: {path}")
    frames = 1 + count * spec["arguments"]["frames_per_action"]
    manifest = path / "frame_manifest.json"
    hashes = None
    if manifest.exists():
        data = read_json(manifest)
        hashes = data["sha256"]
        if data.get("schema") != "vmem_output_frames_v1" or len(hashes) != frames:
            raise ValueError(f"Unexpected frame manifest: {manifest}")
        if any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
               for h in hashes):
            raise ValueError(f"Malformed frame hashes: {manifest}")
    evictions = [row for row in read_json(path / "memory_trace.json")
                 if row.get("event") == "memory_eviction"]
    return {"spec": spec, "retrieval": retrieval, "actions": actions,
            "resources": resources, "hashes": hashes, "evictions": evictions}


def compare_runs(unbounded, bounded):
    left, right = read_run(unbounded), read_run(bounded)
    a, b = left["spec"]["arguments"], right["spec"]["arguments"]
    if a["memory_policy"] != "unbounded" or b["memory_policy"] != "slam_covisibility":
        raise ValueError("Expected an unbounded / slam_covisibility pair, in that order")
    if not isinstance(b["memory_budget"], int) or b["memory_budget"] < 2:
        raise ValueError("Expected a bounded bank with B >= 2")
    mismatches = [f"arguments.{key}" for key in SETTINGS if a.get(key) != b.get(key)]
    missing = [f"arguments.{key}" for key in SETTINGS if key not in a or key not in b]
    for key in PROVENANCE:
        va, vb = [run["spec"].get("provenance", {}).get(key) for run in (left, right)]
        if not va or not vb:
            missing.append(f"provenance.{key}")
        elif va != vb:
            mismatches.append(f"provenance.{key}")
    for key in ("torch_version", "cuda_version"):
        if key not in left["spec"] or key not in right["spec"]:
            missing.append(key)
        elif left["spec"][key] != right["spec"][key]:
            mismatches.append(key)

    first_eviction = min(right["evictions"], key=lambda row: row["global_step"], default=None)
    eviction_step = first_eviction["global_step"] if first_eviction is not None else None
    if eviction_step is not None and not 0 <= eviction_step < len(right["actions"]):
        raise ValueError("Eviction step lies outside the action trace")

    differences = {}
    for name, key in (("context", "selected_context_indices"),
                      ("eligible_bank", "allowed_memory_indices"),
                      ("geometry_at_retrieval", "num_retained_surfels")):
        differences[name] = first_difference(
            [row[key] for row in left["retrieval"]], [row[key] for row in right["retrieval"]])
    for name, key in (("reconstruction_inputs", "reconstruction_input_indices"),
                      ("geometry_after_update", "surfel_count")):
        differences[name] = first_difference(
            [row[key] for row in left["resources"]], [row[key] for row in right["resources"]])
    differences["actions"] = first_difference(
        [{key: row[key] for key in ("action", "current_pose", "num_generated_frames", "total_frames_after")}
         for row in left["actions"]],
        [{key: row[key] for key in ("action", "current_pose", "num_generated_frames", "total_frames_after")}
         for row in right["actions"]])

    early = []
    for key, diff in differences.items():
        if diff is None:
            continue
        # Retrieval/reconstruction happen before eviction in the same action;
        # after-update geometry is sampled after pruning, so exclude that step.
        if eviction_step is None or diff["index"] < eviction_step or (
                key != "geometry_after_update" and diff["index"] == eviction_step):
            early.append(key)

    hash_report = {"status": "unavailable", "note": "Copy both frame_manifest.json files to compare saved PNG hashes."}
    if left["hashes"] is not None and right["hashes"] is not None:
        diff = first_difference(left["hashes"], right["hashes"])
        # Frames generated by the first evicting action also precede eviction.
        prefix_count = (len(right["hashes"]) if eviction_step is None
                        else right["actions"][eviction_step]["total_frames_after"])
        prefix_diff = first_difference(left["hashes"][:prefix_count], right["hashes"][:prefix_count])
        hash_report = {"status": "compared", "first_difference": diff,
                       "frames_compared_before_first_eviction": prefix_count,
                       "prefix_hashes_equal": prefix_diff is None,
                       "note": "Recorded encoded-PNG hashes, not decoded pixels; files are not rehashed by this audit."}
        if prefix_diff is not None:
            early.append("saved_frame_hashes")

    checks = []
    for run in (left, right):
        rows = run["retrieval"]
        checks.append({
            "policy": run["spec"]["arguments"]["memory_policy"],
            "fallback_steps": [row["global_step"] for row in rows if row["fallback_used"]],
            "illegal_selection_steps": [row["global_step"] for row in rows
                                        if set(row["selected_context_indices"]) - set(row["allowed_memory_indices"])],
            "context_slot_counts_after_first_step": sorted({len(row["selected_context_indices"]) for row in rows[1:]}),
            "initial_frame_missing_steps": [row["global_step"] for row in rows if 0 not in row["allowed_memory_indices"]],
            "descriptor_sources": [dict(value) for value in sorted({
                tuple(sorted((row["appearance_descriptor_sources"] or {}).items())) for row in run["resources"]})],
        })
    return {
        "schema": "vmem_pairing_audit_v1", "unbounded": str(unbounded), "bounded": str(bounded),
        "provenance_mismatches": mismatches, "missing_provenance": missing,
        "first_eviction": None if first_eviction is None else {
            key: first_eviction[key] for key in ("global_step", "section_end_frame", "evicted_memory_frame")},
        "first_differences": differences, "pre_eviction_divergences": early,
        "saved_frames": hash_report, "retrieval_checks": checks,
        "interpretation": "Timing/provenance diagnostics only. Same seed is not proof of matched noise; differences do not establish their cause.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unbounded", type=Path, required=True)
    parser.add_argument("--bounded", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare_runs(args.unbounded, args.bounded), indent=2))


if __name__ == "__main__":
    main()
