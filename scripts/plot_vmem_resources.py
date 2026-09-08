#!/usr/bin/env python
"""Plot measured resource traces, or visualize a frozen commanded-path protocol.

No trace data are synthesized. Failed attempts can be included explicitly;
they are not treated as completed quality comparisons.
"""

import argparse
import csv
import json
from pathlib import Path


def read_steps(path):
    result = []
    truncated = False
    with path.open() as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                truncated = True
                break
            if event.get("event") == "step":
                result.append(event)
    return result, truncated


def write_csv(path, records):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in records for key in row}))
        writer.writeheader()
        writer.writerows(records)


def plot_paths(lock, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for case, details in lock["path_validation"].items():
        path = details["commanded_path"]
        axes[0].plot([r["current_pose"][0][3] for r in path], [r["current_pose"][2][3] for r in path], label=case)
        axes[1].plot([r["time_seconds"] for r in path], [r["max_radius"] for r in path], label=case)
    axes[0].set(xlabel="Commanded x (Navigator units)", ylabel="Commanded z (Navigator units)")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[1].set(xlabel="Video time (s)", ylabel="Maximum commanded radius (Navigator units)")
    axes[1].legend(fontsize=6)
    figure.suptitle("Commanded paths only; generated geometric coverage is unmeasured")
    figure.tight_layout()
    figure.savefig(output / "commanded_paths.png", dpi=180)
    plt.close(figure)


def plot_resources(inventory, output, include_incomplete, checkpoints):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    attempts = [arm for pair in inventory["pairs"]
                if pair["validated_pair"] and pair.get("uninterrupted_resource_pair", True)
                for arm in pair["arms"]]
    if include_incomplete:
        attempts.extend(item for item in inventory["attempts"]
                        if item.get("provenance_verified") and not item.get("resumed")
                        and item["status"] in {"failed", "unfinished", "paused"})
    groups, flat, checkpoint_rows = {}, [], []
    for attempt in attempts:
        path = Path(attempt["run_dir"]) / "resource_trace.jsonl"
        if not path.exists():
            continue
        steps, truncated = read_steps(path)
        if not steps:
            continue
        groups.setdefault(attempt["case_id"], []).append((attempt, steps))
        for step in steps:
            memory = step["after_update"]
            entry = {
                "case_id": attempt["case_id"], "run_id": attempt["run_id"],
                "run_dir": attempt["run_dir"], "status": attempt["status"],
                "step": step["step"], "video_seconds": step["video_seconds"],
                "warmup": step["warmup"], "truncated_trace": truncated,
                "eligible_frames": memory["eligible_frames"], "surfel_count": memory["surfel_count"],
                "surfel_reference_count": memory["surfel_reference_count"],
                "component_bytes_estimate": memory["estimated_component_bytes"],
                "accounting_seconds": step["accounting_seconds"],
                "reconstruction_input_count": len(step["reconstruction_input_indices"]),
            }
            for key in ("process_rss_bytes", "cuda_allocated_bytes", "cuda_reserved_bytes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
                entry[key] = memory.get(key)
            entry.update({name + "_seconds": seconds for name, seconds in step["phase_seconds"].items()})
            entry.update({name + "_estimated_bytes": value["estimated_bytes"] for name, value in memory["components"].items()})
            flat.append(entry)
        for target in checkpoints:
            sample = next((step for step in steps if step["video_seconds"] >= target), None)
            if sample is not None:
                checkpoint_rows.append({
                    "case_id": attempt["case_id"], "run_id": attempt["run_id"], "status": attempt["status"],
                    "requested_seconds": target, "sampled_seconds": sample["video_seconds"],
                    "component_bytes_estimate": sample["after_update"]["estimated_component_bytes"],
                    "process_rss_bytes": sample["after_update"].get("process_rss_bytes"),
                    "retrieval_seconds": sample["phase_seconds"]["retrieval"],
                    "warmup": sample["warmup"],
                })
    for case, curves in groups.items():
        memory_figure, axes = plt.subplots(1, 3, figsize=(13, 4))
        latency_figure, latency_axis = plt.subplots(figsize=(7, 4))
        for attempt, steps in curves:
            label = attempt["policy"] + (" (incomplete)" if attempt["status"] != "validated" else "")
            style = "-" if attempt["status"] == "validated" else "--"
            x = [step["video_seconds"] for step in steps]
            rss = [step["after_update"].get("process_rss_bytes") for step in steps]
            axes[0].plot(x, [value / 2**20 if value is not None else float("nan") for value in rss], style, label=label)
            payload = [sum(value["estimated_bytes"] for key, value in step["after_update"]["components"].items()
                           if not key.startswith("weights_")) / 2**20 for step in steps]
            axes[1].plot(x, payload, style, label=label)
            for key, suffix in (("cuda_allocated_bytes", "allocated"), ("cuda_reserved_bytes", "reserved")):
                values = [step["after_update"].get(key) for step in steps]
                axes[2].plot(x, [value / 2**20 if value is not None else float("nan") for value in values],
                             "--" if suffix == "reserved" else style, label=label + " " + suffix)
            steady = [step for step in steps if not step["warmup"]]
            latency_axis.plot([step["video_seconds"] for step in steady],
                              [step["phase_seconds"]["retrieval"] * 1000 for step in steady], style, label=label)
            warmup = [step for step in steps if step["warmup"]]
            latency_axis.scatter([step["video_seconds"] for step in warmup],
                                 [step["phase_seconds"]["retrieval"] * 1000 for step in warmup], marker="x", label=label + " warm-up")
        for axis, title in zip(axes, ("Process RSS (MiB)", "Tracked persistent estimate, excluding weights (MiB)", "PyTorch CUDA allocator (MiB)")):
            axis.set(xlabel="Video duration (s)", ylabel=title)
            axis.legend(fontsize=6)
        memory_figure.suptitle(case)
        memory_figure.tight_layout()
        memory_figure.savefig(output / f"{case}_memory_vs_duration.png", dpi=180)
        plt.close(memory_figure)
        latency_axis.set(xlabel="Video duration (s)", ylabel="Synchronized retrieval wall time (ms)", title=case)
        latency_axis.legend(fontsize=7)
        latency_figure.tight_layout()
        latency_figure.savefig(output / f"{case}_retrieval_latency_vs_duration.png", dpi=180)
        plt.close(latency_figure)
    write_csv(output / "resource_steps.csv", flat)
    write_csv(output / "duration_checkpoints.csv", checkpoint_rows)
    (output / "availability.json").write_text(json.dumps({
        "plotted_cases": len(groups), "measured_steps": len(flat),
        "quality_results": "pending separate user evaluation", "incomplete_included": include_incomplete,
        "resumed_pairs_excluded": sum(pair["validated_pair"] and not pair.get("uninterrupted_resource_pair", True)
                                      for pair in inventory["pairs"]),
    }, indent=2))
    print(f"Plotted {len(groups)} measured cases. No plot is created for missing data.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lock", type=Path, help="Plot commanded paths before generation.")
    group.add_argument("--inventory", type=Path, help="Plot validated measured resource data.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--checkpoints", type=float, nargs="+", default=[10, 20, 30, 60, 120, 180])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.lock:
        plot_paths(json.loads(args.lock.read_text()), args.output)
    else:
        plot_resources(json.loads(args.inventory.read_text()), args.output, args.include_incomplete, args.checkpoints)


if __name__ == "__main__":
    main()
