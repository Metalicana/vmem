"""Portable, data-backed HTML/CSV report for the VMem results controller."""

import csv
from html import escape
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import zipfile

POLICIES = ("unbounded", "slam_covisibility")
LABELS = {"unbounded": "Unbounded", "slam_covisibility": "GeoCov-32"}
DIMENSIONS = ("aesthetic_quality", "imaging_quality", "subject_consistency",
              "background_consistency", "motion_smoothness", "dynamic_degree")


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def copy_if_changed(source, target):
    source = Path(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if (not target.exists() or target.stat().st_size != source.stat().st_size
            or target.stat().st_mtime_ns != source.stat().st_mtime_ns):
        shutil.copy2(source, target)


def number(value, digits=6):
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def scaled(value, divisor):
    return value / divisor if value is not None else None


def prefix_passed(value):
    return bool(value and not value.get("provenance_mismatches") and not value.get("missing_provenance")
                and not value.get("pre_eviction_divergences")
                and value.get("saved_frame_pixels", {}).get("file_hashes_verified")
                and value.get("saved_frame_pixels", {}).get("prefix_pixels_equal") is True
                and value.get("retrieval_checks")
                and all(not row["illegal_selection_steps"] for row in value["retrieval_checks"]))


def resource_summary(arm, metadata, steps):
    last = steps[-1]["after_update"] if steps else {}
    payload = last.get("frame_payloads", {})
    steady = [row for row in steps if not row["warmup"]]
    peak = [row["after_update"].get("cuda_peak_allocated_bytes") for row in steps]
    peak.append(metadata.get("cuda_peak_allocated_bytes"))
    peak = [value for value in peak if value is not None]
    result = {"case_id": arm["case_id"], "policy": arm["policy"], "run_dir": arm["run_dir"],
            "resumed": bool(arm.get("resumed")), "eligible_frames": last.get("eligible_frames"),
            "resident_rgb_frames": payload.get("resident_counts", {}).get("pil_frames"),
            "frame_payload_mib": scaled(payload.get("total_logical_bytes"), 2**20),
            "surfel_count": last.get("surfel_count"), "surfel_references": last.get("surfel_reference_count"),
            "max_recorded_cuda_peak_gib": max(peak) / 2**30 if peak else None,
            "final_process_rss_gib": last.get("process_rss_bytes", 0) / 2**30 if last.get("process_rss_bytes") else None,
            "current_process_wall_hours": scaled(metadata.get("wall_seconds_including_load_and_export"), 3600),
            "median_retrieval_ms_excluding_warmup": statistics.median(
                row["phase_seconds"]["retrieval"] * 1000 for row in steady) if steady else None,
            "steady_updates": len(steady), "measured_updates": len(steps)}
    for phase in ("reconstruction", "generation", "memory_update", "payload_eviction"):
        values = [row["phase_seconds"][phase] for row in steady if phase in row["phase_seconds"]]
        result["median_" + phase + "_seconds"] = statistics.median(values) if values else None
    result.update({"resident_" + name: count for name, count in payload.get("resident_counts", {}).items()})
    for name, key in (("max_recorded_cuda_reserved_peak_gib", "cuda_peak_reserved_bytes"),
                      ("max_sampled_process_rss_gib", "process_rss_bytes")):
        values = [row["after_update"].get(key) for row in steps]
        values.append(metadata.get(key))
        values = [value for value in values if value is not None]
        result[name] = max(values) / 2**30 if values else None
    return result


def revisit_results(arm):
    from evaluate_vmem_revisits import find_returns, image_metrics, read_frame, retrieval_for_frame
    path = Path(arm["run_dir"])
    metadata = read(path / "metadata.json")
    returns = find_returns(read(path / "actions.json"))
    trace = read(path / "retrieval_trace.json")
    anchor = read_frame(path, 0, "png")
    return [{"case_id": arm["case_id"], "policy": arm["policy"],
             "time_seconds": row["frame_index"] / metadata["fps"], **row,
             **image_metrics(anchor, read_frame(path, row["frame_index"], "png")),
             **retrieval_for_frame(trace, row["frame_index"])} for row in returns]


def plot_case(case, curves, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(13, 7))
    styles = {"unbounded": "#0072B2", "slam_covisibility": "#D55E00"}
    definitions = (
        ("Resident RGB frames", lambda row: row["after_update"].get("frame_payloads", {}).get("resident_counts", {}).get("pil_frames")),
        ("Frame payloads (MiB)", lambda row: scaled(row["after_update"].get("frame_payloads", {}).get("total_logical_bytes"), 2**20)),
        ("CUDA allocated (GiB)", lambda row: scaled(row["after_update"].get("cuda_allocated_bytes"), 2**30)),
        ("Retrieval (ms)", lambda row: row["phase_seconds"]["retrieval"] * 1000),
        ("Reconstruction (s)", lambda row: row["phase_seconds"]["reconstruction"]),
        ("Generation (s)", lambda row: row["phase_seconds"]["generation"]),
    )
    for policy, steps in curves:
        sessions = {}
        for row in steps:
            sessions.setdefault(row.get("session_id"), []).append(row)
        for session_index, rows in enumerate(sessions.values()):
            for axis, (label, getter) in zip(axes.flat, definitions):
                data = rows if label.endswith("frames") or "payloads" in label or "CUDA" in label else [r for r in rows if not r["warmup"]]
                if data:
                    axis.plot([r["video_seconds"] for r in data],
                              [float("nan") if getter(r) is None else getter(r) for r in data],
                              color=styles[policy], label=LABELS[policy] if session_index == 0 else None)
            if session_index:
                for axis in axes.flat:
                    axis.axvline(rows[0]["video_seconds"], color=styles[policy], linestyle=":", alpha=.5)
    for axis, (label, _) in zip(axes.flat, definitions):
        axis.set(xlabel="Generated video seconds", ylabel=label)
        axis.grid(alpha=.15)
        if axis.lines:
            axis.legend(fontsize=8)
    figure.suptitle(case + "\nSession boundaries dotted; latency warm-up excluded; no restart-spanning speedup claim", fontsize=10)
    figure.tight_layout()
    figure.savefig(target, dpi=160)
    plt.close(figure)


def quality_rows(root, state, pair, report):
    from evaluate_vmem_quality import sha256, summarize, verify_evaluator_sources
    output = []
    paths = state.get("metric_attempts", {}).get(pair["case_id"], {})
    errors = []
    for dimension in DIMENSIONS:
        empty = {"case_id": pair["case_id"], "dimension": dimension, "unbounded": None,
                 "geocov32": None, "geocov_minus_unbounded": None, "status": "pending"}
        attempt = paths.get(dimension)
        if attempt is None:
            output.append(empty)
            continue
        path = Path(attempt)
        try:
            spec = read(path / "evaluation_spec.json")
            if spec is None or spec["dimensions"] != [dimension]:
                raise ValueError("No matching evaluation spec")
            verify_evaluator_sources(spec, path)
            expected = {arm["policy"]: Path(arm["run_dir"]).resolve() for arm in pair["arms"]}
            if len(spec["inputs"]) != 2:
                raise ValueError("Metric inputs do not form a pair")
            for item in spec["inputs"]:
                if (item["case_id"] != pair["case_id"] or expected.get(item["policy"]) != Path(item["run_dir"]).resolve()
                        or sha256(item["source_video"]) != item["video_sha256"]
                        or sha256(item["staged_video"]) != item["video_sha256"]):
                    raise ValueError("Metric input differs from validated generation")
            rows = summarize(path, display=False)
            if len(rows) != 1 or rows[0]["case_id"] != pair["case_id"]:
                raise ValueError("Unexpected metric cases")
            output.append(rows[0])
            target = report / "raw" / pair["case_id"] / dimension
            for file in ("evaluation_spec.json", "jobs.json", "summary.json"):
                copy_if_changed(path / file, target / file)
            for job in read(path / "jobs.json"):
                if job.get("result"):
                    copy_if_changed(job["result"], target / (f"input_{job['input_index']}_result.json"))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            output.append({**empty, "status": "failed"})
            errors.append({"stage": dimension, "error": str(exc), "attempt": str(path)})
    return output, errors


CSS = """
:root{font-family:system-ui,sans-serif;color:#202427;background:#f5f6f7;letter-spacing:0}
*{box-sizing:border-box}body{margin:0}main{max-width:1250px;margin:auto;padding:24px}
header{border-bottom:3px solid #267265;padding-bottom:16px}h1{font-size:27px;margin:8px 0}h2{font-size:20px}
h3{font-size:16px}.muted,small{color:#59656a}.status{color:#267265;font-weight:650}
section{padding:22px 0;border-bottom:1px solid #d0d7db}.videos{display:grid;grid-template-columns:1fr 1fr;gap:20px}
figure{margin:0}figcaption{padding:8px 0;font-weight:600}video{display:block;width:100%;aspect-ratio:1;background:#161819}
table{width:100%;border-collapse:collapse;font-size:14px;background:white}th,td{padding:10px;border-bottom:1px solid #e0e5e6;text-align:right;font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}th{background:#edf0f1;font-weight:600}.scroll{overflow-x:auto}
.positive{color:#176249}.negative{color:#ad3434}.warning{color:#a13a28}a{color:#17677d}
.plot{width:100%;height:auto;display:block;margin:16px 0}.facts{display:flex;gap:24px;flex-wrap:wrap;margin:12px 0}
.facts span{font-size:14px}details{padding:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
@media(max-width:650px){main{padding:14px}.videos{grid-template-columns:1fr}h1{font-size:23px}td,th{padding:8px}}
"""


def build_report(root):
    from plot_vmem_resources import read_steps
    root = Path(root)
    report = root / "report"
    report.mkdir(parents=True, exist_ok=True)
    state = read(root / "workflow_status.json", {})
    protocol = read(root / "workflow_spec.json", {})
    generation_only = protocol.get("dimensions") == []
    inventory = read(root / "inventory.json", {"pairs": [], "attempts": []})
    all_scores, resources, revisits, errors, sections = [], [], [], list(state.get("errors", [])), []
    resource_steps = []
    pairing_checks = {}
    for pair in inventory["pairs"]:
        case = pair["case_id"]
        if not case.replace("_", "").isalnum():
            raise ValueError("Unsafe report case ID")
        body = [f"<section><h2>{escape(case)}</h2>"]
        if not pair["validated_pair"]:
            body.append("<p class='warning'>Generation pair incomplete. No paired quality result.</p></section>")
            sections.append("".join(body))
            continue
        pairing = read(root / "pairing" / (case + ".json"))
        passed = prefix_passed(pairing)
        pairing_checks[case] = passed
        body.append(f"<p class='{'status' if passed else 'warning'}'>Pre-eviction pairing: "
                    + ("passed" if passed else "pending or failed") + "</p><div class='videos'>")
        curves, case_resources, case_revisits = [], [], []
        for arm in pair["arms"]:
            policy = arm["policy"]
            if policy not in POLICIES:
                raise ValueError("Unexpected policy")
            path = Path(arm["run_dir"])
            metadata = read(path / "metadata.json")
            stem = f"{case}_{policy}"
            video = f"media/{stem}.mp4"
            poster = f"media/{stem}.png"
            copy_if_changed(path / "generated.mp4", report / video)
            copy_if_changed(path / "generated_frames/0000.png", report / poster)
            for file in ("metadata.json", "run_spec.json", "generation_config.yaml", "actions.json"):
                copy_if_changed(path / file, report / "raw" / case / policy / file)
            body.append(f"<figure><figcaption>{LABELS[policy]}{' / resumed' if arm.get('resumed') else ''}</figcaption>"
                        f"<video controls preload='metadata' poster='{poster}' src='{video}'></video></figure>")
            steps, truncated = read_steps(path / "resource_trace.jsonl")
            if truncated:
                raise ValueError(f"Validated resource trace is truncated: {path}")
            curves.append((policy, steps))
            for step in steps:
                memory = step["after_update"]
                resource_steps.append({"case_id": case, "policy": policy, "step": step["step"],
                    "video_seconds": step["video_seconds"], "session_id": step.get("session_id"), "warmup": step["warmup"],
                    "eligible_frames": memory.get("eligible_frames"), "surfel_count": memory.get("surfel_count"),
                    "surfel_reference_count": memory.get("surfel_reference_count"),
                    "process_rss_bytes": memory.get("process_rss_bytes"),
                    "cuda_allocated_bytes": memory.get("cuda_allocated_bytes"),
                    "cuda_reserved_bytes": memory.get("cuda_reserved_bytes"),
                    **{name + "_seconds": value for name, value in step["phase_seconds"].items()},
                    **{name + "_estimated_bytes": value["estimated_bytes"] for name, value in memory.get("components", {}).items()},
                    **{"resident_" + name: value for name, value in memory.get("frame_payloads", {}).get("resident_counts", {}).items()}})
            from profile_vmem_newton import action_profile
            case_resources.append({**resource_summary(arm, metadata, steps),
                                   **action_profile(read(path / "actions.json"), steps)})
            case_revisits.extend(revisit_results(arm))
        resources.extend(case_resources)
        revisits.extend(case_revisits)
        scores, quality_errors = ([], []) if generation_only else quality_rows(root, state, pair, report)
        all_scores.extend(scores)
        errors.extend(quality_errors)
        body.append("</div>")
        if not generation_only:
            body.append("<h3>VBench-Long</h3><div class='scroll'><table><thead><tr><th>Dimension</th>"
                        "<th>Unbounded</th><th>GeoCov-32</th><th>GeoCov minus unbounded</th><th>Status</th></tr></thead><tbody>")
            for row in scores:
                delta = row["geocov_minus_unbounded"]
                color = "" if delta is None or row["dimension"] == "dynamic_degree" else "positive" if delta > 0 else "negative" if delta < 0 else ""
                body.append(f"<tr><td>{escape(row['dimension'].replace('_', ' ').title())}</td>"
                            f"<td>{number(row['unbounded'])}</td><td>{number(row['geocov32'])}</td>"
                            f"<td class='{color}'>{number(delta)}</td><td>{escape(row['status'])}</td></tr>")
            body.append("</tbody></table></div><p class='muted'>Raw scores, not percentages. Higher is better except dynamic degree, "
                        "which measures motion amount. No composite score.</p>")
        body.append("<h3>Recorded Resources</h3><div class='scroll'><table>"
                    "<tr><th>Measurement</th><th>Unbounded</th><th>GeoCov-32</th></tr>")
        by_policy = {row["policy"]: row for row in case_resources}
        for key, label, digits in (
            ("resident_rgb_frames", "Final resident RGB frames", 0),
            ("frame_payload_mib", "Final frame payloads (MiB)", 2),
            ("surfel_count", "Final surfels", 0),
            ("max_recorded_cuda_peak_gib", "Max recorded PyTorch allocated peak (GiB)", 2),
            ("max_recorded_cuda_reserved_peak_gib", "Max recorded PyTorch reserved peak (GiB)", 2),
            ("max_sampled_process_rss_gib", "Max sampled post-update process RSS (GiB, not a peak)", 2),
            ("final_process_rss_gib", "Final process RSS (GiB)", 2),
            ("median_retrieval_ms_excluding_warmup", "Median retrieval, warm-up excluded (ms)", 2),
            ("median_action_seconds_excluding_warmup", "Median action, warm-up excluded (s)", 2),
            ("last_three_median_action_seconds", "Last three actions, median (s)", 2),
            ("first_action_seconds", "First action, including warm-up (s)", 2),
            ("current_process_wall_hours", "Current process elapsed (hours, not full resumed-run time)", 2),
        ):
            body.append(f"<tr><td>{label}</td>" + "".join(f"<td>{number(by_policy[p][key], digits)}</td>" for p in POLICIES) + "</tr>")
        body.append("</table></div><p class='muted'>Process/allocator measurements, not whole-GPU usage. "
                    "Action times exclude frame-save/checkpoint I/O and load/export. "
                    "Resumed traces include inherited steps; allocator sessions differ. No uninterrupted speedup claim.</p>")
        plot = f"{case}_resources.png"
        plot_case(case, curves, report / plot)
        body.append(f"<img class='plot' src='{plot}' alt='Measured memory and stage times versus video duration'>")
        body.append("<h3>Commanded Revisits</h3><div class='scroll'><table><tr><th>Policy</th><th>Returns</th>"
                    "<th>Mean RGB MSE</th><th>Last RGB MSE</th><th>Anchor selection fraction</th></tr>")
        for policy in POLICIES:
            values = [row for row in case_revisits if row["policy"] == policy]
            mean = statistics.mean(row["rgb_mse"] for row in values) if values else None
            selected = [row["anchor_selected"] for row in values if "anchor_selected" in row]
            body.append(f"<tr><td>{LABELS[policy]}</td><td>{len(values)}</td><td>{number(mean)}</td>"
                        f"<td>{number(values[-1]['rgb_mse'] if values else None)}</td>"
                        f"<td>{number(statistics.mean(selected) if selected else None, 3)}</td></tr>")
        body.append("</table></div><p class='muted'>RGB MSE on [0,1] images at commanded returns to the initial pose; "
                    "lower is better. This is self-consistency, not camera-verified GT reconstruction fidelity.</p></section>")
        sections.append("".join(body))
    quality_fields = ["case_id", "dimension", "unbounded", "geocov32", "geocov_minus_unbounded", "status"]
    write_csv(report / "quality.csv", all_scores, quality_fields)
    write_csv(report / "resources.csv", resources, sorted({key for row in resources for key in row}) or ["case_id", "policy"])
    write_csv(report / "resource_steps.csv", resource_steps,
              sorted({key for row in resource_steps for key in row}) or ["case_id", "policy", "step"])
    write_csv(report / "revisits.csv", revisits, sorted({key for row in revisits for key in row}) or ["case_id", "policy"])
    summary = {"workflow_status": state.get("status", "pending"), "quality": all_scores, "resources": resources,
               "quality_requested": not generation_only,
               "pairing_checks": pairing_checks,
               "revisits": revisits, "errors": errors,
               "statistical_scope": "One seed per case; no seed-variance CI or significance claim. Oxford pilot is one paired case.",
               "claim_scope": "Audited-fork VMem end-to-end memory controller; not a same-budget competitor comparison or upstream reproduction."}
    (report / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    for file in ("workflow_spec.json", "workflow_status.json", "inventory.json", "events.jsonl", "vbench_preflight.json", "newton_profile.json"):
        if (root / file).exists():
            copy_if_changed(root / file, report / "raw" / file)
    for path in (root / "pairing").glob("*.json"):
        copy_if_changed(path, report / "raw/pairing" / path.name)
    count = sum(pair["validated_pair"] for pair in inventory["pairs"])
    complete = sum(row["status"] == "complete" for row in all_scores)
    durations = sorted({row.get("num_actions", 195) for row in protocol.get("selected_rows", [])})
    duration_label = ", ".join(str(value) + " actions" for value in durations) or "See run metadata for duration"
    body = ["<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
            f"<title>VMem Results</title><style>{CSS}</style></head><body><main><header><h1>VMem / Controlled Transfer</h1>",
            f"<p class='status'>{escape(state.get('status', 'pending').upper())} / {escape(state.get('stage', 'pending'))}</p>",
            f"<div class='facts'><span>{count}/{len(inventory['pairs'])} validated pairs</span><span>{complete} completed metric pairs</span>",
            f"<span>{len(errors)} reported failures</span><span>{escape(duration_label)}</span></div>",
            "<p class='muted'>Generation-only smoke: quality evaluation not requested. This is not a 60-second quality result.</p>" if generation_only else "",
            "<a href='quality.csv'>Quality CSV</a> &nbsp; <a href='resources.csv'>Resources CSV</a> &nbsp; "
            "<a href='revisits.csv'>Revisit CSV</a> &nbsp; <a href='summary.json'>All results JSON</a></header>", *sections,
            "<section><h2>Scope and Limitations</h2><p>One seed per case. No confidence interval or statistical superiority claim "
            "from this pilot. Clips and revisit frames are not independent trials. Single-image rollouts have no exact-index GT "
            "for novel views; GT LPIPS, FVD and CUT3R accuracy are not reported.</p><p>GeoCov changes archive retention and reconstruction "
            "inputs. B=32 bounds post-update resident frame payloads, not surfels, transient reconstruction, total RAM/VRAM or disk output. "
            "Paths test constrained revisits, not sustained exploration. Results describe our audited VMem fork.</p></section>",
            "<section><h2>Attempts and Failures</h2>"]
    for error in errors:
        body.append(f"<p class='warning'>{escape(error['stage'])}: {escape(error['error'])}</p>")
    body.append("<details><summary>All recorded generation attempts</summary><pre>" + escape(
        json.dumps(inventory["attempts"], indent=2)) + "</pre></details></section></main></body></html>")
    temporary = report / "index.html.tmp"
    temporary.write_text("".join(body))
    temporary.replace(report / "index.html")
    # Avoid repeatedly packaging videos while metrics are still running.
    if state.get("status") in {"complete", "incomplete", "failed", "cancelled", "paused"}:
        for folder in ("logs", "history"):
            for path in (root / folder).glob("*"):
                if path.is_file():
                    copy_if_changed(path, report / "raw" / folder / path.name)
        temporary = root / "results.zip.tmp"
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(report.rglob("*")):
                if path.is_file():
                    archive.write(path, str(Path("report") / path.relative_to(report)),
                                  compress_type=zipfile.ZIP_STORED if path.suffix == ".mp4" else zipfile.ZIP_DEFLATED)
        os.replace(temporary, root / "results.zip")
    return summary
