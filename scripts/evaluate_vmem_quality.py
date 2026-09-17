#!/usr/bin/env python
"""Evaluate validated VMem pairs with six no-reference VBench-Long dimensions.

Run on CECSL in the existing VBench environment. Inputs are copied unchanged;
all derived clips, logs and metric results stay outside generation directories.
"""

import argparse
import csv
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.run_vmem_vbench_long import DIMENSIONS, finite


POLICIES = ("unbounded", "slam_covisibility")
CONSISTENCY = ("subject_consistency", "background_consistency")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def report_failed_log(path):
    """Surface the child traceback without reading a whole evaluation log."""
    print(f"Evaluator log: {path} (last 80 lines, up to 16 KiB)", file=sys.stderr, flush=True)
    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 16384))
            tail = handle.read(16384).decode("utf-8", errors="replace")
        print("\n".join(tail.splitlines()[-80:]) or "(log is empty)", file=sys.stderr, flush=True)
    except OSError as exc:
        print(f"Could not read evaluator log: {exc}", file=sys.stderr, flush=True)


def repo_path(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def select_pairs(inventory, case_id=None):
    if inventory.get("schema") != "vmem_inventory_v1":
        raise ValueError("Use the inventory produced by audit_vmem_runs.py")
    selected, seen = [], set()
    for pair in inventory["pairs"]:
        case = pair["case_id"]
        if case_id is not None and case != case_id:
            continue
        if not pair["validated_pair"]:
            if case_id is not None:
                raise ValueError(f"Case has no validated pair: {case}")
            continue
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case) or case in seen:
            raise ValueError(f"Invalid or duplicate case ID: {case}")
        seen.add(case)
        arms = pair["arms"]
        if (len(arms) != 2 or any(arm is None or arm["status"] != "validated" for arm in arms)
                or {arm["policy"] for arm in arms} != set(POLICIES)):
            raise ValueError(f"Expected one validated arm per policy: {case}")
        if arms[0]["video"] != arms[1]["video"]:
            raise ValueError(f"Paired video formats/durations differ: {case}")
        selected.extend({**arm, "case_id": case} for arm in sorted(arms, key=lambda arm: POLICIES.index(arm["policy"])))
    if not selected:
        raise ValueError("No validated matched cases selected")
    return selected


def probe_video(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-show_entries",
        "stream=codec_type,width,height,avg_frame_rate,nb_read_frames:format=duration",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    if result.stderr.strip():
        raise ValueError(f"Video decoding errors: {path}: {result.stderr}")
    payload = json.loads(result.stdout)
    stream = next(item for item in payload["streams"] if item["codec_type"] == "video")
    return {"frames": int(stream["nb_read_frames"]), "width": stream["width"],
            "height": stream["height"], "fps": float(Fraction(stream["avg_frame_rate"])),
            "seconds": float(payload["format"]["duration"])}


def clip_plan(video):
    """Upstream's 2s custom-input split, including its overlapping tail clip."""
    fps, frames = video["fps"], video["frames"]
    if not finite(fps) or fps <= 0 or not float(fps).is_integer():
        raise ValueError("This protocol requires integer source fps (VMem uses 13)")
    size = 2 * int(fps)
    if not isinstance(frames, int) or frames < 2 * size:
        raise ValueError("At least two complete 2-second clips are required")
    ranges = [(start, start + size) for start in range(0, frames - size + 1, size)]
    if frames % size:
        ranges.append((frames - size, frames))
    return [{"clip_index": index, "start_frame": start, "end_frame_exclusive": end,
             "start_seconds": start / fps, "end_seconds": end / fps,
             "overlapping_tail": bool(index == len(ranges) - 1 and frames % size)}
            for index, (start, end) in enumerate(ranges)]


def verify_clips(stage, video):
    clips_dir = stage / "split_clip" / "video-0"
    plan = clip_plan(video)
    expected = {f"video-0_{row['clip_index']:03d}.mp4" for row in plan}
    if {path.name for path in clips_dir.glob("*.mp4")} != expected:
        raise ValueError(f"Unexpected split coverage: {clips_dir}")
    for row in plan:
        path = clips_dir / f"video-0_{row['clip_index']:03d}.mp4"
        actual = probe_video(path)
        if (actual["frames"] != row["end_frame_exclusive"] - row["start_frame"]
                or any(actual[key] != video[key] for key in ("fps", "width", "height"))):
            raise ValueError(f"Clip format/length mismatch: {path}")
        row.update(path=str(path), sha256=sha256(path))
    return plan


def vbench_sources(root):
    paths = [path for folder in ("vbench", "vbench2_beta_long")
             for path in (root / folder).rglob("*")
             if path.is_file() and path.suffix in {".py", ".yaml", ".json"}]
    if not (root / "vbench2_beta_long/eval_long.py").is_file() or not paths:
        raise ValueError(f"No VBench-Long checkout at {root}")
    return {str(path.relative_to(root)): sha256(path) for path in sorted(paths)}


def verify_evaluator_sources(spec, output):
    if (vbench_sources(Path(spec["vbench_root"])) != spec["vbench_source_sha256"]
            or sha256(__file__) != spec["runner_sha256"]
            or sha256(Path(__file__).with_name("run_vmem_vbench_long.py")) != spec["adapter_sha256"]):
        reason = "Evaluator source changed during evaluation; do not use these scores"
        write_json(output / "invalid.json", {"reason": reason})
        raise ValueError(reason)


def parse_result(path, dimension, staged_video, plan):
    payload = read_json(path)
    if set(payload) != {dimension}:
        raise ValueError(f"Wrong dimensions in {path}")
    result = payload[dimension]
    expected_length = 2 if dimension in CONSISTENCY else 3
    if not isinstance(result, list) or len(result) != expected_length or not finite(result[0]):
        raise ValueError(f"Invalid VBench-Long result: {path}")
    videos = result[-1]
    if (len(videos) != 1 or Path(videos[0]["video_path"]).resolve() != staged_video.resolve()
            or not finite(videos[0]["video_results"])):
        raise ValueError(f"Expected exactly the staged original video in {path}")
    scale = 100 if dimension == "imaging_quality" else 1
    if not math.isclose(result[0], videos[0]["video_results"] / scale, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(f"Aggregate/video score disagreement: {path}")
    clips = []
    if expected_length == 3:
        expected = {str(Path(row["path"]).resolve()): row for row in plan}
        seen = set()
        for detail in result[1]:
            clip_path = str(Path(detail["video_path"]).resolve())
            score = detail["video_results"]
            valid_score = finite(score) or (dimension == "dynamic_degree" and isinstance(score, bool))
            if clip_path not in expected or clip_path in seen or not valid_score:
                raise ValueError(f"Missing/duplicate/invalid clip score in {path}: {clip_path}")
            seen.add(clip_path)
            clips.append({**expected[clip_path], "score": score / scale})
        if seen != set(expected):
            raise ValueError(f"Incomplete clip scores: {path}")
    return result[0], clips


def verify_full_info(path, dimension, plan):
    rows = read_json(path)
    if len(rows) != 1 or rows[0]["dimension"] != [dimension]:
        raise ValueError(f"Unexpected evaluator input selection: {path}")
    selected = [str(Path(item).resolve()) for item in rows[0]["video_list"]]
    expected = {str(Path(row["path"]).resolve()) for row in plan}
    if len(selected) != len(expected) or set(selected) != expected:
        raise ValueError(f"Evaluator did not select every expected clip exactly once: {path}")


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(output, display=True):
    if (output / "invalid.json").exists():
        raise ValueError(read_json(output / "invalid.json")["reason"])
    spec = read_json(output / "evaluation_spec.json")
    jobs = read_json(output / "jobs.json")
    scores, clip_rows = {}, []
    for job in jobs:
        if job["status"] != "complete":
            continue
        result_path = Path(job["result"])
        if sha256(result_path) != job["result_sha256"]:
            raise ValueError(f"Result changed after evaluation: {result_path}")
        item = spec["inputs"][job["input_index"]]
        coverage_path = output / item["case_id"] / item["policy"] / "clip_coverage.json"
        if "clip_coverage_sha256" in job and sha256(coverage_path) != job["clip_coverage_sha256"]:
            raise ValueError(f"Clip coverage changed after evaluation: {coverage_path}")
        plan = read_json(coverage_path)
        if "full_info" in job:
            if sha256(job["full_info"]) != job["full_info_sha256"]:
                raise ValueError(f"Evaluator input selection changed: {job['full_info']}")
            verify_full_info(job["full_info"], job["dimension"], plan)
        score, clips = parse_result(result_path, job["dimension"], Path(item["staged_video"]), plan)
        scores[item["case_id"], item["policy"], job["dimension"]] = score
        clip_rows.extend({**row, "case_id": item["case_id"], "policy": item["policy"],
                          "dimension": job["dimension"]} for row in clips)
    paired = []
    for case in dict.fromkeys(item["case_id"] for item in spec["inputs"]):
        for dimension in spec["dimensions"]:
            left, right = [scores.get((case, policy, dimension)) for policy in POLICIES]
            paired.append({"case_id": case, "dimension": dimension,
                           "unbounded": left, "geocov32": right,
                           "geocov_minus_unbounded": None if left is None or right is None else right - left,
                           "interpretation": "motion amount, not a quality ranking" if dimension == "dynamic_degree" else "higher is better",
                           "status": "complete" if left is not None and right is not None else "incomplete"})
    write_json(output / "summary.json", {"schema": "vmem_quality_summary_v1", "paired": paired,
               "completed_jobs": sum(job["status"] == "complete" for job in jobs), "total_jobs": len(jobs),
               "note": "No overall score or GT fidelity claim; cases, not clips, are paired experimental units."})
    write_csv(output / "paired_scores.csv", paired,
              ["case_id", "dimension", "unbounded", "geocov32", "geocov_minus_unbounded", "interpretation", "status"])
    write_csv(output / "clip_scores.csv", clip_rows,
              ["case_id", "policy", "dimension", "clip_index", "start_frame", "end_frame_exclusive",
               "start_seconds", "end_seconds", "overlapping_tail", "score", "path", "sha256"])
    if display:
        print(f"{'case':36} {'dimension':24} {'unbounded':>11} {'GeoCov32':>11} {'delta':>11}")
        for row in paired:
            values = ["--" if row[key] is None else f"{row[key]:.6f}"
                      for key in ("unbounded", "geocov32", "geocov_minus_unbounded")]
            print(f"{row['case_id']:36} {row['dimension']:24} " + " ".join(f"{value:>11}" for value in values))
        print(f"Raw scores, not percentages. Dynamic degree is not a quality ranking. Results: {output}")
    return paired


def run(args):
    inventory_path, root, output = args.inventory.resolve(), args.vbench_root.expanduser().resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Keep prior attempts intact; choose a new output directory: {output}")
    inventory = read_json(inventory_path)
    selected = select_pairs(inventory, args.case_id)
    sources = vbench_sources(root)
    if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
        raise ValueError("ffprobe and ffmpeg must be on PATH in the VBench environment")
    if len(set(args.dimensions)) != len(args.dimensions):
        raise ValueError("Duplicate dimensions")
    with tempfile.TemporaryDirectory(prefix="vmem-quality-preflight-") as tmp:
        report = Path(tmp) / "runtime.json"
        print("Checking VBench imports and CPU video encoding before staging inputs...", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name("run_vmem_vbench_long.py")),
                        "--vbench-root", str(root), "--check", "--check-report", str(report),
                        "--dimension", *args.dimensions], cwd=root, check=True)
        runtime = read_json(report)
        if runtime.get("status") != "passed" or runtime.get("dimensions") != args.dimensions:
            raise ValueError("Evaluator preflight did not pass the requested dimensions")
    inputs = []
    for arm in selected:
        run_dir = repo_path(arm["run_dir"])
        if run_dir == output or run_dir in output.parents:
            raise ValueError("Keep quality outputs outside the original generation directories")
        video = run_dir / "generated.mp4"
        actual = probe_video(video)
        if actual != arm["video"]:
            raise ValueError(f"Video format changed since inventory: {video}")
        clip_plan(actual)
        inputs.append({"case_id": arm["case_id"], "policy": arm["policy"], "run_dir": str(run_dir),
                       "source_video": str(video), "video_sha256": sha256(video), "video": actual,
                       "generation_provenance": read_json(run_dir / "run_spec.json")["provenance"],
                       "staged_video": str(output / arm["case_id"] / arm["policy"] / "input" / "video-0.mp4")})
    output.mkdir(parents=True, exist_ok=False)
    spec = {"schema": "vmem_vbench_long_v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "inventory": str(inventory_path), "inventory_sha256": sha256(inventory_path),
            "vbench_root": str(root), "vbench_source_sha256": sources,
            "runner_sha256": sha256(__file__), "adapter_sha256": sha256(Path(__file__).with_name("run_vmem_vbench_long.py")),
            "python": sys.version, "executable": sys.executable,
            "evaluator_runtime": runtime,
            "packages": sorted((dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions() if dist.metadata["Name"]),
            "cache_environment": {key: os.environ.get(key) for key in ("VBENCH_CACHE_DIR", "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "HOME")},
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "mode": "long_custom_input", "dev_flag": True, "semantic_splitting": False,
            "imaging_quality_preprocessing_mode": "longer", "evaluation_seed": 0,
            "dimensions": args.dimensions, "inputs": inputs}
    write_json(output / "evaluation_spec.json", spec)
    jobs = [{"input_index": index, "dimension": dimension, "status": "pending"}
            for dimension in args.dimensions for index in range(len(inputs))]
    write_json(output / "jobs.json", jobs)
    for item in inputs:
        staged = Path(item["staged_video"])
        staged.parent.mkdir(parents=True)
        shutil.copy2(item["source_video"], staged)
        if sha256(staged) != item["video_sha256"]:
            raise ValueError(f"Staged video copy differs: {staged}")
    for job in jobs:
        verify_evaluator_sources(spec, output)
        item = inputs[job["input_index"]]
        arm_dir = output / item["case_id"] / item["policy"]
        result_dir = arm_dir / job["dimension"]
        result_dir.mkdir()
        command = [sys.executable, str(Path(__file__).with_name("run_vmem_vbench_long.py")),
                   "--vbench-root", str(root), "--videos_path", str(Path(item["staged_video"]).parent),
                   "--dimension", job["dimension"], "--mode", "long_custom_input", "--dev_flag",
                   "--imaging_quality_preprocessing_mode", "longer", "--output_path", str(result_dir)]
        log = result_dir / "evaluate.log"
        job.update(status="running", command=command, log=str(log))
        write_json(output / "jobs.json", jobs)
        print(f"{item['case_id']} {item['policy']} {job['dimension']}: {log}", flush=True)
        try:
            with log.open("x") as handle:
                subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT, check=True)
            verify_evaluator_sources(spec, output)
            if sha256(item["staged_video"]) != item["video_sha256"] or sha256(item["source_video"]) != item["video_sha256"]:
                raise ValueError("Original or staged video changed during evaluation")
            plan = verify_clips(Path(item["staged_video"]).parent, item["video"])
            coverage_path = arm_dir / "clip_coverage.json"
            if coverage_path.exists() and read_json(coverage_path) != plan:
                raise ValueError("Split clips changed between dimensions")
            write_json(coverage_path, plan)
            candidates = list(result_dir.glob("*_eval_results.json"))
            infos = list(result_dir.glob("*_full_info.json"))
            if len(candidates) != 1 or len(infos) != 1:
                raise ValueError(f"Expected exactly one metric result and input selection in {result_dir}")
            verify_full_info(infos[0], job["dimension"], plan)
            parse_result(candidates[0], job["dimension"], Path(item["staged_video"]), plan)
            job.update(status="complete", result=str(candidates[0]), result_sha256=sha256(candidates[0]),
                       full_info=str(infos[0]), full_info_sha256=sha256(infos[0]),
                       clip_coverage_sha256=sha256(coverage_path))
        except Exception as exc:
            job.update(status="failed", error=str(exc))
            write_json(output / "jobs.json", jobs)
            report_failed_log(log)
            try:
                summarize(output)
            except Exception as summary_error:
                print(f"Partial summary unavailable: {summary_error}", file=sys.stderr, flush=True)
            raise
        write_json(output / "jobs.json", jobs)
        summarize(output, display=False)
    verify_evaluator_sources(spec, output)
    summarize(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("run")
    evaluate.add_argument("--inventory", type=Path, required=True)
    evaluate.add_argument("--vbench-root", type=Path, default=Path.home() / "VBench")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--case-id")
    evaluate.add_argument("--dimensions", nargs="+", choices=DIMENSIONS, default=list(DIMENSIONS))
    report = sub.add_parser("summarize")
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        summarize(args.output.resolve())


if __name__ == "__main__":
    main()
