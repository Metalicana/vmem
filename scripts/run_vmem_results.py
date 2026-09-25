#!/usr/bin/env python
"""Resume frozen VMem generation, validate, evaluate, report on CECSL or Slurm.

No generation implementation is modified. All children are sequential; commands
and failures are durable. GPU admission is a check, not a reservation.
"""

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
DIMENSIONS = ("aesthetic_quality", "imaging_quality", "subject_consistency",
              "background_consistency", "motion_smoothness", "dynamic_degree")


def dimensions(args):
    return () if getattr(args, "generation_only", False) else DIMENSIONS


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def process_token(pid):
    try:
        return Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, ValueError, IndexError, TypeError):
        return None


def process_args(pid):
    try:
        return Path(f"/proc/{int(pid)}/cmdline").read_bytes().decode().strip("\0").split("\0")
    except (OSError, ValueError, TypeError):
        return []


def live_run(path):
    status_path = path / "run_status.json"
    if not status_path.exists():
        return False
    args = process_args(read(status_path).get("pid"))
    spec = read(path / "run_spec.json")
    return (any(Path(arg).name == "run_vmem_demo_actions.py" for arg in args)
            and "--run-id" in args
            and args[args.index("--run-id") + 1] == spec["arguments"]["run_id"])


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another controller owns {path}; use status, not a duplicate launch") from exc
        # Closing releases the lock when no inherited descriptor still owns it.
        yield handle


@contextmanager
def controller_guard(args):
    path = args.output / "controller.lock"
    if args.lock_fd is None:
        with exclusive(path) as handle:
            yield handle
    else:
        with os.fdopen(args.lock_fd, "a+") as handle:
            if os.fstat(handle.fileno()).st_ino != path.stat().st_ino:
                raise ValueError("Inherited controller lock does not match output")
            yield handle


def generation_plan(row, generation_root, lock_path, lock):
    from audit_vmem_runs import inspect_attempt
    from vmem_protocol import verify_lock
    paths = sorted(generation_root.glob(row["run_id"] + "_*"))
    valid, resumable = [], []
    for path in paths:
        if not (path / "run_spec.json").exists():
            raise ValueError(f"Attempt without provenance requires inspection: {path}")
        spec = read(path / "run_spec.json")
        verify_lock(lock_path, spec["arguments"], spec["provenance"])
        if spec["provenance"].get("experiment_lock_sha256") != digest(lock_path):
            raise ValueError(f"Cannot resume under a different lock: {path}")
        if live_run(path):
            raise RuntimeError(f"A generation process is already running: {path}")
        item = inspect_attempt(path, row, lock_path, lock)
        if item["status"] == "validated":
            valid.append(path)
        elif (path / "run_status.json").exists() and read(path / "run_status.json").get("status") == "complete":
            raise ValueError(f"Completed attempt failed validation; not silently rerunning: {item}")
        elif (path / "recovery/latest.pt").is_file():
            resumable.append(path)
    if valid:
        return "reuse", max(valid, key=lambda path: (path / "metadata.json").stat().st_mtime_ns)
    if resumable:
        return "resume", max(resumable, key=lambda path: (path / "recovery/latest.pt").stat().st_mtime_ns)
    if paths:
        raise ValueError(f"No recovery checkpoint for {row['run_id']}; preserved attempts need manual review")
    return "new", None


def query_gpu(gpu):
    result = subprocess.run([
        "nvidia-smi", "-i", str(gpu), "--query-gpu=uuid,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1:
        raise ValueError("Select exactly one GPU index or UUID")
    uuid, free, total, utilization = [value.strip() for value in lines[0].split(",")]
    processes = subprocess.run([
        "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return {"uuid": uuid, "free_mib": int(free), "total_mib": int(total),
            "utilization_percent": int(utilization),
            "compute_processes": [line for line in processes.stdout.strip().splitlines() if line.strip()]}


def gpu_ready(snapshot, minimum):
    return (not snapshot["compute_processes"] and snapshot["free_mib"] >= minimum
            and snapshot["utilization_percent"] <= 5)


def quality_attempt_matches(output, pair, dimension, vbench_root, *, runtime=None, packages=None, executable=None):
    from evaluate_vmem_quality import summarize, verify_evaluator_sources
    spec = read(output / "evaluation_spec.json")
    if spec["dimensions"] != [dimension] or Path(spec["vbench_root"]).resolve() != vbench_root:
        return False
    if executable is not None and spec.get("executable") != str(executable):
        return False
    if packages is not None and spec.get("packages") != packages:
        return False
    if runtime is not None:
        recorded = {key: value for key, value in spec.get("evaluator_runtime", {}).items() if key != "dimensions"}
        current = {key: value for key, value in runtime.items() if key != "dimensions"}
        if recorded != current:
            return False
    expected = {arm["policy"]: Path(arm["run_dir"]).resolve() for arm in pair["arms"]}
    if len(spec["inputs"]) != 2:
        return False
    for item in spec["inputs"]:
        if (item["case_id"] != pair["case_id"] or expected.get(item["policy"]) != Path(item["run_dir"]).resolve()
                or digest(item["source_video"]) != item["video_sha256"]
                or digest(item["staged_video"]) != item["video_sha256"]):
            return False
    verify_evaluator_sources(spec, output)
    rows = summarize(output, display=False)
    return len(rows) == 1 and rows[0]["status"] == "complete"


class Workflow:
    def __init__(self, args):
        self.args, self.root = args, args.output
        self.child = None
        self.state = {"schema": "vmem_results_workflow_v1", "status": "running", "stage": "startup",
                      "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                      "pid": os.getpid(), "process_token": process_token(os.getpid()),
                      "started_at": stamp(), "gpu": args.gpu, "errors": [], "metric_attempts": {},
                      "case_results": {}, "generation_root": str(args.generation_root)}
        previous = self.root / "workflow_status.json"
        if previous.exists():
            save(self.root / "history" / f"status_{stamp()}.json", read(previous))

    def update(self, **values):
        self.state.update(values, updated_at=stamp())
        save(self.root / "workflow_status.json", self.state)

    def event(self, **values):
        with (self.root / "events.jsonl").open("a") as handle:
            handle.write(json.dumps({"at": stamp(), **values}, allow_nan=False) + "\n")

    def stop_child(self):
        if self.child is None or self.child.poll() is not None:
            return
        try:
            os.killpg(self.child.pid, signal.SIGTERM)
        except ProcessLookupError:
            self.child.wait()
            return
        try:
            self.child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.child.wait()

    def command(self, name, command, *, gpu=False):
        from vmem_slurm import child_environment
        if gpu:
            self.wait_gpu(name)
            if self.args.gpu == "inherit":
                command = [command[0], REPO / "scripts/vmem_slurm.py", "--execute", *command[1:]]
        log = self.root / "logs" / f"{stamp()}_{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = child_environment(self.args.gpu, gpu)
        environment.update(PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
        # Use binaries from the selected environment without changing the user's shell.
        environment["PATH"] = str(Path(command[0]).parent) + os.pathsep + environment.get("PATH", "")
        prefix = Path(command[0]).parent.parent
        environment["CONDA_PREFIX"] = str(prefix)
        environment["CONDA_DEFAULT_ENV"] = prefix.name
        print(f"[{name}] {shlex.join(map(str, command))}\n  log: {log}", flush=True)
        self.update(stage=name, active_log=str(log), active_command=list(map(str, command)))
        started = time.monotonic()
        with log.open("x") as handle:
            self.child = subprocess.Popen(list(map(str, command)), cwd=REPO, env=environment,
                                          stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                          start_new_session=True)
            self.update(child_pid=self.child.pid, child_process_token=process_token(self.child.pid))
            try:
                code = self.child.wait()
            finally:
                self.stop_child()
                code = self.child.returncode
                self.event(stage=name, command=list(map(str, command)), log=str(log), returncode=code,
                           wall_seconds=time.monotonic() - started)
                self.child = None
                self.update(child_pid=None, child_process_token=None)
        if code:
            with log.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                handle.seek(max(0, handle.tell() - 8192))
                print(handle.read().decode(errors="replace"), flush=True)
            raise RuntimeError(f"{name} exited {code}; see {log}")
        return log

    def wait_gpu(self, stage):
        if self.args.gpu == "inherit":
            from vmem_slurm import query_allocated_gpu, check_capacity
            snapshot = query_allocated_gpu()
            check_capacity(snapshot, self.args.min_free_mib)
            self.update(stage=f"allocated_gpu:{stage}", gpu_snapshot=snapshot)
            self.event(stage="gpu_admission", next_stage=stage, **snapshot)
            return
        while True:
            snapshot = query_gpu(self.args.gpu)
            if self.args.min_free_mib > snapshot["total_mib"]:
                raise ValueError("--min-free-mib exceeds selected GPU capacity")
            self.update(stage=f"waiting_for_gpu:{stage}", gpu_snapshot=snapshot)
            if gpu_ready(snapshot, self.args.min_free_mib):
                self.event(stage="gpu_admission", next_stage=stage, **snapshot)
                return
            print(f"Waiting for GPU {self.args.gpu}: {snapshot['free_mib']} MiB free, "
                  f"{len(snapshot['compute_processes'])} compute processes. No other jobs will be killed.", flush=True)
            time.sleep(self.args.poll_seconds)

    def error(self, stage, error):
        message = {"stage": stage, "error": str(error), "at": stamp()}
        self.state["errors"].append(message)
        self.event(**message)
        self.update()
        print(f"FAILED: {stage}: {error}", flush=True)

    def refresh_inventory(self):
        self.command("inventory", [sys.executable, REPO / "scripts/audit_vmem_runs.py", "inventory",
                     "--lock", self.args.lock, "--output-root", self.args.generation_root,
                     "--output", self.root / "inventory_all.json"])
        inventory = read(self.root / "inventory_all.json")
        ids = {row["run_id"] for row in self.rows}
        cases = {row["_case_id"] for row in self.rows}
        inventory["pairs"] = [pair for pair in inventory["pairs"] if pair["case_id"] in cases]
        inventory["attempts"] = [arm for arm in inventory["attempts"] if arm["run_id"] in ids]
        save(self.root / "inventory.json", inventory)
        return inventory

    def report(self):
        from report_vmem_results import build_report
        result = build_report(self.root)
        if self.state["status"] == "complete":
            expected = len(self.rows) // 2 * len(dimensions(self.args))
            if (result["errors"] or len(result["quality"]) != expected
                    or any(row["status"] != "complete" for row in result["quality"])
                    or len(result["pairing_checks"]) != len(self.rows) // 2
                    or not all(result["pairing_checks"].values())):
                self.error("report_integrity", "Final artifact verification is incomplete; inspect the report")
                self.update(status="incomplete")
                build_report(self.root)

    def prepare(self):
        from run_vmem_demo_actions import _generation_provenance
        from run_vmem_demo_manifest import _load_manifest
        from vmem_protocol import expected_settings, verify_lock
        from vmem_slurm import environment_identity
        if dimensions(self.args) and not self.args.vbench_python.is_file():
            raise FileNotFoundError(f"VBench Python not found: {self.args.vbench_python}")
        for binary in ("ffmpeg", "ffprobe", "nvidia-smi"):
            if shutil.which(binary) is None:
                raise ValueError(f"Missing executable: {binary}")
        all_rows = [{k: v for k, v in row.items() if k != "_manifest_line"}
                    for row in _load_manifest(self.args.manifest)]
        self.rows = all_rows if self.args.all_cases else all_rows[:2]
        if not self.args.lock.exists():
            if any(self.args.generation_root.glob("*/run_spec.json")):
                raise ValueError("Existing attempts need their original lock; refusing to create a replacement")
            self.command("freeze", [sys.executable, REPO / "scripts/audit_vmem_runs.py", "freeze",
                         self.args.manifest, "--output", self.args.lock])
        self.lock = read(self.args.lock)
        if self.lock["rows"] != all_rows:
            raise ValueError("Existing lock does not contain the selected frozen manifest")
        for row in self.rows:
            provenance = _generation_provenance(Path(row["image"]), Path(self.lock["config"]))
            verify_lock(self.args.lock, expected_settings(row), provenance)
        protocol = {"schema": "vmem_results_protocol_v1", "lock": str(self.args.lock),
                    "lock_sha256": digest(self.args.lock), "selected_rows": self.rows,
                    "generation_root": str(self.args.generation_root), "vbench_root": str(self.args.vbench_root),
                    "vmem_python": sys.executable, "vbench_python": str(self.args.vbench_python),
                    "dimensions": list(dimensions(self.args)), "manifest_sha256": digest(self.args.manifest),
                    "vmem_environment": environment_identity(),
                    "workflow_sources": {file: digest(REPO / "scripts" / file) for file in (
                        "run_vmem_results.py", "report_vmem_results.py", "evaluate_vmem_quality.py",
                        "run_vmem_vbench_long.py", "audit_vmem_runs.py", "audit_vmem_pairing.py",
                        "evaluate_vmem_revisits.py", "vmem_slurm.py", "profile_vmem_newton.py")}}
        path = self.root / "workflow_spec.json"
        if path.exists() and read(path) != protocol:
            raise ValueError("Workflow settings/source changed; use a new --output, preserving this report")
        save(path, protocol)
        self.command("cpu_tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
        self.command("report_dependencies", [sys.executable, "-c", "import matplotlib, numpy, PIL, omegaconf, torch; print('imports OK')"])
        if dimensions(self.args):
            self.command("vbench_preflight", [self.args.vbench_python, REPO / "scripts/run_vmem_vbench_long.py",
                         "--vbench-root", self.args.vbench_root, "--check", "--check-report", self.root / "vbench_preflight.json"])
            log = self.command("vbench_packages", [self.args.vbench_python, "-c",
                "import importlib.metadata as m, json; print(json.dumps(sorted((d.metadata['Name'], d.version) for d in m.distributions() if d.metadata['Name'])))"])
            save(self.root / "vbench_packages.json", read(log))
        # Reject incompatible/live recovery sources before waiting for a GPU.
        for row in self.rows:
            mode, path = generation_plan(row, self.args.generation_root, self.args.lock, self.lock)
            print(f"{row['run_id']}: {mode}" + (f" {path}" if path else ""), flush=True)
        self.refresh_inventory()
        self.report()

    def generate(self, row):
        from run_vmem_demo_manifest import _command_for_row
        mode, path = generation_plan(row, self.args.generation_root, self.args.lock, self.lock)
        self.event(stage="generation_plan", run_id=row["run_id"], mode=mode, source=str(path) if path else None)
        if mode == "reuse":
            return
        command = _command_for_row(row, output_root=self.args.generation_root,
                                  config=Path(self.lock["config"]), device="cuda", dry_run=False,
                                  experiment_lock=self.args.lock, resume_from=path)
        self.command("generate_" + row["run_id"], command, gpu=True)

    def evaluate_pair(self, pair):
        from audit_vmem_pairing import compare_runs
        case = pair["case_id"]
        arms = {arm["policy"]: Path(arm["run_dir"]) for arm in pair["arms"]}
        pairing = compare_runs(arms["unbounded"], arms["slam_covisibility"], compare_pixels=True)
        save(self.root / "pairing" / (case + ".json"), pairing)
        if (pairing["provenance_mismatches"] or pairing["missing_provenance"]
                or pairing["pre_eviction_divergences"] or not pairing["saved_frame_pixels"]["prefix_pixels_equal"]):
            self.error(case + "_pairing", "Pre-eviction pairing check failed; quality is diagnostic, not a clean paired effect")
        if not dimensions(self.args):
            return
        runtime = {"runtime": read(self.root / "vbench_preflight.json"),
                   "packages": read(self.root / "vbench_packages.json"), "executable": self.args.vbench_python}
        for dimension in dimensions(self.args):
            stage = case + "_" + dimension
            base = self.root / "quality" / case / dimension
            result = None
            for attempt in sorted(base.glob("attempt_*"), reverse=True):
                try:
                    if quality_attempt_matches(attempt, pair, dimension, self.args.vbench_root, **runtime):
                        # Revalidate in the scoring environment, not just the controller environment.
                        self.command("verify_" + stage, [self.args.vbench_python, REPO / "scripts/evaluate_vmem_quality.py",
                                     "summarize", "--output", attempt])
                        result = attempt
                        break
                except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                    self.event(stage="quality_attempt_not_reused", attempt=str(attempt), error=str(exc))
            if result is None:
                result = base / ("attempt_" + stamp())
                base.mkdir(parents=True, exist_ok=True)
                try:
                    self.command(stage, [self.args.vbench_python, REPO / "scripts/evaluate_vmem_quality.py", "run",
                                 "--inventory", self.root / "inventory.json", "--case-id", case,
                                 "--vbench-root", self.args.vbench_root, "--dimensions", dimension,
                                 "--output", result], gpu=True)
                    if not quality_attempt_matches(result, pair, dimension, self.args.vbench_root, **runtime):
                        raise ValueError("Metric did not produce a verified completed pair")
                except Exception as exc:
                    self.error(stage, exc)
            self.state["metric_attempts"].setdefault(case, {})[dimension] = str(result)
            self.update()
            self.report()

    def run(self):
        self.update()
        self.prepare()
        for offset in range(0, len(self.rows), 2):
            for row in self.rows[offset:offset + 2]:
                try:
                    self.generate(row)
                except Exception as exc:
                    self.error(row["run_id"], exc)
                self.refresh_inventory()
                self.report()
            case = self.rows[offset]["_case_id"]
            pair = next(pair for pair in read(self.root / "inventory.json")["pairs"] if pair["case_id"] == case)
            if pair["validated_pair"]:
                try:
                    self.evaluate_pair(pair)
                except Exception as exc:
                    self.error(case, exc)
            else:
                self.error(case, "No validated pair; no complete paired quality result is possible")
        self.update(status="incomplete" if self.state["errors"] else "complete", stage="finished")
        self.report()
        if (self.state["status"] == "complete" and self.args.gpu == "inherit"
                and self.args.manifest.name == "vmem_newton_smoke_v1.jsonl"):
            from profile_vmem_newton import verify_smoke
            save(self.root / "newton_profile.json", verify_smoke(self.root))
            self.report()
        print(f"Report: {self.root / 'report/index.html'}\nBundle: {self.root / 'results.zip'}", flush=True)
        return 1 if self.state["errors"] else 0


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", nargs="?", choices=("start", "run", "status", "stop", "report"), default="start")
    cli.add_argument("--gpu", default="1")
    cli.add_argument("--all-cases", action="store_true", help="All 15 frozen cases; default is Oxford pan_45 only")
    cli.add_argument("--manifest", type=Path, default=REPO / "manifests/vmem_transfer_v3.jsonl")
    cli.add_argument("--generation-only", action="store_true", help="Validate/profile generation without quality metrics")
    cli.add_argument("--output", type=Path)
    cli.add_argument("--generation-root", type=Path, default=REPO / "outputs/vmem_transfer_v3_controlled")
    cli.add_argument("--lock", type=Path, default=REPO / "outputs/locks/transfer_v3_controlled.json")
    cli.add_argument("--vbench-root", type=Path, default=Path.home() / "VBench")
    cli.add_argument("--vbench-python", type=Path,
                     default=Path(sys.executable).parent.parent.parent / "vbench/bin/python")
    cli.add_argument("--min-free-mib", type=int, default=85000)
    cli.add_argument("--poll-seconds", type=int, default=60)
    cli.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    return cli


def status(args):
    path = args.output / "workflow_status.json"
    if not path.exists():
        print("No controller state yet. Check the controller log.")
        return
    state = read(path)
    if state.get("hostname", socket.gethostname()) != socket.gethostname():
        state["controller_alive"] = None
        state["liveness_note"] = "PID belongs to another host; use squeue/sacct for the recorded Slurm job"
    else:
        state["controller_alive"] = (state.get("process_token") is not None
                                     and process_token(state.get("pid")) == state["process_token"])
    print(json.dumps(state, indent=2))
    for path in sorted(args.generation_root.glob("*/actions.partial.jsonl")):
        rows = path.read_text().splitlines()
        if rows:
            print(f"{path.parent.name}: {rows[-1]}")


def main():
    args = parser().parse_args()
    if sys.platform != "linux":
        raise SystemExit("Run on CECSL or Newton/Linux, not on the Mac.")
    os.chdir(REPO)
    if args.min_free_mib <= 0 or args.poll_seconds <= 0:
        raise SystemExit("GPU threshold and poll interval must be positive")
    from vmem_slurm import check_launch
    check_launch(args.action, args.gpu)
    if args.gpu != "inherit" and not args.gpu.isdigit() and not args.gpu.startswith("GPU-"):
        raise SystemExit("Select one physical GPU index/UUID, or inherit under Slurm")
    if args.gpu == "inherit" and args.action in {"run", "start"}:
        if args.output is None or args.generation_root == REPO / "outputs/vmem_transfer_v3_controlled" or args.lock == REPO / "outputs/locks/transfer_v3_controlled.json":
            raise SystemExit("Slurm requires explicit new --output, --generation-root and --lock paths")
    args.output = (args.output or REPO / "outputs" / (
        "vmem_results_transfer60" if args.all_cases else "vmem_results_oxford60")).expanduser().resolve()
    for key in ("generation_root", "lock", "vbench_root", "vbench_python", "manifest"):
        setattr(args, key, getattr(args, key).expanduser().absolute())
    args.vbench_root = args.vbench_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == "status":
        status(args)
        return
    if args.action == "stop":
        state = read(args.output / "workflow_status.json")
        if state.get("slurm_job_id") or state.get("hostname", socket.gethostname()) != socket.gethostname():
            raise SystemExit(f"Use scancel for the Slurm job {state.get('slurm_job_id')}; no local PID was signalled")
        if not state.get("process_token") or process_token(state["pid"]) != state["process_token"]:
            raise SystemExit("Recorded controller is no longer alive; no signal sent")
        os.kill(state["pid"], signal.SIGTERM)
        print("Cancellation requested. Only this controller's active child group will be terminated.")
        return
    if args.action == "report":
        with exclusive(args.output / "controller.lock"):
            from report_vmem_results import build_report
            build_report(args.output)
        return
    with controller_guard(args) as controller_lock:
        if args.action == "start":
            log = args.output / "logs" / ("controller_" + stamp() + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "run",
                       "--gpu", args.gpu, "--output", str(args.output),
                       "--generation-root", str(args.generation_root), "--lock", str(args.lock),
                       "--manifest", str(args.manifest),
                       "--vbench-root", str(args.vbench_root), "--vbench-python", str(args.vbench_python),
                       "--min-free-mib", str(args.min_free_mib), "--poll-seconds", str(args.poll_seconds)]
            command.extend(["--lock-fd", str(controller_lock.fileno())])
            if args.all_cases:
                command.append("--all-cases")
            if args.generation_only:
                command.append("--generation-only")
            with log.open("x") as handle:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
                                           stderr=subprocess.STDOUT, start_new_session=True, cwd=REPO,
                                           pass_fds=(controller_lock.fileno(),))
            print(f"Started controller PID {process.pid}.\n"
                  f"Log: {log}\nReport: {args.output / 'report/index.html'}\n"
                  f"Progress: bash scripts/run_vmem_results.sh status --output {shlex.quote(str(args.output))}\n"
                  f"Cancel:   bash scripts/run_vmem_results.sh stop --output {shlex.quote(str(args.output))}")
            return
        with ExitStack() as stack:
            workflow = Workflow(args)
            def cancel(_signal, _frame):
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                signal.signal(signal.SIGUSR1, signal.SIG_IGN)
                workflow.state["interrupt_signal"] = _signal
                raise KeyboardInterrupt("Controller interrupted; retain committed recovery checkpoints")
            signal.signal(signal.SIGTERM, cancel)
            signal.signal(signal.SIGUSR1, cancel)
            try:
                workflow.update()
                stack.enter_context(exclusive(args.generation_root / ".results_controller.lock"))
                if args.gpu == "inherit":
                    from vmem_slurm import query_allocated_gpu, check_capacity
                    gpu = query_allocated_gpu()
                    check_capacity(gpu, args.min_free_mib)
                    workflow.event(stage="allocation", **gpu)
                    # Allocation is exclusive; do not inspect or wait on other node GPUs.
                else:
                    gpu = query_gpu(args.gpu)
                    stack.enter_context(exclusive(REPO / "outputs/locks" / ("results_" + gpu["uuid"] + ".lock")))
                code = workflow.run()
            except KeyboardInterrupt:
                workflow.stop_child()
                paused = workflow.state.get("interrupt_signal") == signal.SIGUSR1
                workflow.update(status="paused" if paused else "cancelled", stage="interrupted")
                workflow.report()
                code = 75 if paused else 130
            except Exception as exc:
                workflow.stop_child()
                workflow.error("controller", exc)
                workflow.update(status="failed")
                try:
                    workflow.report()
                except Exception as report_error:
                    print(f"Report also failed: {report_error}", flush=True)
                code = 1
            raise SystemExit(code)


if __name__ == "__main__":
    main()
