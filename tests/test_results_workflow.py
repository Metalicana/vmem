import csv
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from scripts import run_vmem_results as workflow
from scripts import report_vmem_results as report


class ResultsWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.lock = self.root / "lock.json"
        workflow.save(self.lock, {"rows": []})
        self.row = {"run_id": "transfer_v3_test_unbounded", "memory_policy": "unbounded"}

    def attempt(self, name="1", complete=False, checkpoint=True):
        path = self.root / (self.row["run_id"] + "_" + name)
        path.mkdir()
        workflow.save(path / "run_spec.json", {"arguments": self.row,
                      "provenance": {"experiment_lock_sha256": workflow.digest(self.lock)}})
        workflow.save(path / "run_status.json", {"status": "complete" if complete else "running", "pid": -1})
        if complete:
            workflow.save(path / "metadata.json", {})
        if checkpoint:
            (path / "recovery").mkdir()
            (path / "recovery/latest.pt").write_bytes(b"fixture; never unpickled")
        return path

    def plan(self, status="unfinished"):
        with patch("vmem_protocol.verify_lock"), patch("audit_vmem_runs.inspect_attempt", return_value={"status": status}):
            return workflow.generation_plan(self.row, self.root, self.lock, {})

    def test_new_then_resume_not_restart(self):
        self.assertEqual(self.plan(), ("new", None))
        path = self.attempt()
        before = (path / "run_spec.json").read_bytes()
        self.assertEqual(self.plan(), ("resume", path))
        self.assertEqual((path / "run_spec.json").read_bytes(), before)

    def test_complete_requires_validation_before_reuse(self):
        path = self.attempt(complete=True)
        with self.assertRaisesRegex(ValueError, "failed validation"):
            self.plan("invalid")
        self.assertEqual(self.plan("validated"), ("reuse", path))

    def test_live_job_is_not_duplicated_and_stale_running_is_resumable(self):
        path = self.attempt()
        with patch.object(workflow, "live_run", return_value=True), self.assertRaisesRegex(RuntimeError, "already running"):
            self.plan()
        self.assertEqual(self.plan(), ("resume", path))

    def test_wrong_lock_and_absent_checkpoint_are_not_silently_restarted(self):
        path = self.attempt(checkpoint=False)
        with self.assertRaisesRegex(ValueError, "No recovery checkpoint"):
            self.plan()
        spec = workflow.read(path / "run_spec.json")
        spec["provenance"]["experiment_lock_sha256"] = "wrong"
        workflow.save(path / "run_spec.json", spec)
        with self.assertRaisesRegex(ValueError, "different lock"):
            self.plan()

    def test_latest_committed_checkpoint_selected(self):
        old, new = self.attempt("old"), self.attempt("new")
        os.utime(old / "recovery/latest.pt", ns=(1, 1))
        os.utime(new / "recovery/latest.pt", ns=(2, 2))
        self.assertEqual(self.plan(), ("resume", new))

    def test_controller_lock_can_be_handed_to_detached_worker_without_gap(self):
        path = self.root / "controller.lock"
        with workflow.exclusive(path) as handle:
            inherited = os.dup(handle.fileno())
        try:
            with self.assertRaises(RuntimeError), workflow.exclusive(path):
                pass
            with workflow.controller_guard(SimpleNamespace(output=self.root, lock_fd=inherited)):
                inherited = None
                with self.assertRaises(RuntimeError), workflow.exclusive(path):
                    pass
        finally:
            if inherited is not None:
                os.close(inherited)
        with workflow.exclusive(path):
            pass

    def test_gpu_requires_no_compute_job_even_if_memory_is_free(self):
        data = {"compute_processes": [], "free_mib": 90000, "utilization_percent": 0}
        self.assertTrue(workflow.gpu_ready(data, 85000))
        for change in ({"compute_processes": ["123, 10"]}, {"free_mib": 84999}, {"utilization_percent": 99}):
            self.assertFalse(workflow.gpu_ready({**data, **change}, 85000))

    def test_gpu_queries_are_scoped_and_no_process_is_killed(self):
        outputs = [SimpleNamespace(stdout="GPU-123, 90000, 97887, 0\n"), SimpleNamespace(stdout="999, 1234\n")]
        with patch.object(workflow.subprocess, "run", side_effect=outputs) as run:
            data = workflow.query_gpu("1")
        self.assertEqual(data["compute_processes"], ["999, 1234"])
        for call in run.call_args_list:
            self.assertEqual(call.args[0][1:3], ["-i", "1"])

    def test_termination_only_targets_current_child_group(self):
        instance = workflow.Workflow(SimpleNamespace(output=self.root, gpu="1", generation_root=self.root))
        child = SimpleNamespace(pid=123, poll=lambda: None, wait=lambda timeout=None: 0)
        instance.child = child
        with patch.object(workflow.os, "killpg") as kill:
            instance.stop_child()
        kill.assert_called_once_with(123, workflow.signal.SIGTERM)

    def test_generation_failure_produces_incomplete_report_not_quality_success(self):
        instance = workflow.Workflow(SimpleNamespace(output=self.root, gpu="1", generation_root=self.root))
        instance.rows = [{"run_id": "left", "_case_id": "case"}, {"run_id": "right", "_case_id": "case"}]
        workflow.save(self.root / "inventory.json", {"pairs": [{"case_id": "case", "validated_pair": False}]})
        with patch.object(instance, "prepare"), patch.object(instance, "generate", side_effect=[RuntimeError("OOM"), None]), \
                patch.object(instance, "refresh_inventory"), patch.object(instance, "report"), \
                patch.object(instance, "evaluate_pair") as evaluate:
            self.assertEqual(instance.run(), 1)
            evaluate.assert_not_called()
        self.assertEqual(workflow.read(self.root / "workflow_status.json")["status"], "incomplete")

    def test_final_report_missing_scores_cannot_be_complete(self):
        instance = workflow.Workflow(SimpleNamespace(output=self.root, gpu="1", generation_root=self.root))
        instance.rows = [{}, {}]
        instance.state["status"] = "complete"
        with patch("report_vmem_results.build_report", return_value={"errors": [], "quality": [], "pairing_checks": {}}):
            instance.report()
        self.assertEqual(instance.state["status"], "incomplete")
        self.assertEqual(instance.state["errors"][0]["stage"], "report_integrity")


class MetricReuseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / "score"
        self.output.mkdir()
        self.pair = {"case_id": "case", "arms": []}
        inputs = []
        for policy in ("unbounded", "slam_covisibility"):
            run = self.root / policy
            run.mkdir()
            (run / "generated.mp4").write_bytes(b"video")
            staged = self.output / (policy + ".mp4")
            staged.write_bytes(b"video")
            self.pair["arms"].append({"policy": policy, "run_dir": str(run)})
            inputs.append({"case_id": "case", "policy": policy, "run_dir": str(run),
                           "source_video": str(run / "generated.mp4"), "staged_video": str(staged),
                           "video_sha256": workflow.digest(staged)})
        self.spec = {"dimensions": ["aesthetic_quality"], "vbench_root": str(self.root), "inputs": inputs,
                     "packages": [["torch", "test"]], "executable": "/env/vbench/bin/python",
                     "evaluator_runtime": {"torch": "test", "dimensions": ["aesthetic_quality"]}}
        workflow.save(self.output / "evaluation_spec.json", self.spec)

    def matches(self, status="complete", **kwargs):
        with patch("evaluate_vmem_quality.summarize", return_value=[{"status": status}]), \
                patch("evaluate_vmem_quality.verify_evaluator_sources"):
            return workflow.quality_attempt_matches(self.output, self.pair, "aesthetic_quality", self.root, **kwargs)

    def test_only_complete_unchanged_pair_is_reused(self):
        self.assertTrue(self.matches())
        self.assertFalse(self.matches(status="incomplete"))
        Path(self.spec["inputs"][0]["source_video"]).write_bytes(b"changed")
        self.assertFalse(self.matches())

    def test_different_generation_directory_is_rejected(self):
        self.pair["arms"][0]["run_dir"] = str(self.root / "different")
        self.assertFalse(self.matches())

    def test_scoring_environment_changes_are_rejected(self):
        self.assertTrue(self.matches(runtime={"torch": "test", "dimensions": list(workflow.DIMENSIONS)},
                                     packages=self.spec["packages"], executable=Path(self.spec["executable"])))
        self.assertFalse(self.matches(packages=[["torch", "changed"]]))
        self.assertFalse(self.matches(runtime={"torch": "changed"}))
        self.assertFalse(self.matches(executable=Path("/different/python")))


class ResultsReportTest(unittest.TestCase):
    def test_completed_pair_report_preserves_negative_scores_and_bundles_videos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arms = []
            for policy in report.POLICIES:
                path = root / policy
                (path / "generated_frames").mkdir(parents=True)
                (path / "generated.mp4").write_bytes(b"fixture video, decoding mocked")
                (path / "generated_frames/0000.png").write_bytes(b"fixture poster")
                for name in ("metadata.json", "run_spec.json", "generation_config.yaml", "actions.json"):
                    workflow.save(path / name, {})
                arms.append({"policy": policy, "case_id": "oxford", "run_dir": str(path), "resumed": True})
            workflow.save(root / "workflow_status.json", {"status": "complete", "errors": []})
            workflow.save(root / "inventory.json", {"pairs": [
                {"case_id": "oxford", "validated_pair": True, "arms": arms}], "attempts": arms})
            scores = [{"case_id": "oxford", "dimension": dimension, "unbounded": .6,
                       "geocov32": .5, "geocov_minus_unbounded": -.1, "status": "complete"}
                      for dimension in report.DIMENSIONS]
            step = {"step": 0, "video_seconds": 1, "warmup": True,
                    "phase_seconds": {"retrieval": .01, "reconstruction": 1, "generation": 2},
                    "after_update": {"frame_payloads": {"resident_counts": {"pil_frames": 32},
                                                        "total_logical_bytes": 1000}}}
            def plot(_case, _curves, path):
                path.write_bytes(b"fixture plot")
            with patch("plot_vmem_resources.read_steps", return_value=([step], False)), \
                    patch.object(report, "revisit_results", return_value=[]), \
                    patch.object(report, "quality_rows", return_value=(scores, [])), \
                    patch.object(report, "plot_case", side_effect=plot):
                summary = report.build_report(root)
            self.assertEqual(summary["quality"][0]["geocov_minus_unbounded"], -.1)
            html = (root / "report/index.html").read_text()
            self.assertEqual(html.count("<video controls"), 2)
            self.assertEqual(html.count("class='negative'"), 5)
            self.assertIn("no confidence interval", html.lower())
            self.assertIn("not full resumed-run time", html)
            with zipfile.ZipFile(root / "results.zip") as archive:
                self.assertEqual(len([name for name in archive.namelist() if name.endswith(".mp4")]), 2)
                self.assertIn("report/index.html", archive.namelist())

    def test_prefix_gate_requires_actual_pixel_verification(self):
        valid = {"provenance_mismatches": [], "missing_provenance": [], "pre_eviction_divergences": [],
                 "saved_frame_pixels": {"file_hashes_verified": True, "prefix_pixels_equal": True},
                 "retrieval_checks": [{"illegal_selection_steps": []}]}
        self.assertTrue(report.prefix_passed(valid))
        self.assertFalse(report.prefix_passed(None))
        self.assertFalse(report.prefix_passed({**valid, "pre_eviction_divergences": ["context"]}))
        self.assertFalse(report.prefix_passed({**valid, "saved_frame_pixels": {"prefix_pixels_equal": True}}))

    def test_resumed_resource_scope_and_missing_values(self):
        arm = {"case_id": "case", "policy": "unbounded", "run_dir": "run", "resumed": True}
        steps = [{"warmup": warmup, "phase_seconds": {"retrieval": latency, "reconstruction": 2},
                  "after_update": {"cuda_peak_allocated_bytes": peak}}
                 for warmup, latency, peak in ((True, 1, 8 * 2**30), (False, .01, 4 * 2**30))]
        row = report.resource_summary(arm, {"wall_seconds_including_load_and_export": 3600}, steps)
        self.assertEqual(row["median_retrieval_ms_excluding_warmup"], 10)
        self.assertEqual(row["max_recorded_cuda_peak_gib"], 8)
        self.assertEqual(row["current_process_wall_hours"], 1)
        self.assertTrue(row["resumed"])
        self.assertIsNone(row["frame_payload_mib"])
        self.assertIsNone(row["median_generation_seconds"])
        self.assertEqual(report.number(None), "--")

    def test_failure_report_is_portable_escaped_and_does_not_fabricate_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow.save(root / "workflow_status.json", {"status": "failed", "errors": [
                {"stage": "preflight", "error": "<script>not HTML</script>"}]})
            workflow.save(root / "inventory.json", {"pairs": [{"case_id": "oxford", "validated_pair": False}],
                                                    "attempts": [{"status": "failed"}]})
            summary = report.build_report(root)
            self.assertEqual(summary["quality"], [])
            html = (root / "report/index.html").read_text()
            self.assertNotIn("<script>not HTML", html)
            self.assertIn("&lt;script&gt;", html)
            self.assertIn("pair incomplete", html)
            self.assertTrue((root / "results.zip").is_file())
            with (root / "report/quality.csv").open() as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])


if __name__ == "__main__":
    unittest.main()
